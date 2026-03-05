from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_lorac_state(pool_size: int) -> dict:
    if int(pool_size) < 1:
        raise ValueError(f"pool_size must be >= 1 (got {pool_size}).")
    return {
        "pool_size": int(pool_size),
        # module_key -> per-module state dict
        "modules": {},
    }


def _get_parent_and_child(model: nn.Module, qualified_name: str):
    parts = qualified_name.split(".")
    parent = model
    for p in parts[:-1]:
        if isinstance(parent, (nn.Sequential, nn.ModuleList)) and p.isdigit():
            parent = parent[int(p)]
        elif isinstance(parent, nn.ModuleDict):
            parent = parent[p]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]


def _set_child_module(parent: nn.Module, child_name: str, new_module: nn.Module):
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and child_name.isdigit():
        parent[int(child_name)] = new_module
    elif isinstance(parent, nn.ModuleDict):
        parent[child_name] = new_module
    else:
        setattr(parent, child_name, new_module)


def _default_alpha_for_rank(r: int) -> float:
    # Keep consistent with the repo's LoRA defaults (alpha=2*r) unless overridden.
    return float(2 * int(r))


class LoRACLinear(nn.Module):
    """
    LoRAC / LoRAC-IPC adapter wrapper for a single Linear layer.

    Key design for this repo:
      - Base weight is merged ("single composed model") at the end of each task, then adapters are stripped.
      - Per-task bank state (A/B, omega, omega_snap, IPC stats) lives in an external dict (method_ctx["lorac_state"]).
      - Forward adds only the *residual* delta relative to the merged base to avoid double-counting.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        module_key: str,
        state_ref: dict,
        pool_size: int,
        task_idx: int,
        rank: int,
        lora_alpha: Optional[float],
        ipc_enabled: bool,
        ipc_beta1: float,
        ipc_beta2: float,
        ipc_threshold: float,
        ipc_new_mask: bool,
    ):
        super().__init__()
        if weight.dim() != 2:
            raise ValueError(f"Expected 2D weight, got shape={tuple(weight.shape)}.")
        if int(rank) < 1:
            raise ValueError(f"rank must be >= 1 (got {rank}).")
        if int(pool_size) < 1:
            raise ValueError(f"pool_size must be >= 1 (got {pool_size}).")
        if int(task_idx) < 0 or int(task_idx) >= int(pool_size):
            raise ValueError(f"task_idx must be in [0, pool_size) (got {task_idx}, pool={pool_size}).")

        self.module_key = str(module_key)
        self.state_ref = state_ref
        self.pool_size = int(pool_size)
        self.task_idx = int(task_idx)

        out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)

        # Freeze base weight by keeping it as a buffer (no clone; reuse storage).
        self.register_buffer("weight", weight.detach())
        if bias is not None:
            self.register_buffer("bias", bias.detach())
        else:
            self.bias = None

        alpha_eff = float(_default_alpha_for_rank(self.rank) if lora_alpha is None else float(lora_alpha))
        self.lora_alpha = float(alpha_eff)
        self.scaling = float(self.lora_alpha / float(self.rank))

        # Current-task low-rank factors (trainable)
        self.A = nn.Parameter(torch.empty((self.in_features, self.rank), device=weight.device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros((self.rank, self.out_features), device=weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5.0))

        # Omega weights for history composition (trainable vector across tasks).
        # Stored persistently in state_ref as a CPU fp32 tensor; loaded into a Parameter each task.
        omega_init = state_ref.get("omega")
        if omega_init is None:
            omega_init = torch.ones((self.pool_size,), dtype=torch.float32, device="cpu")
        omega_init = omega_init.detach().to(device=weight.device, dtype=torch.float32)
        if omega_init.numel() != int(self.pool_size):
            raise ValueError(f"omega shape mismatch for {module_key}: got {tuple(omega_init.shape)} pool={pool_size}")
        self.omega = nn.Parameter(omega_init.clone())

        # Snapshot of omegas already merged into the base (CPU fp32 in state_ref).
        omega_snap = state_ref.get("omega_snap")
        if omega_snap is None:
            omega_snap = torch.zeros((self.pool_size,), dtype=torch.float32, device="cpu")
        omega_snap = omega_snap.detach().to(device=weight.device, dtype=torch.float32)
        if omega_snap.numel() != int(self.pool_size):
            raise ValueError(
                f"omega_snap shape mismatch for {module_key}: got {tuple(omega_snap.shape)} pool={pool_size}"
            )
        self.register_buffer("omega_snap", omega_snap.clone())

        # Past-task banks (CPU fp32 in state_ref). We materialize device copies once per task injection.
        a_bank_cpu = state_ref.get("A_bank", []) or []
        b_bank_cpu = state_ref.get("B_bank", []) or []
        if len(a_bank_cpu) != len(b_bank_cpu):
            raise ValueError(f"Bank length mismatch for {module_key}: A={len(a_bank_cpu)} B={len(b_bank_cpu)}")
        if len(a_bank_cpu) != int(task_idx):
            # Expected exactly `task_idx` historical entries.
            raise ValueError(f"Bank length for {module_key} must equal task_idx ({task_idx}), got {len(a_bank_cpu)}")

        self.A_bank = [t.detach().to(device=weight.device, dtype=torch.float32) for t in a_bank_cpu]
        self.B_bank = [t.detach().to(device=weight.device, dtype=torch.float32) for t in b_bank_cpu]

        # IPC config/state.
        self.ipc_enabled = bool(ipc_enabled)
        self.ipc_beta1 = float(ipc_beta1)
        self.ipc_beta2 = float(ipc_beta2)
        self.ipc_threshold = float(ipc_threshold)
        self.ipc_new_mask = bool(ipc_new_mask)

        # mask_prev: 1=adapt allowed, 0=frozen (monotone). Stored in state_ref on CPU.
        mask_prev = int(state_ref.get("mask_prev", 1))
        self.mask_prev = int(1 if mask_prev != 0 else 0)

        # Importance EMA lives in state_ref; keep per-module scalars (on the training device to avoid CPU sync).
        if self.ipc_enabled:
            ema_ipt = state_ref.get("ipc_exp_avg_ipt")
            ema_unc = state_ref.get("ipc_exp_avg_unc")
            if ema_ipt is None:
                ema_ipt = torch.zeros((), device=weight.device, dtype=torch.float32)
            else:
                ema_ipt = torch.as_tensor(ema_ipt, device=weight.device, dtype=torch.float32)
            if ema_unc is None:
                ema_unc = torch.zeros((), device=weight.device, dtype=torch.float32)
            else:
                ema_unc = torch.as_tensor(ema_unc, device=weight.device, dtype=torch.float32)
            self.register_buffer("ipc_exp_avg_ipt", ema_ipt.detach().clone())
            self.register_buffer("ipc_exp_avg_unc", ema_unc.detach().clone())
        else:
            self.ipc_exp_avg_ipt = None
            self.ipc_exp_avg_unc = None

        self._ipc_w: torch.Tensor | None = None

    def _residual_coeffs(self) -> torch.Tensor:
        # Return coeffs for historical tasks (length task_idx) used in residual:
        #   coeff_i = omega_used[i] - omega_snap[i]
        if self.task_idx <= 0:
            return torch.zeros((0,), device=self.omega.device, dtype=torch.float32)
        if self.ipc_enabled and self.mask_prev == 0:
            # Frozen: use omega_snap (no omega changes) => residual coeffs = 0.
            return torch.zeros((int(self.task_idx),), device=self.omega.device, dtype=torch.float32)
        omega_used = self.omega[: int(self.task_idx)]
        return omega_used - self.omega_snap[: int(self.task_idx)]

    def _compose_residual_AB(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Build (A_cat, B_cat) such that:
          A_cat @ B_cat = sum_{i<task} (omega_i - omega_snap_i) * (A_i @ B_i) + (A_cur @ B_cur) * mask_prev
        without materializing the dense residual weight.
        """
        if self.ipc_enabled and self.mask_prev == 0:
            return None

        # Historical terms.
        coeffs = self._residual_coeffs()
        a_parts = []
        b_parts = []
        if int(coeffs.numel()) > 0:
            for i, c in enumerate(coeffs):
                a_parts.append(self.A_bank[i] * c)
                b_parts.append(self.B_bank[i])

        # Current term (always coefficient 1 unless IPC-frozen).
        a_parts.append(self.A)
        b_parts.append(self.B)

        if not a_parts:
            return None
        a_cat = torch.cat(a_parts, dim=1)  # (in, r_total)
        b_cat = torch.cat(b_parts, dim=0)  # (r_total, out)
        return a_cat, b_cat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fast path for non-IPC: base linear + low-rank residual.
        if not self.ipc_enabled:
            base = F.linear(x, self.weight, self.bias)
            ab = self._compose_residual_AB()
            if ab is None:
                return base
            a_cat, b_cat = ab
            x2 = x.reshape(-1, self.in_features).to(dtype=torch.float32)
            h = x2 @ a_cat  # (B*N, r_total)
            out = (h @ b_cat).reshape(*x.shape[:-1], self.out_features)
            out = out.to(dtype=base.dtype)
            if self.scaling != 1.0:
                out = out.mul(float(self.scaling))
            return base + out

        # IPC path: compute the effective (merged-base + residual) weight and retain its grad.
        # NOTE: This is extremely memory-heavy for LLM matrices, but matches the official formula.
        w_base = self.weight.transpose(0, 1).to(dtype=torch.float32)  # (in, out)

        ab = self._compose_residual_AB()
        if ab is None:
            w_eff = w_base
        else:
            a_cat, b_cat = ab
            w_res = a_cat @ b_cat  # (in, out)
            if self.scaling != 1.0:
                w_res = w_res.mul(float(self.scaling))
            w_eff = w_base + w_res

        if self.training and w_eff.requires_grad:
            # Store for importance update and retain its grad.
            self._ipc_w = w_eff
            self._ipc_w.retain_grad()
        else:
            # Avoid holding huge dense weights for frozen modules.
            self._ipc_w = None

        x2 = x.reshape(-1, self.in_features).to(dtype=torch.float32)
        y = (x2 @ w_eff).reshape(*x.shape[:-1], self.out_features)
        if self.bias is not None:
            y = y + self.bias.to(dtype=y.dtype)
        return y.to(dtype=x.dtype if x.dtype.is_floating_point else y.dtype)

    def ortho_loss(self) -> torch.Tensor:
        # Mirror official loss_ortho: Fro norm of (A_all A_all^T - I).
        if self.task_idx <= 0:
            return self.A.new_zeros(())
        if self.ipc_enabled and self.mask_prev == 0:
            return self.A.new_zeros(())

        a_prev = [t.detach() for t in self.A_bank]
        a_all = torch.stack(a_prev + [self.A], dim=0)  # (t+1, in, r)
        # (t+1, r, in) -> ( (t+1)*r, in )
        a_all = a_all.permute(0, 2, 1).reshape(-1, self.in_features)
        gram = a_all @ a_all.t()
        eye = torch.eye(int(gram.shape[0]), device=gram.device, dtype=gram.dtype)
        return torch.norm(gram - eye, p="fro")

    @torch.no_grad()
    def merge_and_update_state(self):
        """
        Merge the current task's *residual* delta into the base weight, and persist bank state to CPU.
        """
        ab = self._compose_residual_AB()
        if ab is not None:
            a_cat, b_cat = ab
            delta = a_cat @ b_cat  # (in, out)
            if self.scaling != 1.0:
                delta = delta.mul(float(self.scaling))
            self.weight.add_(delta.transpose(0, 1).to(dtype=self.weight.dtype))

        # Persist current task factors.
        self.state_ref.setdefault("A_bank", [])
        self.state_ref.setdefault("B_bank", [])
        self.state_ref["A_bank"].append(self.A.detach().to(device="cpu", dtype=torch.float32))
        self.state_ref["B_bank"].append(self.B.detach().to(device="cpu", dtype=torch.float32))

        # Persist omega and update omega_snap to reflect the merged base.
        omega_cpu = self.omega.detach().to(device="cpu", dtype=torch.float32)
        self.state_ref["omega"] = omega_cpu

        omega_snap_cpu = self.state_ref.get("omega_snap")
        if omega_snap_cpu is None:
            omega_snap_cpu = torch.zeros((self.pool_size,), dtype=torch.float32, device="cpu")
        omega_snap_cpu = omega_snap_cpu.detach().to(device="cpu", dtype=torch.float32).clone()

        # For historical tasks < task_idx: base now matches omega values (since we merged residual).
        if self.task_idx > 0:
            omega_snap_cpu[: int(self.task_idx)] = omega_cpu[: int(self.task_idx)]
        # For the task just learned: base includes delta with coefficient 1.
        omega_snap_cpu[int(self.task_idx)] = 1.0

        self.state_ref["omega_snap"] = omega_snap_cpu
        self.omega_snap.copy_(omega_snap_cpu.to(device=self.omega_snap.device, dtype=self.omega_snap.dtype))

        # Persist IPC EMA stats (device scalars).
        if self.ipc_enabled:
            self.state_ref["ipc_exp_avg_ipt"] = self.ipc_exp_avg_ipt.detach().clone()
            self.state_ref["ipc_exp_avg_unc"] = self.ipc_exp_avg_unc.detach().clone()

    @torch.no_grad()
    def ipc_update_ipt(self):
        if not self.ipc_enabled:
            return
        if self._ipc_w is None:
            return
        if self._ipc_w.grad is None:
            # Clear to avoid holding a large tensor when backward didn't populate grad.
            self._ipc_w = None
            return
        w = self._ipc_w
        g = self._ipc_w.grad

        ipt = (w * g).abs().mean().detach()
        # Update sensitivity (exp_avg_ipt), then uncertainty (exp_avg_unc).
        self.ipc_exp_avg_ipt.mul_(float(self.ipc_beta1)).add_(ipt, alpha=(1.0 - float(self.ipc_beta1)))
        unc = (ipt - self.ipc_exp_avg_ipt).abs()
        self.ipc_exp_avg_unc.mul_(float(self.ipc_beta2)).add_(unc, alpha=(1.0 - float(self.ipc_beta2)))

        # Drop the reference to the huge matrix as soon as possible.
        self._ipc_w = None

    @torch.no_grad()
    def ipc_score(self) -> torch.Tensor:
        if not self.ipc_enabled:
            return torch.zeros((), device=self.weight.device, dtype=torch.float32)
        return (self.ipc_exp_avg_ipt * self.ipc_exp_avg_unc).detach()

    @torch.no_grad()
    def ipc_reset_stats(self):
        if not self.ipc_enabled:
            return
        self.ipc_exp_avg_ipt.zero_()
        self.ipc_exp_avg_unc.zero_()
        self.state_ref["ipc_exp_avg_ipt"] = self.ipc_exp_avg_ipt.detach().clone()
        self.state_ref["ipc_exp_avg_unc"] = self.ipc_exp_avg_unc.detach().clone()


def inject_lorac(
    model,
    *,
    layer_indices,
    ffn_module_names,
    attn_module_names,
    lora_ranks: dict,
    lorac_state: dict,
    task_idx: int,
    lora_alpha: Optional[float],
    ipc_enabled: bool,
    ipc_beta1: float,
    ipc_beta2: float,
    ipc_threshold: float,
    ipc_new_mask: bool,
):
    if lorac_state is None or not isinstance(lorac_state, dict):
        raise ValueError("lorac_state must be a dict (use init_lorac_state()).")
    pool_size = int(lorac_state.get("pool_size", 0))
    if pool_size < 1:
        raise ValueError("lorac_state missing pool_size (use init_lorac_state()).")
    modules_state = lorac_state.setdefault("modules", {})

    for layer_index in layer_indices:
        layer = model.model.layers[layer_index]
        # MLP modules (e.g., gate_proj)
        for name in ffn_module_names:
            module = getattr(layer.mlp, name)
            weight = module.weight.detach()
            bias = module.bias.detach() if module.bias is not None else None
            module_key = f"layers.{layer_index}.mlp.{name}"
            rank_key = f"mlp_{name}"
            r = int(lora_ranks[rank_key])

            state_ref = modules_state.get(module_key)
            if state_ref is None:
                state_ref = {
                    "rank": int(r),
                    "A_bank": [],
                    "B_bank": [],
                    "omega": torch.ones((pool_size,), dtype=torch.float32, device="cpu"),
                    "omega_snap": torch.zeros((pool_size,), dtype=torch.float32, device="cpu"),
                    "mask_prev": 1,
                    "ipc_exp_avg_ipt": None,
                    "ipc_exp_avg_unc": None,
                }
                modules_state[module_key] = state_ref
            if int(state_ref.get("rank", r)) != int(r):
                raise ValueError(f"LoRAC rank mismatch for {module_key}: state={state_ref.get('rank')} requested={r}")

            lorac = LoRACLinear(
                weight,
                bias,
                module_key=module_key,
                state_ref=state_ref,
                pool_size=pool_size,
                task_idx=int(task_idx),
                rank=int(r),
                lora_alpha=lora_alpha,
                ipc_enabled=bool(ipc_enabled),
                ipc_beta1=float(ipc_beta1),
                ipc_beta2=float(ipc_beta2),
                ipc_threshold=float(ipc_threshold),
                ipc_new_mask=bool(ipc_new_mask),
            )
            setattr(layer.mlp, name, lorac)

        # Attention modules (e.g., q_proj/k_proj)
        for name in attn_module_names:
            module = getattr(layer.self_attn, name)
            weight = module.weight.detach()
            bias = module.bias.detach() if module.bias is not None else None
            module_key = f"layers.{layer_index}.self_attn.{name}"
            rank_key = f"attn_{name}"
            r = int(lora_ranks[rank_key])

            state_ref = modules_state.get(module_key)
            if state_ref is None:
                state_ref = {
                    "rank": int(r),
                    "A_bank": [],
                    "B_bank": [],
                    "omega": torch.ones((pool_size,), dtype=torch.float32, device="cpu"),
                    "omega_snap": torch.zeros((pool_size,), dtype=torch.float32, device="cpu"),
                    "mask_prev": 1,
                    "ipc_exp_avg_ipt": None,
                    "ipc_exp_avg_unc": None,
                }
                modules_state[module_key] = state_ref
            if int(state_ref.get("rank", r)) != int(r):
                raise ValueError(f"LoRAC rank mismatch for {module_key}: state={state_ref.get('rank')} requested={r}")

            lorac = LoRACLinear(
                weight,
                bias,
                module_key=module_key,
                state_ref=state_ref,
                pool_size=pool_size,
                task_idx=int(task_idx),
                rank=int(r),
                lora_alpha=lora_alpha,
                ipc_enabled=bool(ipc_enabled),
                ipc_beta1=float(ipc_beta1),
                ipc_beta2=float(ipc_beta2),
                ipc_threshold=float(ipc_threshold),
                ipc_new_mask=bool(ipc_new_mask),
            )
            setattr(layer.self_attn, name, lorac)

    return model


def freeze_except_lorac(model):
    for name, param in model.named_parameters():
        param.requires_grad = name.endswith(".A") or name.endswith(".B") or name.endswith(".omega")
    return model


@torch.no_grad()
def update_lorac_ipc_importance(model):
    for module in model.modules():
        if isinstance(module, LoRACLinear):
            module.ipc_update_ipt()
    return model


def lorac_ortho_loss(model) -> torch.Tensor:
    loss = None
    for module in model.modules():
        if isinstance(module, LoRACLinear):
            cur = module.ortho_loss()
            loss = cur if loss is None else (loss + cur)
    if loss is None:
        # No adapters injected.
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        return torch.zeros((), device=dev)
    return loss


@torch.no_grad()
def merge_lorac(model):
    modules: list[LoRACLinear] = []
    for m in model.modules():
        if isinstance(m, LoRACLinear):
            modules.append(m)
    if not modules:
        return model

    ipc_enabled = any(m.ipc_enabled for m in modules)
    ipc_threshold = float(modules[0].ipc_threshold)
    ipc_new_mask = bool(modules[0].ipc_new_mask)

    # 1) Merge per-module residual and persist bank state.
    for m in modules:
        m.merge_and_update_state()

    # 2) IPC masking at end of task (monotone mask, global top-k by importance score).
    if ipc_enabled and ipc_threshold > 0.0:
        # Build score vector on-device.
        scores = []
        active = []
        for m in modules:
            prev_mask = int(m.state_ref.get("mask_prev", 1))
            s = m.ipc_score()
            if prev_mask == 0:
                s = s * 0.0
            scores.append(s)
            active.append(1 if prev_mask != 0 else 0)
        score_vec = torch.stack(scores, dim=0)
        active_vec = torch.as_tensor(active, device=score_vec.device, dtype=torch.int64)

        total_num = int(active_vec.sum().item()) if ipc_new_mask else int(score_vec.numel())
        k = int(total_num * float(ipc_threshold))
        if k > 0:
            k = min(k, int(score_vec.numel()))
            topk_vals, _ = torch.topk(score_vec, k=k)
            mask_threshold = topk_vals[-1]
            for m, s in zip(modules, scores):
                if int(m.state_ref.get("mask_prev", 1)) == 0:
                    continue
                if s >= mask_threshold and float(s.detach().item()) > 0.0:
                    m.state_ref["mask_prev"] = 0

    # 3) Reset IPC stats for the next task.
    for m in modules:
        if m.ipc_enabled:
            m.ipc_reset_stats()

    return model


def strip_lorac(model):
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, LoRACLinear):
            targets.append((name, module))

    for name, module in targets:
        linear = nn.Linear(
            in_features=int(module.weight.size(1)),
            out_features=int(module.weight.size(0)),
            bias=(module.bias is not None),
        )
        linear.to(device=module.weight.device, dtype=module.weight.dtype)
        linear.weight.data = module.weight.detach().clone()
        if module.bias is not None:
            linear.bias.data = module.bias.detach().clone()

        parent, child = _get_parent_and_child(model, name)
        _set_child_module(parent, child, linear)

    return model
