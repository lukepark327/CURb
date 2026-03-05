from __future__ import annotations

import hashlib
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _effective_rank(in_features: int, out_features: int, rank: int) -> int:
    return max(1, min(int(rank), int(in_features), int(out_features)))


def _effective_k(
    *,
    in_features: int,
    out_features: int,
    rank: int,
    k: int | None,
) -> int:
    """
    BiLoRA (paper) allocates k active frequency components in an (out x in) grid.
    For fair comparison against CURb/CURLoRA in this repo, default to k=r_eff^2.
    """
    if k is not None:
        k_eff = int(k)
    else:
        r_eff = _effective_rank(in_features, out_features, rank)
        k_eff = int(r_eff * r_eff)
    k_eff = max(1, int(k_eff))
    # With replacement sampling supports k > out*in, but it becomes degenerate.
    # Clamp for sanity (keeps memory bounded for indices/params).
    k_eff = min(k_eff, int(out_features) * int(in_features))
    return int(k_eff)


def _sample_freq_indices_with_replacement(
    *,
    out_features: int,
    in_features: int,
    k: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample k (u,v) coordinates in an out_features x in_features grid.

    We use with-replacement sampling to avoid torch.randperm(out*in) which is
    infeasible for large LLM matrices.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    total = int(out_features) * int(in_features)
    flat = torch.randint(low=0, high=total, size=(int(k),), generator=gen, device="cpu", dtype=torch.int64)
    u = (flat // int(in_features)).to(device=device, non_blocking=(device.type == "cuda"))
    v = (flat % int(in_features)).to(device=device, non_blocking=(device.type == "cuda"))
    return u, v


def _stable_hash_int(*parts) -> int:
    # Avoid Python's randomized hash(); we need deterministic supports across runs.
    payload = "|".join(str(p) for p in parts)
    h = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


class BiLoRALinear(nn.Module):
    """
    BiLoRA adapter for a single Linear layer.

    This mirrors the paper's Fourier-domain sparse parameterization (Algorithm 1)
    and the official code's practical simplification of taking real(ifft2(.)).

    We keep base weight frozen and train only a 1D coefficient vector `theta`
    that populates a sparse frequency matrix at fixed indices (u_idx, v_idx).

    Forward computes:
      y = x W^T + alpha * x (ifft2(Ffreq).real)^T
    without materializing the dense delta-W by using 1D FFTs + scatter-add.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        rank: int,
        k: int | None = None,
        alpha: float | None = None,
        seed: int = 777,
        chunk_size: int = 0,
        freq_chunk_size: int = 8192,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        if weight.dim() != 2:
            raise ValueError(f"Expected 2D weight, got shape={tuple(weight.shape)}")

        out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # Keep a view of the original weight without cloning to avoid duplicating
        # large LLM matrices. We freeze it by storing as a buffer.
        self.register_buffer("weight", weight.detach())
        if bias is not None:
            self.register_buffer("bias", bias.detach())
        else:
            self.bias = None

        self.k = _effective_k(in_features=in_features, out_features=out_features, rank=rank, k=k)

        # Scaling: keep explicit for parity with other methods; default tries to
        # counteract ifft normalization (1/(m*n)) similarly to the official code.
        if alpha is None:
            # Official uses alpha=300 for dim=768. A simple dimension-aware default
            # is ~0.4*sqrt(m*n); we use 0.5*sqrt(m*n) as a robust heuristic.
            alpha = 0.5 * float(math.sqrt(max(1.0, float(out_features) * float(in_features))))
        self.alpha = float(alpha)

        # Fixed sparse frequency support.
        u, v = _sample_freq_indices_with_replacement(
            out_features=out_features,
            in_features=in_features,
            k=self.k,
            seed=int(seed),
            device=weight.device,
        )
        self.register_buffer("u_idx", u)
        self.register_buffer("v_idx", v)

        # Trainable frequency coefficients (real), initialized to zero as in the paper.
        self.theta = nn.Parameter(torch.zeros((self.k,), device=weight.device, dtype=torch.float32))

        self.chunk_size = int(max(0, chunk_size))  # 0 = no chunking (full batch)
        self.freq_chunk_size = int(max(1, freq_chunk_size))
        self.use_checkpoint = bool(use_checkpoint)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Checkpointing is critical for BiLoRA at LLM scales: without it, autograd
        # tends to retain large FFT/gather intermediates across many layers.
        def _fwd_impl(x_in: torch.Tensor, theta_in: torch.Tensor) -> torch.Tensor:
            base = F.linear(x_in, self.weight, self.bias)

            x_shape = x_in.shape
            x2 = x_in.reshape(-1, self.in_features)
            bsz = int(x2.shape[0])

            # Accumulate delta into base in chunks to avoid allocating large complex tensors.
            out = base.reshape(-1, self.out_features)

            # These buffers live on the module device; ensure indices are on the same device as x.
            u_idx = self.u_idx.to(device=x2.device)
            v_idx = self.v_idx.to(device=x2.device)
            freq_chunk = int(self.freq_chunk_size)

            inv_n = 1.0 / float(self.in_features)
            alpha = float(self.alpha)

            cs = self.chunk_size if self.chunk_size > 0 else bsz
            for start in range(0, bsz, cs):
                end = min(bsz, start + cs)
                xc = x2[start:end].to(dtype=torch.float32)

                # FFT uses negative exponent; conj gives the + exponent used by ifft2 formula.
                x_fft = torch.fft.fft(xc, n=self.in_features, dim=-1)
                x_pos = x_fft.conj()

                out_freq = torch.zeros(
                    (int(end - start), self.out_features),
                    device=x_pos.device,
                    dtype=x_pos.dtype,
                )
                # Process frequency components in blocks to avoid allocating (chunk, k) tensors
                # when k is large (e.g., k=r^2 for CURb-sized budgets).
                for f0 in range(0, int(self.k), freq_chunk):
                    f1 = min(int(self.k), f0 + freq_chunk)
                    vb = v_idx[f0:f1]
                    ub = u_idx[f0:f1]
                    tb = theta_in[f0:f1]
                    gathered = x_pos.index_select(dim=1, index=vb)  # (chunk, block_k), complex
                    contrib = gathered * tb
                    if inv_n != 1.0:
                        contrib = contrib.mul(inv_n)
                    out_freq.index_add_(dim=1, index=ub, source=contrib)

                delta = torch.fft.ifft(out_freq, n=self.out_features, dim=-1).real
                if alpha != 1.0:
                    delta = delta.mul(alpha)

                out[start:end].add_(delta.to(dtype=out.dtype))

            return out.reshape(*x_shape[:-1], self.out_features)

        if self.training and self.use_checkpoint:
            return checkpoint(_fwd_impl, x, self.theta, use_reentrant=False)
        return _fwd_impl(x, self.theta)

    @torch.no_grad()
    def merge(self):
        """
        Merge the current task's BiLoRA delta-W into the frozen base weight.

        This matches the official implementation:
          delta_W = real(ifft2(Ffreq)) * alpha
        """
        dev = self.weight.device
        dt = torch.float32

        # Dense frequency matrix is expensive but done only once per task per module.
        freq = torch.zeros((self.out_features, self.in_features), device=dev, dtype=dt)
        # If any duplicates exist (unlikely with large grids), accumulate to match forward().
        freq.index_put_(
            (self.u_idx, self.v_idx),
            self.theta.to(device=dev, dtype=dt),
            accumulate=True,
        )
        delta_w = torch.fft.ifft2(freq, dim=(-2, -1)).real
        if self.alpha != 1.0:
            delta_w.mul_(float(self.alpha))

        self.weight.add_(delta_w.to(dtype=self.weight.dtype))
        self.theta.zero_()


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


def inject_bilora(
    model,
    *,
    layer_indices,
    ffn_module_names,
    attn_module_names,
    rank: int,
    k: int | None = None,
    alpha: float | None = None,
    seed: int = 777,
    task_idx: int = 0,
    chunk_size: int = 256,
    freq_chunk_size: int = 8192,
    rank_overrides=None,
):
    """
    Replace target Linear modules with BiLoRALinear.

    Args:
      rank: base rank used only to derive default k=r_eff^2.
      k: optional override for number of active frequency components per module.
      alpha: optional scaling (see BiLoRALinear).
      seed: base seed; per-module seeds are derived deterministically.
      rank_overrides: same schema as CURb/CURLoRA overrides (mlp_gate_proj, attn_q_proj, ...)
    """
    base_seed = int(seed) + int(task_idx) * 10
    for layer_index in layer_indices:
        layer = model.model.layers[layer_index]
        for name in ffn_module_names:
            module = getattr(layer.mlp, name)
            # Reuse existing weight storage (no clone) to avoid duplicating large matrices.
            weight = module.weight.detach()
            bias = module.bias.detach() if module.bias is not None else None
            module_key = f"mlp_{name}"
            rank_val = int(rank_overrides[module_key]) if rank_overrides and module_key in rank_overrides else int(rank)
            # Derive a deterministic per-module seed.
            mod_seed = base_seed + (layer_index + 1) * 10007 + (_stable_hash_int("mlp", name) % 1000003)
            bilora = BiLoRALinear(
                weight,
                bias,
                rank=rank_val,
                k=k,
                alpha=alpha,
                seed=mod_seed,
                chunk_size=chunk_size,
                freq_chunk_size=freq_chunk_size,
            )
            setattr(layer.mlp, name, bilora)
        for name in attn_module_names:
            module = getattr(layer.self_attn, name)
            weight = module.weight.detach()
            bias = module.bias.detach() if module.bias is not None else None
            module_key = f"attn_{name}"
            rank_val = int(rank_overrides[module_key]) if rank_overrides and module_key in rank_overrides else int(rank)
            mod_seed = base_seed + (layer_index + 1) * 10007 + (_stable_hash_int("attn", name) % 1000003)
            bilora = BiLoRALinear(
                weight,
                bias,
                rank=rank_val,
                k=k,
                alpha=alpha,
                seed=mod_seed,
                chunk_size=chunk_size,
                freq_chunk_size=freq_chunk_size,
            )
            setattr(layer.self_attn, name, bilora)
    return model


def merge_bilora(model):
    for module in model.modules():
        if isinstance(module, BiLoRALinear):
            module.merge()
    return model


def strip_bilora(model):
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, BiLoRALinear):
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


def freeze_except_bilora_theta(model):
    for name, param in model.named_parameters():
        param.requires_grad = name.endswith(".theta")
    return model
