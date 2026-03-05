import torch
import torch.nn as nn
import torch.nn.functional as F


# CURLinear
class CURLinear(nn.Module):
    def __init__(self, C, U, R, bias=None, row_indices=None, col_indices=None, fuse: str = "none"):
        super(CURLinear, self).__init__()
        self.register_buffer('C', C)
        self.register_buffer('R', R)
        self.U = nn.Parameter(U)  # U is trainable
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

        # Store indices for interpretability
        self.row_indices = row_indices
        self.col_indices = col_indices

        # Placeholders for accumulated activations
        self.capture_activations = False
        self.activation_R_accum = None
        self.activation_C_accum = None
        self.nsamples = 0

        # Fused mode
        # "none" | "cu" | "ur"
        self.fuse = fuse if fuse in ("none", "cu", "ur") else "none"
        self._fused_w = None          # torch.Tensor or None
        self._fused_kind = None       # "cu" | "ur" | None
        self._fused_device = None
        self._fused_dtype = None

    def set_fuse(self, mode: str = "none"):
        assert mode in ("none", "cu", "ur")
        self.fuse = mode
        # invalidate cache
        self._fused_w = None
        self._fused_kind = None
        self._fused_device = None
        self._fused_dtype = None

    @torch.no_grad()
    def _maybe_build_fused(self, device, dtype):
        if self.fuse == "none":
            return
        if self.training or self.capture_activations:
            return
        if (
            self._fused_w is not None and
            self._fused_kind == self.fuse and
            self._fused_device == device and
            self._fused_dtype == dtype
        ):
            return

        if self.fuse == "cu":
            # C @ U : [m, r]
            fused = (self.C.to(device=device, dtype=dtype) @
                     self.U.to(device=device, dtype=dtype))
        else:  # "ur"
            # U @ R : [r, n]
            fused = (self.U.to(device=device, dtype=dtype) @
                     self.R.to(device=device, dtype=dtype))
        self._fused_w = fused.contiguous()
        self._fused_kind = self.fuse
        self._fused_device = device
        self._fused_dtype = dtype

    def forward(self, x):
        # 2-matmul fastpath
        if (self.fuse != "none") and (not self.training) and (not self.capture_activations):
            self._maybe_build_fused(x.device, x.dtype)
            if (self._fused_w is not None) and (self._fused_kind == "cu"):
                # out = (x R^T) (C U)^T + b
                out_R = F.linear(x, self.R, bias=None)
                out = F.linear(out_R, self._fused_w, bias=self.bias)
                return out
            elif (self._fused_w is not None) and (self._fused_kind == "ur"):
                # out = (x (U R)^T) C^T + b
                out_UR = F.linear(x, self._fused_w, bias=None)
                out = F.linear(out_UR, self.C, bias=self.bias)
                return out

        # # y = ((x @ R.T) @ U.T) @ C.T
        # out_R = x.matmul(self.R.t())  # Shape: (batch_size, seq_length, rank)
        # # Shape: (batch_size, seq_length, rank)
        # out_U = out_R.matmul(self.U.t())
        # # Shape: (batch_size, seq_length, output_dim)
        # out_C = out_U.matmul(self.C.t())
        # if self.bias is not None:
        #     out_C += self.bias

        # out_R = x @ R^T
        out_R = F.linear(x, self.R, bias=None)
        # out_U = out_R @ U^T
        out_U = F.linear(out_R, self.U, bias=None)
        # out_C = out_U @ C^T + b
        out_C = F.linear(out_U, self.C, bias=self.bias)

        if self.capture_activations:
            # Accumulate activation_R  # TODO: (out_R)?
            activation_R = out_R.detach()
            # Sum over batch and sequence dimensions
            activation_R_sum = activation_R.sum(dim=(0, 1))
            if self.activation_R_accum is None:
                self.activation_R_accum = activation_R_sum
            else:
                self.activation_R_accum += activation_R_sum

            # Accumulate activation_C  # TODO: (out_U)?
            activation_C = out_U.detach()  # Collect from out_U
            # Sum over batch and sequence dimensions
            activation_C_sum = activation_C.sum(dim=(0, 1))
            if self.activation_C_accum is None:
                self.activation_C_accum = activation_C_sum
            else:
                self.activation_C_accum += activation_C_sum

            # Update nsamples (total positions)
            total_positions = activation_R.size(
                0) * activation_R.size(1)  # batch_size * seq_length
            self.nsamples += total_positions

        return out_C

    def reset_activations(self):
        self.capture_activations = False
        self.activation_R_accum = None
        self.activation_C_accum = None
        self.nsamples = 0

    def activate_capture(self):
        self.capture_activations = True

    def deactivate_capture(self):
        self.capture_activations = False


def enable_cur_fusion(model: nn.Module, mode: str = "ur"):
    assert mode in ("none", "cu", "ur")
    for m in model.modules():
        if isinstance(m, CURLinear):
            m.set_fuse(mode)
    return model


def activate_capture_for_all_CURLinear_modules(model):
    for module in model.modules():
        if isinstance(module, CURLinear):
            module.activate_capture()


def deactivate_capture_for_all_CURLinear_modules(model):
    for module in model.modules():
        if isinstance(module, CURLinear):
            module.deactivate_capture()


def reset_activations_for_all_CURLinear_modules(model):
    for module in model.modules():
        if isinstance(module, CURLinear):
            module.reset_activations()


def rebuild_model_with_W(model):
    """
    Rebuilds the model by replacing CURLinear modules with Linear modules with weight W = C @ U @ R.
    """
    for name, module in model.named_modules():
        if isinstance(module, CURLinear):
            # Reconstruct W = C @ U @ R
            C = module.C
            U = module.U
            R = module.R
            W = C @ U @ R
            bias = module.bias
            # Create a new Linear module
            in_features = R.size(1)  # Original input features
            out_features = C.size(0)  # Original output features
            linear = nn.Linear(
                in_features=in_features, out_features=out_features, bias=(bias is not None))
            linear.weight.data = W
            if bias is not None:
                linear.bias.data = bias
            # Replace the CURLinear module with the new Linear module
            # Navigate to the parent module
            parent_module = model
            name_parts = name.split('.')
            for part in name_parts[:-1]:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, name_parts[-1], linear)
    return model


class FusedCURLinear(nn.Module):
    """
    - fuse='ur': out = F.linear(F.linear(x, UR), C, bias)
    - fuse='cu': out = F.linear(F.linear(x, R),  CU, bias)
    """

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, bias: torch.Tensor = None, fuse: str = "ur"):
        super().__init__()
        assert fuse in ("ur", "cu")
        # Promote to Parameter (counted in param count regardless of requires_grad)
        self.w1 = nn.Parameter(w1.detach().clone(), requires_grad=False)
        self.w2 = nn.Parameter(w2.detach().clone(), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(
                bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None
        self.fuse = fuse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First matmul
        out = F.linear(x, self.w1, bias=None)
        # Second matmul (+ bias)
        out = F.linear(out, self.w2, bias=self.bias)
        return out

    def extra_repr(self) -> str:
        return f"fuse='{self.fuse}', w1={tuple(self.w1.shape)}, w2={tuple(self.w2.shape)}, bias={'True' if self.bias is not None else 'False'}"


def _get_parent_and_child(model: nn.Module, qualified_name: str):
    """
    Given a dot-separated qualified name like 'encoder.layers.3.attn.proj',
    return the parent module and the last child name/index.
    Handles ModuleList/Sequential/ModuleDict.
    """
    parts = qualified_name.split(".")
    parent = model
    for p in parts[:-1]:
        if isinstance(parent, (nn.Sequential, nn.ModuleList)) and p.isdigit():
            parent = parent[int(p)]
        elif isinstance(parent, nn.ModuleDict):
            parent = parent[p]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    return parent, last


def _set_child_module(parent: nn.Module, child_name: str, new_module: nn.Module):
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and child_name.isdigit():
        parent[int(child_name)] = new_module
    elif isinstance(parent, nn.ModuleDict):
        parent[child_name] = new_module
    else:
        setattr(parent, child_name, new_module)


def rebuild_cur_fusion(model: nn.Module, fuse: str = "ur") -> nn.Module:
    """
    Replace CURLinear(C,U,R) modules with 2-matmul FusedCURLinear modules.
    - fuse='ur':  physically fuse (U @ R) and C -> (UR, C)
    - fuse='cu':  physically fuse (C @ U) and R -> (R, CU)
    Returns the model (modified in-place).
    """
    from cur_utils.cur_models import CURLinear  # noqa: F811

    assert fuse in ("ur", "cu")

    # Collect names first to avoid mutation during iteration.
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, CURLinear):
            targets.append((name, module))

    # Replace
    with torch.no_grad():
        for name, module in targets:
            C = module.C            # [m, r] buffer
            U = module.U            # [r, r] parameter
            R = module.R            # [r, n] buffer
            bias = module.bias      # [m] or None

            # Align dtype/device
            dev = U.device
            dt = U.dtype
            C_ = C.to(device=dev, dtype=dt)
            U_ = U.to(device=dev, dtype=dt)
            R_ = R.to(device=dev, dtype=dt)
            b_ = bias.to(device=dev, dtype=dt) if bias is not None else None

            if fuse == "ur":
                # First matmul weight: (UR) [r, n], second: C [m, r]
                w1 = U_ @ R_
                w2 = C_
            else:  # fuse == "cu"
                # First matmul weight: R [r, n], second: (CU) [m, r]
                w1 = R_
                w2 = C_ @ U_

            fused = FusedCURLinear(w1=w1, w2=w2, bias=b_, fuse=fuse)

            # Replace in parent module
            parent, child = _get_parent_and_child(model, name)
            _set_child_module(parent, child, fused)

    return model


# WANDA
class WandaWrappedModule:
    """
    Collects sqrt(E[x^2]) per input feature, excluding PAD tokens via attention_mask.
    Batch-size invariant by design (mean over valid tokens).
    """

    def __init__(self, module, acc_device='cuda', acc_dtype=torch.float64):
        self.module = module
        self.acc_device = acc_device
        self.acc_dtype = acc_dtype
        # (d_in,), accumulated sum of squares over valid tokens
        self.sum_sq = None
        self.nsamples = 0           # # of valid token positions
        # (B, T) attention_mask for the current forward
        self.current_mask = None
        self.handle = None

    def set_current_mask(self, mask: torch.Tensor):
        # mask: (B, T) with 1 for valid tokens
        self.current_mask = mask

    def add_batch(self, module, inp, out):
        with torch.no_grad():
            x = inp[0].detach()  # (B, T, d) or (N, d)
            if x.dim() == 3:
                B, T, D = x.shape
                if self.current_mask is not None:
                    m = self.current_mask.to(x.device).bool()
                    # guard for any off-by-one in T
                    if m.shape[1] != T:
                        Tcommon = min(T, m.shape[1])
                        x = x[:, :Tcommon, :]
                        m = m[:, :Tcommon]
                    x = x[m]  # (N_valid, D)
                else:
                    x = x.view(-1, D)
            elif x.dim() == 2:
                D = x.size(-1)
                # no time dim; treat all as valid
            else:
                x = x.view(-1, x.size(-1))

            if x.numel() == 0:
                return

            # accumulate on cuda/double for numerical stability (batch-order invariant up to fp error)
            xsq_sum = (x.to(self.acc_device, dtype=self.acc_dtype).pow(2)).sum(
                dim=0)  # (D,)
            if self.sum_sq is None:
                self.sum_sq = torch.zeros_like(
                    xsq_sum, device=self.acc_device, dtype=self.acc_dtype)
            self.sum_sq += xsq_sum
            self.nsamples += x.size(0)  # valid token count only

    def register_hook(self):
        if self.handle is None:
            self.handle = self.module.register_forward_hook(self.add_batch)

    def remove_hook(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def get_activation_norm(self):
        # sqrt(E[x^2]) over valid tokens
        if self.sum_sq is None or self.nsamples == 0:
            return None
        # (d_in,), on acc_device
        return torch.sqrt(self.sum_sq / float(self.nsamples))


# Cov
class CovWrappedModule:
    """
    Full second-moment / covariance collector (batch-size invariant).
    Excludes PAD tokens using attention_mask; accumulates on acc_device/acc_dtype.
    """

    def __init__(self, module, acc_device='cuda', acc_dtype=torch.float64):
        self.module = module
        self.acc_device = acc_device
        self.acc_dtype = acc_dtype
        self.nsamples = 0
        self.sum_x = None     # (d_in,)
        self.sum_xxT = None   # (d_in, d_in)
        self.current_mask = None
        self.handle = None

    def set_current_mask(self, mask: torch.Tensor):
        self.current_mask = mask  # (B, T)

    def add_batch(self, module, inp, out):
        with torch.no_grad():
            x = inp[0].detach()  # (B, T, d) or (N, d)
            if x.dim() == 3:
                B, T, D = x.shape
                if self.current_mask is not None:
                    m = self.current_mask.to(x.device).bool()
                    if m.shape[1] != T:
                        Tcommon = min(T, m.shape[1])
                        x = x[:, :Tcommon, :]
                        m = m[:, :Tcommon]
                    x = x[m]  # (N_valid, D)
                else:
                    x = x.view(-1, D)
            else:
                # already (N, d)
                pass

            if x.numel() == 0:
                return

            x_acc = x.to(self.acc_device, dtype=self.acc_dtype)
            d_in = x_acc.size(-1)
            if self.sum_x is None:
                self.sum_x = torch.zeros(
                    d_in, dtype=self.acc_dtype, device=self.acc_device)
                self.sum_xxT = torch.zeros(
                    d_in, d_in, dtype=self.acc_dtype, device=self.acc_device)

            self.sum_x += x_acc.sum(dim=0)
            # mm on double; deterministic across batch segmentations up to fp64 precision
            self.sum_xxT += x_acc.t().mm(x_acc)
            self.nsamples += x_acc.size(0)

    def register_hook(self):
        if self.handle is None:
            self.handle = self.module.register_forward_hook(self.add_batch)

    def remove_hook(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def reset(self):
        self.nsamples = 0
        self.sum_x = None
        self.sum_xxT = None

    @torch.no_grad()
    def get_input_second_moment(self, device=None, dtype=torch.float32):
        if self.sum_xxT is None or self.nsamples == 0:
            return None
        dev = self.acc_device if device is None else device
        return (self.sum_xxT / float(self.nsamples)).to(device=dev, dtype=dtype)

    @torch.no_grad()
    def get_input_covariance(self, unbiased: bool = False, device=None, dtype=torch.float32):
        if self.sum_xxT is None or self.sum_x is None or self.nsamples <= 1:
            return None
        n = float(self.nsamples)
        mu = (self.sum_x / n).unsqueeze(1)
        ExxT = self.sum_xxT / n
        cov = ExxT - mu @ mu.t()
        if unbiased and self.nsamples > 1:
            cov = cov * (n / (n - 1.0))
        dev = self.acc_device if device is None else device
        return cov.to(device=dev, dtype=dtype)

    @torch.no_grad()
    def get_input_cov_sqrt(self, eps: float = 1e-12, unbiased: bool = False,
                           device=None, dtype=torch.float32):
        cov = self.get_input_covariance(
            unbiased=unbiased, device=self.acc_device, dtype=self.acc_dtype)
        if cov is None:
            return None
        cov = 0.5 * (cov + cov.t())
        d = cov.shape[0]
        cov = cov + eps * torch.eye(d, dtype=cov.dtype, device=cov.device)
        evals, evecs = torch.linalg.eigh(cov)
        evals = torch.clamp(evals, min=0.0).sqrt()
        out = (evecs * evals.unsqueeze(0)) @ evecs.t()
        dev = self.acc_device if device is None else device
        return out.to(device=dev, dtype=dtype)

    @torch.no_grad()
    def get_rms(self, eps: float = 1e-12, device=None, dtype=torch.float32):
        ExxT = self.get_input_second_moment(
            device=self.acc_device, dtype=self.acc_dtype)
        if ExxT is None:
            return None
        diag = torch.diag(ExxT).clamp_min(eps)
        out = torch.sqrt(diag)
        dev = self.acc_device if device is None else device
        return out.to(device=dev, dtype=dtype)


# Activation
# Hooks to collect activations (for interpretability)
class ActivationAccumulator:
    def __init__(self, module):
        self.module = module
        self.input_activation_accum = None
        self.output_activation_accum = None
        self.nsamples = 0
        self.capture_activations = False

    def add_batch(self, module, inp, out):
        if not self.capture_activations:
            return

        # Collect activations from module inputs
        # Shape: (batch_size, seq_length, input_dim)
        activation_inp = inp[0].detach()
        # Flatten to (total_positions, input_dim)
        activation_inp = activation_inp.view(-1, activation_inp.size(-1))
        activation_inp_sum = activation_inp.sum(
            dim=0)  # Sum over all positions
        if self.input_activation_accum is None:
            self.input_activation_accum = activation_inp_sum
        else:
            self.input_activation_accum += activation_inp_sum

        # Collect activations from module outputs
        activation_out = out.detach()  # Shape: (batch_size, seq_length, output_dim)
        activation_out = activation_out.view(-1, activation_out.size(-1))
        activation_out_sum = activation_out.sum(dim=0)
        if self.output_activation_accum is None:
            self.output_activation_accum = activation_out_sum
        else:
            self.output_activation_accum += activation_out_sum

        # Update nsamples
        self.nsamples += activation_inp.size(0)  # Total number of positions

    def register_hook(self):
        self.handle = self.module.register_forward_hook(self.add_batch)

    def remove_hook(self):
        self.handle.remove()

    def get_mean_input_activation(self):
        if self.input_activation_accum is not None and self.nsamples > 0:
            return self.input_activation_accum / self.nsamples
        else:
            return None

    def get_mean_output_activation(self):
        if self.output_activation_accum is not None and self.nsamples > 0:
            return self.output_activation_accum / self.nsamples
        else:
            return None

    def reset(self):
        self.input_activation_accum = None
        self.output_activation_accum = None
        self.nsamples = 0
        self.capture_activations = False

    def activate_capture(self):
        self.capture_activations = True

    def deactivate_capture(self):
        self.capture_activations = False


def activate_capture_for_all_ActivationAccumulator(wrapped_modules):
    for wrapped_module in wrapped_modules.values():
        wrapped_module.activate_capture()


def deactivate_capture_for_all_ActivationAccumulator(wrapped_modules):
    for wrapped_module in wrapped_modules.values():
        wrapped_module.deactivate_capture()


def reset_activations_for_all_ActivationAccumulator(wrapped_modules):
    for wrapped_module in wrapped_modules.values():
        wrapped_module.reset()
