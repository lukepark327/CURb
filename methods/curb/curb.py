from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CURbLinear(nn.Module):
    def __init__(self, weight, bias, C, R, alpha: Optional[float] = None):
        super().__init__()
        self.register_buffer("weight", weight.detach().clone())
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None
        self.register_buffer("C", C.detach().clone())
        self.register_buffer("R", R.detach().clone())
        r = int(self.C.size(1))
        self.U = nn.Parameter(torch.zeros((r, r), device=self.C.device, dtype=self.C.dtype))
        # Keep alpha/r aligned with LoRA defaults when not explicitly overridden.
        self.alpha_base = float(alpha) if alpha is not None else float(2 * r)
        self.alpha = self.alpha_base

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        out_r = F.linear(x, self.R, bias=None)
        out_u = F.linear(out_r, self.U, bias=None)
        out_c = F.linear(out_u, self.C, bias=None)
        return base + (self.alpha * out_c)

    @torch.no_grad()
    def merge(self):
        delta = self.C @ self.U @ self.R
        self.weight.add_(self.alpha * delta)
        self.U.zero_()


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


def inject_curb(model, basis, layer_indices, ffn_module_names, attn_module_names, alpha: Optional[float] = None):
    for layer_index in layer_indices:
        layer = model.model.layers[layer_index]
        for name in ffn_module_names:
            key = f"layer_{layer_index}_mlp_{name}"
            module = getattr(layer.mlp, name)
            entry = basis["modules"][key]
            C = entry["C"].to(device=module.weight.device, dtype=module.weight.dtype)
            R = entry["R"].to(device=module.weight.device, dtype=module.weight.dtype)
            bias = module.bias.data.clone() if module.bias is not None else None
            curb = CURbLinear(module.weight.data, bias, C, R, alpha=alpha)
            setattr(layer.mlp, name, curb)
        for name in attn_module_names:
            key = f"layer_{layer_index}_self_attn_{name}"
            module = getattr(layer.self_attn, name)
            entry = basis["modules"][key]
            C = entry["C"].to(device=module.weight.device, dtype=module.weight.dtype)
            R = entry["R"].to(device=module.weight.device, dtype=module.weight.dtype)
            bias = module.bias.data.clone() if module.bias is not None else None
            curb = CURbLinear(module.weight.data, bias, C, R, alpha=alpha)
            setattr(layer.self_attn, name, curb)
    return model


def inject_curb_named_modules(
    model: nn.Module,
    basis: dict,
    module_names: Optional[list[str]] = None,
    alpha: Optional[float] = None,
    strict: bool = True,
    alpha_spectral_norm: bool = False,
):
    """
    Generic CURb injector that replaces arbitrary named modules with CURbLinear.

    This is useful for non-LLaMA architectures (e.g., HuggingFace ViT) where the
    target Linear layers live under different attribute paths.

    Expected basis format:
      basis["modules"][<qualified_module_name>] = {"C": Tensor, "R": Tensor, ...}

    Args:
      model: target nn.Module.
      basis: CURb basis dict (see above).
      module_names: list of qualified module names to replace. If None, uses all
        keys from basis["modules"].
      alpha: optional adapter scaling (defaults to 2*r inside CURbLinear).
      strict: if True, raise on missing modules / missing basis entries / wrong types.
    """
    if module_names is None:
        module_names = list((basis or {}).get("modules", {}).keys())
    if not isinstance(module_names, list):
        raise TypeError(f"module_names must be a list[str] or None, got: {type(module_names)}")

    basis_modules = (basis or {}).get("modules", {})
    if not isinstance(basis_modules, dict):
        raise TypeError("basis['modules'] must be a dict")

    missing_modules = []
    missing_basis = []
    bad_types = []

    for qualified_name in module_names:
        if qualified_name not in basis_modules:
            missing_basis.append(qualified_name)
            if strict:
                continue
            else:
                continue

        parent, child = _get_parent_and_child(model, qualified_name)
        if isinstance(parent, (nn.Sequential, nn.ModuleList)) and child.isdigit():
            module = parent[int(child)]
        elif isinstance(parent, nn.ModuleDict):
            module = parent.get(child, None)
        else:
            module = getattr(parent, child, None)

        if module is None:
            missing_modules.append(qualified_name)
            if strict:
                continue
            else:
                continue

        if isinstance(module, CURbLinear):
            # Avoid accidental double-injection.
            bad_types.append((qualified_name, "already_curb"))
            if strict:
                continue
            else:
                continue

        if not isinstance(module, nn.Linear):
            bad_types.append((qualified_name, type(module).__name__))
            if strict:
                continue
            else:
                continue

        entry = basis_modules[qualified_name]
        if not isinstance(entry, dict) or ("C" not in entry) or ("R" not in entry):
            missing_basis.append(qualified_name)
            if strict:
                continue
            else:
                continue

        C = entry["C"].to(device=module.weight.device, dtype=module.weight.dtype)
        R = entry["R"].to(device=module.weight.device, dtype=module.weight.dtype)
        bias = module.bias.data.clone() if module.bias is not None else None

        module_alpha = alpha
        if alpha_spectral_norm and module_alpha is not None:
            sC = entry.get("sigma_max_C") or float(torch.linalg.matrix_norm(C.float(), ord=2))
            sR = entry.get("sigma_max_R") or float(torch.linalg.matrix_norm(R.float(), ord=2))
            denom = max(sC * sR, 1e-12)
            module_alpha = float(module_alpha) / denom

        curb = CURbLinear(module.weight.data, bias, C, R, alpha=module_alpha)
        _set_child_module(parent, child, curb)

    if strict and (missing_basis or missing_modules or bad_types):
        msgs = []
        if missing_basis:
            msgs.append(f"missing basis entries: {missing_basis[:8]}" + (" ..." if len(missing_basis) > 8 else ""))
        if missing_modules:
            msgs.append(
                f"missing modules in model: {missing_modules[:8]}" + (" ..." if len(missing_modules) > 8 else "")
            )
        if bad_types:
            preview = [f"{n}({t})" for n, t in bad_types[:8]]
            msgs.append(f"non-Linear or invalid targets: {preview}" + (" ..." if len(bad_types) > 8 else ""))
        raise ValueError("inject_curb_named_modules failed (" + "; ".join(msgs) + ")")

    return model


def merge_curb(model):
    for module in model.modules():
        if isinstance(module, CURbLinear):
            module.merge()
    return model


def strip_curb(model):
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, CURbLinear):
            targets.append((name, module))
    for name, module in targets:
        in_features = int(module.weight.size(1))
        out_features = int(module.weight.size(0))
        linear = nn.Linear(in_features=in_features, out_features=out_features, bias=(module.bias is not None))
        linear.to(device=module.weight.device, dtype=module.weight.dtype)
        linear.weight.data = module.weight.detach().clone()
        if module.bias is not None:
            linear.bias.data = module.bias.detach().clone()
        parent, child = _get_parent_and_child(model, name)
        _set_child_module(parent, child, linear)
    return model


def freeze_except_curb_U(model):
    for name, param in model.named_parameters():
        param.requires_grad = (".U" in name)
    return model


def update_curb_alpha(model: nn.Module, factor: float):
    """Set alpha = alpha_base * factor for all CURbLinear modules."""
    for module in model.modules():
        if isinstance(module, CURbLinear):
            module.alpha = module.alpha_base * factor
