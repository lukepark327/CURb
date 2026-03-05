import numpy as np
from torch import nn
import torch


def _sanitize_probabilities(probs: torch.Tensor) -> torch.Tensor:
    probs = torch.nan_to_num(probs.float(), nan=0.0, posinf=0.0, neginf=0.0)
    probs = torch.clamp(probs, min=0.0)
    n = int(probs.numel())
    if n < 1:
        raise ValueError("Probability tensor must have at least one element.")
    total = probs.sum()
    if (not torch.isfinite(total)) or float(total.item()) <= 0.0:
        return torch.full((n,), 1.0 / float(n), dtype=torch.float32, device=probs.device)
    return probs / total


def compute_selection_probabilities(A):
    column_norms_squared = torch.sum(A**2, axis=0)
    row_norms_squared = torch.sum(A**2, axis=1)
    total_sum_squares = torch.sum(column_norms_squared)
    if (not torch.isfinite(total_sum_squares)) or float(total_sum_squares.item()) <= 0.0:
        n_cols = int(column_norms_squared.numel())
        n_rows = int(row_norms_squared.numel())
        column_probs = torch.full(
            (n_cols,), 1.0 / float(max(1, n_cols)), dtype=torch.float32, device=A.device
        )
        row_probs = torch.full(
            (n_rows,), 1.0 / float(max(1, n_rows)), dtype=torch.float32, device=A.device
        )
        return column_probs, row_probs
    column_probs = column_norms_squared / total_sum_squares
    row_probs = row_norms_squared / total_sum_squares
    column_probs = _sanitize_probabilities(column_probs)
    row_probs = _sanitize_probabilities(row_probs)
    return column_probs, row_probs


def select_indices_with_replacement(probs, k):
    probs = _sanitize_probabilities(probs)
    inverted_P = torch.where(probs > 0.0, 1.0 / (probs + 0.001), torch.zeros_like(probs))
    inverted_P = torch.nan_to_num(inverted_P, nan=0.0, posinf=0.0, neginf=0.0)
    inverted_P = torch.clamp(inverted_P, min=0.0)
    probs = _sanitize_probabilities(inverted_P)
    probs_np = probs.detach().cpu().numpy().astype(np.float64, copy=False)
    probs_np = np.nan_to_num(probs_np, nan=0.0, posinf=0.0, neginf=0.0)
    probs_sum = float(probs_np.sum())
    if (not np.isfinite(probs_sum)) or probs_sum <= 0.0:
        probs_np = np.full((len(probs_np),), 1.0 / float(max(1, len(probs_np))), dtype=np.float64)
    else:
        probs_np = probs_np / probs_sum

    return np.random.choice(len(probs_np), size=k, replace=True, p=probs_np)


def adjust_duplicates(selected_indices, A, axis):
    unique_indices, counts = np.unique(selected_indices, return_counts=True)
    adjusted_matrix = A[:,
                        unique_indices] if axis == 1 else A[unique_indices, :]

    for idx, count in enumerate(counts):
        if count > 1:
            scaling_factor = np.sqrt(count)
            if axis == 1:
                adjusted_matrix[:, idx] *= scaling_factor
            else:
                adjusted_matrix[idx, :] *= scaling_factor

    return adjusted_matrix, unique_indices


def cur_decomposition(A, c):
    r = c
    column_probs, row_probs = compute_selection_probabilities(A)
    selected_columns_np = select_indices_with_replacement(column_probs, c)
    selected_rows_np = select_indices_with_replacement(row_probs, r)
    selected_columns = torch.as_tensor(
        selected_columns_np, device=A.device, dtype=torch.long
    )
    selected_rows = torch.as_tensor(
        selected_rows_np, device=A.device, dtype=torch.long
    )

    C = A.index_select(1, selected_columns)
    R = A.index_select(0, selected_rows)

    U = torch.zeros(
        (C.shape[1], R.shape[0]), device=A.device, dtype=A.dtype
    )

    return C, U, R


class CURLoRAModule(nn.Module):
    def __init__(self, W, rank):
        super(CURLoRAModule, self).__init__()
        C, U, R = cur_decomposition(W, rank)
        self.register_buffer('C', C)
        self.register_buffer('R', R)
        self.U = nn.Parameter(U)  # U is trainable

    def forward(self, x):
        # Equivalent to x @ (C @ U @ R)^T, but avoids materializing dense W_approx.
        x = x.matmul(self.R.t())
        x = x.matmul(self.U.t())
        x = x.matmul(self.C.t())
        return x


class CURLoRALinear(nn.Module):
    def __init__(self, weight, bias=None, rank=256, alpha=1):
        super(CURLoRALinear, self).__init__()
        self.register_buffer("weight", weight.detach().clone())
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None
        self.rank = rank
        self.alpha = alpha

        # CURLoRA
        self.curlora_modules = CURLoRAModule(self.weight, self.rank)

    def forward(self, x):
        x_0 = x.matmul(self.weight.t())
        x_adapted = self.curlora_modules(x)
        x = x_0 + (self.alpha * x_adapted)
        if self.bias is not None:
            x += self.bias
        return x


def _effective_rank(in_features: int, out_features: int, rank: int) -> int:
    return max(1, min(int(rank), int(in_features), int(out_features)))


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


def inject_curlora(
    model,
    layer_indices,
    ffn_module_names,
    attn_module_names,
    rank,
    alpha: float | None = None,
    rank_overrides=None,
):
    for layer_index in layer_indices:
        layer = model.model.layers[layer_index]
        for name in ffn_module_names:
            module = getattr(layer.mlp, name)
            weight = module.weight.data.detach().clone()
            bias = module.bias.data.clone() if module.bias is not None else None
            module_key = f"mlp_{name}"
            rank_val = int(rank_overrides[module_key]) if rank_overrides and module_key in rank_overrides else int(rank)
            r_eff = _effective_rank(weight.size(1), weight.size(0), rank_val)
            alpha_eff = float(alpha) if alpha is not None else float(2 * r_eff)
            curlora = CURLoRALinear(weight, bias=bias, rank=r_eff, alpha=alpha_eff)
            setattr(layer.mlp, name, curlora)
        for name in attn_module_names:
            module = getattr(layer.self_attn, name)
            weight = module.weight.data.detach().clone()
            bias = module.bias.data.clone() if module.bias is not None else None
            module_key = f"attn_{name}"
            rank_val = int(rank_overrides[module_key]) if rank_overrides and module_key in rank_overrides else int(rank)
            r_eff = _effective_rank(weight.size(1), weight.size(0), rank_val)
            alpha_eff = float(alpha) if alpha is not None else float(2 * r_eff)
            curlora = CURLoRALinear(weight, bias=bias, rank=r_eff, alpha=alpha_eff)
            setattr(layer.self_attn, name, curlora)
    return model


def merge_curlora(model):
    for module in model.modules():
        if isinstance(module, CURLoRALinear):
            with torch.no_grad():
                delta = module.curlora_modules.C @ module.curlora_modules.U @ module.curlora_modules.R
                module.weight.add_(float(module.alpha) * delta)
                module.curlora_modules.U.zero_()
    return model


def strip_curlora(model):
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, CURLoRALinear):
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


def freeze_except_curlora_U(model):
    for name, param in model.named_parameters():
        param.requires_grad = ("curlora_modules.U" in name)
    return model
