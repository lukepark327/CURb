from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn

from .olora import collect_lora_factors


def init_inflora_state() -> dict:
    """
    DualGPM state (official InfLoRA) maintained separately for MHA vs FFN groups.

    Each group keeps:
      - feature_list: list[torch.Tensor] (U bases)
      - project_type: list[str] in {"remove","retain"}
      - feature_mat: list[torch.Tensor] projection matrices (U U^T), float32 on the training device
    """
    return {
        "attn": {"feature_list": [], "project_type": [], "feature_mat": []},
        "mlp": {"feature_list": [], "project_type": [], "feature_mat": []},
    }


def _iter_llama_group_reps(
    model,
    layer_indices,
    ffn_module_names,
    attn_module_names,
) -> Iterable[Tuple[str, str, nn.Module]]:
    """
    Yield (group_name, group_key, representative_module) where:
      - group_name in {"attn", "mlp"}
      - group_key in {"layers.{i}.self_attn", "layers.{i}.mlp"}
    """
    attn_rep = attn_module_names[0] if attn_module_names else None
    ffn_rep = ffn_module_names[0] if ffn_module_names else None

    for layer_index in layer_indices:
        layer = model.model.layers[layer_index]
        if attn_rep is not None:
            yield "attn", f"layers.{layer_index}.self_attn", getattr(layer.self_attn, attn_rep)
        if ffn_rep is not None:
            yield "mlp", f"layers.{layer_index}.mlp", getattr(layer.mlp, ffn_rep)


@torch.inference_mode()
def _collect_group_curr_matrices(
    *,
    model,
    dataloader,
    device,
    layer_indices,
    ffn_module_names,
    attn_module_names,
) -> Dict[str, torch.Tensor]:
    """
    Official InfLoRA collects a per-layer "cur_matrix" as the mean Gram:
        curr_matrix = mean( x^T x ) over (batch, tokens)
    Here we do the same (but name it curr_matrix to avoid confusion with CUR).

    Returns:
      dict[group_key -> curr_matrix_fp32], where group_key is
      "layers.{i}.self_attn" or "layers.{i}.mlp".
    """
    buffers: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    handles = []

    def _to_3d(x: torch.Tensor) -> torch.Tensor:
        # Expect (B, N, D). Make it robust for (B, D) or weird shapes.
        if x.dim() == 3:
            return x
        if x.dim() == 2:
            return x.unsqueeze(1)
        # Flatten everything except last dim into a pseudo-token axis.
        x = x.reshape(-1, x.shape[-1])
        return x.unsqueeze(1)

    def _make_hook(group_key: str):
        def _hook(_module, inputs, _output):
            if not inputs:
                return
            x = inputs[0]
            if x is None:
                return
            x = x.detach()
            x = _to_3d(x)
            b, n, d = x.shape

            # Official computes: sum_b (x_b^T x_b) for x in (B, N, D) via a batched bmm.
            # Here we compute the *exact same* Gram using a flattened matmul to avoid
            # materializing (B, D, D), which can OOM for large hidden sizes.
            x2 = x.reshape(-1, d)  # (B*N, D)
            if x2.dtype != torch.float32:
                x2 = x2.to(dtype=torch.float32)
            xtx = (x2.transpose(0, 1) @ x2)  # (D, D) on device

            prev_n = int(counts[group_key])
            denom = prev_n + int(b * n)
            if denom <= 0:
                return
            # Keep curr_matrix accumulation on device to avoid per-step CPU sync/transfer.
            # buffers stores the *mean* Gram: mean(x^T x) over (batch, tokens).
            buf = buffers[group_key]
            if prev_n > 0:
                buf.mul_(float(prev_n) / float(denom))
                buf.add_(xtx, alpha=(1.0 / float(denom)))
            else:
                buf.copy_(xtx / float(denom))
            counts[group_key] = denom

        return _hook

    # Init buffers + attach hooks.
    for _group_name, group_key, module in _iter_llama_group_reps(
        model, layer_indices, ffn_module_names, attn_module_names
    ):
        # Dim is module input features for Linear-like modules.
        dim = int(getattr(module, "in_features", 0))
        if dim <= 0:
            raise ValueError(f"Cannot infer in_features for group_key={group_key} (module={type(module).__name__}).")
        buffers[group_key] = torch.zeros((dim, dim), dtype=torch.float32, device=device)
        counts[group_key] = 0
        handles.append(module.register_forward_hook(_make_hook(group_key)))

    was_training = model.training
    use_cache_flag = getattr(model.config, "use_cache", None)
    model.eval()
    if use_cache_flag is not None:
        model.config.use_cache = False

    try:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        if use_cache_flag is not None:
            model.config.use_cache = use_cache_flag
        model.train(was_training)

    return buffers


def _svd_left_u(curr_matrix: torch.Tensor, full_matrices: bool) -> torch.Tensor:
    # Run SVD on the tensor's device (GPU if available) for speed.
    U, _S, _Vh = torch.linalg.svd(curr_matrix, full_matrices=bool(full_matrices))
    return U


def _project_curr_matrix(curr_matrix: torch.Tensor, proj_mat: torch.Tensor, project_type: str) -> torch.Tensor:
    # Mirror official method/inflora.py projection logic.
    if project_type == "remove":
        return curr_matrix - torch.mm(proj_mat, curr_matrix)
    if project_type == "retain":
        return torch.mm(proj_mat, curr_matrix)
    raise ValueError(f"Unknown project_type: {project_type}")


def _design_b_from_curr_matrix(
    *,
    curr_matrix: torch.Tensor,
    rank: int,
    task_idx: int,
    proj_mat: torch.Tensor | None,
    project_type: str | None,
) -> torch.Tensor:
    """
    Official InfLoRA sets B_t from the top-r left singular vectors of (projected) curr_matrix,
    and scales by 1/sqrt(3).
    """
    if int(rank) < 1:
        raise ValueError(f"rank must be >= 1 (got {rank}).")

    mat = curr_matrix
    if int(task_idx) > 0:
        if proj_mat is None or project_type is None:
            raise ValueError("task_idx>0 requires proj_mat and project_type (DualGPM state missing).")
        mat = _project_curr_matrix(mat, proj_mat, project_type)
        U = _svd_left_u(mat, full_matrices=False)
    else:
        U = _svd_left_u(mat, full_matrices=True)

    # B_t = U_r^T / sqrt(3)
    b = U[:, : int(rank)].T.contiguous() / float(math.sqrt(3.0))
    return b


def design_inflora_b_by_module(
    *,
    model,
    dataloader,
    device,
    layer_indices,
    ffn_module_names,
    attn_module_names,
    inflora_ranks: dict,
    inflora_state: dict,
    task_idx: int,
) -> Dict[str, torch.Tensor]:
    """
    Design B_t for each target Linear module key (PEFT LoRA's lora_A weight) following
    the official InfLoRA implementation:
      - collect per-layer curr_matrix = mean(x^T x)
      - for task>0, project curr_matrix using DualGPM's feature_mat with remove/retain
      - SVD and set B_t = U_r^T / sqrt(3)

    Returns:
      dict[module_key -> B_t] where module_key is like "layers.0.self_attn.q_proj".
      B_t is fp32 with shape (r, in_features) on the training device.
    """
    if inflora_state is None or not isinstance(inflora_state, dict):
        raise ValueError("inflora_state must be a dict.")
    if "attn" not in inflora_state or "mlp" not in inflora_state:
        raise ValueError("inflora_state must come from init_inflora_state().")

    curr_mats = _collect_group_curr_matrices(
        model=model,
        dataloader=dataloader,
        device=device,
        layer_indices=layer_indices,
        ffn_module_names=ffn_module_names,
        attn_module_names=attn_module_names,
    )

    b_by_module: Dict[str, torch.Tensor] = {}

    # Fixed ordering: idx corresponds to layer_indices position, matching DualGPM lists.
    for idx, layer_index in enumerate(layer_indices):
        if attn_module_names:
            group_key = f"layers.{layer_index}.self_attn"
            curr = curr_mats.get(group_key)
            if curr is None:
                raise ValueError(f"Missing curr_matrix for group {group_key}")
            mat = curr
            if int(task_idx) > 0:
                proj_list = inflora_state["attn"].get("feature_mat", [])
                type_list = inflora_state["attn"].get("project_type", [])
                if idx >= len(proj_list) or idx >= len(type_list):
                    raise ValueError("DualGPM state for attn missing; did you run update after previous task?")
                mat = _project_curr_matrix(mat, proj_list[idx], type_list[idx])
                U = _svd_left_u(mat, full_matrices=False)
            else:
                U = _svd_left_u(mat, full_matrices=True)

            # Same basis is shared across modules in the group (official).
            for name in attn_module_names:
                r = int(inflora_ranks[f"attn_{name}"])
                b_by_module[f"{group_key}.{name}"] = U[:, :r].T.contiguous() / float(math.sqrt(3.0))

        if ffn_module_names:
            group_key = f"layers.{layer_index}.mlp"
            curr = curr_mats.get(group_key)
            if curr is None:
                raise ValueError(f"Missing curr_matrix for group {group_key}")
            mat = curr
            if int(task_idx) > 0:
                proj_list = inflora_state["mlp"].get("feature_mat", [])
                type_list = inflora_state["mlp"].get("project_type", [])
                if idx >= len(proj_list) or idx >= len(type_list):
                    raise ValueError("DualGPM state for mlp missing; did you run update after previous task?")
                mat = _project_curr_matrix(mat, proj_list[idx], type_list[idx])
                U = _svd_left_u(mat, full_matrices=False)
            else:
                U = _svd_left_u(mat, full_matrices=True)

            for name in ffn_module_names:
                r = int(inflora_ranks[f"mlp_{name}"])
                b_by_module[f"{group_key}.{name}"] = U[:, :r].T.contiguous() / float(math.sqrt(3.0))

    return b_by_module


def _update_dualgpm_official(
    *,
    mat_list: list[torch.Tensor],
    feature_list: list[torch.Tensor],
    project_type: list[str],
    task_idx: int,
    total_sessions: int,
    lamb: float,
    lame: float,
) -> Tuple[list[torch.Tensor], list[str]]:
    """
    Copy of the official DualGPM update logic in methods/inflora.py (with minimal refactoring),
    implemented in torch so it can run on GPU.
    """
    if not mat_list:
        return list(feature_list), list(project_type)

    # Keep everything on the same device/dtype for GPU acceleration.
    dev = mat_list[0].device if torch.is_tensor(mat_list[0]) else torch.device("cpu")

    mats: list[torch.Tensor] = []
    for m in mat_list:
        if torch.is_tensor(m):
            mt = m
            if mt.device != dev:
                mt = mt.to(device=dev)
            if mt.dtype != torch.float32:
                mt = mt.to(dtype=torch.float32)
            mats.append(mt)
        else:
            mats.append(torch.tensor(m, device=dev, dtype=torch.float32))
    mat_list = mats

    feats: list[torch.Tensor] = []
    for f in feature_list:
        if torch.is_tensor(f):
            ft = f
            if ft.device != dev:
                ft = ft.to(device=dev)
            if ft.dtype != torch.float32:
                ft = ft.to(dtype=torch.float32)
            feats.append(ft)
        else:
            feats.append(torch.tensor(f, device=dev, dtype=torch.float32))
    feature_list = feats

    if int(total_sessions) <= 0:
        raise ValueError(f"total_sessions must be >= 1 (got {total_sessions}).")
    threshold = (float(lame) - float(lamb)) * float(task_idx) / float(total_sessions) + float(lamb)

    if len(feature_list) == 0:
        # After First Task
        for activation in mat_list:
            U, S, _Vh = torch.linalg.svd(activation, full_matrices=False)
            sval2 = S.square()
            sval_ratio = sval2 / sval2.sum()
            prefix = torch.cumsum(sval_ratio, dim=0)
            r = int(torch.searchsorted(prefix, torch.tensor(float(threshold), device=dev), right=False).item())
            r = max(int(r), 1)
            if r < (activation.shape[0] / 2):
                feature_list.append(U[:, 0:r])
                project_type.append("remove")
            else:
                feature_list.append(U[:, 0:r])
                project_type.append("retain")
    else:
        for i, activation in enumerate(mat_list):
            if project_type[i] == "remove":
                _U1, S1, _Vh1 = torch.linalg.svd(activation, full_matrices=False)
                sval_total = S1.square().sum()
                proj = feature_list[i] @ feature_list[i].transpose(0, 1)
                act_hat = activation - (proj @ activation)
                U, S, _Vh = torch.linalg.svd(act_hat, full_matrices=False)
                sval_hat = S.square().sum()
                sval_ratio = S.square() / sval_total
                accumulated_sval = (sval_total - sval_hat) / sval_total

                r = 0
                if float(accumulated_sval.item()) < float(threshold):
                    need = float(threshold) - float(accumulated_sval.item())
                    prefix = torch.cumsum(sval_ratio, dim=0)
                    idx = int(torch.searchsorted(prefix, torch.tensor(need, device=dev), right=False).item())
                    r = min(int(idx) + 1, int(prefix.shape[0]))
                if r == 0:
                    continue
                Ui = torch.hstack((feature_list[i], U[:, 0:r]))
                if Ui.shape[1] > Ui.shape[0]:
                    feature_list[i] = Ui[:, 0 : Ui.shape[0]]
                else:
                    feature_list[i] = Ui
            else:
                assert project_type[i] == "retain"
                _U1, S1, _Vh1 = torch.linalg.svd(activation, full_matrices=False)
                sval_total = S1.square().sum()
                proj = feature_list[i] @ feature_list[i].transpose(0, 1)
                act_hat = proj @ activation
                U, S, _Vh = torch.linalg.svd(act_hat, full_matrices=False)
                sval_hat = S.square().sum()
                sval_ratio = S.square() / sval_total
                accumulated_sval = sval_hat / sval_total

                r = 0
                if float(accumulated_sval.item()) >= (1 - float(threshold)):
                    need = float(accumulated_sval.item()) - (1 - float(threshold))
                    prefix = torch.cumsum(sval_ratio, dim=0)
                    idx = int(torch.searchsorted(prefix, torch.tensor(need, device=dev), right=True).item())
                    r = min(int(idx) + 1, int(prefix.shape[0]))
                if r == 0:
                    continue

                Ur = U[:, 0:r]
                act_feature = feature_list[i] - (Ur @ Ur.transpose(0, 1)) @ feature_list[i]
                Ui, _Si, _Vi = torch.linalg.svd(act_feature, full_matrices=True)
                feature_list[i] = Ui[:, : feature_list[i].shape[1] - r]

    # Gradient Constraints Summary adjustments (official)
    for i in range(len(feature_list)):
        if project_type[i] == "remove" and (feature_list[i].shape[1] > (feature_list[i].shape[0] / 2)):
            feature = feature_list[i]
            U, _S, _V = torch.linalg.svd(feature, full_matrices=True)
            new_feature = U[:, feature.shape[1] :]
            feature_list[i] = new_feature
            project_type[i] = "retain"
        elif project_type[i] == "retain":
            assert feature_list[i].shape[1] <= (feature_list[i].shape[0] / 2)

    return feature_list, project_type


def update_inflora_state_after_task(
    *,
    model,
    dataloader,
    device,
    layer_indices,
    ffn_module_names,
    attn_module_names,
    inflora_state: dict,
    task_idx: int,
    total_sessions: int,
    lamb: float,
    lame: float,
) -> None:
    """
    After training task t, official InfLoRA updates DualGPM using curr_matrix collected
    from the (trained) model on current-task data. We do the same for each group.
    """
    if inflora_state is None or not isinstance(inflora_state, dict):
        raise ValueError("inflora_state must be a dict.")
    if "attn" not in inflora_state or "mlp" not in inflora_state:
        raise ValueError("inflora_state must come from init_inflora_state().")

    curr_mats = _collect_group_curr_matrices(
        model=model,
        dataloader=dataloader,
        device=device,
        layer_indices=layer_indices,
        ffn_module_names=ffn_module_names,
        attn_module_names=attn_module_names,
    )

    # Build mat_list in a stable order matching layer_indices.
    if attn_module_names:
        attn_list = []
        for layer_index in layer_indices:
            key = f"layers.{layer_index}.self_attn"
            attn_list.append(curr_mats[key].detach())
        st = inflora_state["attn"]
        st["feature_list"], st["project_type"] = _update_dualgpm_official(
            mat_list=attn_list,
            feature_list=list(st.get("feature_list", [])),
            project_type=list(st.get("project_type", [])),
            task_idx=int(task_idx),
            total_sessions=int(total_sessions),
            lamb=float(lamb),
            lame=float(lame),
        )
        st["feature_mat"] = [
            (f @ f.transpose(0, 1)).to(dtype=torch.float32) for f in st["feature_list"]
        ]

    if ffn_module_names:
        mlp_list = []
        for layer_index in layer_indices:
            key = f"layers.{layer_index}.mlp"
            mlp_list.append(curr_mats[key].detach())
        st = inflora_state["mlp"]
        st["feature_list"], st["project_type"] = _update_dualgpm_official(
            mat_list=mlp_list,
            feature_list=list(st.get("feature_list", [])),
            project_type=list(st.get("project_type", [])),
            task_idx=int(task_idx),
            total_sessions=int(total_sessions),
            lamb=float(lamb),
            lame=float(lame),
        )
        st["feature_mat"] = [
            (f @ f.transpose(0, 1)).to(dtype=torch.float32) for f in st["feature_list"]
        ]


def apply_inflora_to_peft_model(model, b_by_module: Dict[str, torch.Tensor]) -> int:
    """
    After applying PEFT LoRA adapters, overwrite lora_A weights with designed B_t and freeze them.

    Returns: number of modules updated.
    """
    if not b_by_module:
        return 0

    lora_a, _lora_b = collect_lora_factors(model)
    updated = 0
    for key, param in lora_a.items():
        b = b_by_module.get(key)
        if b is None:
            continue
        with torch.no_grad():
            b_t = b.to(device=param.device, dtype=param.dtype)
            if b_t.shape != param.shape:
                out = param.detach().clone()
                out.zero_()
                take_r = min(int(out.shape[0]), int(b_t.shape[0]))
                take_d = min(int(out.shape[1]), int(b_t.shape[1]))
                out[:take_r, :take_d].copy_(b_t[:take_r, :take_d])
                param.copy_(out)
            else:
                param.copy_(b_t)
        # InfLoRA: B_t (LoRA's lora_A) must be frozen.
        param.requires_grad = False
        updated += 1
    return int(updated)
