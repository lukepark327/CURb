import os
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset

from cur_utils.cur_models import WandaWrappedModule
from cur_utils.cur import cur_deim_gpu


# NOTE: helper functions for optional CURb whitening live below.

class _TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str]):
        self.texts = list(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx]}


def _build_text_loader(tokenizer, batch_size, texts: list[str], max_length: int):
    dataset = _TextDataset(texts)

    def collate(batch):
        batch_texts = [b.get("text", "") for b in batch]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        return enc

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    return loader


@torch.no_grad()
def _relative_ridge_eps(G: torch.Tensor, ridge_ratio: float, ridge_abs: float) -> float:
    """
    Scale-aware ridge: eps = ridge_abs + ridge_ratio * trace(G) / r.
    Keeps behavior stable across layers/methods whose Gram matrices have different scales.
    """
    r = int(G.shape[0])
    if r <= 0:
        return float(ridge_abs)
    tr = torch.trace(G)
    tr_val = float(tr.item()) if torch.isfinite(tr) else 0.0
    if tr_val < 0.0:
        tr_val = 0.0
    return float(ridge_abs) + float(ridge_ratio) * (tr_val / float(r))


@torch.no_grad()
def _symmetrize(G: torch.Tensor) -> torch.Tensor:
    return 0.5 * (G + G.t())


@torch.no_grad()
def _inv_sqrt_from_eigh(G: torch.Tensor, eps_floor: float) -> torch.Tensor:
    """
    Returns G^{-1/2} for symmetric PSD G via eigen-decomposition, with eigenvalue flooring.
    """
    evals, evecs = torch.linalg.eigh(G)  # ascending
    evals = torch.clamp(evals, min=float(eps_floor))
    inv_sqrt = evals.rsqrt()
    return (evecs * inv_sqrt.unsqueeze(0)) @ evecs.t()


@torch.no_grad()
def _whiten_C_and_R_diag(
    C: torch.Tensor,
    R: torch.Tensor,
    d: torch.Tensor,
    ridge_ratio: float,
    ridge_abs: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Diagonal-second-moment whitening for CURb anchors.

    - Left:  C <- C @ (C^T C + eps I)^{-1/2}
             Implemented via Cholesky whitening (P = chol(G)^{-1}); eig fallback.
    - Right: R <- (R diag(d^2) R^T + eps I)^{-1/2} @ R
             Implemented via Cholesky whitening (P = chol(G)^{-1}); eig fallback.

    d is the per-input-feature RMS: sqrt(E[x^2]) (same statistic as cov_fast).
    """
    dev = C.device
    r = int(C.shape[1])
    if r <= 0:
        return C, R

    # r is small: do whitening in float64 for stability.
    C64 = C.to(device=dev, dtype=torch.float64)
    R64 = R.to(device=dev, dtype=torch.float64)
    d64 = d.to(device=dev, dtype=torch.float64)

    # --- Left whitening: C ---
    Gc = _symmetrize(C64.t() @ C64)
    eps_c = _relative_ridge_eps(Gc, ridge_ratio=ridge_ratio, ridge_abs=ridge_abs)
    Gc = Gc + eps_c * torch.eye(r, device=dev, dtype=torch.float64)
    try:
        Lc = torch.linalg.cholesky(Gc)  # lower
        # Cw * Lc = C  => Cw = C @ inv(Lc)
        Cw = torch.linalg.solve_triangular(Lc, C64, upper=False, left=False)
    except RuntimeError:
        Pc = _inv_sqrt_from_eigh(Gc, eps_floor=eps_c)
        Cw = C64 @ Pc

    # --- Right whitening: R with diag second moment ---
    d2 = d64.square().view(1, -1)
    Gr = _symmetrize((R64 * d2) @ R64.t())
    eps_r = _relative_ridge_eps(Gr, ridge_ratio=ridge_ratio, ridge_abs=ridge_abs)
    Gr = Gr + eps_r * torch.eye(r, device=dev, dtype=torch.float64)
    try:
        Lr = torch.linalg.cholesky(Gr)  # lower
        # Lr * Rw = R  => Rw = inv(Lr) @ R
        Rw = torch.linalg.solve_triangular(Lr, R64, upper=False, left=True)
    except RuntimeError:
        Pr = _inv_sqrt_from_eigh(Gr, eps_floor=eps_r)
        Rw = Pr @ R64

    # Store in the original dtype (baseline parity).
    return Cw.to(dtype=C.dtype), Rw.to(dtype=R.dtype)


def _remove_cols(ds_like):
    if getattr(ds_like, "features", None) is not None:
        return list(ds_like.features.keys())
    if getattr(ds_like, "column_names", None) is not None:
        return list(ds_like.column_names)
    return []


def _select_first_n(ds_like, n):
    if hasattr(ds_like, "select"):
        take_n = min(n, len(ds_like))
        return ds_like.select(range(take_n))
    return ds_like.take(n)


def _build_c4_loader(tokenizer, batch_size, num_sequences, max_length, dataset_category):
    data_amount = int(num_sequences)
    if data_amount <= 0:
        return DataLoader([], batch_size=batch_size)
    if dataset_category:
        train_stream = load_dataset(
            "allenai/c4",
            dataset_category,
            split="train",
            streaming=True,
        )
    else:
        train_stream = load_dataset(
            "allenai/c4",
            split="train",
            streaming=True,
        )

    dataset = {"train": train_stream.take(data_amount)}

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            return_special_tokens_mask=True,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

    tokenized_dataset = {
        "train": dataset["train"].map(
            tokenize_function,
            batched=True,
            remove_columns=_remove_cols(dataset["train"]),
        )
    }

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, return_tensors="pt"
    )

    loader = DataLoader(
        tokenized_dataset["train"],
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    return loader


def _set_current_mask_for_wrapped_modules(wrapped_modules, attention_mask):
    for wm in wrapped_modules.values():
        if hasattr(wm, "set_current_mask"):
            wm.set_current_mask(attention_mask)


def _collect_activation_stats(model, tokenizer, device, layer_indices, ffn_module_names, attn_module_names,
                              batch_size, num_sequences, max_length, dataset_category,
                              calib_texts: list[str] | None = None):
    wrapped_modules = {}
    for layer_index in layer_indices:
        layer = model.model.layers[layer_index]
        for name in ffn_module_names:
            module = getattr(layer.mlp, name)
            wrapped = WandaWrappedModule(
                module,
                acc_device=device,
                acc_dtype=torch.float64,
            )
            wrapped.register_hook()
            key = f"layer_{layer_index}_mlp_{name}"
            wrapped_modules[key] = wrapped
        for name in attn_module_names:
            module = getattr(layer.self_attn, name)
            wrapped = WandaWrappedModule(
                module,
                acc_device=device,
                acc_dtype=torch.float64,
            )
            wrapped.register_hook()
            key = f"layer_{layer_index}_self_attn_{name}"
            wrapped_modules[key] = wrapped

    num_sequences = int(num_sequences)
    if num_sequences < 0:
        num_sequences = 0
    if calib_texts is not None:
        # Use replay-based texts first; fill remaining with C4 to match the same
        # sequence count as the original C4-only calibration.
        texts = list(calib_texts)[:num_sequences]
        fill_c4 = max(0, num_sequences - len(texts))
        loaders = []
        if texts:
            loaders.append(_build_text_loader(
                tokenizer,
                batch_size=batch_size,
                texts=texts,
                max_length=max_length,
            ))
        if fill_c4:
            loaders.append(_build_c4_loader(
                tokenizer,
                batch_size=batch_size,
                num_sequences=fill_c4,
                max_length=max_length,
                dataset_category=dataset_category,
            ))
    else:
        loaders = [_build_c4_loader(
            tokenizer,
            batch_size=batch_size,
            num_sequences=num_sequences,
            max_length=max_length,
            dataset_category=dataset_category,
        )]

    model.eval()
    use_cache_flag = model.config.use_cache
    model.config.use_cache = False
    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                inputs = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                _set_current_mask_for_wrapped_modules(wrapped_modules, attention_mask)
                _ = model(inputs, attention_mask=attention_mask)

    model.config.use_cache = use_cache_flag

    for wrapped in wrapped_modules.values():
        wrapped.remove_hook()

    return wrapped_modules


def _resolve_module_rank(default_rank, rank_overrides, basis_key, module_key):
    rank_val = int(default_rank)
    if rank_overrides:
        if basis_key in rank_overrides and rank_overrides[basis_key] is not None:
            rank_val = int(rank_overrides[basis_key])
        elif module_key in rank_overrides and rank_overrides[module_key] is not None:
            rank_val = int(rank_overrides[module_key])
    return max(1, int(rank_val))


def load_or_build_curb_basis(
    model,
    tokenizer,
    device,
    layer_indices,
    ffn_module_names,
    attn_module_names,
    rank,
    mode,
    deim_importance_order="high",
    update_whiten: str = "none",
    whiten_ridge_ratio: float = 1e-4,
    whiten_ridge_abs: float = 1e-12,
    rank_overrides=None,
    cache_path=None,
    calib_steps=256,
    batch_size=1,
    max_length=4096,
    dataset_category="en",
    calib_texts: list[str] | None = None,
):
    if isinstance(dataset_category, str) and dataset_category.strip().lower() in ("", "none", "null"):
        dataset_category = None
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu")

    wrapped_modules = None
    need_activation_stats = (mode in ("cov_fast", "hybrid")) or (update_whiten == "diag")
    if need_activation_stats:
        num_sequences = int(batch_size) * int(calib_steps)
        wrapped_modules = _collect_activation_stats(
            model,
            tokenizer,
            device,
            layer_indices,
            ffn_module_names,
            attn_module_names,
            batch_size=batch_size,
            num_sequences=num_sequences,
            max_length=max_length,
            dataset_category=dataset_category,
            calib_texts=calib_texts,
        )

    basis = {
        "metadata": {
            "rank": int(rank),
            "mode": mode,
            "deim_importance_order": deim_importance_order,
            "rank_overrides": rank_overrides or {},
        },
        "modules": {},
    }

    for layer_index in layer_indices:
        layer = model.model.layers[layer_index]
        for name in ffn_module_names:
            key = f"layer_{layer_index}_mlp_{name}"
            module_key = f"mlp_{name}"
            module = getattr(layer.mlp, name)
            weight = module.weight.data
            aux = None
            if mode in ("cov_fast", "hybrid"):
                aux = wrapped_modules[key].get_activation_norm()
                if aux is None:
                    raise RuntimeError(f"Missing activation stats for {key}")
                S = weight * aux.view(1, -1).to(device=weight.device, dtype=weight.dtype)
            elif mode == "weight":
                S = weight
            else:
                raise ValueError(f"Unknown CURb basis mode: {mode}")
            requested_rank = _resolve_module_rank(
                default_rank=rank,
                rank_overrides=rank_overrides,
                basis_key=key,
                module_key=module_key,
            )
            r_eff = min(int(requested_rank), int(weight.shape[0]), int(weight.shape[1]))
            if mode == "hybrid":
                row_idx, _ = cur_deim_gpu(
                    S.float(),
                    r_eff,
                    use_lowrank=True,
                    importance_order=deim_importance_order,
                )
                _, col_idx = cur_deim_gpu(
                    weight.float(),
                    r_eff,
                    use_lowrank=True,
                    importance_order=deim_importance_order,
                )
            else:
                row_idx, col_idx = cur_deim_gpu(
                    S.float(),
                    r_eff,
                    use_lowrank=True,
                    importance_order=deim_importance_order,
                )
            C = weight[:, col_idx]
            R = weight[row_idx, :]
            if update_whiten == "diag":
                if aux is None:
                    aux = wrapped_modules[key].get_activation_norm()
                if aux is None:
                    raise RuntimeError(f"Missing activation stats for whitening: {key}")
                C, R = _whiten_C_and_R_diag(
                    C=C,
                    R=R,
                    d=aux,
                    ridge_ratio=whiten_ridge_ratio,
                    ridge_abs=whiten_ridge_abs,
                )
            C = C.detach().cpu()
            R = R.detach().cpu()
            basis["modules"][key] = {
                "C": C,
                "R": R,
                "row_indices": row_idx,
                "col_indices": col_idx,
                "requested_rank": int(requested_rank),
                "effective_rank": int(r_eff),
            }
        for name in attn_module_names:
            key = f"layer_{layer_index}_self_attn_{name}"
            module_key = f"attn_{name}"
            module = getattr(layer.self_attn, name)
            weight = module.weight.data
            aux = None
            if mode in ("cov_fast", "hybrid"):
                aux = wrapped_modules[key].get_activation_norm()
                if aux is None:
                    raise RuntimeError(f"Missing activation stats for {key}")
                S = weight * aux.view(1, -1).to(device=weight.device, dtype=weight.dtype)
            elif mode == "weight":
                S = weight
            else:
                raise ValueError(f"Unknown CURb basis mode: {mode}")
            requested_rank = _resolve_module_rank(
                default_rank=rank,
                rank_overrides=rank_overrides,
                basis_key=key,
                module_key=module_key,
            )
            r_eff = min(int(requested_rank), int(weight.shape[0]), int(weight.shape[1]))
            if mode == "hybrid":
                row_idx, _ = cur_deim_gpu(
                    S.float(),
                    r_eff,
                    use_lowrank=True,
                    importance_order=deim_importance_order,
                )
                _, col_idx = cur_deim_gpu(
                    weight.float(),
                    r_eff,
                    use_lowrank=True,
                    importance_order=deim_importance_order,
                )
            else:
                row_idx, col_idx = cur_deim_gpu(
                    S.float(),
                    r_eff,
                    use_lowrank=True,
                    importance_order=deim_importance_order,
                )
            C = weight[:, col_idx]
            R = weight[row_idx, :]
            if update_whiten == "diag":
                if aux is None:
                    aux = wrapped_modules[key].get_activation_norm()
                if aux is None:
                    raise RuntimeError(f"Missing activation stats for whitening: {key}")
                C, R = _whiten_C_and_R_diag(
                    C=C,
                    R=R,
                    d=aux,
                    ridge_ratio=whiten_ridge_ratio,
                    ridge_abs=whiten_ridge_abs,
                )
            C = C.detach().cpu()
            R = R.detach().cpu()
            basis["modules"][key] = {
                "C": C,
                "R": R,
                "row_indices": row_idx,
                "col_indices": col_idx,
                "requested_rank": int(requested_rank),
                "effective_rank": int(r_eff),
            }

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(basis, cache_path)
        meta_path = cache_path + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(basis.get("metadata", {}), f, ensure_ascii=True, indent=2)

    return basis


def _move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(v, device) for v in obj)
    return obj


@torch.no_grad()
def _collect_activation_stats_named_modules(
    model,
    device,
    module_names: list[str],
    calib_loader,
    forward_fn=None,
    mask_key: str | None = "attention_mask",
    mask_fn=None,
    max_batches: int | None = None,
):
    """
    Generic activation-stat collector for cov_fast/hybrid CURb basis selection.

    - Wraps the specified named Linear modules with WandaWrappedModule hooks.
    - Runs model forward passes over calib_loader to accumulate sqrt(E[x^2]) per input feature.

    Args:
      model: nn.Module.
      device: torch.device or string (also used as WandaWrappedModule.acc_device).
      module_names: qualified names from model.named_modules().
      calib_loader: iterable of batches (dict/tuple/...).
      forward_fn: optional callable(model, batch) -> outputs. If None, defaults to calling
        model(**batch) when batch is a dict, else model(*batch) or model(batch).
      mask_key: optional key to extract attention_mask from dict batches (LLM-style).
      mask_fn: optional callable(batch) -> attention_mask tensor or None.
      max_batches: optional limit on number of batches consumed from calib_loader.
    """
    module_map = dict(model.named_modules())
    wrapped_modules = {}
    for name in module_names:
        module = module_map.get(name)
        if module is None:
            raise KeyError(f"Module not found in model: {name}")
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear for activation stats, got {type(module)} for: {name}")
        wrapped = WandaWrappedModule(
            module,
            acc_device=device,
            acc_dtype=torch.float64,
        )
        wrapped.register_hook()
        wrapped_modules[name] = wrapped

    model.eval()
    for step, batch in enumerate(calib_loader):
        if max_batches is not None and step >= int(max_batches):
            break

        attention_mask = None
        if mask_fn is not None:
            attention_mask = mask_fn(batch)
        elif mask_key and isinstance(batch, dict) and (mask_key in batch):
            attention_mask = batch.get(mask_key)

        if attention_mask is not None:
            for wm in wrapped_modules.values():
                wm.set_current_mask(attention_mask)

        if forward_fn is not None:
            forward_fn(model, batch)
            continue

        moved = _move_to_device(batch, device)
        if isinstance(moved, dict):
            _ = model(**moved)
        elif isinstance(moved, (list, tuple)):
            _ = model(*moved)
        else:
            _ = model(moved)

    for wrapped in wrapped_modules.values():
        wrapped.remove_hook()

    return wrapped_modules


@torch.no_grad()
def load_or_build_curb_basis_named_modules(
    model,
    device,
    module_names: list[str],
    rank: int,
    mode: str,
    deim_importance_order: str = "high",
    update_whiten: str = "none",
    whiten_ridge_ratio: float = 1e-4,
    whiten_ridge_abs: float = 1e-12,
    rank_overrides: dict | None = None,
    cache_path: str | None = None,
    calib_loader=None,
    forward_fn=None,
    mask_key: str | None = "attention_mask",
    mask_fn=None,
    max_calib_batches: int | None = None,
):
    """
    CURb basis builder for arbitrary named Linear modules.

    This is the ViT-friendly counterpart to load_or_build_curb_basis(), which assumes
    a LLaMA-like model structure (model.model.layers[*].{mlp,self_attn}.*).

    Expected module_names are from model.named_modules(), e.g. for HuggingFace ViT:
      'vit.encoder.layer.0.attention.attention.query'

    Args:
      model: nn.Module.
      device: torch.device or string (used for activation accumulation + batch moves).
      module_names: list[str], qualified module names to build basis for.
      rank: base CURb rank (effective rank is min(rank, in_features, out_features)).
      mode: 'cov_fast' | 'weight' | 'hybrid'.
      calib_loader/forward_fn: required when mode in ('cov_fast','hybrid') or update_whiten == 'diag'.
    """
    if isinstance(mode, str):
        mode = mode.strip()
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu")

    need_activation_stats = (mode in ("cov_fast", "hybrid")) or (update_whiten == "diag")
    wrapped_modules = None
    if need_activation_stats:
        if calib_loader is None:
            raise ValueError(
                "calib_loader is required for CURb cov_fast/hybrid basis or diag whitening "
                "(load_or_build_curb_basis_named_modules)."
            )
        wrapped_modules = _collect_activation_stats_named_modules(
            model=model,
            device=device,
            module_names=module_names,
            calib_loader=calib_loader,
            forward_fn=forward_fn,
            mask_key=mask_key,
            mask_fn=mask_fn,
            max_batches=max_calib_batches,
        )

    basis = {
        "metadata": {
            "rank": int(rank),
            "mode": mode,
            "deim_importance_order": deim_importance_order,
            "rank_overrides": rank_overrides or {},
            "module_names": list(module_names),
        },
        "modules": {},
    }

    module_map = dict(model.named_modules())
    for name in module_names:
        module = module_map.get(name)
        if module is None:
            raise KeyError(f"Module not found in model: {name}")
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(module)} for: {name}")
        weight = module.weight.data

        requested_rank = int(rank)
        if rank_overrides and (name in rank_overrides) and (rank_overrides[name] is not None):
            requested_rank = int(rank_overrides[name])
        r_eff = min(int(requested_rank), int(weight.shape[0]), int(weight.shape[1]))
        r_eff = max(1, int(r_eff))

        aux = None
        if mode in ("cov_fast", "hybrid"):
            aux = wrapped_modules[name].get_activation_norm() if wrapped_modules is not None else None
            if aux is None:
                raise RuntimeError(f"Missing activation stats for {name}")
            S = weight * aux.view(1, -1).to(device=weight.device, dtype=weight.dtype)
        elif mode == "weight":
            S = weight
        else:
            raise ValueError(f"Unknown CURb basis mode: {mode}")

        if mode == "hybrid":
            row_idx, _ = cur_deim_gpu(
                S.float(),
                r_eff,
                use_lowrank=True,
                importance_order=deim_importance_order,
            )
            _, col_idx = cur_deim_gpu(
                weight.float(),
                r_eff,
                use_lowrank=True,
                importance_order=deim_importance_order,
            )
        else:
            row_idx, col_idx = cur_deim_gpu(
                S.float(),
                r_eff,
                use_lowrank=True,
                importance_order=deim_importance_order,
            )

        C = weight[:, col_idx]
        R = weight[row_idx, :]

        if update_whiten == "diag":
            if aux is None:
                aux = wrapped_modules[name].get_activation_norm() if wrapped_modules is not None else None
            if aux is None:
                raise RuntimeError(f"Missing activation stats for whitening: {name}")
            C, R = _whiten_C_and_R_diag(
                C=C,
                R=R,
                d=aux,
                ridge_ratio=whiten_ridge_ratio,
                ridge_abs=whiten_ridge_abs,
            )

        # Spectral norms for alpha normalisation (cheap — one SVD per small matrix).
        sigma_max_C = float(torch.linalg.matrix_norm(C.float(), ord=2))
        sigma_max_R = float(torch.linalg.matrix_norm(R.float(), ord=2))

        basis["modules"][name] = {
            "C": C.detach().cpu(),
            "R": R.detach().cpu(),
            "row_indices": row_idx,
            "col_indices": col_idx,
            "requested_rank": int(requested_rank),
            "effective_rank": int(r_eff),
            "sigma_max_C": sigma_max_C,
            "sigma_max_R": sigma_max_R,
        }

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(basis, cache_path)
        meta_path = cache_path + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(basis.get("metadata", {}), f, ensure_ascii=True, indent=2)

    return basis
