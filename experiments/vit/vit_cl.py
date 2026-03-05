#!/usr/bin/env python
"""
ViT class-IL continual learning benchmark runner.

Design goals:
- Reuse existing CURb/baseline implementations as much as possible.
- Keep fair-comparison protocol similar to CURb/llm/tc_cl.py.
- Single-head class-IL with configurable train/eval class masking.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTForImageClassification, get_scheduler, set_seed

try:
    from torchvision import datasets as tv_datasets
    from torchvision import transforms as tv_transforms

    HAS_TORCHVISION = True
except Exception:
    tv_datasets = None
    tv_transforms = None
    HAS_TORCHVISION = False

try:
    from datasets import load_dataset as hf_load_dataset

    HAS_HF_DATASETS = True
except Exception:
    hf_load_dataset = None
    HAS_HF_DATASETS = False


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_mora_path = os.path.join(REPO_ROOT, "MoRA", "peft-mora")
if os.path.isdir(_mora_path):
    sys.path.append(_mora_path)
    sys.path.append(os.path.join(_mora_path, "src"))

from curb import CURbLinear, inject_curb_named_modules, merge_curb, strip_curb, freeze_except_curb_U, update_curb_alpha  # noqa: E402
from curb_basis import load_or_build_curb_basis_named_modules  # noqa: E402
from curlora import CURLoRALinear, merge_curlora, strip_curlora, freeze_except_curlora_U  # noqa: E402
from bilora import (  # noqa: E402
    BiLoRALinear,
    merge_bilora,
    strip_bilora,
    freeze_except_bilora_theta,
)
from olora import (  # noqa: E402
    collect_lora_factors,
    build_olora_prev_device_map,
    append_olora_subspace,
    compute_olora_losses,
)
from inflora import (  # noqa: E402
    init_inflora_state,
    apply_inflora_to_peft_model,
)
from methods.baselines.inflora import _design_b_from_curr_matrix, _update_dualgpm_official  # noqa: E402
from lorac import (  # noqa: E402
    init_lorac_state,
    LoRACLinear,
    merge_lorac,
    strip_lorac,
    freeze_except_lorac,
    update_lorac_ipc_importance,
    lorac_ortho_loss,
)


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROUND_SEED_STRIDE = 10_000


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Logger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.fp = open(log_path, "a", encoding="utf-8")

    def log(self, msg: str):
        text = f"[{_ts()}] {msg}"
        print(text, flush=True)
        self.fp.write(text + "\n")
        self.fp.flush()

    def close(self):
        try:
            self.fp.close()
        except Exception:
            pass


def _seed_from_parts(base_seed: int, *parts) -> int:
    payload = "|".join(str(p) for p in parts)
    h = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) + int(base_seed)) % (2**32)


def _round_seed(base_seed: int, round_idx: int) -> int:
    return int(base_seed) + int(ROUND_SEED_STRIDE) * int(round_idx)


def _stable_hash_int(text: str) -> int:
    return int(hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8], 16)


def _resolve_torch_dtype(dtype_name: str):
    name = str(dtype_name).lower()
    if name in ("fp32", "float32", "torch.float32"):
        return torch.float32
    if name in ("bf16", "bfloat16", "torch.bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "torch.float16"):
        return torch.float16
    raise ValueError(f"Unsupported model_dtype: {dtype_name}")


def _load_mora_peft():
    mora_src = os.path.join(REPO_ROOT, "MoRA", "peft-mora", "src")
    if not os.path.isdir(mora_src):
        raise ImportError("MoRA source not found at MoRA/peft-mora/src.")

    for key in list(sys.modules.keys()):
        if key == "peft" or key.startswith("peft."):
            del sys.modules[key]
    if mora_src in sys.path:
        sys.path.remove(mora_src)
    sys.path.insert(0, mora_src)

    import importlib

    mora_peft = importlib.import_module("peft")
    if not hasattr(mora_peft, "LoraConfig"):
        raise ImportError("MoRA peft package missing LoraConfig.")
    return mora_peft


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


class IndexedVisionDataset(Dataset):
    def __init__(self, base_ds, indices: Sequence[int], transform):
        self.base_ds = base_ds
        self.indices = [int(i) for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = int(self.indices[idx])
        image, label = self.base_ds[sample_idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)


class HFImageClassificationDataset(Dataset):
    """Minimal wrapper around Hugging Face datasets image-classification split."""

    def __init__(self, ds, image_key: str, label_key: str, class_names: list[str] | None = None):
        self.ds = ds
        self.image_key = str(image_key)
        self.label_key = str(label_key)
        self.targets = [int(x) for x in ds[self.label_key]]
        if class_names is not None:
            self.classes = list(class_names)
        else:
            n_cls = int(max(self.targets) + 1) if self.targets else 0
            self.classes = [str(i) for i in range(n_cls)]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[int(idx)]
        image = row[self.image_key]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image, int(row[self.label_key])


class SimpleImageFolder(Dataset):
    """torchvision-free fallback for ImageFolder-style directory datasets."""

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str):
        self.root = str(root)
        p = Path(self.root)
        if not p.is_dir():
            raise ValueError(f"ImageFolder root not found: {self.root}")

        self.classes = sorted([d.name for d in p.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: list[tuple[str, int]] = []

        for cls_name in self.classes:
            cls_dir = p / cls_name
            label = int(self.class_to_idx[cls_name])
            for fp in sorted(cls_dir.rglob("*")):
                if not fp.is_file():
                    continue
                if fp.suffix.lower() not in self.IMG_EXTS:
                    continue
                self.samples.append((str(fp), label))

        self.targets = [int(y) for _, y in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[int(idx)]
        with Image.open(path) as img:
            image = img.convert("RGB")
        return image, int(label)


def _extract_targets(ds) -> list[int]:
    if hasattr(ds, "targets") and ds.targets is not None:
        return [int(x) for x in ds.targets]
    if hasattr(ds, "samples") and ds.samples is not None:
        return [int(y) for _, y in ds.samples]
    raise ValueError(f"Unsupported dataset type for target extraction: {type(ds)}")


def _build_transforms(image_processor, image_size: int):
    if HAS_TORCHVISION and tv_transforms is not None:
        mean = getattr(image_processor, "image_mean", None) or [0.5, 0.5, 0.5]
        std = getattr(image_processor, "image_std", None) or [0.5, 0.5, 0.5]
        resize_short = int(round(float(image_size) * 256.0 / 224.0))

        train_tf = tv_transforms.Compose(
            [
                tv_transforms.RandomResizedCrop(image_size),
                tv_transforms.RandomHorizontalFlip(),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=mean, std=std),
            ]
        )
        eval_tf = tv_transforms.Compose(
            [
                tv_transforms.Resize(resize_short),
                tv_transforms.CenterCrop(image_size),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=mean, std=std),
            ]
        )
        return train_tf, eval_tf

    class _ProcessorTransform:
        def __init__(self, proc, train_mode: bool):
            self.proc = proc
            self.train_mode = bool(train_mode)

        def __call__(self, image):
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.array(image))
            if image.mode != "RGB":
                image = image.convert("RGB")

            if self.train_mode and random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            enc = self.proc(images=image, return_tensors="pt")
            return enc["pixel_values"][0]

    return _ProcessorTransform(image_processor, True), _ProcessorTransform(image_processor, False)


@dataclass(frozen=True)
class DatasetBundle:
    train_ds: object
    eval_ds: object
    class_names: list[str]
    num_classes: int


def _load_dataset_bundle(dataset: str, data_root: str, train_dir: str | None, val_dir: str | None) -> DatasetBundle:
    key = str(dataset).strip().lower()

    if key == "cifar100":
        if HAS_TORCHVISION and tv_datasets is not None:
            train_ds = tv_datasets.CIFAR100(root=data_root, train=True, download=True)
            eval_ds = tv_datasets.CIFAR100(root=data_root, train=False, download=True)
            class_names = [str(x) for x in getattr(train_ds, "classes", list(range(100)))]
            return DatasetBundle(train_ds=train_ds, eval_ds=eval_ds, class_names=class_names, num_classes=len(class_names))

        if not HAS_HF_DATASETS or hf_load_dataset is None:
            raise ImportError(
                "cifar100 requires torchvision or datasets package. "
                "Install one of them in this environment."
            )
        tr = hf_load_dataset("cifar100", split="train")
        te = hf_load_dataset("cifar100", split="test")
        class_names = [str(x) for x in tr.features["fine_label"].names]
        train_ds = HFImageClassificationDataset(tr, image_key="img", label_key="fine_label", class_names=class_names)
        eval_ds = HFImageClassificationDataset(te, image_key="img", label_key="fine_label", class_names=class_names)
        return DatasetBundle(train_ds=train_ds, eval_ds=eval_ds, class_names=class_names, num_classes=len(class_names))

    if train_dir is None:
        train_dir = os.path.join(data_root, key, "train")
    if val_dir is None:
        # common alternatives
        cand_val = os.path.join(data_root, key, "val")
        cand_test = os.path.join(data_root, key, "test")
        val_dir = cand_val if os.path.isdir(cand_val) else cand_test

    if not train_dir or not os.path.isdir(train_dir):
        raise ValueError(f"Missing train_dir for dataset={dataset}: {train_dir}")
    if not val_dir or not os.path.isdir(val_dir):
        raise ValueError(f"Missing val_dir/test_dir for dataset={dataset}: {val_dir}")

    if HAS_TORCHVISION and tv_datasets is not None:
        train_ds = tv_datasets.ImageFolder(root=train_dir)
        eval_ds = tv_datasets.ImageFolder(root=val_dir)
    else:
        train_ds = SimpleImageFolder(root=train_dir)
        eval_ds = SimpleImageFolder(root=val_dir)

    if train_ds.classes != eval_ds.classes:
        raise ValueError(
            "train/val class sets differ for ImageFolder dataset. "
            f"train_classes={len(train_ds.classes)} val_classes={len(eval_ds.classes)}"
        )

    class_names = [str(x) for x in train_ds.classes]
    return DatasetBundle(train_ds=train_ds, eval_ds=eval_ds, class_names=class_names, num_classes=len(class_names))


def _resolve_external_calib_train_dir(data_root: str, calib_train_dir: str | None) -> str:
    if calib_train_dir is not None and str(calib_train_dir).strip():
        resolved = os.path.abspath(os.path.expanduser(str(calib_train_dir).strip()))
        if not os.path.isdir(resolved):
            raise ValueError(f"External calib train_dir not found: {resolved}")
        return resolved

    candidates = [
        os.path.join(data_root, "imagenet1k", "train"),
        os.path.join(data_root, "imagenet", "train"),
        os.path.join(data_root, "ilsvrc2012", "train"),
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand

    raise ValueError(
        "External calibration source requires an ImageFolder train directory. "
        "Provide --calib_train_dir (e.g., /path/to/imagenet1k/train)."
    )


def _load_external_calib_dataset(data_root: str, calib_train_dir: str | None):
    train_dir = _resolve_external_calib_train_dir(data_root=data_root, calib_train_dir=calib_train_dir)
    if HAS_TORCHVISION and tv_datasets is not None:
        ds = tv_datasets.ImageFolder(root=train_dir)
    else:
        ds = SimpleImageFolder(root=train_dir)
    return ds, train_dir


def _curb_alpha_schedule_factor(
    global_step: int,
    total_steps: int,
    schedule: str,
    min_ratio: float,
    warmup_ratio: float,
) -> float:
    """Compute alpha scale factor in [min_ratio, 1.0] based on global progress."""
    if schedule == "constant" or total_steps <= 0:
        return 1.0
    progress = min(float(global_step) / float(total_steps), 1.0)
    warmup_frac = max(0.0, min(float(warmup_ratio), 1.0))
    if progress < warmup_frac:
        # Warmup: min_ratio → 1.0
        return min_ratio + (1.0 - min_ratio) * (progress / warmup_frac)
    # Decay phase
    decay_progress = (progress - warmup_frac) / max(1.0 - warmup_frac, 1e-12)
    if schedule == "cosine":
        import math
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    elif schedule == "linear":
        return 1.0 - (1.0 - min_ratio) * decay_progress
    return 1.0


def _auto_num_tasks(dataset: str) -> int:
    key = str(dataset).strip().lower()
    if key == "imagenet_r":
        return 10
    if key == "domainnet":
        return 5
    if key == "cifar100":
        return 10
    return 10


def _default_epochs(dataset: str) -> int:
    key = str(dataset).strip().lower()
    if key == "imagenet_r":
        return 50
    if key == "domainnet":
        return 5
    if key == "cifar100":
        return 20
    return 20


def _split_classes(class_order: list[int], num_tasks: int) -> list[list[int]]:
    if int(num_tasks) < 1:
        raise ValueError(f"num_tasks must be >= 1 (got {num_tasks})")
    n = len(class_order)
    base = n // int(num_tasks)
    rem = n % int(num_tasks)
    out = []
    st = 0
    for t in range(int(num_tasks)):
        sz = base + (1 if t < rem else 0)
        out.append([int(x) for x in class_order[st : st + sz]])
        st += sz
    return out


def _sample_or_all(rng: random.Random, arr: list[int], limit: int) -> list[int]:
    if int(limit) < 0 or int(limit) >= len(arr):
        out = list(arr)
        rng.shuffle(out)
        return out
    if int(limit) == 0:
        return []
    out = rng.sample(arr, int(limit))
    rng.shuffle(out)
    return out


def _make_splits(
    *,
    num_classes: int,
    train_targets: list[int],
    eval_targets: list[int],
    num_tasks: int,
    total_round: int,
    seed: int,
    train_samples_per_task: int,
    eval_samples_per_task: int,
    out_path: str,
    logger: Logger | None = None,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    class_to_train_idx: dict[int, list[int]] = {}
    class_to_eval_idx: dict[int, list[int]] = {}
    for i, y in enumerate(train_targets):
        class_to_train_idx.setdefault(int(y), []).append(int(i))
    for i, y in enumerate(eval_targets):
        class_to_eval_idx.setdefault(int(y), []).append(int(i))

    all_classes = list(range(int(num_classes)))
    rounds = []
    for r in range(int(total_round)):
        round_num = r + 1
        round_seed = _round_seed(seed, r)
        rng = random.Random(_seed_from_parts(round_seed, "class_order"))
        class_order = list(all_classes)
        rng.shuffle(class_order)
        task_classes = _split_classes(class_order, int(num_tasks))

        task_splits = []
        for t, cls_list in enumerate(task_classes, start=1):
            train_idx = []
            eval_idx = []
            for c in cls_list:
                train_idx.extend(class_to_train_idx.get(int(c), []))
                eval_idx.extend(class_to_eval_idx.get(int(c), []))

            rng_t = random.Random(_seed_from_parts(round_seed, "task", t))
            train_idx = _sample_or_all(rng_t, train_idx, int(train_samples_per_task))
            eval_idx = _sample_or_all(rng_t, eval_idx, int(eval_samples_per_task))

            task_splits.append(
                {
                    "task_id": int(t),
                    "classes": [int(x) for x in cls_list],
                    "train_indices": [int(x) for x in train_idx],
                    "eval_indices": [int(x) for x in eval_idx],
                }
            )

        rounds.append(
            {
                "round": int(round_num),
                "seed": int(round_seed),
                "class_order": [int(x) for x in class_order],
                "task_splits": task_splits,
            }
        )

    obj = {
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "base_seed": int(seed),
        "round_seed_stride": int(ROUND_SEED_STRIDE),
        "num_classes": int(num_classes),
        "num_tasks": int(num_tasks),
        "total_round": int(total_round),
        "train_samples_per_task": int(train_samples_per_task),
        "eval_samples_per_task": int(eval_samples_per_task),
        "rounds": rounds,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2)
    if logger is not None:
        logger.log(f"[splits] wrote {out_path}")
    return obj


def _load_splits(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_round_splits(splits_obj, round_num: int) -> dict:
    rounds = splits_obj.get("rounds", [])
    for r in rounds:
        if int(r.get("round", -1)) == int(round_num):
            return r
    raise KeyError(f"Round not found in splits: {round_num}")


def _build_vit_module_lists(model) -> tuple[list[str], list[str], list[str], list[str]]:
    n_layers = int(getattr(model.config, "num_hidden_layers", 12))
    q = [f"vit.encoder.layer.{i}.attention.attention.query" for i in range(n_layers)]
    k = [f"vit.encoder.layer.{i}.attention.attention.key" for i in range(n_layers)]
    fc1 = [f"vit.encoder.layer.{i}.intermediate.dense" for i in range(n_layers)]
    all_mods = list(q) + list(k) + list(fc1)
    return q, k, fc1, all_mods


def _build_rank_map(
    q_modules: list[str],
    k_modules: list[str],
    fc1_modules: list[str],
    *,
    rank_q: int,
    rank_k: int,
    rank_fc1: int,
) -> dict[str, int]:
    out = {}
    for n in q_modules:
        out[n] = int(rank_q)
    for n in k_modules:
        out[n] = int(rank_k)
    for n in fc1_modules:
        out[n] = int(rank_fc1)
    return out


def _effective_rank(in_features: int, out_features: int, rank: int) -> int:
    return max(1, min(int(rank), int(in_features), int(out_features)))


def _compute_inflora_ranks_match_trainable(model, lora_rank_map: dict[str, int]) -> dict[str, int]:
    module_map = dict(model.named_modules())
    out = {}
    for name, r_lora in lora_rank_map.items():
        module = module_map.get(name)
        if module is None or not isinstance(module, nn.Linear):
            continue
        in_f = int(module.in_features)
        out_f = int(module.out_features)
        target_trainable = int(r_lora) * (in_f + out_f)
        r = int(target_trainable / max(1, out_f) + 0.5)
        out[name] = int(max(1, min(r, min(in_f, out_f))))
    return out


def _enable_classifier_trainable(model):
    for name, param in model.named_parameters():
        if name.endswith("classifier.weight") or name.endswith("classifier.bias") or ".classifier." in name:
            param.requires_grad = True


def _inject_curlora_named_modules(model, rank_map: dict[str, int], alpha: float | None):
    module_map = dict(model.named_modules())
    for name, rank in rank_map.items():
        module = module_map.get(name)
        if module is None or not isinstance(module, nn.Linear):
            raise KeyError(f"CURLoRA target not found or not Linear: {name}")
        weight = module.weight.detach().clone()
        bias = module.bias.data.clone() if module.bias is not None else None
        r_eff = _effective_rank(weight.size(1), weight.size(0), int(rank))
        alpha_eff = float(alpha) if alpha is not None else float(2 * r_eff)
        wrapped = CURLoRALinear(weight, bias=bias, rank=int(r_eff), alpha=alpha_eff)
        parent, child = _get_parent_and_child(model, name)
        _set_child_module(parent, child, wrapped)
    return model


def _inject_bilora_named_modules(
    model,
    rank_map: dict[str, int],
    *,
    k: int | None,
    alpha: float | None,
    seed: int,
    task_idx: int,
    chunk_size: int,
    freq_chunk_size: int,
):
    module_map = dict(model.named_modules())
    base_seed = int(seed) + int(task_idx) * 10
    for name, rank in rank_map.items():
        module = module_map.get(name)
        if module is None or not isinstance(module, nn.Linear):
            raise KeyError(f"BiLoRA target not found or not Linear: {name}")
        weight = module.weight.detach()
        bias = module.bias.detach() if module.bias is not None else None
        mod_seed = base_seed + (_stable_hash_int(name) % 1000003)
        wrapped = BiLoRALinear(
            weight,
            bias,
            rank=int(rank),
            k=k,
            alpha=alpha,
            seed=int(mod_seed),
            chunk_size=int(chunk_size),
            freq_chunk_size=int(freq_chunk_size),
        )
        parent, child = _get_parent_and_child(model, name)
        _set_child_module(parent, child, wrapped)
    return model


def _inject_lorac_named_modules(
    model,
    *,
    rank_map: dict[str, int],
    lorac_state: dict,
    task_idx: int,
    lora_alpha: float | None,
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
    module_map = dict(model.named_modules())

    for name, rank in rank_map.items():
        module = module_map.get(name)
        if module is None or not isinstance(module, nn.Linear):
            raise KeyError(f"LoRAC target not found or not Linear: {name}")

        state_ref = modules_state.get(name)
        if state_ref is None:
            state_ref = {
                "rank": int(rank),
                "A_bank": [],
                "B_bank": [],
                "omega": torch.ones((pool_size,), dtype=torch.float32, device="cpu"),
                "omega_snap": torch.zeros((pool_size,), dtype=torch.float32, device="cpu"),
                "mask_prev": 1,
                "ipc_exp_avg_ipt": None,
                "ipc_exp_avg_unc": None,
            }
            modules_state[name] = state_ref
        if int(state_ref.get("rank", rank)) != int(rank):
            raise ValueError(f"LoRAC rank mismatch for {name}: state={state_ref.get('rank')} requested={rank}")

        wrapped = LoRACLinear(
            module.weight.detach(),
            module.bias.detach() if module.bias is not None else None,
            module_key=name,
            state_ref=state_ref,
            pool_size=pool_size,
            task_idx=int(task_idx),
            rank=int(rank),
            lora_alpha=lora_alpha,
            ipc_enabled=bool(ipc_enabled),
            ipc_beta1=float(ipc_beta1),
            ipc_beta2=float(ipc_beta2),
            ipc_threshold=float(ipc_threshold),
            ipc_new_mask=bool(ipc_new_mask),
        )
        parent, child = _get_parent_and_child(model, name)
        _set_child_module(parent, child, wrapped)

    return model


def _build_allowed_mask(num_classes: int, allowed_classes: Sequence[int], device: torch.device) -> torch.Tensor:
    mask = torch.zeros((int(num_classes),), dtype=torch.bool, device=device)
    idx = torch.tensor(list(allowed_classes), dtype=torch.long, device=device)
    if idx.numel() > 0:
        mask[idx] = True
    return mask


def _masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    allowed_classes: Sequence[int],
    num_classes: int,
) -> torch.Tensor:
    device = logits.device
    allowed_mask = _build_allowed_mask(int(num_classes), allowed_classes, device=device)
    # Use finite low value to avoid NaNs in reduced precision.
    low = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~allowed_mask.unsqueeze(0), low)
    return F.cross_entropy(masked_logits, targets)


@torch.no_grad()
def evaluate_accuracy(
    model,
    loader,
    *,
    allowed_classes: Sequence[int],
    num_classes: int,
    device: torch.device,
):
    model.eval()
    total = 0
    correct = 0

    allowed_mask = _build_allowed_mask(int(num_classes), allowed_classes, device=device)
    low = None

    pbar = tqdm(loader, desc="eval", leave=False, file=sys.stdout)
    for pixel_values, labels in pbar:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        out = model(pixel_values=pixel_values)
        logits = out.logits if hasattr(out, "logits") else out["logits"]

        if low is None:
            low = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~allowed_mask.unsqueeze(0), low)

        pred = torch.argmax(logits, dim=1)
        total += int(labels.numel())
        correct += int((pred == labels).sum().item())
        acc = (100.0 * correct / total) if total else 0.0
        pbar.set_postfix(acc=f"{acc:.2f}", n=total)

    pbar.close()
    acc = (100.0 * correct / total) if total else 0.0
    return float(acc), int(total)


def _merge_and_strip(model, method: str):
    if method in ("lora", "mora", "olora", "inflora"):
        merged_model = model
        for _ in range(8):
            if not hasattr(merged_model, "merge_and_unload"):
                break
            next_model = merged_model.merge_and_unload()
            if next_model is merged_model:
                break
            merged_model = next_model
        for _ in range(4):
            if not hasattr(merged_model, "get_base_model"):
                break
            if not hasattr(merged_model, "peft_config"):
                break
            base_model = merged_model.get_base_model()
            if base_model is None or base_model is merged_model:
                break
            merged_model = base_model
        if hasattr(merged_model, "peft_config"):
            try:
                delattr(merged_model, "peft_config")
            except Exception:
                merged_model.peft_config = None
        if hasattr(merged_model, "active_adapter"):
            try:
                delattr(merged_model, "active_adapter")
            except Exception:
                merged_model.active_adapter = None
        return merged_model

    if method == "curb":
        merge_curb(model)
        return strip_curb(model)
    if method == "curlora":
        merge_curlora(model)
        return strip_curlora(model)
    if method == "bilora":
        merge_bilora(model)
        return strip_bilora(model)
    if method in ("lorac", "lorac_ipc"):
        merge_lorac(model)
        return strip_lorac(model)
    return model


def _collect_group_curr_matrices_vit(
    *,
    model,
    dataloader,
    device,
    layer_indices: list[int],
    rep_q_name: str,
    rep_fc1_name: str,
):
    """
    Generic InfLoRA-like current matrix collector for ViT.

    Returns dict keys:
      - attn::{layer_idx}
      - mlp::{layer_idx}
    """
    module_map = dict(model.named_modules())
    buffers: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    handles = []

    def _to_3d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x
        if x.dim() == 2:
            return x.unsqueeze(1)
        x = x.reshape(-1, x.shape[-1])
        return x.unsqueeze(1)

    def _make_hook(group_key: str):
        def _hook(_module, inputs, _output):
            if not inputs:
                return
            x = inputs[0]
            if x is None:
                return
            x = _to_3d(x.detach())
            b, n, d = x.shape
            x2 = x.reshape(-1, d)
            if x2.dtype != torch.float32:
                x2 = x2.to(dtype=torch.float32)
            xtx = x2.transpose(0, 1) @ x2

            prev_n = int(counts[group_key])
            denom = prev_n + int(b * n)
            if denom <= 0:
                return
            buf = buffers[group_key]
            if prev_n > 0:
                buf.mul_(float(prev_n) / float(denom))
                buf.add_(xtx, alpha=(1.0 / float(denom)))
            else:
                buf.copy_(xtx / float(denom))
            counts[group_key] = int(denom)

        return _hook

    for layer_idx in layer_indices:
        q_full = f"vit.encoder.layer.{layer_idx}.attention.attention.{rep_q_name}"
        f_full = f"vit.encoder.layer.{layer_idx}.intermediate.{rep_fc1_name}"

        q_mod = module_map.get(q_full)
        f_mod = module_map.get(f_full)
        if q_mod is None or not isinstance(q_mod, nn.Linear):
            raise KeyError(f"InfLoRA attn representative module missing: {q_full}")
        if f_mod is None or not isinstance(f_mod, nn.Linear):
            raise KeyError(f"InfLoRA mlp representative module missing: {f_full}")

        k_attn = f"attn::{layer_idx}"
        k_mlp = f"mlp::{layer_idx}"
        buffers[k_attn] = torch.zeros((int(q_mod.in_features), int(q_mod.in_features)), dtype=torch.float32, device=device)
        buffers[k_mlp] = torch.zeros((int(f_mod.in_features), int(f_mod.in_features)), dtype=torch.float32, device=device)
        counts[k_attn] = 0
        counts[k_mlp] = 0

        handles.append(q_mod.register_forward_hook(_make_hook(k_attn)))
        handles.append(f_mod.register_forward_hook(_make_hook(k_mlp)))

    was_training = model.training
    model.eval()
    try:
        for pixel_values, _labels in dataloader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            model(pixel_values=pixel_values)
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        model.train(was_training)

    return buffers


def _map_target_to_lora_key(lora_keys: list[str], target_modules: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for tgt in target_modules:
        matches = [k for k in lora_keys if k.endswith(tgt)]
        if not matches:
            # fallback: include target as substring
            matches = [k for k in lora_keys if tgt in k]
        if len(matches) != 1:
            raise ValueError(f"Cannot map target module to unique LoRA key: target={tgt} matches={matches[:4]}")
        mapping[tgt] = matches[0]
    return mapping


def _design_inflora_b_by_module_vit(
    *,
    model,
    dataloader,
    device,
    layer_indices: list[int],
    q_modules: list[str],
    k_modules: list[str],
    fc1_modules: list[str],
    inflora_ranks: dict[str, int],
    inflora_state: dict,
    task_idx: int,
) -> dict[str, torch.Tensor]:
    curr_mats = _collect_group_curr_matrices_vit(
        model=model,
        dataloader=dataloader,
        device=device,
        layer_indices=layer_indices,
        rep_q_name="query",
        rep_fc1_name="dense",
    )

    out: dict[str, torch.Tensor] = {}

    # Build fast lookup by layer index.
    q_set = set(q_modules)
    k_set = set(k_modules)
    f_set = set(fc1_modules)

    for layer_idx in layer_indices:
        attn_group = f"attn::{layer_idx}"
        mlp_group = f"mlp::{layer_idx}"

        # Attn projections (query/key) share the same projected curr_matrix by layer.
        if (task_idx > 0) and (inflora_state["attn"].get("feature_mat")):
            idx = layer_indices.index(layer_idx)
            proj_list = inflora_state["attn"].get("feature_mat", [])
            type_list = inflora_state["attn"].get("project_type", [])
            if idx >= len(proj_list) or idx >= len(type_list):
                raise ValueError("InfLoRA attn DualGPM state missing for current layer.")
            proj_mat = proj_list[idx].to(device=device, dtype=torch.float32)
            project_type = str(type_list[idx])
        else:
            proj_mat = None
            project_type = None

        for mod in (f"vit.encoder.layer.{layer_idx}.attention.attention.query", f"vit.encoder.layer.{layer_idx}.attention.attention.key"):
            if mod not in q_set and mod not in k_set:
                continue
            r = int(inflora_ranks[mod])
            b = _design_b_from_curr_matrix(
                curr_matrix=curr_mats[attn_group],
                rank=r,
                task_idx=int(task_idx),
                proj_mat=proj_mat,
                project_type=project_type,
            )
            out[mod] = b

        # MLP fc1
        if (task_idx > 0) and (inflora_state["mlp"].get("feature_mat")):
            idx = layer_indices.index(layer_idx)
            proj_list = inflora_state["mlp"].get("feature_mat", [])
            type_list = inflora_state["mlp"].get("project_type", [])
            if idx >= len(proj_list) or idx >= len(type_list):
                raise ValueError("InfLoRA mlp DualGPM state missing for current layer.")
            proj_mat = proj_list[idx].to(device=device, dtype=torch.float32)
            project_type = str(type_list[idx])
        else:
            proj_mat = None
            project_type = None

        mod = f"vit.encoder.layer.{layer_idx}.intermediate.dense"
        if mod in f_set:
            r = int(inflora_ranks[mod])
            b = _design_b_from_curr_matrix(
                curr_matrix=curr_mats[mlp_group],
                rank=r,
                task_idx=int(task_idx),
                proj_mat=proj_mat,
                project_type=project_type,
            )
            out[mod] = b

    return out


def _remap_inflora_b_to_lora_keys(model, b_by_target: dict[str, torch.Tensor], target_modules: list[str]) -> dict[str, torch.Tensor]:
    if not b_by_target:
        return {}
    lora_a, _ = collect_lora_factors(model)
    key_map = _map_target_to_lora_key(list(lora_a.keys()), list(target_modules))
    out: dict[str, torch.Tensor] = {}
    for tgt in target_modules:
        if tgt in b_by_target:
            out[key_map[tgt]] = b_by_target[tgt]
    return out


def _update_inflora_state_after_task_vit(
    *,
    model,
    dataloader,
    device,
    layer_indices: list[int],
    inflora_state: dict,
    task_idx: int,
    total_sessions: int,
    lamb: float,
    lame: float,
):
    curr_mats = _collect_group_curr_matrices_vit(
        model=model,
        dataloader=dataloader,
        device=device,
        layer_indices=layer_indices,
        rep_q_name="query",
        rep_fc1_name="dense",
    )

    attn_list = [curr_mats[f"attn::{i}"] for i in layer_indices]
    mlp_list = [curr_mats[f"mlp::{i}"] for i in layer_indices]

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
    st["feature_mat"] = [(f @ f.transpose(0, 1)).to(dtype=torch.float32) for f in st["feature_list"]]

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
    st["feature_mat"] = [(f @ f.transpose(0, 1)).to(dtype=torch.float32) for f in st["feature_list"]]


def _build_calib_loader_from_indices(
    base_ds,
    indices: list[int],
    transform,
    *,
    batch_size: int,
    max_samples: int,
    seed: int,
    num_workers: int,
):
    rng = random.Random(int(seed))
    pool = list(indices)
    rng.shuffle(pool)
    if int(max_samples) > 0:
        pool = pool[: int(max_samples)]
    if len(pool) < 1:
        raise ValueError("Calibration sample pool is empty.")
    ds = IndexedVisionDataset(base_ds=base_ds, indices=pool, transform=transform)
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
    )


def _build_calib_loader_from_parts(
    parts,
    *,
    batch_size: int,
    num_workers: int,
):
    ds_parts = []
    for base_ds, indices, transform in parts:
        if not indices:
            continue
        ds_parts.append(IndexedVisionDataset(base_ds=base_ds, indices=list(indices), transform=transform))
    if len(ds_parts) < 1:
        raise ValueError("Calibration sample pool is empty.")
    if len(ds_parts) == 1:
        calib_ds = ds_parts[0]
    else:
        calib_ds = ConcatDataset(ds_parts)
    return DataLoader(
        calib_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
    )


##############################################################################
# CURb diagnostics (--diag_curb)
##############################################################################

@torch.no_grad()
def _diag_curb_snapshot_pretrained(model, module_names):
    """Save ||W||_F for each target module before any CURb injection (call once per round)."""
    snap = {}
    for qname in module_names:
        parts = qname.split(".")
        mod = model
        for p in parts:
            mod = getattr(mod, p) if not p.isdigit() else mod[int(p)]
        if hasattr(mod, "weight"):
            snap[qname] = mod.weight.detach().clone()
    return snap


@torch.no_grad()
def _diag_curb_before_merge(model, logger):
    """Collect per-module stats from CURbLinear BEFORE merge."""
    rows = []
    for name, mod in model.named_modules():
        if not isinstance(mod, CURbLinear):
            continue
        U = mod.U.data
        C = mod.C
        R = mod.R
        alpha = mod.alpha
        u_fro = float(torch.norm(U, p="fro"))
        delta = C @ U @ R
        delta_fro = float(torch.norm(delta, p="fro"))
        alpha_delta_fro = float(abs(alpha)) * delta_fro
        c_fro = float(torch.norm(C, p="fro"))
        r_fro = float(torch.norm(R, p="fro"))
        w_fro = float(torch.norm(mod.weight, p="fro"))
        # λ_max(C^TC) and λ_max(R R^T) — relevant to Theorem 5
        CtC = C.T @ C
        lam_CtC = float(torch.linalg.eigvalsh(CtC)[-1])
        RRt = R @ R.T
        lam_RRt = float(torch.linalg.eigvalsh(RRt)[-1])
        rows.append({
            "module": name,
            "U_fro": u_fro,
            "delta_CUR_fro": delta_fro,
            "alpha_delta_fro": alpha_delta_fro,
            "C_fro": c_fro,
            "R_fro": r_fro,
            "W_fro": w_fro,
            "lam_max_CtC": lam_CtC,
            "lam_max_RRt": lam_RRt,
            "alpha": alpha,
            "r": int(C.size(1)),
        })
        logger.log(
            f"[diag_curb] {name}: ||U||_F={u_fro:.6f} ||αCUR||_F={alpha_delta_fro:.6f} "
            f"||C||_F={c_fro:.4f} ||R||_F={r_fro:.4f} "
            f"λ(C^TC)={lam_CtC:.6f} λ(RR^T)={lam_RRt:.6f} ||W||_F={w_fro:.4f}"
        )
    return rows


@torch.no_grad()
def _diag_curb_after_merge(model, pretrained_snap, module_names, logger):
    """Compute ||W_current - W_pretrained||_F per module AFTER merge+strip."""
    rows = []
    for qname in module_names:
        if qname not in pretrained_snap:
            continue
        parts = qname.split(".")
        mod = model
        for p in parts:
            mod = getattr(mod, p) if not p.isdigit() else mod[int(p)]
        if not hasattr(mod, "weight"):
            continue
        w_cur = mod.weight.data
        w_pre = pretrained_snap[qname].to(device=w_cur.device, dtype=w_cur.dtype)
        drift = float(torch.norm(w_cur - w_pre, p="fro"))
        w_pre_fro = float(torch.norm(w_pre, p="fro"))
        rows.append({
            "module": qname,
            "W_drift_fro": drift,
            "W_pretrained_fro": w_pre_fro,
            "drift_ratio": drift / max(w_pre_fro, 1e-12),
        })
        logger.log(
            f"[diag_curb_drift] {qname}: ||W-W0||_F={drift:.6f} "
            f"||W0||_F={w_pre_fro:.4f} ratio={drift / max(w_pre_fro, 1e-12):.6f}"
        )
    return rows


def _diag_curb_write_csv(rows_list, csv_path):
    """Write accumulated diagnostic rows to CSV."""
    if not rows_list:
        return
    fieldnames = list(rows_list[0].keys())
    for r in rows_list[1:]:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_list)


##############################################################################


def train_on_task(
    *,
    model,
    method_ctx,
    args,
    device,
    train_ds_obj: IndexedVisionDataset,
    calib_loader,
    current_classes: list[int],
    seen_classes: list[int],
    loader_seed: int,
    diag_logger=None,
):
    method = method_ctx["method"]
    train_model = model

    # Method-specific inject.
    if method in ("lora", "mora", "olora", "inflora"):
        q_modules = method_ctx["q_modules"]
        k_modules = method_ctx["k_modules"]
        fc1_modules = method_ctx["fc1_modules"]
        lora_rank_map = method_ctx["lora_rank_map"]

        if method == "mora":
            mora_peft = _load_mora_peft()
            config_cls = mora_peft.LoraConfig
            task_type = getattr(mora_peft.TaskType, "FEATURE_EXTRACTION", TaskType.FEATURE_EXTRACTION)
            _get_peft = mora_peft.get_peft_model
            config_kwargs = {"use_mora": True, "mora_type": 6}
        else:
            config_cls = LoraConfig
            task_type = TaskType.FEATURE_EXTRACTION
            _get_peft = get_peft_model
            config_kwargs = {}

        active_rank_map = dict(lora_rank_map)
        b_by_module = None

        if method == "inflora":
            active_rank_map = dict(method_ctx["inflora_rank_map"])
            b_by_module = _design_inflora_b_by_module_vit(
                model=train_model,
                dataloader=calib_loader,
                device=device,
                layer_indices=method_ctx["layer_indices"],
                q_modules=q_modules,
                k_modules=k_modules,
                fc1_modules=fc1_modules,
                inflora_ranks=active_rank_map,
                inflora_state=method_ctx["inflora_state"],
                task_idx=int(method_ctx.get("inflora_task_idx", 0)),
            )

        def _mk_cfg(target_modules: list[str], rank_lookup: dict[str, int], alpha_override: float | None = None):
            r = int(rank_lookup[target_modules[0]])
            alpha = float(alpha_override) if alpha_override is not None else float(args.lora_alpha)
            return config_cls(
                r=int(r),
                lora_alpha=float(alpha),
                lora_dropout=float(args.lora_dropout),
                bias="none",
                task_type=task_type,
                target_modules=list(target_modules),
                modules_to_save=["classifier"],
                **config_kwargs,
            )

        if method == "inflora":
            # scaling=alpha/r=1
            cfg_q = _mk_cfg(q_modules, active_rank_map, alpha_override=float(active_rank_map[q_modules[0]]))
            cfg_k = _mk_cfg(k_modules, active_rank_map, alpha_override=float(active_rank_map[k_modules[0]]))
            cfg_f = _mk_cfg(fc1_modules, active_rank_map, alpha_override=float(active_rank_map[fc1_modules[0]]))
        else:
            cfg_q = _mk_cfg(q_modules, active_rank_map)
            cfg_k = _mk_cfg(k_modules, active_rank_map)
            cfg_f = _mk_cfg(fc1_modules, active_rank_map)

        train_model = _get_peft(train_model, cfg_q)
        train_model = _get_peft(train_model, cfg_k)
        train_model = _get_peft(train_model, cfg_f)
        train_model.to(device)

        if method == "inflora":
            b_by_lora_key = _remap_inflora_b_to_lora_keys(
                train_model,
                b_by_target=(b_by_module or {}),
                target_modules=list(q_modules) + list(k_modules) + list(fc1_modules),
            )
            apply_inflora_to_peft_model(train_model, b_by_lora_key)

    elif method == "curb":
        # --curb_basis_freeze: reuse the basis from task-1 for all subsequent tasks
        frozen_basis = method_ctx.get("_frozen_curb_basis")
        if frozen_basis is not None and getattr(args, "curb_basis_freeze", False):
            basis = frozen_basis
        else:
            need_calib = (args.curb_basis_mode in ("cov_fast", "hybrid")) or (args.curb_update_whiten == "diag")
            basis = load_or_build_curb_basis_named_modules(
                model=train_model,
                device=device,
                module_names=method_ctx["all_modules"],
                rank=int(args.curb_rank),
                mode=str(args.curb_basis_mode),
                deim_importance_order=str(args.curb_deim_importance_order),
                update_whiten=str(args.curb_update_whiten),
                whiten_ridge_ratio=float(args.curb_whiten_ridge_ratio),
                whiten_ridge_abs=float(args.curb_whiten_ridge_abs),
                rank_overrides=method_ctx.get("curb_rank_map"),
                calib_loader=(calib_loader if need_calib else None),
                forward_fn=(lambda m, b: m(pixel_values=b[0].to(device, non_blocking=True))) if need_calib else None,
                mask_key=None,
                max_calib_batches=int(args.curb_calib_steps) if need_calib else None,
            )
            if getattr(args, "curb_basis_freeze", False) and frozen_basis is None:
                method_ctx["_frozen_curb_basis"] = basis
        inject_curb_named_modules(
            train_model,
            basis=basis,
            module_names=method_ctx["all_modules"],
            alpha=float(args.curb_alpha),
            strict=True,
            alpha_spectral_norm=getattr(args, "curb_alpha_spectral_norm", False),
        )
        freeze_except_curb_U(train_model)
        _enable_classifier_trainable(train_model)

    elif method == "curlora":
        _inject_curlora_named_modules(
            train_model,
            rank_map=method_ctx["curb_rank_map"],
            alpha=float(args.lora_alpha),
        )
        freeze_except_curlora_U(train_model)
        _enable_classifier_trainable(train_model)

    elif method == "bilora":
        _inject_bilora_named_modules(
            train_model,
            rank_map=method_ctx["curb_rank_map"],
            k=(int(args.bilora_k) if args.bilora_k is not None else None),
            alpha=(float(args.bilora_alpha) if args.bilora_alpha is not None else None),
            seed=int(args.bilora_seed),
            task_idx=int(method_ctx.get("bilora_task_idx", 0)),
            chunk_size=int(args.bilora_chunk_size),
            freq_chunk_size=int(args.bilora_freq_chunk_size),
        )
        freeze_except_bilora_theta(train_model)
        _enable_classifier_trainable(train_model)

    elif method in ("lorac", "lorac_ipc"):
        _inject_lorac_named_modules(
            train_model,
            rank_map=method_ctx["lora_rank_map"],
            lorac_state=method_ctx["lorac_state"],
            task_idx=int(method_ctx.get("lorac_task_idx", 0)),
            lora_alpha=float(args.lora_alpha),
            ipc_enabled=(method == "lorac_ipc"),
            ipc_beta1=float(args.lorac_ipc_beta1),
            ipc_beta2=float(args.lorac_ipc_beta2),
            ipc_threshold=float(args.lorac_ipc_threshold),
            ipc_new_mask=bool(args.lorac_ipc_new_mask),
        )
        freeze_except_lorac(train_model)
        _enable_classifier_trainable(train_model)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Safety: always keep classifier trainable for single-head classification.
    _enable_classifier_trainable(train_model)

    gen = torch.Generator()
    gen.manual_seed(int(loader_seed))
    train_loader = DataLoader(
        train_ds_obj,
        batch_size=int(args.train_batch_size),
        shuffle=True,
        generator=gen,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )

    train_model.train()

    trainable_params = [p for p in train_model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"No trainable params after method injection: method={method}")

    if method in ("lorac", "lorac_ipc"):
        base_params = []
        omega_params = []
        for n, p in train_model.named_parameters():
            if not p.requires_grad:
                continue
            if n.endswith(".omega"):
                omega_params.append(p)
            else:
                base_params.append(p)
        param_groups = []
        if base_params:
            param_groups.append({"params": base_params, "lr": float(args.learning_rate)})
        if omega_params:
            param_groups.append({"params": omega_params, "lr": float(args.learning_rate) * float(args.lorac_omega_lr_scale)})
        optimizer = torch.optim.Adam(
            param_groups,
            betas=(float(args.adam_beta1), float(args.adam_beta2)),
            eps=float(args.adam_eps),
            weight_decay=float(args.weight_decay),
        )
    elif method == "curb" and getattr(args, "curb_u_weight_decay", 0.0) > 0:
        u_params = []
        other_params = []
        for n, p in train_model.named_parameters():
            if not p.requires_grad:
                continue
            if ".U" in n:
                u_params.append(p)
            else:
                other_params.append(p)
        param_groups = []
        if u_params:
            param_groups.append({"params": u_params, "lr": float(args.learning_rate),
                                 "weight_decay": float(args.curb_u_weight_decay)})
        if other_params:
            param_groups.append({"params": other_params, "lr": float(args.learning_rate),
                                 "weight_decay": float(args.weight_decay)})
        optimizer = torch.optim.Adam(
            param_groups,
            betas=(float(args.adam_beta1), float(args.adam_beta2)),
            eps=float(args.adam_eps),
        )
    else:
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=float(args.learning_rate),
            betas=(float(args.adam_beta1), float(args.adam_beta2)),
            eps=float(args.adam_eps),
            weight_decay=float(args.weight_decay),
        )

    total_batches = max(1, len(train_loader))
    total_steps_target = int(total_batches) * int(args.epochs)
    if int(args.max_train_steps) > 0:
        total_steps_target = min(total_steps_target, int(args.max_train_steps))

    # Estimate total steps for alpha scheduling
    if method == "curb" and "_curb_global_step" in method_ctx:
        if getattr(args, "curb_alpha_per_task", False):
            # Per-task cycling: schedule spans one task only, reset each task
            method_ctx["_curb_total_global_steps"] = total_steps_target
            method_ctx["_curb_global_step"] = 0
            update_curb_alpha(train_model, 1.0)
        elif "_curb_total_global_steps" not in method_ctx:
            # Global schedule: spans all tasks (set once on first task)
            method_ctx["_curb_total_global_steps"] = total_steps_target * int(args.num_tasks)

    total_optimizer_steps = max(1, math.ceil(total_steps_target / max(1, int(args.grad_accum_steps))))
    warmup_steps = int(total_optimizer_steps * max(0.0, float(args.warmup_ratio)))
    scheduler = get_scheduler(
        name=str(args.lr_scheduler_type),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    olora_a = {}
    olora_b = {}
    olora_prev_a = {}
    if method == "olora":
        olora_a, olora_b = collect_lora_factors(train_model)
        olora_dtype = next(iter(olora_a.values())).dtype if olora_a else torch.float32
        olora_prev_a = build_olora_prev_device_map(method_ctx, device, olora_dtype)

    allowed_for_train = list(current_classes) if args.train_loss_mask == "current_task" else list(seen_classes)

    total_loss = 0.0
    total_steps = 0
    skipped_nonfinite = 0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=total_steps_target, desc="train", leave=False, file=sys.stdout)

    for _epoch in range(int(args.epochs)):
        for pixel_values, labels in train_loader:
            if total_steps >= total_steps_target:
                break

            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            out = train_model(pixel_values=pixel_values)
            logits = out.logits if hasattr(out, "logits") else out["logits"]
            if method in ("curlora", "curb") and (not torch.isfinite(logits).all()):
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                if skipped_nonfinite <= 5:
                    print(
                        f"[warn] skip batch due to non-finite logits "
                        f"(method={method} step={total_steps+1} skipped={skipped_nonfinite})",
                        flush=True,
                    )
                continue
            loss = _masked_cross_entropy(
                logits,
                labels,
                allowed_classes=allowed_for_train,
                num_classes=int(method_ctx["num_classes"]),
            )
            if method in ("curlora", "curb") and (not torch.isfinite(loss)):
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                if skipped_nonfinite <= 5:
                    print(
                        f"[warn] skip batch due to non-finite loss "
                        f"(method={method} step={total_steps+1} skipped={skipped_nonfinite})",
                        flush=True,
                    )
                continue

            if method == "olora":
                orth_loss, l2_loss = compute_olora_losses(loss, olora_a, olora_b, olora_prev_a)
                loss = loss + float(args.olora_lambda_orth) * orth_loss + float(args.olora_lambda_l2) * l2_loss

            if method in ("lorac", "lorac_ipc"):
                loss = loss + float(args.lorac_ortho) * lorac_ortho_loss(train_model)

            loss_scaled = loss / int(args.grad_accum_steps)
            loss_scaled.backward()

            if method == "lorac_ipc":
                update_lorac_ipc_importance(train_model)

            if (total_steps + 1) % int(args.grad_accum_steps) == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(args.max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.detach().item())
            total_steps += 1

            # Per-step CURb alpha scheduling
            if method == "curb" and "_curb_global_step" in method_ctx:
                method_ctx["_curb_global_step"] += 1
                factor = _curb_alpha_schedule_factor(
                    global_step=method_ctx["_curb_global_step"],
                    total_steps=method_ctx.get("_curb_total_global_steps", 1),
                    schedule=str(args.curb_alpha_schedule),
                    min_ratio=float(args.curb_alpha_min_ratio),
                    warmup_ratio=float(args.curb_alpha_warmup_ratio),
                )
                update_curb_alpha(train_model, factor)

            pbar.update(1)
            pbar.set_postfix(loss=f"{float(loss.detach().item()):.4f}")

            # --diag_curb: periodic U norm tracking (every 50 steps + final step)
            if diag_logger is not None and method == "curb":
                if total_steps % 50 == 0 or total_steps >= total_steps_target:
                    u_norms = []
                    for _dn, _dm in train_model.named_modules():
                        if isinstance(_dm, CURbLinear):
                            u_norms.append(float(torch.norm(_dm.U.data, p="fro")))
                    if u_norms:
                        diag_logger.log(
                            f"[diag_curb_step] step={total_steps}/{total_steps_target} "
                            f"loss={float(loss.detach().item()):.6f} "
                            f"||U||_F: mean={sum(u_norms)/len(u_norms):.6f} "
                            f"max={max(u_norms):.6f} min={min(u_norms):.6f}"
                        )

            if total_steps >= total_steps_target:
                break
        if total_steps >= total_steps_target:
            break

    if total_steps > 0 and (total_steps % int(args.grad_accum_steps) != 0):
        torch.nn.utils.clip_grad_norm_(trainable_params, float(args.max_grad_norm))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    pbar.close()

    if method == "olora":
        final_olora_a, _ = collect_lora_factors(train_model)
        append_olora_subspace(method_ctx, final_olora_a)

    # Diagnostic: collect CURb stats BEFORE merge
    diag_before_merge = []
    if diag_logger is not None and method == "curb":
        diag_before_merge = _diag_curb_before_merge(train_model, diag_logger)

    result_model = _merge_and_strip(train_model, method)
    result_model.to(device)

    if method == "inflora":
        _update_inflora_state_after_task_vit(
            model=result_model,
            dataloader=calib_loader,
            device=device,
            layer_indices=method_ctx["layer_indices"],
            inflora_state=method_ctx["inflora_state"],
            task_idx=int(method_ctx.get("inflora_task_idx", 0)),
            total_sessions=int(method_ctx.get("inflora_total_sessions", 1)),
            lamb=float(args.inflora_lamb),
            lame=float(args.inflora_lame),
        )

    duration = time.time() - start_time
    return result_model, {
        "train_steps": int(total_steps),
        "loss_mean": float(total_loss / max(1, total_steps)),
        "duration_sec": float(duration),
        "trainable_params": int(sum(p.numel() for p in trainable_params)),
        "skipped_nonfinite": int(skipped_nonfinite),
        "_diag_before_merge": diag_before_merge,
    }


def main():
    parser = argparse.ArgumentParser(description="ViT class-IL continual learning benchmark runner.")

    parser.add_argument(
        "--method",
        type=str,
        default="lora",
        choices=["curb", "curlora", "bilora", "lora", "mora", "olora", "inflora", "lorac", "lorac_ipc"],
    )
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "imagenet_r", "domainnet"])
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--total_round", type=int, default=3)

    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--model_dtype", type=str, default="fp32")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument(
        "--calib_train_dir",
        type=str,
        default=None,
        help="External calibration ImageFolder train directory (e.g., ImageNet-1K train).",
    )

    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--splits_path", type=str, required=True)
    parser.add_argument("--make_splits_only", action="store_true")
    parser.add_argument("--train_samples_per_task", type=int, default=-1)
    parser.add_argument("--eval_samples_per_task", type=int, default=-1)

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--train_loss_mask", type=str, default="current_task", choices=["current_task", "seen_classes"])
    parser.add_argument("--eval_mask", type=str, default="seen_classes", choices=["seen_classes"])

    # CURb/CURLoRA/BiLoRA
    parser.add_argument("--curb_rank", type=int, default=256)
    # ViT-B/16 parameter-budget matching defaults with LoRA q/k/fc1 fixed at 8/8/8:
    # choose CURb-family ranks so r^2 ~= r_lora*(in+out) per module.
    parser.add_argument("--curb_rank_q", type=int, default=118)
    parser.add_argument("--curb_rank_k", type=int, default=102)
    parser.add_argument("--curb_rank_fc1", type=int, default=176)
    parser.add_argument("--curb_alpha", type=float, default=5.0)
    parser.add_argument("--curb_basis_mode", type=str, default="weight", choices=["cov_fast", "weight", "hybrid"])
    parser.add_argument("--curb_deim_importance_order", type=str, default="low", choices=["high", "low"])
    parser.add_argument("--curb_calib_steps", type=int, default=256)
    parser.add_argument("--curb_batch_size", type=int, default=1)
    parser.add_argument(
        "--curb_calib_source",
        type=str,
        default="train",
        choices=["train", "replay_mix_train", "imagenet1k", "replay_mix_imagenet1k"],
    )
    parser.add_argument("--replay_buffer_per_task", type=int, default=0)
    parser.add_argument("--curb_update_whiten", type=str, default="none", choices=["none", "diag"])
    parser.add_argument("--curb_whiten_ridge_ratio", type=float, default=1e-4)
    parser.add_argument("--curb_whiten_ridge_abs", type=float, default=1e-12)

    parser.add_argument("--bilora_k", type=int, default=None)
    parser.add_argument("--bilora_alpha", type=float, default=None)
    parser.add_argument("--bilora_seed", type=int, default=777)
    parser.add_argument("--bilora_chunk_size", type=int, default=0)
    parser.add_argument("--bilora_freq_chunk_size", type=int, default=8192)

    # LoRA-like baseline (fixed).
    parser.add_argument("--lora_rank_q", type=int, default=8)
    parser.add_argument("--lora_rank_k", type=int, default=8)
    parser.add_argument("--lora_rank_fc1", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    parser.add_argument("--olora_lambda_orth", type=float, default=0.5)
    parser.add_argument("--olora_lambda_l2", type=float, default=0.0)

    parser.add_argument("--inflora_lamb", type=float, default=0.95)
    parser.add_argument("--inflora_lame", type=float, default=1.0)
    parser.add_argument("--inflora_calib_source", type=str, default="train", choices=["train", "imagenet1k"])
    parser.add_argument(
        "--inflora_match_trainable",
        dest="inflora_match_trainable",
        action="store_true",
        help="Match InfLoRA trainable-parameter budget to LoRA (tc_cl-compatible default: enabled).",
    )
    parser.add_argument(
        "--no_inflora_match_trainable",
        dest="inflora_match_trainable",
        action="store_false",
        help="Disable trainable-budget matching (ablation).",
    )
    parser.set_defaults(inflora_match_trainable=True)

    parser.add_argument("--lorac_ortho", type=float, default=1.0)
    parser.add_argument("--lorac_omega_lr_scale", type=float, default=1.0)
    parser.add_argument("--lorac_ipc_beta1", type=float, default=0.85)
    parser.add_argument("--lorac_ipc_beta2", type=float, default=0.85)
    parser.add_argument("--lorac_ipc_threshold", type=float, default=0.05)
    parser.add_argument("--lorac_ipc_new_mask", action="store_true")

    # Diagnostic flags
    parser.add_argument("--diag_curb", action="store_true", help="Enable CURb diagnostic logging (||ΔU||, W-drift, basis norms)")

    # CURb basis freeze: build basis once from pretrained W and reuse for all tasks
    parser.add_argument("--curb_basis_freeze", action="store_true",
                        help="Freeze CUR basis (C,R) from task-1 pretrained weights; skip re-decomposition on later tasks")
    parser.add_argument("--curb_alpha_spectral_norm", action="store_true", default=True,
                        help="Per-module alpha normalisation: alpha_eff = alpha / (sigma_max(C) * sigma_max(R))")
    parser.add_argument("--curb_u_weight_decay", type=float, default=0.05,
                        help="Weight decay applied only to CURb U parameters (separate from global weight_decay)")
    parser.add_argument("--curb_alpha_schedule", type=str, default="constant",
                        choices=["constant", "cosine", "linear"],
                        help="Alpha schedule across global training steps (default: constant)")
    parser.add_argument("--curb_alpha_min_ratio", type=float, default=1.0,
                        help="Final alpha as fraction of initial alpha (default: 1.0 = no decay)")
    parser.add_argument("--curb_alpha_warmup_ratio", type=float, default=0.0,
                        help="Fraction of total steps for alpha warmup (default: 0.0)")
    parser.add_argument("--curb_alpha_per_task", action="store_true", default=False,
                        help="Reset alpha schedule per task (warm-restart). Without this, schedule spans all tasks.")

    args = parser.parse_args()

    if args.num_tasks is None:
        args.num_tasks = _auto_num_tasks(args.dataset)
    if args.epochs < 0:
        args.epochs = _default_epochs(args.dataset)

    if args.total_round < 1:
        raise ValueError("--total_round must be >= 1")
    if args.num_tasks < 1:
        raise ValueError("--num_tasks must be >= 1")
    if args.train_batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("batch sizes must be >= 1")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1")
    if args.replay_buffer_per_task < 0:
        raise ValueError("--replay_buffer_per_task must be >= 0")

    run_id = args.run_name or datetime.now().strftime("vit_run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_path, run_id)
    os.makedirs(run_dir, exist_ok=True)
    logger = Logger(os.path.join(run_dir, "logs", "train.log"))

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=True, indent=2)

    bundle = _load_dataset_bundle(args.dataset, args.data_root, args.train_dir, args.val_dir)
    train_targets = _extract_targets(bundle.train_ds)
    eval_targets = _extract_targets(bundle.eval_ds)

    if len(set(train_targets)) != int(bundle.num_classes):
        logger.log(
            f"[warn] train split classes ({len(set(train_targets))}) != declared num_classes ({bundle.num_classes})"
        )

    logger.log(
        f"Run {run_id} start | dataset={args.dataset} num_classes={bundle.num_classes} "
        f"num_tasks={args.num_tasks} rounds={args.total_round} method={args.method} "
        f"train_mask={args.train_loss_mask} eval_mask={args.eval_mask}"
    )

    splits_path = os.path.abspath(args.splits_path)
    if args.make_splits_only or (not os.path.exists(splits_path)):
        logger.log(f"[splits] creating splits at {splits_path}")
        _make_splits(
            num_classes=int(bundle.num_classes),
            train_targets=train_targets,
            eval_targets=eval_targets,
            num_tasks=int(args.num_tasks),
            total_round=int(args.total_round),
            seed=int(args.seed),
            train_samples_per_task=int(args.train_samples_per_task),
            eval_samples_per_task=int(args.eval_samples_per_task),
            out_path=splits_path,
            logger=logger,
        )
        if args.make_splits_only:
            logger.log("[done] make_splits_only")
            logger.close()
            return

    splits_obj = _load_splits(splits_path)
    if int(splits_obj.get("num_tasks", -1)) != int(args.num_tasks):
        raise ValueError(
            f"splits num_tasks mismatch: splits={splits_obj.get('num_tasks')} cli={args.num_tasks}. "
            "Use a fresh splits_path or regenerate splits."
        )

    dtype = _resolve_torch_dtype(args.model_dtype)
    if str(args.device).lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(str(args.device))

    image_processor = AutoImageProcessor.from_pretrained(args.model_name)
    train_tf, eval_tf = _build_transforms(image_processor, int(args.image_size))

    need_external_calib = False
    if args.method == "curb" and args.curb_calib_source in ("imagenet1k", "replay_mix_imagenet1k"):
        need_external_calib = True
    if args.method == "inflora" and args.inflora_calib_source == "imagenet1k":
        need_external_calib = True

    external_calib_ds = None
    if need_external_calib:
        external_calib_ds, resolved_calib_dir = _load_external_calib_dataset(
            data_root=args.data_root,
            calib_train_dir=args.calib_train_dir,
        )
        logger.log(
            f"[calib] external source=imagenet1k train_dir={resolved_calib_dir} "
            f"num_samples={len(external_calib_ds)}"
        )

    csv_path = os.path.join(run_dir, "eval_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_fp:
        fieldnames = [
            "timestamp",
            "dataset",
            "round",
            "task_step",
            "trained_task",
            "eval_task",
            "acc",
            "AA",
            "BWT",
            "eval_samples",
            "train_loss_mean",
            "train_steps",
            "duration_sec",
        ]
        writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
        writer.writeheader()
        csv_fp.flush()

        for round_idx in range(int(args.total_round)):
            round_num = round_idx + 1
            round_seed = _round_seed(args.seed, round_idx)
            set_seed(round_seed)
            random.seed(round_seed)
            np.random.seed(round_seed)
            logger.log(f"Round {round_num}/{args.total_round} start (seed={round_seed})")

            if device.type == "cuda":
                torch.cuda.set_device(device.index if device.index is not None else 0)
                torch.cuda.empty_cache()

            model = ViTForImageClassification.from_pretrained(
                args.model_name,
                num_labels=int(bundle.num_classes),
                ignore_mismatched_sizes=True,
                torch_dtype=dtype,
            )
            model.to(device)

            q_modules, k_modules, fc1_modules, all_modules = _build_vit_module_lists(model)
            layer_indices = list(range(int(model.config.num_hidden_layers)))

            lora_rank_map = _build_rank_map(
                q_modules,
                k_modules,
                fc1_modules,
                rank_q=int(args.lora_rank_q),
                rank_k=int(args.lora_rank_k),
                rank_fc1=int(args.lora_rank_fc1),
            )
            curb_rank_map = _build_rank_map(
                q_modules,
                k_modules,
                fc1_modules,
                rank_q=int(args.curb_rank_q),
                rank_k=int(args.curb_rank_k),
                rank_fc1=int(args.curb_rank_fc1),
            )

            method_ctx = {
                "method": args.method,
                "num_classes": int(bundle.num_classes),
                "q_modules": q_modules,
                "k_modules": k_modules,
                "fc1_modules": fc1_modules,
                "all_modules": all_modules,
                "layer_indices": layer_indices,
                "lora_rank_map": lora_rank_map,
                "curb_rank_map": curb_rank_map,
            }

            # Alpha scheduling: track global steps across all tasks
            if args.method == "curb" and args.curb_alpha_schedule != "constant":
                method_ctx["_curb_global_step"] = 0

            if args.method == "olora":
                method_ctx["olora_prev_A"] = {}
            if args.method == "inflora":
                method_ctx["inflora_state"] = init_inflora_state()
                method_ctx["inflora_total_sessions"] = int(args.num_tasks)
                if bool(args.inflora_match_trainable):
                    method_ctx["inflora_rank_map"] = _compute_inflora_ranks_match_trainable(model, lora_rank_map)
                else:
                    method_ctx["inflora_rank_map"] = dict(lora_rank_map)
            if args.method in ("lorac", "lorac_ipc"):
                method_ctx["lorac_state"] = init_lorac_state(pool_size=int(args.num_tasks))

            # --diag_curb: snapshot pretrained weights for drift measurement
            diag_pretrained_snap = {}
            diag_all_rows = []
            if getattr(args, "diag_curb", False) and args.method == "curb":
                diag_pretrained_snap = _diag_curb_snapshot_pretrained(model, all_modules)
                logger.log(f"[diag_curb] pretrained snapshot saved for {len(diag_pretrained_snap)} modules")

            round_obj = _get_round_splits(splits_obj, round_num)
            task_splits = list(round_obj.get("task_splits", []))
            if len(task_splits) != int(args.num_tasks):
                raise ValueError(f"round {round_num} task count mismatch: {len(task_splits)} vs {args.num_tasks}")

            diag_acc = {}
            replay_buffers: dict[int, list[int]] = {}
            replay_pool: list[int] = []

            for task_step, task_obj in enumerate(task_splits, start=1):
                cur_classes = [int(x) for x in task_obj["classes"]]
                train_indices = [int(x) for x in task_obj["train_indices"]]

                # seen classes up to current step
                seen_classes = []
                for prev in task_splits[:task_step]:
                    seen_classes.extend([int(x) for x in prev["classes"]])
                seen_classes = sorted(set(seen_classes))

                train_ds_obj = IndexedVisionDataset(bundle.train_ds, train_indices, train_tf)

                # calibration loader budget
                calib_target = int(args.curb_calib_steps) * int(args.curb_batch_size)
                calib_seed = _seed_from_parts(round_seed, "calib", task_step)
                if args.method == "curb":
                    calib_source = str(args.curb_calib_source)
                    calib_batch_size = int(args.curb_batch_size)
                elif args.method == "inflora":
                    calib_source = "imagenet1k" if str(args.inflora_calib_source) == "imagenet1k" else "train"
                    calib_batch_size = int(args.curb_batch_size if calib_source == "imagenet1k" else args.train_batch_size)
                else:
                    calib_source = "train"
                    calib_batch_size = int(args.train_batch_size)

                if calib_source == "train":
                    # InfLoRA official recipe uses current-task training samples as calibration.
                    if args.method == "inflora":
                        train_max = len(train_indices)
                    else:
                        train_max = max(0, int(calib_target)) if int(calib_target) > 0 else len(train_indices)
                    calib_loader = _build_calib_loader_from_indices(
                        bundle.train_ds,
                        list(train_indices),
                        train_tf,
                        batch_size=calib_batch_size,
                        max_samples=int(train_max),
                        seed=calib_seed,
                        num_workers=int(args.num_workers),
                    )
                elif calib_source == "replay_mix_train":
                    calib_rng = random.Random(_seed_from_parts(round_seed, "curb_calib", task_step))
                    if int(calib_target) > 0:
                        need = max(0, int(calib_target))
                        take = min(len(replay_pool), need)
                        use_replay = calib_rng.sample(replay_pool, take) if take > 0 else []
                        remain = max(0, need - len(use_replay))
                        if remain < len(train_indices):
                            use_train = calib_rng.sample(train_indices, remain) if remain > 0 else []
                        else:
                            use_train = list(train_indices)
                    else:
                        use_replay = list(replay_pool)
                        use_train = list(train_indices)
                    calib_indices = list(use_replay) + list(use_train)
                    logger.log(
                        f"[curb_calib] round={round_num} task={task_step} source=replay_mix_train "
                        f"target={calib_target} replay_used={len(use_replay)} train_used={len(use_train)} "
                        f"replay_pool={len(replay_pool)}"
                    )
                    calib_loader = _build_calib_loader_from_indices(
                        bundle.train_ds,
                        calib_indices,
                        train_tf,
                        batch_size=calib_batch_size,
                        max_samples=len(calib_indices),
                        seed=calib_seed,
                        num_workers=int(args.num_workers),
                    )
                elif calib_source == "imagenet1k":
                    if external_calib_ds is None:
                        raise ValueError("External calib source is selected, but external_calib_ds is not initialized.")
                    max_external = max(0, int(calib_target)) if int(calib_target) > 0 else len(external_calib_ds)
                    calib_loader = _build_calib_loader_from_indices(
                        external_calib_ds,
                        list(range(len(external_calib_ds))),
                        train_tf,
                        batch_size=calib_batch_size,
                        max_samples=max_external,
                        seed=calib_seed,
                        num_workers=int(args.num_workers),
                    )
                elif calib_source == "replay_mix_imagenet1k":
                    if external_calib_ds is None:
                        raise ValueError("External calib source is selected, but external_calib_ds is not initialized.")
                    calib_rng = random.Random(_seed_from_parts(round_seed, "curb_calib", task_step))
                    if int(calib_target) > 0:
                        need = max(0, int(calib_target))
                        take = min(len(replay_pool), need)
                        use_replay = calib_rng.sample(replay_pool, take) if take > 0 else []
                        remain = max(0, need - len(use_replay))
                        external_indices = list(range(len(external_calib_ds)))
                        calib_rng.shuffle(external_indices)
                        use_external = external_indices[:remain]
                    else:
                        use_replay = list(replay_pool)
                        use_external = list(range(len(external_calib_ds)))
                        calib_rng.shuffle(use_external)
                    logger.log(
                        f"[curb_calib] round={round_num} task={task_step} source=replay_mix_imagenet1k "
                        f"target={calib_target} replay_used={len(use_replay)} imagenet1k_used={len(use_external)} "
                        f"replay_pool={len(replay_pool)}"
                    )
                    calib_loader = _build_calib_loader_from_parts(
                        [
                            (bundle.train_ds, use_replay, train_tf),
                            (external_calib_ds, use_external, train_tf),
                        ],
                        batch_size=calib_batch_size,
                        num_workers=int(args.num_workers),
                    )
                else:
                    raise ValueError(f"Unknown calibration source: {calib_source}")

                if args.method == "inflora":
                    method_ctx["inflora_task_idx"] = int(task_step - 1)
                if args.method in ("lorac", "lorac_ipc"):
                    method_ctx["lorac_task_idx"] = int(task_step - 1)
                if args.method == "bilora":
                    method_ctx["bilora_task_idx"] = int(task_step - 1)

                logger.log(
                    f"[train] start r{round_num} t{task_step}/{args.num_tasks} "
                    f"classes={len(cur_classes)} train_n={len(train_indices)}"
                )
                model, train_stats = train_on_task(
                    model=model,
                    method_ctx=method_ctx,
                    args=args,
                    device=device,
                    train_ds_obj=train_ds_obj,
                    calib_loader=calib_loader,
                    current_classes=cur_classes,
                    seen_classes=seen_classes,
                    loader_seed=_seed_from_parts(round_seed, "loader", task_step),
                    diag_logger=(logger if getattr(args, "diag_curb", False) else None),
                )

                logger.log(
                    f"[train] done r{round_num} t{task_step} steps={train_stats['train_steps']} "
                    f"loss={train_stats['loss_mean']:.6f} time={train_stats['duration_sec']:.1f}s"
                )

                # --diag_curb: after-merge W drift
                if getattr(args, "diag_curb", False) and args.method == "curb" and diag_pretrained_snap:
                    drift_rows = _diag_curb_after_merge(model, diag_pretrained_snap, all_modules, logger)
                    for bm in train_stats.get("_diag_before_merge", []):
                        bm["round"] = int(round_num)
                        bm["task_step"] = int(task_step)
                        bm["phase"] = "before_merge"
                        diag_all_rows.append(bm)
                    for dr in drift_rows:
                        dr["round"] = int(round_num)
                        dr["task_step"] = int(task_step)
                        dr["phase"] = "after_merge"
                        diag_all_rows.append(dr)

                # replay index buffer for CURb replay calibration
                if int(args.replay_buffer_per_task) > 0 and len(train_indices) > 0:
                    rb_rng = random.Random(_seed_from_parts(round_seed, "replay", task_step))
                    take = min(int(args.replay_buffer_per_task), len(train_indices))
                    replay_buffers[int(task_step)] = rb_rng.sample(train_indices, take)
                replay_pool = []
                for idxs in replay_buffers.values():
                    replay_pool.extend(idxs)

                # evaluate all seen tasks
                acc_map = {}
                for eval_task_idx in range(task_step):
                    eval_task_obj = task_splits[eval_task_idx]
                    eval_indices = [int(x) for x in eval_task_obj["eval_indices"]]
                    eval_ds_obj = IndexedVisionDataset(bundle.eval_ds, eval_indices, eval_tf)
                    eval_loader = DataLoader(
                        eval_ds_obj,
                        batch_size=int(args.eval_batch_size),
                        shuffle=False,
                        num_workers=int(args.num_workers),
                        pin_memory=True,
                    )
                    acc, n_eval = evaluate_accuracy(
                        model,
                        eval_loader,
                        allowed_classes=seen_classes,
                        num_classes=int(bundle.num_classes),
                        device=device,
                    )
                    acc_map[int(eval_task_idx + 1)] = (float(acc), int(n_eval))

                cur_acc = acc_map[int(task_step)][0]
                if int(task_step) not in diag_acc:
                    diag_acc[int(task_step)] = float(cur_acc)

                aa = float(sum(v[0] for v in acc_map.values()) / max(1, len(acc_map)))
                bwt = None
                if task_step > 1:
                    diffs = []
                    for old in range(1, task_step):
                        if old in diag_acc:
                            diffs.append(float(acc_map[old][0]) - float(diag_acc[old]))
                    bwt = float(sum(diffs) / max(1, len(diffs))) if diffs else 0.0

                for eval_task_id, (acc, n_eval) in acc_map.items():
                    row = {
                        "timestamp": datetime.now().isoformat(),
                        "dataset": str(args.dataset),
                        "round": int(round_num),
                        "task_step": int(task_step),
                        "trained_task": f"task_{task_step}",
                        "eval_task": f"task_{eval_task_id}",
                        "acc": float(acc),
                        "AA": float(aa),
                        "BWT": (float(bwt) if bwt is not None else ""),
                        "eval_samples": int(n_eval),
                        "train_loss_mean": float(train_stats["loss_mean"]),
                        "train_steps": int(train_stats["train_steps"]),
                        "duration_sec": float(train_stats["duration_sec"]),
                    }
                    writer.writerow(row)
                csv_fp.flush()

                logger.log(
                    f"[eval] r{round_num} t{task_step} AA={aa:.2f} "
                    f"BWT={(bwt if bwt is not None else 0.0):.2f} "
                    + " ".join(f"t{k}={v[0]:.2f}" for k, v in acc_map.items())
                )

            # --diag_curb: write diagnostic CSV per round
            if getattr(args, "diag_curb", False) and diag_all_rows:
                diag_csv = os.path.join(run_dir, f"diag_curb_round{round_num}.csv")
                _diag_curb_write_csv(diag_all_rows, diag_csv)
                logger.log(f"[diag_curb] saved {len(diag_all_rows)} rows → {diag_csv}")

            logger.log(f"Round {round_num}/{args.total_round} completed")

            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    logger.log(f"Run {run_id} done | eval_csv={csv_path}")
    logger.close()


if __name__ == "__main__":
    main()
