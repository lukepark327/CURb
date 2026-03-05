#!/usr/bin/env python
# autopep8: off
import os
import sys
import argparse
import json
import csv
import time
import random
import shutil
import hashlib
import subprocess
import gc
from datetime import datetime
import multiprocessing as mp
import importlib.util
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, get_scheduler
from peft import LoraConfig, get_peft_model, TaskType

import numpy as np
from tqdm import tqdm
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
import tensorflow as tf

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, repo_root)
_mora_path = os.path.join(repo_root, "MoRA", "peft-mora")
if os.path.isdir(_mora_path):
    sys.path.append(_mora_path)
    sys.path.append(os.path.join(_mora_path, "src"))

from lm_eval.models import huggingface
from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager, get_task_dict
from lm_eval.evaluator_utils import get_subtask_list

from curb import (
    CURbLinear,
    inject_curb,
    merge_curb,
    freeze_except_curb_U,
    strip_curb,
)
from curb_basis import load_or_build_curb_basis
from curlora import inject_curlora, merge_curlora, freeze_except_curlora_U, strip_curlora
from bilora import inject_bilora, merge_bilora, freeze_except_bilora_theta, strip_bilora
from olora import (
    collect_lora_factors,
    build_olora_prev_device_map,
    append_olora_subspace,
    compute_olora_losses,
)
from inflora import (
    init_inflora_state,
    design_inflora_b_by_module,
    update_inflora_state_after_task,
    apply_inflora_to_peft_model,
)
from lorac import (
    init_lorac_state,
    inject_lorac,
    merge_lorac,
    strip_lorac,
    freeze_except_lorac,
    update_lorac_ipc_importance,
    lorac_ortho_loss,
)

# autopep8: on

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
try:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass


DEFAULT_TASKS = [
    "boolq",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "piqa",
    "openbookqa",
    "social_iqa",
    "logiqa",
]

EVAL_MODES = ["base"]

CONSOLE_STREAM = sys.stdout


def _log(message: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def _log_console(message: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", file=CONSOLE_STREAM, flush=True)


def _redirect_output(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8")
    sys.stdout = log_f
    sys.stderr = log_f
    return log_f


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    size = float(value)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if size < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(size)}{unit}"
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TiB"


def _format_mem_stats(mem_stats: dict) -> str:
    alloc = _format_bytes(mem_stats.get("gpu_mem_alloc"))
    peak = _format_bytes(mem_stats.get("gpu_mem_peak"))
    return f"gpu_alloc={alloc} gpu_peak={peak}"


def _summarize_eval_progress(eval_stats, total_tasks: int) -> str:
    parts = []
    for mode in EVAL_MODES:
        count = int(eval_stats.get(f"{mode}_count", 0))
        parts.append(f"{mode}:{count}/{total_tasks}")
    return " ".join(parts)


def _estimate_eta(train_times, train_done, total_tasks, eval_stats):
    train_avg = None
    if train_times:
        train_avg = sum(train_times) / len(train_times)
    train_remaining = None
    if train_avg is not None:
        train_remaining = train_avg * max(0, total_tasks - train_done)

    eval_remaining = None
    for mode in EVAL_MODES:
        count = int(eval_stats.get(f"{mode}_count", 0))
        total_time = float(eval_stats.get(f"{mode}_time", 0.0))
        if count <= 0:
            continue
        avg = total_time / count
        remaining = avg * max(0, total_tasks - count)
        eval_remaining = remaining if eval_remaining is None else max(eval_remaining, remaining)

    if train_remaining is None and eval_remaining is None:
        return None, None, None

    overall = train_remaining if eval_remaining is None else max(train_remaining or 0, eval_remaining)
    return overall, train_remaining, eval_remaining


def _log_progress(train_done, total_tasks, train_times, eval_progress, start_time, context):
    elapsed = time.time() - start_time
    overall_eta, train_eta, eval_eta = _estimate_eta(train_times, train_done, total_tasks, eval_progress)
    eval_summary = _summarize_eval_progress(eval_progress, total_tasks)
    eta_parts = []
    if train_eta is not None:
        eta_parts.append(f"train~{_format_duration(train_eta)}")
    if eval_eta is not None:
        eta_parts.append(f"eval~{_format_duration(eval_eta)}")
    eta_text = f"eta~{_format_duration(overall_eta)}" if overall_eta is not None else "eta~n/a"
    suffix = f"{eta_text} ({', '.join(eta_parts)})" if eta_parts else eta_text
    message = (f"{context} | train {train_done}/{total_tasks} ({train_done/total_tasks:.1%}) "
               f"| eval {eval_summary} | elapsed {_format_duration(elapsed)} | {suffix}")
    _log(message)
    _log_console(message)


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _append_value(bucket, value):
    if value is None:
        return
    if isinstance(value, float) and math.isnan(value):
        return
    bucket.append(value)


def _summarize_round_metrics(eval_csv_path, summary_path):
    if not os.path.exists(eval_csv_path):
        _log(f"[summary] eval_metrics.csv not found: {eval_csv_path}")
        return False
    with open(eval_csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        _log("[summary] eval_metrics.csv is empty")
        return False

    per_round = {}
    for row in rows:
        key = (row.get("round"), row.get("task_step"), row.get("trained_task"), row.get("mode"))
        bucket = per_round.setdefault(key, {"aa": [], "bwt": []})
        _append_value(bucket["aa"], _to_float(row.get("mean_acc")))
        _append_value(bucket["bwt"], _to_float(row.get("bwt")))

    per_round_values = {}
    for key, bucket in per_round.items():
        aa = float(np.mean(bucket["aa"])) if bucket["aa"] else None
        bwt = float(np.mean(bucket["bwt"])) if bucket["bwt"] else None
        per_round_values[key] = {"aa": aa, "bwt": bwt}

    summary = {}
    for (round_id, task_step, trained_task, mode), vals in per_round_values.items():
        s_key = (task_step, trained_task, mode)
        s_bucket = summary.setdefault(
            s_key, {"aa": [], "bwt": []}
        )
        _append_value(s_bucket["aa"], vals["aa"])
        _append_value(s_bucket["bwt"], vals["bwt"])

    fieldnames = [
        "task_step", "trained_task", "mode",
        "aa_mean", "aa_std", "aa_rounds",
        "bwt_mean", "bwt_std", "bwt_rounds",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (task_step, trained_task, mode), bucket in sorted(summary.items(), key=lambda x: (int(x[0][0]), x[0][2])):
            aa_vals = bucket["aa"]
            bwt_vals = bucket["bwt"]
            aa_mean = float(np.mean(aa_vals)) if aa_vals else None
            aa_std = float(np.std(aa_vals)) if aa_vals else None
            bwt_mean = float(np.mean(bwt_vals)) if bwt_vals else None
            bwt_std = float(np.std(bwt_vals)) if bwt_vals else None
            writer.writerow({
                "task_step": task_step,
                "trained_task": trained_task,
                "mode": mode,
                "aa_mean": aa_mean,
                "aa_std": aa_std,
                "aa_rounds": len(aa_vals),
                "bwt_mean": bwt_mean,
                "bwt_std": bwt_std,
                "bwt_rounds": len(bwt_vals),
            })

    return True

def _parse_eval_gpus(value: str) -> list[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts]


def _build_task_config(num_test_steps):
    def _n(x):
        return max(1, int(x)) if x is not None else None

    return {
        "social_iqa":    {"task_type": "classification", "limit": _n(num_test_steps / 3), "fewshot": 0},
        "logiqa":        {"task_type": "classification", "limit": _n(num_test_steps / 4), "fewshot": 5},
        "winogrande":    {"task_type": "classification", "limit": _n(num_test_steps / 2), "fewshot": 5},
        "arc_easy":      {"task_type": "classification", "limit": _n(num_test_steps / 4), "fewshot": 25},
        "arc_challenge": {"task_type": "classification", "limit": _n(num_test_steps / 4), "fewshot": 25},
        "piqa":          {"task_type": "classification", "limit": _n(num_test_steps / 2), "fewshot": 0},
        "openbookqa":    {"task_type": "classification", "limit": _n(num_test_steps / 4), "fewshot": 0},
        "mmlu":          {"task_type": "classification", "limit": _n(num_test_steps / 4), "fewshot": 5},
        "boolq":         {"task_type": "classification", "limit": _n(num_test_steps / 2), "fewshot": 0},
    }


def _seed_from_parts(base_seed: int, *parts) -> int:
    payload = "|".join(str(p) for p in parts)
    h = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) + base_seed) % (2**32)


def count_params(model, trainable_only=False) -> int:
    if hasattr(model, "num_parameters"):
        try:
            return model.num_parameters(only_trainable=trainable_only)
        except TypeError:
            pass
    params = (p for p in model.parameters() if (p.requires_grad or not trainable_only))
    return sum(p.numel() for p in params)


def _curb_param_budget(curb_rank: int) -> int:
    r = max(1, int(curb_rank))
    return r * r


def _effective_curb_rank(in_features: int, out_features: int, curb_rank: int) -> int:
    return max(1, min(int(curb_rank), int(in_features), int(out_features)))


def _compute_curb_ranks(model, curb_rank, ffn_module_names, attn_module_names, rank_overrides=None):
    shapes = _infer_module_shapes(model, ffn_module_names, attn_module_names)
    ranks = {}
    for key, (in_f, out_f) in shapes.items():
        max_rank = int(min(in_f, out_f))
        rank_val = _effective_curb_rank(in_f, out_f, curb_rank)
        if rank_overrides and (key in rank_overrides) and (rank_overrides[key] is not None):
            requested = int(rank_overrides[key])
            if requested < 1:
                raise ValueError(f"Rank override for {key} must be >= 1 (got {requested}).")
            if requested > max_rank:
                raise ValueError(
                    f"Rank override for {key} exceeds module limit: requested={requested}, max={max_rank}."
                )
            rank_val = requested
        ranks[key] = int(max(1, min(rank_val, max_rank)))
    return ranks


def _calc_lora_rank(in_features: int, out_features: int, curb_rank: int) -> int:
    r_eff = _effective_curb_rank(in_features, out_features, curb_rank)
    budget = _curb_param_budget(r_eff)
    denom = max(1, int(in_features + out_features))
    r = budget // denom
    r = max(1, min(r, int(min(in_features, out_features))))
    return int(r)


def _infer_module_shapes(model, ffn_module_names, attn_module_names):
    layer0 = model.model.layers[0]
    shapes = {}
    for name in ffn_module_names:
        module = getattr(layer0.mlp, name)
        shapes[f"mlp_{name}"] = (int(module.in_features), int(module.out_features))
    for name in attn_module_names:
        module = getattr(layer0.self_attn, name)
        shapes[f"attn_{name}"] = (int(module.in_features), int(module.out_features))
    return shapes


def _compute_lora_ranks(model, curb_rank, ffn_module_names, attn_module_names, rank_overrides=None):
    shapes = _infer_module_shapes(model, ffn_module_names, attn_module_names)
    ranks = {}
    for key, (in_f, out_f) in shapes.items():
        rank_val = _calc_lora_rank(in_f, out_f, curb_rank)
        if rank_overrides and (key in rank_overrides) and (rank_overrides[key] is not None):
            requested = int(rank_overrides[key])
            if requested < 1:
                raise ValueError(f"Rank override for {key} must be >= 1 (got {requested}).")
            max_rank = int(min(in_f, out_f))
            if requested > max_rank:
                raise ValueError(
                    f"Rank override for {key} exceeds module limit: requested={requested}, max={max_rank}."
                )
            rank_val = requested
        ranks[key] = int(rank_val)
    return ranks


def _compute_inflora_ranks_match_trainable(model, lora_ranks, ffn_module_names, attn_module_names):
    """
    InfLoRA freezes LoRA's lora_A (= B_t) and only trains lora_B (= A_t).

    For fair comparison against LoRA (which trains both lora_A and lora_B), we
    increase InfLoRA rank so that the number of *trainable* parameters matches:
      out_features * r_inflora  ~=  r_lora * (in_features + out_features)
    """
    shapes = _infer_module_shapes(model, ffn_module_names, attn_module_names)
    inflora_ranks = {}
    for key, (in_f, out_f) in shapes.items():
        if key not in lora_ranks:
            raise KeyError(f"Missing LoRA rank for {key} (needed for InfLoRA rank matching).")
        r_lora = int(lora_ranks[key])
        target_trainable = int(r_lora * (int(in_f) + int(out_f)))
        # Choose the closest integer rank so that out_f * r is near target.
        r = int(target_trainable / max(1, int(out_f)) + 0.5)
        r = max(1, min(r, int(min(in_f, out_f))))
        inflora_ranks[key] = int(r)
    return inflora_ranks


def _mora_new_r(in_features: int, out_features: int, r_lora: int, mora_type: int = 6) -> int:
    new_r = int(math.sqrt((in_features + out_features) * r_lora) + 0.5)
    if mora_type == 6:
        new_r = (new_r // 2) * 2
    return max(1, new_r)


def _compute_param_budget_table(
    model,
    curb_rank,
    ffn_module_names,
    attn_module_names,
    lora_ranks=None,
    curb_ranks=None,
):
    shapes = _infer_module_shapes(model, ffn_module_names, attn_module_names)
    rows = []
    for key, (in_f, out_f) in shapes.items():
        if curb_ranks is not None and key in curb_ranks:
            r_eff = max(1, min(int(curb_ranks[key]), int(in_f), int(out_f)))
        else:
            r_eff = _effective_curb_rank(in_f, out_f, curb_rank)
        if lora_ranks is not None and key in lora_ranks:
            r_lora = int(lora_ranks[key])
        else:
            r_lora = _calc_lora_rank(in_f, out_f, curb_rank)
        lora_params = r_lora * (in_f + out_f)
        curb_params = r_eff * r_eff
        mora_r = _mora_new_r(in_f, out_f, r_lora, mora_type=6)
        mora_params = mora_r * mora_r
        rows.append({
            "module": key,
            "in_features": int(in_f),
            "out_features": int(out_f),
            "r_eff": int(r_eff),
            "r_lora": int(r_lora),
            "lora_params": int(lora_params),
            "curb_params": int(curb_params),
            "mora_new_r": int(mora_r),
            "mora_params": int(mora_params),
        })

    per_layer = {
        "curb": sum(r["curb_params"] for r in rows),
        "curlora": sum(r["curb_params"] for r in rows),
        "bilora": sum(r["curb_params"] for r in rows),
        "lora": sum(r["lora_params"] for r in rows),
        "olora": sum(r["lora_params"] for r in rows),
        "mora": sum(r["mora_params"] for r in rows),
    }
    return rows, per_layer


def _format_rank_triplet(ranks):
    if not ranks:
        return "n/a"
    q = ranks.get("attn_q_proj")
    k = ranks.get("attn_k_proj")
    g = ranks.get("mlp_gate_proj")
    if q is not None and k is not None and g is not None:
        return f"q={int(q)} k={int(k)} gate={int(g)}"
    keys = sorted(ranks.keys())
    return " ".join(f"{k}={int(ranks[k])}" for k in keys)


def _curb_rank_cache_tag(curb_rank, curb_ranks=None):
    if not curb_ranks:
        return f"r{int(curb_rank)}"
    q = curb_ranks.get("attn_q_proj")
    k = curb_ranks.get("attn_k_proj")
    g = curb_ranks.get("mlp_gate_proj")
    if q is not None and k is not None and g is not None:
        return f"rq{int(q)}_rk{int(k)}_rg{int(g)}"
    return f"r{int(curb_rank)}"


def _curb_basis_cache_mode_tag(mode: str, deim_importance_order: str) -> str:
    if deim_importance_order == "high":
        return mode
    return f"{mode}_{deim_importance_order}"


def _param_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _pick_primary_metric_keys(res: dict):
    preferred = [
        "acc_norm,none", "acc_norm",
        "acc,none", "acc",
        "exact_match,flexible-extract", "exact_match_flexible_extract",
        "exact_match,strict-match", "exact_match_strict_extract",
        "exact_match,none", "exact_match",
    ]
    metric_key = next((k for k in preferred if k in res), None)
    if metric_key is None:
        metric_key = next(k for k in res.keys() if "stderr" not in k)

    base, option = metric_key.split(",") if "," in metric_key else (metric_key, None)

    stderr_key_candidates = [
        f"{base}_stderr,{option}",
        f"{base}_stderr",
        f"{metric_key}_stderr",
    ]
    stderr_key = next((k for k in stderr_key_candidates if k in res), None)

    return metric_key, stderr_key


def _get_gpu_mem_stats(device: torch.device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"gpu_mem_alloc": None, "gpu_mem_peak": None}
    return {
        "gpu_mem_alloc": int(torch.cuda.memory_allocated()),
        "gpu_mem_peak": int(torch.cuda.max_memory_allocated()),
    }


class ListDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _collate_batch(batch, tokenizer):
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    enc = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        return_tensors="pt",
    )
    label_tensors = [torch.tensor(x, dtype=torch.long) for x in labels]
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        label_tensors, batch_first=True, padding_value=-100
    )
    enc["labels"] = labels_padded
    return enc


def _get_preferred_docs(task):
    if getattr(task, "has_training_docs", None) and task.has_training_docs():
        return task.training_docs()
    if getattr(task, "has_validation_docs", None) and task.has_validation_docs():
        return task.validation_docs()
    # if getattr(task, "has_test_docs", None) and task.has_test_docs():
    #     return task.test_docs()
    # fewshot_docs = task.fewshot_docs()
    # if fewshot_docs is not None:
    #     return fewshot_docs
    return None


def _select_docs(docs, n, rng: random.Random):
    if docs is None:
        return []
    if hasattr(docs, "select"):
        total = len(docs)
        take = min(n, total)
        if take <= 0:
            return []
        if take == total:
            return list(docs)
        indices = rng.sample(range(total), take)
        return list(docs.select(indices))
    if hasattr(docs, "__len__"):
        total = len(docs)
        take = min(n, total)
        if take <= 0:
            return []
        indices = rng.sample(range(total), take)
        return [docs[i] for i in indices]
    selected = []
    for idx, item in enumerate(docs):
        if idx >= n:
            break
        selected.append(item)
    return selected


def _resolve_subtasks(task_name: str, task_manager: TaskManager):
    task_dict = get_task_dict(task_name, task_manager)
    subtask_map = get_subtask_list(task_dict)
    subtasks = subtask_map.get(task_name, [])
    if not subtasks:
        subtasks = [task_name]
    return subtasks


def _adjust_task_dict(task_dict, fewshot_map: dict, seed: int, fallback_fewshot: int | None = None):
    adjusted = {}
    for name, obj in task_dict.items():
        if isinstance(obj, dict):
            adjusted[name] = _adjust_task_dict(obj, fewshot_map, seed, fallback_fewshot=fallback_fewshot)
        else:
            fewshot = fewshot_map.get(name, fallback_fewshot or 0)
            obj.set_config(key="num_fewshot", value=fewshot)
            obj.set_fewshot_seed(seed=seed)
            adjusted[name] = obj
    return adjusted


def _find_group_entry(task_dict, subtask_name: str):
    for key, val in task_dict.items():
        if isinstance(key, str):
            if key == subtask_name:
                return "task", val
            continue
        key_name = None
        if hasattr(key, "group_name"):
            key_name = getattr(key, "group_name")
        elif hasattr(key, "group"):
            key_name = getattr(key, "group")
        if key_name == subtask_name:
            return "group", val
    return None, None


def _flatten_task_dict(task_dict):
    items = []
    for name, obj in task_dict.items():
        if isinstance(obj, dict):
            items.extend(_flatten_task_dict(obj))
        else:
            items.append((name, obj))
    return items


def _build_prompt_and_target(task, doc):
    num_fewshot = task.get_config("num_fewshot") or 0
    ctx = task.fewshot_context(doc, num_fewshot=num_fewshot)
    if isinstance(ctx, list):
        ctx = ctx[0]
    target_delim = task.get_config("target_delimiter") or " "

    if task.multiple_input:
        choices = task.doc_to_choice(doc)
        gold_idx = task.doc_to_text(doc)
        if isinstance(gold_idx, str):
            if gold_idx in choices:
                gold_idx = choices.index(gold_idx)
            else:
                return None, None
        if not isinstance(gold_idx, int) or gold_idx >= len(choices):
            return None, None
        prompt = ctx + choices[gold_idx]
        target_val = task.doc_to_target(doc)
        if isinstance(target_val, list):
            target_val = target_val[0]
        target = target_delim + str(target_val)
        return prompt, target

    prompt = ctx
    target_val = task.doc_to_target(doc)
    if isinstance(target_val, list):
        target_val = target_val[0]
    if task.config.doc_to_choice is not None and isinstance(target_val, int):
        choices = task.doc_to_choice(doc)
        if 0 <= target_val < len(choices):
            target_val = choices[target_val]
    target = target_delim + str(target_val)
    return prompt, target


def _encode_example(tokenizer, prompt, target, max_length):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    full_ids = prompt_ids + target_ids

    if max_length and len(full_ids) > max_length:
        overflow = len(full_ids) - max_length
        if overflow < len(prompt_ids):
            prompt_ids = prompt_ids[overflow:]
        else:
            trim = overflow - len(prompt_ids)
            prompt_ids = []
            target_ids = target_ids[trim:]
        full_ids = prompt_ids + target_ids

    if not target_ids:
        return None

    labels = [-100] * len(prompt_ids) + target_ids
    return {
        "input_ids": full_ids,
        "labels": labels,
        "prompt_len": len(prompt_ids),
        "total_len": len(full_ids),
    }


def build_training_samples(
    task_name: str,
    task_manager: TaskManager,
    tokenizer,
    max_length: int,
    total_samples: int,
    fewshot_map: dict,
    base_seed: int,
):
    subtasks = _resolve_subtasks(task_name, task_manager)
    per_subtask = total_samples // max(1, len(subtasks))
    remainder = max(0, total_samples - per_subtask * len(subtasks))

    subtask_task_dict = get_task_dict(subtasks, task_manager)
    base_fewshot = fewshot_map.get(task_name, 0)
    subtask_task_dict = _adjust_task_dict(
        subtask_task_dict,
        fewshot_map,
        base_seed,
        fallback_fewshot=base_fewshot,
    )

    samples = []
    subtask_counts = {}

    for idx, subtask_name in enumerate(subtasks):
        take = per_subtask + (1 if idx < remainder else 0)
        subtask_seed = _seed_from_parts(base_seed, task_name, subtask_name)

        entry_type, entry_val = _find_group_entry(subtask_task_dict, subtask_name)
        if entry_type == "task":
            task_obj = entry_val
            docs = _get_preferred_docs(task_obj)
            rng = random.Random(subtask_seed)
            selected_docs = _select_docs(docs, take, rng)
            subtask_counts[subtask_name] = len(selected_docs)

            for doc in selected_docs:
                prompt, target = _build_prompt_and_target(task_obj, doc)
                if prompt is None:
                    continue
                enc = _encode_example(tokenizer, prompt, target, max_length)
                if enc is None:
                    continue
                full_text = prompt + target
                samples.append({
                    "input_ids": enc["input_ids"],
                    "labels": enc["labels"],
                    "text": full_text,
                    "prompt": prompt,
                    "target": target,
                    "task": task_name,
                    "subtask": subtask_name,
                    "prompt_len": enc["prompt_len"],
                    "total_len": enc["total_len"],
                })
            continue

        if entry_type == "group":
            group_tasks = _flatten_task_dict(entry_val)
            if not group_tasks:
                subtask_counts[subtask_name] = 0
                continue
            per_group_task = take // len(group_tasks)
            group_remainder = max(0, take - per_group_task * len(group_tasks))
            group_total = 0
            for g_idx, (g_name, g_task) in enumerate(group_tasks):
                g_take = per_group_task + (1 if g_idx < group_remainder else 0)
                g_seed = _seed_from_parts(subtask_seed, g_name)
                rng = random.Random(g_seed)
                docs = _get_preferred_docs(g_task)
                selected_docs = _select_docs(docs, g_take, rng)
                group_total += len(selected_docs)

                for doc in selected_docs:
                    prompt, target = _build_prompt_and_target(g_task, doc)
                    if prompt is None:
                        continue
                    enc = _encode_example(tokenizer, prompt, target, max_length)
                    if enc is None:
                        continue
                    full_text = prompt + target
                    samples.append({
                        "input_ids": enc["input_ids"],
                        "labels": enc["labels"],
                        "text": full_text,
                        "prompt": prompt,
                        "target": target,
                        "task": task_name,
                        "subtask": subtask_name,
                        "prompt_len": enc["prompt_len"],
                        "total_len": enc["total_len"],
                    })
            subtask_counts[subtask_name] = group_total
            continue

        subtask_counts[subtask_name] = 0

    return samples, subtask_counts


def _build_target_module_lists(layer_indices):
    target_modules_q = [f"layers.{layer}.self_attn.q_proj" for layer in layer_indices]
    target_modules_k = [f"layers.{layer}.self_attn.k_proj" for layer in layer_indices]
    target_modules_gate = [f"layers.{layer}.mlp.gate_proj" for layer in layer_indices]
    return target_modules_q, target_modules_k, target_modules_gate


def _load_mora_peft(repo_root):
    mora_src = os.path.join(repo_root, "MoRA", "peft-mora", "src")
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


def train_on_samples(
    model,
    tokenizer,
    samples,
    args,
    device,
    method_ctx,
    task_label=None,
    loader_seed=None,
    learning_rate=None,
    tf_writer=None,
    tf_global_step_start=0,
    tf_log_every=10,
):
    if not samples:
        return model, {
            "train_steps": 0,
            "loss_mean": None,
            "tokens": 0,
            "duration_sec": 0.0,
            "trainable_params": 0,
            "learning_rate": float(args.learning_rate if learning_rate is None else learning_rate),
        }
    method = method_ctx["method"]
    train_model = model

    if method in ("lora", "mora", "olora"):
        ranks = method_ctx["lora_ranks"]
        target_modules_q = method_ctx["target_modules_q"]
        target_modules_k = method_ctx["target_modules_k"]
        target_modules_gate = method_ctx["target_modules_gate"]
        if method == "mora":
            mora_peft = _load_mora_peft(method_ctx["repo_root"])
            config_cls = mora_peft.LoraConfig
            config_kwargs = {
                "use_mora": True,
                "mora_type": 6,
            }
            task_type = mora_peft.TaskType.CAUSAL_LM
        else:
            mora_peft = None
            config_cls = LoraConfig
            config_kwargs = {}
            task_type = TaskType.CAUSAL_LM

        alpha_q = int(2 * int(ranks["attn_q_proj"]))
        alpha_k = int(2 * int(ranks["attn_k_proj"]))
        alpha_gate = int(2 * int(ranks["mlp_gate_proj"]))
        if args.lora_alpha is not None:
            fixed_alpha = float(args.lora_alpha)
            alpha_q = fixed_alpha
            alpha_k = fixed_alpha
            alpha_gate = fixed_alpha

        lora_config_q = config_cls(
            r=ranks["attn_q_proj"],
            lora_alpha=alpha_q,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=task_type,
            target_modules=target_modules_q,
            **config_kwargs,
        )
        lora_config_k = config_cls(
            r=ranks["attn_k_proj"],
            lora_alpha=alpha_k,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=task_type,
            target_modules=target_modules_k,
            **config_kwargs,
        )
        lora_config_gate = config_cls(
            r=ranks["mlp_gate_proj"],
            lora_alpha=alpha_gate,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=task_type,
            target_modules=target_modules_gate,
            **config_kwargs,
        )
        if method == "mora":
            train_model = mora_peft.get_peft_model(train_model, lora_config_q)
            train_model = mora_peft.get_peft_model(train_model, lora_config_k)
            train_model = mora_peft.get_peft_model(train_model, lora_config_gate)
        else:
            train_model = get_peft_model(train_model, lora_config_q)
            train_model = get_peft_model(train_model, lora_config_k)
            train_model = get_peft_model(train_model, lora_config_gate)
        train_model.to(device)
    elif method == "inflora":
        # NOTE: For fair comparison, we match the number of *trainable* parameters
        # to the LoRA baseline even though InfLoRA freezes lora_A (= B_t). We do
        # this by increasing InfLoRA rank so that:
        #   trainable(InfLoRA) ~= trainable(LoRA)
        # where trainable(InfLoRA) counts only lora_B and trainable(LoRA) counts
        # both lora_A and lora_B.
        inflora_ranks = method_ctx["inflora_ranks"]
        target_modules_q = method_ctx["target_modules_q"]
        target_modules_k = method_ctx["target_modules_k"]
        target_modules_gate = method_ctx["target_modules_gate"]

        inflora_state = method_ctx["inflora_state"]
        task_idx = int(method_ctx.get("inflora_task_idx", 0))

        if args.inflora_calib_source == "train":
            # Official InfLoRA: collect curr_matrix from current-task training samples.
            calib_ds = ListDataset(samples)
            calib_loader = DataLoader(
                calib_ds,
                batch_size=args.train_batch_size,
                shuffle=False,
                collate_fn=lambda b: _collate_batch(b, tokenizer),
            )
        elif args.inflora_calib_source == "c4":
            # Ablation for fair comparison: use external C4 (no task data) with the
            # same sequence count as CURb calibration: curb_calib_steps*curb_batch_size.
            from curb_basis import _build_c4_loader

            target_sequences = int(args.curb_calib_steps) * int(args.curb_batch_size)
            calib_loader = _build_c4_loader(
                tokenizer,
                batch_size=int(args.curb_batch_size),
                num_sequences=int(target_sequences),
                max_length=int(args.max_length),
                dataset_category=str(args.curb_calib_category),
            )
        else:
            raise ValueError(f"Unknown --inflora_calib_source: {args.inflora_calib_source}")
        b_by_module = design_inflora_b_by_module(
            model=train_model,
            dataloader=calib_loader,
            device=device,
            layer_indices=method_ctx["layer_indices"],
            ffn_module_names=method_ctx["ffn_module_names"],
            attn_module_names=method_ctx["attn_module_names"],
            inflora_ranks=inflora_ranks,
            inflora_state=inflora_state,
            task_idx=task_idx,
        )

        # Official InfLoRA has no separate alpha; to make PEFT's LoRA scaling = 1,
        # enforce lora_alpha = r (scaling = alpha/r = 1). B_t itself is scaled by 1/sqrt(3).
        task_type = TaskType.CAUSAL_LM
        rq = int(inflora_ranks["attn_q_proj"])
        rk = int(inflora_ranks["attn_k_proj"])
        rg = int(inflora_ranks["mlp_gate_proj"])
        lora_config_q = LoraConfig(
            r=rq,
            lora_alpha=rq,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=task_type,
            target_modules=target_modules_q,
        )
        lora_config_k = LoraConfig(
            r=rk,
            lora_alpha=rk,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=task_type,
            target_modules=target_modules_k,
        )
        lora_config_gate = LoraConfig(
            r=rg,
            lora_alpha=rg,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=task_type,
            target_modules=target_modules_gate,
        )

        train_model = get_peft_model(train_model, lora_config_q)
        train_model = get_peft_model(train_model, lora_config_k)
        train_model = get_peft_model(train_model, lora_config_gate)
        train_model.to(device)
        apply_inflora_to_peft_model(train_model, b_by_module)
    elif method in ("lorac", "lorac_ipc"):
        lora_ranks = method_ctx["lora_ranks"]
        lorac_state = method_ctx["lorac_state"]
        task_idx = int(method_ctx.get("lorac_task_idx", 0))

        train_model = inject_lorac(
            train_model,
            layer_indices=method_ctx["layer_indices"],
            ffn_module_names=method_ctx["ffn_module_names"],
            attn_module_names=method_ctx["attn_module_names"],
            lora_ranks=lora_ranks,
            lorac_state=lorac_state,
            task_idx=task_idx,
            lora_alpha=args.lora_alpha,
            ipc_enabled=(method == "lorac_ipc"),
            ipc_beta1=float(args.lorac_ipc_beta1),
            ipc_beta2=float(args.lorac_ipc_beta2),
            ipc_threshold=float(args.lorac_ipc_threshold),
            ipc_new_mask=bool(args.lorac_ipc_new_mask),
        )
        freeze_except_lorac(train_model)
    elif method == "curb":
        basis = method_ctx["curb_basis"]
        layer_indices = method_ctx["layer_indices"]
        train_model = inject_curb(
            train_model,
            basis=basis,
            layer_indices=layer_indices,
            ffn_module_names=method_ctx["ffn_module_names"],
            attn_module_names=method_ctx["attn_module_names"],
            alpha=args.curb_alpha,
        )
        freeze_except_curb_U(train_model)
    elif method == "curlora":
        layer_indices = method_ctx["layer_indices"]
        train_model = inject_curlora(
            train_model,
            layer_indices=layer_indices,
            ffn_module_names=method_ctx["ffn_module_names"],
            attn_module_names=method_ctx["attn_module_names"],
            rank=method_ctx["curb_rank"],
            alpha=args.curb_alpha,
            rank_overrides=method_ctx.get("curb_ranks"),
        )
        freeze_except_curlora_U(train_model)
    elif method == "bilora":
        layer_indices = method_ctx["layer_indices"]
        train_model = inject_bilora(
            train_model,
            layer_indices=layer_indices,
            ffn_module_names=method_ctx["ffn_module_names"],
            attn_module_names=method_ctx["attn_module_names"],
            rank=method_ctx["curb_rank"],
            k=args.bilora_k,
            alpha=args.bilora_alpha,
            seed=int(args.bilora_seed),
            task_idx=int(method_ctx.get("bilora_task_idx", 0)),
            chunk_size=int(args.bilora_chunk_size),
            freq_chunk_size=int(args.bilora_freq_chunk_size),
            rank_overrides=method_ctx.get("curb_ranks"),
        )
        freeze_except_bilora_theta(train_model)
    else:
        raise ValueError(f"Unknown method: {method}")

    dataset = ListDataset(samples)
    loader_generator = None
    if loader_seed is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(loader_seed))

    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=loader_generator,
        collate_fn=lambda b: _collate_batch(b, tokenizer),
    )

    train_model.train()
    use_cache_flag = train_model.config.use_cache
    train_model.config.use_cache = False

    trainable_named_params = [(n, p) for n, p in train_model.named_parameters() if p.requires_grad]
    trainable_params = [p for _, p in trainable_named_params]
    current_lr = float(args.learning_rate if learning_rate is None else learning_rate)
    if method in ("lorac", "lorac_ipc"):
        base_params = []
        omega_params = []
        for name, param in trainable_named_params:
            if name.endswith(".omega"):
                omega_params.append(param)
            else:
                base_params.append(param)
        param_groups = []
        if base_params:
            param_groups.append({"params": base_params})
        if omega_params:
            param_groups.append({"params": omega_params, "lr": current_lr * float(args.lorac_omega_lr_scale)})
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=current_lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=current_lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )

    olora_a = {}
    olora_b = {}
    olora_prev_a = {}
    olora_orth_total = 0.0
    olora_l2_total = 0.0
    olora_reg_steps = 0
    if method == "olora":
        olora_a, olora_b = collect_lora_factors(train_model)
        olora_dtype = next(iter(olora_a.values())).dtype if olora_a else torch.float32
        olora_prev_a = build_olora_prev_device_map(method_ctx, device, olora_dtype)

    total_loss = 0.0
    total_steps = 0
    total_tokens = 0
    start_time = time.time()

    optimizer.zero_grad(set_to_none=True)

    total_batches = None
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    total_target = None
    if total_batches is not None:
        total_target = total_batches * args.num_train_epochs
        if args.max_train_steps is not None:
            total_target = min(total_target, args.max_train_steps)

    total_optimizer_steps = None
    scheduler = None
    if total_target is not None:
        total_optimizer_steps = max(1, math.ceil(total_target / max(1, args.grad_accum_steps)))
        warmup_steps = int(total_optimizer_steps * max(0.0, args.warmup_ratio))
        scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )

    desc = f"train {task_label}" if task_label else "train"
    pbar = tqdm(total=total_target, desc=desc, leave=False, file=CONSOLE_STREAM)

    for _ in range(args.num_train_epochs):
        for step, batch in enumerate(loader):
            if args.max_train_steps is not None and total_steps >= args.max_train_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = train_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            orth_loss = outputs.loss.new_zeros(())
            l2_loss = outputs.loss.new_zeros(())
            total_step_loss = outputs.loss
            if method == "olora":
                orth_loss, l2_loss = compute_olora_losses(outputs.loss, olora_a, olora_b, olora_prev_a)

                total_step_loss = (
                    total_step_loss
                    + float(args.olora_lambda_orth) * orth_loss
                    + float(args.olora_lambda_l2) * l2_loss
                )
                olora_orth_total += float(orth_loss.detach().item())
                olora_l2_total += float(l2_loss.detach().item())
                olora_reg_steps += 1
            if method in ("lorac", "lorac_ipc"):
                total_step_loss = total_step_loss + float(args.lorac_ortho) * lorac_ortho_loss(train_model)

            loss = total_step_loss / args.grad_accum_steps
            loss.backward()
            if method == "lorac_ipc":
                update_lorac_ipc_importance(train_model)

            if (total_steps + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            total_steps += 1
            total_tokens += int((labels != -100).sum().item())
            global_step = tf_global_step_start + total_steps
            if pbar is not None:
                if args.grad_accum_steps > 1:
                    display_loss = loss.item() * args.grad_accum_steps
                else:
                    display_loss = loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=f"{display_loss:.4f}")
            if tf_writer is not None and tf_log_every and (total_steps % tf_log_every == 0):
                with tf_writer.as_default():
                    tf.summary.scalar("train/loss", display_loss, step=global_step)
                    tf.summary.scalar("train/step_tokens", int((labels != -100).sum().item()), step=global_step)
                    tf.summary.scalar("train/lr", optimizer.param_groups[0]["lr"], step=global_step)
                    if method == "olora":
                        tf.summary.scalar("train/olora_orth_loss", float(orth_loss.detach().item()), step=global_step)
                        tf.summary.scalar("train/olora_l2_loss", float(l2_loss.detach().item()), step=global_step)

        if args.max_train_steps is not None and total_steps >= args.max_train_steps:
            break

    # Flush remaining accumulated gradients when total_steps is not divisible by grad_accum_steps.
    if total_steps > 0 and (total_steps % args.grad_accum_steps != 0):
        torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    if pbar is not None:
        pbar.close()

    duration = time.time() - start_time
    train_model.config.use_cache = use_cache_flag

    loss_mean = total_loss / max(1, total_steps)
    trainable_param_count = sum(p.numel() for p in trainable_params)
    olora_orth_mean = (olora_orth_total / olora_reg_steps) if olora_reg_steps else None
    olora_l2_mean = (olora_l2_total / olora_reg_steps) if olora_reg_steps else None

    if method in ("lora", "mora", "olora", "inflora"):
        if method == "olora":
            final_olora_a, _ = collect_lora_factors(train_model)
            append_olora_subspace(method_ctx, final_olora_a)

        merged_model = train_model

        # get_peft_model can be applied multiple times (q/k/gate), creating nested wrappers.
        # Merge repeatedly until adapters are fully removed.
        for _ in range(8):
            if not hasattr(merged_model, "merge_and_unload"):
                break
            next_model = merged_model.merge_and_unload()
            if next_model is merged_model:
                break
            merged_model = next_model

        # Final safety unwrapping if a PEFT wrapper still remains.
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
            except AttributeError:
                merged_model.peft_config = None
        if hasattr(merged_model, "active_adapter"):
            try:
                delattr(merged_model, "active_adapter")
            except AttributeError:
                merged_model.active_adapter = None
        merged_model.to(device)
        result_model = merged_model
        if method == "inflora":
            # Official InfLoRA: update DualGPM after finishing the current task.
            update_inflora_state_after_task(
                model=result_model,
                dataloader=calib_loader,  # current-task data, shuffle=False
                device=device,
                layer_indices=method_ctx["layer_indices"],
                ffn_module_names=method_ctx["ffn_module_names"],
                attn_module_names=method_ctx["attn_module_names"],
                inflora_state=method_ctx["inflora_state"],
                task_idx=int(method_ctx.get("inflora_task_idx", 0)),
                total_sessions=int(method_ctx.get("inflora_total_sessions", 1)),
                lamb=float(args.inflora_lamb),
                lame=float(args.inflora_lame),
            )
    elif method == "curb":
        merge_curb(train_model)
        result_model = strip_curb(train_model)
        result_model.to(device)
    elif method == "curlora":
        merge_curlora(train_model)
        result_model = strip_curlora(train_model)
        result_model.to(device)
    elif method == "bilora":
        merge_bilora(train_model)
        result_model = strip_bilora(train_model)
        result_model.to(device)
    elif method in ("lorac", "lorac_ipc"):
        merge_lorac(train_model)
        result_model = strip_lorac(train_model)
        result_model.to(device)
    else:
        result_model = train_model

    return result_model, {
        "train_steps": total_steps,
        "loss_mean": loss_mean,
        "tokens": total_tokens,
        "duration_sec": duration,
        "learning_rate": current_lr,
        "global_step_end": tf_global_step_start + total_steps,
        "trainable_params": trainable_param_count,
        "olora_orth_mean": olora_orth_mean,
        "olora_l2_mean": olora_l2_mean,
    }


def evaluate_tasks(model, tokenizer, task_names, task_config, fewshot_seed, bootstrap_iters):
    results = {}
    per_task = {}
    total_time = 0.0

    model.eval()

    with torch.inference_mode():
        for task_name in task_names:
            if task_name not in task_config:
                raise ValueError(f"Missing task config for: {task_name}")
            cfg = task_config[task_name]
            start = time.time()
            outputs = simple_evaluate(
                model=huggingface.HFLM(
                    pretrained=model,
                    backend="causal",
                    tokenizer=tokenizer,
                    trust_remote_code=True,
                ),
                tasks=[task_name],
                limit=cfg["limit"],
                num_fewshot=cfg["fewshot"],
                fewshot_random_seed=fewshot_seed,
                random_seed=fewshot_seed,
                numpy_random_seed=fewshot_seed,
                torch_random_seed=fewshot_seed,
                bootstrap_iters=bootstrap_iters,
                log_samples=False,
            )
            elapsed = time.time() - start
            total_time += elapsed

            res = outputs["results"].get(task_name)
            if res is None:
                res = outputs.get("groups", {}).get(task_name)
            if res is None:
                raise ValueError(f"Result for task/group not found: {task_name}")

            metric_key, stderr_key = _pick_primary_metric_keys(res)
            acc = res.get(metric_key)
            stderr = res.get(stderr_key, None)

            per_task[task_name] = {
                "metric_key": metric_key,
                "accuracy": acc,
                "stderr": stderr,
                "metrics": res,
                "eval_time_sec": elapsed,
            }
            results[task_name] = acc

    acc_values = [v for v in results.values() if v is not None]
    mean_acc = float(np.mean(acc_values)) if acc_values else None
    return per_task, mean_acc, total_time


def _write_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _ensure_csv(path, fieldnames):
    file_exists = os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    return f, writer


def _write_calibration_file(samples, max_samples, seed, out_dir):
    rng = random.Random(seed)
    if not samples:
        return None, 0
    take = min(max_samples, len(samples))
    selected = rng.sample(samples, take)
    path = os.path.join(out_dir, f"calib_{seed}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for sample in selected:
            f.write(json.dumps({"text": sample["text"]}, ensure_ascii=True) + "\n")
    return path, take


def _load_checkpoint_model(checkpoint_dir, model_dtype, device):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=model_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    return model, tokenizer


def eval_worker(
    mode,
    gpu_id,
    job_queue,
    args,
    task_config,
    eval_fields,
    eval_csv_path,
    eval_jsonl_path,
    eval_lock,
    done_counts,
    eval_progress,
    model_dtype,
    run_dir,
    total_tasks,
    tasks_per_round,
):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    device = torch.device(f"cuda:{gpu_id}" if args.device == "cuda" else args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    logs_dir = os.path.join(run_dir, "logs")
    eval_log_path = os.path.join(logs_dir, f"eval_{mode}.log")
    _redirect_output(eval_log_path)
    _log(f"[eval/{mode}] worker start (gpu={gpu_id}) log={eval_log_path}")
    _log_console(f"[eval/{mode}] worker start (gpu={gpu_id})")

    tf_writer = tf.summary.create_file_writer(os.path.join(run_dir, "tf", f"mode_{mode}"))
    first_acc = {}
    current_round = None

    while True:
        job = job_queue.get()
        if job is None:
            break

        job_round = job["round"]
        if job_round != current_round:
            first_acc = {}
            current_round = job_round

        job_start = time.time()
        task_key = job["task_key"]
        task_step = job["task_step"]
        trained_task = job["trained_task"]
        learned_tasks = job["learned_tasks"]
        eval_seed = job["eval_seed"]
        global_step = job["global_step"]

        _log(f"[eval/{mode}] start round {job['round']} task {task_step}/{tasks_per_round} ({trained_task})")
        _log_console(f"[eval/{mode}] start round {job['round']} task {task_step}/{tasks_per_round} ({trained_task})")

        model, tokenizer = _load_checkpoint_model(job["checkpoint_dir"], model_dtype, device)
        eval_stats, mean_acc, total_eval_time = evaluate_tasks(
            model, tokenizer, learned_tasks, task_config, eval_seed, job["bootstrap_iters"]
        )

        for tname, tstats in eval_stats.items():
            if tname not in first_acc and tstats["accuracy"] is not None:
                first_acc[tname] = tstats["accuracy"]

        bwt = None
        if len(learned_tasks) > 1:
            deltas = []
            for tname in learned_tasks[:-1]:
                curr = eval_stats.get(tname, {}).get("accuracy")
                init = first_acc.get(tname)
                if curr is not None and init is not None:
                    deltas.append(curr - init)
            if deltas:
                bwt = float(np.mean(deltas))

        param_count = count_params(model)
        param_bytes = _param_bytes(model)
        mem_stats = _get_gpu_mem_stats(device)
        mem_text = _format_mem_stats(mem_stats)
        mean_text = f"{mean_acc:.4f}" if mean_acc is not None else "n/a"
        bwt_text = f"{bwt:.4f}" if bwt is not None else "n/a"
        _log_console(f"[eval/{mode}] summary round {job['round']} task {task_step} "
                     f"mean_acc={mean_text} bwt={bwt_text} {mem_text}")

        with tf_writer.as_default():
            if mean_acc is not None:
                tf.summary.scalar(f"{mode}/mean_acc", mean_acc, step=global_step)
            if bwt is not None:
                tf.summary.scalar(f"{mode}/bwt", bwt, step=global_step)
            for tname, tstats in eval_stats.items():
                if tstats["accuracy"] is not None:
                    tf.summary.scalar(f"{mode}/acc/{tname}", tstats["accuracy"], step=global_step)
        tf_writer.flush()

        for eval_task, tstats in eval_stats.items():
            row = {
                "timestamp": datetime.now().isoformat(),
                "round": job["round"],
                "seed": job["seed"],
                "task_step": task_step,
                "trained_task": trained_task,
                "method": args.method,
                "eval_task": eval_task,
                "mode": mode,
                "metric_key": tstats["metric_key"],
                "accuracy": tstats["accuracy"],
                "stderr": tstats["stderr"],
                "mean_acc": mean_acc,
                "bwt": bwt,
                "eval_time_sec": tstats["eval_time_sec"],
                "eval_total_time_sec": total_eval_time,
                "num_train_samples": job["train_samples"],
                "num_replay_used": job["replay_used"],
                "train_loss_mean": job["train_loss_mean"],
                "train_tokens": job["train_tokens"],
                "train_steps": job["train_steps"],
                "model_params": param_count,
                "model_param_bytes": param_bytes,
                **mem_stats,
            }
            with eval_lock:
                eval_f, eval_writer = _ensure_csv(eval_csv_path, eval_fields)
                eval_writer.writerow(row)
                eval_f.flush()
                eval_f.close()
                _write_jsonl(eval_jsonl_path, {
                    **row,
                    "metrics": tstats["metrics"],
                    "subtask_counts": job["subtask_counts"],
                })

        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        with eval_lock:
            done_counts[task_key] = done_counts.get(task_key, 0) + 1
            eval_progress[f"{mode}_count"] = eval_progress.get(f"{mode}_count", 0) + 1
            eval_progress[f"{mode}_time"] = eval_progress.get(f"{mode}_time", 0.0) + (time.time() - job_start)

        elapsed = time.time() - job_start
        _log(f"[eval/{mode}] done round {job['round']} task {task_step} ({trained_task}) "
             f"in {_format_duration(elapsed)} "
             f"({eval_progress.get(f'{mode}_count', 0)}/{total_tasks})")
        _log_console(f"[eval/{mode}] done round {job['round']} task {task_step} ({trained_task}) "
                     f"in {_format_duration(elapsed)} "
                     f"({eval_progress.get(f'{mode}_count', 0)}/{total_tasks})")

    _log(f"[eval/{mode}] worker done")
    _log_console(f"[eval/{mode}] worker done")


def _cleanup_completed(pending, done_counts, expected_done):
    for key, info in list(pending.items()):
        if done_counts.get(key, 0) >= expected_done:
            calib_file = info.get("calib_file")
            if calib_file and os.path.exists(calib_file):
                os.remove(calib_file)
            pending.pop(key, None)


def _run_eval_job_sync_base(
    job,
    args,
    model,
    tokenizer,
    task_config,
    eval_fields,
    eval_csv_path,
    eval_jsonl_path,
    eval_progress,
    done_counts,
    total_tasks,
    tasks_per_round,
    device,
    tf_writer,
    first_acc,
):
    mode = "base"
    if device.type == "cuda":
        torch.cuda.set_device(device)

    job_start = time.time()
    task_key = job["task_key"]
    task_step = job["task_step"]
    trained_task = job["trained_task"]
    learned_tasks = job["learned_tasks"]
    eval_seed = job["eval_seed"]
    global_step = job["global_step"]

    _log(f"[eval/{mode}] start round {job['round']} task {task_step}/{tasks_per_round} ({trained_task})")
    _log_console(f"[eval/{mode}] start round {job['round']} task {task_step}/{tasks_per_round} ({trained_task})")

    original_mode_training = model.training
    original_device = next(model.parameters()).device
    moved = original_device != device
    if moved:
        model.to(device)

    eval_stats, mean_acc, total_eval_time = evaluate_tasks(
        model, tokenizer, learned_tasks, task_config, eval_seed, job["bootstrap_iters"]
    )

    for tname, tstats in eval_stats.items():
        if tname not in first_acc and tstats["accuracy"] is not None:
            first_acc[tname] = tstats["accuracy"]

    bwt = None
    if len(learned_tasks) > 1:
        deltas = []
        for tname in learned_tasks[:-1]:
            curr = eval_stats.get(tname, {}).get("accuracy")
            init = first_acc.get(tname)
            if curr is not None and init is not None:
                deltas.append(curr - init)
        if deltas:
            bwt = float(np.mean(deltas))

    param_count = count_params(model)
    param_bytes = _param_bytes(model)
    mem_stats = _get_gpu_mem_stats(device)
    mem_text = _format_mem_stats(mem_stats)
    mean_text = f"{mean_acc:.4f}" if mean_acc is not None else "n/a"
    bwt_text = f"{bwt:.4f}" if bwt is not None else "n/a"
    _log_console(f"[eval/{mode}] summary round {job['round']} task {task_step} "
                 f"mean_acc={mean_text} bwt={bwt_text} {mem_text}")

    with tf_writer.as_default():
        if mean_acc is not None:
            tf.summary.scalar(f"{mode}/mean_acc", mean_acc, step=global_step)
        if bwt is not None:
            tf.summary.scalar(f"{mode}/bwt", bwt, step=global_step)
        for tname, tstats in eval_stats.items():
            if tstats["accuracy"] is not None:
                tf.summary.scalar(f"{mode}/acc/{tname}", tstats["accuracy"], step=global_step)
    tf_writer.flush()

    for eval_task, tstats in eval_stats.items():
        row = {
            "timestamp": datetime.now().isoformat(),
            "round": job["round"],
            "seed": job["seed"],
            "task_step": task_step,
            "trained_task": trained_task,
            "method": args.method,
            "eval_task": eval_task,
            "mode": mode,
            "metric_key": tstats["metric_key"],
            "accuracy": tstats["accuracy"],
            "stderr": tstats["stderr"],
            "mean_acc": mean_acc,
            "bwt": bwt,
            "eval_time_sec": tstats["eval_time_sec"],
            "eval_total_time_sec": total_eval_time,
            "num_train_samples": job["train_samples"],
            "num_replay_used": job["replay_used"],
            "train_loss_mean": job["train_loss_mean"],
            "train_tokens": job["train_tokens"],
            "train_steps": job["train_steps"],
            "model_params": param_count,
            "model_param_bytes": param_bytes,
            **mem_stats,
        }
        eval_f, eval_writer = _ensure_csv(eval_csv_path, eval_fields)
        eval_writer.writerow(row)
        eval_f.flush()
        eval_f.close()
        _write_jsonl(eval_jsonl_path, {
            **row,
            "metrics": tstats["metrics"],
            "subtask_counts": job["subtask_counts"],
        })

    if moved:
        model.to(original_device)
    if original_mode_training:
        model.train()
    if original_device.type == "cuda":
        torch.cuda.set_device(original_device)
        torch.cuda.empty_cache()

    done_counts[task_key] = done_counts.get(task_key, 0) + 1
    eval_progress[f"{mode}_count"] = eval_progress.get(f"{mode}_count", 0) + 1
    eval_progress[f"{mode}_time"] = eval_progress.get(f"{mode}_time", 0.0) + (time.time() - job_start)

    elapsed = time.time() - job_start
    _log(f"[eval/{mode}] done round {job['round']} task {task_step} ({trained_task}) "
         f"in {_format_duration(elapsed)} "
         f"({eval_progress.get(f'{mode}_count', 0)}/{total_tasks})")
    _log_console(f"[eval/{mode}] done round {job['round']} task {task_step} ({trained_task}) "
                 f"in {_format_duration(elapsed)} "
                 f"({eval_progress.get(f'{mode}_count', 0)}/{total_tasks})")


def main():
    parser = argparse.ArgumentParser(description="Continual Learning with CURb/PEFT evaluation")

    # Model and Paths
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B",
                        help="Base model name/path.")
    parser.add_argument("--model_dtype", type=str, default="fp32",
                        choices=["auto", "bf16", "fp16", "fp32"],
                        help="Model weight dtype to load (auto uses checkpoint dtype).")
    parser.add_argument("--save_path", type=str, default="./cl_runs",
                        help="Root directory to save outputs.")
    parser.add_argument("--model_save_path", type=str, default=None,
                        help="Optional root directory to save model checkpoints.")
    parser.add_argument("--device", type=str, default="cuda",
                        help='Device to run the computations on (e.g., "cpu", "cuda").')
    parser.add_argument("--train_gpu", type=int, default=0,
                        help="GPU id for training (default: 0).")
    parser.add_argument("--eval_gpus", type=str, default="1",
                        help="Comma-separated GPU ids for evaluation workers.")
    parser.add_argument("--eval_pipeline", type=str, default="sync",
                        choices=["sync", "async"],
                        help="Evaluation scheduling: sync=train->eval serial, async=background eval worker.")

    # Tasks
    parser.add_argument("--tasks", type=str, nargs="+", default=DEFAULT_TASKS,
                        help="Task order for continual learning.")
    parser.add_argument("--num_test_steps", type=int, default=128,
                        help="Base number of test steps for eval limits.")

    # Training
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Max sequence length for training.")
    parser.add_argument("--train_samples_per_task", type=int, default=4096,
                        help="Training samples per task.")
    parser.add_argument("--replay_buffer_per_task", type=int, default=0,
                        help="Replay buffer size per task (used only for CURb calibration, not for training).")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--learning_rate_first_task", type=float, default=None,
                        help="Optional LR override for task_step=1 in each round.")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                 "constant_with_warmup"],
                        help="LR scheduler type.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio over optimizer update steps.")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--train_tf_log_every", type=int, default=10,
                        help="Log train scalars every N steps to TensorBoard.")
    parser.add_argument("--mp_start_method", type=str, default="spawn",
                        choices=["fork", "spawn", "forkserver"],
                        help="Multiprocessing start method (Linux only).")

    # Adapter / CURb
    parser.add_argument("--method", type=str, default="curb",
                        choices=["curb", "curlora", "bilora", "lora", "mora", "olora", "inflora", "lorac", "lorac_ipc"],
                        help="Adapter method for continual learning.")
    parser.add_argument("--curb_rank", type=int, default=256,
                        help="CURb rank (U is r x r).")
    parser.add_argument("--curb_rank_q", type=int, default=None,
                        help="Optional CURb/CURLoRA rank override for q_proj.")
    parser.add_argument("--curb_rank_k", type=int, default=None,
                        help="Optional CURb/CURLoRA rank override for k_proj.")
    parser.add_argument("--curb_rank_gate", type=int, default=None,
                        help="Optional CURb/CURLoRA rank override for gate_proj.")
    parser.add_argument("--curb_basis_mode", type=str, default="cov_fast",
                        choices=["cov_fast", "weight", "hybrid"],
                        help="CURb basis selection: cov_fast(S), weight(W), or hybrid(rows from S, cols from W).")
    parser.add_argument("--curb_deim_importance_order", type=str, default="high",
                        choices=["high", "low"],
                        help="DEIM selector order for CURb basis: high (top-importance) or low (inverse-importance).")
    parser.add_argument("--curb_update_whiten", type=str, default="none",
                        choices=["none", "diag"],
                        help="Optional CURb whitening: none or diag (diag 2nd moment + r-space whitening).")
    parser.add_argument("--curb_whiten_ridge_ratio", type=float, default=1e-4,
                        help="Relative ridge for whitening: eps = abs + ratio * trace(G)/r.")
    parser.add_argument("--curb_whiten_ridge_abs", type=float, default=1e-12,
                        help="Absolute ridge floor for whitening.")
    parser.add_argument("--curb_alpha", type=float, default=None,
                        help="Optional fixed adapter scaling for CURb/CURLoRA (default: 2 * effective rank).")
    parser.add_argument("--curb_calib_steps", type=int, default=256)
    parser.add_argument("--curb_batch_size", type=int, default=1)
    parser.add_argument("--curb_max_length", type=int, default=4096)
    parser.add_argument("--curb_calib_category", type=str, default="en")
    parser.add_argument(
        "--curb_calib_source",
        type=str,
        default="c4",
        choices=["c4", "replay_mix_c4"],
        help=(
            "Calibration source for CURb cov_fast/hybrid activation stats. "
            "'c4' uses C4 only. 'replay_mix_c4' uses replay buffer texts (prompt+target) first, "
            "then fills the remaining sequences with C4 to match curb_calib_steps*curb_batch_size."
        ),
    )
    parser.add_argument("--curb_basis_cache", type=str, default=None,
                        help="Optional path to cache CURb basis.")
    parser.add_argument("--curb_ffn_module_names", nargs="*", default=["gate_proj"])
    parser.add_argument("--curb_attn_module_names", nargs="*", default=["q_proj", "k_proj"])

    # BiLoRA (Fourier-domain sparse bilinear update)
    parser.add_argument("--bilora_k", type=int, default=None,
                        help="Optional number of active frequency components per module (default: r_eff^2).")
    parser.add_argument("--bilora_alpha", type=float, default=None,
                        help="Optional scaling for BiLoRA delta-W (default: 0.5*sqrt(out*in)).")
    parser.add_argument("--bilora_seed", type=int, default=777,
                        help="Random seed for BiLoRA frequency support sampling.")
    parser.add_argument("--bilora_chunk_size", type=int, default=256,
                        help="Token chunk size for BiLoRA forward (memory control).")
    parser.add_argument("--bilora_freq_chunk_size", type=int, default=8192,
                        help="Frequency chunk size for BiLoRA forward (memory control).")
    parser.add_argument("--lora_alpha", type=float, default=None,
                        help="Optional fixed alpha override; default uses 2 * effective rank per module.")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_rank_q", type=int, default=None,
                        help="Optional LoRA/MoRA rank override for q_proj.")
    parser.add_argument("--lora_rank_k", type=int, default=None,
                        help="Optional LoRA/MoRA rank override for k_proj.")
    parser.add_argument("--lora_rank_gate", type=int, default=None,
                        help="Optional LoRA/MoRA rank override for gate_proj.")
    parser.add_argument("--olora_lambda_orth", type=float, default=0.5,
                        help="O-LoRA orthogonality regularization coefficient.")
    parser.add_argument("--olora_lambda_l2", type=float, default=0.0,
                        help="O-LoRA L2 regularization coefficient for current LoRA factors.")

    # LoRAC / LoRAC-IPC (official-style defaults)
    parser.add_argument("--lorac_ortho", type=float, default=1.0, help="LoRAC orthogonal regularization coefficient.")
    parser.add_argument(
        "--lorac_omega_lr_scale",
        type=float,
        default=1.0,
        help="Optimizer LR scale applied only to omega parameters (official: omega_lr_scale).",
    )
    parser.add_argument("--lorac_ipc_beta1", type=float, default=0.85, help="IPC beta1 (sensitivity EMA).")
    parser.add_argument("--lorac_ipc_beta2", type=float, default=0.85, help="IPC beta2 (uncertainty EMA).")
    parser.add_argument("--lorac_ipc_threshold", type=float, default=0.1, help="IPC mask fraction in [0,1].")
    parser.add_argument("--lorac_ipc_new_mask", action="store_true", help="IPC new_mask behavior (official).")

    # InfLoRA (official defaults)
    parser.add_argument("--inflora_lamb", type=float, default=0.95,
                        help="DualGPM threshold schedule start (official default: 0.95).")
    parser.add_argument("--inflora_lame", type=float, default=1.0,
                        help="DualGPM threshold schedule end (official default: 1.0).")
    parser.add_argument(
        "--inflora_calib_source",
        type=str,
        default="train",
        choices=["train", "c4"],
        help=(
            "InfLoRA calibration source for B_t design and DualGPM update. "
            "'train' uses current-task training samples (official). "
            "'c4' uses external C4-only sequences with count = curb_calib_steps*curb_batch_size (ablation)."
        ),
    )

    # Misc
    parser.add_argument("--total_round", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap_iters", type=int, default=100000)

    args = parser.parse_args()

    if args.eval_pipeline == "async" and args.device == "cuda" and args.mp_start_method == "fork":
        raise ValueError("CUDA + multiprocessing requires --mp_start_method spawn or forkserver.")

    if args.eval_pipeline == "async" and args.mp_start_method and sys.platform != "win32":
        try:
            mp.set_start_method(args.mp_start_method, force=True)
        except RuntimeError:
            pass

    for rank_arg in (
        "lora_rank_q", "lora_rank_k", "lora_rank_gate",
        "curb_rank_q", "curb_rank_k", "curb_rank_gate",
    ):
        rank_val = getattr(args, rank_arg)
        if rank_val is not None and int(rank_val) < 1:
            raise ValueError(f"--{rank_arg} must be >= 1 (got {rank_val}).")
    if args.olora_lambda_orth < 0:
        raise ValueError(f"--olora_lambda_orth must be >= 0 (got {args.olora_lambda_orth}).")
    if args.olora_lambda_l2 < 0:
        raise ValueError(f"--olora_lambda_l2 must be >= 0 (got {args.olora_lambda_l2}).")
    if args.lorac_ortho < 0:
        raise ValueError("--lorac_ortho must be >= 0.")
    if args.lorac_omega_lr_scale <= 0:
        raise ValueError("--lorac_omega_lr_scale must be > 0.")
    for beta_name in ("lorac_ipc_beta1", "lorac_ipc_beta2"):
        beta_val = float(getattr(args, beta_name))
        if beta_val <= 0.0 or beta_val >= 1.0:
            raise ValueError(f"--{beta_name} must be in (0,1) (got {beta_val}).")
    if args.lorac_ipc_threshold < 0.0 or args.lorac_ipc_threshold > 1.0:
        raise ValueError("--lorac_ipc_threshold must be in [0, 1].")
    if args.inflora_lamb <= 0 or args.inflora_lamb > 1:
        raise ValueError(f"--inflora_lamb must be in (0, 1] (got {args.inflora_lamb}).")
    if args.inflora_lame <= 0 or args.inflora_lame > 1:
        raise ValueError(f"--inflora_lame must be in (0, 1] (got {args.inflora_lame}).")
    if args.inflora_lame < args.inflora_lamb:
        raise ValueError(
            f"--inflora_lame must be >= --inflora_lamb (got lame={args.inflora_lame}, lamb={args.inflora_lamb})."
        )
    if args.learning_rate_first_task is not None and args.learning_rate_first_task <= 0:
        raise ValueError(
            f"--learning_rate_first_task must be > 0 (got {args.learning_rate_first_task})."
        )

    # Handle device selection
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        train_device = torch.device(f"cuda:{args.train_gpu}")
        torch.cuda.set_device(train_device)
    else:
        train_device = torch.device(args.device)

    # Exclude MMLU from training/evaluation
    args.tasks = [t for t in args.tasks if not (t == "mmlu" or t.startswith("mmlu_"))]

    # Setup run directory
    os.makedirs(args.save_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_path, f"cl_run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    model_root = run_dir
    if args.model_save_path:
        os.makedirs(args.model_save_path, exist_ok=True)
        model_root = os.path.join(args.model_save_path, f"cl_run_{run_id}")
        os.makedirs(model_root, exist_ok=True)

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=True)

    eval_csv_path = os.path.join(run_dir, "eval_metrics.csv")
    eval_jsonl_path = os.path.join(run_dir, "eval_metrics.jsonl")
    train_csv_path = os.path.join(run_dir, "train_metrics.csv")
    train_jsonl_path = os.path.join(run_dir, "train_metrics.jsonl")

    eval_fields = [
        "timestamp", "round", "seed", "task_step", "trained_task", "method",
        "eval_task", "mode", "metric_key", "accuracy",
        "stderr", "mean_acc", "bwt", "eval_time_sec", "eval_total_time_sec",
        "num_train_samples",
        "num_replay_used", "train_loss_mean",
        "train_tokens", "train_steps", "model_params", "model_param_bytes",
        "gpu_mem_alloc", "gpu_mem_peak",
    ]
    train_fields = [
        "timestamp", "round", "seed", "task_step", "trained_task", "method",
        "train_samples", "replay_used", "train_steps", "learning_rate", "loss_mean",
        "tokens", "duration_sec", "trainable_params",
        "olora_orth_mean", "olora_l2_mean",
        "gpu_mem_alloc", "gpu_mem_peak",
    ]

    train_f, train_writer = _ensure_csv(train_csv_path, train_fields)
    train_tf_writer = tf.summary.create_file_writer(os.path.join(run_dir, "tf", "train"))

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    model_dtype = "auto" if args.model_dtype == "auto" else dtype_map[args.model_dtype]

    task_config = _build_task_config(args.num_test_steps)
    eval_gpus = _parse_eval_gpus(args.eval_gpus)
    if len(eval_gpus) != len(EVAL_MODES):
        raise ValueError(f"--eval_gpus must provide {len(EVAL_MODES)} GPU ids.")

    total_tasks = len(args.tasks) * args.total_round
    run_start = time.time()
    train_times = []
    global_train_step = 0
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    train_log_path = os.path.join(logs_dir, "train.log")

    use_async_eval = (args.eval_pipeline == "async")
    eval_lock = mp.Lock() if use_async_eval else None
    if use_async_eval:
        manager = mp.Manager()
        done_counts = manager.dict()
        eval_progress = manager.dict()
    else:
        done_counts = {}
        eval_progress = {f"{mode}_count": 0 for mode in EVAL_MODES}
        for mode in EVAL_MODES:
            eval_progress[f"{mode}_time"] = 0.0
    expected_done = len(EVAL_MODES)
    summarized_rounds = set()

    def _round_eval_done(round_id: int) -> bool:
        for task_step in range(1, len(args.tasks) + 1):
            task_key = f"r{round_id}_t{task_step}"
            if done_counts.get(task_key, 0) < expected_done:
                return False
        return True

    def _update_summary_if_round_done(max_round: int) -> None:
        for round_id in range(1, max_round + 1):
            if round_id in summarized_rounds:
                continue
            if not _round_eval_done(round_id):
                continue
            round_summary_dir = os.path.join(run_dir, "summary_task_mode", f"round_{round_id}")
            os.makedirs(round_summary_dir, exist_ok=True)
            summary_path = os.path.join(round_summary_dir, "summary_task_mode.csv")
            if eval_lock is not None:
                with eval_lock:
                    wrote = _summarize_round_metrics(eval_csv_path, summary_path)
            else:
                wrote = _summarize_round_metrics(eval_csv_path, summary_path)
            if wrote:
                _log(f"[summary] wrote {summary_path} after round {round_id}")
                _log_console(f"[summary] wrote {summary_path} after round {round_id}")
            else:
                _log(f"[summary] skipped (no eval data) after round {round_id}")
                _log_console(f"[summary] skipped (no eval data) after round {round_id}")
            summarized_rounds.add(round_id)

    mode_gpu = dict(zip(EVAL_MODES, eval_gpus))
    job_queues = {}
    workers = []
    sync_tf_writers = {}
    sync_eval_devices = {}
    if use_async_eval:
        for mode in EVAL_MODES:
            q = mp.Queue()
            p = mp.Process(
                target=eval_worker,
                args=(
                    mode,
                    mode_gpu[mode],
                    q,
                    args,
                    task_config,
                    eval_fields,
                    eval_csv_path,
                    eval_jsonl_path,
                    eval_lock,
                    done_counts,
                    eval_progress,
                    model_dtype,
                    run_dir,
                    len(args.tasks) * args.total_round,
                    len(args.tasks),
                ),
            )
            p.start()
            job_queues[mode] = q
            workers.append(p)
    else:
        for mode in EVAL_MODES:
            sync_tf_writers[mode] = tf.summary.create_file_writer(os.path.join(run_dir, "tf", f"mode_{mode}"))
            sync_eval_devices[mode] = (
                torch.device(f"cuda:{mode_gpu[mode]}") if args.device == "cuda" else torch.device(args.device)
            )

    _redirect_output(train_log_path)
    _log(f"Run {run_id} start | rounds={args.total_round} tasks/round={len(args.tasks)} "
         f"total_tasks={total_tasks} train_gpu={args.train_gpu} eval_gpus={args.eval_gpus} "
         f"eval_pipeline={args.eval_pipeline} "
         f"log={train_log_path}")
    _log_console(f"Run {run_id} start | rounds={args.total_round} tasks/round={len(args.tasks)} "
                 f"total_tasks={total_tasks} train_gpu={args.train_gpu} eval_gpus={args.eval_gpus} "
                 f"eval_pipeline={args.eval_pipeline}")

    pending_tasks = {}
    method_ctx = {"method": args.method, "repo_root": repo_root}
    curb_ranks = None
    curb_rank_overrides = {
        "attn_q_proj": args.curb_rank_q,
        "attn_k_proj": args.curb_rank_k,
        "mlp_gate_proj": args.curb_rank_gate,
    }
    curb_rank_override_active = any(v is not None for v in curb_rank_overrides.values())
    if not curb_rank_override_active:
        curb_rank_overrides = None
    lora_ranks = None
    lora_rank_overrides = {
        "attn_q_proj": args.lora_rank_q,
        "attn_k_proj": args.lora_rank_k,
        "mlp_gate_proj": args.lora_rank_gate,
    }
    rank_override_active = any(v is not None for v in lora_rank_overrides.values())
    if not rank_override_active:
        lora_rank_overrides = None
    target_modules_q = None
    target_modules_k = None
    target_modules_gate = None

    # Load tokenizer/model once per round (reset per round)
    for round_idx in range(args.total_round):
        round_num = round_idx + 1
        round_seed = args.seed + round_idx
        if use_async_eval:
            round_first_acc = None
        else:
            round_first_acc = {mode: {} for mode in EVAL_MODES}
        set_seed(round_seed)
        random.seed(round_seed)
        np.random.seed(round_seed)
        _log(f"Round {round_num}/{args.total_round} start (seed={round_seed})")
        _log_console(f"Round {round_num}/{args.total_round} start (seed={round_seed})")

        task_manager = TaskManager()
        fixed_indices_path = os.path.join(run_dir, "fixed_indices", f"round_{round_num}", "indices.json")

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=model_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        model.to(train_device)

        layer_indices = list(range(model.config.num_hidden_layers - 1))
        method_ctx["layer_indices"] = layer_indices
        method_ctx["ffn_module_names"] = list(args.curb_ffn_module_names)
        method_ctx["attn_module_names"] = list(args.curb_attn_module_names)
        method_ctx["curb_rank"] = int(args.curb_rank)
        if curb_ranks is None and (args.method in ("curb", "curlora", "bilora") or curb_rank_override_active):
            curb_ranks = _compute_curb_ranks(
                model,
                args.curb_rank,
                args.curb_ffn_module_names,
                args.curb_attn_module_names,
                rank_overrides=curb_rank_overrides,
            )
        if args.method in ("curb", "curlora", "bilora"):
            method_ctx["curb_ranks"] = curb_ranks
            method_ctx["curb_rank_tag"] = _curb_rank_cache_tag(args.curb_rank, curb_ranks)
        if args.method == "olora":
            method_ctx["olora_prev_A"] = {}

        if args.method == "inflora":
            method_ctx["inflora_state"] = init_inflora_state()
            method_ctx["inflora_total_sessions"] = int(len(args.tasks))

        if args.method in ("lora", "mora", "olora", "inflora", "lorac", "lorac_ipc"):
            if lora_ranks is None:
                lora_ranks = _compute_lora_ranks(
                    model,
                    args.curb_rank,
                    args.curb_ffn_module_names,
                    args.curb_attn_module_names,
                    rank_overrides=lora_rank_overrides,
                )
                target_modules_q, target_modules_k, target_modules_gate = _build_target_module_lists(layer_indices)
            method_ctx["lora_ranks"] = lora_ranks
            method_ctx["target_modules_q"] = target_modules_q
            method_ctx["target_modules_k"] = target_modules_k
            method_ctx["target_modules_gate"] = target_modules_gate
            if args.method == "inflora":
                method_ctx["inflora_ranks"] = _compute_inflora_ranks_match_trainable(
                    model,
                    lora_ranks=lora_ranks,
                    ffn_module_names=args.curb_ffn_module_names,
                    attn_module_names=args.curb_attn_module_names,
                )
            if args.method in ("lorac", "lorac_ipc"):
                method_ctx["lorac_state"] = init_lorac_state(pool_size=int(len(args.tasks)))

        if round_idx == 0:
            rows, per_layer = _compute_param_budget_table(
                model,
                args.curb_rank,
                args.curb_ffn_module_names,
                args.curb_attn_module_names,
                lora_ranks=lora_ranks,
                curb_ranks=curb_ranks,
            )
            applied_layers = model.config.num_hidden_layers - 1
            totals = {k: int(v * applied_layers) for k, v in per_layer.items()}
            budget_payload = {
                "model_name": args.model_name,
                "hidden_size": model.config.hidden_size,
                "intermediate_size": getattr(model.config, "intermediate_size", None),
                "num_layers": model.config.num_hidden_layers,
                "applied_layers": applied_layers,
                "curb_rank": int(args.curb_rank),
                "curb_rank_overrides": curb_rank_overrides,
                "curb_rank_override_active": curb_rank_override_active,
                "lora_rank_overrides": lora_rank_overrides,
                "lora_rank_override_active": rank_override_active,
                "per_module": rows,
                "per_layer": per_layer,
                "total": totals,
            }
            budget_path = os.path.join(run_dir, "param_budget.json")
            with open(budget_path, "w", encoding="utf-8") as f:
                json.dump(budget_payload, f, ensure_ascii=True, indent=2)

            budget_csv = os.path.join(run_dir, "param_budget.csv")
            with open(budget_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "module", "in_features", "out_features",
                    "r_eff", "r_lora", "mora_new_r",
                    "curb_params", "lora_params", "olora_params", "mora_params",
                ])
                writer.writeheader()
                for row in rows:
                    out_row = dict(row)
                    out_row["olora_params"] = row.get("lora_params")
                    writer.writerow(out_row)
                writer.writerow({
                    "module": "per_layer_total",
                    "curb_params": per_layer["curb"],
                    "lora_params": per_layer["lora"],
                    "olora_params": per_layer["olora"],
                    "mora_params": per_layer["mora"],
                })
                writer.writerow({
                    "module": "total_all_layers",
                    "curb_params": totals["curb"],
                    "lora_params": totals["lora"],
                    "olora_params": totals["olora"],
                    "mora_params": totals["mora"],
                })

            if args.method in ("lora", "mora", "olora", "inflora", "lorac", "lorac_ipc"):
                rank_source = "override" if rank_override_active else "auto_budget"
                extra = ""
                if args.method == "olora":
                    extra = (f" lambda_orth={args.olora_lambda_orth}"
                             f" lambda_l2={args.olora_lambda_l2}")
                if args.method == "inflora":
                    extra = f" lamb={args.inflora_lamb} lame={args.inflora_lame}"
                if args.method in ("lorac", "lorac_ipc"):
                    extra = (
                        f" alpha={args.lora_alpha} ortho={args.lorac_ortho} "
                        f"omega_lr_scale={args.lorac_omega_lr_scale}"
                    )
                    if args.method == "lorac_ipc":
                        extra = (
                            extra
                            + f" ipc(beta1={args.lorac_ipc_beta1} beta2={args.lorac_ipc_beta2} "
                            + f"thr={args.lorac_ipc_threshold} new_mask={bool(args.lorac_ipc_new_mask)})"
                        )

                active_ranks = lora_ranks
                if args.method == "inflora":
                    active_ranks = method_ctx.get("inflora_ranks", lora_ranks)

                _log(f"[method] {args.method} ranks q={active_ranks['attn_q_proj']} "
                     f"k={active_ranks['attn_k_proj']} gate={active_ranks['mlp_gate_proj']} "
                     f"budget={_curb_param_budget(args.curb_rank)} source={rank_source}{extra}")
                _log_console(f"[method] {args.method} ranks q={active_ranks['attn_q_proj']} "
                             f"k={active_ranks['attn_k_proj']} gate={active_ranks['mlp_gate_proj']} "
                             f"budget={_curb_param_budget(args.curb_rank)} source={rank_source}{extra}")
            elif args.method == "curb":
                alpha_note = "alpha=2*r_eff(auto)" if args.curb_alpha is None else f"alpha={args.curb_alpha}"
                rank_note = _format_rank_triplet(curb_ranks)
                _log(f"[method] curb base_rank={args.curb_rank} ranks({rank_note}) "
                     f"mode={args.curb_basis_mode} deim={args.curb_deim_importance_order} "
                     f"whiten={args.curb_update_whiten} calib_source={args.curb_calib_source} {alpha_note}")
                _log_console(f"[method] curb base_rank={args.curb_rank} ranks({rank_note}) "
                             f"mode={args.curb_basis_mode} deim={args.curb_deim_importance_order} "
                             f"whiten={args.curb_update_whiten} calib_source={args.curb_calib_source} {alpha_note}")
            elif args.method == "curlora":
                alpha_note = "alpha=2*r_eff(auto)" if args.curb_alpha is None else f"alpha={args.curb_alpha}"
                rank_note = _format_rank_triplet(curb_ranks)
                _log(f"[method] curlora base_rank={args.curb_rank} ranks({rank_note}) {alpha_note}")
                _log_console(f"[method] curlora base_rank={args.curb_rank} ranks({rank_note}) {alpha_note}")
            elif args.method == "bilora":
                rank_note = _format_rank_triplet(curb_ranks)
                alpha_note = "alpha=0.5*sqrt(out*in)(auto)" if args.bilora_alpha is None else f"alpha={args.bilora_alpha}"
                k_note = "k=r_eff^2(auto)" if args.bilora_k is None else f"k={args.bilora_k}"
                _log(f"[method] bilora base_rank={args.curb_rank} ranks({rank_note}) {k_note} {alpha_note} seed={int(args.bilora_seed)}")
                _log_console(f"[method] bilora base_rank={args.curb_rank} ranks({rank_note}) {k_note} {alpha_note} seed={int(args.bilora_seed)}")
            else:
                _log(f"[method] {args.method}")
                _log_console(f"[method] {args.method}")

        replay_buffers = {}
        replay_pool = []

        for task_step, task_name in enumerate(args.tasks, start=1):
            _log(f"[train] start round {round_num} task {task_step}/{len(args.tasks)} ({task_name})")
            _log_console(f"[train] start round {round_num} task {task_step}/{len(args.tasks)} ({task_name})")
            # Build training samples for current task
            train_samples, subtask_counts = build_training_samples(
                task_name=task_name,
                task_manager=task_manager,
                tokenizer=tokenizer,
                max_length=args.max_length,
                total_samples=args.train_samples_per_task,
                fewshot_map={k: v["fewshot"] for k, v in task_config.items()},
                base_seed=_seed_from_parts(round_seed, "train", task_name),
            )

            # NOTE: Replay is not used for training in this codebase. We keep a replay
            # buffer only to define the calibration distribution for cov_fast/hybrid.
            mixed_samples = list(train_samples)
            random.Random(_seed_from_parts(round_seed, "mix", task_name)).shuffle(mixed_samples)
            replay_used_count = 0

            if args.method == "curb":
                # Disable CURb basis caching to avoid excessive disk usage.
                cache_path = None
                calib_texts = None
                if args.curb_calib_source == "replay_mix_c4" and (
                    args.curb_basis_mode in ("cov_fast", "hybrid") or args.curb_update_whiten == "diag"
                ):
                    target_sequences = int(args.curb_calib_steps) * int(args.curb_batch_size)
                    calib_rng = random.Random(_seed_from_parts(round_seed, "curb_calib", task_name, task_step))
                    take = min(target_sequences, len(replay_pool))
                    if take > 0:
                        selected = calib_rng.sample(replay_pool, take)
                        calib_texts = [s.get("text", "") for s in selected if s.get("text")]
                    else:
                        calib_texts = []
                    c4_fill = max(0, target_sequences - len(calib_texts))
                    _log(
                        f"[curb_calib] source=replay_mix_c4 target={target_sequences} "
                        f"replay_used={len(calib_texts)} c4_fill={c4_fill} replay_pool={len(replay_pool)}"
                    )
                    _log_console(
                        f"[curb_calib] source=replay_mix_c4 target={target_sequences} "
                        f"replay_used={len(calib_texts)} c4_fill={c4_fill} replay_pool={len(replay_pool)}"
                    )
                method_ctx["curb_basis"] = load_or_build_curb_basis(
                    model=model,
                    tokenizer=tokenizer,
                    device=train_device,
                    layer_indices=layer_indices,
                    ffn_module_names=args.curb_ffn_module_names,
                    attn_module_names=args.curb_attn_module_names,
                    rank=args.curb_rank,
                    mode=args.curb_basis_mode,
                    deim_importance_order=args.curb_deim_importance_order,
                    update_whiten=args.curb_update_whiten,
                    whiten_ridge_ratio=args.curb_whiten_ridge_ratio,
                    whiten_ridge_abs=args.curb_whiten_ridge_abs,
                    rank_overrides=method_ctx.get("curb_ranks"),
                    cache_path=cache_path,
                    calib_steps=args.curb_calib_steps,
                    batch_size=args.curb_batch_size,
                    max_length=args.curb_max_length,
                    dataset_category=args.curb_calib_category,
                    calib_texts=calib_texts,
                )

            task_lr = args.learning_rate
            if task_step == 1 and args.learning_rate_first_task is not None:
                task_lr = float(args.learning_rate_first_task)

            if args.method == "inflora":
                # 0-indexed task id for official InfLoRA threshold schedule / B_t design.
                method_ctx["inflora_task_idx"] = int(task_step - 1)

            if args.method in ("lorac", "lorac_ipc"):
                method_ctx["lorac_task_idx"] = int(task_step - 1)

            if args.method == "bilora":
                # 0-indexed task id for BiLoRA frequency support sampling (official-style: seed + t*10).
                method_ctx["bilora_task_idx"] = int(task_step - 1)

            # Train (LoRA -> merge)
            model, train_stats = train_on_samples(
                model,
                tokenizer,
                mixed_samples,
                args,
                train_device,
                method_ctx,
                task_label=task_name,
                loader_seed=_seed_from_parts(round_seed, "loader", task_name, task_step),
                learning_rate=task_lr,
                tf_writer=train_tf_writer,
                tf_global_step_start=global_train_step,
                tf_log_every=args.train_tf_log_every,
            )
            global_train_step = train_stats.get("global_step_end", global_train_step)
            train_times.append(train_stats["duration_sec"])
            loss_text = f"{train_stats['loss_mean']:.6f}" if train_stats["loss_mean"] is not None else "n/a"
            mem_stats = _get_gpu_mem_stats(train_device)
            mem_text = _format_mem_stats(mem_stats)
            _log(f"[train] done round {round_num} task {task_step} ({task_name}) "
                 f"samples={len(train_samples)} replay=0 "
                 f"steps={train_stats['train_steps']} lr={train_stats.get('learning_rate', args.learning_rate):.3e} "
                 f"loss={loss_text} "
                 f"time={_format_duration(train_stats['duration_sec'])}")
            _log_console(f"[train] done round {round_num} task {task_step} ({task_name}) "
                         f"samples={len(train_samples)} replay=0 "
                         f"steps={train_stats['train_steps']} lr={train_stats.get('learning_rate', args.learning_rate):.3e} "
                         f"loss={loss_text} "
                         f"time={_format_duration(train_stats['duration_sec'])} {mem_text}")
            with train_tf_writer.as_default():
                task_global_step = round_idx * len(args.tasks) + task_step
                if train_stats["loss_mean"] is not None:
                    tf.summary.scalar("train/task_loss_mean", train_stats["loss_mean"], step=task_global_step)
                tf.summary.scalar("train/task_steps", train_stats["train_steps"], step=task_global_step)
                tf.summary.scalar("train/task_tokens", train_stats["tokens"], step=task_global_step)
                tf.summary.scalar("train/task_duration_sec", train_stats["duration_sec"], step=task_global_step)
                tf.summary.scalar("train/task_samples", len(train_samples), step=task_global_step)
                tf.summary.scalar("train/task_replay_used", 0, step=task_global_step)
                if train_stats.get("olora_orth_mean") is not None:
                    tf.summary.scalar("train/task_olora_orth_mean", train_stats["olora_orth_mean"], step=task_global_step)
                if train_stats.get("olora_l2_mean") is not None:
                    tf.summary.scalar("train/task_olora_l2_mean", train_stats["olora_l2_mean"], step=task_global_step)
            train_tf_writer.flush()

            train_row = {
                "timestamp": datetime.now().isoformat(),
                "round": round_num,
                "seed": round_seed,
                "task_step": task_step,
                "trained_task": task_name,
                "method": args.method,
                "train_samples": len(train_samples),
                "replay_used": 0,
                "train_steps": train_stats["train_steps"],
                "learning_rate": train_stats.get("learning_rate"),
                "loss_mean": train_stats["loss_mean"],
                "tokens": train_stats["tokens"],
                "duration_sec": train_stats["duration_sec"],
                "trainable_params": train_stats.get("trainable_params"),
                "olora_orth_mean": train_stats.get("olora_orth_mean"),
                "olora_l2_mean": train_stats.get("olora_l2_mean"),
                **mem_stats,
            }
            train_writer.writerow(train_row)
            train_f.flush()
            _write_jsonl(train_jsonl_path, {
                **train_row,
                "subtask_counts": subtask_counts,
            })

            # Update replay buffer for this task (includes current task)
            if train_samples and int(args.replay_buffer_per_task) > 0:
                rb_rng = random.Random(_seed_from_parts(round_seed, "buffer", task_name))
                take_rb = min(int(args.replay_buffer_per_task), len(train_samples))
                selected = rb_rng.sample(train_samples, take_rb)
                # Store only the text needed for calibration to keep memory bounded.
                buf = []
                for s in selected:
                    ids = s.get("input_ids")
                    if ids is not None:
                        text = tokenizer.decode(ids, skip_special_tokens=True)
                    else:
                        text = s.get("text", "")
                    if not text:
                        continue
                    buf.append({
                        "text": text,
                        "task": task_name,
                        "subtask": s.get("subtask"),
                    })
                replay_buffers[task_name] = buf
            replay_pool = []
            for buf in replay_buffers.values():
                replay_pool.extend(buf)

            # Save checkpoint for evaluation
            ckpt_dir = os.path.join(model_root, "checkpoints", f"round_{round_num}", f"task_{task_step}_{task_name}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            _log(f"[ckpt] saved {ckpt_dir}")
            _log_console(f"[ckpt] saved {ckpt_dir}")

            learned_tasks = args.tasks[:task_step]
            eval_seed = _seed_from_parts(round_seed, "eval", task_name)
            global_step = round_idx * len(args.tasks) + task_step
            task_key = f"r{round_num}_t{task_step}"

            job_base = {
                "task_key": task_key,
                "round": round_num,
                "seed": round_seed,
                "task_step": task_step,
                "trained_task": task_name,
                "learned_tasks": learned_tasks,
                "checkpoint_dir": ckpt_dir,
                "model_root": model_root,
                "eval_seed": eval_seed,
                "global_step": global_step,
                "train_samples": len(train_samples),
                "replay_used": replay_used_count,
                "train_loss_mean": train_stats["loss_mean"],
                "train_tokens": train_stats["tokens"],
                "train_steps": train_stats["train_steps"],
                "subtask_counts": subtask_counts,
                "bootstrap_iters": args.bootstrap_iters,
            }

            if use_async_eval:
                for mode in EVAL_MODES:
                    job = dict(job_base)
                    job_queues[mode].put(job)
                _log(f"[eval] queued round {round_num} task {task_step} "
                     f"({task_name}) modes={','.join(EVAL_MODES)}")
                _log_console(f"[eval] queued round {round_num} task {task_step} "
                             f"({task_name}) modes={','.join(EVAL_MODES)}")

                pending_tasks[task_key] = {
                    "ckpt_dir": ckpt_dir,
                }
                _cleanup_completed(pending_tasks, done_counts, expected_done)
            else:
                for mode in EVAL_MODES:
                    job = dict(job_base)
                    _run_eval_job_sync_base(
                        job=job,
                        args=args,
                        model=model,
                        tokenizer=tokenizer,
                        task_config=task_config,
                        eval_fields=eval_fields,
                        eval_csv_path=eval_csv_path,
                        eval_jsonl_path=eval_jsonl_path,
                        eval_progress=eval_progress,
                        done_counts=done_counts,
                        total_tasks=len(args.tasks) * args.total_round,
                        tasks_per_round=len(args.tasks),
                        device=sync_eval_devices[mode],
                        tf_writer=sync_tf_writers[mode],
                        first_acc=round_first_acc[mode],
                    )
                    if train_device.type == "cuda":
                        torch.cuda.set_device(train_device)
            completed_train = (round_idx * len(args.tasks)) + task_step
            _log_progress(
                completed_train,
                total_tasks,
                train_times,
                eval_progress,
                run_start,
                context=f"[progress] after train r{round_num}t{task_step}",
            )
            _update_summary_if_round_done(round_num)

        # Cleanup model between rounds
        model.to("cpu")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        _log(f"Round {round_num}/{args.total_round} completed")
        _log_console(f"Round {round_num}/{args.total_round} completed")
        _update_summary_if_round_done(round_num)

    if use_async_eval:
        # Wait for all evaluations to finish, then cleanup
        while pending_tasks:
            time.sleep(30)
            _cleanup_completed(pending_tasks, done_counts, expected_done)
            _log_progress(
                total_tasks,
                total_tasks,
                train_times,
                eval_progress,
                run_start,
                context="[progress] waiting eval",
            )
            _update_summary_if_round_done(args.total_round)

        for mode in EVAL_MODES:
            job_queues[mode].put(None)
        for p in workers:
            p.join()
    else:
        for writer in sync_tf_writers.values():
            writer.flush()

    train_f.close()
    summary_path = os.path.join(run_dir, "summary_task_mode.csv")
    if _summarize_round_metrics(eval_csv_path, summary_path):
        _log(f"[summary] wrote {summary_path}")
        _log_console(f"[summary] wrote {summary_path}")
    else:
        _log("[summary] skipped (no eval data)")
        _log_console("[summary] skipped (no eval data)")
    _log("Run completed")
    _log_console("Run completed")


if __name__ == "__main__":
    main()
