#!/usr/bin/env python
# Text-classification continual learning benchmark runner (oLoRA-style).
#
# Key requirements (from user):
# - Tasks order default: DBpedia -> Amazon -> Yahoo -> AG News
# - Per round, resample splits (train from train split, val stratified from test/val split),
#   and use the SAME splits across all methods for fair comparison.
# - Train: 1 epoch, constant LR, wd=0, dropout=0.0 for fairness, effective batch=64 via 8x8.
# - Implement methods following CURb/cl.py flow (inject -> train -> merge/strip per task).

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
_mora_path = os.path.join(REPO_ROOT, "MoRA", "peft-mora")
if os.path.isdir(_mora_path):
    sys.path.append(_mora_path)
    sys.path.append(os.path.join(_mora_path, "src"))

from curb import CURbLinear, inject_curb, merge_curb, strip_curb, freeze_except_curb_U  # noqa: E402
from curb_basis import load_or_build_curb_basis  # noqa: E402
from curlora import inject_curlora, merge_curlora, strip_curlora, freeze_except_curlora_U  # noqa: E402
from bilora import inject_bilora, merge_bilora, strip_bilora, freeze_except_bilora_theta  # noqa: E402
from olora import (  # noqa: E402
    collect_lora_factors,
    build_olora_prev_device_map,
    append_olora_subspace,
    compute_olora_losses,
)
from inflora import (  # noqa: E402
    init_inflora_state,
    design_inflora_b_by_module,
    update_inflora_state_after_task,
    apply_inflora_to_peft_model,
)
from lorac import (  # noqa: E402
    init_lorac_state,
    inject_lorac,
    merge_lorac,
    strip_lorac,
    freeze_except_lorac,
    update_lorac_ipc_importance,
    lorac_ortho_loss,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "true")

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


def _resolve_torch_dtype(dtype_name: str):
    name = str(dtype_name).lower()
    if name in ("fp32", "float32", "torch.float32"):
        return torch.float32
    if name in ("bf16", "bfloat16", "torch.bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "torch.float16"):
        return torch.float16
    raise ValueError(f"Unsupported model_dtype: {dtype_name}")


def _curb_param_budget(curb_rank: int) -> int:
    r = max(1, int(curb_rank))
    return r * r


def _effective_curb_rank(in_features: int, out_features: int, curb_rank: int) -> int:
    return max(1, min(int(curb_rank), int(in_features), int(out_features)))


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
    InfLoRA freezes lora_A (= B_t) and only trains lora_B (= A_t).

    For fair comparison against LoRA, we increase InfLoRA rank so that the number
    of *trainable* parameters matches LoRA's (A+B):
      out_features * r_inflora  ~=  r_lora * (in_features + out_features)
    """
    shapes = _infer_module_shapes(model, ffn_module_names, attn_module_names)
    inflora_ranks = {}
    for key, (in_f, out_f) in shapes.items():
        if key not in lora_ranks:
            raise KeyError(f"Missing LoRA rank for {key} (needed for InfLoRA rank matching).")
        r_lora = int(lora_ranks[key])
        target_trainable = int(r_lora * (int(in_f) + int(out_f)))
        r = int(target_trainable / max(1, int(out_f)) + 0.5)
        r = max(1, min(r, int(min(in_f, out_f))))
        inflora_ranks[key] = int(r)
    return inflora_ranks


def _build_target_module_lists(layer_indices):
    target_modules_q = [f"layers.{layer}.self_attn.q_proj" for layer in layer_indices]
    target_modules_k = [f"layers.{layer}.self_attn.k_proj" for layer in layer_indices]
    target_modules_gate = [f"layers.{layer}.mlp.gate_proj" for layer in layer_indices]
    return target_modules_q, target_modules_k, target_modules_gate


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


class ListDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = list(samples)

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
    # Align label padding with tokenizer padding_side.
    # Each label sequence is the same length as its (unpadded) input_ids.
    max_len = int(enc["input_ids"].shape[1])
    labels_aligned = []
    pad_on_left = (getattr(tokenizer, "padding_side", "right") == "left")
    for lab in labels:
        pad_len = max(0, max_len - len(lab))
        if pad_on_left:
            labels_aligned.append(([-100] * pad_len) + list(lab))
        else:
            labels_aligned.append(list(lab) + ([-100] * pad_len))
    enc["labels"] = torch.tensor(labels_aligned, dtype=torch.long)
    return enc


def _encode_example(tokenizer, prompt: str, target: str, max_length: int):
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


def _humanize_label(name: str) -> str:
    # Insert spaces for CamelCase and underscores for better LM friendliness.
    text = str(name).replace("_", " ")
    text = re.sub(r"(?<!^)(?=[A-Z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass(frozen=True)
class TaskSpec:
    # Canonical task id (used in logs / prompts).
    dataset_name: str
    # HuggingFace datasets path/config (load_dataset(hf_path, hf_name)).
    hf_path: str
    hf_name: str | None
    text_fields: list[str]
    label_field: str
    train_split: str = "train"
    # Optional explicit validation split name. If None, _resolve_val_split() is used.
    val_split: str | None = None
    # Optional label text override (must match label id order). If None, use dataset ClassLabel names.
    label_texts_override: list[str] | None = None


TASK_SPECS: dict[str, TaskSpec] = {
    # CL Benchmark (topic / sentiment)
    "dbpedia_14": TaskSpec("dbpedia_14", "dbpedia_14", None, ["title", "content"], "label", val_split="test"),
    "yahoo_answers_topics": TaskSpec(
        "yahoo_answers_topics",
        "yahoo_answers_topics",
        None,
        ["question_title", "question_content", "best_answer"],
        "topic",
        val_split="test",
    ),
    "ag_news": TaskSpec("ag_news", "ag_news", None, ["text"], "label", val_split="test"),
    "amazon_polarity": TaskSpec("amazon_polarity", "amazon_polarity", None, ["title", "content"], "label", val_split="test"),
    "yelp_polarity": TaskSpec("yelp_polarity", "yelp_polarity", None, ["text"], "label", val_split="test"),
    "imdb": TaskSpec("imdb", "imdb", None, ["text"], "label", val_split="test"),

    # GLUE
    "mnli": TaskSpec("mnli", "glue", "mnli", ["premise", "hypothesis"], "label", val_split="validation_matched"),
    "qqp": TaskSpec(
        "qqp",
        "glue",
        "qqp",
        ["question1", "question2"],
        "label",
        val_split="validation",
        label_texts_override=["no", "yes"],  # not duplicate / duplicate
    ),
    "rte": TaskSpec(
        "rte",
        "glue",
        "rte",
        ["sentence1", "sentence2"],
        "label",
        val_split="validation",
        label_texts_override=["yes", "no"],  # entailment / not_entailment
    ),
    "sst2": TaskSpec("sst2", "glue", "sst2", ["sentence"], "label", val_split="validation"),

    # SuperGLUE
    "wic": TaskSpec(
        "wic",
        "super_glue",
        "wic",
        ["word", "sentence1", "sentence2"],
        "label",
        val_split="validation",
        label_texts_override=["no", "yes"],
    ),
    "cb": TaskSpec("cb", "super_glue", "cb", ["premise", "hypothesis"], "label", val_split="validation"),
    "copa": TaskSpec(
        "copa",
        "super_glue",
        "copa",
        ["premise", "question", "choice1", "choice2"],
        "label",
        val_split="validation",
        label_texts_override=["A", "B"],
    ),
    "boolq": TaskSpec(
        "boolq",
        "super_glue",
        "boolq",
        ["passage", "question"],
        "label",
        val_split="validation",
        label_texts_override=["no", "yes"],
    ),
    "multirc": TaskSpec(
        "multirc",
        "super_glue",
        "multirc",
        ["paragraph", "question", "answer"],
        "label",
        val_split="validation",
        label_texts_override=["no", "yes"],
    ),
}


TASK_ALIASES: dict[str, str] = {
    # Paper-friendly aliases -> canonical ids used by TASK_SPECS
    "amazon": "amazon_polarity",
    "yelp": "yelp_polarity",
    "dbpedia": "dbpedia_14",
    "ag": "ag_news",
    "yahoo": "yahoo_answers_topics",
    "boolqa": "boolq",
    "sst-2": "sst2",
    "sst_2": "sst2",
}


TASK_INSTRUCTIONS: dict[str, str] = {
    # NLI
    "mnli": "Determine the relationship between the premise and the hypothesis.",
    "cb": "Determine the relationship between the premise and the hypothesis.",
    "rte": "Does Sentence1 entail Sentence2?",
    # Paraphrase / boolean QA / WSD
    "qqp": "Are Question1 and Question2 duplicates (asking the same thing)?",
    "boolq": "Answer the question based on the passage.",
    "wic": "Does the word have the same meaning in both sentences?",
    # QA-style classification
    "copa": "Choose the more plausible alternative. Answer with A for Choice A or B for Choice B.",
    "multirc": "Is the answer correct given the paragraph and the question?",
    # Sentiment
    "sst2": "Classify the sentiment of the text.",
    "imdb": "Classify the sentiment of the text.",
    "yelp_polarity": "Classify the sentiment of the text.",
    "amazon_polarity": "Classify the sentiment of the text.",
    # Topic classification
    "dbpedia_14": "Classify the topic of the text.",
    "ag_news": "Classify the topic of the text.",
    "yahoo_answers_topics": "Classify the topic of the text.",
}


def _normalize_task_name(name: str) -> str:
    key = str(name).strip()
    if not key:
        return key
    key = key.lower().replace(" ", "_")
    return TASK_ALIASES.get(key, key)


def _load_task_dataset(spec: TaskSpec):
    if spec.hf_name:
        return load_dataset(spec.hf_path, spec.hf_name)
    return load_dataset(spec.hf_path)


def _resolve_val_split(ds_dict) -> str:
    if "validation" in ds_dict:
        return "validation"
    if "val" in ds_dict:
        return "val"
    if "test" in ds_dict:
        return "test"
    raise ValueError(f"Dataset has no validation/test split. splits={list(ds_dict.keys())}")


def _get_label_names(ds_dict, spec: TaskSpec) -> list[str]:
    if spec.label_texts_override is not None:
        return [str(x) for x in spec.label_texts_override]

    feat = ds_dict[spec.train_split].features.get(spec.label_field)
    if hasattr(feat, "names") and feat.names is not None:
        names = list(feat.names)
    else:
        # Fallback to numeric labels.
        label_vals = ds_dict[spec.train_split][spec.label_field]
        n = int(max(label_vals)) + 1 if label_vals else 0
        names = [str(i) for i in range(n)]

    if spec.dataset_name == "yelp_polarity":
        # The dataset exposes names ['1','2'] but it's a polarity dataset; use semantic labels.
        if len(names) == 2:
            names = ["negative", "positive"]
    if spec.dataset_name == "imdb":
        if len(names) == 2:
            names = ["negative", "positive"]
    if spec.dataset_name == "amazon_polarity":
        # Keep dataset's semantic names when available.
        pass
    names = [_humanize_label(x) for x in names]
    return [str(x) for x in names]


def _extract_text(row: dict, spec: TaskSpec) -> str:
    parts = []
    for field in spec.text_fields:
        val = row.get(field, "")
        if val is None:
            val = ""
        val = str(val).strip()
        if not val:
            continue
        if len(spec.text_fields) > 1:
            field_name = field.replace("_", " ").title()
            if spec.dataset_name == "copa":
                # Make the A/B mapping explicit in the text.
                if field == "choice1":
                    field_name = "Choice A"
                elif field == "choice2":
                    field_name = "Choice B"
                elif field == "question":
                    field_name = "Question Type"
            parts.append(f"{field_name}: {val}")
        else:
            parts.append(val)
    return "\n".join(parts).strip()


def _build_prompt(spec: TaskSpec, text: str, label_texts: list[str]) -> str:
    # Put the instruction close to the end so left-truncation keeps it.
    label_str = "; ".join(label_texts)
    text = text.strip()
    instruction = TASK_INSTRUCTIONS.get(spec.dataset_name, "Choose the correct label for the task.")
    return (
        f"{text}\n\n"
        f"Task: {spec.dataset_name}\n"
        f"Instruction: {instruction}\n"
        f"Choose one label from: {label_str}\n"
        f"Return the label only.\n"
        f"Label:"
    )


def _build_samples_for_indices(ds_split, spec: TaskSpec, indices: list[int], label_texts: list[str], tokenizer, max_length: int):
    samples = []
    for idx in indices:
        row = ds_split[int(idx)]
        text = _extract_text(row, spec)
        if not text:
            continue
        label_id = int(row[spec.label_field])
        if label_id < 0 or label_id >= len(label_texts):
            continue
        prompt = _build_prompt(spec, text, label_texts)
        target = " " + str(label_texts[label_id])
        enc = _encode_example(tokenizer, prompt, target, max_length=max_length)
        if enc is None:
            continue
        samples.append({
            "input_ids": enc["input_ids"],
            "labels": enc["labels"],
            "prompt": prompt,
            "target": target,
            "task": spec.dataset_name,
            "label_id": label_id,
        })
    return samples


@torch.no_grad()
def _score_candidates_with_cache(
    model,
    tokenizer,
    prompts: list[str],
    candidate_token_ids: list[list[int]],
    device: torch.device,
    max_length: int,
):
    # Tokenize prompts (left padding to make cache usage safe).
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past = out.past_key_values
    # With left padding, the last token is always the last position.
    last_logits = out.logits[:, -1, :]
    last_logp = torch.log_softmax(last_logits, dim=-1)  # [B, V]

    B = input_ids.shape[0]
    scores = []

    for cand_ids in candidate_token_ids:
        if not cand_ids:
            scores.append(torch.full((B,), -1e9, device=device, dtype=torch.float32))
            continue
        cand = torch.tensor(cand_ids, device=device, dtype=torch.long)
        first = cand[0].view(1).expand(B)
        score = last_logp.gather(1, first.view(B, 1)).squeeze(1).to(dtype=torch.float32)
        if len(cand_ids) > 1:
            # Feed the full candidate tokens; logits[0] predicts token2, ..., logits[L-2] predicts tokenL.
            cand_batch = cand.view(1, -1).expand(B, -1)
            cand_mask = torch.ones_like(cand_batch, dtype=attention_mask.dtype, device=device)
            full_mask = torch.cat([attention_mask, cand_mask], dim=1)
            cand_out = model(
                input_ids=cand_batch,
                attention_mask=full_mask,
                past_key_values=past,
                use_cache=False,
            )
            cand_logits = cand_out.logits  # [B, L, V]
            cand_logp = torch.log_softmax(cand_logits[:, :-1, :], dim=-1)  # [B, L-1, V]
            rest = cand[1:].view(1, -1).expand(B, -1)  # [B, L-1]
            score = score + cand_logp.gather(2, rest.unsqueeze(-1)).squeeze(-1).sum(dim=1).to(dtype=torch.float32)
        scores.append(score)

    return torch.stack(scores, dim=1)  # [B, K]


@torch.no_grad()
def evaluate_accuracy(
    model,
    tokenizer,
    ds_split,
    spec: TaskSpec,
    indices: list[int],
    label_texts: list[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
):
    model.eval()
    candidate_targets = [" " + lt for lt in label_texts]
    candidate_token_ids = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in candidate_targets]

    total = 0
    correct = 0

    # Stream batches by indexing (no need to pre-build full tokenized dataset).
    pbar = tqdm(range(0, len(indices), batch_size), desc=f"eval {spec.dataset_name}", leave=False, file=sys.stdout)
    for start in pbar:
        batch_indices = indices[start:start + batch_size]
        prompts = []
        gold = []
        for idx in batch_indices:
            row = ds_split[int(idx)]
            text = _extract_text(row, spec)
            if not text:
                continue
            label_id = int(row[spec.label_field])
            if label_id < 0 or label_id >= len(label_texts):
                continue
            prompts.append(_build_prompt(spec, text, label_texts))
            gold.append(label_id)

        if not prompts:
            continue

        scores = _score_candidates_with_cache(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            candidate_token_ids=candidate_token_ids,
            device=device,
            max_length=max_length,
        )
        pred = torch.argmax(scores, dim=1).detach().cpu().numpy().tolist()
        for p, g in zip(pred, gold):
            total += 1
            correct += int(int(p) == int(g))
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


##############################################################################
# CURb diagnostics (--diag_curb)
##############################################################################

@torch.no_grad()
def _diag_curb_snapshot_pretrained_llm(model, layer_indices, ffn_module_names, attn_module_names):
    """Save pretrained W for each CURb target module (LLaMA layer-index based)."""
    snap = {}
    for li in layer_indices:
        layer = model.model.layers[li]
        for name in ffn_module_names:
            mod = getattr(layer.mlp, name)
            key = f"layer_{li}_mlp_{name}"
            if hasattr(mod, "weight"):
                snap[key] = mod.weight.detach().clone()
        for name in attn_module_names:
            mod = getattr(layer.self_attn, name)
            key = f"layer_{li}_self_attn_{name}"
            if hasattr(mod, "weight"):
                snap[key] = mod.weight.detach().clone()
    return snap


@torch.no_grad()
def _diag_curb_before_merge_llm(model, logger):
    """Collect per-module CURbLinear stats BEFORE merge (same as ViT version)."""
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
def _diag_curb_after_merge_llm(model, pretrained_snap, layer_indices, ffn_module_names, attn_module_names, logger):
    """Compute ||W_current - W_pretrained||_F per module AFTER merge+strip."""
    rows = []
    for li in layer_indices:
        layer = model.model.layers[li]
        for name in ffn_module_names:
            key = f"layer_{li}_mlp_{name}"
            if key not in pretrained_snap:
                continue
            mod = getattr(layer.mlp, name)
            w_cur = mod.weight.data
            w_pre = pretrained_snap[key].to(device=w_cur.device, dtype=w_cur.dtype)
            drift = float(torch.norm(w_cur - w_pre, p="fro"))
            w_pre_fro = float(torch.norm(w_pre, p="fro"))
            rows.append({
                "module": key,
                "W_drift_fro": drift,
                "W_pretrained_fro": w_pre_fro,
                "drift_ratio": drift / max(w_pre_fro, 1e-12),
            })
            logger.log(
                f"[diag_curb_drift] {key}: ||W-W0||_F={drift:.6f} "
                f"||W0||_F={w_pre_fro:.4f} ratio={drift / max(w_pre_fro, 1e-12):.6f}"
            )
        for name in attn_module_names:
            key = f"layer_{li}_self_attn_{name}"
            if key not in pretrained_snap:
                continue
            mod = getattr(layer.self_attn, name)
            w_cur = mod.weight.data
            w_pre = pretrained_snap[key].to(device=w_cur.device, dtype=w_cur.dtype)
            drift = float(torch.norm(w_cur - w_pre, p="fro"))
            w_pre_fro = float(torch.norm(w_pre, p="fro"))
            rows.append({
                "module": key,
                "W_drift_fro": drift,
                "W_pretrained_fro": w_pre_fro,
                "drift_ratio": drift / max(w_pre_fro, 1e-12),
            })
            logger.log(
                f"[diag_curb_drift] {key}: ||W-W0||_F={drift:.6f} "
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
    max_train_steps=None,
    diag_logger=None,
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

    if method in ("lora", "mora", "olora", "inflora"):
        lora_ranks = method_ctx["lora_ranks"]
        target_modules_q = method_ctx["target_modules_q"]
        target_modules_k = method_ctx["target_modules_k"]
        target_modules_gate = method_ctx["target_modules_gate"]
        b_by_module = None
        calib_loader = None
        if method == "mora":
            mora_peft = _load_mora_peft()
            config_cls = mora_peft.LoraConfig
            config_kwargs = {
                "use_mora": True,
                "mora_type": 6,
            }
            task_type = mora_peft.TaskType.CAUSAL_LM
            _get_peft = mora_peft.get_peft_model
        else:
            mora_peft = None
            config_cls = LoraConfig
            config_kwargs = {}
            task_type = TaskType.CAUSAL_LM
            _get_peft = get_peft_model

        if method == "inflora":
            # NOTE: For fair comparison, we match the number of *trainable* parameters
            # to the LoRA baseline even though InfLoRA freezes lora_A (= B_t). See
            # _compute_inflora_ranks_match_trainable().
            inflora_state = method_ctx["inflora_state"]
            inflora_ranks = method_ctx["inflora_ranks"]
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

        # Select ranks / scaling per method.
        active_ranks = lora_ranks
        if method == "inflora":
            active_ranks = inflora_ranks
            # Official InfLoRA has no separate alpha; to make PEFT's LoRA scaling = 1,
            # enforce lora_alpha = r (scaling = alpha/r = 1). B_t itself is scaled by 1/sqrt(3).
            alpha_q = int(active_ranks["attn_q_proj"])
            alpha_k = int(active_ranks["attn_k_proj"])
            alpha_gate = int(active_ranks["mlp_gate_proj"])
        else:
            alpha_q = int(2 * int(active_ranks["attn_q_proj"]))
            alpha_k = int(2 * int(active_ranks["attn_k_proj"]))
            alpha_gate = int(2 * int(active_ranks["mlp_gate_proj"]))
            if args.lora_alpha is not None:
                fixed_alpha = float(args.lora_alpha)
                alpha_q = fixed_alpha
                alpha_k = fixed_alpha
                alpha_gate = fixed_alpha

        lora_config_q = config_cls(
            r=active_ranks["attn_q_proj"],
            lora_alpha=alpha_q,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=task_type,
            target_modules=target_modules_q,
            **config_kwargs,
        )
        lora_config_k = config_cls(
            r=active_ranks["attn_k_proj"],
            lora_alpha=alpha_k,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=task_type,
            target_modules=target_modules_k,
            **config_kwargs,
        )
        lora_config_gate = config_cls(
            r=active_ranks["mlp_gate_proj"],
            lora_alpha=alpha_gate,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=task_type,
            target_modules=target_modules_gate,
            **config_kwargs,
        )
        train_model = _get_peft(train_model, lora_config_q)
        train_model = _get_peft(train_model, lora_config_k)
        train_model = _get_peft(train_model, lora_config_gate)
        train_model.to(device)
        if method == "inflora":
            apply_inflora_to_peft_model(train_model, b_by_module or {})
    elif method in ("lorac", "lorac_ipc"):
        lora_ranks = method_ctx["lora_ranks"]
        task_idx = int(method_ctx.get("lorac_task_idx", 0))
        ipc_enabled = (method == "lorac_ipc")
        train_model = inject_lorac(
            train_model,
            layer_indices=method_ctx["layer_indices"],
            ffn_module_names=method_ctx["ffn_module_names"],
            attn_module_names=method_ctx["attn_module_names"],
            lora_ranks=lora_ranks,
            lorac_state=method_ctx["lorac_state"],
            task_idx=task_idx,
            lora_alpha=float(args.lora_alpha) if args.lora_alpha is not None else None,
            ipc_enabled=ipc_enabled,
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

    trainable_params = [p for p in train_model.parameters() if p.requires_grad]
    current_lr = float(args.learning_rate if learning_rate is None else learning_rate)
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
            param_groups.append({"params": base_params, "lr": current_lr})
        if omega_params:
            param_groups.append({"params": omega_params, "lr": current_lr * float(args.lorac_omega_lr_scale)})
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=float(args.weight_decay),
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=current_lr,
            weight_decay=float(args.weight_decay),
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )

    total_steps_target = None
    try:
        total_batches = len(loader)
        total_steps_target = total_batches * int(args.epochs)
        if max_train_steps is not None:
            total_steps_target = min(int(total_steps_target), int(max_train_steps))
    except TypeError:
        total_steps_target = None

    scheduler = None
    if total_steps_target is not None:
        total_optimizer_steps = max(1, math.ceil(total_steps_target / max(1, args.grad_accum_steps)))
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

    total_loss = 0.0
    total_steps = 0
    total_tokens = 0
    skipped_nonfinite = 0
    start_time = time.time()

    optimizer.zero_grad(set_to_none=True)

    desc = f"train {task_label}" if task_label else "train"
    pbar = tqdm(total=total_steps_target, desc=desc, leave=False, file=sys.stdout)

    for _ in range(int(args.epochs)):
        for _, batch in enumerate(loader):
            if max_train_steps is not None and total_steps >= int(max_train_steps):
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = train_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            if method == "curlora":
                logits = outputs.logits if hasattr(outputs, "logits") else None
                if logits is not None and (not torch.isfinite(logits).all()):
                    skipped_nonfinite += 1
                    optimizer.zero_grad(set_to_none=True)
                    if skipped_nonfinite <= 5:
                        print(
                            f"[warn] skip batch due to non-finite logits "
                            f"(method=curlora step={total_steps+1} skipped={skipped_nonfinite})",
                            flush=True,
                        )
                    continue

            total_step_loss = outputs.loss
            if method == "curlora" and (not torch.isfinite(total_step_loss)):
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                if skipped_nonfinite <= 5:
                    print(
                        f"[warn] skip batch due to non-finite loss "
                        f"(method=curlora step={total_steps+1} skipped={skipped_nonfinite})",
                        flush=True,
                    )
                continue

            if method == "olora":
                orth_loss, l2_loss = compute_olora_losses(outputs.loss, olora_a, olora_b, olora_prev_a)
                total_step_loss = (
                    total_step_loss
                    + float(args.olora_lambda_orth) * orth_loss
                    + float(args.olora_lambda_l2) * l2_loss
                )
            if method in ("lorac", "lorac_ipc"):
                total_step_loss = total_step_loss + float(args.lorac_ortho) * lorac_ortho_loss(train_model)

            loss = total_step_loss / int(args.grad_accum_steps)
            loss.backward()
            if method == "lorac_ipc":
                update_lorac_ipc_importance(train_model)

            if (total_steps + 1) % int(args.grad_accum_steps) == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(args.max_grad_norm))
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.detach().item())
            total_steps += 1
            total_tokens += int((labels != -100).sum().item())

            if pbar is not None:
                display_loss = float(loss.detach().item()) * int(args.grad_accum_steps)
                pbar.update(1)
                pbar.set_postfix(loss=f"{display_loss:.4f}")

        if max_train_steps is not None and total_steps >= int(max_train_steps):
            break

    if total_steps > 0 and (total_steps % int(args.grad_accum_steps) != 0):
        torch.nn.utils.clip_grad_norm_(trainable_params, float(args.max_grad_norm))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    if pbar is not None:
        pbar.close()

    duration = time.time() - start_time
    train_model.config.use_cache = use_cache_flag

    loss_mean = (total_loss / max(1, total_steps)) if total_steps else None
    trainable_param_count = sum(p.numel() for p in trainable_params)

    if method == "olora":
        final_olora_a, _ = collect_lora_factors(train_model)
        append_olora_subspace(method_ctx, final_olora_a)

    # Diagnostic: collect CURb stats BEFORE merge
    diag_before_merge = []
    if diag_logger is not None and method == "curb":
        diag_before_merge = _diag_curb_before_merge_llm(train_model, diag_logger)

    result_model = _merge_and_strip(train_model, method)
    result_model.to(device)
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
    return result_model, {
        "train_steps": total_steps,
        "loss_mean": loss_mean,
        "tokens": total_tokens,
        "duration_sec": duration,
        "learning_rate": current_lr,
        "trainable_params": trainable_param_count,
        "skipped_nonfinite": int(skipped_nonfinite),
        "_diag_before_merge": diag_before_merge,
    }


def _make_splits(
    tasks: list[str],
    total_round: int,
    seed: int,
    train_n: int,
    val_per_class: int,
    out_path: str,
    logger: Logger | None = None,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load datasets once and precompute per-class indices for val split.
    ds_cache = {}
    val_idx_by_class = {}

    normalized_tasks = [_normalize_task_name(t) for t in tasks]
    for task in normalized_tasks:
        if task not in TASK_SPECS:
            raise ValueError(f"Unsupported task: {task}. Supported={sorted(TASK_SPECS.keys())}")
        spec = TASK_SPECS[task]
        ds = _load_task_dataset(spec)
        ds_cache[task] = ds
        train_split = spec.train_split
        if train_split not in ds:
            raise ValueError(f"{task} missing train_split={train_split}. splits={list(ds.keys())}")
        val_split = spec.val_split if spec.val_split is not None else _resolve_val_split(ds)
        if val_split not in ds:
            raise ValueError(f"{task} missing val_split={val_split}. splits={list(ds.keys())}")
        labels = ds[val_split][spec.label_field]
        buckets: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            buckets.setdefault(int(lab), []).append(int(i))
        if not buckets:
            raise ValueError(f"{task} split={val_split} has no labeled samples.")

        # Auto-cap val_per_class to the available per-class minimum (balanced, without replacement).
        min_count = min(len(idxs) for idxs in buckets.values())
        if min_count < 1:
            raise ValueError(f"{task} split={val_split} has an empty class bucket.")
        val_per_class_eff = min(int(val_per_class), int(min_count))
        if logger is not None and val_per_class_eff != int(val_per_class):
            logger.log(
                f"[splits] cap val_per_class for {task}: requested={int(val_per_class)} "
                f"available_min={int(min_count)} using={int(val_per_class_eff)}"
            )
        val_idx_by_class[task] = {
            "split": val_split,
            "buckets": buckets,
            "val_per_class_eff": int(val_per_class_eff),
        }
        if logger is not None:
            logger.log(
                f"[splits] loaded {task} train={len(ds[train_split])} {val_split}={len(ds[val_split])} "
                f"classes={len(buckets)}"
            )

    rounds = []
    for round_idx in range(int(total_round)):
        round_num = round_idx + 1
        round_seed = _round_seed(seed, round_idx)
        task_splits = {}
        for task in normalized_tasks:
            spec = TASK_SPECS[task]
            ds = ds_cache[task]
            train_split = spec.train_split
            val_split = val_idx_by_class[task]["split"]
            buckets = val_idx_by_class[task]["buckets"]
            val_per_class_eff = int(val_idx_by_class[task]["val_per_class_eff"])

            rng_train = random.Random(_seed_from_parts(round_seed, task, "train"))
            train_indices = rng_train.sample(range(len(ds[train_split])), min(int(train_n), len(ds[train_split])))

            rng_val = random.Random(_seed_from_parts(round_seed, task, "val"))
            val_indices = []
            for lab, idxs in buckets.items():
                val_indices.extend(rng_val.sample(idxs, int(val_per_class_eff)))
            rng_val.shuffle(val_indices)

            task_splits[task] = {
                "train_split": train_split,
                "train_indices": train_indices,
                "val_split": val_split,
                "val_indices": val_indices,
                "label_field": spec.label_field,
                "val_per_class_eff": int(val_per_class_eff),
            }

        rounds.append({"round": round_num, "seed": round_seed, "task_splits": task_splits})

    obj = {
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "base_seed": int(seed),
        "round_seed_stride": int(ROUND_SEED_STRIDE),
        "tasks": list(normalized_tasks),
        "train_n": int(train_n),
        "val_per_class": int(val_per_class),
        "total_round": int(total_round),
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
            return r.get("task_splits", {}) or {}
    raise KeyError(f"Round not found in splits: {round_num}")


def main():
    parser = argparse.ArgumentParser(description="Text-classification continual learning benchmark (oLoRA-style).")
    parser.add_argument(
        "--method",
        type=str,
        default="lora",
        choices=["curb", "curlora", "bilora", "lora", "mora", "olora", "inflora", "lorac", "lorac_ipc"],
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["dbpedia_14", "amazon_polarity", "yahoo_answers_topics", "ag_news"],
        help="Task stream order.",
    )
    parser.add_argument("--total_round", type=int, default=3)

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--model_dtype", type=str, default="fp32")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--splits_path", type=str, required=True)
    parser.add_argument("--make_splits_only", action="store_true")
    parser.add_argument("--train_samples_per_task", type=int, default=1000)
    parser.add_argument("--val_per_class", type=int, default=500)

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)

    # CURb/CURLoRA params
    parser.add_argument("--curb_rank", type=int, default=256)
    parser.add_argument("--curb_rank_q", type=int, default=256)
    parser.add_argument("--curb_rank_k", type=int, default=202)
    parser.add_argument("--curb_rank_gate", type=int, default=384)
    parser.add_argument("--curb_alpha", type=float, default=1.0)
    parser.add_argument("--curb_basis_mode", type=str, default="weight", choices=["cov_fast", "weight", "hybrid"])
    parser.add_argument("--curb_deim_importance_order", type=str, default="low", choices=["high", "low"])
    # cov_fast/hybrid calibration (C4). Used when curb_basis_mode is cov_fast/hybrid (or update_whiten=diag).
    parser.add_argument("--curb_calib_steps", type=int, default=256)
    parser.add_argument("--curb_batch_size", type=int, default=1)
    parser.add_argument("--curb_max_length", type=int, default=512)
    parser.add_argument("--curb_calib_category", type=str, default="en")
    parser.add_argument(
        "--curb_calib_source",
        type=str,
        default="c4",
        choices=["c4", "replay_mix_c4"],
        help=(
            "Calibration source for CURb cov_fast/hybrid activation stats. "
            "'c4' uses C4 only. 'replay_mix_c4' uses replay buffer texts first, "
            "then fills remaining sequences with C4."
        ),
    )
    parser.add_argument(
        "--replay_buffer_per_task",
        type=int,
        default=0,
        help="Replay buffer size per task used only for calibration text sampling.",
    )
    parser.add_argument("--curb_ffn_module_names", nargs="*", default=["gate_proj"])
    parser.add_argument("--curb_attn_module_names", nargs="*", default=["q_proj", "k_proj"])
    parser.add_argument("--curb_update_whiten", type=str, default="none", choices=["none", "diag"])
    parser.add_argument("--curb_whiten_ridge_ratio", type=float, default=1e-4)
    parser.add_argument("--curb_whiten_ridge_abs", type=float, default=1e-12)

    # BiLoRA (Fourier-domain sparse bilinear update)
    parser.add_argument(
        "--bilora_k",
        type=int,
        default=None,
        help="Optional number of active frequency components per module (default: r_eff^2).",
    )
    parser.add_argument(
        "--bilora_alpha",
        type=float,
        default=None,
        help="Optional scaling for BiLoRA delta-W (default: 0.5*sqrt(out*in)).",
    )
    parser.add_argument("--bilora_seed", type=int, default=777, help="Random seed for BiLoRA frequency support sampling.")
    parser.add_argument("--bilora_chunk_size", type=int, default=256, help="Token chunk size for BiLoRA forward (memory control).")
    parser.add_argument(
        "--bilora_freq_chunk_size",
        type=int,
        default=8192,
        help="Frequency chunk size for BiLoRA forward (memory control).",
    )

    # LoRA/MoRA/O-LoRA params
    parser.add_argument("--lora_rank_q", type=int, default=8)
    parser.add_argument("--lora_rank_k", type=int, default=8)
    parser.add_argument("--lora_rank_gate", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--olora_lambda_orth", type=float, default=0.5)
    parser.add_argument("--olora_lambda_l2", type=float, default=0.0)

    # LoRAC / LoRAC-IPC params (official-style defaults)
    parser.add_argument("--lorac_ortho", type=float, default=1.0, help="LoRAC orthogonal regularization coefficient.")
    parser.add_argument(
        "--lorac_omega_lr_scale",
        type=float,
        default=1.0,
        help="Optimizer LR scale applied only to omega parameters (official: omega_lr_scale).",
    )
    parser.add_argument("--lorac_ipc_beta1", type=float, default=0.85, help="IPC beta1 (sensitivity EMA).")
    parser.add_argument("--lorac_ipc_beta2", type=float, default=0.85, help="IPC beta2 (uncertainty EMA).")
    parser.add_argument("--lorac_ipc_threshold", type=float, default=0.1, help="IPC mask fraction in (0,1).")
    parser.add_argument("--lorac_ipc_new_mask", action="store_true", help="IPC new_mask behavior (official).")

    # InfLoRA (official defaults)
    parser.add_argument("--inflora_lamb", type=float, default=0.95)
    parser.add_argument("--inflora_lame", type=float, default=1.0)
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

    # Diagnostic flags
    parser.add_argument("--diag_curb", action="store_true", help="Enable CURb diagnostic logging")

    args = parser.parse_args()
    args.tasks = [_normalize_task_name(t) for t in (args.tasks or [])]

    if args.total_round < 1:
        raise ValueError("--total_round must be >= 1")
    if args.train_samples_per_task < 1:
        raise ValueError("--train_samples_per_task must be >= 1")
    if args.val_per_class < 1:
        raise ValueError("--val_per_class must be >= 1")
    if args.grad_accum_steps < 1 or args.train_batch_size < 1:
        raise ValueError("batch/grad_accum must be >= 1")
    if args.replay_buffer_per_task < 0:
        raise ValueError("--replay_buffer_per_task must be >= 0")
    need_curb_calib = (args.method == "curb") and (args.curb_basis_mode in ("cov_fast", "hybrid") or args.curb_update_whiten == "diag")
    if need_curb_calib and (args.curb_calib_steps < 1 or args.curb_batch_size < 1 or args.curb_max_length < 1):
        raise ValueError("CURb cov_fast/hybrid/whitening requires curb_calib_steps/batch_size/max_length >= 1.")
    if args.method == "olora" and (args.olora_lambda_orth < 0 or args.olora_lambda_l2 < 0):
        raise ValueError("oLoRA lambdas must be >= 0")
    if args.inflora_lamb <= 0 or args.inflora_lamb > 1:
        raise ValueError("--inflora_lamb must be in (0, 1].")
    if args.inflora_lame <= 0 or args.inflora_lame > 1:
        raise ValueError("--inflora_lame must be in (0, 1].")
    if args.inflora_lame < args.inflora_lamb:
        raise ValueError("--inflora_lame must be >= --inflora_lamb.")
    if args.lorac_ortho < 0:
        raise ValueError("--lorac_ortho must be >= 0.")
    if args.lorac_omega_lr_scale <= 0:
        raise ValueError("--lorac_omega_lr_scale must be > 0.")
    for beta_name in ("lorac_ipc_beta1", "lorac_ipc_beta2"):
        beta_val = float(getattr(args, beta_name))
        if beta_val < 0.0 or beta_val >= 1.0:
            raise ValueError(f"--{beta_name} must be in [0, 1) (got {beta_val}).")
    if args.lorac_ipc_threshold < 0.0 or args.lorac_ipc_threshold > 1.0:
        raise ValueError("--lorac_ipc_threshold must be in [0, 1].")

    run_id = args.run_name or datetime.now().strftime("tc_run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_path, run_id)
    os.makedirs(run_dir, exist_ok=True)
    logger = Logger(os.path.join(run_dir, "logs", "train.log"))

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=True, indent=2)

    logger.log(
        f"Run {run_id} start | method={args.method} rounds={args.total_round} tasks={args.tasks} "
        f"train_n={args.train_samples_per_task} val_per_class={args.val_per_class} "
        f"bs={args.train_batch_size} ga={args.grad_accum_steps} eff_bs={args.train_batch_size*args.grad_accum_steps} "
        f"lr={args.learning_rate} wd={args.weight_decay} dropout={args.lora_dropout} max_length={args.max_length} "
        f"curb_calib_source={args.curb_calib_source} replay_buffer_per_task={args.replay_buffer_per_task}"
    )

    # Ensure splits exist (common across methods). This is typically created by launcher.
    splits_path = os.path.abspath(args.splits_path)
    if args.make_splits_only or (not os.path.exists(splits_path)):
        logger.log(f"[splits] creating splits at {splits_path}")
        _make_splits(
            tasks=list(args.tasks),
            total_round=int(args.total_round),
            seed=int(args.seed),
            train_n=int(args.train_samples_per_task),
            val_per_class=int(args.val_per_class),
            out_path=splits_path,
            logger=logger,
        )
        if args.make_splits_only:
            logger.log("[splits] make_splits_only done; exiting")
            logger.close()
            return
    splits_obj = _load_splits(splits_path)
    splits_tasks = list(splits_obj.get("tasks", []) or [])
    if splits_tasks != list(args.tasks):
        raise ValueError(
            f"splits tasks mismatch. splits_path={splits_path}\n"
            f"- splits tasks: {splits_tasks}\n"
            f"- cli tasks:   {list(args.tasks)}\n"
            f"Use a new splits_path (new EXP_TAG) or regenerate splits."
        )

    # Load datasets once (used for all rounds).
    ds_cache = {}
    label_texts_by_task = {}
    val_split_by_task = {}
    for task in args.tasks:
        if task not in TASK_SPECS:
            raise ValueError(f"Unsupported task: {task}")
        spec = TASK_SPECS[task]
        ds = _load_task_dataset(spec)
        ds_cache[task] = ds
        label_texts_by_task[task] = _get_label_names(ds, spec)
        val_split_by_task[task] = spec.val_split if spec.val_split is not None else _resolve_val_split(ds)
        logger.log(
            f"[data] task={task} train={len(ds[spec.train_split])} {val_split_by_task[task]}={len(ds[val_split_by_task[task]])} "
            f"classes={len(label_texts_by_task[task])}"
        )

    dtype = _resolve_torch_dtype(args.model_dtype)
    device = torch.device(args.device if args.device == "cpu" else "cuda")

    csv_path = os.path.join(run_dir, "eval_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_fp:
        fieldnames = [
            "timestamp",
            "round",
            "task_step",
            "trained_task",
            "eval_task",
            "acc",
            "AA",
            "BWT",
            "val_samples",
            "train_loss_mean",
            "train_steps",
            "train_tokens",
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
                torch.cuda.set_device(0)
                torch.cuda.empty_cache()

            # Load model/tokenizer per round (reset per round), like CURb/cl.py.
            model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            tokenizer.padding_side = "left"
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            model.to(device)

            layer_indices = list(range(model.config.num_hidden_layers - 1))
            ffn_module_names = list(args.curb_ffn_module_names)
            attn_module_names = list(args.curb_attn_module_names)

            method_ctx = {
                "method": args.method,
                "layer_indices": layer_indices,
                "ffn_module_names": ffn_module_names,
                "attn_module_names": attn_module_names,
                "curb_rank": int(args.curb_rank),
            }

            # Rank overrides
            curb_rank_overrides = {
                "attn_q_proj": args.curb_rank_q,
                "attn_k_proj": args.curb_rank_k,
                "mlp_gate_proj": args.curb_rank_gate,
            }
            lora_rank_overrides = {
                "attn_q_proj": args.lora_rank_q,
                "attn_k_proj": args.lora_rank_k,
                "mlp_gate_proj": args.lora_rank_gate,
            }

            if args.method in ("curb", "curlora", "bilora"):
                curb_ranks = _compute_curb_ranks(
                    model,
                    args.curb_rank,
                    ffn_module_names,
                    attn_module_names,
                    rank_overrides=curb_rank_overrides,
                )
                method_ctx["curb_ranks"] = curb_ranks
                if args.method == "bilora":
                    alpha_note = "alpha=0.5*sqrt(out*in)(auto)" if args.bilora_alpha is None else f"alpha={args.bilora_alpha}"
                    k_note = "k=r_eff^2(auto)" if args.bilora_k is None else f"k={args.bilora_k}"
                    logger.log(
                        f"[method] bilora ranks(q={curb_ranks['attn_q_proj']} "
                        f"k={curb_ranks['attn_k_proj']} gate={curb_ranks['mlp_gate_proj']}) "
                        f"{k_note} {alpha_note} seed={int(args.bilora_seed)}"
                    )
                else:
                    logger.log(
                        f"[method] {args.method} ranks(q={curb_ranks['attn_q_proj']} "
                        f"k={curb_ranks['attn_k_proj']} gate={curb_ranks['mlp_gate_proj']}) "
                        f"alpha={args.curb_alpha} mode={args.curb_basis_mode} deim={args.curb_deim_importance_order}"
                    )
            elif args.method in ("lora", "mora", "olora", "inflora", "lorac", "lorac_ipc"):
                lora_ranks = _compute_lora_ranks(
                    model,
                    args.curb_rank,
                    ffn_module_names,
                    attn_module_names,
                    rank_overrides=lora_rank_overrides,
                )
                target_modules_q, target_modules_k, target_modules_gate = _build_target_module_lists(layer_indices)
                method_ctx["lora_ranks"] = lora_ranks
                method_ctx["target_modules_q"] = target_modules_q
                method_ctx["target_modules_k"] = target_modules_k
                method_ctx["target_modules_gate"] = target_modules_gate
                if args.method == "olora":
                    method_ctx["olora_prev_A"] = {}
                if args.method == "inflora":
                    method_ctx["inflora_state"] = init_inflora_state()
                    method_ctx["inflora_total_sessions"] = int(len(args.tasks))
                    method_ctx["inflora_ranks"] = _compute_inflora_ranks_match_trainable(
                        model,
                        lora_ranks=lora_ranks,
                        ffn_module_names=ffn_module_names,
                        attn_module_names=attn_module_names,
                    )
                if args.method in ("lorac", "lorac_ipc"):
                    method_ctx["lorac_state"] = init_lorac_state(pool_size=int(len(args.tasks)))

                active_ranks = lora_ranks
                extra = f"alpha={args.lora_alpha} dropout={args.lora_dropout}"
                if args.method == "inflora":
                    active_ranks = method_ctx["inflora_ranks"]
                    extra = f"lamb={args.inflora_lamb} lame={args.inflora_lame} (alpha implicit)"
                if args.method in ("lorac", "lorac_ipc"):
                    extra = (
                        f"alpha={args.lora_alpha} ortho={args.lorac_ortho} "
                        f"omega_lr_scale={args.lorac_omega_lr_scale}"
                    )
                    if args.method == "lorac_ipc":
                        extra = (
                            extra
                            + f" ipc(beta1={args.lorac_ipc_beta1} beta2={args.lorac_ipc_beta2} "
                            + f"thr={args.lorac_ipc_threshold} new_mask={bool(args.lorac_ipc_new_mask)})"
                        )
                logger.log(
                    f"[method] {args.method} ranks(q={active_ranks['attn_q_proj']} "
                    f"k={active_ranks['attn_k_proj']} gate={active_ranks['mlp_gate_proj']}) {extra}"
                )

            # --diag_curb: snapshot pretrained weights for drift measurement
            diag_pretrained_snap = {}
            diag_all_rows = []
            if getattr(args, "diag_curb", False) and args.method == "curb":
                diag_pretrained_snap = _diag_curb_snapshot_pretrained_llm(
                    model, layer_indices, ffn_module_names, attn_module_names
                )
                logger.log(f"[diag_curb] pretrained snapshot saved for {len(diag_pretrained_snap)} modules")

            round_splits = _get_round_splits(splits_obj, round_num)
            diag_acc = {}  # A[i,i] for BWT
            replay_buffers = {}
            replay_pool = []

            for task_step, task_name in enumerate(args.tasks, start=1):
                if task_name not in round_splits:
                    raise KeyError(f"Missing task in splits for round {round_num}: {task_name}")
                spec = TASK_SPECS[task_name]
                label_texts = label_texts_by_task[task_name]
                train_split_name = round_splits[task_name]["train_split"]
                val_split_name = round_splits[task_name]["val_split"]
                train_indices = list(round_splits[task_name]["train_indices"])
                val_indices = list(round_splits[task_name]["val_indices"])

                logger.log(f"[train] start r{round_num} t{task_step}/{len(args.tasks)} task={task_name} n={len(train_indices)}")

                # Build CURb basis per task-step (matches cl.py flow).
                if args.method == "curb":
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
                        logger.log(
                            f"[curb_calib] source=replay_mix_c4 target={target_sequences} "
                            f"replay_used={len(calib_texts)} c4_fill={c4_fill} replay_pool={len(replay_pool)}"
                        )
                    method_ctx["curb_basis"] = load_or_build_curb_basis(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        layer_indices=layer_indices,
                        ffn_module_names=ffn_module_names,
                        attn_module_names=attn_module_names,
                        rank=args.curb_rank,
                        mode=args.curb_basis_mode,
                        deim_importance_order=args.curb_deim_importance_order,
                        update_whiten=args.curb_update_whiten,
                        whiten_ridge_ratio=args.curb_whiten_ridge_ratio,
                        whiten_ridge_abs=args.curb_whiten_ridge_abs,
                        rank_overrides=method_ctx.get("curb_ranks"),
                        cache_path=cache_path,
                        calib_steps=int(args.curb_calib_steps),
                        batch_size=int(args.curb_batch_size),
                        max_length=int(args.curb_max_length),
                        dataset_category=args.curb_calib_category,
                        calib_texts=calib_texts,
                    )

                ds = ds_cache[task_name]
                train_ds = ds[train_split_name].select(train_indices)
                train_samples = _build_samples_for_indices(
                    ds_split=train_ds,
                    spec=spec,
                    indices=list(range(len(train_ds))),
                    label_texts=label_texts,
                    tokenizer=tokenizer,
                    max_length=int(args.max_length),
                )

                if args.method == "inflora":
                    # 0-indexed task id for official InfLoRA threshold schedule / B_t design.
                    method_ctx["inflora_task_idx"] = int(task_step - 1)
                if args.method in ("lorac", "lorac_ipc"):
                    method_ctx["lorac_task_idx"] = int(task_step - 1)
                if args.method == "bilora":
                    # Deterministic per-task frequency supports (official uses seed + task*10).
                    method_ctx["bilora_task_idx"] = int(task_step - 1)

                model, train_stats = train_on_samples(
                    model=model,
                    tokenizer=tokenizer,
                    samples=train_samples,
                    args=args,
                    device=device,
                    method_ctx=method_ctx,
                    task_label=task_name,
                    loader_seed=_seed_from_parts(round_seed, "loader", task_name, task_step),
                    learning_rate=float(args.learning_rate),
                    diag_logger=(logger if getattr(args, "diag_curb", False) else None),
                )

                loss_text = f"{train_stats['loss_mean']:.6f}" if train_stats["loss_mean"] is not None else "n/a"
                logger.log(
                    f"[train] done r{round_num} t{task_step} task={task_name} "
                    f"steps={train_stats['train_steps']} loss={loss_text} tokens={train_stats['tokens']} "
                    f"time={train_stats['duration_sec']:.1f}s"
                )

                # --diag_curb: after-merge W drift
                if getattr(args, "diag_curb", False) and args.method == "curb" and diag_pretrained_snap:
                    drift_rows = _diag_curb_after_merge_llm(
                        model, diag_pretrained_snap, layer_indices, ffn_module_names, attn_module_names, logger
                    )
                    for bm in train_stats.get("_diag_before_merge", []):
                        bm["round"] = int(round_num)
                        bm["task_step"] = int(task_step)
                        bm["task_name"] = task_name
                        bm["phase"] = "before_merge"
                        diag_all_rows.append(bm)
                    for dr in drift_rows:
                        dr["round"] = int(round_num)
                        dr["task_step"] = int(task_step)
                        dr["task_name"] = task_name
                        dr["phase"] = "after_merge"
                        diag_all_rows.append(dr)

                # Update replay buffer for calibration-only replay mixing.
                if train_samples and int(args.replay_buffer_per_task) > 0:
                    rb_rng = random.Random(_seed_from_parts(round_seed, "buffer", task_name))
                    take_rb = min(int(args.replay_buffer_per_task), len(train_samples))
                    selected_rb = rb_rng.sample(train_samples, take_rb)
                    buf = []
                    for s in selected_rb:
                        ids = s.get("input_ids")
                        if ids is not None:
                            text = tokenizer.decode(ids, skip_special_tokens=True)
                        else:
                            text = (str(s.get("prompt", "")) + str(s.get("target", ""))).strip()
                        if not text:
                            continue
                        buf.append({
                            "text": text,
                            "task": task_name,
                        })
                    replay_buffers[task_name] = buf
                replay_pool = []
                for buf in replay_buffers.values():
                    replay_pool.extend(buf)

                # Evaluate seen tasks
                seen_tasks = list(args.tasks)[:task_step]
                acc_map = {}
                for eval_task in seen_tasks:
                    eval_spec = TASK_SPECS[eval_task]
                    eval_label_texts = label_texts_by_task[eval_task]
                    eval_val_split = round_splits[eval_task]["val_split"]
                    eval_indices = list(round_splits[eval_task]["val_indices"])
                    eval_ds = ds_cache[eval_task][eval_val_split]
                    acc, n_eval = evaluate_accuracy(
                        model=model,
                        tokenizer=tokenizer,
                        ds_split=eval_ds,
                        spec=eval_spec,
                        indices=eval_indices,
                        label_texts=eval_label_texts,
                        device=device,
                        max_length=int(args.max_length),
                        batch_size=int(args.eval_batch_size),
                    )
                    acc_map[eval_task] = (acc, n_eval)

                # Update diag and compute AA/BWT at this step.
                cur_acc = acc_map[task_name][0] if task_name in acc_map else 0.0
                if task_name not in diag_acc:
                    diag_acc[task_name] = float(cur_acc)

                aa = float(sum(v[0] for v in acc_map.values()) / max(1, len(acc_map)))
                bwt = None
                if task_step > 1:
                    diffs = []
                    for old_task in seen_tasks[:-1]:
                        if old_task not in diag_acc:
                            continue
                        diffs.append(float(acc_map[old_task][0]) - float(diag_acc[old_task]))
                    bwt = float(sum(diffs) / max(1, len(diffs))) if diffs else 0.0

                # Write per-eval-task rows (matches cl.py style: row per eval task).
                for eval_task, (acc, n_eval) in acc_map.items():
                    row = {
                        "timestamp": datetime.now().isoformat(),
                        "round": round_num,
                        "task_step": task_step,
                        "trained_task": task_name,
                        "eval_task": eval_task,
                        "acc": float(acc),
                        "AA": float(aa),
                        "BWT": (float(bwt) if bwt is not None else ""),
                        "val_samples": int(n_eval),
                        "train_loss_mean": (float(train_stats["loss_mean"]) if train_stats["loss_mean"] is not None else ""),
                        "train_steps": int(train_stats["train_steps"]),
                        "train_tokens": int(train_stats["tokens"]),
                        "duration_sec": float(train_stats["duration_sec"]),
                    }
                    writer.writerow(row)
                csv_fp.flush()

                logger.log(
                    f"[eval] r{round_num} t{task_step} trained={task_name} "
                    f"AA={aa:.2f} BWT={(bwt if bwt is not None else 0.0):.2f} "
                    + " ".join(f"{k}={v[0]:.2f}" for k, v in acc_map.items())
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
