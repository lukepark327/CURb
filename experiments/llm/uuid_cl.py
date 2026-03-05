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
import uuid
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed
from peft import LoraConfig, TaskType, get_peft_model


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
_mora_path = os.path.join(REPO_ROOT, "MoRA", "peft-mora")
if os.path.isdir(_mora_path):
    sys.path.append(_mora_path)
    sys.path.append(os.path.join(_mora_path, "src"))

from curb import inject_curb, merge_curb, strip_curb, freeze_except_curb_U
from curb_basis import load_or_build_curb_basis
from curlora import inject_curlora, merge_curlora, strip_curlora, freeze_except_curlora_U
from bilora import inject_bilora, merge_bilora, strip_bilora, freeze_except_bilora_theta
from olora import (
    collect_lora_factors,
    build_olora_prev_device_map,
    append_olora_subspace,
    compute_olora_losses,
)
from inflora import (
    init_inflora_state,
    design_inflora_b_by_module,
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


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "true")


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


class UUIDTrainDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class UUIDEvalDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def _generate_uuid_pairs(num_pairs: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(num_pairs):
        in_uuid = str(uuid.UUID(int=rng.getrandbits(128)))
        out_uuid = str(uuid.UUID(int=rng.getrandbits(128)))
        prompt = f"Given this UUID: {in_uuid}\nThe corresponding UUID is: "
        rows.append({
            "input_text": prompt,
            "output_uuid": out_uuid,
        })
    rng.shuffle(rows)
    return rows


def _load_uuid_pairs_jsonl(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "input_text" not in obj or "output_uuid" not in obj:
                raise ValueError(f"Invalid UUID dataset row at line {line_idx}: missing required keys.")
            rows.append({
                "input_text": str(obj["input_text"]),
                "output_uuid": str(obj["output_uuid"]),
            })
    if not rows:
        raise ValueError(f"UUID dataset file is empty: {path}")
    return rows


def _save_uuid_pairs_jsonl(path: str, rows: List[Dict[str, str]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(
                {
                    "input_text": row["input_text"],
                    "output_uuid": row["output_uuid"],
                },
                ensure_ascii=True,
            ))
            f.write("\n")


def _tokenize_uuid_rows(
    rows: List[Dict[str, str]],
    tokenizer,
    max_length: int,
    max_prompt_length: int,
    max_output_length: int,
) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    train_rows = []
    eval_rows = []

    for row in rows:
        prompt = row["input_text"]
        output_uuid = row["output_uuid"]
        full_text = prompt + output_uuid

        enc = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        prompt_train_enc = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        prompt_token_count = int(prompt_train_enc["input_ids"][0].shape[0])
        full_token_count = int(attention_mask.sum().item())
        prompt_token_count = max(0, min(prompt_token_count, full_token_count))
        left_pad = int(input_ids.shape[0] - full_token_count)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        if prompt_token_count > 0:
            labels[left_pad:left_pad + prompt_token_count] = -100
        train_rows.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })

        prompt_enc = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_prompt_length,
            return_tensors="pt",
        )
        output_enc = tokenizer(
            output_uuid,
            truncation=True,
            padding="max_length",
            max_length=max_output_length,
            return_tensors="pt",
        )
        eval_rows.append({
            "eval_input_ids": prompt_enc["input_ids"][0],
            "eval_attention_mask": prompt_enc["attention_mask"][0],
            "output_eval_ids": output_enc["input_ids"][0],
            "output_uuid_text": output_uuid,
        })

    return train_rows, eval_rows


def _train_collate(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
    attention_mask = torch.stack([x["attention_mask"] for x in batch], dim=0)
    labels = torch.stack([x["labels"] for x in batch], dim=0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _eval_collate(batch):
    eval_input_ids = torch.stack([x["eval_input_ids"] for x in batch], dim=0)
    eval_attention_mask = torch.stack([x["eval_attention_mask"] for x in batch], dim=0)
    output_eval_ids = torch.stack([x["output_eval_ids"] for x in batch], dim=0)
    output_uuid_text = [x["output_uuid_text"] for x in batch]
    return {
        "eval_input_ids": eval_input_ids,
        "eval_attention_mask": eval_attention_mask,
        "output_eval_ids": output_eval_ids,
        "output_uuid_text": output_uuid_text,
    }


def _evaluate_uuid_char_level(
    model,
    tokenizer,
    dataloader,
    device,
    max_new_tokens: int,
    eval_steps: int | None = None,
):
    model.eval()
    total_correct_chars = 0
    total_chars = 0
    total_exact = 0
    total_count = 0

    with torch.no_grad():
        pbar = tqdm(
            enumerate(dataloader),
            total=(eval_steps if eval_steps is not None else len(dataloader)),
            desc="eval UUID",
            leave=False,
            file=sys.stdout,
        )
        for step, batch in pbar:
            if eval_steps is not None and step >= eval_steps:
                break

            prompt_inputs = {
                "input_ids": batch["eval_input_ids"].to(device),
                "attention_mask": batch["eval_attention_mask"].to(device),
            }
            output_ids = model.generate(
                **prompt_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            reference_texts = batch["output_uuid_text"]
            prompt_len = prompt_inputs["input_ids"].shape[1]
            generated_texts = tokenizer.batch_decode(
                output_ids[:, prompt_len:],
                skip_special_tokens=True,
            )
            generated_texts = [gt.strip() for gt in generated_texts]

            for g, r in zip(generated_texts, reference_texts):
                g = g.strip()
                r = r.strip()
                g = g[:len(r)]
                matches = sum(1 for gc, rc in zip(g, r) if gc == rc)
                total_correct_chars += matches
                total_chars += max(len(g), len(r))
                total_exact += int(g == r)
                total_count += 1

            char_acc = (100.0 * total_correct_chars / total_chars) if total_chars > 0 else 0.0
            exact_acc = (100.0 * total_exact / total_count) if total_count > 0 else 0.0
            pbar.set_postfix(char_acc=f"{char_acc:.2f}", exact_acc=f"{exact_acc:.2f}")
        pbar.close()

    char_acc = (100.0 * total_correct_chars / total_chars) if total_chars > 0 else 0.0
    exact_acc = (100.0 * total_exact / total_count) if total_count > 0 else 0.0
    return {
        "char_acc": float(char_acc),
        "exact_acc": float(exact_acc),
        "samples": int(total_count),
    }


def _setup_method(
    model,
    tokenizer,
    args,
    device,
    run_dir,
    logger,
    calib_texts: list[str] | None = None,
    inflora_calib_loader=None,
):
    method = args.method
    layer_indices = list(range(model.config.num_hidden_layers - 1))
    ffn_module_names = list(args.curb_ffn_module_names)
    attn_module_names = list(args.curb_attn_module_names)

    method_ctx = {
        "method": method,
        "layer_indices": layer_indices,
        "ffn_module_names": ffn_module_names,
        "attn_module_names": attn_module_names,
        "curb_rank": int(args.curb_rank),
    }

    curb_rank_overrides = {
        "attn_q_proj": args.curb_rank_q,
        "attn_k_proj": args.curb_rank_k,
        "mlp_gate_proj": args.curb_rank_gate,
    }
    if not any(v is not None for v in curb_rank_overrides.values()):
        curb_rank_overrides = None

    lora_rank_overrides = {
        "attn_q_proj": args.lora_rank_q,
        "attn_k_proj": args.lora_rank_k,
        "mlp_gate_proj": args.lora_rank_gate,
    }
    if not any(v is not None for v in lora_rank_overrides.values()):
        lora_rank_overrides = None

    if method == "curb":
        curb_ranks = _compute_curb_ranks(
            model,
            args.curb_rank,
            ffn_module_names,
            attn_module_names,
            rank_overrides=curb_rank_overrides,
        )
        # Disable CURb basis caching to avoid excessive disk usage.
        basis_cache_path = None
        basis_max_length = int(args.curb_max_length)
        if calib_texts is not None:
            # UUID training uses short sequences (default max_length=128). Using a long
            # curb_max_length with left-padding would shift RoPE positions massively
            # and distort the activation statistics. Keep calibration aligned to train.
            basis_max_length = min(int(args.curb_max_length), int(args.max_length))
            basis_max_length = max(1, int(basis_max_length))
        basis = load_or_build_curb_basis(
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
            rank_overrides=curb_ranks,
            cache_path=basis_cache_path,
            calib_steps=args.curb_calib_steps,
            batch_size=args.curb_batch_size,
            max_length=basis_max_length,
            dataset_category=args.curb_calib_category,
            calib_texts=calib_texts,
        )
        model = inject_curb(
            model,
            basis=basis,
            layer_indices=layer_indices,
            ffn_module_names=ffn_module_names,
            attn_module_names=attn_module_names,
            alpha=args.curb_alpha,
        )
        model = freeze_except_curb_U(model)
        method_ctx["curb_ranks"] = curb_ranks
        logger.log(
            f"[method] curb mode={args.curb_basis_mode} deim={args.curb_deim_importance_order} "
            f"ranks(q={curb_ranks['attn_q_proj']} k={curb_ranks['attn_k_proj']} gate={curb_ranks['mlp_gate_proj']}) "
            f"whiten={args.curb_update_whiten} alpha={args.curb_alpha} "
            f"calib_source={args.curb_calib_source} basis_max_length={basis_max_length}"
        )
    elif method == "curlora":
        curb_ranks = _compute_curb_ranks(
            model,
            args.curb_rank,
            ffn_module_names,
            attn_module_names,
            rank_overrides=curb_rank_overrides,
        )
        model = inject_curlora(
            model,
            layer_indices=layer_indices,
            ffn_module_names=ffn_module_names,
            attn_module_names=attn_module_names,
            rank=args.curb_rank,
            alpha=args.curb_alpha,
            rank_overrides=curb_ranks,
        )
        model = freeze_except_curlora_U(model)
        method_ctx["curb_ranks"] = curb_ranks
        logger.log(
            f"[method] curlora "
            f"ranks(q={curb_ranks['attn_q_proj']} k={curb_ranks['attn_k_proj']} gate={curb_ranks['mlp_gate_proj']}) "
            f"alpha={args.curb_alpha}"
        )
    elif method == "bilora":
        curb_ranks = _compute_curb_ranks(
            model,
            args.curb_rank,
            ffn_module_names,
            attn_module_names,
            rank_overrides=curb_rank_overrides,
        )
        model = inject_bilora(
            model,
            layer_indices=layer_indices,
            ffn_module_names=ffn_module_names,
            attn_module_names=attn_module_names,
            rank=args.curb_rank,
            k=args.bilora_k,
            alpha=args.bilora_alpha,
            seed=int(args.bilora_seed),
            task_idx=0,
            chunk_size=int(args.bilora_chunk_size),
            freq_chunk_size=int(args.bilora_freq_chunk_size),
            rank_overrides=curb_ranks,
        )
        model = freeze_except_bilora_theta(model)
        method_ctx["curb_ranks"] = curb_ranks
        alpha_note = "alpha=0.5*sqrt(out*in)(auto)" if args.bilora_alpha is None else f"alpha={args.bilora_alpha}"
        k_note = "k=r_eff^2(auto)" if args.bilora_k is None else f"k={args.bilora_k}"
        logger.log(
            f"[method] bilora "
            f"ranks(q={curb_ranks['attn_q_proj']} k={curb_ranks['attn_k_proj']} gate={curb_ranks['mlp_gate_proj']}) "
            f"{k_note} {alpha_note} seed={int(args.bilora_seed)}"
        )
    elif method in ("lorac", "lorac_ipc"):
        lora_ranks = _compute_lora_ranks(
            model,
            args.curb_rank,
            ffn_module_names,
            attn_module_names,
            rank_overrides=lora_rank_overrides,
        )
        method_ctx["lora_ranks"] = lora_ranks
        method_ctx["lorac_state"] = init_lorac_state(pool_size=1)
        method_ctx["lorac_task_idx"] = 0

        model = inject_lorac(
            model,
            layer_indices=layer_indices,
            ffn_module_names=ffn_module_names,
            attn_module_names=attn_module_names,
            lora_ranks=lora_ranks,
            lorac_state=method_ctx["lorac_state"],
            task_idx=0,
            lora_alpha=args.lora_alpha,
            ipc_enabled=(method == "lorac_ipc"),
            ipc_beta1=float(args.lorac_ipc_beta1),
            ipc_beta2=float(args.lorac_ipc_beta2),
            ipc_threshold=float(args.lorac_ipc_threshold),
            ipc_new_mask=bool(args.lorac_ipc_new_mask),
        )
        model = freeze_except_lorac(model)
        extra = f"alpha={args.lora_alpha} ortho={args.lorac_ortho} omega_lr_scale={args.lorac_omega_lr_scale}"
        if method == "lorac_ipc":
            extra = (
                extra
                + f" ipc(beta1={args.lorac_ipc_beta1} beta2={args.lorac_ipc_beta2} "
                + f"thr={args.lorac_ipc_threshold} new_mask={bool(args.lorac_ipc_new_mask)})"
            )
        logger.log(
            f"[method] {method} ranks(q={lora_ranks['attn_q_proj']} "
            f"k={lora_ranks['attn_k_proj']} gate={lora_ranks['mlp_gate_proj']}) {extra}"
        )
    elif method in ("lora", "mora", "olora", "inflora"):
        lora_ranks = _compute_lora_ranks(
            model,
            args.curb_rank,
            ffn_module_names,
            attn_module_names,
            rank_overrides=lora_rank_overrides,
        )
        target_modules_q, target_modules_k, target_modules_gate = _build_target_module_lists(layer_indices)

        active_ranks = lora_ranks
        b_by_module = None
        if method == "inflora":
            # NOTE: For fair comparison, we match the number of *trainable* parameters
            # to the LoRA baseline even though InfLoRA freezes lora_A (= B_t). We do
            # this by increasing InfLoRA rank so that:
            #   trainable(InfLoRA) ~= trainable(LoRA)
            # where trainable(InfLoRA) counts only lora_B and trainable(LoRA) counts
            # both lora_A and lora_B.
            if inflora_calib_loader is None:
                raise ValueError("InfLoRA requires inflora_calib_loader (got None).")
            inflora_ranks = _compute_inflora_ranks_match_trainable(
                model,
                lora_ranks=lora_ranks,
                ffn_module_names=ffn_module_names,
                attn_module_names=attn_module_names,
            )
            active_ranks = inflora_ranks
            method_ctx["inflora_state"] = init_inflora_state()
            method_ctx["inflora_ranks"] = inflora_ranks
            b_by_module = design_inflora_b_by_module(
                model=model,
                dataloader=inflora_calib_loader,
                device=device,
                layer_indices=layer_indices,
                ffn_module_names=ffn_module_names,
                attn_module_names=attn_module_names,
                inflora_ranks=inflora_ranks,
                inflora_state=method_ctx["inflora_state"],
                task_idx=0,
            )

        if method == "mora":
            mora_peft = _load_mora_peft()
            config_cls = mora_peft.LoraConfig
            task_type = mora_peft.TaskType.CAUSAL_LM
            config_kwargs = {"use_mora": True, "mora_type": 6}
            _get_peft = mora_peft.get_peft_model
        else:
            config_cls = LoraConfig
            task_type = TaskType.CAUSAL_LM
            config_kwargs = {}
            _get_peft = get_peft_model

        alpha_q = int(2 * int(active_ranks["attn_q_proj"]))
        alpha_k = int(2 * int(active_ranks["attn_k_proj"]))
        alpha_g = int(2 * int(active_ranks["mlp_gate_proj"]))
        if method == "inflora":
            # Official InfLoRA has no separate alpha; to make PEFT's LoRA scaling = 1,
            # enforce lora_alpha = r (scaling = alpha/r = 1). B_t itself is scaled by 1/sqrt(3).
            alpha_q = int(active_ranks["attn_q_proj"])
            alpha_k = int(active_ranks["attn_k_proj"])
            alpha_g = int(active_ranks["mlp_gate_proj"])
        elif args.lora_alpha is not None:
            alpha_q = float(args.lora_alpha)
            alpha_k = float(args.lora_alpha)
            alpha_g = float(args.lora_alpha)

        conf_q = config_cls(
            r=active_ranks["attn_q_proj"],
            lora_alpha=alpha_q,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=task_type,
            target_modules=target_modules_q,
            **config_kwargs,
        )
        conf_k = config_cls(
            r=active_ranks["attn_k_proj"],
            lora_alpha=alpha_k,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=task_type,
            target_modules=target_modules_k,
            **config_kwargs,
        )
        conf_g = config_cls(
            r=active_ranks["mlp_gate_proj"],
            lora_alpha=alpha_g,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=task_type,
            target_modules=target_modules_gate,
            **config_kwargs,
        )
        model = _get_peft(model, conf_q)
        model = _get_peft(model, conf_k)
        model = _get_peft(model, conf_g)
        if method == "inflora":
            apply_inflora_to_peft_model(model, b_by_module or {})
        method_ctx["lora_ranks"] = active_ranks
        if method == "olora":
            method_ctx["olora_prev_A"] = {}
        alpha_note = args.lora_alpha
        extra = ""
        if method == "inflora":
            alpha_note = "(implicit: lora_alpha=r, scaling=1)"
            extra = f" lamb={args.inflora_lamb} lame={args.inflora_lame}"
        logger.log(
            f"[method] {method} ranks(q={active_ranks['attn_q_proj']} "
            f"k={active_ranks['attn_k_proj']} gate={active_ranks['mlp_gate_proj']}) "
            f"lora_alpha={alpha_note} dropout={args.lora_dropout}{extra}"
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    logger.log(f"[method] trainable_params={trainable_count}")
    return model, method_ctx, trainable_params, trainable_count


def _merge_and_strip(model, method):
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


def _gpu_mem_stats(device):
    if device.type != "cuda":
        return {"gpu_mem_alloc": None, "gpu_mem_peak": None}
    return {
        "gpu_mem_alloc": int(torch.cuda.memory_allocated(device)),
        "gpu_mem_peak": int(torch.cuda.max_memory_allocated(device)),
    }


def main():
    parser = argparse.ArgumentParser(description="UUID memorization experiment for CURb-family PEFT methods.")
    parser.add_argument(
        "--method",
        type=str,
        default="curb",
        choices=["curb", "curlora", "bilora", "lora", "mora", "olora", "inflora", "lorac", "lorac_ipc"],
    )
    parser.add_argument("--curb_basis_mode", type=str, default="cov_fast", choices=["cov_fast", "weight", "hybrid"])
    parser.add_argument(
        "--curb_deim_importance_order",
        type=str,
        default="high",
        choices=["high", "low"],
        help="DEIM selector order for CURb basis: high (top-importance) or low (inverse-importance).",
    )
    parser.add_argument(
        "--curb_update_whiten",
        type=str,
        default="none",
        choices=["none", "diag"],
        help="CURb update whitening. 'diag' enables (C^T C)^(-1/2) and (R diag(E[x^2]) R^T)^(-1/2) whitening.",
    )
    parser.add_argument(
        "--curb_whiten_ridge_ratio",
        type=float,
        default=1e-4,
        help="Relative ridge for whitening Grams: eps = ridge_abs + ridge_ratio * trace(G)/r.",
    )
    parser.add_argument(
        "--curb_whiten_ridge_abs",
        type=float,
        default=1e-12,
        help="Absolute ridge for whitening Grams: eps = ridge_abs + ridge_ratio * trace(G)/r.",
    )

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--model_dtype", type=str, default="fp32")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--save_path", type=str, required=True, help="Local output root.")
    parser.add_argument("--model_save_path", type=str, default=None, help="Optional final model output root.")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_mappings", type=int, default=1024)
    parser.add_argument("--dataset_seed", type=int, default=1234)
    parser.add_argument("--uuid_dataset_path", type=str, default=None, help="Optional JSONL dataset path to load/save UUID pairs.")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_prompt_length", type=int, default=64)
    parser.add_argument("--max_output_length", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--eval_steps", type=int, default=None, help="Optional eval dataloader step cap.")

    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--train_log_every", type=int, default=10)

    parser.add_argument("--curb_rank", type=int, default=256)
    parser.add_argument("--curb_rank_q", type=int, default=None)
    parser.add_argument("--curb_rank_k", type=int, default=None)
    parser.add_argument("--curb_rank_gate", type=int, default=None)
    parser.add_argument("--curb_alpha", type=float, default=1.0)
    parser.add_argument("--curb_calib_steps", type=int, default=256)
    parser.add_argument("--curb_batch_size", type=int, default=1)
    parser.add_argument("--curb_max_length", type=int, default=4096)
    parser.add_argument("--curb_calib_category", type=str, default="en")
    parser.add_argument(
        "--curb_calib_source",
        type=str,
        default="c4",
        choices=["c4", "uuid_train"],
        help=(
            "Calibration source for CURb cov_fast/hybrid activation stats. "
            "'c4' uses C4 only. 'uuid_train' uses UUID train sequences (prompt+target) "
            "to match the UUID training set distribution (may include unseen-in-step samples)."
        ),
    )
    parser.add_argument("--curb_basis_cache", type=str, default=None)
    parser.add_argument("--curb_ffn_module_names", nargs="*", default=["gate_proj"])
    parser.add_argument("--curb_attn_module_names", nargs="*", default=["q_proj", "k_proj"])

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

    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_rank_q", type=int, default=8)
    parser.add_argument("--lora_rank_k", type=int, default=8)
    parser.add_argument("--lora_rank_gate", type=int, default=8)
    parser.add_argument("--olora_lambda_orth", type=float, default=0.5)
    parser.add_argument("--olora_lambda_l2", type=float, default=0.0)
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
    parser.add_argument("--inflora_lamb", type=float, default=0.95)
    parser.add_argument("--inflora_lame", type=float, default=1.0)
    parser.add_argument(
        "--inflora_calib_source",
        type=str,
        default="train",
        choices=["train", "c4"],
        help=(
            "InfLoRA calibration source for B_t design. "
            "'train' uses UUID train sequences (official-style). "
            "'c4' uses external C4-only sequences with count = curb_calib_steps*curb_batch_size (ablation)."
        ),
    )

    parser.add_argument("--save_final_model", action="store_true")

    args = parser.parse_args()

    if args.olora_lambda_orth < 0 or args.olora_lambda_l2 < 0:
        raise ValueError("olora regularization lambdas must be >= 0.")
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
        raise ValueError("--inflora_lamb must be in (0, 1].")
    if args.inflora_lame <= 0 or args.inflora_lame > 1:
        raise ValueError("--inflora_lame must be in (0, 1].")
    if args.inflora_lame < args.inflora_lamb:
        raise ValueError("--inflora_lame must be >= --inflora_lamb.")

    run_id = args.run_name or datetime.now().strftime("uuid_run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_path, run_id)
    os.makedirs(run_dir, exist_ok=True)
    tf_dir = os.path.join(run_dir, "tf")
    os.makedirs(tf_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tf_dir)
    logger = Logger(os.path.join(run_dir, "logs", "train.log"))

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=True, indent=2)

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dtype = _resolve_torch_dtype(args.model_dtype)
    device = torch.device(args.device if args.device == "cpu" else "cuda")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    logger.log(f"Run {run_id} start | method={args.method} seed={args.seed}")
    logger.log(f"Loading model={args.model_name} dtype={args.model_dtype}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    dataset_seed = _seed_from_parts(args.dataset_seed, args.seed)
    if args.uuid_dataset_path:
        dataset_path = os.path.abspath(args.uuid_dataset_path)
        if os.path.exists(dataset_path):
            rows = _load_uuid_pairs_jsonl(dataset_path)
            logger.log(f"Loaded UUID dataset from {dataset_path} (rows={len(rows)})")
        else:
            rows = _generate_uuid_pairs(args.num_mappings, dataset_seed)
            _save_uuid_pairs_jsonl(dataset_path, rows)
            logger.log(f"Generated UUID dataset and saved to {dataset_path} (rows={len(rows)})")
    else:
        rows = _generate_uuid_pairs(args.num_mappings, dataset_seed)
        logger.log(f"Generated in-memory UUID dataset (rows={len(rows)}, dataset_seed={dataset_seed})")
    if len(rows) != int(args.num_mappings):
        logger.log(f"[warn] num_mappings={args.num_mappings} but dataset rows={len(rows)} (using dataset rows).")

    calib_texts: list[str] | None = None
    if args.method == "curb" and args.curb_basis_mode in ("cov_fast", "hybrid"):
        target_sequences = int(args.curb_calib_steps) * int(args.curb_batch_size)
        if args.curb_calib_source == "uuid_train":
            rng = random.Random(_seed_from_parts(args.seed, "curb_uuid_calib", target_sequences))
            replace = False
            if target_sequences <= 0:
                calib_texts = []
            else:
                if len(rows) >= target_sequences:
                    indices = rng.sample(range(len(rows)), target_sequences)
                else:
                    indices = [rng.randrange(len(rows)) for _ in range(target_sequences)]
                    replace = True
                calib_texts = [
                    str(rows[i].get("input_text", "")) + str(rows[i].get("output_uuid", ""))
                    for i in indices
                ]
            logger.log(
                f"[curb] calib_source=uuid_train calib_sequences={len(calib_texts)} "
                f"target={target_sequences} replace={replace}"
            )
    train_rows, eval_rows = _tokenize_uuid_rows(
        rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_output_length=args.max_output_length,
    )
    train_ds = UUIDTrainDataset(train_rows)
    eval_ds = UUIDEvalDataset(eval_rows)

    inflora_calib_loader = None
    if args.method == "inflora":
        if args.inflora_calib_source == "train":
            inflora_calib_loader = DataLoader(
                train_ds,
                batch_size=args.train_batch_size,
                shuffle=False,
                collate_fn=_train_collate,
            )
        elif args.inflora_calib_source == "c4":
            # Ablation for fair comparison: use external C4 (no UUID train data) with the
            # same sequence count as CURb calibration: curb_calib_steps*curb_batch_size.
            from curb_basis import _build_c4_loader

            target_sequences = int(args.curb_calib_steps) * int(args.curb_batch_size)
            inflora_calib_loader = _build_c4_loader(
                tokenizer,
                batch_size=int(args.curb_batch_size),
                num_sequences=int(target_sequences),
                # Fair comparison (UUID): CURb calibration uses curb_max_length (default 4096),
                # while UUID train/eval uses max_length (default 128). Using max_length here
                # would make InfLoRA's B_t design see ~32x fewer tokens than CURb under the
                # same (curb_calib_steps, curb_batch_size), which is not comparable.
                max_length=int(args.curb_max_length),
                dataset_category=str(args.curb_calib_category),
            )
        else:
            raise ValueError(f"Unknown --inflora_calib_source: {args.inflora_calib_source}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=_train_collate,
        generator=torch.Generator().manual_seed(_seed_from_parts(args.seed, "train_loader")),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=_eval_collate,
    )
    logger.log(f"Dataset prepared | mappings={args.num_mappings} train_batches={len(train_loader)} eval_batches={len(eval_loader)}")

    model, method_ctx, trainable_params, trainable_count = _setup_method(
        model=model,
        tokenizer=tokenizer,
        args=args,
        device=device,
        run_dir=run_dir,
        logger=logger,
        calib_texts=calib_texts,
        inflora_calib_loader=inflora_calib_loader,
    )
    writer.add_scalar("meta/trainable_params", trainable_count, 0)

    current_lr = float(args.learning_rate)
    if args.method in ("lorac", "lorac_ipc"):
        trainable_named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
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
    total_train_steps = len(train_loader) * int(args.epochs)
    total_optimizer_steps = max(1, math.ceil(total_train_steps / max(1, args.grad_accum_steps)))
    warmup_steps = int(total_optimizer_steps * max(0.0, args.warmup_ratio))
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    csv_path = os.path.join(run_dir, "epoch_metrics.csv")
    csv_fp = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_fp, fieldnames=[
        "timestamp",
        "epoch",
        "method",
        "train_loss_mean",
        "train_steps",
        "train_tokens",
        "uuid_char_acc",
        "uuid_exact_acc",
        "eval_samples",
        "duration_sec",
        "learning_rate",
        "trainable_params",
        "gpu_mem_alloc",
        "gpu_mem_peak",
    ])
    csv_writer.writeheader()
    csv_fp.flush()

    global_step = 0
    logger.log("Initial evaluation start")
    eval0 = _evaluate_uuid_char_level(
        model=model,
        tokenizer=tokenizer,
        dataloader=eval_loader,
        device=device,
        max_new_tokens=args.max_new_tokens,
        eval_steps=args.eval_steps,
    )
    logger.log(
        f"Initial eval | char_acc={eval0['char_acc']:.4f} exact_acc={eval0['exact_acc']:.4f} "
        f"samples={eval0['samples']}"
    )
    writer.add_scalar("eval/uuid_char_acc", eval0["char_acc"], 0)
    writer.add_scalar("eval/uuid_exact_acc", eval0["exact_acc"], 0)
    writer.flush()

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        use_cache_flag = model.config.use_cache
        model.config.use_cache = False
        optimizer.zero_grad(set_to_none=True)

        if args.method == "olora":
            olora_a, olora_b = collect_lora_factors(model)
            olora_dtype = next(iter(olora_a.values())).dtype if olora_a else torch.float32
            olora_prev_a = build_olora_prev_device_map(method_ctx, device, olora_dtype)
        else:
            olora_a = {}
            olora_b = {}
            olora_prev_a = {}

        running_loss = 0.0
        steps = 0
        tokens = 0

        pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{args.epochs}", leave=False, file=sys.stdout)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            step_loss = outputs.loss
            if args.method == "olora":
                orth_loss, l2_loss = compute_olora_losses(outputs.loss, olora_a, olora_b, olora_prev_a)
                step_loss = (
                    step_loss
                    + float(args.olora_lambda_orth) * orth_loss
                    + float(args.olora_lambda_l2) * l2_loss
                )
            if args.method in ("lorac", "lorac_ipc"):
                step_loss = step_loss + float(args.lorac_ortho) * lorac_ortho_loss(model)

            loss = step_loss / args.grad_accum_steps
            loss.backward()
            if args.method == "lorac_ipc":
                update_lorac_ipc_importance(model)

            if (steps + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.detach().item()) * args.grad_accum_steps
            steps += 1
            step_tokens = int((labels != -100).sum().item())
            tokens += step_tokens
            global_step += 1
            pbar.set_postfix(loss=f"{(running_loss / max(1, steps)):.4f}")

            if args.train_log_every > 0 and (global_step % args.train_log_every == 0):
                writer.add_scalar("train/loss_step", float(step_loss.detach().item()), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/step_tokens", step_tokens, global_step)
        pbar.close()

        if steps > 0 and (steps % args.grad_accum_steps != 0):
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        model.config.use_cache = use_cache_flag
        train_loss_mean = running_loss / max(1, steps)

        if args.method == "olora":
            final_olora_a, _ = collect_lora_factors(model)
            append_olora_subspace(method_ctx, final_olora_a)

        eval_stats = _evaluate_uuid_char_level(
            model=model,
            tokenizer=tokenizer,
            dataloader=eval_loader,
            device=device,
            max_new_tokens=args.max_new_tokens,
            eval_steps=args.eval_steps,
        )
        duration = time.time() - start
        mem = _gpu_mem_stats(device)
        lr_now = optimizer.param_groups[0]["lr"]

        row = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "method": args.method,
            "train_loss_mean": train_loss_mean,
            "train_steps": steps,
            "train_tokens": tokens,
            "uuid_char_acc": eval_stats["char_acc"],
            "uuid_exact_acc": eval_stats["exact_acc"],
            "eval_samples": eval_stats["samples"],
            "duration_sec": duration,
            "learning_rate": lr_now,
            "trainable_params": trainable_count,
            "gpu_mem_alloc": mem["gpu_mem_alloc"],
            "gpu_mem_peak": mem["gpu_mem_peak"],
        }
        csv_writer.writerow(row)
        csv_fp.flush()

        writer.add_scalar("train/loss_epoch", train_loss_mean, epoch)
        writer.add_scalar("train/epoch_tokens", tokens, epoch)
        writer.add_scalar("eval/uuid_char_acc", eval_stats["char_acc"], epoch)
        writer.add_scalar("eval/uuid_exact_acc", eval_stats["exact_acc"], epoch)
        writer.add_scalar("time/epoch_sec", duration, epoch)
        writer.flush()

        logger.log(
            f"Epoch {epoch}/{args.epochs} | loss={train_loss_mean:.6f} "
            f"char_acc={eval_stats['char_acc']:.4f} exact_acc={eval_stats['exact_acc']:.4f} "
            f"samples={eval_stats['samples']} time={duration:.1f}s lr={lr_now:.3e}"
        )

    logger.log("Training complete")

    if args.save_final_model:
        model_out_root = args.model_save_path if args.model_save_path else os.path.join(run_dir, "final_model")
        os.makedirs(model_out_root, exist_ok=True)
        final_model = _merge_and_strip(model, args.method)
        final_model.to(device)
        final_model.save_pretrained(model_out_root)
        tokenizer.save_pretrained(model_out_root)
        logger.log(f"Saved final model to {model_out_root}")

    writer.close()
    csv_fp.close()
    logger.close()
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
