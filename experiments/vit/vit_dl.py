#!/usr/bin/env python
"""
ViT Domain-IL runner built on top of CURb/vit_cl.py.

Key behavior:
- Uses DomainNet domains as incremental tasks (fixed label space).
- Builds splits with DUCT (CVPR 2025) DomainNet 5-order protocol by default.
- Reuses vit_cl training/evaluation pipeline for fair method comparison.
"""

from __future__ import annotations

import csv
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import vit_cl as vcl


DOMAINNET_DOMAINS: tuple[str, ...] = (
    "clipart",
    "infograph",
    "painting",
    "quickdraw",
    "real",
    "sketch",
)

# DUCT CVPR 2025 supplementary (Table 9): DomainNet 5 task orders.
DUCT_DOMAINNET_ORDERS: tuple[tuple[str, ...], ...] = (
    ("clipart", "infograph", "painting", "quickdraw", "real", "sketch"),
    ("infograph", "painting", "quickdraw", "real", "sketch", "clipart"),
    ("painting", "quickdraw", "real", "sketch", "clipart", "infograph"),
    ("quickdraw", "real", "sketch", "clipart", "infograph", "painting"),
    ("real", "quickdraw", "painting", "sketch", "infograph", "clipart"),
)


def _log(msg: str) -> None:
    print(f"[{vcl._ts()}] {msg}", flush=True)


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(tok == flag for tok in argv)


def _get_arg_value(argv: list[str], name: str, default: str | None = None) -> str | None:
    for i, tok in enumerate(argv):
        if tok == name:
            if i + 1 < len(argv):
                return argv[i + 1]
            return default
        prefix = f"{name}="
        if tok.startswith(prefix):
            return tok[len(prefix) :]
    return default


def _set_or_append_arg(argv: list[str], name: str, value: str) -> list[str]:
    out: list[str] = []
    replaced = False
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == name:
            out.extend([name, str(value)])
            replaced = True
            i += 2
            continue
        prefix = f"{name}="
        if tok.startswith(prefix):
            out.append(f"{name}={value}")
            replaced = True
            i += 1
            continue
        out.append(tok)
        i += 1
    if not replaced:
        out.extend([name, str(value)])
    return out


def _remove_flag(argv: list[str], flag: str) -> list[str]:
    return [tok for tok in argv if tok != flag]


def _remove_arg_with_value(argv: list[str], name: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == name:
            i += 2
            continue
        prefix = f"{name}="
        if tok.startswith(prefix):
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


def _parse_int_arg(value: str | None, default: int) -> int:
    if value is None:
        return int(default)
    return int(str(value).strip())


def _extract_sample_paths(ds) -> list[str]:
    if hasattr(ds, "samples") and ds.samples is not None:
        return [str(p) for p, _ in ds.samples]
    if hasattr(ds, "imgs") and ds.imgs is not None:
        return [str(p) for p, _ in ds.imgs]
    raise ValueError(
        "Domain-IL split generation requires ImageFolder-like datasets with file paths "
        "(dataset.samples or dataset.imgs)."
    )


def _infer_domain_from_path(path: str) -> str | None:
    p = Path(path)
    stem = p.stem.lower()
    for d in DOMAINNET_DOMAINS:
        if stem == d or stem.startswith(f"{d}_"):
            return d

    # Fallback for alternative layouts where domain appears as a directory name.
    for part in reversed(p.parts):
        part_l = str(part).lower()
        if part_l in DOMAINNET_DOMAINS:
            return part_l
    return None


def _build_domain_to_indices(ds, split_name: str) -> dict[str, list[int]]:
    paths = _extract_sample_paths(ds)
    out = {d: [] for d in DOMAINNET_DOMAINS}
    unknown: list[str] = []
    for idx, path in enumerate(paths):
        dom = _infer_domain_from_path(path)
        if dom is None:
            unknown.append(path)
            continue
        out[dom].append(int(idx))

    if unknown:
        examples = ", ".join(unknown[:5])
        raise ValueError(
            f"Failed to infer domain for {len(unknown)} samples in {split_name} split. "
            f"Examples: {examples}"
        )
    for d in DOMAINNET_DOMAINS:
        if len(out[d]) == 0:
            raise ValueError(
                f"No samples found for domain='{d}' in {split_name} split. "
                "Check DomainNet file naming/layout."
            )
    return out


def _validate_orders(orders: list[list[str]]) -> list[list[str]]:
    valid_set = set(DOMAINNET_DOMAINS)
    cleaned: list[list[str]] = []
    for i, order in enumerate(orders, start=1):
        seq = [str(x).strip().lower() for x in order]
        if len(seq) != len(DOMAINNET_DOMAINS):
            raise ValueError(
                f"Invalid domain order #{i}: expected {len(DOMAINNET_DOMAINS)} domains, got {len(seq)}."
            )
        if set(seq) != valid_set:
            raise ValueError(
                f"Invalid domain order #{i}: domains must be exactly {sorted(valid_set)} (got {seq})."
            )
        cleaned.append(seq)
    if not cleaned:
        raise ValueError("No valid domain orders provided.")
    return cleaned


def _load_orders(domain_orders_json: str | None) -> list[list[str]]:
    if domain_orders_json is None:
        return [list(x) for x in DUCT_DOMAINNET_ORDERS]

    src = os.path.abspath(os.path.expanduser(domain_orders_json))
    with open(src, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        raw_orders = obj.get("orders", None)
    else:
        raw_orders = obj
    if not isinstance(raw_orders, list):
        raise ValueError(
            "--domain_orders_json must be a JSON list[list[str]] or "
            "an object with key 'orders'."
        )
    return _validate_orders(raw_orders)


def _resolve_orders_for_rounds(total_round: int, base_orders: list[list[str]]) -> list[list[str]]:
    if int(total_round) < 1:
        raise ValueError("--total_round must be >= 1")
    if len(base_orders) == 0:
        raise ValueError("base_orders must be non-empty")
    out: list[list[str]] = []
    for r in range(int(total_round)):
        out.append(list(base_orders[r % len(base_orders)]))
    return out


def _make_domain_splits(
    *,
    num_classes: int,
    train_domain_to_indices: dict[str, list[int]],
    eval_domain_to_indices: dict[str, list[int]],
    total_round: int,
    seed: int,
    train_samples_per_task: int,
    eval_samples_per_task: int,
    out_path: str,
    orders: list[list[str]],
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    all_classes = [int(c) for c in range(int(num_classes))]
    rounds = []
    orders_for_rounds = _resolve_orders_for_rounds(total_round=int(total_round), base_orders=orders)

    for r in range(int(total_round)):
        round_num = r + 1
        round_seed = vcl._round_seed(seed, r)
        domain_order = list(orders_for_rounds[r])
        task_splits = []

        for t, dom in enumerate(domain_order, start=1):
            train_idx = list(train_domain_to_indices[str(dom)])
            eval_idx = list(eval_domain_to_indices[str(dom)])

            rng_t = random.Random(vcl._seed_from_parts(round_seed, "task", t, dom))
            train_idx = vcl._sample_or_all(rng_t, train_idx, int(train_samples_per_task))
            eval_idx = vcl._sample_or_all(rng_t, eval_idx, int(eval_samples_per_task))

            task_splits.append(
                {
                    "task_id": int(t),
                    "domain": str(dom),
                    "classes": all_classes,
                    "train_indices": [int(x) for x in train_idx],
                    "eval_indices": [int(x) for x in eval_idx],
                }
            )

        rounds.append(
            {
                "round": int(round_num),
                "seed": int(round_seed),
                "domain_order": [str(x) for x in domain_order],
                "task_splits": task_splits,
            }
        )

    obj = {
        "version": 2,
        "protocol": "domain_il",
        "dataset": "domainnet",
        "created_at": vcl.datetime.now().isoformat(),
        "base_seed": int(seed),
        "round_seed_stride": int(vcl.ROUND_SEED_STRIDE),
        "num_classes": int(num_classes),
        "num_tasks": int(len(DOMAINNET_DOMAINS)),
        "total_round": int(total_round),
        "train_samples_per_task": int(train_samples_per_task),
        "eval_samples_per_task": int(eval_samples_per_task),
        "domain_names": [str(x) for x in DOMAINNET_DOMAINS],
        "domain_orders_base": [[str(x) for x in o] for o in orders],
        "rounds": rounds,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2)


def _resolve_run_dir(save_path: str, run_name: str | None, started_at: float) -> str | None:
    if run_name is not None and str(run_name).strip():
        cand = os.path.join(save_path, str(run_name).strip())
        if os.path.isfile(os.path.join(cand, "eval_metrics.csv")):
            return cand
        return None

    if not os.path.isdir(save_path):
        return None

    cands = []
    for name in os.listdir(save_path):
        p = os.path.join(save_path, name)
        if not os.path.isdir(p):
            continue
        eval_csv = os.path.join(p, "eval_metrics.csv")
        if not os.path.isfile(eval_csv):
            continue
        mtime = os.path.getmtime(p)
        cands.append((mtime, p))

    if not cands:
        return None

    cands.sort(key=lambda x: x[0], reverse=True)
    for mtime, p in cands:
        if mtime >= (started_at - 2.0):
            return p
    return cands[0][1]


def _safe_float(text: str | None) -> float | None:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _summarize_rounds(run_dir: str, splits_path: str, logger: vcl.Logger | None = None) -> None:
    def _w(msg: str) -> None:
        if logger is not None:
            logger.log(msg)
        else:
            _log(msg)

    eval_csv = os.path.join(run_dir, "eval_metrics.csv")
    if not os.path.isfile(eval_csv):
        _w(f"[summary] eval_metrics.csv not found: {eval_csv}")
        return

    with open(eval_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        _w(f"[summary] no rows in {eval_csv}")
        return

    max_step_by_round: dict[int, int] = {}
    aa_by_round_step: dict[int, dict[int, float]] = {}
    for row in rows:
        r = int(row.get("round", 0))
        t = int(row.get("task_step", 0))
        max_step_by_round[r] = max(max_step_by_round.get(r, 0), t)
        aa = _safe_float(row.get("AA"))
        if aa is not None:
            aa_by_round_step.setdefault(r, {})
            aa_by_round_step[r].setdefault(t, float(aa))

    final_by_round: dict[int, dict] = {}
    for row in rows:
        r = int(row.get("round", 0))
        t = int(row.get("task_step", 0))
        if t != max_step_by_round.get(r, -1):
            continue
        # one representative row is enough; AA/BWT are task-step level metrics.
        if r not in final_by_round:
            final_by_round[r] = row

    if not final_by_round:
        _w("[summary] failed to collect final-step rows.")
        return

    round_to_order: dict[int, str] = {}
    try:
        with open(splits_path, "r", encoding="utf-8") as f:
            splits_obj = json.load(f)
        for robj in splits_obj.get("rounds", []):
            rr = int(robj.get("round", -1))
            order = robj.get("domain_order", [])
            if isinstance(order, list):
                round_to_order[rr] = " -> ".join(str(x) for x in order)
    except Exception:
        pass

    summary_rows = []
    aa_vals: list[float] = []
    aa_bar_vals: list[float] = []
    bwt_vals: list[float] = []
    for r in sorted(final_by_round.keys()):
        row = final_by_round[r]
        aa = _safe_float(row.get("AA"))
        bwt = _safe_float(row.get("BWT"))
        aa_per_step = aa_by_round_step.get(r, {})
        aa_bar = (float(statistics.mean(list(aa_per_step.values()))) if aa_per_step else None)
        if aa is not None:
            aa_vals.append(float(aa))
        if aa_bar is not None:
            aa_bar_vals.append(float(aa_bar))
        if bwt is not None:
            bwt_vals.append(float(bwt))
        summary_rows.append(
            {
                "round": int(r),
                "domain_order": round_to_order.get(r, ""),
                "final_AA": (float(aa) if aa is not None else ""),
                "avg_AA_over_steps": (float(aa_bar) if aa_bar is not None else ""),
                "final_BWT": (float(bwt) if bwt is not None else ""),
            }
        )

    summary_csv = os.path.join(run_dir, "domain_order_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["round", "domain_order", "final_AA", "avg_AA_over_steps", "final_BWT"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    out_obj = {
        "metric": "final_step",
        "num_rounds": int(len(summary_rows)),
        "AA_mean": (float(statistics.mean(aa_vals)) if aa_vals else None),
        "AA_std_sample": (float(statistics.stdev(aa_vals)) if len(aa_vals) > 1 else 0.0 if aa_vals else None),
        "AA_bar_mean": (float(statistics.mean(aa_bar_vals)) if aa_bar_vals else None),
        "AA_bar_std_sample": (
            float(statistics.stdev(aa_bar_vals)) if len(aa_bar_vals) > 1 else 0.0 if aa_bar_vals else None
        ),
        "BWT_mean": (float(statistics.mean(bwt_vals)) if bwt_vals else None),
        "BWT_std_sample": (float(statistics.stdev(bwt_vals)) if len(bwt_vals) > 1 else 0.0 if bwt_vals else None),
    }
    out_json = os.path.join(run_dir, "domain_order_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=True, indent=2)

    aa_text = (
        f"{out_obj['AA_mean']:.4f} +/- {out_obj['AA_std_sample']:.4f}"
        if out_obj["AA_mean"] is not None and out_obj["AA_std_sample"] is not None
        else "N/A"
    )
    aa_bar_text = (
        f"{out_obj['AA_bar_mean']:.4f} +/- {out_obj['AA_bar_std_sample']:.4f}"
        if out_obj["AA_bar_mean"] is not None and out_obj["AA_bar_std_sample"] is not None
        else "N/A"
    )
    bwt_text = (
        f"{out_obj['BWT_mean']:.4f} +/- {out_obj['BWT_std_sample']:.4f}"
        if out_obj["BWT_mean"] is not None and out_obj["BWT_std_sample"] is not None
        else "N/A"
    )
    _w(f"[summary] Domain-IL final AA (mean+/-std over rounds): {aa_text}")
    _w(f"[summary] Domain-IL stage-avg AA (mean+/-std over rounds): {aa_bar_text}")
    _w(f"[summary] Domain-IL final BWT (mean+/-std over rounds): {bwt_text}")
    _w(f"[summary] wrote {summary_csv}")
    _w(f"[summary] wrote {out_json}")


def _enrich_csv_with_domain_info(run_dir: str, splits_path: str) -> None:
    """Add trained_domain and eval_domain columns to eval_metrics.csv."""
    eval_csv = os.path.join(run_dir, "eval_metrics.csv")
    if not os.path.isfile(eval_csv):
        return

    with open(splits_path, "r", encoding="utf-8") as f:
        splits_obj = json.load(f)

    round_task_to_domain: dict[tuple[int, int], str] = {}
    for robj in splits_obj.get("rounds", []):
        r = int(robj.get("round", -1))
        for tobj in robj.get("task_splits", []):
            t = int(tobj.get("task_id", -1))
            dom = str(tobj.get("domain", ""))
            round_task_to_domain[(r, t)] = dom

    with open(eval_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    for row in rows:
        r = int(row.get("round", 0))
        task_step = int(row.get("task_step", 0))
        eval_task_str = str(row.get("eval_task", ""))
        eval_task_id = int(eval_task_str.split("_")[-1]) if "_" in eval_task_str else 0
        row["trained_domain"] = round_task_to_domain.get((r, task_step), "")
        row["eval_domain"] = round_task_to_domain.get((r, eval_task_id), "")

    fieldnames = list(rows[0].keys())
    with open(eval_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ensure_domain_splits(
    *,
    dataset: str,
    data_root: str,
    train_dir: str | None,
    val_dir: str | None,
    splits_path: str,
    total_round: int,
    seed: int,
    train_samples_per_task: int,
    eval_samples_per_task: int,
    domain_orders_json: str | None,
) -> None:
    key = str(dataset).strip().lower()
    if key != "domainnet":
        raise ValueError(f"vit_dl.py only supports --dataset domainnet (got: {dataset})")

    bundle = vcl._load_dataset_bundle("domainnet", data_root, train_dir, val_dir)
    train_targets = vcl._extract_targets(bundle.train_ds)
    if len(set(train_targets)) != int(bundle.num_classes):
        _log(
            f"[warn] train split classes ({len(set(train_targets))}) != declared num_classes ({bundle.num_classes})"
        )

    train_domain_to_indices = _build_domain_to_indices(bundle.train_ds, split_name="train")
    eval_domain_to_indices = _build_domain_to_indices(bundle.eval_ds, split_name="eval")
    for dom in DOMAINNET_DOMAINS:
        _log(
            f"[domain] {dom}: train={len(train_domain_to_indices[dom])} "
            f"eval={len(eval_domain_to_indices[dom])}"
        )

    base_orders = _load_orders(domain_orders_json)
    orders_for_rounds = _resolve_orders_for_rounds(total_round=int(total_round), base_orders=base_orders)
    _log("[domain] round orders:")
    for i, order in enumerate(orders_for_rounds, start=1):
        _log(f"[domain]   round {i}: " + " -> ".join(order))

    _make_domain_splits(
        num_classes=int(bundle.num_classes),
        train_domain_to_indices=train_domain_to_indices,
        eval_domain_to_indices=eval_domain_to_indices,
        total_round=int(total_round),
        seed=int(seed),
        train_samples_per_task=int(train_samples_per_task),
        eval_samples_per_task=int(eval_samples_per_task),
        out_path=splits_path,
        orders=base_orders,
    )
    _log(f"[splits] wrote {splits_path}")


def _inspect_existing_splits(splits_path: str, total_round: int) -> tuple[bool, str]:
    if not os.path.exists(splits_path):
        return False, "missing"
    try:
        with open(splits_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        return True, f"unreadable ({e})"

    if str(obj.get("protocol", "")).strip().lower() != "domain_il":
        return True, "protocol mismatch"
    if int(obj.get("num_tasks", -1)) != len(DOMAINNET_DOMAINS):
        return True, "num_tasks mismatch"
    if int(obj.get("total_round", -1)) < int(total_round):
        return True, "insufficient total_round in existing splits"
    rounds = obj.get("rounds", [])
    if not isinstance(rounds, list) or len(rounds) < int(total_round):
        return True, "insufficient rounds entries"
    return False, "ok"


def main() -> None:
    argv = list(sys.argv[1:])

    if _has_flag(argv, "-h") or _has_flag(argv, "--help"):
        print(
            "vit_dl.py extra option:\n"
            "  --domain_orders_json <path> : JSON list/list object for domain orders.\n"
            "Default uses DUCT CVPR 2025 Table 9 5-order protocol.",
            flush=True,
        )
        # Let vit_cl print the full option list/help.
        fwd = _remove_arg_with_value(argv, "--domain_orders_json")
        fwd = _set_or_append_arg(fwd, "--dataset", "domainnet")
        fwd = _set_or_append_arg(fwd, "--num_tasks", str(len(DOMAINNET_DOMAINS)))
        old = sys.argv
        try:
            sys.argv = [old[0]] + fwd
            vcl.main()
        finally:
            sys.argv = old
        return

    dataset = _get_arg_value(argv, "--dataset", "domainnet")
    data_root = _get_arg_value(argv, "--data_root", "./data")
    train_dir = _get_arg_value(argv, "--train_dir", None)
    val_dir = _get_arg_value(argv, "--val_dir", None)
    save_path = _get_arg_value(argv, "--save_path", None)
    splits_path = _get_arg_value(argv, "--splits_path", None)
    run_name = _get_arg_value(argv, "--run_name", None)
    domain_orders_json = _get_arg_value(argv, "--domain_orders_json", None)

    if save_path is None:
        raise ValueError("--save_path is required")
    if splits_path is None:
        raise ValueError("--splits_path is required")

    seed = _parse_int_arg(_get_arg_value(argv, "--seed", None), 42)
    total_round = _parse_int_arg(_get_arg_value(argv, "--total_round", None), 5)
    train_samples = _parse_int_arg(_get_arg_value(argv, "--train_samples_per_task", None), -1)
    eval_samples = _parse_int_arg(_get_arg_value(argv, "--eval_samples_per_task", None), -1)
    num_tasks_value = _get_arg_value(argv, "--num_tasks", None)
    if num_tasks_value is not None and int(num_tasks_value) != len(DOMAINNET_DOMAINS):
        raise ValueError(
            f"Domain-IL on DomainNet requires --num_tasks={len(DOMAINNET_DOMAINS)} (got {num_tasks_value})."
        )

    splits_path_abs = os.path.abspath(splits_path)
    make_splits_only = _has_flag(argv, "--make_splits_only")
    need_regen, regen_reason = _inspect_existing_splits(splits_path_abs, total_round=int(total_round))
    if need_regen and os.path.exists(splits_path_abs):
        _log(f"[splits] existing file incompatible ({regen_reason}); regenerating.")

    if make_splits_only or (not os.path.exists(splits_path_abs)) or need_regen:
        _log(f"[splits] creating domain splits at {splits_path_abs}")
        _ensure_domain_splits(
            dataset=str(dataset),
            data_root=str(data_root),
            train_dir=train_dir,
            val_dir=val_dir,
            splits_path=splits_path_abs,
            total_round=int(total_round),
            seed=int(seed),
            train_samples_per_task=int(train_samples),
            eval_samples_per_task=int(eval_samples),
            domain_orders_json=domain_orders_json,
        )
        if make_splits_only:
            _log("[done] make_splits_only")
            return

    fwd = list(argv)
    fwd = _remove_flag(fwd, "--make_splits_only")
    fwd = _remove_arg_with_value(fwd, "--domain_orders_json")
    fwd = _set_or_append_arg(fwd, "--dataset", "domainnet")
    fwd = _set_or_append_arg(fwd, "--num_tasks", str(len(DOMAINNET_DOMAINS)))
    fwd = _set_or_append_arg(fwd, "--total_round", str(int(total_round)))

    # Domain-IL default: 15 epochs/task (DUCT standard).
    # vit_cl uses -1 as "auto" which maps to 5 for domainnet; override to 15 here.
    epochs_val = _get_arg_value(fwd, "--epochs", None)
    if epochs_val is None or str(epochs_val).strip() == "-1":
        fwd = _set_or_append_arg(fwd, "--epochs", "15")

    started_at = time.time()
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + fwd
        vcl.main()
    finally:
        sys.argv = old_argv

    run_dir = _resolve_run_dir(save_path=str(save_path), run_name=run_name, started_at=started_at)
    if run_dir is None:
        _log("[summary] could not resolve run directory; skip domain-order summary.")
        return

    try:
        _enrich_csv_with_domain_info(run_dir=run_dir, splits_path=splits_path_abs)
        _log(f"[enrich] added domain columns to {os.path.join(run_dir, 'eval_metrics.csv')}")
    except Exception as e:
        _log(f"[warn] domain enrichment failed: {e}")

    logger = None
    try:
        logger = vcl.Logger(os.path.join(run_dir, "logs", "train.log"))
        _summarize_rounds(run_dir=run_dir, splits_path=splits_path_abs, logger=logger)
    finally:
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
