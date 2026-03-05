#!/usr/bin/env python3
"""Plot UUID memorization experiment: char_acc per epoch for all methods."""

import csv
import os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Method sources ──────────────────────────────────────────────────────────
UUID_ROOT = Path("cl_runs/uuid_compare")

METHOD_PATHS = {
    # baselines  (update these paths to match your experiment output directories)
    "lora":       "uuid_compare/lora/uuid_lora_seed42",
    "olora":      "uuid_compare/olora/uuid_olora_seed42",
    "mora":       "uuid_compare/mora/uuid_mora_seed42",
    "curlora":    "uuid_compare/curlora/uuid_curlora_seed42",
    "inflora":    "uuid_compare_inflora/inflora/uuid_inflora_seed42",
    "bilora":     "uuid_compare_bilora/bilora/uuid_bilora_seed42",
    "lorac":      "uuid_compare_lorac/lorac/uuid_lorac_seed42",
    "lorac_ipc":  "uuid_compare_lorac/lorac_ipc/uuid_lorac_ipc_seed42",
    # curb variants
    "curb_covfast":                "uuid_compare/curb_covfast/uuid_curb_covfast_seed42",
    "curb_covfast_lowdeim":        "uuid_compare/curb_covfast_lowdeim/uuid_curb_covfast_lowdeim_seed42",
    "curb_weight":                 "uuid_compare/curb_weight/uuid_curb_weight_seed42",
    "curb_weight_lowdeim":         "uuid_compare/curb_weight_lowdeim/uuid_curb_weight_lowdeim_seed42",
    "curb_hybrid_lowdeim":         "uuid_compare_hybrid/curb_hybrid_lowdeim/uuid_curb_hybrid_lowdeim_seed42",
    "curb_hybrid_lowdeim_uuidcalib": "uuid_compare_hybrid/curb_hybrid_lowdeim_uuidcalib/uuid_curb_hybrid_lowdeim_uuidcalib_seed42",
    "curb_covfast_lowdeim_replaycalib": "uuid_replaycalib/curb_covfast_lowdeim_replaycalib/uuid_curb_covfast_lowdeim_replaycalib_seed42",
}

# ── Display config ──────────────────────────────────────────────────────────
# Group 1: Baselines
BASELINES = ["lora", "olora", "mora", "curlora", "inflora", "bilora", "lorac", "lorac_ipc"]
# Group 2: CURb representative
CURB_MAIN = ["curb_hybrid_lowdeim"]
# Group 3: CURb ablations (kept for ablation plot only)
CURB_ABLATION = [
    "curb_covfast", "curb_weight",
    "curb_covfast_lowdeim", "curb_weight_lowdeim",
    "curb_hybrid_lowdeim_uuidcalib",
    "curb_covfast_lowdeim_replaycalib",
]

DISPLAY_NAMES = {
    "lora": "LoRA", "olora": "OLoRA", "mora": "MoRA", "curlora": "CURLoRA",
    "inflora": "InfLoRA", "bilora": "BiLoRA", "lorac": "LoRAC", "lorac_ipc": "LoRAC-IPC",
    "curb_covfast": "CURb-Cov", "curb_covfast_lowdeim": "CURb-Cov-LD",
    "curb_weight": "CURb-W", "curb_weight_lowdeim": "CURb-W-LD",
    "curb_hybrid_lowdeim": "CURb-Hyb-LD",
    "curb_hybrid_lowdeim_uuidcalib": "CURb-Hyb-LD (uuid calib)",
    "curb_covfast_lowdeim_replaycalib": "CURb-Cov-LD (replay calib)",
}

# ── Read data ───────────────────────────────────────────────────────────────
def read_epoch_metrics(method_key):
    csv_path = UUID_ROOT / METHOD_PATHS[method_key] / "epoch_metrics.csv"
    epochs, char_accs, exact_accs = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row["epoch"])
            if ep > 50:
                break
            epochs.append(ep)
            char_accs.append(float(row["uuid_char_acc"]))
            exact_accs.append(float(row["uuid_exact_acc"]))
    return np.array(epochs), np.array(char_accs), np.array(exact_accs)

all_data = {}
for key in METHOD_PATHS:
    all_data[key] = read_epoch_metrics(key)

# ── Color palette ───────────────────────────────────────────────────────────
COLORS_BASELINE = {
    "lora": "#1f77b4", "olora": "#aec7e8", "mora": "#ff7f0e", "curlora": "#ffbb78",
    "inflora": "#2ca02c", "bilora": "#98df8a", "lorac": "#9467bd", "lorac_ipc": "#c5b0d5",
}
COLORS_CURB = {
    "curb_covfast": "#d62728", "curb_covfast_lowdeim": "#e41a1c",
    "curb_weight": "#ff9896", "curb_weight_lowdeim": "#e377c2",
    "curb_hybrid_lowdeim": "#8c564b",
    "curb_hybrid_lowdeim_uuidcalib": "#c49c94",
    "curb_covfast_lowdeim_replaycalib": "#f7b6d2",
}

def get_color(key):
    return COLORS_BASELINE.get(key, COLORS_CURB.get(key, "#333333"))

# ── Plot 1: All methods, char_acc ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))

for key in BASELINES + CURB_MAIN + CURB_ABLATION:
    epochs, char_accs, _ = all_data[key]
    lw = 2.5 if key in CURB_MAIN else 1.5
    ls = "-" if key not in CURB_ABLATION else "--"
    ax.plot(epochs, char_accs, label=DISPLAY_NAMES[key], color=get_color(key),
            linewidth=lw, linestyle=ls, alpha=0.9)

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("UUID Char Accuracy (%)", fontsize=13)
ax.set_title("UUID Memorization — Character Accuracy (all methods)", fontsize=15)
ax.set_xlim(1, 50)
ax.legend(fontsize=8, ncol=3, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("scripts/uuid_char_acc_all.png", dpi=200)
print(f"Saved: scripts/uuid_char_acc_all.png")
plt.close()

# ── Plot 2: All methods, exact_acc ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))

for key in BASELINES + CURB_MAIN + CURB_ABLATION:
    epochs, _, exact_accs = all_data[key]
    lw = 2.5 if key in CURB_MAIN else 1.5
    ls = "-" if key not in CURB_ABLATION else "--"
    ax.plot(epochs, exact_accs, label=DISPLAY_NAMES[key], color=get_color(key),
            linewidth=lw, linestyle=ls, alpha=0.9)

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("UUID Exact Accuracy (%)", fontsize=13)
ax.set_title("UUID Memorization — Exact Accuracy (all methods)", fontsize=15)
ax.set_xlim(1, 50)
ax.legend(fontsize=8, ncol=3, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("scripts/uuid_exact_acc_all.png", dpi=200)
print(f"Saved: scripts/uuid_exact_acc_all.png")
plt.close()

# ── Plot 3: Baselines vs CURb main (char_acc, cleaner view) ────────────────
fig, ax = plt.subplots(figsize=(12, 7))

focus = BASELINES + CURB_MAIN
for key in focus:
    epochs, char_accs, _ = all_data[key]
    lw = 2.5 if key in CURB_MAIN else 1.5
    ax.plot(epochs, char_accs, label=DISPLAY_NAMES[key], color=get_color(key),
            linewidth=lw, alpha=0.9)

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("UUID Char Accuracy (%)", fontsize=13)
ax.set_title("UUID Memorization — Baselines vs CURb (char_acc)", fontsize=15)
ax.set_xlim(1, 50)
ax.legend(fontsize=9, ncol=2, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("scripts/uuid_char_acc_main.png", dpi=200)
print(f"Saved: scripts/uuid_char_acc_main.png")
plt.close()

# ── Plot 3b: Baselines vs CURb main (exact_acc, cleaner view) ───────────────
fig, ax = plt.subplots(figsize=(12, 7))

focus = BASELINES + CURB_MAIN
for key in focus:
    epochs, _, exact_accs = all_data[key]
    lw = 2.5 if key in CURB_MAIN else 1.5
    ax.plot(epochs, exact_accs, label=DISPLAY_NAMES[key], color=get_color(key),
            linewidth=lw, alpha=0.9)

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("UUID Exact Accuracy (%)", fontsize=13)
ax.set_title("UUID Memorization — Baselines vs CURb (exact_acc)", fontsize=15)
ax.set_xlim(1, 50)
ax.legend(fontsize=9, ncol=2, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("scripts/uuid_exact_acc_main.png", dpi=200)
print(f"Saved: scripts/uuid_exact_acc_main.png")
plt.close()

# ── Plot 4: CURb ablations only (char_acc) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))

curb_all = CURB_MAIN + CURB_ABLATION
for key in curb_all:
    epochs, char_accs, _ = all_data[key]
    lw = 2.5 if key in CURB_MAIN else 1.5
    ls = "-" if key in CURB_MAIN else "--"
    ax.plot(epochs, char_accs, label=DISPLAY_NAMES[key], color=get_color(key),
            linewidth=lw, linestyle=ls, alpha=0.9)

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("UUID Char Accuracy (%)", fontsize=13)
ax.set_title("UUID Memorization — CURb Variants Ablation (char_acc)", fontsize=15)
ax.set_xlim(1, 50)
ax.legend(fontsize=9, ncol=2, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("scripts/uuid_char_acc_curb_ablation.png", dpi=200)
print(f"Saved: scripts/uuid_char_acc_curb_ablation.png")
plt.close()

# ── Plot 5: CURb lowDEIM ablation (exact_acc) ─────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))

CURB_LOWDEIM = [
    "curb_hybrid_lowdeim",
    "curb_covfast_lowdeim",
    "curb_weight_lowdeim",
    "curb_hybrid_lowdeim_uuidcalib",
    "curb_covfast_lowdeim_replaycalib",
]
COLORS_LOWDEIM = {
    "curb_hybrid_lowdeim":              "#d62728",  # red
    "curb_covfast_lowdeim":             "#1f77b4",  # blue
    "curb_weight_lowdeim":              "#2ca02c",  # green
    "curb_hybrid_lowdeim_uuidcalib":    "#ff7f0e",  # orange
    "curb_covfast_lowdeim_replaycalib": "#9467bd",  # purple
}
MARKERS_LOWDEIM = {
    "curb_hybrid_lowdeim":              "o",
    "curb_covfast_lowdeim":             "s",
    "curb_weight_lowdeim":              "D",
    "curb_hybrid_lowdeim_uuidcalib":    "^",
    "curb_covfast_lowdeim_replaycalib": "v",
}

for key in CURB_LOWDEIM:
    epochs, _, exact_accs = all_data[key]
    ax.plot(epochs, exact_accs, label=DISPLAY_NAMES[key],
            color=COLORS_LOWDEIM[key], linewidth=2.0, alpha=0.9,
            marker=MARKERS_LOWDEIM[key], markersize=4, markevery=5)

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("UUID Exact Accuracy (%)", fontsize=13)
ax.set_title("UUID Memorization — CURb lowDEIM Variants (exact_acc)", fontsize=15)
ax.set_xlim(1, 50)
ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("scripts/uuid_exact_acc_curb_lowdeim_ablation.png", dpi=200)
print(f"Saved: scripts/uuid_exact_acc_curb_lowdeim_ablation.png")
plt.close()

# ── Summary table ───────────────────────────────────────────────────────────
print("\n=== Summary at Epoch 50 ===")
print(f"{'Method':<40s} {'char_acc@50':>12s} {'exact_acc@50':>13s} {'best_char':>10s} {'best_exact':>11s}")
print("-" * 90)
for key in BASELINES + CURB_MAIN + CURB_ABLATION:
    epochs, char_accs, exact_accs = all_data[key]
    idx50 = 49  # epoch 50
    best_char = char_accs.max()
    best_exact = exact_accs.max()
    print(f"{DISPLAY_NAMES[key]:<40s} {char_accs[idx50]:>12.2f} {exact_accs[idx50]:>13.2f} {best_char:>10.2f} {best_exact:>11.2f}")
