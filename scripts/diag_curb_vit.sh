#!/usr/bin/env bash
# Small-scale CURb diagnostic run for ViT Class-IL.
# Runs curb_weight_lowdeim with --diag_curb on CIFAR-100.
# 1 round, 5 tasks (subset), 5 epochs — quick diagnostic only.
set -euo pipefail

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

export PYENV_VERSION="${PYENV_VERSION:-curb}"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-6}"

SAVE_ROOT="./cl_runs/vit_diag"
EXP_TAG="diag_curb_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="${SAVE_ROOT}/${EXP_TAG}/seed_42"
mkdir -p "${SAVE_DIR}"

# Reuse existing splits (10-task) — we just run fewer tasks via --num_tasks 5
SPLITS_JSON="${SAVE_DIR}/splits.json"

log "=== CURb ViT Diagnostic ==="
log "GPU=${GPU}  SAVE=${SAVE_DIR}"

# Run 1: curb_weight_lowdeim with diagnostic
log "--- [1/2] curb_weight_lowdeim (with --diag_curb) ---"
CUDA_VISIBLE_DEVICES="${GPU}" ${PYTHON_BIN} -u experiments/vit/vit_cl.py \
  --dataset cifar100 \
  --model_name google/vit-base-patch16-224-in21k \
  --model_dtype fp32 \
  --device cuda \
  --data_root ./data \
  --save_path "${SAVE_DIR}/curb_weight_lowdeim" \
  --run_name "diag_curb_weight" \
  --seed 42 \
  --splits_path "${SPLITS_JSON}" \
  --total_round 1 \
  --num_tasks 5 \
  --epochs 5 \
  --image_size 224 \
  --train_batch_size 128 \
  --eval_batch_size 128 \
  --learning_rate 1e-3 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type cosine \
  --num_workers 4 \
  --train_loss_mask current_task \
  --eval_mask seen_classes \
  --curb_rank 256 \
  --curb_rank_q 118 \
  --curb_rank_k 102 \
  --curb_rank_fc1 176 \
  --curb_alpha 1.0 \
  --curb_calib_steps 256 \
  --curb_batch_size 1 \
  --curb_calib_source train \
  --curb_update_whiten none \
  --method curb \
  --curb_basis_mode weight \
  --curb_deim_importance_order low \
  --diag_curb \
  2>&1 | tee "${SAVE_DIR}/curb_weight_lowdeim.log"

log "--- [2/2] lora (baseline, same setting) ---"
CUDA_VISIBLE_DEVICES="${GPU}" ${PYTHON_BIN} -u experiments/vit/vit_cl.py \
  --dataset cifar100 \
  --model_name google/vit-base-patch16-224-in21k \
  --model_dtype fp32 \
  --device cuda \
  --data_root ./data \
  --save_path "${SAVE_DIR}/lora" \
  --run_name "diag_lora" \
  --seed 42 \
  --splits_path "${SPLITS_JSON}" \
  --total_round 1 \
  --num_tasks 5 \
  --epochs 5 \
  --image_size 224 \
  --train_batch_size 128 \
  --eval_batch_size 128 \
  --learning_rate 1e-3 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type cosine \
  --num_workers 4 \
  --train_loss_mask current_task \
  --eval_mask seen_classes \
  --lora_rank_q 8 \
  --lora_rank_k 8 \
  --lora_rank_fc1 8 \
  --lora_alpha 1.0 \
  --lora_dropout 0.0 \
  --method lora \
  2>&1 | tee "${SAVE_DIR}/lora.log"

log "=== Done. Results in ${SAVE_DIR} ==="
log "Diagnostic CSV: ${SAVE_DIR}/curb_weight_lowdeim/diag_curb_weight/diag_curb_round1.csv"
