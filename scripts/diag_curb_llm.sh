#!/usr/bin/env bash
# Small-scale CURb diagnostic run for LLM Text Classification CL.
# Runs curb_weight_lowdeim with --diag_curb on 4 tasks (DBpedia/Amazon/Yahoo/AGNews).
# 1 round, same hyperparams as tc_compare — just adds diagnostic logging.
set -euo pipefail

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

export PYENV_VERSION="${PYENV_VERSION:-curb}"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-7}"

SAVE_ROOT="./cl_runs/llm_diag"
EXP_TAG="diag_curb_llm_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="${SAVE_ROOT}/${EXP_TAG}/seed_42"
mkdir -p "${SAVE_DIR}"

SPLITS_JSON="${SAVE_DIR}/splits.json"

log "=== CURb LLM Diagnostic ==="
log "GPU=${GPU}  SAVE=${SAVE_DIR}"

# Run: curb_weight_lowdeim with diagnostic (4 tasks, 1 round, 1 epoch)
log "--- curb_weight_lowdeim (with --diag_curb) ---"
CUDA_VISIBLE_DEVICES="${GPU}" ${PYTHON_BIN} -u experiments/llm/tc_cl.py \
  --model_name meta-llama/Llama-3.1-8B \
  --model_dtype fp32 \
  --tasks dbpedia_14 amazon_polarity yahoo_answers_topics ag_news \
  --total_round 1 \
  --seed 42 \
  --save_path "${SAVE_DIR}/curb_weight_lowdeim" \
  --run_name "diag_curb_weight_llm" \
  --splits_path "${SPLITS_JSON}" \
  --train_samples_per_task 1000 \
  --val_per_class 500 \
  --epochs 1 \
  --learning_rate 1e-3 \
  --weight_decay 0.0 \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --grad_accum_steps 8 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type constant \
  --max_length 512 \
  --curb_rank 256 \
  --curb_rank_q 256 \
  --curb_rank_k 202 \
  --curb_rank_gate 384 \
  --curb_alpha 1.0 \
  --curb_basis_mode weight \
  --curb_deim_importance_order low \
  --curb_calib_steps 256 \
  --curb_batch_size 1 \
  --curb_max_length 512 \
  --curb_ffn_module_names gate_proj \
  --curb_attn_module_names q_proj k_proj \
  --method curb \
  --diag_curb \
  2>&1 | tee "${SAVE_DIR}/curb_weight_lowdeim.log"

log "=== Done. Results in ${SAVE_DIR} ==="
