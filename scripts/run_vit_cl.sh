#!/usr/bin/env bash
set -euo pipefail

ts() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(ts)] $*"
}

export PYENV_VERSION="${PYENV_VERSION:-curb}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-1}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-google/vit-base-patch16-224-in21k}"
MODEL_DTYPE="${MODEL_DTYPE:-fp32}"
DEVICE="${DEVICE:-cuda}"

DATASET="${DATASET:-cifar100}"
DATA_ROOT="${DATA_ROOT:-./data}"
TRAIN_DIR="${TRAIN_DIR:-}"
VAL_DIR="${VAL_DIR:-}"
NUM_TASKS="${NUM_TASKS:-}"
TOTAL_ROUND="${TOTAL_ROUND:-3}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
METHODS_STR="${METHODS_STR:-lora mora curlora olora inflora bilora lorac lorac_ipc curb_covfast_lowdeim curb_weight_lowdeim curb_hybrid_lowdeim curb_hybrid_lowdeim_replaycalib}"
SEED="${SEED:-42}"

SAVE_ROOT="${SAVE_ROOT:-./cl_runs/vit_compare}"
EXP_TAG="${EXP_TAG:-vit_compare_$(date +%Y%m%d_%H%M%S)}"

TRAIN_SAMPLES_PER_TASK="${TRAIN_SAMPLES_PER_TASK:--1}"
EVAL_SAMPLES_PER_TASK="${EVAL_SAMPLES_PER_TASK:--1}"

IMAGE_SIZE="${IMAGE_SIZE:-224}"
EPOCHS="${EPOCHS:--1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
ADAM_EPS="${ADAM_EPS:-1e-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

TRAIN_LOSS_MASK="${TRAIN_LOSS_MASK:-current_task}"
EVAL_MASK="${EVAL_MASK:-seen_classes}"

CURB_RANK="${CURB_RANK:-256}"
# ViT-B/16 parameter-budget matching defaults with LoRA q/k/fc1 = 8/8/8 fixed:
# curated even-rank profile near target budget (663,552):
# q/k/fc1 = 118/102/176 -> 663,648 (+0.0145% vs target)
CURB_RANK_Q="${CURB_RANK_Q:-118}"
CURB_RANK_K="${CURB_RANK_K:-102}"
CURB_RANK_FC1="${CURB_RANK_FC1:-176}"
CURB_ALPHA="${CURB_ALPHA:-5.0}"
CURB_BASIS_MODE="${CURB_BASIS_MODE:-weight}"
CURB_DEIM_IMPORTANCE_ORDER="${CURB_DEIM_IMPORTANCE_ORDER:-low}"
CURB_CALIB_STEPS="${CURB_CALIB_STEPS:-256}"
CURB_BATCH_SIZE="${CURB_BATCH_SIZE:-1}"
CURB_CALIB_SOURCE="${CURB_CALIB_SOURCE:-train}"
REPLAY_BUFFER_PER_TASK="${REPLAY_BUFFER_PER_TASK:-0}"
CURB_UPDATE_WHITEN="${CURB_UPDATE_WHITEN:-none}"
CURB_WHITEN_RIDGE_RATIO="${CURB_WHITEN_RIDGE_RATIO:-1e-4}"
CURB_WHITEN_RIDGE_ABS="${CURB_WHITEN_RIDGE_ABS:-1e-12}"
CURB_BASIS_FREEZE="${CURB_BASIS_FREEZE:-0}"
CURB_ALPHA_SPECTRAL_NORM="${CURB_ALPHA_SPECTRAL_NORM:-1}"
CURB_U_WEIGHT_DECAY="${CURB_U_WEIGHT_DECAY:-0.05}"
CURB_ALPHA_SCHEDULE="${CURB_ALPHA_SCHEDULE:-constant}"
CURB_ALPHA_MIN_RATIO="${CURB_ALPHA_MIN_RATIO:-1.0}"
CURB_ALPHA_WARMUP_RATIO="${CURB_ALPHA_WARMUP_RATIO:-0.0}"
CURB_ALPHA_PER_TASK="${CURB_ALPHA_PER_TASK:-0}"
CALIB_TRAIN_DIR="${CALIB_TRAIN_DIR:-}"

# Keep LoRA-family baseline fixed at 8/8/8 (tc_cl spirit), and match CURb-family
# ranks via CURb defaults above.
LORA_RANK_Q="${LORA_RANK_Q:-8}"
LORA_RANK_K="${LORA_RANK_K:-8}"
LORA_RANK_FC1="${LORA_RANK_FC1:-8}"
LORA_ALPHA="${LORA_ALPHA:-1.0}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"

OLORA_LAMBDA_ORTH="${OLORA_LAMBDA_ORTH:-0.5}"
OLORA_LAMBDA_L2="${OLORA_LAMBDA_L2:-0.0}"

INFLORA_LAMB="${INFLORA_LAMB:-0.95}"
INFLORA_LAME="${INFLORA_LAME:-1.0}"
INFLORA_CALIB_SOURCE="${INFLORA_CALIB_SOURCE:-train}"
INFLORA_MATCH_TRAINABLE="${INFLORA_MATCH_TRAINABLE:-1}"

BILORA_K="${BILORA_K:-}"
BILORA_ALPHA="${BILORA_ALPHA:-}"
BILORA_SEED="${BILORA_SEED:-777}"
BILORA_CHUNK_SIZE="${BILORA_CHUNK_SIZE:-0}"
BILORA_FREQ_CHUNK_SIZE="${BILORA_FREQ_CHUNK_SIZE:-8192}"

LORAC_ORTH="${LORAC_ORTH:-1.0}"
LORAC_OMEGA_LR_SCALE="${LORAC_OMEGA_LR_SCALE:-1.0}"
LORAC_IPC_BETA1="${LORAC_IPC_BETA1:-0.85}"
LORAC_IPC_BETA2="${LORAC_IPC_BETA2:-0.85}"
LORAC_IPC_THRESHOLD="${LORAC_IPC_THRESHOLD:-0.05}"
LORAC_IPC_NEW_MASK="${LORAC_IPC_NEW_MASK:-0}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
read -r -a METHODS <<< "${METHODS_STR}"

if [[ "${SEED}" == *","* ]]; then
  echo "Use SEED=<int> (single base seed). Do not pass a list; rounds already advance the seed." >&2
  exit 1
fi
if [[ -z "${SEED}" ]]; then
  echo "SEED is empty." >&2
  exit 1
fi
if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "SEED must be an integer (got: ${SEED})." >&2
  exit 1
fi
if (( ${#METHODS[@]} < 1 )); then
  echo "METHODS_STR produced no methods." >&2
  exit 1
fi

if [[ "${DEVICE}" != "cpu" ]] && (( ${#GPUS[@]} < ${#METHODS[@]} )); then
  echo "Need at least ${#METHODS[@]} GPU ids in GPU_LIST (got: ${GPU_LIST})" >&2
  exit 1
fi

method_args() {
  case "$1" in
    lora)
      echo "--method lora"
      ;;
    mora)
      echo "--method mora"
      ;;
    curlora)
      echo "--method curlora"
      ;;
    olora)
      echo "--method olora"
      ;;
    inflora)
      echo "--method inflora"
      ;;
    bilora)
      echo "--method bilora"
      ;;
    lorac)
      echo "--method lorac"
      ;;
    lorac_ipc)
      echo "--method lorac_ipc"
      ;;
    curb_covfast_lowdeim)
      echo "--method curb --curb_basis_mode cov_fast --curb_deim_importance_order low"
      ;;
    curb_weight_lowdeim)
      echo "--method curb --curb_basis_mode weight --curb_deim_importance_order low"
      ;;
    curb_hybrid_lowdeim)
      echo "--method curb --curb_basis_mode hybrid --curb_deim_importance_order low"
      ;;
    curb_hybrid_lowdeim_replaycalib)
      echo "--method curb --curb_basis_mode hybrid --curb_deim_importance_order low --curb_calib_source replay_mix_imagenet1k"
      ;;
    *)
      echo "Unknown method: $1" >&2
      return 1
      ;;
  esac
}

RUN_ROOT_LOCAL="${SAVE_ROOT}/${EXP_TAG}"
LOG_ROOT="${RUN_ROOT_LOCAL}/launcher_logs"
mkdir -p "${RUN_ROOT_LOCAL}" "${LOG_ROOT}"

STATUS_CSV="${RUN_ROOT_LOCAL}/launcher_status.csv"
CMD_TXT="${RUN_ROOT_LOCAL}/launcher_commands.txt"
echo "timestamp,seed,method,gpu,exit_code,log_path,save_path,splits_path" > "${STATUS_CSV}"
: > "${CMD_TXT}"

log "Experiment root(local): ${RUN_ROOT_LOCAL}"
log "Dataset=${DATASET} rounds=${TOTAL_ROUND} seed=${SEED} device=${DEVICE}"
log "Round seed formula: SEED + 10000*round_idx"
log "Methods: ${METHODS[*]}"
if [[ "${DEVICE}" != "cpu" ]]; then
  log "GPUs: ${GPUS[*]}"
fi

overall_failed=0
seed="${SEED}"
seed_root="${RUN_ROOT_LOCAL}/seed_${seed}"
mkdir -p "${seed_root}"
splits_json="${seed_root}/splits.json"
splits_log="${LOG_ROOT}/seed_${seed}_splits.log"

extra_data_args=()
if [[ -n "${TRAIN_DIR}" ]]; then
  extra_data_args+=(--train_dir "${TRAIN_DIR}")
fi
if [[ -n "${VAL_DIR}" ]]; then
  extra_data_args+=(--val_dir "${VAL_DIR}")
fi

extra_calib_args=()
if [[ -n "${CALIB_TRAIN_DIR}" ]]; then
  extra_calib_args+=(--calib_train_dir "${CALIB_TRAIN_DIR}")
fi

extra_task_args=()
if [[ -n "${NUM_TASKS}" ]]; then
  extra_task_args+=(--num_tasks "${NUM_TASKS}")
fi

if [[ ! -f "${splits_json}" ]]; then
  splits_save_path="${seed_root}/splits_gen"
  mkdir -p "${splits_save_path}"
  splits_cmd=(
    "${PYTHON_BIN}" -u experiments/vit/vit_cl.py
    --make_splits_only
    --dataset "${DATASET}"
    --data_root "${DATA_ROOT}"
    --save_path "${splits_save_path}"
    --splits_path "${splits_json}"
    --seed "${seed}"
    --total_round "${TOTAL_ROUND}"
    --train_samples_per_task "${TRAIN_SAMPLES_PER_TASK}"
    --eval_samples_per_task "${EVAL_SAMPLES_PER_TASK}"
    --model_name "${MODEL_NAME}"
    --model_dtype "${MODEL_DTYPE}"
    --device cpu
    "${extra_data_args[@]}"
    "${extra_task_args[@]}"
  )
  printf '%q ' "${splits_cmd[@]}" >> "${CMD_TXT}"
  echo " > ${splits_log} 2>&1" >> "${CMD_TXT}"
  log "seed=${seed} building splits -> ${splits_json}"
  "${splits_cmd[@]}" > "${splits_log}" 2>&1
else
  log "seed=${seed} using existing splits -> ${splits_json}"
fi

pids=()
names=()
logs=()
save_paths=()
gpus=()

for i in "${!METHODS[@]}"; do
  name="${METHODS[$i]}"
  gpu="${GPUS[$i]:-cpu}"
  save_path="${seed_root}/${name}"
  seed_log="${LOG_ROOT}/seed_${seed}_${name}.log"
  mkdir -p "${save_path}"

  read -r -a METH_ARGS <<< "$(method_args "${name}")"

  method_replay_buffer_per_task=0
  if [[ "${name}" == *replaycalib* ]]; then
    method_replay_buffer_per_task="${REPLAY_BUFFER_PER_TASK}"
  fi

  cmd=(
    "${PYTHON_BIN}" -u experiments/vit/vit_cl.py
    --dataset "${DATASET}"
    --model_name "${MODEL_NAME}"
    --model_dtype "${MODEL_DTYPE}"
    --device "${DEVICE}"
    --data_root "${DATA_ROOT}"
    --save_path "${save_path}"
    --run_name "vit_${name}_seed${seed}"
    --seed "${seed}"
    --splits_path "${splits_json}"
    --total_round "${TOTAL_ROUND}"
    --train_samples_per_task "${TRAIN_SAMPLES_PER_TASK}"
    --eval_samples_per_task "${EVAL_SAMPLES_PER_TASK}"
    --image_size "${IMAGE_SIZE}"
    --train_batch_size "${TRAIN_BATCH_SIZE}"
    --eval_batch_size "${EVAL_BATCH_SIZE}"
    --epochs "${EPOCHS}"
    --learning_rate "${LEARNING_RATE}"
    --weight_decay "${WEIGHT_DECAY}"
    --grad_accum_steps "${GRAD_ACCUM_STEPS}"
    --max_grad_norm "${MAX_GRAD_NORM}"
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
    --warmup_ratio "${WARMUP_RATIO}"
    --adam_beta1 "${ADAM_BETA1}"
    --adam_beta2 "${ADAM_BETA2}"
    --adam_eps "${ADAM_EPS}"
    --max_train_steps "${MAX_TRAIN_STEPS}"
    --num_workers "${NUM_WORKERS}"
    --train_loss_mask "${TRAIN_LOSS_MASK}"
    --eval_mask "${EVAL_MASK}"
    --curb_rank "${CURB_RANK}"
    --curb_rank_q "${CURB_RANK_Q}"
    --curb_rank_k "${CURB_RANK_K}"
    --curb_rank_fc1 "${CURB_RANK_FC1}"
    --curb_alpha "${CURB_ALPHA}"
    --curb_basis_mode "${CURB_BASIS_MODE}"
    --curb_deim_importance_order "${CURB_DEIM_IMPORTANCE_ORDER}"
    --curb_calib_steps "${CURB_CALIB_STEPS}"
    --curb_batch_size "${CURB_BATCH_SIZE}"
    --curb_calib_source "${CURB_CALIB_SOURCE}"
    --replay_buffer_per_task "${method_replay_buffer_per_task}"
    --curb_update_whiten "${CURB_UPDATE_WHITEN}"
    --curb_whiten_ridge_ratio "${CURB_WHITEN_RIDGE_RATIO}"
    --curb_whiten_ridge_abs "${CURB_WHITEN_RIDGE_ABS}"
    --curb_u_weight_decay "${CURB_U_WEIGHT_DECAY}"
    --curb_alpha_schedule "${CURB_ALPHA_SCHEDULE}"
    --curb_alpha_min_ratio "${CURB_ALPHA_MIN_RATIO}"
    --curb_alpha_warmup_ratio "${CURB_ALPHA_WARMUP_RATIO}"
    --lora_rank_q "${LORA_RANK_Q}"
    --lora_rank_k "${LORA_RANK_K}"
    --lora_rank_fc1 "${LORA_RANK_FC1}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"
    --olora_lambda_orth "${OLORA_LAMBDA_ORTH}"
    --olora_lambda_l2 "${OLORA_LAMBDA_L2}"
    --inflora_lamb "${INFLORA_LAMB}"
    --inflora_lame "${INFLORA_LAME}"
    --inflora_calib_source "${INFLORA_CALIB_SOURCE}"
    --lorac_ortho "${LORAC_ORTH}"
    --lorac_omega_lr_scale "${LORAC_OMEGA_LR_SCALE}"
    --lorac_ipc_beta1 "${LORAC_IPC_BETA1}"
    --lorac_ipc_beta2 "${LORAC_IPC_BETA2}"
    --lorac_ipc_threshold "${LORAC_IPC_THRESHOLD}"
    --bilora_seed "${BILORA_SEED}"
    --bilora_chunk_size "${BILORA_CHUNK_SIZE}"
    --bilora_freq_chunk_size "${BILORA_FREQ_CHUNK_SIZE}"
    "${extra_data_args[@]}"
    "${extra_calib_args[@]}"
    "${extra_task_args[@]}"
    "${METH_ARGS[@]}"
  )

  if [[ -n "${BILORA_K}" ]]; then
    cmd+=(--bilora_k "${BILORA_K}")
  fi
  if [[ -n "${BILORA_ALPHA}" && "${BILORA_ALPHA}" != "auto" ]]; then
    cmd+=(--bilora_alpha "${BILORA_ALPHA}")
  fi
  if [[ "${INFLORA_MATCH_TRAINABLE}" == "1" || "${INFLORA_MATCH_TRAINABLE}" == "true" ]]; then
    cmd+=(--inflora_match_trainable)
  else
    cmd+=(--no_inflora_match_trainable)
  fi
  if [[ "${name}" == "lorac_ipc" ]]; then
    if [[ "${LORAC_IPC_NEW_MASK}" == "1" || "${LORAC_IPC_NEW_MASK}" == "true" ]]; then
      cmd+=(--lorac_ipc_new_mask)
    fi
  fi
  if [[ "${CURB_BASIS_FREEZE}" == "1" || "${CURB_BASIS_FREEZE}" == "true" ]]; then
    cmd+=(--curb_basis_freeze)
  fi
  if [[ "${CURB_ALPHA_SPECTRAL_NORM}" == "1" || "${CURB_ALPHA_SPECTRAL_NORM}" == "true" ]]; then
    cmd+=(--curb_alpha_spectral_norm)
  fi
  if [[ "${CURB_ALPHA_PER_TASK}" == "1" || "${CURB_ALPHA_PER_TASK}" == "true" ]]; then
    cmd+=(--curb_alpha_per_task)
  fi
  if [[ "${DIAG_CURB:-}" == "1" || "${DIAG_CURB:-}" == "true" ]]; then
    cmd+=(--diag_curb)
  fi

  printf 'CUDA_VISIBLE_DEVICES=%s ' "${gpu}" >> "${CMD_TXT}"
  printf '%q ' "${cmd[@]}" >> "${CMD_TXT}"
  echo " > ${seed_log} 2>&1" >> "${CMD_TXT}"

  log "launch method=${name} device=${DEVICE} gpu=${gpu} log=${seed_log}"
  if [[ "${DEVICE}" == "cpu" ]]; then
    "${cmd[@]}" > "${seed_log}" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "${seed_log}" 2>&1 &
  fi
  pid=$!

  pids+=("${pid}")
  names+=("${name}")
  logs+=("${seed_log}")
  save_paths+=("${save_path}")
  gpus+=("${gpu}")
done

for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  name="${names[$i]}"
  gpu="${gpus[$i]}"
  seed_log="${logs[$i]}"
  save_path="${save_paths[$i]}"
  code=0

  if wait "${pid}"; then
    code=0
    log "method=${name} gpu=${gpu} finished (exit=0)"
  else
    code=$?
    overall_failed=1
    log "method=${name} gpu=${gpu} failed (exit=${code})"
    log "check log: ${seed_log}"
  fi

  echo "$(ts),${seed},${name},${gpu},${code},${seed_log},${save_path},${splits_json}" >> "${STATUS_CSV}"
done

if (( overall_failed )); then
  log "ViT launcher finished with failures. status=${STATUS_CSV}"
  exit 1
fi
log "ViT launcher completed successfully. status=${STATUS_CSV}"
