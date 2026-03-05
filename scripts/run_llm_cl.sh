#!/usr/bin/env bash
set -euo pipefail

ts() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(ts)] $*"
}

# Runtime defaults (override with env vars)
export PYENV_VERSION="${PYENV_VERSION:-curb}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-1}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B}"
MODEL_DTYPE="${MODEL_DTYPE:-fp32}"
CURB_RANK="${CURB_RANK:-256}"
# Llama-3.1-8B defaults for fair comparison with LoRA/MoRA 8/8/8.
CURB_RANK_Q="${CURB_RANK_Q:-256}"
CURB_RANK_K="${CURB_RANK_K:-202}"
CURB_RANK_GATE="${CURB_RANK_GATE:-384}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
# Base seed. Note: experiments/llm/cl.py advances the seed per round internally.
SEED="${SEED:-42}"

SAVE_ROOT="${SAVE_ROOT:-./cl_runs/curb_compare}"
MODEL_SAVE_ROOT="${MODEL_SAVE_ROOT:-./model_saves/curb_compare}"
EXP_TAG="${EXP_TAG:-curb_compare_$(date +%Y%m%d_%H%M%S)}"

TASKS_STR="${TASKS_STR:-boolq winogrande arc_easy arc_challenge piqa openbookqa social_iqa logiqa}"
TOTAL_ROUND="${TOTAL_ROUND:-10}"
NUM_TEST_STEPS="${NUM_TEST_STEPS:-128}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-100000}"
METHODS_STR="${METHODS_STR:-curb_covfast curb_covfast_lowdeim curb_weight curb_weight_lowdeim curlora lora mora olora}"

TRAIN_SAMPLES_PER_TASK="${TRAIN_SAMPLES_PER_TASK:-4096}"
REPLAY_BUFFER_PER_TASK="${REPLAY_BUFFER_PER_TASK:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LEARNING_RATE_FIRST_TASK="${LEARNING_RATE_FIRST_TASK:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
ADAM_EPS="${ADAM_EPS:-1e-8}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
TRAIN_TF_LOG_EVERY="${TRAIN_TF_LOG_EVERY:-10}"

# Fixed alpha across methods for fair comparison.
CURB_ALPHA="${CURB_ALPHA:-1.0}"
CURB_CALIB_STEPS="${CURB_CALIB_STEPS:-256}"
CURB_BATCH_SIZE="${CURB_BATCH_SIZE:-1}"
CURB_MAX_LENGTH="${CURB_MAX_LENGTH:-4096}"
CURB_CALIB_CATEGORY="${CURB_CALIB_CATEGORY:-en}"
CURB_FFN_MODULE_NAMES="${CURB_FFN_MODULE_NAMES:-gate_proj}"
CURB_ATTN_MODULE_NAMES="${CURB_ATTN_MODULE_NAMES:-q_proj k_proj}"
LORA_ALPHA="${LORA_ALPHA:-1.0}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
# BiLoRA defaults for fair comparison: keep alpha aligned to CURB_ALPHA unless overridden.
BILORA_K="${BILORA_K:-}"  # Optional; when unset, methods/baselines/bilora.py uses k=r_eff^2.
BILORA_ALPHA="${BILORA_ALPHA:-${CURB_ALPHA}}"
BILORA_SEED="${BILORA_SEED:-777}"
BILORA_CHUNK_SIZE="${BILORA_CHUNK_SIZE:-256}"
BILORA_FREQ_CHUNK_SIZE="${BILORA_FREQ_CHUNK_SIZE:-8192}"
OLORA_LAMBDA_ORTH="${OLORA_LAMBDA_ORTH:-0.5}"
OLORA_LAMBDA_L2="${OLORA_LAMBDA_L2:-0.0}"
INFLORA_LAMB="${INFLORA_LAMB:-0.95}"
INFLORA_LAME="${INFLORA_LAME:-1.0}"
INFLORA_CALIB_SOURCE="${INFLORA_CALIB_SOURCE:-}"
LORAC_ORTH="${LORAC_ORTH:-1.0}"
LORAC_OMEGA_LR_SCALE="${LORAC_OMEGA_LR_SCALE:-1.0}"
LORAC_IPC_BETA1="${LORAC_IPC_BETA1:-0.85}"
LORAC_IPC_BETA2="${LORAC_IPC_BETA2:-0.85}"
LORAC_IPC_THRESHOLD="${LORAC_IPC_THRESHOLD:-0.1}"
LORAC_IPC_NEW_MASK="${LORAC_IPC_NEW_MASK:-0}"
# LoRA/MoRA fixed rank tuple (Q/K/Gate). Default baseline: 8/8/8.
# Alternative candidates discussed previously: 8/4/6, 4/3/8.
LORA_RANK_Q="${LORA_RANK_Q:-8}"
LORA_RANK_K="${LORA_RANK_K:-8}"
LORA_RANK_GATE="${LORA_RANK_GATE:-8}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"

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

read -r -a TASKS <<< "${TASKS_STR}"
if (( ${#TASKS[@]} < 1 )); then
  echo "TASKS_STR produced no tasks." >&2
  exit 1
fi

read -r -a METHODS <<< "${METHODS_STR}"
if (( ${#METHODS[@]} < 1 )); then
  echo "METHODS_STR produced no methods." >&2
  exit 1
fi

read -r -a CURB_FFN_MODULES <<< "${CURB_FFN_MODULE_NAMES}"
if (( ${#CURB_FFN_MODULES[@]} < 1 )); then
  echo "CURB_FFN_MODULE_NAMES produced no modules." >&2
  exit 1
fi

read -r -a CURB_ATTN_MODULES <<< "${CURB_ATTN_MODULE_NAMES}"
if (( ${#CURB_ATTN_MODULES[@]} < 1 )); then
  echo "CURB_ATTN_MODULE_NAMES produced no modules." >&2
  exit 1
fi

if (( ${#GPUS[@]} < ${#METHODS[@]} )); then
  echo "Need at least ${#METHODS[@]} GPU ids in GPU_LIST (got: ${GPU_LIST})" >&2
  exit 1
fi

method_args() {
  case "$1" in
    curb_covfast)
      echo "--method curb --curb_basis_mode cov_fast"
      ;;
    curb_covfast_lowdeim)
      echo "--method curb --curb_basis_mode cov_fast --curb_deim_importance_order low"
      ;;
    curb_covfast_lowdeim_replaycalib)
      echo "--method curb --curb_basis_mode cov_fast --curb_deim_importance_order low --curb_calib_source replay_mix_c4"
      ;;
    curb_hybrid)
      echo "--method curb --curb_basis_mode hybrid"
      ;;
    curb_hybrid_lowdeim)
      echo "--method curb --curb_basis_mode hybrid --curb_deim_importance_order low"
      ;;
    curb_hybrid_lowdeim_replaycalib)
      echo "--method curb --curb_basis_mode hybrid --curb_deim_importance_order low --curb_calib_source replay_mix_c4"
      ;;
    curb_hybrid_lowdeim_whiten)
      echo "--method curb --curb_basis_mode hybrid --curb_deim_importance_order low --curb_update_whiten diag"
      ;;
    curb_covfast_lowdeim_whiten)
      echo "--method curb --curb_basis_mode cov_fast --curb_deim_importance_order low --curb_update_whiten diag"
      ;;
    curb_weight)
      echo "--method curb --curb_basis_mode weight"
      ;;
    curb_weight_lowdeim)
      echo "--method curb --curb_basis_mode weight --curb_deim_importance_order low"
      ;;
    curb_weight_lowdeim_whiten)
      echo "--method curb --curb_basis_mode weight --curb_deim_importance_order low --curb_update_whiten diag"
      ;;
    curlora)
      echo "--method curlora"
      ;;
    bilora)
      echo "--method bilora"
      ;;
    lora)
      echo "--method lora"
      ;;
    mora)
      echo "--method mora"
      ;;
    lorac)
      echo "--method lorac"
      ;;
    lorac_ipc)
      echo "--method lorac_ipc"
      ;;
    olora)
      echo "--method olora"
      ;;
    inflora)
      echo "--method inflora"
      ;;
    *)
      echo "Unknown method: $1" >&2
      return 1
      ;;
  esac
}

RUN_ROOT_LOCAL="${SAVE_ROOT}/${EXP_TAG}"
RUN_ROOT_MODEL="${MODEL_SAVE_ROOT}/${EXP_TAG}"
LOG_ROOT="${RUN_ROOT_LOCAL}/launcher_logs"
mkdir -p "${RUN_ROOT_LOCAL}" "${RUN_ROOT_MODEL}" "${LOG_ROOT}"

STATUS_CSV="${RUN_ROOT_LOCAL}/launcher_status.csv"
CMD_TXT="${RUN_ROOT_LOCAL}/launcher_commands.txt"
echo "timestamp,seed,method,gpu,exit_code,log_path,save_path,model_save_path" > "${STATUS_CSV}"
: > "${CMD_TXT}"

log "Experiment root(local): ${RUN_ROOT_LOCAL}"
log "Experiment root(model): ${RUN_ROOT_MODEL}"
log "Methods: ${METHODS[*]}"
log "Base seed: ${SEED} (round seeds: ${SEED},$((SEED+1))...)"
log "GPUs: ${GPUS[*]}"

overall_failed=0

seed="${SEED}"
log "=== Seed ${seed} launch start ==="

pids=()
names=()
logs=()
save_paths=()
model_save_paths=()
gpus=()

for i in "${!METHODS[@]}"; do
  name="${METHODS[$i]}"
  gpu="${GPUS[$i]}"
  seed_local="${RUN_ROOT_LOCAL}/seed_${seed}/${name}"
  seed_model="${RUN_ROOT_MODEL}/seed_${seed}/${name}"
  seed_log="${LOG_ROOT}/seed_${seed}_${name}.log"
  mkdir -p "${seed_local}" "${seed_model}"

  read -r -a METH_ARGS <<< "$(method_args "${name}")"
  # Replay buffer is used only to define the calibration distribution for
  # CURb cov_fast (activation-stat) calibration. Keep it off for other
  # methods by default to avoid unnecessary overhead.
  method_replay_buffer_per_task=0
  if [[ "${name}" == *replaycalib* ]]; then
    method_replay_buffer_per_task="${REPLAY_BUFFER_PER_TASK}"
  fi
  method_train_batch_size="${TRAIN_BATCH_SIZE}"
  method_num_train_epochs="${NUM_TRAIN_EPOCHS}"
  method_learning_rate="${LEARNING_RATE}"
  method_learning_rate_first_task="${LEARNING_RATE_FIRST_TASK}"
  method_grad_accum_steps="${GRAD_ACCUM_STEPS}"
  method_lr_scheduler_type="${LR_SCHEDULER_TYPE}"
  method_warmup_ratio="${WARMUP_RATIO}"
  method_curb_alpha=""
  method_lora_alpha=""
  method_bilora_alpha=""
  method_lora_dropout="${LORA_DROPOUT}"
  method_olora_lambda_orth="${OLORA_LAMBDA_ORTH}"
  method_olora_lambda_l2="${OLORA_LAMBDA_L2}"
  if [[ "${name}" == curb_* || "${name}" == "curlora" ]]; then
    method_curb_alpha="${CURB_ALPHA}"
  fi
  if [[ "${name}" == "bilora" ]]; then
    method_bilora_alpha="${BILORA_ALPHA}"
  fi
  if [[ "${name}" == "lora" || "${name}" == "mora" || "${name}" == "olora" || "${name}" == "lorac" || "${name}" == "lorac_ipc" ]]; then
    method_lora_alpha="${LORA_ALPHA}"
  fi

  lora_rank_args=()
  if [[ "${name}" == "lora" || "${name}" == "mora" || "${name}" == "olora" || "${name}" == "inflora" || "${name}" == "lorac" || "${name}" == "lorac_ipc" ]]; then
    if [[ -n "${LORA_RANK_Q}" ]]; then
      lora_rank_args+=(--lora_rank_q "${LORA_RANK_Q}")
    fi
    if [[ -n "${LORA_RANK_K}" ]]; then
      lora_rank_args+=(--lora_rank_k "${LORA_RANK_K}")
    fi
    if [[ -n "${LORA_RANK_GATE}" ]]; then
      lora_rank_args+=(--lora_rank_gate "${LORA_RANK_GATE}")
    fi
  fi

  curb_rank_args=()
	  if [[ "${name}" == curb_* || "${name}" == "curlora" || "${name}" == "bilora" ]]; then
	    if [[ -n "${CURB_RANK_Q}" ]]; then
	      curb_rank_args+=(--curb_rank_q "${CURB_RANK_Q}")
	    fi
    if [[ -n "${CURB_RANK_K}" ]]; then
      curb_rank_args+=(--curb_rank_k "${CURB_RANK_K}")
    fi
	    if [[ -n "${CURB_RANK_GATE}" ]]; then
	      curb_rank_args+=(--curb_rank_gate "${CURB_RANK_GATE}")
	    fi
	  fi

  bilora_args=()
  if [[ "${name}" == "bilora" ]]; then
    if [[ -n "${BILORA_K}" ]]; then
      bilora_args+=(--bilora_k "${BILORA_K}")
    fi
    if [[ -n "${BILORA_ALPHA}" && "${BILORA_ALPHA}" != "auto" ]]; then
      bilora_args+=(--bilora_alpha "${BILORA_ALPHA}")
    fi
    bilora_args+=(
      --bilora_seed "${BILORA_SEED}"
      --bilora_chunk_size "${BILORA_CHUNK_SIZE}"
      --bilora_freq_chunk_size "${BILORA_FREQ_CHUNK_SIZE}"
    )
  fi

	  inflora_calib_args=()
	  if [[ "${name}" == "inflora" && -n "${INFLORA_CALIB_SOURCE}" ]]; then
	    inflora_calib_args+=(--inflora_calib_source "${INFLORA_CALIB_SOURCE}")
	  fi
	
	    cmd=(
	      "${PYTHON_BIN}" -u experiments/llm/cl.py
	      --model_name "${MODEL_NAME}"
      --model_dtype "${MODEL_DTYPE}"
      --save_path "${seed_local}"
      --model_save_path "${seed_model}"
      --device cuda
      --train_gpu 0
      --eval_gpus 0
      --eval_pipeline sync
      --mp_start_method spawn
      --tasks "${TASKS[@]}"
      --total_round "${TOTAL_ROUND}"
      --num_test_steps "${NUM_TEST_STEPS}"
      --bootstrap_iters "${BOOTSTRAP_ITERS}"
      --seed "${seed}"
      --train_samples_per_task "${TRAIN_SAMPLES_PER_TASK}"
      --replay_buffer_per_task "${method_replay_buffer_per_task}"
      --train_batch_size "${method_train_batch_size}"
      --num_train_epochs "${method_num_train_epochs}"
      --learning_rate "${method_learning_rate}"
      --weight_decay "${WEIGHT_DECAY}"
      --grad_accum_steps "${method_grad_accum_steps}"
      --max_grad_norm "${MAX_GRAD_NORM}"
      --lr_scheduler_type "${method_lr_scheduler_type}"
      --warmup_ratio "${method_warmup_ratio}"
      --adam_beta1 "${ADAM_BETA1}"
      --adam_beta2 "${ADAM_BETA2}"
      --adam_eps "${ADAM_EPS}"
      --train_tf_log_every "${TRAIN_TF_LOG_EVERY}"
      --max_length "${MAX_LENGTH}"
      --curb_rank "${CURB_RANK}"
      --curb_calib_steps "${CURB_CALIB_STEPS}"
      --curb_batch_size "${CURB_BATCH_SIZE}"
      --curb_max_length "${CURB_MAX_LENGTH}"
      --curb_calib_category "${CURB_CALIB_CATEGORY}"
      --curb_ffn_module_names "${CURB_FFN_MODULES[@]}"
      --curb_attn_module_names "${CURB_ATTN_MODULES[@]}"
      --lora_dropout "${method_lora_dropout}"
      --olora_lambda_orth "${method_olora_lambda_orth}"
	      --olora_lambda_l2 "${method_olora_lambda_l2}"
      --lorac_ortho "${LORAC_ORTH}"
      --lorac_omega_lr_scale "${LORAC_OMEGA_LR_SCALE}"
      --lorac_ipc_beta1 "${LORAC_IPC_BETA1}"
      --lorac_ipc_beta2 "${LORAC_IPC_BETA2}"
      --lorac_ipc_threshold "${LORAC_IPC_THRESHOLD}"
	      --inflora_lamb "${INFLORA_LAMB}"
	      --inflora_lame "${INFLORA_LAME}"
	      "${inflora_calib_args[@]}"
	      "${METH_ARGS[@]}"
	      "${bilora_args[@]}"
	      "${curb_rank_args[@]}"
	      "${lora_rank_args[@]}"
	    )
    if [[ "${name}" == "lorac_ipc" ]]; then
      if [[ "${LORAC_IPC_NEW_MASK}" == "1" || "${LORAC_IPC_NEW_MASK}" == "true" ]]; then
        cmd+=(--lorac_ipc_new_mask)
      fi
    fi
    if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
      cmd+=(--max_train_steps "${MAX_TRAIN_STEPS}")
    fi
    if [[ -n "${method_learning_rate_first_task}" ]]; then
      cmd+=(--learning_rate_first_task "${method_learning_rate_first_task}")
    fi
    if [[ -n "${method_curb_alpha}" ]]; then
      cmd+=(--curb_alpha "${method_curb_alpha}")
    fi
    if [[ -n "${method_lora_alpha}" ]]; then
      cmd+=(--lora_alpha "${method_lora_alpha}")
    fi

    printf 'CUDA_VISIBLE_DEVICES=%s ' "${gpu}" >> "${CMD_TXT}"
    printf '%q ' "${cmd[@]}" >> "${CMD_TXT}"
    echo " > ${seed_log} 2>&1" >> "${CMD_TXT}"

    lr_first_text="${method_learning_rate_first_task:-none}"
    curb_alpha_text="${method_curb_alpha:-n/a}"
    bilora_alpha_text="${method_bilora_alpha:-n/a}"
    lora_alpha_text="${method_lora_alpha:-n/a}"
    log "seed=${seed} method=${name} gpu=${gpu} lr=${method_learning_rate} lr_first=${lr_first_text} "\
"bs=${method_train_batch_size} ga=${method_grad_accum_steps} sched=${method_lr_scheduler_type} warmup=${method_warmup_ratio} "\
"curb_alpha=${curb_alpha_text} bilora_alpha=${bilora_alpha_text} lora_alpha=${lora_alpha_text} lora_dropout=${method_lora_dropout} log=${seed_log}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "${seed_log}" 2>&1 &
    pid=$!

  pids+=("${pid}")
  names+=("${name}")
  logs+=("${seed_log}")
  save_paths+=("${seed_local}")
  model_save_paths+=("${seed_model}")
  gpus+=("${gpu}")
done

seed_failed=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  name="${names[$i]}"
  gpu="${gpus[$i]}"
  seed_log="${logs[$i]}"
  save_path="${save_paths[$i]}"
  model_save_path="${model_save_paths[$i]}"
  code=0
  if wait "${pid}"; then
    code=0
    log "seed=${seed} method=${name} gpu=${gpu} finished (exit=0)"
  else
    code=$?
    seed_failed=1
    overall_failed=1
    log "seed=${seed} method=${name} gpu=${gpu} failed (exit=${code})"
    log "check log: ${seed_log}"
  fi
  echo "$(ts),${seed},${name},${gpu},${code},${seed_log},${save_path},${model_save_path}" >> "${STATUS_CSV}"
done

if (( seed_failed )); then
  log "=== Seed ${seed} finished with failures ==="
else
  log "=== Seed ${seed} launch done (all success) ==="
fi

if (( overall_failed )); then
  log "Launcher finished with failures. status=${STATUS_CSV}"
  exit 1
fi

log "Launcher completed successfully. status=${STATUS_CSV}"
