#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'USAGE'
Run multi-teacher on-policy distillation on local paths.

Provide at least two teacher directories with repeated --teacher options or
TEACHER_MODEL_PATHS (comma-separated). TEACHER_DOMAINS can name them in the
same order; its default is math,code,if.

The command is printed without execution unless --run is supplied.
USAGE
    local_common_usage
}

local_init
local_parse_common "$@"
local_scope_output mt_opd
if ((LOCAL_HELP)); then
    usage
    exit 0
fi

if ((${#LOCAL_TEACHER_PATHS[@]} == 0)) && [[ -n "${TEACHER_MODEL_PATHS:-}" ]]; then
    IFS=',' read -r -a LOCAL_TEACHER_PATHS <<< "${TEACHER_MODEL_PATHS}"
fi
if ((${#LOCAL_TEACHER_PATHS[@]} == 0)) && [[ -n "${REWARD_MODEL_PATH:-}" ]]; then
    LOCAL_TEACHER_PATHS+=("${REWARD_MODEL_PATH}")
fi
if ((${#LOCAL_TEACHER_PATHS[@]} == 0)) && [[ -n "${TEACHER_PATH:-}" ]]; then
    LOCAL_TEACHER_PATHS+=("${TEACHER_PATH}")
fi

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
PPO_MICRO_BATCH_PER_GPU="${PPO_MICRO_BATCH_PER_GPU:-4}"
LOG_PROB_MICRO_BATCH_PER_GPU="${LOG_PROB_MICRO_BATCH_PER_GPU:-4}"
RM_MICRO_BATCH_PER_GPU="${RM_MICRO_BATCH_PER_GPU:-4}"
LOG_PROB_TOP_K="${LOG_PROB_TOP_K:-256}"
ACTOR_GRAD_CLIP="${ACTOR_GRAD_CLIP:-1.0}"
MOPD_GRADIENT_PROJECTION_MODE="${MOPD_GRADIENT_PROJECTION_MODE:-none}"
MOPD_GRADIENT_PROJECTION_EPSILON="${MOPD_GRADIENT_PROJECTION_EPSILON:-0.0}"
MOPD_GRADIENT_PROJECTION_GRAD_OFFLOAD="${MOPD_GRADIENT_PROJECTION_GRAD_OFFLOAD:-cpu}"
MOPD_GRADIENT_PROJECTION_DECOMPOSITION_RTOL="${MOPD_GRADIENT_PROJECTION_DECOMPOSITION_RTOL:-1e-3}"
MOPD_GRADIENT_PROJECTION_CHUNK_NUMEL="${MOPD_GRADIENT_PROJECTION_CHUNK_NUMEL:-1000000}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
N_RESPONSES="${N_RESPONSES:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"

local_export_runtime
# Derive the logger path after parsing --output so it follows the actual run.
export VERL_FILE_LOGGER_PATH="${VERL_FILE_LOGGER_PATH:-${LOCAL_OUTPUT_DIR}/metrics.jsonl}"
cd "$LOCAL_TRAINING_DIR"
if [[ "$LOCAL_RUN" == 1 ]]; then
    local_validate_common 1 1 1
    ((${#LOCAL_TEACHER_PATHS[@]} >= 2)) || local_die "at least two --teacher paths are required"
    for teacher_path in "${LOCAL_TEACHER_PATHS[@]}"; do
        local_require_local_path teacher "$teacher_path"
        local_require_path teacher "$teacher_path"
    done
    local_prepare_output
fi

IFS=',' read -r -a domain_array <<< "${LOCAL_DOMAINS}"
if ((${#LOCAL_TEACHER_PATHS[@]} > 0)) && ((${#domain_array[@]} != ${#LOCAL_TEACHER_PATHS[@]})); then
    local_die "teacher domain count (${#domain_array[@]}) must match teacher path count (${#LOCAL_TEACHER_PATHS[@]})"
fi

domain_list="["
for i in "${!domain_array[@]}"; do
    [[ "$i" == 0 ]] || domain_list+=","
    domain_list+="${domain_array[$i]}"
done
domain_list+="]"

first_teacher="${LOCAL_TEACHER_PATHS[0]:-${REWARD_MODEL_PATH:-${LOCAL_MODEL_PATH}}}"
teacher_count="${#LOCAL_TEACHER_PATHS[@]}"
if ((teacher_count == 0)); then
    # Keep a readable dry-run command; --run still requires two real teachers.
    teacher_count=1
fi
cmd=(
    "$LOCAL_PYTHON_BIN" -m verl.trainer.main_ppo
    "algorithm.adv_estimator=token_reward_direct"
    "data.train_files=${LOCAL_TRAIN_FILE}"
    "data.val_files=${LOCAL_VAL_FILE}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "actor_rollout_ref.model.path=${LOCAL_MODEL_PATH}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_PER_GPU}"
    "actor_rollout_ref.actor.grad_clip=${ACTOR_GRAD_CLIP}"
    "actor_rollout_ref.actor.mopd_gradient_projection_mode=${MOPD_GRADIENT_PROJECTION_MODE}"
    "actor_rollout_ref.actor.mopd_gradient_projection_epsilon=${MOPD_GRADIENT_PROJECTION_EPSILON}"
    "actor_rollout_ref.actor.mopd_gradient_projection_grad_offload=${MOPD_GRADIENT_PROJECTION_GRAD_OFFLOAD}"
    "actor_rollout_ref.actor.mopd_gradient_projection_decomposition_rtol=${MOPD_GRADIENT_PROJECTION_DECOMPOSITION_RTOL}"
    "actor_rollout_ref.actor.mopd_gradient_projection_chunk_numel=${MOPD_GRADIENT_PROJECTION_CHUNK_NUMEL}"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_PER_GPU}"
    "reward_model.micro_batch_size_per_gpu=${RM_MICRO_BATCH_PER_GPU}"
    "actor_rollout_ref.rollout.name=vllm"
    "+actor_rollout_ref.rollout.reward_mode=mt_opd"
    "+actor_rollout_ref.rollout.log_prob_top_k=${LOG_PROB_TOP_K}"
    "actor_rollout_ref.rollout.n=${N_RESPONSES}"
    "actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
    "reward_model.enable=True"
    "reward_model.model.path=${first_teacher}"
    "reward_model.model.input_tokenizer=null"
    "+reward_model.reward_kwargs.compute_true_reward=False"
    "+mt_opd.teacher_domains=${domain_list}"
    "+mt_opd.n_additional_teachers=$((teacher_count - 1))"
    "trainer.n_gpus_per_node=${LOCAL_GPUS}"
    "trainer.nnodes=${LOCAL_NODES}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.default_local_dir=${LOCAL_CHECKPOINT_DIR}"
    "trainer.project_name=${PROJECT_NAME:-OpenOPD-local}"
    "trainer.experiment_name=${EXPERIMENT_NAME:-mt-opd-local}"
    "trainer.resume_mode=${RESUME_MODE:-auto}"
    "trainer.logger=${TRAINER_LOGGER:-['console']}"
)

if [[ -n "${DATA_SEED:-}" ]]; then
    cmd+=("data.seed=${DATA_SEED}")
fi
if [[ -n "${ROLLOUT_SEED:-}" ]]; then
    cmd+=("+actor_rollout_ref.rollout.seed=${ROLLOUT_SEED}")
fi

for ((i = 1; i < ${#LOCAL_TEACHER_PATHS[@]}; i++)); do
    teacher_index="$i"
    teacher_path="${LOCAL_TEACHER_PATHS[$i]}"
    cmd+=(
        "+mt_reward_model_${teacher_index}.enable=True"
        "+mt_reward_model_${teacher_index}.model.path=${teacher_path}"
        "+mt_reward_model_${teacher_index}.model.input_tokenizer=null"
        "+mt_reward_model_${teacher_index}.model.use_remove_padding=True"
        "+mt_reward_model_${teacher_index}.model.fsdp_config.param_offload=True"
    )
done
if ((${#LOCAL_EXTRA_OVERRIDES[@]})); then
    cmd+=("${LOCAL_EXTRA_OVERRIDES[@]}")
fi
if ((${#LOCAL_REMAINING[@]})); then
    cmd+=("${LOCAL_REMAINING[@]}")
fi

local_run_command "${cmd[@]}"
