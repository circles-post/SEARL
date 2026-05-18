#!/usr/bin/env bash
# Rebuttal epoch-1 GIGPO training entrypoint.

set -euo pipefail
set -x

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
export RL_FACTORY_ROOT="${RL_FACTORY_ROOT:-$REPO_ROOT}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/wandb}"
export EXP_NAME="${EXP_NAME:-epoch1_new}"
export PROJECT_NAME="${PROJECT_NAME:-epoch1_new}"
export DYNAMIC_PROMPT="${DYNAMIC_PROMPT:-False}"

export MODEL_PATH="${MODEL_PATH:-/path/to/rebuttal_cold_start_4B_baseline/checkpoint-965}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-/path/to/Qwen/QwQ-32B}"
export TRAIN_FILES="${TRAIN_FILES:-data/train_v4_plan_merged.parquet}"
export VAL_FILES="${VAL_FILES:-[data/qa_test_our_plan.parquet,data/test_math_v2_our_plan.parquet,data/test_aime25_our_plan.parquet]}"
export RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/outputs/checkpoints/${PROJECT_NAME}}"
export LOG_PATH="${LOG_PATH:-$REPO_ROOT/logs/${PROJECT_NAME}.log}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$RESULT_DIR" "$(dirname "$LOG_PATH")" "$WANDB_DIR"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login "$WANDB_API_KEY"
fi

rm -f "$REPO_ROOT/envs/tools/invented_tools/${PROJECT_NAME}/official_tool_list.py"

"$PYTHON_BIN" -m verl.trainer.main_ppo --config-name=rl_factory_ppo_trainer \
    algorithm.adv_estimator=gigpo\
    data.train_files="$TRAIN_FILES"\
    data.val_files="$VAL_FILES"\
    data.train_batch_size=256\
    data.max_prompt_length="$MAX_PROMPT_LENGTH"\
    data.max_response_length="$MAX_RESPONSE_LENGTH"\
    actor_rollout_ref.model.path="$MODEL_PATH"\
    actor_rollout_ref.model.use_remove_padding=True\
    actor_rollout_ref.model.enable_gradient_checkpointing=True\
    actor_rollout_ref.actor.optim.lr=1e-6\
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1\
    actor_rollout_ref.actor.use_kl_loss=True\
    actor_rollout_ref.actor.kl_loss_coef=0.000\
    actor_rollout_ref.actor.kl_loss_type=low_var_kl\
    actor_rollout_ref.actor.fsdp_config.param_offload=True\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True\
    actor_rollout_ref.actor.state_masking=True\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1\
    actor_rollout_ref.rollout.name=vllm\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9\
    actor_rollout_ref.rollout.n=8\
    actor_rollout_ref.rollout.max_turns=6\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1\
    actor_rollout_ref.ref.fsdp_config.param_offload=True\
    actor_rollout_ref.rollout.enforce_eager=True\
    actor_rollout_ref.rollout.free_cache_engine=True\
    +actor_rollout_ref.rollout.max_tokens=$((MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))\
    actor_rollout_ref.env.name=mcp_base\
    actor_rollout_ref.env.mcp_mode=stdio\
    actor_rollout_ref.env.enable_limiter=True\
    actor_rollout_ref.env.tool_manager=centralized_qwen3\
    actor_rollout_ref.env.enable_thinking=False\
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((6*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((6*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((6*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.env.official_config_path=envs/configs/official_mcps.pydata \
    actor_rollout_ref.env.temp_config_path=envs/configs/temp_mcps.pydata \
    actor_rollout_ref.env.config_path=envs/configs/mcp_tools_local_graph.pydata \
    actor_rollout_ref.env.use_process_reward=True\
    actor_rollout_ref.env.local_search=True\
    actor_rollout_ref.env.mcp_enable=True\
    actor_rollout_ref.env.tool_timeout=30\
    actor_rollout_ref.env.use_graphrag=True\
    actor_rollout_ref.env.max_concurrency=100\
    actor_rollout_ref.env.project_name="$PROJECT_NAME"\
    actor_rollout_ref.env.dynamic_prompt="$DYNAMIC_PROMPT"\
    reward_rollout.if_use_reward_rollout=False\
    reward_rollout.rollout.tensor_model_parallel_size=1\
    reward_rollout.rollout.gpu_memory_utilization=0.8\
    reward_rollout.rollout.model_name="$REWARD_MODEL_PATH"\
    reward_rollout.rollout.free_cache_engine=True\
    reward_rollout.rollout.response_length=2048\
    reward_model.reward_manager=parallel\
    algorithm.kl_ctrl.kl_coef=0.000\
    trainer.critic_warmup=0\
    trainer.logger=['console']\
    trainer.project_name="$PROJECT_NAME"\
    trainer.experiment_name="$EXP_NAME"\
    trainer.n_gpus_per_node=4\
    trainer.nnodes=1\
    trainer.val_before_train=True\
    trainer.default_local_dir="$RESULT_DIR"\
    trainer.default_hdfs_dir=null\
    trainer.save_freq=0\
    trainer.test_freq=5\
    trainer.total_epochs=1 "$@" 2>&1 | tee "$LOG_PATH"
