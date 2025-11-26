#!/usr/bin/bash

# 示例用法:
# 2 卡训练 (GPUs 0 和 1), global batch size 128
# bash GAPartMobile/scripts/train_mm_pi.sh set_table place 024_bowl 

# 4 卡训练 (GPUs 0,1,2,3), global batch size 512
# bash GAPartMobile/scripts/train_mm_pi.sh set_table place 024_bowl "0,1,2,3" 512

SEED=0

TRAJS_PER_OBJ=1000
epochs=100

TASK=$1         # set_table
SUBTASK=$2      # pick
OBJ=$3          # 024_bowl
GPU_IDS="1"      # 【修改】现在是逗号分隔的列表, e.g., "0,1" or "0,1,2,3"
BATCH_SIZE=1   # 【不变】这是 Global Batch Size

SPLIT=train

export WANDB_API_KEY=18fc77deb9293ea52363970a58ba11e3a2f85c82
export WANDB_USER_EMAIL=1297691410@qq.com
export WANDB_USERNAME=wenbocui

# export HTTP_PROXY="http://127.0.0.1:7890"
# export HTTPS_PROXY="http://127.0.0.1:7890"

# --- 1. 新增：计算 GPU 数量 (nproc) ---
# 计算 $GPU_IDS 中逗号的数量, 然后加 1, 得到 GPU 的总数
NPROC=$(( $(echo "$GPU_IDS" | tr -cd ',' | wc -c) + 1 ))

# --- 2. 修改：导出 CUDA_VISIBLE_DEVICES ---
# 告诉 torchrun 哪些 GPU 是可见的
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# (环境变量和路径设置保持不变)
# shellcheck disable=SC2001
ENV_ID="$(echo $SUBTASK | sed 's/\b\(.\)/\u\1/g')SubtaskTrain-v0"
WORKSPACE="mshab_exps_predict"
GROUP=$TASK-rcad-MM-$SUBTASK
EXP_NAME="$ENV_ID/$GROUP/bc-$SUBTASK-$OBJ-local-trajs_per_obj=$TRAJS_PER_OBJ"
# shellcheck disable=SC2001
PROJECT_NAME="MS-HAB-RCAD-bc-world-memory-vision-only"

WANDB=False
TENSORBOARD=False
if [[ -z "${MS_ASSET_DIR}" ]]; then
    MS_ASSET_DIR="$HOME/.maniskill"
fi

RESUME_LOGDIR="$WORKSPACE/$EXP_NAME"
RESUME_CONFIG="$RESUME_LOGDIR/config.yml"

MAX_CACHE_SIZE=300000   # safe num for about 64 GiB system memory
data_dir_fp="/raid/wenbo/project/mshab/mshab_exps/gen_data_save_trajectories/$TASK/$SUBTASK/$SPLIT/$OBJ"

# (args 数组保持不变)
# 您的 DDP 脚本会自动将 "algo.batch_size" 识别为 Global Batch Size
# 并将其除以 NPROC 来得到 per-gpu batch size，所以这里无需更改。
args=(
    "logger.wandb_cfg.group=$GROUP"
    "logger.exp_name=$EXP_NAME"
    "seed=$SEED"
    "eval_env.env_id=$ENV_ID"
    "eval_env.task_plan_fp=$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/$TASK/$SUBTASK/$SPLIT/$OBJ.json"
    "eval_env.spawn_data_fp=$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/spawn_data/$TASK/$SUBTASK/$SPLIT/spawn_data.pt"
    "eval_env.frame_stack=1"
    "algo.epochs=$epochs"
    "algo.batch_size=$BATCH_SIZE"
    "algo.trajs_per_obj=$TRAJS_PER_OBJ"
    "algo.data_dir_fp=$data_dir_fp"
    "algo.max_cache_size=$MAX_CACHE_SIZE"
    "algo.eval_freq=1"
    "algo.log_freq=1"
    "algo.save_freq=1"
    "eval_env.make_env=True"
    "eval_env.num_envs=189"
    "eval_env.max_episode_steps=200"
    "eval_env.record_video=True"
    "eval_env.info_on_video=True"
    "eval_env.save_video_freq=1"
    "logger.wandb=$WANDB"
    "logger.tensorboard=$TENSORBOARD"
    "logger.project_name=$PROJECT_NAME"
    "logger.workspace=$WORKSPACE"
)

# --- 3. 修改：使用 torchrun 执行命令 ---
echo "STARTING A NEW DDP TRAINING RUN"
echo "Visible GPUs: [$GPU_IDS] (Total: $NPROC)"
echo "Global Batch Size: $BATCH_SIZE"

echo "task_plan_fp=$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/$TASK/$SUBTASK/$SPLIT/$OBJ.json"
echo "spawn_data_fp=$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/spawn_data/$TASK/$SUBTASK/$SPLIT/spawn_data.pt"

# 使用 torchrun 启动 DDP 训练
torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC \
    -m GAPartMobile.main_pi \
    logger.clear_out="True" \
    logger.best_stats_cfg="{eval/success_once: 1, eval/return_per_step: 1}" \
    "${args[@]}"