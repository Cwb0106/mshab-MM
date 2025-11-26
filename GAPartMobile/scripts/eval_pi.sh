#!/usr/bin/bash

# bash GAPartMobile/scripts/eval_pi.sh set_table place 024_bowl 4 8



SEED=0

TRAJS_PER_OBJ=1000
epochs=100

TASK=$1         # set_table
SUBTASK=$2      # pick
OBJ=$3          # 024_bowl
GPU_ID=$4       # 0
BATCH_SIZE=$5   # 512

SPLIT=train

export WANDB_API_KEY=18fc77deb9293ea52363970a58ba11e3a2f85c82
export WANDB_USER_EMAIL=1297691410@qq.com
export WANDB_USERNAME=wenbocui

# export HTTP_PROXY="http://127.0.0.1:7890"
# export HTTPS_PROXY="http://127.0.0.1:7890"

# shellcheck disable=SC2001
ENV_ID="$(echo $SUBTASK | sed 's/\b\(.\)/\u\1/g')SubtaskTrain-v0"
WORKSPACE="mshab_exps_predict"
GROUP=$TASK-rcad-MM-$SUBTASK
EXP_NAME="$ENV_ID/$GROUP/bc-$SUBTASK-$OBJ-local-trajs_per_obj=$TRAJS_PER_OBJ"
# shellcheck disable=SC2001
PROJECT_NAME="MS-HAB-RCAD-bccnn-eval"

WANDB=False
TENSORBOARD=False
if [[ -z "${MS_ASSET_DIR}" ]]; then
    MS_ASSET_DIR="$HOME/.maniskill"
fi

RESUME_LOGDIR="$WORKSPACE/$EXP_NAME"
RESUME_CONFIG="$RESUME_LOGDIR/config.yml"

MAX_CACHE_SIZE=300000   # safe num for about 64 GiB system memory


# data_dir_fp="/raid/wenbo/data/mshab/gen_data_save_trajectories/$TASK/$SUBTASK/$SPLIT/$OBJ"
data_dir_fp="/raid/wenbo/project/mshab/mshab_exps/gen_data_save_trajectories/$TASK/$SUBTASK/$SPLIT/$OBJ"

# NOTE: the below args are defaults, however the released checkpoints may use different hyperparameters. To train using the same args, check the config.yml files from the released checkpoints.
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
    "algo.save_backup_ckpts=True"
    "eval_env.make_env=True"
    "eval_env.num_envs=8"
    "eval_env.require_build_configs_repeated_equally_across_envs=False"
    "eval_env.max_episode_steps=100"
    "eval_env.record_video=True"
    "eval_env.info_on_video=True"
    "eval_env.save_video_freq=1"
    "logger.wandb=$WANDB"
    "logger.tensorboard=$TENSORBOARD"
    "logger.project_name=$PROJECT_NAME"
    "logger.workspace=$WORKSPACE"
)

# --- 4. 执行命令 ---
echo "STARTING A NEW TRAINING RUN"

echo "task_plan_fp=$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/$TASK/$SUBTASK/$SPLIT/$OBJ.json"
echo "spawn_data_fp=$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/spawn_data/$TASK/$SUBTASK/$SPLIT/spawn_data.pt"


# 直接运行训练命令，不进行恢复检查
CUDA_VISIBLE_DEVICES=$GPU_ID python -m GAPartMobile.eval.eval_pi \
    logger.clear_out="True" \
    logger.best_stats_cfg="{eval/success_once: 1, eval/return_per_step: 1}" \
    "${args[@]}"