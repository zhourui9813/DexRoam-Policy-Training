#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

#############################
# Environment Configuration
#############################

export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "$NUM_GPUS"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1


#############################
# Training Configuration
#############################

export COSMOS_MODEL_PATH="$ROOT/checkpoints/pretrained/Cosmos-Reason2-2B"
export BASE_MODEL="$ROOT/checkpoints/pretrained/GR00T-N1.7-3B"

NAME="DEXROAM_GROOT_N1.7"
DATASET="$ROOT/../data_lerobot/Astribot-Xhand-data/robot_push_chair_and_close_laptop"
MODALITY_CONFIG="$ROOT/config/modalities/astri_xhand_stereo_config.py"

OUTPUT_DIR="$ROOT/checkpoints/finetuned/$NAME/$(date +%Y-%m%d-%H%M%S)"
MASTER_PORT=29501
SAVE_TOTAL_LIMIT=8
SAVE_STEPS=10000
MAX_STEPS=40000
GLOBAL_BATCH_SIZE=12
DATALOADER_NUM_WORKERS=4

#############################
# Launch Training
#############################

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    gr00t/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DATASET" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path MODALITY_CONFIG \
    --num-gpus "$NUM_GPUS" \
    --output-dir "$OUTPUT_DIR" \
    --save-total-limit "$SAVE_TOTAL_LIMIT" \
    --save-steps "$SAVE_STEPS" \
    --max-steps "$MAX_STEPS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --color-jitter-params \
        brightness 0.3 \
        contrast 0.4 \
        saturation 0.5 \
        hue 0.08 \
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"