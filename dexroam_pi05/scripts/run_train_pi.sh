#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"


#############################
# Environment Configuration
#############################

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

export OPENPI_DATA_HOME="$ROOT/checkpoints/pretrained/paligemma"
export HF_LEROBOT_HOME="$ROOT/../data_lerobot"


#############################
# Training Configuration
#############################

TRAIN_CONFIG_NAME="DEXROAM_EXAMPLE"
RUN_TIMESTAMP=$(date +"%Y-%m%d-%H%M")
EXP_NAME="${TRAIN_CONFIG_NAME}-${RUN_TIMESTAMP}"
ASSETS_BASE_DIR="$ROOT/assets"
CHECKPOINT_BASE_DIR="$ROOT/checkpoints/finetuned"


#############################
# Launch Training
#############################

echo "EXP_NAME=${EXP_NAME}"

uv run --active scripts/train.py \
    "$TRAIN_CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --overwrite \
    --assets-base-dir="$ASSETS_BASE_DIR" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" 