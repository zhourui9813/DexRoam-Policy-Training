#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=0
export OPENPI_DATA_HOME="$ROOT/checkpoints/pretrained/paligemma"
export HF_LEROBOT_HOME="$ROOT/../data_lerobot"

CONFIG_NAME="DEXROAM_EXAMPLE"

uv run python scripts/compute_norm_stats.py \
    --config-name "$CONFIG_NAME"