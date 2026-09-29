#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export COSMOS_MODEL_PATH=$ROOT/checkpoints/pretrained/Cosmos-Reason2-2B

INFER_MODEL=/share/project/zjk/yyb/Isaac-GR00T/.checkpoint/astri_sequential_runs/PUSH_STOOL_ASTRI/2026-0806-032556/checkpoint-25000

python gr00t/eval/run_gr00t_server.py \
  --model-path $INFER_MODEL \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \