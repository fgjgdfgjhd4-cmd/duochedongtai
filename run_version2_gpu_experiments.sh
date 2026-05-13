#!/usr/bin/env bash
set -euo pipefail

cd /home/bit1011/code/duochedongtai

LOG_PATH="experiment_runs/version2_full_gpu_20260513.log"
mkdir -p experiment_runs

{
  echo "[START] $(date) version2 full gpu"
  conda run --no-capture-output -n myenv python -u version2/run_full_gpu_experiments.py \
    --suite all \
    --agent-nums 5 10 \
    --map-indices 0 1 2 \
    --episodes 10 \
    --max-ep-len 200 \
    --population 256 \
    --iterations 30 \
    --checkpoint PPO_preTrained/Map/10_robots/0511-15-16PPO_Map_6000.pth \
    --output-dir version2_results
  echo "[DONE] $(date) version2 full gpu"
} >> "${LOG_PATH}" 2>&1
