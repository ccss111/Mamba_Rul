#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# mamba_ssm.Mamba requires CUDA.
python - <<'PY'
import torch
if not torch.cuda.is_available():
  raise SystemExit("ERROR: CUDA not available, but mamba_ssm.Mamba requires CUDA.\n"
           "Please run on a machine with GPU/CUDA and a CUDA-enabled PyTorch.")
print("CUDA available:", torch.cuda.get_device_name(0))
PY

# Fixed seed plan (as requested).
SEEDS=(17 27 30 33 51 62 80 88 97)
# 2 17 27 30 33 51 62 80 88 97
DATASETS=(FD001 FD002 FD003 FD004)
# FD002 FD003 FD004

MAX_EPOCHS="${MAX_EPOCHS:-30}"
MODEL_CODE="${MODEL_CODE:-MB}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.002}"

for DATASET in "${DATASETS[@]}"; do
  if [[ "$DATASET" == "FD001" || "$DATASET" == "FD003" ]]; then
    SMOOTH_RATE=30
  else
    SMOOTH_RATE=40
  fi

  echo "====================================="
  echo " 训练开始 | DATASET=${DATASET} | smooth_rate=${SMOOTH_RATE} | max_epochs=${MAX_EPOCHS}"
  echo "Seeds: ${SEEDS[*]}"
  echo "====================================="

  PYTHONPATH="$PROJECT_ROOT" python scipt/train_model.py \
    --sub-dataset "$DATASET" \
    --max-epochs "$MAX_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --smooth-rate "$SMOOTH_RATE" \
    --seed-list "${SEEDS[@]}" \
    --model-code "$MODEL_CODE"

done

echo "全部数据集已执行完成: ${DATASETS[*]}"