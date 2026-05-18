#!/bin/bash

set -euo pipefail

PROJECT_ROOT="/CMAPSS-release"
cd "$PROJECT_ROOT"

# Fixed seed plan (as requested).
SEEDS=(2 17 27 30 33 51 62 80 88 97)

DATASETS=(FD001 FD002 FD003 FD004)

for DATASET in "${DATASETS[@]}"; do
  if [[ "$DATASET" == "FD001" || "$DATASET" == "FD003" ]]; then
    SMOOTH_RATE=30
  else
    SMOOTH_RATE=40
  fi

  echo "====================================="
  echo "SGFormer 训练开始 | DATASET=${DATASET} | smooth_rate=${SMOOTH_RATE} | max_epochs=30"
  echo "Seeds: ${SEEDS[*]}"
  echo "====================================="

  PYTHONPATH="$PROJECT_ROOT" python SGFormer_test/train.py \
    --sub-dataset "$DATASET" \
    --max-epochs 30 \
    --smooth-rate "$SMOOTH_RATE" \
    --seed-list "${SEEDS[@]}" \
    --model-code "SGFormer" \
    --disable-code-ablation \
    --spatial-backbone sgformer

done

echo "全部数据集已执行完成: ${DATASETS[*]}"