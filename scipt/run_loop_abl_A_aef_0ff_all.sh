#!/bin/bash

set -euo pipefail

PROJECT_ROOT="/CMAPSS-release"
cd "$PROJECT_ROOT"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DATASETS="${DATASETS:-FD001 FD002 FD003 FD004}"
MODEL_CODE="A_AEF_0FF"

COMMON_ARGS=(
    --gat-num-layers 2
    --gat-embed-dim 8
    --gat-topk 7
    --lr-scheduler step
    --lr 0.002
)

for dataset in $DATASETS; do
    smooth_rate="40"
    if [[ "$dataset" == "FD001" || "$dataset" == "FD003" ]]; then
        smooth_rate="30"
    fi

    echo "====================================="
    echo "Running ${MODEL_CODE} on ${dataset}"
    echo "smooth-rate: ${smooth_rate}"
    echo "RUN_TAG: ${RUN_TAG}"
    echo "====================================="

    PYTHONPATH="$PROJECT_ROOT" python scipt/train_model.py \
        "${COMMON_ARGS[@]}" \
        --sub-dataset "$dataset" \
        --smooth-rate "$smooth_rate" \
        --apply-code-ablation \
        --model-code "$MODEL_CODE"
done

echo "Completed ${MODEL_CODE} ablation on datasets: ${DATASETS}. RUN_TAG=${RUN_TAG}"
