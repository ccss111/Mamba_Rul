#!/bin/bash

set -euo pipefail

PROJECT_ROOT="/CMAPSS-release"
cd "$PROJECT_ROOT"

DATASET="FD002"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
METHOD_CODES="${METHOD_CODES:-A A_aef_0ff B C D}"

# Shared hyper-parameters for latest ablation runs.
COMMON_ARGS=(
    --sub-dataset "$DATASET"
    --smooth-rate 40
    --gat-num-layers 2
    --gat-embed-dim 8
    --gat-topk 7
    --lr-scheduler step
    --lr 0.002
)

method_enabled() {
    local target="$1"
    local target_upper="${target^^}"
    for code in $METHOD_CODES; do
        if [[ "${code^^}" == "$target_upper" ]]; then
            return 0
        fi
    done
    return 1
}

run_ablation() {
    local code="$1"
    local desc="$2"
    local preset_code="${code^^}"

    echo "====================================="
    echo "开始运行方法 ${code}: ${desc}"
    echo "数据集: ${DATASET}"
    echo "====================================="

    PYTHONPATH="$PROJECT_ROOT" python scipt/train_model.py \
        "${COMMON_ARGS[@]}" \
        --apply-code-ablation \
        --model-code "${preset_code}"
}

# (A) KNN graph + GAT-LSTM with encoder and decoder
if method_enabled "A"; then
    run_ablation "A" "KNN graph + GAT-LSTM with encoder and decoder"
fi

# (A_aef_0ff) KNN graph + GAT-LSTM with encoder and decoder (AEF off)
if method_enabled "A_aef_0ff"; then
    run_ablation "A_aef_0ff" "KNN graph + GAT-LSTM with encoder and decoder (AEF off)"
fi

# (B) Full-connected graph + GAT-LSTM with encoder and decoder
if method_enabled "B"; then
    run_ablation "B" "Full-connected graph + GAT-LSTM with encoder and decoder"
fi

# (C) KNN graph + GAT-LSTM with encoder only
if method_enabled "C"; then
    run_ablation "C" "KNN graph + GAT-LSTM with encoder only"
fi

# (D) Original GAT-LSTM without encoder and decoder
if method_enabled "D"; then
    run_ablation "D" "Original GAT-LSTM without encoder and decoder"
fi

echo "FD002 的消融实验已执行完成。RUN_TAG=${RUN_TAG}, METHOD_CODES=${METHOD_CODES}"