#!/bin/bash

set -euo pipefail

PROJECT_ROOT="/CMAPSS-release"
cd "$PROJECT_ROOT"

run_dataset_ablation() {
    local dataset="$1"

    # Shared hyper-parameters for A-E ablation runs.
    local common_args=(
        --sub-dataset "$dataset"
        --gat-num-layers 2
        --gat-embed-dim 8
        --gat-topk 7
        --lr-scheduler step
        --lr 0.002
    )

    run_ablation() {
        local code="$1"
        local desc="$2"

        echo "====================================="
        echo "开始运行方法 ${code}: ${desc}"
        echo "数据集: ${dataset}"
        echo "====================================="

        PYTHONPATH="$PROJECT_ROOT" python scipt/train_model.py \
            "${common_args[@]}" \
            --apply-code-ablation \
            --model-code "${dataset}_${code}"
    }

    # (F) KNN graph + GAT + EncoderDecoder-LSTM without AOF
    run_ablation "F" "KNN graph + GAT + EncoderDecoder-LSTM without AOF"

    unset -f run_ablation
}
run_dataset_ablation "FD001"
run_dataset_ablation "FD002"
run_dataset_ablation "FD003"
run_dataset_ablation "FD004"

echo "FD001/FD002/FD003/FD004 的 F 消融实验已全部执行完成。"
