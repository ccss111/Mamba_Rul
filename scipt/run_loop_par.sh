#!/bin/bash

set -euo pipefail

PROJECT_ROOT="/CMAPSS-release"
cd "$PROJECT_ROOT"

# 格式：层数_隐藏层维度
configs=(
  "1_8"           # I: (8)
  "2_8"           # II: (8,8)
  "2_16"          # III: (16,16)
  "3_8"           # IV: (8,8,8)
  "3_16"          # V: (16,16,16)
  "2_12"          # VI: (12,12)
  "2_10"          # VII: (10,10)
  "2_6"           # VIII: (6,6)
  "2_4"           # IX: (4,4)
)

# 遍历执行你要的配置
for cfg in "${configs[@]}"; do
    # 拆分 gat_num_layers 和 gat_hidden_dim
    IFS='_' read -r gat_num_layers gat_hidden_dim <<< "$cfg"

    echo "====================================="
    echo "当前运行: GAT层数 = $gat_num_layers"
    echo "隐藏层维度: $gat_hidden_dim"
    echo "====================================="

    PYTHONPATH="$PROJECT_ROOT" python scipt/parametric_statistics.py \
        --sub-dataset FD004 \
        --gat-num-layers "$gat_num_layers" \
        --gat-embed-dim 16 \
        --gat-topk 5 \
        --lr-scheduler step \
        --lr 0.002 \
        --decoder-fusion concat \
        --model-code "FD004_GAT_${gat_num_layers}L_hidden_dim${gat_hidden_dim}" \
        --gat-hidden-dim "$gat_hidden_dim"

done

echo "统计GAT 的 参数量已全部启动并执行完成。"