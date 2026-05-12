
#!/bin/bash

set -euo pipefail

PROJECT_ROOT="/CMAPSS-release"
cd "$PROJECT_ROOT"
for gat_embed_dim in 8 16 32 64; do
    for gat_topk in 3 5 7; do
        if [[ $gat_embed_dim -eq 16 && $gat_topk -eq 5 ]]; then
            echo "跳过组合：embed_dim=16, topk=5"
            continue
        fi
        echo "====================================="
        echo "数据集: FD004"
        echo "GAT传感器嵌入维度: $gat_embed_dim"
        echo "GAT邻居数(topk): $gat_topk"
        echo "====================================="

        PYTHONPATH="$PROJECT_ROOT" python scipt/train_model.py \
            --sub-dataset FD004 \
            --gat-num-layers 2 \
            --gat-embed-dim "$gat_embed_dim" \
            --gat-topk "$gat_topk" \
            --lr-scheduler step \
            --lr 0.002 \
            --decoder-fusion concat \
            --model-code "FD004_GAT_embed_dim${gat_embed_dim}_topk${gat_topk}" 
    done
done

echo "FD004 训练已全部启动并执行完成。"