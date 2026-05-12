#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/CMAPSS-release"
TRAIN_SCRIPT="${PROJECT_ROOT}/scipt/train_GAT_LSTM.py"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"

DATASETS=("FD001" "FD003")
# "FD002""FD004"

# 固定实验参数（可通过环境变量覆盖）
GAT_HIDDEN_DIMS="${GAT_HIDDEN_DIMS:-8,8,8}"
GAT_EMBED_DIM="${GAT_EMBED_DIM:-8}"
GAT_TOPK="${GAT_TOPK:-7}"
SEEDS="${SEEDS:-2,17,27,30,33,51,62,80,88,97}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-2e-3}"
SMOOTH_RATE_OVERRIDE="${SMOOTH_RATE_OVERRIDE:-}"

NO_CUDA_ARGS=()
if [[ "${NO_CUDA:-0}" == "1" ]]; then
  NO_CUDA_ARGS+=("--no-cuda")
fi

run_variant() {
  local dataset="$1"
  local tag="$2"
  local graph_mode="$3"
  local aef_flag="$4"
  local aof_flag="$5"
  local model_desc="$6"
  local smooth_rate="${SMOOTH_RATE_OVERRIDE}"

  if [[ -z "${smooth_rate}" ]]; then
    if [[ "${dataset}" == "FD001" || "${dataset}" == "FD003" ]]; then
      smooth_rate="30"
    else
      smooth_rate="40"
    fi
  fi

  echo "==============================================="
  echo "开始训练: 数据集=${dataset}, 方案=${tag}"
  echo "说明: ${model_desc}"
  echo "平滑系数: ${smooth_rate}"
  echo "==============================================="

  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --sub-dataset "${dataset}" \
    --gat-hidden-dims "${GAT_HIDDEN_DIMS}" \
    --gat-embed-dim "${GAT_EMBED_DIM}" \
    --gat-topk "${GAT_TOPK}" \
    --graph-mode "${graph_mode}" \
    "${aef_flag}" \
    "${aof_flag}" \
    --max-epochs "${MAX_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --smooth-rate "${smooth_rate}" \
    --seeds "${SEEDS}" \
    --model-code "${dataset}_${tag}" \
    "${NO_CUDA_ARGS[@]}"
}

for ds in "${DATASETS[@]}"; do
  # A: 动态topk图 + GAT_LSTM + AEF + AOF
  run_variant  "A" "dynamic_topk" "--use-aef" "--use-aof" \
    "动态Topk图 + GAT_LSTM + AEF + AOF"

  # B: 路径图 + GAT_LSTM + AEF + AOF
  run_variant  "B" "path" "--use-aef" "--use-aof" \
    "路径图 + GAT_LSTM + AEF + AOF"

  # C: 动态topk图 + GAT_LSTM + AEF
  run_variant  "C" "dynamic_topk" "--use-aef" "--disable-aof" \
    "动态Topk图 + GAT_LSTM + AEF"

  # D: 动态topk图 + GAT_LSTM + AOF
  run_variant  "D" "dynamic_topk" "--disable-aef" "--use-aof" \
    "动态Topk图 + GAT_LSTM + AOF"

  # E: 原始 GAT + LSTM + FC（路径图，无AEF/AOF）
  run_variant  "E" "path" "--disable-aef" "--disable-aof" \
    "原始 GAT + LSTM + Fully Connected Network"
done

echo "全部消融实验已完成。"
