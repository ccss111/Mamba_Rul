
# 运行时加上PYTHONPATH=/CMAPSS-release
import argparse
import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import *
from dataset import *
from model import *


def _resolve_model_path(model_path, trials_dir, sub_dataset):
    if model_path:
        if os.path.exists(model_path):
            return model_path
        raise FileNotFoundError(f"指定模型不存在: {model_path}")

    legacy_path = os.path.join(trials_dir, f"model_{sub_dataset}.pkl")
    if os.path.exists(legacy_path):
        return legacy_path

    candidates = glob.glob(os.path.join(trials_dir, f"model_{sub_dataset}_*.pkl"))
    if candidates:
        return max(candidates, key=os.path.getmtime)

    best_candidates = glob.glob(os.path.join(trials_dir, f"best_model_{sub_dataset}_*.pkl"))
    if best_candidates:
        return max(best_candidates, key=os.path.getmtime)

    raise FileNotFoundError(
        f"在 {trials_dir} 下找不到 {sub_dataset} 对应模型，请使用 --model-path 显式指定。"
    )


def _collect_per_engine_predictions(model, test_loader_last, max_rul, device, eval_batch_size: int = 32, use_amp: bool = False):
    model.eval()
    dataset = getattr(test_loader_last, "dataset", None)
    if dataset is None or not hasattr(dataset, "x_data") or not hasattr(dataset, "y_data"):
        # Fallback to legacy behavior
        with torch.no_grad():
            x_test, y_test = next(iter(test_loader_last))
            x_test = x_test.to(device)
            y_pred, _ = model.forward(x_test)
        true_rul = (y_test.reshape(-1).cpu().numpy()) * max_rul
        pred_rul = (y_pred.reshape(-1).detach().cpu().numpy()) * max_rul
        return true_rul, pred_rul

    if eval_batch_size is None or int(eval_batch_size) <= 0:
        eval_batch_size = 32
    eval_batch_size = int(eval_batch_size)

    x_all = dataset.x_data
    y_all = dataset.y_data.reshape(-1)

    preds = []
    with torch.no_grad():
        autocast_ctx = (torch.cuda.amp.autocast(dtype=torch.float16)
                        if use_amp and getattr(device, "type", str(device)) == "cuda"
                        else None)
        for start in range(0, len(x_all), eval_batch_size):
            x_batch = x_all[start:start + eval_batch_size].to(device)
            if autocast_ctx is None:
                y_pred, _ = model.forward(x_batch)
            else:
                with autocast_ctx:
                    y_pred, _ = model.forward(x_batch)
            preds.append(y_pred.detach().float().cpu())

    y_pred_all = torch.cat(preds, dim=0).reshape(-1)
    true_rul = (y_all.cpu().numpy()) * max_rul
    pred_rul = (y_pred_all.cpu().numpy()) * max_rul
    return true_rul, pred_rul


def _save_predictions_csv(csv_path, true_rul, pred_rul):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["engine_id", "true_rul", "pred_rul", "error"])
        for idx, (y_true, y_pred) in enumerate(zip(true_rul, pred_rul), start=1):
            error = float(y_pred) - float(y_true)
            writer.writerow([
                idx,
                f"{float(y_true):.6f}",
                f"{float(y_pred):.6f}",
                f"{error:.6f}",
            ])


def _save_predictions_svg(svg_path, sub_dataset, true_rul, pred_rul):
    os.makedirs(os.path.dirname(svg_path), exist_ok=True)

    true_rul = np.asarray(true_rul, dtype=np.float32)
    pred_rul = np.asarray(pred_rul, dtype=np.float32)
    errors = pred_rul - true_rul
    engine_idx = np.arange(1, len(true_rul) + 1)

    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    ax.bar(engine_idx, errors, color="#5fd6e8", width=0.8, alpha=0.9, label="Error (Pred-True)")
    ax.plot(
        engine_idx,
        true_rul,
        linestyle="--",
        color="#dc8a8a",
        marker="s",
        markerfacecolor="none",
        markersize=4,
        linewidth=1.0,
        label="True RUL",
    )
    ax.plot(
        engine_idx,
        pred_rul,
        linestyle="--",
        color="#89d68c",
        marker="^",
        markerfacecolor="none",
        markersize=4,
        linewidth=1.0,
        label="Predicted RUL",
    )

    ax.axhline(0.0, color="#8c8c8c", linewidth=0.8)
    ax.set_xlabel("Engine Id")
    ax.set_ylabel("RUL(Cycle)")
    ax.set_title(f"{sub_dataset} Prediction", fontsize=11)
    ax.grid(axis="y", alpha=0.25)

    y_min = min(float(np.min(errors)), float(np.min(true_rul)), float(np.min(pred_rul)))
    y_max = max(float(np.max(errors)), float(np.max(true_rul)), float(np.max(pred_rul)))
    ax.set_ylim(min(-30.0, y_min - 5.0), y_max + 8.0)
    ax.set_xlim(0.5, len(engine_idx) + 0.5)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(svg_path, format="svg")
    plt.close(fig)

if __name__ == '__main__':
    current_dir = os.getcwd()  # Get the current directory
    parent_dir = os.path.dirname(current_dir)  # Get the upper-level directory
    parent_dir = PROJECT_ROOT
    parser = argparse.ArgumentParser(description='Cmapss Dataset With Pytorch')
    # To evaluate the trained models on different sub-datasets,
    # please change the following two options
    parser.add_argument('--sub-dataset', type=str, default='FD002', help='FD001/2/3/4')
    parser.add_argument('--smooth-rate', type=int, default=30)
    # Below is the default settings
    parser.add_argument('--use-exponential-smoothing', default=True)
    parser.add_argument('--sequence-len', type=int, default=30)
    parser.add_argument('--feature-num', type=int, default=14)
    parser.add_argument('--dataset-root', type=str,
                        default=os.path.join(parent_dir, 'CMAPSSData') + '/', 
                        help='The dir of CMAPSS dataset1')
    parser.add_argument('--max-rul', type=int, default=125, help='piece-wise RUL')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--step-size', type=int, default=10, help='interval of learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='ratio of learning rate scheduler')
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=8, help='Early Stop Patience')
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--log-path', type=str, default='..\\_trials\\', help='The dir of logging path')
    parser.add_argument('--model-path', type=str, default='',
                        help='模型路径，默认自动匹配 model_<sub_dataset>_*.pkl 或 legacy 命名')
    parser.add_argument('--save-pred-csv', action='store_true', default=False,
                        help='将每台发动机的真实/预测RUL导出为CSV')
    parser.add_argument('--pred-csv-path', type=str, default='',
                        help='预测CSV输出路径，默认 logs/predictions/<sub_dataset>_<model_name>_pred.csv')
    parser.add_argument('--save-pred-svg', action='store_true', default=False,
                        help='将每台发动机的真实/预测RUL绘制为SVG')
    parser.add_argument('--pred-svg-path', type=str, default='',
                        help='预测SVG输出路径，默认 figure/<sub_dataset>_<model_name>_pred.svg')
    parser.add_argument('--eval-batch-size', type=int, default=32,
                        help='评估/推理时的batch size（用于避免CUDA OOM）')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='推理时启用AMP(fp16)以节省显存（可能造成轻微数值差异）')
    args = parser.parse_args()

    device = torch.device('cuda' if (not args.no_cuda and torch.cuda.is_available()) else 'cpu')
    trials_dir = os.path.join(parent_dir, 'trials')
    model_path = _resolve_model_path(args.model_path, trials_dir, args.sub_dataset)

    model = torch.load(model_path, map_location=device)
    model_type = type(model).__name__
    model.to(device)
    train_loader, valid_loader, test_loader, test_loader_last, \
        num_test_windows, train_visualize, engine_id = get_dataloader(
            dir_path=args.dataset_root,
            sub_dataset=args.sub_dataset,
            max_rul=args.max_rul,
            seq_length=args.sequence_len,
            batch_size=args.batch_size,
            use_exponential_smoothing=args.use_exponential_smoothing,
            smooth_rate=args.smooth_rate)

    rmse_final, score = evaluate(
            model,
            num_test_windows,
            test_loader,
            args.max_rul,
            device=device,
            eval_batch_size=args.eval_batch_size,
            use_amp=args.amp,
        )

    print('model_path:{}'.format(model_path))
    print('rmse_final:{}, score:{}'.format(rmse_final, score))

    if args.save_pred_csv or args.save_pred_svg:
        true_rul, pred_rul = _collect_per_engine_predictions(
            model=model,
            test_loader_last=test_loader_last,
            max_rul=args.max_rul,
            device=device,
            eval_batch_size=args.eval_batch_size,
            use_amp=args.amp,
        )

    if args.save_pred_csv:

        if args.pred_csv_path:
            pred_csv_path = args.pred_csv_path
        else:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            pred_csv_path = os.path.join(
                parent_dir,
                'logs',
                'predictions',
                f'{args.sub_dataset}_{model_name}_pred.csv',
            )

        _save_predictions_csv(pred_csv_path, true_rul, pred_rul)
        print('prediction_csv:{}'.format(pred_csv_path))

    if args.save_pred_svg:
        if args.pred_svg_path:
            pred_svg_path = args.pred_svg_path
        else:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            pred_svg_path = os.path.join(
                parent_dir,
                'figure',
                f'{args.sub_dataset}_{model_name}_pred.svg',
            )

        _save_predictions_svg(
            svg_path=pred_svg_path,
            sub_dataset=args.sub_dataset,
            true_rul=true_rul,
            pred_rul=pred_rul,
        )
        print('prediction_svg:{}'.format(pred_svg_path))
