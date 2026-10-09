
# 运行时加上PYTHONPATH=/CMAPSS-release
import argparse
import csv
import glob
import os
import sys
from datetime import datetime as datetime_class

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import *
from dataset import *
from model import *

#python test_GAT_EDmamba/test_model.py   --sub-dataset FD001   --model-path   --model-code   


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


def _enable_dropout_only(module):
    if isinstance(module, nn.Dropout):
        module.train()


def _mc_dropout_predict(model, x_batch, mc_samples, device, use_amp: bool = False):
    mc_samples = max(1, int(mc_samples))
    model.eval()
    model.apply(_enable_dropout_only)

    predictions = []
    autocast_ctx = (torch.cuda.amp.autocast(dtype=torch.float16)
                    if use_amp and getattr(device, "type", str(device)) == "cuda"
                    else None)

    with torch.no_grad():
        for _ in range(mc_samples):
            if autocast_ctx is None:
                y_pred, _ = model.forward(x_batch)
            else:
                with autocast_ctx:
                    y_pred, _ = model.forward(x_batch)
            predictions.append(y_pred.detach().float())

    stacked = torch.stack(predictions, dim=0)
    mean_pred = stacked.mean(dim=0)
    variance_pred = stacked.var(dim=0, unbiased=False)
    return mean_pred, variance_pred, stacked


def _explicit_interval_predict(model, x_batch):
    point, outputs = model.forward(x_batch)
    if getattr(model, "uncertainty_loss", "piven") == "gaussian" and isinstance(outputs, dict):
        z_value = float(getattr(model, "gaussian_z", 1.96))
        sigma = outputs["sigma"]
        return point, point - z_value * sigma, point + z_value * sigma
    if isinstance(outputs, dict) and "lower" in outputs and "upper" in outputs:
        return point, outputs["lower"], outputs["upper"]
    return point, point, point



def _collect_per_engine_predictions(
    model,
    test_loader_last,
    max_rul,
    device,
    eval_batch_size: int = 32,
    use_amp: bool = False,
    mc_samples: int = 100,
    interval_alpha: float = 5.0,
):
    model.eval()
    dataset = getattr(test_loader_last, "dataset", None)
    if dataset is None or not hasattr(dataset, "x_data") or not hasattr(dataset, "y_data"):
        # Fallback to legacy behavior
        with torch.no_grad():
            x_test, y_test = next(iter(test_loader_last))
            x_test = x_test.to(device)
            y_pred, y_lower, y_upper = _explicit_interval_predict(model, x_test)
        true_rul = (y_test.reshape(-1).cpu().numpy()) * max_rul
        pred_rul = (y_pred.reshape(-1).detach().cpu().numpy()) * max_rul
        pred_var = np.zeros_like(pred_rul)
        pred_lower = y_lower.reshape(-1).detach().cpu().numpy() * max_rul
        pred_upper = y_upper.reshape(-1).detach().cpu().numpy() * max_rul
        return true_rul, pred_rul, pred_var, pred_lower, pred_upper

    if eval_batch_size is None or int(eval_batch_size) <= 0:
        eval_batch_size = 32
    eval_batch_size = int(eval_batch_size)

    x_all = dataset.x_data
    y_all = dataset.y_data.reshape(-1)

    preds = []
    lower_bounds = []
    upper_bounds = []
    with torch.no_grad():
        for start in range(0, len(x_all), eval_batch_size):
            x_batch = x_all[start:start + eval_batch_size].to(device)
            y_mean, y_lower, y_upper = _explicit_interval_predict(model, x_batch)
            preds.append(y_mean.detach().cpu())
            lower_bounds.append(y_lower.detach().cpu())
            upper_bounds.append(y_upper.detach().cpu())

    y_pred_all = torch.cat(preds, dim=0).reshape(-1)
    y_lower_all = torch.cat(lower_bounds, dim=0).reshape(-1)
    y_upper_all = torch.cat(upper_bounds, dim=0).reshape(-1)
    true_rul = (y_all.cpu().numpy()) * max_rul
    pred_rul = (y_pred_all.cpu().numpy()) * max_rul
    pred_var = np.zeros_like(pred_rul)

    pred_lower = y_lower_all.numpy() * max_rul
    pred_upper = y_upper_all.numpy() * max_rul

    return true_rul, pred_rul, pred_var, pred_lower, pred_upper


def _collect_interval_predictions(
    model,
    test_loader,
    max_rul,
    device,
    eval_batch_size=32,
    use_amp=False,
    mc_samples=100,
    interval_alpha=5.0,
):
    """Collect interval predictions for every window in the test set."""
    dataset = getattr(test_loader, "dataset", None)
    if dataset is None or not hasattr(dataset, "x_data") or not hasattr(dataset, "y_data"):
        raise ValueError("test_loader must have a dataset with x_data/y_data")

    eval_batch_size = max(1, int(eval_batch_size))
    predictions = []
    lower_bounds = []
    upper_bounds = []
    with torch.no_grad():
        for start in range(0, len(dataset.x_data), eval_batch_size):
            x_batch = dataset.x_data[start:start + eval_batch_size].to(device)
            y_mean, y_lower, y_upper = _explicit_interval_predict(model, x_batch)
            predictions.append(y_mean.detach().cpu())
            lower_bounds.append(y_lower.detach().cpu())
            upper_bounds.append(y_upper.detach().cpu())

    pred_rul = torch.cat(predictions).reshape(-1).numpy() * max_rul
    pred_lower = torch.cat(lower_bounds).reshape(-1).numpy() * max_rul
    pred_upper = torch.cat(upper_bounds).reshape(-1).numpy() * max_rul
    true_rul = dataset.y_data.reshape(-1).cpu().numpy() * max_rul

    return true_rul, pred_rul, pred_lower, pred_upper


def _interval_metrics(true_rul, pred_lower, pred_upper, target_picp=0.95,
                      cwc_eta=50.0, cwc_gamma=1.0):
    """Return PICP, PINAW and CWC over the complete test set."""
    true_rul = np.asarray(true_rul, dtype=np.float64).reshape(-1)
    pred_lower = np.asarray(pred_lower, dtype=np.float64).reshape(-1)
    pred_upper = np.asarray(pred_upper, dtype=np.float64).reshape(-1)
    if not (len(true_rul) == len(pred_lower) == len(pred_upper)):
        raise ValueError("true values and interval bounds must have the same length")
    if len(true_rul) == 0:
        raise ValueError("cannot calculate interval metrics for an empty test set")

    lower = np.minimum(pred_lower, pred_upper)
    upper = np.maximum(pred_lower, pred_upper)
    picp = float(np.mean((true_rul >= lower) & (true_rul <= upper)))
    data_range = float(np.max(true_rul) - np.min(true_rul))
    pinaw = float(np.mean(upper - lower) / data_range) if data_range > 0 else 0.0
    target_picp = float(np.clip(target_picp, 0.0, 1.0))
    if picp >= target_picp:
        cwc = pinaw
    else:
        cwc = pinaw * (1.0 + cwc_gamma * np.exp(-cwc_eta * (picp - target_picp)))
    return {"PICP": picp, "PINAW": pinaw, "CWC": float(cwc)}


def _average_interval_score(true_rul, pred_lower, pred_upper, alpha=0.05):
    true_rul = np.asarray(true_rul, dtype=np.float64).reshape(-1)
    lower = np.minimum(np.asarray(pred_lower, dtype=np.float64),
                       np.asarray(pred_upper, dtype=np.float64)).reshape(-1)
    upper = np.maximum(np.asarray(pred_lower, dtype=np.float64),
                       np.asarray(pred_upper, dtype=np.float64)).reshape(-1)
    if not (len(true_rul) == len(lower) == len(upper)):
        raise ValueError("true values and interval bounds must have the same length")
    if len(true_rul) == 0:
        raise ValueError("cannot calculate AIS for an empty test set")
    if not 0.0 < alpha < 1.0:
        raise ValueError("AIS alpha must be between 0 and 1")
    score = -2.0 * alpha * (upper - lower)
    score -= 4.0 * np.where(true_rul < lower, lower - true_rul, 0.0)
    score -= 4.0 * np.where(true_rul > upper, true_rul - upper, 0.0)
    return float(np.mean(score))

def _save_predictions_csv(csv_path, true_rul, pred_rul, pred_var=None, pred_lower=None, pred_upper=None):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if pred_var is None:
        pred_var = np.zeros_like(pred_rul)
    if pred_lower is None:
        pred_lower = pred_rul
    if pred_upper is None:
        pred_upper = pred_rul
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["engine_id", "true_rul", "pred_mean_rul", "pred_variance", "pred_lower_rul", "pred_upper_rul", "error"])
        for idx, (y_true, y_pred, y_var, y_low, y_up) in enumerate(zip(true_rul, pred_rul, pred_var, pred_lower, pred_upper), start=1):
            error = float(y_pred) - float(y_true)
            writer.writerow([
                idx,
                f"{float(y_true):.6f}",
                f"{float(y_pred):.6f}",
                f"{float(y_var):.6f}",
                f"{float(y_low):.6f}",
                f"{float(y_up):.6f}",
                f"{error:.6f}",
            ])


def _save_predictions_svg(svg_path, sub_dataset, true_rul, pred_rul, pred_lower=None, pred_upper=None):
    os.makedirs(os.path.dirname(svg_path), exist_ok=True)

    true_rul = np.asarray(true_rul, dtype=np.float32)
    pred_rul = np.asarray(pred_rul, dtype=np.float32)
    if pred_lower is None:
        pred_lower = pred_rul
    if pred_upper is None:
        pred_upper = pred_rul
    pred_lower = np.asarray(pred_lower, dtype=np.float32)
    pred_upper = np.asarray(pred_upper, dtype=np.float32)
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
    ax.fill_between(
        engine_idx,
        pred_lower,
        pred_upper,
        color="#89d68c",
        alpha=0.18,
        label="Learned prediction interval",
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
    parser.add_argument('--model-path', type=str, nargs='+', default=None,
                        help='一个或多个模型路径；未指定时自动匹配最新模型')
    parser.add_argument('--model-code', type=str, nargs='+', default=None,
                        help='与模型路径按位置对应的模型代号，例如 --model-code dropout01 dropout02')
    parser.add_argument('--save-pred-csv', action='store_true', default=False,
                        help='将每台发动机的真实/预测RUL导出为CSV')
    parser.add_argument('--pred-csv-path', type=str, default='',
                        help='预测CSV输出路径；多模型时建议留空以按模型代号分别输出')
    parser.add_argument('--save-pred-svg', action='store_true', default=False,
                        help='将每台发动机的真实/预测RUL绘制为SVG')
    parser.add_argument('--pred-svg-path', type=str, default='',
                        help='预测SVG输出路径；多模型时建议留空以按模型代号分别输出')
    
    parser.add_argument('--eval-batch-size', type=int, default=32,
                        help='评估/推理时的batch size（用于避免CUDA OOM）')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='推理时启用AMP(fp16)以节省显存（可能造成轻微数值差异）')
    parser.add_argument('--mc-samples', type=int, default=100,
                        help='Monte Carlo Dropout 前向传播次数')
    parser.add_argument('--interval-alpha', type=float, default=5.0,
                        help='正态区间的双侧显著性水平a，95%%置信区间对应5')
    parser.add_argument('--target-picp', type=float, default=0.95,
                        help='CWC目标覆盖率，默认0.95')
    parser.add_argument('--ais-alpha', type=float, default=0.05,
                        help='AIS区间错误率alpha，默认0.05（对应95%%区间）')
    parser.add_argument('--cwc-eta', type=float, default=50.0,
                        help='CWC覆盖率惩罚强度')
    parser.add_argument('--cwc-gamma', type=float, default=1.0,
                        help='CWC覆盖率惩罚系数')
    parser.add_argument('--metrics-csv', type=str, default='',
                        help='区间指标汇总CSV路径，默认输出到本次测试目录')
    args = parser.parse_args()

    device = torch.device('cuda' if (not args.no_cuda and torch.cuda.is_available()) else 'cpu')
    trials_dir = os.path.join(parent_dir, 'trials')
    model_paths = args.model_path or [_resolve_model_path('', trials_dir, args.sub_dataset)]
    if args.model_code is None:
        model_codes = [os.path.splitext(os.path.basename(path))[0] for path in model_paths]
    elif len(args.model_code) != len(model_paths):
        parser.error('--model-code 的数量必须与 --model-path 的数量相同')
    else:
        model_codes = args.model_code
    model_codes = [os.path.basename(code) for code in model_codes]

    train_loader, valid_loader, test_loader, test_loader_last, \
        num_test_windows, train_visualize, engine_id = get_dataloader(
            dir_path=args.dataset_root,
            sub_dataset=args.sub_dataset,
            max_rul=args.max_rul,
            seq_length=args.sequence_len,
            batch_size=args.batch_size,
            use_exponential_smoothing=args.use_exponential_smoothing,
            smooth_rate=args.smooth_rate)

    metric_rows = []
    for model_path, model_code in zip(model_paths, model_codes):
        model = torch.load(model_path, map_location=device)
        model.to(device)
        rmse_final, score = evaluate(
            model, num_test_windows, test_loader, args.max_rul,
            device=device, eval_batch_size=args.eval_batch_size,
            use_amp=args.amp, mc_samples=args.mc_samples,
            use_mc_dropout=True,
        )
        true_all, _, lower_all, upper_all = _collect_interval_predictions(
            model, test_loader, args.max_rul, device,
            eval_batch_size=args.eval_batch_size, use_amp=args.amp,
            mc_samples=args.mc_samples, interval_alpha=args.interval_alpha,
        )
        metrics = _interval_metrics(
            true_all, lower_all, upper_all,
            target_picp=args.target_picp, cwc_eta=args.cwc_eta,
            cwc_gamma=args.cwc_gamma,
        )
        metrics['AIS'] = _average_interval_score(
            true_all, lower_all, upper_all, alpha=args.ais_alpha)
        row = {'model_path': model_path, 'model_code': model_code,
               'rmse': rmse_final, 'score': score, **metrics}
        metric_rows.append(row)
        print('model_path:{}'.format(model_path))
        print('rmse_final:{}, score:{}'.format(rmse_final, score))
        print('interval_metrics: PICP={PICP:.6f}, PINAW={PINAW:.6f}, CWC={CWC:.6f}, AIS={AIS:.6f}'.format(**metrics))

        if args.save_pred_csv or args.save_pred_svg:
            output_dir = os.path.join(
                parent_dir,
                'figure',
                'predictions',
                f'{args.sub_dataset}-{datetime_class.now().strftime("%m%d-%H%M")}',
            )
            os.makedirs(output_dir, exist_ok=True)
            true_rul, pred_rul, pred_var, pred_lower, pred_upper = _collect_per_engine_predictions(
                model=model, test_loader_last=test_loader_last,
                max_rul=args.max_rul, device=device,
                eval_batch_size=args.eval_batch_size, use_amp=args.amp,
                mc_samples=args.mc_samples, interval_alpha=args.interval_alpha,
            )

        if args.save_pred_csv:
            pred_csv_path = os.path.join(
                output_dir, f'{args.sub_dataset}_{model_code}_pred.csv')
            if args.pred_csv_path and len(model_paths) == 1:
                pred_csv_path = os.path.join(output_dir, os.path.basename(args.pred_csv_path))
            _save_predictions_csv(pred_csv_path, true_rul, pred_rul, pred_var, pred_lower, pred_upper)
            print('prediction_csv:{}'.format(pred_csv_path))

        if args.save_pred_svg:
            pred_svg_path = os.path.join(
                output_dir, f'{args.sub_dataset}_{model_code}_pred.svg')
            if args.pred_svg_path and len(model_paths) == 1:
                pred_svg_path = os.path.join(output_dir, os.path.basename(args.pred_svg_path))
            _save_predictions_svg(pred_svg_path, args.sub_dataset, true_rul, pred_rul, pred_lower, pred_upper)
            print('prediction_svg:{}'.format(pred_svg_path))

    print('test_results_written_to_log_only:True')
