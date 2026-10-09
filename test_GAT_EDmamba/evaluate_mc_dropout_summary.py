import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch import nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import get_dataloader
from utils.functions import compute_s_score


def _enable_dropout_only(module):
    if isinstance(module, nn.Dropout):
        module.train()


def _mc_predict(model, x_batch, mc_samples, interval_confidence):
    model.eval()
    model.apply(_enable_dropout_only)
    predictions = []
    with torch.no_grad():
        for _ in range(mc_samples):
            prediction, _ = model(x_batch)
            predictions.append(prediction.detach().float())
    samples = torch.stack(predictions, dim=0)
    tail_probability = (1.0 - interval_confidence) / 2.0
    quantiles = torch.tensor(
        [tail_probability, 1.0 - tail_probability],
        device=samples.device,
        dtype=samples.dtype,
    )
    lower, upper = torch.quantile(samples, quantiles, dim=0)
    return (
        samples.mean(dim=0).reshape(-1),
        lower.reshape(-1),
        upper.reshape(-1),
    )


def _interval_metrics(true_rul, lower, upper, target_picp=0.95, ais_alpha=0.05):
    true_rul = np.asarray(true_rul, dtype=np.float64).reshape(-1)
    lower = np.minimum(lower, upper)
    upper = np.maximum(lower, upper)
    covered = (true_rul >= lower) & (true_rul <= upper)
    picp = float(np.mean(covered))
    data_range = float(np.max(true_rul) - np.min(true_rul))
    pinaw = float(np.mean(upper - lower) / data_range) if data_range > 0 else 0.0
    cwc = pinaw if picp >= target_picp else pinaw * (
        1.0 + np.exp(-50.0 * (picp - target_picp))
    )
    score = -2.0 * ais_alpha * (upper - lower)
    score -= 4.0 * np.where(true_rul < lower, lower - true_rul, 0.0)
    score -= 4.0 * np.where(true_rul > upper, true_rul - upper, 0.0)
    return {
        "PICP": picp,
        "PINAW": pinaw,
        "CWC": float(cwc),
        "AIS": float(np.mean(score)),
    }


def _evaluate_model(model_path, loader, num_test_windows, max_rul, device,
                    eval_batch_size, mc_samples, interval_confidence, target_picp,
                    ais_alpha):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    dataset = loader.dataset
    predictions = []
    lower_bounds = []
    upper_bounds = []
    started = time.perf_counter()
    for start in range(0, len(dataset.x_data), eval_batch_size):
        x_batch = dataset.x_data[start:start + eval_batch_size].to(device)
        mean, lower, upper = _mc_predict(
            model, x_batch, mc_samples, interval_confidence)
        mean = mean * max_rul
        predictions.append(mean.cpu())
        lower_bounds.append((lower * max_rul).clamp(0.0, max_rul).cpu())
        upper_bounds.append((upper * max_rul).clamp(0.0, max_rul).cpu())
    elapsed = time.perf_counter() - started

    pred = torch.cat(predictions)
    lower = torch.cat(lower_bounds).numpy()
    upper = torch.cat(upper_bounds).numpy()
    true = dataset.y_data.reshape(-1).cpu().numpy() * max_rul
    pred_engines = torch.split(pred, num_test_windows)
    true_engines = torch.split(torch.from_numpy(true), num_test_windows)
    true_mean = torch.tensor([values.mean() for values in true_engines])
    pred_mean = torch.tensor([values.mean() for values in pred_engines]).floor()
    true_mean, order = true_mean.sort(descending=True)
    pred_mean = torch.index_select(pred_mean, 0, order)
    rmse = float(mean_squared_error(true_mean.numpy(), pred_mean.numpy(), squared=False))
    score = float(compute_s_score(true_mean, pred_mean).item())
    metrics = _interval_metrics(true, lower, upper, target_picp, ais_alpha)
    return {"rmse_final": rmse, "score": score, "runtime_seconds": elapsed, **metrics}


def _summarize(rows):
    summary = {"FD": rows[0]["dataset"]}
    for source, output in [
        ("rmse_final", "rmse_final_min"),
        ("score", "score_min"),
        ("PINAW", "PINAW_min"),
        ("CWC", "CWC_min"),
    ]:
        summary[output] = min(row[source] for row in rows)
    for source, output in [
        ("rmse_final", "rmse_final_mean"),
        ("score", "score_mean"),
        ("runtime_seconds", "runtime_seconds_mean"),
        ("PICP", "PICP_mean"),
        ("PINAW", "PINAW_mean"),
        ("CWC", "CWC_mean"),
        ("AIS", "AIS_mean"),
    ]:
        summary[output] = float(np.mean([row[source] for row in rows]))
    summary["runtime_minutes_mean"] = summary["runtime_seconds_mean"] / 60.0
    summary["PICP_max"] = max(row["PICP"] for row in rows)
    summary["AIS_max"] = max(row["AIS"] for row in rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description="FD001-FD004 Monte Carlo Dropout 汇总评估")
    parser.add_argument("--mc-samples", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--interval-confidence", type=float, default=0.95)
    parser.add_argument("--sequence-len", type=int, default=30)
    parser.add_argument("--max-rul", type=int, default=125)
    parser.add_argument("--smooth-rate", type=int, default=30)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "figure", "mc_dropout"))
    args = parser.parse_args()
    if args.mc_samples < 2:
        raise ValueError("--mc-samples must be at least 2")
    if not 0.0 < args.interval_confidence < 1.0:
        raise ValueError("--interval-confidence must be between 0 and 1")

    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    detail_rows = []
    summary_rows = []
    for dataset_name in ("FD001", "FD002", "FD003", "FD004"):
        model_paths = sorted(glob.glob(os.path.join(
            PROJECT_ROOT, "logs", f"{dataset_name}_GAT_EDmamba_dropout_0.1",
            "seed*", "best_model_*.pkl")))
        if not model_paths:
            raise FileNotFoundError(f"找不到 {dataset_name} 的 best_model checkpoint")
        _, _, test_loader, _, num_test_windows, _, _ = get_dataloader(
            dir_path=os.path.join(PROJECT_ROOT, "CMAPSSData") + os.sep,
            sub_dataset=dataset_name, max_rul=args.max_rul,
            seq_length=args.sequence_len, batch_size=128,
            use_exponential_smoothing=True, smooth_rate=args.smooth_rate)
        dataset_rows = []
        for model_path in model_paths:
            metrics = _evaluate_model(
                model_path, test_loader, num_test_windows, args.max_rul,
                device, max(1, args.eval_batch_size), args.mc_samples,
                args.interval_confidence, 0.95, 0.05)
            row = {"dataset": dataset_name, "seed": os.path.basename(os.path.dirname(model_path)),
                   "model_path": model_path, **metrics}
            detail_rows.append(row)
            dataset_rows.append(row)
            print(dataset_name, row["seed"], "rmse_final=%.2f score=%.2f runtime=%.2fs" %
                  (row["rmse_final"], row["score"], row["runtime_seconds"]))
        summary_rows.append(_summarize(dataset_rows))

    detail_path = os.path.join(args.output_dir, "mc_dropout_details.csv")
    summary_path = os.path.join(args.output_dir, "mc_dropout_summary.csv")
    with open(detail_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(row)
    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: f"{value:.2f}" if isinstance(value, float) else value
                             for key, value in row.items()})
    print(f"summary_csv:{summary_path}")
    print(f"detail_csv:{detail_path}")
    for row in summary_rows:
        print(", ".join(f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}"
                        for key, value in row.items()))


if __name__ == "__main__":
    main()