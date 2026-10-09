import csv
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs"
SUMMARY_PATH = ROOT / "logs" / "requested_interval_model_summary.csv"
SEEDS = (2, 17, 27, 30, 33, 51, 62, 80, 88, 97)

EXPERIMENTS = (
    ("CQR_95%", "GAT_EDmamba_JointRULLoss_CQR_95%", "test_GAT_EDmamba_CQR/test_model.py"),
    ("CQR_95%_warm_up", "GAT_EDmamba_JointRULLoss_CQR_95%_warm_up", "test_GAT_EDmamba_CQR/test_model.py"),
    ("Gaussian", "GAT_EDmamba_JointRULLoss_Gaussian", "test_GAT_EDmamba_gaussian_piven/test_model.py"),
    ("Gaussian_warm-up_xi", "GAT_EDmamba_JointRULLoss_Gaussian_warm-up_xi", "test_GAT_EDmamba_gaussian_piven/test_model.py"),
)


def latest_checkpoint(directory, seed):
    candidates = sorted((directory / f"seed{seed}").glob("best_model_*.pkl"), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(f"No best checkpoint for seed {seed}: {directory}")
    return candidates[-1]


def matching_log(checkpoint):
    seed_dir = checkpoint.parent
    stamp = checkpoint.stem[len("best_model_"):]
    suffix_match = re.search(r"(\d{4}_\d{6}_seed\d+)$", stamp)
    suffix = suffix_match.group(1) if suffix_match else stamp
    candidates = sorted(seed_dir.glob(f"*{suffix}*.log"), key=os.path.getmtime)
    if not candidates:
        seed_match = re.search(r"seed\d+$", stamp)
        seed = seed_match.group(0) if seed_match else ""
        candidates = sorted(seed_dir.glob(f"*{seed}*.log"), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(f"No training log matching {checkpoint}")
    return candidates[-1]


def training_seconds(log_path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"total_seconds:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not matches:
        raise ValueError(f"No [Train Time] total_seconds in {log_path}")
    return float(matches[-1])


def parse_test_output(output, checkpoint):
    match = re.search(
        r"rmse_final:([0-9.eE+-]+), score:([0-9.eE+-]+).*?"
        r"interval_metrics: PICP=([0-9.eE+-]+), PINAW=([0-9.eE+-]+), "
        r"CWC=([0-9.eE+-]+), AIS=([0-9.eE+-]+)",
        output,
        re.S,
    )
    if not match:
        raise ValueError(f"Unable to parse test output for {checkpoint}:\n{output}")
    rmse, score, picp, pinaw, cwc, ais = map(float, match.groups())
    return {
        "rmse": rmse,
        "score": score,
        "picp": picp,
        "pinaw": pinaw,
        "cwc": cwc,
        "ais": ais,
    }


def run_test(dataset, smooth_rate, script, checkpoint):
    command = [
        sys.executable,
        str(ROOT / script),
        "--sub-dataset", dataset,
        "--smooth-rate", str(smooth_rate),
        "--sequence-len", "30",
        "--feature-num", "14",
        "--dataset-root", str(ROOT / "CMAPSSData") + os.sep,
        "--max-rul", "125",
        "--batch-size", "128",
        "--eval-batch-size", "128",
        "--model-path", str(checkpoint),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return result.stdout, parse_test_output(result.stdout, checkpoint)


def append_result(log_path, command_output, checkpoint):
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("\n[Requested Interval Evaluation]\n")
        stream.write(f"checkpoint: {checkpoint}\n")
        stream.write(command_output)
        if not command_output.endswith("\n"):
            stream.write("\n")


def rounded(value):
    return f"{value:.2f}"


def main():
    rows = []
    for dataset_number in range(1, 5):
        dataset = f"FD{dataset_number:03d}"
        smooth_rate = 30 if dataset_number in (1, 3) else 40
        for label, directory_name, script in EXPERIMENTS:
            values = []
            for seed in SEEDS:
                checkpoint = latest_checkpoint(LOG_ROOT / f"{dataset}_{directory_name}", seed)
                log_path = matching_log(checkpoint)
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                marker = f"checkpoint: {checkpoint}"
                if marker in log_text and "[Requested Interval Evaluation]" in log_text:
                    output = log_text.rsplit("[Requested Interval Evaluation]", 1)[1]
                    metrics = parse_test_output(output, checkpoint)
                    print(f"reused {dataset} {label} seed={seed}", flush=True)
                else:
                    output, metrics = run_test(dataset, smooth_rate, script, checkpoint)
                    append_result(log_path, output, checkpoint)
                    print(f"completed {dataset} {label} seed={seed}", flush=True)
                metrics["runtime"] = training_seconds(log_path)
                values.append(metrics)

            rows.append({
                "数据集": f"{dataset}_{label}",
                "RMSE 最小值": rounded(min(item["rmse"] for item in values)),
                "RMSE 均值": rounded(sum(item["rmse"] for item in values) / len(values)),
                "Score 最小值": rounded(min(item["score"] for item in values)),
                "Score 均值": rounded(sum(item["score"] for item in values) / len(values)),
                "平均运行时间（秒）": rounded(sum(item["runtime"] for item in values) / len(values)),
                "平均运行时间（分钟）": rounded(sum(item["runtime"] for item in values) / len(values) / 60),
                "PICP 最大值": rounded(max(item["picp"] for item in values)),
                "PICP 均值": rounded(sum(item["picp"] for item in values) / len(values)),
                "PINAW 最小值": rounded(min(item["pinaw"] for item in values)),
                "PINAW 均值": rounded(sum(item["pinaw"] for item in values) / len(values)),
                "CWC 最小值": rounded(min(item["cwc"] for item in values)),
                "CWC 均值": rounded(sum(item["cwc"] for item in values) / len(values)),
                "AIS 最大值": rounded(max(item["ais"] for item in values)),
                "AIS 均值": rounded(sum(item["ais"] for item in values) / len(values)),
            })

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()