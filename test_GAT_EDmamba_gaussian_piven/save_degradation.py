import argparse
import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MultipleLocator
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from dataset.preprocessing import CmapssDataSet, gen_sequence, load_dataset

#python test_GAT_EDmamba/save_degradation.py --model-path /Mamba_Rul/logs/FD001_GAT_EDmamba_JointRULLoss_Gaussian/seed2/best_model_FD001_0911_023730_seed2.pkl  --engine-source test --engine-id 100 --all-engines-length

def _resolve_model_path(model_path, trials_dir, sub_dataset):
	if model_path:
		if os.path.exists(model_path):
			return model_path
		raise FileNotFoundError(f"指定模型不存在: {model_path}")

	candidates = [
		os.path.join(trials_dir, f"model_{sub_dataset}.pkl"),
		*glob.glob(os.path.join(trials_dir, f"model_{sub_dataset}_*.pkl")),
		*glob.glob(os.path.join(trials_dir, f"best_model_{sub_dataset}_*.pkl")),
	]
	candidates = [path for path in candidates if os.path.exists(path)]
	if candidates:
		return max(candidates, key=os.path.getmtime)
	raise FileNotFoundError(
		f"在 {trials_dir} 下找不到 {sub_dataset} 对应模型，请使用 --model-path 显式指定。"
	)


def _model_code_and_seed(model_path, sub_dataset):
	seed_dir = os.path.basename(os.path.dirname(model_path))
	model_dir = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
	prefix = f"{sub_dataset}_"
	model_code = model_dir[len(prefix):] if model_dir.startswith(prefix) else model_dir
	seed = seed_dir[4:] if seed_dir.startswith("seed") else seed_dir
	return model_code, seed


def _make_engine_dataset(group, engine_id, seq_length, source, max_rul, y_test=None):
	if engine_id not in group.groups:
		raise ValueError(f"{source} 集不存在发动机 {engine_id}，可选范围为 1-{len(group)}")

	engine = group.get_group(engine_id)
	if source == "train":
		features = engine.iloc[:, 2:-1]
		labels = engine.iloc[:, -1].to_numpy(dtype=np.float32)
	else:
		features = engine.iloc[:, 2:]
		# 测试集分组键可能因预处理变成浮点数，位置索引必须使用整数。
		engine_index = int(engine_id) - 1
		hidden_rul = float(y_test.iloc[engine_index, 0]) * max_rul
		labels = np.arange(
			len(features) - 1 + hidden_rul, hidden_rul - 1, -1, dtype=np.float32
		)
		labels = np.clip(labels, 0.0, max_rul) / max_rul

	if len(features) < seq_length:
		raise ValueError(
			f"发动机 {engine_id} 只有 {len(features)} 个周期，少于 sequence-len={seq_length}"
		)
	windows = np.asarray(list(gen_sequence(features, seq_length)), dtype=np.float32)
	window_labels = labels[seq_length - 1:]
	dataset = CmapssDataSet(windows, window_labels.reshape(-1, 1))
	cycles = np.arange(seq_length, len(features) + 1, dtype=np.int32)
	return DataLoader(dataset, batch_size=32, shuffle=False), cycles, window_labels * max_rul


def _predict_interval(model, x_batch):
	point, outputs = model(x_batch)
	if getattr(model, "uncertainty_loss", "piven") == "gaussian" and isinstance(outputs, dict):
		z_value = float(getattr(model, "gaussian_z", 1.96))
		sigma = outputs["sigma"]
		return point, point - z_value * sigma, point + z_value * sigma
	if isinstance(outputs, dict) and "lower" in outputs and "upper" in outputs:
		return point, outputs["lower"], outputs["upper"]
	# 兼容旧版只输出点预测的模型。
	return point, point, point


def _collect_predictions(model, loader, max_rul, device, eval_batch_size):
	prediction_loader = DataLoader(loader.dataset, batch_size=eval_batch_size, shuffle=False)
	points, lowers, uppers = [], [], []
	model.eval()
	with torch.no_grad():
		for x_batch, _ in prediction_loader:
			point, lower, upper = _predict_interval(model, x_batch.to(device))
			points.append(point.detach().cpu().reshape(-1))
			lowers.append(lower.detach().cpu().reshape(-1))
			uppers.append(upper.detach().cpu().reshape(-1))

	pred_rul = torch.cat(points).numpy() * max_rul
	pred_lower = torch.cat(lowers).numpy() * max_rul
	pred_upper = torch.cat(uppers).numpy() * max_rul
	return pred_rul, np.clip(pred_lower, 0.0, max_rul), np.clip(pred_upper, 0.0, max_rul)


def _average_interval_length(pred_lower, pred_upper):
	lower = np.minimum(np.asarray(pred_lower), np.asarray(pred_upper))
	upper = np.maximum(np.asarray(pred_lower), np.asarray(pred_upper))
	return float(np.mean(upper - lower))


def _collect_dataset_interval_metrics(model, group, seq_length, source, max_rul,
								  y_test, device, eval_batch_size):
	lower_bounds, upper_bounds = [], []
	for engine_id in sorted(group.groups):
		loader, _, _ = _make_engine_dataset(
			group, engine_id, seq_length, source, max_rul, y_test
		)
		_, pred_lower, pred_upper = _collect_predictions(
			model, loader, max_rul, device, eval_batch_size
		)
		lower_bounds.extend(pred_lower)
		upper_bounds.extend(pred_upper)
	return _average_interval_length(lower_bounds, upper_bounds)


def _interval_metrics(true_rul, pred_lower, pred_upper, target_picp=0.95):
	true_rul = np.asarray(true_rul, dtype=np.float64).reshape(-1)
	lower = np.minimum(pred_lower, pred_upper)
	upper = np.maximum(pred_lower, pred_upper)
	covered = (true_rul >= lower) & (true_rul <= upper)
	picp = float(np.mean(covered))
	data_range = float(np.max(true_rul) - np.min(true_rul))
	pinaw = float(np.mean(upper - lower) / data_range) if data_range > 0 else 0.0
	if picp >= target_picp:
		cwc = pinaw
	else:
		cwc = pinaw * (1.0 + np.exp(-50.0 * (picp - target_picp)))
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
		raise ValueError("cannot calculate AIS for an empty engine")
	if not 0.0 < alpha < 1.0:
		raise ValueError("AIS alpha must be between 0 and 1")
	score = -2.0 * alpha * (upper - lower)
	score -= 4.0 * np.where(true_rul < lower, lower - true_rul, 0.0)
	score -= 4.0 * np.where(true_rul > upper, true_rul - upper, 0.0)
	return float(np.mean(score))


def _save_metrics(path, rows):
	with open(path, "w", newline="", encoding="utf-8") as file:
		writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
		writer.writeheader()
		writer.writerows(rows)


def _save_degradation_plot(path, sub_dataset, engine_id, source, cycles, true_rul,
						   pred_rul, pred_lower, pred_upper):
	plot_cycles = cycles - cycles[0]
	fig, ax = plt.subplots(figsize=(8.5, 5.2))
	ax.fill_between(plot_cycles, pred_lower, pred_upper, color="#2a66f0", alpha=0.18,
					label="Learned prediction interval")
	ax.plot(plot_cycles, true_rul, color="#35a853", linewidth=1.7, label="True RUL")
	ax.plot(plot_cycles, pred_rul, color="#2a66f0", linewidth=1.5, label="Predicted RUL")
	ax.plot(plot_cycles, pred_lower, color="#ff5a52", linewidth=0.9, alpha=0.8,
			label="Lower bound")
	ax.plot(plot_cycles, pred_upper, color="#174ea6", linewidth=0.9, alpha=0.8,
			label="Upper bound")
	ax.set_xlabel("Forecast cycles")
	ax.set_ylabel("Remaining useful life (cycles)")
	ax.set_title(f"{sub_dataset} {source} engine {engine_id} degradation curve")
	ax.set_xlim(0, plot_cycles[-1])
	ax.set_ylim(0, max(1.0, float(np.max(np.maximum(true_rul, pred_upper))) * 1.08))
	ax.xaxis.set_minor_locator(MultipleLocator(5))
	ax.yaxis.set_minor_locator(MultipleLocator(5))
	ax.grid(True, which="major", alpha=0.3)
	ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.45)
	ax.legend(loc="upper right", fontsize=8)
	fig.tight_layout()
	fig.savefig(path, format="svg")
	plt.close(fig)


def main():
	parser = argparse.ArgumentParser(description="绘制模型学习的 RUL 退化轨迹与预测区间")
	parser.add_argument("--sub-dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
	parser.add_argument("--engine-source", choices=["train", "test"], default="train")
	parser.add_argument("--engine-id", type=int, default=1)
	parser.add_argument("--sequence-len", type=int, default=30)
	parser.add_argument("--smooth-rate", type=int, default=30)
	parser.add_argument("--max-rul", type=int, default=125)
	parser.add_argument("--dataset-root", default=os.path.join(PROJECT_ROOT, "CMAPSSData"))
	parser.add_argument("--model-path", default="")
	parser.add_argument("--eval-batch-size", type=int, default=32)
	parser.add_argument("--no-cuda", action="store_true")
	parser.add_argument("--degradation-svg-path", default="")
	parser.add_argument("--metrics-csv", default="")
	parser.add_argument("--ais-alpha", type=float, default=0.05,
					help="AIS区间错误率alpha，默认0.05（对应95%%区间）")
	parser.add_argument(
		"--all-engines-length", action="store_true",
		help="计算整个子数据集所有发动机、所有窗口的平均预测区间长度",
	)
	args = parser.parse_args()

	device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
	model_path = _resolve_model_path(
		args.model_path, os.path.join(PROJECT_ROOT, "trials"), args.sub_dataset
	)
	group_train, group_test, y_test = load_dataset(
		args.dataset_root + os.sep, args.sub_dataset, args.max_rul, args.sequence_len,
		True, args.smooth_rate
	)
	group = group_train if args.engine_source == "train" else group_test
	loader, cycles, true_rul = _make_engine_dataset(
		group, args.engine_id, args.sequence_len, args.engine_source, args.max_rul, y_test
	)

	model = torch.load(model_path, map_location=device)
	model.to(device)
	if args.all_engines_length:
		average_interval_length = _collect_dataset_interval_metrics(
			model, group, args.sequence_len, args.engine_source, args.max_rul,
			y_test, device, max(1, args.eval_batch_size)
		)
		print(f"model_path:{model_path}")
		print(f"dataset:{args.sub_dataset}_{args.engine_source}")
		print(f"engine_count:{len(group)}")
		print(f"average_interval_length:{average_interval_length:.6f} cycles")

	pred_rul, pred_lower, pred_upper = _collect_predictions(
		model, loader, args.max_rul, device, max(1, args.eval_batch_size)
	)
	metrics = _interval_metrics(true_rul, pred_lower, pred_upper)
	metrics["AIS"] = _average_interval_score(
		true_rul, pred_lower, pred_upper, alpha=args.ais_alpha
	)
	model_code, seed = _model_code_and_seed(model_path, args.sub_dataset)

	output_dir = os.path.join(
		PROJECT_ROOT, "figure", "save_engine_degradation",
		f"{args.sub_dataset}-engine{args.engine_id}",
	)
	os.makedirs(output_dir, exist_ok=True)
	output_path = os.path.join(
		output_dir,
		os.path.basename(args.degradation_svg_path) if args.degradation_svg_path else
		f"{args.sub_dataset}_{args.engine_source}_engine{args.engine_id}_"
		f"{model_code}_seed{seed}_degradation.svg",
	)
	_save_degradation_plot(
		output_path, args.sub_dataset, args.engine_id, args.engine_source,
		cycles, true_rul, pred_rul, pred_lower, pred_upper
	)
	metrics_path = args.metrics_csv or os.path.join(
		output_dir,
		f"{args.sub_dataset}_{args.engine_source}_engine{args.engine_id}_"
		f"{model_code}_seed{seed}_interval_metrics.csv",
	)
	_save_metrics(metrics_path, [{
		"model_path": model_path,
		"sub_dataset": args.sub_dataset,
		"engine_source": args.engine_source,
		"engine_id": args.engine_id,
		**metrics,
	}])
	print(f"model_path:{model_path}")
	print(f"degradation_svg:{output_path}")
	print("interval_metrics: PICP={PICP:.6f}, PINAW={PINAW:.6f}, CWC={CWC:.6f}, AIS={AIS:.6f}".format(**metrics))
	print("interval_metrics: PICP={PICP:.6f}, PINAW={PINAW:.6f}, CWC={CWC:.6f}".format(**metrics))
	print(f"metrics_csv:{metrics_path}")


if __name__ == "__main__":
	main()
