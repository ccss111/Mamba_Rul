import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from dataset.preprocessing import CmapssDataSet, gen_sequence, load_dataset


def _build_last_window_dataset(group_test, y_test, sequence_len, max_rul):
	windows = []
	true_rul = []
	for engine_id in sorted(group_test.groups):
		features = group_test.get_group(engine_id).iloc[:, 2:]
		if len(features) < sequence_len:
			raise ValueError(
				f"发动机 {int(engine_id)} 只有 {len(features)} 个周期，"
				f"少于 sequence-len={sequence_len}"
			)
		windows.append(np.asarray(list(gen_sequence(features, sequence_len))[-1]))
		true_rul.append(float(y_test.iloc[int(engine_id) - 1, 0]) * max_rul)

	dataset = CmapssDataSet(
		np.asarray(windows, dtype=np.float32),
		np.asarray(true_rul, dtype=np.float32).reshape(-1, 1) / max_rul,
	)
	return dataset, np.asarray(true_rul, dtype=np.float32)


def _predict_interval(model, x_batch):
	point, outputs = model(x_batch)
	if isinstance(outputs, dict) and "lower" in outputs and "upper" in outputs:
		calibration_q = float(getattr(model, "cqr_calibration_q", 0.0))
		return point, outputs["lower"] - calibration_q, outputs["upper"] + calibration_q
	return point, point, point


def _predict(model, dataset, max_rul, device, batch_size):
	loader = DataLoader(dataset, batch_size=max(1, batch_size), shuffle=False)
	points, lowers, uppers = [], [], []
	model.eval()
	with torch.no_grad():
		for x_batch, _ in loader:
			point, lower, upper = _predict_interval(model, x_batch.to(device))
			points.append(point.detach().cpu().reshape(-1).numpy())
			lowers.append(lower.detach().cpu().reshape(-1).numpy())
			uppers.append(upper.detach().cpu().reshape(-1).numpy())
	return tuple(np.concatenate(values) * max_rul for values in (points, lowers, uppers))


def _save_scatter_plot(path, sub_dataset, true_rul, pred_rul, pred_lower, pred_upper):
	unit_numbers = np.arange(1, len(true_rul) + 1)
	lower = np.minimum(pred_lower, pred_upper)
	upper = np.maximum(pred_lower, pred_upper)
	fig, ax = plt.subplots(figsize=(8.5, 5.2))
	ax.errorbar(
		unit_numbers,
		pred_rul,
		yerr=[pred_rul - lower, upper - pred_rul],
		fmt="o",
		markersize=2.8,
		capsize=1.5,
		elinewidth=0.8,
		color="#2f6fbd",
		ecolor="#6f9bd1",
		label="estimates",
	)
	ax.scatter(unit_numbers, true_rul, s=10, color="#d94b4b", label="true", zorder=3)
	ax.set_xlabel("Unit Number")
	ax.set_ylabel("RUL (cycles)")
	ax.set_title(f"{sub_dataset} test-set RUL predictions")
	ax.set_xlim(0, len(unit_numbers) + 1)
	ax.set_ylim(bottom=0)
	ax.grid(True, alpha=0.25)
	ax.legend(loc="lower left", frameon=True, fontsize=8)
	fig.tight_layout()
	fig.savefig(path, dpi=300)
	plt.close(fig)


def main():
	parser = argparse.ArgumentParser(description="绘制测试集中每个 Unit 的 RUL 散点图")
	parser.add_argument("--sub-dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
	parser.add_argument("--model-path", required=True, help="模型文件路径")
	parser.add_argument("--sequence-len", type=int, default=30)
	parser.add_argument("--smooth-rate", type=int, default=30)
	parser.add_argument("--max-rul", type=int, default=125)
	parser.add_argument("--eval-batch-size", type=int, default=32)
	parser.add_argument("--dataset-root", default=os.path.join(PROJECT_ROOT, "CMAPSSData"))
	parser.add_argument("--output", default="", help="输出图片路径")
	parser.add_argument("--no-cuda", action="store_true")
	args = parser.parse_args()

	if not os.path.isfile(args.model_path):
		raise FileNotFoundError(f"指定模型不存在: {args.model_path}")
	device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
	_, group_test, y_test = load_dataset(
		args.dataset_root + os.sep,
		args.sub_dataset,
		args.max_rul,
		args.sequence_len,
		True,
		args.smooth_rate,
	)
	dataset, true_rul = _build_last_window_dataset(
		group_test, y_test, args.sequence_len, args.max_rul,
	)
	model = torch.load(args.model_path, map_location=device)
	model.to(device)
	pred_rul, pred_lower, pred_upper = _predict(
		model, dataset, args.max_rul, device, args.eval_batch_size,
	)

	output_path = args.output or os.path.join(
		os.path.dirname(os.path.abspath(__file__)),
		f"{args.sub_dataset}_scatter_prediction.png",
	)
	os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
	_save_scatter_plot(output_path, args.sub_dataset, true_rul, pred_rul, pred_lower, pred_upper)
	print(f"model_path:{args.model_path}")
	print(f"dataset:{args.sub_dataset}")
	print(f"unit_count:{len(true_rul)}")
	print(f"scatter_plot:{output_path}")


if __name__ == "__main__":
	main()
