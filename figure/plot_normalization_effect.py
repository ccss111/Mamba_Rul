import argparse
import os
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.preprocessing import exponential_smoothing


INDEX_NAMES = ["unit_nr", "time_cycles"]
SETTING_NAMES = ["setting_1", "setting_2", "setting_3"]
SENSOR_NAMES = [f"s_{i}" for i in range(1, 22)]
DROP_SENSORS = ["s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]
VALID_SENSORS = [sensor for sensor in SENSOR_NAMES if sensor not in DROP_SENSORS]
DROP_LABELS = SETTING_NAMES + DROP_SENSORS
COL_NAMES = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES
LINE_ALPHA = 0.65
FONT_AXIS_LABEL_SIZE = 14
FONT_TICK_SIZE = 12
FONT_LEGEND_SIZE = 10
FONT_LEGEND_TITLE_SIZE = 11
FONT_CAPTION_SIZE = 14


def parse_sensors(sensor_text: str) -> List[str]:
    if sensor_text.strip().lower() == "all":
        return VALID_SENSORS

    sensors = [item.strip() for item in sensor_text.split(",") if item.strip()]
    if not sensors:
        raise ValueError("At least one sensor must be provided.")
    return sensors


def load_split_frame(dataset_root: str, sub_dataset: str, split: str) -> pd.DataFrame:
    file_path = os.path.join(dataset_root, f"{split}_{sub_dataset}.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    frame = pd.read_csv(file_path, sep=r"\s+", header=None, names=COL_NAMES)
    frame.drop(labels=DROP_LABELS, axis=1, inplace=True)
    return frame


def build_before_after_frames(
    frame: pd.DataFrame, use_exponential_smoothing: bool, smooth_rate: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    title = frame.iloc[:, :2].reset_index(drop=True)

    if use_exponential_smoothing:
        sensor_values = exponential_smoothing(frame.values, smooth_rate)
        before_sensor = pd.DataFrame(sensor_values, columns=frame.columns[2:])
    else:
        before_sensor = frame.iloc[:, 2:].reset_index(drop=True)

    scaler = StandardScaler()
    after_sensor = pd.DataFrame(
        scaler.fit_transform(before_sensor.values), columns=before_sensor.columns
    )

    before_frame = pd.concat([title, before_sensor], axis=1)
    after_frame = pd.concat([title, after_sensor], axis=1)
    return before_frame, after_frame


def plot_engine_sensors(
    before_frame: pd.DataFrame,
    after_frame: pd.DataFrame,
    engine_id: int,
    sensors: List[str],
    output_path: str,
    output_format: str,
    layout: str,
) -> None:
    engine_before = before_frame[before_frame["unit_nr"] == engine_id]
    engine_after = after_frame[after_frame["unit_nr"] == engine_id]

    if engine_before.empty:
        raise ValueError(f"No data found for engine_id={engine_id}")

    missing = [sensor for sensor in sensors if sensor not in before_frame.columns]
    if missing:
        raise ValueError(f"Unknown sensors: {missing}")

    if layout == "single":
        fig, (ax_before, ax_after) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), sharex=True)
        color_map = plt.get_cmap("tab20")

        for i, sensor in enumerate(sensors):
            color = color_map(i % 20)
            ax_before.plot(
                engine_before["time_cycles"],
                engine_before[sensor],
                color=color,
                linewidth=1.0,
                alpha=LINE_ALPHA,
                linestyle="-",
            )
            ax_after.plot(
                engine_after["time_cycles"],
                engine_after[sensor],
                color=color,
                linewidth=1.0,
                alpha=LINE_ALPHA,
                linestyle="-",
                label=sensor,
            )

        ax_before.set_xlabel("time_cycles", fontsize=FONT_AXIS_LABEL_SIZE)
        ax_before.set_ylabel("sensor value", fontsize=FONT_AXIS_LABEL_SIZE)
        ax_before.tick_params(axis="both", labelsize=FONT_TICK_SIZE)
        ax_before.grid(alpha=0.3)

        ax_after.set_xlabel("time_cycles", fontsize=FONT_AXIS_LABEL_SIZE)
        ax_after.set_ylabel("sensor value", fontsize=FONT_AXIS_LABEL_SIZE)
        ax_after.tick_params(axis="both", labelsize=FONT_TICK_SIZE)
        ax_after.grid(alpha=0.3)
        ax_after.legend(
            title="sensor",
            ncol=2,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=FONT_LEGEND_SIZE,
            title_fontsize=FONT_LEGEND_TITLE_SIZE,
            frameon=True,
        )
        fig.tight_layout(rect=[0, 0.08, 0.86, 1])

        before_pos = ax_before.get_position()
        after_pos = ax_after.get_position()
        fig.text(
            x=(before_pos.x0 + before_pos.x1) / 2,
            y=0.045,
            s="(a)Before normalization",
            ha="center",
            va="center",
            fontsize=FONT_CAPTION_SIZE,
        )
        fig.text(
            x=(after_pos.x0 + after_pos.x1) / 2,
            y=0.045,
            s="(b)After normalization",
            ha="center",
            va="center",
            fontsize=FONT_CAPTION_SIZE,
        )
    else:
        fig, axes = plt.subplots(
            nrows=len(sensors),
            ncols=2,
            figsize=(12, max(3 * len(sensors), 4)),
            squeeze=False,
            sharex=False,
        )

        for i, sensor in enumerate(sensors):
            ax_before = axes[i, 0]
            ax_after = axes[i, 1]

            ax_before.plot(
                engine_before["time_cycles"],
                engine_before[sensor],
                color="#1f77b4",
                alpha=LINE_ALPHA,
                linestyle="-",
            )
            ax_before.set_xlabel("time_cycles", fontsize=FONT_AXIS_LABEL_SIZE)
            ax_before.set_ylabel(sensor, fontsize=FONT_AXIS_LABEL_SIZE)
            ax_before.tick_params(axis="both", labelsize=FONT_TICK_SIZE)
            ax_before.grid(alpha=0.3)

            ax_after.plot(
                engine_after["time_cycles"],
                engine_after[sensor],
                color="#d62728",
                alpha=LINE_ALPHA,
                linestyle="-",
            )
            ax_after.set_xlabel("time_cycles", fontsize=FONT_AXIS_LABEL_SIZE)
            ax_after.set_ylabel(sensor, fontsize=FONT_AXIS_LABEL_SIZE)
            ax_after.tick_params(axis="both", labelsize=FONT_TICK_SIZE)
            ax_after.grid(alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 1])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, format=output_format, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot sensor trajectories before and after StandardScaler normalization."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=os.path.join(PROJECT_ROOT, "CMAPSSData"),
        help="Path to CMAPSSData folder",
    )
    parser.add_argument("--sub-dataset", type=str, default="FD001", help="FD001/FD002/FD003/FD004")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--engine-id", type=int, default=1)
    parser.add_argument(
        "--sensors",
        type=str,
        default="all",
        help="Comma-separated sensor names, or 'all' for all 14 valid sensors",
    )
    parser.add_argument(
        "--use-exponential-smoothing",
        dest="use_exponential_smoothing",
        action="store_true",
        default=True,
        help="Apply exponential smoothing before normalization (same as training default).",
    )
    parser.add_argument(
        "--disable-exponential-smoothing",
        dest="use_exponential_smoothing",
        action="store_false",
        help="Disable smoothing and normalize raw sensor values directly.",
    )
    parser.add_argument("--smooth-rate", type=int, default=40)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "logs", "figures"),
        help="Directory where the figure will be saved",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="single",
        choices=["single", "per-sensor"],
        help="single: two panels (before and after) with all selected sensors; per-sensor: one row per sensor",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="svg",
        choices=["svg", "pdf", "eps"],
        help="Vector output format",
    )

    args = parser.parse_args()
    sensors = parse_sensors(args.sensors)

    frame = load_split_frame(args.dataset_root, args.sub_dataset, args.split)
    before_frame, after_frame = build_before_after_frames(
        frame, args.use_exponential_smoothing, args.smooth_rate
    )

    output_name = (
        f"norm_compare_{args.sub_dataset}_{args.split}_engine{args.engine_id}_{args.layout}.{args.output_format}"
    )
    output_path = os.path.join(args.output_dir, output_name)

    plot_engine_sensors(
        before_frame=before_frame,
        after_frame=after_frame,
        engine_id=args.engine_id,
        sensors=sensors,
        output_path=output_path,
        output_format=args.output_format,
        layout=args.layout,
    )

    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
