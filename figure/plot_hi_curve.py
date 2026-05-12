import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_hi(time_cycles: np.ndarray, tcp: float, eol_cycle: float) -> np.ndarray:
    """Compute piece-wise HI from the provided formula and clamp to [0, 1]."""
    rul = eol_cycle - time_cycles
    hi = 1.0 - np.maximum((tcp - rul) / tcp, 0.0)
    return np.clip(hi, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Plot HI curve: flat then linear degradation.")
    parser.add_argument("--tcp", type=float, default=125.0, help="Transition length Tcp.")
    parser.add_argument("--eol", type=float, default=190.0, help="Failure cycle EOL.")
    parser.add_argument("--x-max", type=float, default=200.0, help="Max x-axis cycle.")
    parser.add_argument("--step", type=float, default=1.0, help="Cycle sampling step.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(project_root / "logs" / "figures" / "hi_curve_tcp125_eol190.svg"),
        help="Output image path.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="auto",
        choices=["auto", "svg", "png", "pdf"],
        help="Image format. 'auto' infers from output file suffix.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    parser.add_argument("--show", action="store_true", help="Show figure window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.tcp <= 0:
        raise ValueError("--tcp must be positive.")
    if args.step <= 0:
        raise ValueError("--step must be positive.")

    x_max = max(args.x_max, args.eol)
    cycles = np.arange(0.0, x_max + args.step, args.step)
    hi = compute_hi(cycles, tcp=args.tcp, eol_cycle=args.eol)

    fig, ax = plt.subplots(figsize=(6.3, 4.7))
    ax.plot(cycles, hi, color="#00bfe7", linewidth=2.0)
    ax.set_xlabel("Operating cycles", fontsize=13)
    ax.set_ylabel("Health index", fontsize=13)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.02, 1.05)
    ax.tick_params(labelsize=11)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix_format = output_path.suffix.lower().lstrip(".")
    if args.output_format == "auto":
        save_format = suffix_format if suffix_format else "svg"
    else:
        save_format = args.output_format

    save_kwargs = {"format": save_format}
    if save_format not in {"svg", "pdf"}:
        save_kwargs["dpi"] = args.dpi
    fig.savefig(output_path, **save_kwargs)

    transition_cycle = args.eol - args.tcp
    print(f"Saved figure to: {output_path} (format={save_format})")
    print(f"Transition starts near cycle: {transition_cycle:.2f}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()