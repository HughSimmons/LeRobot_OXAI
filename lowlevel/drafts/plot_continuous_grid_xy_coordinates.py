#!/usr/bin/env python3
"""Plot commanded pickup XY and final placed XY distributions for a grid sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create initial/final XY histograms from continuous grid lookup JSONs."
    )
    parser.add_argument("grid_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def bins_for_pair(initial: np.ndarray, final: np.ndarray) -> np.ndarray:
    values = np.concatenate((initial, final))
    minimum, maximum = float(np.min(values)), float(np.max(values))
    padding = max(1.0, 0.12 * (maximum - minimum))
    return np.linspace(minimum - padding, maximum + padding, 15)


def main() -> int:
    args = parse_args()
    grid_dir = args.grid_dir.expanduser().resolve()
    lookup_paths = sorted(grid_dir.glob("d4_r*_c*/*.json"))
    if not lookup_paths:
        raise FileNotFoundError(f"No d4_r*_c* lookup JSONs under {grid_dir}")

    initial_xy = []
    final_xy = []
    for lookup_path in lookup_paths:
        lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
        initial_xy.append((lookup["from"]["x"], lookup["from"]["y"]))
        final_xy.append(lookup["metrics"]["final_position"][:2])

    initial_mm = np.array(initial_xy, dtype=float) * 1000.0
    final_mm = np.array(final_xy, dtype=float) * 1000.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis_index, component_name in enumerate(("X", "Y")):
        axis = axes[axis_index]
        bins = bins_for_pair(initial_mm[:, axis_index], final_mm[:, axis_index])
        axis.hist(
            initial_mm[:, axis_index],
            bins=bins,
            alpha=0.65,
            label="initial pickup coordinate",
            color="#2468a2",
            edgecolor="white",
        )
        axis.hist(
            final_mm[:, axis_index],
            bins=bins,
            alpha=0.65,
            label="final placed coordinate",
            color="#e86f20",
            edgecolor="white",
        )
        axis.set_title(f"{component_name} coordinate distribution")
        axis.set_xlabel("world coordinate (mm)")
        axis.set_ylabel("count")
        axis.legend(loc="upper left")
        axis.text(
            0.98,
            0.95,
            (
                f"initial: {np.mean(initial_mm[:, axis_index]):.2f} +/- "
                f"{np.std(initial_mm[:, axis_index]):.2f} mm\n"
                f"final: {np.mean(final_mm[:, axis_index]):.2f} +/- "
                f"{np.std(final_mm[:, axis_index]):.2f} mm\n"
                f"n={len(initial_mm)}"
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.8"},
        )

    fig.suptitle(f"Initial pickup and final placement XY: {grid_dir.name}", y=0.99)
    fig.tight_layout()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else grid_dir / "initial_final_xy_histograms.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved histogram: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

