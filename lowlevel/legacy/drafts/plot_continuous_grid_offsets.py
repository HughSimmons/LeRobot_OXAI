#!/usr/bin/env python3
"""Plot grasp and placement offset histograms from a continuous grid sweep."""

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
        description="Create grasp/place offset histograms for continuous grid JSONs."
    )
    parser.add_argument("grid_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def component_bins(values_mm: np.ndarray) -> np.ndarray:
    minimum = float(np.min(values_mm))
    maximum = float(np.max(values_mm))
    if np.isclose(minimum, maximum):
        return np.linspace(minimum - 0.5, maximum + 0.5, 11)
    padding = max(0.1, 0.15 * (maximum - minimum))
    return np.linspace(minimum - padding, maximum + padding, 11)


def main() -> int:
    args = parse_args()
    grid_dir = args.grid_dir.expanduser().resolve()
    lookup_paths = sorted(grid_dir.glob("d4_r*_c*/*.json"))
    if not lookup_paths:
        raise FileNotFoundError(f"No d4_r*_c* lookup JSONs under {grid_dir}")

    grasp_offsets = []
    place_offsets = []
    labels = []
    for lookup_path in lookup_paths:
        lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
        grasp_offsets.append(lookup["source_grasp_offset"])
        place_offsets.append(lookup["selected_place_offset"])
        labels.append(lookup_path.parent.name)

    grasp_mm = np.array(grasp_offsets, dtype=float) * 1000.0
    place_mm = np.array(place_offsets, dtype=float) * 1000.0
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey="row")
    component_names = ("X", "Y", "Z")
    groups = (("Pickup grasp offset", grasp_mm), ("Placement offset", place_mm))

    for row, (group_name, values) in enumerate(groups):
        for column, component_name in enumerate(component_names):
            axis = axes[row, column]
            component = values[:, column]
            axis.hist(
                component,
                bins=component_bins(component),
                color="#e86f20" if row else "#2468a2",
                edgecolor="white",
            )
            axis.axvline(float(np.mean(component)), color="black", linewidth=1.2)
            axis.set_title(f"{group_name}: {component_name}")
            axis.set_xlabel("offset (mm)")
            if column == 0:
                axis.set_ylabel("count")
            axis.text(
                0.98,
                0.95,
                f"mean {np.mean(component):.3f} mm\nstd {np.std(component):.3f} mm\nn={len(component)}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
            )

    fig.suptitle(f"Continuous grid offsets: {grid_dir.name}", y=0.99)
    fig.tight_layout()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else grid_dir / "offset_histograms.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved histogram: {output_path}")
    print(f"Samples: {', '.join(labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

