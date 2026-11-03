#!/usr/bin/env python3
"""Plot a bullseye map from run_rook_offset_bullseye.py results.

Left: start offsets on the source square, coloured by how far from the target
centre the rook landed. Right: the landing points on the target square, drawn
over bullseye rings with the same colour bands.

Example:
  python lowlevel/drafts/plot_rook_offset_bullseye.py lowlevel/rook_kiri_xy_lookup/<run_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Rectangle
import numpy as np


SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e4e3df"
# Status palette: good / warning / serious / critical, plus a neutral for failures.
BANDS = [
    (2.0, "#0ca30c", "< 2 mm"),
    (5.0, "#fab219", "2–5 mm"),
    (10.0, "#ec835a", "5–10 mm"),
    (np.inf, "#d03b3b", "> 10 mm"),
]
FAILED_COLOR = "#3d3c39"
FAILED_LABEL = "failed: M/✕ missed grasp, F/▼ fell or dropped"
SQUARE_HALF_MM = 20.0
TOLERANCE_MM = 5.0


def band_color(sample: dict) -> str:
    if not sample.get("landed_upright") or sample.get("xy_error_mm") is None:
        return FAILED_COLOR
    for limit, color, _ in BANDS:
        if sample["xy_error_mm"] < limit:
            return color
    return BANDS[-1][1]


def failure_mark(sample: dict) -> str:
    return "M" if not sample.get("pickup_success") else "F"


def within_tolerance(sample: dict) -> bool:
    return bool(sample.get("landed_upright")) and (sample.get("xy_error_mm") or np.inf) < TOLERANCE_MM


def tolerance_radius_mm(samples: list[dict]) -> float:
    """Largest start-offset radius inside which every sample landed within TOLERANCE_MM."""
    radii = sorted((np.hypot(s["dx_mm"], s["dy_mm"]), within_tolerance(s)) for s in samples)
    best = 0.0
    for radius, ok in radii:
        if not ok:
            return best
        best = radius
    return best


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_aspect("equal")


def draw_offset_map(ax, samples: list[dict], config: dict) -> None:
    style_axes(ax)
    src = config["from_square"]
    ax.add_patch(Rectangle((-SQUARE_HALF_MM, -SQUARE_HALF_MM), 2 * SQUARE_HALF_MM, 2 * SQUARE_HALF_MM,
                           fill=False, edgecolor=MUTED, linewidth=1, linestyle=(0, (4, 3))))
    if config.get("grid_size"):
        step = 2 * config["inset_m"] * 1000.0 / max(1, config["grid_size"] - 1)
        cell = step - 0.5  # leaves a surface gap between cells
        for s in samples:
            color = band_color(s)
            ax.add_patch(Rectangle((s["dx_mm"] - cell / 2, s["dy_mm"] - cell / 2), cell, cell,
                                   facecolor=color, edgecolor=SURFACE, linewidth=1))
            if color == FAILED_COLOR:
                ax.text(s["dx_mm"], s["dy_mm"], failure_mark(s), ha="center", va="center",
                        color=SURFACE, fontsize=7.5, fontweight="bold")
    else:
        for s in samples:
            color = band_color(s)
            failed = color == FAILED_COLOR
            ax.scatter(s["dx_mm"], s["dy_mm"], s=44, color=color, edgecolors=SURFACE, linewidths=1.2,
                       marker=("X" if failure_mark(s) == "M" else "v") if failed else "o", zorder=3)
    radius = tolerance_radius_mm(samples)
    if radius > 0:
        ax.add_patch(Circle((0, 0), radius, fill=False, edgecolor=INK, linewidth=1.5, zorder=4))
        ax.annotate(f"all within {TOLERANCE_MM:g} mm\ninside r = {radius:.1f} mm",
                    xy=(radius * 0.707, radius * 0.707), xytext=(SQUARE_HALF_MM - 1, SQUARE_HALF_MM + 1),
                    ha="right", va="bottom", fontsize=8.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.8))
    ax.plot(0, 0, marker="+", color=INK, markersize=10, zorder=5)
    lim = SQUARE_HALF_MM + 3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim + 4)
    ax.set_xlabel(f"start offset along file (mm), + = toward h", color=INK_2, fontsize=9)
    ax.set_ylabel(f"start offset along rank (mm), + = toward 8", color=INK_2, fontsize=9)
    ax.set_title(f"Where the rook started on {src}", color=INK, fontsize=11, loc="left")


def draw_bullseye(ax, samples: list[dict], config: dict) -> int:
    style_axes(ax)
    dst = config["to_square"]
    for limit, color, _ in reversed(BANDS[:-1]):
        ax.add_patch(Circle((0, 0), limit, facecolor=color, alpha=0.16, edgecolor=color, linewidth=1))
    for limit, _, _ in BANDS[:-1]:
        ax.text(limit * 0.707 + 0.3, -limit * 0.707 - 0.3, f"{limit:g} mm", color=INK_2, fontsize=8,
                ha="left", va="top")
    ax.add_patch(Rectangle((-SQUARE_HALF_MM, -SQUARE_HALF_MM), 2 * SQUARE_HALF_MM, 2 * SQUARE_HALF_MM,
                           fill=False, edgecolor=MUTED, linewidth=1, linestyle=(0, (4, 3))))
    off_view = 0
    for s in samples:
        err = s.get("final_error_xy_mm")
        if not s.get("landed_upright") or err is None or any(v is None for v in err):
            off_view += 1
            continue
        if max(abs(err[0]), abs(err[1])) > SQUARE_HALF_MM + 2:
            off_view += 1
            continue
        ax.scatter(err[0], err[1], s=40, color=band_color(s), edgecolors=SURFACE, linewidths=1.2, zorder=3)
    ax.plot(0, 0, marker="+", color=INK, markersize=10, zorder=5)
    lim = SQUARE_HALF_MM + 3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim + 4)
    ax.set_xlabel("landing error along file (mm)", color=INK_2, fontsize=9)
    ax.set_ylabel("landing error along rank (mm)", color=INK_2, fontsize=9)
    ax.set_title(f"Where it landed on {dst} (centre = +)", color=INK, fontsize=11, loc="left")
    if off_view:
        ax.text(-lim + 0.5, lim + 3.5, f"{off_view} not shown: failed or landed off the square",
                color=INK_2, fontsize=8.5, va="top")
    return off_view


def describe(config: dict) -> str:
    if config["mode"] == "known":
        mode = "robot KNOWS the true position"
    else:
        mode = f"robot is BLIND to the offset (always reaches for {config['from_square']} centre)"
    corr = config["placement_corrections"]
    corr_text = "no placement corrections" if corr == 0 else f"{corr} sim-feedback placement corrections"
    if config.get("random_samples"):
        spread = f"{config['random_samples']} Gaussian offsets, σ = {config['noise_sigma_m'] * 1000:g} mm"
    else:
        span = config["inset_m"] * 1000
        spread = f"{config['grid_size']}×{config['grid_size']} grid, ±{span:g} mm"
    return f"{mode} · {corr_text} · {spread}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot a rook offset bullseye map.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads((args.run_dir / "results.json").read_text(encoding="utf-8"))
    config, samples = data["config"], [s for s in data["samples"] if "error" not in s]
    n = len(samples)
    counts = {label: 0 for _, _, label in BANDS}
    counts[FAILED_LABEL] = 0
    for s in samples:
        color = band_color(s)
        label = FAILED_LABEL if color == FAILED_COLOR else next(l for _, c, l in BANDS if c == color)
        counts[label] += 1
    ok = sum(within_tolerance(s) for s in samples)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 6.9), facecolor=SURFACE)
    draw_offset_map(left, samples, config)
    draw_bullseye(right, samples, config)
    fig.suptitle(
        f"Rook {config['from_square']} → {config['to_square']}: {ok}/{n} land within {TOLERANCE_MM:g} mm of centre",
        x=0.06, y=0.975, ha="left", color=INK, fontsize=14, fontweight="bold",
    )
    fig.text(0.06, 0.925, describe(config), ha="left", color=INK_2, fontsize=10)
    handles = [Patch(facecolor=c, edgecolor=c, label=f"{l}  ({counts[l]})") for _, c, l in BANDS]
    handles.append(Patch(facecolor=FAILED_COLOR, edgecolor=FAILED_COLOR,
                         label=f"{FAILED_LABEL}  ({counts[FAILED_LABEL]})"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9.5,
               labelcolor=INK_2, title="distance from target centre", title_fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.17, wspace=0.18)

    output = args.output or args.run_dir / "bullseye.png"
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    print(f"Saved {output}  |  tolerance radius {tolerance_radius_mm(samples):.1f} mm  |  {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
