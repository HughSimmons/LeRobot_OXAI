#!/usr/bin/env python3
"""9x9 source x target matrix for the d3-f5 block (run_block_queue.sh + the earlier e4 sweep).

Two heatmaps on the same scale: within-5 mm count out of 150, before and after the median
place calibration. Rows = source square, columns = target square, both in board order
(rank 5 to 3, file d to f). Also writes the numbers as CSV/markdown next to the figure.

  python lowlevel/drafts/plot_block_matrix.py
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_rook_offset_bullseye import INK, INK_2, MUTED, SURFACE  # noqa: E402

LOOK = Path(__file__).resolve().parents[1] / "rook_kiri_xy_lookup"
OUT = LOOK / "2026-11-07_block_d3-f5" / "figures"
TAB = LOOK / "2026-11-07_block_d3-f5" / "tables"
SQUARES = "d5 e5 f5 d4 e4 f4 d3 e3 f3".split()
BLUE = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load() -> dict[tuple[str, str, str], dict]:
    data = {}
    for src in SQUARES:
        path = (LOOK / "2026-11-06_e4_neighbourhood/tables/FINAL_summary_before_vs_after.csv" if src == "e4"
                else TAB / f"from_{src}__summary_before_vs_after.csv")
        for r in csv.DictReader(path.open(encoding="utf-8")):
            data[(src, r["target"], "before" if r["condition"] == "before" else "after")] = r
    return data


def main() -> int:
    data = load()
    cmap = LinearSegmentedColormap.from_list("blue", BLUE)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6), facecolor=SURFACE)
    for ax, cond, title in zip(axes, ("before", "after"),
                               ("Before: 16 Oct place offset", "After: median place calibration per pair")):
        grid = [[int(data[(s, t, cond)]["within_5mm"]) for t in SQUARES] for s in SQUARES]
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=150)
        for i, s in enumerate(SQUARES):
            for j, t in enumerate(SQUARES):
                v = grid[i][j]
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                        color="white" if v >= 85 else INK)
        total = sum(map(sum, grid))
        ax.set_title(f"{title}\n{total}/{150 * 81} ({100 * total / (150 * 81):.0f}%) within 5 mm",
                     loc="left", fontsize=11, color=INK, pad=10)
        ax.set_xticks(range(9), SQUARES)
        ax.set_yticks(range(9), SQUARES)
        ax.tick_params(colors=INK_2, length=0, labelsize=9.5)
        ax.xaxis.tick_top()
        ax.set_xlabel("target square", color=MUTED, fontsize=9.5)
        ax.xaxis.set_label_position("top")
        ax.set_ylabel("source square", color=MUTED, fontsize=9.5)
        for k in (2.5, 5.5):  # separate ranks
            ax.axhline(k, color=SURFACE, linewidth=3)
            ax.axvline(k, color=SURFACE, linewidth=3)
        for side in ax.spines.values():
            side.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 150))
    cb = fig.colorbar(sm, cax=fig.add_axes([0.925, 0.12, 0.012, 0.62]))
    cb.set_label("landed upright within 5 mm of target centre (of 150)", color=INK_2, fontsize=9.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=8.5)
    fig.suptitle("Rook, every square of the d3–f5 block to every square: blind, 150 noisy starts (σ = 3.3 mm) per pair",
                 x=0.02, y=0.98, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.02, 0.025, "f targets use the pivot-first place. Sources on the f file are picked up in the tilted "
             "far pose (no pickup fix yet): 16–20% of their grasps miss. e4 row: earlier sweep (before = old far place).",
             fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.84, bottom=0.07, wspace=0.12)
    png = OUT / "ALL_81_pairs__within_5mm__before_vs_after.png"
    fig.savefig(png, dpi=150, facecolor=SURFACE)

    with open(TAB / "ALL_81_pairs__within_5mm__before_vs_after.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "target", "condition", "n", "within_5mm", "within_2mm", "missed", "fell", "median_err_mm", "run"])
        for s in SQUARES:
            for t in SQUARES:
                for c in ("before", "after"):
                    r = data[(s, t, c)]
                    w.writerow([s, t, c, r["n"], r["within_5mm"], r["within_2mm"], r["missed"], r["fell"],
                                f"{float(r['median_err_mm']):.2f}", r["run"]])
    lines = ["# d3–f5 block, every source → every target: within 5 mm of 150, before → after (median cal)", "",
             "| source \\ target | " + " | ".join(SQUARES) + " | total | missed grasp | fell (after) |",
             "|---" * 13 + "|"]
    for s in SQUARES:
        b = [int(data[(s, t, "before")]["within_5mm"]) for t in SQUARES]
        a = [int(data[(s, t, "after")]["within_5mm"]) for t in SQUARES]
        miss = sum(int(data[(s, t, "after")]["missed"]) for t in SQUARES)
        fell = sum(int(data[(s, t, "after")]["fell"]) for t in SQUARES)
        lines.append(f"| **{s}** | " + " | ".join(f"{x} → {y}" for x, y in zip(b, a))
                     + f" | {sum(b)} → **{sum(a)}** ({100 * sum(a) / 1350:.0f}%) | {miss} | {fell} |")
    (TAB / "ALL_81_pairs__within_5mm__before_vs_after.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {png} (+ .csv, .md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
