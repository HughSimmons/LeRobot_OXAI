#!/usr/bin/env python3
"""One-page before/after comparison of place-offset calibrations, per target.

Reads the CSV written by summarise_bullseye_conditions.py and draws two dot-plot
panels with one row per target: share of samples landing within 2 mm, and the
median landing error. Each condition is one marker; "before" is grey, so the
calibrations read as moves away from it. Only targets present in every
condition are drawn.

Example:
  python lowlevel/drafts/plot_before_after_summary.py \
      lowlevel/rook_kiri_xy_lookup/2026-11-06_e4_neighbourhood/tables/summary_before_vs_after_all_calibrations.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_rook_offset_bullseye import GRID, INK, INK_2, MUTED, SURFACE  # noqa: E402

# Before = neutral grey; calibrations take categorical slots 1-3 of the reference palette, in order.
COLORS = ["#898781", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "D", "^", "v", "P"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--title", default="e4 → neighbourhood, blind mode: does re-calibrating the place offset help?")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--source", default="e4", help="Source square: its own row is labelled pick & replace.")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    by_key = {(r["target"], r["condition"]): r for r in rows}
    targets = [t for t in dict.fromkeys(r["target"] for r in rows)
               if all((t, c) in by_key for c in conditions)]
    labels = [f"pick & replace ({t})" if t == args.source else f"→ {t}" for t in targets]

    panels = [
        ("within 2 mm of target centre (% of 150)", lambda r: 100.0 * int(r["within_2mm"]) / int(r["n"]), "{:.0f}%"),
        ("within 5 mm of target centre (% of 150)", lambda r: 100.0 * int(r["within_5mm"]) / int(r["n"]), "{:.0f}%"),
        ("median landing error of upright landings (mm)", lambda r: float(r["median_err_mm"]), "{:.2f}"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 0.62 * len(targets) + 2.6), facecolor=SURFACE, sharey=True)
    five = [100.0 * int(r["within_5mm"]) / int(r["n"]) for r in rows if r["target"] in targets]
    narrow_five = min(five) >= 70  # all targets good: zoom the 5 mm panel; otherwise show 0-100
    offsets = [(i - (len(conditions) - 1) / 2) * 0.16 for i in range(len(conditions))]
    for ax, (name, value, fmt) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, length=0, labelsize=9)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for y, t in enumerate(targets):
            vals = [value(by_key[(t, c)]) for c in conditions]
            ax.plot([min(vals), max(vals)], [y, y], color=GRID, linewidth=2, zorder=1)
            for i, (c, v) in enumerate(zip(conditions, vals)):
                ax.plot(v, y + offsets[i], marker=MARKERS[i], markersize=8, color=COLORS[i],
                        markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none", zorder=3)
        # Totals row: pooled over all drawn targets.
        y = len(targets) + 0.55
        for i, c in enumerate(conditions):
            rs = [by_key[(t, c)] for t in targets]
            if "median" in name:
                v = sorted(float(r["median_err_mm"]) for r in rs)[len(rs) // 2 - (len(rs) % 2 == 0):][:2]
                v = sum(v) / len(v)
            else:
                key = "within_2mm" if "2 mm" in name else "within_5mm"
                v = 100.0 * sum(int(r[key]) for r in rs) / sum(int(r["n"]) for r in rs)
            ax.plot(v, y + 1.6 * offsets[i], marker=MARKERS[i], markersize=9, color=COLORS[i],
                    markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none", zorder=3)
            ax.text(v, y + 1.6 * offsets[i], "  " + fmt.format(v), va="center", ha="left", fontsize=8, color=INK_2)
        ax.axhline(len(targets) - 0.3, color=GRID, linewidth=1)
        ax.set_title(name, loc="left", fontsize=10, color=INK, pad=8)
        if "5 mm" in name and narrow_five:
            # Same 0-100-style scale logic as the 2 mm panel would exaggerate 1-4 sample differences; show a wide range.
            ax.set_xlim(70, 95)
    axes[0].set_yticks(list(range(len(targets))) + [len(targets) + 0.55])
    axes[0].set_yticklabels(labels + ["all six (pooled)" if len(targets) == 6 else "all (pooled)"],
                            color=INK_2, fontsize=9.5)
    axes[0].invert_yaxis()
    axes[1].text(0.0, -0.13, "1 sample = 0.7 percentage points" + ("; axis zoomed to 70–95%" if narrow_five else ""),
                 transform=axes[1].transAxes, ha="left", fontsize=8, color=MUTED)
    axes[2].text(1.0, -0.13, "pooled median error = median of the per-target medians",
                 transform=axes[2].transAxes, ha="right", fontsize=8, color=MUTED)

    handles = [plt.Line2D([], [], marker=MARKERS[i], color=COLORS[i], markersize=8, linestyle="none",
                          markeredgecolor=SURFACE, label=c) for i, c in enumerate(conditions)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.12, 0.915), ncol=len(conditions),
               frameon=False, fontsize=9.5, labelcolor=INK_2, handletextpad=0.3, columnspacing=1.6)
    fig.suptitle(args.title, x=0.02, y=0.985, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.8, bottom=0.12, wspace=0.12)

    output = args.output or args.csv.with_suffix(".png")
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
