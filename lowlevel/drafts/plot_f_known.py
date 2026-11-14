#!/usr/bin/env python3
"""f3/f4/f5 -> f3/f4/f5 in KNOWN mode: one panel per (source, target) pair.

Each panel is the 9x9 grid of start offsets (where the rook started, relative to the source-square
centre; the robot is told this offset). Each start is coloured by where the rook ended up (the
same distance bands as the bullseye plots); dark = missed grasp, fell or dropped.

  python lowlevel/drafts/plot_f_known.py [--tag fknown_cal]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_bullseye_neighbourhood import LOOKUP_DIR, find_run  # noqa: E402
from plot_rook_offset_bullseye import (  # noqa: E402
    BANDS, FAILED_COLOR, FAILED_LABEL, INK, INK_2, MUTED, SURFACE, band_color, style_axes, within_tolerance,
)

SOURCES = ["f5", "f4", "f3"]
TARGETS = ["f5", "f4", "f3"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="fknown_cal")
    parser.add_argument("--output", type=Path, default=LOOKUP_DIR / "2026-11-07_block_d3-f5" / "figures" / "f_to_f__robot_knows_start__calibration_pass1.png")
    parser.add_argument("--offset-note", default="place offset = that pair's blind median calibration")
    args = parser.parse_args()

    fig, axes = plt.subplots(3, 3, figsize=(12, 12.4), facecolor=SURFACE)
    counts = {label: 0 for _, _, label in BANDS} | {FAILED_LABEL: 0}
    tot_ok = tot_n = 0
    lines = ["| source \\ target | " + " | ".join(TARGETS) + " |", "|---|---|---|---|"]
    for r, src in enumerate(SOURCES):
        cells = []
        for c, dst in enumerate(TARGETS):
            ax = axes[r][c]
            style_axes(ax)
            run = find_run(src, dst, "known", args.tag)
            if run is None:
                ax.text(0.5, 0.5, f"{src} → {dst}\nnot run", transform=ax.transAxes, ha="center", color=MUTED)
                cells.append("–")
                continue
            samples = [s for s in json.loads((run / "results.json").read_text())["samples"] if "error" not in s]
            ok = sum(within_tolerance(s) for s in samples)
            tot_ok += ok
            tot_n += len(samples)
            fails = [s for s in samples if not s.get("landed_upright")]
            for s in samples:
                color = band_color(s)
                counts[FAILED_LABEL if color == FAILED_COLOR else next(l for _, cc, l in BANDS if cc == color)] += 1
                ax.scatter(s["dx_mm"], s["dy_mm"], s=150, marker="s", color=color, edgecolors=SURFACE,
                           linewidths=1.5, zorder=3)
            ax.plot(0, 0, marker="+", color=INK, markersize=9, zorder=5)
            ax.set_xlim(-12.5, 12.5)
            ax.set_ylim(-12.5, 12.5)
            ax.set_xticks([-10, -5, 0, 5, 10])
            ax.set_yticks([-10, -5, 0, 5, 10])
            ax.set_aspect("equal")
            up = [s for s in samples if s.get("landed_upright")]
            med = np.median([s["xy_error_mm"] for s in up]) if up else float("nan")
            title = ("pick & replace" if src == dst else f"{src} → {dst}") + f"   {ok}/{len(samples)} within 5 mm"
            ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=16)
            ax.text(0, 1.015, f"median error {med:.1f} mm · {len(fails)} failed", transform=ax.transAxes,
                    fontsize=8.5, color=INK_2, ha="left", va="bottom")
            if r == 2:
                ax.set_xlabel("start offset along file, x (mm)", color=INK_2, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"from {src}\nstart offset along rank, y (mm)", color=INK_2, fontsize=9)
            cells.append(f"{ok}/{len(samples)} (med {med:.1f} mm, {len(fails)} failed)")
        lines.append(f"| **{src}** | " + " | ".join(cells) + " |")

    pct = 100 * tot_ok / tot_n if tot_n else 0
    fig.suptitle(f"Rook f-file → f-file, KNOWN start offset: {tot_ok}/{tot_n} ({pct:.0f}%) within 5 mm",
                 x=0.05, y=0.985, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.05, 0.955, "robot is told the true start offset (9×9 grid, ±10 mm) · each square = one start, coloured "
             "by where the rook landed\npivot-first place · " + args.offset_note,
             ha="left", va="top", fontsize=10, color=INK_2, linespacing=1.5)
    handles = [Patch(facecolor=c, edgecolor=c, label=f"landed {l}  ({counts[l]})") for _, c, l in BANDS]
    handles.append(Patch(facecolor=FAILED_COLOR, edgecolor=FAILED_COLOR,
                         label=f"failed: missed grasp, fell or dropped  ({counts[FAILED_LABEL]})"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK_2)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.09, wspace=0.18, hspace=0.25)
    fig.savefig(args.output, dpi=150, facecolor=SURFACE)
    md = args.output.with_suffix(".md")
    md.write_text(f"# f → f, known mode ({args.tag}): within 5 mm of 81\n\n" + "\n".join(lines)
                  + f"\n\nTotal {tot_ok}/{tot_n} ({pct:.0f}%).\n", encoding="utf-8")
    print(f"Saved {args.output} and {md}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
