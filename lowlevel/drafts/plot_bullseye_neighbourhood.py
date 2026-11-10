#!/usr/bin/env python3
"""Small-multiple bullseye summary for one source square and the 3x3 block around it.

Cells are laid out like the board (rank 8 up, file h right), so the figure reads
as "move the rook from the centre square to each neighbour". Each cell shows the
landing points on that target square, coloured with the same distance bands as
plot_rook_offset_bullseye.py, and is titled with the within-5 mm count.

Run directories are found by name under rook_kiri_xy_lookup:
  <from>_to_<to>_bullseye_<mode>_corr*_*<tag>_<stamp>   (newest stamp wins)

Example:
  python lowlevel/drafts/plot_bullseye_neighbourhood.py --from-square e4 --tag e4ring
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_rook_offset_bullseye import (  # noqa: E402
    BANDS,
    FAILED_COLOR,
    FAILED_LABEL,
    GRID,
    INK,
    INK_2,
    MUTED,
    SQUARE_HALF_MM,
    SURFACE,
    TOLERANCE_MM,
    band_color,
    describe,
    style_axes,
    within_tolerance,
)

FILES = "abcdefgh"
LOOKUP_DIR = Path(__file__).resolve().parents[1] / "rook_kiri_xy_lookup"


def neighbourhood(square: str) -> list[list[str | None]]:
    """3x3 rows of target squares, top row = higher rank; None where off the board."""
    f, r = FILES.index(square[0]), int(square[1])
    rows = []
    for dr in (1, 0, -1):
        row = []
        for df in (-1, 0, 1):
            ff, rr = f + df, r + dr
            row.append(f"{FILES[ff]}{rr}" if 0 <= ff < 8 and 1 <= rr <= 8 else None)
        rows.append(row)
    return rows


def find_run(src: str, dst: str, mode: str, tag: str) -> Path | None:
    """Newest run for src->dst with exactly this tag. A comma-separated tag list is tried in order,
    e.g. "e4cal_median,e4fpivot_cal_median" = d/e runs from the first tag, f runs from the second."""
    if tag and "," in tag:
        for one in tag.split(","):
            run = find_run(src, dst, mode, one.strip())
            if run is not None:
                return run
        return None
    # The tag must sit directly before the timestamp, so "e4ring" never matches "e4ring_cal" runs.
    suffix = rf"_{re.escape(tag)}_\d{{8}}_\d{{6}}" if tag else r"_\d{8}_\d{6}"
    name = re.compile(rf"{src}_to_{dst}_bullseye_{mode}_corr\d+_(?:random\d+|grid\d+){suffix}")
    runs = sorted(
        d for d in LOOKUP_DIR.glob(f"{src}_to_{dst}_bullseye_{mode}_*")
        if name.fullmatch(d.name) and (d / "results.json").exists()
    )
    return runs[-1] if runs else None


def draw_cell(ax, samples: list[dict], title: str) -> None:
    style_axes(ax)
    for limit, color, _ in reversed(BANDS[:-1]):
        ax.add_patch(Circle((0, 0), limit, facecolor=color, alpha=0.16, edgecolor=color, linewidth=1))
    ax.add_patch(Rectangle((-SQUARE_HALF_MM, -SQUARE_HALF_MM), 2 * SQUARE_HALF_MM, 2 * SQUARE_HALF_MM,
                           fill=False, edgecolor=MUTED, linewidth=1, linestyle=(0, (4, 3))))
    hidden = 0
    for s in samples:
        err = s.get("final_error_xy_mm")
        if (not s.get("landed_upright") or err is None or any(v is None for v in err)
                or max(abs(err[0]), abs(err[1])) > SQUARE_HALF_MM + 2):
            hidden += 1
            continue
        ax.scatter(err[0], err[1], s=22, color=band_color(s), edgecolors=SURFACE, linewidths=1.0, zorder=3)
    ax.plot(0, 0, marker="+", color=INK, markersize=8, zorder=5)
    landed = [s["final_error_xy_mm"] for s in samples
              if s.get("landed_upright") and s.get("final_error_xy_mm") is not None]
    if landed:
        cx, cy = np.median(np.array(landed, dtype=float), axis=0)
        ax.plot(cx, cy, marker="o", markersize=9, markerfacecolor="none", markeredgecolor=INK,
                markeredgewidth=1.6, zorder=6)
    lim = SQUARE_HALF_MM + 2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=16)
    ax.text(0, 1.015, f"{hidden} failed or landed off-square" if hidden else "none failed",
            transform=ax.transAxes, color=INK_2, fontsize=8.5, ha="left", va="bottom")


def draw_placeholder(ax, text: str) -> None:
    style_axes(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor=GRID, linewidth=1, linestyle=(0, (4, 3))))
    ax.text(0.5, 0.5, text, ha="center", va="center", color=MUTED, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-square", default="e4")
    parser.add_argument("--centre", default=None,
                        help="Centre of the 3x3 block of targets (default: the source square).")
    parser.add_argument("--mode", choices=("known", "blind"), default="blind")
    parser.add_argument("--tag", default="e4ring", help="Run-directory tag; empty string = newest untagged/any.")
    parser.add_argument("--label", default=None, help="Prefix for the figure title, e.g. 'Before'.")
    parser.add_argument("--note", default=None, help="Extra subtitle line, e.g. which place offset was used.")
    parser.add_argument("--compare-tag", default=None,
                        help="Also report the within-5 mm total of this tag's runs over the same targets.")
    parser.add_argument("--missing-text", default="not run yet", help="Placeholder text for targets with no run.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    src = args.from_square

    fig, axes = plt.subplots(3, 3, figsize=(12, 12.6), facecolor=SURFACE)
    totals = {"ok": 0, "n": 0}
    compare = {"ok": 0, "n": 0}
    counts = {label: 0 for _, _, label in BANDS} | {FAILED_LABEL: 0}
    config = None
    rows_out = []
    centre = args.centre or src
    for r, row in enumerate(neighbourhood(centre)):
        for c, dst in enumerate(row):
            ax = axes[r][c]
            if dst is None:
                draw_placeholder(ax, "off board")
                continue
            run = find_run(src, dst, args.mode, args.tag)
            if run is None:
                draw_placeholder(ax, f"{src} → {dst}\n{args.missing_text}")
                continue
            data = json.loads((run / "results.json").read_text(encoding="utf-8"))
            config = config or data["config"]
            samples = [s for s in data["samples"] if "error" not in s]
            ok = sum(within_tolerance(s) for s in samples)
            totals["ok"] += ok
            totals["n"] += len(samples)
            for s in samples:
                color = band_color(s)
                counts[FAILED_LABEL if color == FAILED_COLOR else next(l for _, cc, l in BANDS if cc == color)] += 1
            landed = [s for s in samples if s.get("landed_upright") and s.get("xy_error_mm") is not None]
            med = float(np.median([s["xy_error_mm"] for s in landed])) if landed else float("nan")
            label = "pick & replace" if dst == src else f"→ {dst}"
            draw_cell(ax, samples, f"{label}   {ok}/{len(samples)} within {TOLERANCE_MM:g} mm")
            rows_out.append((dst, ok, len(samples), len(samples) - len(landed), med, run.name))
            if args.compare_tag:
                ref = find_run(src, dst, args.mode, args.compare_tag)
                if ref is not None:
                    ref_samples = [s for s in json.loads((ref / "results.json").read_text(encoding="utf-8"))["samples"]
                                   if "error" not in s]
                    compare["ok"] += sum(within_tolerance(s) for s in ref_samples)
                    compare["n"] += len(ref_samples)
            if r == 2:
                ax.set_xlabel("landing error along file (mm)", color=INK_2, fontsize=9)
            if c == 0:
                ax.set_ylabel("along rank (mm)", color=INK_2, fontsize=9)

    pct = 100.0 * totals["ok"] / totals["n"] if totals["n"] else 0.0
    label = f"{args.label}: " if args.label else ""
    headline = f"{totals['ok']}/{totals['n']} ({pct:.0f}%) within {TOLERANCE_MM:g} mm"
    fig.suptitle(
        (f"{label}rook from {src} to its 3×3 neighbourhood: {headline}" if centre == src else
         f"{label}rook from {src} to the 3×3 block around {centre}: {headline}"),
        x=0.05, y=0.985, ha="left", color=INK, fontsize=13, fontweight="bold",
    )
    if config:
        subtitle = describe(config) + " · same noise draws for every target"
        if args.note:
            subtitle += "\n" + args.note
        if compare["n"]:
            subtitle += (f"\nbefore on the same targets: {compare['ok']}/{compare['n']} "
                         f"({100.0 * compare['ok'] / compare['n']:.0f}%) within {TOLERANCE_MM:g} mm")
        fig.text(0.05, 0.955, subtitle, ha="left", va="top", color=INK_2, fontsize=10, linespacing=1.5)
    handles = [Patch(facecolor=c, edgecolor=c, label=f"{l}  ({counts[l]})") for _, c, l in BANDS]
    handles.append(Patch(facecolor=FAILED_COLOR, edgecolor=FAILED_COLOR,
                         label=f"failed: missed grasp, fell or dropped  ({counts[FAILED_LABEL]})"))
    handles.append(Line2D([], [], marker="o", linestyle="none", markersize=9, markerfacecolor="none",
                          markeredgecolor=INK, markeredgewidth=1.6, label="median landing point"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               labelcolor=INK_2, title="distance from target centre (all targets)", title_fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92 - 0.035 * (bool(args.note) + bool(compare["n"])), bottom=0.105,
                        wspace=0.12, hspace=0.2)

    tag_name = args.tag.replace(",", "+") if args.tag else ""
    stem = f"{src}_neighbourhood" if centre == src else f"{src}_to_block{centre}"
    output = args.output or LOOKUP_DIR / f"{stem}_bullseye_{args.mode}{'_' + tag_name if tag_name else ''}.png"
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    print(f"Saved {output}")
    print(f"{'target':<8}{'≤5mm':>8}{'failed':>8}{'median mm':>11}  run")
    for dst, ok, n, failed, med, name in rows_out:
        print(f"{dst:<8}{f'{ok}/{n}':>8}{failed:>8}{med:>11.2f}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
