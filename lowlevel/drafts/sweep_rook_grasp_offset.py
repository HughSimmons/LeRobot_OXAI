#!/usr/bin/env python3
"""Sweep the rook grasp offset and score each setting by blind-pickup tolerance.

Every grasp-offset setting is run on the same set of Gaussian start offsets
(common random numbers), with the robot blind to the offset, so the settings
can be compared fairly. Scores: share landing within 5 mm of the target
centre, then misses / falls, then mean landing error.

Grasp-offset axes are the gripper's own frame (see chess_traj.xyz_homeref),
not board axes.

Examples:
  # coarse XY sweep, ±3 mm in 1 mm steps around the default
  python -B lowlevel/drafts/sweep_rook_grasp_offset.py --axes xy --span 0.003 --step 0.001 --repeat-baseline
  # height sweep around a chosen centre
  python -B lowlevel/drafts/sweep_rook_grasp_offset.py --axes z --center -0.012 0.003 -0.003 --span 0.002 --step 0.001
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from run_rook_offset_bullseye import LOWLEVEL_DIR, ROOK_ENV, run_sample, to_jsonable, worker_init


DEFAULT_GRASP_OFFSET = (-0.014, 0.002, -0.003)
TOLERANCE_MM = 5.0
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
# Sequential blue ramp, light -> dark (steps 100..700).
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axes", choices=("xy", "z"), default="xy")
    parser.add_argument("--candidates", nargs="+", default=None, metavar="X,Y,Z",
                        help='Compare an explicit list of grasp offsets in mm, bracketed so argparse '
                             'accepts the minus sign (e.g. "(-18,-1,-3)"), instead of a grid.')
    parser.add_argument("--center", nargs=3, type=float, default=DEFAULT_GRASP_OFFSET,
                        help="Grasp offset (m) the sweep is centred on.")
    parser.add_argument("--span", type=float, default=0.003, help="Sweep ± this much (m) on each swept axis.")
    parser.add_argument("--step", type=float, default=0.001)
    parser.add_argument("--from-square", default="e4")
    parser.add_argument("--to-square", default="e5")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--noise-sigma", type=float, default=0.0033)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--place-offset", nargs=3, type=float, default=(-0.011, 0.002, -0.003))
    parser.add_argument("--repeat-baseline", action="store_true",
                        help="Also rerun the centre setting a second time to measure sim repeatability.")
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def sweep_deltas(args: argparse.Namespace) -> list[tuple[float, float, float]]:
    if args.candidates:
        center = np.array(args.center, dtype=float)
        return [
            tuple(np.array([float(v) for v in c.strip(" ()").split(",")]) / 1000.0 - center)
            for c in args.candidates
        ]
    n = int(round(args.span / args.step))
    steps = [i * args.step for i in range(-n, n + 1)]
    if args.axes == "xy":
        return [(d0, d1, 0.0) for d1 in steps for d0 in steps]
    return [(0.0, 0.0, d2) for d2 in steps]


def summarise(samples: list[dict]) -> dict:
    landed = [s for s in samples if s.get("landed_upright")]
    errors = [s["xy_error_mm"] for s in landed]
    return {
        "n": len(samples),
        "within_5mm": sum(e < TOLERANCE_MM for e in errors),
        "within_2mm": sum(e < 2.0 for e in errors),
        "missed": sum(not s.get("pickup_success") for s in samples),
        "fell_or_dropped": sum(bool(s.get("pickup_success")) and not s.get("landed_upright") for s in samples),
        "sim_errors": sum("error" in s for s in samples),
        "mean_landed_error_mm": float(np.mean(errors)) if errors else None,
        "median_landed_error_mm": float(np.median(errors)) if errors else None,
    }


def rank_key(setting: dict):
    s = setting["summary"]
    return (-s["within_5mm"], s["missed"] + s["fell_or_dropped"], s["mean_landed_error_mm"] or np.inf)


def plot_xy(settings: list[dict], meta: dict, output: Path) -> None:
    grid = [s for s in settings if s["label"] != "baseline_repeat"]
    step = meta["step_mm"]
    best = min(grid, key=rank_key)
    fig, ax = plt.subplots(figsize=(8.6, 8.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in grid:
        d0, d1 = s["delta_mm"][0], s["delta_mm"][1]
        frac = s["summary"]["within_5mm"] / max(1, s["summary"]["n"])
        color = BLUES[min(len(BLUES) - 1, int(round(frac * (len(BLUES) - 1))))]
        cell = step - 0.06
        ax.add_patch(Rectangle((d0 - cell / 2, d1 - cell / 2), cell, cell, facecolor=color, edgecolor=SURFACE))
        text_color = SURFACE if frac >= 0.5 else INK
        ax.text(d0, d1 + step * 0.1, f"{frac * 100:.0f}%", ha="center", va="center",
                color=text_color, fontsize=10, fontweight="bold")
        ax.text(d0, d1 - step * 0.22, f"M{s['summary']['missed']} F{s['summary']['fell_or_dropped']}",
                ha="center", va="center", color=text_color, fontsize=7.5)
    half = step / 2 - 0.02
    ax.add_patch(Rectangle((-half, -half), 2 * half, 2 * half, fill=False, edgecolor=INK,
                           linewidth=1.6, linestyle=(0, (3, 2))))
    bd0, bd1 = best["delta_mm"][0], best["delta_mm"][1]
    ax.add_patch(Rectangle((bd0 - half, bd1 - half), 2 * half, 2 * half, fill=False, edgecolor=INK, linewidth=2.6))
    lim = meta["span_mm"] + step / 2 + 0.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_xticks([s["delta_mm"][0] for s in grid[: int(round(2 * meta["span_mm"] / step)) + 1]])
    ax.set_yticks(ax.get_xticks())
    ax.set_xlabel("change to grasp-offset axis 1 (mm, gripper frame)", color=INK_2, fontsize=10)
    ax.set_ylabel("change to grasp-offset axis 2 (mm, gripper frame)", color=INK_2, fontsize=10)
    c = meta["center_mm"]
    b = best["grasp_offset_mm"]
    fig.suptitle(f"Grasp-offset sweep, rook {meta['from_square']} → {meta['to_square']} (robot blind to offset)",
                 x=0.07, y=0.975, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.text(0.07, 0.935,
             f"% of {meta['samples']} random start offsets (σ = {meta['noise_sigma_mm']:g} mm) landing within "
             f"{TOLERANCE_MM:g} mm · M = missed grasp, F = fell/dropped",
             ha="left", color=INK_2, fontsize=9.5)
    fig.text(0.07, 0.905,
             f"dashed = sweep centre ({c[0]:g}, {c[1]:g}, {c[2]:g}) mm · bold = best ({b[0]:g}, {b[1]:g}, {b[2]:g}) mm",
             ha="left", color=INK_2, fontsize=9.5)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.87, bottom=0.08)
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_z(settings: list[dict], meta: dict, output: Path) -> None:
    grid = sorted((s for s in settings if s["label"] != "baseline_repeat"), key=lambda s: s["delta_mm"][2])
    xs = [s["delta_mm"][2] for s in grid]
    ys = [100 * s["summary"]["within_5mm"] / max(1, s["summary"]["n"]) for s in grid]
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.bar(xs, ys, width=meta["step_mm"] * 0.6, color=BLUES[7], edgecolor=SURFACE)
    for x, y, s in zip(xs, ys, grid):
        ax.text(x, y + 1.5, f"{y:.0f}%  M{s['summary']['missed']} F{s['summary']['fell_or_dropped']}",
                ha="center", va="bottom", color=INK_2, fontsize=8.5)
    ax.set_ylim(0, 110)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.yaxis.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("change to grasp-offset axis 3 (mm, gripper frame)", color=INK_2)
    ax.set_ylabel(f"% landing within {TOLERANCE_MM:g} mm", color=INK_2)
    c = meta["center_mm"]
    ax.set_title(f"Grasp height sweep around ({c[0]:g}, {c[1]:g}, {c[2]:g}) mm", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def plot_list(settings: list[dict], meta: dict, output: Path) -> None:
    rows = sorted(settings, key=rank_key, reverse=True)
    fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(rows) + 1.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    labels = []
    for i, s in enumerate(rows):
        m = s["summary"]
        pct = 100 * m["within_5mm"] / max(1, m["n"])
        lo, hi = wilson_interval(m["within_5mm"], m["n"])
        ax.barh(i, pct, height=0.56, color=BLUES[7], edgecolor=SURFACE)
        ax.plot([100 * lo, 100 * hi], [i, i], color=INK, linewidth=1.2)
        ax.text(101.5, i, f"{pct:.0f}%   M{m['missed']} F{m['fell_or_dropped']}   "
                f"mean {m['mean_landed_error_mm']:.1f} mm", va="center", color=INK_2, fontsize=9)
        g = s["grasp_offset_mm"]
        labels.append(f"({g[0]:g}, {g[1]:g}, {g[2]:g})")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, color=INK, fontsize=9.5)
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.xaxis.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"% landing within {TOLERANCE_MM:g} mm (line = 95% confidence range)", color=INK_2)
    ax.set_title(f"Grasp offsets (mm, gripper frame) on {meta['samples']} random start offsets, "
                 f"σ = {meta['noise_sigma_mm']:g} mm, robot blind", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.candidates:
        args.axes = "list"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        or LOWLEVEL_DIR / "rook_kiri_xy_lookup"
        / f"{args.from_square}_to_{args.to_square}_grasp_sweep_{args.axes}_{stamp}"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    log_dir = output_dir / "logs"
    log_dir.mkdir()

    rng = np.random.default_rng(args.seed)
    offsets = rng.normal(0.0, args.noise_sigma, size=(args.samples, 2))
    center = np.array(args.center, dtype=float)
    settings = []
    deltas = sweep_deltas(args)
    labelled = [(f"d{i:03d}", d) for i, d in enumerate(deltas)]
    if args.repeat_baseline:
        labelled.append(("baseline_repeat", (0.0, 0.0, 0.0)))
    for label, delta in labelled:
        grasp = center + np.array(delta)
        settings.append({
            "label": label,
            "delta_mm": [round(float(v) * 1000, 3) for v in delta],
            "grasp_offset": grasp.tolist(),
            "grasp_offset_mm": [round(float(v) * 1000, 3) for v in grasp],
            "config": {
                "mode": "blind",
                "from_square": args.from_square,
                "to_square": args.to_square,
                "placement_corrections": 0,
                "correction_gain": 0.5,
                "grasp_offset": grasp.tolist(),
                "place_offset": list(args.place_offset),
                "placement_lower_steps": 10,
            },
            "samples": [],
        })
    index_by_label = {s["label"]: i for i, s in enumerate(settings)}
    jobs = [
        {"name": f"{s['label']}_s{j:02d}", "dx": float(dx), "dy": float(dy), "config": s["config"]}
        for j, (dx, dy) in enumerate(offsets)
        for s in settings
    ]
    meta = {
        "axes": args.axes,
        "from_square": args.from_square,
        "to_square": args.to_square,
        "center_mm": [round(float(v) * 1000, 3) for v in center],
        "span_mm": args.span * 1000,
        "step_mm": args.step * 1000,
        "samples": args.samples,
        "noise_sigma_mm": args.noise_sigma * 1000,
        "seed": args.seed,
        "place_offset": list(args.place_offset),
        "rook_environment": ROOK_ENV,
    }
    print(f"{len(settings)} settings x {args.samples} samples = {len(jobs)} sims, {args.workers} workers -> {output_dir}",
          flush=True)

    started = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=worker_init, initargs=(str(log_dir),), maxtasksperchild=8) as pool:
        for i, result in enumerate(pool.imap_unordered(run_sample, jobs), start=1):
            settings[index_by_label[result["name"].rsplit("_s", 1)[0]]]["samples"].append(to_jsonable(result))
            if i % 50 == 0 or i == len(jobs):
                elapsed = time.time() - started
                print(f"[{i}/{len(jobs)}] {elapsed / 60:.1f} min elapsed, "
                      f"~{elapsed / i * (len(jobs) - i) / 60:.1f} min left", flush=True)

    for s in settings:
        s["samples"].sort(key=lambda r: r["name"])
        s["summary"] = summarise(s["samples"])
        del s["config"]
    ranked = sorted((s for s in settings if s["label"] != "baseline_repeat"), key=rank_key)
    result = {"meta": meta, "elapsed_seconds": time.time() - started,
              "best": {k: ranked[0][k] for k in ("label", "delta_mm", "grasp_offset", "grasp_offset_mm", "summary")},
              "settings": settings}
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plotter = {"xy": plot_xy, "z": plot_z, "list": plot_list}[args.axes]
    plotter(settings, meta, output_dir / "grasp_sweep.png")

    print("Top settings (grasp offset mm: within5/n, missed, fell, mean err):")
    for s in ranked[:5]:
        m = s["summary"]
        print(f"  {s['grasp_offset_mm']}: {m['within_5mm']}/{m['n']}, M{m['missed']}, F{m['fell_or_dropped']}, "
              f"{m['mean_landed_error_mm']:.2f} mm")
    for s in settings:
        if s["delta_mm"] == [0.0, 0.0, 0.0]:
            m = s["summary"]
            print(f"  [{s['label']}] {s['grasp_offset_mm']}: {m['within_5mm']}/{m['n']}, M{m['missed']}, "
                  f"F{m['fell_or_dropped']}")
    print(f"Results: {output_dir / 'results.json'} ({(time.time() - started) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
