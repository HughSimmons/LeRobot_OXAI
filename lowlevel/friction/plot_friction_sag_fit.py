#!/usr/bin/env python3
"""Plot how well the friction-and-sag model (fit_friction_sag_model.py) predicts the real arm.

Reads the fit's held-out predictions (every reading predicted by a fit that never saw that
angle; saved under "held_out_predictions" when the fit runs without --no-held-out) and makes:

  friction_sag_rmse.png      held-out RMSE per joint, for readings that arrived moving down
                             (with gravity) and up (against it): M3 next to the baselines, and
                             a marker for how much the same move varies when repeated, which no
                             model of the commanded move can beat.
  friction_sag_residuals.png predicted vs measured sag, and what M3 misses vs gravity load,
                             per joint, coloured by approach direction.

"Sag" is how far the joint stopped short of its target in the direction gravity pulls
(e_up in fit_friction_sag_model.py), so positive means drooped.

    python friction/plot_friction_sag_fit.py
    python friction/plot_friction_sag_fit.py --fit path/to/fit.json --out-dir some/folder
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# lowlevel/ on the path, for the friction package (as kinematicscalibration/ does).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from friction.fit_friction_sag_model import GRAVITY_JOINTS, GravityModel, load_sweep_dir, readings_from_sweep
from friction.fit_servo_model_to_sweeps import JOINT_NAMES, SWEEP_DIR

COVERAGE_DIR = SWEEP_DIR.parent / "friction_coverage"
JOINT_TITLES = {"shoulder_lift": "Shoulder lift", "elbow_flex": "Elbow", "wrist_flex": "Wrist flex"}
DIRECTIONS = {False: "arriving down (with gravity)", True: "arriving up (against gravity)"}

# Light chart surface, inks and categorical slots, in fixed order (dataviz reference palette).
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# (key in held_out_predictions, legend name, colour). M3 is the one blue bar; the baselines it is
# compared against are greys, so blue, aqua and orange each mean one thing across both figures.
MODELS = [
    ("m3_deg", "M3 (this model)", BLUE),
    ("shift_plus_sag_deg", "shift + linear sag", "#52514e"),
    ("fixed_shift_deg", "fixed shift (sim today)", "#898781"),
    (None, "no correction", "#c3c2b7"),
]
DIRECTION_COLOR = {False: AQUA, True: ORANGE}


def rmse(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")


def repeat_spread(offsets_deg: np.ndarray) -> dict[tuple[str, bool], float]:
    """How much the same move varies when repeated, by joint and direction: each coverage sweep
    makes every (angle, direction) move twice, and sd = rms(difference) / sqrt(2)."""
    gravity = GravityModel()
    diffs: dict[tuple[str, bool], list[float]] = {}
    for folder in sorted(COVERAGE_DIR.iterdir()):
        for sweep in load_sweep_dir(folder):
            seen: dict[tuple[float, bool], list[float]] = {}
            for r in readings_from_sweep(sweep):
                s = 1.0 if gravity.holding_torque(r.joint, r.pose_deg, offsets_deg) >= 0 else -1.0
                seen.setdefault((r.target_deg, r.move_dir == s), []).append(r.error_deg)
            for (_, lifting), errors in seen.items():
                if len(errors) == 2:
                    diffs.setdefault((sweep.joint, lifting), []).append(errors[0] - errors[1])
    return {key: rmse(d) / np.sqrt(2) for key, d in diffs.items()}


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_rmse(schemes: list[tuple[str, list[dict]]], spread: dict, out: Path) -> None:
    """One row of panels per hold-out scheme (title, held-out predictions)."""
    fig, grid = plt.subplots(len(schemes), 3, figsize=(13, 4.3 * len(schemes)), facecolor=SURFACE,
                             sharey=True, squeeze=False)
    for axes, (title, rows) in zip(grid, schemes):
        rmse_row(axes, rows, spread)
        axes[0].set_ylabel(f"{title}\n\nheld-out RMSE (°)", color=INK2)
    grid[0, 0].set_ylim(bottom=0)
    handles, labels = grid[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.97 if len(schemes) > 1 else 0.92))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def rmse_row(axes, rows: list[dict], spread: dict) -> None:
    width = 0.19
    for ax, joint in zip(axes, GRAVITY_JOINTS):
        style(ax)
        tick_labels = []
        for g, lifting in enumerate((False, True)):
            own = [r for r in rows if r["joint"] == joint and r["lifting"] == lifting]
            tick_labels.append(f"arriving {'up' if lifting else 'down'}\n({'against' if lifting else 'with'} gravity)\nn = {len(own)}")
            measured = np.array([r["measured_error_deg"] for r in own])
            for m, (key, name, color) in enumerate(MODELS):
                predicted = np.array([r[key] for r in own]) if key else np.zeros(len(own))
                value = rmse(measured - predicted)
                x = g + (m - 1.5) * width
                ax.bar(x, value, width * 0.9, color=color, label=name if g == 0 else None)
                if key == "m3_deg":
                    ax.text(x, value + 0.06, f"{value:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
            floor = spread.get((joint, lifting))
            if floor is not None:
                ax.plot([g - 2 * width, g + 2 * width], [floor, floor], color=INK, linewidth=1.4,
                        linestyle=(0, (3, 2)), label="same move repeated (sd)" if g == 0 else None)
        ax.set_xticks([0, 1], tick_labels)
        ax.set_title(JOINT_TITLES[joint], loc="left", fontsize=11, color=INK)
        ax.tick_params(colors=MUTED)


def plot_residuals(rows: list[dict], title: str, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), facecolor=SURFACE)
    fig.suptitle(title, x=0.01, ha="left", y=0.995, fontsize=10, color=INK2)
    for col, joint in enumerate(GRAVITY_JOINTS):
        own = [r for r in rows if r["joint"] == joint]
        s = np.array([1.0 if r["tau_hold"] >= 0 else -1.0 for r in own])
        measured = s * np.array([r["measured_error_deg"] for r in own])   # sag, + = drooped
        predicted = s * np.array([r["m3_deg"] for r in own])
        load = np.abs([r["tau_hold"] for r in own])
        lifting = np.array([r["lifting"] for r in own])

        top, bottom = axes[0, col], axes[1, col]
        for ax in (top, bottom):
            style(ax)
            ax.tick_params(colors=MUTED)
        lo, hi = min(measured.min(), predicted.min()) - 0.5, max(measured.max(), predicted.max()) + 0.5
        top.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        bottom.axhline(0, color=MUTED, linewidth=1)
        for up in (False, True):
            m = lifting == up
            kw = dict(s=26, color=DIRECTION_COLOR[up], edgecolor=SURFACE, linewidth=0.8, alpha=0.9)
            top.scatter(measured[m], predicted[m], label=DIRECTIONS[up], **kw)
            bottom.scatter(load[m], measured[m] - predicted[m], **kw)
        top.set_xlim(lo, hi); top.set_ylim(lo, hi); top.set_aspect("equal", adjustable="box")
        top.set_title(JOINT_TITLES[joint], loc="left", fontsize=11, color=INK)
        top.set_xlabel("measured sag (°)", color=INK2)
        bottom.set_xlabel("gravity load on the joint (N·m)", color=INK2)
    axes[0, 0].set_ylabel("predicted sag, held out (°)", color=INK2)
    axes[1, 0].set_ylabel("measured − predicted (°)", color=INK2)
    axes[0, 0].text(0.04, 0.96, "on the dashed line = predicted exactly", transform=axes[0, 0].transAxes,
                    fontsize=8.5, color=INK2, va="top")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", type=Path, default=SWEEP_DIR / "friction_sag_fit.json")
    parser.add_argument("--out-dir", type=Path, default=SWEEP_DIR / "plots")
    args = parser.parse_args()

    fit = json.loads(args.fit.read_text(encoding="utf-8"))
    schemes = [(title, fit[key]) for key, title in (
        ("held_out_predictions", "angles it never saw"),
        ("held_out_sweep_predictions", "poses it never saw"),
    ) if fit.get(key)]
    if not schemes:
        raise SystemExit(f"{args.fit} has no held-out predictions: rerun fit_friction_sag_model.py "
                         "without --no-held-out, or with --held-out-sweeps")
    offsets = np.array([fit["offsets_deg"][name] for name in JOINT_NAMES], dtype=float)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_rmse(schemes, repeat_spread(offsets), args.out_dir / "friction_sag_rmse.png")
    # The residuals use the strictest scheme available (the last one).
    title, rows = schemes[-1]
    plot_residuals(rows, f"Held out: {title}", args.out_dir / "friction_sag_residuals.png")
    print(f"saved: {args.out_dir / 'friction_sag_rmse.png'}\nsaved: {args.out_dir / 'friction_sag_residuals.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
