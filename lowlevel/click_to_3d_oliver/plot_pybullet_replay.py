"""Plot sim vs hardware joint angles or velocities, or their difference, from a run's pybullet_replay.json.

pybullet_replay.json is written by compare_hardware_to_pybullet.py. One panel per arm joint, all
in URDF degrees:
  --align time (default)  shared time axis: the PyBullet sim at every step (planned schedule) and
                          the hardware at its measured read times
  --align index           shared command-index axis: the sim's reading of each command and the
                          hardware's reading of the same command, whenever each was taken
  --velocity              joint velocity (deg/s) instead of angle, as the replay computed it: a
                          central difference over +/- its velocity_half_window samples, each series
                          against its own clock (see compare_hardware_to_pybullet.py); the first and
                          last few samples have none
  --error                 plot hardware - sim instead of both series (the replay's error_deg or
                          vel_error_deg_s)
With --align index, stretches where the plan holds the arm still are shaded and named from the
gripper target: the grasp (gripper closing) and the release (gripper open).
The gripper is left out, as in the comparison: the real gripper reads LeRobot's 0-100 range, not
degrees.

Usage (IKsim_mj env, from anywhere):
  python lowlevel/click_to_3d_oliver/plot_pybullet_replay.py lowlevel/real_world_runs/click_picks/<run> [--align index] [--velocity] [--error]
Writes <run>/pybullet_replay_<angles|error|velocity|velocity_error>[_by_index].png (or --output).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Colour palette, light mode: series, text and axis colours.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SIM_COLOR = "#2a78d6"  # slot 1, blue
HARDWARE_COLOR = "#eb6834"  # slot 2, orange
ERROR_COLOR = "#1baf7a"  # slot 3, aqua: the difference is its own series, so neither sim's nor hardware's colour
HELD_BAND = "#e1e0d9"  # gridline grey, washed out below
DPI = 150
LINE_PT = 2 * 72 / DPI  # 2 px lines


def held_spans(run_dir: Path) -> list[tuple[int, int, str]]:
    """(first, last command index, label) for each stretch of 50+ commands where the plan holds the arm.

    From timeline_sim.json's target_deg: the arm joints' targets don't change. Named by the gripper
    target from one waypoint before the stretch to its end: closing is the grasp, opening the
    release (the gripper opens as the arm arrives, just before it is held).
    """
    timeline = json.loads((run_dir / "timeline_sim.json").read_text(encoding="utf-8"))
    index = np.asarray(timeline["index"])
    target = np.asarray(timeline["target_deg"])
    lead = int(timeline["steps_per_waypoint"])
    still = np.all(np.abs(np.diff(target[:, :5], axis=0)) < 1e-9, axis=1)
    spans = []
    row = 0
    while row < len(still):
        if not still[row]:
            row += 1
            continue
        end = row
        while end < len(still) and still[end]:
            end += 1
        if end - row >= 50:
            before = target[max(row - lead - 1, 0), 5]
            after = target[end, 5]
            label = "grasp: gripper closing" if after < before else "release: gripper open" if after > before else "arm held"
            spans.append((int(index[row]), int(index[end]), label))
        row = end
    return spans


def rows(values: list) -> np.ndarray:
    """A replay JSON list of rows as a float array (null -> NaN)."""
    return np.array(values, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="run folder with pybullet_replay.json")
    parser.add_argument(
        "--align",
        choices=("time", "index"),
        default="time",
        help="x axis: time since the first command (default) or command index",
    )
    parser.add_argument("--velocity", action="store_true", help="plot joint velocities instead of angles")
    parser.add_argument("--error", action="store_true", help="plot hardware - sim instead of both series")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: <run_dir>/pybullet_replay_<angles|error|velocity|velocity_error>[_by_index].png",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    replay = json.loads((run_dir / "pybullet_replay.json").read_text(encoding="utf-8"))
    if replay.get("schema") != "pybullet_replay_v1":
        raise ValueError(f"{run_dir / 'pybullet_replay.json'} is not a pybullet_replay_v1 file")
    if args.velocity and "velocity_half_window" not in replay:
        raise ValueError(
            f"{run_dir / 'pybullet_replay.json'} has no velocities; rerun compare_hardware_to_pybullet.py on the run"
        )

    joints = replay["compared_joints"]
    vline_x = None
    spans: list[tuple[int, int, str]] = []
    if args.align == "time":
        aligned = replay["same_time"]
        sim_x = np.asarray(replay["sim"]["t_s"])
        sim_deg = np.asarray(replay["sim"]["joint_deg"])
        sim_vel = rows(replay["sim"]["joint_vel_diff_deg_s"]) if args.velocity else None
        hw_x = np.asarray(aligned["t_s"])
        vline_x = float(np.asarray(replay["same_command"]["sim_t_s"])[-1])
        x_label = "time since first command (s)"
        x_min = 0.0
        sim_label = f"PyBullet sim (driven with {replay['drive']}, planned 4.17 ms schedule)"
        hw_label = "hardware (reading_urdf_deg at measured read times)"
        what = "at the same time"
        suffix = ""
    else:
        aligned = replay["same_command"]
        sim_x = hw_x = np.asarray(aligned["index"])
        sim_deg = np.asarray(aligned["sim_deg"])
        sim_vel = rows(aligned["sim_vel_deg_s"]) if args.velocity else None
        x_label = "command index"
        x_min = float(sim_x[0])
        sim_label = f"PyBullet sim's reading of each command (driven with {replay['drive']})"
        hw_label = "hardware's reading of the same command (reading_urdf_deg)"
        what = "for the same command"
        suffix = "_by_index"
        spans = held_spans(run_dir)
    hw_deg = np.asarray(aligned["hardware_deg"])
    subtitle = run_dir.name
    if args.velocity:
        sim_vals = sim_vel
        hw_vals = rows(aligned["hardware_vel_deg_s"])
        error_vals = rows(aligned["vel_error_deg_s"])
        unit, quantity = "deg/s", "velocity"
        subtitle += (
            f"  ·  central difference over ±{replay['velocity_half_window']} samples, each against its own clock"
        )
        names = ("pybullet_replay_velocity_error", "pybullet_replay_velocity")
    else:
        sim_vals, hw_vals = sim_deg, hw_deg
        error_vals = np.asarray(aligned["error_deg"])
        unit, quantity = "deg", "angle"
        names = ("pybullet_replay_error", "pybullet_replay_angles")
    if args.error:
        title = f"Hardware - sim joint {quantity} {what} (URDF {unit})"
        default_name = f"{names[0]}{suffix}.png"
    else:
        title = f"Sim vs hardware joint {quantity} {what} (URDF {unit})"
        default_name = f"{names[1]}{suffix}.png"

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9})
    fig, axes = plt.subplots(
        len(joints), 1, sharex=True, figsize=(10, 1.9 * len(joints) + 1.0), dpi=DPI, facecolor=SURFACE
    )
    for j, (ax, name) in enumerate(zip(axes, joints)):
        ax.set_facecolor(SURFACE)
        for first, last, _ in spans:
            ax.axvspan(first, last, color=HELD_BAND, alpha=0.45, lw=0)
        if args.error:
            ax.axhline(0.0, color=AXIS, lw=0.5, zorder=0.5)
            ax.plot(hw_x, error_vals[:, j], color=ERROR_COLOR, lw=LINE_PT, solid_joinstyle="round", solid_capstyle="round")
            rms = float(np.sqrt(np.nanmean(error_vals[:, j] ** 2)))
            worst = float(np.nanmax(np.abs(error_vals[:, j])))
            ax.annotate(
                f"RMS {rms:.2f}, max {worst:.2f} {unit}",
                xy=(1.0, 1.0),
                xycoords="axes fraction",
                xytext=(0, -2),
                textcoords="offset points",
                ha="right",
                va="top",
                color=TEXT_SECONDARY,
                fontsize=8,
            )
        else:
            ax.plot(sim_x, sim_vals[:, j], color=SIM_COLOR, lw=LINE_PT, solid_joinstyle="round", solid_capstyle="round")
            ax.plot(hw_x, hw_vals[:, j], color=HARDWARE_COLOR, lw=LINE_PT, solid_joinstyle="round", solid_capstyle="round")
            if args.velocity:
                ax.axhline(0.0, color=AXIS, lw=0.5, zorder=0.5)
                ax.annotate(
                    f"peak |v|: sim {np.nanmax(np.abs(sim_vals[:, j])):.0f}, "
                    f"hardware {np.nanmax(np.abs(hw_vals[:, j])):.0f} {unit}",
                    xy=(1.0, 1.0),
                    xycoords="axes fraction",
                    xytext=(0, -2),
                    textcoords="offset points",
                    ha="right",
                    va="top",
                    color=TEXT_SECONDARY,
                    fontsize=8,
                )
        if vline_x is not None:
            ax.axvline(vline_x, color=AXIS, lw=0.5)
        ax.set_ylabel(f"{name}\n({unit})", color=TEXT_SECONDARY, rotation=0, ha="right", va="center", labelpad=8)
        ax.grid(axis="y", color=GRIDLINE, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.spines["bottom"].set_linewidth(0.5)

    for first, last, label in spans:
        axes[0].annotate(
            label,
            xy=((first + last) / 2, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=TEXT_MUTED,
            fontsize=8,
        )
    if vline_x is not None:
        axes[0].annotate(
            "sim's last command",
            xy=(vline_x, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, -2),
            textcoords="offset points",
            ha="left",
            va="top",
            color=TEXT_MUTED,
            fontsize=8,
        )
    # No direct labels: both lines end at the same angle on most joints, so they collide; the
    # legend above the panels names them.
    axes[-1].set_xlabel(x_label, color=TEXT_SECONDARY)
    x_max = max(sim_x[-1], hw_x[-1])
    axes[-1].set_xlim(x_min, x_max + 0.02 * (x_max - x_min))

    if not args.error:
        handles = [
            plt.Line2D([], [], color=SIM_COLOR, lw=LINE_PT, label=sim_label),
            plt.Line2D([], [], color=HARDWARE_COLOR, lw=LINE_PT, label=hw_label),
        ]
        legend = fig.legend(
            handles=handles, loc="upper left", bbox_to_anchor=(0.11, 0.952), ncol=2, frameon=False, fontsize=8
        )
        for text in legend.get_texts():
            text.set_color(TEXT_PRIMARY)
    fig.suptitle(title, x=0.11, y=0.988, ha="left", color=TEXT_PRIMARY, fontsize=10, fontweight="bold")
    fig.text(0.11, 0.968, subtitle, ha="left", va="top", color=TEXT_SECONDARY, fontsize=8)
    # Room above the panels for the title and run name, plus the legend when there is one.
    fig.tight_layout(rect=(0, 0, 1, 0.95 if args.error else 0.925))

    output = (args.output or run_dir / default_name).expanduser().resolve()
    fig.savefig(output, facecolor=SURFACE)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
