"""Tabulate the sim-vs-hardware comparison over many runs, to see whether one run's findings hold.

For each run folder with a pybullet_replay.json (written by compare_hardware_to_pybullet.py), reads
it with the run's timeline_sim.json and timeline_hardware.json and prints one table per measure: a
row per run, then the median, min and max over the runs. Arm joints only, in URDF degrees, in the
order pan / lift / elbow / wrist flex (wflex) / wrist roll (wroll). Nothing is written.

Measures:
  loop timing          the hardware's median time per command (planned: command_delay_s) and its
                       parts: write (t_send to t_sent), sleep (t_sent to t_read_start), read
                       (t_read_start to t_read_end); the slowest read; read retries; and the run's
                       length over the planned schedule's (last t_read_end_s, hardware / sim)
  same-time RMS        the replay's summary; mostly the hardware's drift behind schedule
  same-command RMS     the replay's summary, RMS and largest |error|
  lag                  per joint, the shift k (commands, 0 to MAX_LAG - 1) for which the hardware's
                       reading of command i best matches (least RMS) the sim's reading of command
                       i - k; also k x the run's median time per command, in ms
  lag-shifted RMS      the same-command RMS at that shift, over all commands and without commands
                       whose plan goes past a URDF joint limit (the sim stops at the limit, the real
                       arm doesn't)
  past URDF limit      how far the plan (target_deg) goes past the limits of the URDF the replay
                       loaded
  at rest              hardware reading_urdf_deg - target_deg at the last command of the grasp hold
                       and of the release hold (the stretches plot_pybullet_replay.py shades), with
                       the RMS over each mode's runs (compensate / baseline / plain)
  peak speed           the PyBullet sim's and the hardware's (from the replay), and the plan's own
                       (target_deg) on the hardware's send times, which is what the slower loop
                       alone leaves of the planned speed; then hardware / that plan peak
  velocity error       the replay's same-command velocity RMS

Speeds use the replay's central difference (velocity_half_window samples either side, each series on
its own clock). Runs with timelines but no pybullet_replay.json are listed: replay them first with
compare_hardware_to_pybullet.py.

Usage (IKsim_mj env, from anywhere):
  python lowlevel/click_to_3d_oliver/summarise_pybullet_replays.py [run folders, or folders of runs ...]
Default: every run in lowlevel/real_world_runs/click_picks.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable

import numpy as np

LOWLEVEL_DIR = Path(__file__).resolve().parent.parent  # lowlevel/, which holds paths.py
if str(LOWLEVEL_DIR) not in sys.path:
    sys.path.insert(0, str(LOWLEVEL_DIR))

import paths  # noqa: E402
from compare_hardware_to_pybullet import REPLAY_SCHEMA, central_difference  # noqa: E402
from plot_pybullet_replay import held_spans  # noqa: E402

SHORT_NAMES = {
    "shoulder_pan": "pan",
    "shoulder_lift": "lift",
    "elbow_flex": "elbow",
    "wrist_flex": "wflex",
    "wrist_roll": "wroll",
}
MAX_LAG = 60  # commands; the lag search tries shifts 0 .. MAX_LAG - 1
TIMING_COLUMNS = ["loop ms", "write ms", "sleep ms", "read ms", "readmax", "retries", "x plan"]


def find_runs(locations: list[Path]) -> list[Path]:
    """Run folders (those with timeline_hardware.json): each location is a run or a folder of runs."""
    runs = []
    for location in locations:
        location = location.expanduser().resolve()
        if not location.is_dir():
            raise ValueError(f"{location} is not a folder")
        if (location / "timeline_hardware.json").is_file():
            runs.append(location)
        else:
            runs.extend(sorted(d for d in location.iterdir() if (d / "timeline_hardware.json").is_file()))
    return runs


def urdf_limits_deg(urdf: Path, joints: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper limits (deg) of these joints in the URDF.

    Only the <robot>'s own <joint> elements: the ones nested in <ros2_control> have no <limit>.
    """
    limits = {}
    for joint in ET.parse(urdf).getroot().findall("joint"):
        limit = joint.find("limit")
        if limit is not None:
            limits[joint.get("name")] = (math.degrees(float(limit.get("lower"))), math.degrees(float(limit.get("upper"))))
    return np.array([limits[name][0] for name in joints]), np.array([limits[name][1] for name in joints])


def best_lag(hardware: np.ndarray, sim: np.ndarray) -> tuple[int, float]:
    """The shift k with hardware[i] closest to sim[i - k] (least RMS over i), and that RMS."""
    rms = [np.sqrt(np.mean((hardware[k:] - sim[: len(sim) - k]) ** 2)) for k in range(MAX_LAG)]
    k = int(np.argmin(rms))
    return k, float(rms[k])


def analyse(run_dir: Path) -> dict[str, Any]:
    """Every measure in the module docstring for one run; per-joint ones as arrays over the arm joints."""
    replay = json.loads((run_dir / "pybullet_replay.json").read_text(encoding="utf-8"))
    if replay.get("schema") != REPLAY_SCHEMA:
        raise ValueError(f"{run_dir / 'pybullet_replay.json'} is not a {REPLAY_SCHEMA} file")
    sim_timeline = json.loads((run_dir / "timeline_sim.json").read_text(encoding="utf-8"))
    hw_timeline = json.loads((run_dir / "timeline_hardware.json").read_text(encoding="utf-8"))
    joints = replay["compared_joints"]
    n = len(joints)  # the compared joints are the first n columns of every joint array
    half = int(replay["velocity_half_window"])
    summary = replay["summary"]

    def per_joint(block: dict[str, dict[str, float]], key: str) -> np.ndarray:
        return np.array([block[name][key] for name in joints])

    if hw_timeline["compensated"]:
        mode = "compensate"
    elif hw_timeline["baseline"]:
        mode = "baseline"
    else:
        mode = "plain"
    out: dict[str, Any] = {"mode": mode, "settings": (replay["drive"], half)}

    t_send, t_sent, t_read_start, t_read_end = (
        np.asarray(hw_timeline[key], dtype=float) for key in ("t_send_s", "t_sent_s", "t_read_start_s", "t_read_end_s")
    )
    loop_ms = float(np.median(np.diff(t_send))) * 1e3
    out["timing"] = np.array(
        [
            loop_ms,
            float(np.median(t_sent - t_send)) * 1e3,
            float(np.median(t_read_start - t_sent)) * 1e3,
            float(np.median(t_read_end - t_read_start)) * 1e3,
            float(np.max(t_read_end - t_read_start)) * 1e3,
            float(np.sum(hw_timeline["read_retries"])),
            float(t_read_end[-1] / sim_timeline["t_read_end_s"][-1]),
        ]
    )

    out["same_time_rms"] = per_joint(summary["same_time"], "rms")
    out["same_command_rms"] = per_joint(summary["same_command"], "rms")
    out["same_command_max"] = per_joint(summary["same_command"], "max_abs")
    out["velocity_rms"] = per_joint(summary["velocity"]["same_command"], "rms")
    out["peak_sim"] = np.array([summary["velocity"]["peak_abs_deg_s"][name]["sim"] for name in joints])
    out["peak_hardware"] = np.array([summary["velocity"]["peak_abs_deg_s"][name]["hardware"] for name in joints])

    target = np.asarray(sim_timeline["target_deg"], dtype=float)[:, :n]
    lower, upper = urdf_limits_deg(Path(replay["urdf"]), joints)
    out["past_limit"] = np.maximum(np.maximum(lower - target, target - upper), 0.0).max(axis=0)
    past_any = ((target < lower) | (target > upper)).any(axis=1)  # per command, any joint

    same = replay["same_command"]
    sim = np.asarray(same["sim_deg"], dtype=float)[:, :n]
    hardware = np.asarray(same["hardware_deg"], dtype=float)[:, :n]
    lags, shifted, shifted_in_limits = [], [], []
    for j in range(n):
        k, rms = best_lag(hardware[:, j], sim[:, j])
        error = hardware[k:, j] - sim[: len(sim) - k, j]
        keep = ~past_any[k:] & ~past_any[: len(sim) - k]  # neither command i nor i - k past a limit
        lags.append(k)
        shifted.append(rms)
        shifted_in_limits.append(float(np.sqrt(np.mean(error[keep] ** 2))))
    out["lag"] = np.array(lags, dtype=float)
    out["lag_ms"] = out["lag"] * loop_ms
    out["shifted_rms"] = np.array(shifted)
    out["shifted_rms_in_limits"] = np.array(shifted_in_limits)

    index = list(hw_timeline["index"])
    reading = np.asarray(hw_timeline["reading_urdf_deg"], dtype=float)[:, :n]
    planned = np.asarray(hw_timeline["target_deg"], dtype=float)[:, :n]
    spans = held_spans(run_dir)
    for name in ("grasp", "release"):
        last = next((last for _, last, label in spans if label.startswith(name)), None)
        out[f"{name}_rest"] = np.full(n, np.nan) if last is None else reading[index.index(last)] - planned[index.index(last)]

    plan_on_hw_clock = central_difference(np.asarray(hw_timeline["target_deg"], dtype=float)[:, :n], t_send, half)
    out["peak_plan_hw_clock"] = np.nanmax(np.abs(plan_on_hw_clock), axis=0)
    out["peak_ratio"] = out["peak_hardware"] / out["peak_plan_hw_clock"]
    out["joints"] = joints
    return out


def print_table(
    title: str,
    names: list[str],
    modes: list[str],
    values: list[np.ndarray],
    columns: list[str],
    decimals: int,
    *,
    mode_rms: bool = False,
) -> None:
    """One row per run, then the median, min and max over runs (and each mode's RMS, if asked)."""
    width = max(len(name) for name in names)
    table = np.array(values)

    def line(label: str, mode: str, row: np.ndarray) -> str:
        return f"{label:{width}s}  {mode:10s} " + " ".join(f"{x:8.{decimals}f}" for x in row)

    print(f"\n== {title} ==")
    print(f"{'run':{width}s}  {'mode':10s} " + " ".join(f"{c:>8s}" for c in columns))
    for name, mode, row in zip(names, modes, table):
        print(line(name, mode, row))
    stats: list[tuple[str, Callable[..., np.ndarray]]] = [
        ("median", np.nanmedian),
        ("min", np.nanmin),
        ("max", np.nanmax),
    ]
    for label, stat in stats:
        print(line(label, "", stat(table, axis=0)))
    if mode_rms:
        for mode in sorted(set(modes)):
            rows = table[[m == mode for m in modes]]
            print(line(f"RMS over {mode} runs (n={len(rows)})", "", np.sqrt(np.nanmean(rows**2, axis=0))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "locations",
        type=Path,
        nargs="*",
        default=[paths.REAL_WORLD_RUNS_DIR / "click_picks"],
        help="run folders, or folders holding run folders (default: real_world_runs/click_picks)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = find_runs(args.locations)
    replayed = [run for run in runs if (run / "pybullet_replay.json").is_file()]
    missing = [run for run in runs if run not in replayed]
    print(f"{len(runs)} runs with timelines, {len(replayed)} replayed")
    for run in missing:
        print(f"  not replayed (run compare_hardware_to_pybullet.py on it first): {run.name}")
    if not replayed:
        return 1

    results = [analyse(run) for run in replayed]
    names = [run.name for run in replayed]
    modes = [r["mode"] for r in results]
    joints = results[0]["joints"]
    if any(r["joints"] != joints for r in results):
        raise ValueError("the replays compare different joints")
    settings = sorted({r["settings"] for r in results})
    print("replay settings (drive, velocity half window): " + ", ".join(f"{d}, {k}" for d, k in settings))
    if len(settings) > 1:
        print("  WARNING: the runs were replayed with different settings, so their numbers don't compare")
    columns = [SHORT_NAMES.get(name, name) for name in joints]

    print_table("loop timing, and run length over the plan's", names, modes, [r["timing"] for r in results], TIMING_COLUMNS, 2)
    tables = [
        ("same-time angle RMS (deg)", "same_time_rms", 1, False),
        ("same-command angle RMS (deg)", "same_command_rms", 2, False),
        ("same-command angle max |error| (deg)", "same_command_max", 1, False),
        ("lag behind the sim (commands)", "lag", 0, False),
        ("lag behind the sim (ms: commands x median time per command)", "lag_ms", 0, False),
        ("same-command RMS shifted by the lag (deg)", "shifted_rms", 2, False),
        ("same-command RMS shifted by the lag, without commands past a URDF limit (deg)", "shifted_rms_in_limits", 2, False),
        ("plan's furthest point past a URDF limit (deg)", "past_limit", 1, False),
        ("at rest, end of the grasp hold: hardware - plan (deg)", "grasp_rest", 2, True),
        ("at rest, end of the release hold: hardware - plan (deg)", "release_rest", 2, True),
        ("peak |velocity|: PyBullet sim (deg/s)", "peak_sim", 0, False),
        ("peak |velocity|: plan on the hardware's send times (deg/s)", "peak_plan_hw_clock", 0, False),
        ("peak |velocity|: hardware (deg/s)", "peak_hardware", 0, False),
        ("peak |velocity|: hardware / plan on the hardware's send times", "peak_ratio", 2, False),
        ("same-command velocity error RMS (deg/s)", "velocity_rms", 1, False),
    ]
    for title, key, decimals, mode_rms in tables:
        print_table(title, names, modes, [r[key] for r in results], columns, decimals, mode_rms=mode_rms)
    if any((r["lag"] == MAX_LAG - 1).any() for r in results):
        print(f"\nWARNING: a lag hit the search's edge ({MAX_LAG - 1} commands); raise MAX_LAG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
