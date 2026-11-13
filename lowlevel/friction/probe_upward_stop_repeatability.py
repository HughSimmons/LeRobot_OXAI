#!/usr/bin/env python3
"""Repeat one heavy upward move of the shoulder lift, logging the joint throughout, to find
out why such moves stop in different places.

In the coverage sweeps the lift, moving up with the arm stretched out (03_lift_far_reach),
stopped up to 2 deg apart between two cycles of identical commands, while downward moves
repeated to within about 0.4 deg. The sweeps read the joint once, 2 s after each move,
which can't say why. This probe repeats the move and logs the lift all the way, so each
candidate cause shows up separately:

  genuine randomness   repeats that start from the same place still stop apart
  start dependence     repeats from start A and from start B stop differently on average
  momentum             the joint is still moving up when the ramp ends
  stick-slip           the position moves in steps during the ramp, and the load jumps
  still settling       the position is still changing 2 s after the ramp
  torque limit         the load sits flat at its maximum during the move

Each repeat, from the pose of coverage sweep 3 (its elbow, wrist, pan, roll and gripper):
  1. ramp the lift down to the start target (A: 70.8 deg, B: 65.8 deg) and hold
     --start-hold-s. A and B alternate, so a slow drift such as warming affects both alike.
  2. ramp it up to 55.8 deg the way the sweeps move (--move-steps commands, --command-delay
     apart) and hold --hold-s.
The lift's position is logged after every command and, during holds, as fast as the bus
allows. Its Present_Velocity, Present_Load and Present_Current registers are read every
--extras-every samples and saved raw, as the servo reports them. Each register read is
one more bus round trip, which stretches the ramp slightly, so every sample is timestamped.

These are sweep 3's poses and lift range, which it ran with the readings no closer than
62 mm to the table in the sim. Without --execute it only runs the sim check from
run_friction_coverage_sweeps.py on the moves. With --execute it drives the real arm (run
that yourself), stops if a reading comes within 20 mm of the table, and saves after every
repeat to real_world_runs/upward_stop_probe/<date-time>.json.

    python friction/probe_upward_stop_repeatability.py              # simulation check only
    PYTHONPATH=lerobot/src python friction/probe_upward_stop_repeatability.py --execute
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# lowlevel/ on the path, for the shared arm modules and the friction package (as kinematicscalibration/ does).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_traj import DEFAULT_HOME
from friction.run_friction_coverage_sweeps import (
    DEFAULT_FIT,
    DEFAULT_ROBOT_ID,
    DEFAULT_ROBOT_PORT,
    LOWLEVEL_DIR,
    ArmSession,
    SagModel,
    SimChecks,
    check_plan,
    fit_offsets_deg,
    load_fit,
)
from toggle_joint_calibration import JOINT_NAMES

REFERENCE_POSE_FILE = LOWLEVEL_DIR / "real_world_runs" / "friction_coverage_poses" / "03_lift_far_reach.json"
OUTPUT_DIR = LOWLEVEL_DIR / "real_world_runs" / "upward_stop_probe"
LIFT = "shoulder_lift"
LIFT_INDEX = JOINT_NAMES.index(LIFT)
UP_TARGET_DEG = 55.8
START_TARGETS_DEG = {"A": 70.8, "B": 65.8}
EXTRA_REGISTERS = ("Present_Velocity", "Present_Load", "Present_Current")
# The sweeps read each joint this long after the end of the move.
SWEEP_READ_AFTER_S = 2.0


def load_reference() -> np.ndarray:
    return np.array(json.loads(REFERENCE_POSE_FILE.read_text(encoding="utf-8"))["joints_deg"], dtype=float)


def with_lift(pose: np.ndarray, lift_deg: float) -> np.ndarray:
    out = pose.copy()
    out[LIFT_INDEX] = lift_deg
    return out


def repeat_order(repeats: int) -> list[str]:
    return ["A", "B"] * repeats


def as_sweep(reference: np.ndarray, order: list[str]) -> dict:
    """The probe's moves in the form check_plan takes: the lift through every target in turn."""
    values = [float(reference[LIFT_INDEX])]
    for label in order:
        values += [START_TARGETS_DEG[label], UP_TARGET_DEG]
    return {"order": 1, "name": "upward_stop_repeats", "role": "probe", "joint": LIFT,
            "reference_deg": reference.tolist(), "values": values}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class LiftLog:
    """Samples of the lift during one repeat: position after every command or bus read, and
    the extra registers every `extras_every` samples."""

    def __init__(self, arm: ArmSession, extras_every: int) -> None:
        self.arm = arm
        self.extras_every = extras_every
        self.t0 = time.perf_counter()
        self.rows: list[dict] = []

    def sample(self, phase: str) -> None:
        row = {
            "t_s": time.perf_counter() - self.t0,
            "phase": phase,
            "command_deg": float(self.arm.command[LIFT_INDEX]),
            "position_deg": float(self.arm.joints_deg()[LIFT_INDEX]),
        }
        if len(self.rows) % self.extras_every == 0:
            for register in EXTRA_REGISTERS:
                row[register] = int(self.arm.follower.bus.sync_read(register, [LIFT])[LIFT])
        self.rows.append(row)


def hold_logged(arm: ArmSession, seconds: float, log: LiftLog, phase: str) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        arm.observation = arm.follower.get_observation()
        log.sample(phase)
    arm.check_table("holding")


def summarise(rows: list[dict]) -> dict:
    """Where the lift started and stopped, and how it was moving, from one repeat's samples."""
    up = [r for r in rows if r["phase"] == "up"]
    hold = [r for r in rows if r["phase"] == "hold"]
    ramp_end = up[-1]["t_s"]
    at_read = next((r for r in hold if r["t_s"] >= ramp_end + SWEEP_READ_AFTER_S), hold[-1])
    # Velocity at the end of the ramp: slope of the position over its last 50 ms.
    tail = [r for r in up if r["t_s"] >= ramp_end - 0.05]
    velocity = (float(np.polyfit([r["t_s"] for r in tail], [r["position_deg"] for r in tail], 1)[0])
                if len(tail) >= 3 else float("nan"))
    loads = [abs(r["Present_Load"]) for r in up if "Present_Load" in r]
    return {
        "start_position_deg": [r for r in rows if r["phase"] == "start_hold"][-1]["position_deg"],
        "ramp_duration_s": ramp_end - up[0]["t_s"],
        "velocity_at_ramp_end_deg_s": velocity,
        # Sag = reading - target: positive is drooped, as in the sweeps' "diff".
        "sag_at_2s_deg": at_read["position_deg"] - UP_TARGET_DEG,
        "sag_at_end_deg": hold[-1]["position_deg"] - UP_TARGET_DEG,
        "max_abs_load_raw_during_ramp": max(loads) if loads else None,
        "samples_per_s_in_hold": (len(hold) - 1) / (hold[-1]["t_s"] - hold[0]["t_s"]) if len(hold) > 1 else None,
    }


# ---------------------------------------------------------------------------
# Running on the arm
# ---------------------------------------------------------------------------

def run_repeat(arm: ArmSession, reference: np.ndarray, label: str, args: argparse.Namespace) -> dict:
    log = LiftLog(arm, args.extras_every)
    start = START_TARGETS_DEG[label]
    arm.move_to(with_lift(reference, start), f"{LIFT}={start}", on_step=lambda: log.sample("to_start"))
    hold_logged(arm, args.start_hold_s, log, "start_hold")
    arm.move_to(with_lift(reference, UP_TARGET_DEG), f"{LIFT}={UP_TARGET_DEG}", on_step=lambda: log.sample("up"))
    hold_logged(arm, args.hold_s, log, "hold")
    return {"label": label, "start_target_deg": start, "up_target_deg": UP_TARGET_DEG,
            "summary": summarise(log.rows), "samples": log.rows}


def print_repeat(k: int, total: int, record: dict) -> None:
    s = record["summary"]
    print(f"  [{k}/{total}] {record['label']}: started at {s['start_position_deg']:6.2f}, "
          f"sag at 2 s {s['sag_at_2s_deg']:+5.2f}, at end {s['sag_at_end_deg']:+5.2f}, "
          f"velocity at ramp end {s['velocity_at_ramp_end_deg_s']:+6.1f} deg/s (negative is up), "
          f"ramp {s['ramp_duration_s']:.2f} s")


def print_groups(records: list[dict]) -> None:
    print("\nSag at 2 s (deg, reading - 55.8):")
    for label, start in START_TARGETS_DEG.items():
        sags = [r["summary"]["sag_at_2s_deg"] for r in records if r["label"] == label]
        if sags:
            sd = f"sd {np.std(sags, ddof=1):.2f}" if len(sags) > 1 else "one repeat"
            print(f"  {label} (start {start}): mean {np.mean(sags):+.2f}, {sd}, range {min(sags):+.2f} to {max(sags):+.2f}")


def save(path: Path, args: argparse.Namespace, reference: np.ndarray, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "script": "probe_upward_stop_repeatability.py",
        "robot_id": args.robot_id,
        "reference_deg": reference.tolist(),
        "up_target_deg": UP_TARGET_DEG,
        "start_targets_deg": START_TARGETS_DEG,
        "move_steps": args.move_steps,
        "command_delay_s": args.command_delay,
        "start_hold_s": args.start_hold_s,
        "hold_s": args.hold_s,
        "extras_every": args.extras_every,
        "extra_registers_note": "raw servo values, sign-decoded by LeRobot, not converted to units",
        "repeats": records,
    }
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def execute(args: argparse.Namespace, reference: np.ndarray, order: list[str], checks: SimChecks) -> int:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    out_path = OUTPUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    print(f"\nAbout to make {len(order)} upward moves of the lift on the real arm "
          f"({args.robot_id} on {args.port}), about 11 s each, then return to home.")
    print("Clear the table beside the arm and keep a hand ready to stop it.")
    if input("Start? [yes/no]: ").strip().lower() not in {"yes", "y"}:
        print("Cancelled.")
        return 1

    follower = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
    follower.connect()
    status = 0
    records: list[dict] = []
    try:
        arm = ArmSession(follower, checks, args.command_delay, args.move_steps)
        arm.move_to(reference, "the sweep 3 pose", guard=False)
        arm.settle(args.start_hold_s)
        for k, label in enumerate(order, start=1):
            records.append(run_repeat(arm, reference, label, args))
            save(out_path, args, reference, records)
            print_repeat(k, len(order), records[-1])
        print_groups(records)
        print(f"\nsaved: {out_path}\nMoving back to home.")
        arm.move_to(np.array(DEFAULT_HOME, dtype=float), "home", guard=False)
        arm.settle(args.start_hold_s, guard=False)
    except (KeyboardInterrupt, Exception) as exc:
        reason = "Stopped" if isinstance(exc, KeyboardInterrupt) else f"Stopped by an error: {exc}"
        saved = f" The {len(records)} finished repeats are saved in {out_path}." if records else ""
        print(f"\n{reason}.{saved}")
        status = 1
    finally:
        # Disconnecting disables torque, so the arm drops unless it's at rest or held.
        try:
            input("Support the arm (unless it's resting at home), then press Enter to release it...")
        except (EOFError, KeyboardInterrupt):
            pass
        follower.disconnect()
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="Drive the real arm. Without it, only check in simulation.")
    parser.add_argument("--repeats", type=int, default=10, help="Repeats from each start (A and B alternate).")
    parser.add_argument("--start-hold-s", type=float, default=3.0, help="Hold at the start position before moving up.")
    parser.add_argument("--hold-s", type=float, default=5.0, help="Hold after the upward move, logging.")
    parser.add_argument("--extras-every", type=int, default=5, help="Read velocity, load and current every this many samples.")
    parser.add_argument("--fit", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    # The sweeps' values, so the upward move is the one they made.
    parser.add_argument("--move-steps", type=int, default=200)
    parser.add_argument("--command-delay", type=float, default=1.0 / 240.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fit = load_fit(args.fit)
    # SimChecks first: build_sim takes PyBullet's default client (see run_friction_coverage_sweeps.py).
    checks = SimChecks(fit_offsets_deg(fit))
    sag = SagModel(fit)
    reference = load_reference()
    order = repeat_order(args.repeats)
    print(f"Checking the probe's moves in simulation, from the home pose, with the arm sagged as {args.fit.name} predicts:")
    if not check_plan([as_sweep(reference, order)], 1, args.move_steps, checks, sag):
        print("\nSome checks failed; don't run this on the arm.")
        return 1
    if not args.execute:
        print("\nAll checks passed. Dry run only; add --execute to run it on the real arm.")
        return 0
    return execute(args, reference, order, checks)


if __name__ == "__main__":
    raise SystemExit(main())
