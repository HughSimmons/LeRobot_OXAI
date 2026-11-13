#!/usr/bin/env python3
"""Run every friction sweep in a plan file in one go, keeping the arm powered between sweeps.

Each sweep moves to its reference pose, then steps one joint through its list of angles,
ramping each move in --move-steps commands and reading the angles after --hold-s.
Results are saved in toggle_joint_calibration.py's format (toggle_<joint>_calibration.json,
one folder per sweep) as soon as each sweep finishes, so fit_friction_sag_model.py can
read them.

Two differences from toggle_joint_calibration.py:
- It connects once, so the arm stays powered from the first sweep to the last, moves back
  to the trajectory home pose, and only then asks you to support it before releasing.
- The other joints are held at the reference pose's angles. The toggle script commands
  them to their last *readings*, and as a loaded joint always reads a little below its
  command, every step re-commands it lower: it creeps down by its sag each step (20+ deg
  over a sweep), which put the arm on the table in the first coverage run.

Without --execute nothing moves: it checks the whole plan in simulation, including the
moves from home, between sweeps and back to home (joint limits, clearance to the board and
to the table the base stands on). The sim uses the calibration offsets from the friction
fit, and checks each sweep with every gravity joint sagged as far as the fitted model
predicts. With --execute it drives the real arm; run that yourself and watch it. It also
stops if a reading comes within 20 mm of the table in the sim. Ctrl+C stops after the
current command, and either way it goes through the release prompt.

    python friction/run_friction_coverage_sweeps.py                    # simulation check only
    PYTHONPATH=lerobot/src python friction/run_friction_coverage_sweeps.py --execute --port <port>
    ... --execute --start-at 4                                # resume from sweep 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

# lowlevel/ on the path, for the shared arm modules and the friction package (as kinematicscalibration/ does).
LOWLEVEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LOWLEVEL_DIR))

from chess_traj import DEFAULT_HOME
from friction.fit_friction_sag_model import GRAVITY_JOINTS, GravityModel, JointParams, predicted_up_error
from probe_right_angle_pose import MIN_CLEARANCE_M, build_sim, min_clearance_to_board, set_pose
from probe_xy_pose import (
    action_from_joints,
    clamp_path_to_effective_limits,
    interpolated_steps,
    limit_violations,
    observation_joints_deg,
)
from toggle_joint_calibration import JOINT_NAMES, link_tilt_from_vertical_deg

DEFAULT_PLAN = LOWLEVEL_DIR / "real_world_runs" / "friction_coverage_poses" / "plan.json"
# Written by fit_friction_sag_model.py: calibration offsets and the M3 parameters used to predict sag.
DEFAULT_FIT = LOWLEVEL_DIR / "real_world_runs" / "friction" / "friction_sag_fit.json"
# The friction sweeps and fits so far all use this arm's calibration.
DEFAULT_ROBOT_ID = "orange_follower_arm"
DEFAULT_ROBOT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB9065986-if00"
# Same board as toggle_joint_calibration.py's defaults.
BOARD_ORIGIN = (0.29325, -0.01484, 0.0196)
SQUARE_SIZE = 0.04125
# The dry run flags a sweep that comes closer than this to the table (the arm may rest
# closer, e.g. at home, so moves into and out of home are allowed nearer).
MIN_SWEEP_TABLE_CLEARANCE_M = 0.03
MIN_TRANSIT_TABLE_CLEARANCE_M = 0.005
# On the arm, stop if a reading is closer than this to the table in the sim. In the first
# coverage run the sim put the arm at 0 to 2 mm when it started touching the table.
LIVE_MIN_TABLE_CLEARANCE_M = 0.02


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def load_plan(path: Path) -> list[dict]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    for sweep in plan:
        if sweep["joint"] not in JOINT_NAMES:
            raise ValueError(f"Unknown joint {sweep['joint']!r} in {path}")
        values = sweep["values"]
        if any(a == b for a, b in zip(values, values[1:])):
            raise ValueError(f"Sweep {sweep['name']} repeats an angle twice in a row, which isn't a move")
    return plan


def sweep_poses(sweep: dict) -> list[np.ndarray]:
    """The commanded pose at every value of a sweep (other joints at the reference pose)."""
    reference = np.array(sweep["reference_deg"], dtype=float)
    index = JOINT_NAMES.index(sweep["joint"])
    poses = []
    for value in sweep["values"]:
        pose = reference.copy()
        pose[index] = value
        poses.append(pose)
    return poses


# ---------------------------------------------------------------------------
# Simulation checks
# ---------------------------------------------------------------------------

def load_fit(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run fit_friction_sag_model.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def fit_offsets_deg(fit: dict) -> np.ndarray:
    """Calibration offsets from the fit (URDF angle = motor angle + offset), in JOINT_NAMES order."""
    return np.array([fit["offsets_deg"][name] for name in JOINT_NAMES], dtype=float)


class SagModel:
    """The sag the friction fit predicts at a commanded pose."""

    def __init__(self, fit: dict) -> None:
        self.offsets_deg = fit_offsets_deg(fit)
        values = {name: entry["value"] for name, entry in fit["parameters"].items()}
        self.params = {
            joint: JointParams(kp=values[f"{joint}.kp"], kc=values[f"{joint}.K_c"], kl=values[f"{joint}.K_l"])
            for joint in GRAVITY_JOINTS
        }
        self.gravity = GravityModel()

    def sagged(self, command_deg: np.ndarray, moving: tuple[str, float] | None = None) -> np.ndarray:
        """The readings to expect at a commanded pose, in motor degrees.

        Every gravity joint is sagged as far as M3 predicts for a lifting move (the larger of
        its two errors), since how a held joint last moved isn't known. The exception is
        `moving` = (joint, direction of the move that got it here, +1 or -1): that joint gets
        the case its direction gives, so one that arrived by lowering sags only as far as a
        lowering move does.
        """
        reading = np.asarray(command_deg, dtype=float).copy()
        for joint in GRAVITY_JOINTS:
            tau_hold = self.gravity.holding_torque(joint, command_deg, self.offsets_deg)
            s = 1.0 if tau_hold >= 0 else -1.0  # +1 if up is +theta
            lifting = moving is None or moving[0] != joint or moving[1] == s
            e_up, _ = predicted_up_error(abs(tau_hold), lifting=lifting, joint=self.params[joint])
            # Gravity pulls toward -s, so reading = command - s * e_up.
            reading[JOINT_NAMES.index(joint)] -= float(np.rad2deg(s * e_up))
        return reading


class SimChecks:
    """Joint limits and clearance to the board and table, on the toggle script's sim robot.

    Poses are motor angles, as commanded or read; the calibration offsets are added before
    they're put in the sim."""

    def __init__(self, offsets_deg: np.ndarray) -> None:
        self.offsets_deg = offsets_deg
        self.robot, self.board = build_sim(BOARD_ORIGIN, SQUARE_SIZE)
        # The base stands on the table: put a plane at its lowest point.
        table_z = p.getAABB(self.robot, -1)[0][2]
        plane = p.createCollisionShape(p.GEOM_PLANE)
        self.table = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane, basePosition=[0, 0, table_z])

    def table_clearance(self, pose: np.ndarray) -> float:
        set_pose(self.robot, pose + self.offsets_deg)
        p.performCollisionDetection()
        closest = np.inf
        for link in range(1, p.getNumJoints(self.robot)):  # every link beyond the shoulder
            points = p.getClosestPoints(self.robot, self.table, distance=0.5, linkIndexA=link)
            if points:
                closest = min(closest, min(c[8] for c in points))
        return float(closest)

    def check(self, poses: list[np.ndarray]) -> tuple[list[str], float, float]:
        """Joint-limit violations, and the closest approach to the board and to the table (m)."""
        violations = sorted({v for pose in poses for v in limit_violations(pose)})
        board = min(self.board_clearance(pose) for pose in poses)
        table = min(self.table_clearance(pose) for pose in poses)
        return violations, board, table

    def board_clearance(self, pose: np.ndarray) -> float:
        set_pose(self.robot, pose + self.offsets_deg)
        return min_clearance_to_board(self.robot, self.board)


def check_plan(plan: list[dict], start_at: int, move_steps: int, checks: SimChecks, sag: SagModel) -> bool:
    """Check every sweep and every move between them, starting from the home pose.

    Joint limits are checked on the commanded poses; clearances on the poses the arm is
    expected to sag to (SagModel.sagged: worst case for every joint, except the swept joint
    within a sweep, which gets the case of the direction it arrived from)."""
    ok = True
    previous = np.array(DEFAULT_HOME, dtype=float)

    def report(label: str, poses: list[np.ndarray], min_table: float, sagged: list[np.ndarray] | None = None) -> None:
        nonlocal ok
        violations = sorted({v for pose in poses for v in limit_violations(pose)})
        if sagged is None:
            sagged = [sag.sagged(pose) for pose in poses]
        _, board, table = checks.check(sagged)
        passed = not violations and board >= MIN_CLEARANCE_M and table >= min_table
        ok &= passed
        # min_clearance_to_board only searches 50 mm out, so 50 means "at least 50".
        board_text = f"{board * 1000:4.0f} mm" if board < 0.05 else ">=50 mm"
        print(f"  {label:<36} board {board_text}   table {table * 1000:4.0f} mm   "
              f"{'OK' if passed else 'CHECK ' + ', '.join(violations)}")

    for sweep in plan:
        if sweep["order"] < start_at:
            continue
        poses = sweep_poses(sweep)
        # Leave out the first few steps out of home, where the arm rests almost on the table.
        transit = interpolated_steps(previous, poses[0], move_steps)
        if np.array_equal(previous, np.array(DEFAULT_HOME, dtype=float)):
            transit = transit[move_steps // 10:]
        print(f"#{sweep['order']} {sweep['name']} ({sweep['role']}): {sweep['joint']} "
              f"{min(sweep['values']):+.1f} to {max(sweep['values']):+.1f} deg, {len(sweep['values'])} readings")
        report("move to the reference pose", transit, MIN_TRANSIT_TABLE_CLEARANCE_M)
        # The first value is reached from the transit, so its direction is unknown: worst case.
        directions = [None, *np.sign(np.diff(sweep["values"]))]
        sagged = [sag.sagged(pose, None if d is None else (sweep["joint"], float(d)))
                  for pose, d in zip(poses, directions)]
        report("sweep", poses, MIN_SWEEP_TABLE_CLEARANCE_M, sagged)
        previous = poses[-1]
    back = interpolated_steps(previous, np.array(DEFAULT_HOME, dtype=float), move_steps)
    print("return to home")
    report("move back to home", back[: -move_steps // 10], MIN_TRANSIT_TABLE_CLEARANCE_M)
    return ok


# ---------------------------------------------------------------------------
# Running on the arm
# ---------------------------------------------------------------------------

class ArmSession:
    """One connection to the follower arm: send interpolated moves and read joints back."""

    # Check the readings against the table every this many commands while moving.
    GUARD_EVERY = 10

    def __init__(self, follower, checks: SimChecks, command_delay: float, move_steps: int) -> None:
        self.follower = follower
        self.checks = checks
        self.command_delay = command_delay
        self.move_steps = move_steps
        self.observation = follower.get_observation()
        # Moves ramp from the last command, not the last reading: a loaded joint reads below
        # its command, so starting from the reading would command it lower each move.
        self.command = self.joints_deg()

    def joints_deg(self) -> np.ndarray:
        return observation_joints_deg(self.observation)

    def move_to(self, target_deg: np.ndarray, label: str, guard: bool = True, on_step=None) -> None:
        """Ramp from the last command to target. With guard, stop if the arm nears the table
        (off for moves out of and into home, where the arm rests close to it). on_step, if
        given, is called after every command once its reading is in self.observation."""
        path = interpolated_steps(self.command, np.asarray(target_deg, dtype=float), self.move_steps)
        clipped, _ = clamp_path_to_effective_limits(path)
        violations = sorted({v for step in clipped for v in limit_violations(step)})
        if violations:
            raise RuntimeError(f"Move to {label} exceeds joint limits: " + "; ".join(violations))
        clearance = min(self.checks.board_clearance(step) for step in clipped)
        if clearance < MIN_CLEARANCE_M:
            print(f"  WARNING: move to {label} comes within {clearance * 1000:.1f} mm of the board (advisory only).")
        for k, step in enumerate(clipped, start=1):
            action = self.observation.copy()
            action.update(action_from_joints(step))
            self.follower.send_action(action)
            self.command = step
            time.sleep(self.command_delay)
            self.observation = self.follower.get_observation()
            if on_step is not None:
                on_step()
            if guard and k % self.GUARD_EVERY == 0:
                self.check_table(f"moving to {label}")

    def settle(self, hold_s: float, guard: bool = True) -> np.ndarray:
        time.sleep(hold_s)
        self.observation = self.follower.get_observation()
        if guard:
            self.check_table("holding")
        return self.joints_deg()

    def check_table(self, label: str) -> None:
        clearance = self.checks.table_clearance(self.joints_deg())
        if clearance < LIVE_MIN_TABLE_CLEARANCE_M:
            raise RuntimeError(
                f"while {label}, the readings put the arm {clearance * 1000:.1f} mm from the table "
                f"(limit {LIVE_MIN_TABLE_CLEARANCE_M * 1000:.0f} mm); holding it there"
            )


def run_sweep(arm: ArmSession, sweep: dict, hold_s: float, output_root: Path, from_home: bool) -> Path:
    """Move to the sweep's reference pose, step its joint through the values, save the readings."""
    joint = sweep["joint"]
    index = JOINT_NAMES.index(joint)
    print(f"\n#{sweep['order']} {sweep['name']} ({sweep['role']}): {joint} through {sweep['values']}")
    # Out of home the arm starts almost on the table, so the guard only starts once it's at the reference pose.
    arm.move_to(np.array(sweep["reference_deg"], dtype=float), "the reference pose", guard=not from_home)
    reference = arm.settle(hold_s)
    print(f"  at reference pose: {np.round(reference, 2).tolist()}")

    results = []
    for k, target in enumerate(sweep_poses(sweep), start=1):
        value = float(target[index])
        arm.move_to(target, f"{joint}={value}")
        settled = arm.settle(hold_s)
        entry = {"target": value, "observed_deg": settled.tolist(), "diff": float(settled[index] - value)}
        line = f"  [{k}/{len(sweep['values'])}] target {value:+7.2f}  observed {settled[index]:+7.2f}  diff {entry['diff']:+.3f}"
        if joint == "shoulder_lift":
            entry["predicted_upper_arm_tilt_from_vertical_deg"] = link_tilt_from_vertical_deg(arm.checks.robot, target)
        print(line)
        results.append(entry)

    output_dir = output_root / sweep["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"toggle_{joint}_calibration.json"
    out_path.write_text(json.dumps({"joint": joint, "results": results}, indent=2), encoding="utf-8")
    print(f"  saved: {out_path}")
    return out_path


def execute(args: argparse.Namespace, plan: list[dict], checks: SimChecks) -> int:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    to_run = [s for s in plan if s["order"] >= args.start_at]
    print(f"\nAbout to run {len(to_run)} sweeps on the real arm ({args.robot_id} on {args.port}), "
          f"about a minute each, then return to home.")
    print("Clear the table beside the arm and keep a hand ready to stop it.")
    if input("Start? [yes/no]: ").strip().lower() not in {"yes", "y"}:
        print("Cancelled.")
        return 1

    follower = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
    follower.connect()
    status = 0
    current = None  # the sweep in progress, which isn't saved if it's interrupted
    try:
        arm = ArmSession(follower, checks, args.command_delay, args.move_steps)
        for k, sweep in enumerate(to_run):
            current = sweep
            run_sweep(arm, sweep, args.hold_s, LOWLEVEL_DIR, from_home=k == 0)
        current = None
        print("\nAll sweeps done. Moving back to home.")
        arm.move_to(np.array(DEFAULT_HOME, dtype=float), "home", guard=False)
        arm.settle(args.hold_s, guard=False)
    except (KeyboardInterrupt, Exception) as exc:
        reason = "Stopped" if isinstance(exc, KeyboardInterrupt) else f"Stopped by an error: {exc}"
        resume = f" Resume with --start-at {current['order']}." if current else ""
        print(f"\n{reason}. Sweeps finished before this one are saved.{resume}")
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
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--fit", type=Path, default=DEFAULT_FIT,
                        help="fit_friction_sag_model.py output, for the calibration offsets and the predicted sag.")
    parser.add_argument("--start-at", type=int, default=1, help="Sweep number (the plan's 'order') to start from.")
    parser.add_argument("--execute", action="store_true", help="Drive the real arm. Without it, only check in simulation.")
    parser.add_argument("--port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    # Keep these at toggle_joint_calibration.py's defaults so the readings match the earlier sweeps.
    parser.add_argument("--hold-s", type=float, default=2.0)
    parser.add_argument("--move-steps", type=int, default=200)
    parser.add_argument("--command-delay", type=float, default=1.0 / 240.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = load_plan(args.plan)
    fit = load_fit(args.fit)
    # SimChecks first: build_sim puts everything in PyBullet's default client, which is the
    # first one connected, while GravityModel (in SagModel) passes its own client id.
    checks = SimChecks(fit_offsets_deg(fit))
    sag = SagModel(fit)
    print(f"Checking {args.plan} in simulation, from the home pose, with the arm sagged as {args.fit.name} predicts:")
    ok = check_plan(plan, args.start_at, args.move_steps, checks, sag)
    if not ok:
        print("\nSome checks failed; fix the plan before running it on the arm.")
        return 1
    if not args.execute:
        print("\nAll checks passed. Dry run only; add --execute to run the sweeps on the real arm.")
        return 0
    return execute(args, plan, checks)


if __name__ == "__main__":
    raise SystemExit(main())
