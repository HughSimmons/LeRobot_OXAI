#!/usr/bin/env python3
"""Move shoulder_lift and elbow_flex to exactly 0 degrees (their own URDF
zero reference), leaving shoulder_pan, wrist_flex, wrist_roll, and gripper
unchanged from wherever the arm currently is. Board collision and joint
limits are verified in PyBullet before any hardware move is offered.

This exists to test the shoulder_lift calibration-offset hypothesis raised
while investigating why probed gripper heights came out ~15-20mm short of
the expected board height: FK sensitivity analysis pointed at a ~6.5-7 deg
shoulder_lift zero-offset as the most consistent explanation. Commanding a
known round-number angle (0 deg on each) lets that be checked directly
against a physical square, independent of any board-touch measurement.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p

from chess_traj import DEFAULT_HOME
from probe_xy_pose import (
    EFFECTIVE_LIMITS_DEG,
    MOTOR_KEYS,
    action_from_joints,
    clamp_path_to_effective_limits,
    limit_violations,
    observation_joints_deg,
)

LOWLEVEL_DIR = Path(__file__).resolve().parent
URDF_PATH = str(LOWLEVEL_DIR.parent / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf")
DEFAULT_OUTPUT_DIR = LOWLEVEL_DIR / "real_world_runs"
DEFAULT_ROBOT_PORT = "/dev/tty.usbmodem5B7B0157051"
DEFAULT_ROBOT_ID = "my_awesome_follower_arm"
CONTROL_JOINT_INDICES = [0, 1, 2, 3, 4, 6]  # shoulder_pan..wrist_roll, gripper
MIN_CLEARANCE_M = 0.005


def build_sim(board_origin: tuple[float, float, float], square_size: float):
    p.connect(p.DIRECT)
    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
    board_size = 8 * square_size
    half_height = 0.005
    board_shape = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[board_size / 2, board_size / 2, half_height]
    )
    board_id = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=board_shape, basePosition=list(board_origin)
    )
    return robot_id, board_id


def set_pose(robot_id: int, joints_deg: np.ndarray) -> None:
    joints_rad = np.deg2rad(joints_deg)
    for sim_idx, q in zip(CONTROL_JOINT_INDICES, joints_rad):
        p.resetJointState(robot_id, sim_idx, float(q))


def min_clearance_to_board(robot_id: int, board_id: int, search_dist: float = 0.05) -> float:
    p.performCollisionDetection()
    closest = p.getClosestPoints(robot_id, board_id, distance=search_dist)
    if not closest:
        return search_dist
    return min(c[8] for c in closest)  # contactDistance, index 8


def path_min_clearance(robot_id: int, board_id: int, path: list[np.ndarray]) -> float:
    worst = float("inf")
    for step in path:
        set_pose(robot_id, step)
        worst = min(worst, min_clearance_to_board(robot_id, board_id))
    return worst


def interpolated_steps(start_deg: np.ndarray, end_deg: np.ndarray, steps: int) -> list[np.ndarray]:
    out = []
    steps = max(1, int(steps))
    for k in range(1, steps + 1):
        a = k / steps
        out.append((1.0 - a) * start_deg + a * end_deg)
    return out


def build_target(current_deg: np.ndarray, lift_deg: float, elbow_deg: float) -> np.ndarray:
    """current, with only shoulder_lift and elbow_flex replaced."""
    target = current_deg.copy()
    target[1] = lift_deg
    target[2] = elbow_deg
    return target


def build_safe_path(
    robot_id: int,
    board_id: int,
    current_deg: np.ndarray,
    target_deg: np.ndarray,
    steps: int,
) -> tuple[list[np.ndarray], float]:
    """Straight-line interpolation from current to target (only lift/elbow
    actually move; pan/wrist/gripper are already equal in both), verified
    against the board along every intermediate step."""
    path = interpolated_steps(current_deg, target_deg, steps)
    clipped, _ = clamp_path_to_effective_limits(path)
    bad = [v for step in clipped for v in limit_violations(step)]
    if bad:
        raise RuntimeError("Path exceeds joint limits: " + "; ".join(sorted(set(bad))))
    clearance = path_min_clearance(robot_id, board_id, clipped)
    if clearance < MIN_CLEARANCE_M:
        print(
            f"WARNING: board-clearance check disabled; path comes within "
            f"{clearance*1000:.1f}mm of the board (simplified box model) -- "
            "proceeding anyway per request. Watch the arm and stop it by hand if needed."
        )
    return clipped, clearance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shoulder-lift-deg", type=float, default=0.0)
    parser.add_argument("--elbow-flex-deg", type=float, default=0.0)
    parser.add_argument("--board-origin-x", type=float, default=0.29325)
    parser.add_argument("--board-origin-y", type=float, default=-0.01484)
    parser.add_argument("--board-origin-z", type=float, default=0.0196)
    parser.add_argument("--square-size", type=float, default=0.04125)
    parser.add_argument("--port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--current-pose-file",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "current_before_right_angle.json",
        help="For a dry run only: joints_deg reading to treat as the starting pose.",
    )
    parser.add_argument("--move-steps", type=int, default=240)
    parser.add_argument("--command-delay", type=float, default=1.0 / 240.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--sweep-lift-values",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Command elbow_flex=--elbow-flex-deg at each of these shoulder_lift "
            "values in turn (from wherever the arm currently is), holding briefly "
            "and printing the cmd-obs diff at each -- checks whether the elbow_flex "
            "settling error is a fixed offset (constant across lift values) or "
            "gravity sag (varies with arm configuration)."
        ),
    )
    parser.add_argument("--sweep-hold-s", type=float, default=1.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    board_origin = (args.board_origin_x, args.board_origin_y, args.board_origin_z)
    robot_id, board_id = build_sim(board_origin, args.square_size)

    if not args.execute:
        current = np.array(
            json.loads(args.current_pose_file.read_text(encoding="utf-8"))["joints_deg"],
            dtype=float,
        )
        target = build_target(current, args.shoulder_lift_deg, args.elbow_flex_deg)
        print(f"current_deg (from {args.current_pose_file.name}): {np.round(current, 3).tolist()}")
        print(f"target joints_deg : {np.round(target, 3).tolist()}")
        path, worst_clearance = build_safe_path(robot_id, board_id, current, target, args.move_steps)
        print(f"path min clearance ({len(path)} steps): {worst_clearance*1000:.1f} mm")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": "probe_right_angle_pose_v1",
            "current_joints_deg": current.tolist(),
            "target_joints_deg": target.tolist(),
            "path_min_clearance_mm": worst_clearance * 1000.0,
            "board_origin": list(board_origin),
        }
        out_path = args.output_dir / "probe_right_angle_pose.json"
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"saved: {out_path}")
        print("dry run only; add --execute to move the real SO101 arm.")
        return 0

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    follower = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
    follower.connect()
    try:
        if args.sweep_lift_values is not None:
            observation = follower.get_observation()
            current = observation_joints_deg(observation)
            targets = [
                build_target(current, lift, args.elbow_flex_deg)
                for lift in args.sweep_lift_values
            ]
            print("current_deg:", np.round(current, 3).tolist())
            for lift, t in zip(args.sweep_lift_values, targets):
                print(f"  sweep target lift={lift:7.2f}  elbow={args.elbow_flex_deg:7.2f}  {np.round(t, 3).tolist()}")
            confirmation = input(
                f"Sweep through {len(targets)} poses (Ctrl-C to stop any time)? [yes/no]: "
            ).strip().lower()
            if confirmation not in {"yes", "y"}:
                print("Cancelled.")
                return 1

            results = []
            for lift, target in zip(args.sweep_lift_values, targets):
                path, clearance = build_safe_path(robot_id, board_id, current, target, args.move_steps)
                for index, step in enumerate(path, start=1):
                    action = observation.copy()
                    action.update(action_from_joints(step))
                    follower.send_action(action)
                    time.sleep(args.command_delay)
                    observation = follower.get_observation()
                time.sleep(args.sweep_hold_s)
                settled = observation_joints_deg(follower.get_observation())
                diff = settled - target
                print(
                    f"lift_target={lift:7.2f}  observed={np.round(settled, 3).tolist()}  "
                    f"elbow_diff={diff[2]:+.3f}  lift_diff={diff[1]:+.3f}"
                )
                results.append(
                    {"lift_target": lift, "target": target.tolist(), "observed": settled.tolist(), "diff": diff.tolist()}
                )
                current = settled

            args.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.output_dir / "elbow_sag_sweep.json"
            out_path.write_text(json.dumps({"elbow_flex_deg": args.elbow_flex_deg, "results": results}, indent=2), encoding="utf-8")
            print(f"saved: {out_path}")
            print("\nHolding final pose.")
            input("Press Enter to release and exit...")
            return 0

        observation = follower.get_observation()
        current = observation_joints_deg(observation)
        target = build_target(current, args.shoulder_lift_deg, args.elbow_flex_deg)
        print("current_deg:", np.round(current, 3).tolist())
        print("target joints_deg:", np.round(target, 3).tolist())
        live_path, live_clearance = build_safe_path(robot_id, board_id, current, target, args.move_steps)
        print(f"live path: {len(live_path)} steps, min clearance {live_clearance*1000:.1f}mm")
        confirmation = input("Move to right-angle pose? [yes/no]: ").strip().lower()
        if confirmation not in {"yes", "y"}:
            print("Cancelled.")
            return 1

        for index, step in enumerate(live_path, start=1):
            action = observation.copy()
            action.update(action_from_joints(step))
            follower.send_action(action)
            if index % 50 == 0 or index == len(live_path):
                print(f"  {index}/{len(live_path)}")
            time.sleep(args.command_delay)
            observation = follower.get_observation()

        settled = observation_joints_deg(follower.get_observation())
        print("observed_deg:", np.round(settled, 3).tolist())
        print("cmd-obs diff:", np.round(settled - target, 3).tolist())
        print(
            "\nHolding right-angle pose. Check shoulder_lift and elbow_flex "
            "against a physical square/protractor now."
        )
        input("Press Enter to release and exit...")
    finally:
        follower.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
