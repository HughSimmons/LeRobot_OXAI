#!/usr/bin/env python3
"""Point the gripper straight down at a world XY, just above the playing surface.

A calibration probe: instead of inferring the robot-to-board frame from a chain
of measurements, command the arm to a known world XY and look at where the
gripper actually ends up. The difference between the commanded square and the
observed one is the frame error, read directly.

The pose is solved with the same IK the trajectory builder uses, seeded from the
trajectory home pose, so a probe result is comparable with lookup trajectories.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from board_coordinates import (
    DEFAULT_SQUARE_SIZE,
    location_world_xy,
    validate_square,
)
from scipy.optimize import least_squares

from chess_traj import DEFAULT_HOME
from testkinematics import kinematics

LOWLEVEL_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = LOWLEVEL_DIR / "real_world_runs"
DEFAULT_ROBOT_PORT = "/dev/tty.usbmodem5B7B0157051"
DEFAULT_ROBOT_ID = "my_awesome_follower_arm"
MOTOR_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def action_from_joints(joints_deg: np.ndarray) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in zip(MOTOR_KEYS, np.asarray(joints_deg, dtype=float))
    }


def observation_joints_deg(observation: dict[str, float]) -> np.ndarray:
    return np.array([float(observation[key]) for key in MOTOR_KEYS], dtype=float)


def tilt_deg(pose: np.ndarray) -> float:
    """Angle between the gripper axis and straight down."""
    return float(np.degrees(np.arccos(np.clip(-pose[2, 2], -1.0, 1.0))))


# Revolute limits from so101_new_calib.urdf, degrees, in MOTOR_KEYS order.
# The optimiser must respect these: an unbounded solve happily returns poses
# like wrist_roll = -1100 deg, which are exact in the maths and impossible on
# the hardware.
JOINT_LIMITS_DEG = (
    (-110.00, 110.00),   # shoulder_pan
    (-100.00, 100.00),   # shoulder_lift
    (-96.83, 96.83),     # elbow_flex
    (-95.00, 95.00),     # wrist_flex
    (-157.21, 162.79),   # wrist_roll
)


# DEFAULT_HOME sits outside the URDF limits (shoulder_lift -107.9 vs +/-100,
# elbow_flex 97.4 vs +/-96.8) yet is the pose the hardware actually runs from,
# so the URDF range is narrower than the robot's real one. Widen just enough to
# admit home; still tight enough to reject nonsense like wrist_roll = -1100 deg.
EFFECTIVE_LIMITS_DEG = tuple(
    (min(lo, float(DEFAULT_HOME[i]) - 2.0), max(hi, float(DEFAULT_HOME[i]) + 2.0))
    for i, (lo, hi) in enumerate(JOINT_LIMITS_DEG)
)


def within_limits(joints_deg: np.ndarray, tol: float = 1e-6) -> bool:
    return all(
        lo - tol <= float(q) <= hi + tol
        for q, (lo, hi) in zip(np.asarray(joints_deg)[:5], EFFECTIVE_LIMITS_DEG)
    )


def limit_violations(joints_deg: np.ndarray, tol: float = 1e-6) -> list[str]:
    out = []
    for name, q, (lo, hi) in zip(
        MOTOR_KEYS, np.asarray(joints_deg)[:5], EFFECTIVE_LIMITS_DEG
    ):
        if not (lo - tol <= float(q) <= hi + tol):
            out.append(f"{name}={float(q):.2f} outside [{lo:.2f}, {hi:.2f}]")
    return out


def solve_down_pose(
    target_xyz: np.ndarray,
    *,
    tilt_weight: float = 0.05,
    restarts: int = 12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve joints putting the gripper vertically down at target_xyz.

    The incremental IK used by the trajectory builder converges on position but
    leaves the wrist several degrees off vertical, which makes a probe ambiguous
    to read. Here position and verticality are optimised together instead, with
    multiple seeds because the arm is redundant enough to have local minima.
    """
    target = np.asarray(target_xyz, dtype=float)

    def residual(q: np.ndarray) -> np.ndarray:
        pose = kinematics.forward_kinematics(np.concatenate([q, [0.0]]))
        pos = pose[:3, 3] - target
        # Drive the gripper z-axis to (0, 0, -1); scaled so a degree of tilt
        # trades against roughly a tenth of a millimetre of position.
        axis = pose[:3, 2] - np.array([0.0, 0.0, -1.0])
        return np.concatenate([pos, tilt_weight * axis])

    lo = np.array([b[0] for b in EFFECTIVE_LIMITS_DEG], dtype=float)
    hi = np.array([b[1] for b in EFFECTIVE_LIMITS_DEG], dtype=float)

    rng = np.random.default_rng(0)
    seeds = [np.clip(DEFAULT_HOME[:5].copy(), lo, hi)]
    for _ in range(restarts - 1):
        seeds.append(rng.uniform(lo, hi))

    best = None
    for seed in seeds:
        try:
            sol = least_squares(
                residual, np.clip(seed, lo, hi), bounds=(lo, hi), max_nfev=4000
            )
        except Exception:
            continue
        joints = np.concatenate([sol.x, [DEFAULT_HOME[5]]])
        pose = kinematics.forward_kinematics(joints)
        pos_err = float(np.linalg.norm(pose[:3, 3] - target))
        t = tilt_deg(pose)
        score = pos_err * 1000.0 + t
        if best is None or score < best[0]:
            best = (score, joints, pose, pos_err, t)

    if best is None:
        raise RuntimeError("IK failed to converge for this target")
    _, joints, pose, pos_err, t = best
    return joints, pose[:3, 3], t


def build_approach_path(
    current_deg: np.ndarray,
    via_deg: np.ndarray,
    probe_deg: np.ndarray,
    *,
    steps_to_via: int,
    steps_to_probe: int,
) -> list[np.ndarray]:
    """current -> via (above the square) -> probe.

    A single interpolation from the arm's pose straight to the probe sweeps the
    wrist sideways across the board at probe height. Going up and over first
    means the only descent happens once the gripper is already above the target
    square.
    """
    path: list[np.ndarray] = []
    for k in range(1, steps_to_via + 1):
        a = k / steps_to_via
        path.append((1.0 - a) * current_deg + a * via_deg)
    for k in range(1, steps_to_probe + 1):
        a = k / steps_to_probe
        path.append((1.0 - a) * via_deg + a * probe_deg)
    return path


def resolve_target_xy(args: argparse.Namespace) -> tuple[float, float, str]:
    board_origin = (args.board_origin_x, args.board_origin_y, args.board_origin_z)
    if args.square is not None:
        square = validate_square(args.square)
        x, y = location_world_xy(
            square,
            board_origin=board_origin,
            square_size=args.square_size,
        )
        return x, y, square
    if args.xy is None:
        raise ValueError("Provide either --square or --xy")
    return float(args.xy[0]), float(args.xy[1]), "xy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--square", help="Board square to probe, e.g. d4.")
    parser.add_argument(
        "--xy",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="World XY in metres (robot base frame), instead of --square.",
    )
    parser.add_argument(
        "--height-above-surface",
        type=float,
        default=0.02,
        help="Gripper height above the board playing surface, metres (default 0.02).",
    )
    parser.add_argument("--board-origin-x", type=float, default=0.29325)
    parser.add_argument("--board-origin-y", type=float, default=-0.01484)
    parser.add_argument(
        "--board-origin-z",
        type=float,
        default=0.0196,
        help="Board slab centre z; playing surface is this + 0.005.",
    )
    parser.add_argument(
        "--via-height",
        type=float,
        default=0.12,
        help=(
            "Height above the playing surface for the transit waypoint placed "
            "over the target square (default 0.12). The arm goes up and over "
            "before descending, so it never sweeps across the board."
        ),
    )
    parser.add_argument("--square-size", type=float, default=DEFAULT_SQUARE_SIZE)
    parser.add_argument("--port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--move-steps",
        type=int,
        default=200,
        help="Interpolation steps from the current pose to the probe pose.",
    )
    parser.add_argument("--command-delay", type=float, default=1.0 / 240.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Connect to the SO101 and move. Omit for a dry-run solve.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.move_steps <= 0:
        raise ValueError("--move-steps must be positive")

    x, y, label = resolve_target_xy(args)
    surface_z = args.board_origin_z + 0.005
    z = surface_z + args.height_above_surface
    target = np.array([x, y, z], dtype=float)

    joints, achieved, down_err_deg = solve_down_pose(target)
    via_target = np.array([x, y, surface_z + args.via_height], dtype=float)
    via_joints, via_achieved, via_tilt = solve_down_pose(via_target)

    print(f"probe target : {label}")
    print(f"world xyz    : [{target[0]:.5f}, {target[1]:.5f}, {target[2]:.5f}]")
    print(f"  (surface z = {surface_z:.5f}, +{args.height_above_surface:.3f} clearance)")
    print(f"achieved xyz : [{achieved[0]:.5f}, {achieved[1]:.5f}, {achieved[2]:.5f}]")
    print(f"ik residual  : {np.linalg.norm(achieved - target) * 1000:.2f} mm")
    print(f"off-vertical : {down_err_deg:.2f} deg")
    print(f"joints_deg   : {np.round(joints, 3).tolist()}")
    print(
        f"via waypoint : z={via_target[2]:.4f}  "
        f"residual={np.linalg.norm(via_achieved - via_target) * 1000:.1f} mm  "
        f"tilt={via_tilt:.1f} deg (transit only, need not be vertical)"
    )
    probe_bad = limit_violations(joints)
    via_bad = limit_violations(via_joints)
    if probe_bad or via_bad:
        raise RuntimeError(
            "Solved pose exceeds joint limits: " + "; ".join(probe_bad + via_bad)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema": "probe_xy_pose_v1",
        "label": label,
        "target_world_xyz": target.tolist(),
        "achieved_world_xyz": achieved.tolist(),
        "ik_residual_m": float(np.linalg.norm(achieved - target)),
        "off_vertical_deg": down_err_deg,
        "joints_deg": joints.tolist(),
        "via_joints_deg": via_joints.tolist(),
        "via_height": args.via_height,
        "board_origin": [args.board_origin_x, args.board_origin_y, args.board_origin_z],
        "surface_z": surface_z,
        "height_above_surface": args.height_above_surface,
        "square_size": args.square_size,
    }
    out_path = args.output_dir / "probe_xy_pose.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"saved        : {out_path}")

    if down_err_deg > 1.0:
        print(
            f"WARNING: solved pose is {down_err_deg:.2f} deg off vertical; "
            "this target is past the reach limit for a truly vertical wrist."
        )

    if not args.execute:
        print("dry run only; add --execute to move the real SO101 arm.")
        return 0

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    follower = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
    follower.connect()
    try:
        observation = follower.get_observation()
        current = observation_joints_deg(observation)

        steps_to_via = max(1, int(round(args.move_steps * 0.6)))
        steps_to_probe = max(1, args.move_steps - steps_to_via)
        path = build_approach_path(
            current,
            via_joints,
            joints,
            steps_to_via=steps_to_via,
            steps_to_probe=steps_to_probe,
        )

        bad = [v for step in path for v in limit_violations(step)]
        if bad:
            raise RuntimeError(
                "Approach path exceeds joint limits: " + "; ".join(sorted(set(bad)))
            )

        print("current_deg :", np.round(current, 3).tolist())
        print("via_deg     :", np.round(via_joints, 3).tolist())
        print("target_deg  :", np.round(joints, 3).tolist())
        print(
            f"path: {steps_to_via} steps up to via, then {steps_to_probe} down to "
            f"probe ({len(path)} total); max joint delta "
            f"{float(np.max(np.abs(joints - current))):.2f} deg"
        )
        confirmation = input("Move to probe pose? [yes/no]: ").strip().lower()
        if confirmation not in {"yes", "y"}:
            print("Cancelled.")
            return 1

        for index, step in enumerate(path, start=1):
            action = observation.copy()
            action.update(action_from_joints(step))
            follower.send_action(action)
            if index == steps_to_via:
                print(f"  reached via waypoint ({index}/{len(path)}); descending")
            elif index % 50 == 0 or index == len(path):
                print(f"  {index}/{len(path)}")
            time.sleep(args.command_delay)
            observation = follower.get_observation()

        settled = observation_joints_deg(follower.get_observation())
        print("observed_deg:", np.round(settled, 3).tolist())
        print("cmd-obs diff:", np.round(settled - joints, 3).tolist())
        print(
            "\nHolding probe pose. Read off which square the gripper points at, "
            "then compare with the commanded square above."
        )
        input("Press Enter to release and exit...")
    finally:
        follower.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
