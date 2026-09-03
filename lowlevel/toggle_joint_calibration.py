#!/usr/bin/env python3
"""Cycle one joint through a list of commanded angles, holding at each so the
real angle can be checked against a physical reference (square/protractor).

Meant to be used now that the servo control loop has a nonzero I_Coefficient
(see lerobot/src/lerobot/robots/so_follower/so_follower.py -- I_Coefficient
raised from 0 to 4), which was shown to remove the load-dependent settling
sag that previously made commanded-vs-observed comparisons unreliable. With
that sag mostly gone, a residual commanded-vs-observed gap at a given value
(e.g. elbow_flex commanded to 0) is a more trustworthy signal of a genuine
homing/calibration offset rather than a torque artifact.

All other joints are left at whatever the arm currently reads -- only the
named joint moves. Board-collision checking is advisory only (prints a
warning, does not block); joint limits are still hard-enforced. Watch the
arm and stop it by hand if needed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pybullet as p

from probe_right_angle_pose import build_sim, path_min_clearance, set_pose, MIN_CLEARANCE_M
from probe_xy_pose import (
    MOTOR_KEYS,
    action_from_joints,
    clamp_path_to_effective_limits,
    limit_violations,
    observation_joints_deg,
)

LOWLEVEL_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = LOWLEVEL_DIR / "real_world_runs"
DEFAULT_ROBOT_PORT = "/dev/tty.usbmodem5B7B0157051"
DEFAULT_ROBOT_ID = "my_awesome_follower_arm"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def link_tilt_from_vertical_deg(robot_id: int, joints_deg: np.ndarray) -> float:
    """Angle from true vertical (deg) of the upper_arm_link->lower_arm_link
    vector, given a full 6-value joints_deg pose. The visible upper-arm link
    isn't aligned with pure joint rotation (off by a fixed ~14deg at every
    round shoulder_lift value per URDF geometry), so this predicts the exact
    expected tilt for a commanded shoulder_lift value to compare directly
    against a phone inclinometer reading placed on the arm -- any mismatch
    is the calibration offset, independent of the link's own geometry.
    """
    link_index_by_name = {}
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        link_index_by_name[info[12].decode("utf-8")] = j
    upper_idx = link_index_by_name["upper_arm_link"]
    lower_idx = link_index_by_name["lower_arm_link"]
    set_pose(robot_id, joints_deg)
    upper_pos = np.array(p.getLinkState(robot_id, upper_idx, computeForwardKinematics=True)[4])
    lower_pos = np.array(p.getLinkState(robot_id, lower_idx, computeForwardKinematics=True)[4])
    vec = lower_pos - upper_pos
    vec_n = vec / np.linalg.norm(vec)
    return float(np.degrees(np.arccos(np.clip(vec_n[2], -1.0, 1.0))))


def load_reference_pose(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "joints_deg" in data:
        return np.array(data["joints_deg"], dtype=float)
    if "results" in data and data["results"]:
        return np.array(data["results"][-1]["observed"], dtype=float)
    raise ValueError(f"Don't know how to read a reference pose out of {path}")


def interpolated_steps(start_deg: np.ndarray, end_deg: np.ndarray, steps: int) -> list[np.ndarray]:
    out = []
    steps = max(1, int(steps))
    for k in range(1, steps + 1):
        a = k / steps
        out.append((1.0 - a) * start_deg + a * end_deg)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", choices=JOINT_NAMES, required=True)
    parser.add_argument(
        "--values", type=float, nargs="+", required=True,
        help="Angles (deg) to cycle through in order, e.g. --values 0 90 0 90",
    )
    parser.add_argument("--repeat", type=int, default=1, help="How many times to cycle through --values.")
    parser.add_argument(
        "--hold-s", type=float, default=2.0,
        help="Seconds to hold and settle at each value before reading it back.",
    )
    parser.add_argument(
        "--confirm-each", action="store_true",
        help="Wait for Enter before each move instead of auto-advancing after --hold-s.",
    )
    parser.add_argument(
        "--reference-pose-file",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "reference_pose.json",
        help=(
            "Move to this known pose before starting the toggle sequence, instead "
            "of starting from wherever the arm currently is. Accepts either a "
            "{'joints_deg': [...]} file (default: shoulder_lift=0, elbow_flex=0, "
            "wrist_flex=0 -- the already-checked joints held at their zero) or an "
            "elbow_sag_sweep.json-style file (last result's 'observed' is used). "
            "Pass --no-reference-pose to skip this."
        ),
    )
    parser.add_argument(
        "--no-reference-pose", action="store_true",
        help="Start from wherever the arm currently is instead of --reference-pose-file.",
    )
    parser.add_argument("--board-origin-x", type=float, default=0.29325)
    parser.add_argument("--board-origin-y", type=float, default=-0.01484)
    parser.add_argument("--board-origin-z", type=float, default=0.0196)
    parser.add_argument("--square-size", type=float, default=0.04125)
    parser.add_argument("--port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--move-steps", type=int, default=200)
    parser.add_argument("--command-delay", type=float, default=1.0 / 240.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    joint_idx = JOINT_NAMES.index(args.joint)
    board_origin = (args.board_origin_x, args.board_origin_y, args.board_origin_z)
    robot_id, board_id = build_sim(board_origin, args.square_size)

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    follower = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
    follower.connect()
    results = []
    try:
        observation = follower.get_observation()
        current = observation_joints_deg(observation)
        print("current_deg:", np.round(current, 3).tolist())

        if not args.no_reference_pose and args.reference_pose_file.exists():
            reference = load_reference_pose(args.reference_pose_file)
            print(f"reference pose (from {args.reference_pose_file.name}): {np.round(reference, 3).tolist()}")
            ref_path = interpolated_steps(current, reference, args.move_steps)
            ref_clipped, _ = clamp_path_to_effective_limits(ref_path)
            bad = [v for step in ref_clipped for v in limit_violations(step)]
            if bad:
                raise RuntimeError("Move to reference pose exceeds joint limits: " + "; ".join(sorted(set(bad))))
            ref_clearance = path_min_clearance(robot_id, board_id, ref_clipped)
            if ref_clearance < MIN_CLEARANCE_M:
                print(f"  WARNING: path to reference pose comes within {ref_clearance*1000:.1f}mm of the board (advisory only).")
            confirmation = input("Move to reference pose first? [yes/no]: ").strip().lower()
            if confirmation not in {"yes", "y"}:
                print("Cancelled.")
                return 1
            for step in ref_clipped:
                action = observation.copy()
                action.update(action_from_joints(step))
                follower.send_action(action)
                time.sleep(args.command_delay)
                observation = follower.get_observation()
            time.sleep(args.hold_s)
            current = observation_joints_deg(follower.get_observation())
            print("at reference pose:", np.round(current, 3).tolist())

        sequence = list(args.values) * args.repeat
        print(f"cycling {args.joint} through {sequence}")
        confirmation = input("Start toggle sequence? [yes/no]: ").strip().lower()
        if confirmation not in {"yes", "y"}:
            print("Cancelled.")
            return 1

        for i, value in enumerate(sequence, start=1):
            target = current.copy()
            target[joint_idx] = value

            path = interpolated_steps(current, target, args.move_steps)
            clipped, _ = clamp_path_to_effective_limits(path)
            bad = [v for step in clipped for v in limit_violations(step)]
            if bad:
                raise RuntimeError(f"Move to {args.joint}={value} exceeds joint limits: " + "; ".join(sorted(set(bad))))
            clearance = path_min_clearance(robot_id, board_id, clipped)
            if clearance < MIN_CLEARANCE_M:
                print(f"  WARNING: path comes within {clearance*1000:.1f}mm of the board (advisory only).")

            for step in clipped:
                action = observation.copy()
                action.update(action_from_joints(step))
                follower.send_action(action)
                time.sleep(args.command_delay)
                observation = follower.get_observation()

            time.sleep(args.hold_s)
            settled = observation_joints_deg(follower.get_observation())
            diff = settled[joint_idx] - value
            line = (
                f"[{i}/{len(sequence)}] {args.joint} target={value:+7.2f}  "
                f"observed={settled[joint_idx]:+7.2f}  diff={diff:+.3f}"
            )
            result_entry = {"target": value, "observed_deg": settled.tolist(), "diff": diff}
            if args.joint == "shoulder_lift":
                predicted_tilt = link_tilt_from_vertical_deg(robot_id, target)
                line += f"  predicted_upper_arm_tilt_from_vertical={predicted_tilt:+.2f}deg"
                result_entry["predicted_upper_arm_tilt_from_vertical_deg"] = predicted_tilt
            print(line)
            results.append(result_entry)
            current = settled

            if args.confirm_each:
                input("Press Enter for next value...")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / f"toggle_{args.joint}_calibration.json"
        out_path.write_text(json.dumps({"joint": args.joint, "results": results}, indent=2), encoding="utf-8")
        print(f"saved: {out_path}")
        print("\nHolding final pose. Compare against a physical square/protractor now.")
        input("Press Enter to release and exit...")
    finally:
        follower.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
