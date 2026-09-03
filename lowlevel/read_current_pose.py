#!/usr/bin/env python3
"""Read and print the SO101's current joint positions without moving it."""

import argparse
import json
from pathlib import Path

import numpy as np

MOTOR_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/tty.usbmodem5B7B0157051")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm")
    parser.add_argument("--output", type=Path, default=Path("real_world_runs/current_pose.json"))
    args = parser.parse_args()

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    follower = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
    follower.connect()
    try:
        observation = follower.get_observation()
    finally:
        follower.disconnect()

    joints_deg = np.array([float(observation[k]) for k in MOTOR_KEYS], dtype=float)
    print("current joints_deg:")
    for key, val in zip(MOTOR_KEYS, joints_deg):
        print(f"  {key:20s} {val:9.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "current_pose_v1",
        "motor_keys": list(MOTOR_KEYS),
        "joints_deg": joints_deg.tolist(),
    }
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
