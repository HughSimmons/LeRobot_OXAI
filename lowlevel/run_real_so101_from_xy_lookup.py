#!/usr/bin/env python3
"""Run a saved continuous XY lookup trajectory on a real SO101 follower arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from board_coordinates import DEFAULT_BOARD_ORIGIN, XYPoint
from chess_traj import (
    DEFAULT_HOME,
    normalized_home_joints,
    pickupmove_traj_with_metrics,
    temporary_home_joints,
)


LOWLEVEL_DIR = Path(__file__).resolve().parent
REPO_DIR = LOWLEVEL_DIR.parent
DEFAULT_LOOKUP_JSON = (
    LOWLEVEL_DIR
    / "rook_kiri_xy_lookup"
    / "real_world2_x027_no_edge_d4_d5_full_lookup_20260821"
    / "lookup"
    / "d4_x027_full_to_d5_x027_full.json"
)
DEFAULT_OUTPUT_DIR = LOWLEVEL_DIR / "real_world_runs"
DEFAULT_ROBOT_PORT = "/dev/tty.usbmodem5B7B0157051"
DEFAULT_ROBOT_ID = "my_awesome_follower_arm"
SIM_STEPS_PER_WAYPOINT = 50
SIM_TIMESTEP_S = 1.0 / 240.0
MOTOR_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def point_from_saved(value: dict[str, Any]) -> XYPoint:
    if value.get("type") != "continuous_xy" or value.get("frame") != "world":
        raise ValueError(f"Unsupported saved point: {value}")
    return XYPoint(float(value["x"]), float(value["y"]), name=value.get("name"))


def action_from_joints(joints_deg: np.ndarray) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in zip(MOTOR_KEYS, np.asarray(joints_deg, dtype=float))
    }


def observation_joints_deg(observation: dict[str, float]) -> np.ndarray:
    return np.array([float(observation[key]) for key in MOTOR_KEYS], dtype=float)


def catmull_rom_joints(
    prev_pos: np.ndarray,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    next_pos: np.ndarray,
    alpha: float,
) -> np.ndarray:
    alpha2 = alpha * alpha
    alpha3 = alpha2 * alpha
    return 0.5 * (
        (2.0 * start_pos)
        + (-prev_pos + end_pos) * alpha
        + (2.0 * prev_pos - 5.0 * start_pos + 4.0 * end_pos - next_pos) * alpha2
        + (-prev_pos + 3.0 * start_pos - 3.0 * end_pos + next_pos) * alpha3
    )


def interpolate_sim_style_joints(
    waypoints_deg: list[np.ndarray],
    move_idx: int,
    alpha: float,
) -> np.ndarray:
    start_idx = max(move_idx - 1, 0)
    end_idx = move_idx
    prev_idx = max(start_idx - 1, 0)
    next_idx = min(end_idx + 1, len(waypoints_deg) - 1)

    prev_pos = waypoints_deg[prev_idx]
    start_pos = waypoints_deg[start_idx]
    end_pos = waypoints_deg[end_idx]
    next_pos = waypoints_deg[next_idx]

    target_joints = end_pos.copy()
    target_joints[:5] = catmull_rom_joints(
        prev_pos[:5],
        start_pos[:5],
        end_pos[:5],
        next_pos[:5],
        alpha,
    )
    return target_joints


def interpolate_joint_waypoints(
    waypoints_deg: list[np.ndarray],
    *,
    steps_per_waypoint: int,
    interpolation: str,
) -> list[np.ndarray]:
    if steps_per_waypoint <= 0:
        raise ValueError("steps_per_waypoint must be positive")
    if not waypoints_deg:
        return []

    commands: list[np.ndarray] = []
    if interpolation == "linear":
        current = waypoints_deg[0]
        commands.append(current.copy())
        for target in waypoints_deg[1:]:
            for alpha in np.linspace(0.0, 1.0, steps_per_waypoint + 1)[1:]:
                commands.append((1.0 - alpha) * current + alpha * target)
            current = target.copy()
    elif interpolation == "sim":
        for move_idx in range(len(waypoints_deg)):
            for step in range(steps_per_waypoint):
                alpha = min(step / steps_per_waypoint, 1.0)
                commands.append(
                    interpolate_sim_style_joints(waypoints_deg, move_idx, alpha)
                )
    else:
        raise ValueError(f"Unsupported interpolation mode: {interpolation}")
    return commands


def waypoint_command_index(
    waypoint_index: int,
    *,
    steps_per_waypoint: int,
    interpolation: str,
) -> int:
    if waypoint_index < 0:
        raise ValueError("waypoint_index must be non-negative")
    if steps_per_waypoint <= 0:
        raise ValueError("steps_per_waypoint must be positive")
    if interpolation == "linear":
        return 0 if waypoint_index == 0 else waypoint_index * steps_per_waypoint
    if interpolation == "sim":
        return (waypoint_index + 1) * steps_per_waypoint
    raise ValueError(f"Unsupported interpolation mode: {interpolation}")


def build_trajectory_from_lookup(
    lookup: dict[str, Any],
) -> tuple[list[np.ndarray], int, dict[str, Any]]:
    start = point_from_saved(lookup["from"])
    target = point_from_saved(lookup["to"])
    grasp_offset = np.array(lookup["source_grasp_offset"], dtype=float)
    place_offset = np.array(lookup["selected_place_offset"], dtype=float)
    search = lookup["search"]
    metrics = lookup.get("metrics", {})
    lower_place_path_bias = metrics.get("lower_place_path_bias")
    if lower_place_path_bias is not None:
        lower_place_path_bias = np.array(lower_place_path_bias, dtype=float)

    trajectory_home = metrics.get("trajectory_home_joints_deg")
    if trajectory_home is not None:
        trajectory_home = normalized_home_joints(trajectory_home)

    with temporary_home_joints(trajectory_home):
        movelist, closeidx, traj_metrics = pickupmove_traj_with_metrics(
            start,
            target,
            board_origin=DEFAULT_BOARD_ORIGIN,
            GRASP_OFFSET=grasp_offset,
            PLACE_OFFSET=place_offset,
            placement_lower_steps=int(search["placement_lower_steps"]),
            lift_height=float(search["lift_height"]),
            lower_place_path_bias=lower_place_path_bias,
        )

    waypoints = [np.array(joints, dtype=float) for joints in movelist]
    metadata = {
        "start": start.as_dict(),
        "target": target.as_dict(),
        "grasp_offset": grasp_offset,
        "place_offset": place_offset,
        "lookup_move_id": lookup.get("move_id"),
        "lookup_success": lookup.get("success"),
        "lookup_metrics": metrics,
        "regenerated_trajectory_metrics": traj_metrics,
        "closeidx": closeidx,
        "trajectory_home_joints_deg": (
            DEFAULT_HOME.copy() if trajectory_home is None else trajectory_home.copy()
        ),
    }
    return waypoints, closeidx, metadata


def write_dry_run(
    output_dir: Path,
    *,
    lookup_path: Path,
    waypoints: list[np.ndarray],
    commands: list[np.ndarray],
    metadata: dict[str, Any],
    args: argparse.Namespace,
    pause_command_index: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "lookup_json": str(lookup_path),
        "execute": bool(args.execute),
        "port": args.port,
        "robot_id": args.robot_id,
        "steps_per_waypoint": args.steps_per_waypoint,
        "command_delay_s": args.command_delay,
        "interpolation": args.interpolation,
        "nominal_waypoint_duration_s": args.steps_per_waypoint * args.command_delay,
        "pause_at_pickup": bool(args.pause_at_pickup),
        "pickup_pause_s": args.pickup_pause_s,
        "pause_command_index": pause_command_index,
        "waypoint_count": len(waypoints),
        "command_count": len(commands),
        "metadata": metadata,
        "waypoints_deg": waypoints,
        "commands_deg": commands if args.export_commands else [],
    }
    (output_dir / "real_so101_trajectory_preview.json").write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def execute_commands(
    commands_deg: list[np.ndarray],
    *,
    port: str,
    robot_id: str,
    command_delay: float,
    max_start_delta_deg: float,
    allow_start_bridge: bool,
    start_bridge_steps: int,
    pause_command_index: int | None,
    pickup_pause_s: float,
) -> None:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    follower = SO101Follower(SO101FollowerConfig(port=port, id=robot_id))
    follower.connect()
    try:
        observation = follower.get_observation()
        current = observation_joints_deg(observation)
        first_delta = float(np.max(np.abs(commands_deg[0] - current)))
        if first_delta > max_start_delta_deg:
            if not allow_start_bridge:
                raise RuntimeError(
                    "Current arm position is too far from first command: "
                    f"{first_delta:.2f} deg > {max_start_delta_deg:.2f} deg. "
                    "Move closer to the trajectory start or rerun with "
                    "--allow-start-bridge."
                )
            if start_bridge_steps <= 0:
                raise ValueError("--start-bridge-steps must be positive")

            print("Current arm position is far from the trajectory start.")
            print("current_deg:", np.round(current, 3).tolist())
            print("target_start_deg:", np.round(commands_deg[0], 3).tolist())
            print("delta_deg:", np.round(commands_deg[0] - current, 3).tolist())
            print(
                f"This will move slowly to the trajectory start over "
                f"{start_bridge_steps} steps."
            )
            confirmation = input("Move to trajectory start? [yes/no]: ").strip().lower()
            if confirmation not in {"yes", "y"}:
                raise RuntimeError("Start bridge cancelled by user.")

            for index, alpha in enumerate(
                np.linspace(0.0, 1.0, start_bridge_steps + 1)[1:],
                start=1,
            ):
                joints = (1.0 - alpha) * current + alpha * commands_deg[0]
                action = observation.copy()
                action.update(action_from_joints(joints))
                follower.send_action(action)
                if index % 25 == 0 or index == start_bridge_steps:
                    print(f"start bridge {index}/{start_bridge_steps}")
                time.sleep(command_delay)
                observation = follower.get_observation()

        for index, joints in enumerate(commands_deg, start=1):
            action = observation.copy()
            action.update(action_from_joints(joints))
            follower.send_action(action)
            if pause_command_index is not None and index - 1 == pause_command_index:
                print(f"Paused at pickup command {index}/{len(commands_deg)}.")
                if pickup_pause_s > 0.0:
                    time.sleep(pickup_pause_s)
                else:
                    input("Press Enter to continue trajectory...")
            if index % 50 == 0:
                print(f"sent {index}/{len(commands_deg)} commands")
            time.sleep(command_delay)
            observation = follower.get_observation()
    finally:
        follower.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate a saved continuous XY lookup and optionally send it to a real SO101."
    )
    parser.add_argument("--lookup-json", type=Path, default=DEFAULT_LOOKUP_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--steps-per-waypoint", type=int, default=SIM_STEPS_PER_WAYPOINT)
    parser.add_argument("--command-delay", type=float, default=SIM_TIMESTEP_S)
    parser.add_argument(
        "--interpolation",
        choices=("sim", "linear"),
        default="sim",
        help=(
            "Use the simulator's Catmull-Rom interpolation for arm joints "
            "or the previous linear joint interpolation."
        ),
    )
    parser.add_argument("--max-start-delta-deg", type=float, default=20.0)
    parser.add_argument(
        "--allow-start-bridge",
        action="store_true",
        help=(
            "If current joints are far from the first command, ask for typed "
            "confirmation and slowly move to the trajectory start."
        ),
    )
    parser.add_argument("--start-bridge-steps", type=int, default=150)
    parser.add_argument(
        "--pause-at-pickup",
        action="store_true",
        help=(
            "Pause when the trajectory reaches the pickup/close waypoint. "
            "By default this waits for Enter."
        ),
    )
    parser.add_argument(
        "--pickup-pause-s",
        type=float,
        default=0.0,
        help="Timed pickup pause in seconds. Use 0 to wait for Enter.",
    )
    parser.add_argument(
        "--export-commands",
        action="store_true",
        help="Include every interpolated command in the preview JSON.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually connect to the SO101 and send commands. Omit for dry-run/export.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pickup_pause_s < 0.0:
        raise ValueError("--pickup-pause-s must be non-negative")
    lookup_path = args.lookup_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
    if lookup.get("schema") != "continuous_xy_lookup_v1":
        raise ValueError(f"Unsupported lookup schema in {lookup_path}")

    waypoints, closeidx, metadata = build_trajectory_from_lookup(lookup)
    commands = interpolate_joint_waypoints(
        waypoints,
        steps_per_waypoint=args.steps_per_waypoint,
        interpolation=args.interpolation,
    )
    pause_command_index = None
    if args.pause_at_pickup:
        pause_command_index = min(
            waypoint_command_index(
                closeidx,
                steps_per_waypoint=args.steps_per_waypoint,
                interpolation=args.interpolation,
            ),
            len(commands) - 1,
        )
    write_dry_run(
        output_dir,
        lookup_path=lookup_path,
        waypoints=waypoints,
        commands=commands,
        metadata=metadata,
        args=args,
        pause_command_index=pause_command_index,
    )

    print(f"lookup: {lookup_path}")
    print(f"waypoints: {len(waypoints)} | commands: {len(commands)} | closeidx: {closeidx}")
    if pause_command_index is not None:
        print(f"pickup pause command index: {pause_command_index}")
    print(
        "interpolation: "
        f"{args.interpolation} | nominal waypoint time: "
        f"{args.steps_per_waypoint * args.command_delay:.3f}s"
    )
    print(f"preview: {output_dir / 'real_so101_trajectory_preview.json'}")

    if not args.execute:
        print("dry run only; add --execute to send commands to the real SO101 arm")
        return 0

    execute_commands(
        commands,
        port=args.port,
        robot_id=args.robot_id,
        command_delay=args.command_delay,
        max_start_delta_deg=args.max_start_delta_deg,
        allow_start_bridge=args.allow_start_bridge,
        start_bridge_steps=args.start_bridge_steps,
        pause_command_index=pause_command_index,
        pickup_pause_s=args.pickup_pause_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
