#!/usr/bin/env python3
"""Live visual/real SO101 joint-offset calibration against the chess board."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import select
import sys
import time
from typing import Any

import cv2
import numpy as np
import pybullet as p
import pybullet_data

from board_coordinates import DEFAULT_BOARD_ORIGIN, DEFAULT_SQUARE_SIZE, square_center_world_xy
from chess_traj import gripper_angle_closed, gripper_angle_open
from run_real_so101_from_xy_lookup import (
    DEFAULT_LOOKUP_JSON,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_ROBOT_ID,
    DEFAULT_ROBOT_PORT,
    MOTOR_KEYS,
    action_from_joints,
    build_trajectory_from_lookup,
    json_safe,
    observation_joints_deg,
)


LOWLEVEL_DIR = Path(__file__).resolve().parent
REPO_DIR = LOWLEVEL_DIR.parent
URDF_PATH = REPO_DIR / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
SIM_JOINT_MAP = [0, 1, 2, 3, 4, 6]
BOARD_BASE_HALF_HEIGHT = 0.005
BOARD_TOP_Z = DEFAULT_BOARD_ORIGIN[2] + BOARD_BASE_HALF_HEIGHT
VIEWER_WINDOW = "SO101 real/sim calibration"
CONTROLS_WINDOW = "SO101 motor offsets"
VIEWER_WIDTH = 960
VIEWER_HEIGHT = 540
CONTROLS_WIDTH = 720
CONTROLS_HEIGHT = 180


def deg_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float)]


def find_release_index(waypoints: list[np.ndarray], closeidx: int) -> int:
    for idx in range(closeidx + 1, len(waypoints)):
        gripper_now = waypoints[idx][5]
        gripper_prev = waypoints[idx - 1][5]
        if (
            np.isclose(gripper_now, gripper_angle_open)
            and np.isclose(gripper_prev, gripper_angle_closed)
        ):
            return idx
    return len(waypoints) - 1


def append_pose(
    poses: list[dict[str, Any]],
    name: str,
    index: int | None,
    joints_deg: np.ndarray,
) -> None:
    if any(pose["name"] == name for pose in poses):
        return
    poses.append(
        {
            "name": name,
            "waypoint_index": index,
            "joints_deg": np.asarray(joints_deg, dtype=float).copy(),
        }
    )


def build_guided_poses(
    waypoints: list[np.ndarray],
    closeidx: int,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if not waypoints:
        raise ValueError("Trajectory has no waypoints")

    segments = metadata["regenerated_trajectory_metrics"].get("segments", {})
    above_place = segments.get("destination_above_place", {})
    lower_place = segments.get("destination_lower_place", {})
    release_idx = find_release_index(waypoints, closeidx)
    home = np.array(metadata["trajectory_home_joints_deg"], dtype=float)

    poses: list[dict[str, Any]] = []
    append_pose(poses, "trajectory_home", None, home)
    append_pose(poses, "d4_approach", min(9, len(waypoints) - 1), waypoints[min(9, len(waypoints) - 1)])
    append_pose(poses, "d4_pick", max(0, closeidx - 1), waypoints[max(0, closeidx - 1)])
    append_pose(poses, "d4_close", closeidx, waypoints[closeidx])
    append_pose(poses, "d4_lift", min(closeidx + 1, len(waypoints) - 1), waypoints[min(closeidx + 1, len(waypoints) - 1)])

    above_end = int(above_place.get("end", min(closeidx + 2, len(waypoints))))
    lower_end = int(lower_place.get("end", release_idx))
    append_pose(poses, "d5_approach", max(0, min(above_end - 1, len(waypoints) - 1)), waypoints[max(0, min(above_end - 1, len(waypoints) - 1))])
    append_pose(poses, "d5_place", max(0, min(lower_end - 1, len(waypoints) - 1)), waypoints[max(0, min(lower_end - 1, len(waypoints) - 1))])
    append_pose(poses, "d5_release", release_idx, waypoints[release_idx])
    append_pose(poses, "return_home", len(waypoints) - 1, waypoints[-1])
    return poses


def build_comparison_poses(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    home = np.array(metadata["trajectory_home_joints_deg"], dtype=float)
    gripper = home[5]
    poses: list[dict[str, Any]] = []
    append_pose(poses, "home_reference", None, home)
    append_pose(
        poses,
        "base_forward_90s",
        None,
        np.array([90.0, -90.0, 90.0, 60.0, 0.0, gripper]),
    )
    append_pose(
        poses,
        "base_center_clear",
        None,
        np.array([0.0, -90.0, 75.0, 60.0, 0.0, gripper]),
    )
    append_pose(
        poses,
        "base_mirror_90s",
        None,
        np.array([-90.0, -90.0, 90.0, 60.0, 0.0, gripper]),
    )
    append_pose(
        poses,
        "wrist_roll_plus_90",
        None,
        np.array([90.0, -90.0, 90.0, 60.0, 90.0, gripper]),
    )
    append_pose(
        poses,
        "wrist_roll_minus_90",
        None,
        np.array([90.0, -90.0, 90.0, 60.0, -90.0, gripper]),
    )
    append_pose(
        poses,
        "elbow_open_compare",
        None,
        np.array([90.0, -75.0, 75.0, 50.0, 0.0, gripper]),
    )
    return poses


def setup_pybullet_scene(from_square: str, to_square: str) -> tuple[int, int]:
    if p.isConnected():
        p.disconnect()
    p.connect(p.DIRECT)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    p.loadURDF("plane.urdf", [0, 0, 0])
    robot_id = p.loadURDF(str(URDF_PATH), [0, 0, 0], useFixedBase=True)
    board_id = create_board_visuals()
    create_square_marker(from_square, [0.1, 0.45, 1.0, 0.8])
    create_square_marker(to_square, [1.0, 0.55, 0.0, 0.8])
    return robot_id, board_id


def create_board_visuals() -> int:
    board_x, board_y, board_z = DEFAULT_BOARD_ORIGIN
    board_size = 8 * DEFAULT_SQUARE_SIZE
    board_base_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[board_size / 2, board_size / 2, BOARD_BASE_HALF_HEIGHT],
    )
    board_base_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[board_size / 2, board_size / 2, BOARD_BASE_HALF_HEIGHT],
        rgbaColor=[0.3, 0.3, 0.3, 1.0],
    )
    board_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=board_base_shape,
        baseVisualShapeIndex=board_base_visual,
        basePosition=[board_x, board_y, board_z],
    )

    for row in range(8):
        for col in range(8):
            x = board_x - board_size / 2 + (col + 0.5) * DEFAULT_SQUARE_SIZE
            y = board_y - board_size / 2 + (row + 0.5) * DEFAULT_SQUARE_SIZE
            color = [1, 1, 1, 0.55] if (row + col) % 2 == 0 else [0.1, 0.1, 0.1, 0.55]
            square_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[DEFAULT_SQUARE_SIZE / 2 - 0.001, DEFAULT_SQUARE_SIZE / 2 - 0.001, 0.001],
                rgbaColor=color,
            )
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=square_visual,
                basePosition=[x, y, BOARD_TOP_Z + 0.001],
            )
    return board_id


def create_square_marker(square: str, color: list[float]) -> None:
    x, y = square_center_world_xy(square)
    visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=DEFAULT_SQUARE_SIZE * 0.18,
        length=0.003,
        rgbaColor=color,
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual,
        basePosition=[x, y, BOARD_TOP_Z + 0.006],
    )


def set_sim_joints(robot_id: int, joints_deg: np.ndarray) -> None:
    joints_rad = np.deg2rad(np.asarray(joints_deg, dtype=float))
    for traj_idx, sim_idx in enumerate(SIM_JOINT_MAP):
        p.resetJointState(robot_id, sim_idx, joints_rad[traj_idx], targetVelocity=0.0)
        p.setJointMotorControl2(
            robot_id,
            sim_idx,
            p.POSITION_CONTROL,
            targetPosition=joints_rad[traj_idx],
            force=50,
        )


def board_contact_links(robot_id: int, board_id: int) -> list[str]:
    contacts = p.getContactPoints(bodyA=robot_id, bodyB=board_id)
    links = []
    for contact in contacts:
        link_index = contact[3]
        if link_index < 0:
            links.append("base")
            continue
        joint_info = p.getJointInfo(robot_id, link_index)
        links.append(joint_info[12].decode("utf-8", errors="replace"))
    return sorted(set(links))


def validate_poses_clear_board(poses: list[dict[str, Any]], from_square: str, to_square: str) -> None:
    robot_id, board_id = setup_pybullet_scene(from_square, to_square)
    try:
        failures = []
        for pose in poses:
            set_sim_joints(robot_id, pose["joints_deg"])
            for _ in range(20):
                p.stepSimulation()
            links = board_contact_links(robot_id, board_id)
            if links:
                failures.append((pose["name"], links))
        if failures:
            details = "; ".join(
                f"{name}: {', '.join(links)}" for name, links in failures
            )
            raise RuntimeError(
                "Calibration pose collision with raised board detected: "
                f"{details}"
            )
    finally:
        if p.isConnected():
            p.disconnect()


def create_cv2_controls(max_offset_deg: float) -> None:
    cv2.namedWindow(VIEWER_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VIEWER_WINDOW, VIEWER_WIDTH, VIEWER_HEIGHT)
    cv2.namedWindow(CONTROLS_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WINDOW, CONTROLS_WIDTH, CONTROLS_HEIGHT)
    ticks = int(round(max_offset_deg * 20))
    max_tick = ticks * 2
    for key in MOTOR_KEYS:
        label = key.replace(".pos", "")
        cv2.createTrackbar(label, CONTROLS_WINDOW, ticks, max_tick, lambda _value: None)
    cv2.imshow(CONTROLS_WINDOW, controls_image(np.zeros(len(MOTOR_KEYS), dtype=float)))


def read_cv2_offsets(max_offset_deg: float) -> np.ndarray:
    center = int(round(max_offset_deg * 20))
    offsets = []
    for key in MOTOR_KEYS:
        label = key.replace(".pos", "")
        tick = cv2.getTrackbarPos(label, CONTROLS_WINDOW)
        offsets.append(round((tick - center) / 20.0, 1))
    return np.array(offsets, dtype=float)


def format_pose_line(name: str, values: np.ndarray) -> str:
    rounded = np.round(values, 1).tolist()
    return f"{name}: {rounded}"


def controls_image(offsets_deg: np.ndarray) -> np.ndarray:
    image = np.full((CONTROLS_HEIGHT, CONTROLS_WIDTH, 3), 245, dtype=np.uint8)
    cv2.putText(
        image,
        "Keys or terminal+Enter: 1-6 select | +/- 0.1 deg | [/] 1 deg | s save | n next | q quit",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    for index, (key, offset) in enumerate(zip(MOTOR_KEYS, offsets_deg)):
        row = 64 + index * 18
        cv2.putText(
            image,
            f"{key.replace('.pos', '')}: {offset:+.1f} deg",
            (16, row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
    return image


def set_cv2_offsets(offsets_deg: np.ndarray, max_offset_deg: float) -> None:
    center = int(round(max_offset_deg * 20))
    max_tick = center * 2
    for key, offset in zip(MOTOR_KEYS, offsets_deg):
        label = key.replace(".pos", "")
        tick = int(round(offset * 20.0)) + center
        cv2.setTrackbarPos(label, CONTROLS_WINDOW, int(np.clip(tick, 0, max_tick)))


def adjust_offset(
    offsets_deg: np.ndarray,
    selected_motor: int,
    delta_deg: float,
    max_offset_deg: float,
) -> np.ndarray:
    updated = offsets_deg.copy()
    updated[selected_motor] = round(
        float(np.clip(updated[selected_motor] + delta_deg, -max_offset_deg, max_offset_deg)),
        1,
    )
    set_cv2_offsets(updated, max_offset_deg)
    return updated


def read_terminal_command() -> str | None:
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return None
    command = sys.stdin.readline()
    if command == "":
        return None
    return command.strip().lower()


def key_to_command(key: int) -> str | None:
    if key == 255:
        return None
    if key in {ord("+"), ord("=")}:
        return "+"
    if key in {ord("-"), ord("_")}:
        return "-"
    if key in {ord("["), ord("]"), ord("s"), ord("n"), ord("q")}:
        return chr(key)
    if ord("1") <= key <= ord("6"):
        return chr(key)
    return None


def save_calibration(
    args: argparse.Namespace,
    lookup_path: Path,
    pose: dict[str, Any],
    pose_index: int,
    offsets_deg: np.ndarray,
    observation: dict[str, float] | None,
) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = args.output_dir / f"so101_joint_offsets_{timestamp}.json"
    suffix = 2
    while path.exists():
        path = args.output_dir / f"so101_joint_offsets_{timestamp}_{suffix}.json"
        suffix += 1
    observed = None if observation is None else observation_joints_deg(observation)
    displayed = pose["joints_deg"] + offsets_deg
    payload = {
        "schema": "so101_joint_command_offset_calibration_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "motor_offsets_deg": {
            key: float(offset) for key, offset in zip(MOTOR_KEYS, offsets_deg)
        },
        "board_origin": list(DEFAULT_BOARD_ORIGIN),
        "square_size": DEFAULT_SQUARE_SIZE,
        "lookup_json": str(lookup_path),
        "pose_name": pose["name"],
        "pose_index": pose_index,
        "waypoint_index": pose["waypoint_index"],
        "target_joints_deg": deg_list(pose["joints_deg"]),
        "displayed_sim_joints_deg": deg_list(displayed),
        "observed_real_joints_deg": None if observed is None else deg_list(observed),
        "args": {
            "port": args.port,
            "robot_id": args.robot_id,
            "move_duration_s": args.move_duration_s,
            "command_rate_hz": args.command_rate_hz,
            "max_offset_deg": args.max_offset_deg,
            "execute": args.execute,
        },
    }
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def print_pose_confirmation(pose: dict[str, Any], current: np.ndarray, offsets_deg: np.ndarray) -> bool:
    target = pose["joints_deg"] + offsets_deg
    delta = target - current
    print()
    print(f"Next pose: {pose['name']}")
    print("target_deg:", np.round(target, 3).tolist())
    print("current_deg:", np.round(current, 3).tolist())
    print("delta_deg:", np.round(delta, 3).tolist())
    print(f"max_delta_deg: {np.max(np.abs(delta)):.2f}")
    answer = input("Move real robot to this pose? [yes/no]: ").strip().lower()
    return answer in {"yes", "y"}


def move_real_robot_to_pose(
    follower: Any,
    pose: dict[str, Any],
    offsets_deg: np.ndarray,
    *,
    move_duration_s: float,
    command_rate_hz: float,
) -> dict[str, float]:
    if move_duration_s <= 0:
        raise ValueError("--move-duration-s must be positive")
    if command_rate_hz <= 0:
        raise ValueError("--command-rate-hz must be positive")

    observation = follower.get_observation()
    current = observation_joints_deg(observation)
    if not print_pose_confirmation(pose, current, offsets_deg):
        raise RuntimeError("Calibration move cancelled by user.")

    target = pose["joints_deg"] + offsets_deg
    steps = max(1, int(round(move_duration_s * command_rate_hz)))
    delay = 1.0 / command_rate_hz
    for index, alpha in enumerate(np.linspace(0.0, 1.0, steps + 1)[1:], start=1):
        joints = (1.0 - alpha) * current + alpha * target
        action = observation.copy()
        action.update(action_from_joints(joints))
        follower.send_action(action)
        if index % max(1, int(command_rate_hz)) == 0 or index == steps:
            print(f"move {index}/{steps}")
        time.sleep(delay)
        observation = follower.get_observation()
    return observation


def render_view(status_lines: list[str]) -> np.ndarray:
    projection = p.computeProjectionMatrixFOV(
        fov=55,
        aspect=VIEWER_WIDTH / VIEWER_HEIGHT,
        nearVal=0.01,
        farVal=10.0,
    )
    view = p.computeViewMatrix(
        cameraEyePosition=[0.0, -0.62, 0.28],
        cameraTargetPosition=[DEFAULT_BOARD_ORIGIN[0], DEFAULT_BOARD_ORIGIN[1], DEFAULT_BOARD_ORIGIN[2] + 0.03],
        cameraUpVector=[0.0, 0.0, 1.0],
    )
    _, _, rgba, _, _ = p.getCameraImage(
        VIEWER_WIDTH,
        VIEWER_HEIGHT,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    image = np.array(rgba, dtype=np.uint8).reshape((VIEWER_HEIGHT, VIEWER_WIDTH, 4))[:, :, :3]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    overlay = image.copy()
    cv2.rectangle(overlay, (8, 8), (VIEWER_WIDTH - 8, 178), (245, 245, 245), -1)
    image = cv2.addWeighted(overlay, 0.82, image, 0.18, 0)
    for row, line in enumerate(status_lines[:8]):
        cv2.putText(
            image,
            line,
            (20, 34 + row * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return image


def run_calibration_loop(
    args: argparse.Namespace,
    lookup_path: Path,
    poses: list[dict[str, Any]],
) -> None:
    robot_id, _board_id = setup_pybullet_scene(args.from_square, args.to_square)
    create_cv2_controls(args.max_offset_deg)
    pose_index = 0
    selected_motor = 0
    observation = None

    follower = None
    if args.execute:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        follower = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
        follower.connect()

    print("Controls: s=save offsets, n=next pose, q=quit")
    print("If window keys do not respond, type commands in this terminal and press Enter.")
    print("Examples: 1, +, -, ], [, s, n, q")
    print("Offsets are command corrections in degrees.")

    try:
        offsets_deg = np.zeros(len(MOTOR_KEYS), dtype=float)
        if follower is not None:
            observation = move_real_robot_to_pose(
                follower,
                poses[pose_index],
                offsets_deg,
                move_duration_s=args.move_duration_s,
                command_rate_hz=args.command_rate_hz,
            )

        while p.isConnected():
            pose = poses[pose_index]
            offsets_deg = read_cv2_offsets(args.max_offset_deg)
            commanded = pose["joints_deg"] + offsets_deg
            set_sim_joints(robot_id, commanded)

            if follower is not None:
                observation = follower.get_observation()
                action = observation.copy()
                action.update(action_from_joints(commanded))
                follower.send_action(action)

            observed_joints = (
                np.full(len(MOTOR_KEYS), np.nan)
                if observation is None
                else observation_joints_deg(observation)
            )
            status_lines = [
                f"pose {pose_index + 1}/{len(poses)}: {pose['name']} | waypoint: {pose['waypoint_index']}",
                f"selected motor {selected_motor + 1}: {MOTOR_KEYS[selected_motor]}",
                format_pose_line("target", pose["joints_deg"]),
                format_pose_line("offset", offsets_deg),
                format_pose_line("command", commanded),
                format_pose_line("observed", observed_joints),
                "keys in viewer: s save | n next | q quit",
            ]
            cv2.imshow(VIEWER_WINDOW, render_view(status_lines))
            cv2.imshow(CONTROLS_WINDOW, controls_image(offsets_deg))
            key_command = key_to_command(cv2.waitKey(1) & 0xFF)
            terminal_command = read_terminal_command()
            command = terminal_command or key_command

            if command in {"1", "2", "3", "4", "5", "6"}:
                selected_motor = int(command) - 1
                print(f"selected motor {selected_motor + 1}: {MOTOR_KEYS[selected_motor]}")
            if command == "+":
                offsets_deg = adjust_offset(offsets_deg, selected_motor, 0.1, args.max_offset_deg)
                print(f"{MOTOR_KEYS[selected_motor]} offset: {offsets_deg[selected_motor]:+.1f} deg")
            if command == "-":
                offsets_deg = adjust_offset(offsets_deg, selected_motor, -0.1, args.max_offset_deg)
                print(f"{MOTOR_KEYS[selected_motor]} offset: {offsets_deg[selected_motor]:+.1f} deg")
            if command == "]":
                offsets_deg = adjust_offset(offsets_deg, selected_motor, 1.0, args.max_offset_deg)
                print(f"{MOTOR_KEYS[selected_motor]} offset: {offsets_deg[selected_motor]:+.1f} deg")
            if command == "[":
                offsets_deg = adjust_offset(offsets_deg, selected_motor, -1.0, args.max_offset_deg)
                print(f"{MOTOR_KEYS[selected_motor]} offset: {offsets_deg[selected_motor]:+.1f} deg")
            if command == "s":
                path = save_calibration(
                    args,
                    lookup_path,
                    pose,
                    pose_index,
                    offsets_deg,
                    observation,
                )
                print(f"Saved calibration: {path}")
            if command == "n":
                if pose_index >= len(poses) - 1:
                    print("Already at final pose.")
                else:
                    pose_index += 1
                    if follower is not None:
                        observation = move_real_robot_to_pose(
                            follower,
                            poses[pose_index],
                            offsets_deg,
                            move_duration_s=args.move_duration_s,
                            command_rate_hz=args.command_rate_hz,
                        )
            if command == "q":
                break

            p.stepSimulation()
            time.sleep(1.0 / max(args.viewer_rate_hz, 1.0))
    finally:
        if follower is not None:
            follower.disconnect()
        if p.isConnected():
            p.disconnect()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live SO101 real/sim joint offset calibration using PyBullet sliders."
    )
    parser.add_argument("--lookup-json", type=Path, default=DEFAULT_LOOKUP_JSON)
    parser.add_argument("--port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--move-duration-s", type=float, default=5.0)
    parser.add_argument("--command-rate-hz", type=float, default=30.0)
    parser.add_argument("--viewer-rate-hz", type=float, default=10.0)
    parser.add_argument("--max-offset-deg", type=float, default=10.0)
    parser.add_argument("--from-square", default="d4")
    parser.add_argument("--to-square", default="d5")
    parser.add_argument(
        "--pose-set",
        choices=("comparison", "trajectory"),
        default="comparison",
        help="Use simple visual comparison poses or d4/d5 trajectory keypoints.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Connect to and command the real SO101 arm. Omit for a dry-run pose list.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lookup_path = args.lookup_json.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
    if lookup.get("schema") != "continuous_xy_lookup_v1":
        raise ValueError(f"Unsupported lookup schema in {lookup_path}")

    waypoints, closeidx, metadata = build_trajectory_from_lookup(lookup)
    if args.pose_set == "comparison":
        poses = build_comparison_poses(metadata)
    else:
        poses = build_guided_poses(waypoints, closeidx, metadata)
    validate_poses_clear_board(poses, args.from_square, args.to_square)

    print(f"lookup: {lookup_path}")
    print(f"pose set: {args.pose_set}")
    print(f"guided poses: {len(poses)}")
    for index, pose in enumerate(poses, start=1):
        print(f"{index}. {pose['name']} waypoint={pose['waypoint_index']}")

    if not args.execute:
        print("dry run only; add --execute to open GUI and command the real SO101 arm")
        return 0

    run_calibration_loop(args, lookup_path, poses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
