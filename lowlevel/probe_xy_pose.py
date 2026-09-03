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
    FILES,
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


def clamp_command_to_effective_limits(joints_deg: np.ndarray) -> np.ndarray:
    """Clamp command-only path samples to effective hardware joint limits.

    This is intentionally applied after solving/validating target poses. It
    handles cases where the real observed start pose is a fraction beyond a
    URDF limit, so the bridge can move back into range instead of failing
    before sending any command.
    """
    clipped = np.asarray(joints_deg, dtype=float).copy()
    for idx, (lo, hi) in enumerate(EFFECTIVE_LIMITS_DEG):
        clipped[idx] = np.clip(clipped[idx], lo, hi)
    return clipped


def clamp_path_to_effective_limits(path: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
    clipped_path = [clamp_command_to_effective_limits(step) for step in path]
    max_clip = 0.0
    for original, clipped in zip(path, clipped_path):
        max_clip = max(max_clip, float(np.max(np.abs(np.asarray(original) - clipped))))
    return clipped_path, max_clip


def board_min_xy(
    *,
    board_origin: tuple[float, float, float],
    square_size: float,
) -> tuple[float, float]:
    board_x, board_y = float(board_origin[0]), float(board_origin[1])
    half_board = 4.0 * float(square_size)
    return board_x - half_board, board_y - half_board


def board_grid_corner_world_xy(
    file_line_index: int,
    rank_line_index: int,
    *,
    board_origin: tuple[float, float, float],
    square_size: float,
) -> tuple[float, float]:
    """Return a chessboard grid intersection in world XY.

    file_line_index and rank_line_index run 0..8. A square's SW corner is
    (file_index, rank_index), while its NE corner is (file_index + 1,
    rank_index + 1).
    """
    if not (0 <= file_line_index <= 8 and 0 <= rank_line_index <= 8):
        raise ValueError("Board grid corner indices must be in [0, 8]")
    min_x, min_y = board_min_xy(board_origin=board_origin, square_size=square_size)
    return (
        min_x + float(file_line_index) * float(square_size),
        min_y + float(rank_line_index) * float(square_size),
    )


SQUARE_CORNER_GRID_OFFSETS = {
    "sw": (0, 0),
    "se": (1, 0),
    "nw": (0, 1),
    "ne": (1, 1),
}
DEFAULT_CORNER_SEQUENCE = ("sw", "se", "ne", "nw", "sw")


def square_corner_world_xy(
    square: str,
    corner: str,
    *,
    board_origin: tuple[float, float, float],
    square_size: float,
) -> tuple[float, float]:
    square = validate_square(square)
    corner = corner.lower()
    if corner not in SQUARE_CORNER_GRID_OFFSETS:
        raise ValueError(f"Unknown corner {corner!r}")
    file_index = FILES.index(square[0])
    rank_index = int(square[1]) - 1
    df, dr = SQUARE_CORNER_GRID_OFFSETS[corner]
    return board_grid_corner_world_xy(
        file_index + df,
        rank_index + dr,
        board_origin=board_origin,
        square_size=square_size,
    )


def parse_corner_sequence(raw: str) -> tuple[str, ...]:
    corners = tuple(
        item.strip().lower()
        for item in raw.replace(",", " ").split()
        if item.strip()
    )
    if not corners:
        raise ValueError("--corner-sequence-order must contain at least one corner")
    invalid = [corner for corner in corners if corner not in SQUARE_CORNER_GRID_OFFSETS]
    if invalid:
        raise ValueError(
            "Invalid corner names in --corner-sequence-order: "
            + ", ".join(invalid)
            + ". Use sw, se, nw, ne."
        )
    return corners


def target_axis_from_tilt(target_tilt_deg: float, tilt_azimuth_deg: float = 0.0) -> np.ndarray:
    """Gripper z-axis direction for a given tilt off straight-down.

    tilt_deg=0 gives (0, 0, -1) (straight down). tilt_azimuth_deg picks which
    horizontal direction the gripper leans toward as tilt increases.
    """
    tilt_rad = np.radians(target_tilt_deg)
    az_rad = np.radians(tilt_azimuth_deg)
    return np.array(
        [
            np.sin(tilt_rad) * np.cos(az_rad),
            np.sin(tilt_rad) * np.sin(az_rad),
            -np.cos(tilt_rad),
        ]
    )


def solve_down_pose(
    target_xyz: np.ndarray,
    *,
    tilt_weight: float = 0.05,
    restarts: int = 12,
    preferred_joints_deg: np.ndarray | None = None,
    joint_weight: float = 0.0,
    target_tilt_deg: float = 0.0,
    tilt_azimuth_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve joints putting the gripper at a fixed tilt at target_xyz.

    The incremental IK used by the trajectory builder converges on position but
    leaves the wrist several degrees off the intended tilt, which makes a probe
    ambiguous to read. Here position and tilt are optimised together instead,
    with multiple seeds because the arm is redundant enough to have local
    minima. target_tilt_deg=0 (default) is straight down, matching the
    original vertical-only behaviour exactly.
    """
    target = np.asarray(target_xyz, dtype=float)
    target_axis = target_axis_from_tilt(target_tilt_deg, tilt_azimuth_deg)

    def residual(q: np.ndarray) -> np.ndarray:
        pose = kinematics.forward_kinematics(np.concatenate([q, [0.0]]))
        pos = pose[:3, 3] - target
        # Drive the gripper z-axis to target_axis; scaled so a degree of tilt
        # error trades against roughly a tenth of a millimetre of position.
        axis = pose[:3, 2] - target_axis
        return np.concatenate([pos, tilt_weight * axis])

    lo = np.array([b[0] for b in EFFECTIVE_LIMITS_DEG], dtype=float)
    hi = np.array([b[1] for b in EFFECTIVE_LIMITS_DEG], dtype=float)

    rng = np.random.default_rng(0)
    seeds = [np.clip(DEFAULT_HOME[:5].copy(), lo, hi)]
    if preferred_joints_deg is not None:
        seeds.insert(0, np.clip(np.asarray(preferred_joints_deg, dtype=float)[:5], lo, hi))
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
        t = float(np.degrees(np.arccos(np.clip(np.dot(pose[:3, 2], target_axis), -1.0, 1.0))))
        score = pos_err * 1000.0 + t
        if preferred_joints_deg is not None and joint_weight > 0.0:
            preferred = np.asarray(preferred_joints_deg, dtype=float)[:5]
            score += joint_weight * float(np.linalg.norm(joints[:5] - preferred))
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


def interpolated_steps(start_deg: np.ndarray, end_deg: np.ndarray, steps: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    steps = max(1, int(steps))
    for k in range(1, steps + 1):
        a = k / steps
        out.append((1.0 - a) * start_deg + a * end_deg)
    return out


def build_corner_sequence_path(
    current_deg: np.ndarray,
    sequence: list[dict[str, Any]],
    *,
    steps_per_corner: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """current -> via0 -> corner0 -> via0 -> via1 -> corner1 ..."""
    path: list[np.ndarray] = []
    events: list[dict[str, Any]] = []
    if not sequence:
        return path, events

    up_steps = max(1, int(round(steps_per_corner * 0.30)))
    cross_steps = max(1, int(round(steps_per_corner * 0.40)))
    down_steps = max(1, steps_per_corner - up_steps - cross_steps)

    first_via = np.asarray(sequence[0]["via_joints_deg"], dtype=float)
    path.extend(interpolated_steps(current_deg, first_via, up_steps + cross_steps))
    events.append(
        {
            "path_index": len(path),
            "event": "reached_first_via",
            "corner": sequence[0]["corner"],
        }
    )

    previous_via = first_via
    for idx, pose in enumerate(sequence):
        corner_joints = np.asarray(pose["joints_deg"], dtype=float)
        via_joints = np.asarray(pose["via_joints_deg"], dtype=float)
        if idx > 0:
            path.extend(interpolated_steps(previous_via, via_joints, cross_steps))
            events.append(
                {
                    "path_index": len(path),
                    "event": "reached_via",
                    "corner": pose["corner"],
                }
            )
        path.extend(interpolated_steps(via_joints, corner_joints, down_steps))
        events.append(
            {
                "path_index": len(path),
                "event": "reached_corner",
                "corner": pose["corner"],
                "label": pose["label"],
                "sequence_index": idx + 1,
                "sequence_count": len(sequence),
            }
        )
        path.extend(interpolated_steps(corner_joints, via_joints, up_steps))
        events.append(
            {
                "path_index": len(path),
                "event": "lifted_from_corner",
                "corner": pose["corner"],
                "label": pose["label"],
            }
        )
        previous_via = via_joints
    return path, events


def send_command_path(
    follower: Any,
    observation: dict[str, float],
    path: list[np.ndarray],
    *,
    command_delay: float,
    progress_label: str,
    progress_every: int = 50,
) -> dict[str, float]:
    for index, step in enumerate(path, start=1):
        action = observation.copy()
        action.update(action_from_joints(step))
        follower.send_action(action)
        if index % progress_every == 0 or index == len(path):
            print(f"  {progress_label} {index}/{len(path)}")
        time.sleep(command_delay)
        observation = follower.get_observation()
    return observation


def move_home_from_corner(
    follower: Any,
    observation: dict[str, float],
    current_deg: np.ndarray,
    corner_pose: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, float]:
    via_joints = np.asarray(corner_pose["via_joints_deg"], dtype=float)
    home_joints = np.asarray(DEFAULT_HOME, dtype=float)
    lift_steps = max(1, int(round(args.move_steps * 0.30)))
    home_steps = max(1, int(round(args.move_steps * 0.70)))
    return_path = (
        interpolated_steps(current_deg, via_joints, lift_steps)
        + interpolated_steps(via_joints, home_joints, home_steps)
    )
    return_path, max_path_clip = clamp_path_to_effective_limits(return_path)
    bad = [v for step in return_path for v in limit_violations(step)]
    if bad:
        raise RuntimeError(
            "Return-home path exceeds joint limits: " + "; ".join(sorted(set(bad)))
        )
    print(
        f"Returning home via {corner_pose['label']}_via "
        f"({len(return_path)} steps)."
    )
    if max_path_clip > 1e-6:
        print(f"return clamp: max {max_path_clip:.3f} deg")
    observation = send_command_path(
        follower,
        observation,
        return_path,
        command_delay=args.command_delay,
        progress_label="return_home",
    )
    settled = observation_joints_deg(follower.get_observation())
    print("home observed_deg:", np.round(settled, 3).tolist())
    print("home cmd-obs diff:", np.round(settled - home_joints, 3).tolist())
    return observation


def solve_probe_target(
    target: np.ndarray,
    *,
    ik_restarts: int,
    max_residual_m: float,
    max_tilt_deg: float,
    preferred_joints_deg: np.ndarray | None = None,
    joint_weight: float = 0.0,
    target_tilt_deg: float = 0.0,
    tilt_azimuth_deg: float = 0.0,
    tilt_weight: float = 0.05,
) -> dict[str, Any]:
    joints, achieved, down_err_deg = solve_down_pose(
        target,
        restarts=ik_restarts,
        preferred_joints_deg=preferred_joints_deg,
        joint_weight=joint_weight,
        target_tilt_deg=target_tilt_deg,
        tilt_azimuth_deg=tilt_azimuth_deg,
        tilt_weight=tilt_weight,
    )
    residual_m = float(np.linalg.norm(achieved - target))
    violations = limit_violations(joints)
    reachable = (
        residual_m <= max_residual_m
        and down_err_deg <= max_tilt_deg
        and not violations
    )
    return {
        "target_world_xyz": target.tolist(),
        "achieved_world_xyz": achieved.tolist(),
        "ik_residual_m": residual_m,
        "ik_residual_mm": residual_m * 1000.0,
        "off_vertical_deg": down_err_deg,
        "joints_deg": joints.tolist(),
        "limit_violations": violations,
        "reachable": reachable,
    }


def square_corner_labels(square: str) -> dict[str, tuple[int, int]]:
    square = validate_square(square)
    file_index = FILES.index(square[0])
    rank_index = int(square[1]) - 1
    return {
        label: (file_index + df, rank_index + dr)
        for label, (df, dr) in SQUARE_CORNER_GRID_OFFSETS.items()
    }


def run_corner_coverage(args: argparse.Namespace) -> int:
    board_origin = (args.board_origin_x, args.board_origin_y, args.board_origin_z)
    surface_z = args.board_origin_z + 0.005
    z = surface_z + args.height_above_surface
    max_residual_m = args.max_residual_mm / 1000.0

    corner_results: dict[str, dict[str, Any]] = {}
    for file_line_index in range(9):
        for rank_line_index in range(9):
            x, y = board_grid_corner_world_xy(
                file_line_index,
                rank_line_index,
                board_origin=board_origin,
                square_size=args.square_size,
            )
            label = f"file_line_{file_line_index}_rank_line_{rank_line_index}"
            target = np.array([x, y, z], dtype=float)
            try:
                result = solve_probe_target(
                    target,
                    ik_restarts=args.ik_restarts,
                    max_residual_m=max_residual_m,
                    max_tilt_deg=args.max_tilt_deg,
                )
            except Exception as exc:
                result = {
                    "target_world_xyz": target.tolist(),
                    "achieved_world_xyz": None,
                    "ik_residual_m": None,
                    "ik_residual_mm": None,
                    "off_vertical_deg": None,
                    "joints_deg": None,
                    "limit_violations": [],
                    "reachable": False,
                    "error": str(exc),
                }
            result.update(
                {
                    "label": label,
                    "file_line_index": file_line_index,
                    "rank_line_index": rank_line_index,
                }
            )
            corner_results[f"{file_line_index},{rank_line_index}"] = result

    square_results: dict[str, dict[str, Any]] = {}
    for rank in range(1, 9):
        for file_name in FILES:
            square = f"{file_name}{rank}"
            corner_indices = square_corner_labels(square)
            corner_reach = {
                label: corner_results[f"{idx[0]},{idx[1]}"]["reachable"]
                for label, idx in corner_indices.items()
            }
            reachable_count = sum(1 for ok in corner_reach.values() if ok)
            square_results[square] = {
                "corner_indices": {
                    label: [int(idx[0]), int(idx[1])]
                    for label, idx in corner_indices.items()
                },
                "corner_reachable": corner_reach,
                "reachable_corner_count": reachable_count,
                "all_corners_reachable": reachable_count == 4,
                "any_corner_reachable": reachable_count > 0,
            }

    all_corner_squares = [
        square for square, result in square_results.items()
        if result["all_corners_reachable"]
    ]
    partial_squares = [
        square for square, result in square_results.items()
        if result["any_corner_reachable"] and not result["all_corners_reachable"]
    ]
    no_corner_squares = [
        square for square, result in square_results.items()
        if not result["any_corner_reachable"]
    ]

    print("corner coverage probe")
    print(f"surface z        : {surface_z:.5f}")
    print(f"probe z          : {z:.5f} (+{args.height_above_surface:.3f} m)")
    print(f"max residual     : {args.max_residual_mm:.2f} mm")
    print(f"max off-vertical : {args.max_tilt_deg:.2f} deg")
    print("legend           : A=all four corners, P=partial, .=none")
    for rank in range(8, 0, -1):
        row = []
        for file_name in FILES:
            result = square_results[f"{file_name}{rank}"]
            if result["all_corners_reachable"]:
                row.append("A")
            elif result["any_corner_reachable"]:
                row.append("P")
            else:
                row.append(".")
        print(f"{rank} | {' '.join(row)}")
    print("    a b c d e f g h")
    print(f"all corners : {len(all_corner_squares)} squares")
    print("  " + " ".join(all_corner_squares))
    print(f"partial     : {len(partial_squares)} squares")
    print("  " + " ".join(partial_squares))
    print(f"none        : {len(no_corner_squares)} squares")
    print("  " + " ".join(no_corner_squares))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema": "probe_xy_corner_coverage_v1",
        "board_origin": [args.board_origin_x, args.board_origin_y, args.board_origin_z],
        "surface_z": surface_z,
        "height_above_surface": args.height_above_surface,
        "probe_z": z,
        "square_size": args.square_size,
        "max_residual_mm": args.max_residual_mm,
        "max_tilt_deg": args.max_tilt_deg,
        "ik_restarts": args.ik_restarts,
        "corner_results": corner_results,
        "square_results": square_results,
        "all_corners_reachable_squares": all_corner_squares,
        "partial_corner_reachable_squares": partial_squares,
        "no_corner_reachable_squares": no_corner_squares,
    }
    out_path = args.output_dir / "probe_xy_corner_coverage.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"saved        : {out_path}")
    return 0


def run_square_corner_sequence(args: argparse.Namespace) -> int:
    if args.square is None:
        raise ValueError("Provide --square with --corner-sequence, e.g. --square d4")

    square = validate_square(args.square)
    corner_order = parse_corner_sequence(args.corner_sequence_order)
    board_origin = (args.board_origin_x, args.board_origin_y, args.board_origin_z)
    surface_z = args.board_origin_z + 0.005
    probe_z = surface_z + args.height_above_surface
    via_z = surface_z + args.via_height
    max_residual_m = args.max_residual_mm / 1000.0

    sequence: list[dict[str, Any]] = []
    previous_probe_joints: np.ndarray | None = None
    previous_via_joints: np.ndarray | None = None
    for index, corner in enumerate(corner_order, start=1):
        x, y = square_corner_world_xy(
            square,
            corner,
            board_origin=board_origin,
            square_size=args.square_size,
        )
        target = np.array([x, y, probe_z], dtype=float)
        via_target = np.array([x, y, via_z], dtype=float)
        probe = solve_probe_target(
            target,
            ik_restarts=args.ik_restarts,
            max_residual_m=max_residual_m,
            max_tilt_deg=args.max_tilt_deg,
            preferred_joints_deg=previous_probe_joints,
            joint_weight=args.sequence_joint_weight,
        )
        via_joints, via_achieved, via_tilt = solve_down_pose(
            via_target,
            restarts=args.ik_restarts,
            preferred_joints_deg=(
                previous_via_joints
                if previous_via_joints is not None
                else previous_probe_joints
            ),
            joint_weight=args.sequence_joint_weight,
        )
        via_residual_m = float(np.linalg.norm(via_achieved - via_target))
        via_violations = limit_violations(via_joints)
        if not probe["reachable"]:
            details = probe.get("error") or (
                f"residual={probe['ik_residual_mm']:.2f} mm, "
                f"tilt={probe['off_vertical_deg']:.2f} deg, "
                f"violations={probe['limit_violations']}"
            )
            raise RuntimeError(f"{square}_{corner} is not reachable: {details}")
        if via_violations:
            raise RuntimeError(
                f"{square}_{corner} via pose exceeds joint limits: "
                + "; ".join(via_violations)
            )
        sequence.append(
            {
                "index": index,
                "square": square,
                "corner": corner,
                "label": f"{square}_{corner}_corner",
                "target_world_xyz": target.tolist(),
                "achieved_world_xyz": probe["achieved_world_xyz"],
                "ik_residual_m": probe["ik_residual_m"],
                "ik_residual_mm": probe["ik_residual_mm"],
                "off_vertical_deg": probe["off_vertical_deg"],
                "joints_deg": probe["joints_deg"],
                "via_target_world_xyz": via_target.tolist(),
                "via_achieved_world_xyz": via_achieved.tolist(),
                "via_residual_m": via_residual_m,
                "via_residual_mm": via_residual_m * 1000.0,
                "via_off_vertical_deg": via_tilt,
                "via_joints_deg": via_joints.tolist(),
            }
        )
        previous_probe_joints = np.asarray(probe["joints_deg"], dtype=float)
        previous_via_joints = np.asarray(via_joints, dtype=float)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema": "probe_xy_square_corner_sequence_v1",
        "square": square,
        "corner_order": list(corner_order),
        "board_origin": [args.board_origin_x, args.board_origin_y, args.board_origin_z],
        "surface_z": surface_z,
        "height_above_surface": args.height_above_surface,
        "probe_z": probe_z,
        "via_height": args.via_height,
        "via_z": via_z,
        "square_size": args.square_size,
        "max_residual_mm": args.max_residual_mm,
        "max_tilt_deg": args.max_tilt_deg,
        "ik_restarts": args.ik_restarts,
        "sequence_joint_weight": args.sequence_joint_weight,
        "move_steps_per_corner": args.move_steps,
        "command_delay": args.command_delay,
        "sequence": sequence,
    }
    out_path = args.output_dir / f"probe_xy_{square}_corner_sequence.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"{square} corner sequence")
    print(f"order       : {' -> '.join(corner_order)}")
    print(f"surface z   : {surface_z:.5f}")
    print(f"probe z     : {probe_z:.5f} (+{args.height_above_surface:.3f} m)")
    print(f"via z       : {via_z:.5f}")
    for pose in sequence:
        target = pose["target_world_xyz"]
        joints = np.round(np.asarray(pose["joints_deg"], dtype=float), 3).tolist()
        print(
            f"{pose['index']}. {pose['label']:13s} "
            f"xyz=[{target[0]:.5f}, {target[1]:.5f}, {target[2]:.5f}] "
            f"residual={pose['ik_residual_mm']:.2f}mm "
            f"tilt={pose['off_vertical_deg']:.2f}deg "
            f"joints={joints}"
        )
    print(f"saved       : {out_path}")

    return execute_sequence(args, sequence, prompt_label=f"{square} corner sequence")


def execute_sequence(args: argparse.Namespace, sequence: list[dict[str, Any]], prompt_label: str) -> int:
    if not args.execute:
        print(
            "dry run only; add --execute to cycle the real SO101 arm through "
            "this sequence until q is entered."
        )
        return 0

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    follower = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
    follower.connect()
    try:
        observation = follower.get_observation()
        current = observation_joints_deg(observation)
        path, events = build_corner_sequence_path(
            current,
            sequence,
            steps_per_corner=args.move_steps,
        )
        path, max_path_clip = clamp_path_to_effective_limits(path)
        bad = [v for step in path for v in limit_violations(step)]
        if bad:
            raise RuntimeError(
                "Sequence path exceeds joint limits: "
                + "; ".join(sorted(set(bad)))
            )
        max_delta = float(
            max(np.max(np.abs(np.asarray(step, dtype=float) - current)) for step in path)
        )
        print("current_deg:", np.round(current, 3).tolist())
        print(
            f"path       : {len(path)} total steps through {len(sequence)} positions; "
            f"max delta from current {max_delta:.2f} deg"
        )
        if max_path_clip > 1e-6:
            print(f"path clamp : max {max_path_clip:.3f} deg to stay within effective limits")
        confirmation = input(f"Move through {prompt_label}? [yes/no]: ").strip().lower()
        if confirmation not in {"yes", "y"}:
            print("Cancelled.")
            return 1

        cycle_index = 1
        while True:
            print(f"cycle {cycle_index}: {len(path)} steps")
            event_by_index: dict[int, list[dict[str, Any]]] = {}
            for event in events:
                event_by_index.setdefault(int(event["path_index"]), []).append(event)
            for index, step in enumerate(path, start=1):
                action = observation.copy()
                action.update(action_from_joints(step))
                follower.send_action(action)
                for event in event_by_index.get(index, []):
                    print(
                        f"  {index}/{len(path)} {event['event']} "
                        f"{event.get('label', event['corner'])}"
                    )
                    if event["event"] == "reached_corner":
                        if args.corner_hold_s > 0:
                            time.sleep(args.corner_hold_s)
                        if event["sequence_index"] < event["sequence_count"]:
                            prompt = (
                                f"At {event['label']}. Press Enter to move to next position "
                                "(q to stop): "
                            )
                        else:
                            prompt = (
                                f"At final {event['label']}. Press Enter to start next cycle "
                                "(q to stop): "
                            )
                        response = input(prompt).strip().lower()
                        if response in {"q", "quit", "stop", "no", "n"}:
                            print("Stopped at position by user request.")
                            observation = follower.get_observation()
                            settled = observation_joints_deg(observation)
                            print("observed_deg:", np.round(settled, 3).tolist())
                            corner_pose = sequence[int(event["sequence_index"]) - 1]
                            move_home_from_corner(
                                follower,
                                observation,
                                settled,
                                corner_pose,
                                args,
                            )
                            return 0
                if index % 100 == 0 or index == len(path):
                    print(f"  {index}/{len(path)}")
                time.sleep(args.command_delay)
                observation = follower.get_observation()

            settled = observation_joints_deg(follower.get_observation())
            final_joints = np.asarray(sequence[-1]["via_joints_deg"], dtype=float)
            print("observed_deg:", np.round(settled, 3).tolist())
            print("final-via cmd-obs diff:", np.round(settled - final_joints, 3).tolist())

            current = settled
            path, events = build_corner_sequence_path(
                current,
                sequence,
                steps_per_corner=args.move_steps,
            )
            path, max_path_clip = clamp_path_to_effective_limits(path)
            bad = [v for step in path for v in limit_violations(step)]
            if bad:
                raise RuntimeError(
                    "Next sequence cycle exceeds joint limits: "
                    + "; ".join(sorted(set(bad)))
                )
            cycle_index += 1
            if max_path_clip > 1e-6:
                print(f"next path clamp: max {max_path_clip:.3f} deg")
    finally:
        follower.disconnect()
    return 0


def run_xy_point_sequence(args: argparse.Namespace) -> int:
    if not args.xy_points or len(args.xy_points) % 2 != 0:
        raise ValueError("--xy-points needs an even count of floats (x1 y1 x2 y2 ...)")
    points = [
        (args.xy_points[i], args.xy_points[i + 1]) for i in range(0, len(args.xy_points), 2)
    ]
    max_residual_m = args.max_residual_mm / 1000.0

    sequence: list[dict[str, Any]] = []
    previous_probe_joints: np.ndarray | None = None
    previous_via_joints: np.ndarray | None = None
    for index, (x, y) in enumerate(points, start=1):
        label = f"xy_{index}_({x:.3f},{y:.3f})"
        target = np.array([x, y, args.xy_height], dtype=float)
        via_target = np.array([x, y, args.xy_via_height], dtype=float)
        probe = solve_probe_target(
            target,
            ik_restarts=args.ik_restarts,
            max_residual_m=max_residual_m,
            max_tilt_deg=args.max_tilt_deg,
            preferred_joints_deg=previous_probe_joints,
            joint_weight=args.sequence_joint_weight,
            target_tilt_deg=args.xy_tilt_deg,
            tilt_azimuth_deg=args.xy_tilt_azimuth_deg,
            tilt_weight=args.xy_tilt_weight,
        )
        # Via/transit poses stay vertical (tilt_deg=0) regardless of the
        # target tilt -- only the held position needs to be tilted.
        via_joints, via_achieved, via_tilt = solve_down_pose(
            via_target,
            restarts=args.ik_restarts,
            preferred_joints_deg=(
                previous_via_joints if previous_via_joints is not None else previous_probe_joints
            ),
            joint_weight=args.sequence_joint_weight,
        )
        via_residual_m = float(np.linalg.norm(via_achieved - via_target))
        via_violations = limit_violations(via_joints)
        if not probe["reachable"]:
            details = probe.get("error") or (
                f"residual={probe['ik_residual_mm']:.2f} mm, "
                f"tilt={probe['off_vertical_deg']:.2f} deg, "
                f"violations={probe['limit_violations']}"
            )
            raise RuntimeError(f"{label} is not reachable: {details}")
        if via_violations:
            raise RuntimeError(f"{label} via pose exceeds joint limits: " + "; ".join(via_violations))
        sequence.append(
            {
                "index": index,
                "corner": label,
                "label": label,
                "target_world_xyz": target.tolist(),
                "achieved_world_xyz": probe["achieved_world_xyz"],
                "ik_residual_m": probe["ik_residual_m"],
                "ik_residual_mm": probe["ik_residual_mm"],
                "off_vertical_deg": probe["off_vertical_deg"],
                "joints_deg": probe["joints_deg"],
                "via_target_world_xyz": via_target.tolist(),
                "via_achieved_world_xyz": via_achieved.tolist(),
                "via_residual_m": via_residual_m,
                "via_residual_mm": via_residual_m * 1000.0,
                "via_off_vertical_deg": via_tilt,
                "via_joints_deg": via_joints.tolist(),
            }
        )
        previous_probe_joints = np.asarray(probe["joints_deg"], dtype=float)
        previous_via_joints = np.asarray(via_joints, dtype=float)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema": "probe_xy_point_sequence_v1",
        "points": points,
        "xy_height": args.xy_height,
        "xy_via_height": args.xy_via_height,
        "max_residual_mm": args.max_residual_mm,
        "max_tilt_deg": args.max_tilt_deg,
        "ik_restarts": args.ik_restarts,
        "sequence_joint_weight": args.sequence_joint_weight,
        "move_steps_per_position": args.move_steps,
        "command_delay": args.command_delay,
        "sequence": sequence,
    }
    out_path = args.output_dir / "probe_xy_point_sequence.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print("xy point sequence")
    print(f"height      : {args.xy_height:.5f} (absolute, above robot origin)")
    print(f"via height  : {args.xy_via_height:.5f}")
    for pose in sequence:
        target = pose["target_world_xyz"]
        joints = np.round(np.asarray(pose["joints_deg"], dtype=float), 3).tolist()
        print(
            f"{pose['index']}. {pose['label']:22s} "
            f"xyz=[{target[0]:.5f}, {target[1]:.5f}, {target[2]:.5f}] "
            f"residual={pose['ik_residual_mm']:.2f}mm "
            f"tilt={pose['off_vertical_deg']:.2f}deg "
            f"joints={joints}"
        )
    print(f"saved       : {out_path}")

    return execute_sequence(args, sequence, prompt_label="xy point sequence")


def resolve_target_xy(args: argparse.Namespace) -> tuple[float, float, str]:
    board_origin = (args.board_origin_x, args.board_origin_y, args.board_origin_z)
    if args.square is not None:
        square = validate_square(args.square)
        if args.square_corner == "center":
            x, y = location_world_xy(
                square,
                board_origin=board_origin,
                square_size=args.square_size,
            )
            return x, y, square
        x, y = square_corner_world_xy(
            square,
            args.square_corner,
            board_origin=board_origin,
            square_size=args.square_size,
        )
        return x, y, f"{square}_{args.square_corner}_corner"
    if args.xy is None:
        raise ValueError("Provide either --square or --xy")
    return float(args.xy[0]), float(args.xy[1]), "xy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--square", help="Board square to probe, e.g. d4.")
    parser.add_argument(
        "--square-corner",
        choices=("center", "sw", "se", "nw", "ne"),
        default="center",
        help=(
            "Probe a square centre or one named square corner. "
            "Corners are board-frame directions: sw/se/nw/ne."
        ),
    )
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
    parser.add_argument(
        "--corner-coverage",
        action="store_true",
        help=(
            "Dry-run all 9x9 square-corner grid intersections and report which "
            "squares have all/any corners reachable."
        ),
    )
    parser.add_argument(
        "--corner-sequence",
        action="store_true",
        help=(
            "Solve and optionally execute an up-and-over path through the named "
            "corners of --square, cycling until q is entered."
        ),
    )
    parser.add_argument(
        "--corner-sequence-order",
        default=",".join(DEFAULT_CORNER_SEQUENCE),
        help="Corner order for --corner-sequence, e.g. sw,se,ne,nw,sw.",
    )
    parser.add_argument(
        "--xy-sequence",
        action="store_true",
        help=(
            "Solve and optionally execute an up-and-over path through arbitrary "
            "world XY points at a fixed absolute height (--xy-height), cycling "
            "until q is entered. Use --xy-points x1 y1 x2 y2 ..."
        ),
    )
    parser.add_argument(
        "--xy-points",
        type=float,
        nargs="+",
        default=None,
        help="Flat list of world XY pairs for --xy-sequence, e.g. x1 y1 x2 y2 x3 y3 x4 y4.",
    )
    parser.add_argument(
        "--xy-height",
        type=float,
        default=0.05,
        help="Absolute target height above the robot origin for --xy-sequence (default 0.05m).",
    )
    parser.add_argument(
        "--xy-via-height",
        type=float,
        default=0.15,
        help="Absolute transit height above the robot origin for --xy-sequence (default 0.15m).",
    )
    parser.add_argument(
        "--xy-tilt-deg",
        type=float,
        default=0.0,
        help=(
            "Gripper tilt off straight-down for --xy-sequence target poses "
            "(default 0 = vertical). Via/transit poses stay vertical regardless."
        ),
    )
    parser.add_argument(
        "--xy-tilt-azimuth-deg",
        type=float,
        default=0.0,
        help="Horizontal direction the gripper leans toward for --xy-tilt-deg (default 0).",
    )
    parser.add_argument(
        "--xy-tilt-weight",
        type=float,
        default=2.0,
        help=(
            "IK residual weight for hitting --xy-tilt-deg exactly (default 2.0). "
            "The default tilt_weight used for near-vertical solves is too weak "
            "to converge on larger deliberate tilts; raise further if the "
            "solved tilt still falls noticeably short of the target."
        ),
    )
    parser.add_argument(
        "--corner-hold-s",
        type=float,
        default=1.0,
        help=(
            "Settling seconds before the Enter/q prompt at each reached corner "
            "during --corner-sequence execution. Enter continues; q returns home and quits."
        ),
    )
    parser.add_argument(
        "--sequence-joint-weight",
        type=float,
        default=0.02,
        help=(
            "Continuity preference for --corner-sequence IK branch selection. "
            "Higher values prefer smaller joint jumps."
        ),
    )
    parser.add_argument(
        "--max-residual-mm",
        type=float,
        default=2.0,
        help="Coverage pass threshold for solved probe position residual.",
    )
    parser.add_argument(
        "--max-tilt-deg",
        type=float,
        default=2.0,
        help="Coverage pass threshold for gripper off-vertical angle.",
    )
    parser.add_argument(
        "--ik-restarts",
        type=int,
        default=12,
        help="Number of random IK restarts per target.",
    )
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
    if args.ik_restarts <= 0:
        raise ValueError("--ik-restarts must be positive")
    if args.max_residual_mm <= 0:
        raise ValueError("--max-residual-mm must be positive")
    if args.max_tilt_deg < 0:
        raise ValueError("--max-tilt-deg must be non-negative")
    if args.corner_hold_s < 0:
        raise ValueError("--corner-hold-s must be non-negative")
    if args.sequence_joint_weight < 0:
        raise ValueError("--sequence-joint-weight must be non-negative")
    if sum([args.corner_coverage, args.corner_sequence, args.xy_sequence]) > 1:
        raise ValueError("Use only one of --corner-coverage, --corner-sequence, --xy-sequence")
    if args.corner_coverage:
        if args.execute:
            raise ValueError("--corner-coverage is a dry-run report; omit --execute")
        return run_corner_coverage(args)
    if args.corner_sequence:
        return run_square_corner_sequence(args)
    if args.xy_sequence:
        return run_xy_point_sequence(args)

    x, y, label = resolve_target_xy(args)
    surface_z = args.board_origin_z + 0.005
    z = surface_z + args.height_above_surface
    target = np.array([x, y, z], dtype=float)

    joints, achieved, down_err_deg = solve_down_pose(target, restarts=args.ik_restarts)
    via_target = np.array([x, y, surface_z + args.via_height], dtype=float)
    via_joints, via_achieved, via_tilt = solve_down_pose(
        via_target,
        restarts=args.ik_restarts,
    )

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
        "square_corner": args.square_corner,
        "ik_restarts": args.ik_restarts,
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
        path, max_path_clip = clamp_path_to_effective_limits(path)

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
        if max_path_clip > 1e-6:
            print(f"path clamp : max {max_path_clip:.3f} deg to stay within effective limits")
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
