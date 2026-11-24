#!/usr/bin/env python3
"""Run a saved continuous XY lookup trajectory on a real SO101 follower arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, NamedTuple

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
DEFAULT_REAL_BOARD_ORIGIN = (
    float(DEFAULT_BOARD_ORIGIN[0]),
    float(DEFAULT_BOARD_ORIGIN[1]),
    0.025,
)
# Lookups store their from/to points as world XY for the board at
# DEFAULT_BOARD_ORIGIN (the builders' board). A board placed elsewhere moves
# both points by the same amount, so the move stays the same squares.
LOOKUP_BOARD_ORIGIN_XY = (float(DEFAULT_BOARD_ORIGIN[0]), float(DEFAULT_BOARD_ORIGIN[1]))
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
    *,
    board_origin: tuple[float, float, float],
    zero_grasp_offset: bool = False,
) -> tuple[list[np.ndarray], int, dict[str, Any]]:
    shift_x = float(board_origin[0]) - LOOKUP_BOARD_ORIGIN_XY[0]
    shift_y = float(board_origin[1]) - LOOKUP_BOARD_ORIGIN_XY[1]
    saved_start = point_from_saved(lookup["from"])
    saved_target = point_from_saved(lookup["to"])
    start = XYPoint(saved_start.x + shift_x, saved_start.y + shift_y, name=saved_start.name)
    target = XYPoint(saved_target.x + shift_x, saved_target.y + shift_y, name=saved_target.name)
    grasp_offset = np.array(lookup["source_grasp_offset"], dtype=float)
    place_offset = np.array(lookup["selected_place_offset"], dtype=float)
    search = lookup["search"]
    metrics = lookup.get("metrics", {})
    lower_place_path_bias = metrics.get("lower_place_path_bias")
    if lower_place_path_bias is not None:
        lower_place_path_bias = np.array(lower_place_path_bias, dtype=float)

    if zero_grasp_offset:
        # The saved offsets and the lower-place bias were solved together, so drop
        # them as a set rather than leaving a bias tuned for offsets we removed.
        grasp_offset = np.zeros(3, dtype=float)
        place_offset = np.zeros(3, dtype=float)
        lower_place_path_bias = None

    trajectory_home = metrics.get("trajectory_home_joints_deg")
    if trajectory_home is not None:
        trajectory_home = normalized_home_joints(trajectory_home)

    with temporary_home_joints(trajectory_home):
        movelist, closeidx, traj_metrics = pickupmove_traj_with_metrics(
            start,
            target,
            board_origin=board_origin,
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
        "lookup_board_origin_xy": LOOKUP_BOARD_ORIGIN_XY,
        "board_shift_xy": (shift_x, shift_y),
        "grasp_offset": grasp_offset,
        "place_offset": place_offset,
        "lookup_move_id": lookup.get("move_id"),
        "lookup_success": lookup.get("success"),
        "lookup_metrics": metrics,
        "regenerated_trajectory_metrics": traj_metrics,
        "closeidx": closeidx,
        "board_origin": board_origin,
        "zero_grasp_offset": zero_grasp_offset,
        "trajectory_home_joints_deg": (
            DEFAULT_HOME.copy() if trajectory_home is None else trajectory_home.copy()
        ),
    }
    return waypoints, closeidx, metadata


def compensate_waypoints(waypoints: list[np.ndarray], compensator: Any) -> list[np.ndarray]:
    """Motor command per waypoint that should put the real arm at that URDF waypoint.

    Each waypoint gets command = (q - offset) + f, as in probe_xy_pose.py --compensate
    (see friction/joint_compensation.py). A joint's friction direction is the way it moved into
    the waypoint; one that holds still (under MIN_MOVE_DEG, e.g. while the gripper
    closes) keeps its last direction, since it's still resting where that move left
    it, so its command doesn't jump by f. The first waypoint has no direction yet and
    gets the offsets only. The model is static, so the correction is exact only where
    the arm comes to rest (grasp, release); in transit it's the quasi-static estimate.
    """
    from friction.joint_compensation import approach_dirs

    commands = []
    dirs = np.zeros(len(MOTOR_KEYS))
    previous = None
    for waypoint in waypoints:
        motor_want = np.asarray(waypoint, dtype=float) - compensator.offsets_deg
        if previous is not None:
            step = approach_dirs(motor_want, previous)
            dirs = np.where(step != 0.0, step, dirs)
        command, _ = compensator.compensate(waypoint, move_dirs=dirs)
        commands.append(command)
        previous = motor_want
    return commands


def baseline_offsets_deg() -> np.ndarray:
    """--baseline's offsets (6,): the hand-measured calibration in multisim_chess_fast.py's
    DEFAULT_JOINT_OFFSETS_DEG (URDF angle = motor angle + offset), read without importing it,
    which is slow. The gripper has none.
    """
    import ast

    source = (LOWLEVEL_DIR / "multisim_chess_fast.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "DEFAULT_JOINT_OFFSETS_DEG" for t in node.targets):
            return np.array([*ast.literal_eval(node.value), 0.0], dtype=float)
    raise ValueError("No DEFAULT_JOINT_OFFSETS_DEG in multisim_chess_fast.py")


def slow_gripper_close(
    commands: list[np.ndarray],
    sent_waypoints: list[np.ndarray],
    closeidx: int,
    *,
    steps_per_waypoint: int,
    interpolation: str,
    count: int,
) -> tuple[list[np.ndarray], int, int]:
    """Replace the grasp waypoint's commands with count commands that hold the arm at the grasp
    and ramp the gripper from open to closed.

    The interpolated close jumps the gripper straight to closed, and the smoothing between
    waypoints lets the arm drift a few mm while it does. Returns the new commands, the index
    of the first close command and how many commands were added.
    """
    end = waypoint_command_index(closeidx, steps_per_waypoint=steps_per_waypoint, interpolation=interpolation)
    if interpolation == "linear":
        end += 1
    first = end - steps_per_waypoint
    count = max(count, steps_per_waypoint)
    grasp = np.asarray(sent_waypoints[closeidx], dtype=float)
    opened = float(sent_waypoints[closeidx - 1][5])
    ramp = []
    for alpha in np.linspace(0.0, 1.0, count + 1)[1:]:
        command = grasp.copy()
        command[5] = (1.0 - alpha) * opened + alpha * grasp[5]
        ramp.append(command)
    return commands[:first] + ramp + commands[end:], first, count - steps_per_waypoint


def slow_return(commands: list[np.ndarray], first: int, count: int) -> tuple[list[np.ndarray], int]:
    """Stretch commands[first:] (the move back to rest) to count commands, same path, slower.

    The trajectory swings back to rest in ~3 waypoints (~0.6 s, the pan ~90 deg in one), fast
    enough to rock the base on its clamps. Returns the new commands and how many were added.
    """
    tail = np.asarray(commands[first:], dtype=float)
    if count <= len(tail):
        return commands, 0
    old = np.linspace(0.0, 1.0, len(tail))
    new = np.linspace(0.0, 1.0, count)
    stretched = [np.array([np.interp(t, old, tail[:, j]) for j in range(tail.shape[1])]) for t in new]
    return commands[:first] + stretched, count - len(tail)


class Reading(NamedTuple):
    """One command of the main loop: the joints read after it and when each step happened.

    Times are time.perf_counter() seconds. Per loop: t_send -> send_action -> t_sent ->
    (pickup pause, progress print) -> sleep until this command's read time -> t_read_start ->
    read -> t_read_end. Command k's read time is the first t_send + (k + 1) * command_delay,
    plus any pickup pause; when the loop is behind it, there is no sleep. t_read_start is the
    start of the read attempt that succeeded; read_retries counts the failed attempts before it.
    """

    index: int
    joints_deg: np.ndarray
    t_send: float
    t_sent: float
    t_read_start: float
    t_read_end: float
    read_retries: int


def build_command_sequence(
    waypoints_deg: list[np.ndarray],
    *,
    closeidx: int,
    return_waypoint: int | None,
    steps_per_waypoint: int,
    interpolation: str,
    gripper_close_s: float,
    return_s: float,
    command_delay: float,
) -> tuple[list[np.ndarray], int | None, int, int]:
    """Interpolate waypoints into one entry per command, with the slow gripper close and return.

    Returns (sequence, index of the first gripper-close command or None, commands added by the
    close, commands added by the return). main() calls it on the sent waypoints (commands) and
    on the planned URDF waypoints (targets); the counts depend only on the settings, so the two
    come out the same length with entry i of each belonging to the same command.
    """
    sequence = interpolate_joint_waypoints(
        waypoints_deg,
        steps_per_waypoint=steps_per_waypoint,
        interpolation=interpolation,
    )
    close_first = None
    added = 0
    if gripper_close_s > 0.0:
        sequence, close_first, added = slow_gripper_close(
            sequence,
            waypoints_deg,
            closeidx,
            steps_per_waypoint=steps_per_waypoint,
            interpolation=interpolation,
            count=round(gripper_close_s / command_delay),
        )
    return_added = 0
    if return_s > 0.0:
        if return_waypoint is None:
            raise ValueError("return_s > 0 needs return_waypoint")
        return_first = waypoint_command_index(
            return_waypoint,
            steps_per_waypoint=steps_per_waypoint,
            interpolation=interpolation,
        ) + added
        sequence, return_added = slow_return(sequence, return_first, round(return_s / command_delay))
    return sequence, close_first, added, return_added


TIMELINE_SCHEMA = "so101_timeline_v1"
SIM_TIMELINE_FILENAME = "timeline_sim.json"
HARDWARE_TIMELINE_FILENAME = "timeline_hardware.json"
# Each file saves the entries for the fields it has, so it is self-describing.
TIMELINE_FIELDS = {
    "index": "command index into the run's full command list; row r of every list belongs to command index[r]",
    "target_deg": (
        "desired joint angles, URDF degrees: the planned trajectory interpolated per command, "
        "before any --baseline/--compensate shift. Same in both timelines"
    ),
    "command_deg": "joint command sent to the servos, motor degrees. Same in both timelines",
    "reading_deg": "joint angles read after the command, motor degrees",
    "reading_urdf_deg": "reading_deg + urdf_offsets_deg, URDF degrees, comparable with target_deg",
    "predicted_error_deg": (
        "command_deg - the friction model's predicted resting reading, motor degrees. Static model: "
        "holds only where the arm comes to rest"
    ),
    "t_send_s": "when the command was sent (hardware: just before send_action)",
    "t_sent_s": "when the send finished (hardware: just after send_action; ideal: = t_send_s)",
    "t_read_start_s": "start of the joint read (hardware: of the attempt that succeeded; ideal: = t_read_end_s)",
    "t_read_end_s": (
        "time of the joint reading (ideal: t_send_s + command_delay_s, when the next command is "
        "sent; the ideal joints are then at target_deg)"
    ),
    "read_retries": "failed read attempts before the reading",
}
# Measured on the real arm only; the sim timeline has no readings, so it leaves these out.
HARDWARE_ONLY_FIELDS = ("reading_deg", "reading_urdf_deg", "predicted_error_deg", "read_retries")


def predicted_errors_deg(commands_deg: list[np.ndarray], compensator: Any) -> list[np.ndarray]:
    """Per consecutive command: command - the friction model's predicted resting reading (motor deg).

    The sign is the friction fit's (target - reading, the command being its target). A joint's
    direction is the way it last moved by at least MIN_MOVE_DEG, as in compensate_waypoints. The
    model is static, so the prediction holds where the arm comes to rest, not while it is moving.
    """
    from friction.joint_compensation import approach_dirs

    errors = []
    dirs = np.zeros(len(MOTOR_KEYS))
    previous = None
    for command in commands_deg:
        command = np.asarray(command, dtype=float)
        if previous is not None:
            step = approach_dirs(command, previous)
            dirs = np.where(step != 0.0, step, dirs)
            previous = np.where(step != 0.0, command, previous)
        else:
            previous = command
        errors.append(command - compensator.predict_rest_motor_deg(command, move_dirs=dirs))
    return errors


def ideal_times_s(
    indices: list[int],
    *,
    start_index: int,
    command_delay: float,
    pause_command_index: int | None,
    pause_s: float,
) -> tuple[list[float], list[float]]:
    """(send times, read times) of the ideal timeline, in seconds since command start_index was sent.

    Command k = index - start_index is sent at k * command_delay and read command_delay later, at
    the moment the next command is sent. A pickup pause sits where execute_commands puts it:
    after the pause command is sent and before its read, so it delays that read and every later
    command.
    """
    sends, reads = [], []
    for index in indices:
        t_send = (index - start_index) * command_delay
        if pause_command_index is not None and index > pause_command_index:
            t_send += pause_s
        t_read = t_send + command_delay
        if index == pause_command_index:
            t_read += pause_s
        sends.append(t_send)
        reads.append(t_read)
    return sends, reads


def timeline_payload(
    *,
    timeline: str,
    description: str,
    clock: str,
    run_info: dict[str, Any],
    rows: dict[str, list[Any]],
) -> dict[str, Any]:
    """One timeline file's contents: shared metadata, then one list per field, in TIMELINE_FIELDS order.

    The hardware timeline has every field; the sim timeline has all but HARDWARE_ONLY_FIELDS.
    """
    expected = [
        name for name in TIMELINE_FIELDS if timeline == "hardware" or name not in HARDWARE_ONLY_FIELDS
    ]
    if sorted(rows) != sorted(expected):
        raise ValueError(f"{timeline} timeline fields {sorted(rows)} != expected {sorted(expected)}")
    lengths = {name: len(values) for name, values in rows.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{timeline} timeline rows differ in length: {lengths}")
    return {
        "schema": TIMELINE_SCHEMA,
        "timeline": timeline,
        "description": description,
        "clock": clock,
        "joints": [key.removesuffix(".pos") for key in MOTOR_KEYS],
        **run_info,
        "fields": {name: TIMELINE_FIELDS[name] for name in expected},
        **{name: rows[name] for name in expected},
    }


def write_timelines(
    output_dir: Path,
    *,
    readings: list[Reading],
    commands: list[np.ndarray],
    targets: list[np.ndarray],
    compensator: Any,
    command_delay: float,
    start_index: int,
    pause_command_index: int | None,
    pickup_pause_s: float,
    run_info: dict[str, Any],
) -> tuple[Path, Path]:
    """Write the run as two timelines, one row per logged command.

    timeline_hardware.json: what the real arm did, timed with time.perf_counter() in seconds
    since the first logged t_send.
    timeline_sim.json: the ideal run of the same commands (perfect joints, zero latency): each
    command on schedule, its joints at target_deg. It has the same layout minus the measured
    fields (HARDWARE_ONLY_FIELDS), since nothing is read in it.
    Both cover the same commands, so a run cut short gives two equally short files.
    """
    indices = [reading.index for reading in readings]
    offsets = np.asarray(compensator.offsets_deg, dtype=float)
    row_targets = [np.asarray(targets[index], dtype=float) for index in indices]
    row_commands = [np.asarray(commands[index], dtype=float) for index in indices]
    # No pause, a timed pause, or one that waits for Enter (no scheduled length, so 0 here).
    waits_for_enter = pause_command_index is not None and pickup_pause_s <= 0.0
    pause_s = pickup_pause_s if pause_command_index is not None else 0.0
    shared_info = {
        **run_info,
        "command_delay_s": command_delay,
        "start_index": start_index,
        "command_count": len(commands),
        "pause_command_index": pause_command_index,
        "pause_s": pause_s,
        "pause_waits_for_enter": waits_for_enter,
        "urdf_offsets_deg": offsets,
        "urdf_offsets_source": str(compensator.fit_path),
    }

    t0 = readings[0].t_send
    measured = [np.asarray(reading.joints_deg, dtype=float) for reading in readings]
    hardware = timeline_payload(
        timeline="hardware",
        description="Measured on the real SO101.",
        clock="time.perf_counter, seconds since the first logged t_send",
        run_info=shared_info,
        rows={
            "index": indices,
            "target_deg": row_targets,
            "command_deg": row_commands,
            "reading_deg": measured,
            "reading_urdf_deg": [reading + offsets for reading in measured],
            "predicted_error_deg": predicted_errors_deg(row_commands, compensator),
            "t_send_s": [reading.t_send - t0 for reading in readings],
            "t_sent_s": [reading.t_sent - t0 for reading in readings],
            "t_read_start_s": [reading.t_read_start - t0 for reading in readings],
            "t_read_end_s": [reading.t_read_end - t0 for reading in readings],
            "read_retries": [reading.read_retries for reading in readings],
        },
    )
    sends, reads = ideal_times_s(
        indices,
        start_index=start_index,
        command_delay=command_delay,
        pause_command_index=pause_command_index,
        pause_s=pause_s,
    )
    sim = timeline_payload(
        timeline="sim_ideal",
        description=(
            "Ideal execution of the same commands: perfect joints, zero latency. Command "
            "k = index - start_index is sent at k * command_delay_s (plus pause_s after the pause "
            "command) and its joints would be read command_delay_s later, as the next command is "
            "sent, when they are at target_deg. Nothing is measured, so there are no reading fields."
        ),
        clock="ideal schedule, seconds since command start_index was sent",
        run_info=shared_info,
        rows={
            "index": indices,
            "target_deg": row_targets,
            "command_deg": row_commands,
            "t_send_s": sends,
            "t_sent_s": sends,
            "t_read_start_s": reads,
            "t_read_end_s": reads,
        },
    )
    paths = []
    for filename, payload in ((HARDWARE_TIMELINE_FILENAME, hardware), (SIM_TIMELINE_FILENAME, sim)):
        path = output_dir / filename
        path.write_text(json.dumps(json_safe(payload)) + "\n", encoding="utf-8")
        print(f"{payload['timeline']} timeline: {path}")
        paths.append(path)
    return paths[0], paths[1]


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
        "board_origin": args.board_origin,
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


READ_RETRIES = 3  # extra attempts after a failed position read
READ_RETRY_DELAY_S = 0.02


def read_observation(
    follower,
    *,
    retries: int = READ_RETRIES,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """follower.get_observation(), retried when a read gets no reply.

    The arm reaches WSL over usbipd, sharing the link with the tripod webcam, and lerobot gives a
    sync read one try with a ~50 ms timeout. One stalled reply in the ~3000 reads of a run used to
    end the run, so a ConnectionError is retried a few times before it is raised.

    If timing is given, it gets the time.perf_counter() start and end of the attempt that
    succeeded ("read_start", "read_end") and how many attempts failed before it ("retries").
    """
    for attempt in range(retries + 1):
        read_start = time.perf_counter()
        try:
            observation = follower.get_observation()
            if timing is not None:
                timing.update(read_start=read_start, read_end=time.perf_counter(), retries=attempt)
            return observation
        except ConnectionError as error:
            if attempt == retries:
                raise
            print(f"position read failed ({error}); retry {attempt + 1}/{retries}", flush=True)
            # Drop any late reply still in the input buffer so it is not read as the next answer.
            port_handler = follower.bus.port_handler
            port_handler.is_using = False
            serial_port = getattr(port_handler, "ser", None)
            if serial_port is not None:
                serial_port.reset_input_buffer()
            time.sleep(READ_RETRY_DELAY_S)
    raise AssertionError("unreachable")


def release_arm(follower, *, finished: bool) -> None:
    """Close the port; cut torque only once the user has had the chance to support the arm.

    After a finished run the arm is at rest, so torque goes off as before. After an error or
    Ctrl-C the arm can be anywhere, and cutting torque drops it, so it keeps holding its last
    position until the user presses Enter.
    """
    if finished:
        release = True
    else:
        print(
            "\nRun stopped early. The arm is HOLDING its last position (torque still on).\n"
            "Support the arm, then press Enter to release torque, "
            "or Ctrl-C to leave it holding.",
            flush=True,
        )
        try:
            input()
            release = True
        except (EOFError, KeyboardInterrupt):
            release = False
            print("\nLeaving torque on. Release the arm later, e.g. by reconnecting and disabling torque.")
    try:
        if release:
            follower.bus.port_handler.is_using = False
            follower.bus.disable_torque(num_retry=5)
    except Exception as error:  # noqa: BLE001 - still close the port below
        print(f"could not disable torque: {error}")
    finally:
        follower.disconnect()  # disable_torque_on_disconnect=False: this only closes the port


def execute_commands(
    commands_deg: list[np.ndarray],
    *,
    port: str,
    robot_id: str,
    command_delay: float,
    max_start_delta_deg: float,
    allow_start_bridge: bool,
    start_bridge_steps: int,
    confirm_start_bridge: bool = True,
    pause_command_index: int | None,
    pickup_pause_s: float,
    start_index: int = 0,
    readings: list[Reading] | None = None,
) -> None:
    """Send the commands; if readings is given, append a Reading per command (joints read after it)."""
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    # Torque is cut by release_arm below, not by disconnect(), so an error does not drop the arm.
    follower = SO101Follower(
        SO101FollowerConfig(port=port, id=robot_id, disable_torque_on_disconnect=False)
    )
    follower.connect()
    finished = False
    try:
        observation = read_observation(follower)
        current = observation_joints_deg(observation)
        first_delta = float(np.max(np.abs(commands_deg[start_index] - current)))
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
            print("target_start_deg:", np.round(commands_deg[start_index], 3).tolist())
            print("delta_deg:", np.round(commands_deg[start_index] - current, 3).tolist())
            print(
                f"This will move slowly to the trajectory start over "
                f"{start_bridge_steps} steps."
            )
            if confirm_start_bridge:
                confirmation = input("Move to trajectory start? [yes/no]: ").strip().lower()
                if confirmation not in {"yes", "y"}:
                    raise RuntimeError("Start bridge cancelled by user.")

            for index, alpha in enumerate(
                np.linspace(0.0, 1.0, start_bridge_steps + 1)[1:],
                start=1,
            ):
                joints = (1.0 - alpha) * current + alpha * commands_deg[start_index]
                action = observation.copy()
                action.update(action_from_joints(joints))
                follower.send_action(action)
                if index % 25 == 0 or index == start_bridge_steps:
                    print(f"start bridge {index}/{start_bridge_steps}")
                time.sleep(command_delay)
                observation = read_observation(follower)

        # Fixed schedule: command k is read at the first send + (k + 1) * command_delay (plus any
        # pickup pause), so the write and read come out of each step's sleep instead of adding to
        # it. After a slow write or read the loop skips the sleep until it is back on schedule.
        deadline: float | None = None
        #what deadline is outline a grid of the where the commands should be
        for index, joints in enumerate(commands_deg[start_index:], start=start_index + 1):
            action = observation.copy()
            action.update(action_from_joints(joints))
            t_send = time.perf_counter()
            follower.send_action(action)
            t_sent = time.perf_counter()
            if deadline is None:
                deadline = t_send
            deadline += command_delay
            if pause_command_index is not None and index - 1 == pause_command_index:
                print(f"Paused at pickup command {index}/{len(commands_deg)}.")
                pause_start = time.perf_counter()
                if pickup_pause_s > 0.0:
                    time.sleep(pickup_pause_s)
                else:
                    input("Press Enter to continue trajectory...")
                deadline += time.perf_counter() - pause_start  # the schedule waits with the arm
            if index % 50 == 0:
                print(f"sent {index}/{len(commands_deg)} commands")
            left = deadline - time.perf_counter()
            if left > 0.0:  # behind schedule: no sleep, so the loop catches up
                time.sleep(left)
            timing: dict[str, Any] = {}
            observation = read_observation(follower, timing=timing)
            if readings is not None:
                readings.append(
                    Reading(
                        index=index - 1,
                        joints_deg=observation_joints_deg(observation),
                        t_send=t_send,
                        t_sent=t_sent,
                        t_read_start=timing["read_start"],
                        t_read_end=timing["read_end"],
                        read_retries=timing["retries"],
                    )
                )
        finished = True
    finally:
        release_arm(follower, finished=finished)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate a saved continuous XY lookup and optionally send it to a real SO101."
    )
    parser.add_argument("--lookup-json", type=Path, default=DEFAULT_LOOKUP_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument(
        "--board-origin-x",
        type=float,
        default=DEFAULT_REAL_BOARD_ORIGIN[0],
        help=(
            "Board centre x (m, robot base frame). The lookup's pick and place points "
            "move with the board, keeping the same squares."
        ),
    )
    parser.add_argument("--board-origin-y", type=float, default=DEFAULT_REAL_BOARD_ORIGIN[1])
    parser.add_argument(
        "--board-origin-z",
        type=float,
        default=DEFAULT_REAL_BOARD_ORIGIN[2],
        help=(
            "Board collision-box centre z used for real trajectory regeneration. "
            "Default 0.025 gives a 0.030 m board top with the 5 mm sim half-height."
        ),
    )
    parser.add_argument(
        "--zero-grasp-offset",
        action="store_true",
        help=(
            "Ignore the lookup's grasp/place offsets and regenerate with zeros. "
            "The offsets are tool-frame corrections for gripper geometry, so zeroing "
            "them aims the wrist at the square centre instead of the fingers; the "
            "lower-place path bias is dropped too since it was solved alongside them."
        ),
    )
    parser.add_argument(
        "--compensate",
        action="store_true",
        help=(
            "Send (q - offset) + f per waypoint so the real arm reaches the trajectory's "
            "URDF poses, with calibration offsets and the M3 friction/sag error f from "
            "--compensation-fit (see friction/joint_compensation.py)."
        ),
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "Send q - offset per waypoint with the hand-measured calibration offsets "
            "(DEFAULT_JOINT_OFFSETS_DEG in multisim_chess_fast.py) and no friction/sag model: "
            "the baseline to compare --compensate against. Not with --compensate."
        ),
    )
    parser.add_argument(
        "--compensation-fit",
        type=Path,
        default=LOWLEVEL_DIR / "real_world_runs" / "friction" / "friction_sag_fit.json",
    )
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
        "--yes",
        action="store_true",
        help="With --allow-start-bridge: move to the trajectory start without asking first.",
    )
    parser.add_argument(
        "--start-waypoint",
        type=int,
        default=0,
        help=(
            "Start the trajectory at this waypoint; the start bridge goes straight there. "
            "Lookup trajectories open low beside the robot (tip ~12 mm up) and climb from "
            "waypoint 1, so 3 skips that opening. Must come before the grasp."
        ),
    )
    parser.add_argument(
        "--pause-at-pickup",
        action="store_true",
        help=(
            "Pause when the trajectory reaches the pickup/close waypoint. "
            "By default this waits for Enter."
        ),
    )
    parser.add_argument(
        "--gripper-close-s",
        type=float,
        default=0.0,
        help=(
            "Close the gripper over this many seconds, with the arm held still at the grasp. "
            "0 (default) keeps the trajectory's close: the gripper is commanded closed at once "
            "and the arm lifts one waypoint (~0.2 s) later."
        ),
    )
    parser.add_argument(
        "--return-s",
        type=float,
        default=0.0,
        help=(
            "Take this many seconds to swing back to rest after the place (same path, slower). "
            "0 (default) keeps the trajectory's ~0.6 s."
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
    if args.baseline and args.compensate:
        raise ValueError("--baseline and --compensate are alternatives: pick one")
    args.board_origin = (
        float(args.board_origin_x),
        float(args.board_origin_y),
        float(args.board_origin_z),
    )
    lookup_path = args.lookup_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
    if lookup.get("schema") != "continuous_xy_lookup_v1":
        raise ValueError(f"Unsupported lookup schema in {lookup_path}")

    waypoints, closeidx, metadata = build_trajectory_from_lookup(
        lookup,
        board_origin=args.board_origin,
        zero_grasp_offset=args.zero_grasp_offset,
    )
    # The waypoints are URDF angles; what gets interpolated and sent is
    # sent_waypoints, the same with --compensate turned into motor commands.
    sent_waypoints = waypoints
    if args.compensate:
        from friction.joint_compensation import JointCompensator

        compensator = JointCompensator(args.compensation_fit)
        sent_waypoints = compensate_waypoints(waypoints, compensator)
        metadata["compensation_fit"] = str(args.compensation_fit)
        metadata["compensated_waypoints_deg"] = sent_waypoints
    metadata["compensated"] = bool(args.compensate)
    if args.baseline:
        baseline = baseline_offsets_deg()
        sent_waypoints = [np.asarray(waypoint, dtype=float) - baseline for waypoint in waypoints]
        metadata["baseline_offsets_deg"] = baseline
    metadata["baseline"] = bool(args.baseline)
    # Commands (motor degrees, what is sent) and targets (URDF degrees, the plan before any
    # baseline/compensation shift) go through the same interpolation, slow gripper close and
    # slow return, so targets[i] is the angle commands[i] is meant to reach.
    sequence_kwargs = dict(
        closeidx=closeidx,
        # The last retreat waypoint, so the whole swing back to rest is slowed.
        return_waypoint=(
            metadata["regenerated_trajectory_metrics"]["segments"]["return_home"]["start"] - 1
            if args.return_s > 0.0
            else None
        ),
        steps_per_waypoint=args.steps_per_waypoint,
        interpolation=args.interpolation,
        gripper_close_s=args.gripper_close_s,
        return_s=args.return_s,
        command_delay=args.command_delay,
    )
    commands, close_first, added, return_added = build_command_sequence(sent_waypoints, **sequence_kwargs)
    targets, *target_counts = build_command_sequence(waypoints, **sequence_kwargs)
    if len(targets) != len(commands) or tuple(target_counts) != (close_first, added, return_added):
        raise RuntimeError("targets and commands came out misaligned; they must have the same length")
    metadata["gripper_close_s"] = args.gripper_close_s
    metadata["return_s"] = args.return_s
    pause_command_index = None
    if args.pause_at_pickup:
        pause_command_index = waypoint_command_index(
            closeidx,
            steps_per_waypoint=args.steps_per_waypoint,
            interpolation=args.interpolation,
        )
        if added and pause_command_index >= close_first:
            pause_command_index += added
        pause_command_index = min(pause_command_index, len(commands) - 1)
    start_index = 0
    if args.start_waypoint:
        if not 0 < args.start_waypoint < closeidx:
            raise ValueError(f"--start-waypoint must be between 0 and the grasp waypoint ({closeidx})")
        start_index = waypoint_command_index(
            args.start_waypoint,
            steps_per_waypoint=args.steps_per_waypoint,
            interpolation=args.interpolation,
        )
    metadata["start_waypoint"] = args.start_waypoint
    metadata["start_command_index"] = start_index
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
    print(f"board_origin: {args.board_origin}")
    shift = metadata["board_shift_xy"]
    if any(abs(v) > 1e-9 for v in shift):
        print(
            f"  board moved ({shift[0] * 1000:+.1f}, {shift[1] * 1000:+.1f}) mm from the lookup's "
            f"({LOOKUP_BOARD_ORIGIN_XY[0]}, {LOOKUP_BOARD_ORIGIN_XY[1]}): pick at "
            f"({metadata['start']['x'] * 1000:.1f}, {metadata['start']['y'] * 1000:.1f}) mm, place at "
            f"({metadata['target']['x'] * 1000:.1f}, {metadata['target']['y'] * 1000:.1f}) mm"
        )
    if args.zero_grasp_offset:
        print(
            "zero-grasp-offset: ON (lookup grasp/place offsets and lower-place bias ignored)"
        )
    print(f"waypoints: {len(waypoints)} | commands: {len(commands)} | closeidx: {closeidx}")
    if args.compensate:
        grasp_correction = np.asarray(sent_waypoints[closeidx]) - np.asarray(waypoints[closeidx])
        largest = np.max(np.abs(np.asarray(sent_waypoints) - np.asarray(waypoints)), axis=0)
        print(f"compensation: {args.compensation_fit}")
        print(f"  command - target at grasp (deg): {np.round(grasp_correction, 2).tolist()}")
        print(f"  largest |command - target| (deg): {np.round(largest, 2).tolist()}")
    if args.baseline:
        print(f"baseline: hand-measured offsets only, command = target - {np.round(baseline, 2).tolist()} deg")
    if start_index:
        from testkinematics import kinematics

        tip_z = kinematics.forward_kinematics(np.asarray(waypoints[args.start_waypoint]))[2, 3]
        print(
            f"start: waypoint {args.start_waypoint} (command {start_index}), planned tip "
            f"{tip_z * 1000:.0f} mm up; the start bridge goes straight there"
        )
    if args.gripper_close_s > 0.0:
        print(f"gripper close: {args.gripper_close_s:.2f} s, arm held at the grasp (+{added} commands)")
    if args.return_s > 0.0:
        print(f"return to rest: {args.return_s:.2f} s (+{return_added} commands)")
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

    # Loaded even without --compensate, for the predicted error and URDF offsets in the timelines.
    if not args.compensate:
        from friction.joint_compensation import JointCompensator

        compensator = JointCompensator(args.compensation_fit)
    readings: list[Reading] = []
    try:
        execute_commands(
            commands,
            port=args.port,
            robot_id=args.robot_id,
            command_delay=args.command_delay,
            max_start_delta_deg=args.max_start_delta_deg,
            allow_start_bridge=args.allow_start_bridge,
            confirm_start_bridge=not args.yes,
            start_bridge_steps=args.start_bridge_steps,
            pause_command_index=pause_command_index,
            pickup_pause_s=args.pickup_pause_s,
            start_index=start_index,
            readings=readings,
        )
    finally:
        # Also on Ctrl-C or an error, so a cut-short run keeps what it read.
        if readings:
            write_timelines(
                output_dir,
                readings=readings,
                commands=commands,
                targets=targets,
                compensator=compensator,
                command_delay=args.command_delay,
                start_index=start_index,
                pause_command_index=pause_command_index,
                pickup_pause_s=args.pickup_pause_s,
                run_info={
                    "lookup_json": str(lookup_path),
                    "preview_json": "real_so101_trajectory_preview.json",
                    "compensated": bool(args.compensate),
                    "baseline": bool(args.baseline),
                    "steps_per_waypoint": args.steps_per_waypoint,
                    "interpolation": args.interpolation,
                    "gripper_close_s": args.gripper_close_s,
                    "return_s": args.return_s,
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
