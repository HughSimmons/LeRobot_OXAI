#!/usr/bin/env python3
"""A fixed viewing pose for the wrist camera: save it (e), check a path to it, drive there (w).

Why: c's wrist pose only holds at the joints c was taken at. Forward kinematics re-poses other
views with ~5 mm of error (worst 12-15 mm, fk_transfer_accuracy.py), and on 2026-11-20 a pick
seen from a different angle landed far off. Taking every wrist view from the pose c was taken
at keeps the FK change, and so its error, near zero.

The move follows probe_xy_pose.py's approach: unfold into the view pose's shape at the current
heading, then swing round to its heading, so the swing happens at view height rather than low
over the board. If that order isn't clear, the reverse (swing, then unfold) is tried. Before
anything moves, every step of the path is checked with forward kinematics: the elbow, wrist,
gripper, fixed-jaw tip, moving jaw and wrist camera must stay BOARD_CLEARANCE_MM above the board
surface over the board, and TABLE_CLEARANCE_MM above the table beside it. If no order passes,
nothing moves and the reason is printed; otherwise it moves straight away (no prompt). A released
arm gets torque turned on first (goal = present position first, so it holds where it is); if it shifted since planning,
the path is planned and checked again from where it is.

The gripper keeps its current opening throughout, and wrist roll may turn at most
MAX_ROLL_CHANGE_DEG, so the camera cable can't wind round.

With compensate=True (always, from run_click_to_3d.py's w), the path ends at the friction/sag
model's command for the saved reading rather than at the reading itself (friction/joint_compensation.py):
the saved reading is where the arm should come to rest, so the command overshoots by the
predicted stopping error in the direction each joint approaches from. Only the final pose is
compensated (the model is static). The path to that command is what gets clearance-checked, and
the landing error is still measured against the saved reading.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import numpy as np

from click3d_shared import (
    BOARD_CENTRE_BASE_MM,
    JOINT_NAMES,
    SESSIONS_DIR,
    SQUARE_MM,
    _ensure_lowlevel_on_path,
    _follower,
    arm_held,
    joint_offsets_deg,
    read_joints_robot,
    set_arm_hold,
)
from move_to_click import MoveResult

VIEW_POSE_PATH = SESSIONS_DIR / "view_pose.json"
VIEW_MOVES_LOG = SESSIONS_DIR / "view_moves.jsonl"  # each move's start, target and landing

# Parts of the arm checked along the path (URDF frames; origins, so the margins below allow
# for the links' own size).
CHECK_FRAMES = (
    "lower_arm_link",  # elbow
    "wrist_link",
    "gripper_link",
    "gripper_frame_link",  # fixed-jaw tip
    "moving_jaw_so101_v1_link",
    "wrist_camera_link",
)
# Over the board every checked part stays this far above the surface: clears standing pieces
# (a standard king is ~95 mm; the sim's rook is 40 mm; the lookup trajectories travel at
# 90-130 mm and probe_xy_pose's via is 120 mm).
BOARD_CLEARANCE_MM = 100.0
BOARD_BORDER_MM = 30.0  # the board's wooden frame beyond the 8 x 8 squares
TABLE_CLEARANCE_MM = 15.0  # beside the board, above the table (base z = 0)
MAX_SPEED_DEG_S = 30.0  # fastest any joint moves (probe_xy_pose's default)
COMMAND_DELAY_S = 1.0 / 240.0
MAX_ROLL_CHANGE_DEG = 180.0
START_TOLERANCE_DEG = 1.5  # the held arm may have drifted this much since planning
SETTLE_S = 1.0


class ViewMoveRefused(RuntimeError):
    """The view-pose move isn't safe or possible; nothing was sent to the arm."""


@dataclass
class Clearance:
    margin_mm: float  # lowest (height - required height) over the checked parts; < 0 is unsafe
    frame: str
    point_mm: np.ndarray
    over_board: bool

    def describe(self) -> str:
        where = "over the board" if self.over_board else "beside the board"
        need = (
            f"{BOARD_CLEARANCE_MM:g} mm above the surface"
            if self.over_board
            else f"{TABLE_CLEARANCE_MM:g} mm above the table"
        )
        return (
            f"{self.frame} at {np.round(self.point_mm, 0).tolist()} mm, {where}: "
            f"{self.margin_mm:+.0f} mm vs the {need} needed"
        )


@dataclass
class ViewPlan:
    path: list[np.ndarray]
    order: str
    lowest: Clearance  # the tightest point along the whole path

    @property
    def seconds(self) -> float:
        return len(self.path) * COMMAND_DELAY_S


# --- saving --------------------------------------------------------------------------------


def load_view_pose(path: Path | None = None) -> dict | None:
    path = VIEW_POSE_PATH if path is None else path
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_view_pose(joints_deg: list[float], urdf_path: Path, path: Path | None = None) -> tuple[dict, Clearance]:
    """Save these motor readings as the view pose; returns the record and the pose's clearance."""
    path = VIEW_POSE_PATH if path is None else path
    lowest = clearance(urdf_path, joints_deg)
    record = {
        "joint_reading_deg": [round(float(v), 3) for v in joints_deg],
        "joint_offsets_deg": [round(float(v), 3) for v in joint_offsets_deg()],
        "lowest_part": lowest.frame,
        "lowest_margin_mm": round(lowest.margin_mm, 1),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "note": "motor readings; w drives here. Press c at this pose, so later views match c.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record, lowest


# --- geometry ------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _kinematics(urdf_path: str, frame: str):
    _ensure_lowlevel_on_path()
    from paths import import_robot_kinematics

    return import_robot_kinematics()(urdf_path=urdf_path, target_frame_name=frame, joint_names=list(JOINT_NAMES))


def part_points_mm(urdf_path: Path, joints_deg) -> dict[str, np.ndarray]:
    """Base-frame position (mm) of each checked part, for motor readings joints_deg."""
    urdf_angles = np.asarray(joints_deg, dtype=np.float64) + joint_offsets_deg()
    return {
        frame: np.asarray(_kinematics(str(urdf_path), frame).forward_kinematics(urdf_angles))[:3, 3] * 1000.0
        for frame in CHECK_FRAMES
    }


def over_board(xy_mm: np.ndarray) -> bool:
    half = 4.0 * SQUARE_MM + BOARD_BORDER_MM
    return (
        abs(float(xy_mm[0]) - BOARD_CENTRE_BASE_MM[0]) <= half
        and abs(float(xy_mm[1]) - BOARD_CENTRE_BASE_MM[1]) <= half
    )


def clearance(urdf_path: Path, joints_deg) -> Clearance:
    """The checked part closest to its limit at this pose."""
    worst: Clearance | None = None
    for frame, point in part_points_mm(urdf_path, joints_deg).items():
        above = over_board(point[:2])
        floor = BOARD_CENTRE_BASE_MM[2] + BOARD_CLEARANCE_MM if above else TABLE_CLEARANCE_MM
        candidate = Clearance(float(point[2] - floor), frame, point, above)
        if worst is None or candidate.margin_mm < worst.margin_mm:
            worst = candidate
    return worst


def paced_steps(start_deg: np.ndarray, end_deg: np.ndarray, max_step_deg: float) -> list[np.ndarray]:
    """Evenly spaced poses after start up to and including end; no joint moves > max_step_deg."""
    start, end = np.asarray(start_deg, dtype=np.float64), np.asarray(end_deg, dtype=np.float64)
    count = max(1, int(np.ceil(float(np.max(np.abs(end - start))) / max_step_deg)))
    return [start + (end - start) * (i / count) for i in range(1, count + 1)]


def plan_view_move(
    current_deg,
    view_deg,
    urdf_path: Path,
    max_step_deg: float = MAX_SPEED_DEG_S * COMMAND_DELAY_S,
    check_every: int = 4,
) -> ViewPlan:
    """A checked joint path from the current pose to the view pose, or ViewMoveRefused."""
    current = np.asarray(current_deg, dtype=np.float64)
    target = np.asarray(view_deg, dtype=np.float64).copy()
    target[5] = current[5]  # keep the gripper as it is
    roll_change = abs(float(target[4] - current[4]))
    if roll_change > MAX_ROLL_CHANGE_DEG:
        raise ViewMoveRefused(
            f"wrist roll would turn {roll_change:.0f} deg (limit {MAX_ROLL_CHANGE_DEG:g}): "
            "turn it nearer the view pose's roll by hand first"
        )
    start = clearance(urdf_path, current)
    if start.margin_mm < 0:
        raise ViewMoveRefused(
            f"the arm starts too low: {start.describe()}. Lift it clear (or let a / k return it to rest) first"
        )
    end = clearance(urdf_path, target)
    if end.margin_mm < 0:
        raise ViewMoveRefused(f"the view pose itself is too low: {end.describe()}. Save a higher one with e")

    unfolded = target.copy()
    unfolded[0] = current[0]  # the view pose's shape, at the current heading
    swung = current.copy()
    swung[0] = target[0]  # the current shape, at the view pose's heading
    orders = (
        ("unfold, then swing", [unfolded, target]),
        ("swing, then unfold", [swung, target]),
    )
    failures = []
    for name, waypoints in orders:
        path, previous = [], current
        for waypoint in waypoints:
            path += paced_steps(previous, waypoint, max_step_deg)
            previous = waypoint
        checked = path[check_every - 1 :: check_every] + [path[-1]]
        lowest = min((clearance(urdf_path, step) for step in checked), key=lambda c: c.margin_mm)
        if lowest.margin_mm >= 0:
            return ViewPlan(path, name, lowest)
        failures.append(f"{name}: {lowest.describe()}")
    raise ViewMoveRefused("no clear path to the view pose; " + "; ".join(failures))


# --- compensation --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _compensator():
    _ensure_lowlevel_on_path()
    from friction.joint_compensation import JointCompensator

    return JointCompensator()


def view_command(current_deg, view_deg, compensate: bool) -> tuple[np.ndarray, list]:
    """The motor command that should leave the arm resting at the saved reading view_deg.

    Without compensation that is view_deg itself. With it, the friction/sag model's command for
    URDF pose view_deg + offset, approached from current_deg (each joint moves monotonically
    along the planned path, so its direction is the sign of view - current).
    """
    view = np.asarray(view_deg, dtype=np.float64)
    if not compensate:
        return view.copy(), []
    from friction.joint_compensation import approach_dirs

    model = _compensator()
    dirs = approach_dirs(view, np.asarray(current_deg, dtype=np.float64))
    return model.compensate(model.resting_urdf_deg(view), move_dirs=dirs)


def print_compensation(view_deg, command: np.ndarray, details: list) -> None:
    if not details:
        print("  not compensated: the saved reading is sent as the command")
        return
    print("  compensated (friction/sag model): command = reading + predicted stopping error")
    for detail, reading, cmd in zip(details, view_deg, command):
        print(f"    {detail.joint:<14} reading {float(reading):7.2f} -> command {float(cmd):7.2f} "
              f"({float(cmd) - float(reading):+.2f}, {detail.note})")


# --- moving --------------------------------------------------------------------------------


def go_to_view_pose(
    urdf_path: Path,
    *,
    robot: bool,
    current_deg: list[float] | None = None,
    compensate: bool = False,
) -> MoveResult:
    """w: plan, check and print, then drive to the view pose (with --robot). No prompt: a path
    that fails the clearance checks is refused before anything moves. compensate: end at the
    friction/sag model's command for the saved reading (see the module docstring)."""
    pose = load_view_pose()
    if pose is None:
        return MoveResult(False, "no view pose saved yet: put the arm where it sees the board, then press e")
    view = pose["joint_reading_deg"]
    if robot:
        current_deg = read_joints_robot()  # works with torque off too
    if current_deg is None:
        return MoveResult(False, "no joint readings (run with --robot)")
    command, details = view_command(current_deg, view, compensate)
    try:
        plan = plan_view_move(current_deg, command, urdf_path)
    except ViewMoveRefused as exc:
        print(f"VIEW POSE: refused, nothing moved: {exc}")
        return MoveResult(False, f"view pose refused: {exc}")
    print(f"VIEW POSE: {np.round(current_deg, 1).tolist()} -> {np.round(view, 1).tolist()} (gripper kept)")
    print_compensation(view, command, details)
    print(f"  {plan.order}: {len(plan.path)} steps, {plan.seconds:.1f} s at up to {MAX_SPEED_DEG_S:g} deg/s")
    print(f"  tightest point: {plan.lowest.describe()}")
    if not robot:
        return MoveResult(False, f"view pose: dry run only (no --robot); path clear, {plan.order}")
    released = not arm_held()
    follower = _follower()
    if released:
        # Goal = present position first, then torque on: the arm holds where it is, no jump.
        print("  turning torque on: the arm holds where it is")
        set_arm_hold(True)
    start = np.asarray(read_joints_robot())
    drift = float(np.max(np.abs(start[:5] - np.asarray(current_deg)[:5])))
    if drift > START_TOLERANCE_DEG:
        # It settled or was nudged since planning (e.g. as torque came on): plan again from here.
        print(f"  the arm moved {drift:.1f} deg since planning; planning again from where it is")
        command, details = view_command(start, view, compensate)  # directions may have changed
        try:
            plan = plan_view_move(start, command, urdf_path)
        except ViewMoveRefused as exc:
            print(f"VIEW POSE: refused, nothing moved (the arm is held where it is): {exc}")
            return MoveResult(released, f"view pose refused: {exc}")
    follower.bus.sync_write("Goal_Position", follower.bus.sync_read("Present_Position"))
    for step in plan.path:
        follower.send_action({f"{name}.pos": float(value) for name, value in zip(JOINT_NAMES, step)})
        time.sleep(COMMAND_DELAY_S)
    time.sleep(SETTLE_S)
    landed = np.asarray(read_joints_robot())
    error = landed[:5] - np.asarray(view)[:5]
    with VIEW_MOVES_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "at": datetime.now(timezone.utc).isoformat(),
            "start_deg": np.round(start, 3).tolist(),
            "view_deg": [round(float(v), 3) for v in view],
            "compensated": bool(details),
            "command_deg": np.round(plan.path[-1], 3).tolist(),
            "landed_deg": np.round(landed, 3).tolist(),
            "order": plan.order,
        }) + "\n")
    mode = "compensated" if details else "uncompensated"
    message = f"at the view pose (held, {mode}); landed within {float(np.max(np.abs(error))):.1f} deg of it"
    print(f"  landed {np.round(landed, 2).tolist()}, off by {np.round(error, 2).tolist()} deg")
    return MoveResult(True, message)
