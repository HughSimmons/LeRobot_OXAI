#!/usr/bin/env python3
"""Plan arm poses, and the routes between them, for the wrist camera's lens-calibration photos.

Each view asks for the square (default d4) at a place in the picture, seen from a tilt and
direction at a distance. Inverse kinematics on the wrist-camera URDF (with the friction fit's
joint offsets, as the click tool uses) finds joint readings for it.

Kept to what the arm has already done with the camera mount on, so nothing new can hit it:
- every joint stays inside the range recorded on the real arm with the mount fitted
  (SAFE_LIMITS: 45,961 readings in real_world_runs/ and the click tool's log, 2026-11-03), with
  wrist_roll in probe_xy_pose.py's PROBE_ROLL_RANGE_DEG (outside it the camera swung into the
  motors once) and wrist_flex inside the URDF's +/-95 deg;
- every move goes through HUB, the via pose probe_xy_pose.py uses above d4 (run on the arm
  many times): out of home the way probe does, pan and roll only at hub height, then straight
  down to the view, and back up the same way;
- every interpolated step of every route is checked by forward kinematics: the fixed jaw,
  gripper, wrist, camera mount and camera stay 25-45 mm above the board (10-30 mm above the
  table off it), and nothing goes past the board's far edge.

Planning uses a pinhole camera at the focal length forward kinematics implies (about 516 px;
the bundled lens file is wrong), so the real, distorted view shows a little more than the preview.

Writes, in output/:
  wrist_pose_plan.json   joint readings per view and the route checks (collect_wrist_photos.py runs it)
  wrist_pose_plan.png    the planned camera view and a side/top sketch of the arm per view

Nothing here moves the arm. Run with the click3d venv, from this folder:
  ~/venvs/click3d/bin/python plan_wrist_poses.py [--square d4]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
LOWLEVEL = HERE.parent
for _path in (LOWLEVEL, LOWLEVEL / "click_to_3d_oliver"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from click3d_shared import (  # noqa: E402
    JOINT_NAMES,
    SQUARE_MM,
    board_to_base_mm,
    joint_offsets_deg,
    parse_square_name,
    square_file_rank,
)

OUTPUT = HERE / "output"
DEFAULT_PLAN = OUTPUT / "wrist_pose_plan.json"
WIDTH, HEIGHT = 640, 480
# Between the forward-kinematics estimate from the d4 calibration (516) and the first real view-1
# photo (about 580, 2026-11-03). Each view's grid must stay clear at both FOCAL_CHECK_PX, since
# the real focal isn't known yet (a longer one spreads the picture out from the centre).
FOCAL_PX = 550.0
FOCAL_CHECK_PX = (500.0, 620.0)
K = np.array([[FOCAL_PX, 0.0, WIDTH / 2], [0.0, FOCAL_PX, HEIGHT / 2], [0.0, 0.0, 1.0]])

# Motor readings (lerobot degrees), arm joints only. DEFAULT_HOME is chess_traj's rest pose,
# which probe_xy_pose.py and the lookup runner start and end at (folded beside the base).
HOME = np.array([96.92307692307692, -107.86813186813187, 97.36263736263736, 65.18681318681318, -29.846153846153847])
# probe_xy_pose.py's via above d4 (click_moves/20261103-*_d4/probe_xy_pose.json, via_command_deg).
HUB = np.array([-8.19518577767831, 9.163911644562647, -35.481259446315384, 95.0, -108.11746400705248])
# Recorded range with the mount on (min / max over real readings, rounded outward), except
# wrist_flex capped at the URDF's 95 and wrist_roll at PROBE_ROLL_RANGE_DEG's -150. Roll's top is the
# recorded -28.1, not home's commanded -29.8: the arm reads about -29.1 at rest, so a start there must pass.
SAFE_LIMITS = {
    "shoulder_pan": (-18.5, 97.5),
    "shoulder_lift": (-113.0, 56.0),
    "elbow_flex": (-73.5, 97.5),
    "wrist_flex": (45.0, 95.0),
    "wrist_roll": (-150.0, -28.0),
}
LIMITS_LO = np.array([SAFE_LIMITS[n][0] for n in JOINT_NAMES[:5]])
LIMITS_HI = np.array([SAFE_LIMITS[n][1] for n in JOINT_NAMES[:5]])
# The views themselves stay further inside, so none presses on an edge of that range: shoulder_lift
# no lower than home's, wrist_roll 5 deg in from each end. wrist_flex may reach HUB's 95.
VIEW_LO = np.array([-17.0, -107.0, -72.0, 46.0, -145.0])
VIEW_HI = np.array([96.0, 54.0, 96.0, 95.0, -35.0])
GRIPPER_READING = 5.0  # jaw nearly closed, as in the saved wrist photos; never moved
# The jaws cover the bottom middle of the wrist image (from the saved wrist photo).
JAW_POLYGON = np.array([[225, 335], [375, 335], [465, 480], [90, 480]], dtype=np.float64)
# Each view must show a complete block of inner corners round the square, this far inside the
# frame and outside the jaws: on the first real photo the detector took a jaw tip 9 px above the
# polygon for a corner. GRID_SIZES are tried in order.
GRID_MARGIN_PX = 20.0
GRID_SIZES = (5, 4)
BOARD_SURFACE_Z = float(board_to_base_mm(np.zeros(3))[2])
_board_far = board_to_base_mm(np.array([8 * SQUARE_MM, 8 * SQUARE_MM, 0.0]))
_board_near = board_to_base_mm(np.zeros(3))
FOOTPRINT_PAD_MM = 40.0
BOARD_X = (float(_board_near[0]) - FOOTPRINT_PAD_MM, float(_board_far[0]) + FOOTPRINT_PAD_MM)
BOARD_Y = (float(_board_near[1]) - FOOTPRINT_PAD_MM, float(_board_far[1]) + FOOTPRINT_PAD_MM)
MAX_X_MM = float(_board_far[0]) + 10.0  # nothing past the board's far edge (the tripod is out there)
# Minimum height above the board (over its footprint) / above the table (elsewhere), per frame.
CLEARANCE_MM = {
    "gripper_frame_link": (25.0, 10.0),  # fixed jaw tip
    "gripper_link": (40.0, 30.0),
    "wrist_link": (45.0, 30.0),
    "wrist_camera_mount": (35.0, 25.0),
    "wrist_camera_link": (40.0, 25.0),
}
CHECK_STEP_DEG = 0.5


@dataclass
class View:
    hint: str
    pixel: tuple[float, float]  # where the square's centre should be in the picture
    tilt_deg: float  # optical axis from straight down
    azimuth_deg: float | None  # direction from the square to the camera: 180 = toward the robot, +90 = robot's left
    distance_mm: float
    roll_deg: float | None = None  # image roll away from view 1's (None: free)


VIEWS = [
    View("face-on, square centred", (320, 200), 5, None, 250),
    View("face-on, square TOP-LEFT of frame", (110, 105), 10, None, 230),
    View("face-on, square TOP-RIGHT of frame", (530, 105), 10, None, 230),
    View("face-on, square BOTTOM-LEFT of frame", (105, 335), 10, None, 230),
    View("face-on, square BOTTOM-RIGHT of frame", (535, 335), 10, None, 230),
    View("tilted 40, camera on the robot's side", (320, 200), 40, 180, 230),
    View("tilted 30, camera to the robot's right", (320, 200), 30, -90, 210),
    View("tilted 30, camera to the robot's left", (320, 200), 30, 90, 210),
    View("tilted 45, diagonal from robot's front-right", (320, 200), 45, -135, 230),
    View("tilted 45, diagonal from robot's front-left", (320, 200), 45, 135, 230),
    View("face-on, camera rolled 45 about its axis", (320, 200), 10, None, 240, roll_deg=45.0),
    View("far: whole board in the middle", (320, 220), 15, None, 300),
    View("tilted 30, square at the LEFT edge", (105, 230), 30, 180, 230),
    View("tilted 30, square at the RIGHT edge", (535, 230), 30, 180, 230),
    View("tilted 55, low view from the robot's side", (320, 180), 55, 180, 220),
]


def angle_diff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


class Arm:
    """Forward kinematics of the wrist-camera URDF for motor readings (degrees)."""

    LINKS = ("base_link", "shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link",
             "gripper_link", "gripper_frame_link", "wrist_camera_mount", "wrist_camera_link")

    def __init__(self) -> None:
        from paths import import_robot_kinematics, resolve_wrist_cam_urdf

        urdf_path, _ = resolve_wrist_cam_urdf()
        self.kin = import_robot_kinematics()(
            urdf_path=urdf_path, target_frame_name="wrist_camera_link", joint_names=JOINT_NAMES
        )
        self.offsets = joint_offsets_deg()

    def frames(self, readings5: np.ndarray) -> dict[str, np.ndarray]:
        reading = np.append(np.asarray(readings5, dtype=np.float64)[:5], GRIPPER_READING)
        self.kin.forward_kinematics(reading + self.offsets)
        out = {}
        for name in self.LINKS:
            pose = np.array(self.kin.robot.get_T_world_frame(name), dtype=np.float64)
            pose[:3, 3] *= 1000.0
            out[name] = pose
        return out


def clearance_slack(frames: dict[str, np.ndarray]) -> dict[str, float]:
    """Per checked frame: mm above its required height (negative = too low)."""
    slack = {}
    for name, (board_margin, table_margin) in CLEARANCE_MM.items():
        x, y, z = frames[name][:3, 3]
        over_board = BOARD_X[0] <= x <= BOARD_X[1] and BOARD_Y[0] <= y <= BOARD_Y[1]
        floor = BOARD_SURFACE_Z + board_margin if over_board else table_margin
        slack[name] = float(z - floor)
    return slack


def pose_problems(arm: Arm, q: np.ndarray, frames: dict[str, np.ndarray] | None = None) -> list[str]:
    """Why a pose is unsafe: outside SAFE_LIMITS, too low, or past the board's far edge."""
    problems = []
    q = np.asarray(q, dtype=np.float64)[:5]
    for name, value, lo, hi in zip(JOINT_NAMES, q, LIMITS_LO, LIMITS_HI):
        if not lo - 1e-6 <= value <= hi + 1e-6:
            problems.append(f"{name} {value:.1f} outside [{lo:.1f}, {hi:.1f}]")
    frames = frames if frames is not None else arm.frames(q)
    for name, slack in clearance_slack(frames).items():
        if slack < 0:
            problems.append(f"{name} {-slack:.0f} mm too low")
    far = max(float(frames[name][0, 3]) for name in CLEARANCE_MM)
    if far > MAX_X_MM:
        problems.append(f"reaches x = {far:.0f} mm, past the board's far edge")
    return problems


def hub_at(q: np.ndarray, home_side: bool = False) -> np.ndarray:
    """HUB swung to q's pan and turned to q's roll (HUB's roll on the home side, as probe does)."""
    pose = HUB.copy()
    pose[0] = q[0]
    if not home_side:
        pose[4] = q[4]
    return pose


def near_home(q: np.ndarray, tol_deg: float = 12.0) -> bool:
    return bool(np.all(np.abs(np.asarray(q)[:4] - HOME[:4]) <= tol_deg))


def route(start: np.ndarray, end: np.ndarray) -> list[np.ndarray]:
    """Waypoints from start to end through hub height: up, swing and turn, then down."""
    start = np.asarray(start, dtype=np.float64)[:5]
    end = np.asarray(end, dtype=np.float64)[:5]
    points = [start, hub_at(start, near_home(start)), hub_at(end, near_home(end)), end]
    out = [points[0]]
    for p in points[1:]:
        if np.max(np.abs(p - out[-1])) > 1e-6:
            out.append(p)
    return out


def check_route(arm: Arm, waypoints: list[np.ndarray]) -> dict:
    """Forward-kinematics check of every CHECK_STEP_DEG step along the waypoints."""
    problems: list[str] = []
    min_slack = {name: np.inf for name in CLEARANCE_MM}
    travel = 0.0
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        largest = float(np.max(np.abs(b - a)))
        travel += largest
        steps = max(1, int(np.ceil(largest / CHECK_STEP_DEG)))
        for k in range(steps + 1):
            q = a + (b - a) * (k / steps)
            frames = arm.frames(q)
            for name, slack in clearance_slack(frames).items():
                min_slack[name] = min(min_slack[name], slack)
            for problem in pose_problems(arm, q, frames):
                if problem not in problems:
                    problems.append(problem)
    return {
        "ok": not problems,
        "problems": problems,
        "min_clearance_slack_mm": {k: round(float(v), 1) for k, v in min_slack.items()},
        "largest_joint_travel_deg": round(travel, 1),
    }


def board_points_base(square: str) -> dict[str, np.ndarray]:
    file_i, rank_i = square_file_rank(square)
    s = SQUARE_MM
    corners_board = np.array(
        [[file_i * s, rank_i * s, 0], [(file_i + 1) * s, rank_i * s, 0],
         [(file_i + 1) * s, (rank_i + 1) * s, 0], [file_i * s, (rank_i + 1) * s, 0]], dtype=np.float64
    )
    inner = np.array([[i * s, j * s, 0] for j in range(1, 8) for i in range(1, 8)], dtype=np.float64)
    return {
        "centre": board_to_base_mm(corners_board.mean(axis=0)),
        "corners": np.array([board_to_base_mm(p) for p in corners_board]),
        "inner": np.array([board_to_base_mm(p) for p in inner]),
    }


def project(points_base: np.ndarray, T_base_cam: np.ndarray, focal: float = FOCAL_PX) -> tuple[np.ndarray, np.ndarray]:
    T_cam_base = np.linalg.inv(T_base_cam)
    cam = (T_cam_base[:3, :3] @ np.asarray(points_base, dtype=np.float64).reshape(-1, 3).T).T + T_cam_base[:3, 3]
    depth = cam[:, 2]
    camera = np.array([[focal, 0.0, WIDTH / 2], [0.0, focal, HEIGHT / 2], [0.0, 0.0, 1.0]])
    pix = (camera @ (cam / np.maximum(depth[:, None], 1e-6)).T).T[:, :2]
    return pix, depth


def grid_blocks(square: str) -> list[tuple[int, int, int, list[int]]]:
    """(size, first row, first col, indices into board_points_base()["inner"]) of every square
    block of inner corners that contains the square's four corners, largest first."""
    file_i, rank_i = square_file_rank(square)
    cols, rows = (file_i - 1, file_i), (rank_i - 1, rank_i)  # inner corner i*s is index i - 1
    blocks = []
    for size in GRID_SIZES:
        for r0 in range(0, 8 - size):
            for c0 in range(0, 8 - size):
                if c0 <= min(cols) and max(cols) < c0 + size and r0 <= min(rows) and max(rows) < r0 + size:
                    blocks.append((size, r0, c0, [r * 7 + c for r in range(r0, r0 + size) for c in range(c0, c0 + size)]))
    return blocks


def grid_slack(pix: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Per point: px it has to spare inside the frame and outside the jaws (negative = too close)."""
    out = []
    jaw = JAW_POLYGON.astype(np.float32)
    for (x, y), d in zip(pix, depth):
        if d <= 1:
            out.append(-100.0)
            continue
        frame = min(x, WIDTH - x, y, HEIGHT - y)
        outside_jaw = -cv2.pointPolygonTest(jaw, (float(x), float(y)), True)
        out.append(min(frame, outside_jaw) - GRID_MARGIN_PX)
    return np.asarray(out)


def visible(pix: np.ndarray, depth: np.ndarray, margin: float = 4.0) -> np.ndarray:
    inside = (depth > 1) & (pix[:, 0] > margin) & (pix[:, 0] < WIDTH - margin) & (pix[:, 1] > margin) & (pix[:, 1] < HEIGHT - margin)
    jaw = np.array([cv2.pointPolygonTest(JAW_POLYGON.astype(np.float32), (float(x), float(y)), False) >= 0 for x, y in pix])
    return inside & ~jaw


def view_geometry(T_base_cam: np.ndarray, target: np.ndarray) -> dict[str, float]:
    axis = T_base_cam[:3, 2]
    to_cam = T_base_cam[:3, 3] - target
    x_axis = T_base_cam[:3, 0]
    return {
        "tilt_deg": float(np.degrees(np.arccos(np.clip(-axis[2], -1, 1)))),
        "azimuth_deg": float(np.degrees(np.arctan2(to_cam[1], to_cam[0]))),
        "distance_mm": float(np.linalg.norm(to_cam)),
        "roll_deg": float(np.degrees(np.arctan2(x_axis[1], x_axis[0]))),
    }


def residuals(q: np.ndarray, arm: Arm, view: View, pts: dict, roll_ref: float | None,
              block: list[int] | None = None) -> np.ndarray:
    frames = arm.frames(q)
    T = frames["wrist_camera_link"]
    pix, depth = project(pts["centre"][None], T)
    geo = view_geometry(T, pts["centre"])
    res = [(pix[0, 0] - view.pixel[0]) / 15.0, (pix[0, 1] - view.pixel[1]) / 15.0]
    res.append((geo["tilt_deg"] - view.tilt_deg) / 3.0)
    res.append((geo["distance_mm"] - view.distance_mm) / 20.0)
    if view.azimuth_deg is not None:
        res.append(angle_diff_deg(geo["azimuth_deg"], view.azimuth_deg) / 8.0)
    if view.roll_deg is not None and roll_ref is not None:
        res.append(abs(angle_diff_deg(geo["roll_deg"], roll_ref)) / 4.0 - view.roll_deg / 4.0)
    res.append(0.0 if depth[0] > 1 else 100.0)
    # Keep clear of the board with 10 mm to spare, so the route check has room.
    for slack in clearance_slack(frames).values():
        res.append(max(0.0, 10.0 - slack) / 2.0)
    far = max(float(frames[name][0, 3]) for name in CLEARANCE_MM)
    res.append(max(0.0, far - (MAX_X_MM - 10.0)) / 2.0)
    if block is not None:
        for focal in FOCAL_CHECK_PX:
            res.extend(np.maximum(0.0, -grid_slack(*project(pts["inner"][block], T, focal))) / 2.0)
    return np.asarray(res, dtype=np.float64)


def solve(arm: Arm, view: View, pts: dict, roll_ref: float | None, seeds: list[np.ndarray],
          square: str) -> tuple[np.ndarray, tuple | None]:
    """The view's pose, and the grid block it shows clear of the frame edges and jaws (None if none).

    First without the grid, as a seed; then for each block round the square, largest first,
    keeping the best fit whose block is fully clear. A smaller block is tried only if no larger one fits.
    """
    def fit(start, block):
        return least_squares(residuals, np.clip(start, VIEW_LO + 1e-3, VIEW_HI - 1e-3), bounds=(VIEW_LO, VIEW_HI),
                             args=(arm, view, pts, roll_ref, block), diff_step=1e-3, max_nfev=400)

    free = min((fit(seed, None) for seed in seeds), key=lambda f: f.cost)
    starts = [free.x, HUB.copy()] + seeds[2:6]
    for size in GRID_SIZES:
        best = None
        for block in [b for b in grid_blocks(square) if b[0] == size]:
            for start in starts:
                result = fit(start, block[3])
                T = arm.frames(result.x)["wrist_camera_link"]
                worst = min(grid_slack(*project(pts["inner"][block[3]], T, f)).min() for f in FOCAL_CHECK_PX)
                if worst >= -0.5 and (best is None or result.cost < best[0].cost):
                    best = (result, block)
        if best is not None:
            return best[0].x, best[1]
    return free.x, None


def evaluate(arm: Arm, q: np.ndarray, view: View, pts: dict, roll_ref: float | None, block: tuple | None) -> dict:
    frames = arm.frames(q)
    T = frames["wrist_camera_link"]
    geo = view_geometry(T, pts["centre"])
    cpix, _ = project(pts["centre"][None], T)
    sq_pix, sq_depth = project(pts["corners"], T)
    in_pix, in_depth = project(pts["inner"], T)
    problems = []
    if float(np.hypot(*(cpix[0] - np.asarray(view.pixel)))) > 40:
        problems.append("square not where asked")
    if abs(geo["tilt_deg"] - view.tilt_deg) > 8:
        problems.append(f"tilt {geo['tilt_deg']:.0f} not {view.tilt_deg:.0f}")
    if view.azimuth_deg is not None and abs(angle_diff_deg(geo["azimuth_deg"], view.azimuth_deg)) > 25:
        problems.append(f"direction {geo['azimuth_deg']:.0f} not {view.azimuth_deg:.0f}")
    if abs(geo["distance_mm"] - view.distance_mm) > 50:
        problems.append(f"distance {geo['distance_mm']:.0f} not {view.distance_mm:.0f}")
    if view.roll_deg is not None and roll_ref is not None and abs(abs(angle_diff_deg(geo["roll_deg"], roll_ref)) - view.roll_deg) > 10:
        problems.append(f"roll {abs(angle_diff_deg(geo['roll_deg'], roll_ref)):.0f} not {view.roll_deg:.0f}")
    hidden = ~visible(sq_pix, sq_depth)
    if hidden.any():
        problems.append(f"square corners {[i + 1 for i in np.flatnonzero(hidden)]} out of view or behind the jaws")
    grid = None
    if block is not None:
        slack = {f: float(grid_slack(*project(pts["inner"][block[3]], T, f)).min()) for f in (FOCAL_PX, *FOCAL_CHECK_PX)}
        grid = {"size": block[0], "first_row": block[1], "first_col": block[2],
                "corners_px": np.round(project(pts["inner"][block[3]], T)[0], 1).tolist(),
                "min_slack_px": {str(int(f)): round(v, 1) for f, v in slack.items()}}
        if min(slack.values()) < -1.0:
            problems.append(f"grid touches the edge or jaws at some focal: {grid['min_slack_px']}")
            grid = None
    else:
        problems.append("no clear 4x4 grid round the square")
    safety = pose_problems(arm, q, frames)
    return {
        "joints_reading_deg": [round(float(v), 2) for v in q] + [GRIPPER_READING],
        "achieved": {k: round(v, 1) for k, v in geo.items()},
        "square_centre_px": [round(float(v), 1) for v in cpix[0]],
        "square_corners_px": np.round(sq_pix, 1).tolist(),
        "inner_corners_visible": int(visible(in_pix, in_depth).sum()),
        "grid": grid,
        "grid_ok": grid is not None,
        "clearance_slack_mm": {k: round(v, 1) for k, v in clearance_slack(frames).items()},
        "camera_base_mm": np.round(T[:3, 3], 1).tolist(),
        "view_ok": not problems,
        "view_problems": problems,
        "pose_safety_problems": safety,
    }


def render_view(T_base_cam: np.ndarray, square: str) -> np.ndarray:
    """What the camera should roughly see: the board's squares, the target square, the jaws."""
    image = np.full((HEIGHT, WIDTH, 3), (70, 90, 110), dtype=np.uint8)
    s = SQUARE_MM
    frame = np.array([[-25, -25, 0], [8 * s + 25, -25, 0], [8 * s + 25, 8 * s + 25, 0], [-25, 8 * s + 25, 0]])
    polys = [(frame, (60, 110, 160))]
    for i in range(8):
        for j in range(8):
            corners = np.array([[i * s, j * s, 0], [(i + 1) * s, j * s, 0], [(i + 1) * s, (j + 1) * s, 0], [i * s, (j + 1) * s, 0]])
            polys.append((corners, (40, 60, 95) if (i + j) % 2 == 0 else (200, 215, 225)))
    for corners, color in polys:
        base = np.array([board_to_base_mm(p) for p in corners])
        fine = np.array([base[k] + t * (base[(k + 1) % 4] - base[k]) for k in range(4) for t in np.linspace(0, 1, 12, endpoint=False)])
        pix, depth = project(fine, T_base_cam)
        if (depth > 1).all():
            cv2.fillPoly(image, [np.round(pix).astype(np.int32).reshape(-1, 1, 2)], color, cv2.LINE_AA)
    sq_pix, _ = project(board_points_base(square)["corners"], T_base_cam)
    cv2.polylines(image, [np.round(sq_pix).astype(np.int32).reshape(-1, 1, 2)], True, (255, 200, 0), 2, cv2.LINE_AA)
    cv2.circle(image, tuple(np.round(sq_pix[0]).astype(int)), 7, (0, 255, 0), -1)
    cv2.fillPoly(image, [JAW_POLYGON.astype(np.int32).reshape(-1, 1, 2)], (90, 60, 230))
    return image


def render_arm(frames: dict[str, np.ndarray]) -> np.ndarray:
    """Side (x-z) and top (x-y) sketch of the arm and board, robot base at the left."""
    sketch = np.full((HEIGHT, 300, 3), 245, dtype=np.uint8)
    chain = ["base_link", "shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link", "gripper_frame_link"]
    pts = np.array([frames[n][:3, 3] for n in chain])
    cam = frames["wrist_camera_link"][:3, 3]
    scale = 0.5

    def side(p):
        return int(60 + scale * p[0]), int(225 - scale * p[2])

    def top(p):
        return int(60 + scale * p[0]), int(360 - scale * p[1])

    cv2.putText(sketch, "side", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.line(sketch, side((-120, 0, 0)), side((560, 0, 0)), (150, 150, 150), 1)
    cv2.rectangle(sketch, side((_board_near[0], 0, BOARD_SURFACE_Z)), side((_board_far[0], 0, 0)), (60, 110, 160), -1)
    cv2.putText(sketch, "top", (6, 246), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.rectangle(sketch, top((_board_near[0], _board_near[1], 0)), top((_board_far[0], _board_far[1], 0)), (60, 110, 160), -1)
    for project_fn in (side, top):
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(sketch, project_fn(a), project_fn(b), (40, 40, 40), 3, cv2.LINE_AA)
        cv2.circle(sketch, project_fn(cam), 5, (0, 160, 0), -1)
        axis = frames["wrist_camera_link"][:3, 2]
        cv2.line(sketch, project_fn(cam), project_fn(cam + 80 * axis), (0, 160, 0), 1, cv2.LINE_AA)
    return sketch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--square", default="d4", type=lambda s: parse_square_name(s) or parser.error(f"bad square {s!r}"))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()

    arm = Arm()
    for name, pose in (("home", HOME), ("hub", HUB)):
        bad = pose_problems(arm, pose)
        if bad:
            raise SystemExit(f"{name} pose fails its own checks: {bad}")
    pts = board_points_base(args.square)
    rng = np.random.default_rng(0)
    # The hub and the saved d4 click-tool pose, plus spread-out random starts.
    seeds = [HUB.copy(), np.array([3.8, -111.0, 96.0, 61.7, -89.1])]
    seeds += [VIEW_LO + rng.random(5) * (VIEW_HI - VIEW_LO) for _ in range(12)]

    results = []
    roll_ref = None
    tiles = []
    for index, view in enumerate(VIEWS, start=1):
        q, block = solve(arm, view, pts, roll_ref, seeds + [np.asarray(r["joints_reading_deg"][:5]) for r in results],
                         args.square)
        result = evaluate(arm, q, view, pts, roll_ref, block)
        if index == 1:
            roll_ref = result["achieved"]["roll_deg"]
        route_check = check_route(arm, route(HUB, q))
        result = {"view": index, "hint": view.hint, "asked": view.__dict__, **result, "route_from_hub": route_check}
        # Safe = the pose and the route to and from the hub pass; a view that misses its aim is still
        # taken, but not one without a clear grid (grid_ok), which a lens fit can't use.
        result["safe"] = not result["pose_safety_problems"] and route_check["ok"]
        results.append(result)
        notes = []
        if not result["safe"]:
            notes.append("UNSAFE, skipped: " + "; ".join(result["pose_safety_problems"] + route_check["problems"]))
        if result["view_problems"]:
            notes.append("aim: " + "; ".join(result["view_problems"]))
        slack = min(route_check["min_clearance_slack_mm"].values())
        print(f"{index:2d}. {view.hint:46s} joints {[round(float(v), 1) for v in q]}  tilt {result['achieved']['tilt_deg']:5.1f}  "
              f"dist {result['achieved']['distance_mm']:4.0f}  "
              f"grid {str(result['grid']['size']) + 'x' + str(result['grid']['size']) if result['grid'] else 'NONE'} "
              f"route slack {slack:5.1f} mm  {' | '.join(notes) or 'ok'}")
        frames = arm.frames(q)
        view_image = render_view(frames["wrist_camera_link"], args.square)
        if result["grid"]:
            for x, y in result["grid"]["corners_px"]:
                cv2.circle(view_image, (int(round(x)), int(round(y))), 5, (0, 220, 255), 2, cv2.LINE_AA)
        tile = np.hstack([view_image, render_arm(frames)])
        label = f"{index}. {view.hint}" + ("" if result["safe"] else "  [UNSAFE: skipped]") + (
            "" if result["grid_ok"] else "  [NO GRID: skipped]")
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 26), (255, 255, 255), -1)
        cv2.putText(tile, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0) if result["safe"] else (0, 0, 200), 1, cv2.LINE_AA)
        cv2.putText(tile, "joints " + " ".join(f"{v:.0f}" for v in q), (8, HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(tile, (tile.shape[1] // 2, tile.shape[0] // 2), interpolation=cv2.INTER_AREA))

    home_check = check_route(arm, route(HOME, HUB))
    print(f"home <-> hub: {'ok' if home_check['ok'] else 'FAILS: ' + '; '.join(home_check['problems'])}  "
          f"slack {min(home_check['min_clearance_slack_mm'].values()):.1f} mm")

    coverage = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    cv2.fillPoly(coverage, [JAW_POLYGON.astype(np.int32).reshape(-1, 1, 2)], (220, 220, 255))
    for r in results:
        cv2.polylines(coverage, [np.round(np.asarray(r["square_corners_px"])).astype(np.int32).reshape(-1, 1, 2)], True,
                      (0, 140, 0) if r["safe"] and r["grid_ok"] else (0, 0, 200), 2, cv2.LINE_AA)
        cv2.putText(coverage, str(r["view"]), tuple(np.round(r["square_centre_px"]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(coverage, f"{args.square} in every planned view", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    tiles.append(np.hstack([cv2.resize(coverage, (320, 240)), np.full((240, tiles[0].shape[1] - 320, 3), 255, np.uint8)]))
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])

    args.out.mkdir(parents=True, exist_ok=True)
    plan = {
        "camera": "wrist",
        "square": args.square,
        "square_mm": SQUARE_MM,
        "focal_px_assumed": FOCAL_PX,
        "gripper_reading_deg": GRIPPER_READING,
        "joint_names": list(JOINT_NAMES),
        "joint_offsets_deg": [round(float(v), 3) for v in arm.offsets],
        "home_reading_deg": HOME.tolist(),
        "hub_reading_deg": HUB.tolist(),
        "safe_limits_reading_deg": SAFE_LIMITS,
        "clearance_mm": {k: {"board": v[0], "table": v[1]} for k, v in CLEARANCE_MM.items()},
        "home_hub_route": home_check,
        "note": "joints_reading_deg are motor readings (what the arm reports), not URDF angles.",
        "views": results,
    }
    out_json = args.out / "wrist_pose_plan.json"
    out_json.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    cv2.imwrite(str(args.out / "wrist_pose_plan.png"), sheet)
    print(f"{sum(r['safe'] and r['grid_ok'] for r in results)}/{len(results)} views safe and with a clear grid; wrote {out_json} and wrist_pose_plan.png")


if __name__ == "__main__":
    main()
