#!/usr/bin/env python3
"""Shared helpers for click-to-3D and mount calibration.

Assets resolve CLI path → local project path → bundled backup.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "bundle"
OUTPUT = ROOT / "output"
MOUNT_DIR = OUTPUT / "camera_mount"
SESSIONS_DIR = OUTPUT / "sessions"
LIB = ROOT / "lib"

if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from calib_board_pose import (  # noqa: E402
    invert,
    pixel_ray_board,
    project_points,
    ray_plane_z0,
    se3,
    square_file_rank,
)
from calib_intrinsics import (  # noqa: E402
    detect_largest_inner_grid,
    load_fisheye_calibration,
    undistort_fisheye,
)

SQUARE_MM = 41.75
SESSION_CONFIG = OUTPUT / "session_config.yaml"
OXAI_PATHS_HINT = (
    "Place this folder at LeRobot_OXAI/lowlevel/click_to_3d so the parent "
    "directory contains paths.py. Use --demo-pair without the robot."
)
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def lowlevel_candidates() -> list[Path]:
    """Resolve OXAI lowlevel for FK / robot API.

    Prefer the drop-in layout LeRobot_OXAI/lowlevel/click_to_3d (parent has paths.py).
    Fall back to the coord_finder sibling layout under Documents/Vs.
    """
    candidates: list[Path] = []
    parent = ROOT.parent
    if (parent / "paths.py").is_file():
        candidates.append(parent)
    candidates.extend(
        [
            ROOT.parent.parent / "LeRobot_OXAI" / "lowlevel",
            ROOT.parent.parent / "chessbot" / "LeRobot_OXAI" / "lowlevel",
        ]
    )
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


LOWLEVEL_CANDIDATES = lowlevel_candidates()


@dataclass
class ResolvedAsset:
    name: str
    path: Path
    source: str  # cli | local | bundle


@dataclass
class AssetReport:
    items: list[ResolvedAsset] = field(default_factory=list)

    def add(self, item: ResolvedAsset) -> ResolvedAsset:
        self.items.append(item)
        return item

    @property
    def backups(self) -> list[ResolvedAsset]:
        return [item for item in self.items if item.source == "bundle"]

    def banner_text(self) -> str | None:
        if not self.backups:
            return None
        names = ", ".join(item.name for item in self.backups)
        return f"BACKUP IN USE: {names}"


def ensure_output_dirs() -> None:
    MOUNT_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def resolve_asset(
    name: str,
    *,
    cli_path: Path | None,
    local_candidates: list[Path],
    bundle_path: Path,
    report: AssetReport,
) -> ResolvedAsset:
    if cli_path is not None:
        path = Path(cli_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{name}: CLI path missing: {path}")
        return report.add(ResolvedAsset(name, path, "cli"))
    local = _first_existing([Path(p).expanduser().resolve() for p in local_candidates])
    if local is not None:
        return report.add(ResolvedAsset(name, local, "local"))
    bundle = Path(bundle_path).expanduser().resolve()
    if bundle.is_file():
        return report.add(ResolvedAsset(name, bundle, "bundle"))
    raise FileNotFoundError(f"{name}: not found in project paths or bundle ({bundle})")


def resolve_core_assets(
    report: AssetReport | None = None,
    *,
    fisheye_cli: Path | None = None,
    stationary_cli: Path | None = None,
    urdf_cli: Path | None = None,
) -> tuple[AssetReport, dict[str, ResolvedAsset]]:
    report = report or AssetReport()
    coord = ROOT.parent
    fisheye = resolve_asset(
        "wrist_fisheye.yaml",
        cli_path=fisheye_cli,
        local_candidates=[coord / "output" / "lens_calib" / "wrist_fisheye.yaml"],
        bundle_path=BUNDLE / "wrist_fisheye.yaml",
        report=report,
    )
    stationary = resolve_asset(
        "stationary_intrinsics.yaml",
        cli_path=stationary_cli,
        local_candidates=[
            coord / "output" / "lens_calib" / "stationary_intrinsics.yaml",
            MOUNT_DIR / "stationary_intrinsics.yaml",
        ],
        bundle_path=BUNDLE / "stationary_intrinsics.yaml",
        report=report,
    )
    urdf_locals: list[Path] = []
    for lowlevel in LOWLEVEL_CANDIDATES:
        if str(lowlevel) not in sys.path:
            sys.path.insert(0, str(lowlevel))
        try:
            from paths import resolve_wrist_cam_urdf

            urdf_path, _mesh = resolve_wrist_cam_urdf()
            urdf_locals.append(Path(urdf_path))
            break
        except Exception:
            continue
    urdf_locals.extend(
        [
            ROOT.parent.parent
            / "chessbot"
            / "LeRobot_OXAI"
            / "SO-ARM100"
            / "Simulation"
            / "SO101"
            / "so101_new_calib_wrist_cam.urdf",
            ROOT.parent.parent
            / "LeRobot_OXAI"
            / "SO-ARM100"
            / "Simulation"
            / "SO101"
            / "so101_new_calib_wrist_cam.urdf",
        ]
    )
    urdf = resolve_asset(
        "so101_new_calib_wrist_cam.urdf",
        cli_path=urdf_cli,
        local_candidates=urdf_locals,
        bundle_path=BUNDLE / "so101_new_calib_wrist_cam.urdf",
        report=report,
    )
    return report, {"fisheye": fisheye, "stationary": stationary, "urdf": urdf}


def resolve_demo_pair(report: AssetReport) -> dict[str, ResolvedAsset]:
    demo = BUNDLE / "demo_pair003"
    local_sample = ROOT.parent / "sample images"
    wrist = resolve_asset(
        "demo_wrist.png",
        cli_path=None,
        local_candidates=[local_sample / "pair_003_shot_wrist.png"],
        bundle_path=demo / "pair_003_shot_wrist.png",
        report=report,
    )
    side = resolve_asset(
        "demo_stationary.png",
        cli_path=None,
        local_candidates=[local_sample / "pair_003_shot_stationary.png"],
        bundle_path=demo / "pair_003_shot_stationary.png",
        report=report,
    )
    joints = resolve_asset(
        "demo_joints.json",
        cli_path=None,
        local_candidates=[local_sample / "pair_003_shot_joints.json"],
        bundle_path=demo / "pair_003_shot_joints.json",
        report=report,
    )
    poses = resolve_asset(
        "demo_board_poses.json",
        cli_path=None,
        local_candidates=[ROOT.parent / "output" / "pair03_pnp" / "pair03_pnp.json"],
        bundle_path=demo / "board_poses.json",
        report=report,
    )
    return {"wrist": wrist, "stationary": side, "joints": joints, "poses": poses}


def load_stationary_matrix(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    try:
        data = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        node = data.getNode("K")
        if not node.empty():
            matrix = node.mat()
            data.release()
            return np.asarray(matrix, dtype=np.float64)
        data.release()
    except Exception:
        pass
    raw = yaml.safe_load(text)
    if isinstance(raw, dict) and "K" in raw:
        k = raw["K"]
        if isinstance(k, dict) and "data" in k:
            return np.asarray(k["data"], dtype=np.float64).reshape(3, 3)
        return np.asarray(k, dtype=np.float64).reshape(3, 3)
    raise RuntimeError(f"Could not read stationary K from {path}")


def open_camera(index: int, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


def _ensure_lowlevel_on_path() -> Path | None:
    for lowlevel in lowlevel_candidates():
        if lowlevel.is_dir() and (lowlevel / "paths.py").is_file():
            if str(lowlevel) not in sys.path:
                sys.path.insert(0, str(lowlevel))
            return lowlevel
    return None


def read_joints_robot() -> list[float]:
    if _ensure_lowlevel_on_path() is None:
        raise RuntimeError(f"Robot API not available. {OXAI_PATHS_HINT}")
    try:
        from paths import connect_so101_hardware
    except Exception as exc:  # pragma: no cover - hardware optional
        raise RuntimeError(f"Robot API not available. {OXAI_PATHS_HINT}") from exc
    follower = connect_so101_hardware(calibrate=False)
    obs = follower.get_observation()
    joints = []
    for name in JOINT_NAMES:
        key = f"{name}.pos"
        if key not in obs:
            raise RuntimeError(f"Observation missing {key}")
        joints.append(float(obs[key]))
    return joints


def read_joints_json(path: Path) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [float(v) for v in data["joints_deg"]]


def fk_link_mm(urdf_path: Path, target_frame: str, joints_deg: list[float]) -> np.ndarray:
    if _ensure_lowlevel_on_path() is None:
        raise RuntimeError(f"Could not import robot kinematics. {OXAI_PATHS_HINT}")
    try:
        from paths import import_robot_kinematics
    except Exception as exc:
        raise RuntimeError(f"Could not import robot kinematics. {OXAI_PATHS_HINT}") from exc

    robot_kinematics = import_robot_kinematics()
    kinematics = robot_kinematics(
        urdf_path=str(urdf_path),
        target_frame_name=target_frame,
        joint_names=JOINT_NAMES,
    )
    pose_m = np.asarray(
        kinematics.forward_kinematics(np.asarray(joints_deg, dtype=np.float64)),
        dtype=np.float64,
    )
    pose_mm = pose_m.copy()
    pose_mm[:3, 3] *= 1000.0
    return pose_mm


def load_session_camera_indices(
    wrist_cli: int | None,
    stationary_cli: int | None,
) -> tuple[int, int]:
    """CLI values win when set; otherwise load output/session_config.yaml; else 0/1."""
    wrist = wrist_cli
    stationary = stationary_cli
    if wrist is None or stationary is None:
        if SESSION_CONFIG.is_file():
            data = yaml.safe_load(SESSION_CONFIG.read_text(encoding="utf-8")) or {}
            if wrist is None and "wrist_index" in data:
                wrist = int(data["wrist_index"])
            if stationary is None and "stationary_index" in data:
                stationary = int(data["stationary_index"])
    return int(wrist if wrist is not None else 0), int(stationary if stationary is not None else 1)


def save_session_camera_indices(wrist_index: int, stationary_index: int) -> None:
    ensure_output_dirs()
    payload = {
        "wrist_index": int(wrist_index),
        "stationary_index": int(stationary_index),
        "note": "mounted camera left, stationary right in the click-to-3D window",
    }
    SESSION_CONFIG.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def solve_hand_eye_mount(
    samples: list[dict],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Eye-in-hand hand–eye: X = T_gripper_camera, Y = T_base_board.

    samples need T_base_gripper and T_board_camera (mm). Returns X, Y, mean
    residual of ||(A X)[:3,3] - (Y B)[:3,3]||.
    """
    if len(samples) < 3:
        raise ValueError("need at least 3 samples for hand–eye")
    r_gripper2base = []
    t_gripper2base = []
    r_target2cam = []
    t_target2cam = []
    for sample in samples:
        T_base_grip = np.asarray(sample["T_base_gripper"], dtype=np.float64)
        T_board_cam = np.asarray(sample["T_board_camera"], dtype=np.float64)
        T_cam_board = invert(T_board_cam)
        r_gripper2base.append(T_base_grip[:3, :3])
        t_gripper2base.append(T_base_grip[:3, 3].reshape(3, 1))
        r_target2cam.append(T_cam_board[:3, :3])
        t_target2cam.append(T_cam_board[:3, 3].reshape(3, 1))
    methods = [
        cv2.CALIB_HAND_EYE_TSAI,
        cv2.CALIB_HAND_EYE_PARK,
        cv2.CALIB_HAND_EYE_HORAUD,
        cv2.CALIB_HAND_EYE_DANIILIDIS,
    ]
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for method in methods:
        try:
            rotation, translation = cv2.calibrateHandEye(
                r_gripper2base,
                t_gripper2base,
                r_target2cam,
                t_target2cam,
                method=method,
            )
        except cv2.error:
            continue
        X_try = se3(
            np.asarray(rotation, dtype=np.float64),
            np.asarray(translation, dtype=np.float64).reshape(3),
        )
        Y_list = [
            np.asarray(s["T_base_gripper"], dtype=np.float64)
            @ X_try
            @ invert(np.asarray(s["T_board_camera"], dtype=np.float64))
            for s in samples
        ]
        Y_try = average_se3(Y_list)
        residuals = []
        for sample in samples:
            left = np.asarray(sample["T_base_gripper"], dtype=np.float64) @ X_try
            right = Y_try @ np.asarray(sample["T_board_camera"], dtype=np.float64)
            residuals.append(float(np.linalg.norm(left[:3, 3] - right[:3, 3])))
        residual = float(np.mean(residuals))
        if best is None or residual < best[2]:
            best = (X_try, Y_try, residual)
    if best is None or best[2] > 50.0:
        raise RuntimeError(
            "Hand–eye failed — use larger arm rotations between captures "
            f"(best residual {None if best is None else round(best[2], 2)} mm)."
        )
    return best


def write_board_in_base_file(
    T_base_board_mm: np.ndarray,
    *,
    sample_count: int,
    residual_mm: float,
) -> Path:
    ensure_output_dirs()
    payload = {
        "note": (
            "Diagnostic board-in-base from mount calibration. "
            "The click tool recomputes board-in-base from mount + live PnP."
        ),
        "T_base_board_mm": np.round(np.asarray(T_base_board_mm, dtype=np.float64), 6).tolist(),
        "sample_count": int(sample_count),
        "mean_residual_mm": round(float(residual_mm), 3),
    }
    path = MOUNT_DIR / "board_in_base_mm.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def rotation_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    r = np.asarray(rotation, dtype=np.float64)
    sy = math.sqrt(float(r[0, 0]) ** 2 + float(r[1, 0]) ** 2)
    if sy > 1e-6:
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        pitch = math.atan2(-float(r[2, 0]), sy)
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        roll = math.atan2(-float(r[1, 2]), float(r[1, 1]))
        pitch = math.atan2(-float(r[2, 0]), sy)
        yaw = 0.0
    return roll, pitch, yaw


def average_se3(poses: list[np.ndarray]) -> np.ndarray:
    if not poses:
        raise ValueError("No poses to average")
    translation = np.mean([pose[:3, 3] for pose in poses], axis=0)
    # Chordal mean via SVD of summed rotations.
    accumulated = np.zeros((3, 3), dtype=np.float64)
    for pose in poses:
        accumulated += pose[:3, :3]
    u, _, vt = np.linalg.svd(accumulated)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return se3(rotation, translation)


@dataclass
class MountState:
    kind: str  # calibrated | urdf_estimate
    T_gripper_camera_mm: np.ndarray | None
    urdf_path: Path
    urdf_source: str
    mount_path: Path | None = None


def load_mount(
    urdf_asset: ResolvedAsset,
    report: AssetReport | None = None,
    *,
    mount_yaml: Path | None = None,
    force_estimate: bool = False,
) -> MountState:
    if force_estimate:
        return MountState("urdf_estimate", None, urdf_asset.path, urdf_asset.source, None)
    yaml_path = Path(mount_yaml) if mount_yaml is not None else MOUNT_DIR / "wrist_camera_mount.yaml"
    if yaml_path.is_file():
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        matrix = np.asarray(data["T_gripper_camera_mm"], dtype=np.float64)
        if report is not None:
            report.add(ResolvedAsset("wrist_camera_mount.yaml", yaml_path, "local"))
        return MountState("calibrated", matrix, urdf_asset.path, urdf_asset.source, yaml_path)
    return MountState("urdf_estimate", None, urdf_asset.path, urdf_asset.source, None)


def T_base_camera_mm(joints_deg: list[float], mount: MountState) -> np.ndarray:
    if mount.kind == "calibrated" and mount.T_gripper_camera_mm is not None:
        T_base_gripper = fk_link_mm(mount.urdf_path, "gripper_frame_link", joints_deg)
        # Prefer gripper from stock kinematics; wrist-cam URDF still has gripper_frame_link.
        return T_base_gripper @ mount.T_gripper_camera_mm
    return fk_link_mm(mount.urdf_path, "wrist_camera_link", joints_deg)


def T_base_board_mm(T_board_wrist: np.ndarray, joints_deg: list[float], mount: MountState) -> np.ndarray:
    return T_base_camera_mm(joints_deg, mount) @ invert(T_board_wrist)


def point_board_to_base(point_board_mm: np.ndarray, T_base_board: np.ndarray) -> np.ndarray:
    hom = np.array([point_board_mm[0], point_board_mm[1], point_board_mm[2], 1.0], dtype=np.float64)
    return (T_base_board @ hom)[:3]


def algebraic_from_xy(x_mm: float, y_mm: float, square_mm: float = SQUARE_MM) -> str:
    file_i = int(math.floor(x_mm / square_mm))
    rank_i = int(math.floor(y_mm / square_mm))
    if not (0 <= file_i < 8 and 0 <= rank_i < 8):
        return "off"
    return "abcdefgh"[file_i] + str(rank_i + 1)


def acceptable_pose(pose: np.ndarray | None) -> bool:
    if pose is None or not np.isfinite(pose).all():
        return False
    return float(pose[2, 3]) > 0.0 and float(pose[2, 2]) < 0.0


def pose_from_square_corners(
    image_points: np.ndarray,
    square: str,
    camera_matrix: np.ndarray,
    square_mm: float = SQUARE_MM,
) -> np.ndarray | None:
    """Four corners of one square: index 0 is nearest a1, then +file, +file+rank, +rank."""
    file_i, rank_i = square_file_rank(square)
    object_points = np.array(
        [
            [file_i * square_mm, rank_i * square_mm, 0.0],
            [(file_i + 1) * square_mm, rank_i * square_mm, 0.0],
            [(file_i + 1) * square_mm, (rank_i + 1) * square_mm, 0.0],
            [file_i * square_mm, (rank_i + 1) * square_mm, 0.0],
        ],
        dtype=np.float64,
    )
    dist = np.zeros(5, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2),
        np.asarray(camera_matrix, dtype=np.float64),
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rotation = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    pose = invert(se3(rotation, np.asarray(tvec, dtype=np.float64).reshape(3)))
    if acceptable_pose(pose):
        return pose
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        object_points,
        np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2),
        np.asarray(camera_matrix, dtype=np.float64),
        dist,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok:
        return None
    for rv, tv in zip(rvecs, tvecs):
        rotation = cv2.Rodrigues(np.asarray(rv, dtype=np.float64))[0]
        candidate = invert(se3(rotation, np.asarray(tv, dtype=np.float64).reshape(3)))
        if acceptable_pose(candidate):
            return candidate
    return None


def detect_grid(image_bgr: np.ndarray) -> np.ndarray | None:
    """Return (rows, cols, 2) inner-grid corners or None."""
    detection = detect_largest_inner_grid(image_bgr)
    if detection is None:
        return None
    flat = np.asarray(detection["corners"], dtype=np.float64).reshape(-1, 2)
    rows = int(detection["inner_rows"])
    cols = int(detection["inner_cols"])
    if flat.shape[0] != rows * cols:
        return None
    return flat.reshape(rows, cols, 2)


def map_fisheye_grid_to_undistorted(grid: np.ndarray, calibration: dict) -> np.ndarray:
    """Map raw-fisheye corner pixels into the undistorted (K_undistorted) image."""
    mapped = cv2.fisheye.undistortPoints(
        np.asarray(grid, dtype=np.float64).reshape(-1, 1, 2),
        np.asarray(calibration["K"], dtype=np.float64),
        np.asarray(calibration["D"], dtype=np.float64).reshape(4, 1),
        P=np.asarray(calibration["K_undistorted"], dtype=np.float64),
    )
    return np.asarray(mapped, dtype=np.float64).reshape(grid.shape)


def detect_grid_for_view(
    image_bgr: np.ndarray,
    *,
    raw_bgr: np.ndarray | None = None,
    fisheye_calib: dict | None = None,
) -> tuple[np.ndarray | None, str]:
    """Detect inner corners in the click frame.

    Tries ``image_bgr`` first (undistorted wrist or stationary). If that fails and a
    raw fisheye frame + calibration are provided, detect on the raw image and map
    corners into the undistorted frame via ``fisheye.undistortPoints``.

    Returns ``(grid, source_tag)`` where source_tag is one of
    ``\"undistorted\"``, ``\"image\"``, ``\"raw→undistort\"``, ``\"none\"``.
    """
    grid = detect_grid(image_bgr)
    if grid is not None:
        return grid, ("undistorted" if raw_bgr is not None else "image")
    if raw_bgr is not None and fisheye_calib is not None:
        raw_grid = detect_grid(raw_bgr)
        if raw_grid is not None:
            return map_fisheye_grid_to_undistorted(raw_grid, fisheye_calib), "raw→undistort"
    return None, "none"


def snap_click_to_grid(
    grid: np.ndarray,
    pixel: np.ndarray | tuple[float, float],
    max_dist_px: float = 12.0,
) -> tuple[float, float] | None:
    """Snap a click to the nearest detected corner, or None if too far."""
    index = _snap_grid_index(grid, np.asarray(pixel, dtype=np.float64), max_dist_px=max_dist_px)
    if index is None:
        return None
    row, col = index
    pt = grid[row, col]
    return float(pt[0]), float(pt[1])


def resolve_label_click(
    grid: np.ndarray | None,
    pixel: np.ndarray | tuple[float, float],
    max_dist_px: float = 12.0,
) -> tuple[tuple[float, float] | None, str | None]:
    """Resolve a calib label click.

    If ``grid`` is set, snap within ``max_dist_px`` or return an error.
    If no grid, accept the raw pixel (last-resort fallback).
    Returns ``((x, y), None)`` on success, or ``(None, error_message)``.
    """
    px = (float(pixel[0]), float(pixel[1]))
    if grid is None:
        return px, None
    snapped = snap_click_to_grid(grid, px, max_dist_px=max_dist_px)
    if snapped is None:
        return None, "click nearer a detected corner"
    return snapped, None


def draw_detected_corners(
    image: np.ndarray,
    grid: np.ndarray | None,
    *,
    color: tuple[int, int, int] = (0, 200, 255),
    radius: int = 4,
) -> np.ndarray:
    """Draw a ring on every detected corner (mutates ``image``)."""
    if grid is None:
        return image
    for pt in np.asarray(grid, dtype=np.float64).reshape(-1, 2):
        cv2.circle(image, (int(round(pt[0])), int(round(pt[1]))), radius, color, 1, lineType=cv2.LINE_AA)
    return image


def try_auto_pose(image_bgr: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray | None:
    """Best-effort auto grid; returns None if indexing is ambiguous."""
    grid = detect_grid(image_bgr)
    if grid is None:
        return None
    n_rows, n_cols = grid.shape[:2]
    object_points = []
    image_points = []
    for row in range(n_rows):
        for col in range(n_cols):
            file_i = col + 1
            rank_i = n_rows - row
            if not (0 <= file_i <= 8 and 0 <= rank_i <= 8):
                continue
            object_points.append([file_i * SQUARE_MM, rank_i * SQUARE_MM, 0.0])
            image_points.append(grid[row, col])
    if len(object_points) < 6:
        return None
    dist = np.zeros(5, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        np.asarray(object_points, dtype=np.float64),
        np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2),
        np.asarray(camera_matrix, dtype=np.float64),
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rotation = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    pose = invert(se3(rotation, np.asarray(tvec, dtype=np.float64).reshape(3)))
    return pose if acceptable_pose(pose) else None


def parse_square_name(text: str) -> str | None:
    name = text.strip().lower()
    if len(name) != 2 or name[0] not in "abcdefgh" or name[1] not in "12345678":
        return None
    return name


def _snap_grid_index(grid: np.ndarray, pixel: np.ndarray, max_dist_px: float = 18.0) -> tuple[int, int] | None:
    flat = grid.reshape(-1, 2)
    deltas = flat - np.asarray(pixel, dtype=np.float64).reshape(1, 2)
    dist = np.linalg.norm(deltas, axis=1)
    index = int(np.argmin(dist))
    if float(dist[index]) > max_dist_px:
        return None
    return divmod(index, grid.shape[1])


def _cell_a1_corner_index(grid: np.ndarray, pixel: np.ndarray) -> tuple[int, int] | None:
    """If the click lies in a grid cell, return the image index of that cell's a1-near corner."""
    n_rows, n_cols = grid.shape[:2]
    px = np.asarray(pixel, dtype=np.float64).reshape(2)
    best = None
    best_area = float("inf")
    for row in range(n_rows - 1):
        for col in range(n_cols - 1):
            corners = np.array(
                [grid[row, col], grid[row, col + 1], grid[row + 1, col + 1], grid[row + 1, col]],
                dtype=np.float64,
            )
            # Point-in-quad via winding / ray on axis-aligned bbox first.
            mn = corners.min(axis=0)
            mx = corners.max(axis=0)
            if not (mn[0] - 2 <= px[0] <= mx[0] + 2 and mn[1] - 2 <= px[1] <= mx[1] + 2):
                continue
            # Choose the corner of this cell closest to image bottom-left as a1-near proxy,
            # then the caller remaps using the known square. Prefer smallest (x+y) in image.
            scores = corners[:, 0] + corners[:, 1]
            local = int(np.argmin(scores))
            local_rc = [(row, col), (row, col + 1), (row + 1, col + 1), (row + 1, col)][local]
            area = float((mx[0] - mn[0]) * (mx[1] - mn[1]))
            if area < best_area:
                best_area = area
                best = local_rc
    return best


def pose_from_grid_with_anchor(
    grid: np.ndarray,
    anchor_px: np.ndarray,
    square: str,
    camera_matrix: np.ndarray,
    square_mm: float = SQUARE_MM,
) -> np.ndarray | None:
    """Lock file/rank so the clicked anchor is the square's a1-near corner (or that cell)."""
    file_i, rank_i = square_file_rank(square)
    snapped = _snap_grid_index(grid, anchor_px, max_dist_px=18.0)
    if snapped is None:
        snapped = _cell_a1_corner_index(grid, anchor_px)
    if snapped is None:
        return None
    row_a, col_a = snapped
    n_rows, n_cols = grid.shape[:2]
    best_pose = None
    best_err = float("inf")
    dist = np.zeros(5, dtype=np.float64)
    for swap in (False, True):
        for s_row in (-1, 1):
            for s_col in (-1, 1):
                object_points = []
                image_points = []
                for row in range(n_rows):
                    for col in range(n_cols):
                        if swap:
                            file_line = file_i + (row - row_a) * s_row
                            rank_line = rank_i + (col - col_a) * s_col
                        else:
                            file_line = file_i + (col - col_a) * s_col
                            rank_line = rank_i + (row - row_a) * s_row
                        if not (0 <= file_line <= 8 and 0 <= rank_line <= 8):
                            continue
                        object_points.append([file_line * square_mm, rank_line * square_mm, 0.0])
                        image_points.append(grid[row, col])
                if len(object_points) < 6:
                    continue
                obj = np.asarray(object_points, dtype=np.float64)
                img = np.asarray(image_points, dtype=np.float64)
                ok, rvec, tvec = cv2.solvePnP(
                    obj,
                    img.reshape(-1, 1, 2),
                    np.asarray(camera_matrix, dtype=np.float64),
                    dist,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if not ok:
                    continue
                rotation = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
                pose = invert(se3(rotation, np.asarray(tvec, dtype=np.float64).reshape(3)))
                if not acceptable_pose(pose):
                    continue
                projected, _ = project_points(obj, pose, camera_matrix)
                err = float(np.linalg.norm(projected - img, axis=1).mean())
                if err < best_err:
                    best_err = err
                    best_pose = pose
    return best_pose


def ray_ray_midpoint(
    origin_a: np.ndarray,
    direction_a: np.ndarray,
    origin_b: np.ndarray,
    direction_b: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Closest-approach midpoint of two rays and the gap distance."""
    a = np.asarray(origin_a, dtype=np.float64).reshape(3)
    b = np.asarray(origin_b, dtype=np.float64).reshape(3)
    da = np.asarray(direction_a, dtype=np.float64).reshape(3)
    db = np.asarray(direction_b, dtype=np.float64).reshape(3)
    da = da / (np.linalg.norm(da) + 1e-12)
    db = db / (np.linalg.norm(db) + 1e-12)
    cross = np.cross(da, db)
    denom = float(np.dot(cross, cross))
    if denom < 1e-12:
        # Parallel: project b onto a.
        t = float(np.dot(b - a, da))
        pa = a + t * da
        gap = float(np.linalg.norm(b - pa))
        return pa, gap
    matrix = np.array(
        [
            [float(np.dot(da, da)), -float(np.dot(da, db))],
            [float(np.dot(da, db)), -float(np.dot(db, db))],
        ],
        dtype=np.float64,
    )
    rhs = np.array([float(np.dot(b - a, da)), float(np.dot(b - a, db))], dtype=np.float64)
    try:
        ta, tb = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        ta = float(np.dot(b - a, da))
        tb = 0.0
    pa = a + ta * da
    pb = b + tb * db
    mid = 0.5 * (pa + pb)
    gap = float(np.linalg.norm(pa - pb))
    return mid, gap


def project_ray_polyline(
    origin_board: np.ndarray,
    direction_board: np.ndarray,
    T_board_camera: np.ndarray,
    camera_matrix: np.ndarray,
    travel_mm: tuple[float, float] = (40.0, 700.0),
    samples: int = 80,
) -> dict[str, np.ndarray]:
    """Project a board-frame ray into image pixels, split at the board plane.

    Returns pixels for the full ray (snapping), the part at/above the board
    (z >= 0), and the part below (z < 0).
    """
    direction = np.asarray(direction_board, dtype=np.float64).reshape(3)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    origin = np.asarray(origin_board, dtype=np.float64).reshape(3)
    travels = np.linspace(travel_mm[0], travel_mm[1], samples)
    points = origin.reshape(1, 3) + travels.reshape(-1, 1) * direction.reshape(1, 3)
    # Include the exact board crossing when the ray hits z = 0 in range.
    denom = float(direction[2])
    if abs(denom) > 1e-9:
        t_plane = -float(origin[2]) / denom
        if travel_mm[0] < t_plane < travel_mm[1]:
            points = np.vstack([points, (origin + t_plane * direction).reshape(1, 3)])
            order = np.argsort(np.linalg.norm(points - origin.reshape(1, 3), axis=1))
            points = points[order]
    pixels, depths = project_points(points, T_board_camera, camera_matrix)
    keep = depths > 1.0
    pixels = pixels[keep]
    z_vals = points[keep, 2]
    above = pixels[z_vals >= 0.0]
    below = pixels[z_vals < 0.0]
    return {"all": pixels, "above": above, "below": below}


def snap_to_polyline(pixel: np.ndarray, polyline: np.ndarray | dict, max_dist_px: float = 12.0) -> np.ndarray | None:
    if isinstance(polyline, dict):
        polyline = polyline.get("all")
    if polyline is None or len(polyline) == 0:
        return None
    pixel = np.asarray(pixel, dtype=np.float64).reshape(2)
    deltas = polyline - pixel.reshape(1, 2)
    dist = np.linalg.norm(deltas, axis=1)
    index = int(np.argmin(dist))
    if float(dist[index]) > max_dist_px:
        return None
    return polyline[index].copy()


def snap_ray_click(
    pixel: np.ndarray,
    ray: dict[str, np.ndarray],
    max_dist_px: float = 12.0,
) -> tuple[np.ndarray | None, str | None]:
    """Snap to the ray. Prefer the above-board segment. Reject below-board hits.

    Returns (snapped_pixel, error_message). error_message is set on failure.
    """
    pixel = np.asarray(pixel, dtype=np.float64).reshape(2)
    above = ray.get("above")
    below = ray.get("below")
    all_pts = ray.get("all")
    if above is not None and len(above) > 0:
        deltas = above - pixel.reshape(1, 2)
        dist = np.linalg.norm(deltas, axis=1)
        index = int(np.argmin(dist))
        if float(dist[index]) <= max_dist_px:
            return above[index].copy(), None
    if below is not None and len(below) > 0:
        deltas = below - pixel.reshape(1, 2)
        dist = np.linalg.norm(deltas, axis=1)
        index = int(np.argmin(dist))
        if float(dist[index]) <= max_dist_px:
            return None, "point is beneath the board — click again on the line"
    if all_pts is not None and len(all_pts) > 0:
        snapped = snap_to_polyline(pixel, all_pts, max_dist_px)
        if snapped is None:
            return None, "click closer to the line"
        return snapped, None
    return None, "click closer to the line"


def _draw_dotted_polyline(
    image: np.ndarray,
    pixels: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_px: float = 8.0,
    gap_px: float = 6.0,
) -> None:
    if pixels is None or len(pixels) < 2:
        return
    carry = 0.0
    drawing = True
    for start, end in zip(pixels[:-1], pixels[1:]):
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length < 1e-6:
            continue
        unit = segment / length
        walked = 0.0
        cursor = start.astype(np.float64)
        while walked < length:
            step = (dash_px if drawing else gap_px) - carry
            step = min(step, length - walked)
            nxt = cursor + unit * step
            if drawing:
                cv2.line(
                    image,
                    (int(round(cursor[0])), int(round(cursor[1]))),
                    (int(round(nxt[0])), int(round(nxt[1]))),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            walked += step
            cursor = nxt
            if step >= ((dash_px if drawing else gap_px) - carry) - 1e-9:
                drawing = not drawing
                carry = 0.0
            else:
                carry += step


def draw_ray_polyline(
    image: np.ndarray,
    ray: dict[str, np.ndarray] | np.ndarray | None,
    color: tuple[int, int, int] = (0, 140, 255),
    thickness: int = 2,
    below_alpha: float = 0.35,
) -> None:
    """Solid above/on the board; dotted and faded beneath the board."""
    if ray is None:
        return
    if isinstance(ray, np.ndarray):
        above = ray
        below = np.zeros((0, 2), dtype=np.float64)
    else:
        above = ray.get("above", np.zeros((0, 2)))
        below = ray.get("below", np.zeros((0, 2)))
    if above is not None and len(above) >= 2:
        cv2.polylines(image, [above.astype(np.int32).reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)
    if below is not None and len(below) >= 2:
        overlay = image.copy()
        _draw_dotted_polyline(overlay, below, color, thickness=thickness)
        cv2.addWeighted(overlay, below_alpha, image, 1.0 - below_alpha, 0.0, image)


def draw_backup_banner(panel: np.ndarray, text: str | None) -> np.ndarray:
    if not text:
        return panel
    canvas = panel.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 36), (0, 220, 255), -1)
    cv2.putText(canvas, text[:90], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def _wrap_line(text: str, max_chars: int) -> list[str]:
    text = str(text)
    if len(text) <= max_chars:
        return [text]
    words = text.split(" ")
    rows: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if len(trial) <= max_chars:
            current = trial
            continue
        if current:
            rows.append(current)
        if len(word) <= max_chars:
            current = word
        else:
            while len(word) > max_chars:
                rows.append(word[:max_chars])
                word = word[max_chars:]
            current = word
    if current:
        rows.append(current)
    return rows or [""]


def panel_text(lines: list[str], width: int = 480, height: int = 480) -> np.ndarray:
    """Draw side-panel text with word wrap so instructions stay readable."""
    canvas = np.full((height, width, 3), 32, dtype=np.uint8)
    max_chars = max(20, (width - 24) // 9)
    y = 28
    for line in lines:
        for row in _wrap_line(line, max_chars):
            cv2.putText(canvas, row, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
            y += 20
            if y > height - 12:
                return canvas
    return canvas


def list_asset_candidates() -> dict[str, list[Path]]:
    """Discover YAML/URDF choices for the in-app asset menu."""
    coord = ROOT.parent
    fisheye = []
    for path in [
        BUNDLE / "wrist_fisheye.yaml",
        coord / "output" / "lens_calib" / "wrist_fisheye.yaml",
        MOUNT_DIR / "wrist_fisheye.yaml",
    ]:
        if path.is_file() and path not in fisheye:
            fisheye.append(path)
    stationary = []
    for path in [
        BUNDLE / "stationary_intrinsics.yaml",
        coord / "output" / "lens_calib" / "stationary_intrinsics.yaml",
        MOUNT_DIR / "stationary_intrinsics.yaml",
    ]:
        if path.is_file() and path not in stationary:
            stationary.append(path)
    urdfs = []
    for path in [
        BUNDLE / "so101_new_calib_wrist_cam.urdf",
        MOUNT_DIR / "so101_new_calib_wrist_cam_calibrated.urdf",
        ROOT.parent.parent
        / "chessbot"
        / "LeRobot_OXAI"
        / "SO-ARM100"
        / "Simulation"
        / "SO101"
        / "so101_new_calib_wrist_cam.urdf",
    ]:
        if path.is_file() and path not in urdfs:
            urdfs.append(path)
    mounts: list[Path] = []
    none_marker = MOUNT_DIR / "__none__"
    mounts.append(none_marker)
    calibrated = MOUNT_DIR / "wrist_camera_mount.yaml"
    if calibrated.is_file():
        mounts.append(calibrated)
    for path in sorted(MOUNT_DIR.glob("*.yaml")):
        if path not in mounts:
            mounts.append(path)
    return {
        "fisheye": fisheye,
        "stationary": stationary,
        "urdf": urdfs,
        "mount": mounts,
    }


def append_session_log(record: dict[str, Any]) -> None:
    ensure_output_dirs()
    path = SESSIONS_DIR / "session_log.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load_demo_board_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "wrist" in data and "T_board_camera" in data["wrist"]:
        wrist = np.asarray(data["wrist"]["T_board_camera"], dtype=np.float64)
        side = np.asarray(data["stationary"]["T_board_camera"], dtype=np.float64)
        return wrist, side
    raise RuntimeError(f"No board poses in {path}")


def write_mount_files(
    T_gripper_camera_mm: np.ndarray,
    *,
    sample_count: int,
    residual_mm: float,
    source_urdf: Path,
) -> tuple[Path, Path]:
    ensure_output_dirs()
    xyz = T_gripper_camera_mm[:3, 3] / 1000.0
    rpy = rotation_to_rpy(T_gripper_camera_mm[:3, :3])
    payload = {
        "parent_link": "gripper_frame_link",
        "child_link": "wrist_camera_link",
        "note": "T maps camera points into gripper_frame_link (FK target used during calibration).",
        "T_gripper_camera_mm": np.round(T_gripper_camera_mm, 6).tolist(),
        "origin_xyz_m": [round(float(v), 6) for v in xyz],
        "origin_rpy_rad": [round(float(v), 6) for v in rpy],
        "sample_count": int(sample_count),
        "mean_residual_mm": round(float(residual_mm), 3),
        "source_urdf": str(source_urdf),
    }
    yaml_path = MOUNT_DIR / "wrist_camera_mount.yaml"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    urdf_text = source_urdf.read_text(encoding="utf-8")
    old = None
    for line in urdf_text.splitlines():
        if "wrist_camera_joint" in line:
            break
    # Replace the origin line of wrist_camera_joint.
    lines = urdf_text.splitlines(keepends=True)
    out_lines = []
    in_joint = False
    replaced = False
    for line in lines:
        if 'name="wrist_camera_joint"' in line:
            in_joint = True
        if in_joint and "<origin" in line and not replaced:
            out_lines.append(
                f'    <origin xyz="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" '
                f'rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>\n'
            )
            replaced = True
            if "</joint>" in line:
                in_joint = False
            continue
        if in_joint and "</joint>" in line:
            in_joint = False
        out_lines.append(line)
    if not replaced:
        raise RuntimeError("Could not find wrist_camera_joint origin in URDF")
    urdf_out = MOUNT_DIR / "so101_new_calib_wrist_cam_calibrated.urdf"
    urdf_out.write_text("".join(out_lines), encoding="utf-8")
    return yaml_path, urdf_out


def undistort_wrist(image_bgr: np.ndarray, fisheye_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    calibration = load_fisheye_calibration(fisheye_path)
    balance = float(calibration.get("balance", 0.0))
    undistorted = undistort_fisheye(image_bgr, calibration, balance=balance)
    k = np.asarray(calibration["K_undistorted"], dtype=np.float64)
    return undistorted, k, calibration
