#!/usr/bin/env python3
"""Shared helpers for click-to-3D and mount calibration.

Assets resolve CLI path → local project path → bundled backup.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import subprocess
import sys
import threading
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
from board_roi import (  # noqa: E402
    board_aabb,
    find_board_gate_quad,
    pad_quad,
    point_in_quad,
)
from calib_intrinsics import (  # noqa: E402
    detect_largest_inner_grid,
    load_fisheye_calibration,
    undistort_fisheye,
)

# Same as the sim and lookup tables (board_coordinates.DEFAULT_SQUARE_SIZE, 33.0 cm across 8);
# 42.0 mm (Oliver tape, 33.6 cm) before 2026-11-23.
SQUARE_MM = 41.25
# Board in the robot base, ruler-measured (Oliver's project notes, "Robot origin"): centre at
# (29.26, 0) cm from the robot origin, edges along the base x / y axes. The board is 2.9 cm
# thick and sits on the table the robot stands on, so the playing surface is at z = 29 mm.
BOARD_CENTRE_BASE_MM = (292.6, 0.0, 29.0)
BOARD_BASE_SOURCE = "board_centre_ruler"
# The alternative (b in the click tool): the camera-detected board, "cv": the board in the base
# from the wrist camera's c pose, forward kinematics and the camera mount (T_base_board_mm), as
# Asha's version does. Clicks logged before 2026-11-13 call it "camera_fk" (base_mm_fk).
CV_BASE_SOURCE = "camera_cv"
LEGACY_FK_BASE_SOURCE = "camera_fk"
SESSION_CONFIG = OUTPUT / "session_config.yaml"
OXAI_PATHS_HINT = (
    "Place this folder directly in LeRobot_OXAI/lowlevel so the parent "
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

    Prefer the drop-in layout LeRobot_OXAI/lowlevel/<this folder> (parent has paths.py).
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
    # Default: OpenCV full-point fit. Alternate: wrist_fisheye_status_quo.yaml
    # (dense-grid interim). Pick either via the asset menu (m).
    fisheye = resolve_asset(
        "wrist_fisheye.yaml",
        cli_path=fisheye_cli,
        local_candidates=[
            MOUNT_DIR / "wrist_fisheye.yaml",
            MOUNT_DIR / "wrist_fisheye_status_quo.yaml",
            coord / "output" / "lens_calib" / "wrist_fisheye.yaml",
        ],
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


V4L_BY_ID = Path("/dev/v4l/by-id")


def parse_camera(value: str) -> int | str:
    """CLI camera: an index ("2") or a device path ("/dev/video2" or a /dev/v4l/by-id/ link)."""
    return int(value) if value.isdigit() else value


def camera_by_id(camera: int | str) -> str:
    """The camera's /dev/v4l/by-id/ link, which names the physical camera and survives
    replugs; the plain /dev/videoN path if it has none."""
    device = Path(f"/dev/video{camera}" if isinstance(camera, int) else camera)
    if V4L_BY_ID.is_dir():
        for link in sorted(V4L_BY_ID.iterdir()):
            if link.resolve() == device.resolve():
                return str(link)
    return str(device)


def camera_label(camera: str) -> str:
    """Short name for the status line, e.g. video2."""
    return Path(camera).resolve().name


def _paths_setting(name: str) -> Any:
    """A camera setting from lowlevel/paths.py, or None if it isn't there."""
    if _ensure_lowlevel_on_path() is None:
        return None
    try:
        import paths
    except Exception:
        return None
    return getattr(paths, name, None)


def open_camera(camera: str, role: str, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    """Open a camera by device path and apply paths.py's v4l2 controls for its role
    ("wrist" or "stationary"), e.g. the Innomaker's gain=0."""
    device = str(Path(camera).resolve())
    # By index: the pip opencv-python build can't open V4L2 devices by path.
    cap = cv2.VideoCapture(int(device.removeprefix("/dev/video")), cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {role} camera {camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    controls = _paths_setting(f"{role.upper()}_CAMERA_CONTROLS") or ""
    # One call per control, in order: the C920 rejects a batch, because focus_absolute
    # is read-only until autofocus is off (see capture_camera_pair.py).
    for control in filter(None, controls.split(",")):
        result = subprocess.run(["v4l2-ctl", "-d", device, "-c", control], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"warning: {role} camera {device}: v4l2-ctl -c {control} failed: {result.stderr.strip()}")
    return cap


JPEG_WARNING = b"Corrupt JPEG data"


def hide_jpeg_warnings() -> None:
    """Drop libjpeg's "Corrupt JPEG data: ..." lines from the terminal; everything else on stderr passes.

    The cameras pad their MJPEG frames, and libjpeg prints that line from C for nearly every frame
    it still decodes fine, so Python's warning filters can't reach it. This sends file descriptor
    2 (this process's and its subprocesses') through a pipe and forwards all other output.
    """
    terminal = os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)
    out = os.dup(terminal)

    def forward() -> None:
        pending = b""
        while chunk := os.read(read_fd, 65536):
            *lines, pending = (pending + chunk).split(b"\n")
            text = b"".join(line + b"\n" for line in lines if JPEG_WARNING not in line)
            # Hold back a partial line only while it could still become the warning, so
            # prompts without a newline show up straight away.
            if not JPEG_WARNING.startswith(pending[: len(JPEG_WARNING)]):
                text, pending = text + pending, b""
            if text:
                os.write(out, text)
        if pending and JPEG_WARNING not in pending:
            os.write(out, pending)

    thread = threading.Thread(target=forward, daemon=True)
    thread.start()

    def restore() -> None:
        # At exit (after any traceback is written): put the terminal back, which closes the
        # pipe, and let the forwarder print what is left.
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(terminal, 2)
        thread.join(timeout=2.0)

    atexit.register(restore)


def _ensure_lowlevel_on_path() -> Path | None:
    for lowlevel in lowlevel_candidates():
        if lowlevel.is_dir() and (lowlevel / "paths.py").is_file():
            if str(lowlevel) not in sys.path:
                sys.path.insert(0, str(lowlevel))
            return lowlevel
    return None


_FOLLOWER = None
_ARM_CONFIGURED = False


def _follower():
    """The SO-101 connection, opened once and kept until close_arm() or exit.

    Opens the serial port only: lerobot's full connect() turns torque off on every motor for a
    moment (configure()), so reconnecting for each reading let the arm drop. Torque is left as
    it is until set_arm_hold() changes it.
    """
    global _FOLLOWER
    if _FOLLOWER is None:
        if _ensure_lowlevel_on_path() is None:
            raise RuntimeError(f"Robot API not available. {OXAI_PATHS_HINT}")
        try:
            from paths import connect_so101_hardware
        except Exception as exc:  # pragma: no cover - hardware optional
            raise RuntimeError(f"Robot API not available. {OXAI_PATHS_HINT}") from exc
        _FOLLOWER = connect_so101_hardware(calibrate=False, configure=False)
        atexit.register(close_arm)
    return _FOLLOWER


def close_arm() -> None:
    """Close the serial port without touching torque (a held arm stays held)."""
    global _FOLLOWER
    if _FOLLOWER is not None:
        _FOLLOWER.bus.disconnect(disable_torque=False)
        _FOLLOWER = None


def arm_held() -> bool:
    """True if any arm motor has torque on."""
    return any(bool(v) for v in _follower().bus.sync_read("Torque_Enable").values())


def set_arm_hold(hold: bool) -> None:
    """Hold the arm where it is now (torque on), or release it (torque off: it falls unless supported).

    Holding first sets every Goal_Position to the present position, so the arm doesn't jump to
    an old target. The first hold also applies lerobot's servo settings (P=16, gripper torque
    limits, ...) via configure(); that turns torque off while writing, which is harmless here
    because the arm is limp until this hold anyway.
    """
    global _ARM_CONFIGURED
    follower = _follower()
    if not hold:
        follower.bus.disable_torque()
        return
    follower.bus.sync_write("Goal_Position", follower.bus.sync_read("Present_Position"))
    if _ARM_CONFIGURED:
        follower.bus.enable_torque()
    else:
        follower.configure()  # ends with torque on
        _ARM_CONFIGURED = True


def toggle_arm_hold() -> bool:
    """Flip between held and released; returns True if the arm is now held."""
    hold = not arm_held()
    set_arm_hold(hold)
    return hold


def arm_state_text(held: bool) -> str:
    return "HELD (torque on)" if held else "RELEASED (torque off) - support it"


def read_joints_robot() -> list[float]:
    obs = _follower().get_observation()
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


_JOINT_OFFSETS_DEG: np.ndarray | None = None


def joint_offsets_deg() -> np.ndarray:
    """Calibration offsets in JOINT_NAMES order: URDF angle = motor reading + offset.

    Read from the friction/sag fit (lowlevel/real_world_runs/friction/friction_sag_fit.json),
    the same file run_real_so101_from_xy_lookup.py --compensate uses. Zeros if it is missing.
    Without them FK puts the wrist camera ~9 deg off and clicks land ~5 cm off in base_mm.
    """
    global _JOINT_OFFSETS_DEG
    if _JOINT_OFFSETS_DEG is None:
        fit_path = None
        if _ensure_lowlevel_on_path() is not None:
            try:
                import paths

                fit_path = paths.REAL_WORLD_RUNS_DIR / "friction" / "friction_sag_fit.json"
            except Exception:
                pass
        if fit_path is not None and fit_path.is_file():
            offsets = json.loads(fit_path.read_text(encoding="utf-8"))["offsets_deg"]
            _JOINT_OFFSETS_DEG = np.array([float(offsets.get(name, 0.0)) for name in JOINT_NAMES])
            print(f"FK joint offsets from {fit_path}:")
        else:
            _JOINT_OFFSETS_DEG = np.zeros(len(JOINT_NAMES))
            print(f"FK joint offsets: none (no fit file at {fit_path}); using raw readings")
        print("  " + ", ".join(f"{n} {o:+.2f}" for n, o in zip(JOINT_NAMES, _JOINT_OFFSETS_DEG)))
    return _JOINT_OFFSETS_DEG


def fk_link_mm(urdf_path: Path, target_frame: str, reading_deg: list[float]) -> np.ndarray:
    """Pose of target_frame in the base (mm) for a motor reading; adds joint_offsets_deg() first."""
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
    joints_deg = np.asarray(reading_deg, dtype=np.float64) + joint_offsets_deg()
    pose_m = np.asarray(kinematics.forward_kinematics(joints_deg), dtype=np.float64)
    pose_mm = pose_m.copy()
    pose_mm[:3, 3] *= 1000.0
    return pose_mm


def load_session_cameras(
    wrist_cli: int | str | None,
    stationary_cli: int | str | None,
) -> tuple[str, str]:
    """Camera device paths, first found of: CLI; output/session_config.yaml; paths.py's
    WRIST_CAMERA / STATIONARY_CAMERA; /dev/video0 and /dev/video1.

    Saved and default cameras are /dev/v4l/by-id/ links, so the choice survives replugs
    (plain indices shift: each camera also takes a metadata /dev/videoN).
    """
    data = {}
    if SESSION_CONFIG.is_file():
        data = yaml.safe_load(SESSION_CONFIG.read_text(encoding="utf-8")) or {}

    def pick(cli: int | str | None, role: str, fallback: int) -> str:
        if cli is not None:
            return camera_by_id(cli)
        for candidate in (data.get(f"{role}_camera"), _paths_setting(f"{role.upper()}_CAMERA")):
            if candidate and Path(candidate).exists():
                return str(candidate)
        return camera_by_id(fallback)

    return pick(wrist_cli, "wrist", 0), pick(stationary_cli, "stationary", 1)


def save_session_cameras(wrist_camera: str, stationary_camera: str) -> None:
    ensure_output_dirs()
    payload = {
        "wrist_camera": camera_by_id(wrist_camera),
        "stationary_camera": camera_by_id(stationary_camera),
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
        raise ValueError("need at least 3 samples for hand-eye")
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
            "Hand-eye failed - use larger arm rotations between captures "
            f"(best residual {None if best is None else round(best[2], 2)} mm)."
        )
    return best


@dataclass
class MountFit:
    """fit_mount's result: the mount, one board pose per group, and how well they explain the clicks."""

    T_gripper_camera_mm: np.ndarray
    T_base_board_mm: list[np.ndarray]
    rms_px: float
    sample_rms_px: np.ndarray
    mean_residual_mm: float
    # 1-sigma uncertainty of the mount, about / along the camera's own x, y, z axes.
    rotation_std_deg: np.ndarray
    translation_std_mm: np.ndarray
    start: str


def _se3_from_params(params: np.ndarray) -> np.ndarray:
    rotation = cv2.Rodrigues(np.asarray(params[:3], dtype=np.float64).reshape(3, 1))[0]
    return se3(rotation, np.asarray(params[3:6], dtype=np.float64))


def _refine_mount(
    samples: list[dict],
    groups: list[int],
    camera_matrix: np.ndarray,
    X0: np.ndarray,
    Y0: list[np.ndarray],
    start: str,
) -> MountFit:
    from scipy.optimize import least_squares

    K = np.asarray(camera_matrix, dtype=np.float64)
    A = [np.asarray(s["T_base_gripper"], dtype=np.float64) for s in samples]
    B = [np.asarray(s["T_board_camera"], dtype=np.float64) for s in samples]
    corners_board = [square_corner_points_board(s["square"]) for s in samples]
    seen = [np.asarray(s["corners_px"], dtype=np.float64).reshape(4, 2) for s in samples]
    n_groups = len(Y0)

    # Each pose is the start pose times a small correction, so the parameters stay near zero
    # and the mount's correction comes out about the camera's own axes.
    def unpack(params: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        X = X0 @ _se3_from_params(params[:6])
        Ys = [Y0[g] @ _se3_from_params(params[6 + 6 * g : 12 + 6 * g]) for g in range(n_groups)]
        return X, Ys

    def residuals(params: np.ndarray) -> np.ndarray:
        X, Ys = unpack(params)
        out = []
        for a, g, pts, obs in zip(A, groups, corners_board, seen):
            T_camera_board = invert(a @ X) @ Ys[g]
            cam = pts @ T_camera_board[:3, :3].T + T_camera_board[:3, 3]
            uv = cam[:, :2] / cam[:, 2:3]
            out.append((uv @ K[:2, :2].T + K[:2, 2] - obs).ravel())
        return np.concatenate(out)

    result = least_squares(residuals, np.zeros(6 + 6 * n_groups), x_scale="jac", method="trf")
    X, Ys = unpack(result.x)
    point_err = np.sqrt(np.sum(result.fun.reshape(-1, 4, 2) ** 2, axis=2))
    sample_rms = np.sqrt(np.mean(point_err**2, axis=1))
    # Parameter covariance: residual variance times (J^T J)^-1 (Hartley & Zisserman, ch. 5).
    dof = result.fun.size - result.x.size
    sigma2 = float(result.fun @ result.fun) / dof if dof > 0 else float("nan")
    cov = sigma2 * np.linalg.pinv(result.jac.T @ result.jac)
    std = np.sqrt(np.clip(np.diag(cov)[:6], 0.0, None))
    pose_gap = [float(np.linalg.norm((a @ X)[:3, 3] - (Ys[g] @ b)[:3, 3])) for a, b, g in zip(A, B, groups)]
    return MountFit(
        T_gripper_camera_mm=X,
        T_base_board_mm=Ys,
        rms_px=float(np.sqrt(np.mean(point_err**2))),
        sample_rms_px=sample_rms,
        mean_residual_mm=float(np.mean(pose_gap)),
        rotation_std_deg=np.degrees(std[:3]),
        translation_std_mm=std[3:],
        start=start,
    )


def fit_mount(
    samples: list[dict],
    groups: list[int],
    camera_matrix: np.ndarray,
    X_urdf: np.ndarray,
) -> MountFit:
    """Fit T_gripper_camera to every clicked corner at once.

    samples need T_base_gripper (FK, mm), T_board_camera (that sample's 4-corner pose),
    corners_px (undistorted pixels, click order) and square. groups[i] numbers the board pose
    sample i saw, from 0: all 0 when the board stayed put, one per run when it was moved.

    It minimises the corners' reprojection error, as in Koide & Menegatti (2019), "General
    hand-eye calibration based on reprojection error minimization", rather than the AX = YB
    pose residual, so each sample counts by what its pixels pin down, not by a 4-corner pose
    whose tilt and depth are only loosely fixed. Starts from OpenCV hand-eye (one board only)
    and from the URDF estimate, and keeps the better fit.
    """
    n_groups = int(max(groups)) + 1
    starts: list[tuple[str, np.ndarray]] = [("urdf_estimate", np.asarray(X_urdf, dtype=np.float64))]
    if n_groups == 1 and len(samples) >= 3:
        try:
            starts.insert(0, ("hand_eye", solve_hand_eye_mount(samples)[0]))
        except (RuntimeError, cv2.error):
            pass
    best: MountFit | None = None
    for name, X0 in starts:
        Y0 = [
            average_se3(
                [
                    np.asarray(s["T_base_gripper"], dtype=np.float64)
                    @ X0
                    @ invert(np.asarray(s["T_board_camera"], dtype=np.float64))
                    for s, g in zip(samples, groups)
                    if g == group
                ]
            )
            for group in range(n_groups)
        ]
        fit = _refine_mount(samples, groups, camera_matrix, X0, Y0, name)
        if best is None or fit.rms_px < best.rms_px:
            best = fit
    assert best is not None
    return best


def write_board_in_base_file(
    T_base_board_mm: np.ndarray,
    *,
    sample_count: int,
    residual_mm: float,
    extra: dict | None = None,
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
        **(extra or {}),
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


def board_to_base_mm(point_board_mm: np.ndarray) -> np.ndarray:
    """Board mm (origin a1 corner, +x toward h, +y toward rank 8, +z up) to robot base mm.

    The board's axes are the base's (a1 nearest the robot on its right, files running away
    from it), so this is a shift: the a1 corner is 4 squares from the centre on each axis.
    Uses the ruler-measured BOARD_CENTRE_BASE_MM, not forward kinematics or the mount.
    """
    point = np.asarray(point_board_mm, dtype=np.float64)
    centre_x, centre_y, surface_z = BOARD_CENTRE_BASE_MM
    half = 4.0 * SQUARE_MM
    return np.array([centre_x - half + point[0], centre_y - half + point[1], surface_z + point[2]])


def point_board_to_base(point_board_mm: np.ndarray, T_base_board: np.ndarray) -> np.ndarray:
    hom = np.array([point_board_mm[0], point_board_mm[1], point_board_mm[2], 1.0], dtype=np.float64)
    return (T_base_board @ hom)[:3]


def algebraic_from_xy(x_mm: float, y_mm: float, square_mm: float = SQUARE_MM) -> str:
    file_i = int(math.floor(x_mm / square_mm))
    rank_i = int(math.floor(y_mm / square_mm))
    if not (0 <= file_i < 8 and 0 <= rank_i < 8):
        return "off"
    return "abcdefgh"[file_i] + str(rank_i + 1)


def acceptable_pose(pose: np.ndarray | None, *, upright: bool = False) -> bool:
    """Camera above the board, looking down. ``upright`` (tripod only) also needs image-down
    (the camera's y axis, column 1) to point down at the table, i.e. roll within +-90 deg.

    One square's four corners fit two poses almost equally well (planar ambiguity): the real
    one and one on the far side of the board rolled 180 deg. From a diagonal tripod view on
    2026-11-24 they differed by 0.8 px and the rolled one was kept. The wrist camera can be
    rolled any way, so it doesn't get this check.
    """
    if pose is None or not np.isfinite(pose).all():
        return False
    if not (float(pose[2, 3]) > 0.0 and float(pose[2, 2]) < 0.0):
        return False
    return not upright or float(pose[2, 1]) < 0.0


def square_corner_points_board(square: str, square_mm: float = SQUARE_MM) -> np.ndarray:
    """Board-frame corners (mm, z = 0) of one square in click order: nearest a1, +file, +file+rank, +rank."""
    file_i, rank_i = square_file_rank(square)
    return np.array(
        [
            [file_i * square_mm, rank_i * square_mm, 0.0],
            [(file_i + 1) * square_mm, rank_i * square_mm, 0.0],
            [(file_i + 1) * square_mm, (rank_i + 1) * square_mm, 0.0],
            [file_i * square_mm, (rank_i + 1) * square_mm, 0.0],
        ],
        dtype=np.float64,
    )


def pose_from_square_corners(
    image_points: np.ndarray,
    square: str,
    camera_matrix: np.ndarray,
    square_mm: float = SQUARE_MM,
    *,
    upright: bool = False,
) -> np.ndarray | None:
    """Four corners of one square: index 0 is nearest a1, then +file, +file+rank, +rank.

    ``upright=True`` (tripod) rejects poses with the camera rolled past 90 deg; see
    ``acceptable_pose``. If the first solve gives one, IPPE's other solution is tried.
    """
    object_points = square_corner_points_board(square, square_mm)
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
    if acceptable_pose(pose, upright=upright):
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
        if acceptable_pose(candidate, upright=upright):
            return candidate
    return None


def detect_grid(
    image_bgr: np.ndarray,
    *,
    restrict_to_board: bool = False,
) -> np.ndarray | None:
    """Return (rows, cols, 2) inner-grid corners or None.

    When ``restrict_to_board`` is True (stationary view only):
      1. Find outer board quad (LiveChess2FEN, contour fallback).
      2. Pad that quad (~9% from centroid) — the accept/reject gate.
      3. Pass a rectangular crop (AABB of the padded quad) to OpenCV.
      4. After detection, keep only corners inside the padded quad.

    Wrist / overhead stays ungated (full-frame search, no accept/reject).
    """
    offset = np.array([0.0, 0.0], dtype=np.float64)
    search = image_bgr
    padded_gate: np.ndarray | None = None
    if restrict_to_board:
        quad, _method = find_board_gate_quad(image_bgr)
        if quad is not None:
            padded_gate = pad_quad(quad)
            x0, y0, x1, y1 = board_aabb(padded_gate, image_bgr.shape)
            search = image_bgr[y0:y1, x0:x1]
            offset = np.array([float(x0), float(y0)], dtype=np.float64)
            if search.size == 0:
                search = image_bgr
                offset = np.array([0.0, 0.0], dtype=np.float64)
                padded_gate = None
    # Largest OpenCV grid for click-snap coverage. (usable_inner_grid is for
    # lens-calib gripper filtering only — too strict for rings here.)
    detection = detect_largest_inner_grid(search)
    if detection is None:
        return None
    flat = np.asarray(detection["corners"], dtype=np.float64).reshape(-1, 2) + offset.reshape(1, 2)
    rows = int(detection["inner_rows"])
    cols = int(detection["inner_cols"])
    if flat.shape[0] != rows * cols:
        return None
    if padded_gate is not None:
        keep = np.array([point_in_quad(pt, padded_gate) for pt in flat], dtype=bool)
        if not np.any(keep):
            return None
        if not np.all(keep):
            # Snap rings only need a point list; store as (N, 1, 2).
            return flat[keep].reshape(-1, 1, 2)
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
    restrict_to_board: bool = False,
) -> tuple[np.ndarray | None, str]:
    """Detect inner corners in the click frame.

    Tries ``image_bgr`` first (undistorted wrist or stationary). If that fails and a
    raw fisheye frame + calibration are provided, detect on the raw image and map
    corners into the undistorted frame via ``fisheye.undistortPoints``.

    ``restrict_to_board`` gates the search to a coarse board AABB (stationary only).

    Returns ``(grid, source_tag)`` where source_tag is one of
    ``\"undistorted\"``, ``\"image\"``, ``\"raw->undistort\"``, ``\"none\"``.
    """
    grid = detect_grid(image_bgr, restrict_to_board=restrict_to_board)
    if grid is not None:
        return grid, ("undistorted" if raw_bgr is not None else "image")
    if raw_bgr is not None and fisheye_calib is not None:
        raw_grid = detect_grid(raw_bgr, restrict_to_board=False)
        if raw_grid is not None:
            return map_fisheye_grid_to_undistorted(raw_grid, fisheye_calib), "raw->undistort"
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
    *,
    free_click: bool = False,
) -> tuple[tuple[float, float] | None, str | None]:
    """Resolve a calib label click.

    If ``grid`` is set, snap within ``max_dist_px`` or return an error.
    Hold Shift (``free_click=True``) to accept the raw pixel even when a grid
    is present — escape hatch when rings are partial or double-spaced.
    If no grid, accept the raw pixel (last-resort fallback).
    Returns ``((x, y), None)`` on success, or ``(None, error_message)``.
    """
    px = (float(pixel[0]), float(pixel[1]))
    if grid is None or free_click:
        return px, None
    snapped = snap_click_to_grid(grid, px, max_dist_px=max_dist_px)
    if snapped is None:
        return None, "click nearer a detected corner (or Shift+click for free)"
    return snapped, None


def corner_click_mode_line(grid: np.ndarray | None, manual: bool) -> str:
    """First help line while labelling corners: manual, snapping, or nothing detected."""
    if manual:
        return "MANUAL corners (f: snap) - click exactly on each corner"
    if grid is not None:
        return "Rings = detected corners - clicks snap (Shift+click: free; f: manual)"
    return "no corners detected - click carefully"


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


def draw_board_overlay(
    image: np.ndarray,
    T_board_camera: np.ndarray,
    camera_matrix: np.ndarray,
    square_mm: float = SQUARE_MM,
    alpha: float = 0.5,
) -> np.ndarray:
    """Draw the 8x8 board grid the pose predicts, half transparent, with a1 / h8 labelled
    (mutates ``image``). If the lines don't sit on the real squares, or a1 is at the wrong
    corner, the pose is wrong or the camera moved since it was solved."""
    n = np.arange(9, dtype=np.float64) * square_mm
    lattice = np.array([[x, y, 0.0] for y in n for x in n], dtype=np.float64)
    pix, depth = project_points(lattice, T_board_camera, camera_matrix)
    pix = pix.reshape(9, 9, 2)  # [rank line, file line]
    front = depth.reshape(9, 9) > 1.0
    layer = image.copy()

    def segment(a: tuple[int, int], b: tuple[int, int], edge: bool) -> None:
        if front[a] and front[b]:
            p, q = pix[a], pix[b]
            if np.all(np.abs(np.r_[p, q]) < 1e5):
                colour, width = ((0, 255, 255), 2) if edge else ((255, 255, 0), 1)
                cv2.line(layer, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, width, cv2.LINE_AA)

    for i in range(9):
        for j in range(8):
            segment((i, j), (i, j + 1), i in (0, 8))  # along a rank line, a -> h
            segment((j, i), (j + 1, i), i in (0, 8))  # along a file line, rank 1 -> 8
    cv2.addWeighted(layer, alpha, image, 1.0 - alpha, 0.0, dst=image)
    labels = np.array([[0.5 * square_mm, 0.5 * square_mm, 0.0], [7.5 * square_mm, 7.5 * square_mm, 0.0]])
    label_pix, label_depth = project_points(labels, T_board_camera, camera_matrix)
    for (x, y), d, text in zip(label_pix.reshape(-1, 2), label_depth, ("a1", "h8")):
        if d > 1.0 and abs(x) < 1e5 and abs(y) < 1e5:
            org = (int(x) - 10, int(y) + 6)
            cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
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
            return None, "point is beneath the board - click again on the line"
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
    # Wrist lens menu: only the two project files (default + status-quo alternate).
    # Bundle / lens_calib are resolve fallbacks, not menu choices.
    fisheye: list[Path] = []
    for path in [
        MOUNT_DIR / "wrist_fisheye.yaml",
        MOUNT_DIR / "wrist_fisheye_status_quo.yaml",
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
    skip_mount = {
        "wrist_fisheye.yaml",
        "wrist_fisheye_status_quo.yaml",
        "stationary_intrinsics.yaml",
        "board_in_base_mm.yaml",
    }
    for path in sorted(MOUNT_DIR.glob("*.yaml")):
        if path.name in skip_mount:
            continue
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
    extra: dict | None = None,
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
        # FK used these; the mount is only valid with the same offsets.
        "joint_offsets_deg": [round(float(v), 3) for v in joint_offsets_deg()],
        **(extra or {}),
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
