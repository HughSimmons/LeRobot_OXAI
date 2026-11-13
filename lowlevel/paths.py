"""Shared filesystem paths for the lowlevel/ simulation and calibration code.

Import these instead of hardcoding absolute paths, so scripts work on any
machine the repo is cloned onto.
"""

import os
import sys
from pathlib import Path

LOWLEVEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = LOWLEVEL_DIR.parent
SO_ARM100_DIR = REPO_ROOT / "SO-ARM100" / "Simulation" / "SO101"

ASSETS_DIR = SO_ARM100_DIR / "assets"
MESH_DIR = SO_ARM100_DIR
REAL_WORLD_RUNS_DIR = LOWLEVEL_DIR / "real_world_runs"
URDF_PATH = LOWLEVEL_DIR / "urdf" / "so101_with_mount.urdf"  # SO101 + wrist camera mount
#URDF_PATH = SO_ARM100_DIR / "so101_new_calib.urdf"

# Used by click_to_3d/ (Asha's find_piece branch) and click_to_3d_oliver/ (Oliver's fork of it);
# trimmed from that branch's paths.py.
URDF_WRIST_CAM_PATH = LOWLEVEL_DIR / "urdf" / "so101_new_calib_wrist_cam.urdf"
LEROBOT_SRC = LOWLEVEL_DIR / "lerobot" / "src"
# Oliver's SO101 follower (see SO101 Setup notes); override with SO101_PORT / SO101_ID.
# by-id paths stay fixed across replugs.
SO101_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB9065986-if00"
SO101_ID = "orange_follower_arm"

# Innomaker U20CAM in the SO101 wrist mount; Logitech C920 on the tripod.
WRIST_CAMERA = "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0"
STATIONARY_CAMERA = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_7671151F-video-index0"
# v4l2 controls set when a camera is opened (see `v4l2-ctl -d <camera> --list-ctrls`). The
# Innomaker came up with gain=100, which blew out ~70% of the wrist image.
# exposure_dynamic_framerate=0 stops the C920 dropping below 30 fps in dim light.
# The C920's focus is locked so its lens calibration stays valid; 40 was the
# sharpest setting (Laplacian variance sweep, 2026-10-22) at the tripod's
# distance to the table. Re-check it if the tripod moves.
WRIST_CAMERA_CONTROLS = "auto_exposure=3,gain=0"
STATIONARY_CAMERA_CONTROLS = "exposure_dynamic_framerate=0,focus_automatic_continuous=0,focus_absolute=40"


def setup_lerobot_path() -> Path:
    """Put lowlevel/lerobot/src first on sys.path so `import lerobot` finds the vendored package."""
    if not LEROBOT_SRC.is_dir():
        raise FileNotFoundError(f"LeRobot source not found at {LEROBOT_SRC}")
    src = str(LEROBOT_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    return LEROBOT_SRC


def import_robot_kinematics():
    """Load lerobot's RobotKinematics (placo) without importing lerobot/__init__.py, which pulls in torch."""
    import importlib.util
    import types

    setup_lerobot_path()
    if "lerobot.model.kinematics" in sys.modules:
        return sys.modules["lerobot.model.kinematics"].RobotKinematics

    # Empty parent packages so the two files below import without running lerobot/__init__.py.
    for name in ("lerobot", "lerobot.utils", "lerobot.model"):
        sys.modules.setdefault(name, types.ModuleType(name))

    def load(module_name: str, relative_path: str):
        spec = importlib.util.spec_from_file_location(module_name, LEROBOT_SRC / relative_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    load("lerobot.utils.import_utils", "lerobot/utils/import_utils.py")
    return load("lerobot.model.kinematics", "lerobot/model/kinematics.py").RobotKinematics


def connect_so101_hardware(*, calibrate: bool = True, configure: bool = True):
    """Connect the SO101 follower and return it.

    configure=True is lerobot's connect(): its configure() turns torque off on every motor
    while it writes the servo settings, then back on, so a loaded arm drops for a moment and
    then holds at whatever Goal_Position the servos last had. configure=False only opens
    the serial port and leaves torque and settings as they are (for reading joints).
    """
    setup_lerobot_path()
    # Drop any lerobot stubs left by import_robot_kinematics() so the real package loads.
    for name in [n for n in sys.modules if n == "lerobot" or n.startswith("lerobot.")]:
        del sys.modules[name]
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    follower = SO101Follower(
        SO101FollowerConfig(
            port=os.environ.get("SO101_PORT", SO101_PORT),
            id=os.environ.get("SO101_ID", SO101_ID),
        )
    )
    if configure:
        follower.connect(calibrate=calibrate)
    else:
        follower.bus.connect()
    return follower


def resolve_wrist_cam_urdf() -> tuple[str, str]:
    """Return (urdf_path, mesh_dir) for the SO101 URDF with a wrist_camera_link frame."""
    if not URDF_WRIST_CAM_PATH.is_file():
        raise FileNotFoundError(f"Wrist-camera URDF not found at {URDF_WRIST_CAM_PATH}")
    return str(URDF_WRIST_CAM_PATH), str(MESH_DIR)
