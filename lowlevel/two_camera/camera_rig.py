"""Shared constants for the two-camera PyBullet board-imaging scene.

This module is the single source of truth for the rendering setup used by
`generate_sim_images.py`: image size, field of view, the hardcoded camera
poses, the forward-kinematics wrist camera's placement on the arm, and the
derived pinhole intrinsics. `calibrate_camera_distance.py`
deliberately does not import this module -- it only reads intrinsics and
ground truth back out of the manifest that `generate_sim_images.py` writes
(or takes them as CLI flags), so there is exactly one place where the
FOV-to-intrinsics formula is implemented.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from board_coordinates import DEFAULT_BOARD_ORIGIN, DEFAULT_SQUARE_SIZE  # noqa: E402

WIDTH, HEIGHT = 640, 360
FOV_DEG = 60.0
NEAR, FAR = 0.01, 100.0

BOARD_ORIGIN = DEFAULT_BOARD_ORIGIN
SQUARE_SIZE = DEFAULT_SQUARE_SIZE
BOARD_BASE_HALF_HEIGHT = 0.005
BOARD_TOP_Z = BOARD_ORIGIN[2] + BOARD_BASE_HALF_HEIGHT
SQUARE_SURFACE_Z = BOARD_TOP_Z + 0.001

# Tripod pose is parameterized as (radius, height, azimuth) around the board
# center rather than a raw eye vector, so the "rotated off-axis" check from
# calibration verification is easy to see and adjust. -40 deg is the -90 deg
# (straight-on, -y) pose rotated 50 deg off-axis -- kept as the default since
# it already confirmed the solvePnP pipeline isn't sensitive to one
# particular viewing direction.
_TRIPOD_RADIUS_M = 0.45
_TRIPOD_EYE_Z_M = 0.42
_TRIPOD_AZIMUTH_DEG = -40.0
_tripod_azimuth_rad = math.radians(_TRIPOD_AZIMUTH_DEG)
_TRIPOD_EYE = [
    BOARD_ORIGIN[0] + _TRIPOD_RADIUS_M * math.cos(_tripod_azimuth_rad),
    BOARD_ORIGIN[1] + _TRIPOD_RADIUS_M * math.sin(_tripod_azimuth_rad),
    _TRIPOD_EYE_Z_M,
]

# Camera poses (eye, target, up), world frame, metres. Chosen so the full
# 8x8 board (0.33 m across) sits comfortably inside the 60 deg vertical FOV
# frame with margin -- at this FOV, distance-to-target has to stay roughly
# >= 0.4 m for the board not to overflow the frame.
CAMERAS = {
    "topdown": {  # ground-truth reference view: straight down at the board center
        "eye": [BOARD_ORIGIN[0], BOARD_ORIGIN[1], BOARD_ORIGIN[2] + 0.6],
        "target": [BOARD_ORIGIN[0], BOARD_ORIGIN[1], SQUARE_SURFACE_Z],
        "up": [0.0, 1.0, 0.0],
    },
    "tripod": {  # stationary, off to the side, moderate elevation, further back
        "eye": _TRIPOD_EYE,
        "target": [BOARD_ORIGIN[0], BOARD_ORIGIN[1], SQUARE_SURFACE_Z],
        "up": [0.0, 0.0, 1.0],
    },
    "wrist": {  # simulated arm-mounted camera: closer, steeper look-down
        "eye": [BOARD_ORIGIN[0], -0.08, 0.40],
        "target": [BOARD_ORIGIN[0], BOARD_ORIGIN[1], SQUARE_SURFACE_Z],
        "up": [0.0, 1.0, 0.0],
    },
}

# Wrist camera driven by the arm's forward kinematics (generate_sim_images.py
# --with-robot), replacing the hardcoded "wrist" pose above. Loads
# paths.MOUNT_URDF_PATH, which adds the camera mount + camera as
# WRIST_CAMERA_LINK on gripper_link.
WRIST_CAMERA_LINK = "wrist_camera_mount_link"
# Lens axis in WRIST_CAMERA_LINK's frame, metres: centre of the lens where it
# meets the board -> centre of the lens front face. Measured from the lens
# cylinder in lowlevel/urdf/so101_wrist_cam_mount.stl (STL coords + the
# link's (0, 18.4, -6.3) mm visual offset).
WRIST_LENS_BASE = (-0.049341, -0.019730, 0.011200)
WRIST_LENS_TIP = (-0.040904, -0.001574, 0.011200)
# Image "up" in the link frame. Picked so the jaws sit at the bottom of the
# image; NOTE check against a real photo and flip if the sim image is rotated.
WRIST_IMAGE_UP = (-0.9069, 0.4214, 0.0)
# Vertical FOV of the wrist camera. 
WRIST_FOV_DEG = 80.0
# Default Joint angles in degrees (shoulder_pan, shoulder_lift, elbow_flex,
# wrist_flex, wrist_roll, gripper)
DEFAULT_WRIST_JOINTS_DEG = (-23.2, -93.9, 45.9, 94.9, -101.5, 0.0)


def intrinsics_from_fov(fov_deg: float = FOV_DEG, width: int = WIDTH, height: int = HEIGHT) -> dict:
    """Pinhole intrinsics for PyBullet's computeProjectionMatrixFOV(fov, aspect, ...).

    `fov` there is the vertical field of view in degrees, and `aspect` is
    always passed as width/height (square pixels, no independent horizontal
    FOV), which makes fx == fy exactly:

        fy = fx = (height / 2) / tan(radians(fov) / 2)
        cx, cy = width / 2, height / 2

    The synthetic camera has no lens, so distortion is zero.
    """
    fov_rad = math.radians(fov_deg)
    fy = (height / 2.0) / math.tan(fov_rad / 2.0)
    fx = fy
    cx, cy = width / 2.0, height / 2.0
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
