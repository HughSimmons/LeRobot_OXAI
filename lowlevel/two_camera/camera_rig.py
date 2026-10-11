"""Shared constants for the two-camera PyBullet board-imaging scene.

This module is the single source of truth for the rendering setup used by
`generate_sim_images.py`: image size, field of view, the two hardcoded
camera poses, and the derived pinhole intrinsics. `calibrate_camera_distance.py`
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
