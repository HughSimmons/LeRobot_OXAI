"""Coordinate frame transforms, board geometry presets, and graveyard slots.

Bridges three coordinate conventions used across the project:
1. ``world`` / ``base`` frame (metres): SO-101 base_link at (0, 0, 0), +X forward
   across files a -> h, +Y left across ranks 1 -> 8, +Z vertically up.
2. ``board_center`` frame (metres): origin at the playing-surface centre of the
   8x8 board, axes aligned with world (+X file, +Y rank, +Z above playing surface).
3. ``coord_finder`` / ``two_view_grasp`` board frame (millimetres): origin (0, 0, 0)
   at the outer bottom-left corner of square ``a1`` on the playing surface,
   +X toward ``h1``, +Y toward ``a8``, +Z up out of the board.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

LOWLEVEL_DIR = Path(__file__).resolve().parents[1]
if str(LOWLEVEL_DIR) not in sys.path:
    sys.path.insert(0, str(LOWLEVEL_DIR))

from board_coordinates import FILES, XYPoint, validate_square  # noqa: E402

BOARD_SLAB_HALF_HEIGHT_M = 0.005
DEFAULT_BORDER_MARGIN_M = 0.030


@dataclass(frozen=True)
class BoardConfig:
    """Physical or simulated chessboard placement in the robot base frame (metres)."""

    origin_xyz_m: tuple[float, float, float]
    square_size_m: float
    slab_half_height_m: float = BOARD_SLAB_HALF_HEIGHT_M
    border_margin_m: float = DEFAULT_BORDER_MARGIN_M
    name: str = "custom"

    @property
    def surface_z_m(self) -> float:
        """Playing surface height in robot base frame (metres)."""
        return float(self.origin_xyz_m[2]) + float(self.slab_half_height_m)

    @property
    def board_span_m(self) -> float:
        return 8.0 * float(self.square_size_m)

    @property
    def a1_outer_corner_world_xyz_m(self) -> np.ndarray:
        """World-frame XYZ (m) of the outer a1 corner on the playing surface."""
        half = 0.5 * self.board_span_m
        return np.array(
            [
                float(self.origin_xyz_m[0]) - half,
                float(self.origin_xyz_m[1]) - half,
                self.surface_z_m,
            ],
            dtype=np.float64,
        )

    def square_center_world_xy(self, square: str) -> np.ndarray:
        sq = validate_square(square)
        fi = FILES.index(sq[0])
        ri = int(sq[1]) - 1
        a1 = self.a1_outer_corner_world_xyz_m
        return np.array(
            [
                a1[0] + (fi + 0.5) * self.square_size_m,
                a1[1] + (ri + 0.5) * self.square_size_m,
            ],
            dtype=np.float64,
        )

    def square_center_world_xyz(self, square: str, z_above_surface_m: float = 0.0) -> np.ndarray:
        xy = self.square_center_world_xy(square)
        return np.array([xy[0], xy[1], self.surface_z_m + float(z_above_surface_m)], dtype=np.float64)

    def is_over_board(self, xy_m: Sequence[float], extra_margin_m: float = 0.0) -> bool:
        half = 0.5 * self.board_span_m + float(self.border_margin_m) + float(extra_margin_m)
        return (
            abs(float(xy_m[0]) - float(self.origin_xyz_m[0])) <= half
            and abs(float(xy_m[1]) - float(self.origin_xyz_m[1])) <= half
        )

    def nearest_square(self, xy_m: Sequence[float]) -> str:
        x, y = float(xy_m[0]), float(xy_m[1])
        a1 = self.a1_outer_corner_world_xyz_m
        fi = int(np.clip(math.floor((x - a1[0]) / self.square_size_m), 0, 7))
        ri = int(np.clip(math.floor((y - a1[1]) / self.square_size_m), 0, 7))
        return f"{FILES[fi]}{ri + 1}"

    def graveyard_slot_world_xyz(
        self,
        slot_index: int = 0,
        *,
        z_above_table_m: float = 0.0,
    ) -> np.ndarray:
        """Collision-free off-board capture drop slot in the strict-0-deg vertical zone.

        Slots alternate right (-Y) and left (+Y) of the board inside x in [0.16, 0.27] m
        where SO-101 reaches with 0.0 deg vertical gripper orientation and generous margin.
        """
        if slot_index < 0:
            raise ValueError("slot_index must be non-negative")
        side = -1.0 if (slot_index % 2 == 0) else 1.0
        row_idx = slot_index // 2
        col = row_idx % 4
        tier = row_idx // 4
        half_y = 0.5 * self.board_span_m + self.border_margin_m
        y_m = float(self.origin_xyz_m[1]) + side * (half_y + 0.048 + 0.038 * tier)
        x_start = max(0.160, float(self.origin_xyz_m[0]) - 0.100)
        x_m = x_start + 0.036 * col
        return np.array([x_m, y_m, float(z_above_table_m)], dtype=np.float64)


# Canonical presets matching both local sim and upstream/oliver/two-camera-capture
SIM_BOARD_CONFIG = BoardConfig(
    origin_xyz_m=(0.25, 0.0, 0.0),
    square_size_m=0.040,
    slab_half_height_m=0.005,
    border_margin_m=0.025,
    name="sim_025",
)

REAL_MEASURED_BOARD_CONFIG = BoardConfig(
    origin_xyz_m=(0.2926, 0.0, 0.024),
    square_size_m=0.04125,
    slab_half_height_m=0.005,
    border_margin_m=0.030,
    name="real_02926",
)


def board_config_from_env(default: BoardConfig = SIM_BOARD_CONFIG) -> BoardConfig:
    """Resolve BoardConfig from SIM_BOARD_* environment variables if set."""
    bx = float(os.environ.get("SIM_BOARD_ORIGIN_X", default.origin_xyz_m[0]))
    by = float(os.environ.get("SIM_BOARD_ORIGIN_Y", default.origin_xyz_m[1]))
    bz = float(os.environ.get("SIM_BOARD_ORIGIN_Z", default.origin_xyz_m[2]))
    sq = float(os.environ.get("SIM_BOARD_SQUARE_SIZE", default.square_size_m))
    return BoardConfig(
        origin_xyz_m=(bx, by, bz),
        square_size_m=sq,
        slab_half_height_m=default.slab_half_height_m,
        border_margin_m=default.border_margin_m,
        name="env",
    )


def coord_finder_mm_to_world_m(
    board_xyz_mm: Sequence[float],
    board: BoardConfig = REAL_MEASURED_BOARD_CONFIG,
    *,
    T_base_board_mm: np.ndarray | None = None,
    coord_finder_square_mm: float | None = None,
) -> np.ndarray:
    """Convert ``coord_finder`` / ``two_view_grasp`` (x_mm, y_mm, z_mm) to world metres.

    If ``T_base_board_mm`` (4x4 camera-calibrated board-to-base transform in mm from
    ``click_to_3d``) is provided, uses that rigid transform directly. Otherwise maps
    from the ``a1`` outer corner of ``board`` (scaling XY if ``coord_finder_square_mm``
    differs from ``board.square_size_m * 1000``).
    """
    pts_mm = np.asarray(board_xyz_mm, dtype=np.float64).reshape(3)
    if T_base_board_mm is not None:
        T = np.asarray(T_base_board_mm, dtype=np.float64).reshape(4, 4)
        base_mm = (T @ np.append(pts_mm, 1.0))[:3]
        return base_mm / 1000.0

    scale_xy = 1.0
    if coord_finder_square_mm is not None and coord_finder_square_mm > 0:
        scale_xy = (board.square_size_m * 1000.0) / float(coord_finder_square_mm)

    a1_m = board.a1_outer_corner_world_xyz_m
    return np.array(
        [
            a1_m[0] + (pts_mm[0] * scale_xy) / 1000.0,
            a1_m[1] + (pts_mm[1] * scale_xy) / 1000.0,
            a1_m[2] + pts_mm[2] / 1000.0,
        ],
        dtype=np.float64,
    )


def world_m_to_coord_finder_mm(
    world_xyz_m: Sequence[float],
    board: BoardConfig = REAL_MEASURED_BOARD_CONFIG,
) -> np.ndarray:
    """Inverse of ``coord_finder_mm_to_world_m`` for axis-aligned board placement."""
    w = np.asarray(world_xyz_m, dtype=np.float64).reshape(3)
    a1_m = board.a1_outer_corner_world_xyz_m
    return (w - a1_m) * 1000.0


def grasp_dict_to_world_target(
    grasp: Mapping[str, Any],
    board: BoardConfig = REAL_MEASURED_BOARD_CONFIG,
    *,
    default_pinch_height_m: float = 0.028,
    T_base_board_mm: np.ndarray | None = None,
    coord_finder_square_mm: float | None = None,
) -> tuple[np.ndarray, float | None]:
    """Convert ``two_view_grasp.grasp_from_views`` output dict into (world_xyz_m, yaw_deg).

    Handles ``z_mm=None`` fallback (when side camera ray is occluded) using
    ``default_pinch_height_m`` above the playing surface.
    """
    x_mm = float(grasp["x_mm"])
    y_mm = float(grasp["y_mm"])
    z_mm_raw = grasp.get("z_mm")
    z_mm = float(default_pinch_height_m * 1000.0) if z_mm_raw is None else float(z_mm_raw)
    world_xyz = coord_finder_mm_to_world_m(
        (x_mm, y_mm, z_mm),
        board=board,
        T_base_board_mm=T_base_board_mm,
        coord_finder_square_mm=coord_finder_square_mm,
    )
    yaw_raw = grasp.get("yaw_deg")
    yaw_deg = None if yaw_raw is None else float(yaw_raw)
    return world_xyz, yaw_deg


def resolve_target_world_xyz(
    target: str | XYPoint | Sequence[float],
    board: BoardConfig,
    *,
    default_z_above_surface_m: float = 0.025,
) -> np.ndarray:
    """Resolve a square name, XYPoint, (x, y), or (x, y, z) to world metres."""
    if isinstance(target, str):
        return board.square_center_world_xyz(target, z_above_surface_m=default_z_above_surface_m)
    if isinstance(target, XYPoint):
        return np.array(
            [target.x, target.y, board.surface_z_m + float(default_z_above_surface_m)],
            dtype=np.float64,
        )
    arr = np.asarray(target, dtype=np.float64).reshape(-1)
    if arr.size == 2:
        return np.array(
            [arr[0], arr[1], board.surface_z_m + float(default_z_above_surface_m)],
            dtype=np.float64,
        )
    if arr.size == 3:
        return arr.copy()
    raise ValueError(f"Unsupported target specification: {target!r}")
