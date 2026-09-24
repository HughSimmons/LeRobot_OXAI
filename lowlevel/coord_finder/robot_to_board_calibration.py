#!/usr/bin/env python3
"""Measure the robot-to-board-centre distance using a tape measure visible in
a calibration photo, combined with a LiveChess2FEN board-plane homography.

Click three things on the photo: 2+ points on the board's own playing-surface
edge (its outer wooden frame, not the 8x8 grid boundary), 2+ points on the
robot base's front edge, and 2+ points along EACH of the tape measure's two
long edges (each paired with its real-world reading, typed in after
clicking), plus the tape's known physical width. Clicking both edges of the
tape (not just its centreline) gives 2D-spread correspondences, so a real
table-plane homography can be fit (cv2.findHomography), not just a 1D scale.

Two measurements combine to give the final robot-origin-to-board-centre
distance:
- front edge -> playing-surface edge: both point groups are mapped through
  the table-plane homography and the Euclidean distance is taken. This
  crosses from the table plane to the elevated board plane, so it's a
  cross-check, not fully parallax-corrected.
- playing-surface edge -> board centre: measured precisely on the board
  plane itself (no parallax issue, both points are on the same elevated
  surface) using the LiveChess2FEN-detected board corners/homography and the
  known square length. This is what reveals the true edge-to-centre
  distance, since the outer frame sits further out than the 4-square-length
  distance to the 8x8 grid's own boundary.

A third result, `oriented_board_offset_measurement`, decomposes the
front-edge-to-board-centre offset into forward/lateral components using the
robot's own front-edge orientation (not the tape's), giving both
board_origin_x and board_origin_y directly.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from board_detect_lc2fen import detect_and_warp_board
from board_geometry import (
    BOARD_SIZE_PX,
    board_transform_from_corners,
    detect_board_corners,
    load_image,
    save_debug,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_IMAGE = SCRIPT_DIR / "input" / "robot_to_board_calibration.png"
DEFAULT_OUT_DIR = SCRIPT_DIR / "output" / "robot_to_board_calibration"
DEFAULT_SQUARE_LENGTH_M = 0.04125
WINDOW_NAME = "tape_distance_calibration"
# From so101_new_calib.urdf base_link mesh geometry (base_so101_v2.stl), in
# the base_link frame: x_max of the base plate (front-most point, toward the
# board).
ROBOT_ORIGIN_TO_FRONT_EDGE_M = 0.06463545808017192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--square-length-m", type=float, default=DEFAULT_SQUARE_LENGTH_M)
    parser.add_argument(
        "--tape-width-m",
        type=float,
        default=None,
        help=(
            "Physical width of the tape measure's blade, in metres. Combined "
            "with clicks on both its long edges, this gives 2D-spread "
            "correspondences for a real table-plane homography. Prompted "
            "interactively if not given."
        ),
    )
    parser.add_argument(
        "--no-tape",
        action="store_true",
        help="Skip the tape-measure homography source (default: used).",
    )
    parser.add_argument(
        "--no-table-edges",
        action="store_true",
        help=(
            "Skip the table-edges homography source (default: used). Click "
            ">=2 points on each of two perpendicular table edges, each with "
            "known measured distances from their shared corner."
        ),
    )
    parser.add_argument(
        "--no-local-grid",
        action="store_true",
        help=(
            "Skip clicking the 9 central grid intersections for a locally-fit "
            "board homography (default: used). The global 4-corner homography "
            "assumes an ideal pinhole camera; real lens distortion makes its "
            "interpolated interior grid lines drift from the true squares. "
            "Clicking the 9 points bounding the central 2x2 block of squares "
            "(d4/d5/e4/e5) gives a homography that's only trusted near the "
            "board centre, but isn't biased by distortion far from those "
            "clicks -- used to get a more accurate board centre pixel."
        ),
    )
    parser.add_argument(
        "--board-corners",
        nargs=4,
        metavar=("TL", "TR", "BR", "BL"),
        help="Manual board corners as x,y pairs in original image pixels, "
        "if LiveChess2FEN detection is unavailable or wrong.",
    )
    parser.add_argument(
        "--reclick-local-grid",
        action="store_true",
        help=(
            "Force a fresh click of the 9 central grid points even when "
            "--replay-points-from already has a local_grid saved; all other "
            "points/sources are still replayed as usual."
        ),
    )
    parser.add_argument(
        "--replay-points-from",
        type=Path,
        default=None,
        help=(
            "Path to a previous measurement.json; reuse its clicked points "
            "and typed tape values instead of prompting for new clicks."
        ),
    )
    return parser.parse_args()


# --- board plane (LiveChess2FEN) -------------------------------------------


def parse_corner_pair(value: str) -> list[float]:
    try:
        x_text, y_text = value.split(",", maxsplit=1)
        return [float(x_text), float(y_text)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid corner {value!r}; expected x,y") from exc


def apply_homography(point_xy: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    point = np.array([[[point_xy[0], point_xy[1]]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(point, homography)[0, 0]
    return float(warped[0]), float(warped[1])


def inverse_project(point_xy: tuple[float, float], homography: np.ndarray) -> tuple[int, int]:
    inv_h = np.linalg.inv(homography)
    x, y = apply_homography(point_xy, inv_h)
    return int(round(x)), int(round(y))


def detect_board_for_calibration(
    image_path: Path,
    image: np.ndarray,
    out_dir: Path,
    manual_corners: list[str] | None,
) -> dict:
    """Prefer LiveChess2FEN detection; fall back to local OpenCV geometry."""
    if manual_corners is not None:
        corners = np.asarray([parse_corner_pair(value) for value in manual_corners], dtype=np.float32)
        transform = board_transform_from_corners(corners, board_size_px=BOARD_SIZE_PX)
        warped_path = out_dir / "warped_board.jpg"
        save_debug(warped_path, transform.warp(image))
        meta = {
            "source_image": str(image_path),
            "warped_board": str(warped_path),
            "board_size_px": BOARD_SIZE_PX,
            "board_corners": corners.astype(float).tolist(),
            "method": "manual_board_corners",
        }
        meta_path = out_dir / "board_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        meta["meta_path"] = str(meta_path)
        return meta

    try:
        return detect_and_warp_board(image_path, out_dir)
    except (FileNotFoundError, ImportError) as exc:
        print(f"LiveChess2FEN unavailable, falling back to board_geometry: {exc}")

    corners = detect_board_corners(image)
    transform = board_transform_from_corners(corners, board_size_px=BOARD_SIZE_PX)
    warped_path = out_dir / "warped_board.jpg"
    save_debug(warped_path, transform.warp(image))
    meta = {
        "source_image": str(image_path),
        "warped_board": str(warped_path),
        "board_size_px": BOARD_SIZE_PX,
        "board_corners": np.asarray(corners, dtype=float).tolist(),
        "method": "board_geometry_fallback",
        "fallback_reason": "LiveChess2FEN repo not found",
    }
    meta_path = out_dir / "board_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["meta_path"] = str(meta_path)
    return meta


def board_plane_edge_to_centre_measurement(
    playing_surface_edge_points_px: list[tuple[float, float]],
    homography: np.ndarray,
    board_size_px: int,
    square_length_m: float,
) -> dict:
    """Distance from the board centre to each playing-surface edge point,
    measured on the board plane itself via the LiveChess2FEN homography --
    no parallax issue since both are on the same elevated surface. Reveals
    the true edge-to-centre distance, which is greater than 4 square lengths
    because the outer wooden frame sits further out than the 8x8 grid.
    """
    metres_per_board_px = square_length_m * 8.0 / board_size_px
    center_board_px = (board_size_px / 2.0, board_size_px / 2.0)
    points_board_xy_m = []
    distances_m = []
    for point_px in playing_surface_edge_points_px:
        board_px = apply_homography(point_px, homography)
        xy_m = (
            (board_px[0] - center_board_px[0]) * metres_per_board_px,
            (board_px[1] - center_board_px[1]) * metres_per_board_px,
        )
        points_board_xy_m.append(list(xy_m))
        distances_m.append(float(np.hypot(*xy_m)))

    # Perpendicular distance from centre to the fitted edge *line* -- the
    # true closest approach, not the distance to individual clicked points
    # (which also carries however far along the edge each point happens to
    # be, so it's >= this value).
    perpendicular_distance_m = None
    if len(points_board_xy_m) >= 2:
        line_point, line_dir = fit_line_px(points_board_xy_m)
        perpendicular_distance_m = float(abs(line_point[0] * line_dir[1] - line_point[1] * line_dir[0]))

    return {
        "points_board_xy_m": points_board_xy_m,
        "distances_from_centre_m": distances_m,
        "mean_edge_to_centre_m": float(np.mean(distances_m)),
        "perpendicular_distance_to_edge_line_m": perpendicular_distance_m,
    }


def draw_board_overlay(
    image: np.ndarray,
    corners: np.ndarray,
    homography: np.ndarray,
    board_size_px: int,
    square_corners_px: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    overlay = image.copy()
    corners_i = np.round(corners).astype(np.int32)
    cv2.polylines(overlay, [corners_i.reshape(-1, 1, 2)], True, (0, 255, 255), 3, cv2.LINE_AA)
    for i in range(9):
        p0 = inverse_project((i * board_size_px / 8.0, 0.0), homography)
        p1 = inverse_project((i * board_size_px / 8.0, float(board_size_px)), homography)
        q0 = inverse_project((0.0, i * board_size_px / 8.0), homography)
        q1 = inverse_project((float(board_size_px), i * board_size_px / 8.0), homography)
        cv2.line(overlay, p0, p1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(overlay, q0, q1, (255, 255, 255), 1, cv2.LINE_AA)
    # LiveChess2FEN's own detected 9x9 lattice points (independent of the
    # 4-corner homography interpolation above) -- drawn alongside the white
    # interpolated grid so drift between the two is visible directly.
    if square_corners_px:
        for x, y in square_corners_px:
            cv2.drawMarker(
                overlay, (int(round(x)), int(round(y))), (255, 255, 0), cv2.MARKER_CROSS, 10, 2, cv2.LINE_AA
            )
    center_px = inverse_project((board_size_px / 2.0, board_size_px / 2.0), homography)
    cv2.drawMarker(overlay, center_px, (0, 0, 255), cv2.MARKER_CROSS, 24, 3, cv2.LINE_AA)
    used_rects: list[tuple[int, int, int, int]] = []
    draw_label(overlay, "board centre", center_px, (0, 0, 255), used_rects, font_scale=0.7)
    return overlay, center_px


def fit_line_px(points_px: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares line through 2+ pixel (or metric) points via SVD.

    Returns (point_on_line, unit_direction).
    """
    points = np.asarray(points_px, dtype=float)
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    direction = vh[0]
    return centroid, direction / np.linalg.norm(direction)


def fit_table_homography(
    tape_near_points_px: list[tuple[float, float]],
    tape_near_values_m: list[float],
    tape_far_points_px: list[tuple[float, float]],
    tape_far_values_m: list[float],
    tape_width_m: float,
) -> dict:
    """Real table-plane homography (pixel -> metric (u, v)) fit from points on
    both long edges of a tape measure of known width.

    u is the tape's own along-length reading; v is 0 on the near edge and
    tape_width_m on the far edge. Clicking both edges (not just the
    centreline) gives 2D-spread correspondences, so this is a genuine
    projective homography (cv2.findHomography, >=4 points), not just a 1D
    scale -- it captures perspective properly across the whole photo.
    """
    if len(tape_near_points_px) < 2 or len(tape_far_points_px) < 2:
        raise ValueError("Need >=2 points on each tape edge.")
    if len(tape_near_points_px) != len(tape_near_values_m):
        raise ValueError("Near-edge points/values count mismatch.")
    if len(tape_far_points_px) != len(tape_far_values_m):
        raise ValueError("Far-edge points/values count mismatch.")

    src = np.asarray(tape_near_points_px + tape_far_points_px, dtype=np.float32)
    dst = np.asarray(
        [[v, 0.0] for v in tape_near_values_m] + [[v, tape_width_m] for v in tape_far_values_m],
        dtype=np.float32,
    )
    homography, _ = cv2.findHomography(src, dst, method=0)
    if homography is None:
        raise ValueError("Failed to fit a homography from the tape edge points.")
    return {
        "source": "tape",
        "tape_near_points_px": [[float(x), float(y)] for x, y in tape_near_points_px],
        "tape_near_values_m": [float(v) for v in tape_near_values_m],
        "tape_far_points_px": [[float(x), float(y)] for x, y in tape_far_points_px],
        "tape_far_values_m": [float(v) for v in tape_far_values_m],
        "tape_width_m": float(tape_width_m),
        "homography": homography.tolist(),
    }


def fit_table_edges_homography(
    flush_points_px: list[tuple[float, float]],
    flush_values_m: list[float],
    perpendicular_points_px: list[tuple[float, float]],
    perpendicular_values_m: list[float],
) -> dict:
    """Real table-plane homography (pixel -> metric (u, v)) fit from two
    perpendicular table edges, each with known readings (e.g. measured with a
    tape/ruler from their shared corner).

    flush edge points -> (reading, 0); perpendicular edge points -> (0,
    reading). Assumes both edges' readings are measured from the same shared
    corner and that the edges are genuinely perpendicular -- if that's not
    exactly true, the homography absorbs a small amount of skew error.
    """
    if len(flush_points_px) < 2 or len(perpendicular_points_px) < 2:
        raise ValueError("Need >=2 points on each table edge.")
    if len(flush_points_px) != len(flush_values_m):
        raise ValueError("Flush-edge points/values count mismatch.")
    if len(perpendicular_points_px) != len(perpendicular_values_m):
        raise ValueError("Perpendicular-edge points/values count mismatch.")

    src = np.asarray(flush_points_px + perpendicular_points_px, dtype=np.float32)
    dst = np.asarray(
        [[v, 0.0] for v in flush_values_m] + [[0.0, v] for v in perpendicular_values_m],
        dtype=np.float32,
    )
    homography, _ = cv2.findHomography(src, dst, method=0)
    if homography is None:
        raise ValueError("Failed to fit a homography from the table edge points.")
    return {
        "source": "table_edges",
        "table_flush_points_px": [[float(x), float(y)] for x, y in flush_points_px],
        "table_flush_values_m": [float(v) for v in flush_values_m],
        "table_perpendicular_points_px": [[float(x), float(y)] for x, y in perpendicular_points_px],
        "table_perpendicular_values_m": [float(v) for v in perpendicular_values_m],
        "homography": homography.tolist(),
    }


def fit_local_board_homography(
    central_square_corners_px: list[tuple[float, float]],
    square_length_m: float,
) -> dict:
    """Locally-fit pixel <-> board-plane-metric homography from the 9 grid
    intersections bounding the central 2x2 block of squares (d4/d5/e4/e5),
    clicked in row-major order (3 rows top-to-bottom, 3 points left-to-right
    per row).

    The global 4-corner homography (`board_transform_from_corners`) assumes
    an ideal pinhole camera, so its straight-line interpolation between the
    4 outer corners drifts away from the true (slightly lens-distorted)
    interior grid lines. This local fit is only trusted near the board
    centre -- it should not be extrapolated out to the playing-surface edge
    -- but gives an accurate board-centre pixel and nearby square locations,
    since lens distortion is locally near-linear over such a small patch.

    The middle of the 9 clicked points (index 4) IS the board's true centre
    by construction (it's the point shared by all 4 central squares), so
    `board_center_px` is read off directly, not solved via the homography.
    """
    if len(central_square_corners_px) != 9:
        raise ValueError("Need exactly 9 points (3x3 grid, row-major).")

    src = np.asarray(central_square_corners_px, dtype=np.float32)
    dst = np.asarray(
        [[(col - 1) * square_length_m, (row - 1) * square_length_m] for row in range(3) for col in range(3)],
        dtype=np.float32,
    )
    homography, _ = cv2.findHomography(src, dst, method=0)
    if homography is None:
        raise ValueError("Failed to fit a local homography from the central grid points.")

    return {
        "central_square_corners_px": [[float(x), float(y)] for x, y in central_square_corners_px],
        "square_length_m": float(square_length_m),
        "homography": homography.tolist(),
        "board_center_px": [float(central_square_corners_px[4][0]), float(central_square_corners_px[4][1])],
    }


def local_grid_point_px(local_grid: dict, dx_squares: float, dy_squares: float) -> tuple[int, int]:
    """Pixel location of a point `dx_squares`/`dy_squares` squares away from
    the board centre (in the local grid's row/col click directions), via the
    local homography. Only accurate near the centre -- see
    `fit_local_board_homography`.
    """
    homography = np.asarray(local_grid["homography"], dtype=float)
    square_length_m = local_grid["square_length_m"]
    return inverse_project((dx_squares * square_length_m, dy_squares * square_length_m), homography)


def tape_calibrated_measurement(
    playing_surface_edge_points_px: list[tuple[float, float]],
    robot_base_front_points_px: list[tuple[float, float]],
    table_homography: np.ndarray,
) -> dict:
    """Distance between the robot's front edge and the board's playing-surface
    edge, mapping both point groups through the real table-plane homography.

    This crosses from the table plane to the elevated board plane, so it
    still carries a parallax bias for whichever point sits on the raised
    surface; treat this as a cross-check against the properly board-plane
    corrected `board_plane_edge_to_centre_measurement`, not a final figure.
    """
    if len(playing_surface_edge_points_px) < 2 or len(robot_base_front_points_px) < 2:
        raise ValueError("Need >=2 points each for the playing-surface edge and robot base front.")

    playing_surface_uv = [apply_homography(p, table_homography) for p in playing_surface_edge_points_px]
    robot_front_uv = [apply_homography(p, table_homography) for p in robot_base_front_points_px]
    playing_surface_centroid = np.mean(np.asarray(playing_surface_uv), axis=0)
    robot_front_centroid = np.mean(np.asarray(robot_front_uv), axis=0)
    raw_distance_m = float(np.linalg.norm(playing_surface_centroid - robot_front_centroid))
    robot_origin_distance_m = raw_distance_m + ROBOT_ORIGIN_TO_FRONT_EDGE_M

    return {
        "playing_surface_edge_points_px": [
            [float(x), float(y)] for x, y in playing_surface_edge_points_px
        ],
        "robot_base_front_points_px": [
            [float(x), float(y)] for x, y in robot_base_front_points_px
        ],
        "playing_surface_edge_uv_m": playing_surface_centroid.tolist(),
        "robot_base_front_uv_m": robot_front_centroid.tolist(),
        "raw_tape_distance_m": raw_distance_m,
        "robot_origin_to_front_edge_m": ROBOT_ORIGIN_TO_FRONT_EDGE_M,
        "robot_origin_to_playing_surface_edge_m": robot_origin_distance_m,
    }


def oriented_board_offset_measurement(
    robot_base_front_points_px: list[tuple[float, float]],
    board_center_px: tuple[int, int],
    table_homography: np.ndarray,
) -> dict:
    """Forward/lateral offset from the robot's front edge to the board centre,
    using the front edge's own orientation (not the tape's) to define the
    forward/lateral axes, with distances read directly from the real
    table-plane homography (metric throughout, no separate pixel scale).

    This avoids assuming the tape was laid exactly along the robot's true
    centreline: the front edge is a fixed, known feature of the robot itself,
    so its own fitted direction (mapped into the table-plane) defines
    lateral, and the perpendicular defines forward -- toward the board.
    """
    if len(robot_base_front_points_px) < 2:
        raise ValueError("Need >=2 points for the robot base front edge.")

    front_uv = [apply_homography(p, table_homography) for p in robot_base_front_points_px]
    front_centroid, lateral_unit = fit_line_px(front_uv)
    forward_unit = np.array([-lateral_unit[1], lateral_unit[0]])
    board_uv = np.asarray(apply_homography(board_center_px, table_homography), dtype=float)
    board_vec = board_uv - front_centroid
    if np.dot(board_vec, forward_unit) < 0:
        forward_unit = -forward_unit

    forward_m = float(np.dot(board_vec, forward_unit))
    lateral_m = float(np.dot(board_vec, lateral_unit))

    board_origin_x = ROBOT_ORIGIN_TO_FRONT_EDGE_M + forward_m
    board_origin_y = lateral_m

    return {
        "robot_base_front_points_px": [
            [float(x), float(y)] for x, y in robot_base_front_points_px
        ],
        "robot_base_front_uv_m": [list(p) for p in front_uv],
        "board_center_px": [float(board_center_px[0]), float(board_center_px[1])],
        "board_center_uv_m": board_uv.tolist(),
        "front_centroid_uv_m": front_centroid.tolist(),
        "forward_unit_uv": forward_unit.tolist(),
        "lateral_unit_uv": lateral_unit.tolist(),
        "front_to_centre_forward_m": forward_m,
        "front_to_centre_lateral_m": lateral_m,
        "robot_origin_to_front_edge_m": ROBOT_ORIGIN_TO_FRONT_EDGE_M,
        "board_origin_x": board_origin_x,
        "board_origin_y": board_origin_y,
    }


# --- drawing helpers -------------------------------------------------------


def rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def text_rect(
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    thickness: int,
) -> tuple[int, int, int, int]:
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = origin
    return x, y - height - baseline, x + width, y + baseline


def padded_rect(rect: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return rect[0] - pad, rect[1] - pad, rect[2] + pad, rect[3] + pad


def point_reserved_rect(point: tuple[int, int], radius: int = 20) -> tuple[int, int, int, int]:
    return point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius


def clamp_text_origin(
    origin: tuple[int, int],
    text: str,
    image_shape: tuple[int, ...],
    font_scale: float,
    thickness: int,
    margin: int = 8,
) -> tuple[int, int]:
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    img_h, img_w = image_shape[:2]
    x = min(max(origin[0], margin), max(margin, img_w - width - margin))
    y = min(max(origin[1], height + baseline + margin), max(height + baseline + margin, img_h - baseline - margin))
    return int(x), int(y)


def draw_label(
    image: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int],
    used_rects: list[tuple[int, int, int, int]],
    font_scale: float = 0.55,
    thickness: int = 2,
) -> None:
    offsets = [
        (24, -20),
        (24, 42),
        (-150, -20),
        (-150, 42),
        (32, -64),
        (32, 82),
        (-210, -64),
        (-210, 82),
        (0, -92),
        (0, 110),
    ]
    chosen = None
    chosen_rect = None
    for dx, dy in offsets:
        origin = clamp_text_origin((anchor[0] + dx, anchor[1] + dy), text, image.shape, font_scale, thickness)
        rect = padded_rect(text_rect(text, origin, font_scale, thickness), 4)
        if not any(rects_overlap(rect, used) for used in used_rects):
            chosen = origin
            chosen_rect = rect
            break
    if chosen is None:
        chosen = clamp_text_origin((anchor[0] + 12, anchor[1] - 12), text, image.shape, font_scale, thickness)
        chosen_rect = padded_rect(text_rect(text, chosen, font_scale, thickness), 4)
    used_rects.append(chosen_rect)
    cv2.putText(
        image,
        text,
        chosen,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_corner_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    used_rects: list[tuple[int, int, int, int]],
    font_scale: float = 0.75,
    thickness: int = 2,
) -> None:
    origin = clamp_text_origin(origin, text, image.shape, font_scale, thickness)
    rect = padded_rect(text_rect(text, origin, font_scale, thickness), 6)
    while any(rects_overlap(rect, used) for used in used_rects) and origin[1] < image.shape[0] - 16:
        origin = (origin[0], origin[1] + 32)
        rect = padded_rect(text_rect(text, origin, font_scale, thickness), 6)
    used_rects.append(rect)
    x0, y0, x1, y1 = rect
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(image.shape[1] - 1, x1)
    y1 = min(image.shape[0] - 1, y1)
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_summary_panel(
    image: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
    used_rects: list[tuple[int, int, int, int]],
    origin: tuple[int, int] = (20, 40),
    font_scale: float = 0.7,
    thickness: int = 2,
) -> None:
    if not lines:
        return
    sizes = [
        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        for text, _color in lines
    ]
    line_height = max(size[0][1] + size[1] for size in sizes) + 10
    panel_width = max(size[0][0] for size in sizes) + 24
    panel_height = line_height * len(lines) + 14
    x = origin[0]
    y = origin[1]
    panel_rect = (x, y - 24, x + panel_width, y - 24 + panel_height)
    while any(rects_overlap(panel_rect, used) for used in used_rects) and y + panel_height < image.shape[0] - 12:
        y += line_height
        panel_rect = (x, y - 24, x + panel_width, y - 24 + panel_height)
    x0, y0, x1, y1 = panel_rect
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(image.shape[1] - 1, x1)
    y1 = min(image.shape[0] - 1, y1)
    roi = image[y0:y1, x0:x1]
    if roi.size:
        backing = np.full_like(roi, (0, 0, 0))
        cv2.addWeighted(backing, 0.55, roi, 0.45, 0, dst=roi)
    used_rects.append(panel_rect)
    for idx, (text, color) in enumerate(lines):
        cv2.putText(
            image,
            text,
            (x + 12, y + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


SOURCE_POINT_GROUPS = {
    "tape": [
        ("tape_near_points_px", "tape_near_values_m", "tape_near", (0, 255, 255)),
        ("tape_far_points_px", "tape_far_values_m", "tape_far", (255, 0, 255)),
    ],
    "table_edges": [
        ("table_flush_points_px", "table_flush_values_m", "tbl_flush", (0, 140, 255)),
        ("table_perpendicular_points_px", "table_perpendicular_values_m", "tbl_perp", (140, 255, 0)),
    ],
}
SOURCE_AXIS_COLORS = {
    "tape": ((0, 0, 255), (255, 255, 0)),
    "table_edges": ((0, 0, 180), (180, 180, 0)),
}


def draw_tape_calibration_overlay(
    image: np.ndarray,
    playing_surface_edge_points_px: list[tuple[float, float]],
    robot_base_front_points_px: list[tuple[float, float]],
    homography_sources: dict[str, dict],
    per_source_measurements: dict[str, dict],
    board_plane_measurement: dict,
    combined_board_origin_x: float,
    combined_board_origin_y: float,
    combined_robot_origin_to_board_centre_m: float,
    local_grid: dict | None = None,
    global_board_center_px: tuple[int, int] | None = None,
) -> np.ndarray:
    overlay = image.copy()
    used_rects: list[tuple[int, int, int, int]] = []
    groups = [
        ("edge", playing_surface_edge_points_px, (0, 255, 0)),
        ("front", robot_base_front_points_px, (255, 0, 0)),
    ]
    if local_grid is not None:
        groups.append(("grid", local_grid["central_square_corners_px"], (0, 255, 127)))
    for source_name, info in homography_sources.items():
        for points_key, _values_key, label, color in SOURCE_POINT_GROUPS[source_name]:
            groups.append((label, info[points_key], color))

    for _label, points, _color in groups:
        used_rects.extend(
            point_reserved_rect((int(round(x)), int(round(y)))) for x, y in points
        )
    for label, points, color in groups:
        points_i = [(int(round(x)), int(round(y))) for x, y in points]
        if len(points_i) >= 2:
            cv2.polylines(
                overlay,
                [np.asarray(points_i, dtype=np.int32).reshape(-1, 1, 2)],
                False,
                color,
                2,
                cv2.LINE_AA,
            )
        for idx, point in enumerate(points_i, start=1):
            cv2.drawMarker(overlay, point, color, cv2.MARKER_CROSS, 18, 3, cv2.LINE_AA)
            draw_label(overlay, f"{label}{idx}", point, color, used_rects, font_scale=0.5)

    for source_name, info in homography_sources.items():
        for points_key, values_key, _label, color in SOURCE_POINT_GROUPS[source_name]:
            for point, value in zip(info[points_key], info[values_key]):
                point_i = (int(round(point[0])), int(round(point[1])))
                draw_label(overlay, f"{value:.3f}m", point_i, color, used_rects, font_scale=0.6)

    summary_lines = [
        (
            f"edge->centre (perpendicular)={board_plane_measurement['perpendicular_distance_to_edge_line_m']:.3f}m "
            f"(mean-to-points={board_plane_measurement['mean_edge_to_centre_m']:.3f}m)",
            (0, 0, 255),
        ),
    ]

    if local_grid is not None and global_board_center_px is not None:
        local_center_i = (
            int(round(local_grid["board_center_px"][0])),
            int(round(local_grid["board_center_px"][1])),
        )
        global_center_i = (int(round(global_board_center_px[0])), int(round(global_board_center_px[1])))
        delta_px = float(np.hypot(*(np.array(local_center_i) - np.array(global_center_i))))
        cv2.drawMarker(overlay, local_center_i, (0, 255, 127), cv2.MARKER_CROSS, 20, 3, cv2.LINE_AA)
        draw_label(overlay, "board centre (local)", local_center_i, (0, 255, 127), used_rects, font_scale=0.6)
        summary_lines.append(
            (
                f"local vs global board centre: local={local_center_i} global={global_center_i} "
                f"delta={delta_px:.1f}px (lens-distortion drift)",
                (0, 255, 127),
            )
        )
    for source_name, m in per_source_measurements.items():
        info = homography_sources[source_name]
        table_homography = np.asarray(info["homography"], dtype=float)
        om = m["oriented_board_offset_measurement"]
        front_centroid_uv = np.asarray(om["front_centroid_uv_m"], dtype=float)
        forward_unit = np.asarray(om["forward_unit_uv"], dtype=float)
        lateral_unit = np.asarray(om["lateral_unit_uv"], dtype=float)
        axis_len_m = 0.08
        front_centroid_i = inverse_project(tuple(front_centroid_uv), table_homography)
        forward_tip = inverse_project(tuple(front_centroid_uv + forward_unit * axis_len_m), table_homography)
        lateral_tip = inverse_project(tuple(front_centroid_uv + lateral_unit * axis_len_m), table_homography)
        forward_color, lateral_color = SOURCE_AXIS_COLORS[source_name]
        cv2.arrowedLine(overlay, front_centroid_i, forward_tip, forward_color, 3, cv2.LINE_AA, tipLength=0.15)
        cv2.arrowedLine(overlay, front_centroid_i, lateral_tip, lateral_color, 3, cv2.LINE_AA, tipLength=0.15)
        draw_label(overlay, f"{source_name} +x", forward_tip, forward_color, used_rects, font_scale=0.55)
        draw_label(overlay, f"{source_name} +y", lateral_tip, lateral_color, used_rects, font_scale=0.55)

        summary_lines.append(
            (
                f"[{source_name}] front->edge={m['tape_calibrated_measurement']['raw_tape_distance_m']:.3f}m  "
                f"origin={m['robot_origin_to_board_centre_m']:.3f}m  "
                f"x={om['board_origin_x']:.3f} y={om['board_origin_y']:.3f}",
                forward_color,
            )
        )

    summary_lines.append(
        (
            f"COMBINED: x={combined_board_origin_x:.3f} y={combined_board_origin_y:.3f}  "
            f"robot origin->board centre={combined_robot_origin_to_board_centre_m:.3f}m",
            (0, 255, 0),
        )
    )
    draw_summary_panel(overlay, summary_lines, used_rects, origin=(20, 80), font_scale=0.7)
    return overlay


# --- click collection --------------------------------------------------


def draw_click_preview(
    image: np.ndarray,
    marker_points: list[tuple[int, int]],
    prompt: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    view = image.copy()
    used_rects: list[tuple[int, int, int, int]] = []
    used_rects.extend(point_reserved_rect(point) for point in marker_points)
    if len(marker_points) >= 2:
        cv2.polylines(
            view,
            [np.asarray(marker_points, dtype=np.int32).reshape(-1, 1, 2)],
            False,
            color,
            2,
            cv2.LINE_AA,
        )
    for idx, point in enumerate(marker_points, start=1):
        cv2.drawMarker(view, point, color, cv2.MARKER_CROSS, 20, 3, cv2.LINE_AA)
        draw_label(view, str(idx), point, color, used_rects)
    draw_corner_label(view, prompt, (20, 40), color, used_rects, font_scale=0.8)
    return view


def click_points(
    image: np.ndarray,
    prompt: str,
    color: tuple[int, int, int],
    allow_empty: bool,
) -> list[tuple[float, float]] | None:
    marker_points: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        marker_points.append((x, y))
        cv2.imshow(WINDOW_NAME, draw_click_preview(image, marker_points, prompt, color))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    cv2.imshow(WINDOW_NAME, draw_click_preview(image, marker_points, prompt, color))

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(WINDOW_NAME)
            return None
        if key == ord("u") and marker_points:
            marker_points.pop()
            cv2.imshow(WINDOW_NAME, draw_click_preview(image, marker_points, prompt, color))
        if key == ord("c"):
            marker_points.clear()
            cv2.imshow(WINDOW_NAME, draw_click_preview(image, marker_points, prompt, color))
        if key in (ord("s"), 13, 10) and (marker_points or allow_empty):
            cv2.destroyWindow(WINDOW_NAME)
            return [(float(x), float(y)) for x, y in marker_points]


def click_board_corners(image: np.ndarray) -> list[tuple[float, float]] | None:
    return click_points(
        image,
        "click the 4 outer corners of the FULL 8x8 board, in order "
        "(e.g. top-left, top-right, bottom-right, bottom-left); s/enter saves",
        (255, 0, 255),
        allow_empty=False,
    )


def click_playing_surface_edge_points(image: np.ndarray) -> list[tuple[float, float]] | None:
    return click_points(
        image,
        "click >=2 points on the board's own playing-surface edge (the elevated "
        "top surface, not the platform base); s/enter saves",
        (0, 255, 0),
        allow_empty=False,
    )


def click_robot_base_front_points(image: np.ndarray) -> list[tuple[float, float]] | None:
    return click_points(
        image,
        "click >=2 points on the front-most edge of the robot's base plate; "
        "s/enter saves",
        (255, 0, 0),
        allow_empty=False,
    )


def approx_square_size_px(corners: np.ndarray) -> float:
    corners = np.asarray(corners, dtype=float)
    side_lengths = [float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)]
    return float(np.mean(side_lengths)) / 8.0


def click_central_square_corners(
    image: np.ndarray,
    board_center_px: tuple[float, float],
    corners: np.ndarray,
) -> list[tuple[float, float]] | None:
    """Crop and zoom in on the board centre before clicking, since the 9
    target points are a small fraction of the full photo -- zooming makes
    each click meaningfully more precise.
    """
    square_size_px = approx_square_size_px(corners)
    half_size_px = square_size_px * 2.0
    cx, cy = board_center_px
    x0 = max(0, int(round(cx - half_size_px)))
    y0 = max(0, int(round(cy - half_size_px)))
    x1 = min(image.shape[1], int(round(cx + half_size_px)))
    y1 = min(image.shape[0], int(round(cy + half_size_px)))
    crop = image[y0:y1, x0:x1]
    zoom_scale = 900.0 / max(crop.shape[0], crop.shape[1], 1)
    zoomed = cv2.resize(
        crop,
        (max(1, int(round(crop.shape[1] * zoom_scale))), max(1, int(round(crop.shape[0] * zoom_scale)))),
        interpolation=cv2.INTER_CUBIC,
    )

    while True:
        points = click_points(
            zoomed,
            "ZOOMED: click the 9 grid intersections bounding the central 2x2 "
            "block of squares (d4/d5/e4/e5): 3 rows top-to-bottom, 3 points "
            "left-to-right per row, in that order; s/enter saves",
            (0, 255, 0),
            allow_empty=False,
        )
        if points is None:
            return None
        if len(points) == 9:
            return [(x0 + x / zoom_scale, y0 + y / zoom_scale) for x, y in points]
        print(f"Need exactly 9 points (3x3 grid); got {len(points)}. Try again.")


def click_tape_edge_points(image: np.ndarray, edge_name: str) -> list[tuple[float, float]] | None:
    return click_points(
        image,
        f"click >=2 points along the tape measure's {edge_name} edge, at "
        "markings you can read off (you'll be asked for each point's "
        "real-world value next); s/enter saves",
        (0, 255, 255) if edge_name == "near" else (255, 0, 255),
        allow_empty=False,
    )


def click_table_edge_points(image: np.ndarray, edge_name: str) -> list[tuple[float, float]] | None:
    return click_points(
        image,
        f"click >=2 points along the table's {edge_name} edge, at known "
        "measured distances from the shared corner with the other table "
        "edge (you'll be asked for each point's value next); s/enter saves",
        (0, 140, 255) if edge_name == "flush" else (140, 255, 0),
        allow_empty=False,
    )


def prompt_tape_measure_values(count: int, edge_name: str = "") -> list[float]:
    values: list[float] = []
    label = f"{edge_name} edge " if edge_name else ""
    for index in range(1, count + 1):
        while True:
            raw = input(f"Real-world value at {label}tape point {index}/{count} (metres): ").strip()
            try:
                values.append(float(raw))
                break
            except ValueError:
                print(f"Could not parse {raw!r} as a number; try again.")
    return values


def prompt_tape_width_m() -> float:
    while True:
        raw = input("Tape measure's physical width (metres), e.g. 0.025: ").strip()
        try:
            return float(raw)
        except ValueError:
            print(f"Could not parse {raw!r} as a number; try again.")


# --- main run ------------------------------------------------------------


def run(args: argparse.Namespace) -> dict:
    image_path = args.image.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.is_file():
        raise FileNotFoundError(
            "Calibration image not found: "
            f"{image_path}\n"
            "Place the photo at that path, or pass a different file with --image."
        )

    image = load_image(image_path)

    board_meta = detect_board_for_calibration(image_path, image, out_dir, args.board_corners)
    board_size_px = int(board_meta["board_size_px"])
    corners = np.asarray(board_meta["board_corners"], dtype=np.float32)
    transform = board_transform_from_corners(corners, board_size_px=board_size_px)
    board_overlay, board_center_px = draw_board_overlay(
        image, corners, transform.homography, board_size_px,
        square_corners_px=board_meta.get("square_corners"),
    )
    board_overlay_path = out_dir / "robot_to_board_calibration_board_detected.jpg"
    save_debug(board_overlay_path, board_overlay)

    result: dict = {
        "schema": "tape_distance_calibration_v1",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "image": str(image_path),
        "square_length_m": float(args.square_length_m),
        "board_corners_px": corners.astype(float).tolist(),
        "board_center_px": list(board_center_px),
        "playing_surface_edge_points_px": [],
        "robot_base_front_points_px": [],
        "homography_sources": {},
        "per_source_measurements": {},
        "combined_board_origin_x": None,
        "combined_board_origin_y": None,
        "combined_robot_origin_to_board_centre_m": None,
        "output_paths": {
            "board_detected": str(board_overlay_path),
            "tape_measurement": None,
        },
    }

    replay_data = None
    if args.replay_points_from is not None:
        replay_data = json.loads(args.replay_points_from.read_text())

    def replay_points(key: str) -> list[tuple[float, float]] | None:
        if replay_data is None:
            return None
        points = replay_data.get(key)
        if not points:
            return None
        return [(float(x), float(y)) for x, y in points]

    playing_surface_edge_points_px = replay_points("playing_surface_edge_points_px")
    if playing_surface_edge_points_px is None:
        playing_surface_edge_points_px = click_playing_surface_edge_points(board_overlay)
    robot_base_front_points_px = replay_points("robot_base_front_points_px")
    if robot_base_front_points_px is None:
        robot_base_front_points_px = click_robot_base_front_points(board_overlay)

    global_board_center_px = board_center_px
    local_grid = None
    if not args.no_local_grid:
        replayed_local_grid = None if args.reclick_local_grid else (replay_data or {}).get("local_grid")
        if replayed_local_grid:
            local_grid = replayed_local_grid
        else:
            central_square_corners_px = click_central_square_corners(
                board_overlay, global_board_center_px, corners
            )
            if central_square_corners_px:
                local_grid = fit_local_board_homography(central_square_corners_px, args.square_length_m)
        if local_grid:
            board_center_px = (
                int(round(local_grid["board_center_px"][0])),
                int(round(local_grid["board_center_px"][1])),
            )
            print(
                f"Local grid board centre px={board_center_px} vs global (lens-distortion-affected) "
                f"interpolated centre px={tuple(int(round(v)) for v in global_board_center_px)}  "
                f"delta={np.hypot(*(np.array(board_center_px) - np.array(global_board_center_px))):.1f}px"
            )

    result["board_center_px"] = list(board_center_px)
    result["global_board_center_px"] = list(global_board_center_px)
    result["local_grid"] = local_grid

    replayed_sources = (replay_data or {}).get("homography_sources", {})

    homography_sources: dict[str, dict] = {}
    if not args.no_tape:
        if replayed_sources.get("tape"):
            homography_sources["tape"] = replayed_sources["tape"]
        else:
            tape_near_points_px = click_tape_edge_points(board_overlay, "near")
            tape_near_values_m = (
                prompt_tape_measure_values(len(tape_near_points_px), "near") if tape_near_points_px else None
            )
            tape_far_points_px = click_tape_edge_points(board_overlay, "far")
            tape_far_values_m = (
                prompt_tape_measure_values(len(tape_far_points_px), "far") if tape_far_points_px else None
            )
            tape_width_m = args.tape_width_m if args.tape_width_m is not None else prompt_tape_width_m()
            if tape_near_points_px and tape_far_points_px and tape_near_values_m and tape_far_values_m:
                homography_sources["tape"] = fit_table_homography(
                    tape_near_points_px, tape_near_values_m,
                    tape_far_points_px, tape_far_values_m,
                    tape_width_m,
                )

    if not args.no_table_edges:
        if replayed_sources.get("table_edges"):
            homography_sources["table_edges"] = replayed_sources["table_edges"]
        else:
            flush_points_px = click_table_edge_points(board_overlay, "flush")
            flush_values_m = (
                prompt_tape_measure_values(len(flush_points_px), "flush") if flush_points_px else None
            )
            perpendicular_points_px = click_table_edge_points(board_overlay, "perpendicular")
            perpendicular_values_m = (
                prompt_tape_measure_values(len(perpendicular_points_px), "perpendicular")
                if perpendicular_points_px else None
            )
            if flush_points_px and perpendicular_points_px and flush_values_m and perpendicular_values_m:
                homography_sources["table_edges"] = fit_table_edges_homography(
                    flush_points_px, flush_values_m,
                    perpendicular_points_px, perpendicular_values_m,
                )

    if not (playing_surface_edge_points_px and robot_base_front_points_px and homography_sources):
        print("Incomplete points/values; nothing to measure.")
        measurement_path = out_dir / "measurement.json"
        measurement_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["measurement_json"] = str(measurement_path)
        return result

    board_plane_measurement = board_plane_edge_to_centre_measurement(
        playing_surface_edge_points_px,
        transform.homography,
        board_size_px,
        args.square_length_m,
    )
    edge_to_centre_m = (
        board_plane_measurement["perpendicular_distance_to_edge_line_m"]
        if board_plane_measurement["perpendicular_distance_to_edge_line_m"] is not None
        else board_plane_measurement["mean_edge_to_centre_m"]
    )

    per_source_measurements: dict[str, dict] = {}
    for name, info in homography_sources.items():
        table_homography = np.asarray(info["homography"], dtype=float)
        tape_measurement = tape_calibrated_measurement(
            playing_surface_edge_points_px, robot_base_front_points_px, table_homography,
        )
        oriented_measurement = oriented_board_offset_measurement(
            robot_base_front_points_px, board_center_px, table_homography,
        )
        robot_origin_to_board_centre_m = (
            ROBOT_ORIGIN_TO_FRONT_EDGE_M + tape_measurement["raw_tape_distance_m"] + edge_to_centre_m
        )
        per_source_measurements[name] = {
            "tape_calibrated_measurement": tape_measurement,
            "oriented_board_offset_measurement": oriented_measurement,
            "robot_origin_to_board_centre_m": robot_origin_to_board_centre_m,
        }

    combined_board_origin_x = float(
        np.mean([m["oriented_board_offset_measurement"]["board_origin_x"] for m in per_source_measurements.values()])
    )
    combined_board_origin_y = float(
        np.mean([m["oriented_board_offset_measurement"]["board_origin_y"] for m in per_source_measurements.values()])
    )
    combined_robot_origin_to_board_centre_m = float(
        np.mean([m["robot_origin_to_board_centre_m"] for m in per_source_measurements.values()])
    )

    overlay = draw_tape_calibration_overlay(
        board_overlay,
        playing_surface_edge_points_px,
        robot_base_front_points_px,
        homography_sources,
        per_source_measurements,
        board_plane_measurement,
        combined_board_origin_x,
        combined_board_origin_y,
        combined_robot_origin_to_board_centre_m,
        local_grid=local_grid,
        global_board_center_px=global_board_center_px,
    )
    overlay_path = out_dir / "robot_to_board_calibration_tape_measurement.jpg"
    save_debug(overlay_path, overlay)

    result.update(
        {
            "playing_surface_edge_points_px": [
                [float(x), float(y)] for x, y in playing_surface_edge_points_px
            ],
            "robot_base_front_points_px": [
                [float(x), float(y)] for x, y in robot_base_front_points_px
            ],
            "board_center_px": list(board_center_px),
            "global_board_center_px": list(global_board_center_px),
            "local_grid": local_grid,
            "homography_sources": homography_sources,
            "board_plane_edge_to_centre_measurement": board_plane_measurement,
            "per_source_measurements": per_source_measurements,
            "combined_board_origin_x": combined_board_origin_x,
            "combined_board_origin_y": combined_board_origin_y,
            "combined_robot_origin_to_board_centre_m": combined_robot_origin_to_board_centre_m,
        }
    )
    result["output_paths"]["tape_measurement"] = str(overlay_path)

    print(
        "playing-surface edge -> board centre       = "
        f"{board_plane_measurement['perpendicular_distance_to_edge_line_m']:.4f} m "
        f"(perpendicular to fitted edge line; mean-to-clicked-points={board_plane_measurement['mean_edge_to_centre_m']:.4f} m)"
    )
    for name, m in per_source_measurements.items():
        om = m["oriented_board_offset_measurement"]
        print(
            f"[{name}] front->edge={m['tape_calibrated_measurement']['raw_tape_distance_m']:.4f}m  "
            f"robot origin->board centre={m['robot_origin_to_board_centre_m']:.4f}m  "
            f"oriented board_origin x={om['board_origin_x']:.4f} y={om['board_origin_y']:.4f}"
        )
    print(
        f"COMBINED (mean of {len(per_source_measurements)} source(s)): "
        f"board_origin x={combined_board_origin_x:.4f} y={combined_board_origin_y:.4f}  "
        f"robot origin->board centre={combined_robot_origin_to_board_centre_m:.4f}m"
    )

    measurement_path = out_dir / "measurement.json"
    measurement_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["measurement_json"] = str(measurement_path)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
