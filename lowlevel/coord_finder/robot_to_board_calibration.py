#!/usr/bin/env python3
"""Measure robot-to-board offsets from one calibration photo.

The script detects the chessboard with the local LiveChess2FEN detector,
writes visual verification images, then optionally lets the user click one or
more marker points in the original photo. The clicked points are projected onto
the board plane and reported in metres using the known square length.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
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
DEFAULT_BATCH_IMAGE_DIR = PROJECT_DIR / "sim2realcalims"
DEFAULT_BATCH_OUT_DIR = SCRIPT_DIR / "output" / "robot_to_board_calibration_batch"
DEFAULT_SQUARE_LENGTH_M = 0.04125
DEFAULT_TABLE_FLUSH_EDGE_LENGTH_M = 0.50
MIN_TABLE_HOMOGRAPHY_POINTS = 4
# From so101_new_calib.urdf base_link mesh geometry (base_so101_v2.stl), in the
# base_link frame: |x_min| of the base plate, and the y-separation between its
# two rearmost mounting-hole corners.
ROBOT_ORIGIN_TO_BACK_EDGE_M = 0.02236485131636864
ROBOT_BACK_EDGE_WIDTH_M = 0.08653078228193195
WINDOW_NAME = "robot_to_board_calibration"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_BATCH_IMAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-out-dir", type=Path, default=DEFAULT_BATCH_OUT_DIR)
    parser.add_argument("--square-length-m", type=float, default=DEFAULT_SQUARE_LENGTH_M)
    parser.add_argument(
        "--table-flush-edge-length-m",
        type=float,
        default=DEFAULT_TABLE_FLUSH_EDGE_LENGTH_M,
        help="Physical length spanned by the first shared edge selection; default 0.50 m.",
    )
    parser.add_argument("--marker-name", default="bottom_robot_flush_edge")
    parser.add_argument(
        "--a1-pos",
        choices=("TL", "TR", "BL", "BR"),
        default="BL",
        help="Where a1 appears in the detector's unrotated board image; controls signed dx/dy only.",
    )
    parser.add_argument(
        "--no-click",
        action="store_true",
        help="Only detect the board and write verification images.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process every image in --image-dir, prompting for edge clicks on each.",
    )
    parser.add_argument(
        "--click-table-points",
        action="store_true",
        help="After edge clicks, separately click visible table/robot reference points.",
    )
    parser.add_argument(
        "--click-table-edges",
        action="store_true",
        help=(
            "After measured shared bottom/robot edge endpoint clicks, select right table edge "
            "and elevation-change edge. The table homography is fit from points on both the "
            f"bottom and right edges, so at least {MIN_TABLE_HOMOGRAPHY_POINTS} points combined "
            "(>=2 per edge) are required."
        ),
    )
    parser.add_argument(
        "--replay-points-from",
        type=Path,
        default=None,
        help=(
            "Path to a previous measurement.json; reuse its clicked points "
            "(marker/flush edge, right table edge, elevation edge, table reference) "
            "instead of prompting for new clicks."
        ),
    )
    parser.add_argument(
        "--board-corners",
        nargs=4,
        metavar=("TL", "TR", "BR", "BL"),
        help="Manual board corners as x,y pairs in original image pixels.",
    )
    return parser.parse_args()


def image_files(image_dir: Path) -> list[Path]:
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or "image"


def apply_homography(point_xy: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    point = np.array([[[point_xy[0], point_xy[1]]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(point, homography)[0, 0]
    return float(warped[0]), float(warped[1])


def parse_corner_pair(value: str) -> list[float]:
    try:
        x_text, y_text = value.split(",", maxsplit=1)
        return [float(x_text), float(y_text)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid corner {value!r}; expected x,y"
        ) from exc


def inverse_project(point_xy: tuple[float, float], homography: np.ndarray) -> tuple[int, int]:
    inv_h = np.linalg.inv(homography)
    x, y = apply_homography(point_xy, inv_h)
    return int(round(x)), int(round(y))


def chess_oriented_px(
    board_px: tuple[float, float],
    board_size_px: int,
    a1_pos: str,
) -> tuple[float, float]:
    """Map detector board pixels into chess-oriented pixels.

    Returns coordinates where x=0 is the a-file edge and y=0 is the first-rank
    edge. This matches board_geometry's percentage convention.
    """
    x, y_down = board_px
    y_up = board_size_px - y_down
    pos = a1_pos.upper()
    if pos == "BL":
        return x, y_up
    if pos == "BR":
        return board_size_px - x, y_up
    if pos == "TL":
        return x, board_size_px - y_up
    if pos == "TR":
        return board_size_px - x, board_size_px - y_up
    raise ValueError(f"Unexpected a1_pos: {a1_pos}")


def detector_px_from_chess_oriented_px(
    board_px: tuple[float, float],
    board_size_px: int,
    a1_pos: str,
) -> tuple[float, float]:
    """Inverse of chess_oriented_px."""
    x, y = board_px
    pos = a1_pos.upper()
    if pos == "BL":
        return x, board_size_px - y
    if pos == "BR":
        return board_size_px - x, board_size_px - y
    if pos == "TL":
        return x, y
    if pos == "TR":
        return board_size_px - x, y
    raise ValueError(f"Unexpected a1_pos: {a1_pos}")


def marker_point_measurement(
    marker_px: tuple[float, float],
    homography: np.ndarray,
    board_size_px: int,
    square_length_m: float,
    a1_pos: str,
) -> dict:
    marker_board_unoriented = apply_homography(marker_px, homography)
    marker_board = chess_oriented_px(marker_board_unoriented, board_size_px, a1_pos)
    center_board = (board_size_px / 2.0, board_size_px / 2.0)
    metres_per_px = (8.0 * square_length_m) / board_size_px
    dx_m = (marker_board[0] - center_board[0]) * metres_per_px
    dy_m = (marker_board[1] - center_board[1]) * metres_per_px
    return {
        "marker_board_xy_m": [marker_board[0] * metres_per_px, marker_board[1] * metres_per_px],
        "vector_board_center_to_marker_m": [dx_m, dy_m],
        "distance_m": math.hypot(dx_m, dy_m),
        "marker_board_px_unoriented": list(marker_board_unoriented),
        "marker_board_px_oriented": list(marker_board),
    }


def point_measurements(
    points_px: list[tuple[float, float]],
    homography: np.ndarray,
    board_size_px: int,
    square_length_m: float,
    a1_pos: str,
) -> list[dict]:
    return [
        marker_point_measurement(point, homography, board_size_px, square_length_m, a1_pos)
        for point in points_px
    ]


def edge_measurement(
    marker_points_px: list[tuple[float, float]],
    homography: np.ndarray,
    board_size_px: int,
    square_length_m: float,
    a1_pos: str,
) -> dict:
    point_measurements = [
        marker_point_measurement(point, homography, board_size_px, square_length_m, a1_pos)
        for point in marker_points_px
    ]
    points_board_m = np.asarray([m["marker_board_xy_m"] for m in point_measurements], dtype=float)
    vectors_m = np.asarray([m["vector_board_center_to_marker_m"] for m in point_measurements], dtype=float)
    centroid_board_m = np.mean(points_board_m, axis=0)
    centroid_vector_m = np.mean(vectors_m, axis=0)

    fit = None
    if len(points_board_m) >= 2:
        centered = points_board_m - centroid_board_m
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        if direction[0] < 0:
            direction = -direction
        normal = np.array([-direction[1], direction[0]])
        signed_distance = float(np.dot(-centroid_vector_m, normal))
        projections = centered @ direction
        fit = {
            "centroid_board_xy_m": centroid_board_m.tolist(),
            "unit_direction_board_xy": direction.tolist(),
            "unit_normal_board_xy": normal.tolist(),
            "signed_distance_from_board_center_to_edge_line_m": signed_distance,
            "distance_from_board_center_to_edge_line_m": abs(signed_distance),
            "point_projection_range_m": [float(np.min(projections)), float(np.max(projections))],
        }

    return {
        "point_measurements": point_measurements,
        "edge_points_board_xy_m": points_board_m.tolist(),
        "vectors_board_center_to_edge_points_m": vectors_m.tolist(),
        "edge_centroid_board_xy_m": centroid_board_m.tolist(),
        "vector_board_center_to_edge_centroid_m": centroid_vector_m.tolist(),
        "edge_centroid_distance_m": float(np.linalg.norm(centroid_vector_m)),
        "edge_fit_board_frame": fit,
    }


def fit_line_board_frame(points_board_m: np.ndarray, square_length_m: float) -> dict | None:
    if len(points_board_m) < 2:
        return None
    centroid = np.mean(points_board_m, axis=0)
    centered = points_board_m - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    if direction[0] < 0:
        direction = -direction
    normal = np.array([-direction[1], direction[0]])
    board_center = np.array([4.0 * square_length_m, 4.0 * square_length_m])
    signed_distance = float(np.dot(board_center - centroid, normal))
    projections = centered @ direction
    return {
        "centroid_board_xy_m": centroid.tolist(),
        "unit_direction_board_xy": direction.tolist(),
        "unit_normal_board_xy": normal.tolist(),
        "signed_distance_from_board_center_to_line_m": signed_distance,
        "distance_from_board_center_to_line_m": abs(signed_distance),
        "point_projection_range_m": [float(np.min(projections)), float(np.max(projections))],
    }


def fit_image_line(points_px: list[tuple[float, float]]) -> dict | None:
    if len(points_px) < 2:
        return None
    points = np.asarray(points_px, dtype=float)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    if direction[0] < 0:
        direction = -direction
    normal = np.array([-direction[1], direction[0]])
    projections = centered @ direction
    return {
        "centroid_px": centroid.tolist(),
        "unit_direction_px": direction.tolist(),
        "unit_normal_px": normal.tolist(),
        "point_projection_range_px": [float(np.min(projections)), float(np.max(projections))],
    }


def line_intersection(line_a: dict, line_b: dict) -> tuple[float, float] | None:
    p = np.asarray(line_a["centroid_px"], dtype=float)
    r = np.asarray(line_a["unit_direction_px"], dtype=float)
    q = np.asarray(line_b["centroid_px"], dtype=float)
    s = np.asarray(line_b["unit_direction_px"], dtype=float)
    cross = r[0] * s[1] - r[1] * s[0]
    if abs(cross) < 1e-9:
        return None
    qp = q - p
    t = (qp[0] * s[1] - qp[1] * s[0]) / cross
    hit = p + t * r
    return float(hit[0]), float(hit[1])


def table_basis_from_edges(
    flush_points_px: list[tuple[float, float]],
    perpendicular_points_px: list[tuple[float, float]],
    flush_edge_length_m: float,
    inside_reference_px: tuple[float, float] | None = None,
) -> dict:
    flush_line = fit_image_line(flush_points_px)
    perp_line = fit_image_line(perpendicular_points_px)
    if flush_line is None or perp_line is None:
        raise ValueError("Need at least two points on each table edge.")
    total_points = len(flush_points_px) + len(perpendicular_points_px)
    if total_points < MIN_TABLE_HOMOGRAPHY_POINTS:
        raise ValueError(
            f"Need at least {MIN_TABLE_HOMOGRAPHY_POINTS} table-edge points combined "
            f"(>=2 on the bottom edge, >=2 on the right edge) to fit the table homography; "
            f"got {total_points}."
        )

    bottom_right_px = line_intersection(flush_line, perp_line)
    if bottom_right_px is None:
        raise ValueError("Table edge lines are nearly parallel; cannot build table frame.")

    flush_points = np.asarray(flush_points_px, dtype=float)
    bottom_right = np.asarray(bottom_right_px, dtype=float)
    line_axis = np.asarray(flush_line["unit_direction_px"], dtype=float)
    flush_projections = (flush_points - bottom_right) @ line_axis
    flush_min_idx = int(np.argmin(flush_projections))
    flush_max_idx = int(np.argmax(flush_projections))
    flush_span_px = float(flush_projections[flush_max_idx] - flush_projections[flush_min_idx])
    if flush_span_px <= 1e-9:
        raise ValueError("Clicked flush-edge points have no measurable span.")
    metres_per_px = flush_edge_length_m / flush_span_px
    scale_endpoint_points = [flush_points[flush_min_idx], flush_points[flush_max_idx]]
    endpoint_distances = [float(np.linalg.norm(point - bottom_right)) for point in scale_endpoint_points]
    bottom_left = scale_endpoint_points[int(np.argmax(endpoint_distances))]
    x_axis_px = bottom_right - bottom_left
    x_norm = float(np.linalg.norm(x_axis_px))
    if x_norm <= 1e-9:
        raise ValueError("Could not orient table x axis from bottom-left to bottom-right.")
    x_axis_px = x_axis_px / x_norm

    y_axis_px = np.asarray(perp_line["unit_direction_px"], dtype=float)
    # Diagnostic only: a perspective photo need not show physically perpendicular
    # edges as perpendicular in pixel space, so this does not gate anything.
    orthogonality_warning = abs(float(np.dot(x_axis_px, y_axis_px))) > 0.35
    if inside_reference_px is not None:
        inside_delta = np.asarray(inside_reference_px, dtype=float) - bottom_left
        if float(inside_delta @ y_axis_px) < 0:
            y_axis_px = -y_axis_px

    # Build known table-plane (metres) correspondences for every clicked edge point,
    # then fit a real image->table homography so perspective is corrected properly
    # instead of forcing the drawn axes to be perpendicular in pixel space.
    perp_points = np.asarray(perpendicular_points_px, dtype=float)
    flush_world_xy = [
        [float((point - bottom_left) @ x_axis_px * metres_per_px), 0.0]
        for point in flush_points
    ]
    perp_world_xy = [
        [
            float(flush_edge_length_m),
            float((point - bottom_right) @ y_axis_px * metres_per_px),
        ]
        for point in perp_points
    ]
    image_points = np.concatenate([flush_points, perp_points], axis=0).astype(np.float32)
    world_points = np.asarray(flush_world_xy + perp_world_xy, dtype=np.float32)
    homography, _ = cv2.findHomography(image_points, world_points)
    if homography is None:
        raise ValueError("Could not fit table homography from the clicked edge points.")

    return {
        "origin_px": bottom_left.tolist(),
        "origin_name": "bottom_left_table_corner",
        "bottom_left_table_corner_px": bottom_left.tolist(),
        "bottom_right_table_corner_px": list(bottom_right_px),
        "x_axis_unit_px": x_axis_px.tolist(),
        "y_axis_unit_px": y_axis_px.tolist(),
        "metres_per_px": metres_per_px,
        "flush_span_px": flush_span_px,
        "flush_edge_length_m": flush_edge_length_m,
        "flush_scale_endpoint_indices": [flush_min_idx, flush_max_idx],
        "flush_scale_endpoint_points_px": [
            flush_points[flush_min_idx].tolist(),
            flush_points[flush_max_idx].tolist(),
        ],
        "scale_assumption": (
            "The extreme selected shared bottom-table/robot-back edge points span "
            "flush_edge_length_m; intermediate points only improve the fitted line."
        ),
        "table_y_scale_assumption": (
            "The right table edge has no independently measured physical length, so its "
            "points are assigned table-plane y using the bottom edge's metres_per_px scale "
            "before the homography fit; treat table-plane y as approximate near the shared "
            "corner rather than independently calibrated."
        ),
        "flush_line_px": flush_line,
        "perpendicular_line_px": perp_line,
        "orthogonality_warning": orthogonality_warning,
        "homography": homography.tolist(),
        "homography_point_count": int(len(image_points)),
    }


def image_point_to_table_xy_m(point_px: tuple[float, float], table_basis: dict) -> list[float]:
    homography = np.asarray(table_basis["homography"], dtype=float)
    x, y = apply_homography(point_px, homography)
    return [x, y]


def table_plane_measurement(
    board_center_px: tuple[int, int],
    robot_edge_points_px: list[tuple[float, float]],
    table_flush_points_px: list[tuple[float, float]],
    table_perpendicular_points_px: list[tuple[float, float]],
    flush_edge_length_m: float,
) -> dict:
    table_basis = table_basis_from_edges(
        table_flush_points_px,
        table_perpendicular_points_px,
        flush_edge_length_m,
        board_center_px,
    )
    board_center_table_xy_m = image_point_to_table_xy_m(board_center_px, table_basis)
    robot_points_table_xy_m = [
        image_point_to_table_xy_m(point, table_basis)
        for point in robot_edge_points_px
    ]
    robot_points = np.asarray(robot_points_table_xy_m, dtype=float)
    robot_centroid = np.mean(robot_points, axis=0)
    vector = robot_centroid - np.asarray(board_center_table_xy_m, dtype=float)
    return {
        "table_basis": table_basis,
        "board_center_table_xy_m": board_center_table_xy_m,
        "bottom_left_table_corner_to_board_center_table_m": board_center_table_xy_m,
        "bottom_left_table_corner_to_board_center_table_dx_m": board_center_table_xy_m[0],
        "bottom_left_table_corner_to_board_center_table_dy_m": board_center_table_xy_m[1],
        "robot_edge_points_table_xy_m": robot_points_table_xy_m,
        "robot_edge_centroid_table_xy_m": robot_centroid.tolist(),
        "vector_board_center_to_robot_edge_centroid_table_m": vector.tolist(),
        "distance_board_center_to_robot_edge_centroid_table_m": float(np.linalg.norm(vector)),
    }


def bottom_left_corner_to_board_center_board_plane_measurement(
    table_basis: dict,
    homography: np.ndarray,
    board_size_px: int,
    square_length_m: float,
    a1_pos: str,
) -> dict:
    corner_px = tuple(float(v) for v in table_basis["origin_px"])
    corner_measurement = marker_point_measurement(
        corner_px,
        homography,
        board_size_px,
        square_length_m,
        a1_pos,
    )
    corner_to_center = (
        -np.asarray(corner_measurement["vector_board_center_to_marker_m"], dtype=float)
    )
    return {
        "bottom_left_table_corner_px": list(corner_px),
        "bottom_left_table_corner_board_xy_m": corner_measurement["marker_board_xy_m"],
        "bottom_left_table_corner_to_board_center_board_m": corner_to_center.tolist(),
        "bottom_left_table_corner_to_board_center_board_dx_m": float(corner_to_center[0]),
        "bottom_left_table_corner_to_board_center_board_dy_m": float(corner_to_center[1]),
        "bottom_left_table_corner_to_board_center_board_distance_m": (
            float(np.linalg.norm(corner_to_center))
        ),
    }


def fit_metric_line(points_xy_m: list[list[float]] | np.ndarray) -> dict | None:
    points = np.asarray(points_xy_m, dtype=float)
    if len(points) < 2:
        return None
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    if direction[0] < 0:
        direction = -direction
    normal = np.array([-direction[1], direction[0]])
    projections = centered @ direction
    return {
        "centroid_xy_m": centroid.tolist(),
        "unit_direction_xy": direction.tolist(),
        "unit_normal_xy": normal.tolist(),
        "point_projection_range_m": [float(np.min(projections)), float(np.max(projections))],
    }


def signed_distance_point_to_metric_line(point_xy_m: list[float], line: dict) -> float:
    point = np.asarray(point_xy_m, dtype=float)
    centroid = np.asarray(line["centroid_xy_m"], dtype=float)
    normal = np.asarray(line["unit_normal_xy"], dtype=float)
    return float((point - centroid) @ normal)


def orient_line_normal_toward_point(line: dict, point_xy_m: list[float]) -> dict:
    out = dict(line)
    normal = np.asarray(out["unit_normal_xy"], dtype=float)
    signed = signed_distance_point_to_metric_line(point_xy_m, out)
    if signed < 0:
        normal = -normal
        out["unit_normal_xy"] = normal.tolist()
    return out


def elevation_aware_measurement(
    board_center_board_xy_m: list[float],
    robot_edge_points_px: list[tuple[float, float]],
    elevation_edge_points_px: list[tuple[float, float]],
    table_basis: dict,
    homography: np.ndarray,
    board_size_px: int,
    square_length_m: float,
    a1_pos: str,
) -> dict:
    elevation_board_measurements = point_measurements(
        elevation_edge_points_px,
        homography,
        board_size_px,
        square_length_m,
        a1_pos,
    )
    elevation_edge_board_xy_m = [
        measurement["marker_board_xy_m"] for measurement in elevation_board_measurements
    ]
    elevation_line_board = fit_metric_line(elevation_edge_board_xy_m)
    if elevation_line_board is None:
        raise ValueError("Need at least two elevation-change edge points.")
    elevation_line_board = orient_line_normal_toward_point(
        elevation_line_board,
        board_center_board_xy_m,
    )
    board_center_to_elevation_m = abs(
        signed_distance_point_to_metric_line(board_center_board_xy_m, elevation_line_board)
    )

    elevation_edge_table_xy_m = [
        image_point_to_table_xy_m(point, table_basis)
        for point in elevation_edge_points_px
    ]
    robot_edge_table_xy_m = [
        image_point_to_table_xy_m(point, table_basis)
        for point in robot_edge_points_px
    ]
    elevation_line_table = fit_metric_line(elevation_edge_table_xy_m)
    robot_line_table = fit_metric_line(robot_edge_table_xy_m)
    robot_centroid_table = np.mean(np.asarray(robot_edge_table_xy_m, dtype=float), axis=0)
    if elevation_line_table is None:
        raise ValueError("Need at least two table-projected elevation edge points.")
    elevation_line_table = orient_line_normal_toward_point(
        elevation_line_table,
        robot_centroid_table.tolist(),
    )
    table_origin_xy_m = [0.0, 0.0]
    elevation_line_table_from_origin = orient_line_normal_toward_point(
        elevation_line_table,
        table_origin_xy_m,
    )
    table_origin_to_elevation_m = abs(
        signed_distance_point_to_metric_line(table_origin_xy_m, elevation_line_table_from_origin)
    )
    elevation_to_robot_m = abs(
        signed_distance_point_to_metric_line(robot_centroid_table.tolist(), elevation_line_table)
    )
    total_m = board_center_to_elevation_m + elevation_to_robot_m
    bottom_left_to_board_center_elevation_aware_dy_m = (
        table_origin_to_elevation_m + board_center_to_elevation_m
    )

    return {
        "board_center_board_xy_m": board_center_board_xy_m,
        "elevation_edge_points_px": [[float(x), float(y)] for x, y in elevation_edge_points_px],
        "elevation_edge_board_xy_m": elevation_edge_board_xy_m,
        "elevation_edge_table_xy_m": elevation_edge_table_xy_m,
        "robot_edge_table_xy_m": robot_edge_table_xy_m,
        "robot_edge_centroid_table_xy_m": robot_centroid_table.tolist(),
        "elevation_line_board_frame": elevation_line_board,
        "elevation_line_table_frame": elevation_line_table,
        "robot_line_table_frame": robot_line_table,
        "board_center_to_elevation_edge_board_m": board_center_to_elevation_m,
        "bottom_left_table_corner_to_elevation_edge_table_m": table_origin_to_elevation_m,
        "bottom_left_table_corner_to_board_center_elevation_aware_dy_m": (
            bottom_left_to_board_center_elevation_aware_dy_m
        ),
        "elevation_edge_to_robot_edge_table_m": elevation_to_robot_m,
        "elevation_aware_distance_m": total_m,
    }


def reference_points_measurement(
    points_px: list[tuple[float, float]],
    homography: np.ndarray,
    board_size_px: int,
    square_length_m: float,
    a1_pos: str,
) -> dict:
    measurements = point_measurements(points_px, homography, board_size_px, square_length_m, a1_pos)
    points_board_m = np.asarray([m["marker_board_xy_m"] for m in measurements], dtype=float)
    vectors_m = np.asarray([m["vector_board_center_to_marker_m"] for m in measurements], dtype=float)
    distances_m = np.asarray([m["distance_m"] for m in measurements], dtype=float)
    return {
        "point_measurements": measurements,
        "points_board_xy_m": points_board_m.tolist(),
        "vectors_board_center_to_points_m": vectors_m.tolist(),
        "distances_from_board_center_m": distances_m.tolist(),
        "line_fit_board_frame": fit_line_board_frame(points_board_m, square_length_m),
    }


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


def draw_board_overlay(
    image: np.ndarray,
    corners: np.ndarray,
    homography: np.ndarray,
    board_size_px: int,
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

    center_px = inverse_project((board_size_px / 2.0, board_size_px / 2.0), homography)
    cv2.circle(overlay, center_px, 10, (0, 0, 255), -1, cv2.LINE_AA)
    used_rects: list[tuple[int, int, int, int]] = []
    draw_label(overlay, "board centre", center_px, (0, 0, 255), used_rects, font_scale=0.7)
    return overlay, center_px


def draw_metric_grid(
    overlay: np.ndarray,
    homography: np.ndarray,
    spacing_query: float,
    query_range: tuple[float, float, float, float],
    color: tuple[int, int, int],
) -> np.ndarray:
    """Draw a grid of lines evenly spaced in the homography's query units, mapped
    back into image pixels. Diagnostic only: extrapolating a homography well past
    the points used to fit it can be unreliable, so this is for visual inspection,
    not a source of new measurements."""
    x_min, x_max, y_min, y_max = query_range
    h, w = overlay.shape[:2]
    margin_px = 400.0

    def clip(point: tuple[int, int]) -> tuple[int, int]:
        x, y = point
        return (
            int(round(float(np.clip(x, -margin_px, w + margin_px)))),
            int(round(float(np.clip(y, -margin_px, h + margin_px)))),
        )

    xs = np.arange(x_min, x_max + spacing_query * 0.5, spacing_query)
    ys = np.arange(y_min, y_max + spacing_query * 0.5, spacing_query)
    for x in xs:
        p0 = clip(inverse_project((float(x), y_min), homography))
        p1 = clip(inverse_project((float(x), y_max), homography))
        cv2.line(overlay, p0, p1, color, 1, cv2.LINE_AA)
    for y in ys:
        p0 = clip(inverse_project((x_min, float(y)), homography))
        p1 = clip(inverse_project((x_max, float(y)), homography))
        cv2.line(overlay, p0, p1, color, 1, cv2.LINE_AA)
    return overlay


def draw_grid_overlay_diagnostic(
    image: np.ndarray,
    board_homography: np.ndarray,
    board_size_px: int,
    square_length_m: float,
    table_homography: np.ndarray,
    grid_spacing_m: float = 0.05,
) -> np.ndarray:
    overlay = image.copy()
    metres_per_px_board = (8.0 * square_length_m) / board_size_px
    spacing_board_px = grid_spacing_m / metres_per_px_board
    board_color = (200, 200, 200)
    table_color = (0, 140, 255)

    draw_metric_grid(
        overlay,
        board_homography,
        spacing_board_px,
        (-1.5 * board_size_px, 2.5 * board_size_px, -1.5 * board_size_px, 2.5 * board_size_px),
        board_color,
    )
    draw_metric_grid(
        overlay,
        table_homography,
        grid_spacing_m,
        (-0.3, 0.9, -0.3, 0.7),
        table_color,
    )

    used_rects: list[tuple[int, int, int, int]] = []
    draw_label(
        overlay,
        f"gray = board-plane grid ({grid_spacing_m * 100:.0f}cm)",
        (20, 40),
        board_color,
        used_rects,
        font_scale=0.6,
    )
    draw_label(
        overlay,
        f"orange = table-plane grid ({grid_spacing_m * 100:.0f}cm)",
        (20, 75),
        table_color,
        used_rects,
        font_scale=0.6,
    )
    return overlay


def draw_marker_overlay(
    board_overlay: np.ndarray,
    center_px: tuple[int, int],
    marker_points_px: list[tuple[float, float]],
    marker_name: str,
    measurement: dict,
) -> np.ndarray:
    overlay = board_overlay.copy()
    used_rects: list[tuple[int, int, int, int]] = []
    marker_points_i = [(int(round(x)), int(round(y))) for x, y in marker_points_px]
    used_rects.extend(point_reserved_rect(point) for point in marker_points_i)
    if len(marker_points_i) >= 2:
        cv2.polylines(
            overlay,
            [np.asarray(marker_points_i, dtype=np.int32).reshape(-1, 1, 2)],
            False,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
    for idx, point in enumerate(marker_points_i, start=1):
        cv2.circle(overlay, point, 9, (255, 0, 255), -1, cv2.LINE_AA)
        draw_label(overlay, str(idx), point, (255, 0, 255), used_rects)

    marker_i = (
        int(round(float(np.mean([p[0] for p in marker_points_i])))),
        int(round(float(np.mean([p[1] for p in marker_points_i])))),
    )
    cv2.line(overlay, center_px, marker_i, (255, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(overlay, marker_i, 11, (200, 0, 255), 2, cv2.LINE_AA)
    draw_label(overlay, f"{marker_name} centroid", marker_i, (255, 0, 255), used_rects, font_scale=0.7)
    dx_m, dy_m = measurement["vector_board_center_to_edge_centroid_m"]
    label = f"dx={dx_m:.3f}m dy={dy_m:.3f}m d={measurement['edge_centroid_distance_m']:.3f}m"
    draw_summary_panel(overlay, [(label, (255, 0, 255))], used_rects, font_scale=0.8)
    return overlay


def draw_reference_points_overlay(
    image: np.ndarray,
    center_px: tuple[int, int],
    points_px: list[tuple[float, float]],
    measurements: list[dict],
    label_prefix: str,
) -> np.ndarray:
    overlay = image.copy()
    used_rects: list[tuple[int, int, int, int]] = []
    points_i = [(int(round(x)), int(round(y))) for x, y in points_px]
    used_rects.extend(point_reserved_rect(point) for point in points_i)
    if len(points_i) >= 2:
        cv2.polylines(
            overlay,
            [np.asarray(points_i, dtype=np.int32).reshape(-1, 1, 2)],
            False,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
    for idx, point in enumerate(points_i, start=1):
        cv2.line(overlay, center_px, point, (0, 165, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, point, 9, (0, 165, 255), -1, cv2.LINE_AA)
        vector = measurements[idx - 1]["vector_board_center_to_marker_m"]
        text = f"{label_prefix}{idx} dx={vector[0]:.3f} dy={vector[1]:.3f}"
        draw_label(overlay, text, point, (0, 165, 255), used_rects)
    return overlay


def draw_table_edges_overlay(
    image: np.ndarray,
    board_center_px: tuple[int, int],
    robot_edge_points_px: list[tuple[float, float]],
    flush_points_px: list[tuple[float, float]],
    perpendicular_points_px: list[tuple[float, float]],
    table_measurement: dict,
    elevation_points_px: list[tuple[float, float]] | None = None,
    elevation_measurement: dict | None = None,
) -> np.ndarray:
    overlay = image.copy()
    used_rects: list[tuple[int, int, int, int]] = []
    groups = [
        ("bottom_robot", flush_points_px, (255, 0, 255)),
        ("right", perpendicular_points_px, (255, 128, 0)),
    ]
    if elevation_points_px:
        groups.append(("elev", elevation_points_px, (255, 255, 0)))
    for _label, points, _color in groups:
        used_rects.extend(
            point_reserved_rect((int(round(x)), int(round(y))))
            for x, y in points
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
            cv2.circle(overlay, point, 8, color, -1, cv2.LINE_AA)
            draw_label(overlay, f"{label}{idx}", point, color, used_rects, font_scale=0.5)

    origin = table_measurement["table_basis"]["origin_px"]
    origin_i = (int(round(origin[0])), int(round(origin[1])))
    cv2.circle(overlay, origin_i, 10, (0, 255, 255), -1, cv2.LINE_AA)
    draw_label(overlay, "origin bottom_left", origin_i, (0, 255, 255), used_rects, font_scale=0.6)
    bottom_right = table_measurement["table_basis"].get("bottom_right_table_corner_px")
    if bottom_right is not None:
        bottom_right_i = (int(round(bottom_right[0])), int(round(bottom_right[1])))
        cv2.circle(overlay, bottom_right_i, 10, (0, 255, 255), 2, cv2.LINE_AA)
        draw_label(overlay, "bottom_right", bottom_right_i, (0, 255, 255), used_rects, font_scale=0.6)

    table_basis = table_measurement["table_basis"]
    table_homography = np.asarray(table_basis["homography"], dtype=float)
    axis_len_m = 0.15
    x_tip_i = inverse_project((axis_len_m, 0.0), table_homography)
    y_tip_i = inverse_project((0.0, axis_len_m), table_homography)
    cv2.arrowedLine(overlay, origin_i, x_tip_i, (0, 0, 255), 4, cv2.LINE_AA, tipLength=0.18)
    cv2.arrowedLine(overlay, origin_i, y_tip_i, (0, 255, 0), 4, cv2.LINE_AA, tipLength=0.18)
    draw_label(overlay, "+x along bottom edge", x_tip_i, (0, 0, 255), used_rects, font_scale=0.65)
    draw_label(overlay, "+y toward board", y_tip_i, (0, 255, 0), used_rects, font_scale=0.65)

    robot_centroid_px = (
        int(round(float(np.mean([p[0] for p in robot_edge_points_px])))),
        int(round(float(np.mean([p[1] for p in robot_edge_points_px])))),
    )
    cv2.line(overlay, board_center_px, robot_centroid_px, (0, 255, 255), 3, cv2.LINE_AA)
    vector = table_measurement["vector_board_center_to_robot_edge_centroid_table_m"]
    summary_lines = [(f"table dx={vector[0]:.3f}m dy={vector[1]:.3f}m", (0, 255, 255))]
    if elevation_measurement is not None:
        summary_lines.extend(
            [
                (
                    f"elev aware={elevation_measurement['elevation_aware_distance_m']:.3f}m",
                    (255, 255, 0),
                ),
                (
                    f"board seg={elevation_measurement['board_center_to_elevation_edge_board_m']:.3f}m",
                    (255, 255, 0),
                ),
                (
                    f"table seg={elevation_measurement['elevation_edge_to_robot_edge_table_m']:.3f}m",
                    (255, 255, 0),
                ),
            ]
        )
    draw_summary_panel(overlay, summary_lines, used_rects, origin=(20, 80), font_scale=0.75)
    return overlay


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
            "square_corners_count": 0,
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
        "square_corners_count": 0,
        "method": "board_geometry_fallback",
        "fallback_reason": "LiveChess2FEN repo not found",
    }
    meta_path = out_dir / "board_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["meta_path"] = str(meta_path)
    return meta


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
        cv2.circle(view, point, 9, color, -1, cv2.LINE_AA)
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


def click_marker_points(image: np.ndarray) -> list[tuple[float, float]] | None:
    return click_points(
        image,
        (
            "click measured 50cm shared bottom-table / robot-back edge endpoints "
            f"(>=2 pts; >={MIN_TABLE_HOMOGRAPHY_POINTS} total with right edge for table "
            "homography); optional intermediate points"
        ),
        (255, 0, 255),
        allow_empty=False,
    )


def click_table_points(image: np.ndarray) -> list[tuple[float, float]] | None:
    return click_points(
        image,
        "click table/robot reference points; s/enter saves, u undo, c clear, q skip",
        (0, 165, 255),
        allow_empty=True,
    )


def click_table_perpendicular_edge_points(image: np.ndarray) -> list[tuple[float, float]] | None:
    return click_points(
        image,
        (
            f"click >=2 right table edge points (>={MIN_TABLE_HOMOGRAPHY_POINTS} total with "
            "bottom edge for table homography); endpoints preferred if measured; s/enter saves"
        ),
        (255, 128, 0),
        allow_empty=True,
    )


def click_elevation_change_edge_points(image: np.ndarray) -> list[tuple[float, float]] | None:
    return click_points(
        image,
        "click raised-board to table-height transition edge points; s/enter saves",
        (255, 255, 0),
        allow_empty=True,
    )


def run(args: argparse.Namespace) -> dict:
    if args.square_length_m <= 0:
        raise ValueError("--square-length-m must be positive")

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
        image,
        corners,
        transform.homography,
        board_size_px,
    )
    board_overlay_path = out_dir / "robot_to_board_calibration_board_detected.jpg"
    save_debug(board_overlay_path, board_overlay)

    result = {
        "schema": "robot_to_board_photo_calibration_v1",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "image": str(image_path),
        "square_length_m": float(args.square_length_m),
        "table_flush_edge_length_m": float(args.table_flush_edge_length_m),
        "table_scale_assumption": (
            "For table-plane estimates, the shared bottom-table/robot-back edge selection "
            "must include the measured table endpoints spanning table_flush_edge_length_m. "
            "Extra intermediate points only improve the fitted line. The table-plane "
            f"homography needs at least {MIN_TABLE_HOMOGRAPHY_POINTS} points combined across "
            "the bottom and right table edges (>=2 per edge)."
        ),
        "board_corners_px": corners.astype(float).tolist(),
        "board_center_px": list(board_center_px),
        "marker_name": args.marker_name,
        "marker_px": None,
        "marker_points_px": [],
        "marker_board_xy_m": None,
        "edge_points_board_xy_m": [],
        "vector_board_center_to_marker_m": None,
        "vectors_board_center_to_edge_points_m": [],
        "distance_m": None,
        "edge_dx_m": None,
        "edge_fit_board_frame": None,
        "table_reference_points_px": [],
        "table_reference_points_board_xy_m": [],
        "table_reference_vectors_board_center_to_points_m": [],
        "table_reference_distances_from_board_center_m": [],
        "table_reference_line_fit_board_frame": None,
        "table_edge_flush_points_px": [],
        "table_edge_perpendicular_points_px": [],
        "elevation_change_edge_points_px": [],
        "table_plane_measurement": None,
        "table_plane_vs_board_plane_delta_m": None,
        "elevation_aware_measurement": None,
        "elevation_aware_vs_board_plane_delta_m": None,
        "elevation_aware_vs_table_plane_delta_m": None,
        "a1_pos": args.a1_pos,
        "output_paths": {
            "board_detected": str(board_overlay_path),
            "warped_board": board_meta["warped_board"],
            "marker_measurement": None,
            "table_reference_measurement": None,
            "table_edge_measurement": None,
            "elevation_aware_measurement": None,
            "grid_overlay": None,
        },
        "board_meta": board_meta,
    }

    if args.no_click:
        print(json.dumps(result, indent=2))
        return result

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

    marker_points_px = replay_points("marker_points_px")
    if marker_points_px is None:
        marker_points_px = click_marker_points(board_overlay)
    if marker_points_px is None:
        print("No edge marker points saved.")
        return result

    measurement = edge_measurement(
        marker_points_px,
        transform.homography,
        board_size_px,
        args.square_length_m,
        args.a1_pos,
    )
    marker_overlay = draw_marker_overlay(
        board_overlay,
        board_center_px,
        marker_points_px,
        args.marker_name,
        measurement,
    )
    marker_overlay_path = out_dir / "robot_to_board_calibration_marker_measurement.jpg"
    save_debug(marker_overlay_path, marker_overlay)

    centroid_marker_px = [
        float(np.mean([point[0] for point in marker_points_px])),
        float(np.mean([point[1] for point in marker_points_px])),
    ]
    edge_fit = measurement["edge_fit_board_frame"]
    edge_dx_m = (
        edge_fit["distance_from_board_center_to_edge_line_m"]
        if edge_fit is not None
        else measurement["edge_centroid_distance_m"]
    )
    result.update(
        {
            "marker_px": centroid_marker_px,
            "marker_points_px": [[float(x), float(y)] for x, y in marker_points_px],
            "marker_board_xy_m": measurement["edge_centroid_board_xy_m"],
            "edge_points_board_xy_m": measurement["edge_points_board_xy_m"],
            "vector_board_center_to_marker_m": measurement["vector_board_center_to_edge_centroid_m"],
            "vectors_board_center_to_edge_points_m": measurement["vectors_board_center_to_edge_points_m"],
            "distance_m": measurement["edge_centroid_distance_m"],
            "edge_dx_m": edge_dx_m,
            "edge_fit_board_frame": measurement["edge_fit_board_frame"],
            "point_measurements": measurement["point_measurements"],
        }
    )
    result["output_paths"]["marker_measurement"] = str(marker_overlay_path)

    if args.click_table_points:
        table_points_px = replay_points("table_reference_points_px")
        if table_points_px is None:
            table_points_px = click_table_points(marker_overlay)
        if table_points_px:
            table_measurement = reference_points_measurement(
                table_points_px,
                transform.homography,
                board_size_px,
                args.square_length_m,
                args.a1_pos,
            )
            table_overlay = draw_reference_points_overlay(
                marker_overlay,
                board_center_px,
                table_points_px,
                table_measurement["point_measurements"],
                "table",
            )
            table_overlay_path = out_dir / "robot_to_board_calibration_table_reference_measurement.jpg"
            save_debug(table_overlay_path, table_overlay)
            result.update(
                {
                    "table_reference_points_px": [[float(x), float(y)] for x, y in table_points_px],
                    "table_reference_points_board_xy_m": table_measurement["points_board_xy_m"],
                    "table_reference_vectors_board_center_to_points_m": (
                        table_measurement["vectors_board_center_to_points_m"]
                    ),
                    "table_reference_distances_from_board_center_m": (
                        table_measurement["distances_from_board_center_m"]
                    ),
                    "table_reference_line_fit_board_frame": table_measurement["line_fit_board_frame"],
                    "table_reference_point_measurements": table_measurement["point_measurements"],
                }
            )
            result["output_paths"]["table_reference_measurement"] = str(table_overlay_path)
        else:
            print("No table/reference points saved for this image.")

    if args.click_table_edges:
        table_flush_points_px = marker_points_px
        table_perpendicular_points_px = replay_points("table_edge_perpendicular_points_px")
        if table_perpendicular_points_px is None:
            table_perpendicular_points_px = click_table_perpendicular_edge_points(marker_overlay)
        elevation_change_points_px = None
        if table_perpendicular_points_px:
            elevation_change_points_px = replay_points("elevation_change_edge_points_px")
            if elevation_change_points_px is None:
                elevation_change_points_px = click_elevation_change_edge_points(marker_overlay)

        if table_perpendicular_points_px:
            table_plane = table_plane_measurement(
                board_center_px,
                marker_points_px,
                table_flush_points_px,
                table_perpendicular_points_px,
                args.table_flush_edge_length_m,
            )
            bottom_left_board_plane = bottom_left_corner_to_board_center_board_plane_measurement(
                table_plane["table_basis"],
                transform.homography,
                board_size_px,
                args.square_length_m,
                args.a1_pos,
            )
            elevation_aware = None
            if elevation_change_points_px:
                board_center_board_xy_m = [
                    4.0 * args.square_length_m,
                    4.0 * args.square_length_m,
                ]
                elevation_aware = elevation_aware_measurement(
                    board_center_board_xy_m,
                    marker_points_px,
                    elevation_change_points_px,
                    table_plane["table_basis"],
                    transform.homography,
                    board_size_px,
                    args.square_length_m,
                    args.a1_pos,
                )
            table_edge_overlay = draw_table_edges_overlay(
                marker_overlay,
                board_center_px,
                marker_points_px,
                table_flush_points_px,
                table_perpendicular_points_px,
                table_plane,
                elevation_change_points_px,
                elevation_aware,
            )
            table_edge_overlay_path = out_dir / "robot_to_board_calibration_table_edge_measurement.jpg"
            save_debug(table_edge_overlay_path, table_edge_overlay)

            grid_overlay = draw_grid_overlay_diagnostic(
                marker_overlay,
                transform.homography,
                board_size_px,
                args.square_length_m,
                np.asarray(table_plane["table_basis"]["homography"], dtype=float),
            )
            grid_overlay_path = out_dir / "robot_to_board_calibration_grid_overlay.jpg"
            save_debug(grid_overlay_path, grid_overlay)
            result["output_paths"]["grid_overlay"] = str(grid_overlay_path)

            board_plane_vector = np.asarray(result["vector_board_center_to_marker_m"], dtype=float)
            table_plane_vector = np.asarray(
                table_plane["vector_board_center_to_robot_edge_centroid_table_m"],
                dtype=float,
            )
            result.update(
                {
                    "table_edge_flush_points_px": [
                        [float(x), float(y)] for x, y in table_flush_points_px
                    ],
                    "table_edge_perpendicular_points_px": [
                        [float(x), float(y)] for x, y in table_perpendicular_points_px
                    ],
                    "elevation_change_edge_points_px": [
                        [float(x), float(y)] for x, y in (elevation_change_points_px or [])
                    ],
                    "table_plane_measurement": table_plane,
                    "bottom_left_table_corner_to_board_center_board_plane": bottom_left_board_plane,
                    "table_plane_vs_board_plane_delta_m": (
                        table_plane_vector - board_plane_vector
                    ).tolist(),
                }
            )
            if elevation_aware is not None:
                table_distance = table_plane["distance_board_center_to_robot_edge_centroid_table_m"]
                elevation_distance = elevation_aware["elevation_aware_distance_m"]
                result.update(
                    {
                        "elevation_aware_measurement": elevation_aware,
                        "elevation_aware_vs_board_plane_delta_m": (
                            elevation_distance - float(result["edge_dx_m"])
                        ),
                        "elevation_aware_vs_table_plane_delta_m": (
                            elevation_distance - table_distance
                        ),
                    }
                )
                result["output_paths"]["elevation_aware_measurement"] = str(table_edge_overlay_path)
            result["output_paths"]["table_edge_measurement"] = str(table_edge_overlay_path)
        else:
            print("No table-edge homography points saved for this image.")

    measurement_path = out_dir / "measurement.json"
    measurement_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["measurement_json"] = str(measurement_path)
    print(json.dumps(result, indent=2))
    return result


def summarize_batch(results: list[dict], summary_path: Path) -> dict:
    rows = []
    for result in results:
        if result.get("edge_dx_m") is None:
            continue
        vector = result.get("vector_board_center_to_marker_m") or [None, None]
        table_plane = result.get("table_plane_measurement") or {}
        table_vector = table_plane.get("vector_board_center_to_robot_edge_centroid_table_m")
        table_distance = table_plane.get("distance_board_center_to_robot_edge_centroid_table_m")
        bottom_left_board = result.get("bottom_left_table_corner_to_board_center_board_plane") or {}
        bottom_left_board_vec = bottom_left_board.get(
            "bottom_left_table_corner_to_board_center_board_m"
        )
        bottom_left_table_vec = table_plane.get(
            "bottom_left_table_corner_to_board_center_table_m"
        )
        table_delta = result.get("table_plane_vs_board_plane_delta_m")
        elevation_aware = result.get("elevation_aware_measurement") or {}
        elevation_aware_distance = elevation_aware.get("elevation_aware_distance_m")
        bottom_left_elev_dy = elevation_aware.get(
            "bottom_left_table_corner_to_board_center_elevation_aware_dy_m"
        )
        rows.append(
            {
                "image": result["image"],
                "edge_dx_m": float(result["edge_dx_m"]),
                "edge_dx_cm": float(result["edge_dx_m"]) * 100.0,
                "centroid_vector_dx_m": vector[0],
                "centroid_vector_dy_m": vector[1],
                "centroid_distance_m": result.get("distance_m"),
                "table_plane_vector_dx_m": table_vector[0] if table_vector else None,
                "table_plane_vector_dy_m": table_vector[1] if table_vector else None,
                "table_plane_distance_m": table_distance,
                "table_plane_distance_cm": table_distance * 100.0 if table_distance is not None else None,
                "bottom_left_to_board_center_board_dx_m": (
                    bottom_left_board_vec[0] if bottom_left_board_vec else None
                ),
                "bottom_left_to_board_center_board_dy_m": (
                    bottom_left_board_vec[1] if bottom_left_board_vec else None
                ),
                "bottom_left_to_board_center_table_dx_m": (
                    bottom_left_table_vec[0] if bottom_left_table_vec else None
                ),
                "bottom_left_to_board_center_table_dy_m": (
                    bottom_left_table_vec[1] if bottom_left_table_vec else None
                ),
                "bottom_left_to_board_center_elevation_aware_dy_m": bottom_left_elev_dy,
                "table_plane_delta_dx_m": table_delta[0] if table_delta else None,
                "table_plane_delta_dy_m": table_delta[1] if table_delta else None,
                "elevation_aware_distance_m": elevation_aware_distance,
                "elevation_aware_distance_cm": (
                    elevation_aware_distance * 100.0
                    if elevation_aware_distance is not None
                    else None
                ),
                "elevation_aware_vs_board_plane_delta_m": (
                    result.get("elevation_aware_vs_board_plane_delta_m")
                ),
                "elevation_aware_vs_table_plane_delta_m": (
                    result.get("elevation_aware_vs_table_plane_delta_m")
                ),
                "measurement_json": result.get("measurement_json"),
                "marker_points_count": len(result.get("marker_points_px") or []),
            }
        )

    edge_dx_values = np.asarray([row["edge_dx_m"] for row in rows], dtype=float)
    table_values = np.asarray(
        [row["table_plane_distance_m"] for row in rows if row["table_plane_distance_m"] is not None],
        dtype=float,
    )
    elevation_values = np.asarray(
        [
            row["elevation_aware_distance_m"]
            for row in rows
            if row["elevation_aware_distance_m"] is not None
        ],
        dtype=float,
    )
    bottom_left_board_dy_values = np.asarray(
        [
            row["bottom_left_to_board_center_board_dy_m"]
            for row in rows
            if row["bottom_left_to_board_center_board_dy_m"] is not None
        ],
        dtype=float,
    )
    bottom_left_table_dy_values = np.asarray(
        [
            row["bottom_left_to_board_center_table_dy_m"]
            for row in rows
            if row["bottom_left_to_board_center_table_dy_m"] is not None
        ],
        dtype=float,
    )
    bottom_left_elev_dy_values = np.asarray(
        [
            row["bottom_left_to_board_center_elevation_aware_dy_m"]
            for row in rows
            if row["bottom_left_to_board_center_elevation_aware_dy_m"] is not None
        ],
        dtype=float,
    )
    summary = {
        "schema": "robot_to_board_photo_calibration_batch_v1",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "count": int(len(rows)),
        "edge_dx_m_mean": float(np.mean(edge_dx_values)) if len(edge_dx_values) else None,
        "edge_dx_m_std": float(np.std(edge_dx_values)) if len(edge_dx_values) else None,
        "edge_dx_cm_mean": float(np.mean(edge_dx_values) * 100.0) if len(edge_dx_values) else None,
        "edge_dx_cm_std": float(np.std(edge_dx_values) * 100.0) if len(edge_dx_values) else None,
        "table_plane_distance_m_mean": float(np.mean(table_values)) if len(table_values) else None,
        "table_plane_distance_m_std": float(np.std(table_values)) if len(table_values) else None,
        "elevation_aware_distance_m_mean": float(np.mean(elevation_values)) if len(elevation_values) else None,
        "elevation_aware_distance_m_std": float(np.std(elevation_values)) if len(elevation_values) else None,
        "bottom_left_to_board_center_board_dy_m_mean": (
            float(np.mean(bottom_left_board_dy_values)) if len(bottom_left_board_dy_values) else None
        ),
        "bottom_left_to_board_center_board_dy_m_std": (
            float(np.std(bottom_left_board_dy_values)) if len(bottom_left_board_dy_values) else None
        ),
        "bottom_left_to_board_center_table_dy_m_mean": (
            float(np.mean(bottom_left_table_dy_values)) if len(bottom_left_table_dy_values) else None
        ),
        "bottom_left_to_board_center_table_dy_m_std": (
            float(np.std(bottom_left_table_dy_values)) if len(bottom_left_table_dy_values) else None
        ),
        "bottom_left_to_board_center_elevation_aware_dy_m_mean": (
            float(np.mean(bottom_left_elev_dy_values)) if len(bottom_left_elev_dy_values) else None
        ),
        "bottom_left_to_board_center_elevation_aware_dy_m_std": (
            float(np.std(bottom_left_elev_dy_values)) if len(bottom_left_elev_dy_values) else None
        ),
        "measurements": rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_batch_summary(summary: dict) -> None:
    print("\nBatch edge dx summary")
    print("image | board_m | table_m | elev_aware_m | points")
    print("-" * 92)
    for row in summary["measurements"]:
        image_name = Path(row["image"]).name
        table_text = (
            f"{row['table_plane_distance_m']:.6f}"
            if row["table_plane_distance_m"] is not None
            else "n/a"
        )
        elevation_text = (
            f"{row['elevation_aware_distance_m']:.6f}"
            if row["elevation_aware_distance_m"] is not None
            else "n/a"
        )
        print(
            f"{image_name} | "
            f"{row['edge_dx_m']:.6f} | "
            f"{table_text} | "
            f"{elevation_text} | "
            f"{row['marker_points_count']}"
        )
    if summary["edge_dx_m_mean"] is not None:
        print("-" * 92)
        print(
            "board mean +/- std: "
            f"{summary['edge_dx_m_mean']:.6f} +/- {summary['edge_dx_m_std']:.6f} m "
            f"({summary['edge_dx_cm_mean']:.3f} +/- {summary['edge_dx_cm_std']:.3f} cm)"
        )
    if summary["table_plane_distance_m_mean"] is not None:
        print(
            "table mean +/- std: "
            f"{summary['table_plane_distance_m_mean']:.6f} +/- "
            f"{summary['table_plane_distance_m_std']:.6f} m"
        )
    if summary["elevation_aware_distance_m_mean"] is not None:
        print(
            "elevation-aware mean +/- std: "
            f"{summary['elevation_aware_distance_m_mean']:.6f} +/- "
            f"{summary['elevation_aware_distance_m_std']:.6f} m"
        )
    has_bottom_left_dy = any(
        row["bottom_left_to_board_center_board_dy_m"] is not None
        or row["bottom_left_to_board_center_table_dy_m"] is not None
        or row["bottom_left_to_board_center_elevation_aware_dy_m"] is not None
        for row in summary["measurements"]
    )
    if has_bottom_left_dy:
        print("\nBottom-left table corner to board centre dy")
        print("image | board_dy_m | table_dy_m | elev_aware_dy_m")
        print("-" * 92)
        for row in summary["measurements"]:
            image_name = Path(row["image"]).name
            board_dy_text = (
                f"{row['bottom_left_to_board_center_board_dy_m']:.6f}"
                if row["bottom_left_to_board_center_board_dy_m"] is not None
                else "n/a"
            )
            table_dy_text = (
                f"{row['bottom_left_to_board_center_table_dy_m']:.6f}"
                if row["bottom_left_to_board_center_table_dy_m"] is not None
                else "n/a"
            )
            elevation_dy_text = (
                f"{row['bottom_left_to_board_center_elevation_aware_dy_m']:.6f}"
                if row["bottom_left_to_board_center_elevation_aware_dy_m"] is not None
                else "n/a"
            )
            print(f"{image_name} | {board_dy_text} | {table_dy_text} | {elevation_dy_text}")
        if summary["bottom_left_to_board_center_board_dy_m_mean"] is not None:
            print("-" * 92)
            print(
                "board dy mean +/- std: "
                f"{summary['bottom_left_to_board_center_board_dy_m_mean']:.6f} +/- "
                f"{summary['bottom_left_to_board_center_board_dy_m_std']:.6f} m"
            )
        if summary["bottom_left_to_board_center_table_dy_m_mean"] is not None:
            print(
                "table dy mean +/- std: "
                f"{summary['bottom_left_to_board_center_table_dy_m_mean']:.6f} +/- "
                f"{summary['bottom_left_to_board_center_table_dy_m_std']:.6f} m"
            )
        if summary["bottom_left_to_board_center_elevation_aware_dy_m_mean"] is not None:
            print(
                "elevation-aware dy mean +/- std: "
                f"{summary['bottom_left_to_board_center_elevation_aware_dy_m_mean']:.6f} +/- "
                f"{summary['bottom_left_to_board_center_elevation_aware_dy_m_std']:.6f} m"
            )


def run_batch(args: argparse.Namespace) -> dict:
    image_dir = args.image_dir.resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    paths = image_files(image_dir)
    if not paths:
        raise FileNotFoundError(f"No calibration images found in: {image_dir}")

    batch_out_dir = args.batch_out_dir.resolve()
    batch_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Batch images: {len(paths)} from {image_dir}")
    print(
        "For each image: click measured shared bottom-table/robot-back edge endpoints, "
        "right table edge, then elevation-change edge."
    )

    results = []
    for idx, image_path in enumerate(paths, start=1):
        print(f"\n[{idx}/{len(paths)}] {image_path.name}")
        single_args = argparse.Namespace(**vars(args))
        single_args.image = image_path
        single_args.out_dir = batch_out_dir / f"{idx:02d}_{safe_stem(image_path)}"
        result = run(single_args)
        results.append(result)
        if result.get("edge_dx_m") is not None:
            print(f"edge_dx_m: {result['edge_dx_m']:.6f}")
        elif not args.no_click:
            print("No edge measurement saved for this image.")

    summary_path = batch_out_dir / "batch_summary.json"
    summary = summarize_batch(results, summary_path)
    summary["summary_json"] = str(summary_path)
    print_batch_summary(summary)
    print(f"summary_json: {summary_path}")
    return summary


def main() -> int:
    args = parse_args()
    try:
        if args.batch:
            run_batch(args)
        else:
            run(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
