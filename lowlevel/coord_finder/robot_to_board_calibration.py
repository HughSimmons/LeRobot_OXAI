#!/usr/bin/env python3
"""Measure the robot-to-board-centre distance using a tape measure visible in
a calibration photo, combined with a LiveChess2FEN board-plane homography.

Click three things on the photo: 2+ points on the board's own playing-surface
edge (its outer wooden frame, not the 8x8 grid boundary), 2+ points on the
robot base's front edge, and exactly 2 points on a tape measure lying on the
table (each paired with its real-world reading, typed in after clicking).

Two measurements combine to give the final robot-origin-to-board-centre
distance:
- front edge -> playing-surface edge: the tape's two known readings give a
  direct pixel-to-metres scale; both point groups are projected onto the
  tape's own fitted line. This crosses from the table plane to the elevated
  board plane, so it's a simple cross-check, not fully parallax-corrected.
- playing-surface edge -> board centre: measured precisely on the board
  plane itself (no parallax issue, both points are on the same elevated
  surface) using the LiveChess2FEN-detected board corners/homography and the
  known square length. This is what reveals the true edge-to-centre
  distance, since the outer frame sits further out than the 4-square-length
  distance to the 8x8 grid's own boundary.
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
        "--board-corners",
        nargs=4,
        metavar=("TL", "TR", "BR", "BL"),
        help="Manual board corners as x,y pairs in original image pixels, "
        "if LiveChess2FEN detection is unavailable or wrong.",
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
    return {
        "points_board_xy_m": points_board_xy_m,
        "distances_from_centre_m": distances_m,
        "mean_edge_to_centre_m": float(np.mean(distances_m)),
    }


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


def tape_calibrated_measurement(
    playing_surface_edge_points_px: list[tuple[float, float]],
    robot_base_front_points_px: list[tuple[float, float]],
    tape_measure_points_px: list[tuple[float, float]],
    tape_measure_values_m: list[float],
) -> dict:
    """Distance between the robot's front edge and the board's playing-surface
    edge, using a tape measure visible in the same photo as the metric scale.

    The tape lies flat on the table, so this projects both point groups onto
    the tape's own fitted pixel-space direction and scales by
    metres-per-pixel derived directly from the tape's two known readings --
    no homography needed. That also means the result has a parallax bias for
    whichever segment lies over the elevated playing surface (the tape sits
    lower than that surface); treat this as a quick, self-contained
    cross-check rather than a fully corrected measurement.
    """
    if len(tape_measure_points_px) != 2 or len(tape_measure_values_m) != 2:
        raise ValueError("Tape-measure calibration needs exactly 2 points and 2 values.")
    if len(playing_surface_edge_points_px) < 2 or len(robot_base_front_points_px) < 2:
        raise ValueError("Need >=2 points each for the playing-surface edge and robot base front.")

    tape_a, tape_b = (np.asarray(p, dtype=float) for p in tape_measure_points_px)
    tape_vec = tape_b - tape_a
    tape_pixel_length = float(np.linalg.norm(tape_vec))
    if tape_pixel_length < 1e-6:
        raise ValueError("Tape-measure points are coincident in the image.")
    tape_unit = tape_vec / tape_pixel_length

    value_a, value_b = tape_measure_values_m
    metres_per_pixel = abs(value_b - value_a) / tape_pixel_length

    def projected_value_m(points_px: list[tuple[float, float]]) -> float:
        centroid = np.mean(np.asarray(points_px, dtype=float), axis=0)
        signed_pixels_from_a = float(np.dot(centroid - tape_a, tape_unit))
        # value_a is the reading at tape_a; walking along tape_unit increases
        # the reading toward value_b (or decreases it, if value_b < value_a).
        direction_sign = 1.0 if value_b >= value_a else -1.0
        return value_a + direction_sign * signed_pixels_from_a * metres_per_pixel

    playing_surface_value_m = projected_value_m(playing_surface_edge_points_px)
    robot_front_value_m = projected_value_m(robot_base_front_points_px)
    raw_distance_m = abs(playing_surface_value_m - robot_front_value_m)
    robot_origin_distance_m = raw_distance_m + ROBOT_ORIGIN_TO_FRONT_EDGE_M

    return {
        "tape_measure_points_px": [[float(x), float(y)] for x, y in tape_measure_points_px],
        "tape_measure_values_m": [float(v) for v in tape_measure_values_m],
        "metres_per_pixel": metres_per_pixel,
        "tape_unit_direction_px": tape_unit.tolist(),
        "playing_surface_edge_points_px": [
            [float(x), float(y)] for x, y in playing_surface_edge_points_px
        ],
        "robot_base_front_points_px": [
            [float(x), float(y)] for x, y in robot_base_front_points_px
        ],
        "playing_surface_edge_tape_value_m": playing_surface_value_m,
        "robot_base_front_tape_value_m": robot_front_value_m,
        "raw_tape_distance_m": raw_distance_m,
        "robot_origin_to_front_edge_m": ROBOT_ORIGIN_TO_FRONT_EDGE_M,
        "robot_origin_to_playing_surface_edge_m": robot_origin_distance_m,
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


def draw_tape_calibration_overlay(
    image: np.ndarray,
    playing_surface_edge_points_px: list[tuple[float, float]],
    robot_base_front_points_px: list[tuple[float, float]],
    tape_measurement: dict,
    board_plane_measurement: dict,
    robot_origin_to_board_centre_m: float,
) -> np.ndarray:
    overlay = image.copy()
    used_rects: list[tuple[int, int, int, int]] = []
    groups = [
        ("edge", playing_surface_edge_points_px, (0, 255, 0)),
        ("front", robot_base_front_points_px, (255, 0, 0)),
        ("tape", tape_measurement["tape_measure_points_px"], (0, 255, 255)),
    ]
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
            cv2.circle(overlay, point, 8, color, -1, cv2.LINE_AA)
            draw_label(overlay, f"{label}{idx}", point, color, used_rects, font_scale=0.5)

    for point, value in zip(
        tape_measurement["tape_measure_points_px"], tape_measurement["tape_measure_values_m"]
    ):
        point_i = (int(round(point[0])), int(round(point[1])))
        draw_label(overlay, f"{value:.3f}m", point_i, (0, 255, 255), used_rects, font_scale=0.6)

    summary_lines = [
        (f"front->edge (tape)={tape_measurement['raw_tape_distance_m']:.3f}m", (0, 255, 255)),
        (
            f"edge->centre (board plane)={board_plane_measurement['mean_edge_to_centre_m']:.3f}m",
            (0, 0, 255),
        ),
        (
            f"robot origin->board centre (total)={robot_origin_to_board_centre_m:.3f}m",
            (0, 255, 0),
        ),
    ]
    draw_summary_panel(overlay, summary_lines, used_rects, origin=(20, 80), font_scale=0.75)
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


def click_tape_measure_points(image: np.ndarray) -> list[tuple[float, float]] | None:
    return click_points(
        image,
        "click exactly 2 points on the tape measure's markings (you'll be asked "
        "for each point's real-world value next); s/enter saves",
        (0, 255, 255),
        allow_empty=False,
    )


def prompt_tape_measure_values(count: int) -> list[float]:
    values: list[float] = []
    for index in range(1, count + 1):
        while True:
            raw = input(f"Real-world value at tape point {index}/{count} (metres): ").strip()
            try:
                values.append(float(raw))
                break
            except ValueError:
                print(f"Could not parse {raw!r} as a number; try again.")
    return values


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
        image, corners, transform.homography, board_size_px
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
        "tape_measure_points_px": [],
        "tape_measure_values_m": [],
        "tape_calibrated_measurement": None,
        "board_plane_edge_to_centre_measurement": None,
        "robot_origin_to_board_centre_m": None,
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
    tape_measure_points_px = replay_points("tape_measure_points_px")
    tape_measure_values_m = None
    if tape_measure_points_px is not None and replay_data is not None:
        tape_measure_values_m = replay_data.get("tape_measure_values_m")
    if tape_measure_points_px is None:
        tape_measure_points_px = click_tape_measure_points(board_overlay)
        tape_measure_values_m = (
            prompt_tape_measure_values(len(tape_measure_points_px))
            if tape_measure_points_px
            else None
        )

    if not (
        playing_surface_edge_points_px
        and robot_base_front_points_px
        and tape_measure_points_px
        and tape_measure_values_m
    ):
        print("Incomplete points/values; nothing to measure.")
        measurement_path = out_dir / "measurement.json"
        measurement_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["measurement_json"] = str(measurement_path)
        return result

    tape_measurement = tape_calibrated_measurement(
        playing_surface_edge_points_px,
        robot_base_front_points_px,
        tape_measure_points_px,
        tape_measure_values_m,
    )
    board_plane_measurement = board_plane_edge_to_centre_measurement(
        playing_surface_edge_points_px,
        transform.homography,
        board_size_px,
        args.square_length_m,
    )
    robot_origin_to_board_centre_m = (
        ROBOT_ORIGIN_TO_FRONT_EDGE_M
        + tape_measurement["raw_tape_distance_m"]
        + board_plane_measurement["mean_edge_to_centre_m"]
    )

    overlay = draw_tape_calibration_overlay(
        board_overlay,
        playing_surface_edge_points_px,
        robot_base_front_points_px,
        tape_measurement,
        board_plane_measurement,
        robot_origin_to_board_centre_m,
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
            "tape_measure_points_px": [[float(x), float(y)] for x, y in tape_measure_points_px],
            "tape_measure_values_m": [float(v) for v in tape_measure_values_m],
            "tape_calibrated_measurement": tape_measurement,
            "board_plane_edge_to_centre_measurement": board_plane_measurement,
            "robot_origin_to_board_centre_m": robot_origin_to_board_centre_m,
        }
    )
    result["output_paths"]["tape_measurement"] = str(overlay_path)

    print(f"front edge -> playing-surface edge (tape) = {tape_measurement['raw_tape_distance_m']:.4f} m")
    print(f"playing-surface edge -> board centre       = {board_plane_measurement['mean_edge_to_centre_m']:.4f} m")
    print(f"robot origin -> board centre (total)       = {robot_origin_to_board_centre_m:.4f} m")

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
