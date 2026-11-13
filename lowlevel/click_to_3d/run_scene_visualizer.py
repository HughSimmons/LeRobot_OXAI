#!/usr/bin/env python3
"""OpenCV wireframe view of the board, cameras, and selected point.

Mouse drag orbits about board +z. + / - zoom. q quits.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 900, 700


def _orbit_eye(centre: np.ndarray, yaw: float, pitch: float, distance: float) -> np.ndarray:
    """Camera eye on a sphere about board centre. Yaw about +z; pitch is elevation."""
    return centre + distance * np.array(
        [
            math.cos(pitch) * math.sin(yaw),
            math.cos(pitch) * math.cos(yaw),
            math.sin(pitch),
        ],
        dtype=np.float64,
    )


def _project_world(
    points: np.ndarray,
    yaw: float,
    pitch: float,
    distance: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Z-up orbit camera looking at board centre."""
    centre = np.array([4 * 41.75 / 2, 4 * 41.75 / 2, 0.0], dtype=np.float64)
    eye = _orbit_eye(centre, yaw, pitch, distance)
    forward = centre - eye
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    rel = points - eye.reshape(1, 3)
    cam = np.stack([rel @ right, rel @ up, rel @ forward], axis=1)
    f = 0.9 * min(width, height)
    z = np.clip(cam[:, 2], 1e-3, None)
    u = width * 0.5 + f * cam[:, 0] / z
    v = height * 0.5 - f * cam[:, 1] / z
    return np.stack([u, v], axis=1)


def _frustum_corners(T_board_camera: np.ndarray, K: np.ndarray, depth_mm: float = 120.0) -> np.ndarray:
    """Five points: origin + four far corners of a frustum in board mm."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    # Assume 640x480 if principal near centre.
    w, h = cx * 2.0, cy * 2.0
    corners_cam = np.array(
        [
            [0.0, 0.0, 0.0],
            [(0 - cx) / fx * depth_mm, (0 - cy) / fy * depth_mm, depth_mm],
            [(w - cx) / fx * depth_mm, (0 - cy) / fy * depth_mm, depth_mm],
            [(w - cx) / fx * depth_mm, (h - cy) / fy * depth_mm, depth_mm],
            [(0 - cx) / fx * depth_mm, (h - cy) / fy * depth_mm, depth_mm],
        ],
        dtype=np.float64,
    )
    R = T_board_camera[:3, :3]
    t = T_board_camera[:3, 3]
    return (R @ corners_cam.T).T + t.reshape(1, 3)


def _px(point: np.ndarray) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def build_scene_geometry(scene: dict) -> dict:
    """World-space points that do not change with the orbit camera."""
    square_mm = float(scene.get("square_mm", 41.75))
    board = 8.0 * square_mm
    grid = []
    for i in range(9):
        grid.append([i * square_mm, 0.0, 0.0])
        grid.append([i * square_mm, board, 0.0])
        grid.append([0.0, i * square_mm, 0.0])
        grid.append([board, i * square_mm, 0.0])
    grid = np.asarray(grid, dtype=np.float64)
    a1 = np.array([[square_mm * 0.35, square_mm * 0.35, 0.0]], dtype=np.float64)
    wrist = _frustum_corners(
        np.asarray(scene["T_board_wrist"], dtype=np.float64),
        np.asarray(scene["K_wrist"], dtype=np.float64),
    )
    side = _frustum_corners(
        np.asarray(scene["T_board_side"], dtype=np.float64),
        np.asarray(scene["K_side"], dtype=np.float64),
    )
    chunks = [grid, a1, wrist, side]
    has_point = scene.get("point_board_mm") is not None
    if has_point:
        point = np.asarray(scene["point_board_mm"], dtype=np.float64).reshape(1, 3)
        foot = point.copy()
        foot[0, 2] = 0.0
        chunks.extend([point, foot])
    world = np.vstack(chunks)
    n_grid = len(grid)
    i = n_grid
    slices = {
        "grid": slice(0, n_grid),
        "a1": slice(i, i + 1),
        "wrist": slice(i + 1, i + 6),
        "side": slice(i + 6, i + 11),
        "point": None,
        "foot": None,
    }
    i += 11
    if has_point:
        slices["point"] = slice(i, i + 1)
        slices["foot"] = slice(i + 1, i + 2)
    return {"world": world, "slices": slices}


def draw_scene(
    canvas: np.ndarray,
    geometry: dict,
    yaw: float,
    pitch: float,
    distance: float,
    line_type: int,
) -> np.ndarray:
    height, width = canvas.shape[:2]
    canvas[:] = 28
    proj = _project_world(geometry["world"], yaw, pitch, distance, width, height)
    slices = geometry["slices"]

    grid = proj[slices["grid"]]
    for i in range(0, len(grid), 2):
        cv2.line(canvas, _px(grid[i]), _px(grid[i + 1]), (90, 90, 90), 1, line_type)

    a1 = _px(proj[slices["a1"]][0])
    cv2.putText(canvas, "a1", a1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1, line_type)

    for key, color, label in (
        ("wrist", (0, 200, 255), "wrist"),
        ("side", (0, 255, 120), "stationary"),
    ):
        pix = proj[slices[key]]
        origin = _px(pix[0])
        for i in range(1, 5):
            cv2.line(canvas, origin, _px(pix[i]), color, 1, line_type)
        for i in range(1, 5):
            cv2.line(canvas, _px(pix[i]), _px(pix[1 + (i % 4)]), color, 1, line_type)
        cv2.circle(canvas, origin, 4, color, -1, line_type)
        cv2.putText(
            canvas,
            label,
            (origin[0] + 6, origin[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            line_type,
        )

    if slices["point"] is not None:
        centre = _px(proj[slices["point"]][0])
        foot = _px(proj[slices["foot"]][0])
        cv2.circle(canvas, centre, 7, (0, 0, 255), -1, line_type)
        cv2.circle(canvas, centre, 10, (255, 255, 255), 1, line_type)
        cv2.line(canvas, centre, foot, (0, 0, 180), 1, line_type)

    cv2.putText(
        canvas,
        "drag orbit (z-up)  +/- zoom  q quit",
        (12, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        line_type,
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    geometry = build_scene_geometry(scene)
    canvas = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    yaw, pitch, distance = 0.8, 0.55, 550.0
    dragging = False
    last = (0, 0)
    dirty = True

    def on_mouse(event, x, y, flags, param) -> None:
        nonlocal dragging, last, yaw, pitch, dirty
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            last = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            dirty = True
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            dx = x - last[0]
            dy = y - last[1]
            yaw += dx * 0.01
            pitch = float(np.clip(pitch + dy * 0.01, 0.05, 1.45))
            last = (x, y)
            dirty = True

    window = "click3d_scene"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        if dirty:
            line_type = cv2.LINE_8 if dragging else cv2.LINE_AA
            draw_scene(canvas, geometry, yaw, pitch, distance, line_type)
            cv2.imshow(window, canvas)
            dirty = False
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("+"), ord("=")):
            distance = max(180.0, distance * 0.9)
            dirty = True
        if key in (ord("-"), ord("_")):
            distance = min(1400.0, distance * 1.1)
            dirty = True
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
