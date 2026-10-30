#!/usr/bin/env python3
"""OpenCV wireframe view of the board, cameras, and selected point.

Mouse drag orbits. + / - zoom. q quits.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def _rot_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    return ry @ rx


def _project_world(
    points: np.ndarray,
    yaw: float,
    pitch: float,
    distance: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Simple orbit camera looking at board centre."""
    centre = np.array([4 * 41.75 / 2, 4 * 41.75 / 2, 0.0], dtype=np.float64)
    # Use provided square size from points span if available — centre fixed for chess.
    R = _rot_yaw_pitch(yaw, pitch)
    eye = centre + R @ np.array([0.0, 0.0, distance], dtype=np.float64)
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


def draw_scene(scene: dict, yaw: float, pitch: float, distance: float, size: tuple[int, int] = (900, 700)) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), 28, dtype=np.uint8)
    square_mm = float(scene.get("square_mm", 41.75))
    board = 8.0 * square_mm
    # Grid lines.
    pts = []
    for i in range(9):
        pts.append([i * square_mm, 0.0, 0.0])
        pts.append([i * square_mm, board, 0.0])
        pts.append([0.0, i * square_mm, 0.0])
        pts.append([board, i * square_mm, 0.0])
    grid = np.asarray(pts, dtype=np.float64)
    proj = _project_world(grid, yaw, pitch, distance, width, height)
    for i in range(0, len(proj), 2):
        p0 = (int(round(proj[i, 0])), int(round(proj[i, 1])))
        p1 = (int(round(proj[i + 1, 0])), int(round(proj[i + 1, 1])))
        cv2.line(canvas, p0, p1, (90, 90, 90), 1, cv2.LINE_AA)
    # a1 label
    a1 = _project_world(np.array([[square_mm * 0.35, square_mm * 0.35, 0.0]]), yaw, pitch, distance, width, height)[0]
    cv2.putText(canvas, "a1", (int(a1[0]), int(a1[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1, cv2.LINE_AA)
    # Cameras
    for key, color in (("T_board_wrist", (0, 200, 255)), ("T_board_side", (0, 255, 120))):
        pose = np.asarray(scene[key], dtype=np.float64)
        K = np.asarray(scene["K_wrist" if "wrist" in key else "K_side"], dtype=np.float64)
        frustum = _frustum_corners(pose, K)
        pix = _project_world(frustum, yaw, pitch, distance, width, height)
        origin = (int(round(pix[0, 0])), int(round(pix[0, 1])))
        for i in range(1, 5):
            end = (int(round(pix[i, 0])), int(round(pix[i, 1])))
            cv2.line(canvas, origin, end, color, 1, cv2.LINE_AA)
        for i in range(1, 5):
            a = (int(round(pix[i, 0])), int(round(pix[i, 1])))
            b = (int(round(pix[1 + (i % 4), 0])), int(round(pix[1 + (i % 4), 1])))
            cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, origin, 4, color, -1, cv2.LINE_AA)
        label = "wrist" if "wrist" in key else "stationary"
        cv2.putText(canvas, label, (origin[0] + 6, origin[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    # Selected point
    point = scene.get("point_board_mm")
    if point is not None:
        p = np.asarray(point, dtype=np.float64).reshape(1, 3)
        pix = _project_world(p, yaw, pitch, distance, width, height)[0]
        centre = (int(round(pix[0])), int(round(pix[1])))
        cv2.circle(canvas, centre, 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, centre, 10, (255, 255, 255), 1, cv2.LINE_AA)
        # Drop to board
        foot = p.copy()
        foot[0, 2] = 0.0
        foot_pix = _project_world(foot, yaw, pitch, distance, width, height)[0]
        cv2.line(
            canvas,
            centre,
            (int(round(foot_pix[0])), int(round(foot_pix[1]))),
            (0, 0, 180),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "drag orbit  +/- zoom  q quit",
        (12, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    yaw, pitch, distance = 0.8, 0.55, 550.0
    dragging = False
    last = (0, 0)

    def on_mouse(event, x, y, flags, param) -> None:
        nonlocal dragging, last, yaw, pitch
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            last = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            dx = x - last[0]
            dy = y - last[1]
            yaw += dx * 0.01
            pitch = float(np.clip(pitch + dy * 0.01, -1.2, 1.2))
            last = (x, y)

    window = "click3d_scene"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        frame = draw_scene(scene, yaw, pitch, distance)
        cv2.imshow(window, frame)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("+"), ord("=")):
            distance = max(180.0, distance * 0.9)
        if key in (ord("-"), ord("_")):
            distance = min(1400.0, distance * 1.1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
