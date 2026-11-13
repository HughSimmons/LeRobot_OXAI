#!/usr/bin/env python3
"""Camera poses in the board frame.

The arm camera cannot see the whole board. Its pose for a photo is the
forward-kinematics prediction, refined by solvePnP on the hovered square
and any neighbouring corners actually in frame. The side camera is measured
once and stored.

Pose matrices map points in the source frame into the destination frame:
T_base_camera takes camera-frame points into the robot base, and
T_board_camera takes camera-frame points into the board.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def se3(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float64)
    pose[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return pose


def invert(pose: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    rotation = pose[:3, :3]
    out[:3, :3] = rotation.T
    out[:3, 3] = -rotation.T @ pose[:3, 3]
    return out


def camera_in_board(T_base_camera: np.ndarray, T_base_board: np.ndarray) -> np.ndarray:
    """Kinematics prediction: camera pose expressed in the board frame.

    T_base_camera and T_base_board both map their source points into the
    robot base. The result maps camera-frame points into the board.
    """
    return invert(T_base_board) @ np.asarray(T_base_camera, dtype=np.float64)


def pnp_rt_from_board_camera(T_board_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """solvePnP's rvec/tvec, which map board points into the camera."""
    board_to_camera = invert(T_board_camera)
    rvec, _ = cv2.Rodrigues(board_to_camera[:3, :3])
    tvec = board_to_camera[:3, 3].reshape(3, 1)
    return rvec, tvec


def board_camera_from_pnp(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    return invert(se3(rotation, np.asarray(tvec, dtype=np.float64).reshape(3)))


def camera_origin_board(T_board_camera: np.ndarray) -> np.ndarray:
    """Camera centre in board millimetres."""
    return np.asarray(T_board_camera, dtype=np.float64)[:3, 3].copy()


def rotation_angle_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    relative = pose_a[:3, :3].T @ pose_b[:3, :3]
    trace = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(trace))


def square_file_rank(algebraic: str) -> tuple[int, int]:
    name = algebraic.strip().lower()
    if len(name) != 2 or name[0] not in "abcdefgh" or name[1] not in "12345678":
        raise ValueError(f"Not a square name: {algebraic!r}")
    return ord(name[0]) - ord("a"), int(name[1]) - 1


def local_object_corners(
    algebraic: str,
    square_mm: float,
    neighbour_rings: int = 1,
) -> np.ndarray:
    """Board-frame corners of the hovered square and the surrounding grid.

    Grid indices run from 0 at a1 to 8 at the far edges. A ring of 1 adds
    the neighbouring squares' corners.
    """
    file_idx, rank_idx = square_file_rank(algebraic)
    i0 = max(0, file_idx - neighbour_rings)
    i1 = min(8, file_idx + 1 + neighbour_rings)
    j0 = max(0, rank_idx - neighbour_rings)
    j1 = min(8, rank_idx + 1 + neighbour_rings)
    points = []
    for file_i in range(i0, i1 + 1):
        for rank_i in range(j0, j1 + 1):
            points.append([file_i * square_mm, rank_i * square_mm, 0.0])
    return np.asarray(points, dtype=np.float64)


def project_points(
    points_board: np.ndarray,
    T_board_camera: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project board points. Returns pixels and camera-frame depth."""
    rvec, tvec = pnp_rt_from_board_camera(T_board_camera)
    image, _ = cv2.projectPoints(
        np.asarray(points_board, dtype=np.float64),
        rvec,
        tvec,
        np.asarray(camera_matrix, dtype=np.float64),
        None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64),
    )
    board_to_camera = invert(T_board_camera)
    depths = (board_to_camera[:3, :3] @ np.asarray(points_board, dtype=np.float64).T).T
    depths = depths[:, 2] + board_to_camera[2, 3]
    return image.reshape(-1, 2), depths


def pixel_ray_board(
    pixel_xy: tuple[float, float],
    T_board_camera: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Ray origin and unit direction in the board frame through one pixel.

    The pixel is in the undistorted image that matches camera_matrix.
    """
    origin = camera_origin_board(T_board_camera)
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    direction_camera = np.array(
        [
            (float(pixel_xy[0]) - cx) / fx,
            (float(pixel_xy[1]) - cy) / fy,
            1.0,
        ],
        dtype=np.float64,
    )
    rotation_board_camera = T_board_camera[:3, :3]
    direction = rotation_board_camera @ direction_camera
    direction /= np.linalg.norm(direction)
    return origin, direction


def ray_plane_z0(origin: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
    """Intersection of a ray with the board plane z = 0, in front of the camera."""
    if abs(float(direction[2])) < 1e-9:
        return None
    travel = -float(origin[2]) / float(direction[2])
    if travel <= 0.0:
        return None
    return origin + travel * direction


def refine_with_local_pnp(
    prediction: np.ndarray,
    image_corners: np.ndarray,
    object_corners: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    max_rotation_deg: float = 8.0,
    max_origin_mm: float = 20.0,
) -> tuple[np.ndarray, bool]:
    """Refine the kinematics pose with corners of the hovered square.

    image_corners and object_corners are corresponding pixels and board
    points. The result is kept only when it stays near the prediction, so a
    flipped planar solution is discarded.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    image_corners = np.asarray(image_corners, dtype=np.float64).reshape(-1, 2)
    object_corners = np.asarray(object_corners, dtype=np.float64).reshape(-1, 3)
    if len(image_corners) < 4 or len(image_corners) != len(object_corners):
        return prediction, False

    rvec, tvec = pnp_rt_from_board_camera(prediction)
    dist = np.zeros(5) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)
    ok, rvec_out, tvec_out = cv2.solvePnP(
        object_corners,
        image_corners,
        np.asarray(camera_matrix, dtype=np.float64),
        dist,
        rvec,
        tvec,
        True,
        cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return prediction, False

    refined = board_camera_from_pnp(rvec_out, tvec_out)
    angle = rotation_angle_deg(prediction, refined)
    shift = float(np.linalg.norm(camera_origin_board(refined) - camera_origin_board(prediction)))
    if angle > max_rotation_deg or shift > max_origin_mm:
        return prediction, False
    return refined, True


def side_pose_once(
    pattern_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    T_board_pattern: np.ndarray | None = None,
) -> np.ndarray:
    """Side-camera pose in the board frame, from one view of a known pattern.

    pattern_points are in the pattern frame. T_board_pattern maps those
    points into the board (identity when the pattern is already on a1).
    """
    dist = np.zeros(5) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        np.asarray(pattern_points, dtype=np.float64),
        np.asarray(image_points, dtype=np.float64).reshape(-1, 2),
        np.asarray(camera_matrix, dtype=np.float64),
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed for the side camera pattern")
    T_camera_pattern = se3(cv2.Rodrigues(rvec)[0], tvec)
    if T_board_pattern is None:
        T_board_pattern = np.eye(4)
    T_camera_board = T_camera_pattern @ invert(np.asarray(T_board_pattern, dtype=np.float64))
    return invert(T_camera_board)


def save_pose_yaml(path: Path, poses: dict[str, np.ndarray], scalars: dict[str, float] | None = None) -> None:
    path = Path(path)
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise IOError(f"Could not write {path}")
    for name, value in poses.items():
        storage.write(name, np.asarray(value, dtype=np.float64))
    for name, value in (scalars or {}).items():
        storage.write(name, float(value))
    storage.release()


def load_pose_yaml(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise IOError(f"Could not read {path}")
    out = {}
    for name in names:
        node = storage.getNode(name)
        if node.empty():
            raise KeyError(f"{path} has no node {name}")
        out[name] = node.mat()
    storage.release()
    return out
