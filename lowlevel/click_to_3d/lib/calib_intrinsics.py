#!/usr/bin/env python3
"""Lens calibration for one camera.

Photograph a printed checkerboard at several angles, then save the camera
matrix and distortion. Frames are undistorted with the new matrix from
getOptimalNewCameraMatrix, which is the matrix later passed to solvePnP.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _checkerboard_object_points(inner_cols: int, inner_rows: int, square_mm: float) -> np.ndarray:
    points = np.zeros((inner_rows * inner_cols, 3), dtype=np.float32)
    grid = np.mgrid[0:inner_cols, 0:inner_rows].T.reshape(-1, 2)
    points[:, :2] = grid * float(square_mm)
    return points


def detect_checkerboard(
    image_bgr: np.ndarray,
    inner_cols: int,
    inner_rows: int,
) -> np.ndarray | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (inner_cols, inner_rows), flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def calibrate_from_images(
    images: list[np.ndarray],
    inner_cols: int,
    inner_rows: int,
    square_mm: float,
    max_reprojection_px: float = 1.0,
) -> dict:
    """Calibrate from checkerboard photos. Refuse a large reprojection error."""
    object_points = []
    image_points = []
    image_size = None
    model = _checkerboard_object_points(inner_cols, inner_rows, square_mm)
    for image in images:
        corners = detect_checkerboard(image, inner_cols, inner_rows)
        if corners is None:
            continue
        object_points.append(model)
        image_points.append(corners)
        image_size = (image.shape[1], image.shape[0])
    if len(object_points) < 4 or image_size is None:
        raise RuntimeError(f"Need at least 4 detected boards, found {len(object_points)}")

    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    if float(rms) > max_reprojection_px:
        raise RuntimeError(
            f"Reprojection error {rms:.3f} px is above {max_reprojection_px:.3f} px"
        )
    new_matrix, _roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, image_size, 1, image_size
    )
    return {
        "K": camera_matrix,
        "dist": dist_coeffs,
        "K_undistorted": new_matrix,
        "image_size": image_size,
        "reprojection_px": float(rms),
        "square_mm": float(square_mm),
    }


def undistort(image_bgr: np.ndarray, calibration: dict) -> np.ndarray:
    return cv2.undistort(
        image_bgr,
        calibration["K"],
        calibration["dist"],
        None,
        calibration["K_undistorted"],
    )


def save_calibration(path: Path, calibration: dict) -> None:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise IOError(f"Could not write {path}")
    storage.write("K", np.asarray(calibration["K"], dtype=np.float64))
    storage.write("dist", np.asarray(calibration["dist"], dtype=np.float64))
    storage.write("K_undistorted", np.asarray(calibration["K_undistorted"], dtype=np.float64))
    storage.write("reprojection_px", float(calibration["reprojection_px"]))
    storage.write("square_mm", float(calibration["square_mm"]))
    storage.release()


def load_calibration(path: Path) -> dict:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise IOError(f"Could not read {path}")
    calibration = {
        "K": storage.getNode("K").mat(),
        "dist": storage.getNode("dist").mat(),
        "K_undistorted": storage.getNode("K_undistorted").mat(),
        "reprojection_px": float(storage.getNode("reprojection_px").real()),
        "square_mm": float(storage.getNode("square_mm").real()),
    }
    storage.release()
    return calibration


# Inner-corner counts, largest first. An 8x8 square board has at most 7x7.
_INNER_GRIDS = (
    (7, 7),
    (7, 6),
    (6, 7),
    (6, 6),
    (6, 5),
    (5, 6),
    (5, 5),
    (5, 4),
    (4, 5),
    (4, 4),
)


def _subpix(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return np.asarray(refined, dtype=np.float64).reshape(-1, 1, 2)


def _find_grid(gray: np.ndarray, search: np.ndarray, inner_cols: int, inner_rows: int) -> np.ndarray | None:
    """Sector-based corners first, then the classic detector.

    The sector-based detector does not repeat the same corners on this
    wood board, so a classic hit is kept when both return a grid.
    """
    pattern = (inner_cols, inner_rows)
    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(search, pattern, classic_flags)
    if found and corners is not None and len(corners) == inner_cols * inner_rows:
        return _subpix(gray, np.asarray(corners, dtype=np.float32))
    sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
    if hasattr(cv2, "CALIB_CB_ACCURACY"):
        sb_flags |= cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(search, pattern, sb_flags)
    if not found or corners is None or len(corners) != inner_cols * inner_rows:
        return None
    return _subpix(gray, np.asarray(corners, dtype=np.float32))


def detect_largest_inner_grid(image_bgr: np.ndarray) -> dict | None:
    """Largest complete inner chessboard the frame still shows.

    Close views and the gripper hide part of an 8x8 board, so the search
    walks from 7x7 inner corners down to 4x4 and keeps the first hit.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    search = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    for inner_cols, inner_rows in _INNER_GRIDS:
        corners = _find_grid(gray, search, inner_cols, inner_rows)
        if corners is None:
            continue
        return {
            "inner_cols": inner_cols,
            "inner_rows": inner_rows,
            "corners": corners,
        }
    return None


def _quadrant_mask(gray: np.ndarray, grid: np.ndarray, radius: float, thresh: float = 15.0) -> np.ndarray:
    """True where a corner has the four-square light/dark pattern.

    The gripper and the piece otherwise satisfy the chessboard detector
    without sitting on an intersection.
    """
    radius_px = max(int(round(radius)), 4)
    height, width = gray.shape
    ok = []
    for x_px, y_px in grid.reshape(-1, 2):
        col = int(round(x_px))
        row = int(round(y_px))
        if not (radius_px <= col < width - radius_px and radius_px <= row < height - radius_px):
            ok.append(False)
            continue
        patch = gray[row - radius_px : row + radius_px, col - radius_px : col + radius_px].astype(np.float32)
        upper_left = float(patch[:radius_px, :radius_px].mean())
        upper_right = float(patch[:radius_px, radius_px:].mean())
        lower_left = float(patch[radius_px:, :radius_px].mean())
        lower_right = float(patch[radius_px:, radius_px:].mean())
        opposite = 0.5 * (abs(upper_left - lower_right) + abs(upper_right - lower_left))
        adjacent = 0.25 * (
            abs(upper_left - upper_right)
            + abs(upper_left - lower_left)
            + abs(lower_right - upper_right)
            + abs(lower_right - lower_left)
        )
        ok.append(adjacent - opposite > thresh)
    return np.asarray(ok, dtype=bool).reshape(grid.shape[:2])


def _largest_clean_block(mask: np.ndarray, min_side: int = 4) -> tuple[int, int, int, int] | None:
    rows, cols = mask.shape
    best = None
    for row0 in range(rows):
        for row1 in range(row0 + min_side, rows + 1):
            for col0 in range(cols):
                for col1 in range(col0 + min_side, cols + 1):
                    if not mask[row0:row1, col0:col1].all():
                        continue
                    area = (row1 - row0) * (col1 - col0)
                    if best is None or area > best[0]:
                        best = (area, row0, row1, col0, col1)
    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


def _steps_consistent(grid: np.ndarray) -> bool:
    """Rows and columns each keep a single direction. The board may be rotated."""

    def axis_ok(lines: list[np.ndarray]) -> bool:
        directions = []
        for line in lines:
            steps = np.diff(line, axis=0)
            if np.any(np.linalg.norm(steps, axis=1) < 2.0):
                return False
            for step in steps[1:]:
                if float(np.dot(step, steps[0])) <= 0.0:
                    return False
            directions.append(steps.mean(axis=0))
        directions = np.asarray(directions)
        return bool(np.all(directions @ directions[0] > 0.0))

    n_rows, n_cols = grid.shape[:2]
    return axis_ok([grid[row] for row in range(n_rows)]) and axis_ok(
        [grid[:, col] for col in range(n_cols)]
    )


def detect_usable_inner_grid(image_bgr: np.ndarray) -> dict | None:
    """Largest inner grid whose corners are real intersections.

    The detector's first hit can include the gripper. This walks every size
    from 7x7 down to 4x4, drops corners that fail the four-quadrant test,
    and keeps the largest remaining block whose rows and columns step one way.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    search = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    best = None
    for inner_cols, inner_rows in _INNER_GRIDS:
        corners = _find_grid(gray, search, inner_cols, inner_rows)
        if corners is None:
            continue
        grid = corners.reshape(inner_rows, inner_cols, 2)
        steps = np.concatenate(
            [
                np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
                np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
            ]
        )
        spacing = float(np.median(steps))
        if spacing < 8.0:
            continue
        mask = _quadrant_mask(gray, grid, 0.28 * spacing)
        block = _largest_clean_block(mask)
        if block is None:
            continue
        row0, row1, col0, col1 = block
        piece = grid[row0:row1, col0:col1]
        if not _steps_consistent(piece):
            continue
        area = int(piece.shape[0] * piece.shape[1])
        if best is not None and area <= best["corners_kept"]:
            continue
        best = {
            "inner_cols": int(piece.shape[1]),
            "inner_rows": int(piece.shape[0]),
            "corners": piece.reshape(-1, 1, 2).astype(np.float64),
            "corners_kept": area,
            "detected_cols": inner_cols,
            "detected_rows": inner_rows,
        }
    return best


def fisheye_object_points(inner_cols: int, inner_rows: int, square_mm: float) -> np.ndarray:
    """One view's own grid. The view pose absorbs which physical squares these are."""
    points = np.zeros((inner_rows * inner_cols, 1, 3), dtype=np.float64)
    grid = np.mgrid[0:inner_cols, 0:inner_rows].T.reshape(-1, 2)
    points[:, 0, :2] = grid * float(square_mm)
    return points


def _fisheye_initial(image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    width, height = image_size
    camera = np.eye(3, dtype=np.float64)
    camera[0, 0] = camera[1, 1] = float(max(width, height))
    camera[0, 2] = width / 2.0
    camera[1, 2] = height / 2.0
    return camera, np.zeros((4, 1), dtype=np.float64)


def _fit_fisheye(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray, list, list]:
    """OpenCV fisheye fit, replaced when the polynomial folds the frame.

    Skew is fixed and extrinsics are recomputed. On this wrist lens the free
    higher-order coefficients fit the corners and then fold the rest of the
    picture, so a monotonic (k1, k2) model is kept instead.
    """
    camera, distortion = _fisheye_initial(image_size)
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
    rms, camera, distortion, rvecs, tvecs = cv2.fisheye.calibrate(
        object_points,
        image_points,
        image_size,
        camera,
        distortion,
        flags=flags,
        criteria=criteria,
    )
    distortion = np.asarray(distortion, dtype=np.float64).reshape(4, 1)
    replaced = False
    if _distortion_folds(camera, distortion, image_size):
        camera, distortion, rvecs, tvecs, rms = _monotonic_fisheye(object_points, image_points, image_size)
        replaced = True
    return float(rms), camera, distortion, list(rvecs), list(tvecs), replaced


def _distortion_folds(camera: np.ndarray, distortion: np.ndarray, image_size: tuple[int, int]) -> bool:
    """True when the Kannala-Brandt polynomial is not increasing out to the corners."""
    width, height = image_size
    focal = float(min(camera[0, 0], camera[1, 1]))
    if focal < 1.0 or focal > 2.5 * max(width, height):
        return True
    radius = float(np.hypot(width / 2.0, height / 2.0) / focal)
    theta_max = float(np.arctan(radius))
    coeffs = np.asarray(distortion, dtype=np.float64).reshape(-1)
    coeffs = np.pad(coeffs, (0, max(0, 4 - len(coeffs))))[:4]
    thetas = np.linspace(0.0, theta_max, 50)
    units = thetas ** 2
    # d/dtheta of theta*(1 + k1 u + k2 u^2 + k3 u^3 + k4 u^4)
    derivative = (
        1.0
        + 3.0 * coeffs[0] * units
        + 5.0 * coeffs[1] * units ** 2
        + 7.0 * coeffs[2] * units ** 3
        + 9.0 * coeffs[3] * units ** 4
    )
    return bool(np.any(derivative <= 0.0))


def _pose_reprojection(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera: np.ndarray,
    distortion: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    try:
        ok, rvec, tvec = cv2.fisheye.solvePnP(
            object_points,
            image_points,
            camera,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None
    if not ok:
        return None
    projected, _ = cv2.fisheye.projectPoints(object_points, rvec, tvec, camera, distortion)
    residual = image_points.reshape(-1, 2) - projected.reshape(-1, 2)
    error = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return error, rvec, tvec


def _monotonic_fisheye(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, list, list, float]:
    """Search focal length and k1, k2 for a distortion that stays invertible."""
    width, height = image_size
    best = None
    for focal in (700.0, 1000.0, 1400.0, 1800.0, 2400.0):
        for k1 in (-1.6, -1.2, -0.8, -0.4, -0.1, 0.1):
            for k2 in (-0.5, -0.2, 0.0, 0.3, 0.8):
                camera = np.array(
                    [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
                distortion = np.array([[k1], [k2], [0.0], [0.0]], dtype=np.float64)
                if _distortion_folds(camera, distortion, image_size):
                    continue
                poses = [
                    _pose_reprojection(obj, img, camera, distortion)
                    for obj, img in zip(object_points, image_points)
                ]
                if any(pose is None for pose in poses):
                    continue
                error = float(np.mean([pose[0] for pose in poses]))
                if best is None or error < best[0]:
                    best = (error, camera, distortion, [pose[1] for pose in poses], [pose[2] for pose in poses])
    if best is None:
        raise RuntimeError("No monotonic fisheye model fitted")
    error, camera, distortion, rvecs, tvecs = best
    return camera, distortion, rvecs, tvecs, error


def fisheye_view_error_px(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera: np.ndarray,
    distortion: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    projected, _ = cv2.fisheye.projectPoints(
        np.asarray(object_points, dtype=np.float64),
        np.asarray(rvec, dtype=np.float64),
        np.asarray(tvec, dtype=np.float64),
        camera,
        distortion,
    )
    residual = image_points.reshape(-1, 2) - projected.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def _view_errors(object_points, image_points, camera, distortion, rvecs, tvecs) -> list[float]:
    return [
        fisheye_view_error_px(obj, img, camera, distortion, rvec, tvec)
        for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs)
    ]


def calibrate_fisheye(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
    square_mm: float,
    target_reprojection_px: float = 1.0,
) -> dict:
    """Fit the Kannala-Brandt fisheye model. A high rms is returned, not refused.

    One view whose error sits far above the others is dropped, and the fit
    is run once more. ``meets_target`` records the 1 px printed-pattern rule.
    """
    if len(object_points) < 6:
        raise RuntimeError(f"Need at least 6 grids, found {len(object_points)}")
    rms, camera, distortion, rvecs, tvecs, replaced = _fit_fisheye(object_points, image_points, image_size)
    errors = _view_errors(object_points, image_points, camera, distortion, rvecs, tvecs)
    median = float(np.median(errors))
    worst = int(np.argmax(errors))
    far = errors[worst] > max(2.5 * median, median + 1.0)
    dropped = None
    if far and len(object_points) - 1 >= 6:
        keep = [index for index in range(len(object_points)) if index != worst]
        rms, camera, distortion, rvecs, tvecs, replaced = _fit_fisheye(
            [object_points[index] for index in keep],
            [image_points[index] for index in keep],
            image_size,
        )
        errors = _view_errors(
            [object_points[index] for index in keep],
            [image_points[index] for index in keep],
            camera,
            distortion,
            rvecs,
            tvecs,
        )
        dropped = worst
        object_points = [object_points[index] for index in keep]
        image_points = [image_points[index] for index in keep]

    new_camera = fisheye_new_camera(camera, distortion, image_size, balance=0.0)
    full_camera = fisheye_new_camera(camera, distortion, image_size, balance=1.0)
    distances = [float(np.linalg.norm(np.asarray(tvec, dtype=np.float64))) for tvec in tvecs]
    return {
        "model": "fisheye",
        "K": camera,
        "D": np.asarray(distortion, dtype=np.float64).reshape(4, 1),
        "K_undistorted": new_camera,
        "K_undistorted_balance1": full_camera,
        "balance": 0.0,
        "image_size": image_size,
        "reprojection_px": float(rms),
        "meets_target": bool(float(rms) <= target_reprojection_px),
        "target_reprojection_px": float(target_reprojection_px),
        "square_mm": float(square_mm),
        "view_errors_px": errors,
        "camera_distance_mm": distances,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "dropped_index": dropped,
        "replaced_folded_fit": replaced,
        "object_points": object_points,
        "image_points": image_points,
    }


def fisheye_new_camera(
    camera: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
    balance: float,
) -> np.ndarray:
    """New camera matrix for fisheye.undistortImage at this balance.

    Balance 0 keeps only valid pixels. Balance 1 keeps the whole frame.
    """
    return cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        np.asarray(camera, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64).reshape(4, 1),
        image_size,
        np.eye(3),
        balance=float(balance),
        new_size=image_size,
        fov_scale=1.0,
    )


def undistort_fisheye(
    image_bgr: np.ndarray,
    calibration: dict,
    balance: float = 0.0,
) -> np.ndarray:
    """Undistort with the fisheye model. ``D`` must not be passed to cv2.undistort."""
    size = (image_bgr.shape[1], image_bgr.shape[0])
    if abs(float(balance) - float(calibration.get("balance", 0.0))) < 1e-6 and "K_undistorted" in calibration:
        new_camera = calibration["K_undistorted"]
    elif abs(float(balance) - 1.0) < 1e-6 and "K_undistorted_balance1" in calibration:
        new_camera = calibration["K_undistorted_balance1"]
    else:
        new_camera = fisheye_new_camera(calibration["K"], calibration["D"], size, balance)
    return cv2.fisheye.undistortImage(
        image_bgr,
        np.asarray(calibration["K"], dtype=np.float64),
        np.asarray(calibration["D"], dtype=np.float64).reshape(4, 1),
        Knew=np.asarray(new_camera, dtype=np.float64),
        new_size=size,
    )


def save_fisheye_calibration(path: Path, calibration: dict) -> None:
    """Write a fisheye file. The model tag keeps ``D`` out of the pinhole loader."""
    path = Path(path)
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise IOError(f"Could not write {path}")
    storage.write("model", "fisheye")
    storage.write("K", np.asarray(calibration["K"], dtype=np.float64))
    storage.write("D", np.asarray(calibration["D"], dtype=np.float64).reshape(4, 1))
    storage.write("K_undistorted", np.asarray(calibration["K_undistorted"], dtype=np.float64))
    storage.write("K_undistorted_balance1", np.asarray(calibration["K_undistorted_balance1"], dtype=np.float64))
    storage.write("balance", float(calibration["balance"]))
    storage.write("reprojection_px", float(calibration["reprojection_px"]))
    storage.write("target_reprojection_px", float(calibration["target_reprojection_px"]))
    storage.write("square_mm", float(calibration["square_mm"]))
    storage.write("image_width", int(calibration["image_size"][0]))
    storage.write("image_height", int(calibration["image_size"][1]))
    storage.release()


def load_fisheye_calibration(path: Path) -> dict:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise IOError(f"Could not read {path}")
    model = storage.getNode("model").string()
    if model != "fisheye":
        storage.release()
        raise RuntimeError(f"{path} is model {model!r}, not fisheye")
    calibration = {
        "model": "fisheye",
        "K": storage.getNode("K").mat(),
        "D": storage.getNode("D").mat(),
        "K_undistorted": storage.getNode("K_undistorted").mat(),
        "K_undistorted_balance1": storage.getNode("K_undistorted_balance1").mat(),
        "balance": float(storage.getNode("balance").real()),
        "reprojection_px": float(storage.getNode("reprojection_px").real()),
        "target_reprojection_px": float(storage.getNode("target_reprojection_px").real()),
        "square_mm": float(storage.getNode("square_mm").real()),
        "image_size": (
            int(storage.getNode("image_width").real()),
            int(storage.getNode("image_height").real()),
        ),
        "meets_target": float(storage.getNode("reprojection_px").real())
        <= float(storage.getNode("target_reprojection_px").real()),
    }
    storage.release()
    return calibration
