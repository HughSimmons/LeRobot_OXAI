#!/usr/bin/env python3
"""Grasp point from an arm-camera mask and a side-camera profile.

The arm camera supplies the in-plane point. The side camera supplies the
pinch height and pulls that point off the shadow, along its line of sight.
"""

from __future__ import annotations

import numpy as np

from calib_board_pose import (
    camera_origin_board,
    pixel_ray_board,
    project_points,
    ray_plane_z0,
    square_file_rank,
)
from test_03_svm_servo import principal_axes, svm_segment_single


def _square_centers(square_mm: float) -> dict[str, np.ndarray]:
    centers = {}
    for file_idx in range(8):
        for rank_idx in range(8):
            name = chr(ord("a") + file_idx) + str(rank_idx + 1)
            centers[name] = np.array(
                [(file_idx + 0.5) * square_mm, (rank_idx + 0.5) * square_mm],
                dtype=np.float64,
            )
    return centers


def side_view_blocked(
    square: str,
    occupancy: dict[str, bool],
    T_board_side: np.ndarray,
    square_mm: float,
) -> bool:
    """True when a nearer occupied square stands between the tripod and this one."""
    centers = _square_centers(square_mm)
    target = centers[square.lower()]
    camera_xy = camera_origin_board(T_board_side)[:2]
    to_target = target - camera_xy
    span = float(np.linalg.norm(to_target))
    if span < 1e-6:
        return False
    direction = to_target / span
    for name, occupied in occupancy.items():
        if not occupied or name.lower() == square.lower() or name.lower() not in centers:
            continue
        other = centers[name.lower()]
        along = float(np.dot(other - camera_xy, direction))
        across = float(np.linalg.norm((other - camera_xy) - along * direction))
        if 0.0 < along < span - 0.25 * square_mm and across < 0.65 * square_mm:
            return True
    return False


def _crop_bounds(
    points_px: np.ndarray,
    width: int,
    height: int,
    pad_frac: float = 0.05,
) -> tuple[int, int, int, int] | None:
    finite = points_px[np.isfinite(points_px).all(axis=1)]
    if len(finite) == 0:
        return None
    x0 = float(np.min(finite[:, 0]))
    x1 = float(np.max(finite[:, 0]))
    y0 = float(np.min(finite[:, 1]))
    y1 = float(np.max(finite[:, 1]))
    pad = pad_frac * max(x1 - x0, y1 - y0, 8.0)
    left = int(np.floor(max(0.0, x0 - pad)))
    top = int(np.floor(max(0.0, y0 - pad)))
    right = int(np.ceil(min(width, x1 + pad)))
    bottom = int(np.ceil(min(height, y1 + pad)))
    if right - left < 16 or bottom - top < 16:
        return None
    return left, top, right, bottom


def side_crop_window(
    square: str,
    T_board_side: np.ndarray,
    camera_matrix: np.ndarray,
    image_shape: tuple[int, int],
    square_mm: float,
    piece_height_mm: float = 70.0,
    pad_frac: float = 0.05,
) -> tuple[int, int, int, int] | None:
    """Image window covering the square on the table and the piece above it.

    The window is the axis-aligned box of the projected rectangular prism
    (square corners at z=0 and z=piece_height_mm), padded by pad_frac.
    """
    file_idx, rank_idx = square_file_rank(square)
    corners = []
    for file_i in (file_idx, file_idx + 1):
        for rank_i in (rank_idx, rank_idx + 1):
            for height in (0.0, piece_height_mm):
                corners.append([file_i * square_mm, rank_i * square_mm, height])
    pixels, depths = project_points(np.asarray(corners), T_board_side, camera_matrix)
    pixels = pixels[depths > 1.0]
    height, width = image_shape[:2]
    return _crop_bounds(pixels, width, height, pad_frac=pad_frac)


def _outline(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 2))
    return np.stack([xs, ys], axis=1).astype(np.float64)


def profile_landmarks(mask: np.ndarray) -> dict | None:
    """Base, top, and pinch pixels on a side-view silhouette.

    Image y grows downward, so the top of the piece is the smallest y.
    The pinch is the middle of the narrow neck, not the tip of the crown.
    """
    points = _outline(mask)
    if len(points) < 10:
        return None
    top_i = int(np.argmin(points[:, 1]))
    base_i = int(np.argmax(points[:, 1]))
    y_top = float(points[top_i, 1])
    y_base = float(points[base_i, 1])
    # Skip both tips. A rounded crown or foot is narrower than the neck.
    span = y_base - y_top
    crown = y_top + 0.08 * span
    foot_line = y_base - 0.08 * span
    rows = []
    widths = []
    mids = []
    for row in range(int(round(y_top)), int(round(y_base)) + 1):
        cols = points[np.abs(points[:, 1] - row) < 0.5, 0]
        if len(cols) < 2:
            continue
        rows.append(row)
        widths.append(float(cols.max() - cols.min()))
        mids.append(0.5 * (float(cols.max()) + float(cols.min())))
    pinch = points[top_i]
    if rows:
        rows_a = np.asarray(rows)
        widths_a = np.asarray(widths)
        body = (rows_a >= crown) & (rows_a <= foot_line)
        if not np.any(body):
            body = np.ones(len(rows_a), dtype=bool)
        neck = body & (widths_a <= float(widths_a[body].min()) * 1.15)
        neck_rows = rows_a[neck]
        if len(neck_rows):
            # Longest contiguous run, then its middle row.
            breaks = np.where(np.diff(neck_rows) > 1)[0]
            start = 0
            best = neck_rows[:1]
            for end in list(breaks) + [len(neck_rows) - 1]:
                run = neck_rows[start : end + 1]
                if len(run) > len(best):
                    best = run
                start = int(end) + 1
            pinch_row = float(np.median(best))
            match = np.argmin(np.abs(rows_a - pinch_row))
            pinch = np.array([mids[int(match)], pinch_row])
    return {
        "base_px": points[base_i],
        "top_px": points[top_i],
        "pinch_px": pinch,
    }


def closest_ray_to_vertical(
    origin: np.ndarray,
    direction: np.ndarray,
    foot_xy: np.ndarray,
) -> np.ndarray:
    """Point on the ray closest to the vertical line through foot_xy."""
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    horizontal = float(direction[0] ** 2 + direction[1] ** 2)
    if horizontal < 1e-12:
        travel = 0.0
    else:
        travel = -(
            (float(origin[0]) - float(foot_xy[0])) * float(direction[0])
            + (float(origin[1]) - float(foot_xy[1])) * float(direction[1])
        ) / horizontal
    travel = max(travel, 0.0)
    return origin + travel * direction


def closest_point_on_ray_to_ray(
    origin_a: np.ndarray,
    direction_a: np.ndarray,
    origin_b: np.ndarray,
    direction_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Skew-line closest approach.

    Returns (point_on_a, point_on_b, residual_mm). Travel along ray B is
    clamped to be non-negative (camera ray). Travel along ray A is also
    clamped to be non-negative.
    """
    point_a = np.asarray(origin_a, dtype=np.float64).reshape(3)
    point_b = np.asarray(origin_b, dtype=np.float64).reshape(3)
    dir_a = np.asarray(direction_a, dtype=np.float64).reshape(3)
    dir_b = np.asarray(direction_b, dtype=np.float64).reshape(3)
    norm_a = float(np.linalg.norm(dir_a))
    norm_b = float(np.linalg.norm(dir_b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return point_a.copy(), point_b.copy(), float(np.linalg.norm(point_a - point_b))
    dir_a = dir_a / norm_a
    dir_b = dir_b / norm_b
    offset = point_a - point_b
    aa = float(np.dot(dir_a, dir_a))
    bb = float(np.dot(dir_a, dir_b))
    cc = float(np.dot(dir_b, dir_b))
    dd = float(np.dot(dir_a, offset))
    ee = float(np.dot(dir_b, offset))
    denom = aa * cc - bb * bb
    if abs(denom) < 1e-12:
        # Nearly parallel: drop a perpendicular from A origin onto B.
        travel_b = max(ee / cc, 0.0)
        travel_a = 0.0
    else:
        travel_a = (bb * ee - cc * dd) / denom
        travel_b = (aa * ee - bb * dd) / denom
        travel_a = max(travel_a, 0.0)
        travel_b = max(travel_b, 0.0)
    closest_a = point_a + travel_a * dir_a
    closest_b = point_b + travel_b * dir_b
    residual = float(np.linalg.norm(closest_a - closest_b))
    return closest_a, closest_b, residual


def _rays_hit_silhouette(
    mask: np.ndarray,
    T_board_camera: np.ndarray,
    camera_matrix: np.ndarray,
    side_mask: np.ndarray,
    T_board_side: np.ndarray,
    side_matrix: np.ndarray,
    max_height_mm: float = 90.0,
) -> np.ndarray:
    """Keep mask pixels whose ray passes through the side silhouette above the board."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask
    kept = np.zeros(len(xs), dtype=bool)
    side_h, side_w = side_mask.shape[:2]
    heights = np.linspace(2.0, max_height_mm, 12)
    for index, (x_px, y_px) in enumerate(zip(xs, ys)):
        origin, direction = pixel_ray_board((float(x_px), float(y_px)), T_board_camera, camera_matrix)
        if abs(float(direction[2])) < 1e-8:
            continue
        samples = []
        for height in heights:
            travel = (height - float(origin[2])) / float(direction[2])
            if travel <= 0.0:
                continue
            samples.append(origin + travel * direction)
        if not samples:
            continue
        pixels, depths = project_points(np.asarray(samples), T_board_side, side_matrix)
        for (u_px, v_px), depth in zip(pixels, depths):
            col = int(round(u_px))
            row = int(round(v_px))
            if depth > 1.0 and 0 <= col < side_w and 0 <= row < side_h and side_mask[row, col] > 0:
                kept[index] = True
                break
    out = np.zeros_like(mask)
    out[ys[kept], xs[kept]] = 255
    return out


def _board_hits_near_foot(
    mask: np.ndarray,
    T_board_camera: np.ndarray,
    camera_matrix: np.ndarray,
    T_board_side: np.ndarray,
    side_matrix: np.ndarray,
    side_mask: np.ndarray,
    foot_fraction: float = 0.15,
    below_mm: float = 6.0,
) -> np.ndarray:
    """Keep pixels whose board hit projects onto the foot of the side silhouette.

    The accepted columns are the left and right edges of the bottom of the
    silhouette. A hit may sit a few millimetres below the lowest red row,
    where the board hides the contact.
    """
    side_rows, side_cols = np.where(side_mask > 0)
    if len(side_rows) < 10:
        return mask
    y_top = float(side_rows.min())
    y_base = float(side_rows.max())
    band_top = y_base - foot_fraction * (y_base - y_top)
    in_band = side_rows >= band_top
    if int(np.count_nonzero(in_band)) < 2:
        return mask
    col_min = float(side_cols[in_band].min())
    col_max = float(side_cols[in_band].max())

    rows, cols = np.where(mask > 0)
    if len(cols) == 0:
        return mask
    origin = camera_origin_board(T_board_camera)
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    dirs_cam = np.stack(
        [(cols - cx) / fx, (rows - cy) / fy, np.ones(len(cols))],
        axis=1,
    )
    dirs = (T_board_camera[:3, :3] @ dirs_cam.T).T
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    dz = dirs[:, 2]
    valid = np.abs(dz) > 1e-8
    travel = np.zeros(len(cols), dtype=np.float64)
    travel[valid] = -float(origin[2]) / dz[valid]
    valid &= travel > 0.0
    hits = origin + travel[:, None] * dirs
    pixels, depths = project_points(hits, T_board_side, side_matrix)
    margin = np.zeros(len(cols), dtype=np.float64)
    measurable = depths > 1.0
    margin[measurable] = below_mm * float(side_matrix[1, 1]) / depths[measurable]
    near = (
        valid
        & measurable
        & (pixels[:, 0] >= col_min)
        & (pixels[:, 0] <= col_max)
        & (pixels[:, 1] >= band_top)
        & (pixels[:, 1] <= y_base + margin)
    )
    out = np.zeros_like(mask)
    out[rows[near], cols[near]] = 255
    return out


def _segment_or_use(
    image: np.ndarray,
    mask: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict | None]:
    if mask is not None:
        axes = principal_axes(mask)
        return mask, axes
    seg, axes, _meta = svm_segment_single(
        image,
        center_frac=0.28,
        border_frac=0.12,
        svm_c=1.0,
        svm_gamma="scale",
        rng=rng,
        min_seg_pixels=30,
    )
    return seg, axes


def grasp_from_views(
    overhead_bgr: np.ndarray,
    side_bgr: np.ndarray,
    square: str,
    occupancy: dict[str, bool],
    T_board_arm: np.ndarray,
    T_board_side: np.ndarray,
    K_arm: np.ndarray,
    K_side: np.ndarray,
    square_mm: float,
    overhead_mask: np.ndarray | None = None,
    side_mask: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    fuse_mode: str = "overhead_ray",
) -> dict:
    """Return the grasp in board millimetres.

    Pass overhead_mask to skip the arm-camera classifier (the sim does this
    so the shadow can be included on purpose). The side profile is segmented
    unless side_mask is provided.

    fuse_mode (default matches two_view_pipeline_champions.md):
      - \"overhead_ray\": point on the pinch ray closest to the wrist
        mask-centroid ray in 3D (current champion).
      - \"foot_vertical\": point on the pinch ray closest to the vertical
        through the cleaned foot (legacy A/B).
    """
    if fuse_mode not in ("foot_vertical", "overhead_ray"):
        raise ValueError(f"Unknown fuse_mode: {fuse_mode!r}")
    generator = rng if rng is not None else np.random.default_rng(0)
    mask, axes = _segment_or_use(overhead_bgr, overhead_mask, generator)
    result = {
        "square": square,
        "side_used": False,
        "z_mm": None,
        "yaw_deg": None if axes is None else float(axes["grasp_axis_deg"]),
        "x_mm": None,
        "y_mm": None,
        "shadow_pixels_dropped": 0,
        "pnp_note": "pose supplied by caller",
    }
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return result
    raw_mask = mask.copy()
    raw_centroid = np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)
    raw_pixels = int(len(xs))
    origin, direction = pixel_ray_board(
        (float(raw_centroid[0]), float(raw_centroid[1])),
        T_board_arm,
        K_arm,
    )
    foot = ray_plane_z0(origin, direction)
    if foot is None:
        return result
    naive_foot = foot.copy()
    result["x_mm"] = float(foot[0])
    result["y_mm"] = float(foot[1])

    if side_view_blocked(square, occupancy, T_board_side, square_mm):
        return result

    window = side_crop_window(square, T_board_side, K_side, side_bgr.shape, square_mm)
    if window is None and side_mask is None:
        return result
    if side_mask is None:
        left, top, right, bottom = window
        crop = side_bgr[top:bottom, left:right]
        crop_mask, _axes = _segment_or_use(crop, None, generator)
        full_side = np.zeros(side_bgr.shape[:2], dtype=np.uint8)
        full_side[top:bottom, left:right] = crop_mask
        landmarks = profile_landmarks(crop_mask)
        if landmarks is not None:
            for key in ("base_px", "top_px", "pinch_px"):
                landmarks[key] = landmarks[key] + np.array([left, top], dtype=np.float64)
    else:
        full_side = side_mask
        landmarks = profile_landmarks(side_mask)
    if landmarks is None:
        return result

    cleaned = _rays_hit_silhouette(mask, T_board_arm, K_arm, full_side, T_board_side, K_side)
    cleaned = _board_hits_near_foot(cleaned, T_board_arm, K_arm, T_board_side, K_side, full_side)
    dropped = int(np.count_nonzero(mask) - np.count_nonzero(cleaned))
    cleaned_centroid = raw_centroid.copy()
    if np.count_nonzero(cleaned) >= 20:
        mask = cleaned
        result["shadow_pixels_dropped"] = dropped
        ys, xs = np.where(mask > 0)
        cleaned_centroid = np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)
        origin, direction = pixel_ray_board(
            (float(cleaned_centroid[0]), float(cleaned_centroid[1])),
            T_board_arm,
            K_arm,
        )
        foot = ray_plane_z0(origin, direction)
        if foot is None:
            return result
        axes = principal_axes(mask)
    else:
        dropped = 0

    pinch_origin, pinch_dir = pixel_ray_board(
        (float(landmarks["pinch_px"][0]), float(landmarks["pinch_px"][1])),
        T_board_side,
        K_side,
    )
    overhead_origin, overhead_dir = pixel_ray_board(
        (float(cleaned_centroid[0]), float(cleaned_centroid[1])),
        T_board_arm,
        K_arm,
    )
    yaw = None if axes is None else float(axes["grasp_axis_deg"])
    overhead_closest = None
    skew_residual = None
    if fuse_mode == "overhead_ray":
        # Grasp is the point on the stationary pinch ray nearest the wrist ray.
        overhead_closest, closest, skew_residual = closest_point_on_ray_to_ray(
            overhead_origin,
            overhead_dir,
            pinch_origin,
            pinch_dir,
        )
        corrected = closest.copy()
        shift = 0.0
        across_residual = 0.0
        gaze_unit = np.zeros(2, dtype=np.float64)
        height = float(closest[2])
    else:
        # Height comes from the pinch ray. Only the component along the side
        # camera's gaze is allowed to move the overhead foot; the across-view
        # coordinate stays with the overhead mask.
        closest = closest_ray_to_vertical(pinch_origin, pinch_dir, foot[:2])
        gaze = foot[:2] - camera_origin_board(T_board_side)[:2]
        gaze_norm = float(np.linalg.norm(gaze))
        corrected = foot.copy()
        shift = 0.0
        across_residual = 0.0
        gaze_unit = np.zeros(2, dtype=np.float64)
        if gaze_norm > 1e-6:
            gaze_unit = gaze / gaze_norm
            shift = float(np.dot(closest[:2] - foot[:2], gaze_unit))
            corrected[:2] = foot[:2] + shift * gaze_unit
            across = np.array([-gaze_unit[1], gaze_unit[0]])
            across_residual = float(np.dot(closest[:2] - foot[:2], across))
        height = float(closest[2])
    trace = {
        "raw_centroid_px": raw_centroid,
        "cleaned_centroid_px": cleaned_centroid,
        "naive_foot_mm": naive_foot,
        "cleaned_foot_mm": foot.copy(),
        "raw_pixels": raw_pixels,
        "dropped_pixels": int(result["shadow_pixels_dropped"]),
        "kept_pixels": int(np.count_nonzero(mask)),
        "crop_window": None if window is None else [int(v) for v in window],
        "base_px": np.asarray(landmarks["base_px"], dtype=np.float64),
        "top_px": np.asarray(landmarks["top_px"], dtype=np.float64),
        "pinch_px": np.asarray(landmarks["pinch_px"], dtype=np.float64),
        "closest_mm": closest.copy(),
        "gaze_xy": gaze_unit,
        "along_view_shift_mm": shift,
        "across_view_residual_mm": across_residual,
        "yaw_deg": yaw,
        "raw_mask": raw_mask,
        "cleaned_mask": mask.copy(),
        "side_mask": full_side,
        "pinch_origin_mm": pinch_origin.copy(),
        "pinch_dir": pinch_dir.copy(),
        "fuse_mode": fuse_mode,
        "overhead_origin_mm": overhead_origin.copy(),
        "overhead_dir": overhead_dir.copy(),
    }
    if overhead_closest is not None:
        trace["overhead_closest_mm"] = overhead_closest.copy()
        trace["skew_residual_mm"] = float(skew_residual)
    result.update(
        {
            "x_mm": float(corrected[0]),
            "y_mm": float(corrected[1]),
            "z_mm": height,
            "yaw_deg": yaw,
            "side_used": True,
            "shadow_pixels_dropped": result["shadow_pixels_dropped"],
            "fuse_mode": fuse_mode,
            "trace": trace,
        }
    )
    return result


def naive_mask_centre_mm(
    mask: np.ndarray,
    T_board_arm: np.ndarray,
    K_arm: np.ndarray,
) -> np.ndarray | None:
    """Board point of the raw mask centre, before the side camera corrects it."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    origin, direction = pixel_ray_board((float(xs.mean()), float(ys.mean())), T_board_arm, K_arm)
    return ray_plane_z0(origin, direction)
