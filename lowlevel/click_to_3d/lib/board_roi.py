#!/usr/bin/env python3
"""Coarse outer-board quad for gating corner search (stationary view).

Prefer LiveChess2FEN outer corners (same detector as the piece pipelines).
Contour path is a fallback when LC2FEN is unavailable — never the
chessboard-first detector, which can lock onto the wrist-camera cable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_COORD = Path(__file__).resolve().parents[2]
if str(_COORD) not in sys.path:
    sys.path.insert(0, str(_COORD))


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32)
    top = pts[np.argsort(pts[:, 1])[:2]]
    bottom = pts[np.argsort(pts[:, 1])[2:]]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _quad_from_contour(contour: np.ndarray) -> np.ndarray | None:
    if contour is None or len(contour) < 4:
        return None
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps_scale in np.linspace(0.01, 0.12, 24):
        approx = cv2.approxPolyDP(hull, eps_scale * peri, True)
        if len(approx) == 4:
            return _order_quad_points(approx.reshape(4, 2))
    rect = cv2.minAreaRect(contour)
    if rect[1][0] <= 1 or rect[1][1] <= 1:
        return None
    return _order_quad_points(cv2.boxPoints(rect))


def find_board_contour_quad(image_bgr: np.ndarray) -> np.ndarray | None:
    """Largest board-like quadrilateral in the frame, or None."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    h, w = gray.shape[:2]
    min_area = 0.08 * w * h
    best = None
    best_area = 0.0

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge_variants = [
        cv2.Canny(blur, low, high) for low, high in ((30, 100), (50, 150), (80, 200))
    ]
    for edges in edge_variants:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            quad = _quad_from_contour(contour)
            if quad is None:
                continue
            quad_area = cv2.contourArea(quad.reshape(-1, 1, 2).astype(np.int32))
            if quad_area > best_area:
                best_area = quad_area
                best = quad

    if best is not None:
        return best

    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        quad = _quad_from_contour(contour)
        if quad is None:
            continue
        quad_area = cv2.contourArea(quad.reshape(-1, 1, 2).astype(np.int32))
        if quad_area > best_area:
            best_area = quad_area
            best = quad
    return best


def find_board_gate_quad(image_bgr: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Outer board quad for the stationary cable gate.

    Prefers LiveChess2FEN (piece-pipeline detector). Falls back to contour when
    the LC2FEN tree is missing or detection fails. Returns ``(quad, method)``.
    """
    try:
        from board_detect_lc2fen import detect_board_outer_corners

        corners = detect_board_outer_corners(image_bgr)
        if corners is not None and len(corners) == 4:
            return _order_quad_points(np.asarray(corners, dtype=np.float32)), "livechess2fen"
    except Exception:
        pass
    contour = find_board_contour_quad(image_bgr)
    if contour is not None:
        return contour, "contour"
    return None, "none"


BOARD_GATE_PAD_FRAC = 0.09


def pad_quad(quad: np.ndarray, margin_frac: float = BOARD_GATE_PAD_FRAC) -> np.ndarray:
    """Expand a 4-gon outward from its centroid by ``margin_frac`` of each spoke."""
    pts = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    centre = pts.mean(axis=0)
    return (centre + (1.0 + float(margin_frac)) * (pts - centre)).astype(np.float32)


def point_in_quad(point: np.ndarray | tuple[float, float], quad: np.ndarray) -> bool:
    """True if ``point`` lies inside or on the boundary of ``quad``."""
    pt = (float(point[0]), float(point[1]))
    poly = np.asarray(quad, dtype=np.float32).reshape(-1, 1, 2)
    return float(cv2.pointPolygonTest(poly, pt, False)) >= 0.0


def board_aabb(
    quad: np.ndarray,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Axis-aligned crop box of ``quad``, clipped to the image."""
    height, width = image_shape[:2]
    xs = np.asarray(quad, dtype=np.float64).reshape(-1, 2)[:, 0]
    ys = np.asarray(quad, dtype=np.float64).reshape(-1, 2)[:, 1]
    ix0 = int(max(0, np.floor(float(xs.min()))))
    iy0 = int(max(0, np.floor(float(ys.min()))))
    ix1 = int(min(width, np.ceil(float(xs.max()))))
    iy1 = int(min(height, np.ceil(float(ys.max()))))
    if ix1 - ix0 < 32 or iy1 - iy0 < 32:
        return 0, 0, width, height
    return ix0, iy0, ix1, iy1


def board_aabb_with_margin(
    quad: np.ndarray,
    image_shape: tuple[int, ...],
    margin_frac: float = BOARD_GATE_PAD_FRAC,
) -> tuple[int, int, int, int]:
    """AABB of ``pad_quad(quad)`` — kept for callers that want a single call."""
    return board_aabb(pad_quad(quad, margin_frac), image_shape)
