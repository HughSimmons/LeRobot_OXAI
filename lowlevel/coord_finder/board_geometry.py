"""Shared board geometry utilities for coord_finder scripts.

Coordinate convention (percentage space):
  - (0, 0)   = bottom-left corner of a1
  - (100, 100) = top-right corner of h8
  - x increases toward the h-file; y increases toward rank 8
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

BOARD_SIZE_PX = 800
PATTERN_INNER = (7, 7)


@dataclass
class BoardTransform:
    """Perspective warp from image pixels to a square, axis-aligned board."""

    homography: np.ndarray
    board_size_px: int = BOARD_SIZE_PX

    def warp(self, image: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(
            image,
            self.homography,
            (self.board_size_px, self.board_size_px),
            flags=cv2.INTER_LINEAR,
        )

    def pixel_to_pct(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Convert warped-board pixel coords to 0-100 percentage coords."""
        w = self.board_size_px
        h = self.board_size_px
        x_pct = (x_px / w) * 100.0
        y_pct = ((h - y_px) / h) * 100.0
        return x_pct, y_pct

    def pct_to_pixel(self, x_pct: float, y_pct: float) -> tuple[float, float]:
        w = self.board_size_px
        h = self.board_size_px
        x_px = (x_pct / 100.0) * w
        y_px = h - (y_pct / 100.0) * h
        return x_px, y_px


def parse_square(name: str) -> tuple[int, int]:
    """Parse chess square like 'e4' -> (file_idx, rank_idx) with a1 = (0, 0)."""
    name = name.strip().lower()
    if len(name) != 2:
        raise ValueError(f"Invalid square name: {name!r}")
    file_idx = ord(name[0]) - ord("a")
    rank_idx = int(name[1]) - 1
    if not (0 <= file_idx <= 7 and 0 <= rank_idx <= 7):
        raise ValueError(f"Square out of range: {name!r}")
    return file_idx, rank_idx


def square_name(file_idx: int, rank_idx: int) -> str:
    return f"{chr(ord('a') + file_idx)}{rank_idx + 1}"


def square_roi(file_idx: int, rank_idx: int, board_size_px: int = BOARD_SIZE_PX) -> tuple[int, int, int, int]:
    """Return x0, y0, x1, y1 in warped board pixels (image origin top-left)."""
    sq = board_size_px // 8
    x0 = file_idx * sq
    x1 = (file_idx + 1) * sq
    y1 = board_size_px - rank_idx * sq
    y0 = y1 - sq
    return x0, y0, x1, y1


def reorient_warped_for_a1(image: np.ndarray, a1_pos: str) -> np.ndarray:
    """Rotate/flip a warped board so a1 ends at bottom-left.

    `a1_pos` is where a1 sits in the *current* warped image before remapping:
    TL, TR, BL, or BR.
    """
    pos = a1_pos.upper()
    if pos == "BL":
        return image
    if pos == "BR":
        return cv2.flip(image, 1)
    if pos == "TL":
        return cv2.rotate(image, cv2.ROTATE_180)
    if pos == "TR":
        return cv2.flip(cv2.rotate(image, cv2.ROTATE_180), 1)
    raise ValueError(f"Invalid a1_pos: {a1_pos!r} (expected TL/TR/BL/BR)")


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _preprocess_for_corners(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.equalizeHist(blur)


def _board_corners_from_internal(corners: np.ndarray) -> np.ndarray:
    """Extrapolate outer board corners from a 7x7 internal-corner grid."""
    grid = corners.reshape(7, 7, 2).astype(np.float32)
    right = grid[0, 1] - grid[0, 0]
    down = grid[1, 0] - grid[0, 0]
    tl = grid[0, 0] - 0.5 * right - 0.5 * down
    tr = grid[0, 6] + 0.5 * right - 0.5 * down
    bl = grid[6, 0] - 0.5 * right + 0.5 * down
    br = grid[6, 6] + 0.5 * right + 0.5 * down
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _find_corners_chessboard(gray: np.ndarray) -> np.ndarray | None:
    pre = _preprocess_for_corners(gray)
    flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(pre, PATTERN_INNER, flags)
    if not found:
        found, corners = cv2.findChessboardCorners(pre, PATTERN_INNER, None)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    corners = cv2.cornerSubPix(
        gray,
        corners.astype(np.float32),
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria,
    )
    return _board_corners_from_internal(corners)


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = points.astype(np.float32)
    top = pts[np.argsort(pts[:, 1])[:2]]
    bottom = pts[np.argsort(pts[:, 1])[2:]]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _quad_from_contour(contour: np.ndarray) -> np.ndarray | None:
    """Reduce a contour to an ordered 4-point quadrilateral when possible."""
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


def _find_corners_contour(gray: np.ndarray) -> np.ndarray | None:
    """Fallback: find the largest board-like quadrilateral in the image."""
    h, w = gray.shape[:2]
    min_area = 0.08 * w * h
    best = None
    best_area = 0.0

    edge_variants = []
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    for low, high in ((30, 100), (50, 150), (80, 200)):
        edge_variants.append(cv2.Canny(blur, low, high))

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

    # Last resort: threshold the whole frame and take the dominant rectangle.
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


def detect_board_corners(image: np.ndarray) -> np.ndarray:
    """Detect the four outer board corners in image pixel coordinates."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners = _find_corners_chessboard(gray)
    if corners is None:
        corners = _find_corners_contour(gray)
    if corners is None:
        raise RuntimeError("Could not detect chessboard corners in image.")
    return corners


def board_transform_from_corners(corners: np.ndarray, board_size_px: int = BOARD_SIZE_PX) -> BoardTransform:
    """Build a transform directly from four ordered board corners."""
    src = _order_quad_points(corners)
    dst = np.float32(
        [
            [0, 0],
            [board_size_px, 0],
            [board_size_px, board_size_px],
            [0, board_size_px],
        ]
    )
    homography = cv2.getPerspectiveTransform(src, dst)
    return BoardTransform(homography=homography, board_size_px=board_size_px)


def _refine_corners_with_grid(gray: np.ndarray, coarse: np.ndarray) -> np.ndarray | None:
    """Refine outer corners by snapping to detected interior grid intersections."""
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 120)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(60, w // 12),
        minLineLength=w // 6,
        maxLineGap=w // 40,
    )
    if lines is None:
        return None

    horizontals: list[tuple[float, float, float]] = []
    verticals: list[tuple[float, float, float]] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 1:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle <= 20 or angle >= 160:
            horizontals.append((x1, y1, x2, y2))
        elif 70 <= angle <= 110:
            verticals.append((x1, y1, x2, y2))

    if len(horizontals) < 2 or len(verticals) < 2:
        return None

    def line_to_abc(x1, y1, x2, y2):
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        norm = np.hypot(a, b)
        if norm < 1e-6:
            return None
        return a / norm, b / norm, c / norm

    def intersect(l1, l2):
        a1, b1, c1 = l1
        a2, b2, c2 = l2
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-6:
            return None
        x = (b1 * c2 - b2 * c1) / det
        y = (c1 * a2 - c2 * a1) / det
        return x, y

    h_coeffs = [line_to_abc(*line) for line in horizontals]
    v_coeffs = [line_to_abc(*line) for line in verticals]
    h_coeffs = [c for c in h_coeffs if c is not None]
    v_coeffs = [c for c in v_coeffs if c is not None]
    if len(h_coeffs) < 2 or len(v_coeffs) < 2:
        return None

    points: list[tuple[float, float]] = []
    for hc in h_coeffs:
        for vc in v_coeffs:
            pt = intersect(hc, vc)
            if pt is None:
                continue
            x, y = pt
            if -0.1 * w <= x <= 1.1 * w and -0.1 * h <= y <= 1.1 * h:
                points.append((x, y))

    if len(points) < 4:
        return None

    pts = np.array(points, dtype=np.float32)
    # Keep intersections nearest the coarse quad corners.
    refined = []
    for corner in coarse:
        dists = np.sum((pts - corner) ** 2, axis=1)
        refined.append(pts[int(np.argmin(dists))])
    return _order_quad_points(np.array(refined, dtype=np.float32))


def estimate_board_transform(image: np.ndarray) -> BoardTransform:
    """Estimate homography that maps the board quadrilateral to a square."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    coarse = detect_board_corners(image)
    refined = _refine_corners_with_grid(gray, coarse)
    src = refined if refined is not None else coarse
    return board_transform_from_corners(src)


def largest_component_mask(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] <= 1e-6:
        return None
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return cx, cy


def draw_pct_point(
    warped_board: np.ndarray,
    x_pct: float,
    y_pct: float,
    transform: BoardTransform,
    label: str = "",
    color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    out = warped_board.copy()
    x_px, y_px = transform.pct_to_pixel(x_pct, y_pct)
    cv2.circle(out, (int(round(x_px)), int(round(y_px))), 8, color, -1)
    if label:
        cv2.putText(
            out,
            label,
            (int(round(x_px)) + 10, int(round(y_px)) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def save_debug(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
