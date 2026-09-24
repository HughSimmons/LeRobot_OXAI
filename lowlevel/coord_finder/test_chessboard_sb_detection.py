#!/usr/bin/env python3
"""One-off test: detect the real (genuinely image-detected, not interpolated)
interior 8x8 board grid corners via cv2.findChessboardCornersSB, as a
cross-check against LiveChess2FEN's board_transform_from_corners, which only
straight-line-interpolates between the 4 outer CPS-detected corners and so
can't reveal lens-distortion drift in the interior grid lines.

Checkerboard corner detectors only find the "saddle point" X-junctions where
4 alternating-colour squares meet -- an 8x8 board has at most 7x7=49 such
interior points; the outer boundary of the grid is never detectable this way.

Feeding the whole photo (or even a LiveChess2FEN board_corners bounding-box
crop) into cv2.findChessboardCornersSB picks up a spurious extra row/column
from the wooden frame's inlay trim, which sits inside the LC2FEN-detected
board_corners boundary (CPS pads board_corners outward specifically to
contain the whole board incl. its frame). The fix used here: evaluate
LC2FEN's own corner homography (via board_transform_from_corners) at a
canonical-space inset of half a square width from each edge -- enough to
clear the frame/inlay border without eating into real interior corners --
and crop to that instead of the full board_corners box.

The crop follows the true shape implied by the corner homography rather than
a plain axis-aligned (or even rotated-rectangle) bounding box: because of
perspective, the interior grid's true boundary is a trapezoid in the photo,
not a rectangle, so a rectangular crop is always either loose on one side or
cuts into the pattern on another. The homography gives that trapezoid's
corners directly -- used ONLY to decide which pixels to keep, never to
warp/rectify the image itself (warping the actual pixel content with the
same homography we're trying to independently cross-check would bake its
assumptions into the input and defeat the point of the test). Pixels inside
the (slightly outset, for a quiet-zone margin) quad are left untouched;
pixels outside it, within the quad's bounding box, are blanked to a neutral
grey so the detector isn't confused by frame/background texture.

The saved debug image shows more than just the crop fed to the detector: it
includes a border of the surrounding cropped-out region too (for context),
with a dashed polygon marking the actual quad boundary the detector saw, so
it's clear how much of the frame/inlay/background was excluded.

Also fits a single-photo camera calibration (cv2.calibrateCamera) directly
from these 49 known corners, treating them as correspondences against an
assumed-flat, uniform-spacing 7x7 grid -- this solves for actual lens
distortion coefficients (not just a better-conditioned homography), then
cv2.undistort()s the full original photo so the result can be checked by eye
for straight, evenly-spaced grid lines. A single image under-constrains a
full calibration, so the principal point/aspect ratio are held fixed and
tangential + 3rd-order radial distortion are disabled, leaving only
focal-length and k1/k2 radial distortion to solve for.

Not wired into the main calibration pipeline -- this is a standalone
feasibility test. See board_meta.json / robot_to_board_calibration.py in the
same directory for the production pipeline this could feed into.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from board_geometry import board_transform_from_corners
from robot_to_board_calibration import inverse_project

BOARD_META_PATH = Path(
    "/Users/zhg603/Documents/OXAI/lowlevel/coord_finder/output/"
    "robot_to_board_calibration/board_meta.json"
)
OUT_PATH = Path(
    "/Users/zhg603/Documents/OXAI/lowlevel/coord_finder/output/"
    "robot_to_board_calibration/chessboard_sb_lc2fen_interior_crop.jpg"
)
UNDISTORTED_OUT_PATH = Path(
    "/Users/zhg603/Documents/OXAI/lowlevel/coord_finder/output/"
    "robot_to_board_calibration/chessboard_sb_undistorted.jpg"
)
SQUARE_CROPS_DIR = Path(__file__).resolve().parent / "output" / "robot_to_board_calibration" / "square_crops"
PATTERN_SIZE = (7, 7)  # max detectable interior corners for an 8x8 board
CROP_MARGIN_PX = 15  # quiet-zone margin around the interior bbox; <15 fails to detect
CONTEXT_MARGIN_PX = 150  # extra cropped-out surroundings shown in the debug image
SQUARE_LENGTH_M = 0.04125  # matches robot_to_board_calibration.py's DEFAULT_SQUARE_LENGTH_M
SQUARE_CROP_PX = 100  # side length of each saved, warped individual-square image
ENABLE_UNDISTORT = False  # single-image lens calibration -- disabled for now, see chat


def calibrate_and_undistort(
    image: np.ndarray, corners_full: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Single-image camera calibration from the 49 known interior corners,
    treated as correspondences against an assumed-flat, uniform 7x7 grid of
    real square positions -- solves for actual lens distortion, not just a
    better-fit homography.

    A single view under-constrains a full calibration, so this fixes the
    principal point at the image centre and the aspect ratio at 1, and
    disables tangential distortion and 3rd-order radial distortion --
    leaving only focal length and k1/k2 radial distortion to solve for.

    Returns (camera_matrix, dist_coeffs, mean_reprojection_error_px).
    """
    object_points = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0 : PATTERN_SIZE[0], 0 : PATTERN_SIZE[1]].T.reshape(-1, 2)
    object_points *= SQUARE_LENGTH_M

    image_size = (image.shape[1], image.shape[0])
    flags = (
        cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K3
    )
    reprojection_error, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        [object_points], [corners_full.astype(np.float32)], image_size, None, None, flags=flags
    )
    return camera_matrix, dist_coeffs, reprojection_error


def save_individual_square_crops(
    image: np.ndarray, corners_full: np.ndarray, out_dir: Path
) -> int:
    """Warp each of the 36 individual squares bounded by the 7x7 detected
    interior corners into its own square canonical image, using ONLY that
    cell's own 4 real corners -- a per-square perspective correction with no
    global homography or distortion-model assumption at all. Returns the
    number of squares saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = corners_full.reshape(PATTERN_SIZE[0], PATTERN_SIZE[1], 2)
    dst = np.asarray(
        [[0, 0], [SQUARE_CROP_PX, 0], [SQUARE_CROP_PX, SQUARE_CROP_PX], [0, SQUARE_CROP_PX]],
        dtype=np.float32,
    )
    count = 0
    for row in range(PATTERN_SIZE[0] - 1):
        for col in range(PATTERN_SIZE[1] - 1):
            src = np.asarray(
                [grid[row, col], grid[row, col + 1], grid[row + 1, col + 1], grid[row + 1, col]],
                dtype=np.float32,
            )
            homography = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(image, homography, (SQUARE_CROP_PX, SQUARE_CROP_PX))
            cv2.imwrite(str(out_dir / f"square_r{row}_c{col}.jpg"), warped)
            count += 1
    return count


def half_square_inset_quad_px(board_meta: dict) -> np.ndarray:
    """The board's true interior boundary, inset by half a square width from
    each edge of LC2FEN's 4 detected corners -- just enough to clear a
    typical wooden frame/inlay border without eating into real interior
    corners, in original-image pixel space, TL/TR/BR/BL order.

    Evaluated directly from board_transform_from_corners's homography at the
    canonical-space inset (rather than snapping to LC2FEN's fixed 9-point
    square_corners lattice), so the inset amount is a free, exact parameter
    -- using the homography only to derive *where* to crop, never to
    warp/rectify the image pixels themselves.
    """
    corners = np.asarray(board_meta["board_corners"], dtype=np.float32)
    board_size_px = int(board_meta["board_size_px"])
    transform = board_transform_from_corners(corners, board_size_px=board_size_px)
    half_square = board_size_px / 16.0  # half of one of the 8 squares
    canonical_quad = [
        (half_square, half_square),
        (board_size_px - half_square, half_square),
        (board_size_px - half_square, board_size_px - half_square),
        (half_square, board_size_px - half_square),
    ]
    return np.asarray(
        [inverse_project(p, transform.homography) for p in canonical_quad], dtype=np.float32
    )


def outset_quad(quad: np.ndarray, margin_px: float) -> np.ndarray:
    """Push each quad vertex outward (radially, from the quad's centroid) by
    margin_px, giving the detector's required quiet-zone margin without
    assuming the quad's edges are axis-aligned.
    """
    center = quad.mean(axis=0)
    offsets = quad - center
    unit_offsets = offsets / np.linalg.norm(offsets, axis=1, keepdims=True)
    return quad + unit_offsets * margin_px


def masked_crop_from_quad(
    image: np.ndarray, quad: np.ndarray
) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray]:
    """Crop to the quad's bounding box (no rotation/warping of pixel
    content) and blank everything outside the quad itself to neutral grey,
    so the detector only ever sees real, unmodified board pixels plus a
    quiet uniform border -- never the frame/background beyond the true
    (perspective-trapezoid) interior boundary.

    Returns (masked_crop, (x0, y0, x1, y1), quad_local) -- the bbox within
    the original image, and the quad's coordinates local to the crop, so
    callers can also draw it for context.
    """
    x0, y0 = np.floor(quad.min(axis=0)).astype(int)
    x1, y1 = np.ceil(quad.max(axis=0)).astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)

    crop = image[y0:y1, x0:x1].copy()
    quad_local = (quad - [x0, y0]).astype(np.int32)
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad_local, 255)
    background = np.full_like(crop, 128)
    masked_crop = np.where(mask[:, :, None].astype(bool), crop, background)
    return masked_crop, (x0, y0, x1, y1), quad_local


def draw_dashed_line(
    image: np.ndarray,
    pt1: tuple[float, float],
    pt2: tuple[float, float],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_length: float = 10.0,
    gap_length: float = 8.0,
) -> None:
    start = np.asarray(pt1, dtype=float)
    end = np.asarray(pt2, dtype=float)
    length = float(np.linalg.norm(end - start))
    if length == 0:
        return
    direction = (end - start) / length
    t = 0.0
    while t < length:
        seg_start = start + direction * t
        seg_end = start + direction * min(t + dash_length, length)
        cv2.line(
            image,
            tuple(np.round(seg_start).astype(int)),
            tuple(np.round(seg_end).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )
        t += dash_length + gap_length


def draw_dashed_polygon(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    for i in range(len(points)):
        draw_dashed_line(image, tuple(points[i]), tuple(points[(i + 1) % len(points)]), color, thickness)


def main() -> int:
    board_meta = json.loads(BOARD_META_PATH.read_text())
    image_path = board_meta["source_image"]

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return 1

    quad = outset_quad(half_square_inset_quad_px(board_meta), CROP_MARGIN_PX)
    masked_crop, (x0, y0, x1, y1), quad_local = masked_crop_from_quad(image, quad)
    gray = cv2.cvtColor(masked_crop, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCornersSB(gray, PATTERN_SIZE, flags=cv2.CALIB_CB_LARGER)
    print(f"crop shape={masked_crop.shape[:2]}  found={found}  "
          f"n_corners={corners.shape[0] if found else 0}")
    if not found:
        print("Detection failed -- try adjusting CROP_MARGIN_PX.")
        return 1

    # Show the crop plus its cropped-out surroundings for context, with a
    # dashed polygon marking the actual quad boundary fed to the detector.
    dx0 = max(0, x0 - CONTEXT_MARGIN_PX)
    dy0 = max(0, y0 - CONTEXT_MARGIN_PX)
    dx1 = min(image.shape[1], x1 + CONTEXT_MARGIN_PX)
    dy1 = min(image.shape[0], y1 + CONTEXT_MARGIN_PX)
    vis = image[dy0:dy1, dx0:dx1].copy()

    quad_display = quad_local + [x0 - dx0, y0 - dy0]
    draw_dashed_polygon(vis, quad_display, (0, 255, 255), thickness=2)

    # Plain per-corner markers -- cv2.drawChessboardCorners instead draws one
    # connected polyline through all corners in scan order, which visibly
    # "wraps" diagonally from the end of each row to the start of the next.
    corners_display = corners.copy()
    corners_display[:, 0, 0] += x0 - dx0
    corners_display[:, 0, 1] += y0 - dy0
    for x, y in corners_display.reshape(-1, 2):
        cv2.drawMarker(
            vis, (int(round(x)), int(round(y))), (0, 0, 255), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA
        )

    # LC2FEN's own 4 detected outer board_corners, for comparison against the
    # SB-detected interior corners above.
    board_corners = np.asarray(board_meta["board_corners"], dtype=float)
    for x, y in board_corners:
        cv2.drawMarker(
            vis,
            (int(round(x - dx0)), int(round(y - dy0))),
            (255, 0, 255),
            cv2.MARKER_CROSS,
            24,
            3,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(OUT_PATH), vis)
    print("Saved:", OUT_PATH)

    corners_full = corners.copy()
    corners_full[:, 0, 0] += x0
    corners_full[:, 0, 1] += y0

    if ENABLE_UNDISTORT:
        camera_matrix, dist_coeffs, reprojection_error = calibrate_and_undistort(image, corners_full)
        print(f"camera_matrix fx,fy,cx,cy = {camera_matrix[0,0]:.1f}, {camera_matrix[1,1]:.1f}, "
              f"{camera_matrix[0,2]:.1f}, {camera_matrix[1,2]:.1f}")
        print(f"dist_coeffs (k1,k2,p1,p2,k3) = {dist_coeffs.ravel()}")
        print(f"mean reprojection error = {reprojection_error:.3f}px")

        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
        cv2.imwrite(str(UNDISTORTED_OUT_PATH), undistorted)
        print("Saved:", UNDISTORTED_OUT_PATH)

    n_squares = save_individual_square_crops(image, corners_full, SQUARE_CROPS_DIR)
    print(f"Saved {n_squares} individual square crops to {SQUARE_CROPS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
