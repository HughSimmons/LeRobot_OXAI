"""Recover a camera's distance to the chessboard from a single image.

Detects the board's 7x7 interior corners with cv2.findChessboardCornersSB,
builds 3D object points from the known square size, and runs cv2.solvePnP
to recover the camera pose. Reports the perpendicular distance from the
camera to the board plane (invariant to which corner the detector happened
to return first) and, when run against simulation output, compares it to
the known ground truth recorded in manifest.json.

Needs the OpenCV environment (see ../coord_finder/live2fen_env.yml).

Examples
--------
Sim mode (compares against ground truth)::

    python calibrate_camera_distance.py \\
        --image output/tripod_camera.png \\
        --manifest output/manifest.json --camera tripod

Standalone mode (no ground truth available)::

    python calibrate_camera_distance.py \\
        --image some_real_photo.jpg \\
        --fx 950 --fy 950 --cx 640 --cy 360 --square-size 0.04125
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PATTERN_INNER = (7, 7)  # interior corners of an 8x8 board


def find_corners(gray: np.ndarray) -> np.ndarray:
    flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(gray, PATTERN_INNER, flags)
    if not found:
        pre = cv2.equalizeHist(cv2.GaussianBlur(gray, (5, 5), 0))
        found, corners = cv2.findChessboardCornersSB(pre, PATTERN_INNER, flags)
    if not found:
        found, corners = cv2.findChessboardCorners(gray, PATTERN_INNER, None)
    if not found:
        raise RuntimeError("Could not detect interior chessboard corners")
    return corners.reshape(-1, 2).astype(np.float32)


def build_object_points(square_size: float) -> np.ndarray:
    object_points = np.zeros((PATTERN_INNER[0] * PATTERN_INNER[1], 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0 : PATTERN_INNER[0], 0 : PATTERN_INNER[1]].T.reshape(-1, 2)
    object_points *= square_size
    return object_points


def solve_distance(image_points: np.ndarray, object_points: np.ndarray, camera_matrix: np.ndarray):
    dist_coeffs = np.zeros(5)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise RuntimeError("solvePnP failed to converge")

    rotation, _ = cv2.Rodrigues(rvec)
    perpendicular_distance = abs(float(rotation[:, 2] @ tvec.flatten()))
    euclidean_distance = float(np.linalg.norm(tvec))
    return perpendicular_distance, euclidean_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="Path to the board photo/render")
    parser.add_argument("--manifest", help="manifest.json from generate_sim_images.py (sim mode)")
    parser.add_argument("--camera", choices=["topdown", "tripod", "wrist"], help="Which camera entry in the manifest to compare against")
    parser.add_argument("--fx", type=float, help="Focal length x in pixels (standalone mode)")
    parser.add_argument("--fy", type=float, help="Focal length y in pixels (standalone mode)")
    parser.add_argument("--cx", type=float, help="Principal point x in pixels (standalone mode)")
    parser.add_argument("--cy", type=float, help="Principal point y in pixels (standalone mode)")
    parser.add_argument("--square-size", type=float, default=0.04125, help="Board square side length in metres")
    parser.add_argument("--tolerance-mm", type=float, default=1.0, help="Max allowed error vs. ground truth, in mm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest = None
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        intrinsics = manifest["intrinsics"]
        if args.camera and "intrinsics" in manifest["cameras"][args.camera]:
            intrinsics = manifest["cameras"][args.camera]["intrinsics"]  # per-camera FOV
        fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    else:
        missing = [name for name in ("fx", "fy", "cx", "cy") if getattr(args, name) is None]
        if missing:
            sys.exit(f"Missing intrinsics {missing}: pass --manifest or all of --fx/--fy/--cx/--cy")
        fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy

    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"Could not read image: {args.image}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_points = find_corners(gray)
    object_points = build_object_points(args.square_size)
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    perpendicular_distance, euclidean_distance = solve_distance(image_points, object_points, camera_matrix)

    print(f"Recovered perpendicular distance to board plane: {perpendicular_distance * 1000:.2f} mm")
    print(f"Recovered Euclidean distance (norm(tvec)): {euclidean_distance * 1000:.2f} mm "
          f"(diagnostic only -- depends on corner ordering)")

    if manifest and args.camera:
        camera_info = manifest["cameras"][args.camera]
        ground_truth = camera_info["ground_truth_perpendicular_distance_to_board_plane_m"]
        error_mm = abs(perpendicular_distance - ground_truth) * 1000
        print(f"Ground-truth perpendicular distance: {ground_truth * 1000:.2f} mm")
        print(f"Error: {error_mm:.3f} mm")
        assert error_mm < args.tolerance_mm, (
            f"Distance error {error_mm:.3f} mm exceeds tolerance {args.tolerance_mm} mm"
        )
        print("PASS")


if __name__ == "__main__":
    main()
