"""Relate the tripod and wrist views by synthesizing a common overhead view.

Each camera's image is independently self-calibrated -- corner detection
plus solvePnP against the known board geometry, exactly as in
calibrate_camera_distance.py -- using nothing but that camera's own image
and the shared intrinsics/square size. The recovered pose gives an analytic
board-plane-to-image homography (the board is flat, so this is exact, not
fitted): H = K @ [r1 r2 t], where r1/r2 are the first two columns of the
rotation matrix. Inverting and warping each image with this homography
produces a canonical top-down "common view" -- this is the transform that
relates the two camera views to each other.

The sim's directly-rendered topdown_camera.png is only used afterwards, as
a ground-truth reference to check how well the two synthesized views agree
with reality. It is run through the exact same self-calibration + warp
pipeline (not just displayed) so the comparison is apples-to-apples, but its
pose is never fed into the tripod/wrist synthesis -- only its image, and
only for evaluation.

Needs the OpenCV environment (see ../coord_finder/live2fen_env.yml).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrate_camera_distance import PATTERN_INNER, build_object_points, find_corners

PX_PER_SQUARE = 60
# Interior corners span 6 squares (7 points); add a 1-square margin on each
# side to cover the full 8-square board.
MARGIN_SQUARES = 1
INTERIOR_SPAN_SQUARES = PATTERN_INNER[0] - 1
CANVAS_SQUARES = INTERIOR_SPAN_SQUARES + 2 * MARGIN_SQUARES


def _cell_color_match_score(gray: np.ndarray, corners_grid: np.ndarray) -> int:
    """How many of the 36 interior cells have the light/dark the board's
    known coloring predicts, given this candidate corner-grid orientation.

    Cell (i, j) in 0..5 sits between corners (i, i+1) and (j, j+1), which is
    world square (i+1, j+1) once the 1-square margin is accounted for --
    light iff (i + j) % 2 == 0, same formula generate_sim_images.py uses.
    """
    score = 0
    for i in range(INTERIOR_SPAN_SQUARES):
        for j in range(INTERIOR_SPAN_SQUARES):
            center = corners_grid[i : i + 2, j : j + 2].reshape(-1, 2).mean(axis=0)
            x, y = int(round(center[0])), int(round(center[1]))
            if not (0 <= y < gray.shape[0] and 0 <= x < gray.shape[1]):
                continue
            is_light = gray[y, x] > 127
            if is_light == ((i + j) % 2 == 0):
                score += 1
    return score


def resolve_corner_orientation(gray: np.ndarray, image_points: np.ndarray) -> np.ndarray:
    """Pick whichever of the two color cosets (identity vs. single-axis
    flip) actually matches the board's known coloring.

    cv2.findChessboardCornersSB can return any of the 8 row/column
    orderings of a plain, markerless NxN checkerboard -- for an even cell
    count (ours is 6x6), exactly half of those orderings (identity, a 180
    deg rotation, and the two diagonal transposes) reproduce the true
    coloring, and the other half (a single 90 deg rotation or a single-axis
    mirror) invert every color. The two cosets are otherwise visually
    indistinguishable for a plain board, so testing one representative of
    each -- the raw order and a single row-flip -- is enough to pick the
    correct coset; this uses only the board's own known coloring, not any
    camera's ground-truth pose.
    """
    grid = image_points.reshape(PATTERN_INNER[0], PATTERN_INNER[1], 2)
    flipped = grid[::-1, :, :]
    if _cell_color_match_score(gray, flipped) > _cell_color_match_score(gray, grid):
        return flipped.reshape(-1, 2)
    return grid.reshape(-1, 2)


def estimate_pose(image: np.ndarray, square_size: float, camera_matrix: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_points = find_corners(gray)
    image_points = resolve_corner_orientation(gray, image_points)
    object_points = build_object_points(square_size)
    dist_coeffs = np.zeros(5)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise RuntimeError("solvePnP failed to converge")
    rotation, _ = cv2.Rodrigues(rvec)
    return rotation, tvec


def board_to_image_homography(camera_matrix: np.ndarray, rotation: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """H maps board-plane points in metres [X, Y, 1] to image pixels."""
    board_to_camera = np.column_stack([rotation[:, 0], rotation[:, 1], tvec.flatten()])
    return camera_matrix @ board_to_camera


def warp_to_common_view(image: np.ndarray, homography: np.ndarray, square_size: float, px_per_square: int = PX_PER_SQUARE) -> np.ndarray:
    canvas_px = int(round(CANVAS_SQUARES * px_per_square))
    scale = px_per_square / square_size  # canonical pixels per metre
    # Canonical pixel (0,0) is MARGIN_SQUARES before the first detected
    # interior corner, so it lands on the board's physical edge.
    canonical_to_board = np.array(
        [
            [1 / scale, 0, -MARGIN_SQUARES * square_size],
            [0, 1 / scale, -MARGIN_SQUARES * square_size],
            [0, 0, 1],
        ]
    )
    canonical_to_image = homography @ canonical_to_board
    image_to_canonical = np.linalg.inv(canonical_to_image)
    return cv2.warpPerspective(image, image_to_canonical, (canvas_px, canvas_px))


def synthesize(image_path: Path, square_size: float, camera_matrix: np.ndarray, px_per_square: int) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    rotation, tvec = estimate_pose(image, square_size, camera_matrix)
    homography = board_to_image_homography(camera_matrix, rotation, tvec)
    return warp_to_common_view(image, homography, square_size, px_per_square)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="output/manifest.json", help="manifest.json from generate_sim_images.py")
    parser.add_argument("--output-dir", default="output", help="Directory to write common_view_from_*.png into")
    parser.add_argument("--px-per-square", type=int, default=PX_PER_SQUARE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    intrinsics = manifest["intrinsics"]
    camera_matrix = np.array(
        [[intrinsics["fx"], 0, intrinsics["cx"]], [0, intrinsics["fy"], intrinsics["cy"]], [0, 0, 1]]
    )
    square_size = manifest["board"]["square_size_m"]
    output_dir = Path(args.output_dir)

    # The two-camera setup: synthesized independently, each from only its
    # own image -- this is the actual deliverable.
    views = {}
    for name in ("tripod", "wrist"):
        image_path = output_dir / manifest["cameras"][name]["image_file"]
        views[name] = synthesize(image_path, square_size, camera_matrix, args.px_per_square)
        out_path = output_dir / f"common_view_from_{name}.png"
        cv2.imwrite(str(out_path), views[name])
        print(f"Wrote {out_path}")

    diff = cv2.absdiff(views["tripod"], views["wrist"])
    print(f"tripod vs. wrist common view: mean abs pixel diff = {diff.mean():.2f} / 255")

    # Ground truth, for evaluation only -- same self-calibration pipeline,
    # but its result never feeds into the tripod/wrist views above.
    if "topdown" in manifest["cameras"]:
        topdown_path = output_dir / manifest["cameras"]["topdown"]["image_file"]
        views["topdown"] = synthesize(topdown_path, square_size, camera_matrix, args.px_per_square)
        out_path = output_dir / "common_view_from_topdown.png"
        cv2.imwrite(str(out_path), views["topdown"])
        print(f"Wrote {out_path} (ground-truth reference, not used above)")

        for name in ("tripod", "wrist"):
            diff = cv2.absdiff(views[name], views["topdown"])
            print(f"{name} vs. ground-truth topdown: mean abs pixel diff = {diff.mean():.2f} / 255")


if __name__ == "__main__":
    main()
