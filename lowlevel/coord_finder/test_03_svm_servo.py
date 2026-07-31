#!/usr/bin/env python3
"""Test 03: Gambit-style online SVM segmentation + principal-axis localisation.

Follows the visual-servoing feature extraction from Matthies et al.,
"Gambit: A Robust Chess-Playing Robotic System" (arXiv:2012.06858), Sec III-C:

  1. Take a cropped image of a single square (from board geometry + occupancy).
  2. Train a per-crop kernel SVM to separate "piece" vs "background":
       - positive samples = pixels in a central patch of the crop
       - negative samples = pixels along the crop border
       - features = colour (BGR + HSV) and edge/gradient magnitude
  3. Classify every pixel in the crop -> binary piece mask.
  4. Compute the centre (centroid) and orientation (longest principal axis)
     of the segmented piece via image moments / PCA.
  5. Emit centre in 0-100 percentage coords + grasp roll angle.

Unlike the paper (palm/eye-in-hand camera, closed-loop servoing), this runs
open-loop on an overhead warped board, one crop per occupied square. The
segmentation + moment features are the faithful part.

Usage:
  python test_03_svm_servo.py \\
      --empty ../LiveChess2FEN_setup/testing/example_imA/im1.jpeg \\
      --board ../LiveChess2FEN_setup/testing/example_imA/im2.jpeg \\
      --corners-json output/example_imA/im1_corners.json \\
      --a1-pos TR
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from board_geometry import (
    BoardTransform,
    board_transform_from_corners,
    estimate_board_transform,
    largest_component_mask,
    load_image,
    mask_centroid,
    reorient_warped_for_a1,
    save_debug,
    square_name,
    square_roi,
)
from localise_all_squares import blob_fill_ratio, isolate_square_mask


def list_images(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.name,
    )


def pixel_features(crop_bgr: np.ndarray) -> np.ndarray:
    """Per-pixel feature matrix: [B, G, R, H, S, V, grad_mag]."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    feats = np.dstack(
        [crop_bgr.astype(np.float32), hsv.astype(np.float32), grad]
    )
    return feats.reshape(-1, feats.shape[-1])


def sample_indices(
    shape: tuple[int, int],
    center_frac: float,
    border_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flat pixel indices for positive (centre) and negative (border)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    flat = (yy * w + xx).ravel()

    cy, cx = h / 2.0, w / 2.0
    half = center_frac * min(h, w) / 2.0
    center_mask = (np.abs(yy - cy) <= half) & (np.abs(xx - cx) <= half)

    band = int(round(border_frac * min(h, w)))
    band = max(band, 2)
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:band, :] = True
    border_mask[-band:, :] = True
    border_mask[:, :band] = True
    border_mask[:, -band:] = True

    pos = flat[center_mask.ravel()]
    neg = flat[border_mask.ravel()]

    n = min(len(pos), len(neg))
    if n == 0:
        return pos, neg
    pos = rng.choice(pos, size=n, replace=False)
    neg = rng.choice(neg, size=n, replace=False)
    return pos, neg


def sample_region_bounds(shape: tuple[int, int], center_frac: float, border_frac: float) -> dict[str, int]:
    """Return central positive square and border band widths for visualization."""
    h, w = shape
    cy, cx = h / 2.0, w / 2.0
    half = center_frac * min(h, w) / 2.0
    x0 = int(max(0, math.floor(cx - half)))
    y0 = int(max(0, math.floor(cy - half)))
    x1 = int(min(w, math.ceil(cx + half)))
    y1 = int(min(h, math.ceil(cy + half)))
    band = max(int(round(border_frac * min(h, w))), 2)
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "band": band}


def svm_segment(
    crop_bgr: np.ndarray,
    center_frac: float,
    border_frac: float,
    svm_c: float,
    svm_gamma: str | float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Online kernel-SVM segmentation. Returns a uint8 binary mask (0/255)."""
    h, w = crop_bgr.shape[:2]
    feats = pixel_features(crop_bgr)
    pos_idx, neg_idx = sample_indices((h, w), center_frac, border_frac, rng)
    if len(pos_idx) < 8 or len(neg_idx) < 8:
        return np.zeros((h, w), dtype=np.uint8)

    x_train = np.vstack([feats[pos_idx], feats[neg_idx]])
    y_train = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])

    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=svm_c, gamma=svm_gamma),
    )
    clf.fit(x_train, y_train)

    labels = clf.predict(feats).reshape(h, w)
    mask = (labels > 0.5).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def draw_training_overlay(
    crop_bgr: np.ndarray,
    center_frac: float,
    border_frac: float,
) -> np.ndarray:
    """Visualize the SVM positive/negative training regions like the paper."""
    vis = crop_bgr.copy()
    b = sample_region_bounds(crop_bgr.shape[:2], center_frac, border_frac)
    h, w = crop_bgr.shape[:2]
    band = b["band"]

    overlay = vis.copy()
    red = (0, 0, 255)
    blue = (255, 0, 0)
    cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), red, thickness=band)
    cv2.rectangle(overlay, (b["x0"], b["y0"]), (b["x1"] - 1, b["y1"] - 1), blue, thickness=-1)
    vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)

    cv2.rectangle(vis, (0, 0), (w - 1, h - 1), red, thickness=max(1, band // 2))
    cv2.rectangle(vis, (b["x0"], b["y0"]), (b["x1"] - 1, b["y1"] - 1), blue, thickness=2)
    return vis


def principal_axes(mask: np.ndarray) -> dict | None:
    """Centroid + principal-axis orientation from image moments."""
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] <= 1e-6:
        return None
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]

    mu20 = m["mu20"] / m["m00"]
    mu02 = m["mu02"] / m["m00"]
    mu11 = m["mu11"] / m["m00"]
    cov = np.array([[mu20, mu11], [mu11, mu02]], dtype=np.float64)
    evals, evecs = np.linalg.eigh(cov)
    long_axis = evecs[:, int(np.argmax(evals))]

    long_angle = math.degrees(math.atan2(long_axis[1], long_axis[0]))
    grasp_angle = long_angle + 90.0
    while grasp_angle > 90.0:
        grasp_angle -= 180.0
    while grasp_angle <= -90.0:
        grasp_angle += 180.0
    eccentricity = float(np.sqrt(1.0 - (min(evals) / max(evals)))) if max(evals) > 0 else 0.0

    return {
        "cx": cx,
        "cy": cy,
        "long_axis_deg": round(long_angle, 2),
        "grasp_axis_deg": round(grasp_angle, 2),
        "eccentricity": round(eccentricity, 3),
    }


def draw_mask_axes(mask: np.ndarray, axes: dict | None) -> np.ndarray:
    """Render segmented mask with centroid, principal axes, and ellipse."""
    vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    vis[mask > 0] = (255, 255, 255)
    if axes is None:
        return vis

    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] <= 1e-6:
        return vis
    mu20 = m["mu20"] / m["m00"]
    mu02 = m["mu02"] / m["m00"]
    mu11 = m["mu11"] / m["m00"]
    cov = np.array([[mu20, mu11], [mu11, mu02]], dtype=np.float64)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    cx = float(axes["cx"])
    cy = float(axes["cy"])
    center = (int(round(cx)), int(round(cy)))
    red = (0, 0, 255)

    major = max(8, int(round(2.0 * math.sqrt(max(float(evals[0]), 1e-6)))))
    minor = max(6, int(round(2.0 * math.sqrt(max(float(evals[1]), 1e-6)))))
    angle = math.degrees(math.atan2(float(evecs[1, 0]), float(evecs[0, 0])))
    cv2.ellipse(vis, center, (major, minor), angle, 0, 360, red, 2, cv2.LINE_AA)
    cv2.circle(vis, center, 4, red, -1)

    length_major = max(18, int(round(3.5 * math.sqrt(max(float(evals[0]), 1e-6)))))
    length_minor = max(14, int(round(3.5 * math.sqrt(max(float(evals[1]), 1e-6)))))
    v_major = evecs[:, 0]
    v_minor = evecs[:, 1]
    p1 = (int(round(cx - length_major * v_major[0])), int(round(cy - length_major * v_major[1])))
    p2 = (int(round(cx + length_major * v_major[0])), int(round(cy + length_major * v_major[1])))
    q1 = (int(round(cx - length_minor * v_minor[0])), int(round(cy - length_minor * v_minor[1])))
    q2 = (int(round(cx + length_minor * v_minor[0])), int(round(cy + length_minor * v_minor[1])))
    cv2.line(vis, p1, p2, red, 1, cv2.LINE_AA)
    cv2.line(vis, q1, q2, red, 1, cv2.LINE_AA)
    return vis


def is_occupied(crop_bgr: np.ndarray, diff_threshold: int, min_pixels: int, min_fill: float) -> tuple[bool, int, float]:
    mask = largest_component_mask(isolate_square_mask(crop_bgr, diff_threshold))
    n = int(np.count_nonzero(mask))
    fill = blob_fill_ratio(mask)
    return (n >= min_pixels and fill >= min_fill), n, fill


def load_transform(empty_img, corners_json: Path | None) -> BoardTransform:
    if corners_json is not None:
        data = json.loads(corners_json.read_text(encoding="utf-8"))
        corners = np.array(data["board_corners"], dtype=np.float32)
        return board_transform_from_corners(corners)
    return estimate_board_transform(empty_img)


def run_crop_mode(args) -> dict:
    image_paths = list_images(args.crop_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.crop_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    results: dict[str, dict] = {}

    for image_path in image_paths:
        crop = load_image(image_path)
        training_overlay = draw_training_overlay(crop, args.center_frac, args.border_frac)
        mask = svm_segment(
            crop,
            center_frac=args.center_frac,
            border_frac=args.border_frac,
            svm_c=args.svm_c,
            svm_gamma=args.svm_gamma,
            rng=rng,
        )
        mask = largest_component_mask(mask)
        axes = principal_axes(mask)
        mask_axes = draw_mask_axes(mask, axes)

        stem = image_path.stem
        save_debug(args.out_dir / f"{stem}_train.png", training_overlay)
        save_debug(args.out_dir / f"{stem}_mask_axes.png", mask_axes)
        side_by_side = np.hstack([training_overlay, mask_axes])
        save_debug(args.out_dir / f"{stem}_combined.png", side_by_side)

        entry = {
            "found": axes is not None and int(np.count_nonzero(mask)) >= args.min_seg_pixels,
            "seg_pixels": int(np.count_nonzero(mask)),
        }
        if axes is not None:
            entry.update(
                {
                    "cx_px": round(float(axes["cx"]), 3),
                    "cy_px": round(float(axes["cy"]), 3),
                    "long_axis_deg": axes["long_axis_deg"],
                    "grasp_axis_deg": axes["grasp_axis_deg"],
                    "eccentricity": axes["eccentricity"],
                }
            )
        results[image_path.name] = entry

    summary = {
        "method": "gambit_svm_visual_servo_crop_mode",
        "input_dir": str(args.crop_dir),
        "output_dir": str(args.out_dir),
        "image_count": len(image_paths),
        "results": results,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run(args) -> dict:
    ref_img = load_image(args.empty)
    board_img = load_image(args.board)
    transform = load_transform(ref_img, args.corners_json)
    warped_board = reorient_warped_for_a1(transform.warp(board_img), args.a1_pos)
    board_size = transform.board_size_px
    rng = np.random.default_rng(args.seed)

    if args.square:
        from board_geometry import parse_square

        targets = [parse_square(args.square)]
    else:
        targets = [(f, r) for r in range(8) for f in range(8)]

    results: dict[str, dict] = {}
    vis = warped_board.copy()

    for file_idx, rank_idx in targets:
        sq = square_name(file_idx, rank_idx)
        x0, y0, x1, y1 = square_roi(file_idx, rank_idx, board_size)
        crop = warped_board[y0:y1, x0:x1]

        occupied, n, fill = is_occupied(crop, args.diff_threshold, args.min_diff_pixels, args.min_fill_ratio)
        entry: dict = {
            "x": None,
            "y": None,
            "found": False,
            "occupied": occupied,
            "occupancy_pixels": n,
            "fill_ratio": round(fill, 4),
        }
        if not (occupied or args.square):
            results[sq] = entry
            continue

        mask = svm_segment(
            crop,
            center_frac=args.center_frac,
            border_frac=args.border_frac,
            svm_c=args.svm_c,
            svm_gamma=args.svm_gamma,
            rng=rng,
        )
        mask = largest_component_mask(mask)
        axes = principal_axes(mask)
        if axes is not None and int(np.count_nonzero(mask)) >= args.min_seg_pixels:
            x_pct, y_pct = transform.pixel_to_pct(x0 + axes["cx"], y0 + axes["cy"])
            entry.update(
                {
                    "x": round(x_pct, 4),
                    "y": round(y_pct, 4),
                    "found": True,
                    "long_axis_deg": axes["long_axis_deg"],
                    "grasp_axis_deg": axes["grasp_axis_deg"],
                    "eccentricity": axes["eccentricity"],
                    "seg_pixels": int(np.count_nonzero(mask)),
                }
            )
            cxg, cyg = int(round(x0 + axes["cx"])), int(round(y0 + axes["cy"]))
            cv2.circle(vis, (cxg, cyg), 4, (0, 0, 255), -1)
            length = 18
            rad = math.radians(axes["long_axis_deg"])
            dx, dy = int(length * math.cos(rad)), int(length * math.sin(rad))
            cv2.line(vis, (cxg - dx, cyg - dy), (cxg + dx, cyg + dy), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(vis, sq, (cxg + 5, cyg - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

        results[sq] = entry

        if args.debug_square and sq == args.debug_square:
            save_debug(args.out_dir / f"debug_{sq}_crop.jpg", crop)
            save_debug(args.out_dir / f"debug_{sq}_mask.jpg", mask)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_debug(args.out_dir / "warped_board_svm.jpg", vis)

    occupied = {k: v for k, v in results.items() if v.get("found")}
    summary = {
        "method": "gambit_svm_visual_servo",
        "localised_count": len(occupied),
        "occupied_squares": sorted(occupied.keys(), key=lambda s: (int(s[1]), s[0])),
        "pieces": occupied,
        "all_squares": results,
    }
    (args.out_dir / "coords_svm.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empty", type=Path, help="Reference / pose image")
    parser.add_argument("--board", type=Path, help="Image to localise pieces on")
    parser.add_argument("--corners-json", type=Path, default=None)
    parser.add_argument("--a1-pos", choices=["TL", "TR", "BL", "BR"], default="TR")
    parser.add_argument("--square", type=str, default=None, help="Localise a single square only (e.g. e4).")
    parser.add_argument("--crop-dir", type=Path, default=None, help="If set, run servo-only on each crop image in this folder.")
    parser.add_argument("--center-frac", type=float, default=0.32, help="Central positive-sample region as fraction of crop.")
    parser.add_argument("--border-frac", type=float, default=0.12, help="Border negative-sample band as fraction of crop.")
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", default="scale")
    parser.add_argument("--diff-threshold", type=int, default=25, help="Occupancy pre-filter threshold.")
    parser.add_argument("--min-diff-pixels", type=int, default=400, help="Occupancy pre-filter pixel count.")
    parser.add_argument("--min-fill-ratio", type=float, default=0.15, help="Occupancy pre-filter fill ratio.")
    parser.add_argument("--min-seg-pixels", type=int, default=150, help="Min SVM foreground pixels to accept a localisation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug-square", type=str, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "test_03",
    )
    args = parser.parse_args()

    try:
        args.svm_gamma = float(args.svm_gamma)
    except (TypeError, ValueError):
        pass

    if args.crop_dir is not None:
        summary = run_crop_mode(args)
    else:
        if args.empty is None or args.board is None:
            parser.error("--empty and --board are required unless --crop-dir is used.")
        summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
