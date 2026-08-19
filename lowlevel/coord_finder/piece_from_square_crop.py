#!/usr/bin/env python3
"""From padded square crops → piece-centred crops via classical blob isolation.

Architecture (paper Fig. 5 middle → right, without HOG square/piece detectors):
  1. Start from a square crop (arm already above the square).
  2. Isolate the piece blob with median-deviation + edges + morphology
     (same classical mask as localise_all_squares / test_01 family).
  3. Build a square crop centred on the blob (centroid) with padding.
  4. (Optional later) feed that crop into the Gambit-style servo SVM.

Usage:
  python piece_from_square_crop.py \\
      --square-dir output/square_crop \\
      --out-dir output/piece_crops
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from board_geometry import largest_component_mask, load_image, mask_centroid, save_debug
from localise_all_squares import blob_fill_ratio, isolate_square_mask


def list_images(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.name,
    )


def blob_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return x0, y0, x1, y1 of nonzero mask pixels (x1/y1 exclusive)."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def compact_blob(mask: np.ndarray, keep_percentile: float = 85.0) -> np.ndarray:
    """Trim elongated shadow tails by keeping pixels near the blob core.

    Computes the centroid, then retains only mask pixels whose distance to that
    centroid is below the given percentile of all mask-pixel distances. This
    shrinks connected shadow streaks without needing a second classifier.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 16:
        return mask
    cx, cy = float(xs.mean()), float(ys.mean())
    dists = np.hypot(xs.astype(np.float32) - cx, ys.astype(np.float32) - cy)
    cutoff = float(np.percentile(dists, keep_percentile))
    keep = dists <= max(cutoff, 1.0)
    out = np.zeros_like(mask)
    out[ys[keep], xs[keep]] = 255
    # Re-close small gaps left by the radial trim.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
    return largest_component_mask(out)


def extract_piece_crop(
    square_bgr: np.ndarray,
    mask: np.ndarray,
    pad_frac: float,
    min_size: int,
) -> tuple[np.ndarray, dict] | None:
    """Square crop centred on the blob centroid, sized from bbox + padding."""
    bbox = blob_bbox(mask)
    centroid = mask_centroid(mask)
    if bbox is None or centroid is None:
        return None

    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    side = int(math_ceil(max(bw, bh) * (1.0 + pad_frac)))
    side = max(side, min_size)

    cx, cy = float(centroid[0]), float(centroid[1])
    h, w = square_bgr.shape[:2]

    # Centre the window on the blob; clamp into the square crop.
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    left = max(0, min(left, w - side))
    top = max(0, min(top, h - side))
    # If the square crop itself is smaller than `side`, fall back to full image.
    if side >= w or side >= h:
        left, top = 0, 0
        side = min(w, h)
        crop = square_bgr[top : top + side, left : left + side].copy()
    else:
        crop = square_bgr[top : top + side, left : left + side].copy()

    meta = {
        "bbox": [x0, y0, x1, y1],
        "centroid_px": [round(cx, 3), round(cy, 3)],
        "crop_origin_xy": [left, top],
        "crop_side_px": int(crop.shape[0]),
        "blob_pixels": int(np.count_nonzero(mask)),
        "fill_ratio": round(blob_fill_ratio(mask), 4),
    }
    return crop, meta


def math_ceil(x: float) -> int:
    return int(np.ceil(x))


def draw_debug(square_bgr: np.ndarray, mask: np.ndarray, meta: dict) -> np.ndarray:
    """Overlay mask outline, bbox, centroid, and final crop window."""
    vis = square_bgr.copy()
    color_mask = np.zeros_like(vis)
    color_mask[mask > 0] = (0, 255, 255)
    vis = cv2.addWeighted(vis, 0.7, color_mask, 0.3, 0)

    x0, y0, x1, y1 = meta["bbox"]
    cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 2)

    cx, cy = meta["centroid_px"]
    cv2.circle(vis, (int(round(cx)), int(round(cy))), 4, (255, 0, 0), -1)

    left, top = meta["crop_origin_xy"]
    side = meta["crop_side_px"]
    cv2.rectangle(vis, (left, top), (left + side - 1, top + side - 1), (0, 255, 0), 2)
    return vis


def run(args) -> dict:
    paths = list_images(args.square_dir)
    if not paths:
        raise FileNotFoundError(f"No images in {args.square_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = args.out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for path in paths:
        img = load_image(path)
        # Screenshots may be RGBA → load_image uses imread which drops alpha on 3-channel,
        # but ensure we have BGR.
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        mask = isolate_square_mask(img, args.diff_threshold)
        mask = largest_component_mask(mask)
        mask = compact_blob(mask, keep_percentile=args.compact_percentile)
        n = int(np.count_nonzero(mask))
        fill = blob_fill_ratio(mask)

        entry: dict = {
            "found": False,
            "blob_pixels": n,
            "fill_ratio": round(fill, 4),
        }

        if n < args.min_blob_pixels or fill < args.min_fill_ratio:
            results[path.name] = entry
            save_debug(debug_dir / f"{path.stem}_mask.png", mask)
            continue

        extracted = extract_piece_crop(img, mask, args.pad_frac, args.min_crop_size)
        if extracted is None:
            results[path.name] = entry
            continue

        crop, meta = extracted
        entry.update({"found": True, **meta})
        results[path.name] = entry

        save_debug(args.out_dir / f"{path.stem}_piece.png", crop)
        save_debug(debug_dir / f"{path.stem}_mask.png", mask)
        save_debug(debug_dir / f"{path.stem}_overlay.png", draw_debug(img, mask, meta))

    summary = {
        "method": "classical_blob_piece_crop",
        "input_dir": str(args.square_dir),
        "output_dir": str(args.out_dir),
        "image_count": len(paths),
        "found_count": sum(1 for v in results.values() if v.get("found")),
        "results": results,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--square-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "square_crop",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "piece_crops",
    )
    parser.add_argument("--diff-threshold", type=int, default=22)
    parser.add_argument("--min-blob-pixels", type=int, default=80)
    parser.add_argument("--min-fill-ratio", type=float, default=0.12)
    parser.add_argument(
        "--pad-frac",
        type=float,
        default=0.30,
        help="Extra margin around blob bbox when forming the square piece crop.",
    )
    parser.add_argument(
        "--compact-percentile",
        type=float,
        default=82.0,
        help="Keep mask pixels within this distance-percentile of the blob centroid (trims shadows).",
    )
    parser.add_argument("--min-crop-size", type=int, default=64)
    args = parser.parse_args()

    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
