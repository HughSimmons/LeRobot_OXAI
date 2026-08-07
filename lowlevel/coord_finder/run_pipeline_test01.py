#!/usr/bin/env python3
"""End-to-end pipeline test: board warp → 64 square crops → piece crop → SVM servo.

Pipeline:
  1. Identify + warp board (LiveChess2FEN detect_board)
  2. Naive 8x8 square split
  3. Classical blob → piece-centred crop
  4. Online kernel-SVM segmentation + principal axes (centre + ellipse)

Runs on every square (including empty). Empty/noisy squares still get an
annotated output; occupancy gating is deferred.

Usage:
  python run_pipeline_test01.py \\
      --image input/im2.jpeg \\
      --out-dir output/pipeline_test01 \\
      --a1-pos TR
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from board_detect_lc2fen import detect_and_warp_board
from board_geometry import largest_component_mask, load_image, save_debug
from localise_all_squares import isolate_square_mask
from piece_from_square_crop import compact_blob, extract_piece_crop
from split_squares_trivial import split_board_trivial
from test_03_svm_servo import (
    draw_mask_axes,
    draw_training_overlay,
    principal_axes,
    svm_segment,
)


def annotate_piece_crop(piece_bgr: np.ndarray, axes: dict | None) -> np.ndarray:
    """Draw centroid + principal-axis ellipse on the colour piece crop."""
    vis = piece_bgr.copy()
    if axes is None:
        cv2.putText(
            vis,
            "no axes",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        return vis

    # Rebuild ellipse sizing from a temporary mask of the full crop by using
    # axis metadata only (angles + centre). Scale ellipse to a fraction of crop.
    h, w = vis.shape[:2]
    cx, cy = float(axes["cx"]), float(axes["cy"])
    center = (int(round(cx)), int(round(cy)))
    red = (0, 0, 255)

    # Use eccentricity to set axis ratio; absolute size ~ 30% of crop.
    ecc = float(axes.get("eccentricity", 0.5))
    major = max(12, int(0.32 * min(h, w)))
    minor = max(8, int(major * math.sqrt(max(1.0 - ecc * ecc, 0.05))))
    angle = float(axes["long_axis_deg"])
    cv2.ellipse(vis, center, (major, minor), angle, 0, 360, red, 2, cv2.LINE_AA)
    cv2.circle(vis, center, 4, red, -1)

    rad = math.radians(angle)
    length = max(16, major)
    dx, dy = int(length * math.cos(rad)), int(length * math.sin(rad))
    cv2.line(vis, (center[0] - dx, center[1] - dy), (center[0] + dx, center[1] + dy), red, 1, cv2.LINE_AA)
    # Shorter (grasp) axis
    rad2 = math.radians(float(axes["grasp_axis_deg"]))
    length2 = max(12, minor)
    dx2, dy2 = int(length2 * math.cos(rad2)), int(length2 * math.sin(rad2))
    cv2.line(
        vis,
        (center[0] - dx2, center[1] - dy2),
        (center[0] + dx2, center[1] + dy2),
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return vis


def process_square(
    square_bgr: np.ndarray,
    rng: np.random.Generator,
    diff_threshold: int,
    compact_percentile: float,
    pad_frac: float,
    min_crop_size: int,
    center_frac: float,
    border_frac: float,
    svm_c: float,
    svm_gamma,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Square → piece crop → SVM axes. Always returns an annotated image."""
    mask = isolate_square_mask(square_bgr, diff_threshold)
    mask = largest_component_mask(mask)
    mask = compact_blob(mask, keep_percentile=compact_percentile)

    extracted = extract_piece_crop(square_bgr, mask, pad_frac, min_crop_size)
    if extracted is None:
        piece = square_bgr.copy()
        piece_meta: dict = {"piece_crop_from": "full_square_fallback"}
    else:
        piece, piece_meta = extracted
        piece_meta = {"piece_crop_from": "blob", **piece_meta}

    train_overlay = draw_training_overlay(piece, center_frac, border_frac)
    seg = svm_segment(
        piece,
        center_frac=center_frac,
        border_frac=border_frac,
        svm_c=svm_c,
        svm_gamma=svm_gamma,
        rng=rng,
    )
    seg = largest_component_mask(seg)
    axes = principal_axes(seg)
    annotated = annotate_piece_crop(piece, axes)
    mask_axes = draw_mask_axes(seg, axes)
    combined = np.hstack([train_overlay, annotated, mask_axes])

    meta = {
        **piece_meta,
        "seg_pixels": int(np.count_nonzero(seg)),
        "axes": axes,
    }
    return annotated, combined, meta


def run(args) -> dict:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    squares_dir = out_dir / "squares"
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Detecting and warping board (LiveChess2FEN)...")
    board_meta = detect_and_warp_board(args.image, out_dir)
    warped_path = Path(board_meta["warped_board"])

    print("[2/4] Splitting warped board into 64 squares...")
    board = load_image(warped_path)
    split_meta = split_board_trivial(
        board,
        squares_dir,
        a1_pos=args.a1_pos,
        prefix="sq",
    )

    print("[3/4] Piece-crop + [4/4] SVM servo on every square...")
    rng = np.random.default_rng(args.seed)
    per_square: dict[str, dict] = {}
    annotated_count = 0

    # Process in row-major order of the split summary.
    items = sorted(
        split_meta["squares"].items(),
        key=lambda kv: (kv[1]["row"], kv[1]["col"]),
    )
    for name, info in items:
        square = load_image(info["path"])
        annotated, combined, meta = process_square(
            square,
            rng=rng,
            diff_threshold=args.diff_threshold,
            compact_percentile=args.compact_percentile,
            pad_frac=args.pad_frac,
            min_crop_size=args.min_crop_size,
            center_frac=args.center_frac,
            border_frac=args.border_frac,
            svm_c=args.svm_c,
            svm_gamma=args.svm_gamma,
        )

        alg = info.get("algebraic") or f"r{info['row']}c{info['col']}"
        out_name = f"{info['row']:02d}_{info['col']:02d}_{alg}.png"
        save_debug(out_dir / out_name, annotated)
        save_debug(debug_dir / f"{alg}_combined.png", combined)
        annotated_count += 1

        entry = {
            "file": out_name,
            "row": info["row"],
            "col": info["col"],
            "algebraic": alg,
            "square_path": info["path"],
        }
        if meta.get("axes") is not None:
            axes = meta["axes"]
            entry.update(
                {
                    "cx_px": round(float(axes["cx"]), 3),
                    "cy_px": round(float(axes["cy"]), 3),
                    "long_axis_deg": axes["long_axis_deg"],
                    "grasp_axis_deg": axes["grasp_axis_deg"],
                    "eccentricity": axes["eccentricity"],
                }
            )
        entry["seg_pixels"] = meta.get("seg_pixels")
        entry["piece_crop_from"] = meta.get("piece_crop_from")
        per_square[alg] = entry
        print(f"  wrote {out_name}")

    summary = {
        "pipeline": [
            "board_detect_lc2fen",
            "split_squares_trivial",
            "piece_from_square_crop",
            "svm_servo",
        ],
        "source_image": str(args.image),
        "out_dir": str(out_dir),
        "a1_pos": args.a1_pos,
        "annotated_count": annotated_count,
        "board": board_meta,
        "split": {
            "square_size_px": split_meta["square_size_px"],
            "count": split_meta["count"],
        },
        "squares": per_square,
    }
    (out_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=(
            Path(__file__).resolve().parent.parent
            / "LiveChess2FEN_setup"
            / "testing"
            / "example_imA"
            / "im2.jpeg"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "pipeline_test01",
    )
    parser.add_argument(
        "--a1-pos",
        choices=["TL", "TR", "BL", "BR"],
        default="TR",
        help="Reorient warped board so algebraic labels match (TR worked for example_imA).",
    )
    parser.add_argument("--diff-threshold", type=int, default=22)
    parser.add_argument("--compact-percentile", type=float, default=82.0)
    parser.add_argument("--pad-frac", type=float, default=0.30)
    parser.add_argument("--min-crop-size", type=int, default=64)
    parser.add_argument("--center-frac", type=float, default=0.32)
    parser.add_argument("--border-frac", type=float, default=0.12)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", default="scale")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.image = args.image.resolve()
    args.out_dir = args.out_dir.resolve()
    try:
        args.svm_gamma = float(args.svm_gamma)
    except (TypeError, ValueError):
        pass

    summary = run(args)
    print(json.dumps({k: summary[k] for k in ("annotated_count", "a1_pos", "out_dir", "source_image")}, indent=2))


if __name__ == "__main__":
    main()
