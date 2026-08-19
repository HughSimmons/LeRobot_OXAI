#!/usr/bin/env python3
"""Pipeline test 02: same as test01, with signed light/dark piece isolation.

Rank prior (algebraic, after a1 reorient):
  rank < 5  → light / white pieces
  rank > 5  → dark / black pieces
  rank == 5 → unsigned absolute-deviation fallback (ambiguous mid-board)

Usage:
  python run_pipeline_test02.py \\
      --image input/im2.jpeg \\
      --out-dir output/pipeline_test02 \\
      --a1-pos TR
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from board_detect_lc2fen import detect_and_warp_board
from board_geometry import largest_component_mask, load_image, save_debug
from localise_all_squares import isolate_square_mask, isolate_square_mask_signed
from piece_from_square_crop import compact_blob, extract_piece_crop
from run_pipeline_test01 import annotate_piece_crop
from split_squares_trivial import split_board_trivial
from test_03_svm_servo import draw_mask_axes, draw_training_overlay, principal_axes, svm_segment


def polarity_from_algebraic(alg: str | None) -> str | None:
    """Return 'light', 'dark', or None (unsigned) from algebraic square name."""
    if not alg:
        return None
    match = re.fullmatch(r"([a-h])([1-8])", alg.lower())
    if not match:
        return None
    rank = int(match.group(2))
    if rank < 5:
        return "light"
    if rank > 5:
        return "dark"
    return None


def process_square(
    square_bgr: np.ndarray,
    rng: np.random.Generator,
    polarity: str | None,
    dark_threshold: int,
    light_threshold: int,
    diff_threshold: int,
    compact_percentile: float,
    pad_frac: float,
    min_crop_size: int,
    center_frac: float,
    border_frac: float,
    svm_c: float,
    svm_gamma,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Square → signed (or unsigned) piece crop → SVM axes."""
    if polarity is None:
        mask = isolate_square_mask(square_bgr, diff_threshold)
        mask_mode = "unsigned"
    else:
        mask = isolate_square_mask_signed(
            square_bgr,
            polarity=polarity,
            dark_threshold=dark_threshold,
            light_threshold=light_threshold,
        )
        mask_mode = f"signed_{polarity}"

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
    # Stamp polarity on the annotated crop for quick visual QA.
    label = polarity or "unsigned"
    cv2.putText(
        annotated,
        label,
        (6, annotated.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255) if polarity == "light" else ((255, 180, 0) if polarity == "dark" else (200, 200, 200)),
        1,
        cv2.LINE_AA,
    )
    mask_axes = draw_mask_axes(seg, axes)
    combined = np.hstack([train_overlay, annotated, mask_axes])

    meta = {
        **piece_meta,
        "polarity": polarity,
        "mask_mode": mask_mode,
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

    print("[3/4] Signed piece-crop + [4/4] SVM servo on every square...")
    print("  prior: rank < 5 → light, rank > 5 → dark, rank == 5 → unsigned")
    rng = np.random.default_rng(args.seed)
    per_square: dict[str, dict] = {}
    annotated_count = 0

    items = sorted(
        split_meta["squares"].items(),
        key=lambda kv: (kv[1]["row"], kv[1]["col"]),
    )
    for name, info in items:
        square = load_image(info["path"])
        alg = info.get("algebraic") or f"r{info['row']}c{info['col']}"
        polarity = polarity_from_algebraic(info.get("algebraic"))

        annotated, combined, meta = process_square(
            square,
            rng=rng,
            polarity=polarity,
            dark_threshold=args.dark_threshold,
            light_threshold=args.light_threshold,
            diff_threshold=args.diff_threshold,
            compact_percentile=args.compact_percentile,
            pad_frac=args.pad_frac,
            min_crop_size=args.min_crop_size,
            center_frac=args.center_frac,
            border_frac=args.border_frac,
            svm_c=args.svm_c,
            svm_gamma=args.svm_gamma,
        )

        out_name = f"{info['row']:02d}_{info['col']:02d}_{alg}.png"
        save_debug(out_dir / out_name, annotated)
        save_debug(debug_dir / f"{alg}_combined.png", combined)
        annotated_count += 1

        entry = {
            "file": out_name,
            "row": info["row"],
            "col": info["col"],
            "algebraic": alg,
            "polarity": polarity,
            "mask_mode": meta.get("mask_mode"),
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
        print(f"  wrote {out_name}  polarity={polarity or 'unsigned'}")

    summary = {
        "pipeline": [
            "board_detect_lc2fen",
            "split_squares_trivial",
            "signed_piece_from_square_crop",
            "svm_servo",
        ],
        "rank_prior": {
            "rank_lt_5": "light",
            "rank_gt_5": "dark",
            "rank_eq_5": "unsigned",
        },
        "dark_threshold": args.dark_threshold,
        "light_threshold": args.light_threshold,
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
        default=Path(__file__).resolve().parent / "output" / "pipeline_test02",
    )
    parser.add_argument(
        "--a1-pos",
        choices=["TL", "TR", "BL", "BR"],
        default="TR",
        help="Reorient warped board so algebraic labels match (TR worked for example_imA).",
    )
    parser.add_argument("--dark-threshold", type=int, default=22)
    parser.add_argument("--light-threshold", type=int, default=12)
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
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "annotated_count",
                    "a1_pos",
                    "out_dir",
                    "source_image",
                    "rank_prior",
                    "dark_threshold",
                    "light_threshold",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
