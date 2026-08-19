#!/usr/bin/env python3
"""Localise piece centres on every board square (0-100 percentage coords).

Uses a reference frame for board pose (and optional background subtraction).
When the reference is itself occupied (e.g. starting position), use
`--occupancy-mode square` to detect pieces from within-square contrast instead.

Coordinate convention:
  (0, 0) = bottom-left of a1
  (100, 100) = top-right of h8

Usage:
  python localise_all_squares.py \\
      --empty ../LiveChess2FEN_setup/testing/example_imA/im1.jpeg \\
      --board ../LiveChess2FEN_setup/testing/example_imA/im2.jpeg \\
      --occupancy-mode square \\
      --a1-pos BL \\
      --min-diff-pixels 120 \\
      --diff-threshold 28
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

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


def isolate_diff_mask(
    warped_empty: np.ndarray,
    warped_board: np.ndarray,
    diff_threshold: int,
) -> np.ndarray:
    """Full-board binary mask from absdiff. Keeps *all* components."""
    diff = cv2.absdiff(warped_board, warped_empty)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, diff_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def isolate_square_mask(square_bgr: np.ndarray, diff_threshold: int) -> np.ndarray:
    """Foreground mask inside one square using deviation from local wood colour."""
    gray = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    base = float(np.median(blur))
    abs_dev = cv2.absdiff(blur, np.full_like(blur, int(round(base))))

    # Adaptive floor: captures pale pieces on pale wood when fixed threshold is low.
    thr = max(diff_threshold, int(np.percentile(abs_dev, 70)))
    _, mask = cv2.threshold(abs_dev, thr, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    soft = (abs_dev > max(diff_threshold // 2, 8)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask, cv2.bitwise_and(edges, soft))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    margin = max(2, square_bgr.shape[0] // 14)
    mask[:margin, :] = 0
    mask[-margin:, :] = 0
    mask[:, :margin] = 0
    mask[:, -margin:] = 0
    return mask


def isolate_square_mask_signed(
    square_bgr: np.ndarray,
    polarity: str,
    dark_threshold: int = 22,
    light_threshold: int = 12,
) -> np.ndarray:
    """Foreground mask using *signed* deviation from local wood median.

    polarity:
      - ``"dark"`` / ``"black"``: keep only pixels darker than the wood
        (``I < m - τ₋``). Shadows share this polarity, so ``τ₋`` stays firm.
      - ``"light"`` / ``"white"``: keep only pixels brighter than the wood
        (``I > m + τ₊``). Pale pieces often have small upward contrast, so
        ``τ₊`` is more generous by default and does not get raised by a high
        ``P₇₀`` of absolute deviations.

    Opposite-polarity pixels (e.g. dark shadows when hunting a white piece)
    are discarded.
    """
    pol = polarity.lower().strip()
    if pol in {"dark", "black", "b"}:
        expect_dark = True
    elif pol in {"light", "white", "w"}:
        expect_dark = False
    else:
        raise ValueError(f"polarity must be light/dark (got {polarity!r})")

    gray = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    base = float(np.median(blur))
    signed = blur.astype(np.float32) - base

    if expect_dark:
        # How far below the wood median (0 on the bright side).
        score = np.maximum(-signed, 0.0)
        side = score[score > 0]
        thr = float(dark_threshold)
        if side.size >= 32:
            # Allow adaptive raise only on the dark side (strong dark pieces).
            thr = max(thr, float(np.percentile(side, 70)))
        soft_gate = max(dark_threshold / 2.0, 8.0)
    else:
        # How far above the wood median (0 on the dark side).
        score = np.maximum(signed, 0.0)
        # Pale-on-pale: keep a generous fixed floor; do not raise via P70.
        thr = float(light_threshold)
        soft_gate = max(light_threshold / 2.0, 5.0)

    mask = (score > thr).astype(np.uint8) * 255

    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    soft = (score > soft_gate).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask, cv2.bitwise_and(edges, soft))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    margin = max(2, square_bgr.shape[0] // 14)
    mask[:margin, :] = 0
    mask[-margin:, :] = 0
    mask[:, :margin] = 0
    mask[:, -margin:] = 0
    return mask


def blob_fill_ratio(mask: np.ndarray) -> float:
    """Filled-area / bounding-box area of the largest blob (0 if empty)."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    h = int(ys.max() - ys.min()) + 1
    w = int(xs.max() - xs.min()) + 1
    return float(len(xs)) / float(h * w)


def contact_centroid(mask: np.ndarray, bottom_fraction: float = 0.28) -> tuple[float, float] | None:
    """Centre of the lowest band of the silhouette (approximate base contact)."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 8:
        return mask_centroid(mask)

    y_max = int(ys.max())
    height = max(int(ys.max()) - int(ys.min()), 1)
    cutoff = y_max - int(height * bottom_fraction)
    band = ys >= cutoff
    band_mask = np.zeros_like(mask)
    band_mask[ys[band], xs[band]] = 255
    local = mask_centroid(band_mask)
    return local if local is not None else mask_centroid(mask)


def square_diff_stats(
    warped_ref: np.ndarray,
    warped_board: np.ndarray,
    file_idx: int,
    rank_idx: int,
) -> float:
    """Mean absolute difference of one square vs the reference warp."""
    x0, y0, x1, y1 = square_roi(file_idx, rank_idx, warped_board.shape[0])
    a = warped_board[y0:y1, x0:x1].astype(np.float32)
    b = warped_ref[y0:y1, x0:x1].astype(np.float32)
    return float(np.mean(np.abs(a - b)))


def localise_board(
    warped_ref: np.ndarray,
    warped_board: np.ndarray,
    transform: BoardTransform,
    occupancy_mode: str,
    diff_threshold: int,
    min_diff_pixels: int,
    mean_diff_threshold: float,
    min_fill_ratio: float,
) -> dict[str, dict]:
    """Return localisation for all 64 squares."""
    results: dict[str, dict] = {}
    global_diff_mask = None
    if occupancy_mode == "diff":
        global_diff_mask = isolate_diff_mask(warped_ref, warped_board, diff_threshold)

    board_size = transform.board_size_px
    for rank_idx in range(8):
        for file_idx in range(8):
            sq = square_name(file_idx, rank_idx)
            x0, y0, x1, y1 = square_roi(file_idx, rank_idx, board_size)
            crop = warped_board[y0:y1, x0:x1]

            if occupancy_mode == "diff":
                assert global_diff_mask is not None
                mask = global_diff_mask[y0:y1, x0:x1].copy()
                mean_diff = square_diff_stats(warped_ref, warped_board, file_idx, rank_idx)
            else:
                mask = isolate_square_mask(crop, diff_threshold)
                mean_diff = float(np.mean(mask) / 255.0 * 100.0)

            mask = largest_component_mask(mask)
            pixel_count = int(np.count_nonzero(mask))
            fill = blob_fill_ratio(mask)
            occupied = pixel_count >= min_diff_pixels and fill >= min_fill_ratio
            if occupancy_mode == "diff":
                occupied = occupied and mean_diff >= mean_diff_threshold

            entry: dict = {
                "x": None,
                "y": None,
                "found": False,
                "occupied": occupied,
                "diff_pixels": pixel_count,
                "fill_ratio": round(fill, 4),
                "mean_diff": round(mean_diff, 3),
            }

            if occupied:
                centroid = contact_centroid(mask)
                if centroid is not None:
                    x_pct, y_pct = transform.pixel_to_pct(x0 + centroid[0], y0 + centroid[1])
                    entry.update(
                        {
                            "x": round(x_pct, 4),
                            "y": round(y_pct, 4),
                            "found": True,
                        }
                    )

            results[sq] = entry
    return results


def load_transform(empty_img, corners_json: Path | None) -> BoardTransform:
    if corners_json is not None:
        data = json.loads(corners_json.read_text(encoding="utf-8"))
        corners = np.array(data["board_corners"], dtype=np.float32)
        return board_transform_from_corners(corners)
    return estimate_board_transform(empty_img)


def draw_results(warped_board: np.ndarray, results: dict, transform: BoardTransform) -> np.ndarray:
    vis = warped_board.copy()
    for sq, coords in results.items():
        if not coords.get("found"):
            continue
        x_px, y_px = transform.pct_to_pixel(coords["x"], coords["y"])
        cv2.circle(vis, (int(round(x_px)), int(round(y_px))), 5, (0, 0, 255), -1)
        cv2.putText(
            vis,
            sq,
            (int(round(x_px)) + 4, int(round(y_px)) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empty", type=Path, required=True, help="Reference / pose image")
    parser.add_argument("--board", type=Path, required=True, help="Image to localise pieces on")
    parser.add_argument(
        "--occupancy-mode",
        choices=["diff", "square"],
        default="diff",
        help="diff = subtract reference; square = per-square contrast (use when reference has pieces).",
    )
    parser.add_argument(
        "--a1-pos",
        choices=["TL", "TR", "BL", "BR"],
        default="TR",
        help="Where a1 sits after the initial warp (before remapping to BL).",
    )
    parser.add_argument(
        "--diff-threshold",
        type=int,
        default=25,
        help="Intensity delta used to build the piece mask.",
    )
    parser.add_argument(
        "--min-diff-pixels",
        type=int,
        default=400,
        help="Minimum foreground pixels in a square to count as occupied.",
    )
    parser.add_argument(
        "--min-fill-ratio",
        type=float,
        default=0.15,
        help="Minimum filled-area/bbox ratio of the main blob (rejects thin shadows).",
    )
    parser.add_argument(
        "--mean-diff-threshold",
        type=float,
        default=8.0,
        help="(diff mode) Minimum mean abs pixel difference vs reference for occupancy.",
    )
    parser.add_argument("--corners-json", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "example_imA",
    )
    args = parser.parse_args()

    ref_img = load_image(args.empty)
    board_img = load_image(args.board)
    transform = load_transform(ref_img, args.corners_json)

    warped_ref = reorient_warped_for_a1(transform.warp(ref_img), args.a1_pos)
    warped_board = reorient_warped_for_a1(transform.warp(board_img), args.a1_pos)

    results = localise_board(
        warped_ref=warped_ref,
        warped_board=warped_board,
        transform=transform,
        occupancy_mode=args.occupancy_mode,
        diff_threshold=args.diff_threshold,
        min_diff_pixels=args.min_diff_pixels,
        mean_diff_threshold=args.mean_diff_threshold,
        min_fill_ratio=args.min_fill_ratio,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_debug(args.out_dir / "warped_ref.jpg", warped_ref)
    save_debug(args.out_dir / "warped_board.jpg", warped_board)
    save_debug(args.out_dir / "warped_board_marked.jpg", draw_results(warped_board, results, transform))
    if args.occupancy_mode == "diff":
        save_debug(
            args.out_dir / "difference.jpg",
            cv2.absdiff(warped_board, warped_ref),
        )

    occupied = {k: v for k, v in results.items() if v.get("occupied")}
    summary = {
        "occupied_count": len(occupied),
        "empty_count": 64 - len(occupied),
        "occupied_squares": sorted(occupied.keys(), key=lambda s: (int(s[1]), s[0])),
        "pieces": occupied,
        "all_squares": results,
    }
    (args.out_dir / "coords.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
