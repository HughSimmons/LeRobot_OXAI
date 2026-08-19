#!/usr/bin/env python3
"""(b) Naive 8x8 grid split of a square warped board image.

Matches LiveChess2FEN's `split_board_image_trivial`: divide the NxN board
into 64 equal square tiles. Row 0 is the top of the image; col 0 is left.

Usage:
  python split_squares_trivial.py \\
      --board output/pipeline_test01/warped_board.jpg \\
      --out-dir output/pipeline_test01/squares
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from board_geometry import load_image, reorient_warped_for_a1, save_debug, square_name


def split_board_trivial(
    board_bgr: np.ndarray,
    out_dir: Path,
    a1_pos: str | None = None,
    prefix: str = "sq",
) -> dict:
    """Split a square board into 64 crops. Optionally reorient so a1 is BL first."""
    if board_bgr.shape[0] != board_bgr.shape[1]:
        raise ValueError(
            f"Board must be square; got shape {board_bgr.shape[:2]}"
        )

    if a1_pos is not None:
        board_bgr = reorient_warped_for_a1(board_bgr, a1_pos)

    out_dir.mkdir(parents=True, exist_ok=True)
    side = board_bgr.shape[0]
    square_size = side // 8
    # Trim any remainder so tiles are exact.
    board_bgr = board_bgr[: square_size * 8, : square_size * 8]

    results: dict[str, dict] = {}
    for row in range(8):
        for col in range(8):
            y0, y1 = row * square_size, (row + 1) * square_size
            x0, x1 = col * square_size, (col + 1) * square_size
            tile = board_bgr[y0:y1, x0:x1]

            # After optional a1=BL reorient: row 7 = rank 1, col 0 = file a.
            if a1_pos is not None:
                file_idx, rank_idx = col, 7 - row
                alg = square_name(file_idx, rank_idx)
                name = f"{prefix}_{alg}_r{row}_c{col}.jpg"
            else:
                alg = None
                name = f"{prefix}_r{row}_c{col}.jpg"

            path = out_dir / name
            save_debug(path, tile)
            results[name] = {
                "path": str(path),
                "row": row,
                "col": col,
                "algebraic": alg,
                "roi_xyxy": [x0, y0, x1, y1],
            }

    summary = {
        "method": "split_board_image_trivial",
        "board_size_px": int(board_bgr.shape[0]),
        "square_size_px": int(square_size),
        "a1_pos": a1_pos,
        "count": len(results),
        "squares": results,
    }
    (out_dir / "squares_meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, required=True, help="Square warped board image")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "pipeline_test01" / "squares",
    )
    parser.add_argument(
        "--a1-pos",
        choices=["TL", "TR", "BL", "BR"],
        default=None,
        help="If set, reorient warped board so a1 ends at bottom-left before splitting.",
    )
    parser.add_argument("--prefix", type=str, default="sq")
    args = parser.parse_args()

    board = load_image(args.board)
    summary = split_board_trivial(board, args.out_dir, a1_pos=args.a1_pos, prefix=args.prefix)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
