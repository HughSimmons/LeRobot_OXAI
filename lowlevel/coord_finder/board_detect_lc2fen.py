#!/usr/bin/env python3
"""(a) Detect and warp a chessboard using LiveChess2FEN's detect_board.

Uses the multi-layer iterative detector in
`LiveChess2FEN/lc2fen/detectboard/detect_board.py` (slid → laps → cps),
writes a square rectified board image, and saves the four outer corners
in the original image for later use.

Usage:
  python board_detect_lc2fen.py \\
      --image ../LiveChess2FEN_setup/testing/example_imA/im2.jpeg \\
      --out-dir output/pipeline_test01
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def _lc2fen_repo() -> Path:
    return Path(__file__).resolve().parent.parent / "LiveChess2FEN_setup" / "LiveChess2FEN"


def detect_and_warp_board(
    image_path: Path,
    out_dir: Path,
    board_corners: list[list[int]] | None = None,
) -> dict:
    """Run LC2FEN board detection; return paths + corner metadata."""
    repo = _lc2fen_repo()
    if not repo.is_dir():
        raise FileNotFoundError(f"LiveChess2FEN repo not found at {repo}")

    image_path = Path(image_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "lc2fen_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # LC2FEN writes debug images relative to cwd.
    prev_cwd = Path.cwd()
    prev_path = list(sys.path)
    try:
        os.chdir(repo)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        from lc2fen.detectboard.detect_board import compute_corners, detect

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        warped_path = work_dir / "warped_board.jpg"
        image_object = detect(image, str(warped_path), board_corners)
        corners, square_corners = compute_corners(image_object)

        # Copy warped board into the out_dir root for easy access.
        final_warped = out_dir / "warped_board.jpg"
        warped = cv2.imread(str(warped_path))
        if warped is None:
            raise RuntimeError(f"Detection did not write a board image at {warped_path}")
        # Ensure square (trivial split requires equal sides).
        side = min(warped.shape[0], warped.shape[1])
        warped = warped[:side, :side]
        cv2.imwrite(str(final_warped), warped)

        meta = {
            "source_image": str(image_path),
            "warped_board": str(final_warped),
            "board_size_px": int(side),
            "board_corners": np.asarray(corners, dtype=float).tolist(),
            "square_corners_count": int(len(square_corners)),
            "method": "livechess2fen_detect_board",
        }
        meta_path = out_dir / "board_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        meta["meta_path"] = str(meta_path)
        return meta
    finally:
        os.chdir(prev_cwd)
        sys.path[:] = prev_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "pipeline_test01",
    )
    args = parser.parse_args()
    meta = detect_and_warp_board(args.image, args.out_dir)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
