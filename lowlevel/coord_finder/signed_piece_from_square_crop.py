#!/usr/bin/env python3
"""Piece-centred crops using signed light/dark deviation from wood median.

Same architecture as ``piece_from_square_crop.py``, but the foreground mask
only accepts one polarity:

  dark  → I < m − τ₋   (firm τ₋; dark pieces usually have large contrast)
  light → I > m + τ₊   (generous τ₊; pale pieces often sit close to wood)

Opposite-polarity pixels (e.g. shadows when hunting a white piece) are ignored.

Usage:
  python signed_piece_from_square_crop.py \\
      --square-dir output/square_crop \\
      --out-dir output/piece_crops_signed \\
      --polarity light
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from board_geometry import largest_component_mask, load_image, save_debug
from localise_all_squares import blob_fill_ratio, isolate_square_mask_signed
from piece_from_square_crop import (
    compact_blob,
    draw_debug,
    extract_piece_crop,
    list_images,
)


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
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        mask = isolate_square_mask_signed(
            img,
            polarity=args.polarity,
            dark_threshold=args.dark_threshold,
            light_threshold=args.light_threshold,
        )
        mask = largest_component_mask(mask)
        mask = compact_blob(mask, keep_percentile=args.compact_percentile)
        n = int(np.count_nonzero(mask))
        fill = blob_fill_ratio(mask)

        entry: dict = {
            "found": False,
            "polarity": args.polarity,
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
        "method": "signed_blob_piece_crop",
        "polarity": args.polarity,
        "dark_threshold": args.dark_threshold,
        "light_threshold": args.light_threshold,
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
        default=Path(__file__).resolve().parent / "output" / "piece_crops_signed",
    )
    parser.add_argument(
        "--polarity",
        choices=["light", "dark", "white", "black"],
        required=True,
        help="Expected piece colour polarity relative to local wood median.",
    )
    parser.add_argument(
        "--dark-threshold",
        type=int,
        default=22,
        help="τ₋ for dark pieces: keep I < m − τ₋ (firm).",
    )
    parser.add_argument(
        "--light-threshold",
        type=int,
        default=12,
        help="τ₊ for light pieces: keep I > m + τ₊ (generous; pale-on-pale).",
    )
    parser.add_argument("--min-blob-pixels", type=int, default=80)
    parser.add_argument("--min-fill-ratio", type=float, default=0.12)
    parser.add_argument("--pad-frac", type=float, default=0.30)
    parser.add_argument("--compact-percentile", type=float, default=82.0)
    parser.add_argument("--min-crop-size", type=int, default=64)
    args = parser.parse_args()

    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
