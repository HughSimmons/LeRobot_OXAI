#!/usr/bin/env python3
"""Grow each photo's board corners outward from its marked square, for a lens fit.

Reads a photo folder's labels.json (the four marked corners of one square, green = a1-near,
then counter-clockwise) and, starting from those four, walks outward over the board's inner
corners: each next corner is predicted by a homography of the nearest found ones, refined by
cv2.cornerSubPix, and kept only if the four quarter-squares round it alternate dark/light
(diagonal pairs alike, the pairs contrasting) and it is clear of the frame edge and, for the
wrist camera, the jaws. Corners on the board's outer edge (index 0 or 8) are never taken: they
meet the wooden frame, not other squares.

Writes corners.json into the folder: per photo, {"i,j": [x, y]}, where i counts squares along
the files from a1 (toward h) and j along the ranks (toward 8), so d4 is 3,3 / 4,3 / 4,4 / 3,4.
The automatic grid detector is not used: on the wrist photos it took jaw edges for corners.

  ~/venvs/click3d/bin/python grow_corners.py output/stationary_20261107_001628
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from plan_wrist_poses import JAW_POLYGON  # also puts click_to_3d_oliver/ on sys.path
from click3d_shared import square_file_rank  # noqa: E402

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
FRAME_MARGIN_PX = 8.0
JAW_MARGIN_PX = 12.0


def checker(gray: np.ndarray, H: np.ndarray, i: int, j: int, shift: np.ndarray) -> bool:
    """The four quarter-squares round board corner (i, j): diagonal pairs alike, the pairs contrasting."""
    q = {}
    for di in (-1, 1):
        for dj in (-1, 1):
            c = cv2.perspectiveTransform(np.array([[[i + 0.3 * di, j + 0.3 * dj]]], np.float32), H).reshape(2) + shift
            x, y = int(round(c[0])), int(round(c[1]))
            if not (2 <= x < gray.shape[1] - 2 and 2 <= y < gray.shape[0] - 2):
                return False
            q[di, dj] = float(np.median(gray[y - 2:y + 3, x - 2:x + 3]))
    a = (q[1, 1] + q[-1, -1]) / 2
    b = (q[1, -1] + q[-1, 1]) / 2
    within = max(abs(q[1, 1] - q[-1, -1]), abs(q[1, -1] - q[-1, 1]))
    return abs(a - b) > 30 and within < 0.6 * abs(a - b)


def clear(p: np.ndarray, size: tuple[int, int], jaws: np.ndarray | None) -> bool:
    w, h = size
    if not (FRAME_MARGIN_PX < p[0] < w - FRAME_MARGIN_PX and FRAME_MARGIN_PX < p[1] < h - FRAME_MARGIN_PX):
        return False
    return jaws is None or cv2.pointPolygonTest(jaws, (float(p[0]), float(p[1])), True) < -JAW_MARGIN_PX


def grow(gray: np.ndarray, square: str, marked: list[list[float]], jaws: np.ndarray | None) -> dict[tuple[int, int], list[float]]:
    file_i, rank_i = square_file_rank(square)
    keys = [(file_i, rank_i), (file_i + 1, rank_i), (file_i + 1, rank_i + 1), (file_i, rank_i + 1)]
    found = {k: list(map(float, p)) for k, p in zip(keys, marked)}
    rejected: set[tuple[int, int]] = set()
    size = (gray.shape[1], gray.shape[0])
    while True:
        added = False
        frontier = {(i + di, j + dj) for (i, j) in found for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))}
        for key in sorted(frontier - set(found) - rejected):
            i, j = key
            if not (1 <= i <= 7 and 1 <= j <= 7):
                continue
            near = sorted(found, key=lambda k: (k[0] - i) ** 2 + (k[1] - j) ** 2)[:8]
            src = np.array(near, np.float32)
            dst = np.array([found[k] for k in near], np.float32)
            H, _ = cv2.findHomography(src, dst)
            if H is None:
                continue
            guess = cv2.perspectiveTransform(np.array([[[i, j]]], np.float32), H).reshape(2)
            spacing = min(float(np.linalg.norm(dst[0] - d)) for d in dst[1:])
            win = int(max(3, min(7, spacing / 5)))
            if not clear(guess, size, jaws):
                rejected.add(key)
                continue
            p = cv2.cornerSubPix(gray, guess.reshape(1, 1, 2).astype(np.float32), (win, win), (-1, -1), CRITERIA).reshape(2)
            if (np.linalg.norm(p - guess) > max(4.0, 0.15 * spacing) or not checker(gray, H, i, j, p - guess)
                    or not clear(p, size, jaws)):
                rejected.add(key)
                continue
            found[key] = p.tolist()
            added = True
        if not added:
            return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="photo folder with labels.json")
    args = parser.parse_args()
    labels = json.loads((args.folder / "labels.json").read_text(encoding="utf-8"))
    jaws = JAW_POLYGON.astype(np.float32) if labels["camera"] == "wrist" else None
    out = {}
    for shot in labels["shots"]:
        gray = cv2.cvtColor(cv2.imread(str(args.folder / Path(shot["image"]).name)), cv2.COLOR_BGR2GRAY)
        found = grow(gray, shot["square"], shot["corners_px"], jaws)
        out[Path(shot["image"]).name] = {f"{i},{j}": [round(v, 3) for v in p] for (i, j), p in sorted(found.items())}
        print(f"{shot['image']}: {len(found)} corners")
    record = {
        "note": "Inner board corners grown from the marked square (grow_corners.py). Key 'i,j' = board corner i squares "
                "along the files (toward h) and j along the ranks (toward 8) from a1.",
        "camera": labels["camera"],
        "square_mm": labels.get("square_mm"),
        "corners_px": out,
    }
    (args.folder / "corners.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"{sum(len(v) for v in out.values())} corners in {len(out)} photos -> {args.folder / 'corners.json'}")


if __name__ == "__main__":
    main()
