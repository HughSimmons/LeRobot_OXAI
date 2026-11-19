#!/usr/bin/env python3
"""Batch gold-mask labeling for any mask-batch image (crops or full boards).

Drop images into ``input/``. For each image, build a gold mask of **all**
visible piece silhouettes by committing one SAM piece at a time into a union.

  cd gold_mask_batch
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  python run_gold_mask_batch.py

Workflow per image:
  1. LMB/RMB refine the *current* piece proposal (SAM).
  2. ``a`` (or Space) — add current proposal into the committed multi-piece mask.
  3. Repeat for other pieces; ``b`` undoes the last committed piece.
  4. Enter/``s`` — write image+mask pair and advance.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

APP_ROOT = Path(__file__).resolve().parent
# Documents/Vs/coord_finder layout and LeRobot lowlevel/ + lowlevel/coord_finder.
for candidate in (APP_ROOT.parent, APP_ROOT.parent / "coord_finder"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from piece_mask_sam import _MODEL_ID, mask_from_points  # noqa: E402

WINDOW = "gold_mask_batch"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}

# Always shown on the live window so teammates can pick up without the terminal.
CONTROLS_LINES = (
    "LMB=FG   RMB=BG   a/Space=add piece   b=undo piece   u=undo point   r=reset",
    "Enter/s=accept union   n=next   p=prev   x=skip   q=quit",
)


def collect_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")
    paths = sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix in IMAGE_EXTS
    )
    if not paths:
        raise SystemExit(
            f"No images in {input_dir}. Drop .png/.jpg files there and re-run."
        )
    return paths


def stem_for(path: Path, input_dir: Path) -> str:
    try:
        rel = path.resolve().relative_to(input_dir.resolve())
    except ValueError:
        rel = Path(path.name)
    return "__".join(rel.with_suffix("").parts)


def tint_layers(
    image: np.ndarray,
    committed: np.ndarray | None,
    proposal: np.ndarray | None,
) -> np.ndarray:
    """Green = committed pieces; cyan = current SAM proposal."""
    vis = image.copy()
    paint = vis.copy()
    if committed is not None and int(np.count_nonzero(committed)) > 0:
        paint[committed > 0] = (0, 220, 80)
    if proposal is not None and int(np.count_nonzero(proposal)) > 0:
        paint[proposal > 0] = (255, 220, 0)
    return cv2.addWeighted(paint, 0.45, vis, 0.55, 0)


class BatchApp:
    def __init__(
        self,
        images: list[Path],
        input_dir: Path,
        output_dir: Path,
        *,
        overwrite: bool = False,
    ) -> None:
        self.images = images
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.output_dir / "index.jsonl"
        self.index = 0
        self.points: list[tuple[float, float]] = []
        self.labels: list[int] = []
        self.proposal: np.ndarray | None = None
        self.score: float | None = None
        self.ok: bool | None = None
        self.fail: str | None = None
        self.committed: np.ndarray | None = None
        self.piece_stack: list[np.ndarray] = []
        self.dirty = False
        self.image: np.ndarray | None = None
        print(f"Loaded {len(images)} images from {input_dir}")
        for line in CONTROLS_LINES:
            print(f"  {line}")
        self.load_index(0, warn_dirty=False, skip_done=True)

    def dest_dir(self, path: Path | None = None) -> Path:
        path = path if path is not None else self.images[self.index]
        return self.output_dir / stem_for(path, self.input_dir)

    def already_done(self, path: Path) -> bool:
        return (self.dest_dir(path) / "mask.png").is_file()

    def final_mask(self) -> np.ndarray | None:
        """Committed union, optionally OR current ok proposal."""
        assert self.image is not None
        h, w = self.image.shape[:2]
        out = np.zeros((h, w), dtype=np.uint8)
        if self.committed is not None:
            out = np.maximum(out, (self.committed > 0).astype(np.uint8) * 255)
        if self.proposal is not None and self.ok:
            out = np.maximum(out, (self.proposal > 0).astype(np.uint8) * 255)
        if int(np.count_nonzero(out)) == 0:
            return None
        return out

    def load_index(
        self,
        i: int,
        *,
        warn_dirty: bool = True,
        skip_done: bool = False,
    ) -> None:
        if warn_dirty and self.dirty:
            print(
                "Warning: discarding unsaved work on "
                f"{stem_for(self.images[self.index], self.input_dir)}"
            )
        n = len(self.images)
        start = int(i) % n
        idx = start
        for _ in range(n):
            path = self.images[idx]
            if skip_done and not self.overwrite and self.already_done(path):
                print(f"  skip (exists): {stem_for(path, self.input_dir)}")
                idx = (idx + 1) % n
                if idx == start:
                    print("All images already labeled (use --overwrite to redo).")
                    self.index = idx
                    self.image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    self._clear_all()
                    return
                continue
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Skip unreadable {path}")
                idx = (idx + 1) % n
                continue
            self.index = idx
            self.image = img
            self._clear_all()
            done = " [done]" if self.already_done(path) else ""
            print(f"[{self.index + 1}/{n}] {path}{done}")
            return
        raise SystemExit("No readable images.")

    def _clear_proposal(self) -> None:
        self.points = []
        self.labels = []
        self.proposal = None
        self.score = None
        self.ok = None
        self.fail = None

    def _clear_all(self) -> None:
        self._clear_proposal()
        self.committed = None
        self.piece_stack = []
        self.dirty = False

    def rerun_sam(self) -> None:
        assert self.image is not None
        if not self.points:
            self.proposal = None
            self.score = None
            self.ok = None
            self.fail = None
            return
        try:
            result = mask_from_points(self.image, self.points, self.labels)
        except Exception as exc:  # noqa: BLE001
            print(f"SAM error: {exc}")
            self.proposal = None
            self.score = None
            self.ok = False
            self.fail = f"error:{exc}"
            return
        self.proposal = result["mask"]
        self.score = float(result["score"])
        self.ok = bool(result["ok"])
        self.fail = result.get("fail")
        print(
            f"  proposal points={len(self.points)}  pixels={int(result['pixels'])}  "
            f"score={self.score:.3f}  ok={self.ok}  fail={self.fail}  "
            f"pieces={len(self.piece_stack)}"
        )

    def on_mouse(self, event: int, x: int, y: int, flags: int, _param) -> None:
        del flags
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((float(x), float(y)))
            self.labels.append(1)
            self.dirty = True
            self.rerun_sam()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.append((float(x), float(y)))
            self.labels.append(0)
            self.dirty = True
            self.rerun_sam()

    def undo_point(self) -> None:
        if not self.points:
            return
        self.points.pop()
        self.labels.pop()
        self.dirty = bool(self.points) or bool(self.piece_stack)
        self.rerun_sam()

    def add_piece(self) -> None:
        """Commit current SAM proposal into the multi-piece union."""
        assert self.image is not None
        if self.proposal is None or not self.ok:
            print("No ok proposal to add — refine with FG/BG clicks first.")
            return
        piece = (self.proposal > 0).astype(np.uint8) * 255
        self.piece_stack.append(piece.copy())
        if self.committed is None:
            self.committed = piece.copy()
        else:
            self.committed = np.maximum(self.committed, piece)
        self._clear_proposal()
        self.dirty = True
        print(
            f"  added piece #{len(self.piece_stack)}  "
            f"union_px={int(np.count_nonzero(self.committed))}"
        )

    def undo_piece(self) -> None:
        if not self.piece_stack:
            print("No committed pieces to undo.")
            return
        self.piece_stack.pop()
        if not self.piece_stack:
            self.committed = None
        else:
            assert self.image is not None
            h, w = self.image.shape[:2]
            union = np.zeros((h, w), dtype=np.uint8)
            for piece in self.piece_stack:
                union = np.maximum(union, piece)
            self.committed = union
        self.dirty = bool(self.piece_stack) or bool(self.points)
        n = len(self.piece_stack)
        px = 0 if self.committed is None else int(np.count_nonzero(self.committed))
        print(f"  undone piece — remaining={n}  union_px={px}")

    def reset(self) -> None:
        self._clear_all()
        print("Reset all pieces and points.")

    def compose(self, *, show_controls: bool = True) -> np.ndarray:
        assert self.image is not None
        vis = tint_layers(self.image, self.committed, self.proposal if self.ok else None)
        for (x, y), lab in zip(self.points, self.labels):
            color = (0, 255, 0) if lab == 1 else (0, 0, 255)
            cv2.circle(vis, (int(x), int(y)), 5, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        path = self.images[self.index]
        stem = stem_for(path, self.input_dir)
        fg = sum(1 for L in self.labels if L == 1)
        bg = sum(1 for L in self.labels if L == 0)
        prop_px = int(np.count_nonzero(self.proposal)) if self.proposal is not None else 0
        union_px = int(np.count_nonzero(self.committed)) if self.committed is not None else 0
        line1 = f"{self.index + 1}/{len(self.images)}  {stem}"
        line2 = (
            f"pieces={len(self.piece_stack)} union={union_px}  "
            f"prop FG={fg} BG={bg} px={prop_px} ok={self.ok}"
        )
        if self.fail:
            line2 += f"  fail={self.fail}"
        if self.dirty:
            line2 += "  [unsaved]"
        if self.already_done(path):
            line2 += "  [on disk]"
        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (w, 44), (20, 20, 20), -1)
        cv2.putText(
            vis, line1[:100], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA
        )
        cv2.putText(
            vis, line2[:100], (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 220, 255), 1, cv2.LINE_AA
        )
        if show_controls:
            bar_h = 52
            y0 = max(0, h - bar_h)
            cv2.rectangle(vis, (0, y0), (w, h), (20, 20, 20), -1)
            # ~7 px/char at scale 0.42; keep readable on narrow crops.
            max_chars = max(28, w // 7)
            font = 0.40 if w < 700 else 0.42
            cv2.putText(
                vis,
                CONTROLS_LINES[0][:max_chars],
                (8, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                font,
                (180, 255, 180),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                CONTROLS_LINES[1][:max_chars],
                (8, y0 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                font,
                (180, 255, 180),
                1,
                cv2.LINE_AA,
            )
        return vis

    def accept(self) -> None:
        assert self.image is not None
        # Auto-commit current proposal if ok and not yet added.
        if self.proposal is not None and self.ok:
            self.add_piece()
        mask = self.final_mask()
        if mask is None:
            print("Nothing to save — add at least one piece (FG clicks then a).")
            return
        path = self.images[self.index]
        stem = stem_for(path, self.input_dir)
        dest = self.dest_dir(path)
        if dest.is_dir() and not self.overwrite and (dest / "mask.png").is_file():
            print(f"Already exists (pass --overwrite): {dest}")
            return
        dest.mkdir(parents=True, exist_ok=True)
        overlay = self.compose(show_controls=False)
        cv2.imwrite(str(dest / "image.png"), self.image)
        cv2.imwrite(str(dest / "mask.png"), mask)
        cv2.imwrite(str(dest / "overlay.png"), overlay)
        pixels = int(np.count_nonzero(mask))
        meta = {
            "source": str(path.resolve()),
            "source_relative": str(path.resolve().relative_to(self.input_dir.resolve())),
            "stem": stem,
            "model_id": _MODEL_ID,
            "n_pieces": len(self.piece_stack),
            "points_xy": [[float(x), float(y)] for x, y in self.points],
            "labels": [int(L) for L in self.labels],
            "score": self.score,
            "ok": True,
            "fail": None,
            "pixels": pixels,
            "image_hw": [int(self.image.shape[0]), int(self.image.shape[1])],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "stem": stem,
                        "source": meta["source"],
                        "pixels": pixels,
                        "n_pieces": meta["n_pieces"],
                        "saved_at": meta["saved_at"],
                    }
                )
                + "\n"
            )
        self.dirty = False
        print(f"Accepted {dest}  pieces={meta['n_pieces']}  px={pixels}")
        self.load_index(self.index + 1, warn_dirty=False, skip_done=not self.overwrite)

    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while True:
            cv2.imshow(WINDOW, self.compose())
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (13, 10, ord("s")):
                self.accept()
            elif key in (ord("a"), ord(" ")):
                self.add_piece()
            elif key == ord("b"):
                self.undo_piece()
            elif key == ord("u"):
                self.undo_point()
            elif key == ord("r"):
                self.reset()
            elif key == ord("n"):
                self.load_index(self.index + 1, skip_done=False)
            elif key == ord("p"):
                self.load_index(self.index - 1, skip_done=False)
            elif key == ord("x"):
                print(f"Skip {stem_for(self.images[self.index], self.input_dir)}")
                self.load_index(self.index + 1, warn_dirty=True, skip_done=not self.overwrite)
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=APP_ROOT / "input",
        help="Folder of images (default: ./input)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=APP_ROOT / "output",
        help="Dataset output folder (default: ./output)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow re-labeling images that already have mask.png",
    )
    args = parser.parse_args(argv)
    input_dir = args.input if args.input.is_absolute() else (APP_ROOT / args.input)
    output_dir = args.output if args.output.is_absolute() else (APP_ROOT / args.output)
    images = collect_images(input_dir)
    app = BatchApp(images, input_dir, output_dir, overwrite=args.overwrite)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
