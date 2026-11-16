#!/usr/bin/env python3
"""Gold-mask helpers for click_to_3d: square crop, SAM click state, save."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from click3d_shared import OUTPUT, ROOT, SQUARE_MM

# Documents/Vs/coord_finder layout and LeRobot lowlevel/ + lowlevel/coord_finder.
for candidate in (ROOT.parent, ROOT.parent / "coord_finder"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from two_view_grasp import grasp_from_views, side_crop_window  # noqa: E402

GOLD_MASKS_DIR = OUTPUT / "gold_masks"
PANEL_W = 640
PANEL_H = 480


@dataclass
class PanelLayout:
    """Letterbox mapping from a 640x480 panel into crop pixels."""

    x0: int
    y0: int
    scale: float
    crop_w: int
    crop_h: int
    panel_w: int = PANEL_W
    panel_h: int = PANEL_H


@dataclass
class GoldViewState:
    view: str
    crop: np.ndarray
    window: tuple[int, int, int, int]
    points: list[tuple[float, float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    mask: np.ndarray | None = None
    score: float | None = None
    ok: bool | None = None
    fail: str | None = None
    dirty: bool = False
    model_id: str | None = None

    def reset_points(self) -> None:
        self.points = []
        self.labels = []
        self.mask = None
        self.score = None
        self.ok = None
        self.fail = None
        self.dirty = False

    def undo(self) -> None:
        if not self.points:
            return
        self.points.pop()
        self.labels.pop()
        self.dirty = bool(self.points)


def build_square_crop(
    image_bgr: np.ndarray,
    square: str,
    T_board_camera: np.ndarray,
    K: np.ndarray,
    square_mm: float = SQUARE_MM,
) -> dict[str, Any]:
    """Crop the square prism window (same as two-view grasp)."""
    window = side_crop_window(
        square,
        T_board_camera,
        K,
        image_bgr.shape,
        square_mm,
    )
    if window is None:
        return {
            "ok": False,
            "fail": "crop_window_none",
            "crop": None,
            "window": None,
        }
    x0, y0, x1, y1 = [int(v) for v in window]
    if x1 <= x0 or y1 <= y0:
        return {
            "ok": False,
            "fail": "crop_empty",
            "crop": None,
            "window": (x0, y0, x1, y1),
        }
    crop = image_bgr[y0:y1, x0:x1].copy()
    if crop.size == 0:
        return {
            "ok": False,
            "fail": "crop_empty",
            "crop": None,
            "window": (x0, y0, x1, y1),
        }
    return {
        "ok": True,
        "fail": None,
        "crop": crop,
        "window": (x0, y0, x1, y1),
    }


def tint_overlay(crop: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    vis = crop.copy()
    if mask is not None and int(np.count_nonzero(mask)) > 0:
        paint = vis.copy()
        paint[mask > 0] = (0, 220, 80)
        vis = cv2.addWeighted(paint, 0.45, vis, 0.55, 0)
    return vis


def rerun(state: GoldViewState) -> None:
    """Re-run SAM from current points. Lazy-imports mask_from_points."""
    if not state.points:
        state.mask = None
        state.score = None
        state.ok = None
        state.fail = None
        return
    from piece_mask_sam import _MODEL_ID, mask_from_points

    try:
        result = mask_from_points(state.crop, state.points, state.labels)
    except Exception as exc:  # noqa: BLE001
        state.mask = None
        state.score = None
        state.ok = False
        state.fail = f"error:{exc}"
        state.model_id = _MODEL_ID
        return
    state.mask = result["mask"]
    state.score = float(result["score"])
    state.ok = bool(result["ok"])
    state.fail = result.get("fail")
    state.model_id = result.get("model_id", _MODEL_ID)


def add_point(state: GoldViewState, x: float, y: float, label: int) -> None:
    h, w = state.crop.shape[:2]
    px = float(max(0.0, min(x, w - 1.0)))
    py = float(max(0.0, min(y, h - 1.0)))
    state.points.append((px, py))
    state.labels.append(1 if int(label) else 0)
    state.dirty = True
    rerun(state)


def compose_crop_panel(
    state: GoldViewState,
    title: str,
    active: bool,
    panel_w: int = PANEL_W,
    panel_h: int = PANEL_H,
) -> tuple[np.ndarray, PanelLayout]:
    """Letterbox crop (+ mask/points) into a fixed panel; return panel + layout."""
    vis = tint_overlay(state.crop, state.mask)
    for (x, y), lab in zip(state.points, state.labels):
        color = (0, 255, 0) if lab == 1 else (0, 0, 255)
        cv2.circle(vis, (int(x), int(y)), 5, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    ch, cw = vis.shape[:2]
    scale = min(panel_w / float(cw), panel_h / float(ch))
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))
    resized = cv2.resize(vis, (nw, nh), interpolation=cv2.INTER_AREA)
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    x0 = (panel_w - nw) // 2
    y0 = (panel_h - nh) // 2
    panel[y0 : y0 + nh, x0 : x0 + nw] = resized

    border = (0, 220, 255) if active else (60, 60, 60)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), border, 2 if active else 1)

    fg = sum(1 for L in state.labels if L == 1)
    bg = sum(1 for L in state.labels if L == 0)
    px = int(np.count_nonzero(state.mask)) if state.mask is not None else 0
    score = f"{state.score:.3f}" if state.score is not None else "-"
    line1 = f"{title}{' *' if active else ''}"
    line2 = f"FG={fg} BG={bg} px={px} sc={score}"
    if state.fail:
        line2 += f" {state.fail}"
    if state.dirty:
        line2 += " [unsaved]"
    cv2.rectangle(panel, (0, 0), (panel_w, 40), (20, 20, 20), -1)
    cv2.putText(panel, line1[:70], (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(panel, line2[:70], (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 220, 255), 1, cv2.LINE_AA)

    layout = PanelLayout(x0=x0, y0=y0, scale=float(scale), crop_w=cw, crop_h=ch, panel_w=panel_w, panel_h=panel_h)
    return panel, layout


def map_panel_click(x: float, y: float, layout: PanelLayout) -> tuple[float, float] | None:
    """Map panel pixel to crop pixel, or None if outside the letterboxed image."""
    lx = x - layout.x0
    ly = y - layout.y0
    disp_w = layout.crop_w * layout.scale
    disp_h = layout.crop_h * layout.scale
    if lx < 0 or ly < 0 or lx >= disp_w or ly >= disp_h:
        return None
    if layout.scale <= 0:
        return None
    cx = lx / layout.scale
    cy = ly / layout.scale
    cx = float(max(0.0, min(cx, layout.crop_w - 1.0)))
    cy = float(max(0.0, min(cy, layout.crop_h - 1.0)))
    return cx, cy


def save_gold(
    dest_dir: Path,
    state: GoldViewState,
    meta_fields: dict[str, Any],
) -> Path:
    """Write crop/mask/overlay/meta for an accepted gold view."""
    if state.mask is None or not state.ok:
        raise ValueError("mask not ok — refine before saving")
    dest_dir.mkdir(parents=True, exist_ok=True)
    overlay = tint_overlay(state.crop, state.mask)
    for (x, y), lab in zip(state.points, state.labels):
        color = (0, 255, 0) if lab == 1 else (0, 0, 255)
        cv2.circle(overlay, (int(x), int(y)), 5, color, -1, lineType=cv2.LINE_AA)
    cv2.imwrite(str(dest_dir / "crop.png"), state.crop)
    cv2.imwrite(str(dest_dir / "mask.png"), state.mask)
    cv2.imwrite(str(dest_dir / "overlay.png"), overlay)
    meta = {
        **meta_fields,
        "view": state.view,
        "window": list(state.window),
        "model_id": state.model_id,
        "points_xy": [[float(x), float(y)] for x, y in state.points],
        "labels": [int(L) for L in state.labels],
        "score": state.score,
        "ok": state.ok,
        "fail": state.fail,
        "pixels": int(np.count_nonzero(state.mask)),
        "crop_hw": [int(state.crop.shape[0]), int(state.crop.shape[1])],
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    (dest_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    state.dirty = False
    return dest_dir


def new_session_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def gold_save_dir(session_id: str, square: str, view: str) -> Path:
    GOLD_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    return GOLD_MASKS_DIR / session_id / f"{square}_{view}"


def paste_mask_to_full(
    full_shape_hw: tuple[int, int],
    crop_mask: np.ndarray,
    window: tuple[int, int, int, int],
) -> np.ndarray:
    """Paste a crop-local mask into a zero full-frame mask at ``window``."""
    h, w = int(full_shape_hw[0]), int(full_shape_hw[1])
    out = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = [int(v) for v in window]
    x0 = max(0, min(x0, w))
    x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h))
    y1 = max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return out
    region = crop_mask
    if region.ndim > 2:
        region = region[:, :, 0]
    rh, rw = y1 - y0, x1 - x0
    if region.shape[0] != rh or region.shape[1] != rw:
        region = cv2.resize(region, (rw, rh), interpolation=cv2.INTER_NEAREST)
    out[y0:y1, x0:x1] = (region > 0).astype(np.uint8) * 255
    return out


def _mask_centroid_px(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [round(float(xs.mean()), 2), round(float(ys.mean()), 2)]


def run_gold_two_view(
    *,
    wrist_bgr: np.ndarray,
    side_bgr: np.ndarray,
    wrist_state: GoldViewState,
    side_state: GoldViewState,
    square: str,
    T_board_wrist: np.ndarray,
    T_board_side: np.ndarray,
    K_wrist: np.ndarray,
    K_side: np.ndarray,
    square_mm: float = SQUARE_MM,
) -> dict[str, Any]:
    """Run grasp_from_views on pasted gold masks. Returns ok/fail payload."""
    if wrist_state.mask is None or not wrist_state.ok:
        return {"ok": False, "fail": "wrist mask not ok"}
    if side_state.mask is None or not side_state.ok:
        return {"ok": False, "fail": "side mask not ok"}

    overhead_mask = paste_mask_to_full(wrist_bgr.shape[:2], wrist_state.mask, wrist_state.window)
    side_mask = paste_mask_to_full(side_bgr.shape[:2], side_state.mask, side_state.window)
    if int(np.count_nonzero(overhead_mask)) < 20:
        return {"ok": False, "fail": "wrist full-frame mask empty"}
    if int(np.count_nonzero(side_mask)) < 20:
        return {"ok": False, "fail": "side full-frame mask empty"}

    # Champion 3D fuse: overhead_ray (see two_view_pipeline_champions.md).
    fuse_mode = "overhead_ray"
    grasp = grasp_from_views(
        wrist_bgr,
        side_bgr,
        square,
        occupancy={},
        T_board_arm=T_board_wrist,
        T_board_side=T_board_side,
        K_arm=K_wrist,
        K_side=K_side,
        square_mm=square_mm,
        overhead_mask=overhead_mask,
        side_mask=side_mask,
        fuse_mode=fuse_mode,
    )
    x_mm = grasp.get("x_mm")
    y_mm = grasp.get("y_mm")
    z_mm = grasp.get("z_mm")
    if x_mm is None or y_mm is None:
        return {
            "ok": False,
            "fail": "grasp_from_views returned no xy",
            "fuse_mode": fuse_mode,
            "grasp": {k: v for k, v in grasp.items() if k != "trace"},
        }
    board = np.array(
        [float(x_mm), float(y_mm), float(z_mm) if z_mm is not None else 0.0],
        dtype=np.float64,
    )
    return {
        "ok": True,
        "fail": None,
        "board_mm": board,
        "z_mm": None if z_mm is None else float(z_mm),
        "side_used": bool(grasp.get("side_used")),
        "yaw_deg": grasp.get("yaw_deg"),
        "shadow_pixels_dropped": int(grasp.get("shadow_pixels_dropped") or 0),
        "fuse_mode": grasp.get("fuse_mode") or fuse_mode,
        "wrist_px": _mask_centroid_px(overhead_mask),
        "side_px": _mask_centroid_px(side_mask),
        "grasp": {k: v for k, v in grasp.items() if k != "trace"},
    }
