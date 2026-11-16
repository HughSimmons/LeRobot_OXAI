#!/usr/bin/env python3
"""SAM 2.1 Tiny masks prompted by a DINO box or interactive points.

Laptop-feasible single-image path. Loads facebook/sam2.1-hiera-tiny once
per process and prefers MPS, then CUDA, then CPU.

Near-full multimask and box-fill rectangles are known failure modes when the
prompt box covers most of the crop or SAM paints the box itself. Selection
rejects crop frac >= MAX_CROP_FRAC and high rectangularity rather than
blindly taking max IoU.

``mask_from_points`` supports human FG/BG clicks (optional loose box) for
gold-candidate collection before wiring into click_to_3d.
"""

from __future__ import annotations

import cv2
import numpy as np

from board_geometry import largest_component_mask

_MODEL = None
_PROCESSOR = None
_DEVICE: str | None = None
_MODEL_ID = "facebook/sam2.1-hiera-tiny"

# Extreme invert / surroundings flood (im5/c8 ≈ 0.99). Legitimate large
# silhouettes like im7/c8 ≈ 0.94 stay under this.
MAX_CROP_FRAC = 0.95
MIN_MASK_PIXELS = 80
# Axis-aligned box-fill / prompt-box paint (im4/c4 bad cand ≈ 0.89).
RECT_FILL_MAX = 0.85
BOX_IOU_MAX = 0.85


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device() -> str:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = pick_device()
    return _DEVICE


def load_sam(model_id: str = _MODEL_ID):
    """Lazy-load SAM 2.1 Tiny processor and model."""
    global _MODEL, _PROCESSOR
    if _MODEL is not None and _PROCESSOR is not None:
        return _PROCESSOR, _MODEL, get_device()

    from transformers import Sam2Model, Sam2Processor

    device = get_device()
    try:
        processor = Sam2Processor.from_pretrained(model_id)
        model = Sam2Model.from_pretrained(model_id)
    except Exception:
        # Prefer the on-disk cache when Hub lookups flake (DNS / proxy / chat_template).
        processor = Sam2Processor.from_pretrained(model_id, local_files_only=True)
        model = Sam2Model.from_pretrained(model_id, local_files_only=True)
    model.to(device)
    model.eval()
    _PROCESSOR = processor
    _MODEL = model
    return processor, model, device


def _mask_rect_fill(binary: np.ndarray) -> float:
    """Fraction of the mask AABB that is filled (1.0 ≈ solid rectangle)."""
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return 0.0
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    area = bw * bh
    if area <= 0:
        return 0.0
    return float(np.count_nonzero(binary)) / float(area)


def _mask_box_iou(binary: np.ndarray, box_xyxy: tuple[float, float, float, float]) -> float:
    """IoU between a binary mask and the filled prompt box."""
    h, w = binary.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    ix1 = max(0, int(np.floor(x1)))
    iy1 = max(0, int(np.floor(y1)))
    ix2 = min(w, int(np.ceil(x2)))
    iy2 = min(h, int(np.ceil(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    box = np.zeros((h, w), dtype=np.uint8)
    box[iy1:iy2, ix1:ix2] = 1
    m = (binary > 0).astype(np.uint8)
    inter = int(np.count_nonzero(m & box))
    union = int(np.count_nonzero(m | box))
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _is_box_like(rect_fill: float, box_iou: float) -> bool:
    return bool(rect_fill >= RECT_FILL_MAX or box_iou >= BOX_IOU_MAX)


def _empty_result(
    h: int,
    w: int,
    *,
    fail: str,
    device: str,
    model_id: str,
    box_xyxy: list[float],
    scores: list[float] | None = None,
    fracs: list[float] | None = None,
    rect_fills: list[float] | None = None,
    box_ious: list[float] | None = None,
    gated: bool = True,
) -> dict:
    return {
        "mask": np.zeros((h, w), dtype=np.uint8),
        "score": 0.0,
        "device": device,
        "model_id": model_id,
        "box_xyxy": box_xyxy,
        "pixels": 0,
        "ok": False,
        "fail": fail,
        "candidate_index": None,
        "gated": gated,
        "candidate_scores": scores or [],
        "candidate_fracs": fracs or [],
        "candidate_rect_fills": rect_fills or [],
        "candidate_box_ious": box_ious or [],
    }


def mask_from_box(
    image_bgr: np.ndarray,
    box_xyxy: list[float] | tuple[float, float, float, float],
    model_id: str = _MODEL_ID,
) -> dict:
    """Segment one object with a box + centre-point prompt.

    Among SAM multimask candidates, reject near-full-crop floods and
    rectangular box-fill shapes, then take the highest IoU survivor.
    Hard-fails with ok=False rather than returning argmax IoU floods.
    """
    import torch
    from PIL import Image

    processor, model, device = load_sam(model_id)
    h, w = image_bgr.shape[:2]
    crop_area = float(h * w)
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1 = max(0.0, min(x1, w - 1.0))
    y1 = max(0.0, min(y1, h - 1.0))
    x2 = max(x1 + 1.0, min(x2, float(w)))
    y2 = max(y1 + 1.0, min(y2, float(h)))
    box = [x1, y1, x2, y2]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    # Points: [image][object][point][xy]; labels: [image][object][point].
    inputs = processor(
        images=image,
        input_boxes=[[[x1, y1, x2, y2]]],
        input_points=[[[[cx, cy]]]],
        input_labels=[[[1]]],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # transformers Sam2Processor.post_process_masks(masks, original_sizes, ...)
    # pred_masks: (batch, objects, candidates, H, W)
    processed = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
    )
    first = processed[0]
    if hasattr(first, "detach"):
        first = first.detach().cpu().numpy()
    first = np.asarray(first)
    # Expect (objects, candidates, H, W) or (candidates, H, W).
    while first.ndim > 3 and first.shape[0] == 1:
        first = first[0]
    if first.ndim == 4:
        first = first[0]
    if first.ndim == 2:
        candidates = first[None, ...]
    elif first.ndim == 3:
        candidates = first
    else:
        raise RuntimeError(f"Unexpected SAM mask shape {first.shape}")

    iou = getattr(outputs, "iou_scores", None)
    if iou is not None:
        scores = np.asarray(iou.detach().cpu().numpy()).reshape(-1)
        scores = scores[: len(candidates)]
        if len(scores) < len(candidates):
            scores = np.pad(scores, (0, len(candidates) - len(scores)), constant_values=0.0)
    else:
        scores = np.ones(len(candidates), dtype=np.float64)

    binaries: list[np.ndarray] = []
    fracs: list[float] = []
    rect_fills: list[float] = []
    box_ious: list[float] = []
    pixel_counts: list[int] = []
    for cand in candidates:
        binary = (cand > 0).astype(np.uint8)
        if binary.shape[0] != h or binary.shape[1] != w:
            binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
            binary = (binary > 0).astype(np.uint8)
        px = int(np.count_nonzero(binary))
        binaries.append(binary)
        pixel_counts.append(px)
        fracs.append(float(px) / crop_area if crop_area > 0 else 0.0)
        rect_fills.append(_mask_rect_fill(binary))
        box_ious.append(_mask_box_iou(binary, (x1, y1, x2, y2)))

    score_list = [float(s) for s in scores]
    meta_lists = dict(
        scores=score_list,
        fracs=fracs,
        rect_fills=rect_fills,
        box_ious=box_ious,
    )

    flood_ok = [
        i
        for i, (px, frac) in enumerate(zip(pixel_counts, fracs))
        if px >= MIN_MASK_PIXELS and frac < MAX_CROP_FRAC
    ]
    if not flood_ok:
        fail = "sam_empty" if all(px < MIN_MASK_PIXELS for px in pixel_counts) else "sam_flood"
        return _empty_result(h, w, fail=fail, device=device, model_id=model_id, box_xyxy=box, gated=True, **meta_lists)

    shape_ok = [
        i for i in flood_ok if not _is_box_like(rect_fills[i], box_ious[i])
    ]
    if shape_ok:
        best_i = int(max(shape_ok, key=lambda i: scores[i]))
        gated = True
    elif len(flood_ok) > 1:
        # Prefer the least rectangular flood survivor; still refuse if it is box-like.
        best_i = int(min(flood_ok, key=lambda i: rect_fills[i]))
        if _is_box_like(rect_fills[best_i], box_ious[best_i]):
            return _empty_result(
                h, w, fail="sam_rect", device=device, model_id=model_id, box_xyxy=box, gated=True, **meta_lists
            )
        gated = True
    else:
        only = flood_ok[0]
        if _is_box_like(rect_fills[only], box_ious[only]):
            return _empty_result(
                h, w, fail="sam_rect", device=device, model_id=model_id, box_xyxy=box, gated=True, **meta_lists
            )
        best_i = only
        gated = True

    mask = binaries[best_i] * 255
    return {
        "mask": mask,
        "score": float(scores[best_i]),
        "device": device,
        "model_id": model_id,
        "box_xyxy": box,
        "pixels": int(pixel_counts[best_i]),
        "ok": True,
        "fail": None,
        "candidate_index": int(best_i),
        "gated": gated,
        "candidate_scores": score_list,
        "candidate_fracs": fracs,
        "candidate_rect_fills": rect_fills,
        "candidate_box_ious": box_ious,
    }


def _parse_candidates(processed0, h: int, w: int, outputs) -> tuple[list[np.ndarray], np.ndarray]:
    """Normalize SAM post_process_masks output to per-candidate binaries + scores."""
    first = processed0
    if hasattr(first, "detach"):
        first = first.detach().cpu().numpy()
    first = np.asarray(first)
    while first.ndim > 3 and first.shape[0] == 1:
        first = first[0]
    if first.ndim == 4:
        first = first[0]
    if first.ndim == 2:
        candidates = first[None, ...]
    elif first.ndim == 3:
        candidates = first
    else:
        raise RuntimeError(f"Unexpected SAM mask shape {first.shape}")

    iou = getattr(outputs, "iou_scores", None)
    if iou is not None:
        scores = np.asarray(iou.detach().cpu().numpy()).reshape(-1)
        scores = scores[: len(candidates)]
        if len(scores) < len(candidates):
            scores = np.pad(scores, (0, len(candidates) - len(scores)), constant_values=0.0)
    else:
        scores = np.ones(len(candidates), dtype=np.float64)

    binaries: list[np.ndarray] = []
    for cand in candidates:
        binary = (cand > 0).astype(np.uint8)
        if binary.shape[0] != h or binary.shape[1] != w:
            binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
            binary = (binary > 0).astype(np.uint8)
        binaries.append(binary)
    return binaries, scores


def mask_from_points(
    image_bgr: np.ndarray,
    points_xy: list[tuple[float, float]] | list[list[float]],
    labels: list[int],
    box_xyxy: list[float] | tuple[float, float, float, float] | None = None,
    model_id: str = _MODEL_ID,
) -> dict:
    """Segment one object from FG/BG point prompts (optional loose box).

    ``labels``: 1 = foreground, 0 = background. Requires at least one point.
    Keeps the largest connected component. Rejects empty / near-full floods;
    when a box is provided, also rejects box-fill rectangles.
    """
    import torch
    from PIL import Image

    processor, model, device = load_sam(model_id)
    h, w = image_bgr.shape[:2]
    crop_area = float(h * w)

    if len(points_xy) == 0:
        return _empty_result(
            h,
            w,
            fail="no_points",
            device=device,
            model_id=model_id,
            box_xyxy=list(box_xyxy) if box_xyxy is not None else [],
            gated=False,
        )
    if len(labels) != len(points_xy):
        raise ValueError("points_xy and labels must have the same length")

    pts: list[list[float]] = []
    labs: list[int] = []
    for (x, y), lab in zip(points_xy, labels):
        px = float(max(0.0, min(float(x), w - 1.0)))
        py = float(max(0.0, min(float(y), h - 1.0)))
        pts.append([px, py])
        labs.append(1 if int(lab) else 0)

    box: list[float] | None = None
    if box_xyxy is not None:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        x1 = max(0.0, min(x1, w - 1.0))
        y1 = max(0.0, min(y1, h - 1.0))
        x2 = max(x1 + 1.0, min(x2, float(w)))
        y2 = max(y1 + 1.0, min(y2, float(h)))
        box = [x1, y1, x2, y2]

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    proc_kwargs: dict = {
        "images": image,
        "input_points": [[pts]],
        "input_labels": [[labs]],
        "return_tensors": "pt",
    }
    if box is not None:
        proc_kwargs["input_boxes"] = [[box]]
    inputs = processor(**proc_kwargs)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    processed = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
    )
    binaries, scores = _parse_candidates(processed[0], h, w, outputs)

    fracs: list[float] = []
    rect_fills: list[float] = []
    box_ious: list[float] = []
    pixel_counts: list[int] = []
    for binary in binaries:
        px = int(np.count_nonzero(binary))
        pixel_counts.append(px)
        fracs.append(float(px) / crop_area if crop_area > 0 else 0.0)
        rect_fills.append(_mask_rect_fill(binary))
        if box is not None:
            box_ious.append(_mask_box_iou(binary, (box[0], box[1], box[2], box[3])))
        else:
            box_ious.append(0.0)

    score_list = [float(s) for s in scores]
    meta_lists = dict(
        scores=score_list,
        fracs=fracs,
        rect_fills=rect_fills,
        box_ious=box_ious,
    )
    box_out = box if box is not None else []

    flood_ok = [
        i
        for i, (px, frac) in enumerate(zip(pixel_counts, fracs))
        if px >= MIN_MASK_PIXELS and frac < MAX_CROP_FRAC
    ]
    if not flood_ok:
        fail = "sam_empty" if all(px < MIN_MASK_PIXELS for px in pixel_counts) else "sam_flood"
        return _empty_result(
            h, w, fail=fail, device=device, model_id=model_id, box_xyxy=box_out, gated=True, **meta_lists
        )

    if box is not None:
        shape_ok = [i for i in flood_ok if not _is_box_like(rect_fills[i], box_ious[i])]
        if shape_ok:
            best_i = int(max(shape_ok, key=lambda i: scores[i]))
        elif len(flood_ok) > 1:
            best_i = int(min(flood_ok, key=lambda i: rect_fills[i]))
            if _is_box_like(rect_fills[best_i], box_ious[best_i]):
                return _empty_result(
                    h, w, fail="sam_rect", device=device, model_id=model_id, box_xyxy=box_out, gated=True, **meta_lists
                )
        else:
            only = flood_ok[0]
            if _is_box_like(rect_fills[only], box_ious[only]):
                return _empty_result(
                    h, w, fail="sam_rect", device=device, model_id=model_id, box_xyxy=box_out, gated=True, **meta_lists
                )
            best_i = only
    else:
        best_i = int(max(flood_ok, key=lambda i: scores[i]))

    mask = largest_component_mask(binaries[best_i] * 255)
    pixels = int(np.count_nonzero(mask))
    ok = pixels >= MIN_MASK_PIXELS and (pixels / crop_area if crop_area > 0 else 0.0) < MAX_CROP_FRAC
    return {
        "mask": mask if ok else np.zeros((h, w), dtype=np.uint8),
        "score": float(scores[best_i]),
        "device": device,
        "model_id": model_id,
        "box_xyxy": box_out,
        "points_xy": pts,
        "labels": labs,
        "pixels": pixels if ok else 0,
        "ok": bool(ok),
        "fail": None if ok else ("sam_empty" if pixels < MIN_MASK_PIXELS else "sam_flood"),
        "candidate_index": int(best_i),
        "gated": True,
        "candidate_scores": score_list,
        "candidate_fracs": fracs,
        "candidate_rect_fills": rect_fills,
        "candidate_box_ious": box_ious,
    }


def erode_core(mask: np.ndarray, radius_px: int = 3, min_pixels: int = 80) -> tuple[np.ndarray, bool]:
    """Erode a binary mask. Returns (core, used_erosion)."""
    binary = (mask > 0).astype(np.uint8) * 255
    if radius_px <= 0:
        return binary, False
    k = 2 * int(radius_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(binary, kernel, iterations=1)
    if int(np.count_nonzero(eroded)) < int(min_pixels):
        return binary, False
    return eroded, True


def sam_core_pos_neg(
    sam_mask: np.ndarray,
    erode_radius_px: int = 3,
    unlabelled_pad_px: int = 12,
    min_core_pixels: int = 80,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Three-zone samples: eroded core positive, padded ring unlabelled, rest negative."""
    core, used = erode_core(sam_mask, radius_px=erode_radius_px, min_pixels=min_core_pixels)
    pos = core > 0
    pad = max(int(unlabelled_pad_px), 0)
    if pad > 0:
        k = 2 * pad + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(core, kernel, iterations=1) > 0
    else:
        dilated = pos.copy()
    unlabelled = dilated & ~pos
    neg = ~dilated
    meta = {
        "eroded": used,
        "erode_radius_px": int(erode_radius_px),
        "unlabelled_pad_px": int(pad),
        "pos_pixels": int(np.count_nonzero(pos)),
        "neg_pixels": int(np.count_nonzero(neg)),
        "unlabelled_pixels": int(np.count_nonzero(unlabelled)),
    }
    return pos, neg, meta
