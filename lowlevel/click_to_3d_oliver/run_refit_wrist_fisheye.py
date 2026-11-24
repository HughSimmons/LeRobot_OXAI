#!/usr/bin/env python3
"""Refit the wrist fisheye from Oliver's labeled d4 multi-view set.

Uses OpenCV ``fisheye.calibrate`` on the expanded inner corners in
``corners.json`` (shot_001 dropped). Writes
``output/camera_mount/wrist_fisheye.yaml`` and a small smoke report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
LIB = ROOT / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib_intrinsics import (  # noqa: E402
    calibrate_fisheye,
    load_fisheye_calibration,
    save_fisheye_calibration,
    undistort_fisheye,
)
from click3d_shared import (  # noqa: E402
    MOUNT_DIR,
    SQUARE_MM,
    T_base_camera_mm,
    load_mount,
    resolve_core_assets,
)

DEFAULT_PHOTOS = (
    ROOT.parent / "sample images" / "wrist_photos_20261103_225124"
)
DROP_SHOTS = {"shot_001.png"}
D4_KEYS = ("3,3", "4,3", "4,4", "3,4")


def _load_views(photos_dir: Path, square_mm: float) -> tuple[list[dict], tuple[int, int]]:
    corners_path = photos_dir / "corners.json"
    labels_path = photos_dir / "labels.json"
    data = json.loads(corners_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.is_file() else {}
    joints_by_image = {
        shot["image"]: shot.get("joints_reading_deg")
        for shot in labels.get("shots", [])
    }
    kept_maps = {
        name: corner_map
        for name, corner_map in data["corners_px"].items()
        if name not in DROP_SHOTS
    }
    # Same corner set in every view — OpenCV fisheye.calibrate (and a stable
    # dense fit) need consistent observations; shot_001 already dropped.
    common_keys = sorted(
        set.intersection(*(set(corner_map) for corner_map in kept_maps.values())),
        key=lambda key: tuple(int(part) for part in key.split(",")),
    )
    if len(common_keys) < 4:
        raise RuntimeError(f"Need >=4 shared corner keys, got {len(common_keys)}")
    views: list[dict] = []
    image_size: tuple[int, int] | None = None
    for image_name in sorted(kept_maps):
        corner_map = kept_maps[image_name]
        path = photos_dir / image_name
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        if image_size is None:
            image_size = (image.shape[1], image.shape[0])
        object_points = []
        image_points = []
        for key in common_keys:
            file_i, rank_i = (int(part) for part in key.split(","))
            u, v = corner_map[key]
            object_points.append([file_i * square_mm, rank_i * square_mm, 0.0])
            image_points.append([float(u), float(v)])
        views.append(
            {
                "file": image_name,
                "image": image,
                "object_points": np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3),
                "image_points": np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2),
                "joints_deg": joints_by_image.get(image_name),
                "n_corners": len(common_keys),
                "corner_keys": common_keys,
            }
        )
    if image_size is None or len(views) < 6:
        raise RuntimeError(f"Need >=6 views after drops, got {len(views)}")
    return views, image_size


def _d4_object_points(square_mm: float) -> np.ndarray:
    points = []
    for key in D4_KEYS:
        file_i, rank_i = (int(part) for part in key.split(","))
        points.append([file_i * square_mm, rank_i * square_mm, 0.0])
    return np.asarray(points, dtype=np.float64).reshape(-1, 1, 3)


def _pnp_camera_distance_mm(
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    square_mm: float,
) -> float | None:
    """Camera-to-board origin distance from the four d4 corners (pinhole / undistorted K)."""
    obj = _d4_object_points(square_mm).reshape(-1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(
        obj.astype(np.float32),
        img.astype(np.float32),
        np.asarray(camera_matrix, dtype=np.float64),
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    # Camera origin in board frame = -R^T t
    rotation, _ = cv2.Rodrigues(rvec)
    origin = -rotation.T @ np.asarray(tvec, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(origin))


def _fisheye_pnp_distance_mm(
    image_points_raw: np.ndarray,
    calibration: dict,
    square_mm: float,
) -> float | None:
    """PnP distance in the raw fisheye model (before undistort)."""
    obj = _d4_object_points(square_mm)
    img = np.asarray(image_points_raw, dtype=np.float64).reshape(-1, 1, 2)
    try:
        ok, rvec, tvec = cv2.fisheye.solvePnP(
            obj,
            img,
            np.asarray(calibration["K"], dtype=np.float64),
            np.asarray(calibration["D"], dtype=np.float64).reshape(4, 1),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    origin = -rotation.T @ np.asarray(tvec, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(origin))


def run(photos_dir: Path, out_dir: Path, square_mm: float) -> dict:
    views, image_size = _load_views(photos_dir, square_mm)
    calibration = calibrate_fisheye(
        [view["object_points"] for view in views],
        [view["image_points"] for view in views],
        image_size,
        square_mm,
        target_reprojection_px=1.0,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out_dir / "wrist_fisheye.yaml"
    save_fisheye_calibration(yaml_path, calibration)
    loaded = load_fisheye_calibration(yaml_path)

    # Compare old bundle vs new on one mid view with d4 corners.
    # Prefer shot_007 (far / whole board); fall back to first kept view.
    sample = next((view for view in views if view["file"] == "shot_007.png"), views[0])
    d4_raw = []
    corner_map = json.loads((photos_dir / "corners.json").read_text(encoding="utf-8"))[
        "corners_px"
    ][sample["file"]]
    for key in D4_KEYS:
        d4_raw.append(corner_map[key])
    d4_raw_arr = np.asarray(d4_raw, dtype=np.float64)

    status_path = MOUNT_DIR / "wrist_fisheye_status_quo.yaml"
    compare = (
        load_fisheye_calibration(status_path)
        if status_path.is_file()
        else loaded
    )
    old_dist = _fisheye_pnp_distance_mm(d4_raw_arr, compare, square_mm)
    new_dist = _fisheye_pnp_distance_mm(d4_raw_arr, loaded, square_mm)

    fk_note = None
    if sample.get("joints_deg"):
        try:
            report, assets = resolve_core_assets()
            mount = load_mount(assets["urdf"], report)
            T_base_cam = T_base_camera_mm(list(sample["joints_deg"]), mount)
            fk_note = {
                "mount_kind": mount.kind,
                "camera_in_base_norm_mm": round(float(np.linalg.norm(T_base_cam[:3, 3])), 1),
            }
        except Exception as exc:  # noqa: BLE001 — smoke only
            fk_note = f"unavailable: {exc}"

    undist_dir = out_dir / "undistort_preview"
    undist_dir.mkdir(parents=True, exist_ok=True)
    for view in views:
        cv2.imwrite(
            str(undist_dir / view["file"]),
            undistort_fisheye(view["image"], loaded, balance=0.0),
        )

    dropped = calibration.get("dropped_index")
    dropped_file = views[dropped]["file"] if dropped is not None else None
    kept = [view for index, view in enumerate(views) if index != dropped]
    per_view = []
    for view, err_px, dist_mm in zip(
        kept, calibration["view_errors_px"], calibration["camera_distance_mm"]
    ):
        per_view.append(
            {
                "file": view["file"],
                "n_corners": view["n_corners"],
                "reprojection_px": round(float(err_px), 3),
                "camera_distance_mm": round(float(dist_mm), 1),
            }
        )

    report = {
        "calibrated": True,
        "yaml": str(yaml_path),
        "square_mm": square_mm,
        "square_mm_source": "click3d_shared.SQUARE_MM (41.25 mm, board_coordinates.DEFAULT_SQUARE_SIZE)",
        "dropped_shots": sorted(DROP_SHOTS),
        "dropped_outlier_view": dropped_file,
        "views_used": per_view,
        "reprojection_px": round(float(calibration["reprojection_px"]), 4),
        "meets_target": bool(calibration["meets_target"]),
        "replaced_folded_fit": bool(calibration["replaced_folded_fit"]),
        "K": np.round(calibration["K"], 4).tolist(),
        "D": np.round(np.asarray(calibration["D"]).reshape(-1), 6).tolist(),
        "smoke": {
            "sample": sample["file"],
            "pnp_tvec_norm_mm_old_bundle": None if old_dist is None else round(old_dist, 1),
            "pnp_tvec_norm_mm_new": None if new_dist is None else round(new_dist, 1),
            "fk": fk_note,
            "note": (
                "PnP |t| is camera-to-d4 distance from the four marked corners. "
                "Old bundle f=1400 inflated that ~2.2×; new f≈540 matches pixel span. "
                "These Nov-3 poses are farther than Oliver's original ~155 mm close view."
            ),
        },
    }
    (out_dir / "wrist_fisheye_refit_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", type=Path, default=DEFAULT_PHOTOS)
    parser.add_argument("--out", type=Path, default=MOUNT_DIR)
    parser.add_argument("--square-mm", type=float, default=SQUARE_MM)
    args = parser.parse_args()
    report = run(args.photos, args.out, args.square_mm)
    print(
        json.dumps(
            {
                "yaml": report["yaml"],
                "reprojection_px": report["reprojection_px"],
                "meets_target": report["meets_target"],
                "replaced_folded_fit": report["replaced_folded_fit"],
                "K_focal": report["K"][0][0],
                "D": report["D"],
                "smoke": report["smoke"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
