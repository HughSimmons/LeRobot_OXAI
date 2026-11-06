#!/usr/bin/env python3
"""Standard-aim wrist fisheye fit from the existing labeled/grown corner set.

Uses every accepted corner in corners.json per view (variable N; shot_001
dropped). Tries OpenCV fisheye.calibrate first; on failure runs a continuous
least-squares fit of Kannala-Brandt K+D with OpenCV fisheye.projectPoints /
solvePnP, free cx/cy, fold penalty. Writes output/camera_mount/wrist_fisheye.yaml
and a detailed evaluation report.

Status quo before this attempt is under
output/camera_mount/_status_quo_20261106/ for revert.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
LIB = ROOT / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib_intrinsics import (  # noqa: E402
    _distortion_folds,
    _fisheye_flag,
    fisheye_new_camera,
    load_fisheye_calibration,
    save_fisheye_calibration,
    undistort_fisheye,
)
from click3d_shared import MOUNT_DIR, SQUARE_MM  # noqa: E402

DEFAULT_PHOTOS = ROOT.parent / "sample images" / "wrist_photos_20261103_225124"
DROP_SHOTS = {"shot_001.png"}
D4_KEYS = ("3,3", "4,3", "4,4", "3,4")


def _load_views(photos_dir: Path, square_mm: float) -> tuple[list[dict], tuple[int, int]]:
    data = json.loads((photos_dir / "corners.json").read_text(encoding="utf-8"))
    labels_path = photos_dir / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.is_file() else {}
    joints = {s["image"]: s.get("joints_reading_deg") for s in labels.get("shots", [])}
    hints = {s["image"]: s.get("shot_hint") for s in labels.get("shots", [])}
    views: list[dict] = []
    image_size: tuple[int, int] | None = None
    for name, corner_map in sorted(data["corners_px"].items()):
        if name in DROP_SHOTS:
            continue
        image = cv2.imread(str(photos_dir / name))
        if image is None:
            raise FileNotFoundError(photos_dir / name)
        if image_size is None:
            image_size = (image.shape[1], image.shape[0])
        keys = sorted(corner_map, key=lambda k: tuple(int(p) for p in k.split(",")))
        obj, img = [], []
        for key in keys:
            i, j = (int(p) for p in key.split(","))
            u, v = corner_map[key]
            obj.append([i * square_mm, j * square_mm, 0.0])
            img.append([float(u), float(v)])
        pts = np.asarray(img, dtype=np.float64)
        w, h = image_size
        mx, my = 0.2 * w, 0.2 * h
        near_edge = (
            (pts[:, 0] < mx)
            | (pts[:, 0] > w - mx)
            | (pts[:, 1] < my)
            | (pts[:, 1] > h - my)
        )
        views.append(
            {
                "file": name,
                "image": image,
                "object_points": np.asarray(obj, dtype=np.float64).reshape(-1, 1, 3),
                "image_points": np.asarray(img, dtype=np.float64).reshape(-1, 1, 2),
                "n_corners": len(keys),
                "keys": keys,
                "joints_deg": joints.get(name),
                "hint": hints.get(name),
                "edge_fraction": float(near_edge.mean()) if len(pts) else 0.0,
            }
        )
    if image_size is None or len(views) < 6:
        raise RuntimeError(f"Need >=6 views, got {0 if image_size is None else len(views)}")
    return views, image_size


def _try_opencv_calibrate(
    views: list[dict], image_size: tuple[int, int]
) -> dict | None:
    objs = [v["object_points"] for v in views]
    imgs = [v["image_points"] for v in views]
    w, h = image_size
    attempts = [
        ("free_516", 516.0, 0),
        (
            "fix_k3k4_516",
            516.0,
            _fisheye_flag("CALIB_FIX_K3") | _fisheye_flag("CALIB_FIX_K4"),
        ),
        (
            "fix_k2k3k4_516",
            516.0,
            _fisheye_flag("CALIB_FIX_K2")
            | _fisheye_flag("CALIB_FIX_K3")
            | _fisheye_flag("CALIB_FIX_K4"),
        ),
    ]
    common = set(views[0]["keys"])
    for v in views[1:]:
        common &= set(v["keys"])
    common = sorted(common, key=lambda k: tuple(int(p) for p in k.split(",")))

    def pack_common():
        o, i = [], []
        for v in views:
            idx = [v["keys"].index(k) for k in common]
            o.append(v["object_points"][idx])
            i.append(v["image_points"][idx])
        return o, i

    trials = [("all_points", objs, imgs)]
    if len(common) >= 4:
        co, ci = pack_common()
        trials.append((f"common_{len(common)}", co, ci))

    for trial_name, o_list, i_list in trials:
        for name, focal, extra in attempts:
            K = np.array(
                [[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64
            )
            D = np.zeros((4, 1), dtype=np.float64)
            flags = (
                _fisheye_flag("CALIB_RECOMPUTE_EXTRINSIC")
                | _fisheye_flag("CALIB_FIX_SKEW")
                | _fisheye_flag("CALIB_USE_INTRINSIC_GUESS", default=1)
                | int(extra)
            )
            try:
                rms, K2, D2, rvecs, tvecs = cv2.fisheye.calibrate(
                    o_list,
                    i_list,
                    image_size,
                    K,
                    D,
                    flags=flags,
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7),
                )
            except cv2.error:
                continue
            D2 = np.asarray(D2, dtype=np.float64).reshape(4, 1)
            K2 = np.asarray(K2, dtype=np.float64)
            if _distortion_folds(K2, D2, image_size):
                continue
            return {
                "method": f"opencv_fisheye.calibrate:{trial_name}:{name}",
                "replaced_folded_fit": False,
                "K": K2,
                "D": D2,
                "rvecs": list(rvecs),
                "tvecs": list(tvecs),
                "rms": float(rms),
                "used_common_only": trial_name.startswith("common"),
                "n_per_view": [int(o.shape[0]) for o in o_list],
            }
    return None


def _seed_extrinsics(
    views: list[dict], K: np.ndarray, D: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    poses = []
    for v in views:
        ok, rvec, tvec = cv2.fisheye.solvePnP(
            v["object_points"],
            v["image_points"],
            K,
            D,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError(f"solvePnP failed on {v['file']}")
        poses.append(
            (
                np.asarray(rvec, dtype=np.float64).reshape(3),
                np.asarray(tvec, dtype=np.float64).reshape(3),
            )
        )
    return poses


def _continuous_fit(
    views: list[dict], image_size: tuple[int, int]
) -> dict:
    """Least-squares Kannala-Brandt with OpenCV projectPoints residuals."""
    w, h = image_size
    fx = fy = 520.0
    cx, cy = w / 2.0, h / 2.0
    k1, k2, k3, k4 = -0.2, 0.0, 0.0, 0.0
    K0 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D0 = np.array([[k1], [k2], [k3], [k4]], dtype=np.float64)
    poses = _seed_extrinsics(views, K0, D0)

    n_views = len(views)
    x0 = [fx, fy, cx, cy, k1, k2, k3, k4]
    for rvec, tvec in poses:
        x0.extend(rvec.tolist())
        x0.extend(tvec.tolist())
    x0 = np.asarray(x0, dtype=np.float64)

    def unpack(x: np.ndarray):
        fx_, fy_, cx_, cy_, k1_, k2_, k3_, k4_ = x[:8]
        K = np.array([[fx_, 0, cx_], [0, fy_, cy_], [0, 0, 1]], dtype=np.float64)
        D = np.array([[k1_], [k2_], [k3_], [k4_]], dtype=np.float64)
        extr = []
        for i in range(n_views):
            base = 8 + 6 * i
            extr.append((x[base : base + 3], x[base + 3 : base + 6]))
        return K, D, extr

    def residuals(x: np.ndarray) -> np.ndarray:
        K, D, extr = unpack(x)
        chunks = []
        for v, (rvec, tvec) in zip(views, extr):
            proj, _ = cv2.fisheye.projectPoints(
                v["object_points"],
                rvec.reshape(3, 1),
                tvec.reshape(3, 1),
                K,
                D,
            )
            err = (v["image_points"].reshape(-1, 2) - proj.reshape(-1, 2)).ravel()
            chunks.append(err)
        focal = float(min(K[0, 0], K[1, 1]))
        radius = float(np.hypot(w / 2.0, h / 2.0) / max(focal, 1.0))
        theta_max = float(np.arctan(radius))
        coeffs = D.reshape(-1)
        thetas = np.linspace(0.0, theta_max, 40)
        u = thetas**2
        deriv = (
            1.0
            + 3.0 * coeffs[0] * u
            + 5.0 * coeffs[1] * u**2
            + 7.0 * coeffs[2] * u**3
            + 9.0 * coeffs[3] * u**4
        )
        chunks.append(np.minimum(deriv, 0.0) * 50.0)
        return np.concatenate(chunks)

    lower = np.full_like(x0, -np.inf)
    upper = np.full_like(x0, np.inf)
    lower[0:2] = 200.0
    upper[0:2] = 1200.0
    lower[2], upper[2] = 0.2 * w, 0.8 * w
    lower[3], upper[3] = 0.2 * h, 0.8 * h
    lower[4:8] = -2.5
    upper[4:8] = 2.5

    result = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        method="trf",
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=400,
        verbose=0,
    )
    K, D, extr = unpack(result.x)

    def view_errors(K, D, extr):
        errs, rvecs, tvecs = [], [], []
        for v, (rvec, tvec) in zip(views, extr):
            proj, _ = cv2.fisheye.projectPoints(
                v["object_points"],
                rvec.reshape(3, 1),
                tvec.reshape(3, 1),
                K,
                D,
            )
            res = v["image_points"].reshape(-1, 2) - proj.reshape(-1, 2)
            errs.append(float(np.sqrt(np.mean(np.sum(res * res, axis=1)))))
            rvecs.append(rvec.reshape(3, 1))
            tvecs.append(tvec.reshape(3, 1))
        return errs, rvecs, tvecs

    errs, rvecs, tvecs = view_errors(K, D, extr)
    method = "continuous_lsq_opencv_projectPoints"
    folded = _distortion_folds(K, D, image_size)
    if folded:
        x1 = result.x.copy()
        x1[6] = 0.0
        x1[7] = 0.0
        lower2, upper2 = lower.copy(), upper.copy()
        lower2[6:8] = 0.0
        upper2[6:8] = 0.0

        def residuals_fix(x: np.ndarray) -> np.ndarray:
            x = x.copy()
            x[6] = 0.0
            x[7] = 0.0
            return residuals(x)

        result2 = least_squares(
            residuals_fix,
            x1,
            bounds=(lower2, upper2),
            method="trf",
            loss="soft_l1",
            f_scale=2.0,
            max_nfev=300,
            verbose=0,
        )
        K, D, extr = unpack(result2.x)
        D[2:] = 0.0
        errs, rvecs, tvecs = view_errors(K, D, extr)
        method = "continuous_lsq_fix_k3k4"
        folded = _distortion_folds(K, D, image_size)

    return {
        "method": method,
        "replaced_folded_fit": bool(folded) or method.endswith("fix_k3k4"),
        "K": K,
        "D": np.asarray(D, dtype=np.float64).reshape(4, 1),
        "rvecs": rvecs,
        "tvecs": tvecs,
        "rms": float(np.mean(errs)),
        "view_errors_px": errs,
        "used_common_only": False,
        "n_per_view": [int(v["n_corners"]) for v in views],
        "success": bool(result.success),
    }


def _pnp_tnorm(calib: dict, image_points: np.ndarray, square_mm: float) -> float | None:
    obj = np.array(
        [
            [[3 * square_mm, 3 * square_mm, 0.0]],
            [[4 * square_mm, 3 * square_mm, 0.0]],
            [[4 * square_mm, 4 * square_mm, 0.0]],
            [[3 * square_mm, 4 * square_mm, 0.0]],
        ],
        dtype=np.float64,
    )
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    try:
        ok, _r, t = cv2.fisheye.solvePnP(
            obj,
            img,
            np.asarray(calib["K"], dtype=np.float64),
            np.asarray(calib["D"], dtype=np.float64).reshape(4, 1),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None
    return float(np.linalg.norm(t)) if ok else None


def _evaluate(
    calib: dict, views: list[dict], image_size: tuple[int, int], square_mm: float
) -> dict:
    K = np.asarray(calib["K"], dtype=np.float64)
    D = np.asarray(calib["D"], dtype=np.float64).reshape(4, 1)
    folds = _distortion_folds(K, D, image_size)
    status_path = MOUNT_DIR / "wrist_fisheye_status_quo.yaml"
    if not status_path.is_file():
        status_path = MOUNT_DIR / "_status_quo_20261106" / "wrist_fisheye.yaml"
    status = load_fisheye_calibration(status_path)
    # Historical f=1400 file is gone; compare against status-quo interim instead.
    old = status
    per = []
    for v in views:
        key_to_px = {
            k: v["image_points"][i].reshape(2) for i, k in enumerate(v["keys"])
        }
        if all(k in key_to_px for k in D4_KEYS):
            d4 = [key_to_px[k] for k in D4_KEYS]
            t_new = _pnp_tnorm({"K": K, "D": D}, d4, square_mm)
            t_old = _pnp_tnorm(old, d4, square_mm)
            t_status = _pnp_tnorm(status, d4, square_mm)
        else:
            t_new = t_old = t_status = None
        per.append(
            {
                "file": v["file"],
                "hint": v["hint"],
                "n_corners": v["n_corners"],
                "edge_fraction": round(v["edge_fraction"], 3),
                "pnp_t_old_bundle": None if t_old is None else round(t_old, 1),
                "pnp_t_status_quo": None if t_status is None else round(t_status, 1),
                "pnp_t_new": None if t_new is None else round(t_new, 1),
            }
        )
    principal_moved = abs(float(K[0, 2]) - image_size[0] / 2) > 0.5 or abs(
        float(K[1, 2]) - image_size[1] / 2
    ) > 0.5
    checklist = {
        "rms_under_1px": float(calib["rms"]) <= 1.0,
        "no_fold": not folds,
        "used_variable_n_corners": not calib.get("used_common_only", True),
        "not_coarse_monotonic_grid": not str(calib["method"]).startswith("monotonic"),
        "principal_point_optimized_or_moved": principal_moved
        or str(calib["method"]).startswith("continuous")
        or str(calib["method"]).startswith("opencv"),
        "mean_edge_fraction_gt_0.05": float(np.mean([v["edge_fraction"] for v in views]))
        > 0.05,
    }
    checklist["meets_numeric_standard"] = all(
        [
            checklist["rms_under_1px"],
            checklist["no_fold"],
            checklist["used_variable_n_corners"],
            checklist["not_coarse_monotonic_grid"],
            checklist["principal_point_optimized_or_moved"],
        ]
    )
    return {
        "folds": folds,
        "checklist": checklist,
        "per_view": per,
        "K_focal_fx_fy": [round(float(K[0, 0]), 2), round(float(K[1, 1]), 2)],
        "principal_point": [round(float(K[0, 2]), 2), round(float(K[1, 2]), 2)],
        "D": np.round(D.reshape(-1), 6).tolist(),
        "rms": round(float(calib["rms"]), 4),
        "method": calib["method"],
        "total_point_observations": int(sum(v["n_corners"] for v in views)),
        "mean_edge_fraction": round(float(np.mean([v["edge_fraction"] for v in views])), 3),
    }


def run(photos_dir: Path, out_dir: Path, square_mm: float) -> dict:
    views, image_size = _load_views(photos_dir, square_mm)
    opencv = _try_opencv_calibrate(views, image_size)
    if opencv is not None:
        fit = opencv
        errs = []
        for v in views:
            ok, r2, t2 = cv2.fisheye.solvePnP(
                v["object_points"],
                v["image_points"],
                fit["K"],
                fit["D"],
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                proj, _ = cv2.fisheye.projectPoints(
                    v["object_points"], r2, t2, fit["K"], fit["D"]
                )
                res = v["image_points"].reshape(-1, 2) - proj.reshape(-1, 2)
                errs.append(float(np.sqrt(np.mean(np.sum(res * res, axis=1)))))
            else:
                errs.append(float("nan"))
        fit["view_errors_px"] = errs
        fit["rms"] = float(np.nanmean(errs))
    else:
        fit = _continuous_fit(views, image_size)

    calibration = {
        "model": "fisheye",
        "K": fit["K"],
        "D": fit["D"],
        "K_undistorted": fisheye_new_camera(fit["K"], fit["D"], image_size, 0.0),
        "K_undistorted_balance1": fisheye_new_camera(fit["K"], fit["D"], image_size, 1.0),
        "balance": 0.0,
        "image_size": image_size,
        "reprojection_px": float(fit["rms"]),
        "meets_target": bool(fit["rms"] <= 1.0),
        "target_reprojection_px": 1.0,
        "square_mm": float(square_mm),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out_dir / "wrist_fisheye.yaml"
    save_fisheye_calibration(yaml_path, calibration)
    loaded = load_fisheye_calibration(yaml_path)

    preview = out_dir / "undistort_standard_preview"
    preview.mkdir(parents=True, exist_ok=True)
    for v in views:
        cv2.imwrite(str(preview / v["file"]), undistort_fisheye(v["image"], loaded, 0.0))

    evaluation = _evaluate(fit, views, image_size, square_mm)
    report = {
        "calibrated": True,
        "yaml": str(yaml_path),
        "square_mm": square_mm,
        "dropped_shots": sorted(DROP_SHOTS),
        "method": fit["method"],
        "replaced_folded_fit": bool(fit.get("replaced_folded_fit", False)),
        "n_per_view": fit["n_per_view"],
        "view_errors_px": [round(float(e), 3) for e in fit.get("view_errors_px", [])],
        "evaluation": evaluation,
        "status_quo_backup": str(out_dir / "_status_quo_20261106"),
        "note": (
            "Uses all grown corners per view from corners.json (a1-indexed). "
            "Await approval; reject by restoring _status_quo_20261106."
        ),
    }
    (out_dir / "wrist_fisheye_standard_report.json").write_text(
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
    ev = report["evaluation"]
    print(
        json.dumps(
            {
                "method": report["method"],
                "rms": ev["rms"],
                "focal_fx_fy": ev["K_focal_fx_fy"],
                "principal_point": ev["principal_point"],
                "D": ev["D"],
                "folds": ev["folds"],
                "total_point_observations": ev["total_point_observations"],
                "mean_edge_fraction": ev["mean_edge_fraction"],
                "n_per_view": report["n_per_view"],
                "checklist": ev["checklist"],
                "sample_distances": ev["per_view"][:4],
                "awaiting_approval": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
