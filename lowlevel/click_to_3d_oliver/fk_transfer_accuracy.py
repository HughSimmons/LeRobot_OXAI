#!/usr/bin/env python3
"""How well does one c calibration plus forward kinematics pose the wrist camera after a move?

Offline test on saved wrist photos with joint readings and labelled board corners. For every
ordered pair of photos (anchor a = "the photo you pressed c on", target b = "a photo after the
arm moved"):

    T_base_board  = T_base_camera(joints_a) . inv(T_board_wrist_a)
    T_board_wrist_b_pred = inv(T_base_board) . T_base_camera(joints_b)

with T_base_camera(j) = FK(j, gripper_frame_link) . T_gripper_camera (output/camera_mount/
wrist_camera_mount.yaml). The prediction is scored against photo b's own labelled corners:

- px: board corners projected with the predicted pose vs the observed (undistorted) corners
- board mm: observed corner pixels cast onto the board (z = 0) with the predicted pose vs the
  true corner positions, i.e. where a click on that corner would land
- label margin: worst corner error / half the local corner spacing. Below 1, labelling the
  detected corners by nearest FK-predicted corner works, so the refinement needs no clicks.

Anchor poses come either from all labelled corners (a careful c) or from d4's four corners
only (what c does now). "+ refine" runs refine_with_local_pnp (coord_finder/calib_board_pose)
on the four corners of one square near the image centre of b, then scores the other corners.

Datasets:
- 11 Nov mount run 20261111_193315 (8 parks, snapped corners). The mount was fitted to these
  photos, so its numbers are in-sample and optimistic.
- 3 Nov lens photos wrist_photos_20261103_225124 (shot_001 left out: board not flat there).
  Not used for the mount fit, so this is the out-of-sample check.

Run with the IKsim_mj env from lowlevel/click_to_3d_oliver/:
    ~/miniforge3/envs/IKsim_mj/bin/python fk_transfer_accuracy.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from click3d_shared import MOUNT_DIR, SQUARE_MM, fk_link_mm, load_fisheye_calibration
from lib.calib_board_pose import board_camera_from_pnp, invert, refine_with_local_pnp

ROOT = Path(__file__).resolve().parent
LOWLEVEL = ROOT.parent
D4_KEYS = ("3,3", "4,3", "4,4", "3,4")
DATASETS = {
    "11nov_run193315": dict(
        kind="mount_run",
        samples=ROOT / "output/camera_mount/runs/20261111_193315/samples.json",
        corners=ROOT / "output/camera_mount/runs/20261111_193315/grown/corners_snapped.json",
        note="in-sample: the mount was fitted to these photos",
    ),
    "3nov_lens_photos": dict(
        kind="lens_photos",
        capture=LOWLEVEL / "lens_calibration/output/wrist_photos_20261103_225124/capture.json",
        corners=LOWLEVEL / "lens_calibration/output/wrist_photos_20261103_225124/corners.json",
        skip=("shot_001.png",),
        note="out-of-sample: not used for the mount fit",
    ),
}


def load_views(spec: dict) -> list[dict]:
    if spec["kind"] == "mount_run":
        samples = json.loads(Path(spec["samples"]).read_text())["samples"]
        corners = json.loads(Path(spec["corners"]).read_text())["corners_raw_px"]
        return [
            dict(name=s["image"], joints=s["joint_reading_deg"], corners=corners[s["image"]])
            for s in samples
            if s["image"] in corners
        ]
    shots = json.loads(Path(spec["capture"]).read_text())["shots"]
    corners = json.loads(Path(spec["corners"]).read_text())["corners_px"]
    return [
        dict(name=s["image"], joints=s["joints_reading_deg"], corners=corners[s["image"]])
        for s in shots
        if s["image"] in corners and s["image"] not in spec.get("skip", ())
    ]


def board_point(key: str) -> np.ndarray:
    i, j = (int(v) for v in key.split(","))
    return np.array([i * SQUARE_MM, j * SQUARE_MM, 0.0])


def prepare(view: dict, lens: dict, inlier_px: float) -> dict:
    """Undistort corners, drop ones a board homography can't explain, solve the photo's own pose."""
    keys = sorted(view["corners"])
    raw = np.array([view["corners"][k] for k in keys], dtype=np.float64).reshape(-1, 1, 2)
    und = cv2.fisheye.undistortPoints(raw, lens["K"], lens["D"], P=lens["K_undistorted"]).reshape(-1, 2)
    obj = np.array([board_point(k) for k in keys])
    H, inl = cv2.findHomography(obj[:, :2], und, cv2.RANSAC, inlier_px)
    keep = inl.ravel().astype(bool)
    view = dict(view, keys=[k for k, m in zip(keys, keep) if m], px=und[keep], obj=obj[keep],
                dropped=int((~keep).sum()))
    view["T_board_cam_all"] = solve_pose(view["obj"], view["px"], lens["K_undistorted"])
    view["self_rms_px"] = rms(project(view["obj"], view["T_board_cam_all"], lens["K_undistorted"]) - view["px"])
    idx = [view["keys"].index(k) for k in D4_KEYS if k in view["keys"]]
    view["T_board_cam_d4"] = (
        solve_pose(view["obj"][idx], view["px"][idx], lens["K_undistorted"]) if len(idx) == 4 else None
    )
    return view


def solve_pose(obj: np.ndarray, px: np.ndarray, K: np.ndarray) -> np.ndarray:
    ok, rvec, tvec = cv2.solvePnP(obj, px, K, None, flags=cv2.SOLVEPNP_IPPE)
    ok, rvec, tvec = cv2.solvePnP(obj, px, K, None, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
    return board_camera_from_pnp(rvec, tvec)


def project(obj: np.ndarray, T_board_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    T_cam_board = invert(T_board_cam)
    pts = (T_cam_board[:3, :3] @ obj.T + T_cam_board[:3, 3:4]).T
    uv = (K @ pts.T).T
    return uv[:, :2] / uv[:, 2:3]


def cast_to_board(px: np.ndarray, T_board_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    rays = np.linalg.solve(K, np.column_stack([px, np.ones(len(px))]).T).T
    dirs = (T_board_cam[:3, :3] @ rays.T).T
    origin = T_board_cam[:3, 3]
    t = -origin[2] / dirs[:, 2]
    return origin[None, :2] + t[:, None] * dirs[:, :2]


def rms(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def half_spacing_px(view: dict) -> np.ndarray:
    """Half the distance from each corner to its nearest labelled neighbour, in pixels."""
    d = np.linalg.norm(view["px"][:, None] - view["px"][None], axis=2)
    np.fill_diagonal(d, np.inf)
    return 0.5 * d.min(axis=1)


def centre_square(view: dict, K: np.ndarray) -> list[int] | None:
    """Indices of the four corners of the complete square closest to the image centre."""
    centre = K[:2, 2]
    index = {k: n for n, k in enumerate(view["keys"])}
    best, best_d = None, np.inf
    for key in view["keys"]:
        i, j = (int(v) for v in key.split(","))
        quad = [f"{i},{j}", f"{i + 1},{j}", f"{i + 1},{j + 1}", f"{i},{j + 1}"]
        if all(q in index for q in quad):
            idx = [index[q] for q in quad]
            d = float(np.linalg.norm(view["px"][idx].mean(axis=0) - centre))
            if d < best_d:
                best, best_d = idx, d
    return best


def score(pred: np.ndarray, view: dict, K: np.ndarray, mask: np.ndarray | None = None) -> dict:
    m = np.ones(len(view["px"]), bool) if mask is None else mask
    px_err = project(view["obj"][m], pred, K) - view["px"][m]
    mm_err = cast_to_board(view["px"][m], pred, K) - view["obj"][m, :2]
    px_norm = np.linalg.norm(px_err, axis=1)
    return dict(
        px_rms=rms(px_err),
        px_max=float(px_norm.max()),
        mm_mean=float(np.linalg.norm(mm_err, axis=1).mean()),
        mm_max=float(np.linalg.norm(mm_err, axis=1).max()),
        label_margin=float((px_norm / half_spacing_px(view)[m]).max()),
    )


def run(name: str, spec: dict, lens: dict, mount: np.ndarray, urdf: Path, inlier_px: float) -> dict:
    K = lens["K_undistorted"]
    views = [prepare(v, lens, inlier_px) for v in load_views(spec)]
    for v in views:
        v["T_base_cam"] = fk_link_mm(urdf, "gripper_frame_link", v["joints"]) @ mount
    print(f"\n=== {name}  ({spec['note']})")
    print("photo          corners  dropped  own-pose rms px")
    for v in views:
        print(f"  {v['name']:<14} {len(v['keys']):>5}  {v['dropped']:>5}     {v['self_rms_px']:.2f}")

    rows = []
    for a, b in itertools.permutations(views, 2):
        for anchor in ("all", "d4"):
            T_a = a[f"T_board_cam_{anchor}"]
            if T_a is None:
                continue
            T_base_board = a["T_base_cam"] @ invert(T_a)
            pred = invert(T_base_board) @ b["T_base_cam"]
            row = dict(anchor=anchor, a=a["name"], b=b["name"], fk=score(pred, b, K))
            quad = centre_square(b, K)
            if quad is not None:
                refined, accepted = refine_with_local_pnp(pred, b["px"][quad], b["obj"][quad], K)
                rest = np.ones(len(b["px"]), bool)
                rest[quad] = False
                row["refined"] = score(refined, b, K, rest)
                row["refine_accepted"] = bool(accepted)
            rows.append(row)

    summary = {}
    for anchor in ("all", "d4"):
        sel = [r for r in rows if r["anchor"] == anchor]
        for route in ("fk", "refined"):
            vals = [r[route] for r in sel if route in r]
            if not vals:
                continue
            s = {k: np.array([v[k] for v in vals]) for k in vals[0]}
            summary[f"{anchor}/{route}"] = {
                k: dict(median=float(np.median(x)), p90=float(np.percentile(x, 90)), max=float(x.max()))
                for k, x in s.items()
            }
            summary[f"{anchor}/{route}"]["pairs"] = len(vals)
        acc = [r.get("refine_accepted") for r in sel if "refine_accepted" in r]
        summary[f"{anchor}/refine_accepted"] = f"{sum(acc)}/{len(acc)}"

    labels = {"all/fk": "c on all corners, FK only", "d4/fk": "c on d4 only,   FK only",
              "all/refined": "c on all corners, + refine", "d4/refined": "c on d4 only,   + refine"}
    print(f"\n{len(views)} photos -> {len(views) * (len(views) - 1)} anchor/target pairs. median / p90 / max:")
    print(f"{'route':<28} {'px rms':>18} {'px max':>18} {'board mm mean':>18} {'board mm max':>18} {'label margin':>18}")
    for key, label in labels.items():
        if key not in summary:
            continue
        s = summary[key]
        cells = [f"{s[m]['median']:5.1f}/{s[m]['p90']:5.1f}/{s[m]['max']:5.1f}"
                 for m in ("px_rms", "px_max", "mm_mean", "mm_max", "label_margin")]
        print(f"{label:<28} " + " ".join(f"{c:>18}" for c in cells))
    for anchor in ("all", "d4"):
        print(f"refine accepted ({anchor} anchor): {summary[f'{anchor}/refine_accepted']}")

    worst = sorted((r for r in rows if r["anchor"] == "d4"), key=lambda r: -r["fk"]["mm_mean"])[:5]
    print("worst pairs (d4 anchor, FK only): " + "; ".join(
        f"{r['a'].split('.')[0]}->{r['b'].split('.')[0]} {r['fk']['mm_mean']:.1f} mm" for r in worst))
    return dict(note=spec["note"], summary=summary, rows=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mount", type=Path, default=MOUNT_DIR / "wrist_camera_mount.yaml")
    parser.add_argument("--fisheye", type=Path, default=MOUNT_DIR / "wrist_fisheye.yaml")
    parser.add_argument("--inlier-px", type=float, default=3.0,
                        help="homography inlier threshold for dropping mislabelled corners")
    parser.add_argument("--out", type=Path, default=ROOT / "output/fk_transfer_accuracy.json")
    args = parser.parse_args()

    lens = load_fisheye_calibration(args.fisheye)
    mount_data = yaml.safe_load(args.mount.read_text())
    mount = np.asarray(mount_data["T_gripper_camera_mm"], dtype=np.float64)
    urdf = Path(mount_data["source_urdf"])
    print(f"mount {args.mount.name} (runs {mount_data.get('runs')}), lens f={lens['K_undistorted'][0, 0]:.0f} px, "
          f"square {SQUARE_MM} mm, urdf {urdf.name}")
    results = {name: run(name, spec, lens, mount, urdf, args.inlier_px) for name, spec in DATASETS.items()}
    args.out.write_text(json.dumps(results, indent=1, default=float) + "\n")
    print(f"\nwrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
