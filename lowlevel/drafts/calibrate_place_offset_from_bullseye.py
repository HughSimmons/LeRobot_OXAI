#!/usr/bin/env python3
"""Compute a bias-corrected PLACE_OFFSET from one or more bullseye runs.

For each run, the landing error of the upright landings (landed point minus
target centre, board/world XY) is reduced to one robust bias estimate - the
median by default, so the long tail of far misses does not drag it. That bias
is rotated into the gripper frame at the release waypoint and added to the
run's place offset, exactly as build_general_xy_lookup.corrected_place_offset
does for a single pass:

    new_place = place + R_release^T @ [bias_x, bias_y, 0] * gain

Unlike the builder's per-pass correction loop, this uses the whole noisy
sample set, so random start noise averages out and only the repeatable bias is
removed. Suitable for blind runs.

Example:
  python -B lowlevel/drafts/calibrate_place_offset_from_bullseye.py \
      lowlevel/rook_kiri_xy_lookup/e4_to_*_e4ring_2026* --output /tmp/place_offsets.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np

DRAFTS_DIR = Path(__file__).resolve().parent
LOWLEVEL_DIR = DRAFTS_DIR.parent
sys.path.insert(0, str(DRAFTS_DIR))
from run_rook_offset_bullseye import ROOK_ENV, apply_far_place_env  # noqa: E402

os.environ.update(ROOK_ENV)
sys.path.insert(0, str(LOWLEVEL_DIR))
import sim_no_robot_driver  # noqa: E402

sim_no_robot_driver.install_light_sim_imports()  # same trajectories, no torch / IK meshes
with contextlib.redirect_stdout(io.StringIO()):
    import build_general_xy_lookup as builder  # noqa: E402
    from board_coordinates import XYPoint, square_center_world_xy  # noqa: E402
    from chess_traj import lift_height_for_square  # noqa: E402


def bias_estimate(errors_m: np.ndarray, estimator: str, trim_m: float) -> np.ndarray:
    if estimator == "median":
        return np.median(errors_m, axis=0)
    keep = np.linalg.norm(errors_m, axis=1) < trim_m
    return errors_m[keep].mean(axis=0)


def calibrate_run(run_dir: Path, estimator: str, trim_m: float, gain: float) -> dict:
    data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    cfg = data["config"]
    # The release pose must be rebuilt with the same far-file tilt cap the run used.
    apply_far_place_env(cfg)
    landed = [
        s for s in data["samples"]
        if "error" not in s and s.get("landed_upright") and s.get("final_error_xy_mm") is not None
    ]
    if not landed:
        raise ValueError(f"{run_dir.name}: no upright landings to calibrate from")
    errors_m = np.array([s["final_error_xy_mm"] for s in landed], dtype=float) / 1000.0
    bias = bias_estimate(errors_m, estimator, trim_m)

    src, dst = cfg["from_square"], cfg["to_square"]
    # Blind runs always plan from the source centre, so that is the trajectory to rotate with.
    start = XYPoint(*square_center_world_xy(src), name=f"{src}_center")
    target = XYPoint(*square_center_world_xy(dst), name=f"{dst}_center")
    grasp = np.array(cfg["grasp_offset"], dtype=float)
    place = np.array(cfg["place_offset"], dtype=float)
    with contextlib.redirect_stdout(io.StringIO()):
        rot = builder.release_rotation(
            start, target, grasp, place,
            lift_height=float(lift_height_for_square(target)),
            placement_lower_steps=int(cfg["placement_lower_steps"]),
        )
    new_place = place + rot.T @ (np.array([bias[0], bias[1], 0.0]) * gain)
    return {
        "run": run_dir.name,
        "from_square": src,
        "to_square": dst,
        "mode": cfg["mode"],
        "landed_samples": len(landed),
        "estimator": estimator,
        "bias_world_xy_mm": (bias * 1000.0).round(3).tolist(),
        "gain": gain,
        "grasp_offset": grasp.tolist(),
        "old_place_offset": place.tolist(),
        "new_place_offset": new_place.round(6).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--estimator", choices=("median", "trimmed-mean"), default="median")
    parser.add_argument("--trim-mm", type=float, default=10.0, help="trimmed-mean: ignore landings beyond this.")
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument(
        "--min-landed",
        type=int,
        default=30,
        help="Skip runs with fewer upright landings: a bias from a handful of survivors is not meaningful.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write {to_square: result} JSON here.")
    args = parser.parse_args()

    results = {}
    print(f"{'target':<7}{'landed':>7}  {'bias x,y (mm)':>16}   old place (mm)          ->  new place (mm)")
    for run_dir in args.run_dirs:
        data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        n_landed = sum(1 for s in data["samples"] if "error" not in s and s.get("landed_upright"))
        if n_landed < args.min_landed:
            print(f"{data['config']['to_square']:<7}{n_landed:>7}  skipped: fewer than {args.min_landed} upright landings")
            continue
        r = calibrate_run(run_dir.resolve(), args.estimator, args.trim_mm / 1000.0, args.gain)
        results[r["to_square"]] = r
        fmt = lambda v: "[" + ", ".join(f"{x * 1000:+6.2f}" for x in v) + "]"
        bx, by = r["bias_world_xy_mm"]
        print(f"{r['to_square']:<7}{r['landed_samples']:>7}  {bx:+7.2f},{by:+7.2f}   "
              f"{fmt(r['old_place_offset'])}  ->  {fmt(r['new_place_offset'])}")
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
