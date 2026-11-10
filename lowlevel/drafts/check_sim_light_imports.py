#!/usr/bin/env python3
"""Check that sim_no_robot_driver's light imports leave trajectories unchanged.

Run twice, once with --light, once without; each writes the planned joint lists for a few moves
(including far-file pivot places) to an .npz. Then --compare A B reports the largest difference.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
MOVES = [("e4", "e5", None), ("e4", "f4", "pivot"), ("d3", "f5", "pivot"), ("e5", "d3", None), ("d5", "f3", "pivot")]


def plan(out: Path, light: bool) -> None:
    from run_rook_offset_bullseye import ROOK_ENV, LOWLEVEL_DIR
    os.environ.update(ROOK_ENV)
    sys.path.insert(0, str(LOWLEVEL_DIR))
    if light:
        import sim_no_robot_driver
        sim_no_robot_driver.install_light_sim_imports()
    def rss():
        return int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) // 1024
    with contextlib.redirect_stdout(io.StringIO()):
        import build_general_xy_lookup as b
        from board_coordinates import XYPoint, square_center_world_xy
        from chess_traj import lift_height_for_square
        import multisim_chess_fast as sim
    print(f"light={light}: RSS after imports {rss()} MB, torch loaded: {'torch' in sys.modules}")
    g = np.array([-0.02, -0.001, -0.003]); p = np.array([-0.00875, 0.00223, -0.00267])
    arrays = {}
    for src, dst, style in MOVES:
        os.environ["CHESS_FAR_PLACE_MAX_TILT_DEG"] = "11"
        os.environ["CHESS_FAR_PLACE_STYLE"] = style or "waypoint"
        os.environ["CHESS_FAR_PLACE_PIVOT_HEIGHT"] = "0.02"
        a = XYPoint(*square_center_world_xy(src)); t = XYPoint(*square_center_world_xy(dst))
        with contextlib.redirect_stdout(io.StringIO()):
            moves, closeidx, _ = b.pickupmove_traj_with_metrics(
                a, t, board_origin=sim.board_origin, GRASP_OFFSET=g, PLACE_OFFSET=p,
                placement_lower_steps=10, lift_height=float(lift_height_for_square(t)))
        arrays[f"{src}_{dst}"] = np.array(moves, dtype=float)
    np.savez(out, **arrays)


def compare(a: Path, b: Path) -> int:
    A, B = np.load(a), np.load(b)
    worst = 0.0
    for k in A.files:
        if A[k].shape != B[k].shape:
            print(f"{k}: shape differs {A[k].shape} vs {B[k].shape}")
            return 1
        d = float(np.max(np.abs(A[k] - B[k])))
        worst = max(worst, d)
        print(f"{k}: {A[k].shape[0]} waypoints, max joint difference {d:.3g} deg")
    print("IDENTICAL" if worst == 0 else f"max difference {worst:.3g} deg")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--compare", nargs=2, type=Path)
    args = ap.parse_args()
    raise SystemExit(compare(*args.compare) if args.compare else plan(args.out, args.light))
