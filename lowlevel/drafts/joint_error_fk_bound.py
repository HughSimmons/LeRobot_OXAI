#!/usr/bin/env python3
"""Push joint-angle errors through forward kinematics at the planned grasp pose of each square.

Joint errors (URDF deg, pan / lift / elbow / wrist_flex / wrist_roll), from Oliver's 14 Nov slides:
  T1 hardware - plan at rest, end of the grasp hold, RMS   0.80 0.62 0.22 0.27 0.70
  T2 same-command tracking error during motion, RMS        1.6  3.5  2.7  1.0  1.3
Output: per square, 1-sigma grasp-point error (mm) in world x (away from robot), y, z for each tier
(independent joints), plus the per-joint Jacobian (mm/deg). CSV + JSON.

  python -B lowlevel/drafts/joint_error_fk_bound.py --squares c3 d4 e4 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_rook_offset_bullseye import LOWLEVEL_DIR, PROJECT_DIR, ROOK_ENV  # noqa: E402
sys.path.insert(0, str(LOWLEVEL_DIR))

TIERS = {"T1_rest": [0.80, 0.62, 0.22, 0.27, 0.70], "T2_tracking": [1.6, 3.5, 2.7, 1.0, 1.3]}
JOINTS = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll"]
GRASP = np.array([-0.020, -0.001, -0.003])
PLACE = np.array([-0.00875, 0.00223, -0.00267])


def carry_target(sq: str) -> str:
    r = int(sq[1])
    return f"{sq[0]}{r + 1 if r < 7 else r - 1}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--squares", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.update(ROOK_ENV)
    import sim_no_robot_driver

    sim_no_robot_driver.install_light_sim_imports()
    os.chdir(PROJECT_DIR)
    from board_coordinates import XYPoint, square_center_world_xy
    import build_general_xy_lookup as builder
    from chess_traj import kinematics, lift_height_for_square
    import multisim_chess_fast as sim

    rows, full = [], {}
    for sq in args.squares:
        to = carry_target(sq)
        sx, sy = square_center_world_xy(sq)
        tx, ty = square_center_world_xy(to)
        piece, target = XYPoint(sx, sy, name="p"), XYPoint(tx, ty, name="t")
        sim.ensure_physics_connected()
        world = sim.setup_sim_world(piece, edge_support_margin=builder.EDGE_SUPPORT_MARGIN)
        with sim.temporary_home_joints(world.get("trajectory_home_joints_deg")):
            ml, ci, _ = sim.pickupmove_traj_with_metrics(
                piece, target, board_origin=sim.board_origin, GRASP_OFFSET=GRASP, PLACE_OFFSET=PLACE,
                placement_lower_steps=10, lift_height=float(lift_height_for_square(target)), lower_place_path_bias=None)
        q = np.array(ml[ci], dtype=float)

        def grasp_point(qq):
            t = kinematics.forward_kinematics(qq)
            return t[:3, 3] + t[:3, :3] @ GRASP

        p0 = grasp_point(q)
        # radial unit vector (away from the robot base) and tangential, so x/y read the same on every square
        radial = np.array([p0[0], p0[1], 0.0]) / np.hypot(p0[0], p0[1])
        tang = np.array([-radial[1], radial[0], 0.0])
        basis = np.vstack([radial, tang, [0, 0, 1.0]])
        jac = np.zeros((3, 5))
        for j in range(5):
            qq = q.copy()
            qq[j] += 0.01
            jac[:, j] = basis @ (grasp_point(qq) - p0) / 0.01 * 1000.0  # mm/deg
        row = {"square": sq, "reach_mm": float(np.hypot(p0[0], p0[1]) * 1000), "grasp_joints_deg": q[:5].round(2).tolist()}
        for name, sig in TIERS.items():
            cov = jac @ np.diag(np.square(sig)) @ jac.T
            s = np.sqrt(np.diag(cov))
            row.update({f"{name}_radial_mm": s[0], f"{name}_tangential_mm": s[1], f"{name}_z_mm": s[2]})
            full.setdefault(sq, {})[f"{name}_cov_mm2"] = cov.tolist()
        full[sq]["jacobian_mm_per_deg"] = jac.tolist()
        rows.append(row)
        print(sq, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    import csv

    with (args.out / "fk_bound.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out / "fk_bound.json").write_text(json.dumps({"tiers_deg": TIERS, "joints": JOINTS, "squares": full,
                                                        "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
