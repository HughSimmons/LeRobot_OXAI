#!/usr/bin/env python3
"""Sensitivity analysis: pickup PERCEPTION error on the d4->d5 rook move.

The rook physically sits at the true d4 centre in every trial. The robot's
grasp trajectory is planned for a perturbed pickup point (radius in mm,
angle in degrees, around d4 centre) instead -- simulating a vision/calibration
system that misjudges where the piece actually is. This measures whether the
gripper still finds/grasps/places the piece correctly despite aiming at the
wrong spot, which a naive "spawn the piece wherever you tell it to pick from"
sweep (see run_d4_d5_xy_error_sensitivity.py) cannot test: that sweep always
spawns the piece exactly where the robot aims, so it is trivially 100%
successful and never engages this failure mode.

Place target (d5 centre) is unperturbed -- only pickup perception error is
under test.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys

import numpy as np

LOWLEVEL_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = LOWLEVEL_DIR.parent
sys.path.insert(0, str(LOWLEVEL_DIR))

import os  # noqa: E402

os.environ.setdefault("LOOKUP_PIECE_MODEL", "rook_kiri")
os.environ.setdefault("ROOK_KIRI_MESH_PATH", str(PROJECT_DIR / "rook_kiri2" / "rook2.obj"))
os.environ.setdefault(
    "ROOK_KIRI_VISUAL_MESH_PATH",
    str(PROJECT_DIR / "rook_kiri2" / "rook2_debug_orange_visual.obj"),
)
os.environ.setdefault("ROOK_KIRI_MESH_UP_AXIS", "y")
os.environ.setdefault("ROOK_KIRI_COLLISION_MODEL", "banded_hulls")
os.environ.setdefault(
    "ROOK_KIRI_BAND_COLLISION_MESH_DIR",
    str(LOWLEVEL_DIR / "rook_kiri_lookup" / "collision_geometry_preview_20260801_163534"),
)

from board_coordinates import XYPoint, square_center_world_xy  # noqa: E402
import multisim_chess_fast as sim  # noqa: E402


DEFAULT_GRASP_OFFSET = np.array([-0.014, 0.002, -0.003], dtype=float)
DEFAULT_PLACE_OFFSET = np.array([-0.011, 0.002, -0.003], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-square", default="d4")
    parser.add_argument("--to-square", default="d5")
    parser.add_argument(
        "--radii-mm",
        type=float,
        nargs="+",
        default=[0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0],
    )
    parser.add_argument("--num-angles", type=int, default=8)
    parser.add_argument("--placement-corrections", type=int, default=0)
    parser.add_argument("--grasp-offset", nargs=3, type=float, default=DEFAULT_GRASP_OFFSET)
    parser.add_argument("--place-offset", nargs=3, type=float, default=DEFAULT_PLACE_OFFSET)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_samples(center: np.ndarray, radii_mm: list[float], num_angles: int) -> list[dict]:
    samples = []
    for radius_mm in radii_mm:
        radius_m = radius_mm / 1000.0
        angles = [0.0] if radius_mm == 0.0 else [
            360.0 * step / num_angles for step in range(num_angles)
        ]
        for angle_deg in angles:
            angle_rad = math.radians(angle_deg)
            offset = np.array([math.cos(angle_rad), math.sin(angle_rad)]) * radius_m
            samples.append(
                {
                    "radius_mm": radius_mm,
                    "angle_deg": angle_deg,
                    "commanded_pickup_world_xy": (center + offset).tolist(),
                }
            )
    return samples


def main() -> int:
    args = parse_args()
    if args.num_angles < 1:
        raise ValueError("num-angles must be at least 1")

    true_pickup_center = np.array(square_center_world_xy(args.from_square), dtype=float)
    samples = build_samples(true_pickup_center, args.radii_mm, args.num_angles)

    if args.dry_run:
        for sample in samples:
            print(
                f"DRY RUN r={sample['radius_mm']}mm a={sample['angle_deg']}deg "
                f"true_pos={true_pickup_center.tolist()} "
                f"commanded_pos={sample['commanded_pickup_world_xy']}"
            )
        print(f"{len(samples)} samples total")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        or LOWLEVEL_DIR
        / "rook_kiri_xy_lookup"
        / f"{args.from_square}_pickup_perception_error_to_{args.to_square}_{stamp}"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    to_point = XYPoint(*square_center_world_xy(args.to_square), name=f"{args.to_square}_center")

    world = sim.setup_sim_world(args.from_square)

    results = []
    for sample in samples:
        commanded_point = XYPoint(
            *sample["commanded_pickup_world_xy"],
            name=f"commanded_r{sample['radius_mm']:g}mm_a{sample['angle_deg']:g}deg",
        )
        # The piece was spawned at the true pickup centre when the world was
        # built; only the *label* used for trajectory planning changes here,
        # so relabel world["from_square"] to match rather than passing a
        # mismatched value (which would trip run_sim_move's consistency
        # check) or restore_state=False (which injects an extra "transition
        # from current joints" waypoint that doesn't apply -- the sim always
        # starts each trial from the freshly restored home state).
        world["from_square"] = commanded_point
        result = sim.run_sim_move(
            world,
            commanded_point,
            to_point,
            grasp_offset=np.array(args.grasp_offset, dtype=float),
            place_offset=np.array(args.place_offset, dtype=float),
            return_metrics=True,
            restore_state=True,
            placement_lower_steps=10,
        )
        success = sim.is_successful_move_result(result)
        xy_error = sim.result_xy_error(result)
        tilt_error = sim.result_tilt_error(result)
        record = {
            "radius_mm": sample["radius_mm"],
            "angle_deg": sample["angle_deg"],
            "commanded_pickup_world_xy": sample["commanded_pickup_world_xy"],
            "true_pickup_world_xy": true_pickup_center.tolist(),
            "success": bool(success),
            "pickup_success": bool(result.get("pickup_success", False)),
            "premature_drop": bool(result.get("premature_drop", False)),
            "xy_error": xy_error,
            "final_tilt_deg": tilt_error,
            "trajectory_fk_error": float(result.get("trajectory_fk_error", float("nan"))),
        }
        results.append(record)
        print(
            f"r={record['radius_mm']:>4}mm a={record['angle_deg']:>5}deg  "
            f"success={record['success']!s:5}  pickup={record['pickup_success']!s:5}  "
            f"premature_drop={record['premature_drop']!s:5}  xy_err={xy_error:.5f}  "
            f"tilt={tilt_error:.3f}",
            flush=True,
        )

    by_radius: dict[float, list[bool]] = {}
    for record in results:
        by_radius.setdefault(record["radius_mm"], []).append(record["success"])
    success_rate_by_radius_mm = {
        str(radius): sum(outcomes) / len(outcomes) for radius, outcomes in sorted(by_radius.items())
    }

    summary = {
        "from_square": args.from_square,
        "to_square": args.to_square,
        "true_pickup_world_xy": true_pickup_center.tolist(),
        "to_world_xy": [to_point.x, to_point.y],
        "radii_mm": args.radii_mm,
        "num_angles": args.num_angles,
        "grasp_offset": list(args.grasp_offset),
        "place_offset": list(args.place_offset),
        "success_rate_by_radius_mm": success_rate_by_radius_mm,
        "samples": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
