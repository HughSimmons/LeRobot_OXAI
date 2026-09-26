#!/usr/bin/env python3
"""Sensitivity analysis: initial pickup XY error on the d4->d5 rook move.

Perturbs the assumed pickup position of the rook on d4 radially (fixed radii
in mm, evenly spaced angles per radius) while placing at the exact d5 centre,
and records whether the simulated pick-and-place still succeeds. This isolates
sensitivity to *pickup*-position error rather than sweeping a square grid.

Uses board_coordinates.square_center_world_xy() (the live calibration) for the
d4/d5 centres, unlike the older run_rook_square_pickup_grid.py draft, which
hardcodes a stale square size/board centre.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[3]
LOWLEVEL_DIR = PROJECT_DIR / "lowlevel"
sys.path.insert(0, str(LOWLEVEL_DIR))

from board_coordinates import square_center_world_xy  # noqa: E402

BUILDER = LOWLEVEL_DIR / "build_general_xy_lookup.py"
ROOK_MESH = PROJECT_DIR / "rook_kiri2" / "rook2.obj"
ROOK_VISUAL_MESH = PROJECT_DIR / "rook_kiri2" / "rook2_debug_orange_visual.obj"
BAND_COLLISION_DIR = (
    LOWLEVEL_DIR
    / "rook_kiri_lookup"
    / "collision_geometry_preview_20260801_163534"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-square", default="d4")
    parser.add_argument("--to-square", default="d5")
    parser.add_argument(
        "--radii-mm",
        type=float,
        nargs="+",
        default=[0.0, 4.0, 8.0, 12.0, 16.0],
        help="Pickup-error radii to sample, in mm (0 contributes a single centre sample).",
    )
    parser.add_argument(
        "--num-angles",
        type=int,
        default=8,
        help="Evenly spaced angles per nonzero radius.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--candidate-db", type=Path, default=None)
    parser.add_argument("--placement-corrections", type=int, default=3)
    parser.add_argument("--grasp-offset", nargs=3, type=float, default=(-0.014, 0.002, -0.003))
    parser.add_argument("--place-offset", nargs=3, type=float, default=(-0.011, 0.002, -0.003))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def rook_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "LOOKUP_PIECE_MODEL": "rook_kiri",
            "ROOK_KIRI_MESH_PATH": str(ROOK_MESH),
            "ROOK_KIRI_VISUAL_MESH_PATH": str(ROOK_VISUAL_MESH),
            "ROOK_KIRI_MESH_UP_AXIS": "y",
            "ROOK_KIRI_COLLISION_MODEL": "banded_hulls",
            "ROOK_KIRI_BAND_COLLISION_MESH_DIR": str(BAND_COLLISION_DIR),
        }
    )
    return env


def build_samples(from_center: np.ndarray, radii_mm: list[float], num_angles: int) -> list[dict]:
    samples = []
    sample_index = 0
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
                    "sample_index": sample_index,
                    "radius_mm": radius_mm,
                    "angle_deg": angle_deg,
                    "from_world_xy": (from_center + offset).tolist(),
                }
            )
            sample_index += 1
    return samples


def main() -> int:
    args = parse_args()
    if args.num_angles < 1:
        raise ValueError("num-angles must be at least 1")
    for path in (BUILDER, ROOK_MESH, ROOK_VISUAL_MESH, BAND_COLLISION_DIR):
        if not path.exists():
            raise FileNotFoundError(path)

    from_center = np.array(square_center_world_xy(args.from_square), dtype=float)
    to_center = np.array(square_center_world_xy(args.to_square), dtype=float)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        or LOWLEVEL_DIR
        / "rook_kiri_xy_lookup"
        / f"{args.from_square}_xy_error_sensitivity_to_{args.to_square}_{stamp}"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    candidate_db = (
        args.candidate_db.expanduser().resolve()
        if args.candidate_db is not None
        else LOWLEVEL_DIR / "rook_kiri_xy_lookup" / "continuous_xy_candidates.sqlite3"
    )
    env = rook_environment()
    samples = build_samples(from_center, args.radii_mm, args.num_angles)

    metadata = {
        "from_square": args.from_square,
        "to_square": args.to_square,
        "from_square_world_xy": from_center.tolist(),
        "to_square_world_xy": to_center.tolist(),
        "radii_mm": args.radii_mm,
        "num_angles": args.num_angles,
        "candidate_db": str(candidate_db),
        "placement_corrections": args.placement_corrections,
        "grasp_offset": args.grasp_offset,
        "place_offset": args.place_offset,
        "environment": {
            key: env[key]
            for key in (
                "LOOKUP_PIECE_MODEL",
                "ROOK_KIRI_MESH_PATH",
                "ROOK_KIRI_VISUAL_MESH_PATH",
                "ROOK_KIRI_COLLISION_MODEL",
                "ROOK_KIRI_BAND_COLLISION_MESH_DIR",
            )
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {**metadata, "samples": []}
    for sample in samples:
        sample_name = f"{args.from_square}_r{sample['radius_mm']:g}mm_a{sample['angle_deg']:g}deg"
        sample_dir = output_dir / sample_name
        from_xy = sample["from_world_xy"]
        command = [
            sys.executable,
            "-B",
            str(BUILDER),
            "--from-xy",
            *map(str, from_xy),
            "--to-xy",
            *map(str, to_center),
            "--frame",
            "world",
            "--from-name",
            sample_name,
            "--to-name",
            f"{args.to_square}_center",
            "--output-dir",
            str(sample_dir),
            "--candidate-db",
            str(candidate_db),
            "--database-accept-final-square",
            args.to_square,
            "--grid-radius",
            "0",
            "--grid-z-radius",
            "0",
            "--placement-corrections",
            str(args.placement_corrections),
            "--grasp-offset",
            *map(str, args.grasp_offset),
            "--place-offset",
            *map(str, args.place_offset),
        ]
        sample["name"] = sample_name
        sample["command"] = command

        if args.dry_run:
            summary["samples"].append(sample)
            print(f"DRY RUN {sample_name}: {' '.join(command)}")
            continue

        with (output_dir / f"{sample_name}.log").open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                command,
                cwd=PROJECT_DIR,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        sample["exit_code"] = process.returncode
        lookup_paths = sorted(sample_dir.glob("*.json"))
        if lookup_paths:
            lookup = json.loads(lookup_paths[0].read_text(encoding="utf-8"))
            sample["lookup_json"] = str(lookup_paths[0])
            sample["strict_success"] = bool(lookup.get("success"))
            sample["selected_xy_error"] = lookup.get("metrics", {}).get("xy_error")
            sample["selected_tilt_deg"] = lookup.get("metrics", {}).get("final_tilt_deg")
        else:
            sample["lookup_json"] = None
        summary["samples"].append(sample)
        print(
            f"{sample_name}: strict_success={sample.get('strict_success')} "
            f"xy_error={sample.get('selected_xy_error')} tilt={sample.get('selected_tilt_deg')}",
            flush=True,
        )

    if not args.dry_run:
        by_radius: dict[float, list[bool]] = {}
        for sample in summary["samples"]:
            by_radius.setdefault(sample["radius_mm"], []).append(bool(sample.get("strict_success")))
        summary["success_rate_by_radius_mm"] = {
            str(radius): sum(results) / len(results) for radius, results in sorted(by_radius.items())
        }
        summary["strict_success_count"] = sum(
            bool(sample.get("strict_success")) for sample in summary["samples"]
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
