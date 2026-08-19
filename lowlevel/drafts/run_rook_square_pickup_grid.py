#!/usr/bin/env python3
"""Sweep pickup positions across one square and store stable regional placements."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
LOWLEVEL_DIR = PROJECT_DIR / "lowlevel"
BUILDER = LOWLEVEL_DIR / "build_general_xy_lookup.py"
ROOK_MESH = PROJECT_DIR / "rook_kiri2" / "rook2.obj"
ROOK_VISUAL_MESH = PROJECT_DIR / "rook_kiri2" / "rook2_debug_orange_visual.obj"
BAND_COLLISION_DIR = (
    LOWLEVEL_DIR
    / "rook_kiri_lookup"
    / "collision_geometry_preview_20260801_163534"
)
SQUARE_SIZE_M = 0.04
BOARD_CENTER_WORLD_XY = (0.25, 0.0)


def parse_square(square: str) -> tuple[int, int]:
    square = square.lower()
    if len(square) != 2 or square[0] not in "abcdefgh" or square[1] not in "12345678":
        raise ValueError(f"Invalid square {square!r}")
    return ord(square[0]) - ord("a"), int(square[1]) - 1


def square_center_board_xy(square: str) -> tuple[float, float]:
    file_index, rank_index = parse_square(square)
    return (
        (file_index - 3.5) * SQUARE_SIZE_M,
        (rank_index - 3.5) * SQUARE_SIZE_M,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a coarse grid of continuous pickup coordinates inside one square, "
            "placing toward a target-square centre while storing stable regional results."
        )
    )
    parser.add_argument("--from-square", default="d4")
    parser.add_argument("--to-square", default="d6")
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument(
        "--inset",
        type=float,
        default=0.01,
        help="Maximum centre-relative pickup displacement in metres (default 10 mm).",
    )
    parser.add_argument(
        "--existing-inset",
        type=float,
        default=None,
        help="Exclude the previously covered inner grid up to this inset.",
    )
    parser.add_argument(
        "--new-ring-only",
        action="store_true",
        help="Run only points outside --existing-inset.",
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


def main() -> int:
    args = parse_args()
    parse_square(args.from_square)
    parse_square(args.to_square)
    if args.grid_size < 1:
        raise ValueError("grid-size must be at least 1")
    if args.inset < 0.0 or args.inset > SQUARE_SIZE_M:
        raise ValueError("inset must be between 0 and 0.04 m")
    if args.new_ring_only:
        if args.existing_inset is None:
            raise ValueError("--new-ring-only requires --existing-inset")
        if args.existing_inset < 0.0 or args.existing_inset >= args.inset:
            raise ValueError("existing-inset must be non-negative and smaller than inset")
    for path in (BUILDER, ROOK_MESH, ROOK_VISUAL_MESH, BAND_COLLISION_DIR):
        if not path.exists():
            raise FileNotFoundError(path)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        or LOWLEVEL_DIR / "rook_kiri_xy_lookup" / f"{args.from_square}_pickup_grid_to_{args.to_square}_{stamp}"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    from_center = np.array(square_center_board_xy(args.from_square), dtype=float)
    to_center = np.array(square_center_board_xy(args.to_square), dtype=float)
    offsets = np.linspace(-args.inset, args.inset, args.grid_size)
    samples = [
        (row_index, column_index, from_center + np.array([dx, dy], dtype=float))
        for row_index, dy in enumerate(offsets)
        for column_index, dx in enumerate(offsets)
        if not args.new_ring_only
        or abs(float(dx)) > float(args.existing_inset) + 1e-12
        or abs(float(dy)) > float(args.existing_inset) + 1e-12
    ]
    candidate_db = (
        args.candidate_db.expanduser().resolve()
        if args.candidate_db is not None
        else LOWLEVEL_DIR / "rook_kiri_xy_lookup" / "continuous_xy_candidates.sqlite3"
    )
    env = rook_environment()
    metadata = {
        "from_square": args.from_square,
        "to_square": args.to_square,
        "grid_size": args.grid_size,
        "inset_m": args.inset,
        "existing_inset_m": args.existing_inset,
        "new_ring_only": args.new_ring_only,
        "target_board_xy": to_center.tolist(),
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
    for row_index, column_index, from_xy in samples:
        sample_name = f"{args.from_square}_r{row_index}_c{column_index}"
        sample_dir = output_dir / sample_name
        command = [
            sys.executable,
            "-B",
            str(BUILDER),
            "--from-xy",
            *map(str, from_xy),
            "--to-xy",
            *map(str, to_center),
            "--frame",
            "board",
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
        sample = {
            "row": row_index,
            "column": column_index,
            "from_board_xy": from_xy.tolist(),
            "command": command,
        }
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
            sample["database_stored_candidates"] = int(
                lookup.get("candidate_database", {}).get("stored_candidate_count", 0)
            )
            sample["selected_xy_error"] = lookup.get("metrics", {}).get("xy_error")
            sample["selected_tilt_deg"] = lookup.get("metrics", {}).get("final_tilt_deg")
        else:
            sample["lookup_json"] = None
        summary["samples"].append(sample)
        print(
            f"{sample_name}: strict_success={sample.get('strict_success')} "
            f"database_candidates={sample.get('database_stored_candidates', 0)}",
            flush=True,
        )

    summary["strict_success_count"] = sum(
        bool(sample.get("strict_success")) for sample in summary["samples"]
    )
    summary["database_candidate_count"] = sum(
        int(sample.get("database_stored_candidates", 0))
        for sample in summary["samples"]
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
