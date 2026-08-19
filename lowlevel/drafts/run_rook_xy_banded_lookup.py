#!/usr/bin/env python3
"""Run a continuous XY rook lookup with the current banded collision model."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
LOWLEVEL_DIR = PROJECT_DIR / "lowlevel"
BUILDER = LOWLEVEL_DIR / "build_general_xy_lookup.py"
VERIFIER = LOWLEVEL_DIR / "verify_xy_lookup_move.py"
ROOK_MESH = PROJECT_DIR / "rook_kiri2" / "rook2.obj"
ROOK_VISUAL_MESH = PROJECT_DIR / "rook_kiri2" / "rook2_debug_orange_visual.obj"
BAND_COLLISION_DIR = (
    LOWLEVEL_DIR
    / "rook_kiri_lookup"
    / "collision_geometry_preview_20260801_163534"
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run and optionally verify one continuous rook XY lookup."
    )
    parser.add_argument("--from-xy", nargs=2, type=float, required=True)
    parser.add_argument("--to-xy", nargs=2, type=float, required=True)
    parser.add_argument("--frame", choices=("world", "board"), default="world")
    parser.add_argument("--from-name", default=None)
    parser.add_argument("--to-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--video", action="store_true")
    return parser.parse_known_args()


def run_logged(command: list[str], env: dict[str, str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return process.returncode


def main() -> int:
    args, builder_extra_args = parse_args()
    for required_path in (BUILDER, VERIFIER, ROOK_MESH, ROOK_VISUAL_MESH, BAND_COLLISION_DIR):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        or LOWLEVEL_DIR / "rook_kiri_xy_lookup" / f"xy_lookup_{stamp}"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    lookup_dir = output_dir / "lookup"
    verify_dir = output_dir / "verification"

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
    builder_command = [
        sys.executable,
        "-B",
        str(BUILDER),
        "--from-xy",
        *map(str, args.from_xy),
        "--to-xy",
        *map(str, args.to_xy),
        "--frame",
        args.frame,
        "--output-dir",
        str(lookup_dir),
    ]
    if args.from_name:
        builder_command.extend(("--from-name", args.from_name))
    if args.to_name:
        builder_command.extend(("--to-name", args.to_name))
    builder_command.extend(builder_extra_args)

    metadata = {
        "timestamp": stamp,
        "builder_command": builder_command,
        "environment": {
            key: env[key]
            for key in (
                "LOOKUP_PIECE_MODEL",
                "ROOK_KIRI_MESH_PATH",
                "ROOK_KIRI_VISUAL_MESH_PATH",
                "ROOK_KIRI_MESH_UP_AXIS",
                "ROOK_KIRI_COLLISION_MODEL",
                "ROOK_KIRI_BAND_COLLISION_MESH_DIR",
            )
        },
        "verify_requested": args.verify,
        "video_requested": args.video,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Output dir: {output_dir}")
    builder_code = run_logged(builder_command, env, output_dir / "builder.log")
    print(f"Builder exit code: {builder_code}")
    lookup_paths = sorted(lookup_dir.glob("*.json")) if lookup_dir.exists() else []
    if not lookup_paths:
        print(f"No lookup JSON was produced; see {output_dir / 'builder.log'}")
        return builder_code or 2

    lookup_path = lookup_paths[0]
    lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
    print(f"Lookup: {lookup_path}")
    print(f"Success: {lookup.get('success')}")
    if not args.verify or not lookup.get("success"):
        return builder_code

    verifier_command = [
        sys.executable,
        "-B",
        str(VERIFIER),
        str(lookup_path),
        "--output-dir",
        str(verify_dir),
    ]
    if args.video:
        verifier_command.append("--video")
    verifier_code = run_logged(verifier_command, env, output_dir / "verifier.log")
    print(f"Verifier exit code: {verifier_code}")
    print(f"Verification: {verify_dir / 'summary.json'}")
    return verifier_code


if __name__ == "__main__":
    raise SystemExit(main())

