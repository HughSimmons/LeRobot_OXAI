#!/usr/bin/env python3
"""Run one focused rook builder search and replay the saved best result.

This is intentionally narrow: it threads the current rook visual mesh plus the
latest five-band collision hulls through the general lookup builder, then saves
one verifier video for the selected lookup entry if the builder finds a success.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
LOWLEVEL_DIR = PROJECT_DIR / "lowlevel"
BUILDER = LOWLEVEL_DIR / "build_general_nonh_reverse_lookup.py"
VERIFIER = LOWLEVEL_DIR / "verify_nonh_lookup_moves.py"
ROOK_MESH = PROJECT_DIR / "rook_kiri2" / "rook2.obj"
ROOK_VISUAL_MESH = PROJECT_DIR / "rook_kiri2" / "rook2_debug_orange_visual.obj"
BAND_COLLISION_DIR = (
    LOWLEVEL_DIR
    / "rook_kiri_lookup"
    / "collision_geometry_preview_20260801_163534"
)


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def parse_move_key(move_key: str) -> tuple[str, str]:
    parts = move_key.split("_to_")
    if len(parts) != 2 or any(len(part) != 2 for part in parts):
        raise ValueError(f"Invalid move key {move_key!r}; use e.g. d4_to_d6")
    return parts[0], parts[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one rook lookup through the general builder with the latest "
            "banded collision hulls, then replay the saved best result on video."
        )
    )
    parser.add_argument("move_key", help="Move key, e.g. d4_to_d6")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional exact output directory. Defaults to "
            "lowlevel/rook_kiri_lookup/<move_key>_banded_builder/<timestamp>."
        ),
    )
    return parser.parse_args()


def run_logged(command, env, log_path):
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
    args = parse_args()
    move_key = args.move_key
    from_square, to_square = parse_move_key(move_key)

    require_path(ROOK_MESH, "Rook mesh")
    require_path(ROOK_VISUAL_MESH, "Rook visual mesh")
    require_path(BAND_COLLISION_DIR, "Banded collision directory")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = LOWLEVEL_DIR / "rook_kiri_lookup" / f"{move_key}_banded_builder"
    env_output_dir = os.environ.get("ROOK_BANDED_LOOKUP_OUTPUT_DIR")
    output_dir = Path(
        args.output_dir or env_output_dir or output_root / stamp
    ).expanduser().resolve()
    lookup_output_dir = output_dir / "lookup"
    donor_dir = output_dir / "empty_donor_lookup"
    video_dir = output_dir / "best_candidate_video"
    output_dir.mkdir(parents=True, exist_ok=False)
    lookup_output_dir.mkdir(parents=True, exist_ok=True)
    donor_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "LOOKUP_PIECE_MODEL": "rook_kiri",
            "ROOK_KIRI_MESH_PATH": str(ROOK_MESH),
            "ROOK_KIRI_VISUAL_MESH_PATH": str(ROOK_VISUAL_MESH),
            "ROOK_KIRI_MESH_UP_AXIS": "y",
            "ROOK_KIRI_COLLISION_MODEL": "banded_hulls",
            "ROOK_KIRI_BAND_COLLISION_MESH_DIR": str(BAND_COLLISION_DIR),
            "SOURCE_SQUARES": from_square,
            "TARGET_MOVES": move_key,
            "LOOKUP_OUTPUT_DIR": str(lookup_output_dir),
            "DONOR_LOOKUP_DIR": str(donor_dir),
        }
    )

    builder_command = [sys.executable, "-B", str(BUILDER)]
    verifier_command = [
        sys.executable,
        "-B",
        str(VERIFIER),
        "--lookup-dir",
        str(lookup_output_dir),
        "--output-dir",
        str(video_dir),
        move_key,
    ]
    run_metadata = {
        "purpose": "Focused rook general-builder search using latest banded collision hulls, then video replay of saved best candidate.",
        "timestamp": stamp,
        "move_key": move_key,
        "from_square": from_square,
        "to_square": to_square,
        "python_executable": sys.executable,
        "builder_command": builder_command,
        "verifier_command": verifier_command,
        "environment": {
            key: env[key]
            for key in (
                "LOOKUP_PIECE_MODEL",
                "ROOK_KIRI_MESH_PATH",
                "ROOK_KIRI_VISUAL_MESH_PATH",
                "ROOK_KIRI_MESH_UP_AXIS",
                "ROOK_KIRI_COLLISION_MODEL",
                "ROOK_KIRI_BAND_COLLISION_MESH_DIR",
                "SOURCE_SQUARES",
                "TARGET_MOVES",
                "LOOKUP_OUTPUT_DIR",
                "DONOR_LOOKUP_DIR",
            )
        },
        "lookup_json": str(lookup_output_dir / f"{from_square}_non_h_reverse_move_lookup.json"),
        "video_dir": str(video_dir),
        "builder_log": str(output_dir / "builder.log"),
        "verifier_log": str(output_dir / "verifier.log"),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Output dir: {output_dir}")
    print(f"Running builder: {' '.join(builder_command)}")
    builder_code = run_logged(builder_command, env, output_dir / "builder.log")
    print(f"Builder exit code: {builder_code}")
    if builder_code != 0:
        return builder_code

    lookup_path = lookup_output_dir / f"{from_square}_non_h_reverse_move_lookup.json"
    lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
    entry = lookup.get("moves", {}).get(move_key)
    if not entry or not entry.get("success"):
        print(f"No successful {move_key} entry was saved; see {output_dir / 'builder.log'}")
        return 2

    metrics = entry.get("metrics", {})
    print(
        f"Saved {move_key}: "
        f"xy={metrics.get('xy_error')} "
        f"tilt={metrics.get('final_tilt_deg')} "
        f"fk={metrics.get('trajectory_fk_error')}"
    )
    print(f"Recording verifier video: {' '.join(verifier_command)}")
    verifier_code = run_logged(verifier_command, env, output_dir / "verifier.log")
    print(f"Verifier exit code: {verifier_code}")
    print(f"Best-candidate video/summary dir: {video_dir}")
    return verifier_code


if __name__ == "__main__":
    raise SystemExit(main())
