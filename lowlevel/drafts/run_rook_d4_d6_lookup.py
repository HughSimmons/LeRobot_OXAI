#!/usr/bin/env python3
"""Run the initial rook_kiri d4_to_d6 lookup with the selected mesh candidate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
LOWLEVEL_DIR = PROJECT_DIR / "lowlevel"
BUILDER = LOWLEVEL_DIR / "build_general_nonh_reverse_lookup.py"
DEFAULT_MESH = (
    PROJECT_DIR
    / "rook_kiri2"
    / "rook2.obj"
)
DEFAULT_VISUAL_MESH = DEFAULT_MESH.with_name(DEFAULT_MESH.stem + "_debug_orange_visual.obj")
DEFAULT_BAND_COLLISION_DIR = (
    LOWLEVEL_DIR
    / "rook_kiri_lookup"
    / "collision_geometry_preview_20260801_163534"
)
DEFAULT_OUTPUT_ROOT = LOWLEVEL_DIR / "rook_kiri_lookup" / "d4_to_d6_initial"


def main():
    mesh_path = Path(os.environ.get("ROOK_TEST_MESH_PATH", DEFAULT_MESH)).expanduser().resolve()
    visual_mesh_path = Path(
        os.environ.get("ROOK_TEST_VISUAL_MESH_PATH", DEFAULT_VISUAL_MESH)
    ).expanduser().resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Rook test mesh does not exist: {mesh_path}")
    if not visual_mesh_path.exists():
        raise FileNotFoundError(f"Rook visual test mesh does not exist: {visual_mesh_path}")
    band_collision_dir = Path(
        os.environ.get("ROOK_TEST_BAND_COLLISION_DIR", DEFAULT_BAND_COLLISION_DIR)
    ).expanduser().resolve()
    if not band_collision_dir.exists():
        raise FileNotFoundError(f"Rook band collision dir does not exist: {band_collision_dir}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(os.environ.get("ROOK_TEST_OUTPUT_DIR", DEFAULT_OUTPUT_ROOT / stamp)).expanduser().resolve()
    donor_dir = output_dir / "empty_donor_lookup"
    output_dir.mkdir(parents=True, exist_ok=False)
    donor_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    video_output_dir = output_dir / "videos"
    env.update(
        {
            "LOOKUP_PIECE_MODEL": "rook_kiri",
            "ROOK_KIRI_MESH_PATH": str(mesh_path),
            "ROOK_KIRI_VISUAL_MESH_PATH": str(visual_mesh_path),
            "ROOK_KIRI_MESH_UP_AXIS": "y",
            "ROOK_KIRI_COLLISION_MODEL": "banded_hulls",
            "ROOK_KIRI_BAND_COLLISION_MESH_DIR": str(band_collision_dir),
            "SOURCE_SQUARES": "d4",
            "TARGET_MOVES": "d4_to_d6",
            "LOOKUP_OUTPUT_DIR": str(output_dir),
            "LOOKUP_RECORD_VIDEO": "1",
            "LOOKUP_VIDEO_OUTPUT_DIR": str(video_output_dir),
            # Keep the first rook lookup isolated from archived cylinder donors.
            "DONOR_LOOKUP_DIR": str(donor_dir),
        }
    )

    command = [sys.executable, "-B", str(BUILDER)]
    command_text = " ".join(command)
    shell_command = (
        f'LOOKUP_PIECE_MODEL=rook_kiri \\\n'
        f'ROOK_KIRI_MESH_PATH="{mesh_path}" \\\n'
        f'ROOK_KIRI_VISUAL_MESH_PATH="{visual_mesh_path}" \\\n'
        f'ROOK_KIRI_MESH_UP_AXIS=y \\\n'
        f'ROOK_KIRI_COLLISION_MODEL=banded_hulls \\\n'
        f'ROOK_KIRI_BAND_COLLISION_MESH_DIR="{band_collision_dir}" \\\n'
        f'SOURCE_SQUARES=d4 \\\n'
        f'TARGET_MOVES=d4_to_d6 \\\n'
        f'LOOKUP_OUTPUT_DIR="{output_dir}" \\\n'
        f'DONOR_LOOKUP_DIR="{donor_dir}" \\\n'
        f'{sys.executable} -B {BUILDER}'
    )
    run_metadata = {
        "purpose": "Initial rook_kiri lookup search tightly scoped to d4_to_d6.",
        "timestamp": stamp,
        "python_executable": sys.executable,
        "builder": str(BUILDER),
        "command": command,
        "shell_command": shell_command,
        "environment": {
            "LOOKUP_PIECE_MODEL": env["LOOKUP_PIECE_MODEL"],
            "ROOK_KIRI_MESH_PATH": env["ROOK_KIRI_MESH_PATH"],
            "ROOK_KIRI_VISUAL_MESH_PATH": env["ROOK_KIRI_VISUAL_MESH_PATH"],
            "ROOK_KIRI_MESH_UP_AXIS": env["ROOK_KIRI_MESH_UP_AXIS"],
            "ROOK_KIRI_COLLISION_MODEL": env["ROOK_KIRI_COLLISION_MODEL"],
            "ROOK_KIRI_BAND_COLLISION_MESH_DIR": env["ROOK_KIRI_BAND_COLLISION_MESH_DIR"],
            "SOURCE_SQUARES": env["SOURCE_SQUARES"],
            "TARGET_MOVES": env["TARGET_MOVES"],
            "LOOKUP_OUTPUT_DIR": env["LOOKUP_OUTPUT_DIR"],
            "LOOKUP_RECORD_VIDEO": env["LOOKUP_RECORD_VIDEO"],
            "LOOKUP_VIDEO_OUTPUT_DIR": env["LOOKUP_VIDEO_OUTPUT_DIR"],
            "DONOR_LOOKUP_DIR": env["DONOR_LOOKUP_DIR"],
        },
        "expected_lookup_json": str(output_dir / "d4_non_h_reverse_move_lookup.json"),
    }
    (output_dir / "command_used.txt").write_text(shell_command + "\n", encoding="utf-8")
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Running: {command_text}")
    print(f"Output dir: {output_dir}")
    print(f"Command record: {output_dir / 'command_used.txt'}")

    log_path = output_dir / "builder.log"
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
    print(f"Builder log: {log_path}")
    print(f"Exit code: {process.returncode}")
    if process.returncode != 0:
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
