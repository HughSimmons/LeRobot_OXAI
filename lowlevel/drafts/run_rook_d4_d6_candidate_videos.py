#!/usr/bin/env python3
"""Record inspection videos for selected rook d4_to_d6 grasp candidates."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
LOWLEVEL_DIR = PROJECT_DIR / "lowlevel"
BEST_MESH = (
    PROJECT_DIR
    / "rook_kiri2"
    / "rook2.obj"
)
BEST_VISUAL_MESH = BEST_MESH.with_name(BEST_MESH.stem + "_debug_orange_visual.obj")
BEST_BAND_COLLISION_DIR = (
    LOWLEVEL_DIR
    / "rook_kiri_lookup"
    / "collision_geometry_preview_20260801_163534"
)
OUTPUT_ROOT = LOWLEVEL_DIR / "rook_kiri_lookup" / "d4_to_d6_candidate_videos"

# Set the rook experiment environment before importing builder/sim modules.
os.environ.setdefault("LOOKUP_PIECE_MODEL", "rook_kiri")
os.environ.setdefault("ROOK_KIRI_MESH_PATH", str(BEST_MESH))
os.environ.setdefault("ROOK_KIRI_VISUAL_MESH_PATH", str(BEST_VISUAL_MESH))
os.environ.setdefault("ROOK_KIRI_MESH_UP_AXIS", "y")
os.environ.setdefault("ROOK_KIRI_COLLISION_MODEL", "banded_hulls")
os.environ.setdefault("ROOK_KIRI_BAND_COLLISION_MESH_DIR", str(BEST_BAND_COLLISION_DIR))
os.environ.setdefault("SOURCE_SQUARES", "d4")
os.environ.setdefault("TARGET_MOVES", "d4_to_d6")
os.environ.setdefault("LOOKUP_OUTPUT_DIR", str(OUTPUT_ROOT / "unused_lookup_output"))
os.environ.setdefault("DONOR_LOOKUP_DIR", str(OUTPUT_ROOT / "empty_donor_lookup"))

if str(LOWLEVEL_DIR) not in sys.path:
    sys.path.insert(0, str(LOWLEVEL_DIR))

import build_general_nonh_reverse_lookup as builder  # noqa: E402
import multisim_chess_fast as sim  # noqa: E402


CANDIDATES = (
    {
        "label": "default",
        "grasp_offset": [-0.014, 0.002, -0.003],
        "why": "configured default grasp from the builder run",
    },
    {
        "label": "best_xy_grid_dx0_dy2_dz0",
        "grasp_offset": [-0.014, 0.008, -0.003],
        "why": "best XY result in the failed builder log",
    },
    {
        "label": "lower_z_grid_dx0_dy0_dz_minus1",
        "grasp_offset": [-0.014, 0.002, -0.005],
        "why": "same XY grasp, lower vertical grasp candidate",
    },
    {
        "label": "higher_z_grid_dx0_dy0_dz_plus1",
        "grasp_offset": [-0.014, 0.002, -0.001],
        "why": "same XY grasp, higher vertical grasp candidate",
    },
)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def main():
    if not BEST_MESH.exists():
        raise FileNotFoundError(f"Best rook mesh not found: {BEST_MESH}")
    if not BEST_VISUAL_MESH.exists():
        raise FileNotFoundError(f"Best rook visual mesh not found: {BEST_VISUAL_MESH}")
    if not BEST_BAND_COLLISION_DIR.exists():
        raise FileNotFoundError(f"Best rook band collision dir not found: {BEST_BAND_COLLISION_DIR}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    (OUTPUT_ROOT / "empty_donor_lookup").mkdir(parents=True, exist_ok=True)

    # Redirect multisim's video root for this inspection batch only.
    sim.RECORDINGS_DIR = output_dir
    sim.runid = "videos"

    place_offset = np.array(builder.PLACE_OFFSET, dtype=float)
    lift_height = builder.lookup_lift_height_for_square("d6")
    placement_lower_steps = builder.placement_lower_steps_for_lookup("d6")
    home_joints = builder.lookup_home_joints_for_source("d4")

    summary = {
        "purpose": "Inspection videos for selected failed rook d4_to_d6 pickup candidates.",
        "timestamp": stamp,
        "piece_model": sim.PIECE_MODEL,
        "rook_mesh_path": str(BEST_MESH),
        "rook_visual_mesh_path": str(BEST_VISUAL_MESH),
        "rook_mesh_up_axis": os.environ.get("ROOK_KIRI_MESH_UP_AXIS"),
        "rook_collision_model": os.environ.get("ROOK_KIRI_COLLISION_MODEL", "cylinder_proxy"),
        "rook_band_collision_mesh_dir": os.environ.get("ROOK_KIRI_BAND_COLLISION_MESH_DIR"),
        "piece_config": sim.active_piece_config(),
        "output_dir": str(output_dir),
        "source_square": "d4",
        "target_square": "d6",
        "place_offset": place_offset,
        "lift_height": lift_height,
        "placement_lower_steps": placement_lower_steps,
        "trajectory_home_joints_deg": home_joints,
        "candidates": [],
    }

    for candidate in CANDIDATES:
        label = candidate["label"]
        grasp_offset = np.array(candidate["grasp_offset"], dtype=float)
        print(f"Recording {label}: grasp={grasp_offset}")
        result = builder.run_direct_move_in_fresh_world(
            "d4",
            "d6",
            grasp_offset,
            place_offset,
            move_steps_per_waypoint=builder.DEFAULT_MOVE_STEPS_PER_WAYPOINT,
            placement_lower_steps=placement_lower_steps,
            record_video=True,
            video_label=label,
            lift_height=lift_height,
            trajectory_home_joints=home_joints,
        )
        summary["candidates"].append(
            {
                "label": label,
                "why": candidate["why"],
                "grasp_offset": grasp_offset,
                "pickup_success": result.get("pickup_success"),
                "reject_reason": result.get("reject_reason"),
                "trajectory_fk_error": result.get("trajectory_fk_error"),
                "xy_error": result.get("xy_error"),
                "final_tilt_deg": result.get("final_tilt_deg"),
                "premature_drop": result.get("premature_drop"),
                "premature_drop_step": result.get("premature_drop_step"),
                "premature_drop_z": result.get("premature_drop_z"),
                "min_pre_release_piece_z": result.get("min_pre_release_piece_z"),
                "video_output_dir": result.get("video_output_dir"),
            }
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    raise SystemExit(main())
