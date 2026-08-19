#!/usr/bin/env python3
"""Replay one saved continuous XY lookup entry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as p

from board_coordinates import XYPoint
from build_general_xy_lookup import (
    build_continuous_donor_bridge_override,
    continuous_donor_candidates,
    json_safe,
    result_is_suitable,
)
import multisim_chess_fast as sim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a continuous XY lookup JSON.")
    parser.add_argument("lookup_json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    return parser.parse_args()


def point_from_saved(value: dict[str, object]) -> XYPoint:
    if value.get("type") != "continuous_xy" or value.get("frame") != "world":
        raise ValueError(f"Unsupported saved point: {value}")
    return XYPoint(float(value["x"]), float(value["y"]), name=value.get("name"))


def main() -> int:
    args = parse_args()
    lookup_path = args.lookup_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
    if lookup.get("schema") != "continuous_xy_lookup_v1":
        raise ValueError(f"Unsupported lookup schema in {lookup_path}")

    start = point_from_saved(lookup["from"])
    target = point_from_saved(lookup["to"])
    grasp_offset = np.array(lookup["source_grasp_offset"], dtype=float)
    place_offset = np.array(lookup["selected_place_offset"], dtype=float)
    search = lookup["search"]
    candidate_db_path = Path(lookup.get("candidate_database", {}).get("path", ""))
    saved_metrics = lookup.get("metrics", {})
    lower_place_path_bias = saved_metrics.get("lower_place_path_bias")
    trajectory_override = None
    replay_strategy = "direct"
    bridge_metadata = saved_metrics.get("continuous_donor_bridge")
    if isinstance(bridge_metadata, dict) and bridge_metadata.get("enabled"):
        donor_id = bridge_metadata.get("donor_candidate_id")
        donors = continuous_donor_candidates(candidate_db_path, start=start, target=target)
        donor = next((item for item in donors if item["id"] == donor_id), None)
        if donor is None:
            raise ValueError(f"Saved continuous donor {donor_id!r} is unavailable in {candidate_db_path}")
        trajectory_override, reason = build_continuous_donor_bridge_override(
            start, target, grasp_offset, place_offset, donor,
            lift_height=float(search["lift_height"]),
            placement_lower_steps=int(search["placement_lower_steps"]),
        )
        if trajectory_override is None:
            raise ValueError(f"Could not rebuild saved continuous donor bridge: {reason}")
        replay_strategy = "continuous_donor_bridge"
    elif lower_place_path_bias is not None:
        lower_place_path_bias = np.array(lower_place_path_bias, dtype=float)
        replay_strategy = "continuous_neighbor_lower_place_bias"
    video_context = sim.create_video_context(output_dir / lookup["move_id"]) if args.video else None

    sim.ensure_physics_connected()
    world = sim.setup_sim_world(start, edge_support_margin=0.08)
    try:
        result = sim.run_sim_move(
            world,
            start,
            target,
            grasp_offset,
            place_offset=place_offset,
            return_metrics=True,
            record_video=args.video,
            video_context=video_context,
            move_steps_per_waypoint=int(search["move_steps"]),
            placement_lower_steps=int(search["placement_lower_steps"]),
            lift_height=float(search["lift_height"]),
            lower_place_path_bias=lower_place_path_bias,
            trajectory_override=trajectory_override,
        )
        result["score"] = sim.score_place_result(result)
        verified_success = result_is_suitable(result)
    finally:
        if video_context is not None:
            sim.close_video_context(video_context)
        p.removeState(world["state_id"])

    summary = {
        "schema": "continuous_xy_verification_v1",
        "lookup_json": str(lookup_path),
        "move_id": lookup["move_id"],
        "verified_success": verified_success,
        "video_recorded": args.video,
        "replay_strategy": replay_strategy,
        "result": result,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"{lookup['move_id']}: success={verified_success} "
        f"fk={result.get('trajectory_fk_error')} "
        f"xy={result.get('xy_error')} tilt={result.get('final_tilt_deg')}"
    )
    print(f"Summary: {summary_path}")
    return 0 if verified_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
