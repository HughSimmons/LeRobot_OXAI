#!/usr/bin/env python3
"""Replay a curated set of continuous XY lookup JSONs without videos."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pybullet as p

from build_general_xy_lookup import json_safe, result_is_suitable
from verify_xy_lookup_move import point_from_saved
from build_general_xy_lookup import (
    build_continuous_donor_bridge_override,
    continuous_donor_candidates,
)
import multisim_chess_fast as sim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay continuous XY lookup grid entries.")
    parser.add_argument("lookup_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def lookup_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*.json")):
        if path.name == "summary.json" or path.name.endswith(".verified.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema") == "continuous_xy_lookup_v1":
            files.append(path)
    return files


def replay(payload: dict) -> dict:
    start = point_from_saved(payload["from"])
    target = point_from_saved(payload["to"])
    grasp_offset = np.array(payload["source_grasp_offset"], dtype=float)
    place_offset = np.array(payload["selected_place_offset"], dtype=float)
    search = payload["search"]
    saved_metrics = payload.get("metrics", {})
    lower_place_path_bias = saved_metrics.get("lower_place_path_bias")
    trajectory_override = None
    strategy = "direct"
    bridge = saved_metrics.get("continuous_donor_bridge")
    if isinstance(bridge, dict) and bridge.get("enabled"):
        database = Path(payload["candidate_database"]["path"])
        donors = continuous_donor_candidates(database, start=start, target=target)
        donor = next((item for item in donors if item["id"] == bridge.get("donor_candidate_id")), None)
        if donor is None:
            raise ValueError("saved continuous donor is unavailable")
        trajectory_override, reason = build_continuous_donor_bridge_override(
            start, target, grasp_offset, place_offset, donor,
            lift_height=float(search["lift_height"]),
            placement_lower_steps=int(search["placement_lower_steps"]),
        )
        if trajectory_override is None:
            raise ValueError(f"saved continuous donor bridge cannot rebuild: {reason}")
        strategy = "continuous_donor_bridge"
    elif lower_place_path_bias is not None:
        lower_place_path_bias = np.array(lower_place_path_bias, dtype=float)
        strategy = "continuous_neighbor_lower_place_bias"

    world = sim.setup_sim_world(start, edge_support_margin=0.08)
    try:
        result = sim.run_sim_move(
            world, start, target, grasp_offset, place_offset=place_offset,
            return_metrics=True, record_video=False,
            move_steps_per_waypoint=int(search["move_steps"]),
            placement_lower_steps=int(search["placement_lower_steps"]),
            lift_height=float(search["lift_height"]),
            lower_place_path_bias=lower_place_path_bias,
            trajectory_override=trajectory_override,
        )
        result["score"] = sim.score_place_result(result)
        return {"verified_success": result_is_suitable(result), "strategy": strategy, "result": result}
    finally:
        p.removeState(world["state_id"])


def main() -> int:
    args = parse_args()
    root = args.lookup_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    files = lookup_files(root)
    if not files:
        raise ValueError(f"No continuous lookup JSONs found below {root}")

    sim.ensure_physics_connected()
    entries = []
    for index, path in enumerate(files, start=1):
        payload = json.loads(path.read_text(encoding="utf-8"))
        try:
            replay_result = replay(payload)
            entry = {
                "lookup_json": str(path), "move_id": payload["move_id"],
                **replay_result,
            }
        except Exception as exc:  # Keep a complete grid report when one replay is malformed.
            entry = {"lookup_json": str(path), "move_id": payload.get("move_id"),
                     "verified_success": False, "exception": str(exc)}
        entries.append(entry)
        print(f"[{index}/{len(files)}] {entry['move_id']}: {entry['verified_success']}", flush=True)

    summary = {
        "schema": "continuous_xy_grid_verification_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookup_root": str(root),
        "total": len(entries),
        "verified_successes": sum(bool(entry["verified_success"]) for entry in entries),
        "entries": entries,
    }
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Summary: {output_path}")
    return 0 if summary["verified_successes"] == len(entries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
