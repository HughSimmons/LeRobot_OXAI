import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as p

import build_general_nonh_reverse_lookup as builder
import multisim_chess_fast as sim


DEFAULT_MOVES = (
    "d6_to_a5",
    "d6_to_a6",
    "d6_to_a7",
    "e7_to_a5",
    "e6_to_a5",
    "e5_to_a6",
    "f1_to_a4",
    "f1_to_b6",
    "f1_to_b7",
    "f1_to_c8",
)


def parse_move_key(move_key):
    parts = move_key.split("_to_")
    if len(parts) != 2:
        raise ValueError(f"Invalid move key {move_key!r}; use e.g. f1_to_a4")
    return parts[0], parts[1]


def load_lookup_entry(lookup_dir, move_key):
    from_square, _ = parse_move_key(move_key)
    lookup_path = lookup_dir / f"{from_square}_non_h_reverse_move_lookup.json"
    lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
    entry = lookup.get("moves", {}).get(move_key)
    if entry is None:
        raise KeyError(f"{move_key} not found in {lookup_path}")
    return lookup_path, entry


def run_saved_entry(move_key, entry, output_dir):
    from_square, to_square = parse_move_key(move_key)
    metrics = entry.get("metrics", {})
    grasp_offset = np.array(entry["source_grasp_offset"], dtype=float)
    place_offset = np.array(entry["selected_place_offset"], dtype=float)
    move_steps = int(
        metrics.get("move_steps_per_waypoint")
        or builder.move_steps_per_waypoint_for_lookup(from_square, to_square)
    )
    placement_lower_steps = int(
        metrics.get("placement_lower_steps")
        or builder.placement_lower_steps_for_lookup(to_square)
    )

    trajectory_override = None
    saved_donor = metrics.get("trajectory_fallback_source")
    replay_mode = "direct"
    if saved_donor:
        trajectory_override, reject_reason = builder.build_bridge_override_for_move(
            from_square,
            to_square,
            grasp_offset,
            place_offset,
        )
        if trajectory_override is None:
            raise RuntimeError(f"{move_key}: could not rebuild donor bridge: {reject_reason}")
        replay_mode = "donor_bridge"

    world = builder.setup_sim_world(
        from_square,
        edge_support_margin=builder.LOOKUP_EDGE_SUPPORT_MARGIN,
    )
    video_context = sim.create_video_context(output_dir / move_key)
    try:
        result = sim.run_sim_move(
            world,
            from_square,
            to_square,
            grasp_offset,
            place_offset=place_offset,
            return_metrics=True,
            record_video=True,
            trajectory_override=trajectory_override,
            move_steps_per_waypoint=move_steps,
            placement_lower_steps=placement_lower_steps,
            video_context=video_context,
        )
        result["score"] = sim.score_place_result(result)
        if trajectory_override is not None:
            result = builder.annotate_bridge_result(result, trajectory_override)
    finally:
        sim.close_video_context(video_context)
        p.removeState(world["state_id"])

    return {
        "move_key": move_key,
        "replay_mode": replay_mode,
        "saved_donor": saved_donor,
        "replayed_donor": result.get("trajectory_fallback_source"),
        "verified_success": builder.direct_result_is_suitable(result),
        "reject_reason": result.get("reject_reason"),
        "pickup_success": bool(result.get("pickup_success")),
        "premature_drop": bool(result.get("premature_drop")),
        "trajectory_fk_error": float(result.get("trajectory_fk_error", np.nan)),
        "xy_error": float(result.get("xy_error", np.nan)),
        "z_error": float(result.get("z_error", np.nan)),
        "final_tilt_deg": float(result.get("final_tilt_deg", np.nan)),
        "move_steps_per_waypoint": move_steps,
        "placement_lower_steps": placement_lower_steps,
        "video_output_dir": str(output_dir / move_key),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Replay saved non-h reverse lookup moves and save inspection videos."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Folder to receive one video subfolder per move plus summary.json.",
    )
    parser.add_argument(
        "--lookup-dir",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Folder containing <source>_non_h_reverse_move_lookup.json files.",
    )
    parser.add_argument(
        "moves",
        nargs="*",
        default=DEFAULT_MOVES,
        help="Move keys to verify, e.g. d6_to_a5 f1_to_a4. Defaults to old failures.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    lookup_dir = args.lookup_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    builder.ensure_physics_connected()

    summary = {
        "lookup_dir": str(lookup_dir),
        "output_dir": str(output_dir),
        "moves": [],
    }
    for move_key in args.moves:
        lookup_path, entry = load_lookup_entry(lookup_dir, move_key)
        print(f"\n=== verifying {move_key} from {lookup_path} ===", flush=True)
        row = run_saved_entry(move_key, entry, output_dir)
        row["lookup_path"] = str(lookup_path)
        summary["moves"].append(row)
        print(
            f"{move_key}: success={row['verified_success']} "
            f"mode={row['replay_mode']} donor={row['replayed_donor']} "
            f"fk={row['trajectory_fk_error']:.6f} "
            f"xy={row['xy_error']:.6f} tilt={row['final_tilt_deg']:.3f} "
            f"video={row['video_output_dir']}",
            flush=True,
        )

    summary["verified_count"] = sum(row["verified_success"] for row in summary["moves"])
    summary["total_count"] = len(summary["moves"])
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(builder.json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nsummary: {summary_path}")
    print(f"verified {summary['verified_count']}/{summary['total_count']}")


if __name__ == "__main__":
    main()
