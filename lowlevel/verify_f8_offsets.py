import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pybullet as p

from multisim_chess_fast import (
    FINAL_TILT_TARGET_DEG,
    RECORDINGS_DIR,
    close_video_context,
    create_video_context,
    ensure_physics_connected,
    run_sim_move,
    runid,
    score_place_result,
    setup_sim_world,
)


LOOKUP_PATH = Path(
    os.environ.get(
        "VERIFY_LOOKUP_PATH",
        Path(__file__).resolve().parent / "f8_non_h_reverse_move_lookup.json",
    )
)
VERIFY_OUTPUT_DIR = Path(
    os.environ.get(
        "VERIFY_OUTPUT_DIR",
        RECORDINGS_DIR / runid / "verify" / "f8_offsets",
    )
)
SUMMARY_PATH = VERIFY_OUTPUT_DIR / "verification_summary.json"
LOOKUP_EDGE_SUPPORT_MARGIN = 0.08
XY_SUCCESS_THRESHOLD = 0.001


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, int):
        return value
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]
    return str(value)


def load_lookup():
    with LOOKUP_PATH.open("r", encoding="utf-8") as lookup_file:
        lookup = json.load(lookup_file)

    moves = lookup.get("moves", {})
    if not isinstance(moves, dict):
        raise ValueError(f"Lookup has invalid moves object: {LOOKUP_PATH}")
    return moves


def verify_move(move_key, move):
    from_square = move["from_square"]
    to_square = move["to_square"]
    grasp_offset = np.array(move["source_grasp_offset"], dtype=float)
    place_offset = np.array(move["selected_place_offset"], dtype=float)
    metrics = move.get("metrics", {})
    move_steps_per_waypoint = metrics.get("move_steps_per_waypoint")
    placement_lower_steps = metrics.get("placement_lower_steps", 10)

    output_dir = VERIFY_OUTPUT_DIR / move_key
    world = setup_sim_world(
        from_square,
        edge_support_margin=LOOKUP_EDGE_SUPPORT_MARGIN,
    )
    video_context = create_video_context(output_dir)
    try:
        result = run_sim_move(
            world,
            from_square,
            to_square,
            grasp_offset,
            place_offset=place_offset,
            return_metrics=True,
            record_video=True,
            video_context=video_context,
            move_steps_per_waypoint=move_steps_per_waypoint,
            placement_lower_steps=placement_lower_steps,
        )
        result["score"] = score_place_result(result)
    finally:
        close_video_context(video_context)
        p.removeState(world["state_id"])

    success = (
        result.get("reject_reason") is None
        and bool(result.get("pickup_success", False))
        and np.isfinite(float(result.get("xy_error", np.inf)))
        and np.isfinite(float(result.get("final_tilt_deg", np.inf)))
        and float(result["xy_error"]) < XY_SUCCESS_THRESHOLD
        and float(result["final_tilt_deg"]) < FINAL_TILT_TARGET_DEG
    )
    return {
        "move_key": move_key,
        "from_square": from_square,
        "to_square": to_square,
        "success": success,
        "pickup_success": bool(result.get("pickup_success", False)),
        "premature_drop": bool(result.get("premature_drop", False)),
        "reject_reason": result.get("reject_reason"),
        "xy_error": json_safe(result.get("xy_error")),
        "z_error": json_safe(result.get("z_error")),
        "final_tilt_deg": json_safe(result.get("final_tilt_deg")),
        "trajectory_fk_error": json_safe(result.get("trajectory_fk_error")),
        "score": json_safe(result.get("score")),
        "grasp_offset": json_safe(grasp_offset),
        "place_offset": json_safe(place_offset),
        "move_steps_per_waypoint": json_safe(result.get("move_steps_per_waypoint")),
        "placement_lower_steps": json_safe(result.get("placement_lower_steps")),
        "video_output_dir": str(output_dir),
        "video_files": [
            str(output_dir / "so101_robot_moves.mp4"),
            str(output_dir / "so101_robot_moves_topdown.mp4"),
        ],
    }


def main():
    ensure_physics_connected()
    moves = load_lookup()
    VERIFY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    try:
        for move_key, move in sorted(moves.items()):
            print(f"\nVerifying {move_key}")
            result = verify_move(move_key, move)
            results.append(result)
            print(
                f"{move_key}: success={result['success']} | "
                f"pickup={result['pickup_success']} | "
                f"xy={result['xy_error']} | "
                f"tilt={result['final_tilt_deg']} | "
                f"reject={result['reject_reason']} | "
                f"video={result['video_output_dir']}"
            )
    finally:
        if p.isConnected():
            p.disconnect()

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookup_path": str(LOOKUP_PATH),
        "verify_output_dir": str(VERIFY_OUTPUT_DIR),
        "move_count": len(results),
        "success_count": sum(1 for result in results if result["success"]),
        "pickup_success_count": sum(
            1 for result in results
            if result["pickup_success"]
        ),
        "results": results,
    }
    SUMMARY_PATH.write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    print(f"\nWrote verification summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
