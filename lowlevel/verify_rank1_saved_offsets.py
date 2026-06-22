import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pybullet as p

from multisim_chess_fast import PLACE_OFFSET, run_sim_move, setup_sim_world


INPUT_PATH = Path(__file__).resolve().parent / "rank1_grasp_offset_walk_results.json"
VIDEO_LABEL = "rank1_saved_offsets_home_start"
SUMMARY_PATH = (
    Path(__file__).resolve().parent
    / "recordings"
    / "multisim_place_lookup"
    / "verify"
    / "rank1_saved_offsets_home_start_summary.json"
)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main():
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(INPUT_PATH),
        "video_label": VIDEO_LABEL,
        "results": [],
    }

    try:
        for entry in data.get("results", []):
            from_square = entry["from_square"]
            to_square = entry["to_square"]
            grasp_offset = np.array(entry["selected_grasp_offset"], dtype=float)
            print(
                f"\nVerifying saved offset {from_square}->{to_square} "
                f"grasp={grasp_offset}"
            )

            world = setup_sim_world(
                from_square,
                edge_support_margin=0.08,
            )
            try:
                result = run_sim_move(
                    world,
                    from_square,
                    to_square,
                    grasp_offset,
                    place_offset=PLACE_OFFSET.copy(),
                    return_metrics=True,
                    record_video=True,
                    video_label=VIDEO_LABEL,
                    move_steps_per_waypoint=50,
                    placement_lower_steps=2,
                )
            finally:
                p.removeState(world["state_id"])

            row = {
                "from_square": from_square,
                "to_square": to_square,
                "grasp_offset": grasp_offset,
                "pickup_success": bool(result.get("pickup_success", False)),
                "premature_drop": bool(result.get("premature_drop", False)),
                "reject_reason": result.get("reject_reason"),
                "xy_error": result.get("xy_error"),
                "final_tilt_deg": result.get("final_tilt_deg"),
                "video_output_dir": result.get("video_output_dir"),
            }
            summary["results"].append(row)
            print(
                f"{from_square}->{to_square}: "
                f"pickup={row['pickup_success']} | "
                f"premature={row['premature_drop']} | "
                f"xy={row['xy_error']} | "
                f"tilt={row['final_tilt_deg']} | "
                f"video={row['video_output_dir']}"
            )
    finally:
        if p.isConnected():
            p.disconnect()

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
