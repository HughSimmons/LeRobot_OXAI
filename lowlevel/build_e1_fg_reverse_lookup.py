import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pybullet as p

from multisim_chess_fast import (
    FINAL_TILT_TARGET_DEG,
    MAX_TRAJECTORY_FK_ERROR,
    REVERSED_RELEASE_Z_OFFSET,
    XY_CORRECTION_TARGET_ERROR,
    XY_SELECTION_SIMILAR_ERROR_BAND,
    board_origin,
    calibrate_verified_reverse_move,
    ensure_physics_connected,
)


LOOKUP_FROM_SQUARE = "e1"
LOOKUP_TO_FILES = ("f", "g")
LOOKUP_TO_RANKS = range(1, 9)
# LOOKUP_MOVES = tuple(
#     (LOOKUP_FROM_SQUARE, f"{file}{rank}")
#     for file in LOOKUP_TO_FILES
#     for rank in LOOKUP_TO_RANKS
# )
LOOKUP_MOVES = (

    ("e1", "f6"),
    ("e1", "f7"),
    ("e1", "f8"),
)

# print(LOOKUP_MOVES)
# sys.exit()

OUTPUT_PATH = Path(__file__).resolve().parent / "e1_fg_reverse_move_lookup.json"


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


def summarize_result(result):
    if result is None:
        return None

    fields = (
        "score",
        "xy_error",
        "z_error",
        "final_tilt_deg",
        "final_euler_deg",
        "trajectory_fk_error",
        "pickup_success",
        "premature_drop",
        "reject_reason",
        "place_offset",
        "release_wrist_delta_deg",
        "grasp_wrist_delta_deg",
        "video_output_dir",
    )
    return {
        field: json_safe(result.get(field))
        for field in fields
        if field in result
    }


def summarize_successful_calibration(from_square, to_square, calibration):
    selected_result = calibration["selected_result"]
    metrics = summarize_result(selected_result)

    return {
        "from_square": from_square,
        "to_square": to_square,
        "success": True,
        "selected_place_offset": json_safe(calibration["selected_place_offset"]),
        "selected_release_wrist_delta_deg": json_safe(
            calibration["selected_release_wrist_delta_deg"]
        ),
        "selected_grasp_wrist_delta_deg": json_safe(
            calibration["selected_grasp_wrist_delta_deg"]
        ),
        "source_grasp_offset": json_safe(calibration["source_grasp_offset"]),
        "reversed_grasp_offset": json_safe(calibration["reversed_grasp_offset"]),
        "metrics": metrics,
        "video_output_dir": json_safe(calibration["video_output_dir"]),
    }


def move_meets_current_targets(move):
    if not isinstance(move, dict) or not move.get("success"):
        return False

    metrics = move.get("metrics")
    if not isinstance(metrics, dict):
        return False

    xy_error = metrics.get("xy_error", np.inf)
    final_tilt_deg = metrics.get("final_tilt_deg", np.inf)
    try:
        xy_error = float(xy_error)
        final_tilt_deg = float(final_tilt_deg)
    except (TypeError, ValueError):
        return False

    return (
        np.isfinite(xy_error)
        and np.isfinite(final_tilt_deg)
        and xy_error < XY_CORRECTION_TARGET_ERROR
        and final_tilt_deg < FINAL_TILT_TARGET_DEG
    )


def compact_existing_move(move):
    fields = (
        "from_square",
        "to_square",
        "success",
        "selected_place_offset",
        "selected_release_wrist_delta_deg",
        "selected_grasp_wrist_delta_deg",
        "source_grasp_offset",
        "reversed_grasp_offset",
        "metrics",
        "video_output_dir",
    )
    return {
        field: move[field]
        for field in fields
        if field in move
    }


def build_metadata():
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "lowlevel/build_e1_fg_reverse_lookup.py",
        "board_origin": json_safe(np.array(board_origin)),
        "release_z_offset": REVERSED_RELEASE_Z_OFFSET,
        "fk_error_threshold": MAX_TRAJECTORY_FK_ERROR,
        "final_tilt_target_deg": FINAL_TILT_TARGET_DEG,
        "xy_correction_target_error": XY_CORRECTION_TARGET_ERROR,
        "xy_selection_similar_error_band": XY_SELECTION_SIMILAR_ERROR_BAND,
    }


def load_existing_lookup():
    if not OUTPUT_PATH.exists():
        return {"metadata": {}, "moves": {}}

    try:
        with OUTPUT_PATH.open("r", encoding="utf-8") as lookup_file:
            lookup = json.load(lookup_file)
    except json.JSONDecodeError:
        print(f"Existing lookup is not valid JSON; rebuilding: {OUTPUT_PATH}")
        return {"metadata": {}, "moves": {}}

    if not isinstance(lookup, dict):
        return {"metadata": {}, "moves": {}}
    if not isinstance(lookup.get("metadata"), dict):
        lookup["metadata"] = {}
    if not isinstance(lookup.get("moves"), dict):
        lookup["moves"] = {}
    lookup["moves"] = {
        move_key: compact_existing_move(move)
        for move_key, move in lookup["moves"].items()
        if move_meets_current_targets(move)
    }
    return lookup


def build_lookup():
    ensure_physics_connected()

    lookup = load_existing_lookup()
    lookup["metadata"] = build_metadata()
    lookup.setdefault("moves", {})

    for from_square, to_square in LOOKUP_MOVES:
        move_key = f"{from_square}_to_{to_square}"
        calibration = calibrate_verified_reverse_move(
            from_square,
            to_square,
            record_initial_video=False,
            record_intermediate_video=False,
            record_final_video=True,
            tilt_correction_enabled=True,
            xy_correction_enabled=True,
        )
        if calibration["success"]:
            lookup["moves"][move_key] = summarize_successful_calibration(
                from_square,
                to_square,
                calibration
            )
        else:
            lookup["moves"].pop(move_key, None)
            print(f"{move_key}: None suitable found.")

    return lookup


def main():
    try:
        lookup = build_lookup()
        OUTPUT_PATH.write_text(
            json.dumps(json_safe(lookup), indent=2, sort_keys=True) + "\n",
            encoding="utf-8"
        )
        print(f"\nSaved lookup: {OUTPUT_PATH}")
        for move_key, move in lookup["moves"].items():
            metrics = move["metrics"]
            print(
                f"{move_key}: success={move['success']} | "
                f"xy_error={metrics.get('xy_error')} | "
                f"tilt={metrics.get('final_tilt_deg')} | "
                f"fk={metrics.get('trajectory_fk_error')} | "
                f"reject={metrics.get('reject_reason')}"
            )
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
