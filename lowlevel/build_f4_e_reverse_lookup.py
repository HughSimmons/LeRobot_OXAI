import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pybullet as p

from multisim_chess_fast import (
    FINAL_TILT_TARGET_DEG,
    MAX_TRAJECTORY_FK_ERROR,
    PLACE_OFFSET,
    XY_CORRECTION_TARGET_ERROR,
    board_origin,
    ensure_physics_connected,
    get_release_gripper_rotation,
    run_sim_move,
    score_place_result,
    setup_sim_world,
)


LOOKUP_FROM_SQUARE = "f4"
LOOKUP_TO_FILES = ("e",)
LOOKUP_TO_RANKS = range(1, 9)
LOOKUP_MOVES = tuple(
    (LOOKUP_FROM_SQUARE, f"{file}{rank}")
    for file in LOOKUP_TO_FILES
    for rank in LOOKUP_TO_RANKS
)
DIRECT_GRASP_OFFSET_FOR_LOOKUP = np.array([-0.011, 0.002, -0.003])
DIRECT_PLACE_CORRECTION_ROUNDS = 8

OUTPUT_PATH = Path(__file__).resolve().parent / "f4_e_reverse_move_lookup.json"


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
        "move_steps_per_waypoint",
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
        "source": "lowlevel/build_f4_e_reverse_lookup.py",
        "trajectory_mode": "direct",
        "board_origin": json_safe(np.array(board_origin)),
        "fk_error_threshold": MAX_TRAJECTORY_FK_ERROR,
        "final_tilt_target_deg": FINAL_TILT_TARGET_DEG,
        "xy_correction_target_error": XY_CORRECTION_TARGET_ERROR,
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


def apply_direct_place_correction(from_square, to_square, result, grasp_offset, place_offset):
    world_correction = result["position_error"].copy()
    world_correction[2] = 0.0
    release_rot, _ = get_release_gripper_rotation(
        from_square,
        to_square,
        grasp_offset,
        place_offset
    )
    gripper_frame_correction = release_rot.T @ world_correction
    return place_offset + gripper_frame_correction


def run_direct_move_once(
    world,
    from_square,
    to_square,
    grasp_offset,
    place_offset,
    record_video=False,
    video_label=None,
):
    result = run_sim_move(
        world,
        from_square,
        to_square,
        grasp_offset,
        place_offset=place_offset,
        return_metrics=True,
        record_video=record_video,
        video_label=video_label,
    )
    result["score"] = score_place_result(result)
    return result


def direct_result_is_suitable(result):
    if result is None:
        return False
    return (
        result.get("reject_reason") is None
        and bool(result.get("pickup_success", False))
        and np.isfinite(float(result.get("xy_error", np.inf)))
        and np.isfinite(float(result.get("final_tilt_deg", np.inf)))
        and float(result["xy_error"]) < XY_CORRECTION_TARGET_ERROR
        and float(result["final_tilt_deg"]) < FINAL_TILT_TARGET_DEG
    )


def calibrate_direct_move(from_square, to_square):
    world = setup_sim_world(from_square)
    grasp_offset = DIRECT_GRASP_OFFSET_FOR_LOOKUP.copy()
    place_offset = PLACE_OFFSET.copy()
    selected_result = None
    selected_place_offset = place_offset.copy()

    try:
        for correction_round in range(0, DIRECT_PLACE_CORRECTION_ROUNDS + 1):
            record_final_candidate = correction_round > 0
            result = run_direct_move_once(
                world,
                from_square,
                to_square,
                grasp_offset,
                place_offset,
                record_video=False,
                video_label=(
                    "initial_direct"
                    if correction_round == 0
                    else f"placement_corrected_round_{correction_round}"
                ),
            )
            result["correction_round"] = correction_round
            print(
                f"direct_round={correction_round} | "
                f"place_offset={place_offset} | "
                f"xy={result['xy_error']} | "
                f"tilt={result['final_tilt_deg']} | "
                f"reject={result['reject_reason']}"
            )

            if direct_result_is_suitable(result):
                selected_place_offset = place_offset.copy()
                selected_result = run_direct_move_once(
                    world,
                    from_square,
                    to_square,
                    grasp_offset,
                    selected_place_offset,
                    record_video=False,
                    video_label="final_corrected_lookup",
                )
                break

            if result.get("reject_reason") is not None:
                break

            position_error = np.array(result.get("position_error", np.full(3, np.nan)))
            if not np.all(np.isfinite(position_error)):
                break

            place_offset = apply_direct_place_correction(
                from_square,
                to_square,
                result,
                grasp_offset,
                place_offset
            )
    finally:
        p.removeState(world["state_id"])

    success = direct_result_is_suitable(selected_result)
    return {
        "selected_result": selected_result,
        "selected_place_offset": selected_place_offset.copy(),
        "selected_release_wrist_delta_deg": np.zeros(2),
        "selected_grasp_wrist_delta_deg": np.zeros(2),
        "source_grasp_offset": grasp_offset.copy(),
        "reversed_grasp_offset": grasp_offset.copy(),
        "video_output_dir": (
            selected_result.get("video_output_dir")
            if selected_result is not None else None
        ),
        "success": success,
        "reject_reason": None if success else "none_suitable_found",
    }


def build_lookup():
    ensure_physics_connected()

    lookup = load_existing_lookup()
    lookup["metadata"] = build_metadata()
    lookup.setdefault("moves", {})

    for from_square, to_square in LOOKUP_MOVES:
        move_key = f"{from_square}_to_{to_square}"
        calibration = calibrate_direct_move(from_square, to_square)
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
