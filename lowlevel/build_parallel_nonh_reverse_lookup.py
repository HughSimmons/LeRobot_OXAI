import argparse
import fcntl
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pybullet as p

from visualize_lookup_success import default_output_path, render_svg

from multisim_chess_fast import (
    DEFAULT_MOVE_STEPS_PER_WAYPOINT,
    FINAL_TILT_TARGET_DEG,
    LONG_TRANSPORT_MOVE_STEPS_PER_WAYPOINT,
    MAX_TRAJECTORY_FK_ERROR,
    PIECE_DYNAMICS,
    PLACE_OFFSET,
    XY_CORRECTION_TARGET_ERROR,
    board_origin,
    ensure_physics_connected,
    get_release_gripper_rotation,
    run_sim_move,
    score_place_result,
    setup_sim_world,
)


# Default worker scope. Override at runtime with:
#   python build_parallel_nonh_reverse_lookup.py --sources f1 f2
SOURCE_SQUARES = ("f6",)
LOOKUP_TO_FILES = tuple("abcde")
LOOKUP_TO_RANKS = range(1, 9)

DEFAULT_GRASP_OFFSET_BY_SOURCE = {
    "f1": np.array([-0.014, 0.002, -0.003]),
    "f2": np.array([-0.014, 0.002, -0.003]),
    "f3": np.array([-0.014, 0.002, -0.003]),
    "f4": np.array([-0.011, 0.002, -0.003]),
    "f5": np.array([-0.014, 0.002, -0.003]),
    "f6": np.array([-0.014, 0.002, -0.003]),
    "f7": np.array([-0.014, 0.002, -0.003]),
    "f8": np.array([-0.014, 0.002, -0.003]),
}
GRASP_OFFSET_OVERRIDES_BY_SOURCE = {
    "f1": {},
    "f2": {},
    "f3": {
        "a4": np.array([-0.017, -0.001, -0.003]),
        "b3": np.array([-0.011, 0.005, -0.003]),
    },
    "f4": {},
    "f5": {
        "b2": np.array([-0.011, 0.002, -0.003]),
    },
    "f6": {},
    "f7": {},
    "f8": {},
}

DIRECT_PLACE_CORRECTION_ROUNDS = 3
FALLBACK_CORRECTION_GAIN = 0.5
FALLBACK_GRASP_GRID_ENABLED = True
FALLBACK_GRASP_GRID_XY_DELTA = 0.003
FALLBACK_GRASP_GRID_XY_STEPS = (-1, 0, 1)
USE_LONG_MOVE_STEPS_FOR_AB_DESTINATIONS = True
LONG_MOVE_STEP_DESTINATION_FILES = ("a", "b")
USE_LONG_MOVE_STEPS_FOR_DISTANT_DESTINATIONS = True
LONG_MOVE_STEP_MIN_SQUARE_DISTANCE = 5
LOOKUP_EDGE_SUPPORT_MARGIN = 0.08
RELAXED_PLACEMENT_LOWER_STEPS_BY_TO_SQUARE = {
    f"{file}{rank}": 2
    for file in ("a", "b")
    for rank in range(1, 9)
}

# User-named common output. Multiple script processes append/merge into this file.
COMMON_OUTPUT_FILENAME = "f_non_h_reverse_move_lookup.json"
SKIP_EXISTING_SUCCESSFUL_MOVES = True


OUTPUT_PATH = Path(__file__).resolve().parent / COMMON_OUTPUT_FILENAME
SUCCESS_MAP_PATH = default_output_path(OUTPUT_PATH)
LOCK_PATH = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.name}.lock")


def move_key(from_square, to_square):
    return f"{from_square}_to_{to_square}"


def configured_destination_squares():
    return tuple(
        f"{file}{rank}"
        for file in LOOKUP_TO_FILES
        for rank in LOOKUP_TO_RANKS
    )


def lookup_moves(source_squares=SOURCE_SQUARES, destination_squares=None):
    if destination_squares is None:
        destination_squares = configured_destination_squares()

    return tuple(
        (from_square, to_square)
        for from_square in source_squares
        for to_square in destination_squares
        if to_square != from_square
    )


def default_grasp_offset_for_source(from_square):
    if from_square not in DEFAULT_GRASP_OFFSET_BY_SOURCE:
        raise KeyError(f"No default grasp offset configured for source square {from_square}")
    return DEFAULT_GRASP_OFFSET_BY_SOURCE[from_square].copy()


def configured_grasp_offset_for_lookup(from_square, to_square):
    overrides = GRASP_OFFSET_OVERRIDES_BY_SOURCE.get(from_square, {})
    if to_square in overrides:
        return overrides[to_square].copy(), {
            "source": "configured_override",
            "is_override": True,
            "base_grasp_offset": default_grasp_offset_for_source(from_square),
            "selected_grasp_offset": overrides[to_square].copy(),
        }
    default_grasp_offset = default_grasp_offset_for_source(from_square)
    return default_grasp_offset, {
        "source": "default",
        "is_override": False,
        "base_grasp_offset": default_grasp_offset.copy(),
        "selected_grasp_offset": default_grasp_offset.copy(),
    }


def fallback_grasp_grid_offsets(base_grasp_offset):
    for dx in FALLBACK_GRASP_GRID_XY_STEPS:
        for dy in FALLBACK_GRASP_GRID_XY_STEPS:
            delta = np.array([
                dx * FALLBACK_GRASP_GRID_XY_DELTA,
                dy * FALLBACK_GRASP_GRID_XY_DELTA,
                0.0,
            ])
            yield {
                "grid_dx": dx,
                "grid_dy": dy,
                "grid_delta": delta,
                "grasp_offset": base_grasp_offset + delta,
            }


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
        "placement_lower_steps",
        "relaxed_lowering_move_step_multiplier",
        "slow_waypoint_indices",
        "pickup_success",
        "premature_drop",
        "reject_reason",
        "place_offset",
        "release_wrist_delta_deg",
        "grasp_wrist_delta_deg",
        "video_output_dir",
        "correction_pass",
        "correction_gain",
        "correction_round",
    )
    return {
        field: json_safe(result.get(field))
        for field in fields
        if field in result
    }


def summarize_grid_attempt(attempt):
    return {
        "grid_dx": attempt["grid_dx"],
        "grid_dy": attempt["grid_dy"],
        "grid_delta": json_safe(attempt["grid_delta"]),
        "grasp_offset": json_safe(attempt["grasp_offset"]),
        "success": bool(attempt["calibration"].get("success")),
        "reject_reason": attempt["calibration"].get("reject_reason"),
        "metrics": summarize_result(attempt["calibration"].get("selected_result")),
    }


def summarize_successful_calibration(from_square, to_square, calibration):
    selected_result = calibration["selected_result"]
    metrics = summarize_result(selected_result)
    grasp_selection = calibration["grasp_selection"]

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
        "grasp_offset_source": grasp_selection["source"],
        "uses_grasp_offset_override": bool(grasp_selection["is_override"]),
        "grasp_selection": json_safe(grasp_selection),
        "metrics": metrics,
        "video_output_dir": json_safe(calibration["video_output_dir"]),
    }


def move_meets_current_targets(move):
    if not isinstance(move, dict) or not move.get("success"):
        return False

    from_square = move.get("from_square")
    to_square = move.get("to_square")
    if not isinstance(from_square, str) or not isinstance(to_square, str):
        return False

    expected_grasp_offset, _ = configured_grasp_offset_for_lookup(from_square, to_square)
    if move.get("uses_grasp_offset_override"):
        expected_grasp_offset = np.array(move.get("source_grasp_offset"), dtype=float)

    source_grasp_offset = move.get("source_grasp_offset")
    try:
        source_grasp_offset = np.array(source_grasp_offset, dtype=float)
    except (TypeError, ValueError):
        return False

    if not np.allclose(source_grasp_offset, expected_grasp_offset):
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
        "grasp_offset_source",
        "uses_grasp_offset_override",
        "grasp_selection",
        "metrics",
        "video_output_dir",
    )
    return {
        field: move[field]
        for field in fields
        if field in move
    }


def build_metadata(source_squares=SOURCE_SQUARES):
    single_source_square = source_squares[0] if len(source_squares) == 1 else None
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "last_updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "lowlevel/build_parallel_nonh_reverse_lookup.py",
        "parallel_builder": True,
        "common_output_filename": COMMON_OUTPUT_FILENAME,
        "trajectory_mode": "direct",
        "from_square": single_source_square,
        "source_squares": list(source_squares),
        "destination_files": list(LOOKUP_TO_FILES),
        "destination_ranks": list(LOOKUP_TO_RANKS),
        "excluded_destinations": ["source_square", "h1-h8"],
        "default_move_steps_per_waypoint": DEFAULT_MOVE_STEPS_PER_WAYPOINT,
        "long_move_steps_per_waypoint": LONG_TRANSPORT_MOVE_STEPS_PER_WAYPOINT,
        "long_move_step_destination_files": list(LONG_MOVE_STEP_DESTINATION_FILES),
        "long_move_step_min_square_distance": LONG_MOVE_STEP_MIN_SQUARE_DISTANCE,
        "default_grasp_offset_by_source": json_safe(DEFAULT_GRASP_OFFSET_BY_SOURCE),
        "configured_grasp_offset_overrides_by_source": json_safe(
            GRASP_OFFSET_OVERRIDES_BY_SOURCE
        ),
        "fallback_grasp_grid_enabled": FALLBACK_GRASP_GRID_ENABLED,
        "fallback_grasp_grid_xy_delta": FALLBACK_GRASP_GRID_XY_DELTA,
        "fallback_grasp_grid_xy_steps": list(FALLBACK_GRASP_GRID_XY_STEPS),
        "fallback_correction_gain": FALLBACK_CORRECTION_GAIN,
        "lookup_edge_support_margin": LOOKUP_EDGE_SUPPORT_MARGIN,
        "default_placement_lower_steps": 10,
        "relaxed_placement_lower_steps_by_to_square": RELAXED_PLACEMENT_LOWER_STEPS_BY_TO_SQUARE,
        "piece_dynamics": json_safe(PIECE_DYNAMICS),
        "board_origin": json_safe(np.array(board_origin)),
        "fk_error_threshold": MAX_TRAJECTORY_FK_ERROR,
        "final_tilt_target_deg": FINAL_TILT_TARGET_DEG,
        "xy_correction_target_error": XY_CORRECTION_TARGET_ERROR,
    }


def load_lookup_from_path(path):
    if not path.exists():
        return {"metadata": {}, "moves": {}}

    try:
        with path.open("r", encoding="utf-8") as lookup_file:
            lookup = json.load(lookup_file)
    except json.JSONDecodeError:
        print(f"Existing lookup is not valid JSON; rebuilding: {path}")
        return {"metadata": {}, "moves": {}}

    if not isinstance(lookup, dict):
        return {"metadata": {}, "moves": {}}
    if not isinstance(lookup.get("metadata"), dict):
        lookup["metadata"] = {}
    if not isinstance(lookup.get("moves"), dict):
        lookup["moves"] = {}
    return lookup


def load_existing_lookup():
    if not OUTPUT_PATH.exists():
        return {"metadata": {}, "moves": {}}

    lookup = load_lookup_from_path(OUTPUT_PATH)
    lookup["moves"] = {
        key: compact_existing_move(move)
        for key, move in lookup["moves"].items()
        if move_meets_current_targets(move)
    }
    return lookup


@contextmanager
def common_lookup_lock():
    with LOCK_PATH.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def merge_metadata(existing_metadata, worker_source_squares):
    metadata = build_metadata(worker_source_squares)
    existing_sources = set(existing_metadata.get("source_squares", []))
    source_squares = sorted(existing_sources | set(worker_source_squares))
    metadata["source_squares"] = source_squares
    metadata["from_square"] = source_squares[0] if len(source_squares) == 1 else None
    if existing_metadata.get("generated_at"):
        metadata["generated_at"] = existing_metadata["generated_at"]
    return metadata


def load_common_lookup_locked():
    lookup = load_lookup_from_path(OUTPUT_PATH)
    lookup["metadata"] = merge_metadata(lookup.get("metadata", {}), ())
    lookup.setdefault("moves", {})
    return lookup


def common_move_meets_current_targets(key):
    with common_lookup_lock():
        lookup = load_lookup_from_path(OUTPUT_PATH)
        return move_meets_current_targets(lookup.get("moves", {}).get(key))


def merge_move_into_common_lookup(worker_source_squares, key, move):
    with common_lookup_lock():
        lookup = load_lookup_from_path(OUTPUT_PATH)
        lookup["metadata"] = merge_metadata(
            lookup.get("metadata", {}),
            worker_source_squares,
        )
        lookup.setdefault("moves", {})
        if move is None:
            lookup["moves"].pop(key, None)
        else:
            lookup["moves"][key] = move

        OUTPUT_PATH.write_text(
            json.dumps(json_safe(lookup), indent=2, sort_keys=True) + "\n",
            encoding="utf-8"
        )


def apply_direct_place_correction(
    from_square,
    to_square,
    result,
    grasp_offset,
    place_offset,
    correction_gain=1.0,
):
    world_correction = result["position_error"].copy()
    world_correction[2] = 0.0
    world_correction *= correction_gain
    release_rot, _ = get_release_gripper_rotation(
        from_square,
        to_square,
        grasp_offset,
        place_offset
    )
    gripper_frame_correction = release_rot.T @ world_correction
    return place_offset + gripper_frame_correction


def move_steps_per_waypoint_for_lookup(from_square, to_square):
    from_file_idx = ord(from_square[0]) - ord("a")
    to_file_idx = ord(to_square[0]) - ord("a")
    square_distance = (
        abs(from_file_idx - to_file_idx)
        + abs(int(from_square[1]) - int(to_square[1]))
    )
    if (
        USE_LONG_MOVE_STEPS_FOR_DISTANT_DESTINATIONS
        and square_distance >= LONG_MOVE_STEP_MIN_SQUARE_DISTANCE
    ):
        return LONG_TRANSPORT_MOVE_STEPS_PER_WAYPOINT

    if (
        USE_LONG_MOVE_STEPS_FOR_AB_DESTINATIONS
        and to_square[0] in LONG_MOVE_STEP_DESTINATION_FILES
    ):
        return LONG_TRANSPORT_MOVE_STEPS_PER_WAYPOINT
    return DEFAULT_MOVE_STEPS_PER_WAYPOINT


def placement_lower_steps_for_lookup(to_square):
    return RELAXED_PLACEMENT_LOWER_STEPS_BY_TO_SQUARE.get(to_square, 10)


def run_direct_move_once(
    world,
    from_square,
    to_square,
    grasp_offset,
    place_offset,
    move_steps_per_waypoint,
    placement_lower_steps,
    record_video=True,
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
        move_steps_per_waypoint=move_steps_per_waypoint,
        placement_lower_steps=placement_lower_steps,
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


def run_correction_pass(
    world,
    from_square,
    to_square,
    grasp_offset,
    move_steps_per_waypoint,
    placement_lower_steps,
    correction_gain,
    pass_label,
):
    place_offset = PLACE_OFFSET.copy()

    for correction_round in range(0, DIRECT_PLACE_CORRECTION_ROUNDS + 1):
        result = run_direct_move_once(
            world,
            from_square,
            to_square,
            grasp_offset,
            place_offset,
            move_steps_per_waypoint,
            placement_lower_steps,
            record_video=False,
            video_label=(
                f"{pass_label}_initial_direct"
                if correction_round == 0
                else f"{pass_label}_placement_corrected_round_{correction_round}"
            ),
        )
        result["correction_pass"] = pass_label
        result["correction_gain"] = correction_gain
        result["correction_round"] = correction_round
        print(
            f"{pass_label}_round={correction_round} | "
            f"gain={correction_gain} | "
            f"lower_steps={placement_lower_steps} | "
            f"grasp_offset={grasp_offset} | "
            f"place_offset={place_offset} | "
            f"fk={result['trajectory_fk_error']} | "
            f"xy={result['xy_error']} | "
            f"tilt={result['final_tilt_deg']} | "
            f"reject={result['reject_reason']}"
        )

        if direct_result_is_suitable(result):
            selected_result = run_direct_move_once(
                world,
                from_square,
                to_square,
                grasp_offset,
                place_offset,
                move_steps_per_waypoint,
                placement_lower_steps,
                record_video=False,
                video_label=f"{pass_label}_final_corrected_lookup",
            )
            selected_result["correction_pass"] = pass_label
            selected_result["correction_gain"] = correction_gain
            selected_result["correction_round"] = correction_round
            return {
                "selected_result": selected_result,
                "selected_place_offset": place_offset.copy(),
                "success": True,
                "stopped_early": True,
            }

        if result.get("reject_reason") is not None:
            return {
                "selected_result": None,
                "selected_place_offset": PLACE_OFFSET.copy(),
                "success": False,
                "stopped_early": True,
                "reject_reason": result.get("reject_reason"),
            }

        position_error = np.array(result.get("position_error", np.full(3, np.nan)))
        if not np.all(np.isfinite(position_error)):
            return {
                "selected_result": None,
                "selected_place_offset": PLACE_OFFSET.copy(),
                "success": False,
                "stopped_early": True,
                "reject_reason": "invalid_position_error",
            }

        place_offset = apply_direct_place_correction(
            from_square,
            to_square,
            result,
            grasp_offset,
            place_offset,
            correction_gain=correction_gain,
        )

    return {
        "selected_result": None,
        "selected_place_offset": PLACE_OFFSET.copy(),
        "success": False,
        "stopped_early": False,
        "reject_reason": "correction_exhausted",
    }


def calibrate_with_grasp_offset(from_square, to_square, grasp_offset, pass_prefix):
    world = setup_sim_world(
        from_square,
        edge_support_margin=LOOKUP_EDGE_SUPPORT_MARGIN
    )
    move_steps_per_waypoint = move_steps_per_waypoint_for_lookup(from_square, to_square)
    placement_lower_steps = placement_lower_steps_for_lookup(to_square)

    try:
        calibration = run_correction_pass(
            world,
            from_square,
            to_square,
            grasp_offset,
            move_steps_per_waypoint,
            placement_lower_steps,
            correction_gain=1.0,
            pass_label=f"{pass_prefix}_direct",
        )
        if not calibration["success"] and not calibration["stopped_early"]:
            print(
                f"{from_square}->{to_square}: {pass_prefix} correction exhausted; "
                f"retrying with gain={FALLBACK_CORRECTION_GAIN}"
            )
            calibration = run_correction_pass(
                world,
                from_square,
                to_square,
                grasp_offset,
                move_steps_per_waypoint,
                placement_lower_steps,
                correction_gain=FALLBACK_CORRECTION_GAIN,
                pass_label=f"{pass_prefix}_damped",
            )
    finally:
        p.removeState(world["state_id"])

    selected_result = calibration["selected_result"]
    selected_place_offset = calibration["selected_place_offset"]
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
        "reject_reason": None if success else calibration.get("reject_reason", "none_suitable_found"),
    }


def calibrate_direct_move(from_square, to_square):
    grasp_offset, grasp_selection = configured_grasp_offset_for_lookup(
        from_square,
        to_square
    )
    calibration = calibrate_with_grasp_offset(
        from_square,
        to_square,
        grasp_offset,
        pass_prefix="default",
    )
    calibration["grasp_selection"] = grasp_selection

    if calibration["success"] or not FALLBACK_GRASP_GRID_ENABLED:
        return calibration

    base_grasp_offset = default_grasp_offset_for_source(from_square)
    print(
        f"{from_square}->{to_square}: default grasp failed; "
        "running fallback 3x3 XY grasp grid"
    )
    grid_attempts = []
    successful_attempts = []
    for grid_candidate in fallback_grasp_grid_offsets(base_grasp_offset):
        candidate_grasp_offset = grid_candidate["grasp_offset"]
        if np.allclose(candidate_grasp_offset, grasp_offset):
            continue

        candidate_calibration = calibrate_with_grasp_offset(
            from_square,
            to_square,
            candidate_grasp_offset,
            pass_prefix=(
                "grid_"
                f"dx{grid_candidate['grid_dx']}_dy{grid_candidate['grid_dy']}"
            ),
        )
        attempt = {
            **grid_candidate,
            "calibration": candidate_calibration,
        }
        grid_attempts.append(attempt)

        if candidate_calibration["success"]:
            successful_attempts.append(attempt)

    if successful_attempts:
        successful_attempts.sort(
            key=lambda attempt: (
                float(attempt["calibration"]["selected_result"].get("score", np.inf)),
                float(attempt["calibration"]["selected_result"].get("xy_error", np.inf)),
                float(attempt["calibration"]["selected_result"].get("final_tilt_deg", np.inf)),
            )
        )
        selected_attempt = successful_attempts[0]
        selected_grid_delta = selected_attempt["grid_delta"]
        selected_calibration = selected_attempt["calibration"]
        selected_calibration["grasp_selection"] = {
            "source": "fallback_grid_override",
            "is_override": True,
            "reason": "default_grasp_failed",
            "base_grasp_offset": base_grasp_offset.copy(),
            "selected_grasp_offset": selected_attempt["grasp_offset"].copy(),
            "selected_grid_dx": selected_attempt["grid_dx"],
            "selected_grid_dy": selected_attempt["grid_dy"],
            "selected_grid_delta": selected_grid_delta.copy(),
            "grid_xy_delta": FALLBACK_GRASP_GRID_XY_DELTA,
            "grid_xy_steps": tuple(FALLBACK_GRASP_GRID_XY_STEPS),
            "successful_candidate_count": len(successful_attempts),
            "attempts": [
                summarize_grid_attempt(grid_attempt)
                for grid_attempt in grid_attempts
            ],
        }
        return selected_calibration

    calibration["grasp_selection"] = {
        **grasp_selection,
        "fallback_grid_attempted": True,
        "fallback_grid_success": False,
        "attempts": [
            summarize_grid_attempt(grid_attempt)
            for grid_attempt in grid_attempts
        ],
    }
    return calibration


def build_lookup(worker_source_squares=SOURCE_SQUARES, destination_squares=None):
    ensure_physics_connected()

    for from_square, to_square in lookup_moves(worker_source_squares, destination_squares):
        key = move_key(from_square, to_square)
        if SKIP_EXISTING_SUCCESSFUL_MOVES and common_move_meets_current_targets(key):
            print(f"{key}: already successful in common lookup; skipping.")
            continue

        calibration = calibrate_direct_move(from_square, to_square)
        if calibration["success"]:
            move = summarize_successful_calibration(
                from_square,
                to_square,
                calibration
            )
            merge_move_into_common_lookup(worker_source_squares, key, move)
        else:
            merge_move_into_common_lookup(worker_source_squares, key, None)
            print(f"{key}: None suitable found.")

    with common_lookup_lock():
        return load_common_lookup_locked()


def write_success_map(lookup):
    metadata_source_squares = lookup.get("metadata", {}).get("source_squares", [])
    if len(metadata_source_squares) != 1:
        print("Skipping success map for multi-source lookup.")
        return

    moves = lookup.get("moves", {})
    if not isinstance(moves, dict):
        moves = {}
    SUCCESS_MAP_PATH.write_text(
        render_svg(OUTPUT_PATH, lookup, moves),
        encoding="utf-8"
    )
    print(f"Wrote lookup success map: {SUCCESS_MAP_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build non-h reverse lookup entries and merge each completed move "
            "into the user-named common JSON under a file lock."
        )
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(SOURCE_SQUARES),
        help=(
            "Source squares for this worker process. "
            "Run multiple processes with disjoint source sets."
        ),
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Rebuild moves even if they are already successful in the common JSON.",
    )
    parser.add_argument(
        "--to-squares",
        nargs="+",
        help="Optional destination-square subset for a small test run, e.g. --to-squares e1.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    worker_source_squares = tuple(args.sources)
    destination_squares = tuple(args.to_squares) if args.to_squares else None
    global SKIP_EXISTING_SUCCESSFUL_MOVES
    SKIP_EXISTING_SUCCESSFUL_MOVES = not args.no_skip_existing

    try:
        lookup = build_lookup(worker_source_squares, destination_squares)
        print(f"\nSaved lookup: {OUTPUT_PATH}")
        for key, move in lookup["moves"].items():
            if move.get("from_square") not in worker_source_squares:
                continue
            metrics = move["metrics"]
            grasp_source = move.get("grasp_offset_source")
            print(
                f"{key}: success={move['success']} | "
                f"grasp={grasp_source} | "
                f"xy_error={metrics.get('xy_error')} | "
                f"tilt={metrics.get('final_tilt_deg')} | "
                f"fk={metrics.get('trajectory_fk_error')} | "
                f"reject={metrics.get('reject_reason')}"
            )
        write_success_map(lookup)
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
