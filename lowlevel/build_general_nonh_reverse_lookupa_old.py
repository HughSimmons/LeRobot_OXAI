import json
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


def normalize_square_sequence(config_name, squares):
    if isinstance(squares, str):
        squares = (squares,)
    else:
        squares = tuple(squares)

    for square in squares:
        if (
            not isinstance(square, str)
            or len(square) != 2
            or square[0] not in "abcdefgh"
            or square[1] not in "12345678"
        ):
            raise ValueError(
                f"{config_name} contains invalid square {square!r}; "
                "use values like 'a1' or ('a1', 'f8')"
            )
    return squares


SOURCE_SQUARES = normalize_square_sequence("SOURCE_SQUARES", ("a1",))
LOOKUP_TO_FILES = tuple("bcdef")
LOOKUP_TO_RANKS = range(1, 9)

DEFAULT_SOURCE_GRASP_OFFSET = np.array([-0.014, 0.002, -0.003])
SOURCE_GRASP_OFFSET_OVERRIDES = {
    # "f3": np.array([-0.011, 0.002, -0.003]),
}
MOVE_GRASP_OFFSET_OVERRIDES = {
    # "f3": {
    #     "a1": np.array([-0.014, -0.001, -0.003]),
    # },
}

DEFAULT_GRASP_OFFSET_BY_SOURCE = {
    source_square: np.array(
        SOURCE_GRASP_OFFSET_OVERRIDES.get(
            source_square,
            DEFAULT_SOURCE_GRASP_OFFSET,
        ),
        dtype=float,
    )
    for source_square in SOURCE_SQUARES
}
GRASP_OFFSET_OVERRIDES_BY_SOURCE = {
    source_square: {
        to_square: np.array(grasp_offset, dtype=float)
        for to_square, grasp_offset
        in MOVE_GRASP_OFFSET_OVERRIDES.get(source_square, {}).items()
    }
    for source_square in SOURCE_SQUARES
}
SOURCE_GRASP_SEARCH_RESULTS = {}

DIRECT_PLACE_CORRECTION_ROUNDS = 10
FALLBACK_CORRECTION_GAIN = 0.5
FALLBACK_GRASP_GRID_ENABLED = True
FALLBACK_GRASP_GRID_XY_DELTA = 0.003
FALLBACK_GRASP_GRID_XY_STEPS = (-1, 0, 1)
SOURCE_GRASP_SEARCH_ENABLED = True
SOURCE_GRASP_SEARCH_SAMPLE_COUNT = 3
SOURCE_GRASP_SEARCH_XY_DELTA = 0.003
SOURCE_GRASP_SEARCH_Z_DELTA = 0.002
SOURCE_GRASP_SEARCH_XY_STEPS = (-2, -1, 0, 1, 2)
SOURCE_GRASP_SEARCH_Z_STEPS = (0, -1, 1)
REQUIRE_SOURCE_GRASP_SEARCH_SUCCESS = True
SOURCE_GRASP_SEARCH_TO_SQUARES_BY_SOURCE = {
    # "a1": ("b1", "b2", "c1"),
}
USE_LONG_MOVE_STEPS_FOR_AB_DESTINATIONS = True
LONG_MOVE_STEP_DESTINATION_FILES = ("a", "b")
USE_LONG_MOVE_STEPS_FOR_DISTANT_DESTINATIONS = True
LONG_MOVE_STEP_MIN_SQUARE_DISTANCE = 5
LOOKUP_EDGE_SUPPORT_MARGIN = 0.08
XY_SUCCESS_THRESHOLD = 0.001
RELAXED_PLACEMENT_LOWER_STEPS_BY_TO_SQUARE = {
    f"{file}{rank}": 2
    for file in ("a", "b")
    for rank in range(1, 9)
}


def output_filename():
    source_label = (
        SOURCE_SQUARES[0]
        if len(SOURCE_SQUARES) == 1
        else "_".join(SOURCE_SQUARES)
    )
    return f"{source_label}_non_h_reverse_move_lookup.json"


OUTPUT_PATH = Path(__file__).resolve().parent / output_filename()
SUCCESS_MAP_PATH = default_output_path(OUTPUT_PATH)


def move_key(from_square, to_square):
    return f"{from_square}_to_{to_square}"


def lookup_moves():
    return tuple(
        (from_square, f"{file}{rank}")
        for from_square in SOURCE_SQUARES
        for file in LOOKUP_TO_FILES
        for rank in LOOKUP_TO_RANKS
        if f"{file}{rank}" != from_square
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


def summarize_failed_calibration(from_square, to_square, calibration):
    selected_result = calibration.get("selected_result")
    metrics = summarize_result(selected_result)
    return {
        "from_square": from_square,
        "to_square": to_square,
        "move_key": move_key(from_square, to_square),
        "reject_reason": calibration.get("reject_reason", "none_suitable_found"),
        "grasp_selection": json_safe(calibration.get("grasp_selection", {})),
        "metrics": metrics,
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
        and xy_error < XY_SUCCESS_THRESHOLD
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


def build_metadata():
    single_source_square = SOURCE_SQUARES[0] if len(SOURCE_SQUARES) == 1 else None
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "lowlevel/build_general_nonh_reverse_lookup.py",
        "trajectory_mode": "direct",
        "from_square": single_source_square,
        "source_squares": list(SOURCE_SQUARES),
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
        "source_grasp_search_enabled": SOURCE_GRASP_SEARCH_ENABLED,
        "source_grasp_search_results": json_safe(SOURCE_GRASP_SEARCH_RESULTS),
        "source_grasp_search_sample_count": SOURCE_GRASP_SEARCH_SAMPLE_COUNT,
        "source_grasp_search_xy_delta": SOURCE_GRASP_SEARCH_XY_DELTA,
        "source_grasp_search_z_delta": SOURCE_GRASP_SEARCH_Z_DELTA,
        "source_grasp_search_xy_steps": list(SOURCE_GRASP_SEARCH_XY_STEPS),
        "source_grasp_search_z_steps": list(SOURCE_GRASP_SEARCH_Z_STEPS),
        "require_source_grasp_search_success": REQUIRE_SOURCE_GRASP_SEARCH_SUCCESS,
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
        "xy_correction_target_error": XY_SUCCESS_THRESHOLD,
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
        key: compact_existing_move(move)
        for key, move in lookup["moves"].items()
        if move_meets_current_targets(move)
    }
    return lookup


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


def run_direct_move_in_fresh_world(
    from_square,
    to_square,
    grasp_offset,
    place_offset,
    move_steps_per_waypoint,
    placement_lower_steps,
    record_video=True,
    video_label=None,
):
    world = setup_sim_world(
        from_square,
        edge_support_margin=LOOKUP_EDGE_SUPPORT_MARGIN,
    )
    try:
        return run_direct_move_once(
            world,
            from_square,
            to_square,
            grasp_offset,
            place_offset,
            move_steps_per_waypoint,
            placement_lower_steps,
            record_video=record_video,
            video_label=video_label,
        )
    finally:
        p.removeState(world["state_id"])


def source_grasp_search_to_squares(from_square):
    configured = SOURCE_GRASP_SEARCH_TO_SQUARES_BY_SOURCE.get(from_square)
    if configured:
        return tuple(configured)

    to_squares = []
    for source, to_square in lookup_moves():
        if source != from_square:
            continue
        to_squares.append(to_square)
        if len(to_squares) >= SOURCE_GRASP_SEARCH_SAMPLE_COUNT:
            break
    return tuple(to_squares)


def source_grasp_search_candidates(base_grasp_offset):
    for dz in SOURCE_GRASP_SEARCH_Z_STEPS:
        for dx in SOURCE_GRASP_SEARCH_XY_STEPS:
            for dy in SOURCE_GRASP_SEARCH_XY_STEPS:
                delta = np.array([
                    dx * SOURCE_GRASP_SEARCH_XY_DELTA,
                    dy * SOURCE_GRASP_SEARCH_XY_DELTA,
                    dz * SOURCE_GRASP_SEARCH_Z_DELTA,
                ])
                yield {
                    "grid_dx": dx,
                    "grid_dy": dy,
                    "grid_dz": dz,
                    "grid_delta": delta,
                    "grasp_offset": base_grasp_offset + delta,
                }


def source_grasp_attempt_score(results):
    pickup_failures = sum(
        1 for result in results
        if not bool(result.get("pickup_success", False))
    )
    reject_count = sum(
        1 for result in results
        if result.get("reject_reason") is not None
    )
    premature_drop_count = sum(
        1 for result in results
        if bool(result.get("premature_drop", False))
    )
    fk_error_sum = sum(
        float(result.get("trajectory_fk_error", np.inf))
        for result in results
    )
    return (
        pickup_failures,
        reject_count,
        premature_drop_count,
        fk_error_sum,
    )


def evaluate_source_grasp_candidate(from_square, candidate):
    results = []
    for to_square in source_grasp_search_to_squares(from_square):
        result = run_direct_move_in_fresh_world(
            from_square,
            to_square,
            candidate["grasp_offset"],
            PLACE_OFFSET.copy(),
            move_steps_per_waypoint_for_lookup(from_square, to_square),
            placement_lower_steps_for_lookup(to_square),
            record_video=False,
            video_label="source_grasp_search",
        )
        results.append(result)

    return {
        **candidate,
        "results": results,
        "score": source_grasp_attempt_score(results),
        "pickup_success_count": sum(
            1 for result in results
            if bool(result.get("pickup_success", False))
        ),
    }


def select_source_grasp_offset(from_square):
    base_grasp_offset = DEFAULT_GRASP_OFFSET_BY_SOURCE[from_square].copy()
    if not SOURCE_GRASP_SEARCH_ENABLED:
        return {
            "source": "configured_default",
            "selected_grasp_offset": base_grasp_offset,
            "base_grasp_offset": base_grasp_offset,
            "attempts": [],
        }

    to_squares = source_grasp_search_to_squares(from_square)
    if not to_squares:
        return {
            "source": "configured_default",
            "selected_grasp_offset": base_grasp_offset,
            "base_grasp_offset": base_grasp_offset,
            "attempts": [],
            "reject_reason": "no_source_grasp_search_targets",
        }

    print(
        f"{from_square}: running source grasp search over "
        f"{len(to_squares)} target move(s): {', '.join(to_squares)}"
    )
    attempts = []
    for candidate in source_grasp_search_candidates(base_grasp_offset):
        attempt = evaluate_source_grasp_candidate(from_square, candidate)
        attempts.append(attempt)
        print(
            f"{from_square}: grasp candidate "
            f"dx={candidate['grid_dx']} dy={candidate['grid_dy']} "
            f"dz={candidate['grid_dz']} | "
            f"pickup={attempt['pickup_success_count']}/{len(to_squares)} | "
            f"score={attempt['score']}"
        )

    attempts.sort(key=lambda attempt: attempt["score"])
    selected_attempt = attempts[0]
    selected_grasp_offset = selected_attempt["grasp_offset"].copy()
    return {
        "source": "source_grasp_search",
        "success": selected_attempt["pickup_success_count"] == len(to_squares),
        "selected_grasp_offset": selected_grasp_offset,
        "base_grasp_offset": base_grasp_offset,
        "selected_grid_dx": selected_attempt["grid_dx"],
        "selected_grid_dy": selected_attempt["grid_dy"],
        "selected_grid_dz": selected_attempt["grid_dz"],
        "selected_grid_delta": selected_attempt["grid_delta"].copy(),
        "sample_to_squares": to_squares,
        "pickup_success_count": selected_attempt["pickup_success_count"],
        "sample_count": len(to_squares),
        "score": selected_attempt["score"],
        "attempts": [
            {
                "grid_dx": attempt["grid_dx"],
                "grid_dy": attempt["grid_dy"],
                "grid_dz": attempt["grid_dz"],
                "grid_delta": json_safe(attempt["grid_delta"]),
                "grasp_offset": json_safe(attempt["grasp_offset"]),
                "pickup_success_count": attempt["pickup_success_count"],
                "score": json_safe(attempt["score"]),
                "results": [
                    summarize_result(result)
                    for result in attempt["results"]
                ],
            }
            for attempt in attempts
        ],
    }


def run_source_grasp_searches():
    SOURCE_GRASP_SEARCH_RESULTS.clear()
    for from_square in SOURCE_SQUARES:
        search_result = select_source_grasp_offset(from_square)
        selected_grasp_offset = np.array(
            search_result["selected_grasp_offset"],
            dtype=float,
        )
        DEFAULT_GRASP_OFFSET_BY_SOURCE[from_square] = selected_grasp_offset
        SOURCE_GRASP_SEARCH_RESULTS[from_square] = search_result
        print(
            f"{from_square}: selected source grasp offset "
            f"{selected_grasp_offset}"
        )
        if (
            SOURCE_GRASP_SEARCH_ENABLED
            and
            REQUIRE_SOURCE_GRASP_SEARCH_SUCCESS
            and not search_result.get("success", False)
        ):
            raise RuntimeError(
                f"{from_square}: source grasp search failed; "
                f"best pickup count was "
                f"{search_result.get('pickup_success_count')}/"
                f"{search_result.get('sample_count')}"
            )


def direct_result_is_suitable(result):
    if result is None:
        return False
    return (
        result.get("reject_reason") is None
        and bool(result.get("pickup_success", False))
        and np.isfinite(float(result.get("xy_error", np.inf)))
        and np.isfinite(float(result.get("final_tilt_deg", np.inf)))
        and float(result["xy_error"]) < XY_SUCCESS_THRESHOLD
        and float(result["final_tilt_deg"]) < FINAL_TILT_TARGET_DEG
    )


def run_correction_pass(
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
        result = run_direct_move_in_fresh_world(
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
            selected_result = run_direct_move_in_fresh_world(
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
    move_steps_per_waypoint = move_steps_per_waypoint_for_lookup(from_square, to_square)
    placement_lower_steps = placement_lower_steps_for_lookup(to_square)

    calibration = run_correction_pass(
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
            from_square,
            to_square,
            grasp_offset,
            move_steps_per_waypoint,
            placement_lower_steps,
            correction_gain=FALLBACK_CORRECTION_GAIN,
            pass_label=f"{pass_prefix}_damped",
        )

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


def build_lookup():
    ensure_physics_connected()

    lookup = load_existing_lookup()
    run_source_grasp_searches()
    lookup["metadata"] = build_metadata()
    lookup.setdefault("moves", {})
    report = {
        from_square: {
            "successes": [],
            "failures": [],
        }
        for from_square in SOURCE_SQUARES
    }

    for from_square, to_square in lookup_moves():
        key = move_key(from_square, to_square)
        calibration = calibrate_direct_move(from_square, to_square)
        if calibration["success"]:
            move = summarize_successful_calibration(
                from_square,
                to_square,
                calibration
            )
            lookup["moves"][key] = move
            report[from_square]["successes"].append(key)
        else:
            lookup["moves"].pop(key, None)
            report[from_square]["failures"].append(
                summarize_failed_calibration(from_square, to_square, calibration)
            )
            print(f"{key}: None suitable found.")

    return lookup, report


def print_build_report(report):
    print("\nBuild summary by source square:")
    for from_square in SOURCE_SQUARES:
        source_report = report.get(from_square, {})
        successes = source_report.get("successes", [])
        failures = source_report.get("failures", [])
        total = len(successes) + len(failures)
        print(
            f"{from_square}: "
            f"successes={len(successes)}/{total} | "
            f"failures={len(failures)}/{total}"
        )
        if not failures:
            continue
        for failure in failures:
            metrics = failure.get("metrics") or {}
            print(
                f"  - {failure['move_key']}: "
                f"reject={failure.get('reject_reason')} | "
                f"xy={metrics.get('xy_error')} | "
                f"tilt={metrics.get('final_tilt_deg')} | "
                f"pickup={metrics.get('pickup_success')}"
            )


def write_success_map(lookup):
    if len(SOURCE_SQUARES) != 1:
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


def main():
    try:
        lookup, report = build_lookup()
        OUTPUT_PATH.write_text(
            json.dumps(json_safe(lookup), indent=2, sort_keys=True) + "\n",
            encoding="utf-8"
        )
        print(f"\nSaved lookup: {OUTPUT_PATH}")
        for key, move in lookup["moves"].items():
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
        print_build_report(report)
        write_success_map(lookup)
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
