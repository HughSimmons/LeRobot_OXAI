import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pybullet as p

from visualize_lookup_success import default_output_path, open_in_ide, render_svg
from visualize_lookup_success import expected_destinations, infer_from_square
from visualize_lookup_success import move_key as lookup_move_key
from visualize_lookup_success import move_is_successful

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


LOOKUP_FROM_SQUARE = "f5"
LOOKUP_TO_FILES = tuple("abcde")
# LOOKUP_TO_FILES = tuple("e")
# LOOKUP_TO_RANKS = range(6, 7)
LOOKUP_TO_RANKS = range(1, 9)
LOOKUP_MOVES = tuple(
    (LOOKUP_FROM_SQUARE, f"{file}{rank}")
    for file in LOOKUP_TO_FILES
    for rank in LOOKUP_TO_RANKS
    if f"{file}{rank}" != LOOKUP_FROM_SQUARE
)
DIRECT_GRASP_OFFSET_FOR_LOOKUP = np.array([-0.014, 0.002, -0.003])
B2_GRASP_OFFSET_FOR_LOOKUP = np.array([-0.011, 0.002, -0.003])
DIRECT_PLACE_CORRECTION_ROUNDS = 10
FALLBACK_CORRECTION_GAIN = 0.5
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

OUTPUT_PATH = Path(__file__).resolve().parent / "f5_non_h_reverse_move_lookup.json"
SUCCESS_MAP_PATH = default_output_path(OUTPUT_PATH)


def grasp_offset_for_lookup(to_square):
    if to_square == "b2":
        return B2_GRASP_OFFSET_FOR_LOOKUP.copy()
    return DIRECT_GRASP_OFFSET_FOR_LOOKUP.copy()


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

    source_grasp_offset = move.get("source_grasp_offset")
    try:
        source_grasp_offset = np.array(source_grasp_offset, dtype=float)
    except (TypeError, ValueError):
        return False

    to_square = move.get("to_square")
    if not isinstance(to_square, str):
        return False

    if not np.allclose(source_grasp_offset, grasp_offset_for_lookup(to_square)):
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
        "source": "lowlevel/build_f5_nonh_reverse_lookup.py",
        "trajectory_mode": "direct",
        "from_square": LOOKUP_FROM_SQUARE,
        "destination_files": list(LOOKUP_TO_FILES),
        "destination_ranks": list(LOOKUP_TO_RANKS),
        "excluded_destinations": [LOOKUP_FROM_SQUARE, "h1-h8"],
        "default_move_steps_per_waypoint": DEFAULT_MOVE_STEPS_PER_WAYPOINT,
        "long_move_steps_per_waypoint": LONG_TRANSPORT_MOVE_STEPS_PER_WAYPOINT,
        "long_move_step_destination_files": list(LONG_MOVE_STEP_DESTINATION_FILES),
        "long_move_step_min_square_distance": LONG_MOVE_STEP_MIN_SQUARE_DISTANCE,
        "direct_grasp_offset_for_lookup": json_safe(DIRECT_GRASP_OFFSET_FOR_LOOKUP),
        "b2_grasp_offset_for_lookup": json_safe(B2_GRASP_OFFSET_FOR_LOOKUP),
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


def move_steps_per_waypoint_for_lookup(to_square):
    from_file_idx = ord(LOOKUP_FROM_SQUARE[0]) - ord("a")
    to_file_idx = ord(to_square[0]) - ord("a")
    square_distance = (
        abs(from_file_idx - to_file_idx)
        + abs(int(LOOKUP_FROM_SQUARE[1]) - int(to_square[1]))
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
            }

        position_error = np.array(result.get("position_error", np.full(3, np.nan)))
        if not np.all(np.isfinite(position_error)):
            return {
                "selected_result": None,
                "selected_place_offset": PLACE_OFFSET.copy(),
                "success": False,
                "stopped_early": True,
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
    }


def calibrate_direct_move(from_square, to_square):
    world = setup_sim_world(
        from_square,
        edge_support_margin=LOOKUP_EDGE_SUPPORT_MARGIN
    )
    grasp_offset = grasp_offset_for_lookup(to_square)
    move_steps_per_waypoint = move_steps_per_waypoint_for_lookup(to_square)
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
            pass_label="direct",
        )
        if not calibration["success"] and not calibration["stopped_early"]:
            print(
                f"{from_square}->{to_square}: standard correction exhausted; "
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
                pass_label="damped",
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


def write_and_open_success_map(lookup):
    moves = lookup.get("moves", {})
    if not isinstance(moves, dict):
        moves = {}

    SUCCESS_MAP_PATH.write_text(
        render_svg(OUTPUT_PATH, lookup, moves),
        encoding="utf-8"
    )
    print(f"Wrote lookup success map: {SUCCESS_MAP_PATH}")
    if open_in_ide(SUCCESS_MAP_PATH, "auto"):
        print(f"Opened lookup success map: {SUCCESS_MAP_PATH}")
    else:
        print(f"Open lookup success map manually: {SUCCESS_MAP_PATH}")


def show_success_map_figure(lookup):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch, Rectangle
    except ImportError:
        print("matplotlib is not available; skipping success map figure.")
        return

    moves = lookup.get("moves", {})
    if not isinstance(moves, dict):
        moves = {}

    metadata = lookup.get("metadata", {})
    from_square = infer_from_square(metadata, moves)
    expected = expected_destinations(metadata, moves, from_square)

    colors = {
        "success": "#4caf50",
        "missing": "#e57373",
        "source": "#42a5f5",
        "excluded": "#eeeeee",
    }

    fig, ax = plt.subplots(figsize=(8, 8))
    for file_idx, file in enumerate("abcdefgh"):
        for rank in range(1, 9):
            square = f"{file}{rank}"
            if square == from_square:
                status = "source"
            elif square not in expected:
                status = "excluded"
            elif move_is_successful(moves.get(lookup_move_key(from_square, square))):
                status = "success"
            else:
                status = "missing"

            rect = Rectangle(
                (file_idx, rank - 1),
                1,
                1,
                facecolor=colors[status],
                edgecolor="#263238",
                linewidth=2 if status == "source" else 1,
            )
            ax.add_patch(rect)
            ax.text(
                file_idx + 0.5,
                rank - 0.5,
                square.upper() if status == "source" else square,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold" if status == "source" else "normal",
                color="#1f2933",
            )

    success_count = sum(
        1
        for square in expected
        if move_is_successful(moves.get(lookup_move_key(from_square, square)))
    )
    ax.set_title(
        f"Lookup success map: from {from_square} | "
        f"{success_count}/{len(expected)} successful"
    )
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xticks([idx + 0.5 for idx in range(8)])
    ax.set_xticklabels(list("abcdefgh"))
    ax.set_yticks([idx + 0.5 for idx in range(8)])
    ax.set_yticklabels(range(1, 9))
    ax.tick_params(length=0)
    ax.legend(
        handles=[
            Patch(facecolor=colors["success"], edgecolor="#263238", label="successful"),
            Patch(facecolor=colors["missing"], edgecolor="#263238", label="missing/failed"),
            Patch(facecolor=colors["source"], edgecolor="#263238", label="source"),
            Patch(facecolor=colors["excluded"], edgecolor="#263238", label="not targeted"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )
    plt.tight_layout()
    plt.show()


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
        write_and_open_success_map(lookup)
        show_success_map_figure(lookup)
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
