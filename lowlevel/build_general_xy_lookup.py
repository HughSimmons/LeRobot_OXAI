#!/usr/bin/env python3
"""Build one continuous XY-to-XY pick-and-place lookup entry.

Coordinates are resolved to world-frame metres before trajectory generation.
Square-based builders remain separate and continue to use their existing JSONs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p

from board_coordinates import (
    FILES,
    XYPoint,
    location_label,
    square_center_world_xy,
    point_from_xy,
    validate_square,
    world_xy_within_square,
)
from chess_traj import (
    lift_height_for_square,
    pickupmove_traj_with_metrics,
)
from continuous_xy_candidate_db import (
    DEFAULT_CANDIDATE_DB_PATH,
    find_successful_candidates,
    is_excessive_premature_drop,
    save_candidate,
)
import multisim_chess_fast as sim
from testkinematics import kinematics


LOWLEVEL_DIR = Path(__file__).resolve().parent
DEFAULT_GRASP_OFFSET = np.array([-0.014, 0.002, -0.003], dtype=float)
DEFAULT_PLACE_OFFSET = np.array([-0.011, 0.002, -0.003], dtype=float)
XY_SUCCESS_THRESHOLD = 0.01
MAX_PLACE_OFFSET_XY_ABS = 0.08
EDGE_SUPPORT_MARGIN = 0.08
LONG_MOVE_STEP_MIN_SQUARE_DISTANCE = 5
LONG_MOVE_STEP_DESTINATION_FILES = ("a", "b")
CONTINUOUS_DONOR_TARGET_TOLERANCE_M = 1e-6
CONTINUOUS_NEIGHBOR_BIAS_FACTOR = 0.5
CONTINUOUS_DONOR_BRIDGE_INTERPOLATION_STEPS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search a continuous world/board XY pick-and-place move."
    )
    parser.add_argument("--from-xy", nargs=2, type=float, required=True, metavar=("X", "Y"))
    parser.add_argument("--to-xy", nargs=2, type=float, required=True, metavar=("X", "Y"))
    parser.add_argument(
        "--frame",
        choices=("world", "board"),
        default="world",
        help="Input frame. Board coordinates are metres relative to board centre.",
    )
    parser.add_argument("--from-name", default=None)
    parser.add_argument("--to-name", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-db",
        type=Path,
        default=DEFAULT_CANDIDATE_DB_PATH,
        help="SQLite database receiving every persistable continuous candidate.",
    )
    parser.add_argument(
        "--database-accept-final-square",
        default=None,
        help=(
            "Also persist stable results whose final XY is anywhere within this "
            "square, even when they miss the exact continuous target."
        ),
    )
    parser.add_argument("--grasp-offset", nargs=3, type=float, default=DEFAULT_GRASP_OFFSET)
    parser.add_argument("--place-offset", nargs=3, type=float, default=DEFAULT_PLACE_OFFSET)
    parser.add_argument("--lift-height", type=float, default=None)
    parser.add_argument(
        "--move-steps",
        type=int,
        default=None,
        help=(
            "Steps per waypoint. If omitted, use the square-builder transport "
            "policy for the nearest board squares."
        ),
    )
    parser.add_argument("--placement-lower-steps", type=int, default=10)
    parser.add_argument("--placement-corrections", type=int, default=3)
    parser.add_argument("--correction-gain", type=float, default=0.5)
    parser.add_argument("--grid-radius", type=int, default=1)
    parser.add_argument("--grid-z-radius", type=int, default=1)
    parser.add_argument("--grid-xy-step", type=float, default=0.003)
    parser.add_argument("--grid-z-step", type=float, default=0.002)
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Evaluate every grasp candidate after success instead of stopping early.",
    )
    parser.add_argument(
        "--disable-builder-recovery",
        action="store_true",
        help=(
            "Disable the rook-only donor-bridge and neighbour lower-place-bias "
            "fallbacks normally applied after direct search failures."
        ),
    )
    return parser.parse_args()


def nearest_board_square(point: XYPoint) -> str:
    return min(
        (
            f"{file}{rank}"
            for file in FILES
            for rank in range(1, 9)
        ),
        key=lambda square: (
            (point.x - square_center_world_xy(square)[0]) ** 2
            + (point.y - square_center_world_xy(square)[1]) ** 2,
            square,
        ),
    )


def default_move_steps_for_points(start: XYPoint, target: XYPoint) -> tuple[int, dict[str, Any]]:
    start_square = nearest_board_square(start)
    target_square = nearest_board_square(target)
    square_distance = abs(ord(start_square[0]) - ord(target_square[0])) + abs(
        int(start_square[1]) - int(target_square[1])
    )
    use_long_transport = (
        target_square[0] in LONG_MOVE_STEP_DESTINATION_FILES
        or square_distance >= LONG_MOVE_STEP_MIN_SQUARE_DISTANCE
    )
    return (
        sim.LONG_TRANSPORT_MOVE_STEPS_PER_WAYPOINT
        if use_long_transport
        else sim.DEFAULT_MOVE_STEPS_PER_WAYPOINT,
        {
            "start_square": start_square,
            "target_square": target_square,
            "square_distance": square_distance,
            "use_long_transport": use_long_transport,
            "source": "nearest_square_transport_policy",
        },
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, XYPoint):
        return value.as_dict()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def grasp_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    base = np.array(args.grasp_offset, dtype=float)
    candidates = []
    for dx in range(-args.grid_radius, args.grid_radius + 1):
        for dy in range(-args.grid_radius, args.grid_radius + 1):
            for dz in range(-args.grid_z_radius, args.grid_z_radius + 1):
                delta = np.array(
                    [
                        dx * args.grid_xy_step,
                        dy * args.grid_xy_step,
                        dz * args.grid_z_step,
                    ],
                    dtype=float,
                )
                candidates.append(
                    {
                        "grid_dx": dx,
                        "grid_dy": dy,
                        "grid_dz": dz,
                        "grid_delta": delta,
                        "grasp_offset": base + delta,
                        "distance_sq": float(np.dot(delta, delta)),
                    }
                )
    candidates.sort(
        key=lambda item: (
            item["distance_sq"],
            abs(item["grid_dx"]) + abs(item["grid_dy"]) + abs(item["grid_dz"]),
            abs(item["grid_dz"]),
            item["grid_dx"],
            item["grid_dy"],
            item["grid_dz"],
        )
    )
    return candidates


def result_is_suitable(result: dict[str, Any]) -> bool:
    place_offset = np.array(result.get("place_offset", [np.inf] * 3), dtype=float)
    return (
        result.get("reject_reason") is None
        and bool(result.get("pickup_success"))
        and np.all(np.abs(place_offset[:2]) <= MAX_PLACE_OFFSET_XY_ABS)
        and np.isfinite(float(result.get("xy_error", np.inf)))
        and float(result["xy_error"]) < XY_SUCCESS_THRESHOLD
        and np.isfinite(float(result.get("final_tilt_deg", np.inf)))
        and float(result["final_tilt_deg"]) < sim.FINAL_TILT_TARGET_DEG
    )


def candidate_is_persistable(
    result: dict[str, Any],
    *,
    accept_final_square: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    excessive_drop, drop_policy = is_excessive_premature_drop(result)
    strict_success = result_is_suitable(result)
    final_position = np.array(result.get("final_position", [np.nan] * 3), dtype=float)
    stable_on_board = (
        bool(result.get("trajectory_valid", False))
        and bool(result.get("pickup_success", False))
        and np.all(np.isfinite(final_position))
        and sim.is_xy_on_board(final_position)
        and float(result.get("final_tilt_deg", np.inf)) < sim.FINAL_TILT_TARGET_DEG
        and final_position[2] >= sim.active_piece_config()["dropped_z_threshold"] - 0.003
    )
    accepted_in_square = (
        accept_final_square is not None
        and stable_on_board
        and world_xy_within_square(final_position[:2], accept_final_square)
    )
    accepted = (strict_success or accepted_in_square) and not excessive_drop
    return accepted, {
        **drop_policy,
        "strict_lookup_success": strict_success,
        "stable_on_board": stable_on_board,
        "accepted_final_square": accept_final_square,
        "final_xy_in_accepted_square": accepted_in_square,
        "acceptance_reason": (
            "strict_lookup_success"
            if strict_success
            else (
                "stable_within_accepted_square"
                if accepted_in_square
                else "not_persistable"
            )
        ),
    }


def release_rotation(
    start: XYPoint,
    target: XYPoint,
    grasp_offset: np.ndarray,
    place_offset: np.ndarray,
    *,
    lift_height: float,
    placement_lower_steps: int,
) -> np.ndarray:
    movelist, closeidx, _ = pickupmove_traj_with_metrics(
        start,
        target,
        board_origin=sim.board_origin,
        GRASP_OFFSET=grasp_offset,
        PLACE_OFFSET=place_offset,
        placement_lower_steps=placement_lower_steps,
        lift_height=lift_height,
    )
    release_idx = sim.find_release_move_index(movelist, closeidx)
    release_joints = movelist[max(0, release_idx - 1)]
    return kinematics.forward_kinematics(np.array(release_joints, dtype=float))[:3, :3]


def corrected_place_offset(
    start: XYPoint,
    target: XYPoint,
    result: dict[str, Any],
    grasp_offset: np.ndarray,
    place_offset: np.ndarray,
    *,
    lift_height: float,
    placement_lower_steps: int,
    gain: float,
) -> np.ndarray:
    world_correction = np.array(result["position_error"], dtype=float)
    world_correction[2] = 0.0
    rotation = release_rotation(
        start,
        target,
        grasp_offset,
        place_offset,
        lift_height=lift_height,
        placement_lower_steps=placement_lower_steps,
    )
    return place_offset + rotation.T @ (world_correction * gain)


def segment_bounds(traj_metrics: dict[str, Any], name: str) -> tuple[int, int] | None:
    segment = traj_metrics.get("segments", {}).get(name)
    if not isinstance(segment, dict):
        return None
    try:
        start, end = int(segment["start"]), int(segment["end"])
    except (KeyError, TypeError, ValueError):
        return None
    return (start, end) if start <= end else None


def make_joint_bridge(start: np.ndarray, end: np.ndarray, steps: int) -> list[np.ndarray]:
    return [
        (1.0 - alpha) * start + alpha * end
        for alpha in np.linspace(0.0, 1.0, steps + 2, dtype=float)[1:-1]
    ]


def candidate_matches_active_rook(candidate: dict[str, Any]) -> bool:
    piece_config = candidate.get("piece_config", {})
    return (
        piece_config.get("piece_model") == sim.active_piece_config().get("piece_model")
        and piece_config.get("collision_model")
        == sim.active_piece_config().get("collision_model")
    )


def continuous_donor_candidates(
    candidate_db_path: Path,
    *,
    start: XYPoint,
    target: XYPoint,
) -> list[dict[str, Any]]:
    """Return same-destination rook candidates ordered by local source distance."""

    candidates = []
    for candidate in find_successful_candidates(candidate_db_path, limit=10_000):
        if not candidate_matches_active_rook(candidate):
            continue
        if (
            abs(float(candidate["to_x"]) - target.x) > CONTINUOUS_DONOR_TARGET_TOLERANCE_M
            or abs(float(candidate["to_y"]) - target.y) > CONTINUOUS_DONOR_TARGET_TOLERANCE_M
        ):
            continue
        metrics = candidate.get("metrics", {})
        if (
            metrics.get("reject_reason") is not None
            or not bool(metrics.get("pickup_success"))
            or float(metrics.get("trajectory_fk_error", np.inf))
            > sim.MAX_TRAJECTORY_FK_ERROR
        ):
            continue
        source_delta = np.array(
            [float(candidate["from_x"]) - start.x, float(candidate["from_y"]) - start.y],
            dtype=float,
        )
        candidate["source_distance_m"] = float(np.linalg.norm(source_delta))
        candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            item["source_distance_m"],
            float(item.get("metrics", {}).get("score", np.inf)),
            float(item.get("metrics", {}).get("xy_error", np.inf)),
            int(item["id"]),
        )
    )
    return candidates


def build_continuous_donor_bridge_override(
    start: XYPoint,
    target: XYPoint,
    grasp_offset: np.ndarray,
    place_offset: np.ndarray,
    donor: dict[str, Any],
    *,
    lift_height: float,
    placement_lower_steps: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Keep the requested pickup prefix and splice a local donor destination suffix."""

    target_movelist, target_closeidx, target_metrics = pickupmove_traj_with_metrics(
        start, target, board_origin=sim.board_origin, GRASP_OFFSET=grasp_offset,
        PLACE_OFFSET=place_offset, placement_lower_steps=placement_lower_steps,
        lift_height=lift_height,
    )
    donor_start = point_from_xy(float(donor["from_x"]), float(donor["from_y"]), frame="world")
    donor_place_offset = np.array(donor["place_offset"], dtype=float)
    donor_movelist, _, donor_metrics = pickupmove_traj_with_metrics(
        donor_start, target, board_origin=sim.board_origin,
        GRASP_OFFSET=np.array(donor["grasp_offset"], dtype=float),
        PLACE_OFFSET=donor_place_offset, placement_lower_steps=placement_lower_steps,
        lift_height=lift_height,
    )
    target_bounds = segment_bounds(target_metrics, "destination_above_place")
    donor_bounds = segment_bounds(donor_metrics, "destination_above_place")
    if target_bounds is None or donor_bounds is None:
        return None, "missing_destination_above_place_segment"
    prefix_end, donor_suffix_start = target_bounds[0], donor_bounds[0]
    if prefix_end <= 0 or donor_suffix_start >= len(donor_movelist):
        return None, "invalid_destination_bridge_bounds"
    prefix = [np.array(joints, dtype=float).copy() for joints in target_movelist[:prefix_end]]
    suffix = [np.array(joints, dtype=float).copy() for joints in donor_movelist[donor_suffix_start:]]
    if not prefix or not suffix:
        return None, "empty_destination_bridge_segment"
    bridge = make_joint_bridge(prefix[-1], suffix[0], CONTINUOUS_DONOR_BRIDGE_INTERPOLATION_STEPS)
    combined_fk_error = max(
        float(target_metrics.get("max_fk_error", np.inf)),
        float(donor_metrics.get("max_fk_error", np.inf)),
    )
    if combined_fk_error > sim.MAX_TRAJECTORY_FK_ERROR:
        return None, "bridged_fk_error_too_large"
    combined_metrics = {
        **donor_metrics,
        "max_fk_error": combined_fk_error,
        "slow_waypoint_indices": [
            prefix_end + len(bridge) + index - donor_suffix_start
            for index in donor_metrics.get("slow_waypoint_indices", [])
            if index >= donor_suffix_start
        ],
        "segments": {},
        "continuous_donor_bridge": {
            "enabled": True,
            "reason": "trajectory_fk_error_too_large",
            "donor_candidate_id": donor["id"],
            "donor_move_id": donor["move_id"],
            "donor_source_xy": [donor["from_x"], donor["from_y"]],
            "donor_source_distance_m": donor["source_distance_m"],
            "donor_grasp_offset": donor["grasp_offset"],
            "donor_place_offset": donor["place_offset"],
            "target_prefix_end": prefix_end,
            "donor_suffix_start": donor_suffix_start,
            "bridge_interpolation_steps": CONTINUOUS_DONOR_BRIDGE_INTERPOLATION_STEPS,
            "bridge_inserted_waypoints": len(bridge),
            "combined_fk_error": combined_fk_error,
        },
    }
    return {
        "movelist": prefix + bridge + suffix,
        "closeidx": target_closeidx,
        "traj_metrics": combined_metrics,
    }, None


def run_candidate(
    world: dict[str, Any],
    start: XYPoint,
    target: XYPoint,
    grasp_offset: np.ndarray,
    initial_place_offset: np.ndarray,
    args: argparse.Namespace,
    lift_height: float,
    lower_place_path_bias: np.ndarray | None = None,
    trajectory_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    place_offset = initial_place_offset.copy()
    passes = []
    max_passes = max(1, args.placement_corrections + 1)
    for correction_index in range(max_passes):
        result = sim.run_sim_move(
            world,
            start,
            target,
            grasp_offset,
            place_offset=place_offset,
            return_metrics=True,
            record_video=False,
            move_steps_per_waypoint=args.move_steps,
            placement_lower_steps=args.placement_lower_steps,
            lift_height=lift_height,
            lower_place_path_bias=lower_place_path_bias,
            trajectory_override=trajectory_override,
        )
        result["score"] = sim.score_place_result(result)
        result["success"] = result_is_suitable(result)
        passes.append({"correction_index": correction_index, "result": result})

        if result["success"] or trajectory_override is not None:
            break
        if not result.get("trajectory_valid", True) or not result.get("pickup_success"):
            break
        if correction_index >= args.placement_corrections:
            break

        next_place_offset = corrected_place_offset(
            start,
            target,
            result,
            grasp_offset,
            place_offset,
            lift_height=lift_height,
            placement_lower_steps=args.placement_lower_steps,
            gain=args.correction_gain,
        )
        if np.any(np.abs(next_place_offset[:2]) > MAX_PLACE_OFFSET_XY_ABS):
            break
        place_offset = next_place_offset

    best_pass = min(
        passes,
        key=lambda item: (
            not item["result"]["success"],
            float(item["result"].get("score", np.inf)),
            float(item["result"].get("xy_error", np.inf)),
        ),
    )
    return {
        "success": bool(best_pass["result"]["success"]),
        "selected_place_offset": np.array(best_pass["result"]["place_offset"], dtype=float),
        "selected_result": best_pass["result"],
        "placement_passes": passes,
    }


def persist_attempt(
    attempt: dict[str, Any],
    *,
    candidate_db_path: Path,
    move_id: str,
    start: XYPoint,
    target: XYPoint,
    search_metadata: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    attempt["database_candidates"] = []
    for pass_record in attempt["placement_passes"]:
        pass_result = pass_record["result"]
        persistable, drop_policy = candidate_is_persistable(
            pass_result, accept_final_square=args.database_accept_final_square,
        )
        pass_record["candidate_database_policy"] = drop_policy
        pass_record["persistable_candidate"] = persistable
        if not persistable:
            continue
        database_record = save_candidate(
            candidate_db_path, move_id=move_id, from_xy=start.as_xy(), to_xy=target.as_xy(),
            grasp_offset=np.array(attempt["grasp_offset"], dtype=float),
            place_offset=np.array(pass_result["place_offset"], dtype=float),
            metrics=pass_result,
            search={**search_metadata, "strategy": attempt.get("strategy", "direct"),
                    "grid_dx": attempt.get("grid_dx"), "grid_dy": attempt.get("grid_dy"),
                    "grid_dz": attempt.get("grid_dz"), "correction_index": pass_record["correction_index"]},
            piece_config=sim.active_piece_config(),
            source_lookup={"run_output_dir": str(output_dir), "input_frame": args.frame,
                           "from_input_xy": list(map(float, args.from_xy)),
                           "to_input_xy": list(map(float, args.to_xy))},
            drop_policy=drop_policy,
        )
        attempt["database_candidates"].append(database_record)


def main() -> int:
    args = parse_args()
    if args.grid_radius < 0 or args.grid_z_radius < 0:
        raise ValueError("grid radii cannot be negative")
    if args.placement_corrections < 0:
        raise ValueError("placement corrections cannot be negative")
    if args.database_accept_final_square is not None:
        args.database_accept_final_square = validate_square(
            args.database_accept_final_square
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    start = point_from_xy(*args.from_xy, frame=args.frame, name=args.from_name)
    target = point_from_xy(*args.to_xy, frame=args.frame, name=args.to_name)
    for label, point in (("from", start), ("to", target)):
        if not sim.is_xy_on_board((*point.as_xy(), 0.0), margin=0.0):
            raise ValueError(f"{label} XY point is outside the chessboard: {point.as_xy()}")
    move_id = f"{location_label(start)}_to_{location_label(target)}"
    if args.move_steps is None:
        args.move_steps, move_steps_policy = default_move_steps_for_points(start, target)
    else:
        move_steps_policy = {
            "source": "explicit_cli_override",
            "start_square": nearest_board_square(start),
            "target_square": nearest_board_square(target),
            "square_distance": None,
            "use_long_transport": args.move_steps
            == sim.LONG_TRANSPORT_MOVE_STEPS_PER_WAYPOINT,
        }
    lift_height = (
        float(args.lift_height)
        if args.lift_height is not None
        else float(lift_height_for_square(target))
    )
    candidate_db_path = args.candidate_db.expanduser().resolve()
    search_metadata = {
        "stop_on_first_success": not args.full_grid,
        "grid_order": "nearest_physical_delta",
        "grid_radius": args.grid_radius,
        "grid_z_radius": args.grid_z_radius,
        "grid_xy_step": args.grid_xy_step,
        "grid_z_step": args.grid_z_step,
        "placement_corrections": args.placement_corrections,
        "correction_gain": args.correction_gain,
        "lift_height": lift_height,
        "move_steps": args.move_steps,
        "move_steps_policy": move_steps_policy,
        "placement_lower_steps": args.placement_lower_steps,
        "database_accept_final_square": args.database_accept_final_square,
        "builder_strategy_recovery": not args.disable_builder_recovery,
        "far_square_seed_policy": "native_chess_traj_location_file",
        "continuous_donor_bridge": {
            "enabled": not args.disable_builder_recovery,
            "source": "continuous_rook_candidate_db_same_destination",
            "selection": "nearest_source_xy_then_score_then_xy_error",
            "trigger": "trajectory_fk_error_too_large",
        },
        "continuous_neighbor_lower_place_bias": {
            "enabled": not args.disable_builder_recovery,
            "source": "continuous_rook_candidate_db_same_destination",
            "trigger": "grasp_grid_exhausted",
            "factor": CONTINUOUS_NEIGHBOR_BIAS_FACTOR,
        },
    }

    sim.ensure_physics_connected()
    world = sim.setup_sim_world(start, edge_support_margin=EDGE_SUPPORT_MARGIN)
    attempts = []
    successful_attempts = []
    try:
        for candidate in grasp_candidates(args):
            candidate_result = run_candidate(
                world,
                start,
                target,
                np.array(candidate["grasp_offset"], dtype=float),
                np.array(args.place_offset, dtype=float),
                args,
                lift_height,
            )
            attempt = {**candidate, **candidate_result, "strategy": "direct"}
            persist_attempt(
                attempt, candidate_db_path=candidate_db_path, move_id=move_id,
                start=start, target=target, search_metadata=search_metadata,
                output_dir=output_dir, args=args,
            )
            attempts.append(attempt)
            print(
                f"{move_id} grasp=({candidate['grid_dx']},{candidate['grid_dy']},{candidate['grid_dz']}) "
                f"success={candidate_result['success']} "
                f"fk={candidate_result['selected_result'].get('trajectory_fk_error')} "
                f"xy={candidate_result['selected_result'].get('xy_error')} "
                f"tilt={candidate_result['selected_result'].get('final_tilt_deg')}",
                flush=True,
            )
            if candidate_result["success"]:
                successful_attempts.append(attempt)
                if not args.full_grid:
                    break

            if (
                not args.disable_builder_recovery
                and candidate_result["selected_result"].get("reject_reason")
                == "trajectory_fk_error_too_large"
            ):
                for donor in continuous_donor_candidates(
                    candidate_db_path, start=start, target=target,
                )[:1]:
                    override, bridge_reason = build_continuous_donor_bridge_override(
                        start, target, np.array(candidate["grasp_offset"], dtype=float),
                        np.array(candidate_result["selected_place_offset"], dtype=float), donor,
                        lift_height=lift_height,
                        placement_lower_steps=args.placement_lower_steps,
                    )
                    if override is None:
                        attempts[-1]["continuous_donor_bridge_reject_reason"] = bridge_reason
                        continue
                    bridge_result = run_candidate(
                        world, start, target, np.array(candidate["grasp_offset"], dtype=float),
                        np.array(candidate_result["selected_place_offset"], dtype=float), args,
                        lift_height, trajectory_override=override,
                    )
                    bridge_result["selected_result"]["continuous_donor_bridge"] = (
                        override["traj_metrics"]["continuous_donor_bridge"]
                    )
                    bridge_attempt = {
                        **candidate, **bridge_result, "strategy": "continuous_donor_bridge",
                        "continuous_donor_bridge": override["traj_metrics"]["continuous_donor_bridge"],
                    }
                    persist_attempt(
                        bridge_attempt, candidate_db_path=candidate_db_path, move_id=move_id,
                        start=start, target=target, search_metadata=search_metadata,
                        output_dir=output_dir, args=args,
                    )
                    attempts.append(bridge_attempt)
                    if bridge_result["success"]:
                        successful_attempts.append(bridge_attempt)
                        break
                if successful_attempts and not args.full_grid:
                    break

        if not successful_attempts and not args.disable_builder_recovery:
            donor_candidates = continuous_donor_candidates(
                candidate_db_path, start=start, target=target,
            )
            if donor_candidates:
                best_failed = min(
                    attempts,
                    key=lambda item: (
                        not bool(item["selected_result"].get("pickup_success")),
                        float(item["selected_result"].get("trajectory_fk_error", np.inf)),
                        float(item["selected_result"].get("score", np.inf)),
                        item["distance_sq"],
                    ),
                )
                donor = donor_candidates[0]
                base_place_offset = np.array(best_failed["selected_place_offset"], dtype=float)
                lower_place_path_bias = CONTINUOUS_NEIGHBOR_BIAS_FACTOR * (
                    np.array(donor["place_offset"], dtype=float) - base_place_offset
                )
                bias_result = run_candidate(
                    world, start, target, np.array(best_failed["grasp_offset"], dtype=float),
                    base_place_offset, args, lift_height,
                    lower_place_path_bias=lower_place_path_bias,
                )
                bias_result["selected_result"]["lower_place_path_bias"] = lower_place_path_bias
                bias_result["selected_result"]["continuous_neighbor_lower_place_bias"] = {
                    "enabled": True,
                    "reason": "grasp_grid_exhausted",
                    "donor_candidate_id": donor["id"],
                    "donor_move_id": donor["move_id"],
                    "donor_source_xy": [donor["from_x"], donor["from_y"]],
                    "donor_source_distance_m": donor["source_distance_m"],
                    "donor_place_offset": donor["place_offset"],
                    "base_place_offset": base_place_offset,
                    "factor": CONTINUOUS_NEIGHBOR_BIAS_FACTOR,
                    "bias": lower_place_path_bias,
                }
                bias_attempt = {
                    **{key: best_failed[key] for key in ("grid_dx", "grid_dy", "grid_dz", "grid_delta", "grasp_offset", "distance_sq")},
                    **bias_result,
                    "strategy": "continuous_neighbor_lower_place_bias",
                }
                persist_attempt(
                    bias_attempt, candidate_db_path=candidate_db_path, move_id=move_id,
                    start=start, target=target, search_metadata=search_metadata,
                    output_dir=output_dir, args=args,
                )
                attempts.append(bias_attempt)
                if bias_result["success"]:
                    successful_attempts.append(bias_attempt)
    finally:
        p.removeState(world["state_id"])

    candidate_pool = successful_attempts or attempts
    selected = min(
        candidate_pool,
        key=lambda item: (
            not item["success"],
            float(item["selected_result"].get("score", np.inf)),
            float(item["selected_result"].get("xy_error", np.inf)),
            item["distance_sq"],
        ),
    )
    selected_result = selected["selected_result"]
    payload = {
        "schema": "continuous_xy_lookup_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "move_id": move_id,
        "success": bool(selected["success"]),
        "input_frame": args.frame,
        "from_input_xy": list(map(float, args.from_xy)),
        "to_input_xy": list(map(float, args.to_xy)),
        "from": start,
        "to": target,
        "source_grasp_offset": selected["grasp_offset"],
        "selected_place_offset": selected["selected_place_offset"],
        "metrics": selected_result,
        "search": search_metadata,
        "candidate_database": {
            "path": candidate_db_path,
            "stored_candidate_count": sum(
                len(attempt.get("database_candidates", [])) for attempt in attempts
            ),
        },
        "piece_config": sim.active_piece_config(),
        "attempts": attempts,
    }
    output_path = output_dir / f"{move_id}.json"
    output_path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {output_path}")
    print(f"Success: {payload['success']}")
    return 0 if payload["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
