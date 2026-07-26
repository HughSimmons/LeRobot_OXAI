import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as p

import build_general_nonh_reverse_lookup as builder
from chess_traj import WRIST_ROLL_IDX, kinematics, nearest_equivalent_angle_deg
import multisim_chess_fast as sim


DEFAULT_MOVES = (
    "c3_to_d8",
    # "d6_to_a6",
    # "d6_to_a7",
    # "e7_to_a5",
    # "e6_to_a5",
    # "e5_to_a6",
    # "f1_to_a4",
    # "f1_to_b6",
    # "f1_to_b7",
    # "f1_to_c8",
)


DONOR_CARRY_SPLICE_TEST_MODE = "donor_carry_splice"
DONOR_CARRY_SPLICE_MAX_WRIST_ROLL_DELTA_DEG = 12.0
DONOR_CARRY_SPLICE_RATE_LIMIT_SUFFIX_WAYPOINTS = 6
DONOR_CARRY_SPLICE_LATE_BUFFER_WAYPOINTS = 2
DONOR_CARRY_SPLICE_SCORE_WINDOW = 3
DONOR_CARRY_SPLICE_FORCE_TARGET_IDX = 26
DONOR_CARRY_SPLICE_FORCE_DONOR_IDX = 26


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
    return lookup_path, lookup, entry


def saved_lift_height_for_entry(move_key, lookup, entry, to_square):
    metrics = entry.get("metrics", {})
    metric_lift_height = metrics.get("lift_height")
    if metric_lift_height is not None:
        return float(metric_lift_height)

    metadata = lookup.get("metadata", {})
    target_moves = metadata.get("target_moves")
    metadata_lift_height = metadata.get("lookup_lift_height_override")
    if (
        metadata_lift_height is not None
        and isinstance(target_moves, list)
        and move_key in target_moves
    ):
        return float(metadata_lift_height)

    return builder.lift_height_for_square(to_square)


def regenerate_saved_trajectory(entry, to_square, lift_height):
    return builder.pickupmove_traj_with_metrics(
        entry["from_square"],
        entry["to_square"],
        board_origin=builder.board_origin,
        GRASP_OFFSET=np.array(entry["source_grasp_offset"], dtype=float),
        PLACE_OFFSET=np.array(entry["selected_place_offset"], dtype=float),
        placement_lower_steps=builder.placement_lower_steps_for_lookup(to_square),
        lift_height=lift_height,
    )


def clamp_wrist_roll_delta(joints, reference_roll_deg, max_delta_deg):
    joints = np.array(joints, dtype=float).copy()
    wrapped_roll = nearest_equivalent_angle_deg(
        joints[WRIST_ROLL_IDX],
        reference_roll_deg,
    )
    delta = wrapped_roll - reference_roll_deg
    delta = float(np.clip(delta, -max_delta_deg, max_delta_deg))
    joints[WRIST_ROLL_IDX] = reference_roll_deg + delta
    return joints


def end_effector_pose(joints):
    pose = kinematics.forward_kinematics(np.array(joints, dtype=float))
    return pose[:3, 3].copy(), pose[:3, :3].copy()


def rotation_angle_error_deg(rot_a, rot_b):
    rel = rot_a.T @ rot_b
    trace = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def donor_carry_splice_score_report(lookup_dir, move_key, lookup, entry):
    if move_key != "a1_to_f1":
        raise ValueError(f"Score report only supports a1_to_f1, got {move_key}")

    donor_candidates = [
        candidate
        for candidate in builder.iter_bridge_donor_candidates("a1", "f1")
        if candidate["key"] != move_key
    ]
    donor_candidates.sort(
        key=lambda candidate: (
            candidate["distance"],
            candidate["score"],
            candidate["xy_error"],
            str(candidate["path"].name),
            candidate["key"],
        )
    )
    donor = donor_candidates[0] if donor_candidates else None
    if donor is None or donor["key"] != "b1_to_f1":
        raise ValueError(
            "Expected nearest successful donor for a1_to_f1 to be b1_to_f1"
        )

    _, donor_lookup, donor_entry = load_lookup_entry(lookup_dir, donor["key"])
    donor_lift_height = saved_lift_height_for_entry(
        donor["key"],
        donor_lookup,
        donor_entry,
        donor_entry["to_square"],
    )
    target_lift_height = saved_lift_height_for_entry(
        move_key,
        lookup,
        entry,
        entry["to_square"],
    )
    target_movelist, _, target_metrics = regenerate_saved_trajectory(
        entry,
        entry["to_square"],
        target_lift_height,
    )
    donor_movelist, _, donor_metrics = regenerate_saved_trajectory(
        donor_entry,
        donor_entry["to_square"],
        donor_lift_height,
    )

    target_above_bounds = builder.segment_bounds(target_metrics, "destination_above_place")
    donor_above_bounds = builder.segment_bounds(donor_metrics, "destination_above_place")
    if target_above_bounds is None or donor_above_bounds is None:
        raise ValueError("Missing destination_above_place bounds for score report")

    target_start, target_end = target_above_bounds
    donor_start, donor_end = donor_above_bounds
    target_min_idx = max(target_start, target_end - DONOR_CARRY_SPLICE_SCORE_WINDOW)
    donor_min_idx = max(donor_start, donor_end - DONOR_CARRY_SPLICE_SCORE_WINDOW)

    candidates = []
    for target_idx in range(target_min_idx, target_end):
        target_pos, target_rot = end_effector_pose(target_movelist[target_idx])
        for donor_idx in range(donor_min_idx, donor_end):
            donor_pos, donor_rot = end_effector_pose(donor_movelist[donor_idx])
            pos_error = float(np.linalg.norm(target_pos - donor_pos))
            rot_error_deg = rotation_angle_error_deg(target_rot, donor_rot)
            wrist_roll_delta_deg = abs(
                nearest_equivalent_angle_deg(
                    donor_movelist[donor_idx][WRIST_ROLL_IDX],
                    target_movelist[target_idx][WRIST_ROLL_IDX],
                )
                - float(target_movelist[target_idx][WRIST_ROLL_IDX])
            )
            joint_jump_norm = float(
                np.linalg.norm(
                    np.array(donor_movelist[donor_idx], dtype=float)
                    - np.array(target_movelist[target_idx], dtype=float)
                )
            )
            score = pos_error + 0.002 * rot_error_deg + 0.0005 * joint_jump_norm
            candidates.append(
                {
                    "score": score,
                    "target_idx": target_idx,
                    "donor_idx": donor_idx,
                    "position_error": pos_error,
                    "rotation_error_deg": rot_error_deg,
                    "wrist_roll_delta_deg": wrist_roll_delta_deg,
                    "joint_jump_norm": joint_jump_norm,
                }
            )

    candidates.sort(
        key=lambda item: (
            item["score"],
            item["position_error"],
            item["rotation_error_deg"],
            item["joint_jump_norm"],
            item["target_idx"],
            item["donor_idx"],
        )
    )

    return {
        "test_mode": "donor_carry_splice_score_report",
        "target_move": move_key,
        "donor_move": donor["key"],
        "target_destination_above_place_bounds": list(target_above_bounds),
        "donor_destination_above_place_bounds": list(donor_above_bounds),
        "score_window": DONOR_CARRY_SPLICE_SCORE_WINDOW,
        "top_candidates": candidates[:10],
    }


def build_donor_carry_splice_override(lookup_dir, move_key, lookup, entry):
    if move_key != "a1_to_f1":
        raise ValueError(
            f"{DONOR_CARRY_SPLICE_TEST_MODE} only supports a1_to_f1, got {move_key}"
        )

    donor_candidates = [
        candidate
        for candidate in builder.iter_bridge_donor_candidates("a1", "f1")
        if candidate["key"] != move_key
    ]
    donor_candidates.sort(
        key=lambda candidate: (
            candidate["distance"],
            candidate["score"],
            candidate["xy_error"],
            str(candidate["path"].name),
            candidate["key"],
        )
    )
    donor = donor_candidates[0] if donor_candidates else None
    if donor is None or donor["key"] != "b1_to_f1":
        raise ValueError(
            "Expected nearest successful donor for a1_to_f1 to be b1_to_f1"
        )

    donor_lookup_path, donor_lookup, donor_entry = load_lookup_entry(
        lookup_dir,
        donor["key"],
    )
    donor_lift_height = saved_lift_height_for_entry(
        donor["key"],
        donor_lookup,
        donor_entry,
        donor_entry["to_square"],
    )
    target_lift_height = saved_lift_height_for_entry(
        move_key,
        lookup,
        entry,
        entry["to_square"],
    )
    if abs(target_lift_height - donor_lift_height) > 1e-9:
        raise ValueError(
            "Target and donor lift heights differ; this one-off splice expects them to match"
        )

    target_movelist, target_closeidx, target_metrics = regenerate_saved_trajectory(
        entry,
        entry["to_square"],
        target_lift_height,
    )
    donor_movelist, _, donor_metrics = regenerate_saved_trajectory(
        donor_entry,
        donor_entry["to_square"],
        donor_lift_height,
    )

    target_above_bounds = builder.segment_bounds(
        target_metrics,
        "destination_above_place",
    )
    donor_above_bounds = builder.segment_bounds(
        donor_metrics,
        "destination_above_place",
    )
    if target_above_bounds is None or donor_above_bounds is None:
        raise ValueError("Missing destination_above_place segment for donor carry splice")
    if target_above_bounds != donor_above_bounds:
        raise ValueError(
            f"Mismatched destination_above_place bounds: {target_above_bounds} vs {donor_above_bounds}"
        )

    above_start, above_end = target_above_bounds
    above_len = above_end - above_start
    if above_len < 2:
        raise ValueError(
            f"destination_above_place segment too short for splice: {target_above_bounds}"
        )

    target_prefix_end = DONOR_CARRY_SPLICE_FORCE_TARGET_IDX
    donor_suffix_start = DONOR_CARRY_SPLICE_FORCE_DONOR_IDX
    if target_prefix_end <= 0 or donor_suffix_start >= len(donor_movelist):
        raise ValueError(
            f"Invalid donor carry splice bounds: target_prefix_end={target_prefix_end}, "
            f"donor_suffix_start={donor_suffix_start}"
        )
    if not (above_start <= target_prefix_end < above_end):
        raise ValueError(
            f"Forced target splice index {target_prefix_end} outside destination_above_place "
            f"bounds {target_above_bounds}"
        )
    if not (donor_above_bounds[0] <= donor_suffix_start < donor_above_bounds[1]):
        raise ValueError(
            f"Forced donor splice index {donor_suffix_start} outside destination_above_place "
            f"bounds {donor_above_bounds}"
        )

    prefix = [np.array(joints, dtype=float).copy() for joints in target_movelist[:target_prefix_end]]
    suffix = [np.array(joints, dtype=float).copy() for joints in donor_movelist[donor_suffix_start:]]
    if not prefix or not suffix:
        raise ValueError("Empty target prefix or donor suffix for donor carry splice")

    bridge = builder.make_joint_bridge(
        prefix[-1],
        suffix[0],
        builder.DONOR_BRIDGE_INTERPOLATION_STEPS,
    )
    previous_roll = float(prefix[-1][WRIST_ROLL_IDX])
    for idx, joints in enumerate(bridge):
        bridge[idx] = clamp_wrist_roll_delta(
            joints,
            previous_roll,
            DONOR_CARRY_SPLICE_MAX_WRIST_ROLL_DELTA_DEG,
        )
        previous_roll = float(bridge[idx][WRIST_ROLL_IDX])

    rate_limited_suffix_waypoints = min(
        DONOR_CARRY_SPLICE_RATE_LIMIT_SUFFIX_WAYPOINTS,
        len(suffix),
    )
    for idx, joints in enumerate(suffix):
        max_delta_deg = (
            DONOR_CARRY_SPLICE_MAX_WRIST_ROLL_DELTA_DEG
            if idx < rate_limited_suffix_waypoints
            else 360.0
        )
        suffix[idx] = clamp_wrist_roll_delta(
            joints,
            previous_roll,
            max_delta_deg,
        )
        previous_roll = float(suffix[idx][WRIST_ROLL_IDX])

    movelist = prefix + bridge + suffix
    new_suffix_start = len(prefix) + len(bridge)
    target_prefix_fk_error = builder.prefix_fk_error(target_metrics)
    donor_suffix_fk_error = builder.finite_float(donor_metrics.get("max_fk_error"), 0.0)
    combined_fk_error = max(target_prefix_fk_error, donor_suffix_fk_error)

    traj_metrics = {
        **donor_metrics,
        "max_fk_error": combined_fk_error,
        "fk_error_events": [
            event
            for event in target_metrics.get("fk_error_events", [])
            if isinstance(event, dict)
            and not str(event.get("stage", "")).endswith("_lower_place")
            and not str(event.get("stage", "")).endswith("release_settle")
            and not str(event.get("stage", "")).endswith("retreat")
            and not str(event.get("stage", "")).endswith("return_home")
            and not str(event.get("stage", "")).endswith("return_above_home")
            and not str(event.get("stage", "")).endswith("_far_lift_after_release")
        ] + list(donor_metrics.get("fk_error_events", [])),
        "slow_waypoint_indices": builder.shifted_slow_waypoint_indices(
            donor_metrics,
            donor_suffix_start,
            new_suffix_start,
        ),
        "segments": {},
        "donor_carry_splice_test": {
            "enabled": True,
            "test_mode": DONOR_CARRY_SPLICE_TEST_MODE,
            "target_move": move_key,
            "donor_move": donor["key"],
            "donor_lookup_path": str(donor_lookup_path),
            "target_grasp_offset": np.array(entry["source_grasp_offset"], dtype=float).copy(),
            "target_place_offset": np.array(entry["selected_place_offset"], dtype=float).copy(),
            "donor_grasp_offset": np.array(donor_entry["source_grasp_offset"], dtype=float).copy(),
            "donor_place_offset": np.array(donor_entry["selected_place_offset"], dtype=float).copy(),
            "target_destination_above_place_bounds": list(target_above_bounds),
            "donor_destination_above_place_bounds": list(donor_above_bounds),
            "target_prefix_end": target_prefix_end,
            "donor_suffix_start": donor_suffix_start,
            "forced_target_idx": DONOR_CARRY_SPLICE_FORCE_TARGET_IDX,
            "forced_donor_idx": DONOR_CARRY_SPLICE_FORCE_DONOR_IDX,
            "bridge_interpolation_steps": builder.DONOR_BRIDGE_INTERPOLATION_STEPS,
            "bridge_inserted_waypoints": len(bridge),
            "bridge_waypoints": [joints.copy() for joints in bridge],
            "max_wrist_roll_delta_deg": DONOR_CARRY_SPLICE_MAX_WRIST_ROLL_DELTA_DEG,
            "rate_limited_suffix_waypoints": rate_limited_suffix_waypoints,
            "late_buffer_waypoints": DONOR_CARRY_SPLICE_LATE_BUFFER_WAYPOINTS,
            "target_prefix_last_joints": prefix[-1].copy(),
            "donor_suffix_first_joints": suffix[0].copy(),
            "target_prefix_fk_error": target_prefix_fk_error,
            "donor_suffix_fk_error": donor_suffix_fk_error,
            "combined_fk_error": combined_fk_error,
        },
    }
    return {
        "movelist": movelist,
        "closeidx": target_closeidx,
        "traj_metrics": traj_metrics,
        "active_place_offset": np.array(donor_entry["selected_place_offset"], dtype=float).copy(),
        "lift_height": target_lift_height,
    }


def run_saved_entry(move_key, lookup, entry, output_dir, lookup_dir, test_mode=None):
    from_square, to_square = parse_move_key(move_key)
    metrics = entry.get("metrics", {})
    grasp_offset = np.array(entry["source_grasp_offset"], dtype=float)
    place_offset = np.array(entry["selected_place_offset"], dtype=float)
    lift_height = saved_lift_height_for_entry(move_key, lookup, entry, to_square)
    lower_place_path_bias = metrics.get("lower_place_path_bias")
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
    bridge_rebuild_error = None
    neighbor_prefix_bias_rebuild_error = None
    donor_carry_splice_metadata = None
    saved_bridge_fallback = metrics.get("bridge_fallback")
    saved_neighbor_prefix_bias_fallback = metrics.get("neighbor_prefix_bias_fallback")
    if test_mode == DONOR_CARRY_SPLICE_TEST_MODE:
        custom_override = build_donor_carry_splice_override(
            lookup_dir,
            move_key,
            lookup,
            entry,
        )
        trajectory_override = custom_override
        place_offset = custom_override["active_place_offset"]
        lift_height = custom_override["lift_height"]
        replay_mode = DONOR_CARRY_SPLICE_TEST_MODE
        donor_carry_splice_metadata = custom_override["traj_metrics"].get(
            "donor_carry_splice_test"
        )
    elif isinstance(saved_neighbor_prefix_bias_fallback, dict):
        trajectory_override, reject_reason = builder.build_saved_neighbor_prefix_bias_override(
            from_square,
            to_square,
            grasp_offset,
            place_offset,
            saved_neighbor_prefix_bias_fallback,
            lift_height=lift_height,
        )
        if trajectory_override is None:
            neighbor_prefix_bias_rebuild_error = reject_reason
            replay_mode = "direct_fallback_after_saved_neighbor_prefix_bias_rebuild_failed"
        else:
            replay_mode = "saved_neighbor_prefix_bias"
    elif isinstance(saved_bridge_fallback, dict):
        trajectory_override, reject_reason = builder.build_saved_bridge_override(
            from_square,
            to_square,
            grasp_offset,
            place_offset,
            saved_bridge_fallback,
            lift_height=lift_height,
        )
        if trajectory_override is None:
            bridge_rebuild_error = reject_reason
            replay_mode = "direct_fallback_after_saved_bridge_rebuild_failed"
        else:
            replay_mode = "saved_bridge"
    elif saved_donor:
        trajectory_override, reject_reason = builder.build_bridge_override_for_move(
            from_square,
            to_square,
            grasp_offset,
            place_offset,
            lift_height=lift_height,
        )
        if trajectory_override is None:
            bridge_rebuild_error = reject_reason
            replay_mode = "direct_fallback_after_bridge_rebuild_failed"
        else:
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
            lift_height=lift_height,
            lower_place_path_bias=lower_place_path_bias,
        )
        result["score"] = sim.score_place_result(result)
        if test_mode == DONOR_CARRY_SPLICE_TEST_MODE:
            if donor_carry_splice_metadata is not None:
                result["trajectory_fallback_source"] = donor_carry_splice_metadata.get(
                    "donor_move"
                )
        elif trajectory_override is not None:
            result = builder.annotate_bridge_result(result, trajectory_override)
    finally:
        sim.close_video_context(video_context)
        p.removeState(world["state_id"])

    return {
        "move_key": move_key,
        "test_mode": test_mode,
        "replay_mode": replay_mode,
        "saved_donor": saved_donor,
        "bridge_rebuild_error": bridge_rebuild_error,
        "neighbor_prefix_bias_rebuild_error": neighbor_prefix_bias_rebuild_error,
        "replayed_donor": result.get("trajectory_fallback_source"),
        "neighbor_prefix_bias_fallback": result.get("neighbor_prefix_bias_fallback"),
        "lower_place_path_bias": result.get("lower_place_path_bias"),
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
        "donor_carry_splice_test": donor_carry_splice_metadata,
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
        "--test-mode",
        choices=(DONOR_CARRY_SPLICE_TEST_MODE,),
        default=None,
        help="Run a narrow test-only replay path instead of the normal saved-entry replay.",
    )
    parser.add_argument(
        "--report-donor-carry-splice-scores",
        action="store_true",
        help="Print end-effector continuity scores for a1_to_f1 vs b1_to_f1 without running simulation.",
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
        lookup_path, lookup, entry = load_lookup_entry(lookup_dir, move_key)
        if args.report_donor_carry_splice_scores:
            report = donor_carry_splice_score_report(
                lookup_dir,
                move_key,
                lookup,
                entry,
            )
            print(json.dumps(builder.json_safe(report), indent=2, sort_keys=True))
            summary["moves"].append(report)
            continue
        print(f"\n=== verifying {move_key} from {lookup_path} ===", flush=True)
        row = run_saved_entry(
            move_key,
            lookup,
            entry,
            output_dir,
            lookup_dir,
            test_mode=args.test_mode,
        )
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

    summary["verified_count"] = sum(
        bool(row.get("verified_success", False)) for row in summary["moves"]
    )
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
