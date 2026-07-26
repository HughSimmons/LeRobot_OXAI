"""Isolated full-search test for b8 -> f4 with a -180 degree home pan."""

import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pybullet as p


REFERENCE_COMMIT = "de1d3e9"
MOVE_KEY = "b8_to_f4"
SOURCE_SQUARE = "b8"
TARGET_SQUARE = "f4"
LIFT_HEIGHT_M = 0.07
SHOULDER_PAN_HOME_DEG = -180.0

SCRIPT_PATH = Path(__file__).resolve()
LOWLEVEL_DIR = SCRIPT_PATH.parents[1]
PROJECT_ROOT = LOWLEVEL_DIR.parent
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
RUN_DIR = SCRIPT_PATH.parent / "outputs" / f"b8_f4_home_pan_m180_{RUN_STAMP}"
LOOKUP_PATH = RUN_DIR / "b8_non_h_reverse_move_lookup.json"
SUCCESS_MAP_PATH = RUN_DIR / "b8_non_h_reverse_move_lookup_success_map.svg"
SUMMARY_PATH = RUN_DIR / "experiment_summary.json"
BASELINE_LOOKUP_PATH = LOWLEVEL_DIR / "b8_non_h_reverse_move_lookup_temp.json"


def git_output(*args):
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def load_baseline():
    if not BASELINE_LOOKUP_PATH.exists():
        return {
            "lookup_path": str(BASELINE_LOOKUP_PATH),
            "available": False,
        }

    lookup = json.loads(BASELINE_LOOKUP_PATH.read_text(encoding="utf-8"))
    entry = lookup.get("moves", {}).get(MOVE_KEY)
    if not isinstance(entry, dict):
        return {
            "lookup_path": str(BASELINE_LOOKUP_PATH),
            "available": False,
        }

    metrics = entry.get("metrics") or {}
    return {
        "lookup_path": str(BASELINE_LOOKUP_PATH),
        "available": True,
        "source_grasp_offset": entry.get("source_grasp_offset"),
        "selected_place_offset": entry.get("selected_place_offset"),
        "trajectory_fk_error": metrics.get("trajectory_fk_error"),
        "xy_error": metrics.get("xy_error"),
        "final_tilt_deg": metrics.get("final_tilt_deg"),
        "pickup_success": metrics.get("pickup_success"),
        "score": metrics.get("score"),
    }


def classify_attempt(attempt):
    reason = attempt.get("reject_reason")
    metrics = attempt.get("metrics") or {}
    if reason == "trajectory_fk_error_too_large":
        return "fk_generation"
    if reason == "pickup_failed" or metrics.get("pickup_success") is False:
        return "pickup"
    if metrics.get("premature_drop"):
        return "transport"
    if reason == "place_offset_xy_limit_exceeded":
        return "placement_correction"
    if reason or attempt.get("success") is False:
        return "final_validation"
    return "success"


def summarize_attempts(result_record):
    selection = (result_record or {}).get("grasp_selection") or {}
    attempts = selection.get("attempts") or []
    stage_counts = Counter(classify_attempt(attempt) for attempt in attempts)
    reject_counts = Counter(
        str(attempt.get("reject_reason"))
        for attempt in attempts
        if attempt.get("reject_reason") is not None
    )
    return {
        "attempt_count": len(attempts),
        "stage_counts": dict(sorted(stage_counts.items())),
        "reject_reason_counts": dict(sorted(reject_counts.items())),
        "attempts": attempts,
    }


def compact_runtime_result(result):
    fields = (
        "pickup_success",
        "premature_drop",
        "trajectory_fk_error",
        "trajectory_valid",
        "xy_error",
        "z_error",
        "final_tilt_deg",
        "reject_reason",
        "place_offset",
        "score",
        "trajectory_fallback_source",
        "trajectory_fallback_reason",
        "trajectory_fallback_donor_distance",
    )
    return {
        field: result.get(field)
        for field in fields
        if field in result
    }


def write_json(path, value, builder):
    path.write_text(
        json.dumps(builder.json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=False)

    # Disable the broad, uncommitted override mechanism. This experiment patches
    # only the two modules that own trajectory and simulator home state.
    os.environ.pop("HOME_SHOULDER_PAN_OVERRIDE", None)
    os.environ.pop("DISABLE_WRIST_ROLL_SEED", None)
    os.environ["SOURCE_SQUARES"] = SOURCE_SQUARE
    os.environ["TARGET_MOVES"] = MOVE_KEY
    os.environ["LOOKUP_LIFT_HEIGHT_OVERRIDE"] = str(LIFT_HEIGHT_M)
    os.environ["DONOR_LOOKUP_DIR"] = str(LOWLEVEL_DIR)

    if str(LOWLEVEL_DIR) not in sys.path:
        sys.path.insert(0, str(LOWLEVEL_DIR))

    import chess_traj
    import multisim_chess_fast as sim

    original_traj_home = np.array(chess_traj.home, dtype=float)
    original_sim_home = np.array(sim.home, dtype=float)
    test_home = original_traj_home.copy()
    test_home[0] = SHOULDER_PAN_HOME_DEG

    chess_traj.home = test_home.copy()
    chess_traj.homexyz = chess_traj.kinematics.forward_kinematics(
        chess_traj.home
    )[:3, 3]
    sim.home = test_home.copy()
    sim.home_rad = np.deg2rad(sim.home)

    import build_general_nonh_reverse_lookup as builder
    import verify_nonh_lookup_moves as verifier

    original_pickupmove_traj_with_metrics = builder.pickupmove_traj_with_metrics
    donor_regeneration_events = []

    def pickupmove_with_source_home(
        from_square,
        *args,
        **kwargs,
    ):
        if from_square == SOURCE_SQUARE:
            return original_pickupmove_traj_with_metrics(
                from_square,
                *args,
                **kwargs,
            )

        target_home = np.array(chess_traj.home, dtype=float)
        donor_regeneration_events.append(
            {
                "from_square": from_square,
                "target_home_before_deg": target_home.tolist(),
                "donor_home_deg": original_traj_home.tolist(),
            }
        )
        try:
            chess_traj.home = original_traj_home.copy()
            chess_traj.homexyz = chess_traj.kinematics.forward_kinematics(
                chess_traj.home
            )[:3, 3]
            return original_pickupmove_traj_with_metrics(
                from_square,
                *args,
                **kwargs,
            )
        finally:
            chess_traj.home = target_home
            chess_traj.homexyz = chess_traj.kinematics.forward_kinematics(
                chess_traj.home
            )[:3, 3]

    builder.pickupmove_traj_with_metrics = pickupmove_with_source_home
    original_run_direct_move = builder.run_direct_move_in_fresh_world
    original_run_bridged_move = builder.run_bridged_direct_move_in_fresh_world
    raw_direct_attempts = []
    raw_bridge_attempts = []

    def record_direct_move(
        from_square,
        to_square,
        grasp_offset,
        place_offset,
        *args,
        **kwargs,
    ):
        result = original_run_direct_move(
            from_square,
            to_square,
            grasp_offset,
            place_offset,
            *args,
            **kwargs,
        )
        raw_direct_attempts.append(
            {
                "from_square": from_square,
                "to_square": to_square,
                "grasp_offset": np.array(grasp_offset, dtype=float).tolist(),
                "place_offset": np.array(place_offset, dtype=float).tolist(),
                "result": compact_runtime_result(result),
            }
        )
        return result

    def record_bridged_move(
        from_square,
        to_square,
        grasp_offset,
        place_offset,
        *args,
        **kwargs,
    ):
        result = original_run_bridged_move(
            from_square,
            to_square,
            grasp_offset,
            place_offset,
            *args,
            **kwargs,
        )
        raw_bridge_attempts.append(
            {
                "from_square": from_square,
                "to_square": to_square,
                "grasp_offset": np.array(grasp_offset, dtype=float).tolist(),
                "place_offset": np.array(place_offset, dtype=float).tolist(),
                "result": compact_runtime_result(result),
            }
        )
        return result

    builder.run_direct_move_in_fresh_world = record_direct_move
    builder.run_bridged_direct_move_in_fresh_world = record_bridged_move
    builder.OUTPUT_PATH = LOOKUP_PATH
    builder.SUCCESS_MAP_PATH = SUCCESS_MAP_PATH
    builder.DONOR_LOOKUP_DIR = LOWLEVEL_DIR
    builder.REUSE_EXISTING_SUCCESSFUL_MOVES = False
    builder.FALLBACK_GRASP_GRID_ENABLED = True
    builder.EXPANDED_GRASP_GRID_ALL_MOVES = True

    _, head_commit, _ = git_output("rev-parse", "HEAD")
    _, dirty_paths, _ = git_output("diff", "--name-only")
    builder_matches_code, _, _ = git_output(
        "diff",
        "--quiet",
        REFERENCE_COMMIT,
        "--",
        "lowlevel/build_general_nonh_reverse_lookup.py",
    )

    summary = {
        "test_mode": "b8_f4_home_shoulder_pan_m180_full_search",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "reference_commit": REFERENCE_COMMIT,
        "head_commit": head_commit,
        "reference_checks": {
            "builder_matches_reference_commit": builder_matches_code == 0,
            "normal_wrist_roll_seeding_enabled": (
                "DISABLE_WRIST_ROLL_SEED" not in os.environ
            ),
            "broad_home_override_disabled": (
                "HOME_SHOULDER_PAN_OVERRIDE" not in os.environ
            ),
        },
        "dirty_paths_before_run": dirty_paths.splitlines() if dirty_paths else [],
        "configuration": {
            "source_square": SOURCE_SQUARE,
            "target_square": TARGET_SQUARE,
            "target_move": MOVE_KEY,
            "lift_height_m": LIFT_HEIGHT_M,
            "original_trajectory_home_deg": original_traj_home.tolist(),
            "original_simulator_home_deg": original_sim_home.tolist(),
            "test_home_deg": test_home.tolist(),
            "shoulder_pan_home_deg": SHOULDER_PAN_HOME_DEG,
            "donor_trajectory_home_deg": original_traj_home.tolist(),
            "donor_regeneration_uses_default_home": True,
            "simulator_remains_at_test_home_during_donor_regeneration": True,
            "reuse_existing_successful_moves": False,
            "expanded_grasp_grid_all_moves": True,
            "fallback_grasp_grid_enabled": True,
            "fallback_grasp_grid_stop_on_first_success": (
                builder.FALLBACK_GRASP_GRID_STOP_ON_FIRST_SUCCESS
            ),
            "donor_lookup_dir": str(LOWLEVEL_DIR),
        },
        "baseline_original_home": load_baseline(),
        "output_lookup_path": str(LOOKUP_PATH),
        "output_success_map_path": str(SUCCESS_MAP_PATH),
    }

    try:
        lookup, report = builder.build_lookup()
        lookup["metadata"].update(
            {
                "experiment_test_mode": summary["test_mode"],
                "experiment_reference_commit": REFERENCE_COMMIT,
                "experiment_head_commit": head_commit,
                "experiment_home_deg": test_home.tolist(),
                "experiment_shoulder_pan_home_deg": SHOULDER_PAN_HOME_DEG,
                "experiment_live_lookup_untouched": True,
            }
        )
        write_json(LOOKUP_PATH, lookup, builder)
        builder.write_success_map(lookup)
        builder.print_build_report(report)

        entry = lookup.get("moves", {}).get(MOVE_KEY)
        failure_rows = report.get(SOURCE_SQUARE, {}).get("failures", [])
        result_record = entry if entry is not None else (
            failure_rows[0] if failure_rows else {}
        )
        summary["builder_success"] = entry is not None
        summary["builder_entry"] = entry
        summary["builder_failures"] = failure_rows
        summary["grasp_attempt_summary"] = summarize_attempts(result_record)
        summary["donor_regeneration_events"] = donor_regeneration_events
        summary["raw_direct_attempts"] = raw_direct_attempts
        summary["raw_bridge_attempts"] = raw_bridge_attempts

        if entry is not None:
            verification_dir = RUN_DIR / "verification"
            verification_dir.mkdir(parents=True, exist_ok=False)
            verification = verifier.run_saved_entry(
                MOVE_KEY,
                lookup,
                entry,
                verification_dir,
                RUN_DIR,
                test_mode=None,
            )
            verification["lookup_path"] = str(LOOKUP_PATH)
            summary["strict_verification"] = verification
        else:
            summary["strict_verification"] = None
    except Exception as exc:
        summary["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        summary["donor_regeneration_events"] = donor_regeneration_events
        summary["raw_direct_attempts"] = raw_direct_attempts
        summary["raw_bridge_attempts"] = raw_bridge_attempts
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(SUMMARY_PATH, summary, builder)
        if p.isConnected():
            p.disconnect()
        print(f"\nExperiment output: {RUN_DIR}")
        print(f"Experiment summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
