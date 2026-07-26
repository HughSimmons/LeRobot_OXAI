"""Run an isolated full-search and strict replay for b8 -> b4."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pybullet as p


REFERENCE_COMMIT = "de1d3e9"
MOVE_KEY = "b8_to_b4"
SOURCE_SQUARE = "b8"
TARGET_SQUARE = "b4"

SCRIPT_PATH = Path(__file__).resolve()
LOWLEVEL_DIR = SCRIPT_PATH.parents[1]
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
RUN_DIR = SCRIPT_PATH.parent / "outputs" / f"b8_b4_default_home_{RUN_STAMP}"
LOOKUP_PATH = RUN_DIR / "b8_non_h_reverse_move_lookup.json"
SUCCESS_MAP_PATH = RUN_DIR / "b8_non_h_reverse_move_lookup_success_map.svg"
SUMMARY_PATH = RUN_DIR / "experiment_summary.json"


def write_json(path, value, builder):
    path.write_text(
        json.dumps(builder.json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=False)

    os.environ.pop("HOME_SHOULDER_PAN_OVERRIDE", None)
    os.environ.pop("DISABLE_WRIST_ROLL_SEED", None)
    os.environ.pop("LOOKUP_LIFT_HEIGHT_OVERRIDE", None)
    os.environ["SOURCE_SQUARES"] = SOURCE_SQUARE
    os.environ["TARGET_MOVES"] = MOVE_KEY
    os.environ["DONOR_LOOKUP_DIR"] = str(LOWLEVEL_DIR)

    if str(LOWLEVEL_DIR) not in sys.path:
        sys.path.insert(0, str(LOWLEVEL_DIR))

    import chess_traj
    import multisim_chess_fast as sim
    import build_general_nonh_reverse_lookup as builder
    import verify_nonh_lookup_moves as verifier

    builder.OUTPUT_PATH = LOOKUP_PATH
    builder.SUCCESS_MAP_PATH = SUCCESS_MAP_PATH
    builder.DONOR_LOOKUP_DIR = LOWLEVEL_DIR
    builder.REUSE_EXISTING_SUCCESSFUL_MOVES = False
    builder.FALLBACK_GRASP_GRID_ENABLED = True
    builder.EXPANDED_GRASP_GRID_ALL_MOVES = True

    expected_lift_height = builder.lookup_lift_height_for_square(TARGET_SQUARE)
    summary = {
        "test_mode": "b8_b4_default_home_full_search",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "reference_commit": REFERENCE_COMMIT,
        "configuration": {
            "source_square": SOURCE_SQUARE,
            "target_square": TARGET_SQUARE,
            "target_move": MOVE_KEY,
            "trajectory_home_deg": np.array(chess_traj.home, dtype=float).tolist(),
            "simulator_home_deg": np.array(sim.home, dtype=float).tolist(),
            "lift_height_m": expected_lift_height,
            "lookup_lift_height_override": builder.LOOKUP_LIFT_HEIGHT_OVERRIDE,
            "reuse_existing_successful_moves": False,
            "expanded_grasp_grid_all_moves": True,
            "fallback_grasp_grid_enabled": True,
            "fallback_grasp_grid_stop_on_first_success": (
                builder.FALLBACK_GRASP_GRID_STOP_ON_FIRST_SUCCESS
            ),
            "donor_lookup_dir": str(LOWLEVEL_DIR),
        },
        "output_lookup_path": str(LOOKUP_PATH),
        "output_success_map_path": str(SUCCESS_MAP_PATH),
    }

    try:
        lookup, report = builder.build_lookup()
        lookup["metadata"].update(
            {
                "experiment_test_mode": summary["test_mode"],
                "experiment_reference_commit": REFERENCE_COMMIT,
                "experiment_default_home_verified": True,
                "experiment_live_lookup_untouched": True,
            }
        )
        write_json(LOOKUP_PATH, lookup, builder)
        builder.write_success_map(lookup)
        builder.print_build_report(report)

        entry = lookup.get("moves", {}).get(MOVE_KEY)
        summary["builder_success"] = entry is not None
        summary["builder_entry"] = entry
        summary["builder_report"] = report

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
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(SUMMARY_PATH, summary, builder)
        if p.isConnected():
            p.disconnect()
        print(f"\nExperiment output: {RUN_DIR}")
        print(f"Experiment summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
