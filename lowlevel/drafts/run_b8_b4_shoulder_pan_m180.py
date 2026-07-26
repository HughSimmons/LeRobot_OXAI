"""Run b8 -> b4 with a -180 degree shoulder-pan delta from default home."""

import hashlib
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
SHOULDER_PAN_HOME_DELTA_DEG = -180.0

SCRIPT_PATH = Path(__file__).resolve()
LOWLEVEL_DIR = SCRIPT_PATH.parents[1]
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
RUN_DIR = SCRIPT_PATH.parent / "outputs" / f"b8_b4_shoulder_pan_m180_{RUN_STAMP}"
LOOKUP_PATH = RUN_DIR / "b8_non_h_reverse_move_lookup.json"
SUCCESS_MAP_PATH = RUN_DIR / "b8_non_h_reverse_move_lookup_success_map.svg"
SUMMARY_PATH = RUN_DIR / "experiment_summary.json"
LIVE_LOOKUP_PATH = LOWLEVEL_DIR / "b8_non_h_reverse_move_lookup.json"


def file_sha256(path):
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value, builder):
    path.write_text(
        json.dumps(builder.json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    live_lookup_hash_before = file_sha256(LIVE_LOOKUP_PATH)

    os.environ["SOURCE_SQUARES"] = SOURCE_SQUARE
    os.environ["TARGET_MOVES"] = MOVE_KEY
    os.environ["DONOR_LOOKUP_DIR"] = str(LOWLEVEL_DIR)
    os.environ.pop("LOOKUP_HOME_SHOULDER_PAN_OVERRIDE_DEG", None)
    os.environ["LOOKUP_HOME_SHOULDER_PAN_DELTA_DEG"] = str(
        SHOULDER_PAN_HOME_DELTA_DEG
    )
    os.environ.pop("LOOKUP_LIFT_HEIGHT_OVERRIDE", None)

    if str(LOWLEVEL_DIR) not in sys.path:
        sys.path.insert(0, str(LOWLEVEL_DIR))

    import build_general_nonh_reverse_lookup as builder
    import chess_traj
    import verify_nonh_lookup_moves as verifier

    builder.OUTPUT_PATH = LOOKUP_PATH
    builder.SUCCESS_MAP_PATH = SUCCESS_MAP_PATH
    builder.DONOR_LOOKUP_DIR = LOWLEVEL_DIR
    builder.REUSE_EXISTING_SUCCESSFUL_MOVES = False
    builder.FALLBACK_GRASP_GRID_ENABLED = True
    builder.EXPANDED_GRASP_GRID_ALL_MOVES = True

    requested_home = builder.recorded_lookup_home_joints_for_source(SOURCE_SQUARE)
    expected_home = chess_traj.DEFAULT_HOME.copy()
    expected_home[0] += SHOULDER_PAN_HOME_DELTA_DEG
    if not np.allclose(requested_home, expected_home):
        raise RuntimeError(
            f"Unexpected override home: {requested_home}; expected {expected_home}"
        )

    summary = {
        "test_mode": "b8_b4_shoulder_pan_m180_full_search",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "reference_commit": REFERENCE_COMMIT,
        "configuration": {
            "source_square": SOURCE_SQUARE,
            "target_square": TARGET_SQUARE,
            "target_move": MOVE_KEY,
            "default_trajectory_home_deg": chess_traj.DEFAULT_HOME.copy(),
            "requested_trajectory_home_deg": requested_home,
            "shoulder_pan_home_delta_deg": SHOULDER_PAN_HOME_DELTA_DEG,
            "resolved_shoulder_pan_home_deg": float(requested_home[0]),
            "lift_height_m": builder.lookup_lift_height_for_square(TARGET_SQUARE),
            "reuse_existing_successful_moves": False,
            "expanded_grasp_grid_all_moves": True,
            "fallback_grasp_grid_enabled": True,
            "fallback_grasp_grid_stop_on_first_success": (
                builder.FALLBACK_GRASP_GRID_STOP_ON_FIRST_SUCCESS
            ),
            "donor_home_policy": "legacy_default_home",
            "donor_lookup_dir": str(LOWLEVEL_DIR),
        },
        "output_lookup_path": str(LOOKUP_PATH),
        "output_success_map_path": str(SUCCESS_MAP_PATH),
        "live_lookup_path": str(LIVE_LOOKUP_PATH),
        "live_lookup_sha256_before": live_lookup_hash_before,
    }

    try:
        lookup, report = builder.build_lookup()
        lookup["metadata"].update(
            {
                "experiment_test_mode": summary["test_mode"],
                "experiment_reference_commit": REFERENCE_COMMIT,
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
            summary["strict_verification"] = verifier.run_saved_entry(
                MOVE_KEY,
                lookup,
                entry,
                verification_dir,
                RUN_DIR,
                test_mode=None,
            )
        else:
            summary["strict_verification"] = None
    except Exception as exc:
        summary["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        summary["live_lookup_sha256_after"] = file_sha256(LIVE_LOOKUP_PATH)
        summary["live_lookup_unchanged"] = (
            summary["live_lookup_sha256_before"]
            == summary["live_lookup_sha256_after"]
        )
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(SUMMARY_PATH, summary, builder)
        if p.isConnected():
            p.disconnect()
        print(f"\nExperiment output: {RUN_DIR}")
        print(f"Experiment summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
