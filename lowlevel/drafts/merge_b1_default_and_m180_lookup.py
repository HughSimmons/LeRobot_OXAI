#!/usr/bin/env python3
"""Reproducibly merge b1 default-home and m180-home lookup successes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOWLEVEL_DIR = PROJECT_ROOT / "lowlevel"

DEFAULT_BASELINE = (
    LOWLEVEL_DIR
    / "donor_snapshots"
    / "nonh_20260720_201004"
    / "b1_non_h_reverse_move_lookup.json"
)
DEFAULT_SUPPLEMENT = LOWLEVEL_DIR / "b1_non_h_reverse_move_lookup.json"
DEFAULT_OUTPUT = LOWLEVEL_DIR / "b1_non_h_reverse_move_lookup.json"
DEFAULT_MAP_OUTPUT = LOWLEVEL_DIR / "b1_non_h_reverse_move_lookup_success_map.svg"

FILES = "abcdefgh"
RANKS = range(1, 9)
SOURCE_SQUARE = "b1"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    if not isinstance(data.get("moves"), dict):
        raise ValueError(f"Expected moves object in {path}")
    return data


def successful_moves(lookup: dict) -> dict:
    return {
        key: move
        for key, move in lookup.get("moves", {}).items()
        if isinstance(move, dict) and move.get("success")
    }


def default_home_from(*lookups: dict) -> list[float]:
    for lookup in lookups:
        home = lookup.get("metadata", {}).get("default_trajectory_home_joints_deg")
        if home is not None:
            return home
    return [
        96.92307692307692,
        -107.86813186813187,
        97.36263736263736,
        65.18681318681318,
        -29.846153846153847,
        4.62962962962963,
    ]


def m180_home_from(supplement: dict) -> list[float] | None:
    for move in successful_moves(supplement).values():
        home = move.get("metrics", {}).get("trajectory_home_joints_deg")
        if home is not None:
            return home
    return None


def move_destination(key: str, move: dict) -> str:
    return move.get("to_square") or key.split("_to_", 1)[1]


def family_for_destination(to_square: str) -> str:
    return to_square[0]


def stamp_move(
    move: dict,
    *,
    home_joints_deg: list[float],
    lift_override,
    home_policy: str,
    provenance: str,
) -> dict:
    stamped = deepcopy(move)
    metrics = stamped.setdefault("metrics", {})
    metrics["trajectory_home_joints_deg"] = home_joints_deg
    metrics["lookup_lift_height_override"] = lift_override
    metrics["trajectory_home_policy"] = home_policy
    metrics["merge_provenance"] = provenance
    return stamped


def build_expected_keys() -> set[str]:
    return {
        f"{SOURCE_SQUARE}_to_{file}{rank}"
        for file in FILES
        for rank in RANKS
        if f"{file}{rank}" != SOURCE_SQUARE
    }


def merge_lookups(baseline: dict, supplement: dict) -> tuple[dict, dict]:
    baseline_successes = successful_moves(baseline)
    supplement_successes = successful_moves(supplement)
    default_home = default_home_from(supplement, baseline)
    m180_home = m180_home_from(supplement)

    merged_moves = {}
    source_counts = {
        "baseline_default_home": 0,
        "supplement_m180_home": 0,
    }
    selection = {}

    for key, move in sorted(baseline_successes.items()):
        to_square = move_destination(key, move)
        if family_for_destination(to_square) in {"a", "b", "c", "d", "e"}:
            merged_moves[key] = stamp_move(
                move,
                home_joints_deg=default_home,
                lift_override=None,
                home_policy="default_home_restored_from_snapshot",
                provenance=str(DEFAULT_BASELINE),
            )
            source_counts["baseline_default_home"] += 1
            selection[key] = "baseline_default_home"

    for key, move in sorted(supplement_successes.items()):
        to_square = move_destination(key, move)
        if family_for_destination(to_square) in {"f", "g", "h"}:
            metrics = move.get("metrics", {})
            home = metrics.get("trajectory_home_joints_deg") or m180_home
            lift_override = metrics.get("lookup_lift_height_override")
            if lift_override is None:
                lift_override = supplement.get("metadata", {}).get(
                    "lookup_lift_height_override"
                )
            merged_moves[key] = stamp_move(
                move,
                home_joints_deg=home,
                lift_override=lift_override,
                home_policy="alternate_home_m180",
                provenance=str(DEFAULT_SUPPLEMENT),
            )
            source_counts["supplement_m180_home"] += 1
            selection[key] = "supplement_m180_home"

    expected_keys = build_expected_keys()
    missing_keys = sorted(expected_keys - set(merged_moves))

    metadata = deepcopy(supplement.get("metadata", {}))
    metadata.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "lowlevel/drafts/merge_b1_default_and_m180_lookup.py",
            "from_square": SOURCE_SQUARE,
            "source_squares": [SOURCE_SQUARE],
            "merge_policy": (
                "default-home snapshot for a-e targets; "
                "alternate shoulder-pan -180 supplement for f/g/h targets"
            ),
            "merge_baseline_path": str(DEFAULT_BASELINE),
            "merge_supplement_path": str(DEFAULT_SUPPLEMENT),
            "merge_source_counts": source_counts,
            "merge_missing_keys": missing_keys,
            "merge_selection_by_move": selection,
            "default_trajectory_home_joints_deg": default_home,
            "alternate_home_m180_joints_deg": m180_home,
        }
    )

    merged = {"metadata": metadata, "moves": merged_moves}
    summary = {
        "merged_count": len(merged_moves),
        "success_count": len(successful_moves(merged)),
        "missing_keys": missing_keys,
        "source_counts": source_counts,
    }
    return merged, summary


def write_success_map(output: Path, map_output: Path) -> None:
    script = LOWLEVEL_DIR / "visualize_lookup_success.py"
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            str(output),
            "--output",
            str(map_output),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--supplement", type=Path, default=DEFAULT_SUPPLEMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--success-map", type=Path, default=DEFAULT_MAP_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_json(args.baseline)
    supplement = load_json(args.supplement)
    merged, summary = merge_lookups(baseline, supplement)

    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output.exists():
        backup = args.output.with_name(f"{args.output.name}.bak_remerge_{stamp}")
        shutil.copy2(args.output, backup)
        print(f"Backed up output to {backup}")
    if args.success_map.exists():
        map_backup = args.success_map.with_name(
            f"{args.success_map.name}.bak_remerge_{stamp}"
        )
        shutil.copy2(args.success_map, map_backup)
        print(f"Backed up success map to {map_backup}")

    args.output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote merged lookup to {args.output}")
    write_success_map(args.output, args.success_map)


if __name__ == "__main__":
    main()
