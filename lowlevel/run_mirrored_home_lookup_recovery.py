#!/usr/bin/env python3
"""Run rank-reflected mirrored-home lookup recovery safely from the CLI."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


LOWLEVEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LOWLEVEL_DIR.parent
FILES = "abcdefgh"
RANKS = "12345678"
SOURCE_ORDER = tuple(f"{file}{rank}" for file in "fedcba" for rank in range(8, 0, -1))
TARGET_SQUARES = tuple(f"{file}{rank}" for file in FILES for rank in RANKS)
DEFAULT_HOME0_DEG = 96.92307692307692
MIRRORED_HOME0_DEG = -83.07692307692308
MIRRORED_HOME_DELTA_DEG = -180.0


@dataclass(frozen=True)
class PlannedMove:
    move_key: str
    from_square: str
    to_square: str
    mirror_move_key: str
    mirror_from_square: str
    mirror_to_square: str
    mirror_counterpart_home_policy: str
    lift_override: float | None
    eligible: bool
    reason: str


def parse_square_list(raw_values):
    if not raw_values:
        return None
    squares = []
    for raw in raw_values:
        squares.extend(raw.replace(",", " ").split())
    parsed = tuple(square.strip() for square in squares if square.strip())
    for square in parsed:
        validate_square(square)
    return parsed


def parse_move_list(raw_values):
    if not raw_values:
        return None
    moves = []
    for raw in raw_values:
        moves.extend(raw.replace(",", " ").split())
    parsed = tuple(move.strip() for move in moves if move.strip())
    for move in parsed:
        from_square, to_square = split_move_key(move)
        validate_square(from_square)
        validate_square(to_square)
    return parsed


def validate_square(square: str) -> None:
    if (
        not isinstance(square, str)
        or len(square) != 2
        or square[0] not in FILES
        or square[1] not in RANKS
    ):
        raise argparse.ArgumentTypeError(
            f"Invalid square {square!r}; use values like a1"
        )


def split_move_key(move_key: str) -> tuple[str, str]:
    parts = move_key.split("_to_")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid move key {move_key!r}; use values like a1_to_h1"
        )
    return parts[0], parts[1]


def move_key(from_square: str, to_square: str) -> str:
    return f"{from_square}_to_{to_square}"


def mirror_square(square: str) -> str:
    return f"{square[0]}{9 - int(square[1])}"


def lookup_path_for_source(source_square: str, lookup_dir: Path) -> Path:
    return lookup_dir / f"{source_square}_non_h_reverse_move_lookup.json"


def success_map_path_for_lookup(lookup_path: Path) -> Path:
    return lookup_path.with_name(f"{lookup_path.stem}_success_map.svg")


def load_lookup(path: Path) -> dict:
    if not path.exists():
        return {"metadata": {}, "moves": {}}
    with path.open("r", encoding="utf-8") as handle:
        lookup = json.load(handle)
    if not isinstance(lookup, dict):
        return {"metadata": {}, "moves": {}}
    if not isinstance(lookup.get("metadata"), dict):
        lookup["metadata"] = {}
    if not isinstance(lookup.get("moves"), dict):
        lookup["moves"] = {}
    return lookup


def move_is_successful(move: dict | None) -> bool:
    if not isinstance(move, dict) or not move.get("success"):
        return False
    metrics = move.get("metrics", {})
    if not isinstance(metrics, dict):
        return False
    return bool(metrics.get("pickup_success", True)) and metrics.get("reject_reason") is None


def home_policy_for_move(move: dict | None) -> str:
    if not move_is_successful(move):
        return "not_success"
    metrics = move.get("metrics", {})
    home = metrics.get("trajectory_home_joints_deg")
    if home is None:
        return "default_home_implicit"
    try:
        home0 = float(home[0])
    except (TypeError, ValueError, IndexError):
        return "unknown_home"
    if abs(home0 - DEFAULT_HOME0_DEG) < 1e-3:
        return "default_home_explicit"
    if abs(home0 - MIRRORED_HOME0_DEG) < 1e-3:
        return "alternate_home_m180"
    return f"other_home_{home0:.6f}"


def move_is_default_home_success(move: dict | None) -> bool:
    return home_policy_for_move(move) in {
        "default_home_implicit",
        "default_home_explicit",
    }


def move_is_mirrored_home_success(move: dict | None) -> bool:
    return home_policy_for_move(move) == "alternate_home_m180"


def lift_override_for_counterpart(move: dict | None) -> float | None:
    if not isinstance(move, dict):
        return None
    metrics = move.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    for field in ("lookup_lift_height_override", "lift_height"):
        value = metrics.get(field)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def expected_move_keys_for_source(source_square: str) -> tuple[str, ...]:
    return tuple(
        move_key(source_square, target_square)
        for target_square in TARGET_SQUARES
        if target_square != source_square
    )


def collect_planned_moves(
    *,
    lookup_dir: Path,
    selected_sources: tuple[str, ...] | None,
    selected_target_moves: tuple[str, ...] | None,
    include_unsupported: bool,
) -> tuple[list[PlannedMove], list[PlannedMove]]:
    lookups = {
        source_square: load_lookup(lookup_path_for_source(source_square, lookup_dir))
        for source_square in SOURCE_ORDER
    }
    target_filter = set(selected_target_moves or ())
    sources = selected_sources or SOURCE_ORDER
    planned = []
    unsupported = []

    for source_square in SOURCE_ORDER:
        if source_square not in sources:
            continue
        candidate_keys = expected_move_keys_for_source(source_square)
        if target_filter:
            candidate_keys = tuple(key for key in candidate_keys if key in target_filter)
        source_moves = lookups[source_square].get("moves", {})
        for key in candidate_keys:
            existing_move = source_moves.get(key)
            if move_is_default_home_success(existing_move):
                continue
            if move_is_mirrored_home_success(existing_move):
                continue
            from_square, to_square = split_move_key(key)
            mirror_from_square = mirror_square(from_square)
            mirror_to_square = mirror_square(to_square)
            mirror_key = move_key(mirror_from_square, mirror_to_square)
            mirror_move = lookups.get(mirror_from_square, {}).get("moves", {}).get(
                mirror_key
            )
            mirror_policy = home_policy_for_move(mirror_move)
            eligible = move_is_default_home_success(mirror_move)
            reason = (
                "reflected_default_success"
                if eligible
                else f"reflected_counterpart_{mirror_policy}"
            )
            planned_move = PlannedMove(
                move_key=key,
                from_square=from_square,
                to_square=to_square,
                mirror_move_key=mirror_key,
                mirror_from_square=mirror_from_square,
                mirror_to_square=mirror_to_square,
                mirror_counterpart_home_policy=mirror_policy,
                lift_override=lift_override_for_counterpart(mirror_move),
                eligible=eligible,
                reason=reason,
            )
            if eligible or include_unsupported:
                planned.append(planned_move)
            else:
                unsupported.append(planned_move)

    planned.sort(
        key=lambda item: (
            not item.eligible,
            item.from_square,
            item.to_square[0],
            int(item.to_square[1]),
        )
    )
    unsupported.sort(key=lambda item: (item.from_square, item.to_square))
    return planned, unsupported


def shard_items(items: list[PlannedMove], shard_index: int, shard_count: int) -> list[PlannedMove]:
    if shard_count < 1:
        raise ValueError("--shard-count must be at least 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < count")
    return items[shard_index::shard_count]


def lift_group_key(value: float | None) -> str:
    return "standard" if value is None else f"{value:g}"


def grouped_planned_moves(items: list[PlannedMove]):
    grouped = defaultdict(list)
    for item in items:
        grouped[(item.from_square, item.lift_override)].append(item)
    return dict(sorted(grouped.items(), key=lambda group: (group[0][0], lift_group_key(group[0][1]))))


def mirror_metadata_for_moves(items: list[PlannedMove]) -> dict:
    return {
        item.move_key: {
            "trajectory_home_policy": "alternate_home_m180_rank_reflection",
            "mirror_rank_reflection": True,
            "mirror_reflection_rule": "rank_9_minus_rank",
            "mirror_counterpart_key": item.mirror_move_key,
            "mirror_counterpart_from_square": item.mirror_from_square,
            "mirror_counterpart_to_square": item.mirror_to_square,
            "mirror_counterpart_home_policy": item.mirror_counterpart_home_policy,
        }
        for item in items
    }


def print_plan(planned: list[PlannedMove], unsupported: list[PlannedMove]) -> None:
    print(f"Planned mirrored-home moves: {len(planned)}")
    print(f"Unsupported missing moves listed only: {len(unsupported)}")
    for (source_square, lift_override), items in grouped_planned_moves(planned).items():
        print(
            f"\n{source_square} | lift={lift_group_key(lift_override)} | "
            f"count={len(items)}"
        )
        for item in items:
            print(
                f"  {item.move_key} <- {item.mirror_move_key} "
                f"({item.mirror_counterpart_home_policy})"
            )
    if unsupported:
        print("\nUnsupported by default-home rank reflection:")
        for item in unsupported:
            print(
                f"  {item.move_key} <- {item.mirror_move_key} "
                f"({item.mirror_counterpart_home_policy})"
            )


def run_builder_group(
    *,
    python_executable: str,
    output_dir: Path,
    live_lookup_dir: Path,
    log_dir: Path,
    source_square: str,
    lift_override: float | None,
    items: list[PlannedMove],
) -> tuple[int, Path, Path]:
    group_label = lift_group_key(lift_override).replace(".", "p")
    group_output_dir = output_dir / "supplements" / source_square / f"lift_{group_label}"
    log_path = log_dir / f"{source_square}_lift_{group_label}.log"
    env = os.environ.copy()
    env["SOURCE_SQUARES"] = source_square
    env["TARGET_MOVES"] = " ".join(item.move_key for item in items)
    env["LOOKUP_HOME_SHOULDER_PAN_DELTA_DEG"] = str(MIRRORED_HOME_DELTA_DEG)
    env["LOOKUP_OUTPUT_DIR"] = str(group_output_dir)
    env["DONOR_LOOKUP_DIR"] = str(live_lookup_dir)
    env["MIRROR_LOOKUP_METADATA_BY_MOVE"] = json.dumps(mirror_metadata_for_moves(items))
    if lift_override is None:
        env.pop("LOOKUP_LIFT_HEIGHT_OVERRIDE", None)
    else:
        env["LOOKUP_LIFT_HEIGHT_OVERRIDE"] = str(lift_override)

    command = [
        python_executable,
        "-B",
        str(LOWLEVEL_DIR / "build_general_nonh_reverse_lookup.py"),
    ]
    started_at = datetime.now(timezone.utc)
    group_output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"started_at_utc={started_at.isoformat()}\n")
        log_file.write(f"source_square={source_square}\n")
        log_file.write(f"target_moves={env['TARGET_MOVES']}\n")
        log_file.write(f"lift_override={lift_override}\n")
        log_file.write(f"home_shoulder_pan_delta_deg={MIRRORED_HOME_DELTA_DEG}\n")
        log_file.write(f"lookup_output_dir={group_output_dir}\n")
        log_file.write(f"donor_lookup_dir={live_lookup_dir}\n")
        log_file.write(f"command={shlex.join(command)}\n\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=LOWLEVEL_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
        finished_at = datetime.now(timezone.utc)
        log_file.write(f"\nfinished_at_utc={finished_at.isoformat()}\n")
        log_file.write(
            f"duration_seconds={(finished_at - started_at).total_seconds():.3f}\n"
        )
        log_file.write(f"return_code={return_code}\n")

    output_path = group_output_dir / f"{source_square}_non_h_reverse_move_lookup.json"
    return return_code, output_path, log_path


def successful_supplement_entries(supplement_path: Path) -> dict:
    lookup = load_lookup(supplement_path)
    return {
        key: move
        for key, move in lookup.get("moves", {}).items()
        if move_is_successful(move)
    }


def merge_successes_into_live(
    *,
    source_square: str,
    supplement_paths: list[Path],
    live_lookup_dir: Path,
    no_backup: bool,
) -> tuple[int, Path]:
    live_path = lookup_path_for_source(source_square, live_lookup_dir)
    live_lookup = load_lookup(live_path)
    live_lookup.setdefault("moves", {})
    merged_count = 0
    for supplement_path in supplement_paths:
        for key, move in successful_supplement_entries(supplement_path).items():
            live_lookup["moves"][key] = deepcopy(move)
            merged_count += 1

    if merged_count == 0:
        return 0, live_path

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if live_path.exists() and not no_backup:
        backup = live_path.with_name(f"{live_path.name}.bak_mirror_merge_{stamp}")
        shutil.copy2(live_path, backup)
        print(f"Backed up {live_path} to {backup}")

    metadata = live_lookup.setdefault("metadata", {})
    metadata["last_mirrored_home_merge_at"] = datetime.now(timezone.utc).isoformat()
    metadata["last_mirrored_home_merge_source"] = (
        "lowlevel/run_mirrored_home_lookup_recovery.py"
    )
    metadata["last_mirrored_home_merge_supplements"] = [
        str(path) for path in supplement_paths
    ]
    metadata["last_mirrored_home_merge_count"] = merged_count
    live_path.write_text(json.dumps(live_lookup, indent=2, sort_keys=True) + "\n")
    regenerate_success_map(live_path)
    return merged_count, live_path


def regenerate_success_map(live_path: Path) -> None:
    output_path = success_map_path_for_lookup(live_path)
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(LOWLEVEL_DIR / "visualize_lookup_success.py"),
            str(live_path),
            "--output",
            str(output_path),
            "--no-open",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def write_run_summary(
    *,
    output_dir: Path,
    planned: list[PlannedMove],
    unsupported: list[PlannedMove],
    group_results: list[dict],
    merge_results: list[dict],
) -> None:
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mirrored_home_delta_deg": MIRRORED_HOME_DELTA_DEG,
        "reflection_rule": "rank_9_minus_rank",
        "planned": [item.__dict__ for item in planned],
        "unsupported": [item.__dict__ for item in unsupported],
        "group_results": group_results,
        "merge_results": merge_results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--sources", nargs="*", type=str)
    parser.add_argument("--target-moves", nargs="*", type=str)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--lookup-dir", type=Path, default=LOWLEVEL_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--include-unsupported", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_sources = parse_square_list(args.sources)
    selected_target_moves = parse_move_list(args.target_moves)
    lookup_dir = args.lookup_dir.expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else LOWLEVEL_DIR / "mirror_runs" / f"rank_reflect_m180_{timestamp}"
    )

    planned, unsupported = collect_planned_moves(
        lookup_dir=lookup_dir,
        selected_sources=selected_sources,
        selected_target_moves=selected_target_moves,
        include_unsupported=args.include_unsupported,
    )
    planned = shard_items(planned, args.shard_index, args.shard_count)
    print_plan(planned, unsupported)

    if args.dry_run or not args.execute:
        write_run_summary(
            output_dir=output_dir,
            planned=planned,
            unsupported=unsupported,
            group_results=[],
            merge_results=[],
        )
        print(f"\nSummary: {output_dir / 'summary.json'}")
        return 0

    group_results = []
    supplement_paths_by_source = defaultdict(list)
    log_dir = output_dir / "logs"
    failures = []
    for (source_square, lift_override), items in grouped_planned_moves(planned).items():
        print(
            f"\n=== mirrored recovery {source_square} "
            f"lift={lift_group_key(lift_override)} count={len(items)} ==="
        )
        return_code, supplement_path, log_path = run_builder_group(
            python_executable=args.python,
            output_dir=output_dir,
            live_lookup_dir=lookup_dir,
            log_dir=log_dir,
            source_square=source_square,
            lift_override=lift_override,
            items=items,
        )
        group_result = {
            "source_square": source_square,
            "lift_override": lift_override,
            "target_moves": [item.move_key for item in items],
            "return_code": return_code,
            "supplement_path": str(supplement_path),
            "log_path": str(log_path),
        }
        group_results.append(group_result)
        if supplement_path.exists():
            supplement_paths_by_source[source_square].append(supplement_path)
        if return_code != 0:
            failures.append(group_result)
            if not args.continue_on_failure:
                break

    merge_results = []
    if args.merge and not failures:
        for source_square, supplement_paths in sorted(supplement_paths_by_source.items()):
            merged_count, live_path = merge_successes_into_live(
                source_square=source_square,
                supplement_paths=supplement_paths,
                live_lookup_dir=lookup_dir,
                no_backup=args.no_backup,
            )
            merge_results.append(
                {
                    "source_square": source_square,
                    "merged_count": merged_count,
                    "live_path": str(live_path),
                    "supplement_paths": [str(path) for path in supplement_paths],
                }
            )
            print(f"Merged {merged_count} mirrored successes into {live_path}")
    elif args.merge and failures:
        print("Skipping merge because at least one builder group failed.")

    write_run_summary(
        output_dir=output_dir,
        planned=planned,
        unsupported=unsupported,
        group_results=group_results,
        merge_results=merge_results,
    )
    print(f"\nSummary: {output_dir / 'summary.json'}")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(
                f"  {failure['source_square']} lift={failure['lift_override']} "
                f"return_code={failure['return_code']} log={failure['log_path']}"
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
