import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SOURCE_SQUARES = tuple(
    f"{file}{rank}"
    for file in "fedcba"
    for rank in range(8, 0, -1)
)
TARGET_SQUARES = tuple(
    f"{file}{rank}"
    for file in "abcdefgh"
    for rank in range(1, 9)
)


def parse_square_list(raw_value):
    if raw_value is None or not raw_value.strip():
        return None

    squares = tuple(
        square.strip()
        for square in raw_value.replace(",", " ").split()
        if square.strip()
    )
    for square in squares:
        if square not in SOURCE_SQUARES:
            raise argparse.ArgumentTypeError(
                f"Invalid source square {square!r}; expected one of "
                f"{' '.join(SOURCE_SQUARES)}"
            )
    return squares


def parse_target_moves(raw_value):
    if raw_value is None or not raw_value.strip():
        return None

    moves = tuple(
        move.strip()
        for move in raw_value.replace(",", " ").split()
        if move.strip()
    )
    for move in moves:
        parts = move.split("_to_")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"Invalid target move {move!r}; use values like e4_to_h7"
            )
        from_square, to_square = parts
        if from_square not in SOURCE_SQUARES:
            raise argparse.ArgumentTypeError(
                f"Invalid source square in target move {move!r}"
            )
        if (
            len(to_square) != 2
            or to_square[0] not in "abcdefgh"
            or to_square[1] not in "12345678"
        ):
            raise argparse.ArgumentTypeError(
                f"Invalid destination square in target move {move!r}"
            )
    return moves


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run build_general_nonh_reverse_lookup.py once per source square, "
            "saving one lookup JSON per source."
        )
    )
    parser.add_argument(
        "--start",
        choices=SOURCE_SQUARES,
        default=SOURCE_SQUARES[0],
        help="First source square to run in f8..a1 order.",
    )
    parser.add_argument(
        "--sources",
        type=parse_square_list,
        help=(
            "Space- or comma-separated source squares to run instead of the "
            "full f8..a1 order, e.g. 'e4' or 'e4 e3'."
        ),
    )
    parser.add_argument(
        "--target-moves",
        type=parse_target_moves,
        help=(
            "Space- or comma-separated move keys to pass to the builder, "
            "e.g. 'e4_to_h4 e4_to_h5'. Moves are filtered per source."
        ),
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help=(
            "For each source, run only target moves not already present as "
            "successful entries in that source lookup JSON."
        ),
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue with later source squares if one subprocess exits nonzero.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned source-square order without running builds.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for each build subprocess.",
    )
    parser.add_argument(
        "--donor-lookup-dir",
        type=Path,
        help=(
            "Optional folder of donor lookup JSONs to pass as DONOR_LOOKUP_DIR. "
            "Use the same snapshot folder for all parallel shards for stricter "
            "donor reproducibility."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index for interleaving source squares.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total number of interleaved shards.",
    )
    return parser.parse_args()


def source_order(start_square, selected_sources=None, shard_index=0, shard_count=1):
    if shard_count < 1:
        raise ValueError("--shard-count must be at least 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < count")

    if selected_sources is None:
        start_index = SOURCE_SQUARES.index(start_square)
        sources = SOURCE_SQUARES[start_index:]
    else:
        sources = tuple(
            square
            for square in SOURCE_SQUARES
            if square in set(selected_sources)
        )
    return sources[shard_index::shard_count]


def moves_for_source(target_moves, source_square):
    if target_moves is None:
        return None
    return tuple(
        move
        for move in target_moves
        if move.startswith(f"{source_square}_to_")
    )


def expected_moves_for_source(source_square):
    return tuple(
        f"{source_square}_to_{target_square}"
        for target_square in TARGET_SQUARES
        if target_square != source_square
    )


def successful_move_keys(script_dir, source_square):
    lookup_path = script_dir / f"{source_square}_non_h_reverse_move_lookup.json"
    if not lookup_path.exists():
        return set()

    try:
        lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()

    moves = lookup.get("moves")
    if not isinstance(moves, dict):
        return set()

    return {
        move_key
        for move_key, move in moves.items()
        if isinstance(move, dict) and move.get("success")
    }


def missing_moves_for_source(script_dir, source_square, candidate_moves=None):
    expected_moves = (
        expected_moves_for_source(source_square)
        if candidate_moves is None
        else tuple(candidate_moves)
    )
    successful_moves = successful_move_keys(script_dir, source_square)
    return tuple(
        move_key
        for move_key in expected_moves
        if move_key not in successful_moves
    )


def run_source(
    script_dir,
    python_executable,
    source_square,
    log_dir,
    target_moves=None,
    missing_only=False,
    shard_index=0,
    shard_count=1,
    donor_lookup_dir=None,
):
    output_path = script_dir / f"{source_square}_non_h_reverse_move_lookup.json"
    log_path = log_dir / f"{source_square}.log"
    env = os.environ.copy()
    env["SOURCE_SQUARES"] = source_square
    if target_moves is not None:
        env["TARGET_MOVES"] = " ".join(target_moves)
    else:
        env.pop("TARGET_MOVES", None)
    if donor_lookup_dir is not None:
        env["DONOR_LOOKUP_DIR"] = str(donor_lookup_dir)
    else:
        env.pop("DONOR_LOOKUP_DIR", None)

    command = [
        python_executable,
        "-B",
        str(script_dir / "build_general_nonh_reverse_lookup.py"),
    ]

    started_at = datetime.now(timezone.utc)
    print(f"\n=== {source_square} -> {output_path.name} ===", flush=True)
    print(f"Log: {log_path}", flush=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"started_at_utc={started_at.isoformat()}\n")
        log_file.write(f"source_square={source_square}\n")
        log_file.write(f"target_moves={' '.join(target_moves) if target_moves else ''}\n")
        log_file.write(f"target_move_count={len(target_moves) if target_moves else 'all'}\n")
        log_file.write(f"missing_only={missing_only}\n")
        log_file.write(f"shard_index={shard_index}\n")
        log_file.write(f"shard_count={shard_count}\n")
        log_file.write(f"donor_lookup_dir={donor_lookup_dir or ''}\n")
        log_file.write(f"python_executable={python_executable}\n")
        log_file.write(f"output_path={output_path}\n")
        log_file.write(f"command={shlex.join(command)}\n\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=script_dir,
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
        log_file.write(f"\nreturn_code={return_code}\n")
        log_file.write(f"output_exists={output_path.exists()}\n")

    return return_code, output_path, log_path


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    donor_lookup_dir = (
        args.donor_lookup_dir.expanduser().resolve()
        if args.donor_lookup_dir is not None
        else None
    )
    order = source_order(
        args.start,
        selected_sources=args.sources,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )

    if args.dry_run:
        print(" ".join(order))
        if args.target_moves is not None or args.missing_only:
            for source_square in order:
                target_moves = moves_for_source(args.target_moves, source_square)
                if args.missing_only:
                    target_moves = missing_moves_for_source(
                        script_dir,
                        source_square,
                        candidate_moves=target_moves,
                    )
                elif target_moves is None:
                    target_moves = ()
                print(
                    f"{source_square}: "
                    f"{' '.join(target_moves)}"
                )
        return 0

    log_dir = (
        script_dir
        / "logs"
        / (
            f"nonh_reverse_master_shard{args.shard_index}_of_{args.shard_count}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    print("Source order:", " ".join(order), flush=True)
    print(
        "Target moves:",
        " ".join(args.target_moves) if args.target_moves else "(all)",
        flush=True,
    )
    print(
        "Missing only:",
        bool(args.missing_only),
        flush=True,
    )
    print(
        "Donor lookup dir:",
        donor_lookup_dir if donor_lookup_dir is not None else "(live lookup dir)",
        flush=True,
    )
    print(
        f"Shard: {args.shard_index} of {args.shard_count}",
        flush=True,
    )
    print(f"Logs: {log_dir}", flush=True)

    failures = []
    for source_square in order:
        target_moves = moves_for_source(args.target_moves, source_square)
        if args.missing_only:
            target_moves = missing_moves_for_source(
                script_dir,
                source_square,
                candidate_moves=target_moves,
            )
        if args.target_moves is not None and not target_moves:
            print(f"\n=== {source_square}: no matching target moves; skipping ===")
            continue
        if args.missing_only and not target_moves:
            print(f"\n=== {source_square}: no outstanding target moves; skipping ===")
            continue

        return_code, output_path, log_path = run_source(
            script_dir,
            args.python,
            source_square,
            log_dir,
            target_moves=target_moves,
            missing_only=args.missing_only,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            donor_lookup_dir=donor_lookup_dir,
        )
        if return_code != 0:
            failures.append((source_square, return_code, log_path))
            print(
                f"{source_square}: build subprocess failed with code {return_code}",
                flush=True,
            )
            if not args.continue_on_failure:
                break
        elif not output_path.exists():
            failures.append((source_square, "missing_output", log_path))
            print(f"{source_square}: expected output was not created", flush=True)
            if not args.continue_on_failure:
                break

    if failures:
        print("\nFailures:", flush=True)
        for source_square, reason, log_path in failures:
            print(f"  {source_square}: {reason} ({log_path})", flush=True)
        return 1

    print("\nAll requested source-square builds completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
