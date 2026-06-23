import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SOURCE_SQUARES = tuple(
    f"{file}{rank}"
    # for file in "fedcba"
    for file in "f"
    # for rank in range(8, 0, -1)
    for rank in [1]
)


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


def source_order(start_square, shard_index=0, shard_count=1):
    if shard_count < 1:
        raise ValueError("--shard-count must be at least 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < count")

    start_index = SOURCE_SQUARES.index(start_square)
    return SOURCE_SQUARES[start_index:][shard_index::shard_count]


def run_source(script_dir, python_executable, source_square, log_dir):
    output_path = script_dir / f"{source_square}_non_h_reverse_move_lookup.json"
    log_path = log_dir / f"{source_square}.log"
    env = os.environ.copy()
    env["SOURCE_SQUARES"] = source_square

    command = [
        python_executable,
        "-B",
        str(script_dir / "build_general_nonh_reverse_lookup.py"),
    ]

    print(f"\n=== {source_square} -> {output_path.name} ===", flush=True)
    print(f"Log: {log_path}", flush=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"source_square={source_square}\n")
        log_file.write(f"output_path={output_path}\n")
        log_file.write(f"command={' '.join(command)}\n\n")
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

        log_file.write(f"\nreturn_code={return_code}\n")
        log_file.write(f"output_exists={output_path.exists()}\n")

    return return_code, output_path, log_path


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    order = source_order(args.start, args.shard_index, args.shard_count)

    if args.dry_run:
        print(" ".join(order))
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
        f"Shard: {args.shard_index} of {args.shard_count}",
        flush=True,
    )
    print(f"Logs: {log_dir}", flush=True)

    failures = []
    for source_square in order:
        return_code, output_path, log_path = run_source(
            script_dir,
            args.python,
            source_square,
            log_dir,
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
