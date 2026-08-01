import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))

    import run_all_nonh_reverse_sources as runner

    sys.argv = [
        str(script_dir / "run_all_nonh_reverse_sources.py"),
        "--shard-index",
        "0",
        "--shard-count",
        "3",
        *sys.argv[1:],
    ]
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
