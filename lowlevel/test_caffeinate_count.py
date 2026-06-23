import argparse
import time
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Small counter for testing caffeinate with display sleep allowed."
    )
    parser.add_argument("--seconds", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"count_test_start {datetime.now().isoformat(timespec='seconds')}", flush=True)
    for index in range(1, args.seconds + 1):
        print(
            f"{index}/{args.seconds} {datetime.now().isoformat(timespec='seconds')}",
            flush=True,
        )
        time.sleep(1)
    print(f"count_test_done {datetime.now().isoformat(timespec='seconds')}", flush=True)


if __name__ == "__main__":
    main()
