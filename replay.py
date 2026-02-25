#!/usr/bin/env python3
import subprocess
import argparse
import os
from ids import ROBOT_PORT, ROBOT_ID


def main():
    parser = argparse.ArgumentParser(description="Wrapper for lerobot-replay")

    parser.add_argument(
        "--repo-id",
        # default="seeedstudio123/second_datacollection",
        default="second_datacollection",
        
        help="Dataset repo_id (e.g. user/dataset_name)",
    )

    parser.add_argument(
        "--dataset-root",
        default="/Users/zhg603/.cache/huggingface/lerobot/seeedstudio123/",
        help="Root directory where lerobot datasets are stored",
    )

    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode number to replay",
    )

    parser.add_argument(
        "--display-only",
        action="store_true",
        help="Replay without moving robot",
    )

    args = parser.parse_args()

    cmd = ["lerobot-replay"]

    # Robot section
    if not args.display_only:
        if not os.path.exists(ROBOT_PORT):
            raise RuntimeError(f"Robot port not found: {ROBOT_PORT}")

        cmd += [
            "--robot.type=so101_follower",
            f"--robot.port={ROBOT_PORT}",
            f"--robot.id={ROBOT_ID}",
        ]
    else:
        cmd += ["--robot.type=openarm_follower"]  # dummy safe type if needed

    # Dataset section (correct flags for your version)
    cmd += [
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={args.dataset_root}",
        f"--dataset.episode={args.episode}",
    ]

    print("\nRunning:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
