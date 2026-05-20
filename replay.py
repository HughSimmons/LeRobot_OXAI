# export HF_HUB_OFFLINE=1

# lerobot-replay \
#   --robot.type=so101_follower \
#   --robot.port=/dev/tty.usbmodem5AB90659861 \
#   --robot.id=my_awesome_follower_arm \
#   --dataset.repo_id=seeedstudio123/second_datacollection \
#   --dataset.root=/Users/zhg603/.cache/huggingface/lerobot/seeedstudio123/second_datacollection \
#   --dataset.episode=1

#!/usr/bin/env python3
import subprocess
import argparse
import os
from ids import ROBOT_PORT, ROBOT_ID


def main():
    parser = argparse.ArgumentParser(description="Wrapper for lerobot-replay")

    parser.add_argument(
        "--episode",
        type=int,
        default=4,
        help="Episode number to replay",
    )

    parser.add_argument(
        "--dataset-root",
        default=os.path.expanduser(
            # "~/.cache/huggingface/lerobot/seeedstudio123/second_datacollection"
            "~/.cache/huggingface/lerobot/seeedstudio123/D2"
        ),
        help="Full path to dataset folder",
    )

    args = parser.parse_args()

    if not os.path.exists(ROBOT_PORT):
        raise RuntimeError(f"Robot port not found: {ROBOT_PORT}")

    if not os.path.exists(args.dataset_root):
        raise RuntimeError(f"Dataset folder not found: {args.dataset_root}")

    # Force offline mode
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"

    cmd = [
        "lerobot-replay",
        "--robot.type=so101_follower",
        f"--robot.port={ROBOT_PORT}",
        f"--robot.id={ROBOT_ID}",
        "--dataset.repo_id=seeedstudio123/fourth_datacollect",
        f"--dataset.root={args.dataset_root}",
        f"--dataset.episode={args.episode}",
    ]

    print("\nRunning:")
    print("HF_HUB_OFFLINE=1 " + " ".join(cmd))
    print()

    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
