#!/usr/bin/env python3
import subprocess
from ids import ROBOT_PORT, ROBOT_ID


def main():
    cmd = [
        "lerobot-calibrate",
        "--robot.type=so101_follower",
        f"--robot.port={ROBOT_PORT}",
        f"--robot.id={ROBOT_ID}",
    ]

    print("\nRunning:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
