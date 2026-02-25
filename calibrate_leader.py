#!/usr/bin/env python3
import subprocess
from ids import TELEOP_PORT, TELEOP_ID


def main():
    cmd = [
        "lerobot-calibrate",
        "--teleop.type=so101_leader",
        f"--teleop.port={TELEOP_PORT}",
        f"--teleop.id={TELEOP_ID}",
    ]

    print("\nRunning:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
