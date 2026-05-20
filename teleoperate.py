#!/usr/bin/env python3
import subprocess
from ids import ROBOT_PORT, TELEOP_PORT, ROBOT_ID, TELEOP_ID


def build_camera_config():
    return (
        '{ front: {'
        'type: opencv, '
        'index_or_path: 0, '
        'width: 640, '
        'height: 480, '
        'fps: 30, '
        'fourcc: "MJPG"'
        '}}'
    )


def main():
    # camera_config = build_camera_config()

    cmd = [
        "lerobot-teleoperate",
        "--robot.type=so101_follower",
        f"--robot.port={ROBOT_PORT}",
        f"--robot.id={ROBOT_ID}",
        # f"--robot.cameras={camera_config}",
        "--teleop.type=so101_leader",
        f"--teleop.port={TELEOP_PORT}",
        f"--teleop.id={TELEOP_ID}",
        "--display_data=true",
    ]

    print("\nRunning:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
