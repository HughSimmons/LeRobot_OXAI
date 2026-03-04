#!/usr/bin/env python3
import subprocess
import os
from ids import ROBOT_PORT, ROBOT_ID, TELEOP_PORT, TELEOP_ID


DATASET_REPO = "seeedstudio123/fourth_datacollect"


# def build_camera_config():
#     return (
#         '{ front: {'
#         'type: opencv, '
#         'index_or_path: 0, '
#         'width: 640, '
#         'height: 480, '
#         'fps: 30, '
#         'fourcc: "MJPG"'
#         '}}'
    # )

def build_camera_config():
    return (
        '{ '
        'front: {'
        'type: opencv, '
        'index_or_path: 0, '
        'width: 640, '
        'height: 480, '
        'fps: 30, '
        'fourcc: "MJPG"'
        '}, '
        'side: {'
        'type: opencv, '
        'index_or_path: 1, '
        'width: 640, '
        'height: 480, '
        'fps: 30, '
        'fourcc: "MJPG"'
        '}'
        '}'
    )

def main():
    # Safety checks
    if not os.path.exists(ROBOT_PORT):
        raise RuntimeError(f"Robot port not found: {ROBOT_PORT}")

    if not os.path.exists(TELEOP_PORT):
        raise RuntimeError(f"Teleop port not found: {TELEOP_PORT}")

    camera_config = build_camera_config()

    cmd = [
        "lerobot-record",
        "--robot.type=so101_follower",
        f"--robot.port={ROBOT_PORT}",
        f"--robot.id={ROBOT_ID}",
        f"--robot.cameras={camera_config}",
        "--teleop.type=so101_leader",
        f"--teleop.port={TELEOP_PORT}",
        f"--teleop.id={TELEOP_ID}",
        "--display_data=true",
        f"--dataset.repo_id={DATASET_REPO}",
        "--dataset.num_episodes=5",
        '--dataset.single_task=Grab the black cube',
        "--dataset.push_to_hub=false",
        "--dataset.episode_time_s=30",
        "--dataset.reset_time_s=30",
    ]

    print("\nRunning:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()