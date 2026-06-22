import argparse
import time

import cv2
import numpy as np
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


BOARD_ORIGIN = (0.25, 0.0, 0.0)
SQUARE_SIZE = 0.04
TARGET_SQUARE = "c3"
APPROACH_HEIGHT = 0.13
PLACE_OFFSET = np.array([-0.01845, 0.00115, -0.005])

DEFAULT_PORT = "/dev/tty.usbmodem5B7B0157051"
DEFAULT_ROBOT_ID = "my_awesome_follower_arm"
URDF_PATH = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
TARGET_FRAME_NAME = "gripper_frame_link"

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

HOME_JOINTS = np.array([
    96.92307692307692,
    -107.86813186813187,
    97.36263736263736,
    65.18681318681318,
    -29.846153846153847,
    4.62962962962963,
])
GRIPPER_OPEN_DEG = 25.0


def chess_to_xyz(square, board_origin=BOARD_ORIGIN, square_size=SQUARE_SIZE):
    files = "abcdefgh"
    file_idx = files.index(square[0].lower())
    rank_idx = int(square[1]) - 1
    board_x, board_y, board_z = board_origin
    board_size = 8 * square_size
    return np.array([
        board_x - board_size / 2 + (file_idx + 0.5) * square_size,
        board_y - board_size / 2 + (rank_idx + 0.5) * square_size,
        board_z + 0.04,
    ])


def joints_to_action(joints):
    return {
        f"{name}.pos": float(angle)
        for name, angle in zip(JOINT_NAMES, joints)
    }


def observation_to_joints(observation):
    return np.array([
        observation[f"{name}.pos"]
        for name in JOINT_NAMES
    ])


def solve_pre_lower_joints(square, gripper_angle):
    kinematics = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name=TARGET_FRAME_NAME,
        joint_names=JOINT_NAMES,
    )

    target_xyz = chess_to_xyz(square) + np.array([0.0, 0.0, APPROACH_HEIGHT])
    joint_solution = HOME_JOINTS.copy()

    for _ in range(16):
        fk_pose = kinematics.forward_kinematics(joint_solution)
        target_pose = fk_pose.copy()
        target_pose[:3, 3] = target_xyz - fk_pose[:3, :3] @ PLACE_OFFSET
        joint_solution = kinematics.inverse_kinematics(
            joint_solution,
            target_pose,
            position_weight=10.0,
        )

    joint_solution[5] = gripper_angle
    return joint_solution, target_xyz


def smooth_move(follower, target_joints, steps, delay):
    observation = follower.get_observation()
    start_joints = observation_to_joints(observation)

    for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
        interpolated = (1.0 - alpha) * start_joints + alpha * target_joints
        action = observation.copy()
        action.update(joints_to_action(interpolated))
        follower.send_action(action)
        time.sleep(delay)


def resize_frame_to_fit(frame, max_width, max_height):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame

    display_size = (
        max(1, int(width * scale)),
        max(1, int(height * scale)),
    )
    return cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)


def live_camera_until_robot_release(
    camera_index,
    window_name,
    max_width,
    max_height,
    release_robot,
):
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        robot_released = False
        print("Live camera feed running.")
        print("Press any key in the camera window to disconnect the robot.")
        print("After the robot is disconnected, press q or Escape to close the camera feed.")
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read from camera index {camera_index}")

            display_frame = resize_frame_to_fit(frame, max_width, max_height)
            status = (
                "Robot disconnected. Press q or Esc to close camera."
                if robot_released
                else "Press any key to disconnect robot."
            )
            cv2.putText(
                display_frame,
                status,
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if robot_released else (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1)
            if key == -1:
                continue

            key_code = key & 0xFF
            if not robot_released:
                release_robot()
                robot_released = True
                continue

            if key_code in (27, ord("q"), ord("Q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Move SO101 to the pre-lowering pose above c3, show a live camera "
            "feed, then disconnect the robot when any key is pressed."
        )
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--preview-max-width", type=int, default=1280)
    parser.add_argument("--preview-max-height", type=int, default=720)
    parser.add_argument("--move-steps", type=int, default=50)
    parser.add_argument("--move-delay", type=float, default=0.03)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually connect to and move the robot. Without this, only print the target.",
    )
    args = parser.parse_args()

    target_joints, target_xyz = solve_pre_lower_joints(
        TARGET_SQUARE,
        GRIPPER_OPEN_DEG,
    )

    print(f"Target square: {TARGET_SQUARE}")
    print(f"Pre-lower target xyz: {target_xyz}")
    print(f"Pre-lower target joints: {target_joints}")

    if not args.execute:
        print("Dry run only. Re-run with --execute to move the robot.")
        return

    follower = SO101Follower(SO101FollowerConfig(
        port=args.port,
        id=args.robot_id,
    ))
    connected = False

    def disconnect_robot():
        nonlocal connected
        if connected:
            print("Disconnecting robot. You can now move it freely by hand.")
            follower.disconnect()
            connected = False

    try:
        follower.connect()
        connected = True

        print("Moving to pre-lower c3 pose...")
        smooth_move(follower, target_joints, args.move_steps, args.move_delay)

        live_camera_until_robot_release(
            args.camera_index,
            "c3 pre-lower camera",
            args.preview_max_width,
            args.preview_max_height,
            disconnect_robot,
        )

    finally:
        if connected:
            print("Disconnecting robot.")
            follower.disconnect()


if __name__ == "__main__":
    main()
