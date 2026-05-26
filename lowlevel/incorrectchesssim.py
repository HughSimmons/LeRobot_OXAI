#!/usr/bin/env python3
"""PyBullet SO101 chess move demo using a move_to_square_v2-style sequence."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import imageio
    import numpy as np
    import pybullet as p
    import pybullet_data
except ImportError as exc:
    missing = exc.name or "a simulation dependency"
    raise SystemExit(
        f"Missing Python package: {missing}. Run this script in the same environment used for "
        "the existing PyBullet/kinematics demos, with numpy, pybullet, imageio, and placo available."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = ROOT / "lowlevel" / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.model.kinematics import RobotKinematics  # noqa: E402


URDF_PATH = ROOT / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
MESH_DIR = URDF_PATH.parent

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

HOME = np.array([96.92, -107.87, 97.36, 65.19, -29.85, 4.63])
CORNER1_A1 = np.array([97.32, 0.40, 28.40, 66.55, 177.80, 4.95])

BOARD_X, BOARD_Y, BOARD_Z = 0.3, 0.0, 0.0
SQUARE_SIZE = 0.04
BOARD_SIZE = 8 * SQUARE_SIZE
FILES = "abcdefgh"

WIDTH, HEIGHT = 1280, 720
FPS = 30
LIFT_HEIGHT = 0.10
PIECE_RADIUS = 0.012
PIECE_HEIGHT = 0.04
PIECE_Z = BOARD_Z + 0.008 + PIECE_HEIGHT / 2
GRIPPER_FRAME_LINK = 5
GRASP_DISTANCE = 0.045
PHYSICS_SUBSTEPS = 8


def chess_to_board_xy(square: str) -> np.ndarray:
    """Return the PyBullet board-center coordinates for a square like e4."""
    if len(square) != 2 or square[0].lower() not in FILES or square[1] not in "12345678":
        raise ValueError(f"Invalid chess square: {square!r}")

    file_idx = FILES.index(square[0].lower())
    rank_idx = int(square[1]) - 1
    return np.array(
        [
            BOARD_X - BOARD_SIZE / 2 + SQUARE_SIZE / 2 + file_idx * SQUARE_SIZE,
            BOARD_Y - BOARD_SIZE / 2 + SQUARE_SIZE / 2 + rank_idx * SQUARE_SIZE,
            BOARD_Z,
        ]
    )


def chess_relative_from_a1(square: str) -> np.ndarray:
    """Return an xyz offset from a1 using the same 4 cm square spacing."""
    return chess_to_board_xy(square) - chess_to_board_xy("a1")


def make_kinematics() -> RobotKinematics:
    return RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="gripper_frame_link",
        joint_names=JOINT_NAMES,
    )


def relative_xyz(
    kinematics: RobotKinematics,
    init_joints_deg: np.ndarray,
    change_xyz: np.ndarray,
    iterations: int = 4,
) -> np.ndarray:
    """Move the current end-effector pose by change_xyz and return joints in degrees."""
    target_pose = kinematics.forward_kinematics(init_joints_deg).copy()
    target_pose[:3, 3] += np.asarray(change_xyz, dtype=float)

    joint_solution = np.asarray(init_joints_deg, dtype=float).copy()
    for _ in range(iterations):
        joint_solution = kinematics.inverse_kinematics(
            joint_solution,
            target_pose,
            position_weight=10.0,
            orientation_weight=0.01,
        )
    return joint_solution


def solve_gripper_xyz(
    kinematics: RobotKinematics,
    seed_joints_deg: np.ndarray,
    target_xyz: np.ndarray,
    iterations: int = 8,
) -> np.ndarray:
    """Solve IK for an absolute PyBullet-world gripper-frame position."""
    target_pose = kinematics.forward_kinematics(seed_joints_deg).copy()
    target_pose[:3, 3] = np.asarray(target_xyz, dtype=float)

    joint_solution = np.asarray(seed_joints_deg, dtype=float).copy()
    for _ in range(iterations):
        joint_solution = kinematics.inverse_kinematics(
            joint_solution,
            target_pose,
            position_weight=10.0,
            orientation_weight=0.01,
        )
    return joint_solution


def build_move_to_square_v2_path(
    kinematics: RobotKinematics,
    from_square: str,
    to_square: str,
) -> list[tuple[str, np.ndarray, str | None]]:
    """Build labelled joint/piece waypoints for a pick-and-place chess move."""
    from_xyz = chess_to_board_xy(from_square) + np.array([0.0, 0.0, PIECE_Z])
    to_xyz = chess_to_board_xy(to_square) + np.array([0.0, 0.0, PIECE_Z])

    above_from = solve_gripper_xyz(
        kinematics,
        CORNER1_A1,
        from_xyz + np.array([0.0, 0.0, LIFT_HEIGHT]),
    )
    at_from = solve_gripper_xyz(kinematics, above_from, from_xyz)

    grip_closed = at_from.copy()
    grip_closed[5] = 0.0

    lifted = solve_gripper_xyz(kinematics, grip_closed, from_xyz + np.array([0.0, 0.0, LIFT_HEIGHT]))
    above_to = solve_gripper_xyz(kinematics, lifted, to_xyz + np.array([0.0, 0.0, LIFT_HEIGHT]))
    at_to = solve_gripper_xyz(kinematics, above_to, to_xyz)

    grip_open = at_to.copy()
    grip_open[5] = 5.0

    lifted_after_place = solve_gripper_xyz(
        kinematics,
        grip_open,
        to_xyz + np.array([0.0, 0.0, LIFT_HEIGHT]),
    )

    return [
        ("home", HOME, None),
        ("a1 calibration", CORNER1_A1, None),
        (f"above {from_square}", above_from, None),
        (f"lower to {from_square}", at_from, None),
        ("close gripper", grip_closed, "grasp"),
        ("lift piece", lifted, None),
        (f"move to {to_square}", above_to, None),
        (f"lower to {to_square}", at_to, None),
        ("open gripper", grip_open, "release"),
        ("lift away", lifted_after_place, None),
        ("return home", HOME, None),
    ]


def setup_pybullet() -> int:
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(1 / 240)
    p.setPhysicsEngineParameter(numSolverIterations=100)
    return physics_client


def create_board() -> None:
    p.loadURDF("plane.urdf", [0, 0, 0])

    board_base_shape = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[BOARD_SIZE / 2, BOARD_SIZE / 2, 0.005]
    )
    board_base_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[BOARD_SIZE / 2, BOARD_SIZE / 2, 0.005],
        rgbaColor=[0.3, 0.3, 0.3, 1],
    )
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=board_base_shape,
        baseVisualShapeIndex=board_base_visual,
        basePosition=[BOARD_X, BOARD_Y, BOARD_Z],
    )

    for rank in range(8):
        for file_idx in range(8):
            square_center = chess_to_board_xy(f"{FILES[file_idx]}{rank + 1}")
            color = [1, 1, 1, 0.5] if (rank + file_idx) % 2 == 0 else [0.1, 0.1, 0.1, 0.5]
            square_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[SQUARE_SIZE / 2 - 0.001, SQUARE_SIZE / 2 - 0.001, 0.001],
                rgbaColor=color,
            )
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=square_visual,
                basePosition=[square_center[0], square_center[1], BOARD_Z + 0.008],
            )


def create_piece(square: str) -> int:
    piece_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=PIECE_RADIUS, height=PIECE_HEIGHT)
    piece_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=PIECE_RADIUS,
        length=PIECE_HEIGHT,
        rgbaColor=[1, 0, 0, 1],
    )
    xy = chess_to_board_xy(square)
    piece_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=piece_shape,
        baseVisualShapeIndex=piece_visual,
        basePosition=[xy[0], xy[1], PIECE_Z],
    )
    p.changeDynamics(
        piece_id,
        -1,
        linearDamping=0.04,
        angularDamping=0.04,
        lateralFriction=1.0,
        spinningFriction=0.02,
        rollingFriction=0.02,
        restitution=0.0,
    )
    return piece_id


def load_robot() -> int:
    robot_id = p.loadURDF(str(URDF_PATH), [0, 0, 0], useFixedBase=True)
    print(f"Loaded SO101 from {URDF_PATH}")
    print(f"Number of PyBullet joints: {p.getNumJoints(robot_id)}")
    return robot_id


def set_robot_joints(robot_id: int, joints_deg: np.ndarray) -> None:
    joints_rad = np.deg2rad(joints_deg)
    for joint_idx in range(min(6, p.getNumJoints(robot_id))):
        p.setJointMotorControl2(
            robot_id,
            joint_idx,
            p.POSITION_CONTROL,
            targetPosition=float(joints_rad[joint_idx]),
            force=500,
        )


def gripper_position(robot_id: int) -> np.ndarray:
    link_state = p.getLinkState(robot_id, GRIPPER_FRAME_LINK, computeForwardKinematics=True)
    return np.array(link_state[0])


def try_create_grasp_constraint(robot_id: int, piece_id: int) -> int | None:
    gripper_pos = gripper_position(robot_id)
    piece_pos, piece_orn = p.getBasePositionAndOrientation(piece_id)
    distance = float(np.linalg.norm(gripper_pos - np.array(piece_pos)))

    if distance > GRASP_DISTANCE:
        print(
            f"  Grasp failed: gripper-piece distance {distance:.3f} m "
            f"exceeds threshold {GRASP_DISTANCE:.3f} m"
        )
        return None

    link_state = p.getLinkState(robot_id, GRIPPER_FRAME_LINK, computeForwardKinematics=True)
    parent_pos, parent_orn = link_state[0], link_state[1]
    inv_parent_pos, inv_parent_orn = p.invertTransform(parent_pos, parent_orn)
    parent_frame_pos, parent_frame_orn = p.multiplyTransforms(
        inv_parent_pos,
        inv_parent_orn,
        piece_pos,
        piece_orn,
    )

    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=GRIPPER_FRAME_LINK,
        childBodyUniqueId=piece_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=parent_frame_pos,
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=parent_frame_orn,
        childFrameOrientation=[0, 0, 0, 1],
    )
    print(f"  Grasp attached at distance {distance:.3f} m")
    return constraint_id


def release_grasp_constraint(constraint_id: int | None) -> None:
    if constraint_id is not None:
        p.removeConstraint(constraint_id)
        print("  Grasp released")


def step_physics() -> None:
    for _ in range(PHYSICS_SUBSTEPS):
        p.stepSimulation()


def render_frame(writer: imageio.core.format.Writer, writer_topdown: imageio.core.format.Writer) -> None:
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=WIDTH / HEIGHT, nearVal=0.01, farVal=100
    )

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0.5, 0.5, 0.5],
        cameraTargetPosition=[BOARD_X, BOARD_Y, BOARD_Z],
        cameraUpVector=[0, 0, 1],
    )
    _, _, rgba, _, _ = p.getCameraImage(WIDTH, HEIGHT, viewMatrix=view_matrix, projectionMatrix=proj_matrix)
    writer.append_data(np.array(rgba[:, :, :3], dtype=np.uint8))

    view_matrix_topdown = p.computeViewMatrix(
        cameraEyePosition=[BOARD_X, BOARD_Y, 0.6],
        cameraTargetPosition=[BOARD_X, BOARD_Y, BOARD_Z],
        cameraUpVector=[0, -1, 0],
    )
    _, _, rgba_topdown, _, _ = p.getCameraImage(
        WIDTH, HEIGHT, viewMatrix=view_matrix_topdown, projectionMatrix=proj_matrix
    )
    writer_topdown.append_data(np.array(rgba_topdown[:, :, :3], dtype=np.uint8))


def simulate_path(
    robot_id: int,
    piece_id: int,
    waypoints: list[tuple[str, np.ndarray, str | None]],
) -> None:
    writer = imageio.get_writer("so101_robot_e2e4_realistic.mp4", fps=FPS, codec="libx264", quality=8)
    writer_topdown = imageio.get_writer(
        "so101_robot_e2e4_realistic_topdown.mp4", fps=FPS, codec="libx264", quality=8
    )

    try:
        frame_index = 0
        grasp_constraint_id = None

        for (start_name, start_joints, _), (end_name, end_joints, end_action) in zip(
            waypoints[:-1], waypoints[1:]
        ):
            steps = 80 if end_name in {"home", "return home"} else 50
            print(f"Segment: {start_name} -> {end_name}")

            for local_step in range(steps):
                alpha = local_step / max(steps - 1, 1)
                joints = (1 - alpha) * start_joints + alpha * end_joints

                set_robot_joints(robot_id, joints)
                step_physics()
                render_frame(writer, writer_topdown)

                if frame_index % 50 == 0:
                    print(f"  Frame {frame_index}: {end_name}")
                frame_index += 1

            if end_action == "grasp":
                grasp_constraint_id = try_create_grasp_constraint(robot_id, piece_id)
            elif end_action == "release":
                release_grasp_constraint(grasp_constraint_id)
                grasp_constraint_id = None

        for _ in range(30):
            step_physics()
            render_frame(writer, writer_topdown)
    finally:
        if "grasp_constraint_id" in locals() and grasp_constraint_id is not None:
            release_grasp_constraint(grasp_constraint_id)
        writer.close()
        writer_topdown.close()


def main() -> None:
    from_square = "e2"
    to_square = "e4"

    print(f"Building {from_square}->{to_square} chess move path...")
    kinematics = make_kinematics()
    waypoints = build_move_to_square_v2_path(kinematics, from_square, to_square)

    setup_pybullet()
    try:
        create_board()
        piece_id = create_piece(from_square)
        robot_id = load_robot()

        print(f"{from_square} center: {chess_to_board_xy(from_square)}")
        print(f"{to_square} center: {chess_to_board_xy(to_square)}")
        print(f"Square size: {SQUARE_SIZE} m")
        print("Starting PyBullet move simulation...")

        simulate_path(robot_id, piece_id, waypoints)
        print("Videos saved:")
        print("  - so101_robot_e2e4_realistic.mp4")
        print("  - so101_robot_e2e4_realistic_topdown.mp4")
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
