import pybullet as p
import pybullet_data
import imageio
import numpy as np
import sys
from pathlib import Path
from chess_traj import pickupmove_traj
from chess_traj import chess_to_xy
from testkinematics import kinematics

movelist = pickupmove_traj('c1', 'c5')  # Example move from e2 to e4
renderfreq = 10
WIDTH, HEIGHT = 640, 360
runid = "grasp_calib"


# ----------------------------------------
# Joint mapping
# ----------------------------------------

ARM_JOINTS = [0, 1, 2, 3, 4]
GRIPPER_IDX = 6

CONTROL_JOINTS = ARM_JOINTS + [GRIPPER_IDX]


# ----------------------------------------
# Interpolation
# ----------------------------------------

def interpolate_joints(start, end, alpha):

    q = (1 - alpha) * start + alpha * end

    # Delay gripper interpolation until end
    grip_start_alpha = 0.85

    if alpha < grip_start_alpha:

        q[GRIPPER_IDX] = start[GRIPPER_IDX]

    else:

        grip_alpha = (
            (alpha - grip_start_alpha)
            / (1.0 - grip_start_alpha)
        )

        q[GRIPPER_IDX] = (
            (1 - grip_alpha) * start[GRIPPER_IDX]
            + grip_alpha * end[GRIPPER_IDX]
        )

    return q



##########


def interpolate_joints(start, end, alpha):
    return (1 - alpha) * start + alpha * end


# def interpolate_joints(start, end, alpha):

#     q = (1 - alpha) * start + alpha * end

#     # ----------------------------------------
#     # Delay gripper interpolation
#     # ----------------------------------------

#     grip_start_alpha = 0.85

#     if alpha < grip_start_alpha:

#         # Hold initial gripper value
#         q[5] = start[5]

#     else:

#         # Remap alpha from [0.85,1] -> [0,1]
#         grip_alpha = (
#             (alpha - grip_start_alpha)
#             / (1.0 - grip_start_alpha)
#         )

#         q[5] = (
#             (1 - grip_alpha) * start[5]
#             + grip_alpha * end[5]
#         )

#     return q


create_sim = True

if create_sim:    # Initialize PyBullet in HEADLESS mode
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    print("Setting up physics environment...")

    # Load ground plane
    plane_id = p.loadURDF("plane.urdf", [0, 0, 0])

    # Load SO101 robot
    robot_id = None
    try:
        urdf_path = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
        robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)


        p.changeDynamics(
            robot_id,
            6,
            lateralFriction=5.0,
            spinningFriction=1.0,
            contactStiffness=10000,
            contactDamping=100
        )
        print(f"✓ Loaded SO101 from URDF")
        print(f"✓ Number of joints: {p.getNumJoints(robot_id)}")
    except Exception as e:
        print(f"⚠ Error loading robot: {e}")




    # Create chessboard at position (0.3, 0, 0)
    board_x, board_y, board_z = 0.3, 0, 0
    square_size = 0.04
    board_size = 8 * square_size

    # Board base
    board_base_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[board_size/2, board_size/2, 0.005])
    board_base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[board_size/2, board_size/2, 0.005], 
                                            rgbaColor=[0.3, 0.3, 0.3, 1])
    board_base_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=board_base_shape,
                                    baseVisualShapeIndex=board_base_visual, 
                                    basePosition=[board_x, board_y, board_z])

    # Add checkerboard squares
    for row in range(8):
        for col in range(8):
            x = board_x - board_size/2 + (col + 0.5) * square_size
            y = board_y - board_size/2 + (row + 0.5) * square_size
            z = board_z + 0.008
            
            if (row + col) % 2 == 0:
                color = [1, 1, 1, 0.5]
            else:
                color = [0.1, 0.1, 0.1, 0.5]
            
            square_visual = p.createVisualShape(p.GEOM_BOX, 
                                            halfExtents=[square_size/2 - 0.001, square_size/2 - 0.001, 0.001],
                                            rgbaColor=color)
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=square_visual, basePosition=[x, y, z])

    print(f"✓ Chessboard created")

    # Create LARGER chess pieces (2x scale: 0.024 radius)
    piece_positions = [
        (-0.14, -0.14), (-0.10, -0.14), (-0.06, -0.14), (-0.02, -0.14),
        (0.02, -0.14), (0.06, -0.14), (0.10, -0.14), (0.14, -0.14),
    ]

    piece_ids = []
    for i, (px, py) in enumerate(piece_positions):
        world_x = board_x + px
        world_y = board_y + py
        world_z = board_z + 0.035
        
        # Larger pieces: radius 0.024 (2x), height 0.04 (2x)
        piece_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.012, height=0.04)
        piece_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.012, length=0.04, 
                                        rgbaColor=[1, 0, 0, 1])
        piece_id = p.createMultiBody(baseMass=0.01, baseCollisionShapeIndex=piece_shape,
                                    baseVisualShapeIndex=piece_visual,
                                    basePosition=[world_x, world_y, world_z])
        p.changeDynamics(piece_id, -1, linearDamping=0.04, angularDamping=0.04, lateralFriction=2)
        piece_ids.append(piece_id)

    print(f"✓ Created {len(piece_ids)} large chess pieces")
    print("Piece IDs:", piece_ids)
    print("✓ All objects loaded and ready!")





# import sys
# sys.exit()

# Robot joint waypoints (from simfk.py)

home = np.array([96.92, -107.87, 97.36, 65.19, -29.85, 4.63])
corner1 = np.array([97.32, 0.40, 28.40, 66.55, 177.80, 4.95])
corner2 = np.array([38.59, 60.88, -58.55, 100.48, 178.15, 4.95])



# for i in range(p.getNumJoints(robot_id)):
#     info = p.getJointInfo(robot_id, i)
#     print(i, info[1].decode())

# sys.exit()

# corner1xyz = kinematics.forward_kinematics(corner1)[:3,3]
# print("Corner 1 XYZ:", corner1xyz)

# import sys 
# sys.exit()


# Convert to radians
home_rad = np.deg2rad(home)
corner1_rad = np.deg2rad(corner1)
corner2_rad = np.deg2rad(corner2)

# Define movement sequence
# moves = [
#     (home_rad, 100, "Move to HOME"),
#     (corner1_rad, 150, "Move to CORNER1"),
#     (corner2_rad, 150, "Move to CORNER2"),
#     (home_rad, 150, "Return to HOME"),
# ]

moves = [(np.deg2rad(pos), 50, f"Move to {pos}") for pos in movelist]

# Setup video recording with THREE cameras
camera_params = {
    'eye': [0.0, -0.6, 0.25],
    'target': [0.3, 0.0, 0.05],
    'up': [0, 0, 1],
}

top_down_camera_params = {
    'eye': [0.3, 0, 0.6],
    'target': [0.3, 0, 0],
    'up': [0, -1, 0],
}

# Create subdirectory for this run
output_dir = Path(f"./recordings/{runid}")
output_dir.mkdir(parents=True, exist_ok=True)

video_path = output_dir / "so101_robot_moves.mp4"
video_topdown_path = output_dir / "so101_robot_moves_topdown.mp4"
writer = imageio.get_writer(str(video_path), fps=30, codec='libx264', quality=8)
writer_topdown = imageio.get_writer(str(video_topdown_path), fps=30, codec='libx264', quality=8)

print("Starting robot movement simulation...")

# Calculate total steps
total_steps = sum(steps for _, steps, _ in moves)
current_start_pos = home_rad.copy()

# ----------------------------------------
# Main simulation loop
# ----------------------------------------

move_idx = 0
move_local_step = 0

current_start_pos = moves[0][0].copy()
current_target_pos, move_steps, move_name = moves[0]

SIM_JOINT_MAP = [0, 1, 2, 3, 4, 6]

SIM_JOINT_MAP = [0, 1, 2, 3, 4, 6]

try:

    for global_step in range(total_steps + 100):


        # ----------------------------------------
        # Advance move
        # ----------------------------------------

        if move_local_step >= move_steps and move_idx < len(moves) - 1:

            move_idx += 1

            current_start_pos = current_target_pos.copy()

            current_target_pos, move_steps, move_name = moves[move_idx]

            move_local_step = 0

        # ----------------------------------------
        # Interpolate trajectory
        # ----------------------------------------

        alpha = min(move_local_step / move_steps, 1.0)

        target_joints = interpolate_joints(
            current_start_pos,
            current_target_pos,
            alpha
        )

        # print("Target joints:", target_joints)
        # sys.exit()

        # ----------------------------------------
        # Apply controls
        # ----------------------------------------

        if robot_id is not None:

            for traj_idx, sim_idx in enumerate(SIM_JOINT_MAP):

                force = 50

                if traj_idx == 5:
                    force = 500

                p.setJointMotorControl2(
                    robot_id,
                    sim_idx,
                    p.POSITION_CONTROL,
                    targetPosition=target_joints[traj_idx],
                    force=force
                )

                

        # ----------------------------------------
        # Physics step
        # ----------------------------------------

        p.stepSimulation()


        piece_pos, _ = p.getBasePositionAndOrientation(piece_ids[0])  # Get position of the first piece

        if piece_pos[2]>0.5:

            fk_pose = kinematics.forward_kinematics(np.rad2deg(target_joints))

            gripper_origin = fk_pose[:3,3]
            gripper_rot = fk_pose[:3,:3]

            # Recover grasp offset in LOCAL gripper coordinates
            grasp_offset = (
                gripper_rot.T
                @ (np.array(piece_pos) - gripper_origin)
            )

            print("Recovered GRASP_OFFSET:")
            print(grasp_offset)
            sys.exit()


        if global_step % renderfreq == 0:
            # ----------------------------------------
            # Camera rendering
            # ----------------------------------------

            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=WIDTH / HEIGHT,
                nearVal=0.01,
                farVal=100
            )

            # ---------------- Perspective camera ----------------

            view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_params['eye'],
                cameraTargetPosition=camera_params['target'],
                cameraUpVector=camera_params['up']
            )

            w, h, rgba, _, _ = p.getCameraImage(
                WIDTH,
                HEIGHT,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix
            )

            img = np.array(rgba, dtype=np.uint8).reshape((h, w, 4))

            rgb_array = img[:, :, :3]

            writer.append_data(rgb_array)

            # ---------------- Top-down camera ----------------

            view_matrix_topdown = p.computeViewMatrix(
                cameraEyePosition=top_down_camera_params['eye'],
                cameraTargetPosition=top_down_camera_params['target'],
                cameraUpVector=top_down_camera_params['up']
            )

            w2, h2, rgba2, _, _ = p.getCameraImage(
                WIDTH,
                HEIGHT,
                viewMatrix=view_matrix_topdown,
                projectionMatrix=proj_matrix
            )

            img2 = np.array(rgba2, dtype=np.uint8).reshape((h2, w2, 4))

            rgb_array_topdown = img2[:, :, :3]

            writer_topdown.append_data(rgb_array_topdown)

        # ----------------------------------------
        # Debug
        # ----------------------------------------

        if global_step % 50 == 0:

            actual_gripper = p.getJointState(
                robot_id,
                6
            )[0]

            print(
                f"Frame {global_step} | "
                f"{move_name} | "
                f"alpha={alpha:.2f} | "
                f"target_gripper={target_joints[5]:.2f} | "
                f"actual_gripper={actual_gripper:.2f}"
            )

        move_local_step += 1

finally:

    writer.close()
    writer_topdown.close()

    print("Videos saved.")



p.disconnect()