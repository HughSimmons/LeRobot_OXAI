import pybullet as p
import pybullet_data
import imageio
import numpy as np
import sys
import matplotlib.pyplot as plt

from pathlib import Path
from chess_traj import pickupmove_traj, chess_to_xy, gripper_angle_closed, gripper_angle_open
from testkinematics import kinematics
board_origin = (0.25, 0, 0)  # Must match the origin used in pybsim_chess.py
# video_on = False
video_on = True
# runid = "multisim_dev_pickplace"
runid = "multisim_h1a1_a1h1"


FILES = "abcdefgh"
# FILES = "fgh"

squares = [
    f"{file}{rank}"
    # for rank in range(1, 9)
    for rank in range(1, 2)
    for file in FILES
]




# GRASP_OFFSET = np.array([0, 0, 0.02])  # 2cm offset for grasping
GRASP_OFFSET = np.array([
    -0.025,
    -0.0,
    -0.005
])

# GRASP_OFFSET = np.array([
#     -0.025,
#     -0.0,
#     -0.2
# ])


# GRASP_OFFSET = np.array([
#     -0.015,
#     -0.0,
#     -0.005
# ])


renderfreq = 50
WIDTH, HEIGHT = 640, 360

# Robot joint waypoints (from simfk.py)
home = np.array([96.92, -107.87, 97.36, 65.19, -29.85, 4.63])
corner1 = np.array([97.32, 0.40, 28.40, 66.55, 177.80, 4.95])
corner2 = np.array([38.59, 60.88, -58.55, 100.48, 178.15, 4.95])

# Convert to radians
home_rad = np.deg2rad(home)
corner1_rad = np.deg2rad(corner1)
corner2_rad = np.deg2rad(corner2)

# init_posit = ["a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1"]


# ----------------------------------------
# Interpolation
# ----------------------------------------

def interpolate_joints(start, end, alpha):
    return (1 - alpha) * start + alpha * end

##fn to generate pices 
def create_piece(sq="a1"):
    world_x, world_y, _ = chess_to_xy(sq, board_origin=board_origin)
    world_z = 0.04

    # Larger pieces: radius 0.024 (2x), height 0.04 (2x)
    piece_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.012, height=0.04)
    piece_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.012, length=0.04, 
                                    rgbaColor=[1, 0, 0, 1])
    piece_id = p.createMultiBody(baseMass=0.05, baseCollisionShapeIndex=piece_shape,
                                baseVisualShapeIndex=piece_visual,
                                basePosition=[world_x, world_y, world_z])
    # p.changeDynamics(piece_id, -1, linearDamping=0.04, angularDamping=0.04, lateralFriction=2)
    # p.changeDynamics(piece_id, -1, linearDamping=0.04, angularDamping=0.04, lateralFriction=3)

    p.changeDynamics(
        piece_id,
        -1,
        lateralFriction=10.0,
        rollingFriction=0.2,
        spinningFriction=0.2,
        linearDamping=0.5,
        angularDamping=1.0
    )
    return(piece_id)


# ----------------------------------------
# Joint mapping
# ----------------------------------------

ARM_JOINTS = [0, 1, 2, 3, 4]
GRIPPER_IDX = 6

CONTROL_JOINTS = ARM_JOINTS + [GRIPPER_IDX]


def simchess(i,j, GRASP_OFFSET):
    sq1, sq2 = squares[i], squares[j]
    init_posit = [sq1]
    simid = f"{sq1}_to_{sq2}"

    piece_xyz_log = []
    ee_xyz_log = []
    distance_log = []
    gripper_log = []
    time_log = []

    movelist, closeidx = pickupmove_traj(sq1, sq2, board_origin=board_origin, GRASP_OFFSET=GRASP_OFFSET)  
    movelist2, closeidx2 = pickupmove_traj(sq2, sq1, board_origin=board_origin, GRASP_OFFSET=GRASP_OFFSET)  
    movelist.extend(movelist2)


    create_sim = True

    if create_sim:    # Initialize PyBullet in HEADLESS mode
        # physics_client = p.connect(p.DIRECT)
        # p.setGravity(0, 0, -9.81)
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.resetSimulation()

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
                lateralFriction=10.0,
                spinningFriction=1.0,
                contactStiffness=10000,
                contactDamping=100
            )
            print(f"✓ Loaded SO101 from URDF")
            print(f"✓ Number of joints: {p.getNumJoints(robot_id)}")
        except Exception as e:
            print(f"⚠ Error loading robot: {e}")




        # Create chessboard at position (0.22, 0, 0)
        board_x, board_y, board_z = board_origin
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

        piece_ids = []
        for sq in init_posit:
            piece_id = create_piece(sq)
            piece_ids.append(piece_id)

        print(f"✓ Created {len(piece_ids)} large chess pieces")
        print("Piece IDs:", piece_ids)
        print("✓ All objects loaded and ready!")


    moves = [(np.deg2rad(pos), 50, f"Move to {pos}") for pos in movelist]

    if video_on:
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
        output_dir = Path(f"./recordings/{runid}/{simid}")
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
    pickup_success = False

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

                        if move_idx == closeidx:
                            actual_gripper = p.getJointState(robot_id, 6)[0]


                        if move_idx > closeidx+1:
                            # target_joints[5] = p.getJointState(robot_id, 6)[0]
                            # target_joints[5] = actual_gripper
                        #     target_joints[5] = actual_gripper - np.deg2rad(0.1)
                            force = 50
                            #if gripper closed, dont alter

                        # if move_idx == closeidx+10:
                        #     for _ in range(1000):
                        #         p.stepSimulation()

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
            ###
            piece_pos, _ = p.getBasePositionAndOrientation(
                piece_ids[0]
            )

            fk_pose = kinematics.forward_kinematics(
                np.rad2deg(target_joints)
            )

            ee_xyz = fk_pose[:3,3]

            piece_xyz_log.append(piece_pos)

            ee_xyz_log.append(ee_xyz)

            distance_log.append(
                np.linalg.norm(
                    np.array(piece_pos) - ee_xyz
                )
            )

            gripper_log.append(
                p.getJointState(robot_id, 6)[0]
            )

            time_log.append(global_step / 240.0)
            ###




            piece_pos, _ = p.getBasePositionAndOrientation(piece_ids[0])  # Get position of the first piece

            if piece_pos[2]>0.05 and global_step > 50:  # Check if the piece has been lifted off the board (adjust threshold as needed)
            # if piece_pos[2]>0.1:

                # fk_pose = kinematics.forward_kinematics(np.rad2deg(target_joints))

                # gripper_origin = fk_pose[:3,3]
                # gripper_rot = fk_pose[:3,:3]

                # # Recover grasp offset in LOCAL gripper coordinates
                # grasp_offset = (
                #     gripper_rot.T
                #     @ (np.array(piece_pos) - gripper_origin)
                # )

                # print("Recovered GRASP_OFFSET:")
                # print(grasp_offset)
                pickup_success = True
                # break
                # return(pickup_success)
                # sys.exit()

            if video_on:
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
        if video_on:
            writer.close()
            writer_topdown.close()

            print("Videos saved.")

    # p.disconnect()

    piece_xyz_log = np.array(piece_xyz_log)
    ee_xyz_log = np.array(ee_xyz_log)
    distance_log = np.array(distance_log)
    gripper_log = np.array(gripper_log)
    time_log = np.array(time_log)


    plt.figure(figsize=(10,6))

    plt.plot(time_log, piece_xyz_log[:,0], label="piece x")
    plt.plot(time_log, ee_xyz_log[:,0], "--", label="ee x")

    plt.plot(time_log, piece_xyz_log[:,1], label="piece y")
    plt.plot(time_log, ee_xyz_log[:,1], "--", label="ee y")

    plt.plot(time_log, piece_xyz_log[:,2], label="piece z")
    plt.plot(time_log, ee_xyz_log[:,2], "--", label="ee z")

    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")

    plt.savefig(
        output_dir / "xyz_vs_time.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


    plt.figure(figsize=(10,4))

    plt.plot(
        time_log,
        distance_log
    )

    plt.xlabel("Time (s)")
    plt.ylabel("EE-Piece distance (m)")

    plt.savefig(
        output_dir / "distance_vs_time.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


    print("Pick up success:", pickup_success)
    return pickup_success


# test = simchess(0, 16, GRASP_OFFSET)

# print("Test result:", test)

def grasptest(grasp_offset):

    success_count = 0
    failsquares = []

    # for a in range(len(squares)):
    for a in range(7,len(squares)):
        for b in range(1):
            # print(f"Testing move from {squares[a]} to {squares[b]}...")
            result = simchess(a, b, grasp_offset)

            if result:
                success_count += 1

            else:
                failsquares.append(squares[a])

            # print(f"Result: {'Success' if result else 'Failure'}\n")

    print(f"Total successful pickups: {success_count} out of {len(squares)} moves")
    print(f"Failed squares: {failsquares}")
    return success_count


# base_grasp_offset = np.array([
#     -0.025,
#     0.0,
#     -0.005
# ])

base_grasp_offset = np.array([
    -0.015,
    0.0,
    -0.005
])

# base_grasp_offset = np.array([
#     -0.013,
#     0,
#     -0.005
# ])


delta = 0.00025   # 5 mm spacing

grasp_offsets = []

# dellist = [-2,-1,0,1]
# dellist = [-4,-2,0 ,2,4]
dellist = [0]
# dellist = [delta*x for x in dellist]


# print(-0.025+4*0.0025)

# sys.exit()

for dx in dellist:
    # for dy in [-delta, 0, delta]:
    #     for dz in [-delta, 0, delta]:
    dy,dz = 0, 0

    offset = base_grasp_offset + np.array([
        dx,
        dy,
        dz
    ])

    grasp_offsets.append(offset)


physics_client = p.connect(p.DIRECT)

p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())



# print(grasp_offsets)
cnt = 0
successlist = []
for grasp_offset in grasp_offsets:
    print(f"Testing GRASP_OFFSET: {cnt}")
    cnt+=1
    success = grasptest(grasp_offset)
    successlist.append(success)
    print("-----------------------------------\n")
    print("CNT:", cnt)

p.disconnect()
print(successlist)

