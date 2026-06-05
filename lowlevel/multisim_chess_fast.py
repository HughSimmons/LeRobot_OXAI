import pybullet as p
import pybullet_data
import imageio
import numpy as np
import sys
import time
from pathlib import Path
from chess_traj import pickupmove_traj, chess_to_xy, gripper_angle_closed, gripper_angle_open
from testkinematics import kinematics
board_origin = (0.25, 0, 0)  # Must match the origin used in pybsim_chess.py
video_on = False
# video_on = True
runid = "multisim_place_fast_correction"

# Best PLACE_OFFSET: [-0.0165  0.0015 -0.005 ]
# FILES = "abcdefgh"
FILES = "f"

squares = [
    f"{file}{rank}"
    for rank in range(1, 9)
    # for rank in range(5, 6)
    for file in FILES
]




# GRASP_OFFSET = np.array([0, 0, 0.02])  # 2cm offset for grasping
# GRASP_OFFSET = np.array([
#     -0.025,
#     -0.0,
#     -0.005
# ])

# GRASP_OFFSET = np.array([
#     -0.025,
#     -0.0,
#     -0.2
# ])


GRASP_OFFSET = np.array([
    -0.015,
    -0.0,
    -0.005
])


# GRASP_OFFSET = np.array([
#     -0.0121,
#     -0.0,
#     -0.005
# ])


# PLACE_OFFSET = np.array([
#     -0.015,
#     # 0,
#     0.005,
#     0
# ])

# PLACE_OFFSET = GRASP_OFFSET.copy()
PLACE_OFFSET = np.array([-0.01845, 0.00115, -0.005])


renderfreq = 50
WIDTH, HEIGHT = 640, 360
SOLVER_ITERATIONS = 200
SOLVER_SUBSTEPS = 4
POST_MOVE_SETTLE_STEPS = 1000
SEARCH_SOLVER_ITERATIONS = 200
SEARCH_SOLVER_SUBSTEPS = 4
SEARCH_POST_MOVE_SETTLE_STEPS = 200
VERIFY_TOP_K = 1
MEASURED_CORRECTION_ROUNDS = 1
# COARSE_TO_FINE_PASSES = (
#     {"sample_count": 5, "delta": 0.0010},
#     {"sample_count": 5, "delta": 0.00035},
#     {"sample_count": 5, "delta": 0.00010},
# )
COARSE_TO_FINE_PASSES = (
    {"sample_count": 7, "delta": 0.0015},
    {"sample_count": 5, "delta": 0.00035},
    {"sample_count": 3, "delta": 0.00010},
)# Best PLACE_OFFSET: [-0.0165  0.0015 -0.005 ]
# [-0.01845  0.00115 -0.005  ]

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

def catmull_rom_joints(prev_pos, start_pos, end_pos, next_pos, alpha):
    alpha2 = alpha * alpha
    alpha3 = alpha2 * alpha

    return 0.5 * (
        (2 * start_pos)
        + (-prev_pos + end_pos) * alpha
        + (2 * prev_pos - 5 * start_pos + 4 * end_pos - next_pos) * alpha2
        + (-prev_pos + 3 * start_pos - 3 * end_pos + next_pos) * alpha3
    )


def interpolate_joints(moves, move_idx, alpha):
    start_idx = max(move_idx - 1, 0)
    end_idx = move_idx
    prev_idx = max(start_idx - 1, 0)
    next_idx = min(end_idx + 1, len(moves) - 1)

    prev_pos = moves[prev_idx][0]
    start_pos = moves[start_idx][0]
    end_pos = moves[end_idx][0]
    next_pos = moves[next_idx][0]

    target_joints = end_pos.copy()
    target_joints[:5] = catmull_rom_joints(
        prev_pos[:5],
        start_pos[:5],
        end_pos[:5],
        next_pos[:5],
        alpha
    )

    return target_joints

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
        lateralFriction=1.0,
        rollingFriction=0.02,
        spinningFriction=0.02,
        linearDamping=0.2,
        angularDamping=0.2
    )
    return(piece_id)


# ----------------------------------------
# Joint mapping
# ----------------------------------------

ARM_JOINTS = [0, 1, 2, 3, 4]
GRIPPER_IDX = 6

CONTROL_JOINTS = ARM_JOINTS + [GRIPPER_IDX]


def setup_sim_world(from_square):
    p.resetSimulation()
    p.setPhysicsEngineParameter(
        numSolverIterations=SOLVER_ITERATIONS,
        numSubSteps=SOLVER_SUBSTEPS
    )

    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    print("Setting up physics environment...")

    plane_id = p.loadURDF("plane.urdf", [0, 0, 0])

    robot_id = None
    try:
        urdf_path = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
        robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)

        p.changeDynamics(
            robot_id,
            6,
            lateralFriction=2.0,
            spinningFriction=0.2,
            contactStiffness=10000,
            contactDamping=100
        )
        print(f"✓ Loaded SO101 from URDF")
        print(f"✓ Number of joints: {p.getNumJoints(robot_id)}")
    except Exception as e:
        print(f"⚠ Error loading robot: {e}")

    board_x, board_y, board_z = board_origin
    square_size = 0.04
    board_size = 8 * square_size

    board_base_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[board_size/2, board_size/2, 0.005])
    board_base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[board_size/2, board_size/2, 0.005], 
                                            rgbaColor=[0.3, 0.3, 0.3, 1])
    board_base_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=board_base_shape,
                                    baseVisualShapeIndex=board_base_visual, 
                                    basePosition=[board_x, board_y, board_z])

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

    piece_id = create_piece(from_square)
    piece_ids = [piece_id]

    print(f"✓ Created {len(piece_ids)} large chess pieces")
    print("Piece IDs:", piece_ids)
    print("✓ All objects loaded and ready!")

    state_id = p.saveState()
    return {
        "from_square": from_square,
        "state_id": state_id,
        "plane_id": plane_id,
        "robot_id": robot_id,
        "board_base_id": board_base_id,
        "piece_ids": piece_ids,
    }


def run_sim_move(
    world,
    from_square,
    to_square,
    grasp_offset,
    place_offset=None,
    return_metrics=False,
    record_video=None,
    solver_iterations=SOLVER_ITERATIONS,
    solver_substeps=SOLVER_SUBSTEPS,
    post_move_settle_steps=POST_MOVE_SETTLE_STEPS
):
    if from_square != world["from_square"]:
        raise ValueError(f"world was initialized for {world['from_square']}, not {from_square}")

    p.restoreState(world["state_id"])
    p.setPhysicsEngineParameter(
        numSolverIterations=solver_iterations,
        numSubSteps=solver_substeps
    )

    sq1, sq2 = from_square, to_square
    simid = f"{sq1}_to_{sq2}"
    active_place_offset = PLACE_OFFSET if place_offset is None else place_offset
    video_enabled = video_on if record_video is None else record_video
    robot_id = world["robot_id"]
    piece_ids = world["piece_ids"]

    movelist, closeidx = pickupmove_traj(sq1, sq2, board_origin=board_origin, GRASP_OFFSET=grasp_offset, PLACE_OFFSET=active_place_offset)  


    moves = [(np.deg2rad(pos), 50, f"Move to {pos}") for pos in movelist]

    if video_enabled:
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

        for global_step in range(total_steps + post_move_settle_steps):


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
                moves,
                move_idx,
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

            if video_enabled:
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

                # print(
                #     f"Frame {global_step} | "
                #     f"{move_name} | "
                #     f"alpha={alpha:.2f} | "
                #     f"target_gripper={target_joints[5]:.2f} | "
                #     f"actual_gripper={actual_gripper:.2f}"
                # )

            move_local_step += 1

    finally:
        if video_enabled:
            writer.close()
            writer_topdown.close()

            print("Videos saved.")

    # p.disconnect()

    final_piece_pos, final_piece_orn = p.getBasePositionAndOrientation(piece_ids[0])
    expected_piece_pos = chess_to_xy(sq2, board_origin=board_origin)
    final_piece_pos = np.array(final_piece_pos)
    expected_piece_pos = np.array(expected_piece_pos)
    position_error = final_piece_pos - expected_piece_pos
    xy_error = np.linalg.norm(position_error[:2])
    z_error = abs(position_error[2])
    final_piece_rot = np.array(p.getMatrixFromQuaternion(final_piece_orn)).reshape(3, 3)
    piece_axis_z = final_piece_rot[:, 2]
    final_tilt_deg = np.rad2deg(np.arccos(np.clip(abs(piece_axis_z[2]), -1.0, 1.0)))
    final_euler_deg = np.rad2deg(p.getEulerFromQuaternion(final_piece_orn))

    print("Pick up success:", pickup_success)
    if return_metrics:
        return {
            "pickup_success": pickup_success,
            "from_square": sq1,
            "to_square": sq2,
            "place_offset": active_place_offset.copy(),
            "final_position": final_piece_pos,
            "final_orientation": final_piece_orn,
            "expected_position": expected_piece_pos,
            "position_error": position_error,
            "xy_error": xy_error,
            "z_error": z_error,
            "final_tilt_deg": final_tilt_deg,
            "final_euler_deg": final_euler_deg,
            "solver_iterations": solver_iterations,
            "solver_substeps": solver_substeps,
            "post_move_settle_steps": post_move_settle_steps,
        }

    return pickup_success


def simchess(i, j, GRASP_OFFSET, place_offset=None, return_metrics=False, record_video=None):
    sq1, sq2 = squares[i], squares[j]
    world = setup_sim_world(sq1)
    try:
        return run_sim_move(
            world,
            sq1,
            sq2,
            GRASP_OFFSET,
            place_offset=place_offset,
            return_metrics=return_metrics,
            record_video=record_video
        )
    finally:
        p.removeState(world["state_id"])


# test = simchess(0, 16, GRASP_OFFSET)

# print("Test result:", test)

def grasptest(grasp_offset):

    success_count = 0
    failsquares = []

    for a in range(len(squares)):
    # for a in range(6,len(squares)):
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


def score_place_result(result):
    if not result["pickup_success"]:
        return 1000.0

    return result["xy_error"] + 0.25 * result["z_error"]


def make_centered_samples(sample_count):
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")

    center = (sample_count - 1) / 2
    return np.arange(sample_count) - center


def find_best_place_offset(
    from_square="e1",
    to_square="e5",
    grasp_offset=GRASP_OFFSET,
    base_place_offset=None,
    delta=0.0015,
    sample_count=1,
    world=None,
    solver_iterations=SEARCH_SOLVER_ITERATIONS,
    solver_substeps=SEARCH_SOLVER_SUBSTEPS,
    post_move_settle_steps=SEARCH_POST_MOVE_SETTLE_STEPS
):
    if base_place_offset is None:
        base_place_offset = grasp_offset.copy()

    local_world = world
    should_remove_state = False
    if local_world is None:
        local_world = setup_sim_world(from_square)
        should_remove_state = True

    samples = make_centered_samples(sample_count)
    results = []

    try:
        for dx in samples:
            for dy in samples:
            #     for dz in samples:
                place_offset = base_place_offset + delta * np.array([dx, dy, 0])
                print(f"Testing PLACE_OFFSET: {place_offset}")

                result = run_sim_move(
                    local_world,
                    from_square,
                    to_square,
                    grasp_offset,
                    place_offset=place_offset,
                    return_metrics=True,
                    record_video=False,
                    solver_iterations=solver_iterations,
                    solver_substeps=solver_substeps,
                    post_move_settle_steps=post_move_settle_steps
                )

                result["score"] = score_place_result(result)
                results.append(result)

                print(
                    f"score={result['score']:.5f} | "
                    f"xy_error={result['xy_error']:.5f} | "
                    f"z_error={result['z_error']:.5f} | "
                    f"pickup={result['pickup_success']}"
                )
    finally:
        if should_remove_state:
            p.removeState(local_world["state_id"])

    results.sort(key=lambda item: item["score"])

    print("\nBest PLACE_OFFSET results:")
    for rank, result in enumerate(results[:10], start=1):
        print(
            f"{rank}: offset={result['place_offset']} | "
            f"score={result['score']:.5f} | "
            f"xy_error={result['xy_error']:.5f} | "
            f"z_error={result['z_error']:.5f} | "
            f"final_position={result['final_position']}"
        )

    return results[0], results


def find_best_place_offset_coarse_to_fine(
    world,
    from_square,
    to_square,
    grasp_offset=GRASP_OFFSET,
    base_place_offset=None,
    passes=COARSE_TO_FINE_PASSES
):
    if base_place_offset is None:
        base_place_offset = grasp_offset.copy()

    center_offset = base_place_offset.copy()
    all_results = []

    print("\nStarting coarse-to-fine cheap search...")
    for pass_idx, pass_settings in enumerate(passes, start=1):
        pass_start = time.perf_counter()
        sample_count = pass_settings["sample_count"]
        delta = pass_settings["delta"]

        print(
            f"\nCoarse-to-fine pass {pass_idx}/{len(passes)} | "
            f"center={center_offset} | delta={delta} | sample_count={sample_count}"
        )

        best_result, pass_results = find_best_place_offset(
            from_square=from_square,
            to_square=to_square,
            grasp_offset=grasp_offset,
            base_place_offset=center_offset,
            delta=delta,
            sample_count=sample_count,
            world=world,
            solver_iterations=SEARCH_SOLVER_ITERATIONS,
            solver_substeps=SEARCH_SOLVER_SUBSTEPS,
            post_move_settle_steps=SEARCH_POST_MOVE_SETTLE_STEPS
        )

        for result in pass_results:
            result["search_pass"] = pass_idx
            result["search_delta"] = delta

        all_results.extend(pass_results)
        center_offset = best_result["place_offset"].copy()
        pass_elapsed = time.perf_counter() - pass_start

        print(
            f"Pass {pass_idx} best offset={center_offset} | "
            f"score={best_result['score']:.5f} | elapsed={pass_elapsed:.2f}s"
        )

    all_results.sort(key=lambda item: item["score"])
    return all_results[0], all_results


def get_release_gripper_rotation(from_square, to_square, grasp_offset, place_offset):
    movelist, closeidx = pickupmove_traj(
        from_square,
        to_square,
        board_origin=board_origin,
        GRASP_OFFSET=grasp_offset,
        PLACE_OFFSET=place_offset
    )

    release_idx = None
    for idx in range(closeidx + 1, len(movelist)):
        gripper_now = movelist[idx][5]
        gripper_prev = movelist[idx - 1][5]
        if (
            np.isclose(gripper_now, gripper_angle_open)
            and np.isclose(gripper_prev, gripper_angle_closed)
        ):
            release_idx = idx
            break

    if release_idx is None:
        release_idx = max(len(movelist) - 5, 0)

    release_pose = kinematics.forward_kinematics(movelist[release_idx])
    return release_pose[:3, :3], release_idx


def find_best_place_offset_measured_correction(
    world,
    from_square,
    to_square,
    grasp_offset=GRASP_OFFSET,
    base_place_offset=None,
    rounds=MEASURED_CORRECTION_ROUNDS,
    correction_gain=1.0,
    correct_z=False,
    solver_iterations=SOLVER_ITERATIONS,
    solver_substeps=SOLVER_SUBSTEPS,
    post_move_settle_steps=POST_MOVE_SETTLE_STEPS
):
    if base_place_offset is None:
        base_place_offset = grasp_offset.copy()

    place_offset = base_place_offset.copy()
    results = []

    print("\nStarting measured-error placement correction...")
    for round_idx in range(1, rounds + 1):
        round_start = time.perf_counter()
        print(f"\nCorrection round {round_idx}/{rounds} | PLACE_OFFSET={place_offset}")

        result = run_sim_move(
            world,
            from_square,
            to_square,
            grasp_offset,
            place_offset=place_offset,
            return_metrics=True,
            record_video=False,
            solver_iterations=solver_iterations,
            solver_substeps=solver_substeps,
            post_move_settle_steps=post_move_settle_steps
        )

        result["score"] = score_place_result(result)
        result["correction_round"] = round_idx
        results.append(result)

        measured_error = result["position_error"].copy()
        world_correction = measured_error.copy()
        if not correct_z:
            world_correction[2] = 0.0

        release_rot, release_idx = get_release_gripper_rotation(
            from_square,
            to_square,
            grasp_offset,
            place_offset
        )
        gripper_frame_correction = release_rot.T @ world_correction
        correction = correction_gain * gripper_frame_correction

        next_place_offset = place_offset + correction
        round_elapsed = time.perf_counter() - round_start
        result["release_idx"] = release_idx
        result["world_correction"] = world_correction.copy()
        result["gripper_frame_correction"] = gripper_frame_correction.copy()

        print(
            f"score={result['score']:.5f} | "
            f"xy_error={result['xy_error']:.5f} | "
            f"z_error={result['z_error']:.5f} | "
            f"pickup={result['pickup_success']} | "
            f"measured_error={measured_error} | "
            f"gripper_frame_correction={gripper_frame_correction} | "
            f"next_PLACE_OFFSET={next_place_offset} | "
            f"elapsed={round_elapsed:.2f}s"
        )

        place_offset = next_place_offset

    results.sort(key=lambda item: item["score"])

    print("\nBest measured-correction PLACE_OFFSET results:")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}: round={result['correction_round']} | "
            f"offset={result['place_offset']} | "
            f"score={result['score']:.5f} | "
            f"xy_error={result['xy_error']:.5f} | "
            f"z_error={result['z_error']:.5f} | "
            f"final_position={result['final_position']}"
        )

    return results[0], results


def verify_top_place_offsets(
    world,
    from_square,
    to_square,
    grasp_offset,
    search_results,
    top_k=VERIFY_TOP_K
):
    verified_results = []
    seen_offsets = set()

    print(f"\nVerifying top {top_k} offsets with full physics settings...")
    for result in search_results:
        offset_key = tuple(np.round(result["place_offset"], 9))
        if offset_key in seen_offsets:
            continue

        seen_offsets.add(offset_key)
        place_offset = result["place_offset"]
        print(f"Verifying PLACE_OFFSET: {place_offset}")

        verified_result = run_sim_move(
            world,
            from_square,
            to_square,
            grasp_offset,
            place_offset=place_offset,
            return_metrics=True,
            record_video=False,
            solver_iterations=SOLVER_ITERATIONS,
            solver_substeps=SOLVER_SUBSTEPS,
            post_move_settle_steps=POST_MOVE_SETTLE_STEPS
        )
        verified_result["score"] = score_place_result(verified_result)
        verified_result["cheap_score"] = result["score"]
        verified_results.append(verified_result)

        print(
            f"full_score={verified_result['score']:.5f} | "
            f"cheap_score={result['score']:.5f} | "
            f"xy_error={verified_result['xy_error']:.5f} | "
            f"z_error={verified_result['z_error']:.5f} | "
            f"pickup={verified_result['pickup_success']}"
        )

        if len(verified_results) >= top_k:
            break

    verified_results.sort(key=lambda item: item["score"])
    print("\nBest full-physics verification results:")
    for rank, result in enumerate(verified_results, start=1):
        print(
            f"{rank}: offset={result['place_offset']} | "
            f"score={result['score']:.5f} | "
            f"xy_error={result['xy_error']:.5f} | "
            f"z_error={result['z_error']:.5f} | "
            f"final_position={result['final_position']}"
        )

    return verified_results[0], verified_results


def run_place_offset_search_for_square(world, from_square, to_square, record_video=False):
    print(f"\n========== Searching {from_square} -> {to_square} ==========")

    search_start = time.perf_counter()
    if RUN_MEASURED_CORRECTION_SEARCH:
        best_search_result, place_results = find_best_place_offset_measured_correction(
            world,
            from_square,
            to_square,
            grasp_offset=GRASP_OFFSET,
            base_place_offset=GRASP_OFFSET,
            rounds=MEASURED_CORRECTION_ROUNDS
        )
    else:
        best_search_result, place_results = find_best_place_offset_coarse_to_fine(
            world,
            from_square,
            to_square,
            grasp_offset=GRASP_OFFSET,
            base_place_offset=GRASP_OFFSET
        )
    search_elapsed = time.perf_counter() - search_start

    print("\nBest search PLACE_OFFSET:", best_search_result["place_offset"])
    print(f"Fast sim search time: {search_elapsed:.2f}s")

    verify_start = time.perf_counter()
    best_result, verified_results = verify_top_place_offsets(
        world,
        from_square,
        to_square,
        GRASP_OFFSET,
        place_results,
        top_k=VERIFY_TOP_K
    )
    verify_elapsed = time.perf_counter() - verify_start

    print("\nBest verified PLACE_OFFSET:", best_result["place_offset"])
    print(f"Fast sim full verification time: {verify_elapsed:.2f}s")

    video_result = None
    if record_video:
        print("\nRerunning best PLACE_OFFSET with video...")
        video_start = time.perf_counter()
        video_result = run_sim_move(
            world,
            from_square,
            to_square,
            GRASP_OFFSET,
            place_offset=best_result["place_offset"],
            return_metrics=True,
            record_video=True,
            solver_iterations=SOLVER_ITERATIONS,
            solver_substeps=SOLVER_SUBSTEPS,
            post_move_settle_steps=POST_MOVE_SETTLE_STEPS
        )
        video_elapsed = time.perf_counter() - video_start

        video_output_dir = Path(f"./recordings/{runid}/{from_square}_to_{to_square}")
        print("Video rerun pickup success:", video_result["pickup_success"])
        print("Video rerun final position:", video_result["final_position"])
        print("Video rerun xy error:", video_result["xy_error"])
        print("Video rerun z error:", video_result["z_error"])
        print("Video rerun final tilt deg:", video_result["final_tilt_deg"])
        print("Video rerun final euler deg:", video_result["final_euler_deg"])
        print("Video output directory:", video_output_dir)
        print(f"Fast sim video rerun time: {video_elapsed:.2f}s")

    return best_result, place_results, verified_results, video_result


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


RUN_PLACE_OFFSET_SEARCH = True
RUN_MEASURED_CORRECTION_SEARCH = True
RUN_RANK1_PLACE_OFFSET_BATCH = True
# RECORD_VIDEO_RERUN = False
RECORD_VIDEO_RERUN = True

physics_client = p.connect(p.DIRECT)

p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())


# e1a1 Best verified PLACE_OFFSET: [-0.02246538  0.00114195 -0.00431543]
# e1b1 Best verified PLACE_OFFSET: [-0.02214579  0.00110381 -0.00460133]
# e1e1 Best verified PLACE_OFFSET: [-0.02208837  0.0008782  -0.00150453]
if RUN_PLACE_OFFSET_SEARCH:
    search_from_square = "e1"
    search_to_square = "e1"
    search_to_squares = [f"{file}1" for file in FILES] if RUN_RANK1_PLACE_OFFSET_BATCH else [search_to_square]

    setup_start = time.perf_counter()
    world = setup_sim_world(search_from_square)
    setup_elapsed = time.perf_counter() - setup_start
    print(f"Fast sim setup time: {setup_elapsed:.2f}s")

    try:
        best_place_offsets = []
        batch_start = time.perf_counter()

        for search_to_square in search_to_squares:
            best_result, place_results, verified_results, video_result = run_place_offset_search_for_square(
                world,
                search_from_square,
                search_to_square,
                record_video=RECORD_VIDEO_RERUN
            )
            best_place_offsets.append({
                "to_square": search_to_square,
                "place_offset": best_result["place_offset"].copy(),
                "score": best_result["score"],
                "xy_error": best_result["xy_error"],
                "z_error": best_result["z_error"],
                "pickup_success": best_result["pickup_success"],
                "final_tilt_deg": best_result["final_tilt_deg"],
            })

        batch_elapsed = time.perf_counter() - batch_start

        print("\n========== Rank 1 PLACE_OFFSET summary ==========")
        print("best_place_offsets = [")
        for item in best_place_offsets:
            print(
                "    "
                f"('{item['to_square']}', "
                f"np.array({repr(item['place_offset'].tolist())})), "
                f"# score={item['score']:.5f}, "
                f"xy={item['xy_error']:.5f}, "
                f"z={item['z_error']:.5f}, "
                f"pickup={item['pickup_success']}, "
                f"tilt={item['final_tilt_deg']:.2f}"
            )
        print("]")
        print(f"Rank 1 batch time: {batch_elapsed:.2f}s")
    finally:
        p.removeState(world["state_id"])
else:
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

    print(successlist)

p.disconnect()

#4 delta best so far with 51/64
