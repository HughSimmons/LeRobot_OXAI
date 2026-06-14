import pybullet as p
import pybullet_data
import imageio
import numpy as np
import sys
import time
from pathlib import Path
from chess_traj import pickupmove_traj, pickupmove_traj_with_metrics, chess_to_xy, gripper_angle_closed, gripper_angle_open, solve_xyz_for_traj, xyz_homeref_with_error
from testkinematics import kinematics
board_origin = (0.25, 0, 0)  # Must match the origin used in pybsim_chess.py
video_on = False
# video_on = True
runid = "multisim_place_lookup"
RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
MOVE_FROM_SQUARE = "e1"
MOVE_TO_SQUARE = "f2"

# Best PLACE_OFFSET: [-0.0165  0.0015 -0.005 ]
# FILES = "abcdefgh"
FILES = "".join(sorted({MOVE_FROM_SQUARE[0], MOVE_TO_SQUARE[0]}))

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


# GRASP_OFFSET = np.array([
#     -0.014,
#     -0.002,
#     -0.002
# ])

SOURCE_GRASP_OFFSET = np.array([-0.011, 0.002, -0.002])
SOURCE_GRASP_OFFSET_OVERRIDES = {
    ("f2", "e1"): np.array([-0.011, 0.002, -0.003]),
}
# Previous reversed y offset was too large for the low-resistance-piece retry:
# REVERSED_GRASP_OFFSET = np.array([-0.011, 0.007, -0.002]) #for e1->h1
REVERSED_GRASP_OFFSET = np.array([-0.011, 0.002, -0.002]) #works for e1f1
# REVERSED_GRASP_OFFSET = np.array([-0.01, 0.0085, -0.002])
# REVERSED_GRASP_OFFSET = np.array([-0.012, 0.00, -0.015])
GRASP_OFFSET = SOURCE_GRASP_OFFSET.copy()
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
PIECE_LIFTED_Z_THRESHOLD = 0.05
PIECE_DROPPED_Z_THRESHOLD = 0.035
SOLVER_ITERATIONS = 200
SOLVER_SUBSTEPS = 4
POST_MOVE_SETTLE_STEPS = 1000
SEARCH_SOLVER_ITERATIONS = 200
SEARCH_SOLVER_SUBSTEPS = 4
SEARCH_POST_MOVE_SETTLE_STEPS = 400
VERIFY_TOP_K = 1
MEASURED_CORRECTION_ROUNDS = 1
REVERSED_PLACEMENT_CORRECTION_ROUNDS = 8
REVERSED_PLACEMENT_LOWER_STEPS = 16
REVERSED_RELEASE_Z_OFFSET = 0.03
REVERSED_RELEASE_HOLD_WAYPOINTS = 8
REVERSED_POST_RELEASE_CLEARANCE_STEPS = 8
REVERSED_POST_RELEASE_CLEARANCE_Z = 0.035
REVERSED_RETREAT_HOME_STEPS = 16
DEFAULT_MOVE_STEPS_PER_WAYPOINT = 50
LONG_TRANSPORT_MOVE_STEPS_PER_WAYPOINT = 100
LONG_TRANSPORT_STEP_MOVES = {("e1", "g8")}
RUN_XY_CORRECTION_TEST = True
REVERSED_WAYPOINT_FK_ERROR_THRESHOLD = 0.025
REVERSED_MAX_INTERPOLATED_FK_GAP = 2
MAX_TRAJECTORY_FK_ERROR = 0.025
XY_ERROR_WEIGHT = 10.0
FINAL_TILT_TARGET_DEG = 15.0
XY_CORRECTION_TARGET_ERROR = 1e-4
SOFTEN_GRIPPER_PINCH = True
GRIPPER_CLOSE_FORCE = 500
GRIPPER_TRANSPORT_HOLD_FORCE = 50
SOFT_GRIPPER_CLOSE_FORCE = 120
SOFT_GRIPPER_TRANSPORT_HOLD_FORCE = 20
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


def find_release_move_index(movelist, closeidx):
    for idx in range(closeidx + 1, len(movelist)):
        gripper_now = movelist[idx][5]
        gripper_prev = movelist[idx - 1][5]
        if (
            np.isclose(gripper_now, gripper_angle_open)
            and np.isclose(gripper_prev, gripper_angle_closed)
        ):
            return idx

    return len(movelist) - 1

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

    # Previous piece dynamics pinned tilted cylinders on a single rim contact in
    # no-robot settling tests, preventing the piece from falling flat:
    # p.changeDynamics(
    #     piece_id,
    #     -1,
    #     lateralFriction=1.0,
    #     rollingFriction=0.02,
    #     spinningFriction=0.02,
    #     linearDamping=0.2,
    #     angularDamping=0.2
    # )
    # Lower rolling/spinning resistance and damping lets a tilted piece settle
    # upright without changing the collision shape. Keep lateral friction high
    # enough for gripper pickup.
    # Previous low-resistance test value:
    # lateralFriction=0.5
    p.changeDynamics(
        piece_id,
        -1,
        lateralFriction=1.0,
        rollingFriction=0.001,
        spinningFriction=0.001,
        linearDamping=0.02,
        angularDamping=0.02
    )
    return(piece_id)


def board_xy_bounds(board_origin=board_origin, square_size=0.04):
    board_x, board_y, _ = board_origin
    board_size = 8 * square_size
    return (
        board_x - board_size / 2,
        board_x + board_size / 2,
        board_y - board_size / 2,
        board_y + board_size / 2,
    )


def is_xy_on_board(position, board_origin=board_origin, square_size=0.04, margin=0.0):
    position = np.array(position)
    if position.shape[0] < 2 or not np.all(np.isfinite(position[:2])):
        return False

    x_min, x_max, y_min, y_max = board_xy_bounds(
        board_origin=board_origin,
        square_size=square_size,
    )
    return (
        x_min - margin <= position[0] <= x_max + margin
        and y_min - margin <= position[1] <= y_max + margin
    )


# ----------------------------------------
# Joint mapping
# ----------------------------------------

ARM_JOINTS = [0, 1, 2, 3, 4]
GRIPPER_IDX = 6

CONTROL_JOINTS = ARM_JOINTS + [GRIPPER_IDX]


def ensure_physics_connected():
    if not p.isConnected():
        p.connect(p.DIRECT)

    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    return p.isConnected()


def setup_sim_world(from_square):
    ensure_physics_connected()
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


def create_video_context(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "output_dir": output_dir,
        "writer": imageio.get_writer(
            str(output_dir / "so101_robot_moves.mp4"),
            fps=30,
            codec="libx264",
            quality=8
        ),
        "writer_topdown": imageio.get_writer(
            str(output_dir / "so101_robot_moves_topdown.mp4"),
            fps=30,
            codec="libx264",
            quality=8
        ),
    }


def append_video_frame(video_context):
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=WIDTH / HEIGHT,
        nearVal=0.01,
        farVal=100
    )
    camera_params = {
        "eye": [0.0, -0.6, 0.25],
        "target": [0.3, 0.0, 0.05],
        "up": [0, 0, 1],
    }
    top_down_camera_params = {
        "eye": [0.3, 0, 0.6],
        "target": [0.3, 0, 0],
        "up": [0, -1, 0],
    }

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_params["eye"],
        cameraTargetPosition=camera_params["target"],
        cameraUpVector=camera_params["up"]
    )
    w, h, rgba, _, _ = p.getCameraImage(
        WIDTH,
        HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix
    )
    img = np.array(rgba, dtype=np.uint8).reshape((h, w, 4))
    video_context["writer"].append_data(img[:, :, :3])

    view_matrix_topdown = p.computeViewMatrix(
        cameraEyePosition=top_down_camera_params["eye"],
        cameraTargetPosition=top_down_camera_params["target"],
        cameraUpVector=top_down_camera_params["up"]
    )
    w2, h2, rgba2, _, _ = p.getCameraImage(
        WIDTH,
        HEIGHT,
        viewMatrix=view_matrix_topdown,
        projectionMatrix=proj_matrix
    )
    img2 = np.array(rgba2, dtype=np.uint8).reshape((h2, w2, 4))
    video_context["writer_topdown"].append_data(img2[:, :, :3])


def close_video_context(video_context):
    video_context["writer"].close()
    video_context["writer_topdown"].close()


def run_sim_move(
    world,
    from_square,
    to_square,
    grasp_offset,
    place_offset=None,
    return_metrics=False,
    record_video=None,
    video_label=None,
    trajectory_override=None,
    solver_iterations=SOLVER_ITERATIONS,
    solver_substeps=SOLVER_SUBSTEPS,
    post_move_settle_steps=POST_MOVE_SETTLE_STEPS,
    move_steps_per_waypoint=DEFAULT_MOVE_STEPS_PER_WAYPOINT,
    restore_state=True,
    video_context=None,
):
    if restore_state and from_square != world["from_square"]:
        raise ValueError(f"world was initialized for {world['from_square']}, not {from_square}")

    if restore_state:
        p.restoreState(world["state_id"])
    p.setPhysicsEngineParameter(
        numSolverIterations=solver_iterations,
        numSubSteps=solver_substeps
    )

    sq1, sq2 = from_square, to_square
    simid = f"{sq1}_to_{sq2}"
    active_place_offset = PLACE_OFFSET if place_offset is None else place_offset
    video_enabled = video_context is not None or (video_on if record_video is None else record_video)
    robot_id = world["robot_id"]
    piece_ids = world["piece_ids"]

    if trajectory_override is None:
        movelist, closeidx, traj_metrics = pickupmove_traj_with_metrics(
            sq1,
            sq2,
            board_origin=board_origin,
            GRASP_OFFSET=grasp_offset,
            PLACE_OFFSET=active_place_offset
        )
    else:
        movelist = trajectory_override["movelist"]
        closeidx = trajectory_override["closeidx"]
        traj_metrics = trajectory_override["traj_metrics"]

    trajectory_fk_error = traj_metrics["max_fk_error"]
    release_target_z = traj_metrics["release_target_z"]
    if release_target_z is None:
        release_target_z = PIECE_DROPPED_Z_THRESHOLD

    # premature_drop_z_threshold = release_target_z + 0.003
    premature_drop_z_threshold = release_target_z + 0.01
    if (
        trajectory_override is not None
        and "premature_drop_z_threshold" in trajectory_override
    ):
        premature_drop_z_threshold = trajectory_override["premature_drop_z_threshold"]
    lifted_z_threshold = max(
        PIECE_LIFTED_Z_THRESHOLD,
        release_target_z + 0.02
    )

    if trajectory_fk_error > MAX_TRAJECTORY_FK_ERROR:
        expected_piece_pos = np.array(chess_to_xy(sq2, board_origin=board_origin))
        reject_reason = traj_metrics.get(
            "waypoint_filter_reject_reason",
            "trajectory_fk_error_too_large"
        )
        result = {
            "pickup_success": False,
            "from_square": sq1,
            "to_square": sq2,
            "place_offset": active_place_offset.copy(),
            "final_position": np.full(3, np.nan),
            "final_orientation": None,
            "expected_position": expected_piece_pos,
            "position_error": np.full(3, np.nan),
            "xy_error": np.inf,
            "z_error": np.inf,
            "final_tilt_deg": np.nan,
            "final_euler_deg": np.full(3, np.nan),
            "solver_iterations": solver_iterations,
            "solver_substeps": solver_substeps,
            "move_steps_per_waypoint": move_steps_per_waypoint,
            "post_move_settle_steps": post_move_settle_steps,
            "trajectory_fk_error": trajectory_fk_error,
            "trajectory_fk_error_events": traj_metrics["fk_error_events"],
            "trajectory_valid": False,
            "reject_reason": reject_reason,
            "release_move_idx": None,
            "release_target_z": release_target_z,
            "premature_drop_z_threshold": premature_drop_z_threshold,
            "lifted_z_threshold": lifted_z_threshold,
            "premature_drop": False,
            "premature_drop_step": None,
            "premature_drop_move_idx": None,
            "premature_drop_z": None,
            "min_pre_release_piece_z": None,
            "video_output_dir": None,
        }
        print(
            "Skipping simulation because trajectory FK error is too large: "
            f"{trajectory_fk_error:.5f}"
        )
        if return_metrics:
            return result
        return False

    moves = [
        (np.deg2rad(pos), move_steps_per_waypoint, f"Move to {pos}")
        for pos in movelist
    ]
    release_move_idx = find_release_move_index(movelist, closeidx)
    strong_hold_start_idx = None
    if trajectory_override is not None:
        strong_hold_start_idx = trajectory_override.get("strong_hold_start_idx")

    SIM_JOINT_MAP = [0, 1, 2, 3, 4, 6]
    if not restore_state and robot_id is not None and moves:
        current_joints = np.array([
            p.getJointState(robot_id, sim_idx)[0]
            for sim_idx in SIM_JOINT_MAP
        ])
        moves.insert(
            0,
            (
                current_joints,
                move_steps_per_waypoint,
                "Transition from current robot state",
            )
        )
        closeidx += 1
        release_move_idx += 1
        if strong_hold_start_idx is not None:
            strong_hold_start_idx += 1

    close_video_when_done = False
    output_dir = None
    if video_enabled:
        output_dir = RECORDINGS_DIR / runid / simid
        if video_label is not None:
            output_dir = output_dir / video_label
        if video_context is None:
            video_context = create_video_context(output_dir)
            close_video_when_done = True
        else:
            output_dir = video_context["output_dir"]

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

    pickup_success = False
    piece_was_lifted = False
    premature_drop = False
    premature_drop_step = None
    premature_drop_move_idx = None
    premature_drop_z = None
    min_pre_release_piece_z = np.inf
    soft_gripper_hold_position = None

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
                        force = GRIPPER_CLOSE_FORCE

                        if SOFTEN_GRIPPER_PINCH and move_idx == closeidx:
                            force = SOFT_GRIPPER_CLOSE_FORCE

                        if (
                            SOFTEN_GRIPPER_PINCH
                            and closeidx < move_idx < release_move_idx
                            and not (
                                strong_hold_start_idx is not None
                                and move_idx >= strong_hold_start_idx
                            )
                        ):
                            if soft_gripper_hold_position is None:
                                soft_gripper_hold_position = p.getJointState(robot_id, 6)[0]
                            target_joints[5] = soft_gripper_hold_position
                            force = SOFT_GRIPPER_TRANSPORT_HOLD_FORCE
                        elif (
                            strong_hold_start_idx is not None
                            and strong_hold_start_idx <= move_idx < release_move_idx
                        ):
                            force = GRIPPER_CLOSE_FORCE
                        elif move_idx > closeidx + 1:
                            force = GRIPPER_TRANSPORT_HOLD_FORCE

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
            piece_z = piece_pos[2]

            if move_idx < release_move_idx:
                min_pre_release_piece_z = min(
                    min_pre_release_piece_z,
                    piece_z
                )

            if piece_z > lifted_z_threshold and global_step > 50:  # Check if the piece has been lifted off the board (adjust threshold as needed)
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
                piece_was_lifted = True
                # break
                # return(pickup_success)
                # sys.exit()

            if (
                piece_was_lifted
                and move_idx < release_move_idx
                and piece_z < premature_drop_z_threshold
                and not (
                    strong_hold_start_idx is not None
                    and move_idx >= strong_hold_start_idx
                )
            ):
                premature_drop = True
                if premature_drop_step is None:
                    premature_drop_step = global_step
                    premature_drop_move_idx = move_idx
                    premature_drop_z = piece_z

            if video_enabled:
                if global_step % renderfreq == 0:
                    append_video_frame(video_context)

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
        if video_enabled and close_video_when_done:
            close_video_context(video_context)

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
    if np.isinf(min_pre_release_piece_z):
        min_pre_release_piece_z = None

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
            "move_steps_per_waypoint": move_steps_per_waypoint,
            "post_move_settle_steps": post_move_settle_steps,
            "trajectory_fk_error": trajectory_fk_error,
            "trajectory_fk_error_events": traj_metrics["fk_error_events"],
            "trajectory_valid": True,
            "release_move_idx": release_move_idx,
            "release_target_z": release_target_z,
            "premature_drop_z_threshold": premature_drop_z_threshold,
            "lifted_z_threshold": lifted_z_threshold,
            "premature_drop": premature_drop,
            "premature_drop_step": premature_drop_step,
            "premature_drop_move_idx": premature_drop_move_idx,
            "premature_drop_z": premature_drop_z,
            "min_pre_release_piece_z": min_pre_release_piece_z,
            "video_output_dir": str(output_dir) if video_enabled else None,
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
    if result.get("trajectory_fk_error", 0.0) > MAX_TRAJECTORY_FK_ERROR:
        result["reject_reason"] = (
            result.get("reject_reason")
            or "trajectory_fk_error_too_large"
        )
        return 1000.0

    if not result["pickup_success"]:
        result["reject_reason"] = "pickup_failed"
        return 1000.0

    result["reject_reason"] = None
    return result["xy_error"] + 0.25 * result["z_error"]


def score_grasp_result(result):
    if result.get("trajectory_fk_error", 0.0) > MAX_TRAJECTORY_FK_ERROR:
        result["reject_reason"] = "trajectory_fk_error_too_large"
        return 1000.0

    if not result["pickup_success"]:
        result["reject_reason"] = "pickup_failed"
        return 1000.0

    if result["premature_drop"]:
        result["reject_reason"] = "premature_drop"
        return 1000.0

    result["reject_reason"] = None
    return result["final_tilt_deg"] + XY_ERROR_WEIGHT * result["xy_error"]


def premature_drop_height_rank(result):
    drop_height = result.get("premature_drop_z")
    if drop_height is None:
        return np.inf

    return drop_height


def make_centered_samples(sample_count):
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")

    center = (sample_count - 1) / 2
    return np.arange(sample_count) - center


def find_best_place_offset(
    from_square=MOVE_FROM_SQUARE,
    to_square=MOVE_TO_SQUARE,
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
                    f"traj_fk_error={result['trajectory_fk_error']:.5f} | "
                    f"xy_error={result['xy_error']:.5f} | "
                    f"z_error={result['z_error']:.5f} | "
                    f"pickup={result['pickup_success']} | "
                    f"reject={result['reject_reason']}"
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
            f"traj_fk_error={result['trajectory_fk_error']:.5f} | "
            f"xy_error={result['xy_error']:.5f} | "
            f"z_error={result['z_error']:.5f} | "
            f"reject={result['reject_reason']} | "
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
            f"traj_fk_error={best_result['trajectory_fk_error']:.5f} | "
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

        if result["reject_reason"] == "trajectory_fk_error_too_large":
            round_elapsed = time.perf_counter() - round_start
            print(
                f"score={result['score']:.5f} | "
                f"traj_fk_error={result['trajectory_fk_error']:.5f} | "
                f"reject={result['reject_reason']} | "
                f"elapsed={round_elapsed:.2f}s"
            )
            break

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
            f"traj_fk_error={result['trajectory_fk_error']:.5f} | "
            f"xy_error={result['xy_error']:.5f} | "
            f"z_error={result['z_error']:.5f} | "
            f"pickup={result['pickup_success']} | "
            f"reject={result['reject_reason']} | "
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
            f"traj_fk_error={result['trajectory_fk_error']:.5f} | "
            f"xy_error={result['xy_error']:.5f} | "
            f"z_error={result['z_error']:.5f} | "
            f"reject={result['reject_reason']} | "
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
            f"traj_fk_error={verified_result['trajectory_fk_error']:.5f} | "
            f"xy_error={verified_result['xy_error']:.5f} | "
            f"z_error={verified_result['z_error']:.5f} | "
            f"pickup={verified_result['pickup_success']} | "
            f"reject={verified_result['reject_reason']}"
        )

        if len(verified_results) >= top_k:
            break

    verified_results.sort(key=lambda item: item["score"])
    print("\nBest full-physics verification results:")
    for rank, result in enumerate(verified_results, start=1):
        print(
            f"{rank}: offset={result['place_offset']} | "
            f"score={result['score']:.5f} | "
            f"traj_fk_error={result['trajectory_fk_error']:.5f} | "
            f"xy_error={result['xy_error']:.5f} | "
            f"z_error={result['z_error']:.5f} | "
            f"reject={result['reject_reason']} | "
            f"final_position={result['final_position']}"
        )

    return verified_results[0], verified_results


def run_place_offset_search_for_square(
    world,
    from_square,
    to_square,
    grasp_offset=GRASP_OFFSET,
    base_place_offset=None,
    record_video=False,
    use_measured_correction=None
):
    print(f"\n========== Searching {from_square} -> {to_square} ==========")

    if base_place_offset is None:
        base_place_offset = grasp_offset.copy()

    if use_measured_correction is None:
        use_measured_correction = RUN_MEASURED_CORRECTION_SEARCH

    search_start = time.perf_counter()
    if use_measured_correction:
        best_search_result, place_results = find_best_place_offset_measured_correction(
            world,
            from_square,
            to_square,
            grasp_offset=grasp_offset,
            base_place_offset=base_place_offset,
            rounds=MEASURED_CORRECTION_ROUNDS
        )
    else:
        best_search_result, place_results = find_best_place_offset_coarse_to_fine(
            world,
            from_square,
            to_square,
            grasp_offset=grasp_offset,
            base_place_offset=base_place_offset
        )
    search_elapsed = time.perf_counter() - search_start

    print("\nBest search PLACE_OFFSET:", best_search_result["place_offset"])
    print(f"Fast sim search time: {search_elapsed:.2f}s")

    verify_start = time.perf_counter()
    best_result, verified_results = verify_top_place_offsets(
        world,
        from_square,
        to_square,
        grasp_offset,
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
            grasp_offset,
            place_offset=best_result["place_offset"],
            return_metrics=True,
            record_video=True,
            solver_iterations=SOLVER_ITERATIONS,
            solver_substeps=SOLVER_SUBSTEPS,
            post_move_settle_steps=POST_MOVE_SETTLE_STEPS
        )
        video_elapsed = time.perf_counter() - video_start

        video_output_dir = RECORDINGS_DIR / runid / f"{from_square}_to_{to_square}"
        print("Video rerun pickup success:", video_result["pickup_success"])
        print("Video rerun trajectory FK error:", video_result["trajectory_fk_error"])
        print("Video rerun final position:", video_result["final_position"])
        print("Video rerun xy error:", video_result["xy_error"])
        print("Video rerun z error:", video_result["z_error"])
        print("Video rerun final tilt deg:", video_result["final_tilt_deg"])
        print("Video rerun final euler deg:", video_result["final_euler_deg"])
        print("Video output directory:", video_output_dir)
        print(f"Fast sim video rerun time: {video_elapsed:.2f}s")

    return best_result, place_results, verified_results, video_result


def find_best_grasp_offset_for_move(
    world,
    from_square=MOVE_FROM_SQUARE,
    to_square=MOVE_TO_SQUARE,
    base_grasp_offset=GRASP_OFFSET,
    place_offset=PLACE_OFFSET,
    delta=0.001,
    nsamples=(1, 1, 1)
):
    if len(nsamples) != 3:
        raise ValueError("nsamples must contain three values: x, y, z")

    def centered_samples(sample_count):
        if sample_count < 1 or sample_count % 2 == 0:
            raise ValueError("each nsamples value must be a positive odd integer")

        sample_radius = sample_count // 2
        return list(range(-sample_radius, sample_radius + 1))

    dx_samples, dy_samples, dz_samples = [
        centered_samples(sample_count)
        for sample_count in nsamples
    ]
    results = []

    print(f"\n========== Searching GRASP_OFFSET {from_square} -> {to_square} ==========")
    for dx in dx_samples:
        for dy in dy_samples:
            for dz in dz_samples:
                grasp_offset = base_grasp_offset + delta * np.array([dx, dy, dz])
                print(f"Testing GRASP_OFFSET: {grasp_offset}")
                video_label = f"grasp_dx{dx}_dy{dy}_dz{dz}"

                result = run_sim_move(
                    world,
                    from_square,
                    to_square,
                    grasp_offset,
                    place_offset=place_offset,
                    return_metrics=True,
                    record_video=False,
                    video_label=video_label,
                    solver_iterations=SEARCH_SOLVER_ITERATIONS,
                    solver_substeps=SEARCH_SOLVER_SUBSTEPS,
                    post_move_settle_steps=SEARCH_POST_MOVE_SETTLE_STEPS
                )
                result["grasp_offset"] = grasp_offset.copy()
                result["score"] = score_grasp_result(result)
                result["drop_height_rank"] = premature_drop_height_rank(result)
                results.append(result)

                print(
                    f"score={result['score']:.5f} | "
                    f"drop_height_rank={result['drop_height_rank']:.5f} | "
                    f"traj_fk_error={result['trajectory_fk_error']:.5f} | "
                    f"pickup={result['pickup_success']} | "
                    f"premature_drop={result['premature_drop']} | "
                    f"drop_height={result['premature_drop_z']} | "
                    f"min_pre_release_z={result['min_pre_release_piece_z']} | "
                    f"drop_z={result['premature_drop_z_threshold']:.5f} | "
                    f"xy_error={result['xy_error']:.5f} | "
                    f"z_error={result['z_error']:.5f} | "
                    f"tilt={result['final_tilt_deg']:.2f} | "
                    f"reject={result['reject_reason']} | "
                    f"video_dir={result['video_output_dir']}"
                )

    results.sort(key=lambda item: (item["drop_height_rank"], item["score"]))

    valid_results = [
        result for result in results
        if result["reject_reason"] is None
    ]

    print("\nValid GRASP_OFFSET results:")
    if valid_results:
        for rank, result in enumerate(valid_results, start=1):
            print(
                f"{rank}: grasp_offset={result['grasp_offset']} | "
                f"score={result['score']:.5f} | "
                f"drop_height_rank={result['drop_height_rank']:.5f} | "
                f"xy_error={result['xy_error']:.5f} | "
                f"tilt={result['final_tilt_deg']:.2f}"
            )
    else:
        print("No valid GRASP_OFFSET results found.")

    print("\nDrop-height-ranked GRASP_OFFSET results:")
    for rank, result in enumerate(results[:10], start=1):
        print(
            f"{rank}: grasp_offset={result['grasp_offset']} | "
            f"score={result['score']:.5f} | "
            f"drop_height_rank={result['drop_height_rank']:.5f} | "
            f"traj_fk_error={result['trajectory_fk_error']:.5f} | "
            f"pickup={result['pickup_success']} | "
            f"premature_drop={result['premature_drop']} | "
            f"drop_height={result['premature_drop_z']} | "
            f"min_pre_release_z={result['min_pre_release_piece_z']} | "
            f"drop_z={result['premature_drop_z_threshold']:.5f} | "
            f"xy_error={result['xy_error']:.5f} | "
            f"z_error={result['z_error']:.5f} | "
            f"tilt={result['final_tilt_deg']:.2f} | "
            f"reject={result['reject_reason']} | "
            f"video_dir={result['video_output_dir']} | "
            f"final_position={result['final_position']}"
        )

    top_video_results = []
    print("\nRerunning top 3 drop-height-ranked GRASP_OFFSET candidates with video...")
    for rank, result in enumerate(results[:3], start=1):
        video_label = f"rank{rank}_drop_height"
        video_result = run_sim_move(
            world,
            from_square,
            to_square,
            result["grasp_offset"],
            place_offset=place_offset,
            return_metrics=True,
            record_video=True,
            video_label=video_label,
            solver_iterations=SOLVER_ITERATIONS,
            solver_substeps=SOLVER_SUBSTEPS,
            post_move_settle_steps=POST_MOVE_SETTLE_STEPS
        )
        video_result["grasp_offset"] = result["grasp_offset"].copy()
        video_result["score"] = score_grasp_result(video_result)
        video_result["drop_height_rank"] = premature_drop_height_rank(video_result)
        top_video_results.append(video_result)

        print(
            f"{rank}: grasp_offset={video_result['grasp_offset']} | "
            f"drop_height_rank={video_result['drop_height_rank']:.5f} | "
            f"drop_height={video_result['premature_drop_z']} | "
            f"xy_error={video_result['xy_error']:.5f} | "
            f"tilt={video_result['final_tilt_deg']:.2f} | "
            f"reject={video_result['reject_reason']} | "
            f"video_dir={video_result['video_output_dir']}"
        )

    best_return_result = valid_results[0] if valid_results else results[0]
    return best_return_result, results


def run_grasp_offset_search(record_video=True):
    search_from_square = MOVE_FROM_SQUARE
    search_to_square = MOVE_TO_SQUARE

    setup_start = time.perf_counter()
    world = setup_sim_world(search_from_square)
    setup_elapsed = time.perf_counter() - setup_start
    print(f"Fast sim setup time: {setup_elapsed:.2f}s")

    try:
        search_start = time.perf_counter()
        best_result, grasp_results = find_best_grasp_offset_for_move(
            world,
            from_square=search_from_square,
            to_square=search_to_square,
            base_grasp_offset=GRASP_OFFSET,
            place_offset=PLACE_OFFSET
        )
        search_elapsed = time.perf_counter() - search_start

        print("\nBest GRASP_OFFSET:", best_result["grasp_offset"])
        print(f"Fast sim grasp search time: {search_elapsed:.2f}s")

        video_result = None
        if record_video:
            print("\nRerunning best GRASP_OFFSET with video...")
            video_start = time.perf_counter()
            video_result = run_sim_move(
                world,
                search_from_square,
                search_to_square,
                best_result["grasp_offset"],
                place_offset=PLACE_OFFSET,
                return_metrics=True,
                record_video=True,
                solver_iterations=SOLVER_ITERATIONS,
                solver_substeps=SOLVER_SUBSTEPS,
                post_move_settle_steps=POST_MOVE_SETTLE_STEPS
            )
            video_elapsed = time.perf_counter() - video_start

            video_output_dir = RECORDINGS_DIR / runid / f"{search_from_square}_to_{search_to_square}"
            print("Video rerun pickup success:", video_result["pickup_success"])
            print("Video rerun trajectory FK error:", video_result["trajectory_fk_error"])
            print("Video rerun final position:", video_result["final_position"])
            print("Video rerun xy error:", video_result["xy_error"])
            print("Video rerun z error:", video_result["z_error"])
            print("Video rerun final tilt deg:", video_result["final_tilt_deg"])
            print("Video rerun final euler deg:", video_result["final_euler_deg"])
            print("Video output directory:", video_output_dir)
            print(f"Fast sim video rerun time: {video_elapsed:.2f}s")

        return best_result, grasp_results, video_result
    finally:
        p.removeState(world["state_id"])


def run_grasp_only_until_held(
    from_square=MOVE_FROM_SQUARE,
    to_square=MOVE_TO_SQUARE,
    record_video=True
):
    setup_start = time.perf_counter()
    world = setup_sim_world(from_square)
    setup_elapsed = time.perf_counter() - setup_start
    print(f"Fast sim setup time: {setup_elapsed:.2f}s")

    try:
        search_start = time.perf_counter()
        best_result, grasp_results = find_best_grasp_offset_for_move(
            world,
            from_square=from_square,
            to_square=to_square,
            base_grasp_offset=GRASP_OFFSET,
            place_offset=PLACE_OFFSET
        )
        search_elapsed = time.perf_counter() - search_start

        valid_results = [
            result for result in grasp_results
            if result["reject_reason"] is None
        ]

        if not valid_results:
            print("\nNo valid GRASP_OFFSET found.")
            print("Least-bad GRASP_OFFSET candidates:")
            for rank, result in enumerate(grasp_results[:10], start=1):
                print(
                    f"{rank}: grasp_offset={result['grasp_offset']} | "
                    f"score={result['score']:.5f} | "
                    f"pickup={result['pickup_success']} | "
                    f"premature_drop={result['premature_drop']} | "
                    f"tilt={result['final_tilt_deg']:.2f} | "
                    f"reject={result['reject_reason']}"
                )
            print(f"Fast sim grasp-only search time: {search_elapsed:.2f}s")
            return None, grasp_results, None

        print("\nBest GRASP_OFFSET:", best_result["grasp_offset"])
        print("Best grasp tilt deg:", best_result["final_tilt_deg"])
        print("Best grasp pickup success:", best_result["pickup_success"])
        print("Best grasp premature drop:", best_result["premature_drop"])
        print("Best grasp release target z:", best_result["release_target_z"])
        print("Best grasp premature drop z threshold:", best_result["premature_drop_z_threshold"])
        print("Best grasp trajectory FK error:", best_result["trajectory_fk_error"])
        print("Best grasp xy error:", best_result["xy_error"])
        print("Best grasp z error:", best_result["z_error"])
        print(f"Fast sim grasp-only search time: {search_elapsed:.2f}s")

        video_result = None
        if record_video:
            print("\nRerunning best GRASP_OFFSET with video...")
            video_start = time.perf_counter()
            video_result = run_sim_move(
                world,
                from_square,
                to_square,
                best_result["grasp_offset"],
                place_offset=PLACE_OFFSET,
                return_metrics=True,
                record_video=True,
                solver_iterations=SOLVER_ITERATIONS,
                solver_substeps=SOLVER_SUBSTEPS,
                post_move_settle_steps=POST_MOVE_SETTLE_STEPS
            )
            video_elapsed = time.perf_counter() - video_start

            video_output_dir = RECORDINGS_DIR / runid / f"{from_square}_to_{to_square}"
            print("Video rerun pickup success:", video_result["pickup_success"])
            print("Video rerun premature drop:", video_result["premature_drop"])
            print("Video rerun final tilt deg:", video_result["final_tilt_deg"])
            print("Video rerun final euler deg:", video_result["final_euler_deg"])
            print("Video output directory:", video_output_dir)
            print(f"Fast sim grasp-only video rerun time: {video_elapsed:.2f}s")

        return best_result, grasp_results, video_result
    finally:
        p.removeState(world["state_id"])


def run_single_video_inspection(
    from_square=MOVE_FROM_SQUARE,
    to_square=MOVE_TO_SQUARE
):
    setup_start = time.perf_counter()
    world = setup_sim_world(from_square)
    setup_elapsed = time.perf_counter() - setup_start
    print(f"Fast sim setup time: {setup_elapsed:.2f}s")

    try:
        print(f"\n========== Single video inspection {from_square} -> {to_square} ==========")
        print("Using GRASP_OFFSET:", GRASP_OFFSET)
        print("Using PLACE_OFFSET:", PLACE_OFFSET)

        run_start = time.perf_counter()
        result = run_sim_move(
            world,
            from_square,
            to_square,
            GRASP_OFFSET,
            place_offset=PLACE_OFFSET,
            return_metrics=True,
            record_video=True,
            solver_iterations=SOLVER_ITERATIONS,
            solver_substeps=SOLVER_SUBSTEPS,
            post_move_settle_steps=POST_MOVE_SETTLE_STEPS
        )
        run_elapsed = time.perf_counter() - run_start

        video_output_dir = RECORDINGS_DIR / runid / f"{from_square}_to_{to_square}"
        print("Single run pickup success:", result["pickup_success"])
        print("Single run premature drop:", result["premature_drop"])
        print("Single run trajectory FK error:", result["trajectory_fk_error"])
        print("Single run final position:", result["final_position"])
        print("Single run xy error:", result["xy_error"])
        print("Single run z error:", result["z_error"])
        print("Single run final tilt deg:", result["final_tilt_deg"])
        print("Single run final euler deg:", result["final_euler_deg"])
        print("Video output directory:", video_output_dir)
        print(f"Fast sim single video run time: {run_elapsed:.2f}s")

        return result
    finally:
        p.removeState(world["state_id"])


def run_grasp_then_place_search(
    from_square=MOVE_FROM_SQUARE,
    to_square=MOVE_TO_SQUARE,
    record_video=True
):
    setup_start = time.perf_counter()
    world = setup_sim_world(from_square)
    setup_elapsed = time.perf_counter() - setup_start
    print(f"Fast sim setup time: {setup_elapsed:.2f}s")

    try:
        grasp_start = time.perf_counter()
        best_grasp_result, grasp_results = find_best_grasp_offset_for_move(
            world,
            from_square=from_square,
            to_square=to_square,
            base_grasp_offset=GRASP_OFFSET,
            place_offset=PLACE_OFFSET
        )
        grasp_elapsed = time.perf_counter() - grasp_start
        best_grasp_offset = best_grasp_result["grasp_offset"].copy()

        print("\nBest GRASP_OFFSET:", best_grasp_offset)
        print("Best grasp tilt deg:", best_grasp_result["final_tilt_deg"])
        print("Best grasp pickup success:", best_grasp_result["pickup_success"])
        print("Best grasp premature drop:", best_grasp_result["premature_drop"])
        print("Best grasp release target z:", best_grasp_result["release_target_z"])
        print("Best grasp premature drop z threshold:", best_grasp_result["premature_drop_z_threshold"])
        print("Best grasp trajectory FK error:", best_grasp_result["trajectory_fk_error"])
        print("Best grasp xy error:", best_grasp_result["xy_error"])
        print("Best grasp z error:", best_grasp_result["z_error"])
        print(f"Fast sim grasp search time: {grasp_elapsed:.2f}s")

        place_start = time.perf_counter()
        best_place_result, place_results, verified_results, video_result = run_place_offset_search_for_square(
            world,
            from_square,
            to_square,
            grasp_offset=best_grasp_offset,
            base_place_offset=best_grasp_offset,
            record_video=record_video,
            use_measured_correction=True
        )
        place_elapsed = time.perf_counter() - place_start

        print("\n========== Grasp then PLACE_OFFSET summary ==========")
        print("Best GRASP_OFFSET:", best_grasp_offset)
        print("Best grasp tilt deg:", best_grasp_result["final_tilt_deg"])
        print("Best grasp premature drop:", best_grasp_result["premature_drop"])
        print("Best grasp release target z:", best_grasp_result["release_target_z"])
        print("Best grasp premature drop z threshold:", best_grasp_result["premature_drop_z_threshold"])
        print("Best PLACE_OFFSET:", best_place_result["place_offset"])
        print("Best placement score:", best_place_result["score"])
        print("Best placement final tilt deg:", best_place_result["final_tilt_deg"])
        print("Best placement pickup success:", best_place_result["pickup_success"])
        print("Best placement trajectory FK error:", best_place_result["trajectory_fk_error"])
        print("Best placement xy error:", best_place_result["xy_error"])
        print("Best placement z error:", best_place_result["z_error"])
        print(f"Fast sim placement search+verify+video time: {place_elapsed:.2f}s")

        return best_grasp_result, grasp_results, best_place_result, place_results, verified_results, video_result
    finally:
        p.removeState(world["state_id"])


def make_reversed_traj_metrics(source_traj_metrics):
    return {
        "max_fk_error": float(source_traj_metrics.get("max_fk_error", 0.0)),
        "raw_generated_max_fk_error": 0.0,
        "event_threshold": REVERSED_WAYPOINT_FK_ERROR_THRESHOLD,
        "fk_error_events": list(source_traj_metrics.get("fk_error_events", [])),
        "waypoint_filter_repairs": [],
        "waypoint_filter_reject_reason": None,
        "trajectory_valid": True,
    }


def mark_reversed_trajectory_rejected(traj_metrics, reason, segment_name=None):
    traj_metrics["trajectory_valid"] = False
    traj_metrics["waypoint_filter_reject_reason"] = reason
    if segment_name is not None:
        traj_metrics["waypoint_filter_reject_segment"] = segment_name
    traj_metrics["max_fk_error"] = max(
        traj_metrics.get("max_fk_error", 0.0),
        MAX_TRAJECTORY_FK_ERROR + 1.0
    )


def solve_generated_waypoint(
    target_xyz,
    refjnts,
    frame_offset,
    downflag,
    gripper_angle,
    stage,
    local_idx,
    traj_metrics
):
    solve_error = None
    try:
        joints, fk_error = xyz_homeref_with_error(
            target_xyz,
            refjnts,
            frame_offset,
            downflag,
            traj_metrics=None,
            stage=stage,
        )
    except Exception as exc:
        solve_error = repr(exc)
        joints = np.array(refjnts).copy()
        fk_error = np.inf

    joints[5] = gripper_angle

    fk_error = float(fk_error)
    valid = fk_error <= REVERSED_WAYPOINT_FK_ERROR_THRESHOLD
    traj_metrics["raw_generated_max_fk_error"] = max(
        traj_metrics["raw_generated_max_fk_error"],
        fk_error
    )
    if valid:
        traj_metrics["max_fk_error"] = max(
            traj_metrics["max_fk_error"],
            fk_error
        )
    else:
        event = {
            "stage": stage,
            "idx": int(local_idx),
            "target_xyz": np.array(target_xyz).copy(),
            "fk_error": fk_error,
            "valid": False,
        }
        if solve_error is not None:
            event["solve_error"] = solve_error
        traj_metrics["fk_error_events"].append(event)

    return {
        "stage": stage,
        "idx": int(local_idx),
        "target_xyz": np.array(target_xyz).copy(),
        "joints": joints.copy(),
        "gripper_angle": gripper_angle,
        "fk_error": fk_error,
        "valid": valid,
        "solve_error": solve_error,
    }


def filter_generated_waypoints(records, segment_name, traj_metrics):
    if not records:
        return [], True

    filtered = [
        record["joints"].copy()
        for record in records
    ]
    if all(record["valid"] for record in records):
        return filtered, True

    cursor = 0
    while cursor < len(records):
        if records[cursor]["valid"]:
            cursor += 1
            continue

        gap_start = cursor
        while cursor < len(records) and not records[cursor]["valid"]:
            cursor += 1
        gap_end = cursor - 1
        gap_len = gap_end - gap_start + 1

        if gap_len > REVERSED_MAX_INTERPOLATED_FK_GAP:
            mark_reversed_trajectory_rejected(
                traj_metrics,
                "trajectory_waypoint_fk_gap",
                segment_name
            )
            print(
                f"Rejecting {segment_name}: FK gap length {gap_len} "
                f"exceeds {REVERSED_MAX_INTERPOLATED_FK_GAP}"
            )
            return filtered, False

        if gap_start == 0 or gap_end == len(records) - 1:
            mark_reversed_trajectory_rejected(
                traj_metrics,
                "trajectory_waypoint_fk_gap",
                segment_name
            )
            print(
                f"Rejecting {segment_name}: FK gap touches generated segment endpoint "
                f"({gap_start}-{gap_end})"
            )
            return filtered, False

        before = filtered[gap_start - 1].copy()
        after = filtered[gap_end + 1].copy()
        for idx in range(gap_start, gap_end + 1):
            alpha = (idx - gap_start + 1) / (gap_len + 1)
            repaired = (1 - alpha) * before + alpha * after
            repaired[5] = records[idx]["gripper_angle"]
            filtered[idx] = repaired.copy()

        traj_metrics["waypoint_filter_repairs"].append({
            "segment": segment_name,
            "gap_start": int(records[gap_start]["idx"]),
            "gap_end": int(records[gap_end]["idx"]),
            "gap_len": int(gap_len),
        })
        print(
            f"Repaired {segment_name}: interpolated FK gap "
            f"{records[gap_start]['idx']}-{records[gap_end]['idx']}"
        )

    return filtered, True


def build_reversed_pick_place_trajectory(
    source_movelist,
    source_closeidx,
    source_traj_metrics,
    reversed_from_square,
    reversed_to_square,
    grasp_offset=GRASP_OFFSET,
    place_offset=PLACE_OFFSET
):
    source_release_idx = find_release_move_index(source_movelist, source_closeidx)
    reversed_movelist = [
        np.array(joints).copy()
        for joints in source_movelist[::-1]
    ]

    reversed_closeidx = len(source_movelist) - 1 - source_release_idx
    reversed_release_idx = len(source_movelist) - 1 - source_closeidx
    if reversed_closeidx >= reversed_release_idx:
        raise ValueError(
            "cannot reverse trajectory because close/release indices are out of order: "
            f"close={reversed_closeidx}, release={reversed_release_idx}"
        )
    reversed_traj_metrics = make_reversed_traj_metrics(source_traj_metrics)

    for idx, joints in enumerate(reversed_movelist):
        if idx < reversed_closeidx:
            joints[5] = gripper_angle_open
        elif idx < reversed_release_idx:
            joints[5] = gripper_angle_closed
        else:
            joints[5] = gripper_angle_open

    pickup_target_xyz = chess_to_xy(
        reversed_from_square,
        board_origin=board_origin
    )
    pickup_downflag = reversed_from_square[0] not in ["f", "g", "h"]
    pickup_solve_seed = reversed_movelist[reversed_closeidx].copy()
    pickup_contact_record = solve_generated_waypoint(
        pickup_target_xyz,
        pickup_solve_seed,
        grasp_offset,
        pickup_downflag,
        gripper_angle_closed,
        "reversed_pickup_contact",
        reversed_closeidx,
        reversed_traj_metrics
    )
    pickup_contact_joints = pickup_contact_record["joints"].copy()

    pickup_contact_start = max(0, reversed_closeidx - 4)
    pickup_records = []
    for idx in range(pickup_contact_start, reversed_closeidx + 1):
        corrected_joints = pickup_contact_joints.copy()
        corrected_joints[5] = (
            gripper_angle_closed
            if idx == reversed_closeidx
            else gripper_angle_open
        )
        pickup_records.append({
            **pickup_contact_record,
            "idx": int(idx),
            "joints": corrected_joints.copy(),
            "gripper_angle": corrected_joints[5],
        })
    filtered_pickup_joints, pickup_valid = filter_generated_waypoints(
        pickup_records,
        "reversed_pickup",
        reversed_traj_metrics
    )
    for idx, corrected_joints in zip(
        range(pickup_contact_start, reversed_closeidx + 1),
        filtered_pickup_joints
    ):
        reversed_movelist[idx] = corrected_joints.copy()

    release_target_xyz = chess_to_xy(
        reversed_to_square,
        board_origin=board_origin
    ) + np.array([0, 0, REVERSED_RELEASE_Z_OFFSET])
    release_downflag = reversed_to_square[0] not in ["f", "g", "h"]

    placement_lower_start_idx = max(
        reversed_closeidx + 1,
        reversed_release_idx - 8
    )
    placement_lower_seed = reversed_movelist[placement_lower_start_idx - 1].copy()
    seed_pose = kinematics.forward_kinematics(placement_lower_seed)
    seed_place_xyz = seed_pose[:3, 3] + seed_pose[:3, :3] @ place_offset

    placement_lower_joints = []
    placement_lower_records = []
    current = placement_lower_seed.copy()
    for local_idx, alpha in enumerate(
        np.linspace(0, 1, REVERSED_PLACEMENT_LOWER_STEPS + 1)[1:],
        start=1
    ):
        intermediate_xyz = (
            (1 - alpha) * seed_place_xyz
            + alpha * release_target_xyz
        )
        intermediate_record = solve_generated_waypoint(
            intermediate_xyz,
            current,
            place_offset,
            release_downflag,
            gripper_angle_closed,
            "reversed_placement_lower",
            local_idx,
            reversed_traj_metrics
        )
        intermediate_joints = intermediate_record["joints"].copy()
        intermediate_joints[5] = gripper_angle_closed
        placement_lower_joints.append(intermediate_joints.copy())
        placement_lower_records.append(intermediate_record)
        current = intermediate_joints.copy()

    placement_lower_joints, lower_valid = filter_generated_waypoints(
        placement_lower_records,
        "reversed_placement_lower",
        reversed_traj_metrics
    )
    if placement_lower_joints:
        current = placement_lower_joints[-1].copy()

    release_open_joints = current.copy()
    release_open_joints[5] = gripper_angle_open
    placement_lower_joints.append(release_open_joints)
    for _ in range(REVERSED_RELEASE_HOLD_WAYPOINTS):
        placement_lower_joints.append(release_open_joints.copy())

    clearance_target_xyz = release_target_xyz + np.array([
        0,
        0,
        REVERSED_POST_RELEASE_CLEARANCE_Z
    ])
    clearance_records = []
    clearance_joints = []
    for local_idx, alpha in enumerate(
        np.linspace(0, 1, REVERSED_POST_RELEASE_CLEARANCE_STEPS + 1)[1:],
        start=1
    ):
        intermediate_xyz = (
            (1 - alpha) * release_target_xyz
            + alpha * clearance_target_xyz
        )
        intermediate_record = solve_generated_waypoint(
            intermediate_xyz,
            current,
            place_offset,
            release_downflag,
            gripper_angle_open,
            "reversed_post_release_clearance",
            local_idx,
            reversed_traj_metrics
        )
        intermediate_joints = intermediate_record["joints"].copy()
        intermediate_joints[5] = gripper_angle_open
        clearance_joints.append(intermediate_joints.copy())
        clearance_records.append(intermediate_record)
        current = intermediate_joints.copy()

    clearance_joints, clearance_valid = filter_generated_waypoints(
        clearance_records,
        "reversed_post_release_clearance",
        reversed_traj_metrics
    )
    placement_lower_joints.extend([
        joints.copy()
        for joints in clearance_joints
    ])
    if clearance_joints:
        current = clearance_joints[-1].copy()

    retreat_target = home.copy()
    retreat_target[5] = gripper_angle_open
    retreat_start = placement_lower_joints[-1].copy()
    for alpha in np.linspace(0, 1, REVERSED_RETREAT_HOME_STEPS + 1)[1:]:
        retreat_joints = (
            (1 - alpha) * retreat_start
            + alpha * retreat_target
        )
        retreat_joints[5] = gripper_angle_open
        placement_lower_joints.append(retreat_joints.copy())
    placement_lower_joints.append(retreat_target.copy())

    reversed_movelist = (
        reversed_movelist[:placement_lower_start_idx]
        + placement_lower_joints
    )
    reversed_release_idx = (
        placement_lower_start_idx
        + REVERSED_PLACEMENT_LOWER_STEPS
    )

    reversed_traj_metrics.update({
        "release_target_xyz": release_target_xyz.copy(),
        "release_target_z": float(release_target_xyz[2]),
        "grasp_offset": grasp_offset.copy(),
        "source_release_idx": source_release_idx,
        "reversed_release_idx": reversed_release_idx,
    })

    return {
        "movelist": reversed_movelist,
        "closeidx": reversed_closeidx,
        "traj_metrics": reversed_traj_metrics,
        "release_idx": reversed_release_idx,
        "place_offset": place_offset.copy(),
        "grasp_offset": grasp_offset.copy(),
        "premature_drop_z_threshold": PIECE_DROPPED_Z_THRESHOLD,
        "strong_hold_start_idx": placement_lower_start_idx,
        "source_grasp_idx": source_closeidx,
        "source_place_idx": source_release_idx,
        "source_place_as_reversed_grasp_idx": reversed_closeidx,
        "source_grasp_as_reversed_place_idx": reversed_release_idx,
    }


def apply_reversed_place_correction(result, trajectory_override, correct_z=False):
    world_correction = result["position_error"].copy()
    if not correct_z:
        world_correction[2] = 0.0

    release_joints = trajectory_override["movelist"][trajectory_override["release_idx"]]
    release_rot = kinematics.forward_kinematics(release_joints)[:3, :3]
    gripper_frame_correction = release_rot.T @ world_correction
    corrected_place_offset = trajectory_override["place_offset"] + gripper_frame_correction

    return corrected_place_offset, world_correction, gripper_frame_correction


def print_result_summary(label, result, include_video=False):
    metric_keys = (
        "score",
        "reject_reason",
        "pickup_success",
        "premature_drop",
        "trajectory_fk_error",
        "move_steps_per_waypoint",
        "xy_error",
        "z_error",
        "final_tilt_deg",
    )
    summary = ", ".join(
        f"{key}={result.get(key)}"
        for key in metric_keys
        if key in result
    )
    print(f"{label}: {summary}")
    if include_video and result.get("video_output_dir"):
        print("video_output_dir:", result["video_output_dir"])


def run_scored_simulation(
    from_square,
    to_square,
    grasp_offset,
    place_offset,
    trajectory_override,
    record_video=False,
    video_label=None,
    move_steps_per_waypoint=DEFAULT_MOVE_STEPS_PER_WAYPOINT
):
    world = setup_sim_world(from_square)
    try:
        result = run_sim_move(
            world,
            from_square,
            to_square,
            grasp_offset,
            place_offset=place_offset,
            return_metrics=True,
            record_video=record_video,
            video_label=video_label,
            trajectory_override=trajectory_override,
            move_steps_per_waypoint=move_steps_per_waypoint
        )
        result["score"] = score_place_result(result)
        return result
    finally:
        p.removeState(world["state_id"])


def move_steps_per_waypoint_for_move(from_square, to_square):
    if (from_square, to_square) in LONG_TRANSPORT_STEP_MOVES:
        return LONG_TRANSPORT_MOVE_STEPS_PER_WAYPOINT
    return DEFAULT_MOVE_STEPS_PER_WAYPOINT


def source_grasp_offset_for_move(source_from_square, source_to_square):
    return SOURCE_GRASP_OFFSET_OVERRIDES.get(
        (source_from_square, source_to_square),
        SOURCE_GRASP_OFFSET
    )


def build_source_trajectory(source_from_square, source_to_square):
    source_grasp_offset = source_grasp_offset_for_move(
        source_from_square,
        source_to_square
    )
    movelist, closeidx, traj_metrics = pickupmove_traj_with_metrics(
        source_from_square,
        source_to_square,
        board_origin=board_origin,
        GRASP_OFFSET=source_grasp_offset,
        PLACE_OFFSET=PLACE_OFFSET
    )
    return movelist, closeidx, traj_metrics, {
        "movelist": movelist,
        "closeidx": closeidx,
        "traj_metrics": traj_metrics,
    }, source_grasp_offset


def build_reverse_trajectory(
    source_movelist,
    source_closeidx,
    source_traj_metrics,
    reversed_from_square,
    reversed_to_square,
    place_offset,
    reversed_grasp_offset=REVERSED_GRASP_OFFSET
):
    return build_reversed_pick_place_trajectory(
        source_movelist,
        source_closeidx,
        source_traj_metrics,
        reversed_from_square,
        reversed_to_square,
        grasp_offset=reversed_grasp_offset,
        place_offset=place_offset
    )


def run_xy_correction_round(
    previous_result,
    previous_override,
    source_movelist,
    source_closeidx,
    source_traj_metrics,
    reversed_from_square,
    reversed_to_square,
    correction_round,
    record_video=False,
    move_steps_per_waypoint=DEFAULT_MOVE_STEPS_PER_WAYPOINT,
    reversed_grasp_offset=REVERSED_GRASP_OFFSET
):
    corrected_place_offset, world_correction, gripper_frame_correction = apply_reversed_place_correction(
        previous_result,
        previous_override,
        correct_z=False
    )
    corrected_override = build_reverse_trajectory(
        source_movelist,
        source_closeidx,
        source_traj_metrics,
        reversed_from_square,
        reversed_to_square,
        corrected_place_offset,
        reversed_grasp_offset=reversed_grasp_offset
    )
    corrected_result = run_scored_simulation(
        reversed_from_square,
        reversed_to_square,
        reversed_grasp_offset,
        corrected_place_offset,
        corrected_override,
        record_video=record_video,
        video_label=f"placement_corrected_round_{correction_round}",
        move_steps_per_waypoint=move_steps_per_waypoint
    )
    corrected_result["correction_round"] = correction_round
    corrected_result["world_correction"] = world_correction.copy()
    corrected_result["gripper_frame_correction"] = gripper_frame_correction.copy()
    return corrected_result, corrected_override


def run_verified_reverse_move(
    from_square=MOVE_FROM_SQUARE,
    to_square=MOVE_TO_SQUARE,
    record_video=True,
    correction_rounds=REVERSED_PLACEMENT_CORRECTION_ROUNDS
):
    source_from_square = to_square
    source_to_square = from_square
    reversed_from_square = from_square
    reversed_to_square = to_square
    reverse_move_steps = move_steps_per_waypoint_for_move(
        reversed_from_square,
        reversed_to_square
    )

    print(f"\n========== Verified reverse strategy {from_square} -> {to_square} ==========")
    print(f"Verifying source trajectory {source_from_square} -> {source_to_square}")
    source_grasp_offset = source_grasp_offset_for_move(
        source_from_square,
        source_to_square
    )
    print("source_GRASP_OFFSET:", source_grasp_offset)
    print("reversed_GRASP_OFFSET:", REVERSED_GRASP_OFFSET)

    source_movelist, source_closeidx, source_traj_metrics, source_override, source_grasp_offset = build_source_trajectory(
        source_from_square,
        source_to_square
    )
    source_result = run_scored_simulation(
        source_from_square,
        source_to_square,
        source_grasp_offset,
        PLACE_OFFSET,
        source_override
    )
    print_result_summary("Source verification result", source_result)

    if source_result["reject_reason"] is not None:
        print("Skipping reversed run because source verification failed.")
        return source_result, None

    reversed_override = build_reverse_trajectory(
        source_movelist,
        source_closeidx,
        source_traj_metrics,
        reversed_from_square,
        reversed_to_square,
        PLACE_OFFSET
    )
    reversed_result = run_scored_simulation(
        reversed_from_square,
        reversed_to_square,
        REVERSED_GRASP_OFFSET,
        PLACE_OFFSET,
        reversed_override,
        record_video=record_video,
        video_label="initial_reverse",
        move_steps_per_waypoint=reverse_move_steps
    )
    print_result_summary("Initial reversed result", reversed_result, include_video=True)

    corrected_results = []
    previous_result = reversed_result
    previous_override = reversed_override

    if not RUN_XY_CORRECTION_TEST:
        print("\nSkipping XY placement correction.")
        return source_result, reversed_result, corrected_results

    for correction_round in range(1, correction_rounds + 1):
        previous_position_error = np.array(
            previous_result.get("position_error", np.full(3, np.nan))
        )
        if (
            previous_result.get("reject_reason") is not None
            or not np.all(np.isfinite(previous_position_error))
        ):
            print(
                "\nSkipping remaining XY placement correction because the "
                "previous trajectory result is rejected or has invalid position metrics."
            )
            break

        corrected_result, corrected_override = run_xy_correction_round(
            previous_result,
            previous_override,
            source_movelist,
            source_closeidx,
            source_traj_metrics,
            reversed_from_square,
            reversed_to_square,
            correction_round,
            record_video=record_video,
            move_steps_per_waypoint=reverse_move_steps
        )
        print_result_summary(
            f"XY correction round {correction_round}",
            corrected_result,
            include_video=True
        )
        corrected_results.append(corrected_result)
        previous_result = corrected_result
        previous_override = corrected_override

        if is_xy_correction_good_enough(corrected_result):
            print(
                "Stopping XY placement correction because xy_error "
                f"{corrected_result['xy_error']} <= {XY_CORRECTION_TARGET_ERROR}"
            )
            break

    return source_result, reversed_result, corrected_results


def is_successful_move_result(result):
    return (
        result is not None
        and result.get("reject_reason") is None
        and bool(result.get("pickup_success", False))
        and not bool(result.get("premature_drop", False))
        and np.isfinite(float(result.get("trajectory_fk_error", np.inf)))
    )


def result_xy_error(result):
    return float(result.get("xy_error", np.inf))


def result_tilt_error(result):
    return float(result.get("final_tilt_deg", np.inf))


def is_xy_correction_good_enough(result):
    return (
        is_successful_move_result(result)
        and np.isfinite(result_xy_error(result))
        and result_xy_error(result) <= XY_CORRECTION_TARGET_ERROR
    )


def is_suitable_reverse_solution(result):
    return (
        is_successful_move_result(result)
        and np.isfinite(result_tilt_error(result))
        and np.isfinite(result_xy_error(result))
        and result_tilt_error(result) < FINAL_TILT_TARGET_DEG
        and result_xy_error(result) < XY_CORRECTION_TARGET_ERROR
    )


def calibrate_verified_reverse_move(
    from_square,
    to_square,
    record_initial_video=False,
    record_intermediate_video=False,
    record_final_video=True,
    xy_correction_enabled=True,
    correction_rounds=REVERSED_PLACEMENT_CORRECTION_ROUNDS,
    reversed_grasp_offset=REVERSED_GRASP_OFFSET
):
    source_from_square = to_square
    source_to_square = from_square
    reversed_from_square = from_square
    reversed_to_square = to_square
    reverse_move_steps = move_steps_per_waypoint_for_move(
        reversed_from_square,
        reversed_to_square
    )

    print(f"\n========== Calibrating reverse lookup {from_square} -> {to_square} ==========")
    print(f"Verifying source trajectory {source_from_square} -> {source_to_square}")
    source_grasp_offset = source_grasp_offset_for_move(
        source_from_square,
        source_to_square
    )
    print("source_GRASP_OFFSET:", source_grasp_offset)
    print("reversed_GRASP_OFFSET:", reversed_grasp_offset)

    source_movelist, source_closeidx, source_traj_metrics, source_override, source_grasp_offset = build_source_trajectory(
        source_from_square,
        source_to_square
    )
    source_result = run_scored_simulation(
        source_from_square,
        source_to_square,
        source_grasp_offset,
        PLACE_OFFSET,
        source_override
    )
    print_result_summary("Source verification result", source_result)

    if source_result["reject_reason"] is not None:
        print("Skipping calibration because source verification failed.")
        return {
            "source_result": source_result,
            "initial_reverse_result": None,
            "tilt_results": [],
            "xy_results": [],
            "selected_result": None,
            "selected_place_offset": PLACE_OFFSET.copy(),
            "selected_release_wrist_delta_deg": np.zeros(2),
            "selected_grasp_wrist_delta_deg": np.zeros(2),
            "source_grasp_offset": source_grasp_offset.copy(),
            "reversed_grasp_offset": reversed_grasp_offset.copy(),
            "trajectory_metrics": source_traj_metrics,
            "video_output_dir": None,
            "success": False,
            "reject_reason": source_result["reject_reason"],
        }

    reversed_override = build_reverse_trajectory(
        source_movelist,
        source_closeidx,
        source_traj_metrics,
        reversed_from_square,
        reversed_to_square,
        PLACE_OFFSET,
        reversed_grasp_offset=reversed_grasp_offset
    )
    reversed_result = run_scored_simulation(
        reversed_from_square,
        reversed_to_square,
        reversed_grasp_offset,
        PLACE_OFFSET,
        reversed_override,
        record_video=record_initial_video,
        video_label="initial_reverse",
        move_steps_per_waypoint=reverse_move_steps
    )
    print_result_summary("Initial reversed calibration result", reversed_result)

    tilt_results = []
    xy_results = []
    selected_result = None
    selected_override = None
    selected_place_offset = PLACE_OFFSET.copy()
    selected_trajectory_metrics = reversed_override["traj_metrics"]
    previous_result = reversed_result
    previous_override = reversed_override

    def select_solution(result, trajectory_override):
        nonlocal selected_result
        nonlocal selected_override
        nonlocal selected_place_offset
        nonlocal selected_trajectory_metrics

        selected_result = result
        selected_override = trajectory_override
        selected_place_offset = trajectory_override["place_offset"].copy()
        selected_trajectory_metrics = trajectory_override["traj_metrics"]

    if is_suitable_reverse_solution(reversed_result):
        select_solution(reversed_result, reversed_override)
    elif not is_successful_move_result(reversed_result):
        print("Skipping corrections because initial reversed move failed.")
    elif xy_correction_enabled:
        for correction_round in range(1, correction_rounds + 1):
            previous_position_error = np.array(
                previous_result.get("position_error", np.full(3, np.nan))
            )
            if not np.all(np.isfinite(previous_position_error)):
                print("Stopping XY correction because previous position metrics are invalid.")
                break

            xy_result, xy_override = run_xy_correction_round(
                previous_result,
                previous_override,
                source_movelist,
                source_closeidx,
                source_traj_metrics,
                reversed_from_square,
                reversed_to_square,
                correction_round,
                record_video=record_intermediate_video,
                move_steps_per_waypoint=reverse_move_steps,
                reversed_grasp_offset=reversed_grasp_offset
            )
            xy_results.append(xy_result)
            print_result_summary(f"XY correction round {correction_round}", xy_result)

            if not is_successful_move_result(xy_result):
                print("Stopping XY correction at first rejected/failed round.")
                break

            previous_result = xy_result
            previous_override = xy_override

            if is_suitable_reverse_solution(xy_result):
                select_solution(xy_result, xy_override)
                break

            if is_xy_correction_good_enough(xy_result):
                print(
                    "Stopping XY correction because xy_error "
                    f"{xy_result['xy_error']} <= {XY_CORRECTION_TARGET_ERROR}"
                )
                break

    if record_final_video and is_suitable_reverse_solution(selected_result):
        selected_result = run_scored_simulation(
            reversed_from_square,
            reversed_to_square,
            reversed_grasp_offset,
            selected_place_offset,
            selected_override,
            record_video=True,
            video_label="final_corrected_lookup",
            move_steps_per_waypoint=reverse_move_steps
        )
        print_result_summary("Final video verification result", selected_result, include_video=True)

    success = is_suitable_reverse_solution(selected_result)
    if success:
        reject_reason = None
    else:
        reject_reason = "none_suitable_found"
        if selected_result is None:
            print("\nNone suitable found.")
        else:
            print(
                "\nNone suitable found: "
                f"tilt={selected_result.get('final_tilt_deg')} "
                f"(target < {FINAL_TILT_TARGET_DEG}), "
                f"xy_error={selected_result.get('xy_error')} "
                f"(target < {XY_CORRECTION_TARGET_ERROR})"
            )
            selected_result = None

    return {
        "source_result": source_result,
        "initial_reverse_result": reversed_result,
        "tilt_results": tilt_results,
        "xy_results": xy_results,
        "selected_result": selected_result,
        "selected_place_offset": selected_place_offset.copy(),
        "selected_release_wrist_delta_deg": np.zeros(2),
        "selected_grasp_wrist_delta_deg": np.zeros(2),
        "source_grasp_offset": source_grasp_offset.copy(),
        "reversed_grasp_offset": reversed_grasp_offset.copy(),
        "trajectory_metrics": selected_trajectory_metrics,
        "video_output_dir": selected_result.get("video_output_dir") if selected_result else None,
        "success": success,
        "reject_reason": reject_reason,
    }


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


RUN_SINGLE_VIDEO_INSPECTION = True
RUN_VERIFIED_REVERSE_MOVE = True
RUN_GRASP_OFFSET_SEARCH = False
RUN_GRASP_ONLY_UNTIL_HELD = True
RUN_GRASP_THEN_PLACE_SEARCH = False
RUN_PLACE_OFFSET_SEARCH = False
RUN_MEASURED_CORRECTION_SEARCH = True
RUN_RANK1_PLACE_OFFSET_BATCH = True
RECORD_VIDEO_RERUN = True

def main():
    ensure_physics_connected()

    # e1a1 Best verified PLACE_OFFSET: [-0.02246538  0.00114195 -0.00431543]
    # e1b1 Best verified PLACE_OFFSET: [-0.02214579  0.00110381 -0.00460133]
    # e1e1 Best verified PLACE_OFFSET: [-0.02208837  0.0008782  -0.00150453]
    try:
        if RUN_VERIFIED_REVERSE_MOVE:
            run_verified_reverse_move(
                from_square=MOVE_FROM_SQUARE,
                to_square=MOVE_TO_SQUARE,
                record_video=RECORD_VIDEO_RERUN
            )
        elif RUN_SINGLE_VIDEO_INSPECTION:
            run_single_video_inspection(
                from_square=MOVE_FROM_SQUARE,
                to_square=MOVE_TO_SQUARE
            )
        elif RUN_GRASP_ONLY_UNTIL_HELD:
            run_grasp_only_until_held(
                from_square=MOVE_FROM_SQUARE,
                to_square=MOVE_TO_SQUARE,
                record_video=RECORD_VIDEO_RERUN
            )
        elif RUN_GRASP_THEN_PLACE_SEARCH:
            run_grasp_then_place_search(
                from_square=MOVE_FROM_SQUARE,
                to_square=MOVE_TO_SQUARE,
                record_video=RECORD_VIDEO_RERUN
            )
        elif RUN_GRASP_OFFSET_SEARCH:
            run_grasp_offset_search(record_video=RECORD_VIDEO_RERUN)
        elif RUN_PLACE_OFFSET_SEARCH:
            search_from_square = MOVE_FROM_SQUARE
            search_to_square = MOVE_TO_SQUARE
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
                        "trajectory_fk_error": best_result["trajectory_fk_error"],
                        "reject_reason": best_result["reject_reason"],
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
                        f"traj_fk={item['trajectory_fk_error']:.5f}, "
                        f"xy={item['xy_error']:.5f}, "
                        f"z={item['z_error']:.5f}, "
                        f"pickup={item['pickup_success']}, "
                        f"reject={item['reject_reason']}, "
                        f"tilt={item['final_tilt_deg']:.2f}"
                    )
                print("]")
                print(f"Rank 1 batch time: {batch_elapsed:.2f}s")
            finally:
                p.removeState(world["state_id"])
        else:
            cnt = 0
            successlist = []
            for grasp_offset in grasp_offsets:
                print(f"Testing GRASP_OFFSET: {cnt}")
                cnt += 1
                success = grasptest(grasp_offset)
                successlist.append(success)
                print("-----------------------------------\n")
                print("CNT:", cnt)

            print(successlist)
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()

#4 delta best so far with 51/64
