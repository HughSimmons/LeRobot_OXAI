import pybullet as p
import pybullet_data
import imageio
import numpy as np
from pathlib import Path

# Initialize PyBullet in HEADLESS mode
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
    piece_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=piece_shape,
                                 baseVisualShapeIndex=piece_visual,
                                 basePosition=[world_x, world_y, world_z])
    p.changeDynamics(piece_id, -1, linearDamping=0.04, angularDamping=0.04, lateralFriction=0.5)
    piece_ids.append(piece_id)

print(f"✓ Created {len(piece_ids)} large chess pieces")

print("✓ All objects loaded and ready!")

# Robot joint waypoints (from simfk.py)
home = np.array([96.92, -107.87, 97.36, 65.19, -29.85, 4.63])
corner1 = np.array([97.32, 0.40, 28.40, 66.55, 177.80, 4.95])
corner2 = np.array([38.59, 60.88, -58.55, 100.48, 178.15, 4.95])

# Convert to radians
home_rad = np.deg2rad(home)
corner1_rad = np.deg2rad(corner1)
corner2_rad = np.deg2rad(corner2)

# Define movement sequence
moves = [
    (home_rad, 100, "Move to HOME"),
    (corner1_rad, 150, "Move to CORNER1"),
    (corner2_rad, 150, "Move to CORNER2"),
    (home_rad, 150, "Return to HOME"),
]

# Setup video recording with THREE cameras
camera_params = {
    'eye': [0.5, 0.5, 0.5],
    'target': [0.3, 0, 0],
    'up': [0, 0, 1],
}

top_down_camera_params = {
    'eye': [0.3, 0, 0.6],
    'target': [0.3, 0, 0],
    'up': [0, -1, 0],
}

WIDTH, HEIGHT = 1280, 720
writer = imageio.get_writer('so101_robot_moves.mp4', fps=30, codec='libx264', quality=8)
writer_topdown = imageio.get_writer('so101_robot_moves_topdown.mp4', fps=30, codec='libx264', quality=8)

print("Starting robot movement simulation...")

# Calculate total steps
total_steps = sum(steps for _, steps, _ in moves)
current_start_pos = home_rad.copy()
move_idx = 0

for global_step in range(total_steps + 100):  # Extra steps at end
    current_target_pos, move_steps, move_name = moves[move_idx]
    
    if global_step > 0 and global_step % move_steps == 0 and move_idx < len(moves) - 1:
        move_idx += 1
        current_start_pos = current_target_pos.copy()
        current_target_pos, move_steps, move_name = moves[move_idx]
    
    local_step = global_step % move_steps if move_idx < len(moves) else move_steps
    alpha = min(local_step / move_steps, 1.0)
    target_joints = (1 - alpha) * current_start_pos + alpha * current_target_pos
    
    if robot_id is not None:
        for joint_idx in range(min(6, p.getNumJoints(robot_id))):
            p.setJointMotorControl2(
                robot_id, joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_joints[joint_idx],
                force=500  # Maximum force
            )
    
    p.stepSimulation()
    
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=WIDTH/HEIGHT, nearVal=0.01, farVal=100
    )
    
    # Camera 1 - Perspective view
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_params['eye'],
        cameraTargetPosition=camera_params['target'],
        cameraUpVector=camera_params['up']
    )
    width, height, rgba, _, _ = p.getCameraImage(
        WIDTH, HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix
    )
    rgb_array = np.array(rgba[:, :, :3], dtype=np.uint8)
    writer.append_data(rgb_array)
    
    # Camera 2 - Top-down view
    view_matrix_topdown = p.computeViewMatrix(
        cameraEyePosition=top_down_camera_params['eye'],
        cameraTargetPosition=top_down_camera_params['target'],
        cameraUpVector=top_down_camera_params['up']
    )
    width, height, rgba_topdown, _, _ = p.getCameraImage(
        WIDTH, HEIGHT,
        viewMatrix=view_matrix_topdown,
        projectionMatrix=proj_matrix
    )
    rgb_array_topdown = np.array(rgba_topdown[:, :, :3], dtype=np.uint8)
    writer_topdown.append_data(rgb_array_topdown)
    
    if global_step % 50 == 0:
        move_desc = moves[min(move_idx, len(moves)-1)][2]
        print(f"  Frame {global_step} ({move_desc})...")

writer.close()
writer_topdown.close()
print(f"✓ Videos saved to:")
print(f"  - so101_robot_moves.mp4 (perspective view)")
print(f"  - so101_robot_moves_topdown.mp4 (top-down view)")
p.disconnect()