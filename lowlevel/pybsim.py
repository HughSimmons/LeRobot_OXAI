import pybullet as p
import pybullet_data
import imageio
import numpy as np
import time
from pathlib import Path

# Initialize PyBullet in HEADLESS mode
physics_client = p.connect(p.DIRECT)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

print("Setting up physics environment...")

# Load ground plane
plane_id = p.loadURDF("plane.urdf", [0, 0, 0])

# Try to load SO101 robot
try:
    urdf_path = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
    robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
    print(f"✓ Loaded SO101 from URDF")
except FileNotFoundError:
    print("⚠ URDF not found.")
    robot_id = None
except Exception as e:
    print(f"⚠ Error loading URDF: {e}")
    robot_id = None

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

# Create chess pieces
piece_positions = [
    (-0.14, -0.14), (-0.10, -0.14), (-0.06, -0.14), (-0.02, -0.14),
    (0.02, -0.14), (0.06, -0.14), (0.10, -0.14), (0.14, -0.14),
]

piece_ids = []
for i, (px, py) in enumerate(piece_positions):
    world_x = board_x + px
    world_y = board_y + py
    world_z = board_z + 0.035
    
    piece_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.012, height=0.02)
    piece_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.012, length=0.02, 
                                       rgbaColor=[1, 0, 0, 1])
    piece_id = p.createMultiBody(baseMass=0.05, baseCollisionShapeIndex=piece_shape,
                                 baseVisualShapeIndex=piece_visual,
                                 basePosition=[world_x, world_y, world_z])
    p.changeDynamics(piece_id, -1, linearDamping=0.04, angularDamping=0.04, lateralFriction=0.5)
    piece_ids.append(piece_id)

print(f"✓ Created {len(piece_ids)} chess pieces")

# Create test piece
test_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.012, height=0.02)
test_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.012, length=0.02, 
                                  rgbaColor=[0, 1, 0, 1])
test_piece_id = p.createMultiBody(baseMass=0.05, baseCollisionShapeIndex=test_shape,
                                  baseVisualShapeIndex=test_visual,
                                  basePosition=[0, 0, 0.4])
p.changeDynamics(test_piece_id, -1, linearDamping=0.04, angularDamping=0.04, lateralFriction=0.5)

print("✓ All objects loaded and ready!")

# NOW setup video recording
camera_params = {
    'eye': [0.5, 0.5, 0.5],
    'target': [0.3, 0, 0],
    'up': [0, 0, 1],
}

orthogonal_camera_params = {
    'eye': [0.3, 0.5, 0.3],
    'target': [0.3, 0, 0.1],
    'up': [0, 0, 1],
}

WIDTH, HEIGHT = 1280, 720

writer = imageio.get_writer('so101_chess_sim.mp4', fps=30, codec='libx264', quality=8)
writer2 = imageio.get_writer('so101_chess_sim_orthogonal.mp4', fps=30, codec='libx264', quality=8)

# In the simulation loop:
for step in range(400):
    if 50 < step < 150:
        p.applyExternalForce(piece_ids[0], -1, [0.5, 0, 0], [0, 0, 0], p.WORLD_FRAME)
    
    p.stepSimulation()
    
    # Camera 1
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_params['eye'],
        cameraTargetPosition=camera_params['target'],
        cameraUpVector=camera_params['up']
    )
    
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=WIDTH/HEIGHT, nearVal=0.01, farVal=100
    )
    
    width, height, rgba, _, _ = p.getCameraImage(
        WIDTH, HEIGHT,  # Changed resolution
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix
    )
    rgb_array = np.array(rgba[:, :, :3], dtype=np.uint8)
    writer.append_data(rgb_array)
    
    # Camera 2
    view_matrix2 = p.computeViewMatrix(
        cameraEyePosition=orthogonal_camera_params['eye'],
        cameraTargetPosition=orthogonal_camera_params['target'],
        cameraUpVector=orthogonal_camera_params['up']
    )
    
    width, height, rgba2, _, _ = p.getCameraImage(
        WIDTH, HEIGHT,  # Changed resolution
        viewMatrix=view_matrix2,
        projectionMatrix=proj_matrix
    )
    rgb_array2 = np.array(rgba2[:, :, :3], dtype=np.uint8)
    writer2.append_data(rgb_array2)
    
    if step % 100 == 0:
        print(f"  Frame {step}...")

writer.close()
writer2.close()
print(f"✓ Videos saved to: so101_chess_sim.mp4 and so101_chess_sim_orthogonal.mp4")
p.disconnect()