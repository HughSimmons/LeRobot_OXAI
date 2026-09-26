import pybullet as p
import pybullet_data
import imageio
import numpy as np
from pathlib import Path
import time

# Initialize PyBullet in headless mode
# physics_client = p.connect(p.DIRECT)
physics_client = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load ground plane
plane_id = p.loadURDF("plane.urdf", [0, 0, 0])

# Create two spheres for collision demo
pusher_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
pusher_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
pusher_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=pusher_shape,
                              baseVisualShapeIndex=pusher_visual, basePosition=[-0.4, 0, 0.05])

target_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 0, 1, 1])
target_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=target_shape,
                              baseVisualShapeIndex=target_visual, basePosition=[0.3, 0, 0.05])

# Set friction for both spheres
p.changeDynamics(pusher_id, -1, lateralFriction=0.5)
p.changeDynamics(target_id, -1, lateralFriction=0.5)

print("✓ Physics environment initialized: 2 spheres, ground plane")

# Define three camera viewpoints
cameras = {
    'default': {
        'eye': [0, 0.8, 0.5],
        'target': [0, 0, 0],
        'up': [0, 0, 1],
    },
    'side': {
        'eye': [0.8, 0, 0.5],
        'target': [0, 0, 0],
        'up': [0, 0, 1],
    },
    'top': {
        'eye': [0, 0, 1.0],
        'target': [0, 0, 0],
        'up': [0, -1, 0],
    }
}

# Create video writers for each camera
video_path = Path('.')
# writers = {
#     name: imageio.get_writer(video_path / f"collision_demo_{name}.mp4", fps=30)
#     for name in cameras
# }

print("Running PyBullet collision demo with 3 cameras...")
for step in range(400):
    if step < 100:
        p.applyExternalForce(pusher_id, -1, [1.0, 0, 0], [0, 0, 0], p.WORLD_FRAME)
    
    p.stepSimulation()
    # p.resetDebugVisualizerCamera()
    time.sleep(0.01)  # Add small delay
    
    # Skip all the camera rendering for GUI mode

# Close writers
# for writer in writers.values():
#     writer.close()

print("✓ Demo complete!")
print("  - collision_demo_default.mp4 (front view)")
print("  - collision_demo_side.mp4 (side view)")
print("  - collision_demo_top.mp4 (top view)")

p.disconnect()