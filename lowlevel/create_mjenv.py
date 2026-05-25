import os
import mujoco
import imageio
import numpy as np

# Load the existing SO101 XML (already in MuJoCo format)
xml_path = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/so101_new_calib.xml"

try:
    model = mujoco.MjModel.from_xml_path(xml_path)
    print("✓ Successfully loaded SO101 model!")
    print(f"✓ Number of bodies: {model.nbody}")
    print(f"✓ Number of joints: {model.njnt}")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

data = mujoco.MjData(model)

# Setup rendering
renderer = mujoco.Renderer(model, height=480, width=640)
# Set background color to white (RGBA)
# renderer.scene.bgcolor = [1, 1, 1, 1]  # White
# Or light gray: [0.8, 0.8, 0.8, 1]

# Setup video writer
video_path = "so101_simulation.mp4"
writer = imageio.get_writer(video_path, fps=30)

print("Running simulation and recording video...")
for step in range(500):  # 500 steps = ~17 seconds at 30fps
    mujoco.mj_step(model, data)
    
    renderer.update_scene(data)
    pixels = renderer.render()
    
    # pixels are already in RGB format
    writer.append_data(pixels)
    
    if step % 50 == 0:
        print(f"  Frame {step}...")

writer.close()
print(f"✓ Video saved to: {video_path}")
print("Open it with: open so101_simulation.mp4")