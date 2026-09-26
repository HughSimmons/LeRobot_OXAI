import mujoco
import imageio
import numpy as np

# Two spheres on floor, one gets pushed horizontally
xml_string = """
<mujoco model="collision_test">
  <option gravity="0 0 -9.81"/>
  
  <worldbody>
    <!-- Ground floor -->
    <geom type="plane" size="2 2 0.1" rgba="0.7 0.7 0.7 1" contype="1" conaffinity="1"/>
    
    <!-- Target sphere (stationary on floor) -->
    <body name="target_sphere" pos="0.3 0 0.05">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="0.1" rgba="0 0 1 1" contype="1" conaffinity="1" friction="0.5 0.5 0.01"/>
    </body>
    
    <!-- Pusher sphere (will slide and collide) -->
    <body name="pusher_sphere" pos="-0.4 0 0.05">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="0.1" rgba="1 0 0 1" contype="1" conaffinity="1" friction="0.5 0.5 0.01"/>
    </body>
  </worldbody>
</mujoco>
"""

# Load model
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

print(f"✓ Model loaded: {model.nbody} bodies, {model.ngeom} geoms")

# Setup rendering (default camera only)
renderer = mujoco.Renderer(model, height=480, width=640)
writer = imageio.get_writer("collision_demo.mp4", fps=30)

print("Running horizontal collision demo...")
for step in range(400):
    # Apply horizontal push force to red sphere for first 100 steps
    if step < 100:
        data.body("pusher_sphere").xfrc_applied[0] = 1.0  # Push in +x direction
    
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    pixels = renderer.render()
    writer.append_data(pixels)
    
    if step % 50 == 0:
        pusher_pos = data.body("pusher_sphere").xpos
        target_pos = data.body("target_sphere").xpos
        distance = np.linalg.norm(pusher_pos - target_pos)
        print(f"  Frame {step}: distance={distance:.4f}m, pusher_x={pusher_pos[0]:.3f}, target_x={target_pos[0]:.3f}")

writer.close()
print(f"✓ Demo saved to: collision_demo.mp4")
print("Open it with: open collision_demo.mp4")