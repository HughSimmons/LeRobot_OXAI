import os
import mujoco
import imageio
import numpy as np

# Load the original SO101 XML
xml_path = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/so101_new_calib.xml"

with open(xml_path, 'r') as f:
    xml_content = f.read()

# Fix the meshdir to use absolute path
assets_path = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/assets"
xml_content = xml_content.replace('meshdir="assets"', f'meshdir="{assets_path}"')

# Add chessboard and pieces before the closing </worldbody> tag
chessboard_xml = '''    <!-- Chessboard with squares -->
    <body name="chessboard" pos="0.3 0 0">
      <!-- Board base -->
      <geom name="board_base" type="box" size="0.16 0.16 0.005" rgba="0.3 0.3 0.3 1" contype="1" conaffinity="1"/>
      
      <!-- Checkerboard squares (8x8) -->
'''

# Add 64 squares programmatically
square_size = 0.04  # 4cm per square (8 squares * 4cm = 32cm)
for row in range(8):
    for col in range(8):
        x = -0.16 + (col + 0.5) * square_size
        y = -0.16 + (row + 0.5) * square_size
        # Alternate colors - white and dark
        if (row + col) % 2 == 0:
            color = "1 1 1 0.3"  # White
        else:
            color = "0.1 0.1 0.1 0.3"  # Dark
        
        chessboard_xml += f'''      <geom type="box" pos="{x} {y} 0.005" size="{square_size/2} {square_size/2} 0.003" rgba="{color}" contype="0" conaffinity="0"/>
'''

# Close the chessboard body (no pieces nested inside)
chessboard_xml += '''    </body>
'''

# Add chess pieces as separate worldbody-level bodies with correct world positions
piece_positions = [
    (-0.14, -0.14),
    (-0.10, -0.14),
    (-0.06, -0.14),
    (-0.02, -0.14),
    (0.02, -0.14),
    (0.06, -0.14),
    (0.10, -0.14),
    (0.14, -0.14),
]

# Board is at x=0.3, y=0. Piece local coords map to world coords:
# world_x = 0.3 + px
# world_y = py (since board center y is 0)
for i, (px, py) in enumerate(piece_positions):
    world_x = 0.3 + px
    world_y = py
    world_z = 0.035  # Place on top of board surface (0.005 + 0.02 + some clearance)
    chessboard_xml += f'''    <body name="piece_{i}" pos="{world_x} {world_y} {world_z}">
        <freejoint/>
        <geom type="cylinder" size="0.012 0.02" mass="0.05" rgba="1 0 0 1" contype="1" conaffinity="1" friction="0.1 0.1 0.1"/>
      </body>
'''

# Add a test piece above robot origin to check collision response
test_piece_xml = '''    <!-- Test piece dropping on robot origin -->
    <body name="test_piece" pos="0 0 0.4">
        <freejoint/>
        <geom type="cylinder" size="0.012 0.02" mass="0.05" rgba="0 1 0 1" contype="1" conaffinity="1" friction="0.1 0.1 0.1"/>
    </body>
'''

# Insert before closing worldbody tag
xml_modified = xml_content.replace('  </worldbody>', chessboard_xml + test_piece_xml + '  </worldbody>')

# Save modified XML
xml_output_path = "/Users/zhg603/Documents/OXAI/lowlevel/so101_with_chess.xml"
with open(xml_output_path, 'w') as f:
    f.write(xml_modified)

print(f"✓ Created modified XML: {xml_output_path}")

# Now load and render
try:
    model = mujoco.MjModel.from_xml_path(xml_output_path)
    print("✓ Successfully loaded SO101 model with chessboard!")
    print(f"✓ Number of bodies: {model.nbody}")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

data = mujoco.MjData(model)

# Setup rendering - two cameras
renderer = mujoco.Renderer(model, height=480, width=640)

# Create a second renderer for side view (just duplicate for now)
renderer_ortho = mujoco.Renderer(model, height=480, width=640)

# Setup video writers
video_path = "so101_with_chess.mp4"
writer = imageio.get_writer(video_path, fps=30)

video_path_ortho = "so101_with_chess_ortho.mp4"
writer_ortho = imageio.get_writer(video_path_ortho, fps=30)

# From simfk.py - these are in DEGREES
home = np.array([96.92, -107.87, 97.36, 65.19, -29.85, 4.63])
corner1 = np.array([97.32, 0.40, 28.40, 66.55, 177.80, 4.95])
corner2 = np.array([38.59, 60.88, -58.55, 100.48, 178.15, 4.95])

# Simple smooth move sequence
moves = [
    (home, 50),           # Start at home
    (home, 200),          # Hold home while we push pieces
    (home, 100),          # Stay there
]

current_move_idx = 0
current_start_pos = np.deg2rad(home)
current_target_pos = np.deg2rad(moves[0][0])
steps_in_move = 0
move_duration = moves[0][1]

print("Running simulation and recording video...")
for step in range(350):
    # Interpolate between current start and target
    alpha = steps_in_move / move_duration
    target_joints = (1 - alpha) * current_start_pos + alpha * current_target_pos
    
    # Send to MuJoCo
    data.ctrl[:] = target_joints
    
    # First camera (default perspective)
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    pixels = renderer.render()
    writer.append_data(pixels)
    
    # Second camera - change lookat position for different view
    renderer_ortho.update_scene(data)
    
    # Try accessing camera attributes
    if hasattr(renderer_ortho, '_scene'):
        try:
            cam = renderer_ortho._scene.camera[0]
            # Try these attributes instead
            if hasattr(cam, 'lookat'):
                cam.lookat = np.array([0.3, 0, 0.1])  # Focus on board
            if hasattr(cam, 'azimuth'):
                cam.azimuth = 90
            if hasattr(cam, 'elevation'):
                cam.elevation = 0
        except:
            pass  # If camera modification fails, just render with default view
    
    pixels_ortho = renderer_ortho.render()
    writer_ortho.append_data(pixels_ortho)
    


    steps_in_move += 1
    
    # Move to next waypoint
    if steps_in_move >= move_duration and current_move_idx < len(moves) - 1:
        current_move_idx += 1
        current_start_pos = current_target_pos.copy()
        current_target_pos = np.deg2rad(moves[current_move_idx][0])
        move_duration = moves[current_move_idx][1]
        steps_in_move = 0
    
    if step % 50 == 0:
        print(f"  Frame {step}...")

    # Add direct push to piece_0 (like the sphere demo)
    if 50 < step < 150:
        data.body("piece_0").xfrc_applied[0] = 0.5  # Push piece horizontally

writer.close()
writer_ortho.close()
print(f"✓ Video saved to: {video_path}")
print(f"✓ Orthogonal video saved to: {video_path_ortho}")
print("Open it with: open so101_with_chess.mp4")

# # Debug: check collision geometry
# print(f"\nGeometry collision info:")
# for geom_id in range(model.ngeom):
#     body_id = model.geom_bodyid[geom_id]
#     body_name = model.body(body_id).name
#     contype = model.geom_contype[geom_id]
#     conaffinity = model.geom_conaffinity[geom_id]
#     print(f"  Geom {geom_id}: body={body_name}, contype={contype}, conaffinity={conaffinity}")