import sys
from pathlib import Path
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
URDF_PATH = str(paths.URDF_PATH)
OUT_DIR = Path(__file__).resolve().parent / "output" / "urdf_visualization"
W, H = 1280, 960
TABLE_Z = -0.00240  # exact vertex-based lowest point of base_so101_v2.stl

p.connect(p.DIRECT)
robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
for link_idx in range(p.getNumJoints(robot_id)):
    p.changeVisualShape(robot_id, link_idx, rgbaColor=[0, 0, 0, 0])

def project(view, proj, pt):
    c = np.array(proj).reshape(4, 4, order="F") @ np.array(view).reshape(4, 4, order="F") @ np.array([*pt, 1.0])
    n = c[:3] / c[3]
    return ((n[0] * 0.5 + 0.5) * W, (1 - (n[1] * 0.5 + 0.5)) * H)

view = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.0, 0.0, 0.02], distance=0.18, yaw=0, pitch=-10, roll=0, upAxisIndex=2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=W / H, nearVal=0.01, farVal=3.0)
img = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER)[2]
arr = np.reshape(img, (H, W, 4))[:, :, :3].astype(np.uint8)
im = Image.fromarray(arr)
d = ImageDraw.Draw(im)

origin = np.array([0.0, 0.0, 0.0])
d.line([project(view, proj, origin), project(view, proj, origin + [0.03, 0, 0])], fill=(255, 40, 40), width=3)
d.line([project(view, proj, origin), project(view, proj, origin + [0, 0.03, 0])], fill=(0, 170, 0), width=3)
d.line([project(view, proj, origin), project(view, proj, origin + [0, 0, 0.03])], fill=(40, 90, 255), width=3)
x, y = project(view, proj, origin)
d.ellipse([x-9, y-9, x+9, y+9], fill=(255, 0, 255), outline=(0,0,0), width=2)
d.text((x+14, y-10), "robot origin (0,0,0)", fill=(255,0,255))

table_pt = np.array([0.06, -0.02, TABLE_Z])
tx, ty = project(view, proj, table_pt)
d.line([project(view, proj, [-0.02, -0.02, TABLE_Z]), project(view, proj, [0.08, -0.02, TABLE_Z])], fill=(255, 140, 0), width=3)
d.ellipse([tx-7, ty-7, tx+7, ty+7], fill=(255, 140, 0), outline=(0,0,0), width=2)
d.text((tx+12, ty+6), f"table surface z={TABLE_Z:.5f}\n(2.4mm below origin)", fill=(255,140,0))

out_path = OUT_DIR / "base_origin_with_table_line.png"
im.save(out_path)
print("saved", out_path)
p.disconnect()
