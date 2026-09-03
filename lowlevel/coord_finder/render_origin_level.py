from pathlib import Path
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

URDF_PATH = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
OUT_DIR = Path(__file__).resolve().parent / "output" / "urdf_visualization"
W, H = 1280, 960
TABLE_Z = -0.00240

p.connect(p.DIRECT)
robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
for link_idx in range(p.getNumJoints(robot_id)):
    p.changeVisualShape(robot_id, link_idx, rgbaColor=[0, 0, 0, 0])

def project(view, proj, pt):
    c = np.array(proj).reshape(4, 4, order="F") @ np.array(view).reshape(4, 4, order="F") @ np.array([*pt, 1.0])
    n = c[:3] / c[3]
    return ((n[0] * 0.5 + 0.5) * W, (1 - (n[1] * 0.5 + 0.5)) * H)

# pitch=0, so table-line endpoints and origin (all at world y=0) share depth
# along the camera's optical axis -> screen_y depends only on world z.
view = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.03, 0.0, 0.02], distance=0.18, yaw=0, pitch=0, roll=0, upAxisIndex=2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=W / H, nearVal=0.01, farVal=3.0)
img = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER)[2]
arr = np.reshape(img, (H, W, 4))[:, :, :3].astype(np.uint8)
im = Image.fromarray(arr)
d = ImageDraw.Draw(im)

origin = np.array([0.0, 0.0, 0.0])
d.line([project(view, proj, origin - [0.02,0,0]), project(view, proj, origin + [0.09,0,0])], fill=(255,40,40), width=2)
ox, oy = project(view, proj, origin)
d.ellipse([ox-8, oy-8, ox+8, oy+8], fill=(255,0,255), outline=(0,0,0), width=2)
d.text((ox+12, oy-24), "robot origin z=0", fill=(255,0,255))

d.line([project(view, proj, [-0.02, 0, TABLE_Z]), project(view, proj, [0.09, 0, TABLE_Z])], fill=(255,140,0), width=2)
tx, ty = project(view, proj, [0.07, 0, TABLE_Z])
d.text((tx, ty+8), f"table z={TABLE_Z:.5f} (2.4mm below origin)", fill=(255,140,0))

out_path = OUT_DIR / "base_origin_level_side.png"
im.save(out_path)
print("saved", out_path)
p.disconnect()
