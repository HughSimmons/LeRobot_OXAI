from pathlib import Path
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

URDF_PATH = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
OUT_DIR = Path(__file__).resolve().parent / "output" / "urdf_visualization"
W, H = 1280, 960

p.connect(p.DIRECT)
robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
for link_idx in range(p.getNumJoints(robot_id)):
    p.changeVisualShape(robot_id, link_idx, rgbaColor=[0, 0, 0, 0])

def project(view, proj, pt):
    c = np.array(proj).reshape(4, 4, order="F") @ np.array(view).reshape(4, 4, order="F") @ np.array([*pt, 1.0])
    n = c[:3] / c[3]
    return ((n[0] * 0.5 + 0.5) * W, (1 - (n[1] * 0.5 + 0.5)) * H)

view = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.02, 0.0, 0.0], distance=0.16, yaw=0, pitch=0, roll=0, upAxisIndex=2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=W / H, nearVal=0.01, farVal=3.0)
img = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER)[2]
arr = np.reshape(img, (H, W, 4))[:, :, :3].astype(np.uint8)
im = Image.fromarray(arr)
d = ImageDraw.Draw(im)

origin = np.array([0.0, 0.0, 0.0])
ox, oy = project(view, proj, origin)
d.ellipse([ox-8, oy-8, ox+8, oy+8], fill=(255,0,255), outline=(0,0,0), width=2)
d.text((ox+12, oy-10), "origin z=0", fill=(255,0,255))

lowest = np.array([0.06434536, -0.02419871, -0.00240026])
lx, ly = project(view, proj, lowest)
d.ellipse([lx-8, ly-8, lx+8, ly+8], fill=(255,140,0), outline=(0,0,0), width=2)
d.text((lx+12, ly-10), f"computed lowest pt of\nbase_so101_v2.stl\nz={lowest[2]:.5f}", fill=(255,140,0))

out_path = OUT_DIR / "base_only_full_view.png"
im.save(out_path)
print("saved", out_path)
p.disconnect()
