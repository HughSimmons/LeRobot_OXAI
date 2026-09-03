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

# base_link (-1) has 4 visual shapes in this order:
# 0 base_motor_holder_so101_v1.stl, 1 base_so101_v2.stl, 2 sts3215_03a_v1.stl,
# 3 waveshare_mounting_plate_so101_v2.stl -- keep only #1 (the plate itself).
for shape_idx in (0, 2, 3):
    p.changeVisualShape(robot_id, -1, shapeIndex=shape_idx, rgbaColor=[0, 0, 0, 0])

def project(view, proj, pt):
    c = np.array(proj).reshape(4, 4, order="F") @ np.array(view).reshape(4, 4, order="F") @ np.array([*pt, 1.0])
    n = c[:3] / c[3]
    return ((n[0] * 0.5 + 0.5) * W, (1 - (n[1] * 0.5 + 0.5)) * H)

def render(out_name, target, dist, yaw, pitch, axis_len=0.03):
    view = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=target, distance=dist, yaw=yaw, pitch=pitch, roll=0, upAxisIndex=2)
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=W / H, nearVal=0.01, farVal=3.0)
    img = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER)[2]
    arr = np.reshape(img, (H, W, 4))[:, :, :3].astype(np.uint8)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)

    origin = np.array([0.0, 0.0, 0.0])
    d.line([project(view, proj, origin), project(view, proj, origin + [axis_len, 0, 0])], fill=(255,40,40), width=3)
    d.line([project(view, proj, origin), project(view, proj, origin + [0, axis_len, 0])], fill=(0,170,0), width=3)
    d.line([project(view, proj, origin), project(view, proj, origin + [0, 0, axis_len])], fill=(40,90,255), width=3)
    ox, oy = project(view, proj, origin)
    d.ellipse([ox-9, oy-9, ox+9, oy+9], fill=(255,0,255), outline=(0,0,0), width=2)
    label = "robot origin (0,0,0)\nURDF base_link frame"
    d.text((ox+14, oy-10), label, fill=(255,0,255))
    d.text((ox+13, oy-11), label, fill=(255,255,255))

    out_path = OUT_DIR / out_name
    im.save(out_path)
    print("saved", out_path)

render("base_plate_isolated_side_origin.png", target=[0.02, 0.0, 0.0], dist=0.16, yaw=0, pitch=0)
render("base_plate_isolated_topdown_origin.png", target=[0.02, 0.0, 0.01], dist=0.16, yaw=0, pitch=-89.9)
render("base_plate_isolated_front_origin.png", target=[0.02, 0.0, 0.0], dist=0.16, yaw=90, pitch=0)
p.disconnect()
