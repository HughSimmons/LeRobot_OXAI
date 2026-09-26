#!/usr/bin/env python3
"""Render the base plate in isolation with the robot's kinematic origin
(world (0,0,0), the base_link frame) marked clearly."""

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

p.connect(p.DIRECT)
robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)

# Hide every link except base_link (index -1) so only the base plate remains visible.
num_joints = p.getNumJoints(robot_id)
for link_idx in range(num_joints):
    p.changeVisualShape(robot_id, link_idx, rgbaColor=[0, 0, 0, 0])


def project(view, proj, pt):
    c = np.array(proj).reshape(4, 4, order="F") @ np.array(view).reshape(4, 4, order="F") @ np.array([*pt, 1.0])
    n = c[:3] / c[3]
    return ((n[0] * 0.5 + 0.5) * W, (1 - (n[1] * 0.5 + 0.5)) * H)


def render(out_name, cam_target, cam_dist, cam_yaw, cam_pitch, axis_len=0.03):
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_target, distance=cam_dist,
        yaw=cam_yaw, pitch=cam_pitch, roll=0, upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=W / H, nearVal=0.01, farVal=3.0)
    img = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER)[2]
    arr = np.reshape(img, (H, W, 4))[:, :, :3].astype(np.uint8)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)

    origin = np.array([0.0, 0.0, 0.0])

    # axis lines through the origin
    def line3(p0, p1, color, width=3):
        d.line([project(view, proj, p0), project(view, proj, p1)], fill=color, width=width)

    line3(origin, origin + [axis_len, 0, 0], (255, 40, 40))   # +X red
    line3(origin, origin + [0, axis_len, 0], (0, 170, 0))     # +Y green
    line3(origin, origin + [0, 0, axis_len], (40, 90, 255))   # +Z blue

    x, y = project(view, proj, origin)
    r = 9
    d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 255), outline=(0, 0, 0), width=2)
    label = "robot origin (0,0,0)\nbase_link frame"
    d.text((x + 14, y - 10), label, fill=(255, 0, 255))
    d.text((x + 13, y - 11), label, fill=(255, 255, 255))

    out_path = OUT_DIR / out_name
    im.save(out_path)
    print(f"saved {out_path}")


OUT_DIR.mkdir(parents=True, exist_ok=True)
render("base_origin_topdown.png", cam_target=[0.0, 0.0, 0.01], cam_dist=0.18, cam_yaw=0, cam_pitch=-89.9)
render("base_origin_side.png", cam_target=[0.0, 0.0, 0.02], cam_dist=0.18, cam_yaw=0, cam_pitch=-10)
render("base_origin_rear.png", cam_target=[0.0, 0.0, 0.02], cam_dist=0.18, cam_yaw=180, cam_pitch=-15)
p.disconnect()
