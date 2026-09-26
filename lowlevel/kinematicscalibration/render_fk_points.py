#!/usr/bin/env python3
"""Render the robot at a recorded pose and mark the gripper_frame_link
origin (reported FK point) and the exact jaw-tip contact point on top."""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
URDF_PATH = str(paths.URDF_PATH)
ASSETS_DIR = Path(URDF_PATH).parent / "assets"
CONTROL_JOINT_INDICES = [0, 1, 2, 3, 4, 6]
LOWLEVEL_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "output" / "urdf_visualization"


def read_binary_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    tri_count = struct.unpack_from("<I", data, 80)[0]
    verts = np.empty((tri_count * 3, 3), dtype=np.float64)
    offset = 84
    for t in range(tri_count):
        for v in range(3):
            x, y, z = struct.unpack_from("<fff", data, offset + 12 + v * 12)
            verts[t * 3 + v] = (x, y, z)
        offset += 50
    return verts


jaw_verts = read_binary_stl_vertices(ASSETS_DIR / "moving_jaw_so101_v1.stl")
jaw_verts_link = jaw_verts + np.array([0.0, 0.0, 0.0189])

p.connect(p.DIRECT)
robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
link_index_by_name = {}
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    link_index_by_name[info[12].decode("utf-8")] = j
gf_idx = link_index_by_name["gripper_frame_link"]
jaw_idx = link_index_by_name["moving_jaw_so101_v1_link"]

W, H = 1280, 960


def project(view, proj, pt):
    c = np.array(proj).reshape(4, 4, order="F") @ np.array(view).reshape(4, 4, order="F") @ np.array([*pt, 1.0])
    n = c[:3] / c[3]
    return ((n[0] * 0.5 + 0.5) * W, (1 - (n[1] * 0.5 + 0.5)) * H)


def render_pose(pose_path: Path, out_name: str, cam_target, cam_dist, cam_yaw, cam_pitch):
    data = json.loads(pose_path.read_text(encoding="utf-8"))
    joints_deg = np.array(data["joints_deg"], dtype=float)
    joints_rad = np.deg2rad(joints_deg)
    for sim_idx, q in zip(CONTROL_JOINT_INDICES, joints_rad):
        p.resetJointState(robot_id, sim_idx, float(q))

    gf_state = p.getLinkState(robot_id, gf_idx, computeForwardKinematics=True)
    gf_pos = np.array(gf_state[4])

    jaw_state = p.getLinkState(robot_id, jaw_idx, computeForwardKinematics=True)
    jaw_pos, jaw_orn = np.array(jaw_state[4]), jaw_state[5]
    jaw_rot = np.array(p.getMatrixFromQuaternion(jaw_orn)).reshape(3, 3)
    world_pts = (jaw_rot @ jaw_verts_link.T).T + jaw_pos
    tip_world = world_pts[world_pts[:, 2].argmin()]

    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_target, distance=cam_dist,
        yaw=cam_yaw, pitch=cam_pitch, roll=0, upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=W / H, nearVal=0.01, farVal=3.0)
    img = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER)[2]
    arr = np.reshape(img, (H, W, 4))[:, :, :3].astype(np.uint8)

    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)

    def dot(pt, color, r=8, label=None, dx=14, dy=-8):
        x, y = project(view, proj, pt)
        d.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(0, 0, 0), width=2)
        if label:
            d.text((x + dx, y + dy), label, fill=color)
            d.text((x + dx - 1, y + dy - 1), label, fill=(255, 255, 255))

    dot(gf_pos, (255, 0, 255), r=9, label=f"gripper_frame_link\n[{gf_pos[0]:.4f},{gf_pos[1]:.4f},{gf_pos[2]:.4f}]")
    dot(tip_world, (0, 220, 255), r=7, label=f"jaw-tip contact\n[{tip_world[0]:.4f},{tip_world[1]:.4f},{tip_world[2]:.4f}]", dy=14)

    out_path = OUT_DIR / out_name
    im.save(out_path)
    print(f"saved {out_path}")


OUT_DIR.mkdir(parents=True, exist_ok=True)
render_pose(
    LOWLEVEL_DIR / "real_world_runs" / "vertical_reading_1.json",
    "vertical_reading_1_fk_points.png",
    cam_target=[0.17, -0.10, 0.02], cam_dist=0.35, cam_yaw=50, cam_pitch=-20,
)
render_pose(
    LOWLEVEL_DIR / "real_world_runs" / "vertical_reading_3.json",
    "vertical_reading_3_fk_points.png",
    cam_target=[0.19, -0.02, 0.02], cam_dist=0.35, cam_yaw=50, cam_pitch=-20,
)
p.disconnect()
