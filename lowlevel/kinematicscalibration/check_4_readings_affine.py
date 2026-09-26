#!/usr/bin/env python3
import sys
import json
import struct
from pathlib import Path
import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
URDF_PATH = str(paths.URDF_PATH)
ASSETS_DIR = Path(URDF_PATH).parent / "assets"
CONTROL_JOINT_INDICES = [0, 1, 2, 3, 4, 6]
LOWLEVEL_DIR = Path(__file__).resolve().parent.parent


def read_binary_stl_vertices(path):
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
jaw_verts_link = jaw_verts + np.array([0.0, 0.0, 0.0189])  # mesh -> link-local frame

p.connect(p.DIRECT)
robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
link_index_by_name = {}
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    link_index_by_name[info[12].decode("utf-8")] = j
moving_jaw_idx = link_index_by_name["moving_jaw_so101_v1_link"]
gripper_link_idx = link_index_by_name["gripper_link"]

# First: find the jaw's local vertex that is lowest specifically in the
# near-vertical regime (already established as the true contact vertex there).
# Reuse pose 1 (near corner, only 7.7 deg tilt, jaw confirmed as lowest link).
paths = sorted((LOWLEVEL_DIR / "real_world_runs").glob("current_pose*.json"))
ref = json.loads(paths[0].read_text(encoding="utf-8"))
ref_joints_rad = np.deg2rad(np.array(ref["joints_deg"], dtype=float))
for sim_idx, q in zip(CONTROL_JOINT_INDICES, ref_joints_rad):
    p.resetJointState(robot_id, sim_idx, float(q))
jaw_state = p.getLinkState(robot_id, moving_jaw_idx, computeForwardKinematics=True)
jaw_pos, jaw_orn = np.array(jaw_state[4]), jaw_state[5]
jaw_rot = np.array(p.getMatrixFromQuaternion(jaw_orn)).reshape(3, 3)
world_pts = (jaw_rot @ jaw_verts_link.T).T + jaw_pos
contact_vertex_idx = world_pts[:, 2].argmin()
contact_vertex_local = jaw_verts_link[contact_vertex_idx]
print(f"identified contact vertex (jaw-local frame): {contact_vertex_local}")

print()
for path in paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    joints_deg = np.array(data["joints_deg"], dtype=float)
    joints_rad = np.deg2rad(joints_deg)
    for sim_idx, q in zip(CONTROL_JOINT_INDICES, joints_rad):
        p.resetJointState(robot_id, sim_idx, float(q))

    jaw_state = p.getLinkState(robot_id, moving_jaw_idx, computeForwardKinematics=True)
    jaw_pos, jaw_orn = np.array(jaw_state[4]), jaw_state[5]
    jaw_rot = np.array(p.getMatrixFromQuaternion(jaw_orn)).reshape(3, 3)
    predicted_world = jaw_rot @ contact_vertex_local + jaw_pos

    jaw_aabb_min, _ = p.getAABB(robot_id, moving_jaw_idx)
    grip_aabb_min, _ = p.getAABB(robot_id, gripper_link_idx)
    true_lowest_z = min(jaw_aabb_min[2], grip_aabb_min[2])
    lowest_link = "moving_jaw" if jaw_aabb_min[2] < grip_aabb_min[2] else "gripper_link"

    print(
        f"{path.name:24s} tip_xyz=[{predicted_world[0]:+.5f}, {predicted_world[1]:+.5f}, {predicted_world[2]:+.5f}]"
    )
p.disconnect()
