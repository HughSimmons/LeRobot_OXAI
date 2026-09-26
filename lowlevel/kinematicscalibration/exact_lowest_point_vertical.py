#!/usr/bin/env python3
"""Exact (no PyBullet collision margin) lowest-point ground truth for the
gripper assembly, per recorded pose, using raw STL vertices transformed by
FK through the real URDF link + mesh-origin chain."""

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


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def mesh_to_link_frame(stl_name: str, xyz, rpy) -> np.ndarray:
    verts = read_binary_stl_vertices(ASSETS_DIR / stl_name)
    R = rpy_to_matrix(*rpy)
    t = np.array(xyz)
    return (R @ verts.T).T + t


# (link_name, [(stl_file, xyz, rpy), ...]) -- from so101_new_calib.urdf
LINK_MESH_PARTS = {
    "wrist_link": [
        ("sts3215_03a_no_horn_v1.stl", (8.32667e-17, -0.0424, 0.0306), (1.5708, 1.5708, 0)),
        ("wrist_roll_pitch_so101_v2.stl", (0, -0.028, 0.0181), (-1.5708, -1.5708, 0)),
    ],
    "gripper_link": [
        ("sts3215_03a_v1.stl", (0.0077, 0.0001, -0.0234), (-1.5708, -5.19179e-17, -1.66533e-16)),
        ("wrist_roll_follower_so101_v1.stl", (8.32667e-17, -0.000218214, 0.000949706), (-3.14159, -5.55112e-17, 0)),
    ],
    "moving_jaw_so101_v1_link": [
        ("moving_jaw_so101_v1.stl", (-5.55112e-17, -5.55112e-17, 0.0189), (9.53145e-17, 6.93889e-18, 1.24077e-24)),
    ],
}

link_local_points: dict[str, list[tuple[str, np.ndarray]]] = {}
for link_name, parts in LINK_MESH_PARTS.items():
    entries = []
    for stl_name, xyz, rpy in parts:
        verts_link = mesh_to_link_frame(stl_name, xyz, rpy)
        entries.append((stl_name, verts_link))
    link_local_points[link_name] = entries

p.connect(p.DIRECT)
robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
link_index_by_name = {}
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    link_index_by_name[info[12].decode("utf-8")] = j

paths = sorted((LOWLEVEL_DIR / "real_world_runs").glob("vertical_reading*.json"))
for path in paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    joints_deg = np.array(data["joints_deg"], dtype=float)
    joints_rad = np.deg2rad(joints_deg)
    for sim_idx, q in zip(CONTROL_JOINT_INDICES, joints_rad):
        p.resetJointState(robot_id, sim_idx, float(q))

    best_z = None
    best_desc = None
    best_world = None
    for link_name, parts in link_local_points.items():
        link_idx = link_index_by_name[link_name]
        state = p.getLinkState(robot_id, link_idx, computeForwardKinematics=True)
        pos, orn = np.array(state[4]), state[5]
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        for stl_name, verts_link in parts:
            world_pts = (rot @ verts_link.T).T + pos
            i = world_pts[:, 2].argmin()
            z = world_pts[i, 2]
            if best_z is None or z < best_z:
                best_z = z
                best_desc = f"{link_name}/{stl_name}"
                best_world = world_pts[i]

    print(
        f"{path.name:24s} exact_lowest_z={best_z:+.5f}  at {best_desc:45s}  "
        f"world=[{best_world[0]:+.5f}, {best_world[1]:+.5f}, {best_world[2]:+.5f}]"
    )

p.disconnect()
