import json
import struct
from pathlib import Path
import numpy as np
import pybullet as p

URDF_PATH = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
ASSETS_DIR = Path(URDF_PATH).parent / "assets"
CONTROL_JOINT_INDICES = [0, 1, 2, 3, 4, 6]
LOWLEVEL_DIR = Path(__file__).resolve().parent.parent
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

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
jaw_verts_link = jaw_verts + np.array([0.0, 0.0, 0.0189])

p.connect(p.DIRECT)
robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
link_index_by_name = {}
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    link_index_by_name[info[12].decode("utf-8")] = j
jaw_idx = link_index_by_name["moving_jaw_so101_v1_link"]

def jaw_tip_z(joints_deg):
    joints_rad = np.deg2rad(joints_deg)
    for sim_idx, q in zip(CONTROL_JOINT_INDICES, joints_rad):
        p.resetJointState(robot_id, sim_idx, float(q))
    state = p.getLinkState(robot_id, jaw_idx, computeForwardKinematics=True)
    pos, orn = np.array(state[4]), state[5]
    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    world_pts = (rot @ jaw_verts_link.T).T + pos
    return world_pts[:, 2].min()

paths = sorted((LOWLEVEL_DIR / "real_world_runs").glob("vertical_reading*.json"))
for path in paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    base_joints = np.array(data["joints_deg"], dtype=float)
    base_z = jaw_tip_z(base_joints)
    sens = {}
    for i, name in enumerate(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]):
        j = base_joints.copy()
        j[i] += 1.0
        z_plus = jaw_tip_z(j)
        sens[name] = (z_plus - base_z) * 1000.0  # mm per deg
    print(f"{path.name:24s} base_z={base_z*1000:.2f}mm  sensitivity(mm/deg): " +
          "  ".join(f"{k}={v:+.3f}" for k, v in sens.items()))
p.disconnect()
