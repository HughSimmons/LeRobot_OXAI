import json, struct
from pathlib import Path
import numpy as np
import pybullet as p

URDF_PATH = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
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
print("raw jaw mesh vertex count:", len(jaw_verts))
print("raw jaw mesh z range:", jaw_verts[:,2].min(), jaw_verts[:,2].max())
jaw_verts_link = jaw_verts + np.array([0.0, 0.0, 0.0189])

p.connect(p.DIRECT)
robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
link_index_by_name = {}
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    link_index_by_name[info[12].decode("utf-8")] = j
moving_jaw_idx = link_index_by_name["moving_jaw_so101_v1_link"]

paths = sorted((LOWLEVEL_DIR / "real_world_runs").glob("current_pose*.json"))
ref = json.loads(paths[0].read_text(encoding="utf-8"))
print("ref pose file:", paths[0].name, "joints_deg:", ref["joints_deg"])
ref_joints_rad = np.deg2rad(np.array(ref["joints_deg"], dtype=float))
for sim_idx, q in zip(CONTROL_JOINT_INDICES, ref_joints_rad):
    p.resetJointState(robot_id, sim_idx, float(q))

jaw_state = p.getLinkState(robot_id, moving_jaw_idx, computeForwardKinematics=True)
jaw_pos, jaw_orn = np.array(jaw_state[4]), jaw_state[5]
jaw_rot = np.array(p.getMatrixFromQuaternion(jaw_orn)).reshape(3, 3)
print("jaw_pos:", jaw_pos)
world_pts = (jaw_rot @ jaw_verts_link.T).T + jaw_pos
print("world_pts z min/max:", world_pts[:,2].min(), world_pts[:,2].max())
idx = world_pts[:, 2].argmin()
print("min z vertex world:", world_pts[idx])
print("min z vertex local (link frame):", jaw_verts_link[idx])

aabb_min, aabb_max = p.getAABB(robot_id, moving_jaw_idx)
print("pybullet AABB min:", aabb_min)
p.disconnect()
