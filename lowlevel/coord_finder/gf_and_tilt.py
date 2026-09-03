import json
from pathlib import Path
import numpy as np
import pybullet as p

URDF_PATH = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
CONTROL_JOINT_INDICES = [0, 1, 2, 3, 4, 6]
LOWLEVEL_DIR = Path(__file__).resolve().parent.parent

p.connect(p.DIRECT)
robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
link_index_by_name = {}
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    link_index_by_name[info[12].decode("utf-8")] = j
gf_idx = link_index_by_name["gripper_frame_link"]

paths = sorted((LOWLEVEL_DIR / "real_world_runs").glob("vertical_reading*.json"))
for path in paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    joints_deg = np.array(data["joints_deg"], dtype=float)
    joints_rad = np.deg2rad(joints_deg)
    for sim_idx, q in zip(CONTROL_JOINT_INDICES, joints_rad):
        p.resetJointState(robot_id, sim_idx, float(q))
    state = p.getLinkState(robot_id, gf_idx, computeForwardKinematics=True)
    pos, orn = np.array(state[4]), state[5]
    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    local_z_world = rot @ np.array([0, 0, 1.0])
    tilt_from_down = 180.0 - np.degrees(np.arccos(np.clip(local_z_world[2], -1, 1)))
    print(f"{path.name:24s} gripper_frame_link xyz=[{pos[0]:+.5f}, {pos[1]:+.5f}, {pos[2]:+.5f}]  tilt_from_vertical={tilt_from_down:.2f}deg")
p.disconnect()
