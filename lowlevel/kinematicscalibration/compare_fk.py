import json
import sys
from pathlib import Path
import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from testkinematics import kinematics
import paths
URDF_PATH = str(paths.URDF_PATH)
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

    # testkinematics
    pose = kinematics.forward_kinematics(joints_deg)
    tk_xyz = pose[:3, 3]

    # pybullet
    joints_rad = np.deg2rad(joints_deg)
    for sim_idx, q in zip(CONTROL_JOINT_INDICES, joints_rad):
        p.resetJointState(robot_id, sim_idx, float(q))
    state = p.getLinkState(robot_id, gf_idx, computeForwardKinematics=True)
    pb_xyz = np.array(state[4])

    diff_mm = (tk_xyz - pb_xyz) * 1000.0
    print(f"{path.name:24s} testkinematics={np.round(tk_xyz,5)}  pybullet={np.round(pb_xyz,5)}  diff_mm={np.round(diff_mm,3)}")
p.disconnect()
