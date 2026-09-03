#!/usr/bin/env python3
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
moving_jaw_idx = link_index_by_name["moving_jaw_so101_v1_link"]
gripper_link_idx = link_index_by_name["gripper_link"]
gf_idx = link_index_by_name["gripper_frame_link"]

paths = sorted((LOWLEVEL_DIR / "real_world_runs").glob("current_pose*.json"))
for path in paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    joints_deg = np.array(data["joints_deg"], dtype=float)
    joints_rad = np.deg2rad(joints_deg)
    for sim_idx, q in zip(CONTROL_JOINT_INDICES, joints_rad):
        p.resetJointState(robot_id, sim_idx, float(q))

    gf_state = p.getLinkState(robot_id, gf_idx, computeForwardKinematics=True)
    gf_pos, gf_orn = np.array(gf_state[4]), gf_state[5]
    gf_rot = np.array(p.getMatrixFromQuaternion(gf_orn)).reshape(3, 3)
    local_z_world = gf_rot @ np.array([0, 0, 1.0])
    tilt_deg = np.degrees(np.arccos(np.clip(local_z_world[2], -1, 1)))

    jaw_aabb_min, _ = p.getAABB(robot_id, moving_jaw_idx)
    grip_aabb_min, _ = p.getAABB(robot_id, gripper_link_idx)
    true_lowest_z = min(jaw_aabb_min[2], grip_aabb_min[2])
    lowest_link = "moving_jaw" if jaw_aabb_min[2] < grip_aabb_min[2] else "gripper_link"

    predicted_z = gf_pos[2] + 0.0112 * local_z_world[2]  # vertical-model prediction along approach axis

    print(
        f"{path.name:24s} gf_z={gf_pos[2]:+.5f}  tilt={tilt_deg:6.2f}deg  "
        f"true_lowest_z={true_lowest_z:+.5f} ({lowest_link})  "
        f"vertical_model_pred={predicted_z:+.5f}  "
        f"pred_err={predicted_z - true_lowest_z:+.5f}"
    )
p.disconnect()
