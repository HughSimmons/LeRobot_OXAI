#!/usr/bin/env python3
"""Same contact-point identity sweep, restricted to near-exact-vertical wrist
poses (as probe_xy_pose.py actually produces) with only XY and wrist_roll
varying -- the realistic calibration-probe orientation family."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from probe_xy_pose import solve_down_pose, JOINT_LIMITS_DEG  # noqa: E402
import paths
URDF_PATH = str(paths.URDF_PATH)
CONTROL_JOINT_INDICES = [0, 1, 2, 3, 4, 6]
GRIPPER_ASSEMBLY_LINKS = ("wrist_link", "gripper_link", "moving_jaw_so101_v1_link")


def main() -> int:
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    rng = np.random.default_rng(1)

    p.connect(p.DIRECT)
    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
    link_index_by_name = {}
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        link_index_by_name[info[12].decode("utf-8")] = j
    target_links = {name: link_index_by_name[name] for name in GRIPPER_ASSEMBLY_LINKS}
    gripper_frame_idx = link_index_by_name["gripper_frame_link"]

    identity_counter: Counter[str] = Counter()
    local_offsets: dict[str, list[np.ndarray]] = {}
    solved = 0

    for _ in range(n_samples):
        x = rng.uniform(0.15, 0.35)
        y = rng.uniform(-0.15, 0.15)
        z = rng.uniform(0.02, 0.06)
        target = np.array([x, y, z])
        try:
            joints5, achieved, tilt = solve_down_pose(target, restarts=6)
        except Exception:
            continue
        if tilt > 1.0 or np.linalg.norm(achieved - target) > 0.005:
            continue
        solved += 1
        joints_deg = np.concatenate([joints5, [10.0]])
        joints_rad = np.deg2rad(joints_deg)
        for sim_idx, q in zip(CONTROL_JOINT_INDICES, joints_rad):
            p.resetJointState(robot_id, sim_idx, float(q))

        gf_state = p.getLinkState(robot_id, gripper_frame_idx, computeForwardKinematics=True)
        gf_pos, gf_orn = np.array(gf_state[4]), gf_state[5]
        gf_rot = np.array(p.getMatrixFromQuaternion(gf_orn)).reshape(3, 3)

        min_z = None
        min_link = None
        min_pt_world = None
        for name, link_idx in target_links.items():
            aabb_min, _ = p.getAABB(robot_id, link_idx)
            if min_z is None or aabb_min[2] < min_z:
                min_z = aabb_min[2]
                min_link = name
                min_pt_world = np.array(aabb_min)

        identity_counter[min_link] += 1
        local_offset = gf_rot.T @ (min_pt_world - gf_pos)
        local_offsets.setdefault(min_link, []).append(local_offset)

    print(f"solved {solved}/{n_samples} near-vertical (<1deg tilt) poses")
    for name, count in identity_counter.most_common():
        offs = np.array(local_offsets[name])
        mean = offs.mean(axis=0)
        spread_mm = (offs.max(axis=0) - offs.min(axis=0)) * 1000.0
        print(
            f"  {name:28s} {count:4d}/{solved} ({100*count/solved:5.1f}%)  "
            f"local_offset_mean={np.round(mean, 4).tolist()}  "
            f"spread_mm={np.round(spread_mm, 2).tolist()}"
        )
    p.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
