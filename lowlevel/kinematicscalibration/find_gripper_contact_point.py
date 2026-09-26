#!/usr/bin/env python3
"""Find the lowest world-Z point of the gripper assembly for saved poses.

Uses PyBullet's actual (posed) collision-mesh AABB per link -- not the
gripper_frame_link FK origin -- to find where the gripper really touches
down, for each of a set of recorded joint readings.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
URDF_PATH = str(paths.URDF_PATH)
CONTROL_JOINT_INDICES = [0, 1, 2, 3, 4, 6]  # shoulder_pan..wrist_roll, gripper
LOWLEVEL_DIR = Path(__file__).resolve().parent.parent

GRIPPER_ASSEMBLY_LINKS = (
    "wrist_link",
    "gripper_link",
    "gripper_frame_link",
    "moving_jaw_so101_v1_link",
)


def main() -> int:
    paths = [Path(p_) for p_ in sys.argv[1:]] or sorted(
        (LOWLEVEL_DIR / "real_world_runs").glob("current_pose*.json")
    )

    p.connect(p.DIRECT)
    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)

    link_index_by_name = {}
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        link_index_by_name[info[12].decode("utf-8")] = j
    target_links = {
        name: link_index_by_name[name]
        for name in GRIPPER_ASSEMBLY_LINKS
        if name in link_index_by_name
    }

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        joints_deg = np.array(data["joints_deg"], dtype=float)
        joints_rad = np.deg2rad(joints_deg)
        for sim_idx, q in zip(CONTROL_JOINT_INDICES, joints_rad):
            p.resetJointState(robot_id, sim_idx, float(q))

        min_z = None
        min_link = None
        min_xyz = None
        for name, link_idx in target_links.items():
            aabb_min, aabb_max = p.getAABB(robot_id, link_idx)
            if min_z is None or aabb_min[2] < min_z:
                min_z = aabb_min[2]
                min_link = name
                min_xyz = aabb_min

        gf_state = p.getLinkState(
            robot_id, link_index_by_name["gripper_frame_link"], computeForwardKinematics=True
        )
        gf_pos = gf_state[4]  # world position of the link (URDF) frame

        print(
            f"{path.name:24s} lowest_z={min_z:.5f} at {min_link:28s} "
            f"xyz=[{min_xyz[0]:.5f}, {min_xyz[1]:.5f}, {min_xyz[2]:.5f}]  "
            f"gripper_frame_link_z={gf_pos[2]:.5f}  "
            f"offset_z={gf_pos[2] - min_z:.5f}"
        )

    p.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
