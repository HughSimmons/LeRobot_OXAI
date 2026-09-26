#!/usr/bin/env python3
"""Sweep many simulated orientations to see whether the gripper's lowest
point (the one that would touch a flat surface below) is a single fixed
local point, or changes identity with orientation.

Exact (noise-free) since it's pure simulation: for each sampled joint
configuration, the true lowest point of the gripper assembly mesh is found
via PyBullet's per-link AABB (tight to the transformed mesh), then expressed
as a local offset from gripper_frame_link. If contact identity is orientation
dependent, this shows up as multiple frequently-occurring links/offsets
rather than one dominant point.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
URDF_PATH = str(paths.URDF_PATH)
CONTROL_JOINT_INDICES = [0, 1, 2, 3, 4, 6]  # shoulder_pan..wrist_roll, gripper

# Same effective limits as probe_xy_pose.py (URDF limits widened to admit
# DEFAULT_HOME), restricted here to configurations plausible for a downward
# touch: shoulder_lift/elbow_flex/wrist_flex ranges biased toward "gripper
# pointing down-ish", pan/roll swept broadly.
SAMPLE_RANGES_DEG = {
    "shoulder_pan": (-90.0, 90.0),
    "shoulder_lift": (-20.0, 70.0),
    "elbow_flex": (-90.0, 10.0),
    "wrist_flex": (20.0, 95.0),
    "wrist_roll": (-157.0, 162.0),
    "gripper": (0.0, 20.0),
}
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

GRIPPER_ASSEMBLY_LINKS = ("wrist_link", "gripper_link", "moving_jaw_so101_v1_link")


def main() -> int:
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    rng = np.random.default_rng(0)

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
    kept = 0

    for _ in range(n_samples):
        joints_deg = np.array(
            [rng.uniform(*SAMPLE_RANGES_DEG[name]) for name in JOINT_ORDER], dtype=float
        )
        joints_rad = np.deg2rad(joints_deg)
        for sim_idx, q in zip(CONTROL_JOINT_INDICES, joints_rad):
            p.resetJointState(robot_id, sim_idx, float(q))

        # keep only configs where the gripper z-axis points reasonably downward
        gf_state = p.getLinkState(robot_id, gripper_frame_idx, computeForwardKinematics=True)
        gf_pos, gf_orn = np.array(gf_state[4]), gf_state[5]
        gf_rot = np.array(p.getMatrixFromQuaternion(gf_orn)).reshape(3, 3)
        local_z_axis_world = gf_rot @ np.array([0.0, 0.0, 1.0])
        if local_z_axis_world[2] > -0.5:  # require pointing at least ~60deg downward
            continue
        kept += 1

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

    print(f"kept {kept}/{n_samples} samples with gripper pointing downward")
    print("lowest-point identity frequency:")
    for name, count in identity_counter.most_common():
        offs = np.array(local_offsets[name])
        mean = offs.mean(axis=0)
        spread_mm = (offs.max(axis=0) - offs.min(axis=0)) * 1000.0
        print(
            f"  {name:28s} {count:4d}/{kept} ({100*count/kept:5.1f}%)  "
            f"local_offset_mean={np.round(mean, 4).tolist()}  "
            f"spread_mm={np.round(spread_mm, 2).tolist()}"
        )

    p.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
