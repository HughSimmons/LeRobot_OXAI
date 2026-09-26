#!/usr/bin/env python3
"""Identify the gripper's physical board-contact point.

Approach: the four recorded hand-positioned poses were each resting a jaw
tip on the board surface, at different overall arm orientations. Take a set
of candidate points (STL extreme vertices) on the jaw meshes, express each
as a fixed local offset from gripper_frame_link, then use forward kinematics
to see where each candidate lands in world Z for all four poses. The true
contact point should land at ~constant world Z across all four (since it was
physically touching the same board-top plane each time); any other point on
the rigid gripper body will vary with orientation. Lowest z-spread wins.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
URDF_PATH = str(paths.URDF_PATH)
ASSETS_DIR = Path(URDF_PATH).parent / "assets"
CONTROL_JOINT_INDICES = [0, 1, 2, 3, 4, 6]  # shoulder_pan..wrist_roll, gripper
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


def candidate_points(verts: np.ndarray) -> dict[str, np.ndarray]:
    idx = {
        "xmin": verts[:, 0].argmin(), "xmax": verts[:, 0].argmax(),
        "ymin": verts[:, 1].argmin(), "ymax": verts[:, 1].argmax(),
        "zmin": verts[:, 2].argmin(), "zmax": verts[:, 2].argmax(),
    }
    return {name: verts[i] for name, i in idx.items()}


def main() -> int:
    paths = [Path(p_) for p_ in sys.argv[1:]] or sorted(
        (LOWLEVEL_DIR / "real_world_runs").glob("current_pose*.json")
    )

    # moving_jaw_so101_v1_link: mesh origin xyz=(~0,~0,0.0189), rpy~identity
    jaw_verts = read_binary_stl_vertices(ASSETS_DIR / "moving_jaw_so101_v1.stl")
    jaw_verts_link = jaw_verts + np.array([0.0, 0.0, 0.0189])
    jaw_candidates = {
        f"moving_jaw:{name}": pt for name, pt in candidate_points(jaw_verts_link).items()
    }

    # gripper_link's wrist_roll_follower part (fixed jaw body): mesh origin
    # xyz~(0,-0.000218,0.00095), rpy~(pi,0,0) -> local z flips sign.
    follower_verts = read_binary_stl_vertices(ASSETS_DIR / "wrist_roll_follower_so101_v1.stl")
    follower_verts_link = follower_verts * np.array([1.0, -1.0, -1.0]) + np.array(
        [8.32667e-17, -0.000218214, 0.000949706]
    )
    follower_candidates = {
        f"fixed_jaw:{name}": pt for name, pt in candidate_points(follower_verts_link).items()
    }

    all_candidates = {**jaw_candidates, **follower_candidates}

    p.connect(p.DIRECT)
    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
    link_index_by_name = {}
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        link_index_by_name[info[12].decode("utf-8")] = j
    moving_jaw_idx = link_index_by_name["moving_jaw_so101_v1_link"]
    gripper_link_idx = link_index_by_name["gripper_link"]
    gripper_frame_idx = link_index_by_name["gripper_frame_link"]

    world_z_by_candidate: dict[str, list[float]] = {name: [] for name in all_candidates}
    world_pos_by_candidate: dict[str, list[np.ndarray]] = {name: [] for name in all_candidates}
    gf_offset_by_candidate: dict[str, list[np.ndarray]] = {name: [] for name in all_candidates}

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        joints_deg = np.array(data["joints_deg"], dtype=float)
        joints_rad = np.deg2rad(joints_deg)
        for sim_idx, q in zip(CONTROL_JOINT_INDICES, joints_rad):
            p.resetJointState(robot_id, sim_idx, float(q))

        jaw_state = p.getLinkState(robot_id, moving_jaw_idx, computeForwardKinematics=True)
        jaw_pos, jaw_orn = jaw_state[4], jaw_state[5]
        jaw_rot = np.array(p.getMatrixFromQuaternion(jaw_orn)).reshape(3, 3)

        grip_state = p.getLinkState(robot_id, gripper_link_idx, computeForwardKinematics=True)
        grip_pos, grip_orn = grip_state[4], grip_state[5]
        grip_rot = np.array(p.getMatrixFromQuaternion(grip_orn)).reshape(3, 3)

        gf_state = p.getLinkState(robot_id, gripper_frame_idx, computeForwardKinematics=True)
        gf_pos, gf_orn = np.array(gf_state[4]), gf_state[5]
        gf_rot = np.array(p.getMatrixFromQuaternion(gf_orn)).reshape(3, 3)

        for name, local_pt in all_candidates.items():
            if name.startswith("moving_jaw:"):
                world_pt = jaw_rot @ local_pt + jaw_pos
            else:
                world_pt = grip_rot @ local_pt + grip_pos
            world_z_by_candidate[name].append(world_pt[2])
            world_pos_by_candidate[name].append(world_pt)
            gf_offset_by_candidate[name].append(gf_rot.T @ (world_pt - gf_pos))

    print(f"{'candidate':22s} {'z_spread_mm':>12s}  world_z per pose")
    ranked = sorted(
        world_z_by_candidate.items(), key=lambda kv: max(kv[1]) - min(kv[1])
    )
    for name, zs in ranked:
        spread_mm = (max(zs) - min(zs)) * 1000.0
        z_str = ", ".join(f"{z:.4f}" for z in zs)
        print(f"{name:22s} {spread_mm:12.2f}  [{z_str}]")

    best_name, best_zs = ranked[0]
    offsets = gf_offset_by_candidate[best_name]
    offset_mean = np.mean(offsets, axis=0)
    offset_spread_mm = (np.max(offsets, axis=0) - np.min(offsets, axis=0)) * 1000.0
    print(f"\nbest candidate: {best_name}")
    print(f"  world z per pose : {[round(z, 5) for z in best_zs]}")
    print(f"  offset in gripper_frame_link local frame (mean): {offset_mean.tolist()}")
    print(f"  offset spread across poses (mm): {offset_spread_mm.tolist()}")

    p.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
