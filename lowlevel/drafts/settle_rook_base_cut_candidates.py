#!/usr/bin/env python3
"""Settle base-cut rook candidates and compare visual top axis to proxy axis."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as p
import pybullet_data


PROJECT_DIR = Path(__file__).resolve().parents[2]
LOWLEVEL_DIR = PROJECT_DIR / "lowlevel"
CANDIDATE_DIR = PROJECT_DIR / "rook_kiri" / "corrected_candidates" / "base_circle_cuts"
AXIS_TRIM_DIR = PROJECT_DIR / "rook_kiri" / "corrected_candidates"
CANDIDATE_SETS = {
    "base_circle_cuts": (
        CANDIDATE_DIR / "candidate_a_base_circle_cut_090.obj",
        CANDIDATE_DIR / "candidate_a_base_circle_cut_105.obj",
        CANDIDATE_DIR / "candidate_a_base_circle_cut_120.obj",
    ),
    "axis_trim": (
        AXIS_TRIM_DIR / "candidate_a_top_q995.obj",
        AXIS_TRIM_DIR / "candidate_b_top_q99.obj",
        AXIS_TRIM_DIR / "candidate_c_top_max_minus_002.obj",
    ),
}
CANDIDATE_SET = os.environ.get("ROOK_SETTLE_CANDIDATE_SET", "base_circle_cuts")
if CANDIDATE_SET not in CANDIDATE_SETS:
    raise ValueError(
        f"ROOK_SETTLE_CANDIDATE_SET must be one of {sorted(CANDIDATE_SETS)}, got {CANDIDATE_SET!r}"
    )
CANDIDATE_OBJS = CANDIDATE_SETS[CANDIDATE_SET]

TARGET_HEIGHT = 0.04
BOARD_TOP_Z = 0.005
MASS = 0.05
WIDTH, HEIGHT = 640, 360
SETTLE_STEPS = 600
RENDER_FREQ = 10
VISUAL_EULER = [math.pi / 2.0, 0.0, 0.0]
VISUAL_ROT = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(VISUAL_EULER))).reshape(3, 3)
DYNAMICS = {
    "lateralFriction": 1.0,
    "rollingFriction": 0.001,
    "spinningFriction": 0.001,
    "linearDamping": 0.04,
    "angularDamping": 0.04,
}


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def read_obj_vertices(path):
    vertices = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"No vertices found in {path}")
    return np.array(vertices, dtype=float)


def fit_plane(points):
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    if normal[1] < 0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    d = -float(normal @ center)
    distances = points @ normal + d
    return {
        "normal": normal,
        "center": center,
        "rms": float(np.sqrt(np.mean(distances * distances))),
        "distance_min": float(distances.min()),
        "distance_max": float(distances.max()),
    }


def top_normal(vertices):
    y = vertices[:, 1]
    mask = y >= np.quantile(y, 0.995)
    return fit_plane(vertices[mask])


def angle_deg(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


def render_png(path, camera):
    projection = p.computeProjectionMatrixFOV(60, WIDTH / HEIGHT, 0.01, 4.0)
    view = p.computeViewMatrix(camera["eye"], camera["target"], camera["up"])
    width, height, rgba, _, _ = p.getCameraImage(
        WIDTH,
        HEIGHT,
        viewMatrix=view,
        projectionMatrix=projection,
    )
    image = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
    imageio.imwrite(path, image[:, :, :3])


def camera_set(piece_pos):
    x, y, z = piece_pos
    return {
        "side": {
            "eye": [x, y - 0.20, z + 0.035],
            "target": [x, y, z + 0.005],
            "up": [0, 0, 1],
        },
        "topdown": {
            "eye": [x, y, z + 0.35],
            "target": [x, y, z],
            "up": [0, -1, 0],
        },
        "board_square": {
            "eye": [x + 0.10, y - 0.16, z + 0.11],
            "target": [x, y, z + 0.01],
            "up": [0, 0, 1],
        },
    }


def create_board():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", [0, 0, 0])
    board_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.16, 0.16, 0.005])
    board_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.16, 0.16, 0.005],
        rgbaColor=[0.3, 0.3, 0.3, 1],
    )
    p.createMultiBody(0, board_shape, board_visual, basePosition=[0.25, 0.0, 0.0])
    square = 0.04
    for row in range(8):
        for col in range(8):
            x = 0.25 - 0.16 + (col + 0.5) * square
            y = -0.16 + (row + 0.5) * square
            color = [1, 1, 1, 0.5] if (row + col) % 2 == 0 else [0.1, 0.1, 0.1, 0.5]
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.019, 0.019, 0.001], rgbaColor=color)
            p.createMultiBody(0, -1, visual, basePosition=[x, y, 0.008])


def video_context(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "front": imageio.get_writer(output_dir / "front.mp4", fps=24, macro_block_size=1),
        "top": imageio.get_writer(output_dir / "top.mp4", fps=24, macro_block_size=1),
    }


def append_video_frame(context, piece_pos):
    for key, camera in {
        "front": {
            "eye": [piece_pos[0] + 0.10, piece_pos[1] - 0.18, piece_pos[2] + 0.09],
            "target": [piece_pos[0], piece_pos[1], piece_pos[2] + 0.005],
            "up": [0, 0, 1],
        },
        "top": {
            "eye": [piece_pos[0], piece_pos[1], piece_pos[2] + 0.35],
            "target": [piece_pos[0], piece_pos[1], piece_pos[2]],
            "up": [0, -1, 0],
        },
    }.items():
        projection = p.computeProjectionMatrixFOV(60, WIDTH / HEIGHT, 0.01, 4.0)
        view = p.computeViewMatrix(camera["eye"], camera["target"], camera["up"])
        width, height, rgba, _, _ = p.getCameraImage(
            WIDTH,
            HEIGHT,
            viewMatrix=view,
            projectionMatrix=projection,
        )
        image = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
        context[key].append_data(image[:, :, :3])


def close_video_context(context):
    for writer in context.values():
        writer.close()


def settle_candidate(obj_path, output_dir):
    vertices = read_obj_vertices(obj_path)
    raw_dims = vertices.max(axis=0) - vertices.min(axis=0)
    mesh_scale = TARGET_HEIGHT / raw_dims[1]
    scaled_dims = raw_dims * mesh_scale
    collision_radius = float(max(scaled_dims[0], scaled_dims[2]) / 2.0)
    top_fit = top_normal(vertices)
    top_normal_body = VISUAL_ROT @ top_fit["normal"]

    p.connect(p.DIRECT)
    try:
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        create_board()
        collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=collision_radius,
            height=TARGET_HEIGHT,
        )
        visual = p.createVisualShape(
            p.GEOM_MESH,
            fileName=str(obj_path),
            meshScale=[mesh_scale] * 3,
            visualFramePosition=[0.0, 0.0, -TARGET_HEIGHT / 2.0],
            visualFrameOrientation=p.getQuaternionFromEuler(VISUAL_EULER),
            rgbaColor=[0.8, 0.8, 0.85, 1.0],
        )
        piece_id = p.createMultiBody(
            baseMass=MASS,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[0.25, 0.0, BOARD_TOP_Z + TARGET_HEIGHT / 2.0],
        )
        p.changeDynamics(piece_id, -1, **DYNAMICS)

        candidate_dir = output_dir / obj_path.stem
        candidate_dir.mkdir(parents=True, exist_ok=True)
        context = video_context(candidate_dir / "settle_video")
        min_z = np.inf
        max_axis_tilt = 0.0
        for step in range(SETTLE_STEPS):
            p.stepSimulation()
            pos, orn = p.getBasePositionAndOrientation(piece_id)
            min_z = min(min_z, float(pos[2]))
            body_rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            physical_axis_world = body_rot @ np.array([0.0, 0.0, 1.0])
            max_axis_tilt = max(max_axis_tilt, angle_deg(physical_axis_world, [0, 0, 1]))
            if step % RENDER_FREQ == 0:
                append_video_frame(context, pos)
        close_video_context(context)

        pos, orn = p.getBasePositionAndOrientation(piece_id)
        body_rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        physical_axis_world = body_rot @ np.array([0.0, 0.0, 1.0])
        top_normal_world = body_rot @ top_normal_body

        stills = {}
        for label, camera in camera_set(pos).items():
            still_path = candidate_dir / f"{label}.png"
            render_png(still_path, camera)
            stills[label] = str(still_path)

        return {
            "name": obj_path.stem,
            "obj_path": str(obj_path),
            "mesh_scale_for_40mm_height": mesh_scale,
            "scaled_dims_m": scaled_dims,
            "collision_radius_m": collision_radius,
            "top_fit_local_obj_y_up": top_fit,
            "top_normal_body_frame": top_normal_body,
            "settled_position": list(pos),
            "settled_orientation_xyzw": list(orn),
            "physical_axis_world_body_z": physical_axis_world,
            "top_normal_world_after_settle": top_normal_world,
            "angle_top_normal_to_physical_axis_deg": angle_deg(top_normal_world, physical_axis_world),
            "angle_physical_axis_to_world_up_deg": angle_deg(physical_axis_world, [0, 0, 1]),
            "angle_top_normal_to_world_up_deg": angle_deg(top_normal_world, [0, 0, 1]),
            "min_base_z_during_settle": min_z,
            "max_physical_axis_tilt_during_settle_deg": max_axis_tilt,
            "stills": stills,
            "video_dir": str(candidate_dir / "settle_video"),
        }
    finally:
        if p.isConnected():
            p.disconnect()


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = LOWLEVEL_DIR / "recordings" / f"rook_{CANDIDATE_SET}_settle_compare_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    results = [settle_candidate(path, output_dir) for path in CANDIDATE_OBJS]
    summary = {
        "summary": "Base-cut candidate settle comparison using current simplified cylinder proxy physics.",
        "candidate_set": CANDIDATE_SET,
        "target_height_m": TARGET_HEIGHT,
        "settle_steps": SETTLE_STEPS,
        "visual_euler": VISUAL_EULER,
        "dynamics": DYNAMICS,
        "candidates": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
