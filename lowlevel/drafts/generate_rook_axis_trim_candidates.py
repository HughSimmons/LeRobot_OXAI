#!/usr/bin/env python3
"""Generate and preview non-destructive rook axis/base correction candidates."""

from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as p
import pybullet_data


PROJECT_DIR = Path(__file__).resolve().parents[2]
ROOK_DIR = PROJECT_DIR / "rook_kiri"
SOURCE_OBJ = ROOK_DIR / "Rook_cleaned.obj"
SOURCE_MTL = ROOK_DIR / "Rook_cleaned.mtl"
SOURCE_TEXTURE = ROOK_DIR / "3DModel.jpg"
OUTPUT_DIR = ROOK_DIR / "corrected_candidates"
WIDTH, HEIGHT = 640, 360
TARGET_HEIGHT = 0.04
BASE_BAND_FRACTION = 0.01


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    selector: str
    label: str


CANDIDATES = (
    CandidateSpec("candidate_a_top_q995", "top_q995", "top 0.5% vertices"),
    CandidateSpec("candidate_b_top_q99", "top_q99", "top 1% vertices"),
    CandidateSpec("candidate_c_top_max_minus_002", "top_max_minus_002", "top max_y - 0.002"),
)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def read_obj(path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    vertices = []
    vertex_line_indices = []
    for idx, line in enumerate(lines):
        if not line.startswith("v "):
            continue
        parts = line.split()
        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        vertex_line_indices.append(idx)
    if not vertices:
        raise ValueError(f"No vertices found in {path}")
    return lines, np.array(vertices, dtype=float), vertex_line_indices


def fit_plane(points):
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    if normal[1] < 0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)
    d = -float(normal @ center)
    distances = points @ normal + d
    return {
        "normal": normal,
        "d": d,
        "center": center,
        "rms": float(np.sqrt(np.mean(distances * distances))),
        "distance_min": float(distances.min()),
        "distance_max": float(distances.max()),
    }


def top_selector(vertices, selector):
    y = vertices[:, 1]
    if selector == "top_q995":
        return y >= np.quantile(y, 0.995)
    if selector == "top_q99":
        return y >= np.quantile(y, 0.99)
    if selector == "top_max_minus_002":
        return y >= y.max() - 0.002
    raise ValueError(f"Unknown selector {selector}")


def rotation_between_vectors(source, target):
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    sin_angle = np.linalg.norm(cross)
    cos_angle = float(np.clip(source @ target, -1.0, 1.0))
    if sin_angle < 1e-12:
        if cos_angle > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(source @ axis) > 0.9:
            axis = np.array([0.0, 0.0, 1.0])
        axis = axis - source * (source @ axis)
        axis = axis / np.linalg.norm(axis)
        return rodrigues(axis, math.pi)
    axis = cross / sin_angle
    angle = math.atan2(sin_angle, cos_angle)
    return rodrigues(axis, angle)


def rodrigues(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def format_vertex(vertex):
    return f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}"


def write_candidate_obj(source_lines, vertex_line_indices, vertices, output_path):
    lines = list(source_lines)
    for idx, vertex in zip(vertex_line_indices, vertices):
        lines[idx] = format_vertex(vertex)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def camera_set(center, radius):
    distance = max(radius * 3.0, 0.20)
    return {
        "side": {
            "eye": [float(center[0]), float(center[1] - distance), float(center[2] + radius * 0.6)],
            "target": center.tolist(),
            "up": [0, 0, 1],
        },
        "topdown": {
            "eye": [float(center[0]), float(center[1]), float(center[2] + max(distance, 0.35))],
            "target": center.tolist(),
            "up": [0, -1, 0],
        },
        "board_square": {
            "eye": [float(center[0] + 0.10), float(center[1] - 0.16), float(center[2] + 0.11)],
            "target": [float(center[0]), float(center[1]), float(center[2] + 0.005)],
            "up": [0, 0, 1],
        },
    }


def render_png(path, camera):
    projection = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=WIDTH / HEIGHT,
        nearVal=0.01,
        farVal=4.0,
    )
    view = p.computeViewMatrix(
        cameraEyePosition=camera["eye"],
        cameraTargetPosition=camera["target"],
        cameraUpVector=camera["up"],
    )
    width, height, rgba, _, _ = p.getCameraImage(
        WIDTH,
        HEIGHT,
        viewMatrix=view,
        projectionMatrix=projection,
    )
    image = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
    imageio.imwrite(path, image[:, :, :3])


def render_candidate(candidate_obj, preview_dir, mesh_scale, scaled_dims):
    p.connect(p.DIRECT)
    try:
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", [0, 0, 0])

        board_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.16, 0.16, 0.005])
        board_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.16, 0.16, 0.005],
            rgbaColor=[0.3, 0.3, 0.3, 1],
        )
        p.createMultiBody(0, board_shape, board_visual, basePosition=[0.25, 0.0, 0.0])
        square_size = 0.04
        for row in range(8):
            for col in range(8):
                x = 0.25 - 0.16 + (col + 0.5) * square_size
                y = -0.16 + (row + 0.5) * square_size
                color = [1, 1, 1, 0.5] if (row + col) % 2 == 0 else [0.1, 0.1, 0.1, 0.5]
                visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[0.019, 0.019, 0.001],
                    rgbaColor=color,
                )
                p.createMultiBody(0, -1, visual, basePosition=[x, y, 0.008])

        visual_shape = p.createVisualShape(
            p.GEOM_MESH,
            fileName=str(candidate_obj),
            meshScale=[mesh_scale] * 3,
            visualFramePosition=[0.0, 0.0, -TARGET_HEIGHT / 2.0],
            visualFrameOrientation=p.getQuaternionFromEuler([math.pi / 2.0, 0.0, 0.0]),
            rgbaColor=[0.8, 0.8, 0.85, 1.0],
        )
        body = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0.25, 0.0, 0.005 + TARGET_HEIGHT / 2.0],
        )
        p.stepSimulation()
        # PyBullet can report a degenerate AABB for visual-only mesh bodies, so
        # use the known scaled dimensions for framing.
        center = np.array([0.25, 0.0, 0.005 + TARGET_HEIGHT / 2.0])
        extents = np.array([scaled_dims[0], scaled_dims[2], scaled_dims[1]])
        aabb_min = (center - extents / 2.0).tolist()
        aabb_max = (center + extents / 2.0).tolist()
        radius = max(float(np.linalg.norm(extents) / 2.0), 0.05)
        cameras = camera_set(center, radius)
        stills = {}
        for label, camera in cameras.items():
            path = preview_dir / f"{label}.png"
            render_png(path, camera)
            stills[label] = str(path)
        return {
            "aabb_min": aabb_min,
            "aabb_max": aabb_max,
            "stills": stills,
        }
    finally:
        if p.isConnected():
            p.disconnect()


def process_candidate(spec, source_lines, vertices, vertex_line_indices):
    selected = top_selector(vertices, spec.selector)
    top_fit = fit_plane(vertices[selected])
    rotation = rotation_between_vectors(top_fit["normal"], np.array([0.0, 1.0, 0.0]))
    top_center = top_fit["center"]
    rotated = (vertices - top_center) @ rotation.T + top_center

    base_threshold = np.quantile(rotated[:, 1], BASE_BAND_FRACTION)
    base_mask = rotated[:, 1] <= base_threshold
    base_y = float(rotated[base_mask, 1].min())
    corrected = rotated.copy()
    corrected[base_mask, 1] = base_y

    # Normalize candidate to object-space base at y=0 and center footprint in x/z.
    corrected[:, 1] -= corrected[:, 1].min()
    corrected[:, 0] -= (corrected[:, 0].min() + corrected[:, 0].max()) / 2.0
    corrected[:, 2] -= (corrected[:, 2].min() + corrected[:, 2].max()) / 2.0

    output_obj = OUTPUT_DIR / f"{spec.name}.obj"
    preview_dir = OUTPUT_DIR / spec.name
    preview_dir.mkdir(parents=True, exist_ok=True)
    write_candidate_obj(source_lines, vertex_line_indices, corrected, output_obj)

    raw_dims = corrected.max(axis=0) - corrected.min(axis=0)
    mesh_scale = TARGET_HEIGHT / raw_dims[1]
    scaled_dims = raw_dims * mesh_scale
    preview = render_candidate(output_obj, preview_dir, mesh_scale, scaled_dims)
    base_fit_after = fit_plane(corrected[base_mask])
    top_fit_after = fit_plane(corrected[selected])
    normal_angle_deg = float(
        np.degrees(
            np.arccos(
                np.clip(
                    top_fit["normal"] @ np.array([0.0, 1.0, 0.0]),
                    -1.0,
                    1.0,
                )
            )
        )
    )
    return {
        "name": spec.name,
        "label": spec.label,
        "selector": spec.selector,
        "output_obj": str(output_obj),
        "top_vertex_count": int(selected.sum()),
        "base_vertex_count": int(base_mask.sum()),
        "source_top_fit": top_fit,
        "source_top_normal_angle_from_raw_y_deg": normal_angle_deg,
        "rotation_matrix": rotation,
        "corrected_dims_raw": raw_dims,
        "mesh_scale_for_40mm_height": mesh_scale,
        "corrected_scaled_dims_m": scaled_dims,
        "corrected_top_fit": top_fit_after,
        "corrected_base_fit": base_fit_after,
        "preview": preview,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_MTL, OUTPUT_DIR / SOURCE_MTL.name)
    shutil.copy2(SOURCE_TEXTURE, OUTPUT_DIR / SOURCE_TEXTURE.name)
    source_lines, vertices, vertex_line_indices = read_obj(SOURCE_OBJ)
    source_digest_note = {
        "source_obj": str(SOURCE_OBJ),
        "source_mtl": str(SOURCE_MTL),
        "source_texture": str(SOURCE_TEXTURE),
        "vertex_count": int(vertices.shape[0]),
        "source_min": vertices.min(axis=0),
        "source_max": vertices.max(axis=0),
        "source_dims": vertices.max(axis=0) - vertices.min(axis=0),
    }
    candidates = [
        process_candidate(spec, source_lines, vertices, vertex_line_indices)
        for spec in CANDIDATES
    ]
    summary = {
        "summary": "Derived rook candidates; original Rook_cleaned.obj left unchanged.",
        "target_height_m": TARGET_HEIGHT,
        "base_band_fraction": BASE_BAND_FRACTION,
        "source": source_digest_note,
        "candidates": candidates,
    }
    summary_path = OUTPUT_DIR / "candidate_summary.json"
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
