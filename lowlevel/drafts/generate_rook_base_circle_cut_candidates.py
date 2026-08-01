#!/usr/bin/env python3
"""Generate base-cut rook candidates where the support section is rounder.

This is intentionally a draft mesh-prep utility. It leaves the source OBJ and
the first axis-corrected candidates untouched, then writes derived clipped
meshes for visual inspection.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as p
import pybullet_data


PROJECT_DIR = Path(__file__).resolve().parents[2]
ROOK_DIR = PROJECT_DIR / "rook_kiri"
INPUT_OBJ = ROOK_DIR / "corrected_candidates" / "candidate_a_top_q995.obj"
SOURCE_MTL = ROOK_DIR / "corrected_candidates" / "Rook_cleaned.mtl"
SOURCE_TEXTURE = ROOK_DIR / "corrected_candidates" / "3DModel.jpg"
OUTPUT_DIR = ROOK_DIR / "corrected_candidates" / "base_circle_cuts"

TARGET_HEIGHT = 0.04
WIDTH, HEIGHT = 640, 360
MATERIAL_NAME = "3DModel"

# Fractions of the current corrected mesh height. The circle scan showed the
# first robust circular region around 0.10-0.12, i.e. about 4-5 mm at 40 mm.
CUT_FRACTIONS = (0.09, 0.105, 0.12)


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
    vertices = []
    faces = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                face = []
                for item in line.split()[1:]:
                    face.append(int(item.split("/")[0]) - 1)
                if len(face) == 3:
                    faces.append(face)
                elif len(face) > 3:
                    for idx in range(1, len(face) - 1):
                        faces.append([face[0], face[idx], face[idx + 1]])
    if not vertices or not faces:
        raise ValueError(f"Could not read vertices/faces from {path}")
    return np.array(vertices, dtype=float), np.array(faces, dtype=int)


def section_points(vertices, faces, cut_y):
    points = []
    for tri in vertices[faces]:
        ys = tri[:, 1]
        if cut_y < ys.min() - 1e-12 or cut_y > ys.max() + 1e-12:
            continue
        for a, b in ((0, 1), (1, 2), (2, 0)):
            p0 = tri[a]
            p1 = tri[b]
            y0 = p0[1]
            y1 = p1[1]
            if abs(y1 - y0) < 1e-12:
                continue
            if (cut_y - y0) * (cut_y - y1) <= 1e-12:
                t = (cut_y - y0) / (y1 - y0)
                if -1e-9 <= t <= 1.0 + 1e-9:
                    point = p0 + t * (p1 - p0)
                    points.append([point[0], point[2]])
    if not points:
        return np.empty((0, 2))
    return np.unique(np.round(np.array(points), 7), axis=0)


def circle_fit(points):
    if len(points) < 8:
        return None
    x = points[:, 0]
    z = points[:, 1]
    matrix = np.column_stack([2 * x, 2 * z, np.ones_like(x)])
    rhs = x * x + z * z
    cx, cz, c = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    radius_squared = cx * cx + cz * cz + c
    if radius_squared <= 0:
        return None
    radius = float(np.sqrt(radius_squared))
    distance = np.sqrt((x - cx) ** 2 + (z - cz) ** 2)
    rms = float(np.sqrt(np.mean((distance - radius) ** 2)))
    width = float(x.max() - x.min())
    depth = float(z.max() - z.min())
    eccentricity = abs(width - depth) / max(width, depth)
    return {
        "point_count": int(len(points)),
        "center_xz": [float(cx), float(cz)],
        "radius_raw": radius,
        "rms_raw": rms,
        "relative_rms": rms / radius,
        "width_raw": width,
        "depth_raw": depth,
        "bbox_eccentricity": eccentricity,
    }


def add_vertex(vertex_map, output_vertices, point):
    key = tuple(np.round(point, 9))
    if key not in vertex_map:
        vertex_map[key] = len(output_vertices)
        output_vertices.append(np.array(point, dtype=float))
    return vertex_map[key]


def clip_triangle_to_above(tri, cut_y):
    above = tri[:, 1] >= cut_y - 1e-12
    if above.all():
        return [tri]
    if not above.any():
        return []

    clipped = []
    for idx in range(3):
        current = tri[idx]
        previous = tri[(idx - 1) % 3]
        current_above = current[1] >= cut_y - 1e-12
        previous_above = previous[1] >= cut_y - 1e-12
        if current_above != previous_above:
            t = (cut_y - previous[1]) / (current[1] - previous[1])
            clipped.append(previous + t * (current - previous))
        if current_above:
            clipped.append(current)

    if len(clipped) < 3:
        return []
    clipped = np.array(clipped)
    if len(clipped) == 3:
        return [clipped]
    if len(clipped) == 4:
        return [
            clipped[[0, 1, 2]],
            clipped[[0, 2, 3]],
        ]
    return [clipped[[0, idx, idx + 1]] for idx in range(1, len(clipped) - 1)]


def cap_faces(contour_points, cut_y, vertex_map, output_vertices):
    if len(contour_points) < 3:
        return []
    center_xz = contour_points.mean(axis=0)
    angles = np.arctan2(contour_points[:, 1] - center_xz[1], contour_points[:, 0] - center_xz[0])
    ordered = contour_points[np.argsort(angles)]
    center_idx = add_vertex(vertex_map, output_vertices, [center_xz[0], cut_y, center_xz[1]])
    ring_indices = [
        add_vertex(vertex_map, output_vertices, [point[0], cut_y, point[1]])
        for point in ordered
    ]
    faces = []
    for idx, current in enumerate(ring_indices):
        nxt = ring_indices[(idx + 1) % len(ring_indices)]
        faces.append([center_idx, nxt, current])
    return faces


def make_cut_mesh(vertices, faces, cut_y):
    output_vertices = []
    output_faces = []
    vertex_map = {}
    for tri in vertices[faces]:
        for clipped_tri in clip_triangle_to_above(tri, cut_y):
            face = [add_vertex(vertex_map, output_vertices, point) for point in clipped_tri]
            if len(set(face)) == 3:
                output_faces.append(face)

    contour = section_points(vertices, faces, cut_y)
    output_faces.extend(cap_faces(contour, cut_y, vertex_map, output_vertices))
    out = np.array(output_vertices, dtype=float)
    out[:, 1] -= out[:, 1].min()
    out[:, 0] -= (out[:, 0].min() + out[:, 0].max()) / 2.0
    out[:, 2] -= (out[:, 2].min() + out[:, 2].max()) / 2.0
    return out, np.array(output_faces, dtype=int), contour


def write_obj(path, vertices, faces):
    lines = [
        "mtllib Rook_cleaned.mtl",
        f"o {path.stem}",
        f"usemtl {MATERIAL_NAME}",
    ]
    lines.extend(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}" for v in vertices)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def render_candidate(obj_path, preview_dir, mesh_scale, scaled_dims):
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
        square = 0.04
        for row in range(8):
            for col in range(8):
                x = 0.25 - 0.16 + (col + 0.5) * square
                y = -0.16 + (row + 0.5) * square
                color = [1, 1, 1, 0.5] if (row + col) % 2 == 0 else [0.1, 0.1, 0.1, 0.5]
                visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.019, 0.019, 0.001], rgbaColor=color)
                p.createMultiBody(0, -1, visual, basePosition=[x, y, 0.008])

        visual_shape = p.createVisualShape(
            p.GEOM_MESH,
            fileName=str(obj_path),
            meshScale=[mesh_scale] * 3,
            visualFramePosition=[0.0, 0.0, -TARGET_HEIGHT / 2.0],
            visualFrameOrientation=p.getQuaternionFromEuler([math.pi / 2.0, 0.0, 0.0]),
            rgbaColor=[0.8, 0.8, 0.85, 1.0],
        )
        p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0.25, 0.0, 0.005 + TARGET_HEIGHT / 2.0],
        )
        p.stepSimulation()
        center = np.array([0.25, 0.0, 0.005 + TARGET_HEIGHT / 2.0])
        extents = np.array([scaled_dims[0], scaled_dims[2], scaled_dims[1]])
        radius = max(float(np.linalg.norm(extents) / 2.0), 0.05)
        stills = {}
        for label, camera in camera_set(center, radius).items():
            path = preview_dir / f"{label}.png"
            render_png(path, camera)
            stills[label] = str(path)
        return stills
    finally:
        if p.isConnected():
            p.disconnect()


def process_cut(vertices, faces, cut_fraction):
    height = vertices[:, 1].max() - vertices[:, 1].min()
    cut_y = vertices[:, 1].min() + cut_fraction * height
    cut_vertices, cut_faces, contour = make_cut_mesh(vertices, faces, cut_y)
    raw_dims = cut_vertices.max(axis=0) - cut_vertices.min(axis=0)
    mesh_scale = TARGET_HEIGHT / raw_dims[1]
    scaled_dims = raw_dims * mesh_scale
    metrics = circle_fit(contour)

    name = f"candidate_a_base_circle_cut_{int(round(cut_fraction * 1000)):03d}"
    obj_path = OUTPUT_DIR / f"{name}.obj"
    preview_dir = OUTPUT_DIR / name
    preview_dir.mkdir(parents=True, exist_ok=True)
    write_obj(obj_path, cut_vertices, cut_faces)
    stills = render_candidate(obj_path, preview_dir, mesh_scale, scaled_dims)
    return {
        "name": name,
        "input_obj": str(INPUT_OBJ),
        "output_obj": str(obj_path),
        "cut_fraction_of_input_height": cut_fraction,
        "removed_height_raw": float(cut_y - vertices[:, 1].min()),
        "removed_height_at_40mm_scale_m": cut_fraction * TARGET_HEIGHT,
        "vertex_count": int(cut_vertices.shape[0]),
        "face_count": int(cut_faces.shape[0]),
        "circle_metrics_at_cut": metrics,
        "raw_dims": raw_dims,
        "mesh_scale_for_40mm_height": mesh_scale,
        "scaled_dims_m": scaled_dims,
        "preview": {
            "stills": stills,
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SOURCE_MTL.exists():
        shutil.copy2(SOURCE_MTL, OUTPUT_DIR / SOURCE_MTL.name)
    if SOURCE_TEXTURE.exists():
        shutil.copy2(SOURCE_TEXTURE, OUTPUT_DIR / SOURCE_TEXTURE.name)
    vertices, faces = read_obj(INPUT_OBJ)
    candidates = [process_cut(vertices, faces, cut_fraction) for cut_fraction in CUT_FRACTIONS]
    summary = {
        "summary": "Derived base-cut rook candidates; source meshes left unchanged.",
        "input_obj": str(INPUT_OBJ),
        "target_height_m": TARGET_HEIGHT,
        "cut_fractions": CUT_FRACTIONS,
        "candidates": candidates,
    }
    summary_path = OUTPUT_DIR / "base_circle_cut_summary.json"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
