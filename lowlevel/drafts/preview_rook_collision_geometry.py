#!/usr/bin/env python3
"""Preview rook visual mesh against collision mesh/hull from several views."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial import ConvexHull


PROJECT_DIR = Path(__file__).resolve().parents[2]
LOWLEVEL_DIR = PROJECT_DIR / "lowlevel"
MESH_DIR = PROJECT_DIR / "rook_kiri2"
COLLISION_MESH = MESH_DIR / "rook2.obj"
VISUAL_MESH = MESH_DIR / "rook2_debug_orange_visual.obj"
MESH_UP_AXIS = "y"
TARGET_HEIGHT = 0.04
WIDTH, HEIGHT = 900, 560
BAND_COUNT = 5
BAND_OVERLAP_M = 0.001


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


def read_obj(path):
    vertices = []
    faces = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                face = [int(item.split("/")[0]) - 1 for item in line.split()[1:]]
                if len(face) == 3:
                    faces.append(face)
                elif len(face) > 3:
                    for idx in range(1, len(face) - 1):
                        faces.append([face[0], face[idx], face[idx + 1]])
    if not vertices or not faces:
        raise ValueError(f"Could not read {path}")
    return np.array(vertices, dtype=float), np.array(faces, dtype=int)


def to_sim_z_up(vertices):
    if MESH_UP_AXIS == "y":
        converted = np.column_stack([vertices[:, 0], vertices[:, 2], vertices[:, 1]])
    elif MESH_UP_AXIS == "z":
        converted = vertices.copy()
    else:
        raise ValueError(f"Unsupported MESH_UP_AXIS {MESH_UP_AXIS!r}")
    converted[:, 2] -= converted[:, 2].min()
    converted[:, 0] -= (converted[:, 0].min() + converted[:, 0].max()) / 2.0
    converted[:, 1] -= (converted[:, 1].min() + converted[:, 1].max()) / 2.0
    return converted


def write_obj(path, vertices, faces):
    lines = [f"o {path.stem}"]
    lines.extend(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}" for v in vertices)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mesh_scale(vertices):
    dims = vertices.max(axis=0) - vertices.min(axis=0)
    return TARGET_HEIGHT / dims[2]


def make_hull_obj(vertices, output_dir):
    hull = ConvexHull(vertices)
    hull_obj = output_dir / "collision_convex_hull_preview.obj"
    write_obj(hull_obj, vertices, hull.simplices)
    return hull_obj, hull


def make_band_hull_objs(vertices, output_dir, band_count=BAND_COUNT):
    z_min = float(vertices[:, 2].min())
    z_max = float(vertices[:, 2].max())
    height = z_max - z_min
    overlap_raw = BAND_OVERLAP_M / (TARGET_HEIGHT / height)
    band_objs = []
    band_summaries = []
    for band_idx in range(band_count):
        low = z_min + band_idx * height / band_count
        high = z_min + (band_idx + 1) * height / band_count
        low_query = low - overlap_raw if band_idx > 0 else low
        high_query = high + overlap_raw if band_idx < band_count - 1 else high
        mask = (vertices[:, 2] >= low_query) & (vertices[:, 2] <= high_query)
        band_vertices = vertices[mask]
        if band_vertices.shape[0] < 8:
            continue
        hull = ConvexHull(band_vertices)
        path = output_dir / f"collision_band_hull_{band_idx:02d}.obj"
        write_obj(path, band_vertices, hull.simplices)
        band_objs.append(path)
        dims = band_vertices.max(axis=0) - band_vertices.min(axis=0)
        band_summaries.append(
            {
                "band_index": band_idx,
                "obj": str(path),
                "z_low_raw": low,
                "z_high_raw": high,
                "vertex_count": int(band_vertices.shape[0]),
                "hull_vertex_count": int(len(hull.vertices)),
                "hull_face_count": int(hull.simplices.shape[0]),
                "raw_dims": dims,
            }
        )
    return band_objs, band_summaries


def setup_board():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", [0, 0, 0])
    board_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.16, 0.16, 0.005])
    board_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.16, 0.16, 0.005],
        rgbaColor=[0.28, 0.28, 0.28, 1],
    )
    p.createMultiBody(0, board_shape, board_visual, basePosition=[0.25, 0.0, 0.0])
    square = 0.04
    for row in range(8):
        for col in range(8):
            x = 0.25 - 0.16 + (col + 0.5) * square
            y = -0.16 + (row + 0.5) * square
            color = [1, 1, 1, 0.45] if (row + col) % 2 == 0 else [0.02, 0.02, 0.02, 0.45]
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.019, 0.019, 0.001], rgbaColor=color)
            p.createMultiBody(0, -1, visual, basePosition=[x, y, 0.008])


def add_mesh_body(obj_path, scale, pos, color, alpha=1.0):
    visual = p.createVisualShape(
        p.GEOM_MESH,
        fileName=str(obj_path),
        meshScale=[scale] * 3,
        visualFramePosition=[0.0, 0.0, -TARGET_HEIGHT / 2.0],
        rgbaColor=[color[0], color[1], color[2], alpha],
    )
    body = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual,
        basePosition=[pos[0], pos[1], 0.005 + TARGET_HEIGHT / 2.0],
    )
    p.changeVisualShape(
        body,
        -1,
        rgbaColor=[color[0], color[1], color[2], alpha],
        specularColor=[0, 0, 0],
    )
    return body


def add_band_hull_body(band_objs, scale, pos, color, alpha=0.45):
    bodies = []
    for obj_path in band_objs:
        bodies.append(add_mesh_body(obj_path, scale, pos, color, alpha))
    return bodies


def render_png(path, eye, target, up):
    projection = p.computeProjectionMatrixFOV(55, WIDTH / HEIGHT, 0.01, 4.0)
    view = p.computeViewMatrix(eye, target, up)
    width, height, rgba, _, _ = p.getCameraImage(
        WIDTH,
        HEIGHT,
        viewMatrix=view,
        projectionMatrix=projection,
    )
    image = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
    imageio.imwrite(path, image[:, :, :3])


def render_pybullet_views(output_dir, visual_obj, hull_obj, band_objs, scale):
    p.connect(p.DIRECT)
    try:
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        setup_board()
        centers = {
            "visual": [0.18, -0.055],
            "collision_hull": [0.18, 0.055],
            "band_hulls": [0.31, -0.055],
            "overlay": [0.31, 0.055],
        }
        add_mesh_body(visual_obj, scale, centers["visual"], [1.0, 0.62, 0.0], 1.0)
        add_mesh_body(hull_obj, scale, centers["collision_hull"], [0.0, 0.85, 1.0], 0.55)
        add_band_hull_body(band_objs, scale, centers["band_hulls"], [0.15, 1.0, 0.05], 0.55)
        add_mesh_body(visual_obj, scale, centers["overlay"], [1.0, 0.62, 0.0], 1.0)
        add_band_hull_body(band_objs, scale, centers["overlay"], [0.15, 1.0, 0.05], 0.38)
        p.stepSimulation()

        views = {
            "side": ([0.25, -0.31, 0.055], [0.25, 0.0, 0.024], [0, 0, 1]),
            "front": ([0.02, 0.0, 0.055], [0.25, 0.0, 0.024], [0, 0, 1]),
            "topdown": ([0.25, 0.0, 0.38], [0.25, 0.0, 0.024], [0, -1, 0]),
            "isometric": ([0.02, -0.24, 0.12], [0.25, 0.0, 0.025], [0, 0, 1]),
        }
        stills = {}
        for label, (eye, target, up) in views.items():
            path = output_dir / f"pybullet_{label}.png"
            render_png(path, eye, target, up)
            stills[label] = str(path)
        return stills
    finally:
        if p.isConnected():
            p.disconnect()


def hull_edges(hull):
    edges = set()
    for tri in hull.simplices:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges.add(tuple(sorted((int(a), int(b)))))
    return sorted(edges)


def render_projection_plots(output_dir, vertices, hull):
    edges = hull_edges(hull)
    views = {
        "top_xy": (0, 1, "x", "y"),
        "side_xz": (0, 2, "x", "z"),
        "front_yz": (1, 2, "y", "z"),
    }
    paths = {}
    for label, (ix, iy, xlabel, ylabel) in views.items():
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(vertices[:, ix], vertices[:, iy], s=0.5, c="#f5a000", alpha=0.25, label="visual/collision mesh vertices")
        for a, b in edges:
            ax.plot(
                [vertices[a, ix], vertices[b, ix]],
                [vertices[a, iy], vertices[b, iy]],
                color="#00b7ff",
                linewidth=0.35,
                alpha=0.8,
            )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.legend(loc="best")
        path = output_dir / f"hull_projection_{label}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths[label] = str(path)
    return paths


def render_band_projection_plots(output_dir, vertices, band_summaries, scale):
    views = {
        "top_xy": (0, 1, "x", "y"),
        "side_xz": (0, 2, "x", "z"),
        "front_yz": (1, 2, "y", "z"),
    }
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(band_summaries)))
    paths = {}
    for label, (ix, iy, xlabel, ylabel) in views.items():
        fig, ax = plt.subplots(figsize=(7, 7))
        scaled_vertices = vertices * scale
        ax.scatter(
            scaled_vertices[:, ix],
            scaled_vertices[:, iy],
            s=0.5,
            c="#f5a000",
            alpha=0.18,
            label="visual mesh vertices",
        )
        for color, band in zip(colors, band_summaries):
            band_vertices, _ = read_obj(Path(band["obj"]))
            band_vertices = band_vertices * scale
            band_hull = ConvexHull(band_vertices)
            for a, b in hull_edges(band_hull):
                ax.plot(
                    [band_vertices[a, ix], band_vertices[b, ix]],
                    [band_vertices[a, iy], band_vertices[b, iy]],
                    color=color,
                    linewidth=0.45,
                    alpha=0.9,
                )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"banded_{label}")
        path = output_dir / f"band_hull_projection_{label}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths[label] = str(path)
    return paths


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = LOWLEVEL_DIR / "rook_kiri_lookup" / f"collision_geometry_preview_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    raw_vertices, faces = read_obj(COLLISION_MESH)
    vertices = to_sim_z_up(raw_vertices)
    visual_preview_obj = output_dir / "visual_mesh_z_up_preview.obj"
    write_obj(visual_preview_obj, vertices, faces)
    scale = mesh_scale(vertices)
    hull_obj, hull = make_hull_obj(vertices, output_dir)
    band_objs, band_summaries = make_band_hull_objs(vertices, output_dir)
    pybullet_stills = render_pybullet_views(output_dir, visual_preview_obj, hull_obj, band_objs, scale)
    projection_stills = render_projection_plots(output_dir, vertices * scale, hull)
    band_projection_stills = render_band_projection_plots(output_dir, vertices, band_summaries, scale)

    dims = vertices.max(axis=0) - vertices.min(axis=0)
    summary = {
        "purpose": "Visual comparison of rook visual mesh and convex hull collision envelope.",
        "collision_mesh": str(COLLISION_MESH),
        "visual_mesh": str(VISUAL_MESH),
        "visual_preview_obj": str(visual_preview_obj),
        "mesh_up_axis": MESH_UP_AXIS,
        "convex_hull_preview_obj": str(hull_obj),
        "source_mesh_raw_dims": raw_vertices.max(axis=0) - raw_vertices.min(axis=0),
        "mesh_raw_dims": dims,
        "mesh_scale_for_40mm_height": scale,
        "mesh_scaled_dims_m": dims * scale,
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "hull_vertex_count": int(len(hull.vertices)),
        "hull_face_count": int(hull.simplices.shape[0]),
        "band_count": BAND_COUNT,
        "band_overlap_m": BAND_OVERLAP_M,
        "band_hull_objs": [str(path) for path in band_objs],
        "band_hulls": band_summaries,
        "pybullet_stills": pybullet_stills,
        "projection_stills": projection_stills,
        "band_projection_stills": band_projection_stills,
        "notes": [
            "Orange body is the visual mesh.",
            "Cyan body/lines are the convex hull envelope of the collision mesh.",
            "Green bodies/lines are the height-banded convex hull proxy.",
            "This previews the mesh/hull geometry; PyBullet's internal dynamic mesh handling may still differ in contact details.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary_path)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
