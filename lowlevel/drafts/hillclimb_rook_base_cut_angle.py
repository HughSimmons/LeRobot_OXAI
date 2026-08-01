#!/usr/bin/env python3
"""Hill-climb base cut angle for rook mesh static settling.

This draft script tests whether a deliberately angled base cut can make the
settled physical axis line up better with the fitted top-circle/top-plane
normal. Unlike the normal rook sim, this uses mesh collision so the base cut
angle can affect the settled pose.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as p
import pybullet_data


PROJECT_DIR = Path(__file__).resolve().parents[2]
LOWLEVEL_DIR = PROJECT_DIR / "lowlevel"
ROOK_DIR = PROJECT_DIR / "rook_kiri"
INPUT_OBJ = ROOK_DIR / "corrected_candidates" / "candidate_b_top_q99.obj"
OUTPUT_DIR = ROOK_DIR / "corrected_candidates" / "base_angle_hillclimb"
TARGET_HEIGHT = 0.04
CUT_FRACTION = 0.105
SETTLE_STEPS = 700
MASS = 0.05
BOARD_TOP_Z = 0.005
WIDTH, HEIGHT = 640, 360
MATERIAL_NAME = "3DModel"

# Mesh-collision settle is only a diagnostic. Keep damping modest so the body
# can find the support plane, but still comes to rest quickly.
DYNAMICS = {
    "lateralFriction": 1.0,
    "rollingFriction": 0.001,
    "spinningFriction": 0.001,
    "linearDamping": 0.05,
    "angularDamping": 0.05,
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
        raise ValueError(f"Could not read vertices/faces from {path}")
    return np.array(vertices, dtype=float), np.array(faces, dtype=int)


def fit_plane(points):
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    if normal[1] < 0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    distances = (points - center) @ normal
    return {
        "normal": normal,
        "center": center,
        "rms": float(np.sqrt(np.mean(distances * distances))),
    }


def top_fit(vertices):
    y = vertices[:, 1]
    return fit_plane(vertices[y >= np.quantile(y, 0.995)])


def rot_x(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_z(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def angle_deg(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


def add_vertex(vertex_map, output_vertices, point):
    key = tuple(np.round(point, 9))
    if key not in vertex_map:
        vertex_map[key] = len(output_vertices)
        output_vertices.append(np.array(point, dtype=float))
    return vertex_map[key]


def clip_triangle_to_plane(tri, normal, point):
    signed = (tri - point) @ normal
    keep = signed >= -1e-12
    if keep.all():
        return [tri]
    if not keep.any():
        return []
    clipped = []
    for idx in range(3):
        current = tri[idx]
        previous = tri[(idx - 1) % 3]
        current_signed = signed[idx]
        previous_signed = signed[(idx - 1) % 3]
        current_keep = current_signed >= -1e-12
        previous_keep = previous_signed >= -1e-12
        if current_keep != previous_keep:
            t = previous_signed / (previous_signed - current_signed)
            clipped.append(previous + t * (current - previous))
        if current_keep:
            clipped.append(current)
    if len(clipped) < 3:
        return []
    clipped = np.array(clipped)
    if len(clipped) == 3:
        return [clipped]
    return [clipped[[0, idx, idx + 1]] for idx in range(1, len(clipped) - 1)]


def plane_basis(normal):
    normal = normal / np.linalg.norm(normal)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(ref @ normal)) > 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    u = ref - normal * float(ref @ normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def cap_faces(contour, normal, vertex_map, output_vertices):
    if len(contour) < 3:
        return []
    contour = np.unique(np.round(np.array(contour), 9), axis=0)
    center = contour.mean(axis=0)
    u, v = plane_basis(normal)
    rel = contour - center
    angles = np.arctan2(rel @ v, rel @ u)
    ordered = contour[np.argsort(angles)]
    center_idx = add_vertex(vertex_map, output_vertices, center)
    ring = [add_vertex(vertex_map, output_vertices, point) for point in ordered]
    return [[center_idx, ring[(idx + 1) % len(ring)], ring[idx]] for idx in range(len(ring))]


def make_cut_mesh_y_up(vertices, faces, tilt_x_deg, tilt_z_deg):
    height = vertices[:, 1].max() - vertices[:, 1].min()
    plane_point = np.array([0.0, vertices[:, 1].min() + CUT_FRACTION * height, 0.0])
    normal = rot_z(math.radians(tilt_z_deg)) @ rot_x(math.radians(tilt_x_deg)) @ np.array([0.0, 1.0, 0.0])
    normal /= np.linalg.norm(normal)

    out_vertices = []
    out_faces = []
    vertex_map = {}
    contour = []
    for tri in vertices[faces]:
        signed = (tri - plane_point) @ normal
        if signed.min() <= 1e-12 and signed.max() >= -1e-12:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                s0 = signed[a]
                s1 = signed[b]
                if s0 * s1 <= 1e-12 and abs(s1 - s0) > 1e-12:
                    t = s0 / (s0 - s1)
                    if -1e-9 <= t <= 1 + 1e-9:
                        contour.append(tri[a] + t * (tri[b] - tri[a]))
        for clipped in clip_triangle_to_plane(tri, normal, plane_point):
            face = [add_vertex(vertex_map, out_vertices, point) for point in clipped]
            if len(set(face)) == 3:
                out_faces.append(face)
    out_faces.extend(cap_faces(contour, normal, vertex_map, out_vertices))
    out = np.array(out_vertices, dtype=float)
    return out, np.array(out_faces, dtype=int), normal


def y_up_to_z_up(vertices):
    converted = np.column_stack([vertices[:, 0], vertices[:, 2], vertices[:, 1]])
    converted[:, 2] -= converted[:, 2].min()
    converted[:, 0] -= (converted[:, 0].min() + converted[:, 0].max()) / 2.0
    converted[:, 1] -= (converted[:, 1].min() + converted[:, 1].max()) / 2.0
    return converted


def normal_y_up_to_z_up(normal):
    return np.array([normal[0], normal[2], normal[1]], dtype=float)


def write_obj(path, vertices_z_up, faces):
    lines = [
        "mtllib Rook_cleaned.mtl",
        f"o {path.stem}",
        f"usemtl {MATERIAL_NAME}",
    ]
    lines.extend(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}" for v in vertices_z_up)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def setup_board():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", [0, 0, 0])
    board_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.16, 0.16, 0.005])
    board_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.16, 0.16, 0.005], rgbaColor=[0.3, 0.3, 0.3, 1])
    p.createMultiBody(0, board_shape, board_visual, basePosition=[0.25, 0.0, 0.0])


def settle_score(obj_path, top_normal_z_up):
    vertices = read_obj_vertices_only(obj_path)
    raw_dims = vertices.max(axis=0) - vertices.min(axis=0)
    mesh_scale = TARGET_HEIGHT / raw_dims[2]
    p.connect(p.DIRECT)
    try:
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        setup_board()
        collision = p.createCollisionShape(p.GEOM_MESH, fileName=str(obj_path), meshScale=[mesh_scale] * 3)
        visual = p.createVisualShape(p.GEOM_MESH, fileName=str(obj_path), meshScale=[mesh_scale] * 3, rgbaColor=[0.8, 0.8, 0.85, 1.0])
        piece = p.createMultiBody(
            baseMass=MASS,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[0.25, 0.0, BOARD_TOP_Z + TARGET_HEIGHT * 0.75],
        )
        p.changeDynamics(piece, -1, **DYNAMICS)
        for _ in range(SETTLE_STEPS):
            p.stepSimulation()
        pos, orn = p.getBasePositionAndOrientation(piece)
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        physical_axis = rot @ np.array([0.0, 0.0, 1.0])
        top_world = rot @ top_normal_z_up
        lin, ang = p.getBaseVelocity(piece)
        return {
            "angle_top_normal_to_physical_axis_deg": angle_deg(top_world, physical_axis),
            "angle_top_normal_to_world_up_deg": angle_deg(top_world, [0, 0, 1]),
            "angle_physical_axis_to_world_up_deg": angle_deg(physical_axis, [0, 0, 1]),
            "settled_position": list(pos),
            "settled_orientation_xyzw": list(orn),
            "linear_speed": float(np.linalg.norm(lin)),
            "angular_speed": float(np.linalg.norm(ang)),
        }
    finally:
        if p.isConnected():
            p.disconnect()


def read_obj_vertices_only(path):
    vertices = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
    return np.array(vertices, dtype=float)


def render_best(obj_path, output_dir):
    vertices = read_obj_vertices_only(obj_path)
    raw_dims = vertices.max(axis=0) - vertices.min(axis=0)
    mesh_scale = TARGET_HEIGHT / raw_dims[2]
    p.connect(p.DIRECT)
    try:
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        setup_board()
        collision = p.createCollisionShape(p.GEOM_MESH, fileName=str(obj_path), meshScale=[mesh_scale] * 3)
        visual = p.createVisualShape(p.GEOM_MESH, fileName=str(obj_path), meshScale=[mesh_scale] * 3, rgbaColor=[0.8, 0.8, 0.85, 1.0])
        piece = p.createMultiBody(
            baseMass=MASS,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[0.25, 0.0, BOARD_TOP_Z + TARGET_HEIGHT * 0.75],
        )
        p.changeDynamics(piece, -1, **DYNAMICS)
        for _ in range(SETTLE_STEPS):
            p.stepSimulation()
        pos, _ = p.getBasePositionAndOrientation(piece)
        cameras = {
            "side": ([pos[0], pos[1] - 0.20, pos[2] + 0.035], [pos[0], pos[1], pos[2] + 0.005], [0, 0, 1]),
            "topdown": ([pos[0], pos[1], pos[2] + 0.35], [pos[0], pos[1], pos[2]], [0, -1, 0]),
        }
        stills = {}
        for label, (eye, target, up) in cameras.items():
            projection = p.computeProjectionMatrixFOV(60, WIDTH / HEIGHT, 0.01, 4.0)
            view = p.computeViewMatrix(eye, target, up)
            width, height, rgba, _, _ = p.getCameraImage(WIDTH, HEIGHT, viewMatrix=view, projectionMatrix=projection)
            image = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
            path = output_dir / f"best_{label}.png"
            imageio.imwrite(path, image[:, :, :3])
            stills[label] = str(path)
        return stills
    finally:
        if p.isConnected():
            p.disconnect()


def evaluate_grid(vertices, faces, top_normal_z_up, output_dir, angles_x, angles_z, prefix):
    rows = []
    for tilt_x in angles_x:
        for tilt_z in angles_z:
            cut_y_up, cut_faces, cut_normal_y_up = make_cut_mesh_y_up(vertices, faces, tilt_x, tilt_z)
            cut_z_up = y_up_to_z_up(cut_y_up)
            obj_path = output_dir / f"{prefix}_x{tilt_x:+05.2f}_z{tilt_z:+05.2f}.obj"
            write_obj(obj_path, cut_z_up, cut_faces)
            score = settle_score(obj_path, top_normal_z_up)
            row = {
                "tilt_x_deg": float(tilt_x),
                "tilt_z_deg": float(tilt_z),
                "obj_path": str(obj_path),
                "cut_normal_y_up": cut_normal_y_up,
                "cut_normal_z_up": normal_y_up_to_z_up(cut_normal_y_up),
                **score,
            }
            rows.append(row)
            print(
                f"{prefix} x={tilt_x:+.2f} z={tilt_z:+.2f} "
                f"score={row['angle_top_normal_to_world_up_deg']:.3f} "
                f"top-vs-axis={row['angle_top_normal_to_physical_axis_deg']:.3f} "
                f"axis-up={row['angle_physical_axis_to_world_up_deg']:.3f}"
            )
    return rows


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOWLEVEL_DIR / "recordings" / f"rook_base_angle_hillclimb_{stamp}"
    mesh_dir = OUTPUT_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    mesh_dir.mkdir(parents=True, exist_ok=False)
    source_mtl = INPUT_OBJ.parent / "Rook_cleaned.mtl"
    if source_mtl.exists():
        (mesh_dir / "Rook_cleaned.mtl").write_text(source_mtl.read_text(encoding="utf-8"), encoding="utf-8")

    vertices, faces = read_obj(INPUT_OBJ)
    fit = top_fit(vertices)
    top_normal_z_up = normal_y_up_to_z_up(fit["normal"])

    coarse_angles = np.array([-6.0, -3.0, 0.0, 3.0, 6.0])
    coarse = evaluate_grid(vertices, faces, top_normal_z_up, mesh_dir, coarse_angles, coarse_angles, "coarse")
    best_coarse = min(coarse, key=lambda row: row["angle_top_normal_to_world_up_deg"])
    center_x = best_coarse["tilt_x_deg"]
    center_z = best_coarse["tilt_z_deg"]

    fine_x = center_x + np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
    fine_z = center_z + np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
    fine = evaluate_grid(vertices, faces, top_normal_z_up, mesh_dir, fine_x, fine_z, "fine")
    all_rows = coarse + fine
    best = min(all_rows, key=lambda row: row["angle_top_normal_to_world_up_deg"])
    best_stills = render_best(Path(best["obj_path"]), run_dir)

    summary = {
        "summary": "Hill-climbed angled base cut with mesh-collision settling.",
        "input_obj": str(INPUT_OBJ),
        "mesh_output_dir": str(mesh_dir),
        "cut_fraction": CUT_FRACTION,
        "settle_steps": SETTLE_STEPS,
        "target_height_m": TARGET_HEIGHT,
        "top_fit_y_up": fit,
        "top_normal_z_up": top_normal_z_up,
        "coarse_count": len(coarse),
        "fine_count": len(fine),
        "best": {**best, "stills": best_stills},
        "coarse_results": coarse,
        "fine_results": fine,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(json_safe({"summary_path": str(summary_path), "best": summary["best"]}), indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
