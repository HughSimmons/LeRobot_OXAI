from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import pybullet as p
import pybullet_data


MESH_PATH = "/Users/zhg603/Downloads/Rook.obj"
OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "recordings"
    / "stl_preview"
)

WIDTH, HEIGHT = 640, 360
# MESH_SCALE = [1.0, 1.0, 1.0]
MESH_SCALE = [0.001, 0.001, 0.001]
MESH_POSITION = [0.0, 0.0, 0.0]
MESH_ORIENTATION_EULER = [0.0, 0.0, 0.0]
USE_COLLISION_MESH = False


def render_png(path, camera_params, distance):
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=WIDTH / HEIGHT,
        nearVal=0.01,
        farVal=max(100, distance * 4),
    )
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_params["eye"],
        cameraTargetPosition=camera_params["target"],
        cameraUpVector=camera_params["up"],
    )
    width, height, rgba, _, _ = p.getCameraImage(
        WIDTH,
        HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
    )
    image = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
    imageio.imwrite(path, image[:, :, :3])


def obj_visual_bounds(mesh_path):
    if mesh_path.suffix.lower() != ".obj":
        return None

    vertices = []
    with mesh_path.open("r", encoding="utf-8", errors="ignore") as mesh_file:
        for line in mesh_file:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue

    if not vertices:
        return None

    vertices = np.array(vertices, dtype=float)
    ##
    raw_min = vertices.min(axis=0)
    raw_max = vertices.max(axis=0)

    center_x = (raw_min[0] + raw_max[0]) / 2
    center_y = (raw_min[1] + raw_max[1]) / 2
    base_z   = raw_min[2]

    offset = np.array([center_x, center_y, base_z])

    vertices -= offset
    ##

    vertices *= np.array(MESH_SCALE, dtype=float)
    vertices += np.array(MESH_POSITION, dtype=float)
    return vertices.min(axis=0), vertices.max(axis=0)


def describe_body(body_id, mesh_path):
    aabb_min, aabb_max = p.getAABB(body_id)
    aabb_min = np.array(aabb_min, dtype=float)
    aabb_max = np.array(aabb_max, dtype=float)
    visual_bounds = obj_visual_bounds(mesh_path)
    if visual_bounds is not None:
        visual_min, visual_max = visual_bounds
        print(f"OBJ visual min: {visual_min}", flush=True)
        print(f"OBJ visual max: {visual_max}", flush=True)
        if np.linalg.norm(aabb_max - aabb_min) <= 0.0:
            aabb_min = visual_min
            aabb_max = visual_max

    center = (aabb_min + aabb_max) / 2
    extents = aabb_max - aabb_min
    radius = max(float(np.linalg.norm(extents) / 2), 0.05)
    print(f"AABB min: {aabb_min}", flush=True)
    print(f"AABB max: {aabb_max}", flush=True)
    print(f"AABB extents: {extents}", flush=True)
    print(f"AABB center: {center}", flush=True)
    print(f"Visual shape data: {p.getVisualShapeData(body_id)}", flush=True)
    return center, extents, radius


def camera_params_for_body(center, radius):
    distance = max(radius * 3.0, 0.25)
    side = {
        "eye": [
            float(center[0]),
            float(center[1] - distance),
            float(center[2] + radius * 0.7),
        ],
        "target": center.tolist(),
        "up": [0, 0, 1],
    }
    topdown = {
        "eye": [
            float(center[0]),
            float(center[1]),
            float(center[2] + distance),
        ],
        "target": center.tolist(),
        "up": [0, -1, 0],
    }
    return side, topdown, distance


def load_mesh(mesh_path):
    print(f"Creating mesh visual shape: {mesh_path}", flush=True)
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=str(mesh_path),
        meshScale=MESH_SCALE,
        rgbaColor=[0.8, 0.8, 0.85, 1.0],
    )
    print(f"Visual shape id: {visual_shape}", flush=True)
    collision_shape = -1
    if USE_COLLISION_MESH:
        print(f"Creating mesh collision shape: {mesh_path}", flush=True)
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=str(mesh_path),
            meshScale=MESH_SCALE,
        )
        print(f"Collision shape id: {collision_shape}", flush=True)
    print("Creating mesh body", flush=True)
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=MESH_POSITION,
        baseOrientation=p.getQuaternionFromEuler(MESH_ORIENTATION_EULER),
    )


def main():
    mesh_path = Path(sys.argv[1] if len(sys.argv) > 1 else MESH_PATH).expanduser()
    if not mesh_path.is_absolute():
        mesh_path = Path(__file__).resolve().parent / mesh_path
    mesh_path = mesh_path.resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(
            f"Mesh file not found: {mesh_path}. "
            "Edit MESH_PATH or pass a mesh path as the first argument."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    p.connect(p.DIRECT)
    try:
        print("Resetting PyBullet simulation", flush=True)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", [0, 0, 0])
        mesh_id = load_mesh(mesh_path)
        p.stepSimulation()
        center, extents, radius = describe_body(mesh_id, mesh_path)
        side_camera, topdown_camera, camera_distance = camera_params_for_body(
            center,
            radius,
        )
        print(f"Camera distance: {camera_distance}", flush=True)

        side_path = OUTPUT_DIR / f"{mesh_path.stem}_side.png"
        topdown_path = OUTPUT_DIR / f"{mesh_path.stem}_topdown.png"
        print(f"Rendering side frame: {side_path}", flush=True)
        render_png(side_path, side_camera, camera_distance)
        print(f"Rendering top-down frame: {topdown_path}", flush=True)
        render_png(topdown_path, topdown_camera, camera_distance)

        print(f"Loaded mesh: {mesh_path}")
        print(f"Mesh body id: {mesh_id}")
        print(f"Side render: {side_path}")
        print(f"Top-down render: {topdown_path}")
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
