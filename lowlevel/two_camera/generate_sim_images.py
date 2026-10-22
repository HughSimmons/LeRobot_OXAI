"""Render the chessboard from hardcoded camera viewpoints in PyBullet.

Builds a board-only scene (no robot, no pieces by default) and renders it
from a "topdown" pose (common overhead reference view), a "tripod" pose
(stationary, off to the side), and a "wrist" pose (closer, steeper,
simulating a camera mounted on the arm). Writes the images plus a
manifest.json carrying the camera intrinsics and ground-truth distances, so
`calibrate_camera_distance.py` can be checked against a known answer.

Run with no flags first and calibrate against that piece-free output. Only
once the camera views are confirmed accurate should `--with-piece` be used
to render a second set of images (written to output/with_piece/, the
piece-free baseline is never overwritten) with a single piece placed on the
board, for downstream piece-detection work.

`--with-robot` also loads the SO101 with the wrist camera mount and replaces
the hardcoded "wrist" pose with the camera's real pose from forward
kinematics at `--wrist-joints-deg` (output goes to output/with_robot/, so
the robot-free baseline is never overwritten). The arm then also appears in
the other cameras' views.

Needs the pybullet/imageio environment (see ../realsimenv.yml).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as p
import pybullet_data

from camera_rig import (
    BOARD_ORIGIN,
    CAMERAS,
    DEFAULT_WRIST_JOINTS_DEG,
    FAR,
    FOV_DEG,
    HEIGHT,
    NEAR,
    SQUARE_SIZE,
    SQUARE_SURFACE_Z,
    WIDTH,
    WRIST_CAMERA_LINK,
    WRIST_FOV_DEG,
    WRIST_IMAGE_UP,
    WRIST_LENS_BASE,
    WRIST_LENS_TIP,
    intrinsics_from_fov,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402
from board_coordinates import square_center_world_xy  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Simple cylinder piece, dimensions matching multisim_chess_fast.py's
# CYLINDER_PIECE_HEIGHT/RADIUS -- only needed here for a visual sanity
# check, not physics, so it's created kinematic (baseMass=0).
PIECE_HEIGHT = 0.04
PIECE_RADIUS = 0.012
PIECE_RGBA = [0.8, 0.1, 0.1, 1.0]


def build_board_only_scene() -> None:
    """Board base box + 64 checker-square visuals. No robot, no pieces.

    Uses a solid-color ground plane rather than pybullet_data's default
    plane.urdf, whose checkered texture looks enough like a second
    chessboard to confuse the corner detector.
    """
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    ground_half_extent = 3.0
    ground_shape = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[ground_half_extent, ground_half_extent, 0.01]
    )
    ground_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[ground_half_extent, ground_half_extent, 0.01],
        rgbaColor=[0.55, 0.55, 0.55, 1],
    )
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=ground_shape,
        baseVisualShapeIndex=ground_visual,
        basePosition=[0, 0, -0.01],
    )

    board_x, board_y, board_z = BOARD_ORIGIN
    board_size = 8 * SQUARE_SIZE

    base_shape = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[board_size / 2, board_size / 2, 0.005]
    )
    base_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[board_size / 2, board_size / 2, 0.005],
        rgbaColor=[0.3, 0.3, 0.3, 1],
    )
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=base_shape,
        baseVisualShapeIndex=base_visual,
        basePosition=[board_x, board_y, board_z],
    )

    for row in range(8):
        for col in range(8):
            x = board_x - board_size / 2 + (col + 0.5) * SQUARE_SIZE
            y = board_y - board_size / 2 + (row + 0.5) * SQUARE_SIZE
            z = SQUARE_SURFACE_Z
            # Opaque squares (unlike multisim_chess_fast.py's alpha=0.5) --
            # solid contrast helps the corner detector.
            color = [1, 1, 1, 1] if (row + col) % 2 == 0 else [0.1, 0.1, 0.1, 1]
            square_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[SQUARE_SIZE / 2 - 0.001, SQUARE_SIZE / 2 - 0.001, 0.001],
                rgbaColor=color,
            )
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=square_visual, basePosition=[x, y, z])


def add_piece(square: str) -> int:
    """Place a single kinematic cylinder piece on `square`'s center."""
    x, y = square_center_world_xy(square)
    z = SQUARE_SURFACE_Z + PIECE_HEIGHT / 2
    collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=PIECE_RADIUS, height=PIECE_HEIGHT)
    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER, radius=PIECE_RADIUS, length=PIECE_HEIGHT, rgbaColor=PIECE_RGBA
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[x, y, z],
    )


def load_robot(joints_deg) -> int:
    """
    Load the SO101 + wrist camera mount at the world origin, posed at `joints_deg`.

    Joints are set directly (resetJointState), not driven by motors: this
    scene is only rendered, never stepped.
    """
    robot = p.loadURDF(str(paths.MOUNT_URDF_PATH), [0, 0, 0], useFixedBase=True)
    movable = [i for i in range(p.getNumJoints(robot)) if p.getJointInfo(robot, i)[2] != p.JOINT_FIXED]
    if len(joints_deg) != len(movable):
        raise ValueError(f"expected {len(movable)} joint angles, got {len(joints_deg)}")
    for joint_index, angle_deg in zip(movable, joints_deg):
        p.resetJointState(robot, joint_index, np.radians(angle_deg))
    return robot


def find_link(robot: int, link_name: str) -> int:
    for i in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, i)[12].decode() == link_name:
            return i
    raise ValueError(f"link {link_name!r} not found in the robot URDF")


def wrist_camera_pose(robot: int) -> dict:
    """Wrist camera eye/target/up in the world, from the camera link's FK pose."""
    state = p.getLinkState(robot, find_link(robot, WRIST_CAMERA_LINK), computeForwardKinematics=True)
    link_pos = np.array(state[4])  # URDF link frame, not state[0] (the centre of mass)
    rotation = np.array(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
    #rotation matrix aligning world axis with mount axis

    lens_base, lens_tip = np.array(WRIST_LENS_BASE), np.array(WRIST_LENS_TIP)
    #coordinates of lens base and tip in frame of mount

    forward = rotation @ (lens_tip - lens_base)
    #axis of the lens in world frame

    forward /= np.linalg.norm(forward)
    # 1 mm in front of the lens face, so the camera doesn't see its own lens
    eye = link_pos + rotation @ lens_tip + 0.001 * forward
    #position of eye in world frame
    return {
        "eye": eye.tolist(),
        "target": (eye + forward).tolist(),
        "up": (rotation @ np.array(WRIST_IMAGE_UP)).tolist(),
        "fov_deg": WRIST_FOV_DEG,
    }


def render_camera(pose: dict) -> np.ndarray:
    proj =p.computeProjectionMatrixFOV(pose.get("fov_deg", FOV_DEG), WIDTH / HEIGHT, NEAR, FAR)
    view = p.computeViewMatrix(pose["eye"], pose["target"], pose["up"])
    w, h, rgba, _, _ = p.getCameraImage(WIDTH, HEIGHT, viewMatrix=view, projectionMatrix=proj)
    image = np.array(rgba, dtype=np.uint8).reshape((h, w, 4))
    return image[:, :, :3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--with-piece",
        action="store_true",
        help="Also place a piece on the board. Only use after the piece-free camera views are calibrated.",
    )
    parser.add_argument("--piece-square", default="e4", help="Square to place the piece on (default: e4)")
    parser.add_argument(
        "--with-robot",
        action="store_true",
        help="Load the arm and place the wrist camera by forward kinematics instead of the hardcoded pose.",
    )
    parser.add_argument(
        "--wrist-joints-deg",
        type=float,
        nargs=6,
        default=list(DEFAULT_WRIST_JOINTS_DEG),
        metavar=("PAN", "LIFT", "ELBOW", "WRIST_FLEX", "WRIST_ROLL", "GRIPPER"),
        help="Arm pose for --with-robot, in degrees (default: camera_rig.DEFAULT_WRIST_JOINTS_DEG)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = OUTPUT_DIR / "with_robot" if args.with_robot else OUTPUT_DIR
    if args.with_piece:
        output_dir = output_dir / "with_piece"
    output_dir.mkdir(parents=True, exist_ok=True)

    build_board_only_scene()
    if args.with_piece:
        add_piece(args.piece_square)

    cameras = dict(CAMERAS)
    if args.with_robot:
        robot = load_robot(args.wrist_joints_deg)
        cameras["wrist"] = wrist_camera_pose(robot)

    manifest = {
        "board": {
            "origin_m": list(BOARD_ORIGIN),
            "square_size_m": SQUARE_SIZE,
            "square_surface_z_m": SQUARE_SURFACE_Z,
        },
        "image_width": WIDTH,
        "image_height": HEIGHT,
        "fov_deg": FOV_DEG,
        "intrinsics": intrinsics_from_fov(),
        "cameras": {},
    }
    if args.with_piece:
        manifest["piece_square"] = args.piece_square
    if args.with_robot:
        manifest["wrist_joints_deg"] = args.wrist_joints_deg

    for name, pose in cameras.items():
        image = render_camera(pose)
        image_file = f"{name}_camera.png"
        imageio.imwrite(str(output_dir / image_file), image)

        eye = np.array(pose["eye"], dtype=np.float64)
        target = np.array(pose["target"], dtype=np.float64)
        fov_deg = pose.get("fov_deg", FOV_DEG)
        manifest["cameras"][name] = {
            "image_file": image_file,
            "eye": pose["eye"],
            "target": pose["target"],
            "up": pose["up"],
            # Per-camera intrinsics; the top-level "intrinsics" stays the shared default
            "fov_deg": fov_deg,
            "intrinsics": intrinsics_from_fov(fov_deg),
            "ground_truth_perpendicular_distance_to_board_plane_m": abs(eye[2] - SQUARE_SURFACE_Z),
            "ground_truth_euclidean_distance_to_target_m": float(np.linalg.norm(eye - target)),
        }
        print(f"Wrote {output_dir / image_file}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
