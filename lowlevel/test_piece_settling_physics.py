import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import pybullet as p
import pybullet_data

from chess_traj import chess_to_xy


BOARD_ORIGIN = (0.25, 0.0, 0.0)
SQUARE_SIZE = 0.04
BOARD_THICKNESS = 0.01
BOARD_TOP_Z = BOARD_ORIGIN[2] + BOARD_THICKNESS / 2
PIECE_RADIUS = 0.012
PIECE_HEIGHT = 0.04
PIECE_MASS = 0.05
SOLVER_ITERATIONS = 200
SOLVER_SUBSTEPS = 4
SETTLE_STEPS = 2400
VIDEO_WIDTH = 320
VIDEO_HEIGHT = 210
VIDEO_CAPTURE_EVERY_STEPS = 8
VIDEO_FPS = 50
VIDEO_OUTPUT_DIR = Path(__file__).resolve().parent / "recordings" / "piece_settling_physics"
BASELINE_DYNAMICS = {
    "lateralFriction": 1.0,
    "rollingFriction": 0.02,
    "spinningFriction": 0.02,
    "linearDamping": 0.2,
    "angularDamping": 0.2,
}
LOW_RESISTANCE_DYNAMICS = {
    "lateralFriction": 0.5,
    "rollingFriction": 0.0,
    "spinningFriction": 0.0,
    "linearDamping": 0.0,
    "angularDamping": 0.0,
}


def ensure_physics_connected(gui=False):
    if p.isConnected():
        return

    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())


def setup_world(gui=False):
    ensure_physics_connected(gui=gui)
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(
        numSolverIterations=SOLVER_ITERATIONS,
        numSubSteps=SOLVER_SUBSTEPS,
    )

    plane_id = p.loadURDF("plane.urdf", [0, 0, 0])
    board_size = 8 * SQUARE_SIZE
    board_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[board_size / 2, board_size / 2, BOARD_THICKNESS / 2],
    )
    board_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[board_size / 2, board_size / 2, BOARD_THICKNESS / 2],
        rgbaColor=[0.3, 0.3, 0.3, 1],
    )
    board_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=board_shape,
        baseVisualShapeIndex=board_visual,
        basePosition=BOARD_ORIGIN,
    )
    return plane_id, board_id


def make_camera():
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0.33, -0.30, 0.16],
        cameraTargetPosition=[0.27, -0.14, 0.025],
        cameraUpVector=[0, 0, 1],
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=45,
        aspect=VIDEO_WIDTH / VIDEO_HEIGHT,
        nearVal=0.01,
        farVal=1.0,
    )
    return view_matrix, projection_matrix


def capture_frame(view_matrix, projection_matrix):
    _, _, rgba, _, _ = p.getCameraImage(
        VIDEO_WIDTH,
        VIDEO_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    return rgba[:, :, :3]


def piece_center_z_for_tilt(tilt_deg, clearance):
    tilt_rad = math.radians(abs(tilt_deg))
    vertical_half_extent = (
        PIECE_HEIGHT / 2 * math.cos(tilt_rad)
        + PIECE_RADIUS * math.sin(tilt_rad)
    )
    return BOARD_TOP_Z + vertical_half_extent + clearance


def create_piece_at(
    square,
    tilt_deg=0.0,
    clearance=0.02,
    dynamics=None,
    angular_velocity=None,
):
    if dynamics is None:
        dynamics = BASELINE_DYNAMICS

    world_x, world_y, _ = chess_to_xy(square, board_origin=BOARD_ORIGIN)
    center_z = piece_center_z_for_tilt(tilt_deg, clearance)
    orn = p.getQuaternionFromEuler([math.radians(tilt_deg), 0.0, 0.0])

    piece_shape = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=PIECE_RADIUS,
        height=PIECE_HEIGHT,
    )
    piece_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=PIECE_RADIUS,
        length=PIECE_HEIGHT,
        rgbaColor=[1, 0, 0, 1],
    )
    piece_id = p.createMultiBody(
        baseMass=PIECE_MASS,
        baseCollisionShapeIndex=piece_shape,
        baseVisualShapeIndex=piece_visual,
        basePosition=[world_x, world_y, center_z],
        baseOrientation=orn,
    )
    p.changeDynamics(piece_id, -1, **dynamics)
    if angular_velocity is not None:
        p.resetBaseVelocity(piece_id, angularVelocity=angular_velocity)
    return piece_id


def final_piece_metrics(piece_id, board_id):
    final_pos, final_orn = p.getBasePositionAndOrientation(piece_id)
    final_rot = p.getMatrixFromQuaternion(final_orn)
    final_tilt_deg = math.degrees(
        math.acos(max(-1.0, min(1.0, abs(final_rot[8]))))
    )
    final_euler_deg = [
        math.degrees(value)
        for value in p.getEulerFromQuaternion(final_orn)
    ]
    board_contacts = p.getContactPoints(piece_id, board_id)
    other_contacts = [
        contact for contact in p.getContactPoints(piece_id)
        if contact[2] != board_id and contact[1] != board_id
    ]
    return {
        "final_position": final_pos,
        "final_tilt_deg": float(final_tilt_deg),
        "final_euler_deg": final_euler_deg,
        "board_contact_count": len(board_contacts),
        "other_contact_count": len(other_contacts),
    }


def run_settle_test(
    name,
    square,
    tilt_deg,
    clearance,
    settle_steps,
    dynamics=None,
    angular_velocity=None,
    video_path=None,
    gui=False,
):
    _, board_id = setup_world(gui=gui)
    piece_id = create_piece_at(
        square,
        tilt_deg=tilt_deg,
        clearance=clearance,
        dynamics=dynamics,
        angular_velocity=angular_velocity,
    )
    writer = None
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(video_path, fps=VIDEO_FPS)
        view_matrix, projection_matrix = make_camera()

    for _ in range(settle_steps):
        p.stepSimulation()
        if writer is not None and _ % VIDEO_CAPTURE_EVERY_STEPS == 0:
            writer.append_data(capture_frame(view_matrix, projection_matrix))

    if writer is not None:
        writer.close()

    metrics = final_piece_metrics(piece_id, board_id)
    metrics.update({
        "name": name,
        "initial_tilt_deg": tilt_deg,
        "initial_clearance": clearance,
        "angular_velocity": angular_velocity,
        "video_path": str(video_path) if video_path is not None else None,
    })
    return metrics


def print_result(result):
    pos = result["final_position"]
    euler = result["final_euler_deg"]
    print(
        f"{result['name']}: "
        f"initial_tilt={result['initial_tilt_deg']:.1f} deg | "
        f"clearance={result['initial_clearance']:.4f} m | "
        f"angular_velocity={result['angular_velocity']} | "
        f"final_tilt={result['final_tilt_deg']:.4f} deg | "
        f"final_pos=[{pos[0]:.5f}, {pos[1]:.5f}, {pos[2]:.5f}] | "
        f"final_euler=[{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}] | "
        f"board_contacts={result['board_contact_count']} | "
        f"other_contacts={result['other_contact_count']}"
    )
    if result["video_path"] is not None:
        print(f"  video={result['video_path']}")


def main():
    parser = argparse.ArgumentParser(
        description="Run no-robot piece settling sanity tests in PyBullet."
    )
    parser.add_argument("--square", default="e1")
    parser.add_argument("--settle-steps", type=int, default=SETTLE_STEPS)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=VIDEO_OUTPUT_DIR,
    )
    args = parser.parse_args()

    tests = [
        {
            "name": "baseline_upright_drop",
            "tilt_deg": 0.0,
            "clearance": 0.04,
            "dynamics": BASELINE_DYNAMICS,
            "angular_velocity": None,
        },
        {
            "name": "baseline_tilt_recovery_15deg",
            "tilt_deg": 15.0,
            "clearance": 0.002,
            "dynamics": BASELINE_DYNAMICS,
            "angular_velocity": None,
            "record_video": True,
        },
        {
            "name": "low_resistance_tilt_recovery_15deg",
            "tilt_deg": 15.0,
            "clearance": 0.002,
            "dynamics": LOW_RESISTANCE_DYNAMICS,
            "angular_velocity": None,
            "record_video": True,
        },
        {
            "name": "higher_drop_tilt_recovery_15deg",
            "tilt_deg": 15.0,
            "clearance": 0.02,
            "dynamics": LOW_RESISTANCE_DYNAMICS,
            "angular_velocity": None,
            "record_video": False,
        },
        {
            "name": "nudged_tilt_recovery_15deg",
            "tilt_deg": 15.0,
            "clearance": 0.002,
            "dynamics": LOW_RESISTANCE_DYNAMICS,
            "angular_velocity": [-0.5, 0.0, 0.0],
            "record_video": False,
        },
        {
            "name": "higher_drop_nudged_tilt_recovery_15deg",
            "tilt_deg": 15.0,
            "clearance": 0.02,
            "dynamics": LOW_RESISTANCE_DYNAMICS,
            "angular_velocity": [-0.5, 0.0, 0.0],
            "record_video": False,
        },
    ]

    print(f"Running piece settling tests at square {args.square}")
    print(f"Script: {Path(__file__).resolve()}")
    try:
        for test in tests:
            video_path = None
            if args.save_videos and test.get("record_video", False):
                video_path = args.video_dir / f"{test['name']}.mp4"

            result = run_settle_test(
                test["name"],
                args.square,
                test["tilt_deg"],
                test["clearance"],
                args.settle_steps,
                dynamics=test["dynamics"],
                angular_velocity=test["angular_velocity"],
                video_path=video_path,
                gui=args.gui,
            )
            print_result(result)
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
