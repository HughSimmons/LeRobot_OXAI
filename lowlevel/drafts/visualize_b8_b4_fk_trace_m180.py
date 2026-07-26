"""Trace b8 -> b4 FK with the confirmed -180 degree shoulder-pan delta."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import imageio
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw


SOURCE_SQUARE = "b8"
TARGET_SQUARE = "b4"
SHOULDER_PAN_HOME_DELTA_DEG = -180.0
GRASP_OFFSET = np.array([-0.014, 0.002, -0.003])
PLACE_OFFSET = np.array([-0.01845, 0.00115, -0.005])
LIFT_HEIGHT = 0.13
PLACEMENT_LOWER_STEPS = 2
FRAMES_PER_SAMPLE = 6
INITIAL_HOLD_FRAMES = 36
FINAL_HOLD_FRAMES = 24
FPS = 18

SCRIPT_PATH = Path(__file__).resolve()
LOWLEVEL_DIR = SCRIPT_PATH.parents[1]
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
RUN_DIR = SCRIPT_PATH.parent / "outputs" / f"b8_b4_fk_trace_m180_{RUN_STAMP}"
FRONT_VIDEO_PATH = RUN_DIR / "target_vs_achieved_front.mp4"
TOP_VIDEO_PATH = RUN_DIR / "target_vs_achieved_topdown.mp4"
SUMMARY_PATH = RUN_DIR / "summary.json"
HOME_FRONT_PATH = RUN_DIR / "confirmed_home_front.png"
HOME_TOP_PATH = RUN_DIR / "confirmed_home_topdown.png"

TARGET_COLOR = [0.1, 1.0, 0.2, 1.0]
ACHIEVED_COLOR = [1.0, 0.1, 0.8, 1.0]
TARGET_TEXT_COLOR = (75, 255, 105)
ACHIEVED_TEXT_COLOR = (255, 70, 220)


def add_marker(position, color, radius=0.0028):
    shape = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color,
    )
    return p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=shape,
        basePosition=np.array(position, dtype=float).tolist(),
    )


def add_segment(start, end, color, radius=0.0008):
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        return None

    direction = vector / length
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, direction)
    axis_norm = float(np.linalg.norm(axis))
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if axis_norm < 1e-8:
        axis = np.array([1.0, 0.0, 0.0])
        angle = 0.0 if dot > 0 else np.pi
    else:
        axis /= axis_norm
        angle = float(np.arccos(dot))

    orientation = p.getQuaternionFromAxisAngle(axis.tolist(), angle)
    shape = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=length,
        rgbaColor=color,
    )
    return p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=shape,
        basePosition=((start + end) / 2.0).tolist(),
        baseOrientation=orientation,
    )


def raw_camera_frame(eye, target, up):
    projection = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=640 / 360,
        nearVal=0.01,
        farVal=100,
    )
    view = p.computeViewMatrix(
        cameraEyePosition=eye,
        cameraTargetPosition=target,
        cameraUpVector=up,
    )
    width, height, rgba, _, _ = p.getCameraImage(
        640,
        360,
        viewMatrix=view,
        projectionMatrix=projection,
    )
    return np.array(rgba, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]


def camera_frame(
    eye,
    target,
    up,
    sample=None,
    sample_index=None,
    sample_count=None,
    home_joints=None,
):
    frame = raw_camera_frame(eye, target, up)

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 632, 104), fill=(0, 0, 0))
    if home_joints is not None:
        home_joints = np.array(home_joints, dtype=float)
        draw.text(
            (18, 16),
            "CONFIRMED OVERRIDE HOME | shoulder-pan delta = -180.0 deg",
            fill=(255, 255, 255),
        )
        draw.text(
            (18, 42),
            f"resolved shoulder pan = {home_joints[0]:.6f} deg",
            fill=(255, 220, 80),
        )
        draw.text(
            (18, 68),
            "home joints ["
            + ", ".join(f"{joint:.2f}" for joint in home_joints)
            + "] deg",
            fill=(220, 220, 220),
        )
        return np.asarray(image)

    target_xyz = np.array(sample["target_xyz"], dtype=float)
    achieved_xyz = np.array(sample["achieved_xyz"], dtype=float)
    endpoint_error = float(sample["endpoint_error"])
    draw.text(
        (18, 16),
        f"{SOURCE_SQUARE} -> {TARGET_SQUARE} | {sample['stage']} "
        f"| sample {sample_index + 1}/{sample_count}",
        fill=(255, 255, 255),
    )
    draw.text(
        (18, 38),
        "TARGET   XYZ "
        f"[{target_xyz[0]: .4f}, {target_xyz[1]: .4f}, {target_xyz[2]: .4f}] m",
        fill=TARGET_TEXT_COLOR,
    )
    draw.text(
        (18, 60),
        "ACHIEVED XYZ "
        f"[{achieved_xyz[0]: .4f}, {achieved_xyz[1]: .4f}, {achieved_xyz[2]: .4f}] m",
        fill=ACHIEVED_TEXT_COLOR,
    )
    draw.text(
        (18, 82),
        f"endpoint delta = {endpoint_error:.4f} m"
        f" | max solve-step error = {float(sample['max_step_error']):.4f} m",
        fill=(255, 220, 80),
    )
    return np.asarray(image)


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    os.environ["SOURCE_SQUARES"] = SOURCE_SQUARE
    os.environ["TARGET_MOVES"] = f"{SOURCE_SQUARE}_to_{TARGET_SQUARE}"
    os.environ.pop("LOOKUP_HOME_SHOULDER_PAN_OVERRIDE_DEG", None)
    os.environ["LOOKUP_HOME_SHOULDER_PAN_DELTA_DEG"] = str(
        SHOULDER_PAN_HOME_DELTA_DEG
    )

    if str(LOWLEVEL_DIR) not in sys.path:
        sys.path.insert(0, str(LOWLEVEL_DIR))

    import build_general_nonh_reverse_lookup as builder
    import multisim_chess_fast as sim

    home_joints = builder.recorded_lookup_home_joints_for_source(SOURCE_SQUARE)
    movelist, _, metrics = builder.generate_lookup_trajectory(
        SOURCE_SQUARE,
        TARGET_SQUARE,
        GRASP_OFFSET,
        PLACE_OFFSET,
        placement_lower_steps=PLACEMENT_LOWER_STEPS,
        lift_height=LIFT_HEIGHT,
        home_joints=home_joints,
    )
    samples = metrics.get("fk_trace_samples", [])
    if not samples:
        raise RuntimeError("Trajectory generation produced no FK trace samples")

    world = sim.setup_sim_world(
        SOURCE_SQUARE,
        edge_support_margin=builder.LOOKUP_EDGE_SUPPORT_MARGIN,
        home_joints=home_joints,
    )
    front_writer = imageio.get_writer(
        str(FRONT_VIDEO_PATH),
        fps=FPS,
        codec="libx264",
        quality=8,
    )
    top_writer = imageio.get_writer(
        str(TOP_VIDEO_PATH),
        fps=FPS,
        codec="libx264",
        quality=8,
    )
    sim_joint_map = [0, 1, 2, 3, 4, 6]
    previous_joints = home_joints.copy()
    previous_target = None
    previous_achieved = None

    try:
        front_eye = [0.0, -0.6, 0.25]
        front_target = [0.3, 0.0, 0.05]
        top_eye = [0.3, 0.0, 0.6]
        top_target = [0.3, 0.0, 0.0]
        imageio.imwrite(
            HOME_FRONT_PATH,
            raw_camera_frame(front_eye, front_target, [0, 0, 1]),
        )
        imageio.imwrite(
            HOME_TOP_PATH,
            raw_camera_frame(top_eye, top_target, [0, -1, 0]),
        )
        for _ in range(INITIAL_HOLD_FRAMES):
            front_writer.append_data(
                camera_frame(
                    front_eye,
                    front_target,
                    [0, 0, 1],
                    home_joints=home_joints,
                )
            )
            top_writer.append_data(
                camera_frame(
                    top_eye,
                    top_target,
                    [0, -1, 0],
                    home_joints=home_joints,
                )
            )

        for sample_index, sample in enumerate(samples):
            target_xyz = np.array(sample["target_xyz"], dtype=float)
            achieved_xyz = np.array(sample["achieved_xyz"], dtype=float)
            sample_joints = np.array(sample["joints"], dtype=float)

            add_marker(target_xyz, TARGET_COLOR)
            add_marker(achieved_xyz, ACHIEVED_COLOR)
            if previous_target is not None:
                add_segment(previous_target, target_xyz, TARGET_COLOR)
                add_segment(previous_achieved, achieved_xyz, ACHIEVED_COLOR)

            for alpha in np.linspace(0.0, 1.0, FRAMES_PER_SAMPLE):
                joints = (1.0 - alpha) * previous_joints + alpha * sample_joints
                for traj_idx, sim_idx in enumerate(sim_joint_map):
                    p.resetJointState(
                        world["robot_id"],
                        sim_idx,
                        np.deg2rad(joints[traj_idx]),
                        targetVelocity=0.0,
                    )
                front_writer.append_data(
                    camera_frame(
                        front_eye,
                        front_target,
                        [0, 0, 1],
                        sample,
                        sample_index,
                        len(samples),
                    )
                )
                top_writer.append_data(
                    camera_frame(
                        top_eye,
                        top_target,
                        [0, -1, 0],
                        sample,
                        sample_index,
                        len(samples),
                    )
                )

            previous_joints = sample_joints
            previous_target = target_xyz
            previous_achieved = achieved_xyz

        for _ in range(FINAL_HOLD_FRAMES):
            front_writer.append_data(
                camera_frame(
                    front_eye,
                    front_target,
                    [0, 0, 1],
                    samples[-1],
                    len(samples) - 1,
                    len(samples),
                )
            )
            top_writer.append_data(
                camera_frame(
                    top_eye,
                    top_target,
                    [0, -1, 0],
                    samples[-1],
                    len(samples) - 1,
                    len(samples),
                )
            )
    finally:
        front_writer.close()
        top_writer.close()
        p.removeState(world["state_id"])
        if p.isConnected():
            p.disconnect()

    endpoint_errors = [float(sample["endpoint_error"]) for sample in samples]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_mode": "fk_target_vs_achieved_trace",
        "source_square": SOURCE_SQUARE,
        "target_square": TARGET_SQUARE,
        "home_joints_deg": home_joints,
        "shoulder_pan_home_delta_deg": SHOULDER_PAN_HOME_DELTA_DEG,
        "resolved_shoulder_pan_home_deg": float(home_joints[0]),
        "grasp_offset": GRASP_OFFSET,
        "place_offset": PLACE_OFFSET,
        "lift_height": LIFT_HEIGHT,
        "trajectory_waypoint_count": len(movelist),
        "trace_sample_count": len(samples),
        "trajectory_max_fk_error": metrics["max_fk_error"],
        "max_endpoint_error": max(endpoint_errors),
        "front_video": str(FRONT_VIDEO_PATH),
        "topdown_video": str(TOP_VIDEO_PATH),
        "confirmed_home_front": str(HOME_FRONT_PATH),
        "confirmed_home_topdown": str(HOME_TOP_PATH),
        "legend": {
            "target_path": "green",
            "achieved_fk_path": "magenta",
        },
        "samples": samples,
    }
    SUMMARY_PATH.write_text(
        json.dumps(builder.json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Front video: {FRONT_VIDEO_PATH}")
    print(f"Top-down video: {TOP_VIDEO_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
