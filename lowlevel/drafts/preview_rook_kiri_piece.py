#!/usr/bin/env python3
"""Visual/static physics gate for the rook_kiri piece model."""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as p

os.environ.setdefault("LOOKUP_PIECE_MODEL", "rook_kiri")

LOWLEVEL_DIR = Path(__file__).resolve().parents[1]
if str(LOWLEVEL_DIR) not in sys.path:
    sys.path.insert(0, str(LOWLEVEL_DIR))

import multisim_chess_fast as sim  # noqa: E402


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


def render_png(path, camera):
    projection = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=sim.WIDTH / sim.HEIGHT,
        nearVal=0.01,
        farVal=4.0,
    )
    view = p.computeViewMatrix(
        cameraEyePosition=camera["eye"],
        cameraTargetPosition=camera["target"],
        cameraUpVector=camera["up"],
    )
    width, height, rgba, _, _ = p.getCameraImage(
        sim.WIDTH,
        sim.HEIGHT,
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


def step_with_video(piece_id, output_dir, label, steps, reset_orientation=None):
    if reset_orientation is not None:
        pos, _ = p.getBasePositionAndOrientation(piece_id)
        p.resetBasePositionAndOrientation(
            piece_id,
            pos,
            p.getQuaternionFromEuler(reset_orientation),
        )
        p.resetBaseVelocity(piece_id, [0, 0, 0], [0, 0, 0])

    context = sim.create_video_context(output_dir / label)
    min_z = np.inf
    max_tilt = 0.0
    try:
        for idx in range(steps):
            p.stepSimulation()
            pos, _ = p.getBasePositionAndOrientation(piece_id)
            min_z = min(min_z, float(pos[2]))
            max_tilt = max(max_tilt, float(sim.piece_tilt_deg(piece_id)))
            if idx % sim.renderfreq == 0:
                sim.append_video_frame(context)
    finally:
        sim.close_video_context(context)

    pos, orn = p.getBasePositionAndOrientation(piece_id)
    return {
        "label": label,
        "steps": steps,
        "final_position": list(pos),
        "final_orientation": list(orn),
        "final_tilt_deg": sim.piece_tilt_deg(piece_id),
        "min_z": min_z,
        "max_tilt_deg": max_tilt,
        "video_output_dir": str(output_dir / label),
    }


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = LOWLEVEL_DIR / "recordings" / f"rook_kiri_static_gate_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    world = sim.setup_sim_world("d4")
    piece_id = world["piece_ids"][0]
    try:
        piece_pos, _ = p.getBasePositionAndOrientation(piece_id)
        cameras = camera_set(piece_pos)
        stills = {}
        for label, camera in cameras.items():
            path = output_dir / f"{label}.png"
            render_png(path, camera)
            stills[label] = str(path)

        upright = step_with_video(
            piece_id,
            output_dir,
            "upright_settle",
            steps=300,
            reset_orientation=[0, 0, 0],
        )
        tilted = step_with_video(
            piece_id,
            output_dir,
            "tilt_15deg_settle",
            steps=600,
            reset_orientation=[math.radians(15.0), 0, 0],
        )

        summary = {
            "piece_model": sim.PIECE_MODEL,
            "piece_config": sim.active_piece_config(),
            "world_piece_config": world.get("piece_config"),
            "stills": stills,
            "upright_settle": upright,
            "tilt_15deg_settle": tilted,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(json_safe(summary), indent=2, sort_keys=True))
    finally:
        if "state_id" in world:
            p.removeState(world["state_id"])
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
