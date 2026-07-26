"""Render shoulder-pan home interpretations against the saved confirmation."""

import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as p


SCRIPT_PATH = Path(__file__).resolve()
LOWLEVEL_DIR = SCRIPT_PATH.parents[1]
OUTPUT_DIR = SCRIPT_PATH.parent / "outputs" / "shoulder_pan_home_candidates"
REFERENCE_FRONT = (
    LOWLEVEL_DIR
    / "recordings"
    / "tmp_home_shoulderpan_m180_frame_20260726"
    / "home_shoulderpan_m180_front.png"
)
REFERENCE_TOP = (
    LOWLEVEL_DIR
    / "recordings"
    / "tmp_home_shoulderpan_m180_frame_20260726"
    / "home_shoulderpan_m180_topdown.png"
)


def capture(eye, target, up):
    projection = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=640 / 360,
        nearVal=0.01,
        farVal=100,
    )
    view = p.computeViewMatrix(eye, target, up)
    width, height, rgba, _, _ = p.getCameraImage(
        640,
        360,
        viewMatrix=view,
        projectionMatrix=projection,
    )
    return np.array(rgba, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]


def pixel_mae(candidate, reference):
    candidate = candidate.astype(float)
    reference = reference.astype(float)
    return float(np.mean(np.abs(candidate - reference)))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if str(LOWLEVEL_DIR) not in sys.path:
        sys.path.insert(0, str(LOWLEVEL_DIR))

    import chess_traj
    import multisim_chess_fast as sim

    default_home = chess_traj.DEFAULT_HOME.copy()
    candidates = {
        "absolute_m180": -180.0,
        "default_minus_180": float(default_home[0] - 180.0),
        "negative_default_minus_180": float(-default_home[0] - 180.0),
        "negative_default": float(-default_home[0]),
    }
    reference_front = imageio.imread(REFERENCE_FRONT)[:, :, :3]
    reference_top = imageio.imread(REFERENCE_TOP)[:, :, :3]
    report = {}

    for name, shoulder_pan in candidates.items():
        home_joints = default_home.copy()
        home_joints[0] = shoulder_pan
        world = sim.setup_sim_world("b8", home_joints=home_joints)
        try:
            front = capture(
                [0.0, -0.6, 0.25],
                [0.3, 0.0, 0.05],
                [0, 0, 1],
            )
            top = capture(
                [0.3, 0.0, 0.6],
                [0.3, 0.0, 0.0],
                [0, -1, 0],
            )
        finally:
            p.removeState(world["state_id"])

        front_path = OUTPUT_DIR / f"{name}_front.png"
        top_path = OUTPUT_DIR / f"{name}_topdown.png"
        imageio.imwrite(front_path, front)
        imageio.imwrite(top_path, top)
        report[name] = {
            "home_joints_deg": home_joints.tolist(),
            "front_pixel_mae": pixel_mae(front, reference_front),
            "topdown_pixel_mae": pixel_mae(top, reference_top),
            "combined_pixel_mae": (
                pixel_mae(front, reference_front)
                + pixel_mae(top, reference_top)
            ),
            "front_path": str(front_path),
            "topdown_path": str(top_path),
        }

    report["best_match"] = min(
        candidates,
        key=lambda name: report[name]["combined_pixel_mae"],
    )
    (OUTPUT_DIR / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if p.isConnected():
        p.disconnect()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
