#!/usr/bin/env python3
"""Render a topdown board-frame diagnostic for the real_world2 board settings."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import imageio.v3 as iio
import numpy as np
import pybullet as p

LOWLEVEL_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = LOWLEVEL_DIR.parent
sys.path.insert(0, str(LOWLEVEL_DIR))

from board_coordinates import square_center_world_xy  # noqa: E402
from chess_traj import pickupmove_traj_with_metrics  # noqa: E402
import multisim_chess_fast as sim  # noqa: E402
from testkinematics import kinematics  # noqa: E402


OUTPUT_DIR = (
    LOWLEVEL_DIR
    / "rook_kiri_xy_lookup"
    / "real_world2_board_frame_debug_20260821"
)
SQUARE_SIZE = 0.04125
GRASP_OFFSET = np.array([-0.014, 0.002, -0.003], dtype=float)
PLACE_OFFSET = np.array([-0.011, 0.002, -0.003], dtype=float)


def add_marker(xyz: np.ndarray, color: list[float], radius: float = 0.004) -> int:
    visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    return p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual,
        basePosition=xyz.tolist(),
    )


def add_line_marker(start: np.ndarray, end: np.ndarray, color: list[float]) -> None:
    p.addUserDebugLine(start.tolist(), end.tolist(), color[:3], lineWidth=4)


def render_topdown(path: Path) -> None:
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=1.0,
        nearVal=0.01,
        farVal=100,
    )
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0.25, 0.0, 0.75],
        cameraTargetPosition=[0.25, 0.0, 0.03],
        cameraUpVector=[0, 1, 0],
    )
    width = height = 900
    _, _, rgba, _, _ = p.getCameraImage(
        width,
        height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
    )
    image = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
    iio.imwrite(path, image[:, :, :3])


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sim.ensure_physics_connected()
    start_square = "d4"
    target_square = "d5"
    world = sim.setup_sim_world(start_square, edge_support_margin=0.08)

    try:
        points = {
            "d4": np.array([*square_center_world_xy("d4", board_origin=sim.board_origin, square_size=SQUARE_SIZE), 0.095]),
            "d5": np.array([*square_center_world_xy("d5", board_origin=sim.board_origin, square_size=SQUARE_SIZE), 0.095]),
            "e4": np.array([*square_center_world_xy("e4", board_origin=sim.board_origin, square_size=SQUARE_SIZE), 0.095]),
        }
        add_marker(points["d4"], [1.0, 0.0, 0.0, 1.0], radius=0.006)
        add_marker(points["d5"], [0.0, 0.9, 0.0, 1.0], radius=0.006)
        add_marker(points["e4"], [0.0, 0.25, 1.0, 1.0], radius=0.006)

        from_xy = points["d4"][:2]
        to_xy = points["d5"][:2]
        right_xy = points["e4"][:2]
        add_line_marker(
            np.array([from_xy[0], from_xy[1], 0.09]),
            np.array([to_xy[0], to_xy[1], 0.09]),
            [0.0, 0.9, 0.0, 1.0],
        )
        add_line_marker(
            np.array([from_xy[0], from_xy[1], 0.085]),
            np.array([right_xy[0], right_xy[1], 0.085]),
            [0.0, 0.25, 1.0, 1.0],
        )

        movelist, closeidx, traj_metrics = pickupmove_traj_with_metrics(
            start_square,
            target_square,
            sim.board_origin,
            GRASP_OFFSET,
            PLACE_OFFSET,
            lift_height=0.13,
            placement_lower_steps=10,
        )
        close_joints = np.array(movelist[closeidx], dtype=float)
        close_pose = kinematics.forward_kinematics(close_joints)
        planned_grasp = close_pose[:3, 3] + close_pose[:3, :3] @ GRASP_OFFSET
        add_marker(planned_grasp + np.array([0.0, 0.0, 0.01]), [1.0, 0.0, 1.0, 1.0], radius=0.004)

        render_topdown(OUTPUT_DIR / "topdown_board_frame_markers.png")

        summary = {
            "board_origin": list(map(float, sim.board_origin)),
            "square_size": SQUARE_SIZE,
            "legend": {
                "red": "d4 pickup square center",
                "green": "d5 target square center and d4_to_d5 direction",
                "blue": "e4 square center and file-positive direction",
                "magenta": "planned grasp point at close index",
            },
            "points": {name: value.tolist() for name, value in points.items()},
            "planned_grasp_xyz": planned_grasp.tolist(),
            "planned_grasp_error_from_d4": (
                planned_grasp - np.array(traj_metrics["from_world_xyz"], dtype=float)
            ).tolist(),
            "closeidx": closeidx,
            "trajectory_fk_error": float(traj_metrics["max_fk_error"]),
        }
        (OUTPUT_DIR / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        p.removeState(world["state_id"])

    print(f"Saved diagnostic to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
