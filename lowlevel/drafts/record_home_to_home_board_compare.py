#!/usr/bin/env python3
"""Record default-home to alternate-home motion under old and new board settings."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
LOWLEVEL_DIR = SCRIPT_PATH.parents[1]
REPO_DIR = SCRIPT_PATH.parents[2]
DEFAULT_OUTPUT_DIR = (
    LOWLEVEL_DIR
    / "rook_kiri_xy_lookup"
    / "real_world2_home_to_home_board_compare_20260821"
)
PYTHON = Path(sys.executable)

PIECE_ENV = {
    "LOOKUP_PIECE_MODEL": "rook_kiri",
    "ROOK_KIRI_COLLISION_MODEL": "banded_hulls",
    "ROOK_KIRI_MESH_UP_AXIS": "y",
    "ROOK_KIRI_MESH_PATH": str(REPO_DIR / "rook_kiri2" / "rook2.obj"),
    "ROOK_KIRI_VISUAL_MESH_PATH": str(
        REPO_DIR / "rook_kiri2" / "rook2_debug_orange_visual.obj"
    ),
    "ROOK_KIRI_BAND_COLLISION_MESH_DIR": str(
        LOWLEVEL_DIR
        / "rook_kiri_lookup"
        / "collision_geometry_preview_20260801_163534"
    ),
}

CASES = (
    {
        "name": "01_old_board_z0_square4000",
        "board_origin_z": 0.0,
        "square_size": 0.04,
    },
    {
        "name": "02_new_board_z3cm_square4125",
        "board_origin_z": 0.03,
        "square_size": 0.04125,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-name")
    parser.add_argument("--board-origin-z", type=float)
    parser.add_argument("--square-size", type=float)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--edge-support-margin", type=float, default=0.08)
    return parser.parse_args()


def run_parent(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "description": "Default home to mirrored_compromise home, old board first then new board.",
        "cases": [],
    }

    for case in CASES:
        case_dir = output_dir / case["name"]
        env = os.environ.copy()
        env.update(PIECE_ENV)
        env["SIM_BOARD_ORIGIN_Z"] = str(case["board_origin_z"])
        env["SIM_BOARD_SQUARE_SIZE"] = str(case["square_size"])
        command = [
            str(PYTHON),
            "-B",
            str(SCRIPT_PATH),
            "--output-dir",
            str(output_dir),
            "--case-name",
            case["name"],
            "--board-origin-z",
            str(case["board_origin_z"]),
            "--square-size",
            str(case["square_size"]),
            "--steps",
            str(args.steps),
            "--settle-steps",
            str(args.settle_steps),
            "--edge-support-margin",
            str(args.edge_support_margin),
        ]
        subprocess.run(command, cwd=REPO_DIR, env=env, check=True)
        index["cases"].append(
            {
                **case,
                "directory": str(case_dir),
                "front_video": str(
                    case_dir
                    / "home_to_mirrored_compromise"
                    / "so101_robot_moves.mp4"
                ),
                "topdown_video": str(
                    case_dir
                    / "home_to_mirrored_compromise"
                    / "so101_robot_moves_topdown.mp4"
                ),
                "summary": str(case_dir / "summary.json"),
            }
        )

    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Home-to-home comparison saved: {output_dir}")
    return 0


def run_case(args: argparse.Namespace) -> int:
    if str(LOWLEVEL_DIR) not in sys.path:
        sys.path.insert(0, str(LOWLEVEL_DIR))

    import numpy as np
    import pybullet as p

    import multisim_chess_fast as sim
    from chess_traj import DEFAULT_HOME

    output_dir = args.output_dir.expanduser().resolve() / args.case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    alternate_home = np.array(
        [
            -96.937906118107,
            -108.574743187855,
            97.506932803653,
            65.749431985183,
            35.411444525975,
            4.62962962962963,
        ],
        dtype=float,
    )
    joint_map = (0, 1, 2, 3, 4, 6)
    default_home_rad = np.deg2rad(DEFAULT_HOME)
    alternate_home_rad = np.deg2rad(alternate_home)

    sim.ensure_physics_connected()
    world = sim.setup_sim_world(
        "d4",
        edge_support_margin=args.edge_support_margin,
        home_joints=DEFAULT_HOME,
    )
    video_context = sim.create_video_context(output_dir / "home_to_mirrored_compromise")
    contact_samples = []
    max_joint_error_rad = 0.0
    final_joint_positions_rad = None
    try:
        robot_id = world["robot_id"]
        board_body_names = {
            world["board_base_id"]: "board_base",
            **{
                support_id: f"edge_support_{index}"
                for index, support_id in enumerate(world["edge_support_ids"])
            },
        }
        board_body_ids = tuple(board_body_names)
        for step in range(1, args.steps + 1):
            alpha = step / args.steps
            target = (1.0 - alpha) * default_home_rad + alpha * alternate_home_rad
            for traj_idx, sim_idx in enumerate(joint_map):
                force = 500 if traj_idx == 5 else 50
                p.setJointMotorControl2(
                    robot_id,
                    sim_idx,
                    p.POSITION_CONTROL,
                    targetPosition=target[traj_idx],
                    force=force,
                )
            p.stepSimulation()
            actual = np.array(
                [p.getJointState(robot_id, sim_idx)[0] for sim_idx in joint_map],
                dtype=float,
            )
            max_joint_error_rad = max(
                max_joint_error_rad,
                float(np.max(np.abs(actual - target))),
            )
            if step % 10 == 0:
                contact_details = []
                for board_body_id in board_body_ids:
                    for contact in p.getContactPoints(robot_id, board_body_id):
                        contact_details.append(
                            {
                                "board_body": board_body_names[board_body_id],
                                "robot_link_index": int(contact[3]),
                                "board_link_index": int(contact[4]),
                                "distance": float(contact[8]),
                                "normal_force": float(contact[9]),
                            }
                        )
                if contact_details:
                    contact_samples.append(
                        {
                            "step": step,
                            "contact_count": len(contact_details),
                            "contacts": contact_details[:8],
                        }
                    )
            if step % sim.renderfreq == 0:
                sim.append_video_frame(video_context)

        for step in range(args.settle_steps):
            p.stepSimulation()
            if step % sim.renderfreq == 0:
                sim.append_video_frame(video_context)
        final_joint_positions_rad = np.array(
            [p.getJointState(robot_id, sim_idx)[0] for sim_idx in joint_map],
            dtype=float,
        )
    finally:
        sim.close_video_context(video_context)
        p.removeState(world["state_id"])

    summary = {
        "case_name": args.case_name,
        "board_origin": list(sim.board_origin),
        "square_size": sim.BOARD_SQUARE_SIZE,
        "start_home_name": "default",
        "start_home_deg": DEFAULT_HOME.tolist(),
        "end_home_name": "mirrored_compromise",
        "end_home_deg": alternate_home.tolist(),
        "steps": args.steps,
        "settle_steps": args.settle_steps,
        "edge_support_margin": args.edge_support_margin,
        "front_video": str(
            output_dir / "home_to_mirrored_compromise" / "so101_robot_moves.mp4"
        ),
        "topdown_video": str(
            output_dir
            / "home_to_mirrored_compromise"
            / "so101_robot_moves_topdown.mp4"
        ),
        "max_joint_error_deg": float(np.rad2deg(max_joint_error_rad)),
        "final_joint_positions_deg": (
            None
            if final_joint_positions_rad is None
            else np.rad2deg(final_joint_positions_rad).tolist()
        ),
        "final_joint_error_deg": (
            None
            if final_joint_positions_rad is None
            else np.rad2deg(final_joint_positions_rad - alternate_home_rad).tolist()
        ),
        "contact_samples": contact_samples[:50],
        "contact_sample_count": len(contact_samples),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Recorded {args.case_name}: {output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    if args.case_name is None:
        return run_parent(args)
    return run_case(args)


if __name__ == "__main__":
    raise SystemExit(main())
