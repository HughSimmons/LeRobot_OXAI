#!/usr/bin/env python3
"""Dry-run-by-default SO-101 trajectory runner for the ``dls_nullspace`` planner.

Algebraic squares or ``coord_finder`` continuous 3D piece detections ->
Coupled Attitude-Matched IK -> DLS + Null-Space SDF trajectory planning ->
safety gate -> optional calibration-offset / friction-sag compensation ->
paced execution on ``SO101Follower`` (``planner.hardware``).

For clicks from Oliver's ``click_to_3d_oliver`` tool use the ``d`` hotkey instead (see
``planner/README.md``, "Running on Oliver's hardware"); this CLI is for squares / grasp JSON.

Safety Gates:
1. Dry-run by default (never moves hardware unless ``--execute`` is explicitly passed).
2. Verifies Coupled IK convergence (< 2.0 mm), matched piece release tilt, joint bounds,
   and analytical Capsule-SDF clearance across standing board pieces & held piece.
3. Reads live follower joint angles on connect and replans from the robot's exact current
   configuration (compensation-aware, so the first command equals the reading).
4. On error or Ctrl-C the arm keeps holding (torque on) until the operator presses Enter.

(There is no PyBullet physics rollout here; ``planner_benchmarks`` has the simulator-based
evaluation.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

LOWLEVEL_DIR = Path(__file__).resolve().parents[1]
if str(LOWLEVEL_DIR) not in sys.path:
    sys.path.insert(0, str(LOWLEVEL_DIR))

from planner import (  # noqa: E402
    ALL_JOINT_NAMES,
    BoardConfig,
    CaptureSequenceResult,
    DEFAULT_HOME_DEG,
    DLSNullspacePlanner,
    GRIPPER_CLOSED_DEG,
    GRIPPER_OPEN_DEG,
    MoveRequest,
    PieceCylinder,
    PlannedTrajectory,
    REAL_MEASURED_BOARD_CONFIG,
    SIM_BOARD_CONFIG,
    SO101Kinematics,
    SceneCollisionChecker,
    build_pieces_from_coord_finder_detections,
    build_pieces_from_square_map,
    grasp_dict_to_world_target,
    plan_move_or_capture,
)
from planner import hardware as hw  # noqa: E402
from planner.safety import safety_gate, tilt_notices  # noqa: E402


def parse_board_config(preset: str, origin_override: Optional[List[float]], square_mm: Optional[float]) -> BoardConfig:
    base = REAL_MEASURED_BOARD_CONFIG if preset == "real" else SIM_BOARD_CONFIG
    origin = tuple(origin_override) if origin_override is not None else base.origin_xyz_m
    sq_m = (square_mm / 1000.0) if square_mm is not None else base.square_size_m
    return BoardConfig(
        origin_xyz_m=(float(origin[0]), float(origin[1]), float(origin[2])),
        square_size_m=float(sq_m),
        slab_half_height_m=base.slab_half_height_m,
        border_margin_m=base.border_margin_m,
        name=preset,
    )


def load_obstacles(
    board_json: Optional[Path],
    board: BoardConfig,
    use_staunton_heights: bool,
) -> List[PieceCylinder]:
    """Load standing piece obstacles from either a square->symbol map or coord_finder list."""
    if board_json is None:
        return []
    payload = json.loads(board_json.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "pieces" in payload and isinstance(payload["pieces"], list):
        return build_pieces_from_coord_finder_detections(payload["pieces"], board)
    if isinstance(payload, list):
        return build_pieces_from_coord_finder_detections(payload, board)
    if isinstance(payload, dict):
        sq_map = {str(k): str(v) for k, v in payload.items()}
        return build_pieces_from_square_map(sq_map, board, use_staunton_heights=use_staunton_heights)
    raise ValueError(f"Unsupported board JSON format in {board_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan, gate, compensate, and (with --execute) run an SO-101 chess move.")
    parser.add_argument("--from-sq", type=str, default="e2", help="Source square (e.g. e2) unless --from-grasp-json is set.")
    parser.add_argument("--to-sq", type=str, default="e4", help="Destination square (e.g. e4) unless --to-grasp-json is set.")
    parser.add_argument("--from-grasp-json", type=Path, default=None, help="JSON output from coord_finder two_view_grasp.")
    parser.add_argument("--to-grasp-json", type=Path, default=None, help="Optional continuous destination JSON.")
    parser.add_argument("--board-json", type=Path, default=None, help="JSON map of standing board pieces (square->symbol or coord_finder list).")
    parser.add_argument("--piece", type=str, default="P", help="Moving piece FEN symbol (P, N, B, R, Q, K).")
    parser.add_argument("--capture", action="store_true", help="Execute 2-leg capture to off-board graveyard first.")
    parser.add_argument("--captured-piece", type=str, default="p", help="Captured piece FEN symbol.")
    parser.add_argument("--graveyard-slot", type=int, default=0, help="Off-board graveyard slot index (0..15).")
    parser.add_argument(
        "--board-preset",
        choices=["real", "sim"],
        default="real",
        help="Board geometry preset ('real' = 292.6 mm / 41.25 mm; 'sim' = 250 mm / 40 mm).",
    )
    parser.add_argument("--board-origin", type=float, nargs=3, default=None, help="Override board origin XYZ in metres.")
    parser.add_argument("--square-mm", type=float, default=None, help="Override square size in mm.")
    parser.add_argument("--staunton-heights", action="store_true", default=True, help="Use standard Staunton piece heights.")
    parser.add_argument(
        "--compensation",
        choices=list(hw.COMPENSATION_MODES),
        default=None,
        help="off = send planned angles; baseline = calibration offsets only (default); on = offsets + "
        "friction/sag (uses real_world_runs/friction/friction_sag_fit.json when present).",
    )
    parser.add_argument("--compensate", dest="compensation", action="store_const", const="on", help="Alias of --compensation on.")
    parser.add_argument("--no-compensate", dest="compensation", action="store_const", const="off", help="Alias of --compensation off.")
    parser.add_argument("--control-hz", type=float, default=hw.DEFAULT_CONTROL_HZ, help="Real hardware streaming rate (Hz).")
    parser.add_argument("--execute", action="store_true", help="Connect to real SO101Follower and execute move.")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation prompt when --execute is set.")
    parser.add_argument("--save-trajectory", type=Path, default=None, help="Optional .npy path to save planned dense commands.")
    args = parser.parse_args()

    board = parse_board_config(args.board_preset, args.board_origin, args.square_mm)
    kin = SO101Kinematics()
    pieces = load_obstacles(args.board_json, board, args.staunton_heights)
    checker = SceneCollisionChecker(kin, board, pieces)
    planner = DLSNullspacePlanner(kin, board)

    pick_yaw_deg: Optional[float] = None
    if args.from_grasp_json is not None:
        g_dict = json.loads(args.from_grasp_json.read_text(encoding="utf-8"))
        from_target, pick_yaw_deg = grasp_dict_to_world_target(g_dict, board=board)
    else:
        from_target = args.from_sq

    if args.to_grasp_json is not None:
        t_dict = json.loads(args.to_grasp_json.read_text(encoding="utf-8"))
        to_target, _ = grasp_dict_to_world_target(t_dict, board=board)
    else:
        to_target = args.to_sq

    mode = args.compensation or "baseline"
    compensator = hw.make_compensator(mode)

    follower = None
    start_joints = DEFAULT_HOME_DEG.copy()
    finished = False

    try:
        if args.execute:
            print("Connecting to real SO-101 follower arm...")
            follower = hw.connect_follower()
            reading = hw.read_joints_deg(follower)
            start_joints = hw.start_urdf_from_reading(reading, compensator)
            print(f"Live start joints (deg): {np.round(reading, 2).tolist()}")

        req = MoveRequest(
            from_target=from_target,
            to_target=to_target,
            moving_piece_symbol=args.piece,
            pick_yaw_deg=pick_yaw_deg,
            is_capture=args.capture,
            captured_piece_symbol=args.captured_piece,
            graveyard_slot_index=args.graveyard_slot,
            start_joints_deg=start_joints,
            end_joints_deg=DEFAULT_HOME_DEG.copy(),
            gripper_open_deg=GRIPPER_OPEN_DEG,
            gripper_closed_deg=GRIPPER_CLOSED_DEG,
        )

        plan_result = plan_move_or_capture(planner, req, checker, dt_s=hw.DEFAULT_SOURCE_DT_S)
        raw_cmds = hw.compensate_plan(plan_result, None, "off")
        final_cmds = hw.compensate_plan(plan_result, compensator, mode)
        if mode != "off":
            max_comp_deg = float(np.max(np.abs(final_cmds[:, :5] - raw_cmds[:, :5])))
            print(f"Compensation '{mode}' applied (max delta = {max_comp_deg:.2f}°).")

        safe_ok, _ = safety_gate(plan_result, raw_cmds, kin)
        for note in tilt_notices(plan_result):
            print(f"  [note] {note}")

        if args.save_trajectory is not None:
            args.save_trajectory.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.save_trajectory, final_cmds)
            print(f"Saved {final_cmds.shape[0]}x6 trajectory commands to {args.save_trajectory}")

        if not args.execute:
            print("\nDry-run complete (hardware NOT moved). Pass --execute to run on SO101Follower.")
            finished = True
            return

        if not safe_ok:
            raise RuntimeError("Refusing hardware execution because Safety Gate reported warnings.")

        if not args.yes:
            ans = input("\nProceed with real SO-101 motion? [y/N]: ").strip().lower()
            if ans not in ("y", "yes"):
                print("Aborted by user.")
                finished = True  # nothing moved: release normally
                return

        assert follower is not None
        print(f"Streaming trajectory at {args.control_hz:.0f} Hz...")
        run_dir = hw.pcompat.real_world_runs_dir() / "planned_moves" / time.strftime("%Y%m%d-%H%M%S")
        follower_to_release, follower = follower, None  # execute_commands releases the arm itself
        hw.execute_commands(
            follower_to_release,
            final_cmds,
            control_hz=args.control_hz,
            run_dir=run_dir,
            metadata={"from": str(from_target), "to": str(to_target), "piece": args.piece, "compensation": mode},
        )
        print("Move execution complete.")
    finally:
        if follower is not None:
            hw.release_arm(follower, finished=finished)


if __name__ == "__main__":
    main()
