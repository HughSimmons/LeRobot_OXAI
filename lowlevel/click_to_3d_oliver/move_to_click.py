#!/usr/bin/env python3
"""Move the SO-101's fixed-jaw tip to a clicked point by running lowlevel/probe_xy_pose.py.

A click's base_mm is its board position shifted by the ruler-measured board centre
(click3d_shared.board_to_base_mm), or, after b in the click tool, the camera-detected board
(base_mm_cv: wrist camera c pose + forward kinematics + mount). Either is in the robot base frame that
probe_xy_pose.py's --xy uses, in mm rather than metres. probe_xy_pose.py runs as a subprocess, unchanged: IK straight
down at the point, up-and-over via waypoint, optional --compensate, its own yes/no prompt,
then return to DEFAULT_HOME.

Each move first runs probe as a dry run. Nothing connects to the arm unless the dry run
solved the exact target, within reach and pointing straight down. Every move is saved
under lowlevel/real_world_runs/click_moves/<time>_<square>_<mode>/: click.json (the click and the
plan) and probe_xy_pose.json (probe's record, with the joint readings after the move).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from click3d_shared import (
    BOARD_BASE_SOURCE,
    BOARD_CENTRE_BASE_MM,
    CV_BASE_SOURCE,
    LEGACY_FK_BASE_SOURCE,
    OXAI_PATHS_HINT,
    SQUARE_MM,
    _ensure_lowlevel_on_path,
    board_to_base_mm,
    close_arm,
    square_file_rank,
)

# The playing surface in the robot base: the floor no target may go below.
BOARD_SURFACE_Z_MM = BOARD_CENTRE_BASE_MM[2]
# probe_xy_pose.py puts the target at its own board surface (default --board-origin-z 0.0196
# + 0.005) + --height-above-surface, so heights are passed relative to that. The dry run
# checks that the target it solved is the one asked for, so a changed probe default is
# caught rather than silently shifting z.
PROBE_SURFACE_Z_MM = 24.6
PROBE_VIA_HEIGHT_MM = 120.0  # probe's default --via-height, above its board surface
VIA_CLEARANCE_MM = 50.0  # via at least this far above a raised target
# A target this far below the board surface is refused; within it, it is raised to the
# surface so the jaw never aims into the board.
MAX_BELOW_SURFACE_MM = 5.0
MAX_ABOVE_SURFACE_MM = 200.0  # well above any piece
# Pick-and-place (p): the sim piece for build_general_xy_lookup.py and run_real_so101 (they
# must match), and the builder's grasp search size: 0 tries one grasp (~35 s), 1 searches more.
PICK_SIM_ENV = {"LOOKUP_PIECE_MODEL": "rook_kiri", "ROOK_KIRI_COLLISION_MODEL": "banded_hulls"}
PICK_GRID_RADIUS = 0
# Seconds p takes to close the gripper on the piece, arm held still (run_real's --gripper-close-s).
PICK_GRIPPER_CLOSE_S = 1.0
# Seconds p takes to swing back to rest after the place (run_real's --return-s); faster rocks
# the base on its clamps.
PICK_RETURN_S = 4.0
# The sim's arm normally misses by the real arm's offsets and friction band (multisim_chess_fast.py).
# p plans in an ideal sim instead (angle sent = angle reached), so one trajectory serves both
# runs and --compensate only changes the commands sent to the real arm.
EXACT_TRACKING_ENV = {
    name: "0 0 0 0 0"
    for name in ("SIM_JOINT_OFFSETS_DEG", "SIM_JOINT_FRICTION_DEG")
}
# probe_xy_pose.py's --max-residual-mm / --max-tilt-deg defaults (its reachability test).
MAX_IK_RESIDUAL_MM = 2.0
MAX_TILT_DEG = 2.0
TARGET_MATCH_MM = 0.01


class MoveRefused(Exception):
    """The click can't be moved to safely; the message says why and what to do."""


@dataclass
class MovePlan:
    target_mm: np.ndarray  # fixed-jaw tip target, base frame
    height_above_board_mm: float  # above BOARD_SURFACE_Z_MM
    probe_height_mm: float  # probe's --height-above-surface (relative to PROBE_SURFACE_Z_MM)
    via_height_mm: float  # probe's --via-height (relative to PROBE_SURFACE_Z_MM)
    square: str
    notes: list[str] = field(default_factory=list)


@dataclass
class MoveResult:
    arm_moved: bool  # the execute run was started, so the arm may have moved
    message: str
    output_dir: Path | None = None  # the run's folder, where the click tool saves its tripod video


@dataclass
class PickTargets:
    """Where p picks the piece from and places it, in the route chosen with b."""

    source: str  # BOARD_BASE_SOURCE (ruler) or CV_BASE_SOURCE (camera-detected board)
    from_base_xy_mm: np.ndarray  # piece, robot base frame
    to_base_xy_mm: np.ndarray  # destination square centre, robot base frame
    from_board_xy_m: tuple[float, float]  # handed to the lookup builder: metres from the ruler board centre
    to_board_xy_m: tuple[float, float]

    def record(self) -> dict[str, Any]:
        return {
            "target_source": self.source,
            "from_target_base_xy_mm": [round(float(v), 3) for v in self.from_base_xy_mm],
            "to_target_base_xy_mm": [round(float(v), 3) for v in self.to_base_xy_mm],
            "lookup_from_board_xy_m": [round(float(v), 6) for v in self.from_board_xy_m],
            "lookup_to_board_xy_m": [round(float(v), 6) for v in self.to_board_xy_m],
        }


def pick_targets(from_click: dict[str, Any], to_click: dict[str, Any]) -> PickTargets:
    """p's piece and destination x, y, from the route b is on (the clicks' base_source).

    ruler: the piece click's board position and the destination square's centre on the
    ruler-measured board (what p always did).
    cv: the same two board points mapped through each click's camera-detected board pose
    (T_base_board_cv_mm).

    Only x, y change: the board height stays the fixed BOARD_CENTRE_BASE_MM z (29 mm). They go to
    the lookup builder as metres from the ruler board centre, the frame the real-arm runner places
    with --board-origin, so the arm targets exactly these base-frame x, y.
    """
    cv = to_click.get("base_source") in (CV_BASE_SOURCE, LEGACY_FK_BASE_SOURCE)
    file_i, rank_i = square_file_rank(to_click["square"])
    centre_board = np.array([(file_i + 0.5) * SQUARE_MM, (rank_i + 0.5) * SQUARE_MM, 0.0])
    piece_board = np.array([*np.asarray(from_click["board_mm"], dtype=np.float64)[:2], 0.0])
    if cv:
        poses = [click.get("T_base_board_cv_mm", click.get("T_base_board_fk_mm")) for click in (from_click, to_click)]
        if any(T is None for T in poses):
            raise MoveRefused("these clicks have no camera board pose (made before p followed b): click them again")
        from_base = (np.asarray(poses[0], dtype=np.float64) @ np.append(piece_board, 1.0))[:2]
        to_base = (np.asarray(poses[1], dtype=np.float64) @ np.append(centre_board, 1.0))[:2]
    else:
        from_base, to_base = board_to_base_mm(piece_board)[:2], board_to_base_mm(centre_board)[:2]
    centre_xy = np.asarray(BOARD_CENTRE_BASE_MM[:2], dtype=np.float64)
    return PickTargets(
        source=CV_BASE_SOURCE if cv else BOARD_BASE_SOURCE,
        from_base_xy_mm=from_base,
        to_base_xy_mm=to_base,
        from_board_xy_m=tuple(float(v) for v in (from_base - centre_xy) / 1000.0),
        to_board_xy_m=tuple(float(v) for v in (to_base - centre_xy) / 1000.0),
    )


def mode_suffix(compensate: bool, baseline: bool = False) -> str:
    """The run folder's suffix: which compensation mode the arm ran in."""
    return "baseline" if baseline else "compensate" if compensate else "off"


def plan_move(click: dict[str, Any], above_mm: float = 0.0) -> MovePlan:
    """Target for a click record from the session log; raises MoveRefused if unsafe."""
    # Clicks logged before both routes were stored (or with the old f=1400 lens) have neither
    # source; the floor and height checks below apply to both routes alike.
    if click.get("base_source") not in (BOARD_BASE_SOURCE, CV_BASE_SOURCE, LEGACY_FK_BASE_SOURCE):
        raise MoveRefused(
            "this click's base_mm has no known source (an old click, from before the "
            "ruler / camera+kinematics choice). Click it again (space)."
        )
    base = np.asarray(click.get("base_mm", []), dtype=np.float64)
    if base.shape != (3,) or not np.all(np.isfinite(base)):
        raise MoveRefused(f"click has no valid base_mm: {click.get('base_mm')}")

    target = base.copy()
    target[2] += above_mm
    notes = []
    below = BOARD_SURFACE_Z_MM - target[2]
    if below > MAX_BELOW_SURFACE_MM:
        raise MoveRefused(
            f"target z {target[2]:.1f} mm is {below:.1f} mm below the board surface "
            f"({BOARD_SURFACE_Z_MM:.1f} mm). Check the click, or raise it with --above-mm."
        )
    if below > 0:
        notes.append(f"target raised {below:.1f} mm to the board surface (it was below it)")
        target[2] = BOARD_SURFACE_Z_MM
    height = float(target[2] - BOARD_SURFACE_Z_MM)
    if height > MAX_ABOVE_SURFACE_MM:
        raise MoveRefused(
            f"target is {height:.0f} mm above the board, higher than any piece. "
            "Check the click (and its ray gap)."
        )
    probe_height = float(target[2] - PROBE_SURFACE_Z_MM)
    return MovePlan(
        target_mm=target,
        height_above_board_mm=height,
        probe_height_mm=probe_height,
        via_height_mm=max(PROBE_VIA_HEIGHT_MM, probe_height + VIA_CLEARANCE_MM),
        square=str(click.get("square", "off")),
        notes=notes,
    )


def probe_command(
    plan: MovePlan, *, lowlevel: Path, output_dir: Path, compensate: bool, execute: bool
) -> list[str]:
    """probe_xy_pose.py arguments that put the fixed-jaw tip at plan.target_mm."""
    port = os.environ.get("SO101_PORT")
    robot_id = os.environ.get("SO101_ID")
    if port is None or robot_id is None:
        import paths

        port = port or paths.SO101_PORT
        robot_id = robot_id or paths.SO101_ID
    x_m, y_m = plan.target_mm[:2] / 1000.0
    cmd = [
        sys.executable,
        str(lowlevel / "probe_xy_pose.py"),
        "--xy", f"{x_m:.6f}", f"{y_m:.6f}",
        f"--height-above-surface={plan.probe_height_mm / 1000.0:.6f}",
        f"--via-height={plan.via_height_mm / 1000.0:.6f}",
        "--output-dir", str(output_dir),
        "--port", port,
        "--robot-id", robot_id,
    ]
    if compensate:
        cmd.append("--compensate")
    if execute:
        cmd.append("--execute")
    return cmd

def pick_piece_command(
    lookup_json: Path, *, lowlevel: Path, output_dir: Path, compensate: bool, baseline: bool, execute: bool
) -> list[str]:
    """run_real_so101_from_xy_lookup.py arguments that play lookup_json on the measured board."""
    import paths

    cmd = [
        sys.executable,
        str(lowlevel / "run_real_so101_from_xy_lookup.py"),
        "--lookup-json", str(lookup_json),
        f"--board-origin-x={BOARD_CENTRE_BASE_MM[0] / 1000.0:.6f}",
        f"--board-origin-y={BOARD_CENTRE_BASE_MM[1] / 1000.0:.6f}",
        # run_real's z is the centre of the 10 mm sim board slab: 5 mm below its top.
        f"--board-origin-z={(BOARD_CENTRE_BASE_MM[2] - 5.0) / 1000.0:.6f}",
        "--output-dir", str(output_dir),
        "--port", os.environ.get("SO101_PORT", paths.SO101_PORT),
        "--robot-id", os.environ.get("SO101_ID", paths.SO101_ID),
        # The arm starts wherever it was parked for the clicks: bridge to the trajectory
        # start (run_real asks first) over 1000 commands, a few seconds, not its 150 (~0.6 s).
        "--allow-start-bridge", "--start-bridge-steps", "1000",
        # Skip the trajectory's opening at table height beside the robot: bridge straight to
        # waypoint 3, already ~7 cm up, so the gripper doesn't drag on the way to the start.
        "--start-waypoint", "3",
        "--gripper-close-s", str(PICK_GRIPPER_CLOSE_S),
        "--return-s", str(PICK_RETURN_S),
    ]
    if compensate:
        cmd.append("--compensate")
    if baseline:
        cmd.append("--baseline")  # hand-measured offsets only, to compare --compensate against
    if execute:
        cmd += ["--execute", "--yes"]
    return cmd


def check_dry_run(record_path: Path, plan: MovePlan) -> None:
    """Raise MoveRefused unless probe's dry run solved plan's target, in reach and vertical."""
    if not record_path.is_file():
        raise MoveRefused(f"probe dry run wrote no record at {record_path}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    solved_mm = np.asarray(record["target_world_xyz"], dtype=np.float64) * 1000.0
    mismatch = float(np.max(np.abs(solved_mm - plan.target_mm)))
    if mismatch > TARGET_MATCH_MM:
        raise MoveRefused(
            f"probe aimed at {np.round(solved_mm, 2).tolist()} mm, not {np.round(plan.target_mm, 2).tolist()} mm "
            f"({mismatch:.2f} mm off). Has probe_xy_pose.py's --board-origin-z default changed? "
            "Update PROBE_SURFACE_Z_MM in move_to_click.py to match."
        )
    residual_mm = float(record["ik_residual_m"]) * 1000.0
    tilt = float(record["off_vertical_deg"])
    if residual_mm > MAX_IK_RESIDUAL_MM or tilt > MAX_TILT_DEG:
        raise MoveRefused(
            f"out of reach: the closest straight-down pose is {residual_mm:.1f} mm from the point "
            f"and {tilt:.1f} deg off vertical (limits {MAX_IK_RESIDUAL_MM:.0f} mm, {MAX_TILT_DEG:.0f} deg)"
        )


def run_probe(cmd: list[str], lowlevel: Path, extra_env: dict[str, str] | None = None) -> int:
    env = {**os.environ, **(extra_env or {})}
    lerobot_src = str(lowlevel / "lerobot" / "src")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (lerobot_src, env.get("PYTHONPATH")) if p)
    print("running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(lowlevel), env=env)


def move_to_click(
    click: dict[str, Any], *, above_mm: float, compensate: bool, robot: bool, on_move=None
) -> MoveResult:
    """Plan, dry-run, and (with robot) run probe_xy_pose.py --execute for one click.

    Prompts in the terminal. Closes the click tool's serial connection before probe opens
    its own; the next joint read reopens it.
    """
    lowlevel = _ensure_lowlevel_on_path()
    if lowlevel is None:
        return MoveResult(False, f"move unavailable: {OXAI_PATHS_HINT}")
    try:
        plan = plan_move(click, above_mm)
    except MoveRefused as exc:
        print(f"\nMOVE REFUSED: {exc}")
        return MoveResult(False, f"move refused: {exc}")

    import paths

    stamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = paths.REAL_WORLD_RUNS_DIR / "click_moves" / f"{stamp}_{re.sub(r'[^A-Za-z0-9]', '_', plan.square)}_{mode_suffix(compensate)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "click.json").write_text(
        json.dumps(
            {
                "click": click,
                "target_mm": [round(float(v), 3) for v in plan.target_mm],
                "above_mm": above_mm,
                "height_above_board_mm": round(plan.height_above_board_mm, 3),
                "board_surface_z_mm": BOARD_SURFACE_Z_MM,
                "via_height_mm": round(plan.via_height_mm, 3),
                "compensate": compensate,
                "notes": plan.notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    x, y, z = plan.target_mm
    print("\n" + "=" * 72)
    print(f"MOVE TO CLICK ({plan.square}): fixed-jaw tip to base ({x:.1f}, {y:.1f}, {z:.1f}) mm,")
    print(f"  {plan.height_above_board_mm:.1f} mm above the board surface (z {BOARD_SURFACE_Z_MM:.1f} mm), pointing straight down.")
    print(f"  compensate: {'on' if compensate else 'off'}   saving to {output_dir}")
    for note in plan.notes:
        print(f"  {note}")
    print("Step 1/2: dry run (no arm), about 30 s ...")
    print("=" * 72, flush=True)
    dry_cmd = probe_command(plan, lowlevel=lowlevel, output_dir=output_dir, compensate=compensate, execute=False)
    if run_probe(dry_cmd, lowlevel) != 0:
        print("\nMOVE REFUSED: the probe dry run failed (see the error above).")
        return MoveResult(False, "move refused: probe dry run failed (see terminal)")
    try:
        check_dry_run(output_dir / "probe_xy_pose.json", plan)
    except MoveRefused as exc:
        print(f"\nMOVE REFUSED: {exc}")
        return MoveResult(False, f"move refused: {exc}")

    if not robot:
        print("\nDry run OK. No --robot, so the arm is not moved.")
        return MoveResult(False, f"dry run ok for {plan.square} (no --robot: arm not moved)")

    print("\nDry run OK. Step 2/2: move the arm.")
    print("  Connecting turns torque off for a moment while the servos are configured, so")
    print("  support the arm. probe then asks 'Move to probe pose? [yes/no]' before it moves.")
    try:
        answer = input("Press Enter to connect, or type n then Enter to cancel: ").strip().lower()
    except EOFError:
        answer = "n"
    if answer in {"n", "no", "q"}:
        print("Cancelled; the arm was not touched.")
        return MoveResult(False, "move cancelled")

    close_arm()  # probe opens the port itself
    exec_cmd = probe_command(plan, lowlevel=lowlevel, output_dir=output_dir, compensate=compensate, execute=True)
    if on_move is not None:
        on_move()  # the arm is about to move (the click tool starts its tripod video here)
    code = run_probe(exec_cmd, lowlevel)
    if code == 0:
        message = f"moved to {plan.square} and back; saved {output_dir.name}"
        """
        This just clips the commanded coordinates to a square, the actual command is not 
        going to the square
        """
    else:
        message = f"probe exited with code {code} (cancelled or failed; see terminal)"
    print(f"\n{message}")
    return MoveResult(True, message, output_dir)


def centre_error_mm(board_mm: list[float]) -> dict[str, Any]:
    """A click's x / y offset from the nearest square's centre (board frame, mm; height ignored).

    Off the board, the nearest edge square. For a piece centred on its square, the click's error.
    The v scene view draws the same thing.
    """
    x, y = float(board_mm[0]), float(board_mm[1])
    file_i = min(7, max(0, int(np.floor(x / SQUARE_MM))))
    rank_i = min(7, max(0, int(np.floor(y / SQUARE_MM))))
    dx, dy = x - (file_i + 0.5) * SQUARE_MM, y - (rank_i + 0.5) * SQUARE_MM
    return {
        "square": "abcdefgh"[file_i] + str(rank_i + 1),
        "dx": round(dx, 3),
        "dy": round(dy, 3),
        "dist": round(float(np.hypot(dx, dy)), 3),
    }


def pick_piece(
    from_click: dict[str, Any], to_click: dict[str, Any], *, compensate: bool, baseline: bool, robot: bool, on_move=None
) -> MoveResult:
    """Pick the piece at from_click and place it on the centre of to_click's square.

    The two x, y come from the route b is on (pick_targets): the ruler-measured board, or the
    camera-detected board (cv). Builds a lookup (build_general_xy_lookup.py, board frame: metres
    from the ruler board centre), then runs it with run_real_so101_from_xy_lookup.py: --execute
    with robot (it plans and checks the commands before moving, so there is no separate dry run),
    a dry run without. After the arm run, metadata.json records which route it used.
    """
    lowlevel = _ensure_lowlevel_on_path()
    if lowlevel is None:
        return MoveResult(False, f"pick unavailable: {OXAI_PATHS_HINT}")
    import paths

    to_square = to_click["square"]
    if to_square == "off":
        return MoveResult(False, "pick refused: the destination click is off the board")
    try:
        targets = pick_targets(from_click, to_click)
    except MoveRefused as exc:
        return MoveResult(False, f"pick refused: {exc}")
    from_xy, to_xy = targets.from_board_xy_m, targets.to_board_xy_m
    route = "cv" if targets.source == CV_BASE_SOURCE else "ruler"
    from_name = f"{from_click['square']}_click"
    suffix = mode_suffix(compensate, baseline) + ("_cv" if route == "cv" else "")
    output_dir = paths.REAL_WORLD_RUNS_DIR / "click_picks" / f"{time.strftime('%Y%m%d-%H%M%S')}_{from_name}_to_{to_square}_{suffix}"
    lookup_dir = output_dir / "lookup"  # the builder refuses an existing directory
    sim_env = {**PICK_SIM_ENV, **EXACT_TRACKING_ENV}

    print("\n" + "=" * 72)
    print(f"PICK ({route}) {from_name} {np.round(from_xy, 4).tolist()} -> {to_square} centre {np.round(to_xy, 4).tolist()} (m from board centre)")
    print(f"Step 1/2: build the lookup in the ideal sim (~35 s with --grid-radius {PICK_GRID_RADIUS}). Saving to {output_dir}")
    print("=" * 72, flush=True)
    build = [
        sys.executable, str(lowlevel / "build_general_xy_lookup.py"),
        "--from-xy", f"{from_xy[0]:.6f}", f"{from_xy[1]:.6f}",
        "--to-xy", f"{to_xy[0]:.6f}", f"{to_xy[1]:.6f}",
        "--frame", "board",
        "--from-name", from_name, "--to-name", to_square,
        "--output-dir", str(lookup_dir),
        "--grid-radius", str(PICK_GRID_RADIUS),
    ]
    code = run_probe(build, lowlevel, sim_env)
    if code != 0:
        why = "the pick failed in sim" if code == 2 else "the builder errored"
        print(f"\nPICK REFUSED: {why} (exit {code}). A larger PICK_GRID_RADIUS searches more grasps.")
        return MoveResult(False, f"pick refused: {why} (exit {code}, see terminal)")
    lookup_json = lookup_dir / f"{from_name}_to_{to_square}.json"

    if not robot:
        # Without the arm, the dry run is all k can do. With it there is none: --execute builds
        # and checks the same commands (and writes the same preview) before anything moves.
        print("\nStep 2/2: run_real_so101 dry run (no --robot: the arm won't move).", flush=True)
        dry = pick_piece_command(
            lookup_json, lowlevel=lowlevel, output_dir=output_dir, compensate=compensate, baseline=baseline, execute=False
        )
        if run_probe(dry, lowlevel, sim_env) != 0:
            return MoveResult(False, "pick refused: run_real_so101 dry run failed (see terminal)")
        return MoveResult(False, f"pick dry run ok ({route}), {from_name} -> {to_square} (no --robot: arm not moved)")

    # k itself is the go-ahead: no Enter here and no yes/no before the start bridge (--yes).
    print("\nStep 2/2: moving the arm now. Support it: connecting turns torque off for a moment.", flush=True)
    close_arm()  # run_real opens the port itself
    run = pick_piece_command(
        lookup_json, lowlevel=lowlevel, output_dir=output_dir, compensate=compensate, baseline=baseline, execute=True
    )
    if on_move is not None:
        on_move()  # the arm is about to move (the click tool starts its tripod video here)
    code = run_probe(run, lowlevel, sim_env)
    preview_path = output_dir / "real_so101_trajectory_preview.json"
    if not preview_path.is_file():
        # run_real stopped before planning finished (it writes the preview before connecting),
        # so the arm didn't move and there is no run to record.
        print(f"\nPICK FAILED before the arm moved: run_real_so101 exited with code {code} and wrote no preview")
        return MoveResult(False, f"pick failed before moving: run_real_so101 exit {code} (see terminal)")

    metadata_path = output_dir / "metadata.json"
    metadata = {}
    metadata["from_click"] = from_click
    metadata["to_click"] = to_click
    metadata["from_click_centre_error_mm"] = centre_error_mm(from_click["board_mm"])
    metadata.update(targets.record())

    preview = json.loads(preview_path.read_text(encoding="utf-8"))

    if baseline:
        metadata["compensation_type"] = "baseline"
        metadata["calib_offset"] = preview["metadata"]["baseline_offsets_deg"]
    elif compensate:
        metadata["compensation_type"] = "compensation"
        fit_path = Path(preview["metadata"]["compensation_fit"])
        metadata["calib_offset"] = json.loads(fit_path.read_text(encoding="utf-8"))
    else:
        metadata["compensation_type"] = "None"
        metadata["calib_offset"] = (0.0, 0.0, 0.0, 0.0, 0.0)

    metadata["board_centre_base_mm"] = BOARD_CENTRE_BASE_MM
    # The sim gripper the lookup was planned with (SIM_GRIPPER_MODEL); replay with the same one.
    # Lookups from before the setting existed have no entry and used the legacy gripper.
    lookup = json.loads(lookup_json.read_text(encoding="utf-8"))
    metadata["gripper_model"] = lookup["search"].get("gripper_model", "legacy")

    with open(metadata_path, "w+") as f:
        json.dump(metadata, f)

    message = f"pick ({route}) {from_name} -> {to_square} done; saved {output_dir.name}" if code == 0 else f"run_real_so101 exited with code {code} (see terminal)"
    print(f"\n{message}")
    start_replay_plots(output_dir, lowlevel)
    return MoveResult(True, message, output_dir)


# Plots made after every k run (plot_pybullet_replay.py --align values): over time, and by command.
REPLAY_PLOT_ALIGNS = ("time", "index")


def start_replay_plots(output_dir: Path, lowlevel: Path) -> subprocess.Popen | None:
    """After a k run: replay it in PyBullet and plot sim vs hardware, in a background process.

    Runs compare_hardware_to_pybullet.py (writes pybullet_replay.json), then plot_pybullet_replay.py
    once per REPLAY_PLOT_ALIGNS (pybullet_replay_angles.png, pybullet_replay_angles_by_index.png).
    Detached so the click tool's window, keys and tripod video don't wait for it (~1 min); its
    output goes to <run>/replay_plots.log. Sim only: it never opens the arm's port.
    """
    if not (output_dir / "timeline_hardware.json").is_file() or not (output_dir / "timeline_sim.json").is_file():
        print(f"replay plots skipped: no timelines in {output_dir.name}")
        return None
    here = Path(__file__).resolve().parent
    steps = [[sys.executable, str(here / "compare_hardware_to_pybullet.py"), str(output_dir)]]
    steps += [
        [sys.executable, str(here / "plot_pybullet_replay.py"), str(output_dir), "--align", align]
        for align in REPLAY_PLOT_ALIGNS
    ]
    script = " && ".join(" ".join(shlex.quote(part) for part in step) for step in steps)
    env = {**os.environ, "MPLBACKEND": "Agg"}  # no plot window from a background process
    lerobot_src = str(lowlevel / "lerobot" / "src")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (lerobot_src, env.get("PYTHONPATH")) if p)
    log_path = output_dir / "replay_plots.log"
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            ["bash", "-c", script], cwd=str(lowlevel), env=env,
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
        )
    print(f"replay + plots running in the background -> {output_dir.name}/ (log: {log_path.name})")
    return process
