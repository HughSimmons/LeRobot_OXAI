"""Pick-and-place from two ``click_to_3d_oliver`` clicks, planned by ``dls_nullspace``.

Drop-in alternative to ``move_to_click.pick_piece`` (key ``k``: ~35 s PyBullet lookup build, then
``run_real_so101_from_xy_lookup.py``). Here the same two clicks go straight to
:class:`planner.DLSNullspacePlanner` in ~0.2 s and the commands stream to the arm in-process:

    from planner.click_bridge import pick_piece_planned
    result = pick_piece_planned(from_click, to_click, compensate=..., baseline=..., robot=...)

What is reused from his tool (imported at call time, so this file works on any machine that has his
``click_to_3d_oliver`` folder next to it):

* ``move_to_click.pick_targets``: ruler or camera-detected (``b``) route -> base-frame x, y;
* ``MoveResult`` / ``MoveRefused`` / ``centre_error_mm``;
* ``click3d_shared.BOARD_CENTRE_BASE_MM``, ``SQUARE_MM``, ``close_arm``;
* his run-folder + ``metadata.json`` conventions (``real_world_runs/click_picks/<stamp>_..._planner``).

Scene model: empty board, only the moving piece (no neighbours are modelled yet), default piece
height / radius of the sim rook. Override with ``PLANNER_PIECE_HEIGHT_MM`` / ``PLANNER_PIECE_RADIUS_MM``.
``robot=False`` is a pure dry run that never opens the serial port.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Optional

import numpy as np

from . import _paths_compat as pcompat
from . import hardware as hw
from .base import MoveRequest, plan_move_or_capture
from .collision_sdf import SceneCollisionChecker
from .dls_nullspace import DLSNullspacePlanner
from .frames import BOARD_SLAB_HALF_HEIGHT_M, BoardConfig
from .kinematics_so101 import DEFAULT_HOME_DEG, GRIPPER_CLOSED_DEG, GRIPPER_OPEN_DEG, SO101Kinematics
from .safety import far_reach_notice, motor_range_notice, safety_gate, tilt_notices

PLANNER_NAME = "dls_nullspace"
DEFAULT_PIECE_SYMBOL = "r"
MAX_BELOW_SURFACE_MM = 5.0  # same limits as move_to_click.plan_move
MAX_ABOVE_SURFACE_MM = 200.0


@dataclass
class _OliverTools:
    move_to_click: Any
    shared: Any

    @property
    def MoveResult(self):  # noqa: N802
        return self.move_to_click.MoveResult


def _load_oliver_tools() -> _OliverTools:
    """Import his ``move_to_click`` / ``click3d_shared`` (adding his folder to ``sys.path`` if needed)."""
    folder = pcompat.LOWLEVEL_DIR / "click_to_3d_oliver"
    if folder.is_dir() and str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    try:
        import click3d_shared
        import move_to_click
    except ImportError as exc:  # pragma: no cover - environment problem, reported to the operator
        raise ImportError(
            "click_bridge needs Oliver's click_to_3d_oliver folder (move_to_click.py, click3d_shared.py) "
            f"directly in {pcompat.LOWLEVEL_DIR}: {exc}"
        ) from exc
    return _OliverTools(move_to_click, click3d_shared)


def board_from_oliver(shared: Any) -> BoardConfig:
    """``BoardConfig`` for his ruler-measured board (centre ``BOARD_CENTRE_BASE_MM``, playing surface z)."""
    cx, cy, surface_z = (float(v) for v in shared.BOARD_CENTRE_BASE_MM)
    return BoardConfig(
        origin_xyz_m=(cx / 1000.0, cy / 1000.0, surface_z / 1000.0 - BOARD_SLAB_HALF_HEIGHT_M),
        square_size_m=float(shared.SQUARE_MM) / 1000.0,
        slab_half_height_m=BOARD_SLAB_HALF_HEIGHT_M,
        border_margin_m=0.030,
        name="oliver_ruler_board",
    )


_PLANNING_STACK: dict[str, Any] = {}


def _planning_stack(board: BoardConfig):
    """Kinematics / planner / empty-scene checker, built once per process."""
    key = repr(board)
    if _PLANNING_STACK.get("key") != key:
        kin = SO101Kinematics()
        _PLANNING_STACK.update(
            key=key,
            kin=kin,
            planner=DLSNullspacePlanner(kin, board),
            checker=SceneCollisionChecker(kin, board, []),
        )
    return _PLANNING_STACK["kin"], _PLANNING_STACK["planner"], _PLANNING_STACK["checker"]


def _env_m(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    return float(raw) / 1000.0 if raw else None


def compensation_mode(compensate: bool, baseline: bool) -> str:
    return "baseline" if baseline else "on" if compensate else "off"


def check_click_heights(click: dict[str, Any], surface_z_mm: float) -> None:
    """Same floor / ceiling checks as ``move_to_click.plan_move`` (raises ``ValueError``)."""
    base = np.asarray(click.get("base_mm", []), dtype=np.float64)
    if base.shape != (3,) or not np.all(np.isfinite(base)):
        return  # no height to check; the planner uses its own grasp height
    below = surface_z_mm - float(base[2])
    if below > MAX_BELOW_SURFACE_MM:
        raise ValueError(
            f"piece click z {base[2]:.1f} mm is {below:.1f} mm below the board surface "
            f"({surface_z_mm:.1f} mm). Click it again."
        )
    height = float(base[2]) - surface_z_mm
    if height > MAX_ABOVE_SURFACE_MM:
        raise ValueError(f"piece click is {height:.0f} mm above the board, higher than any piece. Click it again.")


def plan_pick_place(
    board: BoardConfig,
    from_xy_m: np.ndarray,
    to_xy_m: np.ndarray,
    *,
    start_joints_deg: np.ndarray,
    piece: str = DEFAULT_PIECE_SYMBOL,
    piece_height_m: Optional[float] = None,
    piece_radius_m: Optional[float] = None,
):
    kin, planner, checker = _planning_stack(board)
    request = MoveRequest(
        from_target=(float(from_xy_m[0]), float(from_xy_m[1])),
        to_target=(float(to_xy_m[0]), float(to_xy_m[1])),
        moving_piece_symbol=piece,
        moving_piece_height_m=piece_height_m,
        moving_piece_radius_m=piece_radius_m,
        start_joints_deg=np.asarray(start_joints_deg, dtype=np.float64).copy(),
        end_joints_deg=DEFAULT_HOME_DEG.copy(),
        gripper_open_deg=GRIPPER_OPEN_DEG,
        gripper_closed_deg=GRIPPER_CLOSED_DEG,
    )
    return kin, plan_move_or_capture(planner, request, checker, dt_s=hw.DEFAULT_SOURCE_DT_S)


def _calib_offset_record(comp: Optional[Any], mode: str) -> Any:
    if mode == "off" or comp is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    if mode == "on" and getattr(comp, "fit_path", None) is not None and Path(comp.fit_path).is_file():
        return json.loads(Path(comp.fit_path).read_text(encoding="utf-8"))
    return [round(float(v), 4) for v in comp.offsets_deg[:5]]


def pick_piece_planned(
    from_click: dict[str, Any],
    to_click: dict[str, Any],
    *,
    compensate: bool,
    baseline: bool,
    robot: bool,
    on_move: Optional[Callable[[], None]] = None,
    piece: Optional[str] = None,
):
    """Pick the piece at ``from_click`` and place it on the centre of ``to_click``'s square.

    Returns his ``MoveResult``. The key press that calls this is the go-ahead: there is no extra
    confirmation once the dry plan passes the safety gate (as with ``k``).
    """
    tools = _load_oliver_tools()
    mtc, shared = tools.move_to_click, tools.shared
    MoveResult = tools.MoveResult  # noqa: N806

    to_square = to_click["square"]
    if to_square == "off":
        return MoveResult(False, "pick refused: the destination click is off the board")
    try:
        targets = mtc.pick_targets(from_click, to_click)
    except mtc.MoveRefused as exc:
        return MoveResult(False, f"pick refused: {exc}")

    board = board_from_oliver(shared)
    from_xy = np.asarray(targets.from_base_xy_mm, dtype=np.float64) / 1000.0
    to_xy = np.asarray(targets.to_base_xy_mm, dtype=np.float64) / 1000.0
    route = "cv" if targets.source == shared.CV_BASE_SOURCE else "ruler"
    mode = compensation_mode(compensate, baseline)
    symbol = (piece or DEFAULT_PIECE_SYMBOL).strip() or DEFAULT_PIECE_SYMBOL
    height_m, radius_m = _env_m("PLANNER_PIECE_HEIGHT_MM"), _env_m("PLANNER_PIECE_RADIUS_MM")

    if not board.is_over_board(to_xy, extra_margin_m=-board.border_margin_m):
        return MoveResult(False, f"pick refused: destination {to_square} maps off the board ({np.round(to_xy * 1000, 1).tolist()} mm)")
    if not board.is_over_board(from_xy):
        return MoveResult(False, f"pick refused: the piece click maps off the board ({np.round(from_xy * 1000, 1).tolist()} mm)")
    try:
        check_click_heights(from_click, float(shared.BOARD_CENTRE_BASE_MM[2]))
    except ValueError as exc:
        return MoveResult(False, f"pick refused: {exc}")

    from_name = f"{from_click['square']}_click"
    suffix = {"off": "off", "baseline": "baseline", "on": "compensate"}[mode] + ("_cv" if route == "cv" else "") + "_planner"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = pcompat.real_world_runs_dir() / "click_picks" / f"{stamp}_{re.sub(r'[^A-Za-z0-9_]', '_', from_name)}_to_{to_square}_{suffix}"

    print("\n" + "=" * 72)
    print(
        f"PLANNED PICK ({route}, {PLANNER_NAME}) {from_name} {np.round(from_xy * 1000, 1).tolist()} mm -> "
        f"{to_square} {np.round(to_xy * 1000, 1).tolist()} mm (robot base frame), compensation: {mode}"
    )
    print(f"Saving to {output_dir}")
    print("Step 1/2: plan from the home pose (no arm) ...")
    print("=" * 72, flush=True)

    try:
        kin, dry_plan = plan_pick_place(
            board, from_xy, to_xy, start_joints_deg=DEFAULT_HOME_DEG, piece=symbol,
            piece_height_m=height_m, piece_radius_m=radius_m,
        )
    except Exception as exc:  # noqa: BLE001 - IK/planner failures are refusals, not crashes of the click tool
        print(f"\nPICK REFUSED: planning failed: {exc}")
        return MoveResult(False, f"pick refused: planning failed ({type(exc).__name__}: {exc})")

    compensator = hw.make_compensator(mode)
    dry_cmds = hw.compensate_plan(dry_plan, compensator, mode)
    ok, warnings = safety_gate(dry_plan, hw.compensate_plan(dry_plan, None, "off"), kin)
    notices = (
        tilt_notices(dry_plan)
        + far_reach_notice([("pick", from_xy), ("place", to_xy)])
        + motor_range_notice(dry_cmds, kin)
    )
    for note in notices:
        print(f"  [note] {note}")
    if not ok:
        print("\nPICK REFUSED: safety gate blocked the plan (see above).")
        return MoveResult(False, f"pick refused: safety gate ({'; '.join(warnings)})")

    metadata: dict[str, Any] = {
        "planner": PLANNER_NAME,
        "from_click": from_click,
        "to_click": to_click,
        "from_click_centre_error_mm": mtc.centre_error_mm(from_click["board_mm"]),
        **targets.record(),
        "compensation_type": {"baseline": "baseline", "on": "compensation", "off": "None"}[mode],
        "calib_offset": _calib_offset_record(compensator, mode),
        "board_centre_base_mm": list(shared.BOARD_CENTRE_BASE_MM),
        "gripper_model": "planner_capsule_sdf",
        "piece_symbol": symbol,
        "tilt_notices": notices,
        "dry_plan_duration_s": dry_plan.total_duration_s,
    }

    def write_metadata(extra: dict[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metadata.json").write_text(json.dumps({**metadata, **extra}, indent=2, default=str) + "\n", encoding="utf-8")

    if not robot:
        hw.save_run_log(output_dir, commands_deg=dry_cmds, log=None, metadata=metadata, finished=False, error="dry run (no --robot)", filename="plan_preview.json")
        write_metadata({"executed": False})
        print("\nPlan OK. No --robot, so the arm is not moved.")
        return MoveResult(False, f"planned pick dry run ok ({route}), {from_name} -> {to_square} (no --robot: arm not moved)", output_dir)

    # Step 2: connect, re-plan from the live pose, move. The key press is the go-ahead.
    print("\nStep 2/2: connecting and moving the arm now.", flush=True)
    shared.close_arm()  # the click tool's own serial connection; we open ours
    follower = hw.connect_follower()
    try:
        reading = hw.read_joints_deg(follower)
        print(f"Live start joints (deg): {np.round(reading, 2).tolist()}")
        start_urdf = hw.start_urdf_from_reading(reading, compensator)
        kin, plan = plan_pick_place(
            board, from_xy, to_xy, start_joints_deg=start_urdf, piece=symbol,
            piece_height_m=height_m, piece_radius_m=radius_m,
        )
        cmds = hw.compensate_plan(plan, compensator, mode)
        ok, warnings = safety_gate(plan, hw.compensate_plan(plan, None, "off"), kin)
        if not ok:
            raise RuntimeError("safety gate blocked the plan from the live pose: " + "; ".join(warnings))
    except BaseException as exc:
        print(f"\nPICK FAILED before the arm moved: {exc}")
        hw.release_arm(follower, finished=False)
        write_metadata({"executed": False, "error": f"{type(exc).__name__}: {exc}"})
        if isinstance(exc, KeyboardInterrupt):
            raise
        return MoveResult(False, f"pick failed before moving: {exc}", output_dir)

    if on_move is not None:
        on_move()  # the arm is about to move (the click tool starts its tripod video here)
    try:
        log = hw.execute_commands(follower, cmds, run_dir=output_dir, metadata=metadata)  # releases the arm
    except BaseException as exc:
        write_metadata({"executed": True, "finished": False, "error": f"{type(exc).__name__}: {exc}"})
        if isinstance(exc, KeyboardInterrupt):
            raise
        message = f"planned pick stopped early: {exc} (see terminal)"
        print(f"\n{message}")
        return MoveResult(True, message, output_dir)

    write_metadata({"executed": True, "finished": True, "stream_duration_s": log.duration_s, "behind_schedule_steps": log.behind_schedule_steps})
    message = f"planned pick ({route}) {from_name} -> {to_square} done; saved {output_dir.name}"
    print(f"\n{message}")
    return MoveResult(True, message, output_dir)
