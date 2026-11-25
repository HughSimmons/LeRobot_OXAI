"""Pre-motion safety gate and operator-facing report for planned moves."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .base import CaptureSequenceResult, PlannedTrajectory
from .kinematics_so101 import SO101Kinematics

MAX_IK_ERR_MM = 2.5
MIN_TRANSIT_CLEARANCE_MM = -2.0
JOINT_LIMIT_SLACK_DEG = 0.5
TILT_NOTICE_DEG = 2.0  # Oliver's straight-down tool refuses above this; the planner may tilt more.
FAR_REACH_NOTICE_M = 0.36  # radial base distance where the arm-span ceiling starts forcing tilt


def plan_legs(plan_result: PlannedTrajectory | CaptureSequenceResult) -> list[tuple[str, PlannedTrajectory]]:
    if isinstance(plan_result, CaptureSequenceResult):
        return [("leg1_capture", plan_result.leg1_clear_captured), ("leg2_move", plan_result.leg2_move_attacker)]
    return [("single_leg", plan_result)]


def tilt_notices(plan_result: PlannedTrajectory | CaptureSequenceResult) -> List[str]:
    """Non-blocking notes: gripper tilt vs Oliver's straight-down limit, and far-file reach."""
    notes: List[str] = []
    for label, leg in plan_legs(plan_result):
        ik = leg.coupled_ik
        if max(ik.pick_tilt_deg, ik.place_tilt_deg) > TILT_NOTICE_DEG:
            notes.append(
                f"{label}: gripper tilt {ik.pick_tilt_deg:.1f} deg at pick, {ik.place_tilt_deg:.1f} deg at place "
                f"(release tilt {ik.piece_release_tilt_deg:.1f} deg); his tools refuse > {TILT_NOTICE_DEG:.0f} deg, "
                "this planner matches pick and place attitude instead."
            )
    return notes


def far_reach_notice(points: Sequence[tuple[str, Sequence[float]]], base_xy_m=(0.0, 0.0)) -> List[str]:
    """Non-blocking note for each labelled world (x, y) in metres far from the base."""
    notes: List[str] = []
    for tag, xy in points:
        radius = float(np.hypot(xy[0] - base_xy_m[0], xy[1] - base_xy_m[1]))
        if radius > FAR_REACH_NOTICE_M:
            notes.append(
                f"{tag} is {radius * 1000:.0f} mm from the base: near the arm-span ceiling, expect forced tilt "
                "and little lift headroom (move the board closer if this keeps happening)."
            )
    return notes


def motor_range_notice(motor_commands_deg: np.ndarray, kin: SO101Kinematics) -> List[str]:
    """Non-blocking: compensated motor commands that leave the URDF joint range (calibration offsets
    shift commands by a few degrees, so a pose planned at a joint limit lands just outside it)."""
    arm = np.asarray(motor_commands_deg)[:, :5]
    over = float(max(np.max(arm - kin.arm_hi_deg[None, :]), np.max(kin.arm_lo_deg[None, :] - arm)))
    if over > JOINT_LIMIT_SLACK_DEG:
        return [
            f"compensated motor commands go {over:.1f} deg beyond the URDF joint range (offset/friction "
            "shift at a joint limit); the arm may stop short of the planned pose there."
        ]
    return []


def safety_gate(
    plan_result: PlannedTrajectory | CaptureSequenceResult,
    commands_deg: np.ndarray,
    kin: SO101Kinematics,
    *,
    verbose: bool = True,
) -> Tuple[bool, List[str]]:
    """IK error, transit clearance and joint limits. Returns ``(ok, warnings)``; prints a report.

    ``commands_deg`` must be the planner's URDF-space commands (before motor compensation): the
    limits describe the plan, and compensation is checked separately by ``motor_range_notice``.
    """
    warnings: List[str] = []
    if verbose:
        print("\n=== SO-101 Planned Move Safety & Attitude Report ===")
    for label, leg in plan_legs(plan_result):
        ik = leg.coupled_ik
        if verbose:
            print(
                f"[{label}] planner={leg.planner_name} (fallback={leg.fallback_used}) | "
                f"plan_time={leg.planning_time_ms:.1f} ms | duration={leg.total_duration_s:.2f} s"
            )
            print(
                f"  Coupled IK: pick_err={ik.pick_pos_err_mm:.2f} mm, place_err={ik.place_pos_err_mm:.2f} mm | "
                f"tilt=({ik.pick_tilt_deg:.2f}° -> {ik.place_tilt_deg:.2f}°), "
                f"predicted_piece_release_tilt={ik.piece_release_tilt_deg:.2f}°"
            )
            print(
                f"  SDF Clearance: min_transit={leg.min_transit_clearance_mm:+.2f} mm | "
                f"path_length={leg.joint_path_length_deg:.1f}°"
            )
            print(f"  q_pick_deg : {np.round(ik.q_pick_deg, 2).tolist()}")
            print(f"  q_place_deg: {np.round(ik.q_place_deg, 2).tolist()}")
        if not ik.converged or max(ik.pick_pos_err_mm, ik.place_pos_err_mm) > MAX_IK_ERR_MM:
            warnings.append(
                f"{label}: Coupled IK position error > {MAX_IK_ERR_MM} mm "
                f"({ik.pick_pos_err_mm:.2f}, {ik.place_pos_err_mm:.2f})"
            )
        if leg.min_transit_clearance_mm < MIN_TRANSIT_CLEARANCE_MM:
            warnings.append(
                f"{label}: Transit SDF clearance below {MIN_TRANSIT_CLEARANCE_MM} mm "
                f"({leg.min_transit_clearance_mm:+.2f} mm)"
            )

    arm_cmds = np.asarray(commands_deg)[:, :5]
    below = float(np.min(arm_cmds - kin.arm_lo_deg[None, :]))
    above = float(np.max(arm_cmds - kin.arm_hi_deg[None, :]))
    if below < -JOINT_LIMIT_SLACK_DEG or above > JOINT_LIMIT_SLACK_DEG:
        warnings.append(f"Arm joint command exceeded URDF limits (min_margin={below:.2f}°, max_over={above:.2f}°)")

    ok = not warnings
    if verbose:
        print(f"Safety Gate Status: {'PASS' if ok else 'BLOCKED'}")
        for w in warnings:
            print(f"  [!] {w}")
    return ok, warnings
