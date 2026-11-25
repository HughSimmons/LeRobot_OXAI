"""Common planner interfaces, minimum-jerk trajectory timing, and 2-leg capture sequencing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from board_coordinates import XYPoint
from .collision_sdf import (
    HeldPieceAttachment,
    PieceCylinder,
    SIM_ROOK_DIMS_M,
    STAUNTON_PIECE_DIMS_M,
    SceneCollisionChecker,
)
from .frames import BoardConfig, resolve_target_world_xyz
from .kinematics_so101 import (
    CoupledIKSolution,
    DEFAULT_HOME_DEG,
    DEFAULT_PINCH_OFFSET_TCP_M,
    GRIPPER_CLOSED_DEG,
    GRIPPER_OPEN_DEG,
    SO101Kinematics,
)


@dataclass
class MoveRequest:
    """Specification of a single chess move (or 2-leg capture) to plan."""

    from_target: str | XYPoint | Sequence[float]
    to_target: str | XYPoint | Sequence[float]
    moving_piece_label: str | None = None
    moving_piece_symbol: str = "r"
    moving_piece_height_m: float | None = None
    moving_piece_radius_m: float | None = None
    pick_yaw_deg: float | None = None
    # Optional explicit grasp/place Z above surface (defaults to ~56% of piece height, clamped)
    grasp_z_above_surface_m: float | None = None
    place_release_clearance_m: float = 0.0008
    # Capture support
    is_capture: bool = False
    captured_piece_label: str | None = None
    captured_piece_symbol: str = "p"
    captured_piece_height_m: float | None = None
    captured_piece_radius_m: float | None = None
    graveyard_slot_index: int = 0
    # Start / terminal arm poses
    start_joints_deg: np.ndarray = field(default_factory=lambda: DEFAULT_HOME_DEG.copy())
    end_joints_deg: np.ndarray = field(default_factory=lambda: DEFAULT_HOME_DEG.copy())
    return_to_end: bool = True
    # Gripper open/close angles (deg)
    gripper_open_deg: float = GRIPPER_OPEN_DEG
    gripper_closed_deg: float = GRIPPER_CLOSED_DEG
    pinch_offset_tcp_m: np.ndarray = field(default_factory=lambda: DEFAULT_PINCH_OFFSET_TCP_M.copy())


@dataclass
class TrajectorySegmentSpec:
    """One logical phase of a pick-and-place trajectory prior to dense timing."""

    name: str
    waypoints_deg: np.ndarray       # (M, 6)
    duration_s: float
    is_dwell: bool = False
    is_rest_endpoint: bool = False


@dataclass
class PlannedTrajectory:
    """Complete time-parameterized joint trajectory with kinematic & clearance diagnostics."""

    planner_name: str
    waypoints_deg: np.ndarray                 # (W, 6) key waypoints
    commands_deg: np.ndarray                  # (T, 6) dense 240 Hz commands (zero Catmull-Rom overshoot)
    dt_s: float
    segment_ranges: dict[str, tuple[int, int]]
    rest_command_indices: list[int]
    close_gripper_cmd_idx: int
    open_gripper_cmd_idx: int
    coupled_ik: CoupledIKSolution
    planning_time_ms: float
    min_transit_clearance_mm: float
    min_overall_clearance_mm: float
    joint_path_length_deg: float
    mean_squared_jerk: float
    fallback_used: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def total_duration_s(self) -> float:
        return float(self.commands_deg.shape[0] * self.dt_s)


def resolve_piece_dims(
    symbol: str,
    height_override_m: float | None = None,
    radius_override_m: float | None = None,
    *,
    default_staunton: bool = False,
) -> tuple[float, float]:
    s = symbol.strip().lower()[0] if symbol else "r"
    if default_staunton and s in STAUNTON_PIECE_DIMS_M:
        h, r = STAUNTON_PIECE_DIMS_M[s]
    else:
        h, r = SIM_ROOK_DIMS_M
    if height_override_m is not None:
        h = float(height_override_m)
    if radius_override_m is not None:
        r = float(radius_override_m)
    return h, r


def adaptive_hover_height_m(
    xy_world_m: Sequence[float],
    piece_height_m: float,
    neighbor_max_height_m: float,
) -> float:
    """Select hover height above board surface balancing tall-piece clearance vs SO-101 far-reach ceiling.

    On SO-101, near squares (R < 0.31 m) easily lift 120-150 mm above the board, whereas far
    squares (R in [0.35, 0.45] m) have a spherical workspace ceiling that tapers down to ~85-100 mm.
    """
    dx = float(xy_world_m[0]) - 0.0388
    dy = float(xy_world_m[1])
    radial_m = float(np.hypot(dx, dy))

    desired = max(0.110, piece_height_m + 0.055, neighbor_max_height_m + 0.040)
    if radial_m <= 0.29:
        ceiling = 0.165
    elif radial_m >= 0.41:
        ceiling = 0.084
    else:
        alpha = (radial_m - 0.29) / (0.41 - 0.29)
        ceiling = 0.165 * (1.0 - alpha) + 0.084 * alpha
    return float(min(desired, ceiling))


def adaptive_open_gripper_deg(
    piece_radius_m: float,
    *,
    has_tight_neighbors: bool,
    default_open_deg: float = GRIPPER_OPEN_DEG,
) -> float:
    """Choose jaw opening wide enough to clear the piece's upper collar/waist without hitting neighbors.

    From our V-HACD mesh calibration:
    - ``gripper = 5 deg``  -> 22.0 mm gap
    - ``gripper = 14 deg`` -> 32.9 mm gap (outer jaw span < 43 mm, fits inside a 40 mm square!)
    - ``gripper = 18 deg`` -> 37.5 mm gap
    - ``gripper = 25 deg`` -> 44.7 mm gap
    On crowded squares, opening to 14-17 deg keeps both jaws within the square's clear lane
    while providing ~10 mm total margin around a 20-25 mm piece collar/crown!
    """
    if not has_tight_neighbors:
        return float(default_open_deg)
    collar_diam_mm = 2.0 * min(11.6, 0.72 * piece_radius_m * 1000.0)
    needed_gap_mm = collar_diam_mm + 14.5
    # Linear fit: gap_mm(deg) ~ 16.3 + 1.14 * deg
    deg = (needed_gap_mm - 16.3) / 1.14
    return float(np.clip(deg, 18.5, default_open_deg))


def minimum_jerk_s(u: np.ndarray) -> np.ndarray:
    """5th-order minimum-jerk scalar profile ``s(u) in [0, 1]`` with zero endpoint vel & accel."""
    u2 = u * u
    u3 = u2 * u
    return u3 * (10.0 - 15.0 * u + 6.0 * u2)


def interpolate_waypoints_monotone_cubic(waypoints: np.ndarray, n_samples: int) -> np.ndarray:
    """Interpolate ``(M, D)`` waypoints into ``(n_samples, D)`` with minimum-jerk time scaling and zero overshoot.

    Uses Fritsch-Carlson / Steffen-style monotone tangent clamping in parameter ``s in [0, 1]``
    combined with ``s = minimum_jerk_s(t/T)`` so:
    1. Start and end have zero velocity and zero acceleration (no base rocking!).
    2. Monotone joint moves NEVER overshoot or dip below ``place`` / ``pick`` (fixing the 7%
       Catmull-Rom dip bug in ``multisim_chess_fast.py``!).
    """
    W = np.asarray(waypoints, dtype=np.float64)
    m, d = W.shape
    if n_samples <= 1:
        return W[-1:, :].copy()
    if m == 1:
        return np.repeat(W, n_samples, axis=0)
    if m == 2:
        u = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)[:, None]
        s = minimum_jerk_s(u)
        return (1.0 - s) * W[0:1, :] + s * W[1:2, :]

    # Cumulative chord-length parameterization along arm joints
    diffs = np.linalg.norm(np.diff(W[:, :5], axis=0), axis=1)
    diffs = np.maximum(diffs, 1e-3)
    knots = np.concatenate([[0.0], np.cumsum(diffs)])
    knots /= knots[-1]

    # Compute Fritsch-Carlson monotone tangents per joint dimension
    h = np.diff(knots)                                     # (m-1,)
    secants = np.diff(W, axis=0) / h[:, None]              # (m-1, d)
    tangents = np.zeros_like(W)
    tangents[0] = secants[0]
    tangents[-1] = secants[-1]
    for k in range(1, m - 1):
        s_prev = secants[k - 1]
        s_next = secants[k]
        same_sign = (s_prev * s_next) > 0.0
        avg = 0.5 * (s_prev + s_next)
        # Clamp tangent magnitude to 3 * min(|s_prev|, |s_next|) to guarantee zero overshoot
        max_mag = 3.0 * np.minimum(np.abs(s_prev), np.abs(s_next))
        clamped = np.sign(avg) * np.minimum(np.abs(avg), max_mag)
        tangents[k] = np.where(same_sign, clamped, 0.0)

    u_grid = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    s_grid = minimum_jerk_s(u_grid)

    idx = np.clip(np.searchsorted(knots, s_grid, side="right") - 1, 0, m - 2)
    hk = h[idx][:, None]
    tau = ((s_grid - knots[idx]) / h[idx])[:, None]
    tau2 = tau * tau
    tau3 = tau2 * tau

    h00 = 2.0 * tau3 - 3.0 * tau2 + 1.0
    h10 = tau3 - 2.0 * tau2 + tau
    h01 = -2.0 * tau3 + 3.0 * tau2
    h11 = tau3 - tau2

    out = (
        h00 * W[idx]
        + h10 * hk * tangents[idx]
        + h01 * W[idx + 1]
        + h11 * hk * tangents[idx + 1]
    )
    # Gripper channel interpolated piecewise linearly in minimum-jerk s
    out[:, 5] = (1.0 - tau[:, 0]) * W[idx, 5] + tau[:, 0] * W[idx + 1, 5]
    return out


def compile_segments_to_commands(
    segments: Sequence[TrajectorySegmentSpec],
    *,
    dt_s: float = 1.0 / 240.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, int]], list[int], float, float]:
    """Compile logical segments into dense 240 Hz commands + smoothness metrics."""
    cmd_blocks: list[np.ndarray] = []
    waypoint_blocks: list[np.ndarray] = []
    seg_ranges: dict[str, tuple[int, int]] = {}
    rest_indices: list[int] = []
    cursor = 0

    for seg_i, seg in enumerate(segments):
        n_steps = max(2, int(round(seg.duration_s / dt_s)))
        if seg.is_dwell:
            pose = np.asarray(seg.waypoints_deg[-1], dtype=np.float64).reshape(1, 6)
            block = np.repeat(pose, n_steps, axis=0)
        else:
            block = interpolate_waypoints_monotone_cubic(seg.waypoints_deg, n_steps)
            if seg_i > 0:
                # Drop duplicate boundary sample so velocity stays smooth
                block = block[1:]

        start_idx = cursor
        cursor += block.shape[0]
        seg_ranges[seg.name] = (start_idx, cursor)
        if seg.is_rest_endpoint and cursor > 0:
            rest_indices.append(cursor - 1)
        cmd_blocks.append(block)
        waypoint_blocks.append(
            seg.waypoints_deg if seg_i == 0 else seg.waypoints_deg[1:]
        )

    commands = np.vstack(cmd_blocks) if cmd_blocks else np.zeros((0, 6), dtype=np.float64)
    waypoints = np.vstack(waypoint_blocks) if waypoint_blocks else np.zeros((0, 6), dtype=np.float64)

    if commands.shape[0] >= 4:
        diffs = np.diff(commands[:, :5], axis=0)
        path_len_deg = float(np.sum(np.linalg.norm(diffs, axis=1)))
        jerk = np.diff(commands[:, :5], n=3, axis=0) / (dt_s ** 3)
        ms_jerk = float(np.mean(np.sum(jerk * jerk, axis=1)))
    else:
        path_len_deg = 0.0
        ms_jerk = 0.0

    return waypoints, commands, seg_ranges, rest_indices, path_len_deg, ms_jerk


class TrajectoryPlanner(ABC):
    """Base class for SO-101 chess trajectory planners."""

    name: str = "base"

    def __init__(self, kin: SO101Kinematics, board: BoardConfig) -> None:
        self.kin = kin
        self.board = board

    @abstractmethod
    def plan_single_leg(
        self,
        request: MoveRequest,
        checker: SceneCollisionChecker,
        *,
        dt_s: float = 1.0 / 240.0,
    ) -> PlannedTrajectory:
        """Plan one pick-and-place move on ``checker``'s obstacle scene."""

    def solve_endpoints_and_vertical_strokes(
        self,
        request: MoveRequest,
        checker: SceneCollisionChecker,
    ) -> dict[str, object]:
        """Shared Layer-1 Coupled Attitude-Matched IK + Locked-Attitude DLS vertical strokes."""
        moving_h, moving_r = resolve_piece_dims(
            request.moving_piece_symbol,
            request.moving_piece_height_m,
            request.moving_piece_radius_m,
            default_staunton=(len(checker.pieces) > 0 and any(p.height_m > 0.042 for p in checker.pieces)),
        )
        grasp_z = (
            float(request.grasp_z_above_surface_m)
            if request.grasp_z_above_surface_m is not None
            else float(np.clip(0.56 * moving_h, 0.020, 0.052))
        )

        p_pick = resolve_target_world_xyz(request.from_target, self.board, default_z_above_surface_m=grasp_z)
        p_place_raw = resolve_target_world_xyz(
            request.to_target,
            self.board,
            default_z_above_surface_m=grasp_z + float(request.place_release_clearance_m),
        )
        # If to_target is an off-board graveyard slot with z == table_z_m (e.g. 0.0), add grasp_z + clearance
        if not self.board.is_over_board(p_place_raw[:2]) and p_place_raw[2] <= checker.table_z_m + 0.006:
            p_place = np.array(
                [p_place_raw[0], p_place_raw[1], checker.table_z_m + grasp_z + float(request.place_release_clearance_m)],
                dtype=np.float64,
            )
            place_surface_z = checker.table_z_m
        else:
            p_place = p_place_raw
            place_surface_z = self.board.surface_z_m

        pick_label = (
            request.moving_piece_label.lower()
            if request.moving_piece_label is not None
            else self.board.nearest_square(p_pick[:2])
        )
        excl = (pick_label,)

        # Check local neighbor density within 65 mm of pick or place to adapt jaw opening & hover height
        obs_xy, _, obs_z1, _ = checker.active_obstacles(excl)
        tight_pick = False
        tight_place = False
        local_max_h_pick = moving_h
        local_max_h_place = moving_h
        if obs_xy.shape[0] > 0:
            d_pick = np.linalg.norm(obs_xy - p_pick[:2], axis=1)
            d_place = np.linalg.norm(obs_xy - p_place[:2], axis=1)
            if np.any(d_pick < 0.068):
                tight_pick = True
            if np.any(d_place < 0.068):
                tight_place = True
            near_p = obs_z1[d_pick < 0.095] - self.board.surface_z_m
            near_d = obs_z1[d_place < 0.095] - self.board.surface_z_m
            if near_p.size > 0:
                local_max_h_pick = max(local_max_h_pick, float(np.max(near_p)))
            if near_d.size > 0:
                local_max_h_place = max(local_max_h_place, float(np.max(near_d)))

        grip_open = adaptive_open_gripper_deg(
            moving_r, has_tight_neighbors=tight_pick, default_open_deg=request.gripper_open_deg
        )
        grip_open_place = (
            adaptive_open_gripper_deg(moving_r, has_tight_neighbors=True, default_open_deg=request.gripper_open_deg)
            if tight_place
            else max(float(request.gripper_open_deg), 28.0)
        )
        grip_closed = float(request.gripper_closed_deg)

        def endpoint_clearance(q_pick_6: np.ndarray, q_place_6: np.ndarray) -> float:
            q_place_open = q_place_6.copy()
            q_place_open[5] = grip_open_place
            Q_pair = np.vstack([q_pick_6, q_place_open])
            c_vals, _ = checker.evaluate_clearance_batch(
                Q_pair, exclude_labels=excl, allow_tip_near_board=True
            )
            return float(np.min(c_vals))

        coupled = self.kin.solve_coupled_pick_place_ik(
            p_pick,
            p_place,
            pinch_offset_tcp_m=request.pinch_offset_tcp_m,
            preferred_pick_yaw_deg=request.pick_yaw_deg,
            roll_candidates_deg=(0.0, -45.0, 45.0, -90.0, 90.0, -135.0, 135.0),
            clearance_fn=endpoint_clearance,
            gripper_open_deg=grip_open,
            gripper_closed_deg=grip_closed,
        )

        hover_z_pick = adaptive_hover_height_m(p_pick[:2], moving_h, local_max_h_pick)
        hover_z_place = adaptive_hover_height_m(p_place[:2], moving_h, local_max_h_place)
        dz_pick = max(0.045, (self.board.surface_z_m + hover_z_pick) - p_pick[2])
        dz_place = max(0.045, (place_surface_z + hover_z_place) - p_place[2])

        # Closed-jaw lift & open-jaw approach at pick move along world vertical (+Z / -Z)
        # so the open jaws descend straight into the 40 mm square column without sweeping neighbors!
        lift_from_pick_closed, _, _ = self.kin.solve_locked_attitude_cartesian_path(
            coupled.q_pick_deg,
            [0.0, 0.0, dz_pick],
            n_waypoints=8,
            gripper_deg=grip_closed,
            pinch_offset_tcp_m=request.pinch_offset_tcp_m,
        )
        lower_to_pick_open = lift_from_pick_closed[::-1].copy()
        lower_to_pick_open[:, 5] = grip_open

        place_lift_vertical, _, _ = self.kin.solve_locked_attitude_cartesian_path(
            coupled.q_place_deg,
            [0.0, 0.0, dz_place],
            n_waypoints=8,
            gripper_deg=grip_closed,
            pinch_offset_tcp_m=request.pinch_offset_tcp_m,
        )
        lower_to_place_closed = place_lift_vertical[::-1].copy()
        lower_to_place_closed[:, 5] = grip_closed

        # Open-jaw retreat from place backs out along the gripper's throat axis (-Z_tcp):
        # identical to pure world +Z when tilt == 0 deg, and when tilted > 0 deg slides both
        # slanted open jaws cleanly off the placed piece without hooking it!
        T_place, _ = self.kin.fk_tcp(coupled.q_place_deg, request.pinch_offset_tcp_m)
        place_throat_up = -T_place[:3, 2]
        place_slide_len = dz_place / max(float(place_throat_up[2]), 0.72)
        retreat_from_place_open, _, _ = self.kin.solve_locked_attitude_cartesian_path(
            coupled.q_place_deg,
            place_throat_up * place_slide_len,
            n_waypoints=8,
            gripper_deg=grip_open_place,
            pinch_offset_tcp_m=request.pinch_offset_tcp_m,
        )

        held_piece = HeldPieceAttachment.from_pick_pose(
            self.kin,
            coupled.q_pick_deg,
            p_pick[:2],
            self.board.surface_z_m,
            moving_h,
            moving_r,
        )

        return {
            "coupled": coupled,
            "p_pick": p_pick,
            "p_place": p_place,
            "pick_label": pick_label,
            "moving_h": moving_h,
            "moving_r": moving_r,
            "grip_open": grip_open,
            "grip_open_place": grip_open_place,
            "grip_closed": grip_closed,
            "lower_to_pick_open": lower_to_pick_open,
            "lift_from_pick_closed": lift_from_pick_closed,
            "lower_to_place_closed": lower_to_place_closed,
            "retreat_from_place_open": retreat_from_place_open,
            "q_hover_pick_open": lower_to_pick_open[0].copy(),
            "q_hover_pick_closed": lift_from_pick_closed[-1].copy(),
            "q_hover_place_closed": lower_to_place_closed[0].copy(),
            "q_hover_place_open": retreat_from_place_open[-1].copy(),
            "held_piece": held_piece,
        }


@dataclass
class CaptureSequenceResult:
    """Two-leg capture plan: Leg 1 clears opponent piece to graveyard; Leg 2 moves attacker."""

    leg1_clear_captured: PlannedTrajectory
    leg2_move_attacker: PlannedTrajectory
    combined_commands_deg: np.ndarray
    total_planning_time_ms: float


def plan_move_or_capture(
    planner: TrajectoryPlanner,
    request: MoveRequest,
    checker: SceneCollisionChecker,
    *,
    dt_s: float = 1.0 / 240.0,
) -> PlannedTrajectory | CaptureSequenceResult:
    """Plan either a standard single-leg chess move or a seamless 2-leg capture sequence."""
    if not request.is_capture:
        return planner.plan_single_leg(request, checker, dt_s=dt_s)

    graveyard_xyz = planner.board.graveyard_slot_world_xyz(
        request.graveyard_slot_index, z_above_table_m=checker.table_z_m
    )
    cap_label = (
        request.captured_piece_label
        if request.captured_piece_label is not None
        else planner.board.nearest_square(
            resolve_target_world_xyz(request.to_target, planner.board)[:2]
        )
    )

    leg1_req = MoveRequest(
        from_target=request.to_target,
        to_target=graveyard_xyz[:2],
        moving_piece_label=cap_label,
        moving_piece_symbol=request.captured_piece_symbol,
        moving_piece_height_m=request.captured_piece_height_m,
        moving_piece_radius_m=request.captured_piece_radius_m,
        place_release_clearance_m=request.place_release_clearance_m,
        is_capture=False,
        start_joints_deg=request.start_joints_deg.copy(),
        end_joints_deg=request.end_joints_deg.copy(),
        return_to_end=False,  # Stay safely at hover above graveyard to flow directly into Leg 2!
        gripper_open_deg=request.gripper_open_deg,
        gripper_closed_deg=request.gripper_closed_deg,
        pinch_offset_tcp_m=request.pinch_offset_tcp_m.copy(),
    )
    leg1 = planner.plan_single_leg(leg1_req, checker, dt_s=dt_s)

    # Update obstacle scene: remove captured piece from target square, place it in graveyard
    cap_h, cap_r = resolve_piece_dims(
        request.captured_piece_symbol,
        request.captured_piece_height_m,
        request.captured_piece_radius_m,
    )
    updated_pieces = [p for p in checker.pieces if p.label != cap_label.lower()]
    updated_pieces.append(
        PieceCylinder(
            label=f"graveyard_{request.graveyard_slot_index}",
            x_m=float(graveyard_xyz[0]),
            y_m=float(graveyard_xyz[1]),
            z_base_m=checker.table_z_m,
            height_m=cap_h,
            radius_m=cap_r,
            piece_type=request.captured_piece_symbol,
        )
    )
    checker_leg2 = checker.with_pieces(updated_pieces)

    leg2_req = MoveRequest(
        from_target=request.from_target,
        to_target=request.to_target,
        moving_piece_label=request.moving_piece_label,
        moving_piece_symbol=request.moving_piece_symbol,
        moving_piece_height_m=request.moving_piece_height_m,
        moving_piece_radius_m=request.moving_piece_radius_m,
        pick_yaw_deg=request.pick_yaw_deg,
        grasp_z_above_surface_m=request.grasp_z_above_surface_m,
        place_release_clearance_m=request.place_release_clearance_m,
        is_capture=False,
        start_joints_deg=leg1.commands_deg[-1].copy(),
        end_joints_deg=request.end_joints_deg.copy(),
        return_to_end=request.return_to_end,
        gripper_open_deg=request.gripper_open_deg,
        gripper_closed_deg=request.gripper_closed_deg,
        pinch_offset_tcp_m=request.pinch_offset_tcp_m.copy(),
    )
    leg2 = planner.plan_single_leg(leg2_req, checker_leg2, dt_s=dt_s)

    combined = np.vstack([leg1.commands_deg, leg2.commands_deg[1:]])
    return CaptureSequenceResult(
        leg1_clear_captured=leg1,
        leg2_move_attacker=leg2,
        combined_commands_deg=combined,
        total_planning_time_ms=leg1.planning_time_ms + leg2.planning_time_ms,
    )
