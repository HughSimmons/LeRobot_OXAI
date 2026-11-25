"""Candidate B: Task-Priority Damped Least Squares (DLS) Jacobian Planner with Null-Space Repulsion.

Directly generalizes the ``hardack`` QArm Mini DLS Jacobian controller
(``hardack/Hardware:Robotics/non-final/test_script1.py``) to the 6-DOF SO-101:
- Uses Layer-1 Coupled Attitude-Matched IK for ``q_pick`` and ``q_place``.
- Tracks raised parabolic/safe-arch Cartesian references for transit using the 3x5
  translational damped pseudo-inverse ``J^# = J^T (J J^T + lambda^2 I)^(-1)``.
- Projects analytical joint-limit centering, attitude interpolation, and finite-difference
  SDF obstacle repulsion gradients into the 2D null space ``(I_5 - J^# J)``, plus a
  task-space upward/azimuthal repulsive shift when clearance drops below ``repulse_margin_m``.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from .base import (
    MoveRequest,
    PlannedTrajectory,
    TrajectoryPlanner,
    TrajectorySegmentSpec,
    compile_segments_to_commands,
)
from .collision_sdf import HeldPieceAttachment, SceneCollisionChecker
from .kinematics_so101 import WORLD_UP


class DLSNullspacePlanner(TrajectoryPlanner):
    """Task-priority DLS Jacobian trajectory planner with null-space SDF obstacle avoidance."""

    name = "dls_nullspace"

    def __init__(
        self,
        kin,
        board,
        *,
        damping: float = 0.035,
        repulse_margin_m: float = 0.022,
        nullspace_obs_gain: float = 0.35,
        transit_waypoints: int = 16,
    ) -> None:
        super().__init__(kin, board)
        self.damping = float(damping)
        self.repulse_margin_m = float(repulse_margin_m)
        self.nullspace_obs_gain = float(nullspace_obs_gain)
        self.transit_waypoints = int(transit_waypoints)

    def _sdf_gradient_rad(
        self,
        q5_rad: np.ndarray,
        gripper_deg: float,
        checker: SceneCollisionChecker,
        exclude_labels: Sequence[str],
        held_piece: HeldPieceAttachment | None,
    ) -> tuple[float, np.ndarray]:
        """Vectorized 11-pose batch finite-difference gradient of ``min_clearance`` wrt ``q5_rad``."""
        eps_deg = 0.8
        eps_rad = np.deg2rad(eps_deg)
        base_deg = np.append(np.rad2deg(q5_rad), gripper_deg)
        Q_batch = np.repeat(base_deg[None, :], 11, axis=0)
        for j in range(5):
            Q_batch[1 + 2 * j, j] += eps_deg
            Q_batch[2 + 2 * j, j] -= eps_deg
        c_vals, _ = checker.evaluate_clearance_batch(
            Q_batch, exclude_labels=exclude_labels, held_piece=held_piece, allow_tip_near_board=False
        )
        c0 = float(c_vals[0])
        grad = (c_vals[1::2] - c_vals[2::2]) / (2.0 * eps_rad)
        return c0, grad

    def _plan_dls_transit(
        self,
        q_start_deg: np.ndarray,
        q_goal_deg: np.ndarray,
        *,
        gripper_deg: float,
        checker: SceneCollisionChecker,
        exclude_labels: Sequence[str],
        held_piece: HeldPieceAttachment | None,
        pinch_offset_tcp_m: np.ndarray,
        n_steps: int,
    ) -> np.ndarray:
        """Roll out differential DLS + null-space obstacle & posture tracking from ``q_start`` to ``q_goal``."""
        _, p_start = self.kin.fk_tcp(q_start_deg, pinch_offset_tcp_m)
        _, p_goal = self.kin.fk_tcp(q_goal_deg, pinch_offset_tcp_m)

        # Compute safe arch apex height above the tallest piece in the corridor
        obs_xy, _, obs_z1, _ = checker.active_obstacles(exclude_labels)
        corridor_top = self.board.surface_z_m + 0.045
        if obs_xy.shape[0] > 0:
            seg = p_goal[:2] - p_start[:2]
            seg_len_sq = float(np.dot(seg, seg))
            if seg_len_sq > 1e-8:
                t_proj = np.clip(((obs_xy - p_start[:2]) @ seg) / seg_len_sq, 0.0, 1.0)
                closest = p_start[:2] + t_proj[:, None] * seg[None, :]
                dists = np.linalg.norm(obs_xy - closest, axis=1)
            else:
                dists = np.linalg.norm(obs_xy - p_start[:2], axis=1)
            in_corridor = obs_z1[dists < 0.075]
            if in_corridor.size > 0:
                corridor_top = max(corridor_top, float(np.max(in_corridor)))

        held_extra = 0.0 if held_piece is None else held_piece.height_m
        mid_xy = 0.5 * (p_start[:2] + p_goal[:2])
        mid_radial = float(np.hypot(mid_xy[0] - 0.0388, mid_xy[1]))
        # Respect SO-101 workspace ceiling
        z_ceiling = self.board.surface_z_m + float(np.clip(0.24 - 0.36 * max(mid_radial - 0.25, 0.0), 0.085, 0.175))
        z_apex = min(max(p_start[2], p_goal[2], corridor_top + 0.55 * held_extra + 0.030), z_ceiling)
        arch_boost = max(0.0, z_apex - 0.5 * (p_start[2] + p_goal[2]))

        q_rad = np.deg2rad(q_start_deg[:5]).copy()
        q_goal_rad = np.deg2rad(q_goal_deg[:5]).copy()
        q_mid_limits = 0.5 * (self.kin.arm_lo_rad + self.kin.arm_hi_rad)
        q_span = np.maximum(self.kin.arm_hi_rad - self.kin.arm_lo_rad, 1e-3)

        waypoints = np.zeros((n_steps + 1, 6), dtype=np.float64)
        waypoints[0, :5] = q_start_deg[:5]
        waypoints[0, 5] = gripper_deg

        I5 = np.eye(5, dtype=np.float64)
        lam2 = self.damping ** 2

        for k in range(1, n_steps):
            s = k / n_steps
            p_ref = (1.0 - s) * p_start + s * p_goal
            p_ref[2] += 4.0 * s * (1.0 - s) * arch_boost
            q_posture_ref = (1.0 - s) * np.deg2rad(q_start_deg[:5]) + s * q_goal_rad

            for _ in range(4):
                p_cur, _, J_pos, _, _ = self.kin.fk_jacobian(q_rad, pinch_offset_tcp_m)
                c0, grad_c = self._sdf_gradient_rad(
                    q_rad, gripper_deg, checker, exclude_labels, held_piece
                )

                e_pos = p_ref - p_cur
                if c0 < self.repulse_margin_m:
                    # Task-space upward bias if approaching an obstacle
                    e_pos[2] += 0.35 * (self.repulse_margin_m - c0)

                # 3x5 damped Moore-Penrose pseudo-inverse (hardack DLS generalized)
                JJT = J_pos @ J_pos.T + lam2 * np.eye(3, dtype=np.float64)
                J_pinv = J_pos.T @ np.linalg.inv(JJT)
                N_null = I5 - J_pinv @ J_pos

                # Secondary task in 2D null space: SDF clearance gradient + goal posture tracking + limit centering
                g_sec = 0.35 * (q_posture_ref - q_rad) - 0.05 * ((q_rad - q_mid_limits) / q_span)
                if c0 < self.repulse_margin_m:
                    weight_obs = self.nullspace_obs_gain * ((self.repulse_margin_m - c0) / self.repulse_margin_m)
                    g_sec = g_sec + weight_obs * grad_c

                dq = 0.65 * (J_pinv @ e_pos) + N_null @ g_sec
                dq = np.clip(dq, -0.18, 0.18)
                q_rad = np.clip(q_rad + dq, self.kin.arm_lo_rad, self.kin.arm_hi_rad)

            waypoints[k, :5] = np.rad2deg(q_rad)
            waypoints[k, 5] = gripper_deg

        waypoints[-1, :5] = q_goal_deg[:5]
        waypoints[-1, 5] = gripper_deg
        return waypoints

    def plan_single_leg(
        self,
        request: MoveRequest,
        checker: SceneCollisionChecker,
        *,
        dt_s: float = 1.0 / 240.0,
    ) -> PlannedTrajectory:
        t0 = time.perf_counter()
        ep = self.solve_endpoints_and_vertical_strokes(request, checker)
        coupled = ep["coupled"]
        excl = (ep["pick_label"],)
        held_piece = ep["held_piece"]
        grip_open = ep["grip_open"]
        grip_closed = ep["grip_closed"]

        q_start = request.start_joints_deg.copy()
        q_start[5] = grip_open

        approach_wp = self._plan_dls_transit(
            q_start,
            ep["q_hover_pick_open"],
            gripper_deg=grip_open,
            checker=checker,
            exclude_labels=(),
            held_piece=None,
            pinch_offset_tcp_m=request.pinch_offset_tcp_m,
            n_steps=self.transit_waypoints,
        )
        carry_wp = self._plan_dls_transit(
            ep["q_hover_pick_closed"],
            ep["q_hover_place_closed"],
            gripper_deg=grip_closed,
            checker=checker,
            exclude_labels=excl,
            held_piece=held_piece,
            pinch_offset_tcp_m=request.pinch_offset_tcp_m,
            n_steps=self.transit_waypoints,
        )

        segments: list[TrajectorySegmentSpec] = [
            TrajectorySegmentSpec("approach_hover_pick", approach_wp, duration_s=1.1),
            TrajectorySegmentSpec("lower_to_pick", ep["lower_to_pick_open"], duration_s=0.65, is_rest_endpoint=True),
            TrajectorySegmentSpec(
                "close_gripper",
                np.vstack([coupled.q_pick_deg, np.append(coupled.q_pick_deg[:5], grip_closed)]),
                duration_s=0.45,
                is_dwell=False,
                is_rest_endpoint=True,
            ),
            TrajectorySegmentSpec("lift_from_pick", ep["lift_from_pick_closed"], duration_s=0.65),
            TrajectorySegmentSpec("transit_carry", carry_wp, duration_s=max(1.25, float(np.sum(np.linalg.norm(np.diff(carry_wp[:, :5], axis=0), axis=1))) / 120.0)),
            TrajectorySegmentSpec("lower_to_place", ep["lower_to_place_closed"], duration_s=0.75, is_rest_endpoint=True),
            TrajectorySegmentSpec(
                "open_gripper",
                np.vstack([np.append(coupled.q_place_deg[:5], grip_closed), np.append(coupled.q_place_deg[:5], ep["grip_open_place"])]),
                duration_s=0.45,
                is_dwell=False,
                is_rest_endpoint=True,
            ),
            TrajectorySegmentSpec("retreat_from_place", ep["retreat_from_place_open"], duration_s=0.60),
        ]

        if request.return_to_end:
            q_end = request.end_joints_deg.copy()
            q_end[5] = ep["grip_open_place"]
            return_wp = self._plan_dls_transit(
                ep["q_hover_place_open"],
                q_end,
                gripper_deg=ep["grip_open_place"],
                checker=checker,
                exclude_labels=excl,
                held_piece=None,
                pinch_offset_tcp_m=request.pinch_offset_tcp_m,
                n_steps=max(10, self.transit_waypoints - 4),
            )
            segments.append(TrajectorySegmentSpec("return_end", return_wp, duration_s=1.1, is_rest_endpoint=True))

        waypoints, commands, seg_ranges, rest_idx, path_len, ms_jerk = compile_segments_to_commands(
            segments, dt_s=dt_s
        )

        # Evaluate clearance metrics along transit and full move
        tr_s, tr_e = seg_ranges["transit_carry"]
        c_transit, _ = checker.evaluate_clearance_batch(
            commands[tr_s:tr_e:4], exclude_labels=excl, held_piece=held_piece, allow_tip_near_board=False
        )
        ap_s, ap_e = seg_ranges["approach_hover_pick"]
        c_approach, _ = checker.evaluate_clearance_batch(
            commands[ap_s:ap_e:4], exclude_labels=(), held_piece=None, allow_tip_near_board=False
        )
        min_transit_mm = float(min(np.min(c_transit), np.min(c_approach)) * 1000.0)
        c_pick_place, _ = checker.evaluate_clearance_batch(
            np.vstack([ep["lower_to_pick_open"], ep["lower_to_place_closed"]]),
            exclude_labels=excl,
            allow_tip_near_board=True,
        )
        min_all_mm = min(min_transit_mm, float(np.min(c_pick_place) * 1000.0))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return PlannedTrajectory(
            planner_name=self.name,
            waypoints_deg=waypoints,
            commands_deg=commands,
            dt_s=dt_s,
            segment_ranges=seg_ranges,
            rest_command_indices=rest_idx,
            close_gripper_cmd_idx=seg_ranges["close_gripper"][0],
            open_gripper_cmd_idx=seg_ranges["open_gripper"][0],
            coupled_ik=coupled,
            planning_time_ms=elapsed_ms,
            min_transit_clearance_mm=min_transit_mm,
            min_overall_clearance_mm=min_all_mm,
            joint_path_length_deg=path_len,
            mean_squared_jerk=ms_jerk,
        )
