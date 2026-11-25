"""Candidate A (Primary Recommendation): Safe-Corridor Cubic B-Spline Trajectory Optimizer.

Combines:
- Layer 1: Coupled Attitude-Matched Pick-Place IK + Locked-Attitude DLS vertical strokes.
- Layer 2: Multi-seed high-clearance arch / unfold-swing initialization followed by
  vectorized L-BFGS-B cubic B-spline optimization over interior joint control points,
  minimizing joint velocity/acceleration energy + quadratic hinge SDF collision penalties
  for all robot links, wrist camera, jaws, and the rigidly attached held piece.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from planner.base import (
    MoveRequest,
    PlannedTrajectory,
    TrajectoryPlanner,
    TrajectorySegmentSpec,
    compile_segments_to_commands,
)
from planner.collision_sdf import HeldPieceAttachment, SceneCollisionChecker


def build_clamped_cubic_bspline_basis(n_samples: int, n_interior: int = 3) -> np.ndarray:
    """Build ``(n_samples, n_interior + 2)`` clamped cubic B-spline basis matrix.

    Control points ``P_0, P_1, ..., P_{m-1}`` (``m = n_interior + 2``) interpolate ``P_0`` at
    ``u = 0`` and ``P_{m-1}`` at ``u = 1`` while staying strictly inside the convex hull of ``P``.
    """
    m = n_interior + 2
    degree = 3
    # Clamped knot vector of length m + degree + 1
    n_internal_knots = max(0, m - degree - 1)
    if n_internal_knots > 0:
        internal = np.linspace(0.0, 1.0, n_internal_knots + 2)[1:-1]
    else:
        internal = np.array([], dtype=np.float64)
    knots = np.concatenate(
        [np.zeros(degree + 1), internal, np.ones(degree + 1)]
    )

    u_vals = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    # Cox-de Boor recursion
    B = np.zeros((n_samples, m, degree + 1), dtype=np.float64)
    for i in range(m):
        if i == m - 1:
            B[:, i, 0] = np.where((u_vals >= knots[i]) & (u_vals <= knots[i + 1]), 1.0, 0.0)
        else:
            B[:, i, 0] = np.where((u_vals >= knots[i]) & (u_vals < knots[i + 1]), 1.0, 0.0)
    # Ensure exact endpoint u=1.0 lands on last basis function
    B[-1, :, 0] = 0.0
    B[-1, m - 1, 0] = 1.0

    for d in range(1, degree + 1):
        for i in range(m):
            denom1 = knots[i + d] - knots[i]
            term1 = ((u_vals - knots[i]) / denom1) * B[:, i, d - 1] if denom1 > 1e-12 else 0.0
            denom2 = knots[i + d + 1] - knots[i + 1]
            term2 = ((knots[i + d + 1] - u_vals) / denom2) * B[:, i + 1, d - 1] if (i + 1 < m and denom2 > 1e-12) else 0.0
            B[:, i, d] = term1 + term2

    basis = B[:, :, degree]
    basis[0, :] = 0.0
    basis[0, 0] = 1.0
    basis[-1, :] = 0.0
    basis[-1, -1] = 1.0
    return basis


class TrajOptSplinePlanner(TrajectoryPlanner):
    """Deterministic safe-corridor cubic B-spline trajectory optimizer (~10-25 ms)."""

    name = "trajopt_spline"

    def __init__(
        self,
        kin,
        board,
        *,
        n_interior_ctrl: int = 3,
        n_eval_samples: int = 24,
        safe_clearance_m: float = 0.018,
        collision_weight: float = 2.5e6,
        max_opt_iters: int = 28,
    ) -> None:
        super().__init__(kin, board)
        self.n_interior = int(n_interior_ctrl)
        self.n_eval = int(n_eval_samples)
        self.safe_clearance_m = float(safe_clearance_m)
        self.collision_weight = float(collision_weight)
        self.max_opt_iters = int(max_opt_iters)

        self._basis = build_clamped_cubic_bspline_basis(self.n_eval, self.n_interior)  # (N, K+2)
        self._B0 = self._basis[:, 0:1]      # (N, 1)
        self._B_int = self._basis[:, 1:-1]  # (N, K)
        self._B1 = self._basis[:, -1:]      # (N, 1)

        # Smoothness Laplacian (velocity + acceleration finite-difference matrices)
        D1 = np.diff(np.eye(self.n_eval), n=1, axis=0)
        D2 = np.diff(np.eye(self.n_eval), n=2, axis=0)
        self._L_smooth = 0.002 * (D1.T @ D1) + 0.025 * (D2.T @ D2)

    def _eval_spline_arm(self, q0_5: np.ndarray, q1_5: np.ndarray, C_int: np.ndarray) -> np.ndarray:
        """Evaluate ``(N, 5)`` arm trajectory from endpoints and ``(K, 5)`` interior control points."""
        return self._B0 @ q0_5[None, :] + self._B_int @ C_int + self._B1 @ q1_5[None, :]

    def _candidate_seed_controls(self, q0_5: np.ndarray, q1_5: np.ndarray) -> list[np.ndarray]:
        """Generate smart candidate interior control matrices ``(K, 5)`` (arch, unfold-swing, swing-unfold)."""
        k = self.n_interior
        alphas = np.linspace(0.0, 1.0, k + 2)[1:-1][:, None]  # (K, 1)
        lin = (1.0 - alphas) * q0_5[None, :] + alphas * q1_5[None, :]

        # Seed 1: Direct mild high-clearance arch (lift shoulder_lift toward -35 deg, tuck elbow slightly)
        arch1 = lin.copy()
        arch_bell = 4.0 * alphas * (1.0 - alphas)
        arch1[:, 1] = np.clip(lin[:, 1] - 22.0 * arch_bell[:, 0], self.kin.arm_lo_deg[1], self.kin.arm_hi_deg[1])
        arch1[:, 2] = np.clip(lin[:, 2] + 10.0 * arch_bell[:, 0], self.kin.arm_lo_deg[2], self.kin.arm_hi_deg[2])

        # Seed 2: Taller clearance arch (clears 95 mm Kings + 50 mm held piece)
        arch2 = lin.copy()
        high_pose = np.array(
            [
                0.5 * (q0_5[0] + q1_5[0]),
                min(-35.0, 0.5 * (q0_5[1] + q1_5[1]) - 32.0),
                min(45.0, max(0.0, 0.5 * (q0_5[2] + q1_5[2]) + 15.0)),
                55.0,
                0.5 * (q0_5[4] + q1_5[4]),
            ],
            dtype=np.float64,
        )
        high_pose = np.clip(high_pose, self.kin.arm_lo_deg, self.kin.arm_hi_deg)
        arch2 = (1.0 - arch_bell) * lin + arch_bell * high_pose[None, :]

        # Seed 3: Unfold-then-swing (keep start pan while lifting high, then swing pan to goal)
        unfold_swing = lin.copy()
        if k >= 3:
            unfold_swing[0] = np.array([q0_5[0], min(q0_5[1], -35.0), max(q0_5[2], 25.0), 55.0, q0_5[4]])
            unfold_swing[1] = np.array([0.5 * (q0_5[0] + q1_5[0]), -38.0, 25.0, 55.0, 0.5 * (q0_5[4] + q1_5[4])])
            unfold_swing[2] = np.array([q1_5[0], min(q1_5[1], -30.0), max(q1_5[2], 15.0), 55.0, q1_5[4]])
            unfold_swing = np.clip(unfold_swing, self.kin.arm_lo_deg, self.kin.arm_hi_deg)

        # Seed 4: Swing-then-unfold (for returning to folded DEFAULT_HOME safely outside board)
        swing_unfold = lin.copy()
        if k >= 3:
            swing_unfold[0] = np.array([q0_5[0], min(q0_5[1], -30.0), max(q0_5[2], 15.0), 55.0, q0_5[4]])
            swing_unfold[1] = np.array([q1_5[0], -45.0, 45.0, 60.0, q1_5[4]])
            swing_unfold[2] = np.array([q1_5[0], 0.5 * (-45.0 + q1_5[1]), 0.5 * (45.0 + q1_5[2]), q1_5[3], q1_5[4]])
            swing_unfold = np.clip(swing_unfold, self.kin.arm_lo_deg, self.kin.arm_hi_deg)

        return [lin, arch1, arch2, unfold_swing, swing_unfold]

    def optimize_transit_spline(
        self,
        q_start_deg: np.ndarray,
        q_goal_deg: np.ndarray,
        *,
        gripper_deg: float,
        checker: SceneCollisionChecker,
        exclude_labels: Sequence[str],
        held_piece: HeldPieceAttachment | None,
    ) -> np.ndarray:
        """Optimize interior cubic B-spline control points for a collision-free smooth transit."""
        q0 = np.asarray(q_start_deg[:5], dtype=np.float64)
        q1 = np.asarray(q_goal_deg[:5], dtype=np.float64)
        k = self.n_interior
        dim = k * 5

        seeds = self._candidate_seed_controls(q0, q1)
        # Evaluate all seeds in a single batch SDF pass!
        seed_trajs = []
        for C in seeds:
            Q_arm = self._eval_spline_arm(q0, q1, C)
            Q6 = np.column_stack([Q_arm, np.full(self.n_eval, gripper_deg, dtype=np.float64)])
            seed_trajs.append(Q6)

        Q_all_seeds = np.vstack(seed_trajs)
        c_all, _ = checker.evaluate_clearance_batch(
            Q_all_seeds, exclude_labels=exclude_labels, held_piece=held_piece, allow_tip_near_board=False
        )
        c_per_seed = c_all.reshape(len(seeds), self.n_eval)

        best_seed_idx = 0
        best_seed_cost = np.inf
        for idx, C in enumerate(seeds):
            Q_arm = seed_trajs[idx][:, :5]
            smooth = 0.5 * float(np.sum(Q_arm * (self._L_smooth @ Q_arm)))
            viol = np.maximum(0.0, self.safe_clearance_m - c_per_seed[idx, 1:-1])
            col = 0.5 * self.collision_weight * float(np.sum(viol * viol))
            total = smooth + col
            if total < best_seed_cost:
                best_seed_cost = total
                best_seed_idx = idx

        C_init = seeds[best_seed_idx].copy()
        min_init_clear = float(np.min(c_per_seed[best_seed_idx, 1:-1]))

        # Fast-path: if already well above safe_clearance_m on the linear/mild-arch seed, return directly!
        if min_init_clear >= self.safe_clearance_m and best_seed_idx in (0, 1):
            Q_arm = self._eval_spline_arm(q0, q1, C_init)
            return np.column_stack([Q_arm, np.full(self.n_eval, gripper_deg, dtype=np.float64)])

        lo_flat = np.tile(self.kin.arm_lo_deg, k)
        hi_flat = np.tile(self.kin.arm_hi_deg, k)
        bounds = list(zip(lo_flat.tolist(), hi_flat.tolist()))

        def loss_and_grad(x_flat: np.ndarray) -> tuple[float, np.ndarray]:
            C0 = x_flat.reshape(k, 5)
            Q0_arm = self._eval_spline_arm(q0, q1, C0)

            # Analytical smoothness value + gradient
            LQ = self._L_smooth @ Q0_arm
            smooth_val = 0.5 * float(np.sum(Q0_arm * LQ))
            grad_smooth = (self._B_int.T @ LQ).reshape(-1)

            # Single-pass analytical SDF hinge loss + exact Jacobian backprop to spline control points!
            Q6_int = np.column_stack(
                [Q0_arm[1:-1], np.full(self.n_eval - 2, gripper_deg, dtype=np.float64)]
            )
            col_unit_loss, grad_Q_int_deg, _ = checker.hinge_penalty_and_grad_batch(
                Q6_int,
                self.safe_clearance_m,
                exclude_labels=exclude_labels,
                held_piece=held_piece,
            )
            col_val = self.collision_weight * col_unit_loss
            grad_col = self.collision_weight * (self._B_int[1:-1, :].T @ grad_Q_int_deg[:, :5]).reshape(-1)
            return smooth_val + col_val, grad_smooth + grad_col

        res = minimize(
            loss_and_grad,
            C_init.reshape(-1),
            method="SLSQP",
            jac=True,
            bounds=bounds,
            options={"maxiter": self.max_opt_iters, "ftol": 1e-4},
        )
        C_opt = np.clip(res.x.reshape(k, 5), self.kin.arm_lo_deg, self.kin.arm_hi_deg)
        Q_opt_arm = self._eval_spline_arm(q0, q1, C_opt)
        return np.column_stack([Q_opt_arm, np.full(self.n_eval, gripper_deg, dtype=np.float64)])

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

        approach_wp = self.optimize_transit_spline(
            q_start,
            ep["q_hover_pick_open"],
            gripper_deg=grip_open,
            checker=checker,
            exclude_labels=(),
            held_piece=None,
        )
        carry_wp = self.optimize_transit_spline(
            ep["q_hover_pick_closed"],
            ep["q_hover_place_closed"],
            gripper_deg=grip_closed,
            checker=checker,
            exclude_labels=excl,
            held_piece=held_piece,
        )

        segments: list[TrajectorySegmentSpec] = [
            TrajectorySegmentSpec("approach_hover_pick", approach_wp, duration_s=1.05),
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
            return_wp = self.optimize_transit_spline(
                ep["q_hover_place_open"],
                q_end,
                gripper_deg=ep["grip_open_place"],
                checker=checker,
                exclude_labels=excl,
                held_piece=None,
            )
            segments.append(TrajectorySegmentSpec("return_end", return_wp, duration_s=1.05, is_rest_endpoint=True))

        waypoints, commands, seg_ranges, rest_idx, path_len, ms_jerk = compile_segments_to_commands(
            segments, dt_s=dt_s
        )

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
