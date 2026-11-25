"""Candidate C: Joint-Space Bi-Directional RRT-Connect + Shortcutting, plus Unified Hybrid Planner.

- ``RRTConnectPlanner``: Probabilistically complete 5D joint-space Bi-Directional RRT-Connect
  using vectorized batch Capsule-SDF edge checking (including the rigidly attached held piece),
  high-clearance bridge sampling bias, and randomized shortcutting.
- ``HybridPlanner`` (``hybrid_trajopt_rrt``): Runs ``TrajOptSplinePlanner`` as the deterministic
  ~15 ms primary solver, and invokes ``RRTConnectPlanner`` only on any transit segment whose
  minimum SDF clearance falls below ``fallback_clearance_mm``.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from planner.base import (
    MoveRequest,
    PlannedTrajectory,
    TrajectoryPlanner,
    TrajectorySegmentSpec,
    compile_segments_to_commands,
)
from planner.collision_sdf import HeldPieceAttachment, SceneCollisionChecker
from .trajopt_spline import TrajOptSplinePlanner


class RRTConnectPlanner(TrajectoryPlanner):
    """Bi-Directional RRT-Connect + shortcut smoothing in 5D SO-101 arm joint space."""

    name = "rrt_connect"

    def __init__(
        self,
        kin,
        board,
        *,
        step_size_deg: float = 16.0,
        edge_resolution_deg: float = 3.5,
        min_clearance_m: float = 0.004,
        max_iters: int = 350,
        shortcut_iters: int = 30,
        seed: int = 7,
    ) -> None:
        super().__init__(kin, board)
        self.step_size_deg = float(step_size_deg)
        self.edge_resolution_deg = float(edge_resolution_deg)
        self.min_clearance_m = float(min_clearance_m)
        self.max_iters = int(max_iters)
        self.shortcut_iters = int(shortcut_iters)
        self.seed = int(seed)

    def _edge_is_valid(
        self,
        q_from_5: np.ndarray,
        q_to_5: np.ndarray,
        *,
        gripper_deg: float,
        checker: SceneCollisionChecker,
        exclude_labels: Sequence[str],
        held_piece: HeldPieceAttachment | None,
        clearance_thresh_m: float,
    ) -> bool:
        dist = float(np.linalg.norm(q_to_5 - q_from_5))
        n_samples = max(2, int(np.ceil(dist / self.edge_resolution_deg)) + 1)
        alphas = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)[:, None]
        Q_arm = (1.0 - alphas) * q_from_5[None, :] + alphas * q_to_5[None, :]
        # Exclude exact endpoints (index 0 and -1) if start/hover sits slightly inside soft margin,
        # while strictly checking all interior configurations.
        Q_mid = Q_arm[1:-1] if n_samples > 2 else Q_arm
        if Q_mid.shape[0] == 0:
            return True
        Q6 = np.column_stack([Q_mid, np.full(Q_mid.shape[0], gripper_deg, dtype=np.float64)])
        c_vals, _ = checker.evaluate_clearance_batch(
            Q6, exclude_labels=exclude_labels, held_piece=held_piece, allow_tip_near_board=False
        )
        return bool(np.all(c_vals >= clearance_thresh_m))

    def _shortcut_path(
        self,
        path_5: list[np.ndarray],
        *,
        gripper_deg: float,
        checker: SceneCollisionChecker,
        exclude_labels: Sequence[str],
        held_piece: HeldPieceAttachment | None,
        clearance_thresh_m: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        pts = [p.copy() for p in path_5]
        for _ in range(self.shortcut_iters):
            if len(pts) <= 2:
                break
            i = int(rng.integers(0, len(pts) - 2))
            j = int(rng.integers(i + 2, len(pts)))
            if self._edge_is_valid(
                pts[i],
                pts[j],
                gripper_deg=gripper_deg,
                checker=checker,
                exclude_labels=exclude_labels,
                held_piece=held_piece,
                clearance_thresh_m=clearance_thresh_m + 0.002,
            ):
                pts = pts[: i + 1] + pts[j:]

        # Subdivide shortcutted polyline to ~14 smooth waypoints
        raw = np.vstack(pts)
        if raw.shape[0] == 2:
            alphas = np.linspace(0.0, 1.0, 12, dtype=np.float64)[:, None]
            dense_arm = (1.0 - alphas) * raw[0:1] + alphas * raw[1:2]
        else:
            dense_blocks = []
            for k in range(raw.shape[0] - 1):
                seg_len = float(np.linalg.norm(raw[k + 1] - raw[k]))
                n_sub = max(2, int(np.ceil(seg_len / 12.0)))
                alphas = np.linspace(0.0, 1.0, n_sub + 1, dtype=np.float64)[:, None]
                seg_pts = (1.0 - alphas) * raw[k : k + 1] + alphas * raw[k + 1 : k + 2]
                dense_blocks.append(seg_pts if k == 0 else seg_pts[1:])
            dense_arm = np.vstack(dense_blocks)
        return np.column_stack([dense_arm, np.full(dense_arm.shape[0], gripper_deg, dtype=np.float64)])

    def plan_rrt_transit(
        self,
        q_start_deg: np.ndarray,
        q_goal_deg: np.ndarray,
        *,
        gripper_deg: float,
        checker: SceneCollisionChecker,
        exclude_labels: Sequence[str],
        held_piece: HeldPieceAttachment | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Connect ``q_start_deg`` to ``q_goal_deg`` via Bi-Directional RRT-Connect + shortcutting."""
        q0 = np.asarray(q_start_deg[:5], dtype=np.float64).copy()
        q1 = np.asarray(q_goal_deg[:5], dtype=np.float64).copy()

        # Check direct straight line first
        if self._edge_is_valid(
            q0,
            q1,
            gripper_deg=gripper_deg,
            checker=checker,
            exclude_labels=exclude_labels,
            held_piece=held_piece,
            clearance_thresh_m=self.min_clearance_m + 0.008,
        ):
            alphas = np.linspace(0.0, 1.0, 12, dtype=np.float64)[:, None]
            arm = (1.0 - alphas) * q0[None, :] + alphas * q1[None, :]
            return np.column_stack([arm, np.full(12, gripper_deg, dtype=np.float64)])

        for thresh in (self.min_clearance_m, 0.0, -0.003):
            nodes_a = [q0]
            parents_a = [-1]
            nodes_b = [q1]
            parents_b = [-1]
            swapped = False

            for it in range(self.max_iters):
                u = float(rng.random())
                if u < 0.15:
                    q_rand = nodes_b[0].copy()
                elif u < 0.55:
                    # High-clearance bridge bias: keeps shoulder lifted above standing pieces
                    pan = float(rng.uniform(min(q0[0], q1[0]) - 25.0, max(q0[0], q1[0]) + 25.0))
                    lift = float(rng.uniform(-75.0, -10.0))
                    elbow = float(rng.uniform(-20.0, 75.0))
                    wflex = float(rng.uniform(15.0, 85.0))
                    wroll = float(rng.uniform(min(q0[4], q1[4]) - 35.0, max(q0[4], q1[4]) + 35.0))
                    q_rand = np.clip([pan, lift, elbow, wflex, wroll], self.kin.arm_lo_deg, self.kin.arm_hi_deg)
                else:
                    q_rand = rng.uniform(self.kin.arm_lo_deg, self.kin.arm_hi_deg)

                # Extend tree A one step toward q_rand
                arr_a = np.vstack(nodes_a)
                nearest_a = int(np.argmin(np.sum((arr_a - q_rand[None, :]) ** 2, axis=1)))
                q_near_a = nodes_a[nearest_a]
                diff_a = q_rand - q_near_a
                dist_a = float(np.linalg.norm(diff_a))
                if dist_a < 1e-6:
                    continue
                q_new_a = q_near_a + min(1.0, self.step_size_deg / dist_a) * diff_a
                q_new_a = np.clip(q_new_a, self.kin.arm_lo_deg, self.kin.arm_hi_deg)

                if not self._edge_is_valid(
                    q_near_a,
                    q_new_a,
                    gripper_deg=gripper_deg,
                    checker=checker,
                    exclude_labels=exclude_labels,
                    held_piece=held_piece,
                    clearance_thresh_m=thresh,
                ):
                    nodes_a, nodes_b = nodes_b, nodes_a
                    parents_a, parents_b = parents_b, parents_a
                    swapped = not swapped
                    continue

                new_idx_a = len(nodes_a)
                nodes_a.append(q_new_a)
                parents_a.append(nearest_a)

                # Greedily connect tree B toward q_new_a
                arr_b = np.vstack(nodes_b)
                curr_b = int(np.argmin(np.sum((arr_b - q_new_a[None, :]) ** 2, axis=1)))
                connected = False
                for _ in range(32):
                    q_curr_b = nodes_b[curr_b]
                    diff_b = q_new_a - q_curr_b
                    dist_b = float(np.linalg.norm(diff_b))
                    if dist_b <= self.step_size_deg:
                        if self._edge_is_valid(
                            q_curr_b,
                            q_new_a,
                            gripper_deg=gripper_deg,
                            checker=checker,
                            exclude_labels=exclude_labels,
                            held_piece=held_piece,
                            clearance_thresh_m=thresh,
                        ):
                            conn_idx_b = len(nodes_b)
                            nodes_b.append(q_new_a.copy())
                            parents_b.append(curr_b)
                            connected = True
                        break
                    q_step_b = q_curr_b + (self.step_size_deg / dist_b) * diff_b
                    q_step_b = np.clip(q_step_b, self.kin.arm_lo_deg, self.kin.arm_hi_deg)
                    if not self._edge_is_valid(
                        q_curr_b,
                        q_step_b,
                        gripper_deg=gripper_deg,
                        checker=checker,
                        exclude_labels=exclude_labels,
                        held_piece=held_piece,
                        clearance_thresh_m=thresh,
                    ):
                        break
                    prev_b = curr_b
                    curr_b = len(nodes_b)
                    nodes_b.append(q_step_b)
                    parents_b.append(prev_b)
                    arr_b = np.vstack([arr_b, q_step_b[None, :]])

                if connected:
                    # Reconstruct full path
                    chain_a = []
                    idx = new_idx_a
                    while idx != -1:
                        chain_a.append(nodes_a[idx])
                        idx = parents_a[idx]
                    chain_a.reverse()

                    chain_b = []
                    idx = parents_b[conn_idx_b]
                    while idx != -1:
                        chain_b.append(nodes_b[idx])
                        idx = parents_b[idx]

                    full_chain = chain_a + chain_b
                    if swapped:
                        full_chain.reverse()
                    return self._shortcut_path(
                        full_chain,
                        gripper_deg=gripper_deg,
                        checker=checker,
                        exclude_labels=exclude_labels,
                        held_piece=held_piece,
                        clearance_thresh_m=thresh,
                        rng=rng,
                    )

                nodes_a, nodes_b = nodes_b, nodes_a
                parents_a, parents_b = parents_b, parents_a
                swapped = not swapped

        # Fallback high-arch bridge if tree budget exhausted
        mid = 0.5 * (q0 + q1)
        mid[1] = min(-45.0, mid[1] - 30.0)
        mid = np.clip(mid, self.kin.arm_lo_deg, self.kin.arm_hi_deg)
        return self._shortcut_path(
            [q0, mid, q1],
            gripper_deg=gripper_deg,
            checker=checker,
            exclude_labels=exclude_labels,
            held_piece=held_piece,
            clearance_thresh_m=-1.0,
            rng=rng,
        )

    def plan_single_leg(
        self,
        request: MoveRequest,
        checker: SceneCollisionChecker,
        *,
        dt_s: float = 1.0 / 240.0,
    ) -> PlannedTrajectory:
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        ep = self.solve_endpoints_and_vertical_strokes(request, checker)
        coupled = ep["coupled"]
        excl = (ep["pick_label"],)
        held_piece = ep["held_piece"]
        grip_open = ep["grip_open"]
        grip_closed = ep["grip_closed"]

        q_start = request.start_joints_deg.copy()
        q_start[5] = grip_open

        approach_wp = self.plan_rrt_transit(
            q_start,
            ep["q_hover_pick_open"],
            gripper_deg=grip_open,
            checker=checker,
            exclude_labels=(),
            held_piece=None,
            rng=rng,
        )
        carry_wp = self.plan_rrt_transit(
            ep["q_hover_pick_closed"],
            ep["q_hover_place_closed"],
            gripper_deg=grip_closed,
            checker=checker,
            exclude_labels=excl,
            held_piece=held_piece,
            rng=rng,
        )

        segments: list[TrajectorySegmentSpec] = [
            TrajectorySegmentSpec("approach_hover_pick", approach_wp, duration_s=1.15),
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
            return_wp = self.plan_rrt_transit(
                ep["q_hover_place_open"],
                q_end,
                gripper_deg=ep["grip_open_place"],
                checker=checker,
                exclude_labels=excl,
                held_piece=None,
                rng=rng,
            )
            segments.append(TrajectorySegmentSpec("return_end", return_wp, duration_s=1.10, is_rest_endpoint=True))

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


class HybridPlanner(TrajectoryPlanner):
    """Production Hybrid Planner: ``TrajOptSplinePlanner`` fast path + ``RRTConnectPlanner`` fallback."""

    name = "hybrid_trajopt_rrt"

    def __init__(
        self,
        kin,
        board,
        *,
        fallback_clearance_mm: float = -1.0,
        seed: int = 7,
    ) -> None:
        super().__init__(kin, board)
        self.trajopt = TrajOptSplinePlanner(kin, board)
        self.rrt = RRTConnectPlanner(kin, board, seed=seed)
        self.fallback_clearance_mm = float(fallback_clearance_mm)

    def plan_single_leg(
        self,
        request: MoveRequest,
        checker: SceneCollisionChecker,
        *,
        dt_s: float = 1.0 / 240.0,
    ) -> PlannedTrajectory:
        t0 = time.perf_counter()
        primary = self.trajopt.plan_single_leg(request, checker, dt_s=dt_s)
        if primary.min_transit_clearance_mm >= self.fallback_clearance_mm:
            return PlannedTrajectory(
                planner_name=self.name,
                waypoints_deg=primary.waypoints_deg,
                commands_deg=primary.commands_deg,
                dt_s=primary.dt_s,
                segment_ranges=primary.segment_ranges,
                rest_command_indices=primary.rest_command_indices,
                close_gripper_cmd_idx=primary.close_gripper_cmd_idx,
                open_gripper_cmd_idx=primary.open_gripper_cmd_idx,
                coupled_ik=primary.coupled_ik,
                planning_time_ms=(time.perf_counter() - t0) * 1000.0,
                min_transit_clearance_mm=primary.min_transit_clearance_mm,
                min_overall_clearance_mm=primary.min_overall_clearance_mm,
                joint_path_length_deg=primary.joint_path_length_deg,
                mean_squared_jerk=primary.mean_squared_jerk,
                fallback_used=False,
                notes=["trajopt_spline_cleared"],
            )

        rrt_plan = self.rrt.plan_single_leg(request, checker, dt_s=dt_s)
        chosen = (
            rrt_plan
            if rrt_plan.min_transit_clearance_mm > primary.min_transit_clearance_mm + 1.5
            else primary
        )
        return PlannedTrajectory(
            planner_name=self.name,
            waypoints_deg=chosen.waypoints_deg,
            commands_deg=chosen.commands_deg,
            dt_s=chosen.dt_s,
            segment_ranges=chosen.segment_ranges,
            rest_command_indices=chosen.rest_command_indices,
            close_gripper_cmd_idx=chosen.close_gripper_cmd_idx,
            open_gripper_cmd_idx=chosen.open_gripper_cmd_idx,
            coupled_ik=chosen.coupled_ik,
            planning_time_ms=(time.perf_counter() - t0) * 1000.0,
            min_transit_clearance_mm=chosen.min_transit_clearance_mm,
            min_overall_clearance_mm=chosen.min_overall_clearance_mm,
            joint_path_length_deg=chosen.joint_path_length_deg,
            mean_squared_jerk=chosen.mean_squared_jerk,
            fallback_used=(chosen is rrt_plan),
            notes=[f"fallback_triggered_primary_clear={primary.min_transit_clearance_mm:.1f}mm"],
        )
