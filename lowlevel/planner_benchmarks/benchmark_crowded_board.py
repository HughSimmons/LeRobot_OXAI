#!/usr/bin/env python3
"""Head-to-head PyBullet V-HACD benchmark across crowded multi-piece chessboards.

Evaluates five trajectory planners on 16 standardized crowded-board moves:
1. ``legacy_chess_traj``  -- Existing Placo soft-IK + Catmull-Rom baseline (``chess_traj.py``)
2. ``dls_nullspace``      -- Task-Priority 6x5 Damped Least Squares + Null-Space SDF Repulsion
3. ``trajopt_spline``     -- Coupled IK + Analytical Capsule-SDF Cubic B-Spline Optimizer
4. ``rrt_connect``        -- Coupled IK + Joint-Space Bi-Directional RRT-Connect + Shortcutting
5. ``hybrid_trajopt_rrt`` -- Production Hybrid (``trajopt_spline`` fast path + ``rrt_connect`` fallback)

Categories tested:
- Near-to-near strict ``0.0 deg`` vertical moves over crowded pawn ranks
- Near-to-far **preemptive angled pickup** moves with tall center/rank blockers
- Far-to-near **matched angled drop** moves
- Far-to-far reach-ceiling moves detouring around tall blockers
- Two-leg capture sequences (``captured -> off-board graveyard``, ``attacker -> target``)
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data

LOWLEVEL_DIR = Path(__file__).resolve().parents[1]
if str(LOWLEVEL_DIR) not in sys.path:
    sys.path.insert(0, str(LOWLEVEL_DIR))

from planner import (  # noqa: E402
    BOARD_FILES,
    CaptureSequenceResult,
    DEFAULT_HOME_DEG,
    DLSNullspacePlanner,
    MoveRequest,
    PieceCylinder,
    PlannedTrajectory,
    SIM_BOARD_CONFIG,
    SO101Kinematics,
    SceneCollisionChecker,
    build_pieces_from_square_map,
    plan_move_or_capture,
    resolve_vhacd_urdf,
)
from planner_benchmarks.alternatives import (  # noqa: E402
    HybridPlanner,
    RRTConnectPlanner,
    TrajOptSplinePlanner,
)

PYBULLET_CONTROL_JOINTS = (0, 1, 2, 3, 4, 6)
SIM_GRIPPER_CLOSED_DEG = -6.5
SIM_GRIPPER_OPEN_DEG = 26.0


@dataclass(frozen=True)
class BenchmarkScenario:
    """One reproducible crowded-board chess move scenario."""

    scenario_id: str
    category: str
    description: str
    board_squares: Dict[str, str]
    from_square: str
    to_square: str
    moving_symbol: str
    is_capture: bool = False
    captured_symbol: str = "p"
    graveyard_slot: int = 0


@dataclass
class ScenarioRolloutMetrics:
    """Complete kinematic + 240 Hz PyBullet physics rollout metrics for one planner on one move."""

    scenario_id: str
    category: str
    planner: str
    is_capture: bool
    planning_time_ms: float
    fallback_used: bool
    pick_tilt_deg: float
    place_tilt_deg: float
    predicted_release_tilt_deg: float
    min_sdf_transit_clearance_mm: float
    joint_path_length_deg: float
    mean_squared_jerk: float
    pickup_success: bool
    premature_drop: bool
    collateral_max_disp_mm: float
    collateral_knockdowns: int
    collateral_collisions: int
    final_xy_error_mm: float
    final_piece_tilt_deg: float
    captured_cleared_ok: bool
    overall_success: bool
    notes: str = ""


def standard_opening_board() -> Dict[str, str]:
    board = {
        **{f"{f}1": s for f, s in zip(BOARD_FILES, "RNBQKBNR")},
        **{f"{f}2": "P" for f in BOARD_FILES},
        **{f"{f}7": "p" for f in BOARD_FILES},
        **{f"{f}8": s for f, s in zip(BOARD_FILES, "rnbqkbnr")},
    }
    return board


def midgame_crowded_board() -> Dict[str, str]:
    b = standard_opening_board()
    # Open classic Italian / Queen's Gambit style crowded lanes + tall blockers
    del b["e2"]
    b["e4"] = "P"
    del b["d7"]
    b["d5"] = "p"
    del b["g8"]
    b["f6"] = "n"
    del b["b8"]
    b["c6"] = "n"
    return b


def build_benchmark_suite() -> List[BenchmarkScenario]:
    """Build 16 standardized crowded-board scenarios spanning all 5 required move classes."""
    opening = standard_opening_board()
    midgame = midgame_crowded_board()

    # Board for d1->h5 over tall f6 Queen blocker while surrounded on rank 1
    b_d1_h5 = dict(midgame)
    b_d1_h5["f6"] = "q"  # 82 mm Queen directly near transit corridor

    # Board for e2->g6 preemptive angled pawn test with tall f4 King/Queen blocker
    b_e2_g6 = dict(opening)
    b_e2_g6["f4"] = "q"

    # Board for c1->h6 diagonal over tall e4/f5 pieces
    b_c1_h6 = dict(midgame)
    del b_c1_h6["d2"]
    b_c1_h6["d3"] = "P"
    b_c1_h6["f5"] = "q"

    # Board for extreme corner-to-corner a1->h8 with center tall pieces
    b_a1_h8 = {
        "a1": "R", "b1": "N", "e1": "K", "b2": "P", "c2": "P",
        "d4": "Q", "e5": "k", "f6": "b", "g7": "p", "f8": "r",
    }

    # Board for far-to-near h5->d1 return into crowded rank 1
    b_h5_d1 = dict(b_d1_h5)
    del b_h5_d1["d1"]
    b_h5_d1["h5"] = "Q"

    # Board for far-to-far f1->h6 around tall g4 Queen
    b_f1_h6 = dict(opening)
    del b_f1_h6["e2"]
    b_f1_h6["e4"] = "P"
    b_f1_h6["g4"] = "q"

    # Board for far-to-far g5->g8 detouring around tall g7 Queen
    b_g5_g8 = {
        "e1": "K", "d2": "P", "e2": "P",
        "g5": "R", "g7": "q", "f7": "p", "h7": "p", "e8": "k",
    }

    # Board for 2-leg capture c1 x h6 on crowded board
    b_cap_h6 = dict(midgame)
    del b_cap_h6["d2"]
    b_cap_h6["d3"] = "P"
    b_cap_h6["h6"] = "n"

    return [
        # 1. Near-to-near strict 0.0 deg vertical jumps
        BenchmarkScenario(
            "near_knight_b1_c3",
            "1_near_vertical",
            "32-piece opening: Knight b1->c3 jumping over crowded a2/b2/c2 pawns at strict 0 deg tilt",
            dict(opening),
            "b1",
            "c3",
            "N",
        ),
        BenchmarkScenario(
            "near_pawn_e2_e4",
            "1_near_vertical",
            "32-piece opening: King's pawn e2->e4 surrounded by full ranks 1 and 2",
            dict(opening),
            "e2",
            "e4",
            "P",
        ),
        BenchmarkScenario(
            "near_queen_d1_d4",
            "1_near_vertical",
            "Crowded rank-1 Queen d1->d4 jumping over d2 pawn between c1 Bishop and e1 King",
            dict(opening),
            "d1",
            "d4",
            "Q",
        ),
        BenchmarkScenario(
            "near_bishop_c1_e3",
            "1_near_vertical",
            "32-piece opening: Bishop c1->e3 over crowded d2 pawn",
            dict(opening),
            "c1",
            "e3",
            "B",
        ),
        # 2. Near-to-far preemptive angled pickup moves
        BenchmarkScenario(
            "preemptive_queen_d1_h5",
            "2_near_to_far_preemptive",
            "Queen d1->h5: preemptive 28.7 deg pickup on crowded rank 1, flying over tall f6 Queen",
            b_d1_h5,
            "d1",
            "h5",
            "Q",
        ),
        BenchmarkScenario(
            "preemptive_pawn_e2_g6",
            "2_near_to_far_preemptive",
            "Pawn e2->g6: preemptive angled pickup on crowded rank 2 over tall f4 Queen blocker",
            b_e2_g6,
            "e2",
            "g6",
            "P",
        ),
        BenchmarkScenario(
            "preemptive_bishop_c1_h6",
            "2_near_to_far_preemptive",
            "Bishop c1->h6: preemptive angled pickup on rank 1 over tall f5 Queen to far h6",
            b_c1_h6,
            "c1",
            "h6",
            "B",
        ),
        BenchmarkScenario(
            "extreme_corner_a1_h8",
            "2_near_to_far_preemptive",
            "Rook a1->h8: extreme corner-to-corner a->h reach-ceiling move around center Queen & King",
            b_a1_h8,
            "a1",
            "h8",
            "R",
        ),
        # 3. Far-to-near matched angled drop moves
        BenchmarkScenario(
            "matched_drop_h5_d1",
            "3_far_to_near_matched",
            "Queen h5->d1: angled far pickup on h5 with matched 28.7 deg drop into crowded rank 1",
            b_h5_d1,
            "h5",
            "d1",
            "Q",
        ),
        BenchmarkScenario(
            "matched_drop_g8_f6",
            "3_far_to_near_matched",
            "32-piece opening: Black Knight g8->f6 over f7/g7/h7 pawns",
            dict(opening),
            "g8",
            "f6",
            "n",
        ),
        BenchmarkScenario(
            "matched_drop_h8_d5",
            "3_far_to_near_matched",
            "Rook h8->d5: far corner h8 pickup over rank-7 pawns to center d5",
            {**opening, "d5": "P", **{k: v for k, v in opening.items() if k != "d5"}},
            "h8",
            "e5",
            "r",
        ),
        # 4. Far-to-far reach-ceiling moves around tall blockers
        BenchmarkScenario(
            "far_ceiling_g1_h3",
            "4_far_to_far_ceiling",
            "32-piece opening: Knight g1->h3 jumping over g2/h2 pawns along outer reach shell",
            dict(opening),
            "g1",
            "h3",
            "N",
        ),
        BenchmarkScenario(
            "far_ceiling_f1_h6",
            "4_far_to_far_ceiling",
            "Bishop f1->h6 detouring around tall g4 Queen near outer workspace ceiling",
            b_f1_h6,
            "f1",
            "h6",
            "B",
        ),
        BenchmarkScenario(
            "far_ceiling_g5_g8",
            "4_far_to_far_ceiling",
            "Rook g5->g8 jumping over tall 82 mm g7 Queen near outer spherical boundary",
            b_g5_g8,
            "g5",
            "g8",
            "R",
        ),
        # 5. Two-leg capture sequences
        BenchmarkScenario(
            "capture_center_e4xd5",
            "5_two_leg_capture",
            "Crowded midgame 2-leg capture e4xd5: clear d5 pawn to graveyard slot 0, then move e4->d5",
            dict(midgame),
            "e4",
            "d5",
            "P",
            is_capture=True,
            captured_symbol="p",
            graveyard_slot=0,
        ),
        BenchmarkScenario(
            "capture_far_c1xh6",
            "5_two_leg_capture",
            "Crowded 2-leg far capture c1xh6: clear far h6 Knight to graveyard slot 1, then preemptive c1->h6",
            b_cap_h6,
            "c1",
            "h6",
            "B",
            is_capture=True,
            captured_symbol="n",
            graveyard_slot=1,
        ),
    ]


def catmull_rom_expand(waypoints_deg: np.ndarray, steps_per_seg: int = 36) -> np.ndarray:
    """Reproduce ``multisim_chess_fast.py``'s Catmull-Rom interpolation for legacy baseline."""
    W = np.asarray(waypoints_deg, dtype=np.float64)
    n = W.shape[0]
    if n <= 1:
        return W.copy()
    out: List[np.ndarray] = []
    alphas = np.linspace(0.0, 1.0, steps_per_seg, endpoint=False, dtype=np.float64)
    a2 = alphas * alphas
    a3 = a2 * alphas
    for idx in range(1, n):
        p0 = W[max(idx - 2, 0)]
        p1 = W[idx - 1]
        p2 = W[idx]
        p3 = W[min(idx + 1, n - 1)]
        seg = 0.5 * (
            2.0 * p1[None, :]
            + (-p0 + p2)[None, :] * alphas[:, None]
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)[None, :] * a2[:, None]
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3)[None, :] * a3[:, None]
        )
        out.append(seg)
    out.append(W[-1:, :])
    return np.vstack(out)


def plan_legacy_chess_traj(
    scenario: BenchmarkScenario,
    kin: SO101Kinematics,
    checker: SceneCollisionChecker,
) -> Tuple[np.ndarray, int, int, float, float, float, float, float, float]:
    """Run existing ``chess_traj.pickupmove_traj`` + Catmull-Rom interpolation as legacy baseline."""
    import chess_traj

    t0 = time.perf_counter()
    board_origin = SIM_BOARD_CONFIG.origin_xyz_m
    grasp_off = np.array([-0.014, 0.002, -0.003], dtype=np.float64)
    place_off = np.array([-0.014, 0.002, -0.003], dtype=np.float64)

    def build_one_leg(f_sq, t_sq) -> Tuple[np.ndarray, int, int, np.ndarray, np.ndarray]:
        wp_list, close_wp_i = chess_traj.pickupmove_traj(
            f_sq, t_sq, board_origin, grasp_off, place_off
        )
        W = np.asarray(wp_list, dtype=np.float64)
        open_candidates = [i for i in range(close_wp_i + 1, len(W)) if W[i, 5] > 6.0]
        open_wp_i = open_candidates[0] if open_candidates else max(close_wp_i + 1, len(W) - 2)
        # Map closed gripper (<= 6 deg) to SIM_GRIPPER_CLOSED_DEG for identical V-HACD squeeze
        W[:, 5] = np.where(W[:, 5] <= 6.0, SIM_GRIPPER_CLOSED_DEG, SIM_GRIPPER_OPEN_DEG)
        steps_per_seg = 32
        cmds = catmull_rom_expand(W, steps_per_seg=steps_per_seg)
        q_pick_pose = W[max(0, close_wp_i - 1)].copy()
        q_place_pose = W[max(0, open_wp_i - 1)].copy()
        return cmds, close_wp_i * steps_per_seg, open_wp_i * steps_per_seg, q_pick_pose, q_place_pose

    if scenario.is_capture:
        slot_xyz = SIM_BOARD_CONFIG.graveyard_slot_world_xyz(scenario.graveyard_slot, z_above_table_m=0.0)
        cmds1, c1, o1, _, _ = build_one_leg(scenario.to_square, ( float(slot_xyz[0]), float(slot_xyz[1]) ))
        cmds2, c2, o2, qp, qd = build_one_leg(scenario.from_square, scenario.to_square)
        cmds = np.vstack([cmds1, cmds2])
        close_idx = c1
        open_idx = len(cmds1) + o2
    else:
        cmds, close_idx, open_idx, qp, qd = build_one_leg(scenario.from_square, scenario.to_square)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    T1, _ = kin.fk_tcp(qp)
    T2, _ = kin.fk_tcp(qd)
    from planner.kinematics_so101 import gripper_tilt_deg, piece_landing_tilt_deg

    pick_tilt = gripper_tilt_deg(T1[:3, :3])
    place_tilt = gripper_tilt_deg(T2[:3, :3])
    rel_tilt = piece_landing_tilt_deg(T1[:3, :3], T2[:3, :3])

    excl = (scenario.to_square, scenario.from_square) if scenario.is_capture else (scenario.from_square,)
    c_vals, _ = checker.evaluate_clearance_batch(cmds[::6], exclude_labels=excl, allow_tip_near_board=True)
    min_sdf_mm = float(np.min(c_vals) * 1000.0)

    diffs = np.diff(cmds[:, :5], axis=0)
    path_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    dt = 1.0 / 240.0
    jerk = np.diff(cmds[:, :5], n=3, axis=0) / (dt ** 3)
    ms_jerk = float(np.mean(np.sum(jerk * jerk, axis=1)))
    return cmds, close_idx, open_idx, elapsed_ms, pick_tilt, place_tilt, rel_tilt, min_sdf_mm, path_len, ms_jerk


def create_staunton_piece_body(pc: PieceCylinder, surface_z_m: float) -> int:
    """Spawn a 3-part Staunton hourglass rigid body (wide base + ~19.5 mm stem + collar/crown)."""
    h = pc.height_m
    base_h = 0.26 * h
    head_h = 0.18 * h
    stem_r = float(np.clip(pc.radius_m * 0.65, 0.0094, 0.0102))
    head_r = float(min(0.0114, pc.radius_m * 0.70))
    c_shape = p.createCollisionShapeArray(
        shapeTypes=[p.GEOM_CYLINDER, p.GEOM_CYLINDER, p.GEOM_CYLINDER],
        radii=[
            pc.radius_m * 0.88,
            stem_r,
            head_r,
        ],
        lengths=[base_h, h, head_h],
        collisionFramePositions=[
            [0.0, 0.0, -0.5 * h + 0.5 * base_h],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5 * h - 0.5 * head_h],
        ],
    )
    bid = p.createMultiBody(
        baseMass=0.040,
        baseCollisionShapeIndex=c_shape,
        basePosition=[pc.x_m, pc.y_m, surface_z_m + 0.5 * h],
    )
    p.changeDynamics(
        bid,
        -1,
        lateralFriction=1.25,
        spinningFriction=0.015,
        rollingFriction=0.003,
        linearDamping=0.04,
        angularDamping=0.04,
    )
    return bid


def piece_tilt_from_quat(orn: Sequence[float]) -> float:
    R = np.asarray(p.getMatrixFromQuaternion(orn), dtype=np.float64).reshape(3, 3)
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


class CrowdedPyBulletHarness:
    """Reusable 240 Hz headless PyBullet simulator with V-HACD SO-101 gripper & wrist camera."""

    def __init__(self) -> None:
        self.client_id = p.connect(p.DIRECT)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        self.urdf_path = str(resolve_vhacd_urdf())
        self.board = SIM_BOARD_CONFIG
        # Optional video capture (enabled via --record-dir): frames from the last rollout
        self.record_frames: bool = False
        self.record_stride: int = 8  # 240 Hz / 8 = 30 fps
        self.last_frames: List[np.ndarray] = []
        self._cam_view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.24, 0.0, 0.05],
            distance=0.82,
            yaw=-58.0,
            pitch=-32.0,
            roll=0.0,
            upAxisIndex=2,
            physicsClientId=self.client_id,
        )
        self._cam_proj = p.computeProjectionMatrixFOV(
            fov=48.0, aspect=640.0 / 480.0, nearVal=0.05, farVal=3.0, physicsClientId=self.client_id
        )

    def _grab_frame(self) -> None:
        _, _, rgba, _, _ = p.getCameraImage(
            640,
            480,
            viewMatrix=self._cam_view,
            projectionMatrix=self._cam_proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client_id,
        )
        self.last_frames.append(np.asarray(rgba, dtype=np.uint8).reshape(480, 640, 4)[:, :, :3].copy())

    def close(self) -> None:
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)

    def _reset_world_and_spawn(self, pieces: Sequence[PieceCylinder]) -> Tuple[int, Dict[str, int]]:
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client_id)

        plane_id = p.loadURDF("plane.urdf", [0.0, 0.0, 0.0], physicsClientId=self.client_id)
        p.changeDynamics(plane_id, -1, lateralFriction=1.0, spinningFriction=0.01, rollingFriction=0.005)

        robot_id = p.loadURDF(self.urdf_path, [0.0, 0.0, 0.0], useFixedBase=True, physicsClientId=self.client_id)
        # Silicone grip pads on both fixed jaw (link 4) and moving jaw (link 6)
        for jaw_link in (4, 6):
            p.changeDynamics(
                robot_id,
                jaw_link,
                lateralFriction=2.8,
                spinningFriction=0.80,
                rollingFriction=0.02,
                contactStiffness=15000.0,
                contactDamping=150.0,
            )

        b_half = [
            0.5 * self.board.board_span_m + self.board.border_margin_m,
            0.5 * self.board.board_span_m + self.board.border_margin_m,
            self.board.slab_half_height_m,
        ]
        b_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=b_half)
        board_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=b_col,
            basePosition=list(self.board.origin_xyz_m),
        )
        p.changeDynamics(board_id, -1, lateralFriction=1.0, spinningFriction=0.01, rollingFriction=0.005)

        body_ids: Dict[str, int] = {}
        for pc in pieces:
            body_ids[pc.label] = create_staunton_piece_body(pc, self.board.surface_z_m)
        return robot_id, body_ids

    def execute_commands(
        self,
        scenario: BenchmarkScenario,
        pieces: Sequence[PieceCylinder],
        commands_deg: np.ndarray,
        hold_windows: Sequence[Tuple[int, int]],
        *,
        planner_name: str,
        planning_time_ms: float,
        fallback_used: bool,
        pick_tilt_deg: float,
        place_tilt_deg: float,
        predicted_release_tilt_deg: float,
        min_sdf_transit_clearance_mm: float,
        joint_path_length_deg: float,
        mean_squared_jerk: float,
        notes: str = "",
    ) -> ScenarioRolloutMetrics:
        robot_id, body_ids = self._reset_world_and_spawn(pieces)
        self.last_frames = []

        q0_rad = np.deg2rad(commands_deg[0])
        for j_i, sim_j in enumerate(PYBULLET_CONTROL_JOINTS):
            p.resetJointState(robot_id, sim_j, float(q0_rad[j_i]), targetVelocity=0.0)
        for _ in range(25):
            p.stepSimulation()

        init_xy = {
            lbl: np.asarray(p.getBasePositionAndOrientation(bid)[0][:2], dtype=np.float64)
            for lbl, bid in body_ids.items()
        }
        piece_h_map = {pc.label: pc.height_m for pc in pieces}

        attacker_id = body_ids[scenario.from_square]
        attacker_rest_z = self.board.surface_z_m + 0.5 * piece_h_map[scenario.from_square]
        max_attacker_z = attacker_rest_z
        premature_drop = False

        # For 2-leg capture, the attacker moves during the last hold window; for 1-leg, window 0
        atk_close_idx, atk_open_idx = hold_windows[-1]

        for step_i, cmd_deg in enumerate(commands_deg):
            cmd_rad = np.deg2rad(cmd_deg)
            for j_i in range(5):
                p.setJointMotorControl2(
                    robot_id,
                    PYBULLET_CONTROL_JOINTS[j_i],
                    p.POSITION_CONTROL,
                    targetPosition=float(cmd_rad[j_i]),
                    force=60.0,
                    positionGain=0.45,
                    velocityGain=1.0,
                )
            holding_now = any(c_i <= step_i < o_i for (c_i, o_i) in hold_windows)
            grip_force = 8.5 if holding_now else 4.0
            p.setJointMotorControl2(
                robot_id,
                6,
                p.POSITION_CONTROL,
                targetPosition=float(cmd_rad[5]),
                force=grip_force,
                positionGain=0.30,
                velocityGain=0.9,
            )
            p.stepSimulation()
            if self.record_frames and step_i % self.record_stride == 0:
                self._grab_frame()

            if atk_close_idx <= step_i < atk_open_idx:
                atk_pos, _ = p.getBasePositionAndOrientation(attacker_id)
                max_attacker_z = max(max_attacker_z, float(atk_pos[2]))
                # Check premature drop during mid-transit carry (middle 50% of hold window)
                mid_start = atk_close_idx + int(0.30 * (atk_open_idx - atk_close_idx))
                mid_end = atk_close_idx + int(0.75 * (atk_open_idx - atk_close_idx))
                if mid_start <= step_i <= mid_end and max_attacker_z > attacker_rest_z + 0.030:
                    if float(atk_pos[2]) < attacker_rest_z + 0.012:
                        premature_drop = True

        # Let all pieces settle under gravity for 0.5 s (120 steps)
        for settle_i in range(120):
            p.stepSimulation()
            if self.record_frames and settle_i % self.record_stride == 0:
                self._grab_frame()

        pickup_success = (max_attacker_z - attacker_rest_z) >= 0.030

        # Evaluate attacker final pose on destination square
        atk_pos, atk_orn = p.getBasePositionAndOrientation(attacker_id)
        target_xy = self.board.square_center_world_xy(scenario.to_square)
        final_xy_err_mm = float(np.linalg.norm(np.asarray(atk_pos[:2]) - target_xy) * 1000.0)
        final_tilt_deg = piece_tilt_from_quat(atk_orn)

        # Evaluate captured piece clearance if capture scenario
        captured_ok = True
        excluded_bystanders = {scenario.from_square}
        if scenario.is_capture and scenario.to_square in body_ids:
            excluded_bystanders.add(scenario.to_square)
            cap_pos, _ = p.getBasePositionAndOrientation(body_ids[scenario.to_square])
            # Captured piece must be off the 8x8 playing grid!
            captured_ok = not self.board.is_over_board(cap_pos[:2], extra_margin_m=-self.board.border_margin_m)

        # Evaluate collateral bystander disturbance across ALL other standing pieces
        max_collat_mm = 0.0
        collat_collisions = 0
        collat_knockdowns = 0
        for lbl, bid in body_ids.items():
            if lbl in excluded_bystanders:
                continue
            b_pos, b_orn = p.getBasePositionAndOrientation(bid)
            disp_mm = float(np.linalg.norm(np.asarray(b_pos[:2]) - init_xy[lbl]) * 1000.0)
            b_tilt = piece_tilt_from_quat(b_orn)
            max_collat_mm = max(max_collat_mm, disp_mm)
            if disp_mm > 3.5 or b_tilt > 15.0:
                collat_collisions += 1
            if b_tilt > 45.0:
                collat_knockdowns += 1

        overall_ok = (
            pickup_success
            and (not premature_drop)
            and collat_collisions == 0
            and captured_ok
            and final_tilt_deg < 15.0
            and final_xy_err_mm < 22.0
        )

        return ScenarioRolloutMetrics(
            scenario_id=scenario.scenario_id,
            category=scenario.category,
            planner=planner_name,
            is_capture=scenario.is_capture,
            planning_time_ms=round(planning_time_ms, 2),
            fallback_used=fallback_used,
            pick_tilt_deg=round(pick_tilt_deg, 2),
            place_tilt_deg=round(place_tilt_deg, 2),
            predicted_release_tilt_deg=round(predicted_release_tilt_deg, 2),
            min_sdf_transit_clearance_mm=round(min_sdf_transit_clearance_mm, 2),
            joint_path_length_deg=round(joint_path_length_deg, 1),
            mean_squared_jerk=round(mean_squared_jerk, 1),
            pickup_success=pickup_success,
            premature_drop=premature_drop,
            collateral_max_disp_mm=round(max_collat_mm, 3),
            collateral_knockdowns=collat_knockdowns,
            collateral_collisions=collat_collisions,
            final_xy_error_mm=round(final_xy_err_mm, 2),
            final_piece_tilt_deg=round(final_tilt_deg, 2),
            captured_cleared_ok=captured_ok,
            overall_success=overall_ok,
            notes=notes,
        )


def write_mp4(path: Path, frames: Sequence[np.ndarray], fps: int = 30) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7, macro_block_size=1) as w:
        for fr in frames:
            w.append_data(fr)


def annotate_frame(frame: np.ndarray, title: str, subtitle: str, ok: bool) -> np.ndarray:
    import cv2

    out = np.ascontiguousarray(frame.copy())
    cv2.rectangle(out, (0, 0), (out.shape[1], 46), (24, 24, 24), -1)
    cv2.putText(out, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    col = (80, 220, 120) if ok else (70, 70, 240)
    cv2.putText(out, subtitle, (10, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return out


def write_montage(path: Path, clips: Sequence[Tuple[str, str, bool, List[np.ndarray]]], cols: int = 3) -> None:
    """Tile per-planner rollouts into one labeled grid video (shorter clips freeze on last frame)."""
    n_frames = max(len(c[3]) for c in clips)
    h, w = clips[0][3][0].shape[:2]
    rows = (len(clips) + cols - 1) // cols
    out_frames: List[np.ndarray] = []
    for t in range(n_frames):
        canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for k, (title, sub, ok, frs) in enumerate(clips):
            fr = annotate_frame(frs[min(t, len(frs) - 1)], title, sub, ok)
            r, c = divmod(k, cols)
            canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = fr
        out_frames.append(canvas)
    write_mp4(path, out_frames)


def format_markdown_report(results: Sequence[ScenarioRolloutMetrics], scenarios: Sequence[BenchmarkScenario]) -> str:
    planner_order = [
        "legacy_chess_traj",
        "dls_nullspace",
        "trajopt_spline",
        "rrt_connect",
        "hybrid_trajopt_rrt",
    ]
    lines: List[str] = []
    lines.append("# SO-101 Crowded-Board Trajectory Planning Benchmark Report\n")
    lines.append(
        f"- **Physics Engine**: PyBullet `DIRECT` @ 240 Hz with 24-hull V-HACD gripper + wrist camera (`so101_with_mount_vhacd_gripper.urdf`)\n"
        f"- **Scenarios Tested**: {len(scenarios)} crowded-board scenarios (32-piece Staunton heights `45–92 mm`, tall blockers, near<->far moves, 2-leg captures)\n"
    )

    lines.append("## 1. Aggregate Planner Comparison\n")
    lines.append(
        "| Planner | Overall Success | Collateral-Free | Upright Landing (<15°) | Median Plan Time (ms) | Mean XY Error on Landed (mm) | Mean Settled Tilt on Landed (°) | Mean Max Collateral Disp (mm) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for pl_name in planner_order:
        rows = [r for r in results if r.planner == pl_name]
        if not rows:
            continue
        n = len(rows)
        n_succ = sum(1 for r in rows if r.overall_success)
        n_colfree = sum(1 for r in rows if r.collateral_collisions == 0)
        n_upright = sum(
            1 for r in rows if r.pickup_success and r.final_xy_error_mm < 35.0 and r.final_piece_tilt_deg < 15.0
        )
        med_ms = float(np.median([r.planning_time_ms for r in rows]))
        landed = [
            r for r in rows if r.pickup_success and r.final_xy_error_mm < 35.0 and r.final_piece_tilt_deg < 25.0
        ]
        mean_xy = float(np.mean([r.final_xy_error_mm for r in landed])) if landed else float("nan")
        mean_tilt = float(np.mean([r.final_piece_tilt_deg for r in landed])) if landed else float("nan")
        mean_collat = float(np.mean([r.collateral_max_disp_mm for r in rows]))
        lines.append(
            f"| `{pl_name}` | **{n_succ}/{n} ({100.0*n_succ/n:.0f}%)** | {n_colfree}/{n} | {n_upright}/{n} | "
            f"{med_ms:.1f} ms | {mean_xy:.2f} mm | {mean_tilt:.2f}° | {mean_collat:.2f} mm |"
        )

    lines.append("\n## 2. Per-Scenario Breakdown\n")
    lines.append(
        "| Scenario | Category | Planner | Success | Plan (ms) | Pick->Place Tilt (°) | Min SDF (mm) | Collateral Collisions | XY Err (mm) | Final Tilt (°) |"
    )
    lines.append("|---|---|---|:---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        ok_str = "PASS" if r.overall_success else "FAIL"
        lines.append(
            f"| `{r.scenario_id}` | `{r.category}` | `{r.planner}` | **{ok_str}** | {r.planning_time_ms:.1f} | "
            f"{r.pick_tilt_deg:.1f}->{r.place_tilt_deg:.1f} | {r.min_sdf_transit_clearance_mm:+.1f} | "
            f"{r.collateral_collisions} ({r.collateral_max_disp_mm:.1f} mm) | {r.final_xy_error_mm:.1f} | {r.final_piece_tilt_deg:.1f} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run crowded-board PyBullet benchmark across SO-101 planners.")
    parser.add_argument(
        "--planners",
        nargs="+",
        default=["legacy_chess_traj", "dls_nullspace", "trajopt_spline", "rrt_connect", "hybrid_trajopt_rrt"],
        help="Subset of planners to evaluate.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Optional subset of scenario_ids to run.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for JSON and Markdown benchmark reports.",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help="If set, render each rollout to MP4 (per planner) plus a per-scenario side-by-side montage.",
    )
    args = parser.parse_args()

    kin = SO101Kinematics()
    board = SIM_BOARD_CONFIG
    planner_objs = {
        "dls_nullspace": DLSNullspacePlanner(kin, board),
        "trajopt_spline": TrajOptSplinePlanner(kin, board),
        "rrt_connect": RRTConnectPlanner(kin, board, seed=7),
        "hybrid_trajopt_rrt": HybridPlanner(kin, board, seed=7),
    }

    suite = build_benchmark_suite()
    if args.scenarios:
        wanted = set(args.scenarios)
        suite = [s for s in suite if s.scenario_id in wanted]

    harness = CrowdedPyBulletHarness()
    harness.record_frames = args.record_dir is not None
    all_metrics: List[ScenarioRolloutMetrics] = []

    print(f"Running {len(suite)} crowded-board scenarios across {len(args.planners)} planners...", flush=True)
    try:
        for sc in suite:
            pieces = build_pieces_from_square_map(sc.board_squares, board, use_staunton_heights=True)
            checker = SceneCollisionChecker(kin, board, pieces)
            print(f"\n=== [{sc.category}] {sc.scenario_id}: {sc.description} ===", flush=True)
            scenario_clips: List[Tuple[str, str, bool, List[np.ndarray]]] = []

            for pl_name in args.planners:
                if pl_name == "legacy_chess_traj":
                    (
                        cmds,
                        c_idx,
                        o_idx,
                        plan_ms,
                        p_tilt,
                        d_tilt,
                        rel_tilt,
                        min_sdf,
                        plen,
                        ms_j,
                    ) = plan_legacy_chess_traj(sc, kin, checker)
                    # For legacy 2-leg capture, construct both hold windows
                    if sc.is_capture:
                        half = cmds.shape[0] // 2
                        windows = [(c_idx, half - 96), (half + 96, o_idx)]
                    else:
                        windows = [(c_idx, o_idx)]
                    m = harness.execute_commands(
                        sc,
                        pieces,
                        cmds,
                        windows,
                        planner_name=pl_name,
                        planning_time_ms=plan_ms,
                        fallback_used=False,
                        pick_tilt_deg=p_tilt,
                        place_tilt_deg=d_tilt,
                        predicted_release_tilt_deg=rel_tilt,
                        min_sdf_transit_clearance_mm=min_sdf,
                        joint_path_length_deg=plen,
                        mean_squared_jerk=ms_j,
                    )
                else:
                    pl = planner_objs[pl_name]
                    req = MoveRequest(
                        from_target=sc.from_square,
                        to_target=sc.to_square,
                        moving_piece_symbol=sc.moving_symbol,
                        is_capture=sc.is_capture,
                        captured_piece_symbol=sc.captured_symbol,
                        graveyard_slot_index=sc.graveyard_slot,
                        gripper_closed_deg=SIM_GRIPPER_CLOSED_DEG,
                        gripper_open_deg=SIM_GRIPPER_OPEN_DEG,
                    )
                    res = plan_move_or_capture(pl, req, checker)
                    if isinstance(res, CaptureSequenceResult):
                        leg1 = res.leg1_clear_captured
                        leg2 = res.leg2_move_attacker
                        offset2 = leg1.commands_deg.shape[0]
                        windows = [
                            (leg1.close_gripper_cmd_idx, leg1.open_gripper_cmd_idx),
                            (offset2 + leg2.close_gripper_cmd_idx, offset2 + leg2.open_gripper_cmd_idx),
                        ]
                        m = harness.execute_commands(
                            sc,
                            pieces,
                            res.combined_commands_deg,
                            windows,
                            planner_name=pl_name,
                            planning_time_ms=res.total_planning_time_ms,
                            fallback_used=(leg1.fallback_used or leg2.fallback_used),
                            pick_tilt_deg=leg2.coupled_ik.pick_tilt_deg,
                            place_tilt_deg=leg2.coupled_ik.place_tilt_deg,
                            predicted_release_tilt_deg=leg2.coupled_ik.piece_release_tilt_deg,
                            min_sdf_transit_clearance_mm=min(
                                leg1.min_transit_clearance_mm, leg2.min_transit_clearance_mm
                            ),
                            joint_path_length_deg=leg1.joint_path_length_deg + leg2.joint_path_length_deg,
                            mean_squared_jerk=0.5 * (leg1.mean_squared_jerk + leg2.mean_squared_jerk),
                            notes=",".join(leg1.notes + leg2.notes),
                        )
                    else:
                        assert isinstance(res, PlannedTrajectory)
                        windows = [(res.close_gripper_cmd_idx, res.open_gripper_cmd_idx)]
                        m = harness.execute_commands(
                            sc,
                            pieces,
                            res.commands_deg,
                            windows,
                            planner_name=pl_name,
                            planning_time_ms=res.planning_time_ms,
                            fallback_used=res.fallback_used,
                            pick_tilt_deg=res.coupled_ik.pick_tilt_deg,
                            place_tilt_deg=res.coupled_ik.place_tilt_deg,
                            predicted_release_tilt_deg=res.coupled_ik.piece_release_tilt_deg,
                            min_sdf_transit_clearance_mm=res.min_transit_clearance_mm,
                            joint_path_length_deg=res.joint_path_length_deg,
                            mean_squared_jerk=res.mean_squared_jerk,
                            notes=",".join(res.notes),
                        )

                all_metrics.append(m)
                if args.record_dir is not None and harness.last_frames:
                    write_mp4(args.record_dir / sc.scenario_id / f"{pl_name}.mp4", harness.last_frames)
                    scenario_clips.append(
                        (
                            f"{sc.scenario_id} | {pl_name}",
                            f"{'PASS' if m.overall_success else 'FAIL'}  xy_err={m.final_xy_error_mm:.1f}mm  "
                            f"tilt={m.final_piece_tilt_deg:.1f}deg  collat={m.collateral_collisions}",
                            m.overall_success,
                            list(harness.last_frames),
                        )
                    )
                status = "PASS" if m.overall_success else "FAIL"
                print(
                    f"  {pl_name:20s} | {status} | plan={m.planning_time_ms:6.1f} ms | "
                    f"tilt={m.pick_tilt_deg:4.1f}->{m.place_tilt_deg:4.1f}° | "
                    f"sdf={m.min_sdf_transit_clearance_mm:+5.1f} mm | "
                    f"collat={m.collateral_collisions} ({m.collateral_max_disp_mm:4.1f} mm) | "
                    f"xy_err={m.final_xy_error_mm:5.1f} mm | final_tilt={m.final_piece_tilt_deg:5.1f}°",
                    flush=True,
                )
            if args.record_dir is not None and scenario_clips:
                write_montage(args.record_dir / f"{sc.scenario_id}_comparison.mp4", scenario_clips)
    finally:
        harness.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "benchmark_results.json"
    md_path = args.out_dir / "benchmark_report.md"

    json_path.write_text(
        json.dumps([asdict(m) for m in all_metrics], indent=2),
        encoding="utf-8",
    )
    report_md = format_markdown_report(all_metrics, suite)
    md_path.write_text(report_md, encoding="utf-8")

    print("\n" + "=" * 96)
    print(report_md)
    print(f"Saved JSON metrics to: {json_path}")
    print(f"Saved Markdown report to: {md_path}")


if __name__ == "__main__":
    main()
