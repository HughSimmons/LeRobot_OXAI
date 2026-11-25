"""Fast vectorized SO-101 forward kinematics, analytical Jacobians, and Coupled Pick-Place IK.

Implements:
1. Exact URDF forward kinematics (single & batch ``(N, 6)``) matching Pinocchio / PyBullet
   to machine precision (~1e-13 m), including collision frame transforms for all arm links,
   fixed jaw, moving jaw (as a function of gripper angle ``q[5]``), and wrist camera mount.
2. Exact analytical geometric Jacobian for TCP/pinch position and local vertical attitude
   ``v(q) = R(q)^T z_world``.
3. Coupled Pick-Place Attitude-Matched IK (``solve_coupled_pick_place_ik``):
   - Enforces strict ``0.0 deg`` vertical gripper on reachable near squares (``a-d``, ``e``).
   - Enforces minimum-feasible outward tilt on far squares (``f-h``).
   - Enforces ``R(q_pick)^T z_world == R(q_place)^T z_world`` so pieces picked up at an angle
     on far squares are dropped at the exact same angle, and pieces picked up on near squares
     destined for far squares are preemptively grasped at (or as close as reachable to) the
     destination's required drop angle.
   - Evaluates candidate jaw yaw/roll angles against board obstacles to avoid hitting
     neighboring standing pieces with the open jaws or wrist camera.
4. Locked-attitude Damped Least Squares (DLS) Cartesian stroke generator for collision-free
   vertical lowering and lifting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from .compensation import resolve_vhacd_urdf

ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
ALL_JOINT_NAMES = (*ARM_JOINT_NAMES, "gripper")

DEFAULT_HOME_DEG = np.array(
    [
        96.92307692307692,
        -107.86813186813187,
        97.36263736263736,
        65.18681318681318,
        -29.846153846153847,
        4.62962962962963,
    ],
    dtype=np.float64,
)

GRIPPER_OPEN_DEG = 25.0
GRIPPER_CLOSED_DEG = 5.0

# Effective hardware joint limits (deg): URDF bounds widened slightly on lift/elbow to admit DEFAULT_HOME,
# matching upstream/real_world2 probe_xy_pose.py while keeping wrist_roll cable-safe.
EFFECTIVE_ARM_LIMITS_DEG = np.array(
    [
        [-110.00, 110.00],   # shoulder_pan
        [-109.87, 100.00],   # shoulder_lift
        [-96.83, 99.36],     # elbow_flex
        [-95.00, 95.00],     # wrist_flex
        [-155.00, 160.00],   # wrist_roll (cable-safe within URDF [-157.2, +162.8])
    ],
    dtype=np.float64,
)

# Midpoint between V-HACD fixed jaw inner face (x=0) and open/closed moving jaw channel
# in gripper_frame_link coordinates (+Z of gripper_frame_link points downward out of jaw tips).
DEFAULT_PINCH_OFFSET_TCP_M = np.array([-0.0155, 0.0005, -0.0030], dtype=np.float64)

WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
WORLD_DOWN = np.array([0.0, 0.0, -1.0], dtype=np.float64)


def _rpy_to_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def _make_transform(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rpy_to_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def _batch_rz(angles_rad: np.ndarray) -> np.ndarray:
    """Build (N, 4, 4) Z-axis rotation transforms from (N,) angles in radians."""
    n = angles_rad.shape[0]
    c = np.cos(angles_rad)
    s = np.sin(angles_rad)
    T = np.zeros((n, 4, 4), dtype=np.float64)
    T[:, 0, 0] = c
    T[:, 0, 1] = -s
    T[:, 1, 0] = s
    T[:, 1, 1] = c
    T[:, 2, 2] = 1.0
    T[:, 3, 3] = 1.0
    return T


def gripper_tilt_deg(R_tcp: np.ndarray) -> float:
    """Angle (deg) between TCP +Z axis and straight down [0, 0, -1]."""
    return float(np.degrees(np.arccos(np.clip(-float(R_tcp[2, 2]), -1.0, 1.0))))


def piece_landing_tilt_deg(R_pick: np.ndarray, R_place: np.ndarray) -> float:
    """Angle (deg) off world vertical of a piece grasped upright at ``R_pick`` and released at ``R_place``."""
    u_world = R_place @ (R_pick.T @ WORLD_UP)
    return float(np.degrees(np.arccos(np.clip(float(u_world[2]), -1.0, 1.0))))


@dataclass(frozen=True)
class CoupledIKSolution:
    q_pick_deg: np.ndarray          # (6,) with open gripper
    q_place_deg: np.ndarray         # (6,) with closed gripper
    pick_pos_err_mm: float
    place_pos_err_mm: float
    pick_tilt_deg: float
    place_tilt_deg: float
    piece_release_tilt_deg: float
    min_feasible_pick_tilt_deg: float
    min_feasible_place_tilt_deg: float
    clearance_m: float
    converged: bool


class SO101Kinematics:
    """Exact analytical FK, batch FK, and geometric Jacobians for SO-101."""

    def __init__(self, urdf_path: Path | None = None) -> None:
        self.urdf_path = resolve_vhacd_urdf() if urdf_path is None else Path(urdf_path)
        root = ET.parse(self.urdf_path).getroot()
        joints_xml = {j.attrib["name"]: j for j in root.findall("joint")}

        def parse_joint_T(jname: str, fallback_xyz: Sequence[float], fallback_rpy: Sequence[float]) -> np.ndarray:
            if jname not in joints_xml:
                return _make_transform(fallback_xyz, fallback_rpy)
            orig = joints_xml[jname].find("origin")
            xyz = [float(v) for v in orig.attrib.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in orig.attrib.get("rpy", "0 0 0").split()]
            return _make_transform(xyz, rpy)

        self.T_chain_fixed = tuple(
            parse_joint_T(name, [0, 0, 0], [0, 0, 0]) for name in ARM_JOINT_NAMES
        )
        self.T_gripper_to_tcp = parse_joint_T(
            "gripper_frame_joint", [-0.0079, -0.000218121, -0.0981274], [0.0, np.pi, 0.0]
        )
        self.T_gripper_to_jaw_joint = parse_joint_T(
            "gripper", [0.0202, 0.0188, -0.0234], [np.pi / 2.0, 0.0, 0.0]
        )
        self.T_gripper_to_cam_mount = parse_joint_T(
            "wrist_camera_mount_joint", [-0.0088, 0.024, -0.01315], [-np.pi / 2.0, 0.0, -np.pi / 2.0]
        )

        self.arm_limits_deg = EFFECTIVE_ARM_LIMITS_DEG.copy()
        self.arm_lo_deg = self.arm_limits_deg[:, 0].copy()
        self.arm_hi_deg = self.arm_limits_deg[:, 1].copy()
        self.arm_lo_rad = np.deg2rad(self.arm_lo_deg)
        self.arm_hi_rad = np.deg2rad(self.arm_hi_deg)

    def clamp_arm_deg(self, q_deg: Sequence[float]) -> np.ndarray:
        arr = np.asarray(q_deg, dtype=np.float64).copy()
        arr[:5] = np.clip(arr[:5], self.arm_lo_deg, self.arm_hi_deg)
        return arr

    def within_arm_limits(self, q_deg: Sequence[float], tol_deg: float = 1e-3) -> bool:
        arr = np.asarray(q_deg, dtype=np.float64)[:5]
        return bool(np.all(arr >= self.arm_lo_deg - tol_deg) and np.all(arr <= self.arm_hi_deg + tol_deg))

    def fk_tcp(
        self,
        q_deg: Sequence[float],
        pinch_offset_tcp_m: np.ndarray = DEFAULT_PINCH_OFFSET_TCP_M,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(T_world_tcp, pinch_world_xyz)`` for a single joint configuration."""
        q_rad = np.deg2rad(np.asarray(q_deg, dtype=np.float64)[:5])
        T = np.eye(4, dtype=np.float64)
        for i in range(5):
            T = T @ self.T_chain_fixed[i]
            c, s = np.cos(q_rad[i]), np.sin(q_rad[i])
            R = T[:3, :3].copy()
            T[:3, 0] = c * R[:, 0] + s * R[:, 1]
            T[:3, 1] = -s * R[:, 0] + c * R[:, 1]
        T_tcp = T @ self.T_gripper_to_tcp
        pinch = T_tcp[:3, 3] + T_tcp[:3, :3] @ pinch_offset_tcp_m
        return T_tcp, pinch

    def fk_all_links_batch(self, Q_deg: np.ndarray) -> dict[str, np.ndarray]:
        """Vectorized batch FK for ``Q_deg`` of shape ``(N, 6)`` or ``(6,)``.

        Returns ``(N, 4, 4)`` world transforms for:
        ``shoulder``, ``upper_arm``, ``lower_arm``, ``wrist``, ``gripper``,
        ``tcp``, ``moving_jaw``, and ``wrist_cam``.
        """
        Q = np.asarray(Q_deg, dtype=np.float64)
        if Q.ndim == 1:
            Q = Q[None, :]
        n = Q.shape[0]
        Q_rad = np.deg2rad(Q[:, :6])

        link_keys = ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper")
        out: dict[str, np.ndarray] = {}
        T = np.broadcast_to(np.eye(4, dtype=np.float64), (n, 4, 4)).copy()
        for i, key in enumerate(link_keys):
            T = T @ self.T_chain_fixed[i]
            T = T @ _batch_rz(Q_rad[:, i])
            out[key] = T

        T_grip = out["gripper"]
        out["tcp"] = T_grip @ self.T_gripper_to_tcp
        out["wrist_cam"] = T_grip @ self.T_gripper_to_cam_mount
        T_jaw_pre = T_grip @ self.T_gripper_to_jaw_joint
        out["moving_jaw"] = T_jaw_pre @ _batch_rz(Q_rad[:, 5])
        return out

    def fk_jacobian(
        self,
        q5_rad: np.ndarray,
        pinch_offset_tcp_m: np.ndarray = DEFAULT_PINCH_OFFSET_TCP_M,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Analytical FK + Jacobians wrt 5 arm joints in radians.

        Returns:
            p_pinch: (3,) world grasp point (m)
            R_tcp:   (3, 3) world rotation of gripper_frame_link
            J_pos:   (3, 5) dp_pinch / dq_rad (m/rad)
            J_vert:  (3, 5) d(R_tcp^T z_world) / dq_rad (1/rad)
            J_omega: (3, 5) spatial angular velocity axes z_j
        """
        T = np.eye(4, dtype=np.float64)
        origins = np.empty((5, 3), dtype=np.float64)
        axes = np.empty((5, 3), dtype=np.float64)

        for i in range(5):
            T = T @ self.T_chain_fixed[i]
            origins[i] = T[:3, 3]
            axes[i] = T[:3, 2]
            c, s = np.cos(q5_rad[i]), np.sin(q5_rad[i])
            R = T[:3, :3].copy()
            T[:3, 0] = c * R[:, 0] + s * R[:, 1]
            T[:3, 1] = -s * R[:, 0] + c * R[:, 1]

        T_tcp = T @ self.T_gripper_to_tcp
        R_tcp = T_tcp[:3, :3]
        p_pinch = T_tcp[:3, 3] + R_tcp @ pinch_offset_tcp_m

        J_pos = np.cross(axes, p_pinch[None, :] - origins).T  # (3, 5)
        J_omega = axes.T                                       # (3, 5)
        # d(R^T z_w)/dq_j = R^T (z_w x z_j)
        zw_cross_z = np.cross(WORLD_UP[None, :], axes).T       # (3, 5)
        J_vert = R_tcp.T @ zw_cross_z                          # (3, 5)
        return p_pinch, R_tcp, J_pos, J_vert, J_omega

    def _lm_solve_single(
        self,
        target_xyz: np.ndarray,
        seed_deg: Sequence[float],
        *,
        pinch_offset_tcp_m: np.ndarray = DEFAULT_PINCH_OFFSET_TCP_M,
        target_local_vertical: np.ndarray | None = None,
        pos_weight: float = 100.0,
        vert_weight: float = 2.0,
        min_tilt_weight: float = 0.25,
        preferred_roll_deg: float | None = None,
        roll_weight: float = 0.0,
        preferred_q5_deg: np.ndarray | None = None,
        posture_weight: float = 0.005,
        max_iters: int = 35,
    ) -> tuple[np.ndarray, float, float]:
        """Fast bounded Levenberg-Marquardt IK for 5 arm joints using analytical Jacobians."""
        q = np.clip(np.deg2rad(np.asarray(seed_deg, dtype=np.float64)[:5]), self.arm_lo_rad, self.arm_hi_rad)
        q_pref = q.copy() if preferred_q5_deg is None else np.deg2rad(np.asarray(preferred_q5_deg, dtype=np.float64)[:5])
        roll_pref_rad = 0.0 if preferred_roll_deg is None else float(np.deg2rad(preferred_roll_deg))
        damping = 1e-3

        def eval_res_jac(q_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            p_pinch, R_tcp, J_pos, J_vert, _ = self.fk_jacobian(q_vec, pinch_offset_tcp_m)
            res_parts = [pos_weight * (p_pinch - target_xyz)]
            jac_parts = [pos_weight * J_pos]

            if target_local_vertical is not None:
                v_cur = R_tcp.T @ WORLD_UP
                res_parts.append(vert_weight * (v_cur - target_local_vertical))
                jac_parts.append(vert_weight * J_vert)
            elif min_tilt_weight > 0.0:
                # Penalize horizontal components of (R_tcp^T z_w)[:2] (magnitude = sin(tilt),
                # non-vanishing unit gradient at tilt=0) plus (R_tcp[2,2] + 1) so tilt -> 0.00 deg!
                v_cur = R_tcp.T @ WORLD_UP
                res_parts.append(min_tilt_weight * np.array([v_cur[0], v_cur[1], R_tcp[2, 2] + 1.0], dtype=np.float64))
                jac_parts.append(min_tilt_weight * J_vert)

            if preferred_roll_deg is not None and roll_weight > 0.0:
                res_parts.append(np.array([roll_weight * (q_vec[4] - roll_pref_rad)], dtype=np.float64))
                j_roll = np.zeros((1, 5), dtype=np.float64)
                j_roll[0, 4] = roll_weight
                jac_parts.append(j_roll)

            if posture_weight > 0.0:
                res_parts.append(posture_weight * (q_vec - q_pref))
                jac_parts.append(posture_weight * np.eye(5, dtype=np.float64))

            return np.concatenate(res_parts), np.vstack(jac_parts)

        r, J = eval_res_jac(q)
        cost = 0.5 * float(np.dot(r, r))

        for _ in range(max_iters):
            H = J.T @ J + damping * np.eye(5, dtype=np.float64)
            g = J.T @ r
            if float(np.max(np.abs(g))) < 1e-7:
                break
            try:
                dq = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break

            q_cand = np.clip(q + dq, self.arm_lo_rad, self.arm_hi_rad)
            if np.max(np.abs(q_cand - q)) < 1e-7:
                # Projected gradient fallback along active box bounds
                q_cand = np.clip(q - 0.05 * g / (np.diag(H) + 1e-6), self.arm_lo_rad, self.arm_hi_rad)
                if np.max(np.abs(q_cand - q)) < 1e-8:
                    break

            r_cand, J_cand = eval_res_jac(q_cand)
            cost_cand = 0.5 * float(np.dot(r_cand, r_cand))
            if cost_cand < cost:
                q, r, J, cost = q_cand, r_cand, J_cand, cost_cand
                damping = max(damping * 0.4, 1e-6)
                if cost < 1e-10:
                    break
            else:
                damping = min(damping * 4.0, 1e2)

        q_deg = np.rad2deg(q)
        T_tcp, pinch = self.fk_tcp(q_deg, pinch_offset_tcp_m)
        pos_err_mm = float(np.linalg.norm(pinch - target_xyz) * 1000.0)
        tilt = gripper_tilt_deg(T_tcp[:3, :3])
        return q_deg, pos_err_mm, tilt

    def canonical_arm_seeds_for_xyz(self, target_xyz: Sequence[float], roll_deg: float = 0.0) -> list[np.ndarray]:
        """Diverse elbow-up / extended / folded seeds aimed at ``target_xyz``'s azimuth."""
        dx = float(target_xyz[0]) - 0.0388
        dy = float(target_xyz[1])
        pan_guess = float(np.clip(-np.degrees(np.arctan2(dy, max(dx, 0.03))), -105.0, 105.0))
        return [
            np.array([pan_guess, -55.0, 80.0, 65.0, roll_deg], dtype=np.float64),
            np.array([pan_guess, -15.0, 45.0, 60.0, roll_deg], dtype=np.float64),
            np.array([pan_guess, 35.0, -35.0, 85.0, roll_deg], dtype=np.float64),
            np.array([pan_guess, 68.0, -68.0, 55.0, roll_deg], dtype=np.float64),
            np.array([pan_guess, 82.0, -25.0, 15.0, roll_deg], dtype=np.float64),
            np.array([pan_guess, -85.0, 95.0, 45.0, roll_deg], dtype=np.float64),
        ]

    def solve_min_tilt_ik(
        self,
        target_xyz: Sequence[float],
        *,
        pinch_offset_tcp_m: np.ndarray = DEFAULT_PINCH_OFFSET_TCP_M,
        preferred_roll_deg: float = 0.0,
        roll_weight: float = 0.02,
        gripper_deg: float = GRIPPER_OPEN_DEG,
        extra_seeds_deg: Sequence[Sequence[float]] = (),
    ) -> tuple[np.ndarray, float, float]:
        """Solve 6-DOF joints reaching ``target_xyz`` with minimum physically feasible tilt."""
        target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        seeds = list(extra_seeds_deg) + self.canonical_arm_seeds_for_xyz(target, preferred_roll_deg)

        best: tuple[float, np.ndarray, float, float] | None = None
        for seed in seeds:
            q5, pos_err_mm, tilt = self._lm_solve_single(
                target,
                seed,
                pinch_offset_tcp_m=pinch_offset_tcp_m,
                min_tilt_weight=0.18,
                preferred_roll_deg=preferred_roll_deg,
                roll_weight=roll_weight,
            )
            # Strict position priority (< 0.5 mm), then minimum tilt
            penalty_pos = pos_err_mm * (500.0 if pos_err_mm > 0.5 else 25.0)
            score = penalty_pos + tilt + 0.002 * abs(q5[4] - preferred_roll_deg)
            if best is None or score < best[0]:
                best = (score, q5, pos_err_mm, tilt)

        assert best is not None
        _, q5_best, err_mm, tilt_deg = best
        return np.append(q5_best, float(gripper_deg)), err_mm, tilt_deg

    def _lm_solve_coupled_pair(
        self,
        pick_xyz: np.ndarray,
        place_xyz: np.ndarray,
        seed_pick_deg: np.ndarray,
        seed_place_deg: np.ndarray,
        *,
        pinch_offset_tcp_m: np.ndarray,
        roll_pick_pref_deg: float,
        roll_place_pref_deg: float,
        roll_weight: float = 0.03,
        match_weight: float = 18.0,
        vert_weight: float = 0.22,
        max_iters: int = 40,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Jointly optimize ``(q_pick, q_place)`` in R^10 with analytical block Jacobians."""
        lo10 = np.concatenate([self.arm_lo_rad, self.arm_lo_rad])
        hi10 = np.concatenate([self.arm_hi_rad, self.arm_hi_rad])
        x = np.clip(
            np.deg2rad(np.concatenate([seed_pick_deg[:5], seed_place_deg[:5]])),
            lo10,
            hi10,
        )
        x_pref = x.copy()
        r1_pref = float(np.deg2rad(roll_pick_pref_deg))
        r2_pref = float(np.deg2rad(roll_place_pref_deg))
        pos_w = 120.0
        posture_w = 0.004
        damping = 1e-3

        def eval_coupled(vec10: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            q1 = vec10[:5]
            q2 = vec10[5:]
            p1, R1, Jp1, Jv1, _ = self.fk_jacobian(q1, pinch_offset_tcp_m)
            p2, R2, Jp2, Jv2, _ = self.fk_jacobian(q2, pinch_offset_tcp_m)
            v1 = R1.T @ WORLD_UP
            v2 = R2.T @ WORLD_UP

            # Residuals (25,): 3 pick_pos + 3 place_pos + 3 attitude_match + 6 verticality + 2 roll + 8 posture
            r = np.concatenate(
                [
                    pos_w * (p1 - pick_xyz),
                    pos_w * (p2 - place_xyz),
                    match_weight * (v1 - v2),
                    vert_weight * np.array([v1[0], v1[1], R1[2, 2] + 1.0, v2[0], v2[1], R2[2, 2] + 1.0], dtype=np.float64),
                    np.array([roll_weight * (q1[4] - r1_pref), roll_weight * (q2[4] - r2_pref)], dtype=np.float64),
                    posture_w * (vec10 - x_pref),
                ]
            )

            J = np.zeros((27, 10), dtype=np.float64)
            J[0:3, 0:5] = pos_w * Jp1
            J[3:6, 5:10] = pos_w * Jp2
            J[6:9, 0:5] = match_weight * Jv1
            J[6:9, 5:10] = -match_weight * Jv2
            J[9:12, 0:5] = vert_weight * Jv1
            J[12:15, 5:10] = vert_weight * Jv2
            J[15, 4] = roll_weight
            J[16, 9] = roll_weight
            J[17:27, :] = posture_w * np.eye(10, dtype=np.float64)
            return r, J

        r, J = eval_coupled(x)
        cost = 0.5 * float(np.dot(r, r))
        I10 = np.eye(10, dtype=np.float64)

        for _ in range(max_iters):
            H = J.T @ J + damping * I10
            g = J.T @ r
            if float(np.max(np.abs(g))) < 1e-7:
                break
            try:
                dx = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break

            x_cand = np.clip(x + dx, lo10, hi10)
            if np.max(np.abs(x_cand - x)) < 1e-7:
                x_cand = np.clip(x - 0.05 * g / (np.diag(H) + 1e-6), lo10, hi10)
                if np.max(np.abs(x_cand - x)) < 1e-8:
                    break

            r_cand, J_cand = eval_coupled(x_cand)
            cost_cand = 0.5 * float(np.dot(r_cand, r_cand))
            if cost_cand < cost:
                x, r, J, cost = x_cand, r_cand, J_cand, cost_cand
                damping = max(damping * 0.4, 1e-6)
                if cost < 1e-10:
                    break
            else:
                damping = min(damping * 4.0, 1e2)

        return np.rad2deg(x[:5]), np.rad2deg(x[5:])

    def solve_coupled_pick_place_ik(
        self,
        pick_xyz: Sequence[float],
        place_xyz: Sequence[float],
        *,
        pinch_offset_tcp_m: np.ndarray = DEFAULT_PINCH_OFFSET_TCP_M,
        preferred_pick_yaw_deg: float | None = None,
        roll_candidates_deg: Sequence[float] = (0.0, -45.0, 45.0, -90.0, 90.0),
        clearance_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
        gripper_open_deg: float = GRIPPER_OPEN_DEG,
        gripper_closed_deg: float = GRIPPER_CLOSED_DEG,
    ) -> CoupledIKSolution:
        """Solve coupled ``(q_pick, q_place)`` matching grasp-to-release piece orientation.

        Guarantees:
        - When both ``pick_xyz`` and ``place_xyz`` are within the vertical reach zone (``a-e``),
          both gripper poses converge to ``0.0 deg`` vertical downward.
        - When ``place_xyz`` is on a far square requiring tilt ``theta_place > 0``, ``q_pick``
          preemptively tilts to match ``R(q_pick)^T z_w == R(q_place)^T z_w`` so the piece
          lands upright (``piece_release_tilt_deg ~ 0 deg``).
        - When ``pick_xyz`` is on a far square requiring tilt ``theta_pick > 0``, ``q_place``
          drops the piece at the exact same gripper angle.
        - If an extreme near-to-far move (e.g. ``a4 -> h4``) saturates ``shoulder_lift`` before
          reaching ``theta_place``, TCP position accuracy is preserved (< 1 mm) while ``q_pick``
          tilts as close as physically reachable to ``theta_place``.
        """
        p_pick = np.asarray(pick_xyz, dtype=np.float64).reshape(3)
        p_place = np.asarray(place_xyz, dtype=np.float64).reshape(3)

        q_pick_ind, _, min_pick_tilt = self.solve_min_tilt_ik(p_pick, pinch_offset_tcp_m=pinch_offset_tcp_m)
        q_place_ind, _, min_place_tilt = self.solve_min_tilt_ik(p_place, pinch_offset_tcp_m=pinch_offset_tcp_m)

        # Build candidate wrist_roll values (including coord_finder yaw preference if supplied)
        rolls: list[float] = []
        if preferred_pick_yaw_deg is not None:
            for base_r in (preferred_pick_yaw_deg, preferred_pick_yaw_deg + 180.0, preferred_pick_yaw_deg - 180.0):
                if self.arm_lo_deg[4] <= base_r <= self.arm_hi_deg[4]:
                    rolls.append(float(base_r))
        for r_deg in roll_candidates_deg:
            if self.arm_lo_deg[4] <= r_deg <= self.arm_hi_deg[4] and all(abs(r_deg - ex) > 5.0 for ex in rolls):
                rolls.append(float(r_deg))

        best_sol: CoupledIKSolution | None = None
        best_score = np.inf

        # Seed bank: individual min-tilt solutions + folded/extended seeds so extreme a<->h pairs find global optimum
        pick_seeds_base = [
            q_pick_ind[:5],
            self.canonical_arm_seeds_for_xyz(p_pick)[0],
            self.canonical_arm_seeds_for_xyz(p_pick)[3],
            self.canonical_arm_seeds_for_xyz(p_pick)[5],
        ]
        place_seeds_base = [
            q_place_ind[:5],
            self.canonical_arm_seeds_for_xyz(p_place)[0],
            self.canonical_arm_seeds_for_xyz(p_place)[3],
            self.canonical_arm_seeds_for_xyz(p_place)[5],
        ]

        # Try the warm-started min-tilt pair first; only try backup folded/extended seeds
        # if the primary seed saturates a joint limit (> 1.0 mm error).
        for roll_deg in rolls:
            primary_pair = (pick_seeds_base[0].copy(), place_seeds_base[0].copy())
            backup_pairs = [
                (pick_seeds_base[3].copy(), place_seeds_base[2].copy()),
                (pick_seeds_base[2].copy(), place_seeds_base[3].copy()),
            ]
            for pair_idx, (sp1, sp2) in enumerate([primary_pair, *backup_pairs]):
                sp1[4] = roll_deg
                sp2[4] = roll_deg
                q1_5, q2_5 = self._lm_solve_coupled_pair(
                    p_pick,
                    p_place,
                    sp1,
                    sp2,
                    pinch_offset_tcp_m=pinch_offset_tcp_m,
                    roll_pick_pref_deg=roll_deg,
                    roll_place_pref_deg=roll_deg,
                    match_weight=22.0,
                    max_iters=28,
                )
                T1, pinch1 = self.fk_tcp(q1_5, pinch_offset_tcp_m)
                T2, pinch2 = self.fk_tcp(q2_5, pinch_offset_tcp_m)
                e1_mm = float(np.linalg.norm(pinch1 - p_pick) * 1000.0)
                e2_mm = float(np.linalg.norm(pinch2 - p_place) * 1000.0)

                if max(e1_mm, e2_mm) > 1.2 and pair_idx == 2:
                    # Near-square shoulder_lift saturated while trying to match extreme far tilt:
                    # relax match_weight so both endpoints hit < 0.5 mm while maximizing preemptive tilt!
                    q1_5, q2_5 = self._lm_solve_coupled_pair(
                        p_pick,
                        p_place,
                        q1_5,
                        q2_5,
                        pinch_offset_tcp_m=pinch_offset_tcp_m,
                        roll_pick_pref_deg=roll_deg,
                        roll_place_pref_deg=roll_deg,
                        match_weight=1.8,
                        max_iters=28,
                    )
                    T1, pinch1 = self.fk_tcp(q1_5, pinch_offset_tcp_m)
                    T2, pinch2 = self.fk_tcp(q2_5, pinch_offset_tcp_m)
                    e1_mm = float(np.linalg.norm(pinch1 - p_pick) * 1000.0)
                    e2_mm = float(np.linalg.norm(pinch2 - p_place) * 1000.0)

                t1 = gripper_tilt_deg(T1[:3, :3])
                t2 = gripper_tilt_deg(T2[:3, :3])
                piece_tilt = piece_landing_tilt_deg(T1[:3, :3], T2[:3, :3])

                q_pick_6 = np.append(q1_5, float(gripper_open_deg))
                q_place_6 = np.append(q2_5, float(gripper_closed_deg))

                clearance_m = 0.05
                if clearance_fn is not None:
                    clearance_m = float(clearance_fn(q_pick_6, q_place_6))

                # Composite objective:
                # 1. Position exactness (< 1 mm hard preference)
                # 2. Collision clearance (heavily penalize negative or < 4 mm jaw/camera clearance)
                # 3. Zero piece release tilt (R_pick^T z_w == R_place^T z_w)
                # 4. Minimum overall gripper tilt + minimal wrist_roll excursion
                pos_pen = (e1_mm + e2_mm) * (200.0 if max(e1_mm, e2_mm) > 1.0 else 20.0)
                col_pen = 0.0
                if clearance_m < 0.006:
                    col_pen = 5000.0 * (0.006 - clearance_m) ** 2 + (500.0 if clearance_m < 0.0 else 0.0)
                else:
                    col_pen = -2.0 * min(clearance_m, 0.025)

                yaw_pen = 0.01 * abs(q1_5[4] - (0.0 if preferred_pick_yaw_deg is None else preferred_pick_yaw_deg))
                roll_delta_pen = 0.005 * abs(q2_5[4] - q1_5[4])
                score = pos_pen + col_pen + 8.0 * piece_tilt + 0.35 * (t1 + t2) + yaw_pen + roll_delta_pen

                if score < best_score:
                    best_score = score
                    best_sol = CoupledIKSolution(
                        q_pick_deg=q_pick_6,
                        q_place_deg=q_place_6,
                        pick_pos_err_mm=e1_mm,
                        place_pos_err_mm=e2_mm,
                        pick_tilt_deg=t1,
                        place_tilt_deg=t2,
                        piece_release_tilt_deg=piece_tilt,
                        min_feasible_pick_tilt_deg=min_pick_tilt,
                        min_feasible_place_tilt_deg=min_place_tilt,
                        clearance_m=clearance_m,
                        converged=(max(e1_mm, e2_mm) <= 2.0),
                    )

        assert best_sol is not None
        return best_sol

    def solve_locked_attitude_cartesian_path(
        self,
        q_anchor_deg: Sequence[float],
        delta_xyz_world_m: Sequence[float],
        *,
        n_waypoints: int = 10,
        gripper_deg: float | None = None,
        pinch_offset_tcp_m: np.ndarray = DEFAULT_PINCH_OFFSET_TCP_M,
        damping: float = 0.025,
    ) -> tuple[np.ndarray, float, float]:
        """Generate a straight-line Cartesian stroke from ``q_anchor_deg`` to ``p_anchor + delta_xyz``.

        Uses 6x5 Damped Least Squares locking the gripper's local vertical vector ``R^T z_w``
        and ``wrist_roll`` as closely as reachable along the stroke, while guaranteeing smooth
        monotone Cartesian progression (used for vertical lift from pick and vertical descent to place).

        Returns ``(waypoints_deg (n_waypoints+1, 6), max_pos_err_mm, max_attitude_drift_deg)``.
        """
        q_start = np.asarray(q_anchor_deg, dtype=np.float64).copy()
        grip = float(q_start[5] if gripper_deg is None else gripper_deg)
        T0, p0 = self.fk_tcp(q_start, pinch_offset_tcp_m)
        v_lock = T0[:3, :3].T @ WORLD_UP
        roll_lock_rad = float(np.deg2rad(q_start[4]))
        delta = np.asarray(delta_xyz_world_m, dtype=np.float64).reshape(3)

        waypoints = np.zeros((n_waypoints + 1, 6), dtype=np.float64)
        waypoints[0, :5] = q_start[:5]
        waypoints[0, 5] = grip

        q_rad = np.deg2rad(q_start[:5]).copy()
        max_pos_err_mm = 0.0
        max_att_deg = 0.0

        for step_idx in range(1, n_waypoints + 1):
            alpha = step_idx / n_waypoints
            p_des = p0 + alpha * delta
            q_prev = q_rad.copy()
            # Strong attitude lock in bottom 30 mm right at grasp/release, smoothly relaxing near
            # the outer spherical ceiling so wrist_flex absorbs reach without flipping the elbow!
            att_w = 5.5 * (1.0 - 0.78 * alpha)
            lm_damp = max(damping, 0.08)

            def stroke_res_jac(q_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                p_cur, R_cur, J_pos, J_vert, _ = self.fk_jacobian(q_vec, pinch_offset_tcp_m)
                v_cur = R_cur.T @ WORLD_UP
                res = np.concatenate(
                    [
                        120.0 * (p_cur - p_des),
                        att_w * (v_cur - v_lock),
                        np.array([0.5 * (q_vec[4] - roll_lock_rad)], dtype=np.float64),
                        0.35 * (q_vec - q_prev),
                    ]
                )
                J = np.zeros((12, 5), dtype=np.float64)
                J[0:3, :] = 120.0 * J_pos
                J[3:6, :] = att_w * J_vert
                J[6, 4] = 0.5
                J[7:12, :] = 0.35 * np.eye(5, dtype=np.float64)
                return res, J

            r_cur, J_cur = stroke_res_jac(q_rad)
            cost_cur = 0.5 * float(np.dot(r_cur, r_cur))

            for _ in range(12):
                H = J_cur.T @ J_cur + (lm_damp ** 2) * np.eye(5, dtype=np.float64)
                dq = -np.linalg.solve(H, J_cur.T @ r_cur)
                max_abs = float(np.max(np.abs(dq)))
                if max_abs > 0.12:
                    dq = dq * (0.12 / max_abs)
                q_cand = np.clip(q_rad + dq, self.arm_lo_rad, self.arm_hi_rad)
                if float(np.max(np.abs(q_cand - q_rad))) < 1e-6:
                    break
                r_cand, J_cand = stroke_res_jac(q_cand)
                cost_cand = 0.5 * float(np.dot(r_cand, r_cand))
                if cost_cand < cost_cur:
                    q_rad, r_cur, J_cur, cost_cur = q_cand, r_cand, J_cand, cost_cand
                    lm_damp = max(lm_damp * 0.6, 0.02)
                else:
                    lm_damp = min(lm_damp * 3.0, 5.0)

            q_step_deg = np.rad2deg(q_rad)
            T_s, p_s = self.fk_tcp(q_step_deg, pinch_offset_tcp_m)
            max_pos_err_mm = max(max_pos_err_mm, float(np.linalg.norm(p_s - p_des) * 1000.0))
            att_cos = float(np.clip(np.dot(T_s[:3, :3].T @ WORLD_UP, v_lock), -1.0, 1.0))
            max_att_deg = max(max_att_deg, float(np.degrees(np.arccos(att_cos))))

            waypoints[step_idx, :5] = q_step_deg
            waypoints[step_idx, 5] = grip

        return waypoints, max_pos_err_mm, max_att_deg
