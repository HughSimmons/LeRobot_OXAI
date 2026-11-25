"""URDF asset resolution and M3 gravity-sag + static friction compensation.

Ports the realistic 24-hull V-HACD gripper URDF setup and the Duclusaud et al.
(ICRA 2025) M3 load-dependent friction + gravity-sag joint compensator from
``upstream/oliver/two-camera-capture``, using Pinocchio ``rnea`` for fast,
zero-overhead inverse-dynamics holding torques when Pinocchio is available
(with automatic PyBullet ``calculateInverseDynamics`` fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Sequence

import numpy as np

LOWLEVEL_DIR = Path(__file__).resolve().parents[1]
if str(LOWLEVEL_DIR) not in sys.path:
    sys.path.insert(0, str(LOWLEVEL_DIR))

from . import _paths_compat as pcompat  # noqa: E402

PLANNER_DIR = Path(__file__).resolve().parent
URDF_ASSETS_DIR = PLANNER_DIR / "assets" / "urdf"
VHACD_TEMPLATE_URDF = URDF_ASSETS_DIR / "so101_with_mount_vhacd_gripper.template.urdf"
VHACD_RESOLVED_URDF = URDF_ASSETS_DIR / "so101_with_mount_vhacd_gripper.urdf"

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
GRAVITY_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")
NO_GRAVITY_JOINTS = ("shoulder_pan", "wrist_roll")

# Calibrated hardware constants from upstream/oliver/two-camera-capture
DEFAULT_JOINT_OFFSETS_DEG = np.array([2.6, 2.0, -3.0, -4.6, 0.0, 0.0], dtype=np.float64)
DEFAULT_JOINT_FRICTION_DEG = np.array([0.29, 0.84, 0.76, 0.47, 0.29, 0.0], dtype=np.float64)
MIN_MOVE_DEG = 0.3


def resolve_vhacd_urdf() -> Path:
    """Materialize and return the V-HACD gripper + wrist-camera URDF with valid local mesh paths."""
    if not VHACD_TEMPLATE_URDF.is_file():
        raise FileNotFoundError(f"Missing V-HACD template URDF at {VHACD_TEMPLATE_URDF}")
    assets = pcompat.assets_dir()
    rel_assets = os.path.relpath(assets, URDF_ASSETS_DIR).replace(os.sep, "/")
    template_text = VHACD_TEMPLATE_URDF.read_text(encoding="utf-8")
    resolved_text = re.sub(
        r'filename="\.\./\.\./SO-ARM100/Simulation/SO101/assets/',
        f'filename="{rel_assets}/',
        template_text,
    )
    if not VHACD_RESOLVED_URDF.is_file() or VHACD_RESOLVED_URDF.read_text(encoding="utf-8") != resolved_text:
        VHACD_RESOLVED_URDF.write_text(resolved_text, encoding="utf-8")
    return VHACD_RESOLVED_URDF


def resolve_standard_urdf() -> Path:
    """Return the standard SO-101 URDF path."""
    return pcompat.standard_urdf()


@dataclass(frozen=True)
class M3JointParams:
    """Duclusaud M3 load-dependent servo friction + stiffness parameters."""

    kp: float  # N*m/rad
    kc: float  # N*m Coulomb friction at zero load
    kl: float  # dimensionless load-dependent friction gain in [0, 1)


@dataclass(frozen=True)
class JointCompensationDetail:
    joint: str
    want_urdf_deg: float
    offset_deg: float
    stopping_error_deg: float
    command_motor_deg: float
    move_dir: float
    mode: str


def predicted_m3_up_error_rad(tau_g: float, lifting: bool, params: M3JointParams) -> tuple[float, int]:
    """M3 stopping error along gravity-opposing direction (rad, positive = sagged)."""
    kp, kc, kl = params.kp, params.kc, params.kl
    if lifting:
        return (kc + (1.0 + kl) * tau_g) / ((1.0 - kl) * kp), 1
    if tau_g >= 0.5 * kc:
        return (-kc + (1.0 - kl) * tau_g) / ((1.0 + kl) * kp), 2
    return (-kc + (1.0 + kl) * tau_g) / ((1.0 - kl) * kp), 3


def approach_dirs(target_deg: Sequence[float], previous_deg: Sequence[float]) -> np.ndarray:
    delta = np.asarray(target_deg, dtype=np.float64) - np.asarray(previous_deg, dtype=np.float64)
    return np.where(np.abs(delta) >= MIN_MOVE_DEG, np.sign(delta), 0.0)


class PinocchioGravityModel:
    """Microsecond inverse-dynamics holding torques for SO-101 using Pinocchio (or PyBullet fallback)."""

    def __init__(self, urdf_path: Path | None = None) -> None:
        self.urdf_path = resolve_vhacd_urdf() if urdf_path is None else Path(urdf_path)
        self._use_pinocchio = False
        try:
            import pinocchio as pin

            self._pin = pin
            self.model = pin.buildModelFromUrdf(str(self.urdf_path))
            self.model.gravity.linear = np.array([0.0, 0.0, -9.81], dtype=np.float64)
            self.data = self.model.createData()
            self.q_idx = {}
            self.v_idx = {}
            for name in JOINT_NAMES:
                jid = self.model.getJointId(name)
                self.q_idx[name] = self.model.joints[jid].idx_q
                self.v_idx[name] = self.model.joints[jid].idx_v
            self._use_pinocchio = True
        except Exception:
            import pybullet as p

            self._p = p
            self._client = p.connect(p.DIRECT)
            p.setGravity(0.0, 0.0, -9.81, physicsClientId=self._client)
            self._robot = p.loadURDF(
                str(self.urdf_path),
                useFixedBase=True,
                flags=p.URDF_USE_INERTIA_FROM_FILE,
                physicsClientId=self._client,
            )
            infos = [
                p.getJointInfo(self._robot, j, physicsClientId=self._client)
                for j in range(p.getNumJoints(self._robot, physicsClientId=self._client))
            ]
            self._movable = [j for j, info in enumerate(infos) if info[2] != p.JOINT_FIXED]
            by_name = {info[1].decode(): j for j, info in enumerate(infos)}
            self._slot = {name: self._movable.index(by_name[name]) for name in JOINT_NAMES}

    def holding_torques_nm(self, urdf_angles_deg: Sequence[float]) -> dict[str, float]:
        """Static gravity holding torque (N*m) for each joint at ``urdf_angles_deg``."""
        angles_rad = np.deg2rad(np.asarray(urdf_angles_deg, dtype=np.float64)[:6])
        if self._use_pinocchio:
            q = self._pin.neutral(self.model)
            v = np.zeros(self.model.nv, dtype=np.float64)
            a = np.zeros(self.model.nv, dtype=np.float64)
            for i, name in enumerate(JOINT_NAMES):
                q[self.q_idx[name]] = angles_rad[i]
            tau = self._pin.rnea(self.model, self.data, q, v, a)
            return {name: float(tau[self.v_idx[name]]) for name in JOINT_NAMES}

        positions = [0.0] * len(self._movable)
        zeros = [0.0] * len(self._movable)
        for i, name in enumerate(JOINT_NAMES):
            positions[self._slot[name]] = float(angles_rad[i])
        torques = self._p.calculateInverseDynamics(
            self._robot, positions, zeros, zeros, physicsClientId=self._client
        )
        return {name: float(torques[self._slot[name]]) for name in JOINT_NAMES}


class JointCompensator:
    """Converts desired URDF resting poses into real STS3215 servo commands.

    Supports both:
    - Full Duclusaud M3 JSON fit (``friction_sag_fit.json`` from ``fit_friction_sag_model.py``)
    - Measured static offset + direction-band fallback (``DEFAULT_JOINT_OFFSETS_DEG`` and
      ``DEFAULT_JOINT_FRICTION_DEG`` from ``oliver/two-camera-capture``) when no fit JSON
      is present on disk.
    """

    def __init__(
        self,
        fit_path: Path | None = None,
        *,
        offsets_deg: Sequence[float] = DEFAULT_JOINT_OFFSETS_DEG,
        friction_band_deg: Sequence[float] = DEFAULT_JOINT_FRICTION_DEG,
        urdf_path: Path | None = None,
        use_default_fit: bool = False,
        use_oliver_model: bool = True,
    ) -> None:
        """``use_default_fit=True`` loads ``<real_world_runs>/friction/friction_sag_fit.json`` if present.

        With a fit file and ``use_oliver_model`` (default), Oliver's validated
        ``friction.joint_compensation.JointCompensator`` computes the stopping error whenever it is
        importable on this machine; otherwise the equivalent model below is used.
        """
        if fit_path is None and use_default_fit:
            fit_path = pcompat.default_friction_fit()
        self.fit_path = Path(fit_path) if fit_path is not None else None
        self.has_m3_fit = False
        self._oliver = None
        self.friction_band_deg = np.asarray(DEFAULT_JOINT_FRICTION_DEG, dtype=np.float64).copy()
        self.m3_params: dict[str, M3JointParams] = {}
        self.no_gravity_bands: dict[str, tuple[float, float]] = {}

        if self.fit_path is not None and self.fit_path.is_file():
            fit = json.loads(self.fit_path.read_text(encoding="utf-8"))
            if fit.get("error_sign") != "target - reading":
                raise ValueError(f"{self.fit_path}: expected error_sign 'target - reading'")
            params = fit["parameters"]
            self.offsets_deg = np.array(
                [float(fit["offsets_deg"][name]) for name in JOINT_NAMES], dtype=np.float64
            )
            self.m3_params = {
                name: M3JointParams(
                    kp=float(params[f"{name}.kp"]["value"]),
                    kc=float(params[f"{name}.K_c"]["value"]),
                    kl=float(params[f"{name}.K_l"]["value"]),
                )
                for name in GRAVITY_JOINTS
            }
            self.no_gravity_bands = {
                name: (float(band["constant_deg"]), float(band["half_width_deg"]))
                for name, band in fit["no_gravity_bands"].items()
            }
            self.has_m3_fit = True
            if use_oliver_model:
                self._oliver = self._try_load_oliver_model(self.fit_path)
        else:
            off = np.asarray(offsets_deg, dtype=np.float64).reshape(-1)
            fric = np.asarray(friction_band_deg, dtype=np.float64).reshape(-1)
            self.offsets_deg = off[:6] if off.size >= 6 else np.append(off[:5], 0.0)
            self.friction_band_deg = fric[:6] if fric.size >= 6 else np.append(fric[:5], 0.0)
            # Physically grounded STS3215 M3 defaults matching kp ~ 17.5 N*m/rad (P=16) and measured bands
            self.m3_params = {
                "shoulder_lift": M3JointParams(kp=18.0, kc=0.22, kl=0.22),
                "elbow_flex": M3JointParams(kp=18.0, kc=0.20, kl=0.20),
                "wrist_flex": M3JointParams(kp=18.0, kc=0.12, kl=0.15),
            }
            self.no_gravity_bands = {
                "shoulder_pan": (0.0, float(self.friction_band_deg[0])),
                "wrist_roll": (0.0, float(self.friction_band_deg[4])),
            }
        # Gravity torques are only needed by the built-in model (Oliver's carries its own).
        self.gravity = None if self._oliver is not None else PinocchioGravityModel(urdf_path=urdf_path)
        self.model_source = "oliver_friction" if self._oliver is not None else (
            "m3_fit" if self.has_m3_fit else "offset_band_fallback"
        )

    @staticmethod
    def _try_load_oliver_model(fit_path: Path):
        """Oliver's ``friction.joint_compensation.JointCompensator`` for ``fit_path``, or ``None``."""
        try:
            from friction.joint_compensation import JointCompensator as OliverJointCompensator

            return OliverJointCompensator(fit_path)
        except Exception as exc:  # missing package, mujoco/pybullet import, schema mismatch...
            print(f"[compensation] Oliver friction model unavailable ({exc!r}); using built-in M3 model.")
            return None

    def resting_urdf_deg(self, observed_motor_deg: Sequence[float]) -> np.ndarray:
        obs = np.asarray(observed_motor_deg, dtype=np.float64).reshape(6)
        return obs + self.offsets_deg

    def compensate_pose(
        self,
        want_urdf_deg: Sequence[float],
        previous_urdf_deg: Sequence[float] | None = None,
        *,
        move_dirs: Sequence[float] | None = None,
        use_gravity_sag: bool | None = None,
    ) -> tuple[np.ndarray, list[JointCompensationDetail]]:
        """Compute 6-DOF motor command (deg) that brings the real arm to rest at ``want_urdf_deg``."""
        want = np.asarray(want_urdf_deg, dtype=np.float64).reshape(6)
        motor_want = want - self.offsets_deg
        if move_dirs is None:
            if previous_urdf_deg is None:
                dirs = np.zeros(6, dtype=np.float64)
            else:
                prev_motor = np.asarray(previous_urdf_deg, dtype=np.float64).reshape(6) - self.offsets_deg
                dirs = approach_dirs(motor_want, prev_motor)
        else:
            dirs = np.asarray(move_dirs, dtype=np.float64).reshape(6)

        apply_m3 = self.has_m3_fit if use_gravity_sag is None else bool(use_gravity_sag)
        if self._oliver is not None and apply_m3:
            command, olv_details = self._oliver.compensate(want, move_dirs=dirs)
            return np.asarray(command, dtype=np.float64), [
                JointCompensationDetail(
                    joint=d.joint,
                    want_urdf_deg=d.want_deg,
                    offset_deg=d.offset_deg,
                    stopping_error_deg=d.f_deg,
                    command_motor_deg=d.command_deg,
                    move_dir=d.move_dir,
                    mode=f"oliver_{d.note.replace(' ', '_')}",
                )
                for d in olv_details
            ]
        if apply_m3 and self.gravity is None:
            self.gravity = PinocchioGravityModel()
        torques = self.gravity.holding_torques_nm(want) if apply_m3 else {}

        cmd = motor_want.copy()
        details: list[JointCompensationDetail] = []
        for i, name in enumerate(JOINT_NAMES):
            d = float(dirs[i])
            if name == "gripper":
                f_deg, mode = 0.0, "gripper"
            elif d == 0.0:
                f_deg, mode = 0.0, "offset_only"
            elif apply_m3 and name in GRAVITY_JOINTS:
                tau_hold = torques[name]
                s = 1.0 if tau_hold >= 0.0 else -1.0
                lifting = d == s
                e_up, _ = predicted_m3_up_error_rad(abs(tau_hold), lifting, self.m3_params[name])
                f_deg = float(np.rad2deg(s * e_up))
                mode = "m3_lifting" if lifting else "m3_lowering"
            elif name in self.no_gravity_bands:
                c_deg, h_deg = self.no_gravity_bands[name]
                f_deg = c_deg + h_deg * d
                mode = "band"
            else:
                f_deg = float(self.friction_band_deg[i]) * d
                mode = "static_band"

            cmd[i] = motor_want[i] + f_deg
            details.append(
                JointCompensationDetail(
                    joint=name,
                    want_urdf_deg=float(want[i]),
                    offset_deg=float(self.offsets_deg[i]),
                    stopping_error_deg=f_deg,
                    command_motor_deg=float(cmd[i]),
                    move_dir=d,
                    mode=mode,
                )
            )
        return cmd, details

    def compensate_trajectory_rest_waypoints(
        self,
        commands_urdf_deg: np.ndarray,
        rest_indices: Sequence[int],
        *,
        blend_window_steps: int = 12,
        use_gravity_sag: bool = False,
        offset_only: bool = False,
    ) -> np.ndarray:
        """Apply calibration offset throughout and blend stopping-error compensation into rest poses.

        As documented in ``oliver/two-camera-capture``, static friction/sag models predict where
        a joint stops at rest (grasp, release, home), not dynamic mid-swing tracking. Thus:
        - Calibration offsets (``motor = urdf - offset``) apply across the entire trajectory.
        - Direction-aware friction/sag overshoot ramps smoothly over ``blend_window_steps`` into
          each designated rest index (e.g., ``pick``, ``place``, ``end``) and decays smoothly out.
        - ``offset_only=True`` is Oliver's BASELINE mode: calibration offsets only, no friction/sag.
        """
        traj = np.asarray(commands_urdf_deg, dtype=np.float64).copy()
        n = traj.shape[0]
        if n == 0:
            return traj

        motor_traj = traj.copy()
        motor_traj[:, :5] -= self.offsets_deg[:5]
        if offset_only:
            return motor_traj

        extra_f = np.zeros_like(motor_traj)
        for idx in sorted({int(i) for i in rest_indices if 0 <= int(i) < n}):
            lookback = max(0, idx - max(blend_window_steps, 4))
            # Find last non-trivial approach direction per joint leading into rest pose
            prev = traj[lookback]
            _, details = self.compensate_pose(
                traj[idx], prev, use_gravity_sag=use_gravity_sag
            )
            f_vec = np.array([d.stopping_error_deg for d in details], dtype=np.float64)
            # Smooth cosine ramp into idx and out of idx (preserving dwell plateaus)
            dwell_end = idx
            while dwell_end + 1 < n and np.max(np.abs(traj[dwell_end + 1, :5] - traj[idx, :5])) < 1e-3:
                dwell_end += 1

            w_in = min(blend_window_steps, idx)
            if w_in > 0:
                for k in range(w_in):
                    alpha = 0.5 * (1.0 - np.cos(np.pi * (k + 1) / w_in))
                    extra_f[idx - w_in + k, :5] = alpha * f_vec[:5]
            extra_f[idx : dwell_end + 1, :5] = f_vec[:5]

            w_out = min(blend_window_steps, n - 1 - dwell_end)
            if w_out > 0:
                for k in range(w_out):
                    alpha = 0.5 * (1.0 + np.cos(np.pi * (k + 1) / w_out))
                    extra_f[dwell_end + 1 + k, :5] = alpha * f_vec[:5]

        return motor_traj + extra_f
