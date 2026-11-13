#!/usr/bin/env python3
"""Turn a wanted URDF pose into the motor command that makes the real SO101 stop there.

The real arm misses a commanded angle for two reasons: its calibration offsets (URDF
angle = motor angle + offset) and where gravity and gearbox friction leave the joint
at rest. fit_friction_sag_model.py fits both from the friction sweeps and predicts
the stopping error e = target - reading of a command. Inverted, per joint:

    q_command = (q_want - offset) + f(q_want)

  * offset: the fitted calibration offset (0 for pan, roll and the gripper).
  * f, joints carrying gravity (shoulder_lift, elbow_flex, wrist_flex): the M3
    stopping error. tau_hold comes from inverse dynamics at q_want, which is where
    the joint will be at rest, so f is known directly and needs no iteration. The
    lifting / lowering case comes from the direction the command approaches from.
  * f, joints without gravity load (shoulder_pan, wrist_roll): the fitted band,
    c + h * direction.
  * The gripper is left alone.

Only the pose the arm comes to rest at is compensated; the model is static, so it says
nothing about transit waypoints. A joint that barely moves (under MIN_MOVE_DEG) into the
pose gets the offset only, since its approach direction is unknown.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# lowlevel/ on the path, for paths.py and the friction package (as kinematicscalibration/ does).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paths

DEFAULT_FIT_PATH = paths.REAL_WORLD_RUNS_DIR / "friction" / "friction_sag_fit.json"

# Same order as MOTOR_KEYS in the real-arm scripts.
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GRAVITY_JOINTS = ["shoulder_lift", "elbow_flex", "wrist_flex"]
NO_GRAVITY_JOINTS = ["shoulder_pan", "wrist_roll"]

# A move into the pose smaller than this (deg) has no trustworthy direction; same
# threshold the fit uses to drop readings.
MIN_MOVE_DEG = 0.3


@dataclass(frozen=True)
class JointCompensation:
    joint: str
    want_deg: float      # URDF angle the joint should stop at
    offset_deg: float
    f_deg: float         # predicted target - reading of the command
    command_deg: float
    move_dir: float      # +1 / -1 toward +/- theta, 0 if unknown
    note: str            # "lifting", "lowering", "band", "offset only", "gripper"


class JointCompensator:
    def __init__(self, fit_path: Path = DEFAULT_FIT_PATH) -> None:
        # Imported here so scripts that don't compensate don't pay for pybullet + mujoco.
        from friction.fit_friction_sag_model import GravityModel, JointParams

        self.fit_path = Path(fit_path)
        fit = json.loads(self.fit_path.read_text(encoding="utf-8"))
        if fit.get("error_sign") != "target - reading":
            raise ValueError(f"{self.fit_path}: expected error_sign 'target - reading'")
        params = fit["parameters"]
        self.offsets_deg = np.array([float(fit["offsets_deg"][name]) for name in JOINT_NAMES])
        self.joint_params = {
            name: JointParams(
                kp=float(params[f"{name}.kp"]["value"]),
                kc=float(params[f"{name}.K_c"]["value"]),
                kl=float(params[f"{name}.K_l"]["value"]),
            )
            for name in GRAVITY_JOINTS
        }
        self.bands = {
            name: (float(band["constant_deg"]), float(band["half_width_deg"]))
            for name, band in fit["no_gravity_bands"].items()
        }
        self.gravity = GravityModel()

    def compensate(
        self,
        want_deg: np.ndarray,
        previous_command_deg: np.ndarray | None = None,
        *,
        move_dirs: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[JointCompensation]]:
        """Motor command (6,) that should come to rest at URDF pose want_deg (6,).

        previous_command_deg is the command the arm moves from, which sets each joint's
        approach direction. Or pass move_dirs (+1 / -1 / 0 per joint) directly.
        """
        want = np.asarray(want_deg, dtype=float)
        motor_want = want - self.offsets_deg
        if move_dirs is None:
            if previous_command_deg is None:
                raise ValueError("compensate needs previous_command_deg or move_dirs")
            move_dirs = approach_dirs(motor_want, previous_command_deg)
        command = motor_want.copy()
        details = []
        for i, name in enumerate(JOINT_NAMES):
            f_deg, note = self._stopping_error_deg(i, motor_want, move_dirs[i])
            command[i] = motor_want[i] + f_deg
            details.append(
                JointCompensation(
                    name, float(want[i]), float(self.offsets_deg[i]), f_deg, float(command[i]), float(move_dirs[i]), note
                )
            )
        return command, details

    def predict_rest_motor_deg(
        self,
        command_deg: np.ndarray,
        previous_command_deg: np.ndarray | None = None,
        *,
        move_dirs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Where the model says command_deg (6,) comes to rest, in motor degrees.

        The forward direction: reading = command - f(reading). f depends on the resting
        pose, which isn't known up front, so this iterates; f changes little over a few
        degrees, so it settles in a couple of passes. Directions as in compensate().
        """
        command = np.asarray(command_deg, dtype=float)
        if move_dirs is None:
            if previous_command_deg is None:
                raise ValueError("predict_rest_motor_deg needs previous_command_deg or move_dirs")
            move_dirs = approach_dirs(command, previous_command_deg)
        rest = command.copy()
        for _ in range(5):
            f = np.array([self._stopping_error_deg(i, rest, move_dirs[i])[0] for i in range(len(JOINT_NAMES))])
            rest = command - f
        return rest

    def _stopping_error_deg(self, i: int, motor_pose_deg: np.ndarray, move_dir: float) -> tuple[float, str]:
        """Predicted target - reading (deg) of joint i at rest at motor_pose_deg, and which case applied."""
        from friction.fit_friction_sag_model import predicted_up_error

        name = JOINT_NAMES[i]
        if name == "gripper":
            return 0.0, "gripper"
        if move_dir == 0.0:
            return 0.0, "offset only"
        if name in GRAVITY_JOINTS:
            tau_hold = self.gravity.holding_torque(name, motor_pose_deg, self.offsets_deg)
            s = 1.0 if tau_hold >= 0 else -1.0
            lifting = move_dir == s
            e_up, _ = predicted_up_error(abs(tau_hold), lifting, self.joint_params[name])
            return float(np.rad2deg(s * e_up)), "lifting" if lifting else "lowering"
        c, h = self.bands[name]
        return c + h * move_dir, "band"

    def resting_urdf_deg(self, observed_deg: np.ndarray) -> np.ndarray:
        """URDF angles of a reading (motor degrees), using the fitted offsets."""
        return np.asarray(observed_deg, dtype=float) + self.offsets_deg


def approach_dirs(target_deg: np.ndarray, previous_deg: np.ndarray) -> np.ndarray:
    """Per joint +1 / -1 for a move toward +/- theta, 0 for one under MIN_MOVE_DEG."""
    delta = np.asarray(target_deg, dtype=float) - np.asarray(previous_deg, dtype=float)
    return np.where(np.abs(delta) >= MIN_MOVE_DEG, np.sign(delta), 0.0)
