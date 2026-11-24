#!/usr/bin/env python3
"""Fit a static friction-and-sag model to the real-arm friction sweeps by least squares.

Where does a servo joint stop when commanded to an angle? It is pulled toward the
target like a spring (stiffness kp), held back by gravity and by gearbox friction,
and it sticks at the edge of the friction band. This script fits that model,
the load-dependent friction model M3 of Duclusaud et al. (ICRA 2025) with the
joint at rest, to the readings recorded by toggle_joint_calibration.py in
real_world_runs/friction/. Nothing talks to the arm.

Model, per reading:
  * "up" is the direction that opposes gravity. tau_hold, from inverse dynamics at
    the recorded pose, is the torque the motor needs to hold it; s = sign(tau_hold)
    is +1 if up is +theta and -1 if up is -theta, and the load is tau_g = |tau_hold|.
  * A move is "lifting" if it went up and "lowering" otherwise.
  * Friction at rest is K_c + K_l * |tau_m - tau_e| (M3), solved together with the
    torque balance. With e_up = s * (target - reading), positive when sagged:

        lifting:                    e_up = ( K_c + (1 + K_l) tau_g) / ((1 - K_l) kp)
        lowering, tau_g >= K_c/2:   e_up = (-K_c + (1 - K_l) tau_g) / ((1 + K_l) kp)
        lowering, tau_g <  K_c/2:   e_up = (-K_c + (1 + K_l) tau_g) / ((1 - K_l) kp)

    The last case is a lowering move where friction ends up larger than twice the
    gravity torque, so the motor pulls with gravity. The case is chosen from the
    parameters being tried, so the optimiser moves readings between cases as it goes;
    both lowering formulas give -K_c / (2 kp) at the switch, so the curve is continuous.

What is fitted:
  * kp, K_c and K_l for each joint that carries gravity (shoulder_lift, elbow_flex,
    wrist_flex). Without gravity load the three only appear as K_c / ((1 - K_l) kp),
    so for shoulder_pan and wrist_roll only the band half-width in degrees is reported.
  * The calibration offsets of those three joints (URDF angle = motor angle + offset),
    since they change tau_hold. They're shared by all sweeps, and each is pulled
    toward the value in multisim_chess_fast.py by a prior with that file's stated
    uncertainty (0.7 deg for elbow and wrist flex; the lift was never measured, so it
    gets a loose 5 deg). A fitted offset also absorbs any error in the URDF's mass
    model, so compare it with the board calibration rather than replacing it.
    --fix-offsets keeps them at the sim's values.

Scoring: every model is refitted without each angle of each joint (both approach
directions together) and scored on predicting it, next to the current fixed-shift
model (c + h * direction) and a shift with a linear sag term (c + h * direction +
tau_hold / k). The first reading of each sweep is not used, because the direction
it was approached from isn't recorded. Nor is a reading where the joint moved less
than MIN_MOVE_DEG or against the command, since its direction can't be trusted; these
are printed.

--train-dirs adds more sweep folders to the fit (e.g. the coverage sweeps from
run_friction_coverage_sweeps.py, one joint per folder); --test-dirs scores the fit
on folders it never saw, next to the two baselines fitted on the same training data.

Errors use target - reading, positive when the joint stopped below its target in
theta (fit_servo_model_to_sweeps.py uses the opposite sign).

Limitation: it assumes every reading sits exactly at the edge of the friction band.
A soft joint may still be creeping when read (see fit_servo_model_to_sweeps.py),
which leaves it slightly short of the edge.

Run from lowlevel/ in the IKsim_mj env (takes a few minutes):
    python friction/fit_friction_sag_model.py
    python friction/fit_friction_sag_model.py --fix-offsets
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pybullet as p
from scipy.optimize import least_squares

# lowlevel/ on the path, for paths.py and the friction package (as kinematicscalibration/ does).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paths
from friction.fit_servo_model_to_sweeps import (
    JOINT_NAMES,
    SWEEP_DIR,
    Sweep,
    held_out_folds,
    joint_offsets_deg,
    load_sweeps,
    rmse,
)

GRAVITY_JOINTS = ["shoulder_lift", "elbow_flex", "wrist_flex"]
NO_GRAVITY_JOINTS = ["shoulder_pan", "wrist_roll"]

# Prior uncertainty (deg) of multisim_chess_fast.py's DEFAULT_JOINT_OFFSETS_DEG (pan, lift,
# elbow, wrist flex, wrist roll): the ruler's ~2 mm at the tip for the two fitted joints
# (about 3 mm per degree), 0 for the ones never measured. Moved here from the sim's
# DEFAULT_JOINT_OFFSET_SD_DEG when the sim stopped drawing random offsets (2026-11-19).
OFFSET_PRIOR_SD_DEG = (0.0, 0.0, 0.7, 0.7, 0.0)
# Prior uncertainty (deg) for an offset listed above with sd 0, i.e. one that was never measured.
UNMEASURED_OFFSET_SD_DEG = 5.0
# How far a fitted offset may move from the sim's value (deg).
OFFSET_SEARCH_RANGE_DEG = 15.0

# Parameter bounds: kp (N*m/rad), K_c (N*m), K_l (dimensionless, must stay below 1).
KP_BOUNDS = (0.5, 2000.0)
KC_BOUNDS = (0.0, 2.0)
KL_BOUNDS = (0.0, 0.95)

# Starting points for the optimiser; the best result is kept.
KP_STARTS = (5.0, 10.0, 30.0)
KC_STARTS = (0.02, 0.08)
KL_STARTS = (0.1, 0.4)

# A move smaller than this (deg) can't show which way the joint went. The servo reads in
# steps of 0.088 deg, so this is a few steps.
MIN_MOVE_DEG = 0.3


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Reading:
    """One scored reading of a sweep: the swept joint's error after one move."""

    joint: str
    target_deg: float
    pose_deg: np.ndarray   # (6,) all joints' readings, motor degrees
    error_deg: float       # target - reading of the swept joint, in theta
    move_dir: float        # +1 if the move went toward +theta, -1 if toward -theta
    sweep: str = ""        # Sweep.label of the sweep it came from
    index: int = 0         # its position in that sweep (0 is the first reading)


def load_sweep_dir(sweep_dir: Path) -> list[Sweep]:
    """Every toggle_<joint>_calibration.json in a folder (e.g. one coverage sweep's folder)."""
    joints = sorted(
        path.name.removeprefix("toggle_").removesuffix("_calibration.json")
        for path in sweep_dir.glob("toggle_*_calibration.json")
    )
    if not joints:
        raise FileNotFoundError(f"No toggle_<joint>_calibration.json in {sweep_dir}")
    return load_sweeps(sweep_dir, joints)


def unclear_moves(sweep: Sweep) -> dict[int, str]:
    """Readings whose approach direction can't be trusted, by index, with the reason.

    The model needs the direction the joint last slipped. That's the way the command moved
    (Sweep.approach_directions) only if the joint actually moved: if it changed by less
    than MIN_MOVE_DEG it may have stayed stuck where the previous move left it, and if it
    moved the other way something else (contact, overshoot, a load change) moved it.
    """
    readings = sweep.readings_deg[:, sweep.joint_index]
    directions = sweep.approach_directions
    unclear = {}
    for k in range(1, len(readings)):
        moved = readings[k] - readings[k - 1]
        if abs(moved) < MIN_MOVE_DEG:
            unclear[k] = f"moved only {moved:+.2f} deg"
        elif np.sign(moved) != directions[k]:
            unclear[k] = f"moved {moved:+.2f} deg, against the command"
    return unclear


def readings_from_sweep(sweep: Sweep) -> list[Reading]:
    """Every reading except the first, whose approach direction is unknown, and the
    unclear_moves ones, whose direction can't be trusted."""
    errors = -sweep.errors_deg  # Sweep.errors_deg is reading - target
    directions = sweep.approach_directions
    unclear = unclear_moves(sweep)
    return [
        Reading(sweep.joint, float(sweep.targets_deg[k]), sweep.readings_deg[k], float(errors[k]), float(directions[k]),
                sweep.label, k)
        for k in range(1, len(sweep.targets_deg))
        #disregard the first reading
        if k not in unclear
    ]


class GravityModel:
    """Gravity holding torques of the URDF robot, from PyBullet's inverse dynamics."""

    def __init__(self) -> None:
        self.client = p.connect(p.DIRECT)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client)
        self.robot = p.loadURDF(
            str(paths.URDF_PATH), useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=self.client
        )
        n = p.getNumJoints(self.robot, physicsClientId=self.client)
        # One tuple per URDF joint: (index, name as bytes, joint type, ...), e.g.
        #   infos[1] = (1, b"shoulder_lift", p.JOINT_REVOLUTE, ...)
        #   infos[5] = (5, b"gripper_frame_joint", p.JOINT_FIXED, ...)
        infos = [p.getJointInfo(self.robot, j, physicsClientId=self.client) for j in range(n)]
        # URDF indices of the joints that can move; the two fixed ones (5 gripper_frame_joint,
        # 7 wrist_camera_mount_joint) are left out:
        #   [0, 1, 2, 3, 4, 6]
        self.movable = [j for j, info in enumerate(infos) if info[2] != p.JOINT_FIXED]
        # Joint name -> URDF index, all 8 joints:
        #   {"shoulder_pan": 0, "shoulder_lift": 1, "elbow_flex": 2, "wrist_flex": 3, "wrist_roll": 4,
        #    "gripper_frame_joint": 5, "gripper": 6, "wrist_camera_mount_joint": 7}
        by_name = {info[1].decode(): j for j, info in enumerate(infos)}
        # Joint name -> position in calculateInverseDynamics' lists, which only have the movable joints
        # (so the gripper is slot 5, not its URDF index 6):
        #   {"shoulder_pan": 0, "shoulder_lift": 1, "elbow_flex": 2, "wrist_flex": 3, "wrist_roll": 4,
        #    "gripper": 5}
        self.position_slot = {name: self.movable.index(by_name[name]) for name in JOINT_NAMES}

    def holding_torque(self, joint: str, pose_deg: np.ndarray, offsets_deg: np.ndarray) -> float:
        """Torque joint `joint` must apply to hold the pose still (N*m, in +theta)."""
        # A list of zeros, one per movable joint ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), filled in below with
        # each joint's angle in radians.
        positions = [0.0] * len(self.movable)
        for i, name in enumerate(JOINT_NAMES):
            positions[self.position_slot[name]] = float(np.deg2rad(pose_deg[i] + offsets_deg[i]))
        # Zero velocities and accelerations: the arm is held still, so the torques are gravity only.
        zeros = [0.0] * len(self.movable)
        torques = p.calculateInverseDynamics(self.robot, positions, zeros, zeros, physicsClientId=self.client)
        return float(torques[self.position_slot[joint]])


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JointParams:
    kp: float  # N*m/rad
    kc: float  # N*m, friction at zero load
    kl: float  # extra friction per N*m of gearbox load


def predicted_up_error(tau_g: float, lifting: bool, joint: JointParams) -> tuple[float, int]:
    """M3 stopping error along up (rad, positive = sagged) and which case applied (1, 2, 3)."""
    kp, kc, kl = joint.kp, joint.kc, joint.kl
    if lifting:
        return (kc + (1 + kl) * tau_g) / ((1 - kl) * kp), 1
    if tau_g >= kc / 2:
        return (-kc + (1 - kl) * tau_g) / ((1 + kl) * kp), 2
    return (-kc + (1 + kl) * tau_g) / ((1 - kl) * kp), 3


@dataclass(frozen=True)
class Prediction:
    error_deg: float     # predicted target - reading, in theta
    tau_hold: float      # N*m, signed
    lifting: bool
    case: int


def predict(reading: Reading, joint: JointParams, offsets_deg: np.ndarray, gravity: GravityModel) -> Prediction:
    tau_hold = gravity.holding_torque(reading.joint, reading.pose_deg, offsets_deg)
    s = 1.0 if tau_hold >= 0 else -1.0
    lifting = reading.move_dir == s
    e_up, case = predicted_up_error(abs(tau_hold), lifting, joint)
    return Prediction(float(np.rad2deg(s * e_up)), tau_hold, lifting, case)


class ParameterVector:
    """Packs the fitted parameters into the flat vector least_squares works on:
    (kp, K_c, K_l) per gravity joint, then the fitted offsets in degrees.
    E.g. (kp_joint1,K_c_joint1, K_l_joint1, kp_joint2,....)
    """

    def __init__(self, offset_joints: list[str]) -> None:
        self.offset_joints = offset_joints

    def unpack(self, x: np.ndarray, base_offsets_deg: np.ndarray) -> tuple[dict[str, JointParams], np.ndarray]:
        joints = {
            name: JointParams(kp=x[3 * i], kc=x[3 * i + 1], kl=x[3 * i + 2])
            for i, name in enumerate(GRAVITY_JOINTS)
        }
        offsets = base_offsets_deg.copy()
        for i, name in enumerate(self.offset_joints):
            offsets[JOINT_NAMES.index(name)] = x[3 * len(GRAVITY_JOINTS) + i]
        return joints, offsets

    def names(self) -> list[str]:
        servo = [f"{j}.{p}" for j in GRAVITY_JOINTS for p in ("kp", "K_c", "K_l")]
        return servo + [f"{j}.offset_deg" for j in self.offset_joints]

    def bounds(self, base_offsets_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        low = [KP_BOUNDS[0], KC_BOUNDS[0], KL_BOUNDS[0]] * len(GRAVITY_JOINTS)
        high = [KP_BOUNDS[1], KC_BOUNDS[1], KL_BOUNDS[1]] * len(GRAVITY_JOINTS)
        for name in self.offset_joints:
            centre = base_offsets_deg[JOINT_NAMES.index(name)]
            low.append(centre - OFFSET_SEARCH_RANGE_DEG)
            high.append(centre + OFFSET_SEARCH_RANGE_DEG)
        return np.array(low), np.array(high)

    def starts(self, base_offsets_deg: np.ndarray) -> list[np.ndarray]:
        offsets = [base_offsets_deg[JOINT_NAMES.index(n)] for n in self.offset_joints]
        return [
            np.array([kp, kc, kl] * len(GRAVITY_JOINTS) + offsets)
            for kp, kc, kl in itertools.product(KP_STARTS, KC_STARTS, KL_STARTS)
        ]


@dataclass(frozen=True)
class OffsetPrior:
    mean_deg: np.ndarray  # (6,) the sim's offsets
    sd_deg: np.ndarray    # (6,)


def offset_prior(base_offsets_deg: np.ndarray) -> OffsetPrior:
    """The sim's offsets, with OFFSET_PRIOR_SD_DEG's uncertainty (or a loose one where it
    lists none)."""
    sd = np.array([*OFFSET_PRIOR_SD_DEG, 0.0], dtype=float)
    sd[sd <= 0] = UNMEASURED_OFFSET_SD_DEG
    return OffsetPrior(base_offsets_deg.copy(), sd)


@dataclass
class FitResult:
    joints: dict[str, JointParams]
    offsets_deg: np.ndarray
    x: np.ndarray
    sd: np.ndarray  # one-sigma uncertainty of each entry of x
    rmse_deg: float


def fit(
    readings: list[Reading],
    layout: ParameterVector,
    prior: OffsetPrior,
    noise_deg: float,
    gravity: GravityModel,
) -> FitResult:
    """Least-squares fit of every gravity joint's parameters and the shared offsets.

    Each residual is (measured - predicted) / noise_deg, so the data and the offset
    priors are weighed against each other in units of their uncertainty.
    """
    fitted = [r for r in readings if r.joint in GRAVITY_JOINTS]

    def residuals(x: np.ndarray) -> np.ndarray:
        joints, offsets = layout.unpack(x, prior.mean_deg)
        data = [
            (r.error_deg - predict(r, joints[r.joint], offsets, gravity).error_deg) / noise_deg
            for r in fitted
        ]
        priors = [
            (offsets[JOINT_NAMES.index(n)] - prior.mean_deg[JOINT_NAMES.index(n)]) / prior.sd_deg[JOINT_NAMES.index(n)]
            for n in layout.offset_joints
            if np.isfinite(prior.sd_deg[JOINT_NAMES.index(n)])  # an infinite sd means no prior
        ]
        """
        This priors term is a penalty. The fit is allowed to move an offset away from the 
        sim's value, but it pays (change / sd)² for doing so. Moving the elbow offset by 0.7° 
        costs the same as missing one reading by 0.1°.
        """
        return np.array(data + priors)
        #for python lists + joins them end to end, not elementwise
    
    low, high = layout.bounds(prior.mean_deg)
    best = None
    for start in layout.starts(prior.mean_deg):
        result = least_squares(residuals, start, bounds=(low, high), x_scale="jac")
        if best is None or result.cost < best.cost:
            best = result

    # One-sigma uncertainties from the Jacobian, scaled by how well the model fits
    # (as scipy's curve_fit does when the noise level isn't known exactly).
    dof = max(1, len(best.fun) - len(best.x))
    scale = 2 * best.cost / dof
    covariance = np.linalg.pinv(best.jac.T @ best.jac) * scale
    joints, offsets = layout.unpack(best.x, prior.mean_deg)
    data_residuals = best.fun[: len(fitted)] * noise_deg
    return FitResult(joints, offsets, best.x, np.sqrt(np.diag(covariance)), rmse(data_residuals))


# ---------------------------------------------------------------------------
# Baselines and scoring
# ---------------------------------------------------------------------------

def linear_fit_predict(columns: list[np.ndarray], y: np.ndarray, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Least-squares fit of y = sum(coef * column) on `train`, predicted on `test`."""
    design = np.column_stack(columns)
    coef, *_ = np.linalg.lstsq(design[train], y[train], rcond=None)
    return design[test] @ coef


@dataclass
class JointScore:
    rmse_no_correction: float
    rmse_fixed_shift: float
    rmse_shift_plus_sag: float
    rmse_m3: float


def held_out_scores(
    sweeps: list[Sweep],
    readings: list[Reading],
    layout: ParameterVector,
    prior: OffsetPrior,
    noise_deg: float,
    gravity: GravityModel,
    by: str = "angle",
) -> tuple[dict[str, JointScore], list[dict]]:
    """Leave-one-out: refit without a group of readings and predict them, for every group.
    Scores pool all of a joint's sweeps.

    by="angle": a group is one angle of one sweep (approached from both sides), so the model
    is scored on angles it never saw, at poses it did see.
    by="sweep": a group is a whole sweep, so the model is scored on poses it never saw.

    Also returns every reading's held-out predictions (M3 and both baselines), so they can be
    split or plotted, e.g. by approach direction."""
    scores = {}
    predictions = []
    for joint in GRAVITY_JOINTS:
        own = [i for i, r in enumerate(readings) if r.joint == joint]
        if not own:
            continue
        if by == "sweep" and sum(s.joint == joint for s in sweeps) < 2:
            print(f"  {joint}: only one sweep, so leaving it out leaves nothing to fit; skipped")
            continue
        errors = np.array([readings[i].error_deg for i in own])
        directions = np.array([readings[i].move_dir for i in own])
        tau = np.array([gravity.holding_torque(joint, readings[i].pose_deg, prior.mean_deg) for i in own])
        ones = np.ones_like(errors)

        m3 = np.full(len(own), np.nan)
        shift = np.full(len(own), np.nan)
        shift_sag = np.full(len(own), np.nan)
        lifting = np.zeros(len(own), dtype=bool)
        tau_hold = np.full(len(own), np.nan)  # at the held-out fit's offsets
        for sweep in (s for s in sweeps if s.joint == joint):
            # Each fold is reading indices within the sweep; find those readings in `own`
            # (some may be missing, left out by unclear_moves).
            folds = held_out_folds(sweep.targets_deg) if by == "angle" else [np.arange(1, len(sweep.targets_deg))]
            for fold in folds:
                test = np.array([k for k, i in enumerate(own)
                                 if readings[i].sweep == sweep.label and readings[i].index in fold], dtype=int)
                if len(test) == 0:
                    continue
                train = np.setdiff1d(np.arange(len(own)), test)
                held_out = {own[k] for k in test}
                result = fit([r for i, r in enumerate(readings) if i not in held_out], layout, prior, noise_deg, gravity)
                for k in test:
                    pred = predict(readings[own[k]], result.joints[joint], result.offsets_deg, gravity)
                    m3[k], lifting[k], tau_hold[k] = pred.error_deg, pred.lifting, pred.tau_hold
                shift[test] = linear_fit_predict([ones, directions], errors, train, test)
                shift_sag[test] = linear_fit_predict([ones, directions, tau], errors, train, test)
        scores[joint] = JointScore(
            rmse_no_correction=rmse(errors),
            rmse_fixed_shift=rmse(shift - errors),
            rmse_shift_plus_sag=rmse(shift_sag - errors),
            rmse_m3=rmse(m3 - errors),
        )
        for k, i in enumerate(own):
            r = readings[i]
            predictions.append({
                "joint": joint, "sweep": r.sweep, "index": r.index, "target_deg": r.target_deg,
                "measured_error_deg": r.error_deg, "lifting": bool(lifting[k]), "tau_hold": float(tau_hold[k]),
                "m3_deg": float(m3[k]), "fixed_shift_deg": float(shift[k]), "shift_plus_sag_deg": float(shift_sag[k]),
            })
    return scores, predictions


def test_scores(
    test_sweeps: list[Sweep],
    train_readings: list[Reading],
    result: FitResult,
    prior: OffsetPrior,
    gravity: GravityModel,
) -> dict[str, JointScore]:
    """Score the fit on sweeps it never saw, by Sweep.label. The baselines are fitted on
    the same training readings of the joint."""
    scores = {}
    for sweep in test_sweeps:
        if sweep.joint not in GRAVITY_JOINTS:
            continue
        test = readings_from_sweep(sweep)
        both = [r for r in train_readings if r.joint == sweep.joint] + test
        train_rows = np.arange(len(both) - len(test))
        test_rows = np.arange(len(both) - len(test), len(both))
        errors = np.array([r.error_deg for r in both])
        directions = np.array([r.move_dir for r in both])
        tau = np.array([gravity.holding_torque(sweep.joint, r.pose_deg, prior.mean_deg) for r in both])
        ones = np.ones_like(errors)
        shift = linear_fit_predict([ones, directions], errors, train_rows, test_rows)
        shift_sag = linear_fit_predict([ones, directions, tau], errors, train_rows, test_rows)
        m3 = np.array([predict(r, result.joints[sweep.joint], result.offsets_deg, gravity).error_deg for r in test])
        measured = errors[test_rows]
        scores[sweep.label] = JointScore(
            rmse_no_correction=rmse(measured),
            rmse_fixed_shift=rmse(shift - measured),
            rmse_shift_plus_sag=rmse(shift_sag - measured),
            rmse_m3=rmse(m3 - measured),
        )
    return scores


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_fit(result: FitResult, layout: ParameterVector, readings: list[Reading], gravity: GravityModel) -> None:
    print(f"\nFitted on all readings (RMSE {result.rmse_deg:.3f} deg)")
    for name, value, sd in zip(layout.names(), result.x, result.sd):
        print(f"  {name:<26} {value:9.4f}  +- {sd:.4f}")

    for joint in GRAVITY_JOINTS:
        params = result.joints[joint]
        print(f"\n{joint}: kp {params.kp:.1f} N*m/rad, K_c {params.kc:.3f} N*m, K_l {params.kl:.2f}"
              f"  (third case below tau_g = K_c/2 = {params.kc / 2:.3f} N*m)")
        sweep_width = max(len(r.sweep) for r in readings if r.joint == joint)
        print(f"   {'sweep':<{sweep_width}}   target   move      tau_hold   measured   predicted   case   data says third case")
        for r in readings:
            if r.joint != joint:
                continue
            pred = predict(r, params, result.offsets_deg, gravity)
            s = 1.0 if pred.tau_hold >= 0 else -1.0
            e_up = np.deg2rad(s * r.error_deg)
            data_third = (not pred.lifting) and e_up < -abs(pred.tau_hold) / params.kp
            flag = "yes" if data_third else "no"
            if data_third != (pred.case == 3):
                flag += "  (disagrees)"
            print(f"   {r.sweep:<{sweep_width}}   {r.target_deg:+6.1f}   {'lifting ' if pred.lifting else 'lowering'}  {pred.tau_hold:+.3f}"
                  f"     {r.error_deg:+.2f}      {pred.error_deg:+.2f}       {pred.case}      {flag}")


def print_scores(scores: dict[str, JointScore], title: str) -> None:
    width = max([14, *(len(name) for name in scores)])
    print(f"\n{title}")
    print(f"  {'':<{width}} {'no correction':>14} {'fixed shift':>12} {'shift + sag':>12} {'M3 fit':>8}")
    for name, s in scores.items():
        print(f"  {name:<{width}} {s.rmse_no_correction:14.3f} {s.rmse_fixed_shift:12.3f}"
              f" {s.rmse_shift_plus_sag:12.3f} {s.rmse_m3:8.3f}")


def print_per_sweep(predictions: list[dict]) -> None:
    """RMSE of each held-out sweep on its own, split by the joint's approach direction."""
    if not predictions:
        return
    print("\nPer held-out sweep, M3 RMSE (deg)       all   arriving down   arriving up")
    labels = list(dict.fromkeys(p["sweep"] for p in predictions))
    width = max(len(label) for label in labels)
    for label in labels:
        own = [p for p in predictions if p["sweep"] == label]
        miss = lambda rows: rmse(np.array([p["measured_error_deg"] - p["m3_deg"] for p in rows])) if rows else float("nan")
        print(f"  {label:<{width}}  {miss(own):6.3f}   {miss([p for p in own if not p['lifting']]):9.3f}"
              f"   {miss([p for p in own if p['lifting']]):9.3f}")


def no_gravity_bands(sweeps: list[Sweep], readings: list[Reading]) -> dict[str, dict[str, float]]:
    """Fixed-shift fit (error = c + h * direction) for the joints without gravity load."""
    bands = {}
    for sweep in sweeps:
        if sweep.joint not in NO_GRAVITY_JOINTS:
            continue
        own = [r for r in readings if r.joint == sweep.joint]
        errors = np.array([r.error_deg for r in own])
        design = np.column_stack([np.ones(len(own)), [r.move_dir for r in own]])
        (c, h), *_ = np.linalg.lstsq(design, errors, rcond=None)
        bands[sweep.joint] = {"half_width_deg": float(h), "constant_deg": float(c)}
    return bands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-dir", type=Path, default=SWEEP_DIR,
                        help="Folder with one toggle sweep of each of the five arm joints.")
    parser.add_argument("--train-dirs", type=Path, nargs="*", default=[],
                        help="More folders of toggle sweeps (any joints) to fit on, e.g. the coverage *_train folders.")
    parser.add_argument("--test-dirs", type=Path, nargs="*", default=[],
                        help="Folders of toggle sweeps to score the fit on, never fitted on.")
    parser.add_argument("--fix-offsets", action="store_true", help="Keep the calibration offsets at the sim's values.")
    parser.add_argument("--no-offset-prior", action="store_true",
                        help="Fit the offsets from the data alone, without the pull toward the sim's values "
                             "(they still start there and stay within OFFSET_SEARCH_RANGE_DEG of them).")
    parser.add_argument("--noise-deg", type=float, default=0.1,
                        help="Assumed reading noise; sets how strongly the data pull against the offset priors.")
    parser.add_argument("--no-held-out", action="store_true", help="Skip the leave-one-angle-out scoring.")
    parser.add_argument("--held-out-sweeps", action="store_true",
                        help="Also score leaving out one whole sweep at a time (poses the fit never saw).")
    parser.add_argument("--output", type=Path, default=SWEEP_DIR / "friction_sag_fit.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sweeps = load_sweeps(args.sweep_dir, NO_GRAVITY_JOINTS[:1] + GRAVITY_JOINTS + NO_GRAVITY_JOINTS[1:])
    sweeps += [sweep for folder in args.train_dirs for sweep in load_sweep_dir(folder)]
    test_sweeps = [sweep for folder in args.test_dirs for sweep in load_sweep_dir(folder)]
    readings = [r for sweep in sweeps for r in readings_from_sweep(sweep)]
    print(f"Fitting on {len(sweeps)} sweeps ({len(readings)} readings); testing on {len(test_sweeps)}.")
    for sweep in sweeps + test_sweeps:
        for k, reason in unclear_moves(sweep).items():
            print(f"Left out {sweep.label} reading {k + 1} (target {sweep.targets_deg[k]:+.1f} deg): {reason}")
    gravity = GravityModel()
    prior = offset_prior(joint_offsets_deg())
    if args.no_offset_prior:
        prior = OffsetPrior(prior.mean_deg, np.full_like(prior.sd_deg, np.inf))
    layout = ParameterVector([] if args.fix_offsets else GRAVITY_JOINTS)

    print("Offset priors (deg, URDF = motor + offset):")
    for name in GRAVITY_JOINTS:
        i = JOINT_NAMES.index(name)
        status = "fixed" if args.fix_offsets else "no prior" if args.no_offset_prior else f"+- {prior.sd_deg[i]:.1f}"
        print(f"  {name:<14} {prior.mean_deg[i]:+.2f}  {status}")

    result = fit(readings, layout, prior, args.noise_deg, gravity)
    print_fit(result, layout, readings, gravity)

    bands = no_gravity_bands(sweeps, readings)
    print("\nJoints without gravity load (only the band half-width in degrees can be found):")
    for joint, band in bands.items():
        print(f"  {joint:<14} half-width {band['half_width_deg']:.2f} deg, constant {band['constant_deg']:+.2f} deg")

    scores, held_out = {}, []
    if not args.no_held_out:
        print("\nScoring on held-out angles (refits the model once per angle)...")
        scores, held_out = held_out_scores(sweeps, readings, layout, prior, args.noise_deg, gravity)
        print_scores(scores, "Held-out RMSE (deg): each angle predicted by a fit that never saw it")

    sweep_scores, sweep_held_out = {}, []
    if args.held_out_sweeps:
        print("\nScoring on held-out sweeps (refits the model once per sweep)...")
        sweep_scores, sweep_held_out = held_out_scores(sweeps, readings, layout, prior, args.noise_deg, gravity, by="sweep")
        print_scores(sweep_scores, "Held-out RMSE (deg): each sweep predicted by a fit that never saw its pose")
        print_per_sweep(sweep_held_out)

    tested = {}
    if test_sweeps:
        tested = test_scores(test_sweeps, readings, result, prior, gravity)
        print_scores(tested, "Test RMSE (deg): sweeps the fit never saw")

    output = {
        "model": "M3 at rest, see fit_friction_sag_model.py",
        "error_sign": "target - reading",
        "noise_deg": args.noise_deg,
        "fixed_offsets": args.fix_offsets,
        "offset_prior": not (args.fix_offsets or args.no_offset_prior),
        "parameters": {name: {"value": float(v), "sd": float(s)} for name, v, s in zip(layout.names(), result.x, result.sd)},
        "offsets_deg": dict(zip(JOINT_NAMES, result.offsets_deg.tolist())),
        "rmse_fitted_deg": result.rmse_deg,
        "no_gravity_bands": bands,
        "held_out_rmse_deg": {joint: asdict(s) for joint, s in scores.items()},
        "held_out_predictions": held_out,
        "held_out_sweep_rmse_deg": {joint: asdict(s) for joint, s in sweep_scores.items()},
        "held_out_sweep_predictions": sweep_held_out,
        "train_sweeps": [sweep.label for sweep in sweeps],
        "test_rmse_deg": {name: asdict(s) for name, s in tested.items()},
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nsaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
