#!/usr/bin/env python3
"""Fit a physical servo model to the real-arm friction sweeps by replaying them in PyBullet.

The sim currently fakes joint friction. multisim_chess_fast.py commands each joint to
`target + offset - h * motion_dir`, so it always stops h short of the target, and
PyBullet's POSITION_CONTROL motor tracks that exactly. This script tests a physical
model instead, where the stopping error comes out of the forces:

  * the servo pulls toward its target like a damped spring with a torque limit
    (torque = kp * (target - q) - kv * qdot, clipped to +-max_torque),
  * dry friction resists motion with up to a fixed torque, and
  * gravity loads the joint, so how far it sags depends on the pose.

Everything runs in simulation; nothing talks to the arm.

  Step 1  Build the servo model from the STS3215 values in the SO-ARM100 MuJoCo
          model (so101_new_calib.xml), translated to PyBullet as in the table below.
  Step 2  Replay every sweep in real_world_runs/friction/ (recorded with
          toggle_joint_calibration.py) using the upstream values as they are, and
          compare the sim's settled angles with the real arm's.
  Step 3  Fit kp and the friction torque per joint on a grid. Every model is scored
          on readings at angles it was not fitted to (leave-one-angle-out), next to
          the current fixed-shift model fitted the same way.

MuJoCo -> PyBullet, using the values MuJoCo compiles from the MJCF (actuator
attributes override the class defaults, e.g. forcerange is 3.35 not 2.94):

  <position kp>          PD_CONTROL positionGain. Both are N*m/rad (torque = kp * error).
  <position kv>          PD_CONTROL velocityGain, with targetVelocity 0.
  <position forcerange>  PD_CONTROL force, the torque cap.
  <joint frictionloss>   VELOCITY_CONTROL motor with targetVelocity 0, velocityGain 1 and
                         force = frictionloss. velocityGain must be 1: at PyBullet's 0.1
                         the motor only removes 10% of the speed per step, which is drag,
                         not dry friction. MuJoCo's friction is a soft constraint and this
                         one is hard: with the same value, MuJoCo's joint creeps about 10%
                         further into the friction band.
  <joint damping>        changeDynamics(jointDamping). PyBullet adds it as a torque once
                         per stepSimulation call, which is why the sim runs without substeps.
  <joint armature>       PyBullet has no equivalent. It is added to the swept joint's child
                         link as isotropic inertia, which raises the swept joint's inertia by
                         exactly `armature` because every other joint is held still. That is
                         exact for these one-joint sweeps, not a general substitute.

Replay of one sweep, mirroring toggle_joint_calibration.py:
  * The swept joint uses the servo model. The script ramped each move in 200 commands
    sent every 1/240 s from the previous settled reading, held for 2 s, then read the
    angle. The real command period also includes the bus round trip, so set
    --command-hz lower if the ramps were slower.
  * The other joints are held at the angles the real arm read: the swept joint's
    gravity load depends on them, and they drift during a sweep (shoulder_lift moved
    5 deg during the elbow sweep).
  * The first reading only sets the starting state, because the direction the arm
    approached it from isn't recorded. The rest are scored.
  * Angles: URDF angle = motor angle + calibration offset, with the offsets read from
    multisim_chess_fast.py (or SIM_JOINT_OFFSETS_DEG), as in the main sim. The error
    being compared, reading - target, is in motor degrees on both sides.

Run from lowlevel/ in the IKsim_mj env. All five joints take about 35 minutes on 8 cores
(1,795 replays); --joints runs a subset:
    python friction/fit_servo_model_to_sweeps.py
    python friction/fit_servo_model_to_sweeps.py --joints elbow_flex --workers 4
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
import pybullet as p

# lowlevel/ on the path, for paths.py and the shared arm modules (as kinematicscalibration/ does).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paths

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINT_NAMES = JOINT_NAMES[:5]

SWEEP_DIR = paths.REAL_WORLD_RUNS_DIR / "friction"
UPSTREAM_MJCF = paths.SO_ARM100_DIR / "so101_new_calib.xml"
# The defaults file so101_new_calib.xml was made from. It still has the original
# kp = 17.8 and kv = 0 that so101_new_calib.xml replaced (see the comment there).
UPSTREAM_DEFAULTS = paths.SO_ARM100_DIR / "joints_properties.xml"
SERVO_CLASS = "sts3215"

# Torque cap of the motors holding the non-swept joints (N*m): far above any load here.
HOLD_TORQUE = 100.0

# Constraint solver settings. The friction motor is a constraint, and with PyBullet's
# defaults (50 iterations) the solver stops before it converges, so a joint under a
# torque below the friction limit still creeps at about 0.1 deg/s. These make it hold.
SOLVER_ITERATIONS = 200
SOLVER_RESIDUAL_THRESHOLD = 1e-12

# Step 3 search grid. kp spans the measured sag (around 10) up to the upstream 998.
KP_GRID = np.geomspace(2.0, 2000.0, 21)  # N*m/rad
FRICTION_GRID = np.concatenate([[0.0], np.geomspace(0.005, 1.0, 16)])  # N*m

# A reading taken while the swept joint still moves faster than this counts as unsettled.
# A soft servo creeps up to its friction limit slowly, so its reading depends on the
# hold time and --hold-s has to match the real one.
SETTLED_SPEED_DEG_S = 0.1


# ---------------------------------------------------------------------------
# Step 1: the servo model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServoParams:
    """One servo's parameters, in the MJCF's units."""

    kp: float          # N*m/rad, position gain
    kv: float          # N*m*s/rad, velocity gain of the position controller
    max_torque: float  # N*m, cap on the controller's torque
    friction: float    # N*m, dry friction (MJCF frictionloss)
    damping: float     # N*m*s/rad, passive joint damping
    armature: float    # kg*m^2, reflected rotor inertia


def servo_params_from_model(model: mujoco.MjModel, joint_name: str) -> ServoParams:
    """Read a joint's servo parameters out of a compiled MuJoCo model.

    Reading the compiled model, rather than the XML attributes, applies MuJoCo's own
    rules for default classes and per-element overrides.
    """
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"No joint called {joint_name!r} in the MuJoCo model")
    actuators = [a for a in range(model.nu) if model.actuator_trnid[a, 0] == joint_id]
    if len(actuators) != 1:
        raise ValueError(f"Expected one actuator on {joint_name!r}, found {len(actuators)}")
    actuator = actuators[0]

    # A position actuator's force is gain*ctrl + bias, with gain = kp and
    # bias = [0, -kp, -kv]. Check it is one before reading kp and kv off it.
    gain = model.actuator_gainprm[actuator]
    bias = model.actuator_biasprm[actuator]
    if not (np.isclose(bias[1], -gain[0]) and bias[0] == 0.0):
        raise ValueError(f"The actuator on {joint_name!r} is not a <position> actuator")
    if not model.actuator_forcelimited[actuator]:
        raise ValueError(f"The actuator on {joint_name!r} has no forcerange")
    low, high = model.actuator_forcerange[actuator]
    if not np.isclose(low, -high):
        raise ValueError(f"The actuator on {joint_name!r} has an asymmetric forcerange")

    dof = model.jnt_dofadr[joint_id]
    return ServoParams(
        kp=float(gain[0]),
        kv=float(-bias[2]),
        max_torque=float(high),
        friction=float(model.dof_frictionloss[dof]),
        damping=float(model.dof_damping[dof]),
        armature=float(model.dof_armature[dof]),
    )


def servo_params_from_mjcf(mjcf_path: Path, joint_name: str) -> ServoParams:
    """Servo parameters of one joint of a full MJCF robot model."""
    return servo_params_from_model(mujoco.MjModel.from_xml_path(str(mjcf_path)), joint_name)


def servo_params_from_defaults_file(defaults_path: Path, class_name: str) -> ServoParams:
    """Servo parameters from a file that only holds MJCF <default> classes.

    MuJoCo can't compile such a file alone, so it is wrapped in a one-joint model
    that uses the class.
    """
    wrapper = f"""
    <mujoco>
      {defaults_path.read_text(encoding="utf-8")}
      <worldbody>
        <body>
          <joint name="servo" type="hinge" class="{class_name}"/>
          <geom type="sphere" size="0.01" mass="0.1"/>
        </body>
      </worldbody>
      <actuator>
        <position joint="servo" class="{class_name}"/>
      </actuator>
    </mujoco>
    """
    return servo_params_from_model(mujoco.MjModel.from_xml_string(wrapper), "servo")


def apply_servo_model(robot: int, joint: int, servo: ServoParams) -> None:
    """Set up the friction, damping and armature of the swept joint (see the module docstring)."""
    # Dry friction. This replaces the joint's default motor, which would lock it.
    p.setJointMotorControl2(
        robot, joint, p.VELOCITY_CONTROL, targetVelocity=0.0, velocityGain=1.0, force=servo.friction
    )
    p.changeDynamics(robot, joint, jointDamping=servo.damping)
    # Isotropic, so it adds `armature` about the joint axis whatever the link's principal axes.
    inertia = np.array(p.getDynamicsInfo(robot, joint)[2])
    p.changeDynamics(robot, joint, localInertiaDiagonal=list(inertia + servo.armature))


def command_servo(robot: int, joint: int, target_rad: float, servo: ServoParams) -> None:
    """Point the servo's position controller at a new target."""
    p.setJointMotorControl2(
        robot, joint, p.PD_CONTROL,
        targetPosition=target_rad, targetVelocity=0.0,
        positionGain=servo.kp, velocityGain=servo.kv, force=servo.max_torque,
    )


def remove_servo_controller(robot: int, joint: int) -> None:
    """Remove the joint's PD controller.

    PD_CONTROL runs in a PyBullet plugin that keys controllers by body id and link,
    and they survive resetSimulation, so a leftover one would push on the next robot
    loaded. Sending force 0 removes it.
    """
    p.setJointMotorControl2(
        robot, joint, p.PD_CONTROL, targetPosition=0.0, positionGain=0.0, velocityGain=0.0, force=0.0
    )


def hold_joint(robot: int, joint: int, target_rad: float) -> None:
    """Hold a non-swept joint at an angle with a stiff motor."""
    p.setJointMotorControl2(
        robot, joint, p.POSITION_CONTROL,
        targetPosition=target_rad, targetVelocity=0.0, positionGain=1.0, velocityGain=1.0, force=HOLD_TORQUE,
    )


# ---------------------------------------------------------------------------
# Sweep data and replay settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sweep:
    """One toggle_joint_calibration.py run: a list of commanded targets for one joint
    and, after each move, the angles all six joints read (motor degrees)."""

    joint: str
    targets_deg: np.ndarray   # (n,) commanded angle of the swept joint
    readings_deg: np.ndarray  # (n, 6) settled readings, in JOINT_NAMES order
    source: str = ""          # name of the folder it was loaded from

    @property
    def joint_index(self) -> int:
        return JOINT_NAMES.index(self.joint)

    @property
    def label(self) -> str:
        """e.g. "friction/elbow_flex" or "03_lift_far_reach_train/shoulder_lift"."""
        return f"{self.source}/{self.joint}"

    @property
    def errors_deg(self) -> np.ndarray:
        """Measured stopping error of the swept joint: reading - target."""
        return self.readings_deg[:, self.joint_index] - self.targets_deg

    @property
    def approach_directions(self) -> np.ndarray:
        """+1 if move k went up, -1 if down. NaN for the first reading (unknown).

        Taken from the change in target, which is the way the command moved and so the
        only way the joint can slip from rest. Comparing the target with the previous
        reading instead goes wrong when the joint stopped short by more than a step: the
        far-reach lift read 49.2 deg at target 40.8, then moved down to 52.4 for target
        48.3, which that comparison calls up.
        """
        return np.concatenate([[np.nan], np.sign(np.diff(self.targets_deg))])


def load_sweeps(sweep_dir: Path, joints: list[str]) -> list[Sweep]:
    sweeps = []
    for joint in joints:
        path = sweep_dir / f"toggle_{joint}_calibration.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["joint"] != joint:
            raise ValueError(f"{path} is a sweep of {data['joint']}, not {joint}")
        sweeps.append(Sweep(
            joint=joint,
            targets_deg=np.array([r["target"] for r in data["results"]], dtype=float),
            readings_deg=np.array([r["observed_deg"] for r in data["results"]], dtype=float),
            source=sweep_dir.name,
        ))
    return sweeps


def read_constant_from_source(path: Path, name: str):
    """Read a literal module-level constant from a Python file without importing it.

    Used for multisim_chess_fast.py, whose import takes about 30 s and prints setup output.
    """
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"No constant {name} in {path}")


def joint_offsets_deg() -> np.ndarray:
    """Calibration offsets (URDF angle = motor angle + offset), in JOINT_NAMES order.

    Same source as multisim_chess_fast.py: SIM_JOINT_OFFSETS_DEG if set, else its
    DEFAULT_JOINT_OFFSETS_DEG. The gripper has none.
    """
    override = os.environ.get("SIM_JOINT_OFFSETS_DEG", "").strip()
    if override:
        arm = [float(v) for v in override.replace(",", " ").split()]
    else:
        arm = read_constant_from_source(paths.LOWLEVEL_DIR / "multisim_chess_fast.py", "DEFAULT_JOINT_OFFSETS_DEG")
    if len(arm) != len(ARM_JOINT_NAMES):
        raise ValueError(f"Expected {len(ARM_JOINT_NAMES)} joint offsets, got {arm}")
    return np.array([*arm, 0.0], dtype=float)


@dataclass(frozen=True)
class ReplaySettings:
    physics_hz: int = 960          # PyBullet step rate
    command_hz: int = 240          # rate the script sent ramp commands at
    ramp_commands: int = 200       # commands per move (toggle_joint_calibration.py --move-steps)
    hold_s: float = 2.0            # settle time before each reading (--hold-s)
    urdf_path: str = str(paths.URDF_PATH)
    offsets_deg: tuple[float, ...] = (0.0,) * len(JOINT_NAMES)

    @property
    def steps_per_command(self) -> int:
        return self.physics_hz // self.command_hz

    @property
    def hold_steps(self) -> int:
        return round(self.hold_s * self.physics_hz)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def load_robot(settings: ReplaySettings) -> tuple[int, dict[str, int]]:
    """Fresh world with gravity and the robot. Returns its id and joint indices by name."""
    p.resetSimulation()
    p.setGravity(0.0, 0.0, -9.81)
    p.setTimeStep(1.0 / settings.physics_hz)
    p.setPhysicsEngineParameter(
        numSubSteps=0,
        numSolverIterations=SOLVER_ITERATIONS,
        solverResidualThreshold=SOLVER_RESIDUAL_THRESHOLD,
    )
    # Without this flag PyBullet ignores the URDF's inertias and recomputes them from
    # the collision meshes.
    robot = p.loadURDF(settings.urdf_path, useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
    by_name = {p.getJointInfo(robot, j)[1].decode(): j for j in range(p.getNumJoints(robot))}
    return robot, {name: by_name[name] for name in JOINT_NAMES}


@dataclass(frozen=True)
class ReplayResult:
    readings_deg: np.ndarray       # (n,) simulated reading of the swept joint, motor degrees
    speeds_deg_s: np.ndarray       # (n,) its speed when read
    held_joint_slip_deg: float     # largest gap between a held joint and its target

    @property
    def unsettled(self) -> bool:
        return bool(np.nanmax(self.speeds_deg_s) > SETTLED_SPEED_DEG_S)


def replay_sweep(sweep: Sweep, servo: ServoParams, settings: ReplaySettings) -> ReplayResult:
    """Run one sweep in PyBullet with the swept joint driven by the servo model."""
    offsets = np.asarray(settings.offsets_deg)
    robot, joint_ids = load_robot(settings)
    swept = joint_ids[sweep.joint]
    held = {name: j for name, j in joint_ids.items() if name != sweep.joint}

    def urdf_rad(name: str, motor_deg: float) -> float:
        return float(np.deg2rad(motor_deg + offsets[JOINT_NAMES.index(name)]))

    def swept_reading_deg() -> float:
        return float(np.rad2deg(p.getJointState(robot, swept)[0])) - offsets[sweep.joint_index]

    def swept_speed_deg_s() -> float:
        return abs(float(np.rad2deg(p.getJointState(robot, swept)[1])))

    held_slip = 0.0

    def hold_others_at(motor_deg: np.ndarray) -> None:
        for name, j in held.items():
            hold_joint(robot, j, urdf_rad(name, motor_deg[JOINT_NAMES.index(name)]))

    def step(n: int, held_targets_deg: np.ndarray) -> None:
        nonlocal held_slip
        for _ in range(n):
            p.stepSimulation()
        for name, j in held.items():
            target = held_targets_deg[JOINT_NAMES.index(name)]
            actual = np.rad2deg(p.getJointState(robot, j)[0]) - offsets[JOINT_NAMES.index(name)]
            held_slip = max(held_slip, abs(actual - target))

    try:
        # Start where the real arm was at the first reading, at rest, with the
        # command it had then, and let it settle.
        for name, j in joint_ids.items():
            p.resetJointState(robot, j, urdf_rad(name, sweep.readings_deg[0, JOINT_NAMES.index(name)]), 0.0)
        apply_servo_model(robot, swept, servo)
        hold_others_at(sweep.readings_deg[0])
        command_servo(robot, swept, urdf_rad(sweep.joint, sweep.targets_deg[0]), servo)
        step(settings.hold_steps, sweep.readings_deg[0])

        readings = [swept_reading_deg()]
        speeds = [swept_speed_deg_s()]
        for k in range(1, len(sweep.targets_deg)):
            start = readings[-1]  # the script ramps from the last settled reading
            for i in range(1, settings.ramp_commands + 1):
                a = i / settings.ramp_commands
                command_servo(robot, swept, urdf_rad(sweep.joint, (1 - a) * start + a * sweep.targets_deg[k]), servo)
                # The held joints drift between readings; walk them along with the ramp.
                others = (1 - a) * sweep.readings_deg[k - 1] + a * sweep.readings_deg[k]
                hold_others_at(others)
                step(settings.steps_per_command, others)
            step(settings.hold_steps, sweep.readings_deg[k])
            readings.append(swept_reading_deg())
            speeds.append(swept_speed_deg_s())
    finally:
        remove_servo_controller(robot, swept)

    return ReplayResult(np.array(readings), np.array(speeds), held_slip)


# Each worker process keeps its own PyBullet connection.
def _connect_worker() -> None:
    p.connect(p.DIRECT)


def _replay_task(task: tuple[Sweep, ServoParams, ReplaySettings]) -> ReplayResult:
    return replay_sweep(*task)


def replay_many(tasks: list[tuple[Sweep, ServoParams, ReplaySettings]], workers: int) -> list[ReplayResult]:
    with ProcessPoolExecutor(max_workers=workers, initializer=_connect_worker) as pool:
        return list(pool.map(_replay_task, tasks, chunksize=4))


def gravity_torque_range(sweep: Sweep, settings: ReplaySettings) -> tuple[float, float]:
    """Smallest and largest gravity torque on the swept joint over the recorded poses (N*m)."""
    robot, joint_ids = load_robot(settings)
    movable = [j for j in range(p.getNumJoints(robot)) if p.getJointInfo(robot, j)[2] != p.JOINT_FIXED]
    torques = []
    for reading in sweep.readings_deg:
        q = {joint_ids[name]: np.deg2rad(reading[i] + settings.offsets_deg[i]) for i, name in enumerate(JOINT_NAMES)}
        positions = [float(q.get(j, 0.0)) for j in movable]
        tau = p.calculateInverseDynamics(robot, positions, [0.0] * len(movable), [0.0] * len(movable))
        torques.append(tau[movable.index(joint_ids[sweep.joint])])
    return float(min(torques)), float(max(torques))


# ---------------------------------------------------------------------------
# Scoring and fitting
# ---------------------------------------------------------------------------

def rmse(residuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(residuals))))


def held_out_folds(targets_deg: np.ndarray) -> list[np.ndarray]:
    """Leave-one-angle-out folds over the scored readings (index 1 onwards).

    Both readings at an angle (approached from above and from below) are held out
    together, so a model is never scored at an angle it was fitted on.
    """
    scored = np.arange(1, len(targets_deg))
    return [scored[targets_deg[scored] == angle] for angle in np.unique(targets_deg[scored])]


def fit_fixed_shift(errors: np.ndarray, directions: np.ndarray, rows: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of the current sim's model, error = c - h * direction.

    c is the constant part (in the sim it ends up in the calibration offsets) and h is
    the friction shift (JOINT_FRICTION_DEG).
    """
    design = np.column_stack([np.ones(len(rows)), -directions[rows]])
    (c, h), *_ = np.linalg.lstsq(design, errors[rows], rcond=None)
    return float(c), float(h)


@dataclass
class JointReport:
    joint: str
    gravity_torque_nm: tuple[float, float]
    rmse_no_correction: float
    rmse_upstream: dict[str, float]
    fixed_shift_c_h_deg: tuple[float, float]
    rmse_fixed_shift_fitted: float
    rmse_fixed_shift_held_out: float
    fitted_kp: float
    fitted_friction: float
    rmse_servo_fitted: float
    rmse_servo_held_out: float
    on_grid_edge: list[str]
    unsettled_readings: bool
    held_joint_slip_deg: float
    measured_errors_deg: list[float]
    servo_fitted_errors_deg: list[float]


def evaluate_joint(
    sweep: Sweep,
    upstream_results: dict[str, ReplayResult],
    grid_results: list[ReplayResult],
    settings: ReplaySettings,
) -> JointReport:
    """Score every model on the sweep's readings, fitting and cross-validating step 3's two."""
    errors = sweep.errors_deg
    directions = sweep.approach_directions
    scored = np.arange(1, len(errors))
    folds = held_out_folds(sweep.targets_deg)

    # Simulated errors for every grid point: shape (len(KP_GRID), len(FRICTION_GRID), n).
    simulated = np.array([r.readings_deg - sweep.targets_deg for r in grid_results])
    simulated = simulated.reshape(len(KP_GRID), len(FRICTION_GRID), len(errors))

    def best_grid_point(rows: np.ndarray) -> tuple[int, int]:
        cost = np.sqrt(np.mean((simulated[:, :, rows] - errors[rows]) ** 2, axis=2))
        return np.unravel_index(np.argmin(cost), cost.shape)

    # Fitted on all scored readings, for the reported parameters.
    i_kp, i_f = best_grid_point(scored)
    c, h = fit_fixed_shift(errors, directions, scored)

    # Held-out predictions: refit without each angle, predict that angle.
    servo_held_out = np.full(len(errors), np.nan)
    shift_held_out = np.full(len(errors), np.nan)
    for test in folds:
        train = np.setdiff1d(scored, test)
        fold_kp, fold_f = best_grid_point(train)
        servo_held_out[test] = simulated[fold_kp, fold_f, test]
        fold_c, fold_h = fit_fixed_shift(errors, directions, train)
        shift_held_out[test] = fold_c - fold_h * directions[test]

    edges = []
    if i_kp in (0, len(KP_GRID) - 1):
        edges.append("kp")
    if i_f == len(FRICTION_GRID) - 1:  # zero friction, the bottom edge, is a valid answer
        edges.append("friction")

    best = grid_results[i_kp * len(FRICTION_GRID) + i_f]
    reported = [*upstream_results.values(), best]
    return JointReport(
        joint=sweep.joint,
        gravity_torque_nm=gravity_torque_range(sweep, settings),
        rmse_no_correction=rmse(errors[scored]),
        rmse_upstream={
            name: rmse(r.readings_deg[scored] - sweep.targets_deg[scored] - errors[scored])
            for name, r in upstream_results.items()
        },
        fixed_shift_c_h_deg=(c, h),
        rmse_fixed_shift_fitted=rmse(c - h * directions[scored] - errors[scored]),
        rmse_fixed_shift_held_out=rmse(shift_held_out[scored] - errors[scored]),
        fitted_kp=float(KP_GRID[i_kp]),
        fitted_friction=float(FRICTION_GRID[i_f]),
        rmse_servo_fitted=rmse(simulated[i_kp, i_f, scored] - errors[scored]),
        rmse_servo_held_out=rmse(servo_held_out[scored] - errors[scored]),
        on_grid_edge=edges,
        unsettled_readings=any(r.unsettled for r in reported),
        held_joint_slip_deg=max(r.held_joint_slip_deg for r in reported),
        measured_errors_deg=errors.tolist(),
        servo_fitted_errors_deg=(best.readings_deg - sweep.targets_deg).tolist(),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_servo_params(label: str, servo: ServoParams) -> None:
    print(
        f"  {label:<24} kp {servo.kp:8.2f} N*m/rad   kv {servo.kv:6.3f}   max torque {servo.max_torque:5.2f} N*m   "
        f"friction {servo.friction:6.3f} N*m   damping {servo.damping:5.2f}   armature {servo.armature:6.3f} kg*m^2"
    )


def print_report(report: JointReport, sweep: Sweep) -> None:
    low, high = report.gravity_torque_nm
    print(f"\n{report.joint}  (gravity torque {low:+.3f} to {high:+.3f} N*m over the sweep)")
    print(f"  {'model':<44} {'RMSE fitted':>12} {'RMSE held-out':>14}")
    print(f"  {'no correction (error = 0)':<44} {report.rmse_no_correction:12.3f} {'-':>14}")
    for name, value in report.rmse_upstream.items():
        print(f"  {'servo model, ' + name + ' as is':<44} {value:12.3f} {'-':>14}")
    c, h = report.fixed_shift_c_h_deg
    print(
        f"  {f'fixed shift  (c {c:+.2f}, h {h:.2f} deg)':<44} "
        f"{report.rmse_fixed_shift_fitted:12.3f} {report.rmse_fixed_shift_held_out:14.3f}"
    )
    ratio = np.rad2deg(report.fitted_friction / report.fitted_kp)
    print(
        f"  {f'servo model  (kp {report.fitted_kp:.1f}, friction {report.fitted_friction:.3f})':<44} "
        f"{report.rmse_servo_fitted:12.3f} {report.rmse_servo_held_out:14.3f}"
    )
    print(f"  fitted friction/kp = {ratio:.2f} deg (the friction band's half-width)")
    print("  readings (deg):  target   measured error   fitted servo error")
    for k, target in enumerate(sweep.targets_deg):
        note = "  (start, not scored)" if k == 0 else ""
        print(
            f"                  {target:+7.1f}   {report.measured_errors_deg[k]:+8.2f}"
            f"         {report.servo_fitted_errors_deg[k]:+8.2f}{note}"
        )
    if abs(high) < 0.05 and abs(low) < 0.05:
        # Without gravity every kp, friction pair with the same ratio fits about equally
        # well, so the best one can land anywhere along that line, grid edges included.
        print("  note: almost no gravity load, so kp and friction can't be told apart here;"
              " only their ratio is meaningful.")
    elif report.on_grid_edge:
        print(f"  WARNING: best fit is at the edge of the grid for {', '.join(report.on_grid_edge)}; widen the grid.")
    if report.unsettled_readings:
        print(f"  note: some sim readings were taken while the joint still crept faster than {SETTLED_SPEED_DEG_S} deg/s,"
              " so they depend on --hold-s matching the hold time the sweep was recorded with.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joints", nargs="+", choices=ARM_JOINT_NAMES, default=ARM_JOINT_NAMES)
    parser.add_argument("--sweep-dir", type=Path, default=SWEEP_DIR)
    parser.add_argument("--hold-s", type=float, default=ReplaySettings.hold_s,
                        help="Settle time before each reading, as passed to toggle_joint_calibration.py.")
    parser.add_argument("--command-hz", type=int, default=ReplaySettings.command_hz,
                        help="Rate of the ramp commands. Must divide --physics-hz.")
    parser.add_argument("--physics-hz", type=int, default=ReplaySettings.physics_hz)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--output", type=Path, default=SWEEP_DIR / "servo_model_fit.json")
    args = parser.parse_args()
    if args.physics_hz % args.command_hz:
        parser.error("--command-hz must divide --physics-hz")
    return args


def main() -> int:
    args = parse_args()
    settings = ReplaySettings(
        physics_hz=args.physics_hz,
        command_hz=args.command_hz,
        hold_s=args.hold_s,
        offsets_deg=tuple(float(x) for x in joint_offsets_deg()),
    )
    sweeps = load_sweeps(args.sweep_dir, args.joints)

    # Step 1: servo parameters from the upstream MuJoCo files.
    upstream = {
        UPSTREAM_MJCF.name: {j: servo_params_from_mjcf(UPSTREAM_MJCF, j) for j in args.joints},
        UPSTREAM_DEFAULTS.name: {j: servo_params_from_defaults_file(UPSTREAM_DEFAULTS, SERVO_CLASS) for j in args.joints},
    }
    print("Step 1: servo parameters read from the upstream MuJoCo files")
    for name, per_joint in upstream.items():
        for joint, servo in per_joint.items():
            print_servo_params(f"{name} {joint}", servo)
    print(f"  joint offsets (deg, URDF = motor + offset): {list(settings.offsets_deg)}")

    # Steps 2 and 3 run as one batch so all replays share the worker pool.
    # Step 3 keeps kv, max torque, damping and armature from so101_new_calib.xml.
    tasks = []
    for sweep in sweeps:
        for name in upstream:
            tasks.append((sweep, upstream[name][sweep.joint], settings))
        base = upstream[UPSTREAM_MJCF.name][sweep.joint]
        for kp in KP_GRID:
            for friction in FRICTION_GRID:
                tasks.append((sweep, replace(base, kp=float(kp), friction=float(friction)), settings))
    print(f"\nReplaying {len(tasks)} sweeps in PyBullet on {args.workers} workers...")
    results = iter(replay_many(tasks, args.workers))

    p.connect(p.DIRECT)  # for the gravity torques in the report
    reports = []
    print("\nSteps 2 and 3: stopping error (reading - target) RMSE in degrees, over the scored readings")
    for sweep in sweeps:
        upstream_results = {name: next(results) for name in upstream}
        grid_results = [next(results) for _ in range(len(KP_GRID) * len(FRICTION_GRID))]
        report = evaluate_joint(sweep, upstream_results, grid_results, settings)
        print_report(report, sweep)
        reports.append(report)
    p.disconnect()

    output = {
        "settings": asdict(settings),
        "kp_grid": KP_GRID.tolist(),
        "friction_grid": FRICTION_GRID.tolist(),
        "upstream_servo_params": {name: {j: asdict(s) for j, s in per.items()} for name, per in upstream.items()},
        "joints": [asdict(r) for r in reports],
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nsaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
