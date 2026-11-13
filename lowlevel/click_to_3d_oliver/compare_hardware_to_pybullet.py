"""Replay a real-arm run's commands in PyBullet on the planned schedule and compare with the hardware.

Reads a run folder written by run_real_so101_from_xy_lookup.py --execute:
  timeline_sim.json       the commands and the times they were meant to be sent
                          (one every command_delay_s = 1/240 s)
  timeline_hardware.json  what the real arm did, with its measured read times

PyBullet steps once per command_delay_s (setTimeStep). Before each step the joints are given the
command whose ideal send time has been reached; after it they are read. So command k is read at
(k + 1) * command_delay_s, its t_read_end_s in timeline_sim.json. After the last command the sim
holds it until the hardware's last reading, as the real arm held its last command.

The robot is set up like the trajectory-finding sim (multisim_chess_fast.py): paths.URDF_PATH,
gravity, its solver iterations and substeps, POSITION_CONTROL with force 50 on the arm joints and
its default gripper force. No board or piece is loaded, and the sim has none of the joint offset or
friction draws, so its joints are ideal apart from PyBullet's own dynamics.

The sim is driven with target_deg (the planned URDF angles), or with --drive command, command_deg
(the motor commands actually sent, turned into URDF angles with the run's urdf_offsets_deg). In a
--compensate run the commands include the sag correction, so a sim driven with them, having no
sag, settles past the target by that correction.

Compared (hardware reading_urdf_deg minus sim joint angle, URDF degrees, arm joints only):
  same time     the sim interpolated to each hardware read time (t_read_end_s)
  same command  the sim's reading of command i against the hardware's reading of command i
The gripper is replayed but left out of the comparison: the real gripper's position is LeRobot's
0-100 range, not degrees, while the sim, like multisim_chess_fast.py, treats it as degrees.

Joint velocities (deg/s) are compared the same two ways. Both the sim's and the hardware's come
from their angles by a central difference, (q[i+k] - q[i-k]) / (t[i+k] - t[i-k]) with
k = --velocity-half-window (default 5), each against its own clock: the sim's 4.17 ms steps, the
hardware's measured read times. A one-step difference is mostly noise on the hardware: a reading
moves in 0.088 deg encoder steps and the reads are unevenly spaced, so over one ~7 ms step a
steady 30 deg/s move reads anywhere from 0 to over 100 deg/s. Over +/-5 samples (about 70 ms on the
hardware) one encoder step is about 1 deg/s. The first and last k samples have no velocity (null).
The sim's own joint velocity from PyBullet is also saved, but the comparison uses the difference,
so both sides are smoothed alike.

Usage (IKsim_mj env, from anywhere):
  python lowlevel/click_to_3d_oliver/compare_hardware_to_pybullet.py lowlevel/real_world_runs/click_picks/<run>
Writes <run>/pybullet_replay.json (or --output).
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p

LOWLEVEL_DIR = Path(__file__).resolve().parent.parent  # lowlevel/, which holds paths.py
if str(LOWLEVEL_DIR) not in sys.path:
    sys.path.insert(0, str(LOWLEVEL_DIR))

import paths  # noqa: E402

TIMELINE_SCHEMA = "so101_timeline_v1"
REPLAY_SCHEMA = "pybullet_replay_v1"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM = slice(0, 5)  # the joints compared; the gripper's units differ (see module docstring)
# PyBullet joint indices in trajectory order, as multisim_chess_fast.py's CONTROL_JOINTS
# (index 5 is the fixed gripper_frame_joint).
CONTROL_JOINTS = [0, 1, 2, 3, 4, 6]
# Arm motor force, the literal multisim_chess_fast.py's run loop uses for joints 0-4.
ARM_FORCE = 50.0


def multisim_constant(name: str) -> Any:
    """A module-level constant of multisim_chess_fast.py, read without importing it (slow)."""
    source = (paths.LOWLEVEL_DIR / "multisim_chess_fast.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"No {name} in multisim_chess_fast.py")


def load_timeline(path: Path, timeline: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != TIMELINE_SCHEMA or data.get("timeline") != timeline:
        raise ValueError(f"{path} is not a {TIMELINE_SCHEMA} {timeline} timeline")
    return data


def simulate(
    drive_urdf_deg: np.ndarray,
    send_times_s: np.ndarray,
    *,
    dt: float,
    end_time_s: float,
    settle_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step PyBullet every dt, each step driving the latest command whose send time has come.

    Returns sample times (s, step n read at (n + 1) * dt), joint angles (URDF deg) and joint
    velocities (deg/s), one row per step, in trajectory joint order.
    """
    solver_iterations = int(multisim_constant("SOLVER_ITERATIONS"))
    solver_substeps = int(multisim_constant("SOLVER_SUBSTEPS"))
    gripper_force = float(multisim_constant("GRIPPER_CLOSE_FORCE"))
    forces = [ARM_FORCE] * 5 + [gripper_force]

    client = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        p.setTimeStep(dt, physicsClientId=client)
        p.setPhysicsEngineParameter(
            numSolverIterations=solver_iterations, numSubSteps=solver_substeps, physicsClientId=client
        )
        robot = p.loadURDF(str(paths.URDF_PATH), [0, 0, 0], useFixedBase=True, physicsClientId=client)

        def drive(command_deg: np.ndarray) -> None:
            for sim_idx, angle, force in zip(CONTROL_JOINTS, np.deg2rad(command_deg), forces):
                p.setJointMotorControl2(
                    robot, sim_idx, p.POSITION_CONTROL, targetPosition=float(angle), force=force,
                    physicsClientId=client,
                )

        def read() -> tuple[np.ndarray, np.ndarray]:
            #returns joint angles and joint velocities
            states = p.getJointStates(robot, CONTROL_JOINTS, physicsClientId=client)
            return np.rad2deg([s[0] for s in states]), np.rad2deg([s[1] for s in states])

        # Start at rest on the first command, as the real arm was after its start bridge.
        for sim_idx, angle in zip(CONTROL_JOINTS, np.deg2rad(drive_urdf_deg[0])):
            p.resetJointState(robot, sim_idx, float(angle), targetVelocity=0.0, physicsClientId=client)
        drive(drive_urdf_deg[0])
        for _ in range(round(settle_s / dt)):
            p.stepSimulation(physicsClientId=client)

        steps = math.ceil(end_time_s / dt - 1e-9)
        times, angles, velocities = [], [], []
        active = -1
        for n in range(steps):
            # Latest command sent by the start of this step (1e-9 s guards float rounding).
            latest = int(np.searchsorted(send_times_s, n * dt + 1e-9, side="right")) - 1
            if latest != active:
                drive(drive_urdf_deg[latest])
                active = latest
            p.stepSimulation(physicsClientId=client)
            q, qd = read()
            times.append((n + 1) * dt)
            angles.append(q)
            velocities.append(qd)
    finally:
        p.disconnect(physicsClientId=client)
    return np.array(times), np.array(angles), np.array(velocities)


def central_difference(values: np.ndarray, times: np.ndarray, half_window: int) -> np.ndarray:
    """Velocity of each joint at every sample, by a central difference.

    With k = half_window, the velocity at sample i is the slope between the samples k before and
    k after it:

        rate[i] = (values[i + k] - values[i - k]) / (times[i + k] - times[i - k])

    Taking one sample each side (rather than i and i + 1) centres the slope on sample i. Spreading
    them k samples apart averages out the encoder's 0.088 deg steps and the uneven read times,
    which swamp a one-step difference on the hardware (see the module docstring).

    Args:
        values: joint angles in degrees, shape (n_samples, n_joints): one row per sample (a sim
            step, or a hardware reading), one column per joint.
        times: when each row was sampled, in seconds, shape (n_samples,), increasing. Each series
            uses its own clock: the sim's 4.17 ms steps, or the hardware's measured read times.
        half_window: k, how many samples either side of sample i to take the slope across (at
            least 1). The slope spans 2k gaps between samples: for k = 5, about 70 ms on the
            hardware (2k x ~6.9 ms) and 42 ms in the sim (2k x 4.17 ms).

    Returns:
        Velocities in degrees per second, the same shape as values. The first and last k rows
        are NaN, since they have no sample k away on one side (every row is NaN if there are 2k
        samples or fewer).
    """
    k = half_window
    rates = np.full(values.shape, np.nan)
    if len(values) > 2 * k:
        # Row r of these slices belongs to sample i = r + k: "ahead" is sample i + k, "behind"
        # is sample i - k.
        ahead_values, behind_values = values[2 * k :], values[: -2 * k]
        ahead_times, behind_times = times[2 * k :], times[: -2 * k]
        # [:, None] turns the time gaps into a column, so each row's gap divides all its joints.
        rates[k:-k] = (ahead_values - behind_values) / (ahead_times - behind_times)[:, None]
    return rates


def interp_columns(query_times: np.ndarray, times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Each column of values (sampled at times) linearly interpolated to query_times.

    NaN samples are skipped, and a query outside the remaining samples gets NaN, not the end value.
    """
    out = np.full((len(query_times), values.shape[1]), np.nan)
    for j in range(values.shape[1]):
        known = ~np.isnan(values[:, j])
        if known.any():
            out[:, j] = np.interp(query_times, times[known], values[known, j], left=np.nan, right=np.nan)
    return out


def error_summary(error: np.ndarray) -> dict[str, dict[str, float]]:
    """Per arm joint: RMS, mean and largest absolute error, ignoring NaN (no value) rows."""
    arm = error[:, ARM]
    return {
        name: {
            "rms": float(np.sqrt(np.nanmean(arm[:, j] ** 2))),
            "mean": float(np.nanmean(arm[:, j])),
            "max_abs": float(np.nanmax(np.abs(arm[:, j]))),
        }
        for j, name in enumerate(JOINTS[ARM])
    }


def peak_speeds(sim_vel: np.ndarray, hardware_vel: np.ndarray) -> dict[str, dict[str, float]]:
    """Per arm joint: the largest |velocity| (deg/s) of the sim and of the hardware."""
    return {
        name: {
            "sim": float(np.nanmax(np.abs(sim_vel[:, j]))),
            "hardware": float(np.nanmax(np.abs(hardware_vel[:, j]))),
        }
        for j, name in enumerate(JOINTS[ARM])
    }


def json_rows(values: np.ndarray) -> list[Any]:
    """values as nested lists for JSON, NaN written as null."""
    return np.where(np.isnan(values), None, values).tolist()


def print_summary(title: str, summary: dict[str, dict[str, float]]) -> None:
    print(title)
    for name, values in summary.items():
        print(f"  {name:13s} rms {values['rms']:7.2f}  mean {values['mean']:+7.2f}  max {values['max_abs']:7.2f}")


def print_peaks(title: str, peaks: dict[str, dict[str, float]]) -> None:
    print(title)
    for name, values in peaks.items():
        print(f"  {name:13s} sim {values['sim']:7.1f}  hardware {values['hardware']:7.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="run folder with timeline_sim.json and timeline_hardware.json")
    parser.add_argument(
        "--drive",
        choices=("target", "command"),
        default="target",
        help="drive the sim with the planned target angles (default) or the commands sent",
    )
    parser.add_argument(
        "--settle-s", type=float, default=1.0, help="seconds to hold the first command before t = 0 (default 1)"
    )
    parser.add_argument(
        "--velocity-half-window",
        type=int,
        default=5,
        help="samples either side for the velocities' central difference (default 5)",
    )
    parser.add_argument("--output", type=Path, default=None, help="default: <run_dir>/pybullet_replay.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    sim_timeline = load_timeline(run_dir / "timeline_sim.json", "sim_ideal")
    hardware = load_timeline(run_dir / "timeline_hardware.json", "hardware")
    if sim_timeline["index"] != hardware["index"]:
        raise ValueError("timeline_sim.json and timeline_hardware.json cover different commands")
    if args.settle_s < 0.0:
        raise ValueError("--settle-s must be non-negative")
    if args.velocity_half_window < 1:
        raise ValueError("--velocity-half-window must be at least 1")

    dt = float(sim_timeline["command_delay_s"])
    offsets = np.asarray(sim_timeline["urdf_offsets_deg"], dtype=float)
    if args.drive == "command":
        drive_urdf = np.asarray(sim_timeline["command_deg"], dtype=float) + offsets
    else:
        drive_urdf = np.asarray(sim_timeline["target_deg"], dtype=float)
    send_times = np.asarray(sim_timeline["t_send_s"], dtype=float)
    sim_read_times = np.asarray(sim_timeline["t_read_end_s"], dtype=float)
    hw_times = np.asarray(hardware["t_read_end_s"], dtype=float)
    hw_urdf = np.asarray(hardware["reading_urdf_deg"], dtype=float)
    end_time = max(float(hw_times[-1]), float(sim_read_times[-1]))

    times, sim_deg, sim_vel = simulate(drive_urdf, send_times, dt=dt, end_time_s=end_time, settle_s=args.settle_s)

    def sim_at(query_times: np.ndarray) -> np.ndarray:
        #wot is this
        return np.stack([np.interp(query_times, times, sim_deg[:, j]) for j in range(len(JOINTS))], axis=1)

    sim_at_hw_times = sim_at(hw_times)
    sim_at_own_reads = sim_at(sim_read_times)
    same_time = hw_urdf - sim_at_hw_times
    same_command = hw_urdf - sim_at_own_reads

    # Velocities: the same central difference on both, each against its own clock.
    half = args.velocity_half_window
    sim_vel_diff = central_difference(sim_deg, times, half)
    hw_vel = central_difference(hw_urdf, hw_times, half)
    sim_vel_at_hw_times = interp_columns(hw_times, times, sim_vel_diff)
    sim_vel_at_own_reads = interp_columns(sim_read_times, times, sim_vel_diff)
    vel_same_time = hw_vel - sim_vel_at_hw_times
    vel_same_command = hw_vel - sim_vel_at_own_reads

    summary = {
        "same_time": error_summary(same_time),
        "same_command": error_summary(same_command),
        "velocity": {
            "same_time": error_summary(vel_same_time),
            "same_command": error_summary(vel_same_command),
            "peak_abs_deg_s": peak_speeds(sim_vel_at_own_reads, hw_vel),
        },
    }

    print(f"run: {run_dir}")
    print(
        f"sim driven with {args.drive}_deg, {len(times)} steps of {dt * 1e3:.3f} ms; commands end at "
        f"{sim_read_times[-1]:.2f} s on the planned schedule, the hardware's at {hw_times[-1]:.2f} s"
    )
    print_summary("same time: hardware reading_urdf - sim, deg", summary["same_time"])
    print_summary("same command: hardware reading_urdf - sim, deg", summary["same_command"])
    print(f"velocities: central difference over +/-{half} samples")
    print_summary("same time: hardware - sim velocity, deg/s", summary["velocity"]["same_time"])
    print_summary("same command: hardware - sim velocity, deg/s", summary["velocity"]["same_command"])
    print_peaks("peak |velocity| over the logged commands, deg/s", summary["velocity"]["peak_abs_deg_s"])

    output = (args.output or run_dir / "pybullet_replay.json").expanduser().resolve()
    payload = {
        "schema": REPLAY_SCHEMA,
        "run_dir": str(run_dir),
        "joints": JOINTS,
        "compared_joints": JOINTS[ARM],
        "drive": f"{args.drive}_deg",
        "settle_s": args.settle_s,
        "dt_s": dt,
        "urdf": str(paths.URDF_PATH),
        "solver_iterations": int(multisim_constant("SOLVER_ITERATIONS")),
        "solver_substeps": int(multisim_constant("SOLVER_SUBSTEPS")),
        "arm_force": ARM_FORCE,
        "gripper_force": float(multisim_constant("GRIPPER_CLOSE_FORCE")),
        "urdf_offsets_deg": offsets.tolist(),
        "velocity_half_window": half,
        "notes": (
            "error = hardware reading_urdf_deg - sim, URDF degrees; vel_error = hardware - sim velocity, "
            "deg/s, both from a central difference over +/- velocity_half_window samples on their own "
            "clocks (null where it has no value). Gripper replayed but not compared: the real gripper "
            "reads LeRobot's 0-100 range, the sim treats the value as degrees."
        ),
        "sim": {
            "t_s": times.tolist(),
            "joint_deg": sim_deg.tolist(),
            "joint_vel_deg_s": sim_vel.tolist(),
            "joint_vel_diff_deg_s": json_rows(sim_vel_diff),
        },
        "same_time": {
            "t_s": hw_times.tolist(),
            "sim_deg": sim_at_hw_times.tolist(),
            "hardware_deg": hw_urdf.tolist(),
            "error_deg": same_time.tolist(),
            "sim_vel_deg_s": json_rows(sim_vel_at_hw_times),
            "hardware_vel_deg_s": json_rows(hw_vel),
            "vel_error_deg_s": json_rows(vel_same_time),
        },
        "same_command": {
            "index": hardware["index"],
            "sim_t_s": sim_read_times.tolist(),
            "hardware_t_s": hw_times.tolist(),
            "sim_deg": sim_at_own_reads.tolist(),
            "hardware_deg": hw_urdf.tolist(),
            "error_deg": same_command.tolist(),
            "sim_vel_deg_s": json_rows(sim_vel_at_own_reads),
            "hardware_vel_deg_s": json_rows(hw_vel),
            "vel_error_deg_s": json_rows(vel_same_command),
        },
        "summary": summary,
    }
    output.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
