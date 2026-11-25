"""Real-arm execution for the ``dls_nullspace`` planner (SO-101 follower).

Everything that touches the servo bus lives here so ``click_bridge.py`` and
``run_planned_move_real.py`` share one tested path:

* connect through ``paths.connect_so101_hardware`` (works with both ``paths.py`` variants);
* read joints with the same retry-on-``ConnectionError`` pattern as ``run_real_so101_from_xy_lookup``;
* start the plan from the LIVE pose (compensation-aware, see ``start_urdf_from_reading``);
* stream the dense 240 Hz plan down-sampled to ``control_hz`` on a fixed schedule (speed is
  exactly the planner's trajectory timing; nothing here changes it);
* on error or Ctrl-C keep torque ON and wait for the operator before releasing the arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

from . import _paths_compat as pcompat
from .base import CaptureSequenceResult, PlannedTrajectory
from .compensation import JointCompensator
from .kinematics_so101 import ALL_JOINT_NAMES

COMPENSATION_MODES = ("off", "baseline", "on")
JOINT_KEYS = tuple(f"{name}.pos" for name in ALL_JOINT_NAMES)

READ_RETRIES = 3
READ_RETRY_DELAY_S = 0.02
DEFAULT_CONTROL_HZ = 60.0
DEFAULT_SOURCE_DT_S = 1.0 / 240.0
DEFAULT_MAX_START_DELTA_DEG = 6.0

SUPPORT_ARM_WARNING = (
    "SUPPORT THE ARM: connecting can turn torque off for a moment (lerobot configure()), "
    "so a loaded arm may sag. Hold it until it stiffens."
)


# --------------------------------------------------------------------------- compensation helpers
def make_compensator(mode: str) -> Optional[JointCompensator]:
    """``off`` -> None; ``baseline`` -> calibration offsets only; ``on`` -> offsets + friction/sag."""
    if mode not in COMPENSATION_MODES:
        raise ValueError(f"compensation mode must be one of {COMPENSATION_MODES}, got {mode!r}")
    if mode == "off":
        return None
    if mode == "baseline":
        return JointCompensator()
    comp = JointCompensator(use_default_fit=True)
    if not comp.has_m3_fit:
        print(
            "[compensation] no friction_sag_fit.json found "
            f"({pcompat.real_world_runs_dir() / pcompat.FRICTION_FIT_RELATIVE}); "
            "'on' uses the generic built-in M3 defaults, not a fit of this arm."
        )
    else:
        print(f"[compensation] model source: {comp.model_source}")
    return comp


def compensate_plan(
    plan_result: PlannedTrajectory | CaptureSequenceResult,
    compensator: Optional[JointCompensator],
    mode: str,
) -> np.ndarray:
    """Dense 6-DOF motor commands (deg) for ``plan_result`` under ``mode``."""
    raw = (
        plan_result.combined_commands_deg
        if isinstance(plan_result, CaptureSequenceResult)
        else plan_result.commands_deg
    )
    if compensator is None or mode == "off":
        return np.asarray(raw, dtype=np.float64).copy()
    offset_only = mode == "baseline"
    kwargs = {"use_gravity_sag": not offset_only, "offset_only": offset_only}
    if isinstance(plan_result, CaptureSequenceResult):
        c1 = compensator.compensate_trajectory_rest_waypoints(
            plan_result.leg1_clear_captured.commands_deg,
            plan_result.leg1_clear_captured.rest_command_indices,
            **kwargs,
        )
        c2 = compensator.compensate_trajectory_rest_waypoints(
            plan_result.leg2_move_attacker.commands_deg,
            plan_result.leg2_move_attacker.rest_command_indices,
            **kwargs,
        )
        return np.vstack([c1, c2[1:]])
    return compensator.compensate_trajectory_rest_waypoints(
        plan_result.commands_deg, plan_result.rest_command_indices, **kwargs
    )


def start_urdf_from_reading(reading_deg: Sequence[float], compensator: Optional[JointCompensator]) -> np.ndarray:
    """URDF-frame start pose for planning from a live motor reading.

    Compensation writes ``motor = urdf - offset``, so planning from the raw reading would make the
    FIRST command jump by the calibration offset (a few degrees). Planning from
    ``reading + offset`` makes command[0] equal the reading exactly.
    """
    reading = np.asarray(reading_deg, dtype=np.float64).reshape(6).copy()
    if compensator is None:
        return reading
    start = reading.copy()
    start[:5] = reading[:5] + compensator.offsets_deg[:5]
    return start


# --------------------------------------------------------------------------- bus helpers
def read_observation(follower, *, retries: int = READ_RETRIES, timing: dict[str, Any] | None = None) -> dict[str, Any]:
    """``follower.get_observation()``, retried when a read gets no reply (as in the lookup runner)."""
    for attempt in range(retries + 1):
        read_start = time.perf_counter()
        try:
            observation = follower.get_observation()
            if timing is not None:
                timing.update(read_start=read_start, read_end=time.perf_counter(), retries=attempt)
            return observation
        except ConnectionError as error:
            if attempt == retries:
                raise
            print(f"position read failed ({error}); retry {attempt + 1}/{retries}", flush=True)
            port_handler = getattr(getattr(follower, "bus", None), "port_handler", None)
            if port_handler is not None:
                port_handler.is_using = False
                serial_port = getattr(port_handler, "ser", None)
                if serial_port is not None:
                    serial_port.reset_input_buffer()
            time.sleep(READ_RETRY_DELAY_S)
    raise AssertionError("unreachable")


def observation_joints_deg(observation: dict[str, Any]) -> np.ndarray:
    return np.array([float(observation[key]) for key in JOINT_KEYS], dtype=np.float64)


def read_joints_deg(follower) -> np.ndarray:
    return observation_joints_deg(read_observation(follower))


def action_from_joints(joints_deg: Sequence[float]) -> dict[str, float]:
    return {key: float(v) for key, v in zip(JOINT_KEYS, np.asarray(joints_deg, dtype=np.float64))}


def connect_follower(*, configure: bool = True):
    """Connect via ``paths.connect_so101_hardware`` and keep torque on if we later fail.

    ``disable_torque_on_disconnect`` is cleared so that an error does not drop the arm; torque is
    cut by ``release_arm`` only once the operator has had the chance to support it.
    """
    print(SUPPORT_ARM_WARNING, flush=True)
    follower = pcompat.connect_follower(calibrate=True, configure=configure)
    config = getattr(follower, "config", None)
    if config is not None and hasattr(config, "disable_torque_on_disconnect"):
        config.disable_torque_on_disconnect = False
    return follower


def release_arm(follower, *, finished: bool, release_on_finish: bool = True, input_fn: Callable[[], Any] = input) -> None:
    """Close the port; cut torque only after a clean finish or once the operator pressed Enter."""
    if finished:
        release = release_on_finish
    else:
        print(
            "\nRun stopped early. The arm is HOLDING its last position (torque still on).\n"
            "Support the arm, then press Enter to release torque, or Ctrl-C to leave it holding.",
            flush=True,
        )
        try:
            input_fn()
            release = True
        except (EOFError, KeyboardInterrupt):
            release = False
            print("\nLeaving torque on. Release the arm later, e.g. by reconnecting and disabling torque.")
    try:
        if release:
            bus = getattr(follower, "bus", None)
            if bus is not None:
                if getattr(bus, "port_handler", None) is not None:
                    bus.port_handler.is_using = False
                bus.disable_torque(num_retry=5)
    except Exception as error:  # noqa: BLE001 - still close the port below
        print(f"could not disable torque: {error}")
    finally:
        follower.disconnect()


# --------------------------------------------------------------------------- streaming
@dataclass
class StreamLog:
    stride: int
    step_dt_s: float
    sent_commands_deg: list[list[float]] = field(default_factory=list)
    read_joints_deg: list[list[float]] = field(default_factory=list)
    read_retries: int = 0
    behind_schedule_steps: int = 0
    max_lag_s: float = 0.0
    duration_s: float = 0.0


def subsample_commands(commands_deg: np.ndarray, *, source_dt_s: float, control_hz: float) -> tuple[np.ndarray, int, float]:
    """Down-sample the dense plan to ``control_hz``; always keep the final command."""
    stride = max(1, int(round((1.0 / control_hz) / source_dt_s)))
    sampled = commands_deg[::stride]
    if not np.allclose(sampled[-1], commands_deg[-1]):
        sampled = np.vstack([sampled, commands_deg[-1:]])
    return sampled, stride, stride * source_dt_s


def stream_commands(
    follower,
    commands_deg: np.ndarray,
    *,
    source_dt_s: float = DEFAULT_SOURCE_DT_S,
    control_hz: float = DEFAULT_CONTROL_HZ,
    max_start_delta_deg: float = DEFAULT_MAX_START_DELTA_DEG,
    on_step: Optional[Callable[[int, int], None]] = None,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> StreamLog:
    """Stream ``commands_deg`` (N x 6) to the follower on a fixed schedule and log readings.

    Command k goes out at ``t0 + k * step_dt``; after a slow write/read the loop skips sleeping until
    it is back on schedule, so the planner's trajectory duration is preserved. Raises
    ``RuntimeError`` before sending anything if the first command is far from the live pose.
    """
    sampled, stride, step_dt = subsample_commands(commands_deg, source_dt_s=source_dt_s, control_hz=control_hz)
    log = StreamLog(stride=stride, step_dt_s=step_dt)

    observation = read_observation(follower)
    current = observation_joints_deg(observation)
    # Arm joints only: the plan opens the gripper from its first command (a deliberate step).
    first_delta = float(np.max(np.abs(sampled[0, :5] - current[:5])))
    if first_delta > max_start_delta_deg:
        raise RuntimeError(
            f"Live arm pose is {first_delta:.2f} deg (arm joints) from the first command (> {max_start_delta_deg:.2f}). "
            "The arm moved since planning; re-plan from the current pose."
        )

    t0 = clock()
    n = len(sampled)
    try:
        for idx, q_deg in enumerate(sampled):
            follower.send_action(action_from_joints(q_deg))
            log.sent_commands_deg.append([float(v) for v in q_deg])
            deadline = t0 + (idx + 1) * step_dt
            remaining = deadline - clock()
            if remaining > 0.0:
                sleep(remaining)
            else:
                log.behind_schedule_steps += 1
                log.max_lag_s = max(log.max_lag_s, -remaining)
            timing: dict[str, Any] = {}
            obs = read_observation(follower, timing=timing)
            log.read_retries += int(timing.get("retries", 0))
            log.read_joints_deg.append([float(v) for v in observation_joints_deg(obs)])
            if on_step is not None and (idx % 60 == 0 or idx == n - 1):
                on_step(idx + 1, n)
    except BaseException as exc:
        log.duration_s = clock() - t0
        exc.stream_log = log  # type: ignore[attr-defined]  # partial log for the run record
        raise
    log.duration_s = clock() - t0
    return log


# --------------------------------------------------------------------------- one-shot executor
def save_run_log(
    run_dir: Path,
    *,
    commands_deg: np.ndarray,
    log: Optional[StreamLog],
    metadata: Optional[dict[str, Any]] = None,
    finished: bool,
    error: Optional[str] = None,
    filename: str = "planned_move.json",
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "planner": "dls_nullspace",
        "finished": finished,
        "error": error,
        "n_commands": int(len(commands_deg)),
        "commands_deg": np.asarray(commands_deg, dtype=np.float64).round(4).tolist(),
        "metadata": metadata or {},
    }
    if log is not None:
        payload["stream"] = {
            "stride": log.stride,
            "step_dt_s": log.step_dt_s,
            "duration_s": log.duration_s,
            "read_retries": log.read_retries,
            "behind_schedule_steps": log.behind_schedule_steps,
            "max_lag_s": log.max_lag_s,
            "read_joints_deg": np.asarray(log.read_joints_deg).round(4).tolist(),
        }
    out = run_dir / filename
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def execute_commands(
    follower,
    commands_deg: np.ndarray,
    *,
    source_dt_s: float = DEFAULT_SOURCE_DT_S,
    control_hz: float = DEFAULT_CONTROL_HZ,
    run_dir: Optional[Path] = None,
    metadata: Optional[dict[str, Any]] = None,
    release_on_finish: bool = True,
    max_start_delta_deg: float = DEFAULT_MAX_START_DELTA_DEG,
    on_step: Optional[Callable[[int, int], None]] = None,
) -> StreamLog:
    """Stream ``commands_deg`` then release the arm (``release_arm`` semantics); always writes a log."""
    finished = False
    log: Optional[StreamLog] = None
    error: Optional[str] = None
    try:
        log = stream_commands(
            follower,
            commands_deg,
            source_dt_s=source_dt_s,
            control_hz=control_hz,
            max_start_delta_deg=max_start_delta_deg,
            on_step=on_step,
        )
        finished = True
        return log
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
        error = f"{type(exc).__name__}: {exc}"
        log = getattr(exc, "stream_log", None)
        raise
    finally:
        try:
            if run_dir is not None:
                path = save_run_log(
                    run_dir, commands_deg=commands_deg, log=log, metadata=metadata, finished=finished, error=error
                )
                print(f"Run log: {path}")
        finally:
            release_arm(follower, finished=finished, release_on_finish=release_on_finish)
