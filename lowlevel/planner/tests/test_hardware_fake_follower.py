"""Offline checks for ``planner.hardware`` using a fake follower (no serial port, no lerobot).

Run:  python lowlevel/planner/tests/test_hardware_fake_follower.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np

LOWLEVEL_DIR = Path(__file__).resolve().parents[2]
if str(LOWLEVEL_DIR) not in sys.path:
    sys.path.insert(0, str(LOWLEVEL_DIR))

from planner import hardware as hw  # noqa: E402
from planner import JointCompensator  # noqa: E402


class FakeBus:
    def __init__(self) -> None:
        self.port_handler = type("PH", (), {"is_using": False, "ser": None})()
        self.torque_disabled = False

    def disable_torque(self, num_retry: int = 0) -> None:
        self.torque_disabled = True


class FakeFollower:
    """Servo that reaches each command instantly; can inject read failures or a send crash."""

    def __init__(self, start, *, fail_read_at=(), crash_send_at=None) -> None:
        self.pos = np.asarray(start, dtype=float)
        self.bus = FakeBus()
        self.sent: list[dict] = []
        self.reads = 0
        self.fail_read_at = set(fail_read_at)
        self.crash_send_at = crash_send_at
        self.disconnected = False
        self.config = type("Cfg", (), {"disable_torque_on_disconnect": True})()

    def get_observation(self):
        self.reads += 1
        if self.reads in self.fail_read_at:
            raise ConnectionError("no status packet")
        return {k: float(v) for k, v in zip(hw.JOINT_KEYS, self.pos)}

    def send_action(self, action):
        if self.crash_send_at is not None and len(self.sent) == self.crash_send_at:
            raise RuntimeError("bus error")
        assert set(action) == set(hw.JOINT_KEYS), action.keys()
        self.sent.append(action)
        self.pos = np.array([action[k] for k in hw.JOINT_KEYS])

    def disconnect(self):
        self.disconnected = True


def test_pacing_and_keys():
    start = np.array([0.0, -100.0, 95.0, 60.0, 0.0, 5.0])
    cmds = np.linspace(start, start + np.array([20, 10, -10, 5, 30, 20.0]), 481)  # 2 s at 240 Hz
    follower = FakeFollower(start)
    now = [0.0]
    slept = []

    def clock():
        return now[0]

    def sleep(dt):
        slept.append(dt)
        now[0] += dt

    log = hw.stream_commands(follower, cmds, clock=clock, sleep=sleep)
    expected_steps = len(np.arange(0, 481, 4))
    assert len(follower.sent) in (expected_steps, expected_steps + 1), len(follower.sent)
    np.testing.assert_allclose(
        [follower.sent[-1][k] for k in hw.JOINT_KEYS], cmds[-1], atol=1e-9
    )
    assert abs(log.duration_s - (len(follower.sent)) / 60.0) < 1e-6, log.duration_s  # speed unchanged
    assert log.stride == 4 and abs(log.step_dt_s - 1 / 60) < 1e-9
    assert all(k.endswith(".pos") for k in follower.sent[0])
    print("pacing/keys OK:", len(follower.sent), "sends,", f"{log.duration_s:.3f}s")


def test_read_retry():
    start = np.zeros(6)
    follower = FakeFollower(start, fail_read_at={1, 2})
    obs = hw.read_observation(follower)
    assert follower.reads == 3 and "gripper.pos" in obs
    follower = FakeFollower(start, fail_read_at={1, 2, 3, 4})
    try:
        hw.read_observation(follower)
    except ConnectionError:
        pass
    else:
        raise AssertionError("expected ConnectionError after retries exhausted")
    print("read retry OK")


def test_start_guard():
    follower = FakeFollower(np.zeros(6))
    cmds = np.tile(np.array([30.0, 0, 0, 0, 0, 0]), (50, 1))
    try:
        hw.stream_commands(follower, cmds, sleep=lambda dt: None)
    except RuntimeError as exc:
        assert "moved since planning" in str(exc), exc
    else:
        raise AssertionError("start guard did not trigger")
    assert not follower.sent
    print("start guard OK")


def test_hold_on_error_and_finish():
    start = np.zeros(6)
    cmds = np.tile(start, (100, 1))
    cmds[:, 0] = np.linspace(0, 5, 100)
    # crash mid-stream: torque must stay ON until the operator presses Enter
    follower = FakeFollower(start, crash_send_at=5)
    with tempfile.TemporaryDirectory() as tmp:
        presses = []
        orig = hw.release_arm

        def release_with_enter(f, **kw):
            orig(f, input_fn=lambda: presses.append(1), **kw)
            assert presses, "operator prompt was not shown"

        hw_release = hw.release_arm
        hw.release_arm = release_with_enter
        try:
            try:
                hw.execute_commands(follower, cmds, run_dir=Path(tmp))
            except RuntimeError as exc:
                assert "bus error" in str(exc)
            else:
                raise AssertionError("expected crash")
        finally:
            hw.release_arm = hw_release
        assert follower.disconnected and follower.bus.torque_disabled  # only after Enter
        saved = (Path(tmp) / "planned_move.json").read_text()
        assert '"finished": false' in saved and "bus error" in saved
    # Ctrl-C at the prompt: torque stays on
    follower = FakeFollower(start)
    def ctrl_c():
        raise KeyboardInterrupt
    hw.release_arm(follower, finished=False, input_fn=ctrl_c)
    assert follower.disconnected and not follower.bus.torque_disabled
    # clean finish: torque released
    follower = FakeFollower(start)
    hw.release_arm(follower, finished=True)
    assert follower.bus.torque_disabled
    print("hold-on-error / release OK")


def test_start_urdf_offset():
    comp = JointCompensator()
    reading = np.array([5.0, -90.0, 90.0, 50.0, 10.0, 6.0])
    urdf = hw.start_urdf_from_reading(reading, comp)
    motor0 = comp.compensate_trajectory_rest_waypoints(np.tile(urdf, (3, 1)), [], offset_only=True)[0]
    np.testing.assert_allclose(motor0, reading, atol=1e-9)
    np.testing.assert_allclose(hw.start_urdf_from_reading(reading, None), reading)
    print("start pose offset round-trip OK")


if __name__ == "__main__":
    test_pacing_and_keys()
    test_read_retry()
    test_start_guard()
    test_hold_on_error_and_finish()
    test_start_urdf_offset()
    print("ALL OK")
