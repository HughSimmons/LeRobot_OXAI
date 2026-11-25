"""Offline check of ``planner.click_bridge.pick_piece_planned`` on synthetic click records.

Needs Oliver's ``move_to_click.py`` (from ``oliver/two-camera-capture``): point ``OLIVER_CLICK_DIR`` at
a folder containing it (default ``lowlevel/click_to_3d_oliver``). His ``click3d_shared.py`` imports cv2 and
the camera stack, so this test substitutes a stub with the same constants and pure helpers. No serial
port is opened: the follower is a fake.

Run:  python lowlevel/planner/tests/test_click_bridge_synthetic.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types

import numpy as np

HERE = Path(__file__).resolve().parent
LOWLEVEL_DIR = HERE.parents[1]
for p in (str(LOWLEVEL_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

OLIVER_DIR = Path(os.environ.get("OLIVER_CLICK_DIR", LOWLEVEL_DIR / "click_to_3d_oliver"))
SQUARE_MM = 41.25
CENTRE = (292.6, 0.0, 29.0)


def install_stub_shared() -> types.ModuleType:
    stub = types.ModuleType("click3d_shared")
    stub.SQUARE_MM = SQUARE_MM
    stub.BOARD_CENTRE_BASE_MM = CENTRE
    stub.BOARD_BASE_SOURCE = "board_centre_ruler"
    stub.CV_BASE_SOURCE = "camera_cv"
    stub.LEGACY_FK_BASE_SOURCE = "camera_fk"
    stub.OXAI_PATHS_HINT = "(stub)"
    stub._ensure_lowlevel_on_path = lambda: LOWLEVEL_DIR
    stub.square_file_rank = lambda sq: ("abcdefgh".index(sq[0]), int(sq[1]) - 1)
    stub.close_arm = lambda: None

    def board_to_base_mm(point):
        point = np.asarray(point, dtype=float)
        half = 4.0 * SQUARE_MM
        return np.array([CENTRE[0] - half + point[0], CENTRE[1] - half + point[1], CENTRE[2] + point[2]])

    stub.board_to_base_mm = board_to_base_mm
    sys.modules["click3d_shared"] = stub
    if str(OLIVER_DIR) not in sys.path:
        sys.path.insert(0, str(OLIVER_DIR))
    return stub


def click(square: str, *, cv_T=None, z_mm: float = CENTRE[2] + 22.0, board_xy=None, source="board_centre_ruler"):
    f, r = "abcdefgh".index(square[0]), int(square[1]) - 1
    bx, by = board_xy if board_xy is not None else ((f + 0.5) * SQUARE_MM + 0.7, (r + 0.5) * SQUARE_MM - 0.4)
    half = 4.0 * SQUARE_MM
    rec = {
        "square": square,
        "board_mm": [bx, by, 0.0],
        "base_mm": [CENTRE[0] - half + bx, CENTRE[1] - half + by, z_mm],
        "base_source": source,
    }
    if cv_T is not None:
        rec["T_base_board_cv_mm"] = cv_T.tolist()
    return rec


def cv_pose(yaw_deg: float = 1.2, dx: float = 1.5, dy: float = -2.0) -> np.ndarray:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    half = 4.0 * SQUARE_MM
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = [CENTRE[0] - half + dx, CENTRE[1] - half + dy, CENTRE[2]]
    return T


def main() -> None:
    install_stub_shared()
    from planner import _paths_compat as pcompat
    from planner import click_bridge as cb
    from planner import hardware as hw
    from test_hardware_fake_follower import FakeFollower

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pcompat.real_world_runs_dir = lambda: tmp_path  # type: ignore[assignment]

        # --- dry runs, both routes, all compensation modes ---------------------------------
        T = cv_pose()
        cases = [
            ("ruler/off", click("e2"), click("e4"), dict(compensate=False, baseline=False)),
            ("ruler/baseline", click("d2"), click("d4"), dict(compensate=False, baseline=True)),
            ("ruler/on", click("c2"), click("f5"), dict(compensate=True, baseline=False)),
            (
                "cv/off",
                click("e2", cv_T=T, source="camera_cv"),
                click("e4", cv_T=T, source="camera_cv"),
                dict(compensate=False, baseline=False),
            ),
        ]
        for label, a, b, flags in cases:
            res = cb.pick_piece_planned(a, b, robot=False, **flags)
            assert not res.arm_moved and "dry run ok" in res.message, (label, res.message)
            meta = json.loads((res.output_dir / "metadata.json").read_text())
            assert meta["planner"] == "dls_nullspace" and meta["executed"] is False
            for key in ("from_click", "to_click", "from_click_centre_error_mm", "target_source", "compensation_type",
                        "calib_offset", "board_centre_base_mm", "gripper_model", "from_target_base_xy_mm"):
                assert key in meta, (label, key)
            assert (res.output_dir / "plan_preview.json").is_file()
            print(f"dry run OK [{label}] -> {res.output_dir.name}")

        # --- refusals ---------------------------------------------------------------------
        refusals = [
            ("destination off", click("e2"), {**click("e4"), "square": "off"}),
            (
                "cv pose maps destination off board",
                click("e2", cv_T=cv_pose(dx=500.0), source="camera_cv"),
                click("e4", cv_T=cv_pose(dx=500.0), source="camera_cv"),
            ),
            ("piece click off board", click("e2", board_xy=(900.0, 100.0)), click("e4")),
            ("piece below surface", click("e2", z_mm=CENTRE[2] - 20.0), click("e4")),
            ("piece too high", click("e2", z_mm=CENTRE[2] + 400.0), click("e4")),
            ("cv without pose", click("e2", source="camera_cv"), click("e4", source="camera_cv")),
        ]
        for label, a, b in refusals:
            res = cb.pick_piece_planned(a, b, robot=False, compensate=False, baseline=False)
            assert not res.arm_moved and "refused" in res.message, (label, res.message)
            print(f"refusal OK [{label}]: {res.message[:90]}")

        # --- robot path with a fake follower, starting from his parked view pose -----------
        parked = np.array([2.10989, -62.28571, 45.89011, 79.60440, -81.09890, 2.84392])
        follower = FakeFollower(parked)
        hw.connect_follower = lambda **kw: follower  # type: ignore[assignment]
        moved_flag = []
        res = cb.pick_piece_planned(
            click("e2"), click("e4"), compensate=False, baseline=True, robot=True,
            on_move=lambda: moved_flag.append(True),
        )
        assert res.arm_moved and moved_flag and "done" in res.message, res.message
        assert follower.sent, "no commands were sent"
        first = np.array([follower.sent[0][k] for k in hw.JOINT_KEYS])
        assert np.max(np.abs(first[:5] - parked[:5])) < 1.0, f"first command jumps {np.max(np.abs(first[:5] - parked[:5])):.2f} deg"
        assert follower.bus.torque_disabled and follower.disconnected  # clean finish releases the arm
        meta = json.loads((res.output_dir / "metadata.json").read_text())
        assert meta["executed"] and meta["finished"]
        saved = json.loads((res.output_dir / "planned_move.json").read_text())
        assert saved["finished"] and saved["stream"]["behind_schedule_steps"] <= 2
        end = np.array([follower.sent[-1][k] for k in hw.JOINT_KEYS])
        print(
            f"robot path OK: {len(follower.sent)} sends, first-command jump "
            f"{np.max(np.abs(first[:5] - parked[:5])):.3f} deg, final cmd {np.round(end, 1).tolist()}"
        )
    print("ALL OK")


if __name__ == "__main__":
    main()
