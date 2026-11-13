#!/usr/bin/env python3
"""Drive the arm through the planned wrist-camera views and take a raw photo at each.

Reads output/wrist_pose_plan.json (plan_wrist_poses.py). Without --execute it is a dry run:
nothing connects, and every route from rest through each view and back is rebuilt and
checked by forward kinematics (see plan_wrist_poses.py for what is checked). With --execute:

- the wrist camera opens first; if it doesn't, the arm is never touched;
- the arm connects without lerobot's configure step, so torque is not switched on or off;
  every Goal_Position is set to the present position before torque goes on, so the arm can't
  jump to an old target. A limp arm must start at rest (DEFAULT_HOME, within 12 deg);
- the gripper must be nearly closed (the clearance checks assume it) and is never moved;
- before every move the route is rebuilt from the last command, checked again, and shown,
  and nothing moves until you press Enter (s skips a view, q goes back to rest);
- moves are slow (12 deg/s by default) and go up to hub height, swing, then down;
- if any joint falls more than --max-lag-deg behind its command (something in the way), or on
  Ctrl-C or any error, the arm stops where it is and keeps holding (torque stays on). Then
  support it, check the camera mount, and run --return-home to bring it back to rest.

Photos go to output/wrist_photos_<time>/: shot_NNN.png (raw frame) and capture.json (view,
commands, joint readings, the planned square outline). Mark the corners afterwards with
  ~/venvs/click3d/bin/python label_photos.py --camera wrist --images output/wrist_photos_<time>

Run from this folder with the click3d venv:
  ~/venvs/click3d/bin/python collect_wrist_photos.py                  # dry run
  ~/venvs/click3d/bin/python collect_wrist_photos.py --execute        # moves the arm
  ~/venvs/click3d/bin/python collect_wrist_photos.py --return-home --execute
  ~/venvs/click3d/bin/python collect_wrist_photos.py --release --execute    # torque off, nothing else
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import plan_wrist_poses as plan_lib
from plan_wrist_poses import HOME, JOINT_NAMES, OUTPUT, Arm, check_route, near_home, pose_problems, route

MOTOR_KEYS = tuple(f"{name}.pos" for name in JOINT_NAMES)
MAX_GRIPPER_READING = 15.0


class Abort(Exception):
    pass


def paced(waypoints: list[np.ndarray], max_step_deg: float) -> list[np.ndarray]:
    """Linear steps between waypoints, no arm joint moving more than max_step_deg per step."""
    steps: list[np.ndarray] = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        n = max(1, int(np.ceil(float(np.max(np.abs(b - a))) / max_step_deg)))
        steps.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
    return steps


def describe(check: dict, steps: int, delay: float) -> str:
    slack = check["min_clearance_slack_mm"]
    lowest = min(slack, key=slack.get)
    return (f"{steps} steps, at least {steps * delay:.0f} s; closest: {lowest} {slack[lowest]:.0f} mm above its "
            f"required height; joint travel {check["largest_joint_travel_deg"]:.0f} deg")


class Collector:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        self.arm = Arm()
        if np.max(np.abs(np.asarray(self.plan["joint_offsets_deg"]) - self.arm.offsets)) > 1e-3:
            raise SystemExit("The friction fit's joint offsets changed since the plan was made: rerun plan_wrist_poses.py")
        wanted = set(args.views) if args.views else None
        usable = [v for v in self.plan["views"] if v["safe"] and v.get("grid_ok", True)]
        self.views = [v for v in usable if wanted is None or v["view"] in wanted]
        skipped = [v["view"] for v in self.plan["views"] if v not in usable]
        if skipped:
            print(f"plan marks views {skipped} unsafe or without a clear grid; they are left out")
        self.max_step = args.max_speed_deg_s * args.command_delay
        self.follower = None
        self.cap = None
        self.gripper = None
        self.command = None  # last arm command sent (5 joints)

    # --- planning --------------------------------------------------------------------

    def checked_route(self, start: np.ndarray, end: np.ndarray) -> tuple[list[np.ndarray], dict]:
        waypoints = route(start, end)
        check = check_route(self.arm, waypoints)
        return waypoints, check

    def dry_run(self) -> int:
        print(f"dry run: {len(self.views)} views, assuming the arm starts at rest {np.round(HOME, 1).tolist()}")
        current = HOME.copy()
        total = 0
        bad = 0
        for view in self.views:
            target = np.asarray(view["joints_reading_deg"][:5])
            waypoints, check = self.checked_route(current, target)
            steps = len(paced(waypoints, self.max_step))
            total += steps
            flag = "ok" if check["ok"] else "FAILS: " + "; ".join(check["problems"])
            bad += not check["ok"]
            print(f"  view {view['view']:2d} {view['hint']:46s} {describe(check, steps, self.args.command_delay)}  {flag}")
            current = target
        waypoints, check = self.checked_route(current, HOME)
        steps = len(paced(waypoints, self.max_step))
        total += steps
        bad += not check["ok"]
        print(f"  back to rest {'':47s} {describe(check, steps, self.args.command_delay)}  "
              f"{'ok' if check['ok'] else 'FAILS: ' + '; '.join(check['problems'])}")
        print(f"moving time at least {total * self.args.command_delay / 60:.1f} min plus settling; "
              f"{'all routes pass' if not bad else f'{bad} routes FAIL'}")
        return 0 if not bad else 1

    # --- hardware --------------------------------------------------------------------

    def open_camera(self) -> None:
        from click3d_shared import load_session_cameras, open_camera

        device = load_session_cameras(self.args.device, None)[0]
        self.cap = open_camera(device, "wrist")
        frame = self.grab()
        print(f"wrist camera {device}: {frame.shape[1]}x{frame.shape[0]}")
        self.device = device

    def grab(self) -> np.ndarray:
        frame = None
        for _ in range(6):  # drop buffered frames, keep the newest
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError("wrist camera read failed")
        return frame

    def reading(self) -> np.ndarray:
        observation = self.follower.get_observation()
        return np.array([float(observation[key]) for key in MOTOR_KEYS])

    def connect(self) -> None:
        from paths import connect_so101_hardware

        # Port only: no configure(), so torque stays as it is.
        self.follower = connect_so101_hardware(calibrate=False, configure=False)
        present = self.reading()
        self.gripper = float(present[5])
        held = any(bool(v) for v in self.follower.bus.sync_read("Torque_Enable").values())
        print(f"arm connected; torque {'ON' if held else 'off'}; reading {np.round(present, 1).tolist()}")
        if self.gripper > MAX_GRIPPER_READING:
            raise Abort(f"gripper reads {self.gripper:.1f}: close it (below {MAX_GRIPPER_READING:.0f}) first; "
                        "the clearance checks assume a closed jaw")
        problems = pose_problems(self.arm, present[:5])
        if not held and not near_home(present):
            raise Abort("the arm is limp and not at rest: put it at rest (folded beside the base) and run again")
        if held and problems and not near_home(present):
            raise Abort(f"the arm is held at a pose that fails the checks ({'; '.join(problems)}), so no route is "
                        f"planned from it. Support the arm, run --release, and put it at rest by hand")
        if held:
            # Carry on from the goal it is holding; rewriting it to the present (sagged) reading
            # would let it droop a little further.
            goal = self.follower.bus.sync_read("Goal_Position")
            self.command = np.array([float(goal[name]) for name in JOINT_NAMES[:5]])
            return
        # Hold exactly where it is, so torque coming on can't make it jump to an old goal.
        self.follower.bus.sync_write("Goal_Position", self.follower.bus.sync_read("Present_Position"))
        input("Torque goes on at the present pose (servo settings are written first). Press Enter...")
        self.follower.configure()  # ends with torque on, holding the goal just written
        self.command = present[:5].copy()

    def hold_here(self, reason: str) -> None:
        """Stop: hold the present pose with torque on, and close the port without releasing."""
        print(f"\nSTOPPING: {reason}")
        if self.follower is None:
            return
        try:
            self.follower.bus.sync_write("Goal_Position", self.follower.bus.sync_read("Present_Position"))
            print(f"holding at {np.round(self.reading(), 1).tolist()} (torque stays ON)")
        finally:
            self.follower.bus.disconnect(disable_torque=False)
            self.follower = None
        print("Support the arm and check the camera mount. To bring it back to rest:\n"
              f"  ~/venvs/click3d/bin/python {Path(__file__).name} --return-home --execute\n"
              "or, if that refuses, support it, release it, and put it at rest by hand:\n"
              f"  ~/venvs/click3d/bin/python {Path(__file__).name} --release --execute")

    def release_port(self, reason: str) -> None:
        """Nothing moved: report why and close the port, leaving torque as it was."""
        print(f"\nNOT STARTED: {reason}")
        if self.follower is not None:
            self.follower.bus.disconnect(disable_torque=False)
            self.follower = None

    def move(self, target: np.ndarray, label: str, home: bool = False) -> bool:
        """Route from the last command to target, after checks and Enter. False if skipped.

        home: the move back to rest, which can't be skipped: s or q there stops and holds.
        """
        waypoints, check = self.checked_route(self.command, target)
        steps = paced(waypoints, self.max_step)
        print(f"\n{label}\n  route: {describe(check, len(steps), self.args.command_delay)}")
        if not check["ok"]:
            print("  route FAILS the checks, not moving: " + "; ".join(check["problems"]))
            return False
        if home:
            answer = input("  Enter = move back to rest, h = stop and hold here: ").strip().lower()
            if answer:
                raise Abort("stopped before the move back to rest")
        else:
            answer = input("  Enter = move, s = skip, q = back to rest and stop: ").strip().lower()
            if answer == "s":
                return False
            if answer == "q":
                raise KeyboardInterrupt("q")
        for index, step in enumerate(steps, start=1):
            action = {key: float(v) for key, v in zip(MOTOR_KEYS[:5], step)}
            action[MOTOR_KEYS[5]] = self.gripper
            self.follower.send_action(action)
            self.command = step
            time.sleep(self.args.command_delay)
            lag = np.abs(self.reading()[:5] - step)
            if float(lag.max()) > self.args.max_lag_deg:
                joint = JOINT_NAMES[int(np.argmax(lag))]
                raise Abort(f"{joint} is {lag.max():.1f} deg behind its command (limit {self.args.max_lag_deg}): "
                            "something may be in the way")
            if index % 200 == 0:
                print(f"  {index}/{len(steps)}")
        time.sleep(self.args.settle_s)
        off = np.abs(self.reading()[:5] - target)
        if float(off.max()) > self.args.max_lag_deg:
            raise Abort(f"{JOINT_NAMES[int(np.argmax(off))]} settled {off.max():.1f} deg from its target")
        return True

    def collect(self) -> int:
        self.open_camera()
        try:
            self.connect()
        except BaseException as exc:
            self.release_port(str(exc) or type(exc).__name__)
            self.cap.release()
            return 1
        out_dir = OUTPUT / f"wrist_photos_{datetime.now():%Y%m%d_%H%M%S}"
        out_dir.mkdir(parents=True)
        record = {
            "camera": "wrist",
            "device": self.device,
            "square": self.plan["square"],
            "square_mm": self.plan["square_mm"],
            "plan": str(Path(self.args.plan).resolve()),
            "joint_offsets_deg": self.plan["joint_offsets_deg"],
            "note": "Raw frames. joints are motor readings; command = what was sent.",
            "shots": [],
        }
        try:
            for n, view in enumerate(self.views, start=1):
                target = np.asarray(view["joints_reading_deg"][:5])
                if not self.move(target, f"view {view['view']} ({n}/{len(self.views)}): {view['hint']}"):
                    continue
                frame = self.grab()
                reading = self.reading()
                name = f"shot_{len(record['shots']) + 1:03d}.png"
                cv2.imwrite(str(out_dir / name), frame)
                record["shots"].append({
                    "image": name,
                    "plan_view": view["view"],
                    "hint": view["hint"],
                    "square": self.plan["square"],
                    "command_deg": [round(float(v), 3) for v in self.command] + [self.gripper],
                    "joints_reading_deg": [round(float(v), 3) for v in reading],
                    "planned_square_corners_px": view["square_corners_px"],
                    "time": datetime.now().isoformat(timespec="seconds"),
                })
                (out_dir / "capture.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
                print(f"  saved {name}")
            self.go_home()
        except BaseException as exc:  # Abort, Ctrl-C, serial errors, anything: stop and hold
            if not (isinstance(exc, KeyboardInterrupt) and str(exc) == "q"):
                self.hold_here(str(exc) or type(exc).__name__)
                return 1
            try:
                self.go_home()
            except BaseException as again:
                self.hold_here(str(again) or type(again).__name__)
                return 1
        finally:
            if self.cap is not None:
                self.cap.release()
        print(f"\n{len(record['shots'])} photos in {out_dir}")
        return 0

    def go_home(self) -> None:
        if near_home(self.reading(), tol_deg=5.0):
            print("\nalready at rest, not moving")
        elif not self.move(HOME.copy(), "back to rest", home=True):
            raise Abort("the route back to rest fails the checks")
        input("At rest. Press Enter to release torque (the arm rests folded)...")
        self.follower.bus.disable_torque()
        self.follower.bus.disconnect(disable_torque=False)
        self.follower = None

    def return_home(self) -> int:
        try:
            self.connect()
        except BaseException as exc:
            self.release_port(str(exc) or type(exc).__name__)
            return 1
        try:
            self.go_home()
        except BaseException as exc:
            self.hold_here(str(exc) or type(exc).__name__)
            return 1
        return 0


def release() -> int:
    """Switch torque off, for an arm the user is supporting, without moving it."""
    from paths import connect_so101_hardware

    follower = connect_so101_hardware(calibrate=False, configure=False)
    try:
        input("Support the arm with your hand: torque goes OFF and it will drop. Press Enter to release...")
        follower.bus.disable_torque()
        print("released")
    finally:
        follower.bus.disconnect(disable_torque=False)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", default=str(plan_lib.DEFAULT_PLAN))
    parser.add_argument("--views", type=lambda s: [int(v) for v in s.split(",")], default=None,
                        help="only these planned views, e.g. 1,2,6 (default: all safe ones)")
    parser.add_argument("--execute", action="store_true", help="move the real arm (default: dry run)")
    parser.add_argument("--return-home", action="store_true", help="bring a held arm back to rest, then release")
    parser.add_argument("--release", action="store_true", help="switch torque off (support the arm first), nothing else")
    parser.add_argument("--device", default=None, help="wrist camera index or path (default: click tool's saved one)")
    parser.add_argument("--max-speed-deg-s", type=float, default=12.0)
    parser.add_argument("--command-delay", type=float, default=0.01)
    parser.add_argument("--settle-s", type=float, default=1.5)
    parser.add_argument("--max-lag-deg", type=float, default=10.0,
                        help="stop if any joint is this far behind its command")
    args = parser.parse_args()
    if not 0 < args.max_speed_deg_s <= 20:
        parser.error("--max-speed-deg-s must be in (0, 20]")
    if args.device is not None and args.device.isdigit():
        args.device = int(args.device)
    if args.release:
        if not args.execute:
            parser.error("--release needs --execute")
        return release()
    collector = Collector(args)
    if not args.execute:
        return collector.dry_run()
    from click3d_shared import hide_jpeg_warnings

    hide_jpeg_warnings()
    return collector.return_home() if args.return_home else collector.collect()


if __name__ == "__main__":
    raise SystemExit(main())
