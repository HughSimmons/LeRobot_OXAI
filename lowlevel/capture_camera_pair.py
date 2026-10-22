#!/usr/bin/env python3
"""Live-preview the wrist and stationary cameras and save synchronised image pairs.

The preview uses matplotlib because the conda envs' OpenCV builds are headless
(no cv2.imshow). Run it in IKsim_mj, which has the Tk backend.

Keys (with the preview window focused):
    space   save a pair
    c       save a pair labelled "calib" (ChArUco / empty-board shot)
    q, Esc  quit

Each camera is read continuously in its own thread, so a saved pair is the two
most recent frames rather than stale buffered ones. The gap between the two
frames' capture times is logged as skew_ms in pairs.csv.

With --teleop the script also runs leader -> follower teleoperation itself
(instead of lerobot-teleoperate, which would hold the follower's serial port)
and logs the follower's measured joint positions with every pair. This MOVES
THE REAL ARM. When you quit, the follower's torque is released, so support the
arm first; the script waits for Enter before letting go.

Output, one folder per session:
    outputs/camera_pairs/<session>/
        session.json                      camera (and arm) settings
        pairs.csv                         one row per saved pair
        pair_001_shot_wrist.png
        pair_001_shot_stationary.png
        pair_001_shot_joints.json         --teleop only; fk_from_pose_json.py reads it
        ...

Cameras default to their /dev/v4l/by-id/ paths, which stay fixed whatever
/dev/videoN numbers they get; list them with `ls /dev/v4l/by-id/`.

Examples:
    python lowlevel/capture_camera_pair.py
    PYTHONPATH=lowlevel/lerobot/src python lowlevel/capture_camera_pair.py --teleop
"""

import argparse
import csv
import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from paths import REPO_ROOT

DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "camera_pairs"
CAMERA_NAMES = ("wrist", "stationary")
# Innomaker U20CAM in the SO101 wrist mount; Logitech C920 on the tripod.
DEFAULT_WRIST_CAMERA = "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0"
DEFAULT_STATIONARY_CAMERA = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_7671151F-video-index0"
# v4l2 controls set at startup (see `v4l2-ctl -d <camera> --list-ctrls`). The
# Innomaker came up with gain=100, which blew out ~70% of the wrist image.
# exposure_dynamic_framerate=0 stops the C920 dropping below 30 fps in dim light.
# The C920's focus is locked so its lens calibration stays valid; 40 was the
# sharpest setting (Laplacian variance sweep, 2026-10-22) at the tripod's
# distance to the table. Re-check it if the tripod moves.
DEFAULT_WRIST_CONTROLS = "auto_exposure=3,gain=0"
DEFAULT_STATIONARY_CONTROLS = "exposure_dynamic_framerate=0,focus_automatic_continuous=0,focus_absolute=40"

# Oliver's SO101 pair (see SO101 Setup notes); by-id paths stay fixed across replugs.
DEFAULT_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB9065986-if00"
DEFAULT_FOLLOWER_ID = "orange_follower_arm"
DEFAULT_LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AAF219942-if00"
DEFAULT_LEADER_ID = "black_leader_arm"
MOTOR_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
# Refuse to save if the newest joint reading is older than this (teleop thread died).
MAX_JOINT_AGE_S = 0.5


def parse_camera(value: str) -> int | str:
    """Accept a camera index ("0") or a device path ("/dev/video0")."""
    return int(value) if value.isdigit() else value


class CameraReader:
    """Keeps the latest frame from one camera, read in a background thread."""

    def __init__(
        self, name: str, source: int | str, width: int | None, height: int | None, fourcc: str, controls: str = ""
    ):
        self.name = name
        self.source = source
        self.device = f"/dev/video{source}" if isinstance(source, int) else source
        self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open {name} camera {source!r}")
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # One call per control, in order: in a single call the C920 rejects the
        # whole batch, because focus_absolute is read-only until autofocus is off.
        for control in filter(None, controls.split(",")):
            subprocess.run(["v4l2-ctl", "-d", self.device, "-c", control], check=True)

        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            stamp = time.time()
            if ok:
                with self._lock:
                    self._frame = frame
                    self._stamp = stamp

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return self._frame, self._stamp

    def settings(self) -> dict:
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        controls = subprocess.run(["v4l2-ctl", "-d", self.device, "--list-ctrls"], capture_output=True, text=True)
        return {
            "source": self.source,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "fourcc": "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)),
            "v4l2_controls": controls.stdout.splitlines(),
        }

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


class TeleopArm:
    """Leader -> follower teleop in a background thread, keeping the follower's latest joint reading.

    Does what lerobot-teleoperate's loop does (with its default identity
    processors), so the serial port stays in this process and joints can be read.
    """

    def __init__(self, args: argparse.Namespace, fps: float = 60.0):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        self.settings = {
            "follower_port": args.follower_port,
            "follower_id": args.follower_id,
            "leader_port": args.leader_port,
            "leader_id": args.leader_id,
            "max_relative_target": args.max_relative_target,
        }
        self.leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id))
        self.follower = SO101Follower(
            SO101FollowerConfig(
                port=args.follower_port,
                id=args.follower_id,
                max_relative_target=args.max_relative_target,
            )
        )
        self.leader.connect()
        try:
            self.follower.connect()
        except Exception:
            self.leader.disconnect()
            raise

        self.period_s = 1.0 / fps
        self.error: Exception | None = None
        self._lock = threading.Lock()
        self._measured: dict | None = None
        self._commanded: dict | None = None
        self._stamp = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            while self._running:
                start = time.perf_counter()
                measured = self.follower.get_observation()
                commanded = self.follower.send_action(self.leader.get_action())
                with self._lock:
                    self._measured = measured
                    self._commanded = commanded
                    self._stamp = time.time()
                time.sleep(max(self.period_s - (time.perf_counter() - start), 0.0))
        except Exception as exc:  # surfaced when the next pair is saved
            self.error = exc

    def latest(self) -> tuple[dict | None, dict | None, float]:
        with self._lock:
            return self._measured, self._commanded, self._stamp

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        input("Support the follower arm, then press Enter to release its torque... ")
        self.follower.disconnect()
        self.leader.disconnect()


def save_pair(
    cameras: list[CameraReader],
    arm: TeleopArm | None,
    session_dir: Path,
    index: int,
    label: str,
    writer,
    csv_file,
) -> bool:
    frames = [cam.latest() for cam in cameras]
    if any(frame is None for frame, _ in frames):
        print("not saved: a camera has no frame yet")
        return False
    if arm is not None:
        measured, commanded, joint_stamp = arm.latest()
        if measured is None or time.time() - joint_stamp > MAX_JOINT_AGE_S:
            print(f"not saved: no fresh joint reading (teleop error: {arm.error!r})")
            return False

    prefix = f"pair_{index:03d}_{label}"
    row = {"index": index, "label": label, "saved_at": datetime.now().isoformat(timespec="milliseconds")}
    for cam, (frame, stamp) in zip(cameras, frames):
        filename = f"{prefix}_{cam.name}.png"
        cv2.imwrite(str(session_dir / filename), frame)
        row[f"{cam.name}_file"] = filename
        row[f"{cam.name}_stamp"] = f"{stamp:.3f}"
    row["skew_ms"] = f"{abs(frames[0][1] - frames[1][1]) * 1000:.1f}"

    if arm is not None:
        joints = [float(measured[k]) for k in MOTOR_KEYS]
        # Same schema as read_current_pose.py, so fk_from_pose_json.py works on it.
        record = {
            "schema": "current_pose_v1",
            "motor_keys": list(MOTOR_KEYS),
            "joints_deg": joints,
            "commanded_deg": [float(commanded[k]) for k in MOTOR_KEYS],
            "stamp": joint_stamp,
        }
        (session_dir / f"{prefix}_joints.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        row["joints_stamp"] = f"{joint_stamp:.3f}"
        row.update({k: f"{v:.3f}" for k, v in zip(MOTOR_KEYS, joints)})

    writer.writerow(row)
    csv_file.flush()
    print(f"saved pair {index:03d} ({label}), skew {row['skew_ms']} ms")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wrist", type=parse_camera, default=DEFAULT_WRIST_CAMERA, help="wrist camera index or device path")
    parser.add_argument(
        "--stationary", type=parse_camera, default=DEFAULT_STATIONARY_CAMERA, help="stationary camera index or device path"
    )
    parser.add_argument("--width", type=int, default=None, help="requested frame width (default: camera default)")
    parser.add_argument("--height", type=int, default=None, help="requested frame height (default: camera default)")
    parser.add_argument("--fourcc", default="MJPG", help="pixel format; MJPG lets two USB cameras share a bus")
    parser.add_argument("--wrist-controls", default=DEFAULT_WRIST_CONTROLS, help="v4l2-ctl -c string; '' to skip")
    parser.add_argument("--stationary-controls", default=DEFAULT_STATIONARY_CONTROLS, help="v4l2-ctl -c string; '' to skip")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--session", default=None, help="session folder name (default: current date and time)")
    parser.add_argument("--teleop", action="store_true", help="run leader->follower teleop and log joints (moves the arm)")
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--follower-id", default=DEFAULT_FOLLOWER_ID)
    parser.add_argument("--leader-port", default=DEFAULT_LEADER_PORT)
    parser.add_argument("--leader-id", default=DEFAULT_LEADER_ID)
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=None,
        help="cap per-step follower motion in degrees (lerobot's safety clip; default: off)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_dir = args.out_root / (args.session or datetime.now().strftime("%Y%m%d_%H%M%S"))
    session_dir.mkdir(parents=True, exist_ok=True)

    cameras: list[CameraReader] = []
    arm: TeleopArm | None = None
    try:
        sources = (args.wrist, args.stationary)
        controls = (args.wrist_controls, args.stationary_controls)
        for name, source, ctrl in zip(CAMERA_NAMES, sources, controls):
            cameras.append(CameraReader(name, source, args.width, args.height, args.fourcc, ctrl))
        if args.teleop:
            arm = TeleopArm(args)

        # Let the first frames arrive before reporting the negotiated settings.
        time.sleep(1.0)
        session = {"created": datetime.now().isoformat(timespec="seconds")}
        session.update({cam.name: cam.settings() for cam in cameras})
        if arm is not None:
            session["arm"] = arm.settings
        (session_dir / "session.json").write_text(json.dumps(session, indent=2), encoding="utf-8")
        for cam in cameras:
            print(f"{cam.name}: {({k: v for k, v in session[cam.name].items() if k != 'v4l2_controls'})}")
        print(f"saving to {session_dir}")
        print("space = save pair, c = save calib pair, q/Esc = quit")

        csv_path = session_dir / "pairs.csv"
        # Re-using a --session name appends to its pairs.csv instead of overwriting pairs.
        has_header = csv_path.exists() and csv_path.stat().st_size > 0
        index = len(list(csv.DictReader(csv_path.open()))) + 1 if has_header else 1
        fields = ["index", "label", "saved_at"]
        for name in CAMERA_NAMES:
            fields += [f"{name}_file", f"{name}_stamp"]
        fields.append("skew_ms")
        if arm is not None:
            fields += ["joints_stamp", *MOTOR_KEYS]

        with csv_path.open("a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            if not has_header:
                writer.writeheader()

            # Free up keys that matplotlib binds by default ("c" is toolbar "back").
            plt.rcParams["keymap.back"] = [k for k in plt.rcParams["keymap.back"] if k != "c"]
            pressed: list[str] = []
            fig, axes = plt.subplots(1, len(cameras), figsize=(12, 5))
            fig.canvas.mpl_connect("key_press_event", lambda event: pressed.append(event.key))
            images = [None] * len(cameras)
            for ax, cam in zip(axes, cameras):
                ax.set_title(cam.name)
                ax.set_axis_off()

            while plt.fignum_exists(fig.number):
                for i, (ax, cam) in enumerate(zip(axes, cameras)):
                    frame = cam.latest()[0]
                    if frame is None:
                        continue
                    rgb = frame[..., ::-1]
                    if images[i] is None:
                        images[i] = ax.imshow(rgb)
                    else:
                        images[i].set_data(rgb)
                title = f"saved: {index - 1}    space = pair, c = calib pair, q/Esc = quit"
                if arm is not None and arm.error is not None:
                    title = f"TELEOP STOPPED: {arm.error!r}\n{title}"
                fig.suptitle(title, color="red" if arm is not None and arm.error is not None else "black")
                plt.pause(0.03)

                while pressed:
                    key = pressed.pop(0)
                    if key in ("q", "escape"):
                        plt.close(fig)
                    elif key in (" ", "c"):
                        label = "calib" if key == "c" else "shot"
                        if save_pair(cameras, arm, session_dir, index, label, writer, csv_file):
                            index += 1
    finally:
        for cam in cameras:
            cam.close()
        if arm is not None:
            arm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
