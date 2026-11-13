#!/usr/bin/env python3
"""Calibrate T_gripper_camera from several parked poses + wrist board PnP.

Keys:
  space  capture a sample at the current park pose
  t      choose / change the square for later samples
  f      corner clicks: snap to detected corners / manual (--manual-corners starts manual)
  s      solve this run, write YAML + calibrated URDF, quit
  q      quit without solving (the captured samples stay saved)

Each capture uses the same corner circuit as the click tool: green is the
corner nearest a1, then the other three counter-clockwise in the image
(toward the h-file, then the far corner, then the rank-8 side).

Every sample is saved as it is taken, under output/camera_mount/runs/<date_time>/:
the raw wrist frame, the clicked corners (raw fisheye pixels too) and the raw joint
reading. --resolve solves several saved runs together, redoing each sample with
the current lens file, URDF and joint offsets:

  python run_mount_calibration.py --resolve all
  python run_mount_calibration.py --resolve 20261111_153000 20261112_101500 --board-per-run

Minimum 3 samples. Writes under click_to_3d_oliver/output/camera_mount/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import click3d_shared as shared
from click3d_shared import (
    BOARD_CENTRE_BASE_MM,
    SQUARE_MM,
    MountFit,
    arm_held,
    arm_state_text,
    detect_grid_for_view,
    draw_backup_banner,
    corner_click_mode_line,
    draw_detected_corners,
    ensure_output_dirs,
    fit_mount,
    fk_link_mm,
    hide_jpeg_warnings,
    invert,
    joint_offsets_deg,
    load_demo_board_poses,
    load_fisheye_calibration,
    load_mount,
    load_session_cameras,
    open_camera,
    parse_camera,
    panel_text,
    parse_square_name,
    pose_from_square_corners,
    read_joints_json,
    read_joints_robot,
    resolve_core_assets,
    resolve_demo_pair,
    resolve_label_click,
    save_session_cameras,
    se3,
    square_corner_points_board,
    toggle_arm_hold,
    undistort_wrist,
    write_board_in_base_file,
    write_mount_files,
)

WINDOW = "mount_calibration"
CORNER_CIRCUIT = [
    "corners: GREEN nearest a1, then counter-clockwise",
    "next toward h, then far corner, then rank 8",
]
RUN_SCHEMA = "mount_run_v1"


# --- saved runs ---------------------------------------------------------------------------


def runs_dir() -> Path:
    """Where capture runs are saved; read at call time so the smoke test can redirect it."""
    return shared.MOUNT_DIR / "runs"


def _fisheye_arrays(calib: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(calib["K"], dtype=np.float64),
        np.asarray(calib["D"], dtype=np.float64).reshape(4, 1),
        np.asarray(calib["K_undistorted"], dtype=np.float64),
    )


def undistorted_to_raw(points: np.ndarray, calib: dict) -> np.ndarray:
    """Pixels in undistort_wrist's image -> the same points in the raw fisheye frame."""
    K, D, K_und = _fisheye_arrays(calib)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    normalised = (np.linalg.inv(K_und) @ np.column_stack([pts, np.ones(len(pts))]).T).T[:, :2]
    return cv2.fisheye.distortPoints(normalised.reshape(-1, 1, 2), K, D).reshape(-1, 2)


def raw_to_undistorted(points: np.ndarray, calib: dict) -> np.ndarray:
    """Raw fisheye pixels -> pixels in undistort_wrist's image for this lens file."""
    K, D, K_und = _fisheye_arrays(calib)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.fisheye.undistortPoints(pts, K, D, R=np.eye(3), P=K_und).reshape(-1, 2)


def _write_run(run_dir: Path, meta: dict) -> None:
    tmp = run_dir / "samples.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(run_dir / "samples.json")


def start_run(source: str, fisheye_path: Path, urdf_path: Path, wrist_camera) -> tuple[Path, dict]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir() / stamp
    suffix = 2
    while run_dir.exists():
        run_dir = runs_dir() / f"{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    meta = {
        "schema": RUN_SCHEMA,
        "created": datetime.now().isoformat(timespec="seconds"),
        # robot: real joint readings. demo / camera_only / smoke runs are skipped by --resolve all.
        "source": source,
        "wrist_camera": str(wrist_camera),
        "fisheye_at_capture": str(fisheye_path),
        "urdf_at_capture": str(urdf_path),
        "joint_offsets_deg_at_capture": [round(float(v), 3) for v in joint_offsets_deg()],
        "square_mm": SQUARE_MM,
        "samples": [],
    }
    _write_run(run_dir, meta)
    return run_dir, meta


def save_sample(
    run_dir: Path,
    meta: dict,
    *,
    raw_bgr: np.ndarray | None,
    square: str,
    corners_px: np.ndarray,
    joints: list[float],
    fisheye_calib: dict,
    T_board_camera: np.ndarray,
) -> int:
    """Append one capture to the run: raw frame, clicked corners and the raw joint reading."""
    index = len(meta["samples"]) + 1
    image = None
    if raw_bgr is not None:
        image = f"sample_{index:02d}.png"
        cv2.imwrite(str(run_dir / image), raw_bgr)
    corners = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    meta["samples"].append(
        {
            "index": index,
            "stamp": round(time.time(), 3),
            "image": image,
            "square": square,
            # Same order as clicked: nearest a1, then counter-clockwise in the image.
            "corners_raw_px": np.round(undistorted_to_raw(corners, fisheye_calib), 3).tolist(),
            "corners_undistorted_px": np.round(corners, 3).tolist(),
            "joint_reading_deg": [round(float(v), 4) for v in joints],
            "T_board_camera_mm_at_capture": np.round(T_board_camera, 6).tolist(),
        }
    )
    _write_run(run_dir, meta)
    return index


def find_runs(names: list[str]) -> list[Path]:
    """Run folders by name (under runs/) or path; 'all' means every robot run."""
    if names == ["all"]:
        found = []
        for path in sorted(runs_dir().glob("*/samples.json")):
            meta = json.loads(path.read_text(encoding="utf-8"))
            if meta.get("source") == "robot" and meta.get("samples"):
                found.append(path.parent)
            else:
                print(f"skipping {path.parent.name}: source {meta.get('source')}, {len(meta.get('samples', []))} samples")
        return found
    runs = []
    for name in names:
        path = Path(name)
        if not (path / "samples.json").is_file():
            path = runs_dir() / name
        if not (path / "samples.json").is_file():
            raise SystemExit(f"no saved run {name!r} (looked for {path / 'samples.json'})")
        runs.append(path)
    return runs


def load_samples(
    run_dirs: list[Path],
    fisheye_calib: dict,
    urdf_path: Path,
    exclude: set[tuple[str, int]] = frozenset(),
) -> list[dict]:
    """Every saved sample, redone with the current lens file, URDF and joint offsets."""
    _, _, K_und = _fisheye_arrays(fisheye_calib)
    samples = []
    for run_dir in run_dirs:
        meta = json.loads((run_dir / "samples.json").read_text(encoding="utf-8"))
        if meta.get("schema") != RUN_SCHEMA:
            raise SystemExit(f"{run_dir}: unknown schema {meta.get('schema')!r}")
        if abs(float(meta.get("square_mm", SQUARE_MM)) - SQUARE_MM) > 1e-6:
            print(f"warning: {run_dir.name} was captured with square_mm {meta['square_mm']}, now {SQUARE_MM}")
        for item in meta["samples"]:
            name = f"{run_dir.name}#{item['index']}"
            if (run_dir.name, int(item["index"])) in exclude:
                print(f"excluded {name}")
                continue
            corners = raw_to_undistorted(np.asarray(item["corners_raw_px"], dtype=np.float64), fisheye_calib)
            pose = pose_from_square_corners(corners, item["square"], K_und)
            if pose is None:
                print(f"skipping {name}: no valid corner pose with the current lens file")
                continue
            joints = [float(v) for v in item["joint_reading_deg"]]
            samples.append(
                {
                    "run": run_dir.name,
                    "name": name,
                    "square": item["square"],
                    "joints": joints,
                    "corners_px": corners,
                    "T_board_camera": pose,
                    "T_base_gripper": fk_link_mm(urdf_path, "gripper_frame_link", joints),
                }
            )
    return samples


def parse_exclude(items: list[str]) -> set[tuple[str, int]]:
    out = set()
    for item in items:
        run, _, index = item.rpartition(":")
        if not run or not index.isdigit():
            raise SystemExit(f"--exclude wants RUN:N (e.g. 20261111_153000:3), got {item!r}")
        out.add((Path(run).name, int(index)))
    return out


# --- solve and report ---------------------------------------------------------------------


def _rotvec_deg(rotation: np.ndarray) -> np.ndarray:
    return np.degrees(cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))[0].reshape(3))


def _fmt(values, digits: int = 2) -> str:
    return "[" + ", ".join(f"{float(v):+.{digits}f}" for v in values) + "]"


def solve_samples(
    samples: list[dict],
    *,
    board_per_run: bool,
    urdf_path: Path,
    fisheye_calib: dict,
    fisheye_path: Path,
    write: bool = True,
) -> tuple[MountFit, dict]:
    """Fit the mount to samples from one or more runs, print a report, and write the mount files."""
    if len(samples) < 3:
        raise RuntimeError(f"need at least 3 samples, have {len(samples)}")
    runs = list(dict.fromkeys(s["run"] for s in samples))
    counts = {run: sum(s["run"] == run for s in samples) for run in runs}
    if board_per_run:
        groups = [runs.index(s["run"]) for s in samples]
        for run, count in counts.items():
            if count < 3:
                print(f"warning: {run} has {count} sample(s); with its own board pose it adds little to the mount")
    else:
        groups = [0] * len(samples)
    _, _, K_und = _fisheye_arrays(fisheye_calib)
    first = samples[0]
    X_urdf = invert(first["T_base_gripper"]) @ fk_link_mm(urdf_path, "wrist_camera_link", first["joints"])
    fit = fit_mount(samples, groups, K_und, X_urdf)
    X = fit.T_gripper_camera_mm

    correction = invert(X_urdf) @ X  # in the URDF camera frame
    run_rms = {
        run: float(np.sqrt(np.mean([fit.sample_rms_px[i] ** 2 for i, s in enumerate(samples) if s["run"] == run])))
        for run in runs
    }
    centre_board = np.array([4.0 * SQUARE_MM, 4.0 * SQUARE_MM, 0.0, 1.0])
    boards = []
    for group, Y in enumerate(fit.T_base_board_mm):
        boards.append(
            {
                "runs": [runs[group]] if board_per_run else runs,
                "tilt_deg": round(float(np.degrees(np.arccos(np.clip(Y[2, 2], -1.0, 1.0)))), 2),
                "centre_minus_ruler_mm": np.round((Y @ centre_board)[:3] - np.array(BOARD_CENTRE_BASE_MM), 1).tolist(),
            }
        )
    worst = sorted(range(len(samples)), key=lambda i: -fit.sample_rms_px[i])[:3]
    report = {
        "runs": runs,
        "samples_per_run": counts,
        "board_pose": "one per run" if board_per_run else "one for all runs",
        "fit_start": fit.start,
        "reprojection_rms_px": round(fit.rms_px, 3),
        "run_rms_px": {run: round(value, 3) for run, value in run_rms.items()},
        "worst_samples_rms_px": {samples[i]["name"]: round(float(fit.sample_rms_px[i]), 3) for i in worst},
        "mount_std_rotation_deg_camera_xyz": np.round(fit.rotation_std_deg, 3).tolist(),
        "mount_std_translation_mm_camera_xyz": np.round(fit.translation_std_mm, 3).tolist(),
        "change_vs_urdf_rotation_deg_camera_xyz": np.round(_rotvec_deg(correction[:3, :3]), 3).tolist(),
        "change_vs_urdf_translation_mm_camera_xyz": np.round(correction[:3, 3], 3).tolist(),
        "boards": boards,
        "fisheye": str(fisheye_path),
    }

    print()
    print(f"mount fit: {len(samples)} samples from {len(runs)} run(s), board pose {report['board_pose']}")
    for run in runs:
        print(f"  {run}: {counts[run]} samples, rms {run_rms[run]:.2f} px")
    print(f"  reprojection rms {fit.rms_px:.2f} px (start: {fit.start}); worst samples: "
          + ", ".join(f"{k} {v:.2f}" for k, v in report["worst_samples_rms_px"].items()))
    print(f"  mount 1-sigma: rotation {_fmt(fit.rotation_std_deg)} deg, translation {_fmt(fit.translation_std_mm, 1)} mm"
          " (camera x, y, z)")
    print(f"  change vs URDF estimate: rotation {_fmt(report['change_vs_urdf_rotation_deg_camera_xyz'])} deg,"
          f" translation {_fmt(correction[:3, 3], 1)} mm (camera x, y, z)")
    for board in boards:
        label = ", ".join(board["runs"])
        print(f"  board ({label}): tilt {board['tilt_deg']:.2f} deg, centre - ruler {board['centre_minus_ruler_mm']} mm")
    if not board_per_run and len(runs) > 1 and fit.rms_px > 2.0:
        print("  poor fit across runs: if the board moved between them, compare with --board-per-run")

    if write:
        extra = {
            "fit": "reprojection (Koide & Menegatti 2019), start " + fit.start,
            "reprojection_rms_px": round(fit.rms_px, 3),
            "runs": runs,
            "board_pose": report["board_pose"],
            "mount_std_rotation_deg_camera_xyz": report["mount_std_rotation_deg_camera_xyz"],
            "mount_std_translation_mm_camera_xyz": report["mount_std_translation_mm_camera_xyz"],
            "fisheye": str(fisheye_path),
        }
        yaml_path, urdf_out = write_mount_files(
            X,
            sample_count=len(samples),
            residual_mm=fit.mean_residual_mm,
            source_urdf=urdf_path,
            extra=extra,
        )
        board_note = {"board_from_runs": boards[-1]["runs"]}
        board_path = write_board_in_base_file(
            fit.T_base_board_mm[-1],
            sample_count=len(samples),
            residual_mm=fit.mean_residual_mm,
            extra=board_note,
        )
        print(f"wrote {yaml_path}\nwrote {urdf_out}\nwrote {board_path}")
    else:
        print("(--no-write: nothing written)")
    return fit, report


def resolve_main(args: argparse.Namespace) -> None:
    """--resolve: solve saved runs together, headless (no camera, no arm)."""
    report, assets = resolve_core_assets()
    if report.banner_text():
        print(report.banner_text())
    fisheye_path = assets["fisheye"].path
    urdf_path = assets["urdf"].path
    fisheye_calib = load_fisheye_calibration(fisheye_path)
    run_dirs = find_runs(args.resolve)
    if not run_dirs:
        raise SystemExit(f"no robot runs with samples under {runs_dir()}")
    samples = load_samples(run_dirs, fisheye_calib, urdf_path, parse_exclude(args.exclude))
    try:
        solve_samples(
            samples,
            board_per_run=args.board_per_run,
            urdf_path=urdf_path,
            fisheye_calib=fisheye_calib,
            fisheye_path=fisheye_path,
            write=not args.no_write,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


# --- capture GUI --------------------------------------------------------------------------


class MountApp:
    def __init__(self, args: argparse.Namespace) -> None:
        ensure_output_dirs()
        self.args = args
        self.report, assets = resolve_core_assets()
        self.fisheye = assets["fisheye"]
        self.urdf = assets["urdf"]
        self.mount = load_mount(self.urdf, self.report)
        self.samples: list[dict] = []
        self.run_dir: Path | None = None
        self.run_meta: dict | None = None
        self.T_base_board = None
        self.label_square = parse_square_name(args.square) if args.square else None
        self.square_buffer = ""
        self.help_lines: list[str] = list(CORNER_CIRCUIT)
        self.status = "park arm, space=capture (need >=3), s=solve"
        self.demo = None
        self.cap = None
        self.joints = None
        self.frame = None
        self.frame_raw = None
        self.K_wrist = None
        self.fisheye_calib = None
        self.label_points: list[tuple[float, float]] = []
        self.mode = "live"
        self.pending_image = None
        self.pending_raw = None
        self.pending_joints = None
        self.pending_grid = None
        self.pending_grid_source = "none"
        self.done = False
        self.manual_corners = args.manual_corners
        if not args.smoke and self.label_square is None:
            self.mode = "enter_square"
            self.status = "type square name (e.g. e4) then Enter"
            self.help_lines = [
                "Pick a square with clear corners for this session",
                "Press t later to change square between samples",
                *CORNER_CIRCUIT,
            ]
        self.wrist_camera, _stationary = load_session_cameras(
            getattr(args, "wrist_camera", None),
            getattr(args, "stationary_camera", None),
        )
        args.wrist_camera = self.wrist_camera
        if args.demo_pair:
            self.source = "smoke" if args.smoke else "demo"
            self.demo = resolve_demo_pair(self.report)
            self.joints = read_joints_json(self.demo["joints"].path)
            raw = cv2.imread(str(self.demo["wrist"].path))
            self.frame_raw = raw.copy() if raw is not None else None
            self.frame, self.K_wrist, self.fisheye_calib = undistort_wrist(raw, self.fisheye.path)
        elif args.robot:
            self.source = "robot"
            self.cap = open_camera(self.wrist_camera, "wrist")
            save_session_cameras(self.wrist_camera, _stationary)
            self.joints = read_joints_robot()
            self.arm_state = arm_state_text(arm_held())
        else:
            # No arm: joints stay zero, so these samples can't calibrate a mount.
            self.source = "camera_only"
            self.cap = open_camera(self.wrist_camera, "wrist")
            save_session_cameras(self.wrist_camera, _stationary)
            self.joints = [0.0] * 6

    def grab(self) -> np.ndarray:
        if self.demo is not None:
            return self.frame.copy()
        ok, raw = self.cap.read()
        if not ok:
            raise RuntimeError("wrist camera read failed")
        self.frame_raw = raw.copy()
        image, self.K_wrist, self.fisheye_calib = undistort_wrist(raw, self.fisheye.path)
        if self.args.robot:
            self.joints = read_joints_robot()
        return image

    def capture(self) -> None:
        if self.label_square is None:
            self.mode = "enter_square"
            self.status = "type square name (e.g. e4) then Enter"
            self.help_lines = ["Pick a square with clear corners for this session"]
            return
        image = self.grab()
        grid, source = detect_grid_for_view(
            image,
            raw_bgr=self.frame_raw,
            fisheye_calib=self.fisheye_calib,
            restrict_to_board=False,
        )
        self.pending_image = image
        self.pending_raw = None if self.frame_raw is None else self.frame_raw.copy()
        self.pending_joints = list(self.joints)
        self.pending_grid = grid
        self.pending_grid_source = source
        self.label_points = []
        self.mode = "label"
        self.corner_prompt()

    def corner_prompt(self) -> None:
        """Status and help for labelling the corners of the captured frame."""
        self.status = f"click 4 corners of {self.label_square}"
        if self.pending_grid is not None and not self.manual_corners:
            self.status += f" (corners: {self.pending_grid_source})"
        self.help_lines = [
            corner_click_mode_line(self.pending_grid, self.manual_corners),
            "1st click: GREEN = corner nearest a1",
            "then 3 cyan, counter-clockwise in the image",
            "next is toward h, then far corner, then rank 8",
        ]

    def _store_sample(self, T_board_camera: np.ndarray, corners_px: np.ndarray) -> None:
        if self.run_dir is None:
            self.run_dir, self.run_meta = start_run(self.source, self.fisheye.path, self.urdf.path, self.wrist_camera)
            print(f"saving samples to {self.run_dir}")
        index = save_sample(
            self.run_dir,
            self.run_meta,
            raw_bgr=self.pending_raw,
            square=self.label_square,
            corners_px=corners_px,
            joints=self.pending_joints,
            fisheye_calib=self.fisheye_calib,
            T_board_camera=T_board_camera,
        )
        self.samples.append({"index": index, "square": self.label_square, "joints": list(self.pending_joints)})
        self.status = f"samples={len(self.samples)} (saved, run {self.run_dir.name})  space=more  s=solve"
        self.help_lines = list(CORNER_CIRCUIT)
        self.mode = "live"

    def solve(self) -> None:
        if len(self.samples) < 3:
            self.status = "need at least 3 samples"
            return
        fisheye_calib = load_fisheye_calibration(self.fisheye.path)
        samples = load_samples([self.run_dir], fisheye_calib, self.urdf.path)
        try:
            fit, _ = solve_samples(
                samples,
                board_per_run=False,
                urdf_path=self.urdf.path,
                fisheye_calib=fisheye_calib,
                fisheye_path=self.fisheye.path,
            )
        except RuntimeError as exc:
            self.status = str(exc)
            print(exc)
            return
        self.T_base_board = fit.T_base_board_mm[0]
        print("To solve this run together with earlier ones (same board position):")
        print("  python run_mount_calibration.py --resolve all")
        self.status = f"saved rms={fit.rms_px:.2f}px  mean residual={fit.mean_residual_mm:.2f}mm"
        self.done = True

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if x >= 640:
            return
        if self.mode == "label":
            free = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
            grid = None if self.manual_corners else self.pending_grid
            snapped, err = resolve_label_click(grid, (float(x), float(y)), free_click=free)
            if err is not None:
                self.status = err
                return
            self.label_points.append(snapped)
            if len(self.label_points) >= 4:
                pts = np.asarray(self.label_points[:4], dtype=np.float64)
                pose = pose_from_square_corners(pts, self.label_square, self.K_wrist)
                if pose is None:
                    self.status = "pose rejected — green = a1-near, then the other three counter-clockwise in the image"
                    self.label_points = []
                    return
                self._store_sample(pose, pts)

    def compose(self) -> np.ndarray:
        if self.mode == "label" and self.pending_image is not None:
            image = self.pending_image.copy()
            if not self.manual_corners:
                draw_detected_corners(image, self.pending_grid)
            for i, (px, py) in enumerate(self.label_points):
                color = (0, 255, 0) if i == 0 else (255, 200, 0)
                cv2.circle(image, (int(px), int(py)), 5, color, -1)
        else:
            image = self.grab()
        backup = self.report.banner_text()
        lines = [
            "MOUNT CALIBRATION",
            f"status: {self.status}",
            f"square: {self.label_square or '-'}",
            f"samples: {len(self.samples)}" + (f"  (run {self.run_dir.name})" if self.run_dir else ""),
            f"urdf: {self.urdf.source}",
            "space capture   t square   s solve   q quit",
            f"f corner clicks: {'MANUAL' if self.manual_corners else 'snap to detected'}",
        ]
        if self.args.robot:
            lines.append(f"arm: {self.arm_state}  h hold/release")
        if self.mode == "enter_square":
            lines.append(f"buffer: {self.square_buffer or '_'}")
        lines.extend(self.help_lines)
        if backup:
            lines.insert(1, backup)
        panel = panel_text(lines, width=480, height=480)
        panel = draw_backup_banner(panel, backup)
        blank = np.full_like(image, 40)
        return np.hstack([image, blank, panel])

    def handle_key(self, key: int) -> bool:
        if self.mode == "enter_square":
            if key in (ord("q"),):
                return False
            if key == 27:
                self.square_buffer = ""
                return True
            if key in (13, 10):
                name = parse_square_name(self.square_buffer)
                if name is None:
                    self.status = "invalid square — try e4"
                    self.square_buffer = ""
                    return True
                self.label_square = name
                self.mode = "live"
                self.status = "park arm, space=capture (need >=3), s=solve, t=square"
                self.help_lines = list(CORNER_CIRCUIT)
                return True
            ch = chr(key) if 32 <= key < 127 else ""
            if ch.lower() in "abcdefgh12345678" and len(self.square_buffer) < 2:
                self.square_buffer += ch.lower()
            return True
        if key in (ord("q"), 27):
            return False
        if key == ord("t"):
            self.mode = "enter_square"
            self.square_buffer = ""
            self.status = "type square name (e.g. e4) then Enter"
            self.help_lines = [
                "Pick a square with clear corners",
                f"current was: {self.label_square or '-'}",
            ]
            return True
        if key == ord("f"):
            self.manual_corners = not self.manual_corners
            if self.mode == "label":
                self.corner_prompt()
            else:
                self.status = f"corner clicks: {'manual' if self.manual_corners else 'snap to detected corners'}"
            return True
        if key == ord("h") and self.args.robot:
            self.arm_state = arm_state_text(toggle_arm_hold())
            self.status = f"arm {self.arm_state}"
            return True
        if key == ord(" "):
            self.capture()
        if key == ord("s"):
            self.solve()
        return True

    def run_gui(self) -> None:
        if self.report.banner_text():
            print(self.report.banner_text())
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while not self.done:
            cv2.imshow(WINDOW, self.compose())
            key = cv2.waitKey(30) & 0xFF
            if key != 255 and not self.handle_key(key):
                break
            if self.done:
                break
        cv2.destroyAllWindows()
        if self.cap is not None:
            self.cap.release()
        if self.run_dir is not None and not self.done:
            print(f"{len(self.samples)} sample(s) kept in {self.run_dir}; solve later with --resolve")

    # --- smoke test (no camera, no arm; writes only to a temporary folder) ---

    def _synthetic_run(
        self,
        X_true: np.ndarray,
        Y_true: np.ndarray,
        joint_sets: list[list[float]],
        rng: np.random.Generator,
    ) -> Path:
        """Save a run whose corners are what a camera mounted at X_true would see of board Y_true."""
        _, _, K_und = _fisheye_arrays(self.fisheye_calib)
        run_dir, meta = start_run("smoke", self.fisheye.path, self.urdf.path, "smoke")
        for joints in joint_sets:
            A = fk_link_mm(self.urdf.path, "gripper_frame_link", joints)
            T_camera_board = invert(A @ X_true) @ Y_true
            best = None
            for square in (f + r for f in "abcdefgh" for r in "12345678"):
                cam = square_corner_points_board(square) @ T_camera_board[:3, :3].T + T_camera_board[:3, 3]
                if (cam[:, 2] <= 50.0).any():
                    continue
                uv = cam[:, :2] / cam[:, 2:3] @ K_und[:2, :2].T + K_und[:2, 2]
                if (uv < 40).any() or (uv[:, 0] > 600).any() or (uv[:, 1] > 440).any():
                    continue
                gap = float(np.linalg.norm(uv.mean(axis=0) - (320.0, 240.0)))
                if best is None or gap < best[0]:
                    best = (gap, square, uv)
            assert best is not None, f"smoke: no square fully in view at joints {joints}"
            _, square, uv = best
            uv = uv + rng.normal(0.0, 0.3, uv.shape)
            pose = pose_from_square_corners(uv, square, K_und)
            assert pose is not None
            save_sample(
                run_dir,
                meta,
                raw_bgr=None,
                square=square,
                corners_px=uv,
                joints=joints,
                fisheye_calib=self.fisheye_calib,
                T_board_camera=pose,
            )
        return run_dir

    def run_smoke(self) -> None:
        real_mount_dir = shared.MOUNT_DIR
        tmp = Path(tempfile.mkdtemp(prefix="mount_smoke_"))
        shared.MOUNT_DIR = tmp  # nothing under output/ is written
        try:
            self._smoke(tmp)
        finally:
            shared.MOUNT_DIR = real_mount_dir
            shutil.rmtree(tmp, ignore_errors=True)

    def _smoke(self, tmp: Path) -> None:
        calib = self.fisheye_calib
        pts = np.array([[100.0, 80.0], [320.0, 240.0], [600.0, 400.0]])
        round_trip = raw_to_undistorted(undistorted_to_raw(pts, calib), calib)
        assert np.abs(round_trip - pts).max() < 1e-3, round_trip

        joints0 = list(self.joints)
        A0 = fk_link_mm(self.urdf.path, "gripper_frame_link", joints0)
        X_urdf = invert(A0) @ fk_link_mm(self.urdf.path, "wrist_camera_link", joints0)
        # A mount off the URDF estimate like the real one looks (about 5 deg roll, 2 deg pitch).
        tilt = cv2.Rodrigues(np.radians([-2.3, 4.8, 0.5]).reshape(3, 1))[0]
        X_true = X_urdf @ se3(tilt, np.array([2.0, -3.0, 1.5]))
        T_board_cam0, _ = load_demo_board_poses(self.demo["poses"].path)
        Y_true = A0 @ X_true @ invert(T_board_cam0)
        deltas = [
            (0, 0, 0, 0, 0),
            (10, 0, 0, 0, 30),
            (-10, 0, 0, 0, -30),
            (0, 5, -5, 10, 0),
            (5, -5, 5, -10, 45),
            (-5, 0, 0, 5, -40),
        ]
        joint_sets = [[j + d for j, d in zip(joints0, (*delta, 0.0))] for delta in deltas]
        rng = np.random.default_rng(0)
        run_a = self._synthetic_run(X_true, Y_true, joint_sets, rng)
        moved = se3(cv2.Rodrigues(np.radians([0.0, 0.0, 2.0]).reshape(3, 1))[0], np.array([15.0, -8.0, 0.0]))
        run_b = self._synthetic_run(X_true, moved @ Y_true, joint_sets[::-1], rng)

        def mount_error(fit: MountFit) -> tuple[float, float]:
            delta = invert(X_true) @ fit.T_gripper_camera_mm
            return float(np.linalg.norm(_rotvec_deg(delta[:3, :3]))), float(np.linalg.norm(delta[:3, 3]))

        def check(fit: MountFit) -> None:
            # Within 3 sigma of the fit's own uncertainty: translation is the weakly fixed part.
            rot_err, trans_err = mount_error(fit)
            assert rot_err < 3.0 * np.linalg.norm(fit.rotation_std_deg) + 0.1, (rot_err, fit.rotation_std_deg)
            assert trans_err < 3.0 * np.linalg.norm(fit.translation_std_mm) + 0.5, (trans_err, fit.translation_std_mm)
            assert rot_err < 1.0, rot_err  # the 5 deg tilt is recovered

        common = dict(urdf_path=self.urdf.path, fisheye_calib=calib, fisheye_path=self.fisheye.path)
        one = load_samples([run_a], calib, self.urdf.path)
        fit_a, _ = solve_samples(one, board_per_run=False, **common)
        check(fit_a)
        mount = load_mount(self.urdf)
        assert mount.kind == "calibrated" and mount.mount_path.parent == tmp, mount
        assert np.allclose(mount.T_gripper_camera_mm, fit_a.T_gripper_camera_mm, atol=1e-3)

        found = find_runs(["all"])
        assert found == [], found  # smoke runs aren't robot runs
        both = load_samples([run_a, run_b], calib, self.urdf.path)
        shared_fit, _ = solve_samples(both, board_per_run=False, write=False, **common)
        per_run_fit, _ = solve_samples(both, board_per_run=True, write=False, **common)
        rot_err, trans_err = mount_error(per_run_fit)
        assert shared_fit.rms_px > 3.0 * per_run_fit.rms_px, (shared_fit.rms_px, per_run_fit.rms_px)
        check(per_run_fit)
        dropped = load_samples([run_a], calib, self.urdf.path, parse_exclude([f"{run_a.name}:2"]))
        assert len(dropped) == len(one) - 1
        print(
            "mount smoke ok: one run "
            f"{mount_error(fit_a)[0]:.2f} deg / {mount_error(fit_a)[1]:.2f} mm off; "
            f"board moved: shared rms {shared_fit.rms_px:.1f} px, per run {per_run_fit.rms_px:.2f} px, "
            f"{rot_err:.2f} deg / {trans_err:.2f} mm off"
        )


def main() -> None:
    hide_jpeg_warnings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo-pair", action="store_true")
    parser.add_argument("--robot", action="store_true")
    parser.add_argument(
        "--wrist-camera",
        "--wrist-index",
        type=parse_camera,
        default=None,
        help="Mounted camera: index or device path (default: session_config.yaml, then paths.WRIST_CAMERA)",
    )
    parser.add_argument(
        "--stationary-camera",
        "--stationary-index",
        type=parse_camera,
        default=None,
        help="Unused here; kept so session_config keeps both cameras",
    )
    parser.add_argument("--square", type=str, default=None)
    parser.add_argument(
        "--manual-corners",
        action="store_true",
        help="Start with manual corner clicks (no snapping to detected corners); f toggles",
    )
    parser.add_argument(
        "--resolve",
        nargs="+",
        metavar="RUN",
        help="Solve saved runs together instead of capturing: run folder names under "
        "output/camera_mount/runs/ (or paths), or 'all' for every run taken with --robot",
    )
    parser.add_argument(
        "--board-per-run",
        action="store_true",
        help="With --resolve: the board moved between runs, so fit one board pose per run (the mount is shared)",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        metavar="RUN:N",
        help="With --resolve: leave out sample N of a run, e.g. 20261111_153000:3",
    )
    parser.add_argument("--no-write", action="store_true", help="With --resolve: print the fit, write nothing")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.resolve:
        resolve_main(args)
        return
    if args.smoke:
        args.demo_pair = True
    app = MountApp(args)
    if args.smoke:
        app.run_smoke()
    else:
        app.run_gui()


if __name__ == "__main__":
    main()
