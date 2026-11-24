#!/usr/bin/env python3
"""Click-to-3D: park the arm, calibrate cameras to the board, click a point.

Keys:
  c      calibrate both cameras to the board
  f      corner clicks: snap to detected corners / manual (--manual-corners starts manual)
  m      asset menu (fisheye / stationary / URDF / mount)
  u      launch mount calibration (separate process)
  v      launch 3D scene visualiser
  l      tripod grid overlay on / off (default on): the board grid c's tripod pose predicts,
         a1 / h8 labelled. Off the real lines = a bad c pose or a moved tripod: press c
  space  click on the views frozen at c (the arm may move between clicks). With --fk-repose
         (and --robot): click on a fresh wrist frame instead, posed from c + forward kinematics
  r      resume live preview
  y / n  after a wrist click: on the board / above the board
  g      gold-mask (Asha's): type a square name, segment the piece with SAM on both views,
         then p inside it gives the grasp point as a click (needs requirements-sam.txt)
  o      compensation for a and k: OFF -> BASELINE (hand-measured offsets only, k only) -> ON
  k      pick the piece at the click before last, place it on the last click's square centre,
         at the positions from the route b is on (ruler or cv); the run's metadata records it
  a      move the gripper to the last click (prompts in the terminal; --compensate, --above-mm)
  b      base_mm (a's target, and both of k's points) from: the ruler-measured board centre /
         the camera-detected board, cv (wrist camera c pose + kinematics + mount) (--base)
  e      save the arm's current joints as the view pose (--robot; replacing one takes e twice)
  w      drive to the view pose (no prompt): path checked clear of the board by FK first,
         refused if not; turns torque on if the arm is released; always friction/sag-
         compensated (o doesn't affect it). See view_pose.py
  q      quit

Gold-mask (from ready, after c), as in Asha's click_to_3d:
  g        ask for the square name (poses reused; no corner clicks)
  Enter    accept the name: crop both views to that square's prism
  LMB/RMB  foreground / background point on the panel under the cursor
  u / r    undo / reset points on the active view
  t        toggle the active view (wrist / tripod)
  Enter/s  save the active view's mask under output/gold_masks/
  p        save both masks, run the two-view grasp, and use its point as the last click
  Esc      back to ready
The wrist view is c's frame, or with --fk-repose a fresh frame posed by FK (crop grown
FK_CROP_MARGIN_MM); the tripod view is a fresh frame, frozen while you segment.

Use --demo-pair for offline stills. Use --smoke for a headless math check.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import shutil
import tempfile
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from click3d_shared import (
    BOARD_BASE_SOURCE,
    BOARD_CENTRE_BASE_MM,
    CV_BASE_SOURCE,
    ROOT,
    SESSIONS_DIR,
    SQUARE_MM,
    board_to_base_mm,
    T_base_board_mm,
    algebraic_from_xy,
    append_session_log,
    arm_held,
    arm_state_text,
    detect_grid_for_view,
    draw_backup_banner,
    draw_board_overlay,
    draw_detected_corners,
    draw_ray_polyline,
    ensure_output_dirs,
    hide_jpeg_warnings,
    list_asset_candidates,
    load_demo_board_poses,
    load_mount,
    camera_label,
    close_arm,
    corner_click_mode_line,
    joint_offsets_deg,
    load_session_cameras,
    load_stationary_matrix,
    open_camera,
    panel_text,
    parse_camera,
    parse_square_name,
    pixel_ray_board,
    point_board_to_base,
    pose_from_square_corners,
    project_points,
    project_ray_polyline,
    ray_plane_z0,
    ray_ray_midpoint,
    read_joints_json,
    read_joints_robot,
    resolve_core_assets,
    resolve_demo_pair,
    resolve_label_click,
    save_session_cameras,
    snap_click_to_grid,
    toggle_arm_hold,
    snap_ray_click,
    try_auto_pose,
    undistort_wrist,
    load_fisheye_calibration,
    ResolvedAsset,
    T_base_camera_mm,
)
from calib_board_pose import camera_origin_board, invert, rotation_angle_deg  # lib/, on sys.path via click3d_shared
from move_to_click import MoveRefused, move_to_click, pick_piece, pick_targets, plan_move
from view_pose import (
    VIEW_POSE_PATH,
    ViewMoveRefused,
    go_to_view_pose,
    load_view_pose,
    plan_view_move,
    save_view_pose,
)

# Asha's gold-mask mode (g). Optional: its two-view grasp needs scikit-learn at import and SAM
# (torch + transformers) when you click, so without requirements-sam.txt the rest still runs.
try:
    from gold_mask_session import (
        GoldViewState,
        PANEL_W as GOLD_PANEL_W,
        add_point,
        build_square_crop,
        compose_crop_panel,
        gold_save_dir,
        map_panel_click,
        new_session_id,
        rerun,
        run_gold_two_view,
        save_gold,
    )

    GOLD_IMPORT_ERROR: Exception | None = None
except ImportError as exc:
    GOLD_IMPORT_ERROR = exc
    GOLD_PANEL_W = 640

WINDOW = "click_to_3d_oliver"
# The last c calibration: loaded at start, overwritten by the next c.
CALIBRATION_DIR = SESSIONS_DIR / "calibration"
# --fk-repose: each space's wrist frame, joints and pose, kept so FK's error can be measured later.
FK_REPOSE_DIR = SESSIONS_DIR / "fk_repose"
# Two-view clicks (n) whose wrist and tripod rays pass further apart than this get a warning.
# A starting guess, not a measured limit: FK alone was ~5 mm (worst 12-15 mm) in the
# 2026-11-19 offline test (fk_transfer_accuracy.py), and a click adds its own pixel error.
RAY_GAP_WARN_MM = 10.0
# Gold-mask on an FK-posed wrist frame: grow the wrist crop by this much on every side of the
# square, so FK's error (worst 12-15 mm offline) doesn't cut the piece out of the crop.
FK_CROP_MARGIN_MM = 15.0
# Tripod videos of a / k moves whose run folder isn't known (e.g. the move raised).
VIDEOS_DIR = ROOT / "output" / "videos"


class TripodRecorder:
    """Writes the tripod view to an mp4 in real time: the window refreshes at an uneven rate,
    so each frame is repeated (or skipped) to keep a steady FPS."""

    FPS = 30.0

    def __init__(self, path: Path) -> None:
        self.path = path
        self.writer: cv2.VideoWriter | None = None
        self.start = time.monotonic()
        self.written = 0

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.FPS, (width, height))
        due = int((time.monotonic() - self.start) * self.FPS) + 1
        while self.written < due:
            self.writer.write(frame)
            self.written += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
PANEL_W = 480


def shown_path(path: Path) -> str:
    """A saved file's path for the log and status: relative to this folder when it's inside it."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


class App:
    def __init__(self, args: argparse.Namespace) -> None:
        ensure_output_dirs()
        self.args = args
        self.report, assets = resolve_core_assets()
        self.fisheye = assets["fisheye"]
        self.stationary_intr = assets["stationary"]
        self.urdf = assets["urdf"]
        self.K_side = load_stationary_matrix(self.stationary_intr.path)
        self.mount = load_mount(self.urdf, self.report)
        self.demo = None
        self.cap_wrist = None
        self.cap_side = None
        self.joints: list[float] | None = None
        # The wrist pose for the wrist image on screen (frozen_wrist, taken at frozen_joints).
        self.T_board_wrist = None
        self.T_board_side = None
        # (wrist, side, joints) from c, and the wrist pose c solved for that image. Without
        # --fk-repose every click is made on this view, however the arm has moved since. With it,
        # space takes a new wrist frame and poses it from these plus forward kinematics.
        self.cal_view: tuple[np.ndarray, np.ndarray, list[float]] | None = None
        self.T_board_wrist_c = None
        self.wrist_pose_source = "c"  # or "fk_repose": where T_board_wrist came from
        self.repose_info: dict | None = None  # how far the arm and camera moved since c
        self.move_thread: threading.Thread | None = None  # a running a / k move
        self.move_result = None  # its MoveResult, None if it raised
        self.recorder: TripodRecorder | None = None  # films the tripod view during a --robot move
        self.side_view: np.ndarray | None = None  # the last tripod frame shown, markers drawn
        self.show_side_grid = True  # l: the board grid c's tripod pose predicts, on the tripod view
        self.mode = "live"
        self.status = "press c to calibrate"
        self.help_lines: list[str] = []
        self.frozen_wrist = None
        self.frozen_wrist_raw = None
        self.frozen_side = None
        self.frozen_joints = None
        self.K_wrist = None
        self.fisheye_calib = None
        self.wrist_click = None
        self.polyline = None
        self.last_result = None
        self.prev_result = None  # the click before last_result: p picks from it to last_result
        self.label_points: list[tuple[float, float]] = []
        self.label_square: str | None = None
        self.square_buffer = ""
        self.wrist_grid = None
        self.wrist_grid_source = "none"
        self.side_grid = None
        self.side_grid_source = "none"
        self.menu_category = "fisheye"
        self.menu_index = 0
        self.menu_categories = ["fisheye", "stationary", "urdf", "mount"]
        self.manual_corners = args.manual_corners
        # Gold-mask mode (g): square being segmented, SAM state per view, panel layouts.
        self.gold_session_id: str | None = None
        self.gold_square: str | None = None
        self.gold_square_buffer = ""
        self.gold_active: str = "wrist"
        self.gold_wrist = None  # GoldViewState
        self.gold_side = None  # GoldViewState
        self.gold_layouts: dict = {}
        self.view_save_armed_at = -1e9  # when e last asked to confirm replacing the view pose
        self.wrist_camera, self.stationary_camera = load_session_cameras(
            getattr(args, "wrist_camera", None),
            getattr(args, "stationary_camera", None),
        )
        args.wrist_camera = self.wrist_camera
        args.stationary_camera = self.stationary_camera
        if args.demo_pair:
            self.demo = resolve_demo_pair(self.report)
            self.joints = read_joints_json(self.demo["joints"].path)
            wrist_raw = cv2.imread(str(self.demo["wrist"].path))
            side = cv2.imread(str(self.demo["stationary"].path))
            if wrist_raw is None or side is None:
                raise FileNotFoundError("demo pair images missing")
            self.frozen_wrist_raw = wrist_raw.copy()
            self.frozen_wrist, self.K_wrist, self.fisheye_calib = undistort_wrist(
                wrist_raw, self.fisheye.path
            )
            self.frozen_side = side.copy()
            self.frozen_joints = list(self.joints)
            if args.use_saved_poses:
                self.T_board_wrist, self.T_board_side = load_demo_board_poses(self.demo["poses"].path)
                self.T_board_wrist_c = self.T_board_wrist
                self.cal_view = (self.frozen_wrist, self.frozen_side, list(self.frozen_joints))
                self.label_square = "d6"
                self.status = "demo poses loaded - space to click"
                self.mode = "ready"
        elif args.robot:
            self.open_live_cameras()
            self.joints = read_joints_robot()
            self.arm_state = arm_state_text(arm_held())
            self.load_calibration()
        else:
            self.open_live_cameras()
            self.joints = [0.0] * 6
            self.status = "no --robot: joints are zeros until demo/robot"
            self.load_calibration()

    def banner(self) -> str | None:
        return self.report.banner_text()

    def open_live_cameras(self) -> None:
        if self.cap_wrist is not None:
            self.cap_wrist.release()
        if self.cap_side is not None:
            self.cap_side.release()
        self.cap_wrist = open_camera(self.wrist_camera, "wrist")
        self.cap_side = open_camera(self.stationary_camera, "stationary")
        save_session_cameras(self.wrist_camera, self.stationary_camera)
        self.args.wrist_camera = self.wrist_camera
        self.args.stationary_camera = self.stationary_camera

    def swap_cameras(self) -> None:
        if self.demo is not None:
            self.status = "camera swap is for live cameras only"
            return
        self.wrist_camera, self.stationary_camera = self.stationary_camera, self.wrist_camera
        try:
            self.open_live_cameras()
        except RuntimeError as exc:
            self.wrist_camera, self.stationary_camera = self.stationary_camera, self.wrist_camera
            self.status = f"swap failed: {exc}"
            return
        self.status = (
            f"swapped - mounted left ({camera_label(self.wrist_camera)}), "
            f"stationary right ({camera_label(self.stationary_camera)})"
        )
        self.mode = "live" if self.T_board_wrist is None else "ready"
    
    def grab_live(self) -> tuple[np.ndarray, np.ndarray]:
        if self.demo is not None:
            return self.frozen_wrist.copy(), self.frozen_side.copy()
        ok_w, wrist_raw = self.cap_wrist.read()
        ok_s, side = self.cap_side.read()
        if not ok_w or not ok_s:
            raise RuntimeError("camera read failed")
        self.frozen_wrist_raw = wrist_raw.copy()
        wrist, self.K_wrist, self.fisheye_calib = undistort_wrist(wrist_raw, self.fisheye.path)
        if self.args.robot:
            # Joints read with every frame, so frozen frames and frozen_joints match.
            self.joints = read_joints_robot()
        return wrist, side

    def grab_side(self) -> np.ndarray:
        """A live stationary frame (the demo's still without live cameras)."""
        if self.demo is not None:
            return self.frozen_side.copy()
        ok, side = self.cap_side.read()
        if not ok:
            raise RuntimeError("stationary camera read failed")
        return side

    def calibrate(self) -> None:
        wrist, side = self.grab_live()
        if self.K_wrist is None:
            _, self.K_wrist, self.fisheye_calib = undistort_wrist(wrist, self.fisheye.path)
        self.frozen_wrist, self.frozen_side = wrist, side
        self.frozen_joints = list(self.joints or [])
        self.square_buffer = ""
        self.label_square = None
        self.label_points = []
        # Detect on both views immediately so the user can pick a square that
        # has a complete cell in both cameras. Stationary search is gated to a
        # coarse board AABB so the wrist-cam cable is not treated as a grid.
        self.wrist_grid, self.wrist_grid_source = detect_grid_for_view(
            wrist,
            raw_bgr=self.frozen_wrist_raw,
            fisheye_calib=self.fisheye_calib,
            restrict_to_board=False,
        )
        self.side_grid, self.side_grid_source = detect_grid_for_view(
            side,
            restrict_to_board=True,
        )
        self.mode = "enter_square"
        self.status = "type square name (e.g. d6) then Enter"
        self.help_lines = [
            "Rings = corners on BOTH views - pick a square complete in both",
            "Type file then rank, e.g. d then 6",
            "Enter confirms, Esc clears",
            f"wrist corners: {self.wrist_grid_source}  side: {self.side_grid_source}",
        ]

    def start_wrist_label(self) -> None:
        self.mode = "label_wrist"
        self.label_points = []
        # Both views' corners were detected at c (calibrate), so the rings are already up.
        self.corner_prompt()

    def corner_prompt(self) -> None:
        """Status and help for labelling the corners on the current view."""
        wrist = self.mode == "label_wrist"
        grid, source = (self.wrist_grid, self.wrist_grid_source) if wrist else (self.side_grid, self.side_grid_source)
        self.status = f"{'wrist' if wrist else 'stationary'}: 4 corners of {self.label_square}"
        if grid is not None and not self.manual_corners:
            self.status += f" (corners: {source})"
        self.help_lines = [
            corner_click_mode_line(grid, self.manual_corners),
            "1st click: GREEN = corner nearest a1",
            "then 3 cyan, counter-clockwise in the image",
            "next is toward h, then far corner, then rank 8",
            f"Click on the {'LEFT (wrist)' if wrist else 'RIGHT (stationary)'} image",
        ]

    def finish_wrist_label(self) -> None:
        pts = np.asarray(self.label_points[:4], dtype=np.float64)
        pose = pose_from_square_corners(pts, self.label_square, self.K_wrist)
        if pose is None:
            self.status = "pose rejected - green = a1-near, then the other three counter-clockwise in the image"
            self.label_points = []
            return
        self.T_board_wrist = self.T_board_wrist_c = pose
        self.cal_view = (self.frozen_wrist, self.frozen_side, list(self.frozen_joints or []))
        self.wrist_pose_source, self.repose_info = "c", None
        self.label_points = []
        self.mode = "label_side"
        self.corner_prompt()

    def finish_side_corners(self) -> None:
        pts = np.asarray(self.label_points[:4], dtype=np.float64)
        pose = pose_from_square_corners(pts, self.label_square, self.K_side, upright=True)
        if pose is None:
            self.status = "pose rejected - green = a1-near, then the other three counter-clockwise in the image"
            self.label_points = []
            return
        self.T_board_side = pose
        self.mode = "ready"
        self.status = "calib ok (saved) - space to click"
        self.help_lines = []
        self.save_calibration()

    def save_calibration(self, calib_dir = CALIBRATION_DIR) -> None:
        """Save this c calibration, replacing the last one; the next start loads it."""
        wrist, side, joints = self.cal_view
        calib_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(calib_dir / "wrist.png"), wrist)
        cv2.imwrite(str(calib_dir / "stationary.png"), side)
        calibration = {
            "square": self.label_square,
            "joints": joints,
            # c's pose, which goes with these joints and wrist.png (not a later FK re-pose).
            "T_board_wrist": np.asarray(self.T_board_wrist_c, dtype=np.float64).tolist(),
            "T_board_side": np.asarray(self.T_board_side, dtype=np.float64).tolist(),
            "K_wrist": np.asarray(self.K_wrist, dtype=np.float64).tolist(),
            # The tripod lens T_board_side was solved with; a reload under another one is refused.
            "K_side": np.asarray(self.K_side, dtype=np.float64).tolist(),
        }
        (calib_dir / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")

    def load_calibration(self) -> None:
        """Start from the last saved c calibration (frozen wrist image, joints, both board poses)."""
        path = CALIBRATION_DIR / "calibration.json"
        if not path.exists():
            return
        wrist = cv2.imread(str(CALIBRATION_DIR / "wrist.png"))
        side = cv2.imread(str(CALIBRATION_DIR / "stationary.png"))
        if wrist is None or side is None:
            return
        calibration = json.loads(path.read_text(encoding="utf-8"))
        # The saved wrist image and pose belong to the lens file in use when c was pressed;
        # after a lens change (e.g. the f=1400 -> 541 recalibration) they are stale.
        saved_K = np.asarray(calibration["K_wrist"], dtype=np.float64)
        current_K = np.asarray(load_fisheye_calibration(self.fisheye.path)["K_undistorted"], dtype=np.float64)
        if not np.allclose(saved_K, current_K, atol=0.5):
            self.status = "saved calibration used a different wrist lens file - press c to recalibrate"
            print(f"not loading {path}: saved with wrist f={saved_K[0, 0]:.0f}, lens now f={current_K[0, 0]:.0f}")
            return
        # Same for the tripod: T_board_side only holds for the lens it was solved with. Saves from
        # before K_side was recorded (2026-11-19) used the assumed f=457 C920 lens.
        saved_K_side = np.asarray(calibration.get("K_side", np.zeros((3, 3))), dtype=np.float64)
        if not np.allclose(saved_K_side, self.K_side, atol=0.5):
            self.status = "saved calibration used a different tripod lens file - press c to recalibrate"
            old = f"f={saved_K_side[0, 0]:.0f}" if "K_side" in calibration else "not recorded (f=457 assumed lens)"
            print(f"not loading {path}: saved with tripod {old}, lens now f={self.K_side[0, 0]:.0f}")
            return
        self.T_board_wrist = self.T_board_wrist_c = np.asarray(calibration["T_board_wrist"], dtype=np.float64)
        self.T_board_side = np.asarray(calibration["T_board_side"], dtype=np.float64)
        self.K_wrist = np.asarray(calibration["K_wrist"], dtype=np.float64)
        self.label_square = calibration["square"]
        self.frozen_wrist, self.frozen_side, self.frozen_joints = wrist, side, list(calibration["joints"])
        self.cal_view = (wrist, side, list(calibration["joints"]))
        self.mode = "ready"
        self.status = f"loaded saved calibration ({self.label_square}) - space to click, c to redo"

    def show_c_view(self) -> None:
        """Put c's wrist image, tripod image, joints and wrist pose back on screen."""
        wrist, side, joints = self.cal_view
        self.frozen_wrist, self.frozen_side, self.frozen_joints = wrist, side, list(joints)
        self.T_board_wrist = self.T_board_wrist_c
        self.wrist_pose_source, self.repose_info = "c", None

    def fk_repose_wrist(self, joints: list[float]) -> np.ndarray:
        """Wrist pose in the board for these joints, from c and forward kinematics + mount.

        The board's pose in the robot base comes from c (c's wrist pose and joints) and is held
        fixed; the camera's pose in the base is FK at `joints` times the mount. At c's own joints
        this gives back c's pose. Expect ~5 mm of board error, worst 12-15 mm (offline test,
        fk_transfer_accuracy.py, 2026-11-19).
        """
        _, _, joints_c = self.cal_view
        T_base_board = T_base_board_mm(self.T_board_wrist_c, joints_c, self.mount)
        return invert(T_base_board) @ T_base_camera_mm(joints, self.mount)

    def take_reposed_view(self) -> None:
        """space with --fk-repose: a fresh wrist frame, posed from c + FK at its joints."""
        wrist, side = self.grab_live()  # with --robot this also reads the joints for this frame
        joints = list(self.joints or [])
        _, _, joints_c = self.cal_view
        pose = self.fk_repose_wrist(joints)
        self.frozen_wrist, self.frozen_side, self.frozen_joints = wrist, side, joints
        self.T_board_wrist = pose
        self.wrist_pose_source = "fk_repose"
        self.repose_info = {
            "max_joint_change_deg": round(float(np.max(np.abs(np.subtract(joints[:5], joints_c[:5])))), 2),
            "camera_shift_mm": round(float(np.linalg.norm(camera_origin_board(pose) - camera_origin_board(self.T_board_wrist_c))), 1),
            "camera_turn_deg": round(rotation_angle_deg(pose, self.T_board_wrist_c), 2),
            "photo": self.save_repose_photo(joints, pose),
        }

    def save_repose_photo(self, joints: list[float], pose: np.ndarray) -> str | None:
        """Keep the raw wrist frame behind a re-posed click, so FK's error can be checked later
        (e.g. against the board corners in it). Live robot runs only."""
        if not self.args.robot or self.demo is not None or self.frozen_wrist_raw is None:
            return None
        FK_REPOSE_DIR.mkdir(parents=True, exist_ok=True)
        stem = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        cv2.imwrite(str(FK_REPOSE_DIR / f"{stem}_wrist_raw.png"), self.frozen_wrist_raw)
        _, _, joints_c = self.cal_view
        record = {
            "joint_reading_deg": [round(float(v), 3) for v in joints],
            "joint_reading_deg_at_c": [round(float(v), 3) for v in joints_c],
            "joint_offsets_deg": [round(float(v), 3) for v in joint_offsets_deg()],
            "T_board_wrist_fk_mm": np.round(pose, 6).tolist(),
            "T_board_wrist_c_mm": np.round(np.asarray(self.T_board_wrist_c), 6).tolist(),
            "K_wrist": np.asarray(self.K_wrist, dtype=np.float64).tolist(),
            "fisheye": str(self.fisheye.path),
            "mount": self.mount.kind,
            "mount_yaml": None if self.mount.mount_path is None else str(self.mount.mount_path),
        }
        (FK_REPOSE_DIR / f"{stem}.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        return str((FK_REPOSE_DIR / f"{stem}_wrist_raw.png").relative_to(ROOT))

    # Gold-mask mode (g): Asha's SAM segmentation + two-view grasp, on this fork's views.

    def take_gold_views(self) -> None:
        """Freeze the two frames gold-mask segments, until it ends.

        Wrist: with --fk-repose (and joint readings) a fresh frame posed by FK, otherwise c's
        frame and pose, as space does. Tripod: a fresh frame either way (its pose is fixed).
        """
        if self.args.fk_repose and (self.args.robot or self.demo is not None):
            self.take_reposed_view()  # fresh wrist + tripod frames, FK pose
        else:
            self.show_c_view()
            self.frozen_side = self.grab_side()

    def start_gold_square_prompt(self) -> None:
        """g: ask only for a square name; the poses come from c (and FK)."""
        if GOLD_IMPORT_ERROR is not None:
            self.status = f"gold-mask needs the SAM add-on (pip install -r requirements-sam.txt): {GOLD_IMPORT_ERROR}"
            print(self.status)
            return
        if self.T_board_wrist is None or self.T_board_side is None or self.cal_view is None:
            self.status = "calibrate first (c)"
            return
        self.take_gold_views()
        self.gold_square = None
        # Always start empty so the user must type a name (including demo / d6 calib).
        self.gold_square_buffer = ""
        self.mode = "enter_gold_square"
        self.status = "gold square? type name (e.g. e4) then Enter"
        self.help_lines = [
            "type square name only - poses from calib, no corner clicks",
            "Backspace edit  Esc cancel",
        ]
        if self.label_square:
            self.help_lines.append(f"calib square was {self.label_square} (not auto-selected)")

    def enter_gold_mask(self) -> None:
        """Enter on the square name: crop both frozen views to the square's prism."""
        if self.T_board_wrist is None or self.T_board_side is None or self.frozen_wrist is None:
            self.status = "calibrate first (c)"
            return
        # An FK wrist pose can be ~5-15 mm off, so grow its crop; c's pose and the tripod's are
        # measured, so they keep Asha's window.
        margin = FK_CROP_MARGIN_MM if self.wrist_pose_source == "fk_repose" else 0.0
        wrist_built = build_square_crop(
            self.frozen_wrist, self.gold_square, self.T_board_wrist, self.K_wrist, SQUARE_MM, margin_mm=margin
        )
        side_built = build_square_crop(self.frozen_side, self.gold_square, self.T_board_side, self.K_side, SQUARE_MM)
        if not wrist_built["ok"] or not side_built["ok"]:
            self.status = f"gold crop failed - wrist={wrist_built.get('fail')} side={side_built.get('fail')}"
            self.mode = "enter_gold_square"
            return
        self.gold_session_id = new_session_id()
        self.gold_active = "wrist"
        self.gold_wrist = GoldViewState(view="wrist", crop=wrist_built["crop"], window=tuple(wrist_built["window"]))
        self.gold_side = GoldViewState(view="side", crop=side_built["crop"], window=tuple(side_built["window"]))
        self.gold_layouts = {}
        self.mode = "gold_mask"
        self.status = f"gold mask {self.gold_square} - LMB FG / RMB BG  (session {self.gold_session_id})"
        self.help_lines = [
            "CONTROLS",
            "LMB = FG point   RMB = BG point",
            "u = undo point   r = reset view",
            "t = toggle active view (wrist/side)",
            "Enter / s = save active view mask",
            "p = save both + grasp point -> last click",
            "Esc = cancel gold mode",
            f"saves -> output/gold_masks/{self.gold_session_id}/",
        ]
        if margin:
            self.help_lines.append(f"wrist crop grown {margin:g} mm (FK pose)")
        print(self.status)
        for line in self.help_lines:
            print(f"  {line}")

    def gold_meta(self) -> dict:
        """What each saved gold mask records about how its frames were taken."""
        return {
            "session": self.gold_session_id,
            "square": self.gold_square,
            "calib_square": self.label_square,
            "square_mm": SQUARE_MM,
            "wrist_pose": self.wrist_pose_source,
            "joint_reading_deg_at_photo": [round(float(v), 3) for v in (self.frozen_joints or [])],
            "fk_repose": self.repose_info,
        }

    def run_gold_pipeline(self) -> None:
        """p in gold-mask: save both masks, run the two-view grasp, and report its point like a
        click, so it becomes the last click (a moves to it; k picks from it)."""
        if self.gold_wrist is None or self.gold_side is None or self.gold_square is None or self.gold_session_id is None:
            self.status = "gold pipeline needs session + poses"
            return
        if self.gold_wrist.mask is None or not self.gold_wrist.ok:
            self.status = "refine wrist mask first (need ok mask)"
            return
        if self.gold_side.mask is None or not self.gold_side.ok:
            self.status = "refine side mask first (need ok mask)"
            return
        result = run_gold_two_view(
            wrist_bgr=self.frozen_wrist,
            side_bgr=self.frozen_side,
            wrist_state=self.gold_wrist,
            side_state=self.gold_side,
            square=self.gold_square,
            T_board_wrist=self.T_board_wrist,
            T_board_side=self.T_board_side,
            K_wrist=self.K_wrist,
            K_side=self.K_side,
            square_mm=SQUARE_MM,
        )
        if not result["ok"]:
            self.status = f"gold two-view failed: {result.get('fail')}"
            print(self.status)
            return
        gold_dirs = []
        for state in (self.gold_wrist, self.gold_side):
            dest = gold_save_dir(self.gold_session_id, self.gold_square, state.view)
            save_gold(dest, state, self.gold_meta())
            gold_dirs.append(shown_path(dest))
            print(f"saved {shown_path(dest)}")
        skew = result.get("skew_residual_mm")
        self.wrist_click = tuple(result["wrist_px"]) if result.get("wrist_px") else None  # wrist mask centre
        self.report_point(
            np.asarray(result["board_mm"], dtype=np.float64),
            method="gold_two_view",
            ray_gap_mm=skew,
            side_px=result.get("side_px"),
            extra={
                "gold_square": self.gold_square,
                "calib_square": self.label_square,
                "side_used": result.get("side_used"),
                "yaw_deg": result.get("yaw_deg"),
                "shadow_pixels_dropped": result.get("shadow_pixels_dropped"),
                "fuse_mode": result.get("fuse_mode") or "overhead_ray",
                "z_mm": None if result.get("z_mm") is None else round(float(result["z_mm"]), 3),
                "skew_residual_mm": None if skew is None else round(float(skew), 3),
                "foot_filter": result.get("foot_filter"),  # False: skipped for gold masks, see gold_mask_session
                "gold_session": self.gold_session_id,
                "gold_dirs": gold_dirs,
            },
        )
        if skew is not None:
            self.status += f"  skew {skew:.1f} mm"
        if not result.get("side_used"):
            self.status += "  (tripod not used: wrist-only point)"
        self.gold_wrist = self.gold_side = None
        self.gold_layouts = {}

    def exit_gold_mask(self) -> None:
        dirty = any(s is not None and s.dirty for s in (self.gold_wrist, self.gold_side))
        if dirty:
            self.status = "left gold mask (discarded unsaved points)"
            print("Warning: discarding unsaved gold-mask points")
        else:
            self.status = "left gold mask"
        self.mode = "ready"
        self.gold_wrist = self.gold_side = None
        self.gold_layouts = {}
        self.help_lines = []

    def _gold_active_state(self):
        return self.gold_side if self.gold_active == "side" else self.gold_wrist

    def on_gold_mouse(self, event, x, y) -> None:
        """LMB / RMB on the wrist or tripod crop panel: a foreground / background SAM point."""
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        if self.gold_wrist is None or self.gold_side is None:
            return
        if x < GOLD_PANEL_W:
            view, state, px = "wrist", self.gold_wrist, float(x)
        elif x < 2 * GOLD_PANEL_W:
            view, state, px = "side", self.gold_side, float(x - GOLD_PANEL_W)
        else:
            return
        layout = self.gold_layouts.get(view)
        if layout is None:
            return
        mapped = map_panel_click(px, float(y), layout)
        if mapped is None:
            self.status = "click inside the crop"
            return
        self.gold_active = view
        add_point(state, mapped[0], mapped[1], 1 if event == cv2.EVENT_LBUTTONDOWN else 0)
        px_count = int(np.count_nonzero(state.mask)) if state.mask is not None else 0
        self.status = f"{view} points={len(state.points)} px={px_count} ok={state.ok} fail={state.fail}"

    def on_mouse(self, event, x, y, flags, param) -> None:
        if self.mode == "gold_mask":
            self.on_gold_mouse(event, x, y)
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        free = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        if self.mode == "label_wrist" and x < 640:
            grid = None if self.manual_corners else self.wrist_grid
            snapped, err = resolve_label_click(grid, (float(x), float(y)), free_click=free)
            if err is not None:
                self.status = err
                return
            self.label_points.append(snapped)
            if len(self.label_points) >= 4:
                self.label_points = self.label_points[:4]
                self.finish_wrist_label()
            return
        if self.mode == "label_side" and x >= 640:
            grid = None if self.manual_corners else self.side_grid
            snapped, err = resolve_label_click(grid, (float(x - 640), float(y)), free_click=free)
            if err is not None:
                self.status = err
                return
            self.label_points.append(snapped)
            if len(self.label_points) >= 4:
                self.label_points = self.label_points[:4]
                self.finish_side_corners()
            return
        if self.mode == "click_wrist" and x >= 640:
            self.finish_side_on_board((float(x - 640), float(y)))
            return
        if self.mode == "click_wrist" and x < 640:
            self.wrist_click = (float(x), float(y))
            self.mode = "ask_plane"
            self.status = "on the board? y=yes  n=above"
            self.help_lines = []
            return
        if self.mode == "click_side" and x >= 640 and self.polyline is not None:
            snapped, err = snap_ray_click(np.array([x - 640.0, float(y)]), self.polyline, 12.0)
            if err is not None:
                self.status = err
                return
            self.finish_above(snapped)

    def finish_on_board(self) -> None:
        assert self.wrist_click is not None
        origin, direction = pixel_ray_board(self.wrist_click, self.T_board_wrist, self.K_wrist)
        self.report_board_plane(origin, direction, method="board_plane")

    def finish_side_on_board(self, side_px: tuple[float, float]) -> None:
        """A click on the tripod view, taken as on the board surface (no above-board option).

        The tripod view is live, so this sees pieces moved since c; the wrist view is frozen.
        """
        self.wrist_click = None
        origin, direction = pixel_ray_board(side_px, self.T_board_side, self.K_side)
        self.report_board_plane(origin, direction, method="side_board_plane", side_px=side_px)

    def report_board_plane(self, origin, direction, *, method: str, side_px=None) -> None:
        """Report where the click's ray meets the board surface, if that's on the board."""
        foot = ray_plane_z0(origin, direction)
        if foot is None:
            self.status = "ray misses the board plane"
            self.mode = "ready"
            return
        if not (-20 <= foot[0] <= 8 * SQUARE_MM + 20 and -20 <= foot[1] <= 8 * SQUARE_MM + 20):
            self.status = f"hit outside board ({foot[0]:.1f},{foot[1]:.1f}) - retry"
            self.mode = "ready"
            return
        self.report_point(foot, method=method, ray_gap_mm=0.0, side_px=side_px)

    def finish_above(self, side_px: np.ndarray) -> None:
        assert self.wrist_click is not None
        o1, d1 = pixel_ray_board(self.wrist_click, self.T_board_wrist, self.K_wrist)
        o2, d2 = pixel_ray_board((float(side_px[0]), float(side_px[1])), self.T_board_side, self.K_side)
        point, gap = ray_ray_midpoint(o1, d1, o2, d2)
        if float(point[2]) < -1.0:
            self.status = "point is beneath the board - click again on the line"
            return
        self.report_point(point, method="two_view", ray_gap_mm=gap, side_px=side_px)
        if gap > RAY_GAP_WARN_MM:
            # The wrist and tripod rays should meet at the clicked point; a wide gap means a bad
            # wrist pose (or the two clicks weren't on the same point).
            self.status += f"  RAY GAP {gap:.1f} mm (> {RAY_GAP_WARN_MM:g}): check the wrist pose"
            print(self.status)

    def report_point(
        self,
        point_board: np.ndarray,
        *,
        method: str,
        ray_gap_mm: float | None,
        side_px: np.ndarray | tuple[float, float] | list[float] | None = None,
        extra: dict | None = None,
    ) -> None:
        """Make point_board the last click: both base routes, logged. extra adds fields (gold)."""
        # The joints of the wrist frame clicked on: c's, or (--fk-repose) the frame's own.
        joints = self.frozen_joints or self.joints or [0.0] * 6
        joints_c = self.cal_view[2] if self.cal_view is not None else joints
        point_base = board_to_base_mm(point_board)
        # The camera + forward kinematics + mount route. Both are logged; b picks which one is
        # base_mm (the ruler one by default, until the mount is calibrated). After an FK
        # re-pose this is the same board pose c gave, since the pose was built from it.
        T_bb = T_base_board_mm(self.T_board_wrist, joints, self.mount)
        point_base_cv = point_board_to_base(point_board, T_bb)
        square = algebraic_from_xy(float(point_board[0]), float(point_board[1]))
        wrist_px = None  # a tripod click has none
        if self.wrist_click is not None:
            wrist_px = [round(float(self.wrist_click[0]), 2), round(float(self.wrist_click[1]), 2)]
        recorded_side = None
        if side_px is not None:
            recorded_side = [round(float(side_px[0]), 2), round(float(side_px[1]), 2)]
        self.prev_result = self.last_result
        self.last_result = {
            "method": method,
            "wrist_px": wrist_px,
            "side_px": recorded_side,
            "board_mm": [round(float(v), 3) for v in point_board],
            "base_mm": None,  # set by use_base_source: base_mm_ruler or base_mm_cv
            "base_source": None,
            "base_mm_ruler": [round(float(v), 3) for v in point_base],
            "board_centre_base_mm": list(BOARD_CENTRE_BASE_MM),
            "square_mm": SQUARE_MM,
            "base_mm_cv": [round(float(v), 3) for v in point_base_cv],
            # The camera-detected board pose behind base_mm_cv: with b on cv, p maps the piece
            # and the destination square's centre through it (x, y; the height stays 29 mm).
            "T_base_board_cv_mm": np.round(np.asarray(T_bb, dtype=np.float64), 6).tolist(),
            "square": square,
            "ray_gap_mm": None if ray_gap_mm is None else round(float(ray_gap_mm), 3),
            "mount": self.mount.kind,
            "mount_urdf_source": self.mount.urdf_source,
            "joint_reading_deg_at_c": [round(float(v), 3) for v in joints_c],
            "joint_offsets_deg_at_c": [round(float(v), 3) for v in joint_offsets_deg()],
            # Which wrist pose the click went through: "c" (c's frame) or "fk_repose" (a later
            # frame posed from c + FK), with the joints and pose of that frame.
            "wrist_pose": self.wrist_pose_source,
            "joint_reading_deg_at_photo": [round(float(v), 3) for v in joints],
            "T_board_wrist_mm": np.round(np.asarray(self.T_board_wrist, dtype=np.float64), 6).tolist(),
            "fk_repose": self.repose_info,
            **(extra or {}),
        }
        self.use_base_source(self.last_result)
        append_session_log(self.last_result)
        print(json.dumps(self.last_result, indent=2))
        self.status = f"board {self.last_result['board_mm']}  base ({self.args.base}) {self.last_result['base_mm']}  {square}"
        self.mode = "ready"
        self.polyline = None
        self.help_lines = []

    def use_base_source(self, result: dict | None) -> None:
        """Point a click's base_mm / base_source at the route chosen with b (args.base)."""
        if not result or "base_mm_ruler" not in result:
            return  # a click logged before both routes were stored
        cv = self.args.base == "cv"
        cv_point = result.get("base_mm_cv", result.get("base_mm_fk"))  # _fk: clicks before 2026-11-13
        if cv and cv_point is None:
            return
        result["base_mm"] = list(cv_point if cv else result["base_mm_ruler"])
        result["base_source"] = CV_BASE_SOURCE if cv else BOARD_BASE_SOURCE

    def toggle_base_source(self) -> None:
        """b: switch base_mm between the ruler board centre and camera + kinematics + mount.

        The last two clicks switch too (both routes are stored with each), so g can go to the
        same click either way without clicking again. The session log keeps what was logged.
        """
        self.args.base = "cv" if self.args.base == "ruler" else "ruler"
        for result in (self.prev_result, self.last_result):
            self.use_base_source(result)
        self.status = f"base_mm from {self.base_source_label()}"
        if self.last_result and "base_mm_ruler" in self.last_result:
            diff = np.subtract(self.last_result.get("base_mm_cv", self.last_result.get("base_mm_fk")), self.last_result["base_mm_ruler"])
            self.status += f"; last click cv - ruler = {np.round(diff, 1).tolist()} mm"

    def wrist_pose_label(self) -> str:
        mode = "fk-repose on" if self.args.fk_repose else "fk-repose off"
        if self.wrist_pose_source == "fk_repose" and self.repose_info:
            info = self.repose_info
            return (f"wrist pose: FK, cam {info['camera_shift_mm']:.0f} mm / "
                    f"{info['camera_turn_deg']:.0f} deg from c [{mode}]")
        return f"wrist pose: c [{mode}]"

    def base_source_label(self) -> str:
        return "camera-detected board (cv)" if self.args.base == "cv" else "measured board centre (ruler)"

    def launch_mount(self) -> None:
        script = ROOT / "run_mount_calibration.py"
        cmd = [sys.executable, str(script)]
        if self.args.demo_pair:
            cmd.append("--demo-pair")
        if self.args.robot:
            cmd.append("--robot")
        if self.manual_corners:
            cmd.append("--manual-corners")
        cmd.extend(["--wrist-camera", self.wrist_camera, "--stationary-camera", self.stationary_camera])
        print("launching", " ".join(cmd))
        # Free the serial port for the mount tool; reopened on the next joint read.
        close_arm()
        # And the cameras: a second process can't open one this process holds.
        live = self.demo is None
        if live:
            for cap in (self.cap_wrist, self.cap_side):
                if cap is not None:
                    cap.release()
        code = subprocess.call(cmd)
        if live:
            self.open_live_cameras()
        if self.args.robot:
            self.arm_state = arm_state_text(arm_held())
        self.mount = load_mount(self.urdf, self.report)
        if code == 0:
            self.status = f"mount reloaded: {self.mount.kind}"
        else:
            self.status = f"mount calibration failed (exit {code}, see terminal); mount: {self.mount.kind}"

    def run_in_background(self, label: str, move, needs_c: bool = True) -> None:
        """Run an a / k / w move in a thread so the window, and its live tripod view, keep updating.

        Until it finishes, keys are ignored and the window shows the wrist view frozen at c
        beside the live tripod view; neither reads the arm, whose serial port the move uses.
        needs_c=False (w) runs before any c too, on a wrist frame frozen now.
        """
        if self.cal_view is not None:
            # The arm's serial port is the move's, so no fresh frame + joints: show c's view,
            # with c's pose (the last FK re-pose belonged to the frame it replaces).
            self.show_c_view()
        elif needs_c:
            self.status = "calibrate first (c)"
            return
        elif self.frozen_wrist is None:
            # No c yet: freeze a frame now, so the window doesn't read joints during the move.
            self.frozen_wrist, self.frozen_side = self.grab_live()
            self.frozen_joints = list(self.joints or [])
        self.mode = "ready"
        self.status = f"{label}: running, see the terminal (keys ignored until done)"
        self.move_result = None

        def start_recording() -> None:
            # Called by the move just before the arm moves, so the video skips the planning.
            if self.args.robot and self.demo is None:
                self.recorder = TripodRecorder(VIDEOS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{label.lower()}.mp4")

        def work() -> None:
            try:
                result = move(start_recording)
            except Exception as exc:
                traceback.print_exc()
                self.status = f"{label} failed: {exc}"
                return
            self.move_result = result
            if result.arm_moved:
                self.arm_state = arm_state_text(arm_held())
            self.status = result.message

            if result.output_dir is not None and result.output_dir.exists():
                self.save_calibration(self.move_result.output_dir / "calibration")

        self.move_thread = threading.Thread(target=work, daemon=True)
        self.move_thread.start()

    def finish_recording(self) -> None:
        """Called from the window loop once the move thread ends: keep the video only if the arm
        may have moved, in the run's folder when the move says where that is."""
        recorder, self.recorder = self.recorder, None
        recorder.close()
        result = self.move_result
        if result is not None and not result.arm_moved:
            recorder.path.unlink(missing_ok=True)
            return
        path = recorder.path
        if result is not None and result.output_dir is not None and path.exists():
            path = Path(shutil.move(str(path), str(result.output_dir / "tripod.mp4")))
        print(f"tripod video: {path}")

    def move_to_last_click(self) -> None:
        if self.last_result is None:
            self.status = "click a point first (space)"
            return
        if self.mode not in ("live", "ready"):
            self.status = "finish the current step before moving (a)"
            return
        if self.args.baseline:
            self.status = "a has no baseline mode (probe_xy_pose): press o for OFF or ON"
            return
        click, above_mm, compensate, robot = self.last_result, self.args.above_mm, self.args.compensate, self.args.robot
        self.run_in_background(
            "MOVE",
            lambda on_move: move_to_click(click, above_mm=above_mm, compensate=compensate, robot=robot, on_move=on_move),
        )

    def move_to_view_pose(self) -> None:
        """w: drive to the saved view pose, after a clearance check (no prompt)."""
        if self.mode not in ("live", "ready"):
            self.status = "finish the current step before moving (w)"
            return
        urdf, robot, demo_joints = self.urdf.path, self.args.robot, None if self.args.robot else self.joints
        self.run_in_background(
            "VIEW",
            # Always compensated, whatever o is set to (o only affects a and k).
            lambda _on_move: go_to_view_pose(urdf, robot=robot, current_deg=demo_joints, compensate=True),
            needs_c=False,
        )

    def save_view_pose_here(self) -> None:
        """e: save the arm's current joints as the view pose (it may be held or hand-supported).

        Replacing a saved pose takes a second e within 3 s, so a stray key can't move the pose
        c was taken at.
        """
        if not self.args.robot:
            self.status = "e needs --robot (it saves the arm's joint readings)"
            return
        if self.move_thread is not None and self.move_thread.is_alive():
            return
        now = time.monotonic()
        if load_view_pose() is not None and now - self.view_save_armed_at > 3.0:
            self.view_save_armed_at = now
            self.status = "a view pose is already saved: press e again within 3 s to replace it"
            return
        self.view_save_armed_at = -1e9
        joints = read_joints_robot()
        _record, lowest = save_view_pose(joints, self.urdf.path)
        verdict = "w will refuse it, too low" if lowest.margin_mm < 0 else "press c here next"
        self.status = (f"view pose saved ({shown_path(VIEW_POSE_PATH)}): tightest {lowest.frame} "
                       f"{lowest.margin_mm:+.0f} mm; {verdict}")
        print(self.status)
        print(f"  joints {np.round(joints, 2).tolist()}; {lowest.describe()}")

    def compensation_label(self) -> str:
        if self.args.baseline:
            return "BASELINE (hand-measured offsets, k only)"
        return "ON" if self.args.compensate else "OFF"

    def pick_piece_at_click(self) -> None:
        if self.prev_result is None or self.last_result is None:
            self.status = "click the piece, then its destination square, then k"
            return
        if self.mode not in ("live", "ready"):
            self.status = "finish the current step before picking (k)"
            return
        piece, square, robot = self.prev_result, self.last_result, self.args.robot
        compensate, baseline = self.args.compensate, self.args.baseline
        self.run_in_background(
            "PICK",
            lambda on_move: pick_piece(
                piece, square, compensate=compensate, baseline=baseline, robot=robot, on_move=on_move
            ),
        )
        


    def launch_visualizer(self) -> None:
        if self.T_board_wrist is None or self.T_board_side is None or self.K_wrist is None:
            self.status = "calibrate first (c)"
            return
        payload = {
            "square_mm": SQUARE_MM,
            "T_board_wrist": np.asarray(self.T_board_wrist, dtype=np.float64).tolist(),
            "T_board_side": np.asarray(self.T_board_side, dtype=np.float64).tolist(),
            "K_wrist": np.asarray(self.K_wrist, dtype=np.float64).tolist(),
            "K_side": np.asarray(self.K_side, dtype=np.float64).tolist(),
            "point_board_mm": None if self.last_result is None else self.last_result["board_mm"],
            "label_square": self.label_square,
        }
        path = Path(tempfile.gettempdir()) / "click3d_scene.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        cmd = [sys.executable, str(ROOT / "run_scene_visualizer.py"), "--scene", str(path)]
        print("launching", " ".join(cmd))
        subprocess.Popen(cmd)

    def open_asset_menu(self) -> None:
        self.mode = "asset_menu"
        self.menu_category = "fisheye"
        self.menu_index = 0
        self.status = "asset menu - Tab category, j/k select, Enter apply, Esc back"
        self.help_lines = []

    def apply_menu_selection(self) -> None:
        choices = list_asset_candidates()
        options = choices.get(self.menu_category, [])
        if not options:
            self.status = f"no options for {self.menu_category}"
            return
        index = int(np.clip(self.menu_index, 0, len(options) - 1))
        chosen = options[index]
        if self.menu_category == "fisheye":
            self.fisheye = ResolvedAsset("wrist_fisheye.yaml", chosen, "cli")
            self.status = f"fisheye -> {chosen.name}"
        elif self.menu_category == "stationary":
            self.stationary_intr = ResolvedAsset("stationary_intrinsics.yaml", chosen, "cli")
            self.K_side = load_stationary_matrix(chosen)
            self.status = f"stationary K -> {chosen.name}"
        elif self.menu_category == "urdf":
            self.urdf = ResolvedAsset("so101_new_calib_wrist_cam.urdf", chosen, "cli")
            self.mount = load_mount(self.urdf, self.report)
            self.status = f"urdf -> {chosen.name}"
        elif self.menu_category == "mount":
            if chosen.name == "__none__":
                self.mount = load_mount(self.urdf, self.report, force_estimate=True)
                self.status = "mount -> URDF estimate"
            else:
                self.mount = load_mount(self.urdf, self.report, mount_yaml=chosen)
                self.status = f"mount -> {chosen.name}"
        print(self.status)
        self.mode = "live" if self.T_board_wrist is None else "ready"

    def compose_gold(self) -> np.ndarray:
        """Gold-mask window: the wrist and tripod crops (640 x 480 each) and the panel."""
        sq = self.gold_square or "-"
        wrist, lay_w = compose_crop_panel(self.gold_wrist, f"wrist {sq}", self.gold_active == "wrist")
        side, lay_s = compose_crop_panel(self.gold_side, f"side {sq}", self.gold_active == "side")
        self.gold_layouts = {"wrist": lay_w, "side": lay_s}
        backup = self.banner()
        lines = [
            f"mode: {self.mode}",
            f"status: {self.status}",
            f"gold square: {sq}   calib square: {self.label_square or '-'}",
            f"active: {self.gold_active}   session: {self.gold_session_id or '-'}",
            self.wrist_pose_label(),
            "",
            *self.help_lines,
        ]
        if backup:
            lines.insert(0, backup)
        panel = panel_text(lines, width=PANEL_W, height=480)
        panel = draw_backup_banner(panel, backup)
        return np.hstack([wrist, side, panel])

    def compose(self) -> np.ndarray:
        if self.mode == "gold_mask" and self.gold_wrist is not None and self.gold_side is not None:
            return self.compose_gold()
        frozen_modes = (
            "click_wrist",
            "ask_plane",
            "click_side",
            "label_wrist",
            "label_side",
            "enter_square",
            "enter_gold_square",
            "ready",
            "asset_menu",
        )
        if self.mode in frozen_modes and self.frozen_wrist is not None:
            wrist = self.frozen_wrist.copy()
            if self.mode == "enter_gold_square":
                side = self.frozen_side.copy()  # the tripod frame gold-mask will segment
            else:
                side = self.grab_side()  # the tripod camera is fixed, so its view stays live
        else:
            wrist, side = self.grab_live()
            if self.mode == "live":
                self.frozen_wrist = None
        if self.wrist_click is not None and self.mode in ("ask_plane", "click_side", "ready"):
            cv2.circle(wrist, (int(self.wrist_click[0]), int(self.wrist_click[1])), 6, (0, 255, 255), 2)
        # The grid c's tripod pose predicts: off lines (or a1 in the wrong corner) mean a bad
        # pose or a moved tripod. Hidden while c is being clicked, which replaces the pose.
        if (
            self.show_side_grid
            and self.T_board_side is not None
            and self.mode not in ("enter_square", "label_wrist", "label_side")
        ):
            draw_board_overlay(side, self.T_board_side, self.K_side)
        if self.mode == "click_side" and self.polyline is not None:
            draw_ray_polyline(side, self.polyline)
        # Rings on both views from c onward, so a square complete in both can be picked.
        if self.mode in ("enter_square", "label_wrist", "label_side") and not self.manual_corners:
            draw_detected_corners(wrist, self.wrist_grid)
            draw_detected_corners(side, self.side_grid)
        if self.mode == "label_wrist":
            for i, (px, py) in enumerate(self.label_points):
                color = (0, 255, 0) if i == 0 else (255, 200, 0)
                cv2.circle(wrist, (int(px), int(py)), 5, color, -1)
        if self.mode == "label_side":
            for i, (px, py) in enumerate(self.label_points):
                color = (0, 255, 0) if i == 0 else (255, 200, 0)
                cv2.circle(side, (int(px), int(py)), 5, color, -1)
        if self.T_board_wrist is not None and self.T_board_side is not None:
            # k's two points: 1 = piece (click before last, orange), 2 = destination (last click, white).
            for result, label, color in ((self.prev_result, "1", (0, 165, 255)), (self.last_result, "2", (255, 255, 255))):
                if result is None:
                    continue
                pt = np.asarray(result["board_mm"], dtype=np.float64).reshape(1, 3)
                for image, pose, K in (
                    (wrist, self.T_board_wrist, self.K_wrist),
                    (side, self.T_board_side, self.K_side),
                ):
                    pix, depth = project_points(pt, pose, K)
                    if depth[0] > 1:
                        x, y = int(pix[0, 0]), int(pix[0, 1])
                        cv2.circle(image, (x, y), 5, color, -1)
                        cv2.putText(image, label, (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        self.side_view = side
        backup = self.banner()
        lines = [
            f"mode: {self.mode}",
            f"status: {self.status}",
            f"square: {self.label_square or '-'}",
            f"mount: {self.mount.kind} ({self.mount.urdf_source})  assets: {'BACKUP' if backup else 'local'}",
            f"cams: wrist={camera_label(self.wrist_camera)} side={camera_label(self.stationary_camera)}",
            f"c calib  m assets  u mount  v 3d  g gold{'' if GOLD_IMPORT_ERROR is None else ' (n/a)'}",
            f"l tripod grid: {'on' if self.show_side_grid else 'off'}",
            "x swap  space click  r live  q quit",
            f"e save view pose  w go to view pose ({'saved' if VIEW_POSE_PATH.is_file() else 'none saved'})",
            f"f corner clicks: {'MANUAL' if self.manual_corners else 'snap to detected'}",
            f"a move to click (compensate {'on' if self.args.compensate else 'off'}, +{self.args.above_mm:g} mm)",
            "k pick: click before last -> last click's square",
            f"o compensation: {self.compensation_label()}",
            f"b base_mm from: {self.base_source_label()}",
            self.wrist_pose_label(),
        ]
        if self.args.robot:
            lines.append(f"arm: {self.arm_state}  h hold/release")
        if self.mode == "enter_square":
            lines.append(f"buffer: {self.square_buffer or '_'}")
        if self.mode == "enter_gold_square":
            lines.append(f"gold buffer: {self.gold_square_buffer or '_'}")
        lines.extend(self.help_lines)
        if self.mode == "asset_menu":
            lines.append(f"category: {self.menu_category}")
            options = list_asset_candidates().get(self.menu_category, [])
            for i, path in enumerate(options[:9]):
                mark = ">" if i == self.menu_index else " "
                label = "URDF estimate (none)" if path.name == "__none__" else path.name
                lines.append(f"{mark} {i + 1}. {label}")
        if backup:
            lines.insert(0, backup)
        if self.prev_result and self.last_result:
            lines.append(f"k ({self.args.base}): piece 1 ({self.prev_result['square']}) -> square 2 ({self.last_result['square']} centre)")
        if self.last_result:
            lines.append(f"board {self.last_result['board_mm']}")
            if "base_mm_ruler" in self.last_result:
                for name in ("ruler", "cv"):
                    mark = ">" if name == self.args.base else " "
                    lines.append(f"{mark}base {name:5s} {self.last_result[f'base_mm_{name}']}")
            else:
                lines.append(f"base  {self.last_result['base_mm']}")
        panel = panel_text(lines, width=PANEL_W, height=480)
        panel = draw_backup_banner(panel, backup)
        return np.hstack([wrist, side, panel])

    def handle_gold_key(self, key: int) -> bool:
        """Keys inside gold-mask mode (Asha's bindings)."""
        if key == ord("q"):
            return False
        if key == 27:
            self.exit_gold_mask()
            return True
        state = self._gold_active_state()
        if key == ord("u"):
            if state is not None:
                state.undo()
                rerun(state)
                self.status = f"{self.gold_active} undo -> {len(state.points)} points"
        elif key == ord("r"):
            if state is not None:
                state.reset_points()
                self.status = f"{self.gold_active} reset"
        elif key == ord("t"):
            self.gold_active = "side" if self.gold_active == "wrist" else "wrist"
            self.status = f"active view: {self.gold_active}"
        elif key == ord("p"):
            self.run_gold_pipeline()
        elif key in (13, 10, ord("s")):
            if state is None or self.gold_session_id is None or not self.gold_square:
                return True
            if not state.points:
                self.status = "nothing to save - add FG clicks"
            elif state.mask is None or not state.ok:
                self.status = "mask not ok - refine before saving"
            else:
                dest = gold_save_dir(self.gold_session_id, self.gold_square, state.view)
                save_gold(dest, state, self.gold_meta())
                self.status = f"saved {shown_path(dest)}"
                print(self.status)
        return True

    def handle_gold_square_key(self, key: int) -> bool:
        """Typing the gold-mask square name."""
        if key == ord("q"):
            return False
        if key == 27:
            self.mode = "ready"
            self.gold_square_buffer = ""
            self.status = "cancelled gold square"
            self.help_lines = []
            return True
        if key in (8, 127):
            self.gold_square_buffer = self.gold_square_buffer[:-1]
            return True
        if key in (13, 10):
            name = parse_square_name(self.gold_square_buffer)
            if name is None:
                self.status = "invalid square - try e4"
                self.gold_square_buffer = ""
                return True
            self.gold_square = name
            self.enter_gold_mask()
            return True
        ch = chr(key) if 32 <= key < 127 else ""
        if ch.lower() in "abcdefgh12345678":
            self.gold_square_buffer = ch.lower() if len(self.gold_square_buffer) >= 2 else self.gold_square_buffer + ch.lower()
        return True

    def handle_key(self, key: int) -> bool:
        if self.move_thread is not None and self.move_thread.is_alive():
            self.status = "moving: answer the prompts in the terminal (keys ignored until done)"
            return True
        if self.mode == "gold_mask":
            return self.handle_gold_key(key)
        if self.mode == "enter_gold_square":
            return self.handle_gold_square_key(key)
        if self.mode == "enter_square":
            if key in (ord("q"),):
                return False
            if key == 27:
                self.square_buffer = ""
                self.status = "type square name (e.g. d6) then Enter"
                return True
            if key in (13, 10):
                name = parse_square_name(self.square_buffer)
                if name is None:
                    self.status = "invalid square - try d6"
                    self.square_buffer = ""
                    return True
                self.label_square = name
                self.start_wrist_label()
                return True
            ch = chr(key) if 32 <= key < 127 else ""
            if ch.lower() in "abcdefgh12345678" and len(self.square_buffer) < 2:
                self.square_buffer += ch.lower()
            return True
        if self.mode == "asset_menu":
            if key == 27:
                self.mode = "ready" if self.T_board_wrist is not None else "live"
                self.status = "left asset menu"
                return True
            if key == 9:  # Tab
                idx = self.menu_categories.index(self.menu_category)
                self.menu_category = self.menu_categories[(idx + 1) % len(self.menu_categories)]
                self.menu_index = 0
                return True
            if key in (ord("j"), ord("s")):
                self.menu_index += 1
                return True
            if key in (ord("k"), ord("w")):
                self.menu_index = max(0, self.menu_index - 1)
                return True
            if ord("1") <= key <= ord("9"):
                self.menu_index = key - ord("1")
                self.apply_menu_selection()
                return True
            if key in (13, 10):
                self.apply_menu_selection()
                return True
            if key == ord("q"):
                return False
            return True
        if key in (ord("q"), 27):
            return False
        if key == ord("c"):
            self.calibrate()
        elif key == ord("m"):
            self.open_asset_menu()
        elif key == ord("u"):
            self.launch_mount()
        elif key == ord("v"):
            self.launch_visualizer()
        elif key == ord("l"):
            self.show_side_grid = not self.show_side_grid
            self.status = f"tripod grid overlay {'on' if self.show_side_grid else 'off'}"
        elif key == ord("f"):
            self.manual_corners = not self.manual_corners
            if self.mode in ("label_wrist", "label_side"):
                self.corner_prompt()
            else:
                self.status = f"corner clicks: {'manual' if self.manual_corners else 'snap to detected corners'}"
        elif key == ord("g") and self.mode in ("ready", "live"):
            self.start_gold_square_prompt()
        elif key == ord("a"):
            self.move_to_last_click()
        elif key == ord("k"):
            self.pick_piece_at_click()
        elif key == ord("p"):
            self.status = "pick is k now (and move is a); p runs the grasp inside gold-mask (g)"
        elif key == ord("w"):
            self.move_to_view_pose()
        elif key == ord("e"):  # not Shift+W: OpenCV's Qt window reports it as w
            self.save_view_pose_here()
        elif key == ord("o"):
            # OFF -> BASELINE -> ON -> OFF
            self.args.compensate, self.args.baseline = (
                (False, True) if not (self.args.compensate or self.args.baseline)
                else (True, False) if self.args.baseline
                else (False, False)
            )
            self.status = f"compensation {self.compensation_label()} for a and k"
        elif key == ord("b"):
            self.toggle_base_source()
        elif key == ord("r"):
            self.mode = "live"
            self.status = "live preview"
            self.wrist_click = None
            self.polyline = None
            self.help_lines = []
        elif key == ord("h") and self.args.robot:
            self.arm_state = arm_state_text(toggle_arm_hold())
            self.status = f"arm {self.arm_state}"
        elif key == ord("x") and self.mode in ("live", "ready"):
            self.swap_cameras()
        elif key == ord(" "):
            if self.T_board_wrist is None or self.T_board_side is None or self.cal_view is None:
                self.status = "calibrate first (c)"
            elif self.args.fk_repose and (self.args.robot or self.demo is not None):
                self.take_reposed_view()
                info = self.repose_info
                self.mode = "click_wrist"
                self.status = (
                    f"click on the WRIST (left, new frame, FK pose: joints moved up to "
                    f"{info['max_joint_change_deg']:g} deg since c) or the TRIPOD (right, live: board surface only)"
                )
                self.wrist_click = None
            else:
                self.show_c_view()
                self.mode = "click_wrist"
                self.status = "click on the WRIST (left, frozen at c) or the TRIPOD (right, live: board surface only)"
                if self.args.fk_repose:
                    # Without --robot the joints never change, so a live frame would get c's pose.
                    self.status = "--fk-repose needs --robot (no joint readings): " + self.status
                self.wrist_click = None
        elif key == ord("y") and self.mode == "ask_plane":
            self.finish_on_board()
        elif key == ord("n") and self.mode == "ask_plane":
            o, d = pixel_ray_board(self.wrist_click, self.T_board_wrist, self.K_wrist)
            self.polyline = project_ray_polyline(o, d, self.T_board_side, self.K_side)
            self.mode = "click_side"
            self.status = "click the same point on the LINE (stationary)"
        return True

    def run_gui(self) -> None:
        if self.banner():
            print(self.banner())
        for item in self.report.items:
            print(f"  {item.name}: {item.source} -> {item.path}")
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while True:
            frame = self.compose()
            if self.recorder is not None:
                self.recorder.write(self.side_view)
                if not self.move_thread.is_alive():
                    self.finish_recording()
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(30) & 0xFF
            if key != 255 and not self.handle_key(key):
                break
        cv2.destroyAllWindows()
        if self.cap_wrist is not None:
            self.cap_wrist.release()
        if self.cap_side is not None:
            self.cap_side.release()

    def smoke_pick_targets(self) -> None:
        """p follows b: x, y from the chosen route, in the lookup builder's frame."""
        half = 4.0 * SQUARE_MM
        c, s_ = np.cos(np.radians(2.0)), np.sin(np.radians(2.0))
        T_cv = np.eye(4)
        T_cv[:3, :3] = [[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]
        T_cv[:3, 3] = board_to_base_mm(np.zeros(3)) + [5.0, 12.0, 6.0]  # ruler board turned 2 deg and shifted

        def click(square, board, source):
            board = np.asarray(board, dtype=np.float64)
            return {"square": square, "board_mm": board.tolist(), "base_source": source, "T_base_board_cv_mm": T_cv.tolist()}

        piece, dest = click("d5", [160.0, 200.0, 0.0], CV_BASE_SOURCE), click("e4", [180.0, 130.0, 0.0], CV_BASE_SOURCE)
        t = pick_targets(piece, dest)
        e4_centre = np.array([4.5 * SQUARE_MM, 3.5 * SQUARE_MM, 0.0])
        assert t.source == CV_BASE_SOURCE
        assert np.allclose(t.from_base_xy_mm, point_board_to_base(np.asarray(piece["board_mm"]), T_cv)[:2])
        assert np.allclose(t.to_base_xy_mm, (T_cv @ np.append(e4_centre, 1.0))[:2])
        centre_xy = np.asarray(BOARD_CENTRE_BASE_MM[:2])
        assert np.allclose(centre_xy + np.asarray(t.from_board_xy_m) * 1000.0, t.from_base_xy_mm)
        assert t.record()["target_source"] == CV_BASE_SOURCE
        # ruler: the same lookup inputs p always used
        r = pick_targets({**piece, "base_source": BOARD_BASE_SOURCE}, {**dest, "base_source": BOARD_BASE_SOURCE})
        assert r.source == BOARD_BASE_SOURCE
        assert np.allclose(r.from_board_xy_m, (np.asarray(piece["board_mm"][:2]) - half) / 1000.0)
        assert np.allclose(r.to_board_xy_m, (e4_centre[:2] - half) / 1000.0)
        print("pick targets ok: cv", np.round(t.from_base_xy_mm, 1).tolist(), "vs ruler", np.round(r.from_base_xy_mm, 1).tolist())

    def run_smoke(self) -> None:
        # The smoke clicks and gold masks go to a temp folder, not output/sessions/session_log.jsonl.
        import click3d_shared

        scratch = Path(tempfile.mkdtemp(prefix="click3d_smoke_"))
        click3d_shared.SESSIONS_DIR = scratch / "sessions"
        if GOLD_IMPORT_ERROR is None:
            import gold_mask_session

            gold_mask_session.GOLD_MASKS_DIR = scratch / "gold_masks"
        import view_pose

        view_pose.VIEW_POSE_PATH = scratch / "sessions" / "view_pose.json"
        view_pose.VIEW_MOVES_LOG = scratch / "sessions" / "view_moves.jsonl"
        print("smoke output (session log, gold masks, view pose):", scratch)
        self.args.base = "ruler"  # the checks below start from the ruler route; b is tested too
        if self.demo is None:
            self.demo = resolve_demo_pair(self.report)
            self.joints = read_joints_json(self.demo["joints"].path)
            wrist_raw = cv2.imread(str(self.demo["wrist"].path))
            self.frozen_wrist_raw = wrist_raw.copy()
            self.frozen_wrist, self.K_wrist, self.fisheye_calib = undistort_wrist(
                wrist_raw, self.fisheye.path
            )
            self.frozen_side = cv2.imread(str(self.demo["stationary"].path))
            self.frozen_joints = list(self.joints)
        assert parse_square_name("d6") == "d6"
        assert parse_square_name("zz") is None
        # Snap helper: near a synthetic corner snaps; far click is rejected.
        synth = np.zeros((2, 2, 2), dtype=np.float64)
        synth[0, 0] = [100.0, 200.0]
        synth[0, 1] = [140.0, 200.0]
        synth[1, 0] = [100.0, 240.0]
        synth[1, 1] = [140.0, 240.0]
        near = snap_click_to_grid(synth, (103.0, 198.0), max_dist_px=12.0)
        assert near == (100.0, 200.0)
        assert snap_click_to_grid(synth, (180.0, 180.0), max_dist_px=12.0) is None
        snapped, err = resolve_label_click(synth, (103.0, 198.0))
        assert err is None and snapped == (100.0, 200.0)
        _, err_far = resolve_label_click(synth, (180.0, 180.0))
        assert err_far is not None
        raw_ok, none_err = resolve_label_click(None, (11.0, 22.0))
        assert none_err is None and raw_ok == (11.0, 22.0)
        # Wrist detect: undistorted first, else raw→undistort mapping path.
        wrist_grid, wrist_src = detect_grid_for_view(
            self.frozen_wrist,
            raw_bgr=self.frozen_wrist_raw,
            fisheye_calib=self.fisheye_calib,
        )
        print("wrist grid source:", wrist_src, "corners:", "yes" if wrist_grid is not None else "no")
        self.T_board_wrist, self.T_board_side = load_demo_board_poses(self.demo["poses"].path)
        self.T_board_wrist_c = self.T_board_wrist
        self.cal_view = (self.frozen_wrist, self.frozen_side, list(self.frozen_joints))
        print("SMOKE")
        print("banner:", self.banner())
        print("mount:", self.mount.kind, self.mount.urdf_source)
        centre = np.array([[3.5 * SQUARE_MM, 5.5 * SQUARE_MM, 0.0]], dtype=np.float64)
        pix, depth = project_points(centre, self.T_board_wrist, self.K_wrist)
        assert depth[0] > 1
        self.wrist_click = (float(pix[0, 0]), float(pix[0, 1]))
        self.finish_on_board()
        board = self.last_result["board_mm"]
        assert abs(board[0] - 3.5 * SQUARE_MM) < 8 and abs(board[1] - 5.5 * SQUARE_MM) < 8
        # d6 centre: file d = 3.5 squares, rank 6 = 5.5 squares from a1; the board centre is 4 and 4.
        expected = np.array(BOARD_CENTRE_BASE_MM) + [-0.5 * SQUARE_MM, 1.5 * SQUARE_MM, 0.0]
        assert np.allclose(self.last_result["base_mm"], expected + [board[0] - 3.5 * SQUARE_MM, board[1] - 5.5 * SQUARE_MM, 0.0], atol=1e-3)
        assert self.last_result["base_source"] == BOARD_BASE_SOURCE
        # b: the same click switches to the camera + kinematics route and back.
        ruler = list(self.last_result["base_mm"])
        self.toggle_base_source()
        assert self.last_result["base_source"] == CV_BASE_SOURCE
        assert self.last_result["base_mm"] == self.last_result["base_mm_cv"]
        self.toggle_base_source()
        assert self.last_result["base_source"] == BOARD_BASE_SOURCE and self.last_result["base_mm"] == ruler
        print("base toggle ok: cv - ruler =", np.round(np.subtract(self.last_result["base_mm_cv"], ruler), 1).tolist())
        # Each click keeps the camera-detected board pose behind its base_mm_cv.
        T_click = np.asarray(self.last_result["T_base_board_cv_mm"], dtype=np.float64)
        assert np.allclose(point_board_to_base(np.asarray(self.last_result["board_mm"]), T_click), self.last_result["base_mm_cv"], atol=0.01)
        self.smoke_pick_targets()
        assert self.last_result["side_px"] is None
        assert abs(self.last_result["wrist_px"][0] - self.wrist_click[0]) < 0.02
        assert abs(self.last_result["wrist_px"][1] - self.wrist_click[1]) < 0.02
        o, d = pixel_ray_board(self.wrist_click, self.T_board_wrist, self.K_wrist)
        lifted = np.array([146.125, 229.625, 40.0], dtype=np.float64)
        side_pix, side_depth = project_points(lifted.reshape(1, 3), self.T_board_side, self.K_side)
        assert side_depth[0] > 1
        self.polyline = project_ray_polyline(o, d, self.T_board_side, self.K_side)
        # Under-board rejection
        if self.polyline.get("below") is not None and len(self.polyline["below"]) > 0:
            bad = self.polyline["below"][len(self.polyline["below"]) // 2]
            snapped, err = snap_ray_click(bad, self.polyline, 30.0)
            assert err is not None and "beneath" in err
            print("under-board guard ok:", err)
        self.finish_above(side_pix[0])
        assert self.last_result["method"] == "two_view"
        assert self.last_result["board_mm"][2] > 5
        assert abs(self.last_result["wrist_px"][0] - self.wrist_click[0]) < 0.02
        assert abs(self.last_result["side_px"][0] - float(side_pix[0, 0])) < 0.02
        assert abs(self.last_result["side_px"][1] - float(side_pix[0, 1])) < 0.02
        # Force under-board triangulation rejection
        below_pt = np.array([146.125, 229.625, -20.0])
        side_below, d2 = project_points(below_pt.reshape(1, 3), self.T_board_side, self.K_side)
        if d2[0] > 1:
            before = self.last_result
            self.finish_above(side_below[0])
            # either rejected (status message) or accepted if ray geometry differs
            print("beneath triangulation check status:", self.status)
            self.last_result = before
        # Move planning (no subprocess, no arm). Board surface is BOARD_CENTRE_BASE_MM[2].
        surface = BOARD_CENTRE_BASE_MM[2]
        click = {"base_source": BOARD_BASE_SOURCE, "method": "board_plane", "base_mm": [250.0, -50.0, surface - 2.0], "square": "d6"}
        plan = plan_move(click)
        assert np.allclose(plan.target_mm, [250.0, -50.0, surface]) and plan.height_above_board_mm == 0.0
        assert abs(plan.probe_height_mm - (surface - 24.6)) < 1e-9 and plan.via_height_mm == 120.0
        assert "raised 2.0 mm" in plan.notes[0]
        plan = plan_move({**click, "base_source": CV_BASE_SOURCE})  # b: camera-detected clicks move too
        assert np.allclose(plan.target_mm, [250.0, -50.0, surface])
        plan = plan_move({**click, "method": "two_view", "base_mm": [250.0, -50.0, surface + 100.0]}, above_mm=10.0)
        assert abs(plan.height_above_board_mm - 110.0) < 1e-9
        assert abs(plan.via_height_mm - max(120.0, surface + 110.0 - 24.6 + 50.0)) < 1e-9
        bad_cases = (
            ({**click, "base_source": None}, 0.0),  # old click, from before the source was recorded
            (click, -10.0),  # 12 mm below the surface
            ({**click, "base_mm": [0, 0, surface + 400]}, 0.0),
        )
        for bad, above in bad_cases:
            try:
                plan_move(bad, above)
            except MoveRefused as exc:
                print("move guard ok:", str(exc)[:60])
            else:
                raise AssertionError(f"plan_move accepted {bad} +{above} mm")
        self.smoke_fk_repose()
        self.smoke_gold()
        self.smoke_view_pose()
        print("smoke ok", self.last_result)

    def smoke_view_pose(self) -> None:
        """w's planning, no arm: a clear path from rest to a high pose, and the refusals."""
        import view_pose

        urdf = self.urdf.path
        rest = [96.923, -107.868, 97.363, 65.187, -29.846, 4.630]  # chess_traj.DEFAULT_HOME
        high = [-7.1, -34.2, -14.9, 103.0, -90.5, 1.7]  # over the board, jaw tip ~225 mm up
        low = [float(v) for v in self.cal_view[2]]  # the demo c pose: jaw tip ~116 mm up over the board
        plan = plan_view_move(rest, high, urdf)
        step = view_pose.MAX_SPEED_DEG_S * view_pose.COMMAND_DELAY_S
        poses = [np.asarray(rest)] + plan.path
        assert max(float(np.max(np.abs(b[:5] - a[:5]))) for a, b in zip(poses, poses[1:])) <= step + 1e-9
        assert np.allclose(plan.path[-1][:5], high[:5]) and all(abs(p[5] - rest[5]) < 1e-9 for p in plan.path)
        assert plan.lowest.margin_mm >= 0
        refusals = (
            (rest, low, "itself is too low"),
            (rest, [*high[:4], rest[4] + 190.0, high[5]], "wrist roll"),
            (low, high, "starts too low"),
        )
        for current, target, words in refusals:
            try:
                plan_view_move(current, target, urdf)
            except ViewMoveRefused as exc:
                assert words in str(exc), exc
            else:
                raise AssertionError(f"plan_view_move accepted {target} from {current}")
        # e then w without --robot: saved, planned, nothing sent.
        save_view_pose(high, urdf)
        assert view_pose.load_view_pose()["joint_reading_deg"] == [round(v, 3) for v in high]
        result = view_pose.go_to_view_pose(urdf, robot=False, current_deg=rest)
        assert not result.arm_moved and "dry run" in result.message, result.message
        print(f"view pose ok: rest -> high {plan.order}, {len(plan.path)} steps / {plan.seconds:.1f} s, "
              f"tightest {plan.lowest.frame} {plan.lowest.margin_mm:+.0f} mm; low, roll and low-start refused")

    def smoke_gold(self) -> None:
        """g -> square -> masks -> p, on both the c view and an FK-posed view, with SAM replaced by
        the exact silhouette of a synthetic piece (a cylinder) drawn from the known poses. The
        grasp must land on the cylinder's axis and become the last click."""
        if GOLD_IMPORT_ERROR is not None:
            self.handle_key(ord("g"))
            assert "requirements-sam" in self.status
            print(f"gold smoke skipped (SAM add-on missing: {GOLD_IMPORT_ERROR}); g says so")
            return
        centre = np.array([3.5 * SQUARE_MM, 5.5 * SQUARE_MM])  # d6, the demo's calibration square

        def silhouette(image: np.ndarray, pose: np.ndarray, K: np.ndarray) -> np.ndarray:
            ring = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            pts = np.array([[centre[0] + 12 * np.cos(t), centre[1] + 12 * np.sin(t), z]
                            for t in ring for z in (0.0, 45.0)])
            pix, depth = project_points(pts, pose, K)
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(pix[depth > 1].astype(np.int32)), 255)
            return mask

        def segment(state, full: np.ndarray) -> None:
            x0, y0, x1, y1 = state.window
            state.mask = full[y0:y1, x0:x1].copy()
            state.points, state.labels, state.ok, state.score = [(1.0, 1.0)], [1], True, 1.0

        prev_last = self.last_result
        for repose in (False, True):
            self.args.fk_repose = repose
            if repose:
                self.joints = [float(v) for v in np.add(self.cal_view[2], [3.0, -2.0, 2.0, -3.0, 10.0, 0.0])]
            self.mode = "ready"
            self.handle_key(ord("g"))
            assert self.mode == "enter_gold_square", self.status
            assert self.wrist_pose_source == ("fk_repose" if repose else "c")
            for ch in "d6":
                self.handle_key(ord(ch))
            self.handle_key(13)
            assert self.mode == "gold_mask", self.status
            # An FK pose grows the wrist crop; the tripod crop never changes.
            wx0, wy0, wx1, wy1 = self.gold_wrist.window
            plain = build_square_crop(self.frozen_wrist, "d6", self.T_board_wrist, self.K_wrist, SQUARE_MM)["window"]
            grown = (wx1 - wx0) * (wy1 - wy0) > (plain[2] - plain[0]) * (plain[3] - plain[1])
            assert grown == repose, (self.gold_wrist.window, plain)
            segment(self.gold_wrist, silhouette(self.frozen_wrist, self.T_board_wrist, self.K_wrist))
            segment(self.gold_side, silhouette(self.frozen_side, self.T_board_side, self.K_side))
            # What Asha's grasp gives on the same pasted masks and poses, called directly.
            from gold_mask_session import FOOT_FILTER_FOR_GOLD, grasp_from_views, paste_mask_to_full

            direct = grasp_from_views(
                self.frozen_wrist, self.frozen_side, "d6", {}, self.T_board_wrist, self.T_board_side,
                self.K_wrist, self.K_side, SQUARE_MM,
                overhead_mask=paste_mask_to_full(self.frozen_wrist.shape[:2], self.gold_wrist.mask, self.gold_wrist.window),
                side_mask=paste_mask_to_full(self.frozen_side.shape[:2], self.gold_side.mask, self.gold_side.window),
                fuse_mode="overhead_ray",
                foot_filter=FOOT_FILTER_FOR_GOLD,
            )
            before = self.last_result
            self.handle_key(ord("p"))
            r = self.last_result
            assert self.mode == "ready" and r is not before and r["method"] == "gold_two_view", self.status
            assert self.prev_result is before  # so k can pick from it
            assert r["side_used"] and r["skew_residual_mm"] is not None and len(r["gold_dirs"]) == 2
            assert r["foot_filter"] == FOOT_FILTER_FOR_GOLD  # recorded with the click
            assert r["wrist_pose"] == ("fk_repose" if repose else "c")
            # Plumbing: the click is exactly Asha's grasp point, and both base routes come from it.
            assert np.allclose(r["board_mm"], [direct["x_mm"], direct["y_mm"], direct["z_mm"]], atol=2e-3)
            assert r["base_mm"] == r["base_mm_ruler"]
            # Method, with exact poses and masks: a plain cylinder has no neck, so the pinch height
            # is arbitrary and the two rays cross the axis at different heights (~4 mm here).
            error = float(np.hypot(r["board_mm"][0] - centre[0], r["board_mm"][1] - centre[1]))
            assert error < 10.0 and 0.0 < r["board_mm"][2] < 45.0, r["board_mm"]
            print(f"gold ok ({'fk re-pose' if repose else 'c view'}): board {r['board_mm']}, "
                  f"{error:.1f} mm off the cylinder axis, skew {r['skew_residual_mm']} mm, "
                  f"wrist crop {(wx0, wy0, wx1, wy1)} vs plain {tuple(plain)}")
        # Esc leaves without a click; the top-level p only points at the new keys.
        self.handle_key(ord("g"))
        self.handle_key(27)
        assert self.mode == "ready"
        self.handle_key(ord("p"))
        assert "pick is k" in self.status
        self.args.fk_repose = False
        self.joints = list(self.cal_view[2])
        self.show_c_view()
        assert prev_last is not self.last_result

    def smoke_fk_repose(self) -> None:
        """--fk-repose: c's joints give back c's pose; after a move the camera moves but the board
        stays where c put it, so a click on the same board point lands in the same place."""
        joints_c = list(self.cal_view[2])
        T_c = self.T_board_wrist_c
        assert np.allclose(self.fk_repose_wrist(joints_c), T_c, atol=1e-6)
        moved = [float(v) for v in np.add(joints_c, [12.0, -6.0, 8.0, -10.0, 35.0, 0.0])]
        self.joints = moved  # what grab_live would have read with the new frame (demo: stills)
        self.take_reposed_view()
        pose = self.T_board_wrist
        assert self.wrist_pose_source == "fk_repose" and self.frozen_joints == moved
        shift = self.repose_info["camera_shift_mm"]
        assert shift > 10 and self.repose_info["photo"] is None  # no photos off the robot
        assert np.allclose(T_base_board_mm(pose, moved, self.mount), T_base_board_mm(T_c, joints_c, self.mount), atol=1e-6)
        d4 = np.array([[3.5 * SQUARE_MM, 3.5 * SQUARE_MM, 0.0]])
        pix, depth = project_points(d4, pose, self.K_wrist)
        assert depth[0] > 1
        self.wrist_click = (float(pix[0, 0]), float(pix[0, 1]))
        self.finish_on_board()
        moved_click = self.last_result
        assert moved_click["wrist_pose"] == "fk_repose"
        assert moved_click["joint_reading_deg_at_photo"] == [round(v, 3) for v in moved]
        assert moved_click["joint_reading_deg_at_c"] == [round(float(v), 3) for v in joints_c]
        self.show_c_view()
        pix, _ = project_points(d4, T_c, self.K_wrist)
        self.wrist_click = (float(pix[0, 0]), float(pix[0, 1]))
        self.finish_on_board()
        assert self.last_result["wrist_pose"] == "c" and self.last_result["fk_repose"] is None
        for key in ("board_mm", "base_mm_cv", "base_mm_ruler"):
            assert np.allclose(moved_click[key], self.last_result[key], atol=0.01), key
        # Two-view click whose tripod point is 30 mm off the wrist ray: warned.
        lifted = np.array([[3.5 * SQUARE_MM, 3.5 * SQUARE_MM, 40.0]])
        pix, _ = project_points(lifted, T_c, self.K_wrist)
        self.wrist_click = (float(pix[0, 0]), float(pix[0, 1]))
        side_pix, _ = project_points(lifted + [30.0, 0.0, 0.0], self.T_board_side, self.K_side)
        self.finish_above(side_pix[0])
        assert self.last_result["ray_gap_mm"] > RAY_GAP_WARN_MM and "RAY GAP" in self.status
        self.joints = joints_c
        print(f"fk re-pose ok: camera moved {shift} mm in the moved view; same click board/base either way; "
              f"ray gap warning at {self.last_result['ray_gap_mm']} mm")


def main() -> None:
    hide_jpeg_warnings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-pair", action="store_true", help="Use bundled/local pair 003 stills")
    parser.add_argument("--use-saved-poses", action="store_true", help="Load demo board poses without interactive calib")
    parser.add_argument("--robot", action="store_true", help="Read live joint angles from the SO-101")
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
        help="Stationary camera: index or device path (default: session_config.yaml, then paths.STATIONARY_CAMERA)",
    )
    parser.add_argument(
        "--compensate",
        action="store_true",
        help="Start with compensation on for a and k (calibration offsets + friction/sag model); o toggles",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Start in baseline mode for k (hand-measured offsets only, no friction/sag model); o toggles",
    )
    parser.add_argument(
        "--above-mm",
        type=float,
        default=0.0,
        help="a: put the fixed-jaw tip this far above the clicked point (default 0 = at the point)",
    )
    parser.add_argument(
        "--manual-corners",
        action="store_true",
        help="Start with manual corner clicks (no snapping to detected corners); f toggles",
    )
    parser.add_argument(
        "--base",
        choices=("ruler", "cv"),
        default="ruler",
        help="base_mm (a's target, and k's points) from the ruler-measured board centre (default) or "
        "the camera-detected board (cv: wrist camera c pose + forward kinematics + mount); b toggles",
    )
    parser.add_argument(
        "--fk-repose",
        action="store_true",
        help="space takes a fresh wrist frame and poses it from the c calibration + forward kinematics "
        "+ mount at that frame's joints, so c needn't be redone after the arm moves (needs --robot). "
        "Expect ~5 mm board error, worst 12-15 mm; each frame is saved under output/sessions/fk_repose/",
    )
    parser.add_argument("--smoke", action="store_true", help="Headless verification on demo pair")
    args = parser.parse_args()
    if args.compensate and args.baseline:
        parser.error("--compensate and --baseline are alternatives: pick one")
    if args.smoke:
        args.demo_pair = True
        args.use_saved_poses = True
    app = App(args)
    if args.smoke:
        app.run_smoke()
    else:
        app.run_gui()


if __name__ == "__main__":
    main()
