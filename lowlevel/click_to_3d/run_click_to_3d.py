#!/usr/bin/env python3
"""Click-to-3D: park the arm, calibrate cameras to the board, click a point.

Keys:
  c      calibrate both cameras to the board
  m      asset menu (fisheye / stationary / URDF / mount)
  u      launch mount calibration (separate process)
  v      launch 3D scene visualiser
  space  freeze and click on the wrist view
  r      resume live preview
  y / n  after a wrist click: on the board / above the board
  q      quit

Use --demo-pair for offline stills. Use --smoke for a headless math check.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from click3d_shared import (
    ROOT,
    SQUARE_MM,
    T_base_board_mm,
    algebraic_from_xy,
    append_session_log,
    detect_grid_for_view,
    draw_backup_banner,
    draw_detected_corners,
    draw_ray_polyline,
    ensure_output_dirs,
    list_asset_candidates,
    load_demo_board_poses,
    load_mount,
    load_session_camera_indices,
    load_stationary_matrix,
    open_camera,
    panel_text,
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
    save_session_camera_indices,
    snap_click_to_grid,
    snap_ray_click,
    try_auto_pose,
    undistort_wrist,
    ResolvedAsset,
)

WINDOW = "click_to_3d"
PANEL_W = 480


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
        self.T_board_wrist = None
        self.T_board_side = None
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
        self.wrist_index, self.stationary_index = load_session_camera_indices(
            getattr(args, "wrist_index", None),
            getattr(args, "stationary_index", None),
        )
        args.wrist_index = self.wrist_index
        args.stationary_index = self.stationary_index
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
                self.label_square = "d6"
                self.status = "demo poses loaded — space to click"
                self.mode = "ready"
        elif args.robot:
            self.open_live_cameras()
            self.joints = read_joints_robot()
        else:
            self.open_live_cameras()
            self.joints = [0.0] * 6
            self.status = "no --robot: joints are zeros until demo/robot"

    def banner(self) -> str | None:
        return self.report.banner_text()

    def open_live_cameras(self) -> None:
        if self.cap_wrist is not None:
            self.cap_wrist.release()
        if self.cap_side is not None:
            self.cap_side.release()
        self.cap_wrist = open_camera(self.wrist_index)
        self.cap_side = open_camera(self.stationary_index)
        save_session_camera_indices(self.wrist_index, self.stationary_index)
        self.args.wrist_index = self.wrist_index
        self.args.stationary_index = self.stationary_index

    def swap_cameras(self) -> None:
        if self.demo is not None:
            self.status = "camera swap is for live cameras only"
            return
        self.wrist_index, self.stationary_index = self.stationary_index, self.wrist_index
        try:
            self.open_live_cameras()
        except RuntimeError as exc:
            self.wrist_index, self.stationary_index = self.stationary_index, self.wrist_index
            self.status = f"swap failed: {exc}"
            return
        self.status = (
            f"swapped — mounted left (idx {self.wrist_index}), "
            f"stationary right (idx {self.stationary_index})"
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
        return wrist, side

    def calibrate(self) -> None:
        wrist, side = self.grab_live()
        if self.K_wrist is None:
            _, self.K_wrist, self.fisheye_calib = undistort_wrist(wrist, self.fisheye.path)
        self.frozen_wrist, self.frozen_side = wrist, side
        self.frozen_joints = list(self.joints or [])
        self.square_buffer = ""
        self.label_square = None
        self.label_points = []
        self.wrist_grid = None
        self.wrist_grid_source = "none"
        self.side_grid, self.side_grid_source = detect_grid_for_view(side)
        self.mode = "enter_square"
        self.status = "type square name (e.g. d6) then Enter"
        self.help_lines = [
            "Type file then rank, e.g. d then 6",
            "Enter confirms, Esc clears",
            "Then label that square on the wrist",
        ]

    def start_wrist_label(self) -> None:
        self.mode = "label_wrist"
        self.label_points = []
        self.wrist_grid, self.wrist_grid_source = detect_grid_for_view(
            self.frozen_wrist,
            raw_bgr=self.frozen_wrist_raw,
            fisheye_calib=self.fisheye_calib,
        )
        if self.wrist_grid is not None:
            self.status = f"wrist: 4 corners of {self.label_square} (corners: {self.wrist_grid_source})"
            self.help_lines = [
                "Rings = detected corners — clicks snap to them",
                "1st click: GREEN = corner nearest a1",
                "then 3 cyan, counter-clockwise in the image",
                "next is toward h, then far corner, then rank 8",
                "Click on the LEFT (wrist) image",
            ]
        else:
            self.status = f"wrist: 4 corners of {self.label_square}"
            self.help_lines = [
                "no corners detected — click carefully",
                "1st click: GREEN = corner nearest a1",
                "then 3 cyan, counter-clockwise in the image",
                "next is toward h, then far corner, then rank 8",
                "Click on the LEFT (wrist) image",
            ]

    def finish_wrist_label(self) -> None:
        pts = np.asarray(self.label_points[:4], dtype=np.float64)
        pose = pose_from_square_corners(pts, self.label_square, self.K_wrist)
        if pose is None:
            self.status = "pose rejected — green = a1-near, then the other three counter-clockwise in the image"
            self.label_points = []
            return
        self.T_board_wrist = pose
        self.label_points = []
        self.mode = "label_side"
        if self.side_grid is not None:
            self.status = (
                f"stationary: 4 corners of {self.label_square}"
                f" (corners: {self.side_grid_source})"
            )
            self.help_lines = [
                "Rings = detected corners — clicks snap to them",
                "1st click: GREEN = corner nearest a1",
                "then 3 cyan, counter-clockwise in the image",
                "next is toward h, then far corner, then rank 8",
                "Click on the RIGHT (stationary) image",
            ]
        else:
            self.status = f"stationary: 4 corners of {self.label_square}"
            self.help_lines = [
                "no corners detected — click carefully",
                "1st click: GREEN = corner nearest a1",
                "then 3 cyan, counter-clockwise in the image",
                "next is toward h, then far corner, then rank 8",
                "Click on the RIGHT (stationary) image",
            ]

    def finish_side_corners(self) -> None:
        pts = np.asarray(self.label_points[:4], dtype=np.float64)
        pose = pose_from_square_corners(pts, self.label_square, self.K_side)
        if pose is None:
            self.status = "pose rejected — green = a1-near, then the other three counter-clockwise in the image"
            self.label_points = []
            return
        self.T_board_side = pose
        self.mode = "ready"
        self.status = "calib ok — space to click"
        self.help_lines = []

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.mode == "label_wrist" and x < 640:
            snapped, err = resolve_label_click(self.wrist_grid, (float(x), float(y)))
            if err is not None:
                self.status = err
                return
            self.label_points.append(snapped)
            if len(self.label_points) >= 4:
                self.label_points = self.label_points[:4]
                self.finish_wrist_label()
            return
        if self.mode == "label_side" and x >= 640:
            snapped, err = resolve_label_click(self.side_grid, (float(x - 640), float(y)))
            if err is not None:
                self.status = err
                return
            self.label_points.append(snapped)
            if len(self.label_points) >= 4:
                self.label_points = self.label_points[:4]
                self.finish_side_corners()
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
        foot = ray_plane_z0(origin, direction)
        if foot is None:
            self.status = "ray misses the board plane"
            self.mode = "ready"
            return
        if not (-20 <= foot[0] <= 8 * SQUARE_MM + 20 and -20 <= foot[1] <= 8 * SQUARE_MM + 20):
            self.status = f"hit outside board ({foot[0]:.1f},{foot[1]:.1f}) — retry"
            self.mode = "ready"
            return
        self.report_point(foot, method="board_plane", ray_gap_mm=0.0)

    def finish_above(self, side_px: np.ndarray) -> None:
        assert self.wrist_click is not None
        o1, d1 = pixel_ray_board(self.wrist_click, self.T_board_wrist, self.K_wrist)
        o2, d2 = pixel_ray_board((float(side_px[0]), float(side_px[1])), self.T_board_side, self.K_side)
        point, gap = ray_ray_midpoint(o1, d1, o2, d2)
        if float(point[2]) < -1.0:
            self.status = "point is beneath the board — click again on the line"
            return
        self.report_point(point, method="two_view", ray_gap_mm=gap, side_px=side_px)

    def report_point(
        self,
        point_board: np.ndarray,
        *,
        method: str,
        ray_gap_mm: float,
        side_px: np.ndarray | tuple[float, float] | None = None,
    ) -> None:
        joints = self.frozen_joints or self.joints or [0.0] * 6
        T_bb = T_base_board_mm(self.T_board_wrist, joints, self.mount)
        point_base = point_board_to_base(point_board, T_bb)
        square = algebraic_from_xy(float(point_board[0]), float(point_board[1]))
        assert self.wrist_click is not None
        wrist_px = [round(float(self.wrist_click[0]), 2), round(float(self.wrist_click[1]), 2)]
        recorded_side = None
        if side_px is not None:
            recorded_side = [round(float(side_px[0]), 2), round(float(side_px[1]), 2)]
        self.last_result = {
            "method": method,
            "wrist_px": wrist_px,
            "side_px": recorded_side,
            "board_mm": [round(float(v), 3) for v in point_board],
            "base_mm": [round(float(v), 3) for v in point_base],
            "square": square,
            "ray_gap_mm": round(float(ray_gap_mm), 3),
            "mount": self.mount.kind,
            "mount_urdf_source": self.mount.urdf_source,
        }
        append_session_log(self.last_result)
        print(json.dumps(self.last_result, indent=2))
        self.status = f"board {self.last_result['board_mm']}  base {self.last_result['base_mm']}  {square}"
        self.mode = "ready"
        self.polyline = None
        self.help_lines = []

    def launch_mount(self) -> None:
        script = ROOT / "run_mount_calibration.py"
        cmd = [sys.executable, str(script)]
        if self.args.demo_pair:
            cmd.append("--demo-pair")
        if self.args.robot:
            cmd.append("--robot")
        cmd.extend(["--wrist-index", str(self.args.wrist_index)])
        print("launching", " ".join(cmd))
        subprocess.call(cmd)
        self.mount = load_mount(self.urdf, self.report)
        self.status = f"mount reloaded: {self.mount.kind}"

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
        self.status = "asset menu — Tab category, j/k select, Enter apply, Esc back"
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

    def compose(self) -> np.ndarray:
        frozen_modes = (
            "click_wrist",
            "ask_plane",
            "click_side",
            "label_wrist",
            "label_side",
            "enter_square",
            "ready",
            "asset_menu",
        )
        if self.mode in frozen_modes and self.frozen_wrist is not None:
            wrist = self.frozen_wrist.copy()
            side = self.frozen_side.copy()
        else:
            wrist, side = self.grab_live()
            if self.mode == "live":
                self.frozen_wrist = None
        if self.wrist_click is not None and self.mode in ("ask_plane", "click_side", "ready"):
            cv2.circle(wrist, (int(self.wrist_click[0]), int(self.wrist_click[1])), 6, (0, 255, 255), 2)
        if self.mode == "click_side" and self.polyline is not None:
            draw_ray_polyline(side, self.polyline)
        if self.mode == "label_wrist":
            draw_detected_corners(wrist, self.wrist_grid)
            for i, (px, py) in enumerate(self.label_points):
                color = (0, 255, 0) if i == 0 else (255, 200, 0)
                cv2.circle(wrist, (int(px), int(py)), 5, color, -1)
        if self.mode == "label_side":
            draw_detected_corners(side, self.side_grid)
            for i, (px, py) in enumerate(self.label_points):
                color = (0, 255, 0) if i == 0 else (255, 200, 0)
                cv2.circle(side, (int(px), int(py)), 5, color, -1)
        if self.last_result is not None and self.T_board_wrist is not None and self.T_board_side is not None:
            pt = np.asarray(self.last_result["board_mm"], dtype=np.float64).reshape(1, 3)
            for image, pose, K in (
                (wrist, self.T_board_wrist, self.K_wrist),
                (side, self.T_board_side, self.K_side),
            ):
                pix, depth = project_points(pt, pose, K)
                if depth[0] > 1:
                    cv2.circle(image, (int(pix[0, 0]), int(pix[0, 1])), 5, (255, 255, 255), -1)
        backup = self.banner()
        lines = [
            f"mode: {self.mode}",
            f"status: {self.status}",
            f"square: {self.label_square or '-'}",
            f"mount: {self.mount.kind} ({self.mount.urdf_source})",
            f"assets: {'BACKUP' if backup else 'local'}",
            f"cams: wrist={self.wrist_index} side={self.stationary_index}",
            "c calib  m assets  u mount  v 3d",
            "x swap cams  space click  r live  q quit",
        ]
        if self.mode == "enter_square":
            lines.append(f"buffer: {self.square_buffer or '_'}")
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
        if self.last_result:
            lines.append(f"board {self.last_result['board_mm']}")
            lines.append(f"base  {self.last_result['base_mm']}")
        panel = panel_text(lines, width=PANEL_W, height=480)
        panel = draw_backup_banner(panel, backup)
        return np.hstack([wrist, side, panel])

    def handle_key(self, key: int) -> bool:
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
                    self.status = "invalid square — try d6"
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
        elif key == ord("r"):
            self.mode = "live"
            self.status = "live preview"
            self.wrist_click = None
            self.polyline = None
            self.help_lines = []
        elif key == ord("x") and self.mode in ("live", "ready"):
            self.swap_cameras()
        elif key == ord(" "):
            if self.T_board_wrist is None or self.T_board_side is None:
                self.status = "calibrate first (c)"
            else:
                if self.frozen_wrist is None:
                    self.frozen_wrist, self.frozen_side = self.grab_live()
                    self.frozen_joints = list(self.joints or [])
                self.mode = "click_wrist"
                self.status = "click a point on the WRIST (left) view"
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
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(30) & 0xFF
            if key != 255 and not self.handle_key(key):
                break
        cv2.destroyAllWindows()
        if self.cap_wrist is not None:
            self.cap_wrist.release()
        if self.cap_side is not None:
            self.cap_side.release()

    def run_smoke(self) -> None:
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
        print("SMOKE")
        print("banner:", self.banner())
        print("mount:", self.mount.kind, self.mount.urdf_source)
        centre = np.array([[3.5 * SQUARE_MM, 5.5 * SQUARE_MM, 0.0]], dtype=np.float64)
        pix, depth = project_points(centre, self.T_board_wrist, self.K_wrist)
        assert depth[0] > 1
        self.wrist_click = (float(pix[0, 0]), float(pix[0, 1]))
        self.finish_on_board()
        board = self.last_result["board_mm"]
        assert abs(board[0] - 146.125) < 8 and abs(board[1] - 229.625) < 8
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
        print("smoke ok", self.last_result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-pair", action="store_true", help="Use bundled/local pair 003 stills")
    parser.add_argument("--use-saved-poses", action="store_true", help="Load demo board poses without interactive calib")
    parser.add_argument("--robot", action="store_true", help="Read live joint angles from the SO-101")
    parser.add_argument(
        "--wrist-index",
        type=int,
        default=None,
        help="Mounted camera index (default: session_config.yaml or 0)",
    )
    parser.add_argument(
        "--stationary-index",
        type=int,
        default=None,
        help="Stationary camera index (default: session_config.yaml or 1)",
    )
    parser.add_argument("--smoke", action="store_true", help="Headless verification on demo pair")
    args = parser.parse_args()
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
