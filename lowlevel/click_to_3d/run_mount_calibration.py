#!/usr/bin/env python3
"""Calibrate T_gripper_camera from several parked poses + wrist board PnP.

Keys:
  space  capture a sample at the current park pose
  t      choose / change the square for later samples
  s      solve, write YAML + calibrated URDF, quit
  q      quit without saving

Each capture uses the same corner circuit as the click tool: green is the
corner nearest a1, then the other three counter-clockwise in the image
(toward the h-file, then the far corner, then the rank-8 side).

Minimum 3 samples. Writes under click_to_3d/output/camera_mount/.
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from click3d_shared import (
    detect_grid_for_view,
    draw_backup_banner,
    draw_detected_corners,
    ensure_output_dirs,
    fk_link_mm,
    invert,
    load_demo_board_poses,
    load_mount,
    load_session_camera_indices,
    open_camera,
    panel_text,
    parse_square_name,
    pose_from_square_corners,
    read_joints_json,
    read_joints_robot,
    resolve_core_assets,
    resolve_demo_pair,
    resolve_label_click,
    save_session_camera_indices,
    se3,
    solve_hand_eye_mount,
    undistort_wrist,
    write_board_in_base_file,
    write_mount_files,
)

WINDOW = "mount_calibration"
CORNER_CIRCUIT = [
    "corners: GREEN nearest a1, then counter-clockwise",
    "next toward h, then far corner, then rank 8",
]


class MountApp:
    def __init__(self, args: argparse.Namespace) -> None:
        ensure_output_dirs()
        self.args = args
        self.report, assets = resolve_core_assets()
        self.fisheye = assets["fisheye"]
        self.urdf = assets["urdf"]
        self.mount = load_mount(self.urdf, self.report)
        self.samples: list[dict] = []
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
        self.pending_grid = None
        self.pending_grid_source = "none"
        self.done = False
        if not args.smoke and self.label_square is None:
            self.mode = "enter_square"
            self.status = "type square name (e.g. e4) then Enter"
            self.help_lines = [
                "Pick a square with clear corners for this session",
                "Press t later to change square between samples",
                *CORNER_CIRCUIT,
            ]
        self.wrist_index, _stationary = load_session_camera_indices(
            getattr(args, "wrist_index", None),
            getattr(args, "stationary_index", None),
        )
        args.wrist_index = self.wrist_index
        if args.demo_pair:
            self.demo = resolve_demo_pair(self.report)
            self.joints = read_joints_json(self.demo["joints"].path)
            raw = cv2.imread(str(self.demo["wrist"].path))
            self.frame_raw = raw.copy() if raw is not None else None
            self.frame, self.K_wrist, self.fisheye_calib = undistort_wrist(raw, self.fisheye.path)
            if args.smoke:
                self.label_square = self.label_square or "d6"
                self._smoke_fill()
        elif args.robot:
            self.cap = open_camera(self.wrist_index)
            save_session_camera_indices(self.wrist_index, _stationary)
            self.joints = read_joints_robot()
        else:
            self.cap = open_camera(self.wrist_index)
            save_session_camera_indices(self.wrist_index, _stationary)
            self.joints = [0.0] * 6

    def _smoke_fill(self) -> None:
        """Synthetic multi-pose samples with a fixed mount and board-in-base."""
        T_board_cam0, _ = load_demo_board_poses(self.demo["poses"].path)
        T_base_cam = fk_link_mm(self.urdf.path, "wrist_camera_link", self.joints)
        T_base_grip0 = fk_link_mm(self.urdf.path, "gripper_frame_link", self.joints)
        Y = T_base_cam @ invert(T_board_cam0)
        X = invert(T_base_grip0) @ Y @ T_board_cam0
        # Large, distinct rotations so calibrateHandEye has informative motions.
        angles = [
            (0.55, 0.0, 0.0),
            (0.0, 0.65, 0.0),
            (0.0, 0.0, 0.75),
            (0.4, 0.35, -0.3),
        ]
        for ax, ay, az in angles:
            cx, sx = float(np.cos(ax)), float(np.sin(ax))
            cy, sy = float(np.cos(ay)), float(np.sin(ay))
            cz, sz = float(np.cos(az)), float(np.sin(az))
            rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
            ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
            rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            rot = rz @ ry @ rx
            translation = T_base_grip0[:3, 3] + np.array([20.0 * ax, 15.0 * ay, 10.0 * az])
            T_grip = se3(rot @ T_base_grip0[:3, :3], translation)
            T_board_cam = invert(Y) @ T_grip @ X
            self.samples.append(
                {
                    "joints": list(self.joints),
                    "T_board_camera": T_board_cam,
                    "T_base_gripper": T_grip,
                }
            )
        self.status = f"smoke samples={len(self.samples)}"

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
        )
        self.pending_image = image
        self.pending_grid = grid
        self.pending_grid_source = source
        self.label_points = []
        self.mode = "label"
        if grid is not None:
            self.status = (
                f"click 4 corners of {self.label_square}"
                f" (corners: {source})"
            )
            self.help_lines = [
                "Rings = detected corners — clicks snap to them",
                "1st click: GREEN = corner nearest a1",
                "then 3 cyan, counter-clockwise in the image",
                "next is toward h, then far corner, then rank 8",
            ]
        else:
            self.status = f"click 4 corners of {self.label_square}"
            self.help_lines = [
                "no corners detected — click carefully",
                "1st click: GREEN = corner nearest a1",
                "then 3 cyan, counter-clockwise in the image",
                "next is toward h, then far corner, then rank 8",
            ]

    def _store_sample(self, T_board_camera: np.ndarray) -> None:
        joints = list(self.joints)
        T_base_grip = fk_link_mm(self.urdf.path, "gripper_frame_link", joints)
        self.samples.append(
            {
                "joints": joints,
                "T_board_camera": T_board_camera,
                "T_base_gripper": T_base_grip,
            }
        )
        self.status = f"samples={len(self.samples)}  space=more  s=solve"
        self.help_lines = list(CORNER_CIRCUIT)
        self.mode = "live"

    def solve(self) -> None:
        if len(self.samples) < 3:
            self.status = "need at least 3 samples"
            return
        X, Y, residual = solve_hand_eye_mount(self.samples)
        self.T_base_board = Y
        yaml_path, urdf_path = write_mount_files(
            X,
            sample_count=len(self.samples),
            residual_mm=residual,
            source_urdf=self.urdf.path,
        )
        board_path = write_board_in_base_file(
            Y,
            sample_count=len(self.samples),
            residual_mm=residual,
        )
        est = fk_link_mm(self.urdf.path, "wrist_camera_link", self.samples[0]["joints"])
        grip = self.samples[0]["T_base_gripper"]
        old = invert(grip) @ est
        delta_t = float(np.linalg.norm(X[:3, 3] - old[:3, 3]))
        print(
            json.dumps(
                {
                    "wrote_yaml": str(yaml_path),
                    "wrote_urdf": str(urdf_path),
                    "wrote_board_in_base": str(board_path),
                    "samples": len(self.samples),
                    "mean_residual_mm": round(residual, 3),
                    "delta_vs_urdf_estimate_mm": round(delta_t, 3),
                    "T_gripper_camera_mm": np.round(X, 6).tolist(),
                    "T_base_board_mm": np.round(Y, 6).tolist(),
                },
                indent=2,
            )
        )
        self.status = f"saved residual={residual:.2f}mm  delta_urdf={delta_t:.2f}mm"
        self.done = True

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if x >= 640:
            return
        if self.mode == "label":
            snapped, err = resolve_label_click(self.pending_grid, (float(x), float(y)))
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
                self._store_sample(pose)

    def compose(self) -> np.ndarray:
        if self.mode == "label" and self.pending_image is not None:
            image = self.pending_image.copy()
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
            f"samples: {len(self.samples)}",
            f"urdf: {self.urdf.source}",
            "space capture   t square   s solve   q quit",
        ]
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

    def run_smoke(self) -> None:
        self.solve()
        mount = load_mount(self.urdf)
        assert mount.kind == "calibrated"
        print("mount smoke ok", mount.mount_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-pair", action="store_true")
    parser.add_argument("--robot", action="store_true")
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
        help="Unused here; kept so session_config loads both indices",
    )
    parser.add_argument("--square", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.demo_pair = True
    app = MountApp(args)
    if args.smoke:
        app.run_smoke()
    else:
        app.run_gui()


if __name__ == "__main__":
    main()
