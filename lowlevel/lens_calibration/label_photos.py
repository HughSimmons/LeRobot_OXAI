#!/usr/bin/env python3
"""Take lens-calibration photos for one camera and mark one square's four corners in each.

For Asha's lens recalibration: every photo shows the same square, and its corners are
marked the way the click tool marks them (green = corner nearest a1, then the other three
counter-clockwise in the picture: toward the h-file, the far corner, the rank-8 side). The
saved photos are RAW frames (not undistorted), as a lens calibration needs.

Keys:
  space  freeze the live frame (live), or save the marked photo (after 4 corners)
  Enter  save the marked photo (after 4 corners)
  u      undo the last corner
  f      corner clicks: snap to detected corners / manual (exactly where you click)
  Esc    drop this photo, back to live (or to the photo, in --images mode)
  n / b  next / previous suggested shot (live), or skip / go back a photo (--images)
  s      change the square (type e.g. e4, then Enter)
  h      hold / release the arm (--robot only)
  q      quit

Clicks snap to a detected chessboard corner within 12 px; with none near, a click is
refined to the nearest corner by cv2.cornerSubPix (if that moves it under 6 px). A view
whose four corners run clockwise (the mirror image) or cross over is refused.

With --capture-only, space saves the raw frame straight away, with no corners to click, into
capture.json (the format collect_wrist_photos.py writes), for marking later with --images. The
live view dots the chessboard corners detected in earlier photos, to show what the frame covers.

With --images on a folder from collect_wrist_photos.py, the photos' capture.json supplies the
square, the joint readings and the planned view, the planned square is drawn in magenta to find
it by, and labels.json and the marked photos go into that same folder.

With --plan (default for the live wrist camera: output/wrist_pose_plan.json from
plan_wrist_poses.py, if it exists) the suggested shots are the planned arm poses: the live
view shows where the plan puts the square (magenta) and, with --robot, each joint's reading
against its target, so the arm can be guided there by hand (h releases it, h holds it again).

Output, one folder per run, in output/ (or the --images folder):
  shot_001.png              raw frame
  shot_001_marked.png       the same with the square's corners marked
  labels.json               camera settings, then per photo: square, corners_px, joints

Examples (click3d venv, from this folder):
  ~/venvs/click3d/bin/python label_photos.py --camera wrist --images output/wrist_photos_<time>
  ~/venvs/click3d/bin/python label_photos.py --camera stationary --square d4
  ~/venvs/click3d/bin/python label_photos.py --camera stationary --square d4 --capture-only
  ~/venvs/click3d/bin/python label_photos.py --camera wrist --square d4 --robot
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent, HERE.parent / "click_to_3d_oliver"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from click3d_shared import (  # noqa: E402
    SQUARE_MM,
    _paths_setting,
    arm_held,
    arm_state_text,
    detect_grid,
    draw_detected_corners,
    hide_jpeg_warnings,
    load_session_cameras,
    open_camera,
    panel_text,
    parse_camera,
    parse_square_name,
    read_joints_robot,
    snap_click_to_grid,
    toggle_arm_hold,
)

WINDOW = "lens calibration photos"
PHOTOS_DIR = HERE / "output"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
SNAP_PX = 12.0
REFINE_MAX_SHIFT_PX = 6.0
GREEN = (0, 255, 0)
CYAN = (255, 200, 0)
PANEL_WIDTH = 420
MAGENTA = (255, 0, 255)
ON_TARGET_DEG = 3.0
DEFAULT_PLAN = PHOTOS_DIR / "wrist_pose_plan.json"
CAPTURE_FILE = "capture.json"
CONVENTION = (
    "corners_px[0] is the square's corner nearest a1, then +file (toward h), "
    "+file+rank (far corner), +rank (toward rank 8): counter-clockwise in the picture. "
    "Pixels are in the raw (not undistorted) frame, origin top-left."
)

# Suggested views, in robot-relative words (a1 is nearest the robot, on its right). Slanted
# views pin down the focal length (straight-down views can't tell a longer focal length from
# a farther board), and the board in each corner of the picture pins down the distortion.
SHOTS = [
    "straight down at the board, board filling most of the picture",
    "board pushed into the TOP-LEFT corner of the picture",
    "board pushed into the TOP-RIGHT corner of the picture",
    "board pushed into the BOTTOM-LEFT corner of the picture",
    "board pushed into the BOTTOM-RIGHT corner of the picture",
    "slanted view (~40 deg from straight down) from the robot's side of the board",
    "slanted view (~40 deg) from the far side of the board, opposite the robot",
    "slanted view (~40 deg) from the robot's left side of the board",
    "slanted view (~40 deg) from the robot's right side of the board",
    "slanted view (~45 deg) from one corner of the board",
    "slanted view (~45 deg) from the opposite corner of the board",
    "straight down, camera turned ~45 deg so the squares look like diamonds",
    "from farther back: whole board small in the middle",
    "slanted view (~30 deg), board at the LEFT edge of the picture",
    "slanted view (~30 deg), board at the RIGHT edge of the picture",
]


def visually_counter_clockwise(points: np.ndarray) -> bool:
    """True for a convex quad toured counter-clockwise on screen (image y points down)."""
    edges = np.roll(points, -1, axis=0) - points
    turns = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    # With y down, an on-screen counter-clockwise tour turns the negative way at every corner.
    return bool(np.all(turns < 0))


def refine_corner(gray: np.ndarray, pixel: tuple[float, float]) -> tuple[float, float] | None:
    """Nearest chessboard corner by cornerSubPix, or None if it is more than a few px away."""
    start = np.array([[pixel]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    refined = cv2.cornerSubPix(gray, start.copy(), (5, 5), (-1, -1), criteria).reshape(2)
    if float(np.linalg.norm(refined - start.reshape(2))) > REFINE_MAX_SHIFT_PX:
        return None
    return float(refined[0]), float(refined[1])


def draw_marks(image: np.ndarray, points: list[tuple[float, float]], square: str) -> np.ndarray:
    """The square's corners as the click tool draws them: green first, then cyan, numbered."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 4:
        cv2.polylines(image, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], True, CYAN, 1, cv2.LINE_AA)
        cx, cy = pts.mean(axis=0)
        cv2.putText(image, square, (int(cx) - 12, int(cy) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        color = GREEN if i == 0 else CYAN
        cv2.circle(image, (int(round(x)), int(round(y))), 6, color, 2, cv2.LINE_AA)
        cv2.circle(image, (int(round(x)), int(round(y))), 1, color, -1)
        cv2.putText(image, str(i + 1), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image


class Labeller:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.role = args.camera
        self.square = args.square
        self.manual = args.manual_corners
        self.mode = "live"  # live, label, square
        self.previous_mode = "live"
        self.status = ""
        self.square_buffer = ""
        self.shot_hint = 0
        self.frame: np.ndarray | None = None
        self.gray: np.ndarray | None = None
        self.grid: np.ndarray | None = None
        self.points: list[tuple[float, float]] = []
        self.sources: list[str] = []
        self.joints: list[float] | None = None
        self.cap = None
        self.held = False

        self.shots = list(SHOTS)
        self.plan_views: list[dict] = []
        plan_path = Path(args.plan) if args.plan else (DEFAULT_PLAN if self.role == "wrist" and args.images is None else None)
        if plan_path is not None and plan_path.is_file():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.plan_views = plan["views"]
            self.shots = [view["hint"] for view in self.plan_views]
            if plan["square"] != self.square:
                print(f"note: the plan is for {plan['square']}, not {self.square}; using {plan['square']}")
                self.square = plan["square"]
            print(f"shots from {plan_path} ({len(self.plan_views)} planned poses)")
        elif args.plan:
            raise SystemExit(f"No plan at {args.plan}")

        self.images: list[Path] = []
        self.image_index = 0
        self.captured: dict[str, dict] = {}  # image name -> its collect_wrist_photos.py record
        default_dir = None
        if args.images is not None:
            folder = Path(args.images).expanduser().resolve()
            self.images = sorted(p for p in folder.iterdir()
                                 if p.suffix.lower() in IMAGE_SUFFIXES and not p.stem.endswith("_marked"))
            if not self.images:
                raise SystemExit(f"No images in {args.images}")
            self.device = f"images from {folder}"
            default_name = f"{self.role}_labels_{folder.name}"
            if (folder / CAPTURE_FILE).is_file():
                capture = json.loads((folder / CAPTURE_FILE).read_text(encoding="utf-8"))
                self.captured = {shot["image"]: shot for shot in capture["shots"]}
                self.device = capture.get("device", self.device)
                if capture.get("square") and capture["square"] != self.square:
                    print(f"note: the photos are of {capture['square']}, not {self.square}; using {capture['square']}")
                    self.square = capture["square"]
                default_dir = folder
        else:
            if self.role == "wrist":
                self.device = load_session_cameras(args.device, None)[0]
            else:
                self.device = load_session_cameras(None, args.device)[1]
            default_name = f"{self.role}_{datetime.now():%Y%m%d_%H%M%S}"
        self.out_dir = Path(args.out) if args.out else (default_dir or PHOTOS_DIR / default_name)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.labels_path = self.out_dir / "labels.json"
        self.capture_path = self.out_dir / CAPTURE_FILE
        self.capture = None
        if args.capture_only:
            if self.capture_path.is_file():
                self.capture = json.loads(self.capture_path.read_text(encoding="utf-8"))
                print(f"Adding to {len(self.capture['shots'])} photos in {self.out_dir}")
            else:
                self.capture = {
                    "camera": self.role,
                    "device": self.device,
                    "v4l2_controls": _paths_setting(f"{self.role.upper()}_CAMERA_CONTROLS"),
                    "square": self.square,
                    "square_mm": SQUARE_MM,
                    "note": "Raw frames, saved without marking; detected_corners_px is cv2's grid, unchecked.",
                    "shots": [],
                }
        if self.labels_path.is_file():
            self.labels = json.loads(self.labels_path.read_text(encoding="utf-8"))
            print(f"Adding to {len(self.labels['shots'])} labelled photos in {self.out_dir}")
        else:
            self.labels = {
                "camera": self.role,
                "device": self.device,
                "v4l2_controls": _paths_setting(f"{self.role.upper()}_CAMERA_CONTROLS") if not self.images else None,
                "square_mm": SQUARE_MM,
                "convention": CONVENTION,
                "created": datetime.now().isoformat(timespec="seconds"),
                "shots": [],
            }
        if self.images:
            # Carry on after the photos already labelled.
            done = {Path(shot["image"]).name for shot in self.labels["shots"]}
            remaining = [i for i, p in enumerate(self.images) if p.name not in done]
            self.image_index = remaining[0] if remaining else len(self.images) - 1

    # --- input -----------------------------------------------------------------------

    def open(self) -> None:
        if self.images:
            self.load_image()
            return
        self.cap = open_camera(self.device, self.role)
        print(f"{self.role} camera: {self.device}")
        if self.args.robot:
            self.joints = read_joints_robot()
            self.held = arm_held()
            print(f"arm connected, joints {np.round(self.joints, 2).tolist()}")

    def load_image(self) -> None:
        path = self.images[self.image_index]
        frame = cv2.imread(str(path))
        if frame is None:
            self.status = f"could not read {path.name}"
            return
        self.freeze(frame)
        self.status = f"photo {self.image_index + 1}/{len(self.images)}: {path.name}"

    def live_frame(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"{self.role} camera read failed")
        return frame

    def freeze(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.grid = detect_grid(frame)
        self.points, self.sources = [], []
        if self.args.robot:
            self.joints = read_joints_robot()
        self.mode = "label"
        if self.grid is None:
            self.status = "no chessboard grid detected: click carefully (clicks are sub-pixel refined)"
        else:
            self.status = f"detected {self.grid.shape[1]}x{self.grid.shape[0]} inner corners"

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self.mode != "label" or len(self.points) >= 4:
            return
        if x >= self.frame.shape[1]:
            return
        click = (float(x), float(y))
        if self.manual:
            point, source = click, "manual"
        else:
            point = snap_click_to_grid(self.grid, click, SNAP_PX) if self.grid is not None else None
            source = "detected"
            if point is None:
                point, source = refine_corner(self.gray, click), "subpix"
            if point is None:
                point, source = click, "manual"
        self.points.append(point)
        self.sources.append(source)
        if len(self.points) == 4:
            pts = np.asarray(self.points, dtype=np.float64)
            if not visually_counter_clockwise(pts):
                self.status = "refused: corners run clockwise (mirror) or cross over - click them again"
                self.points, self.sources = [], []
                return
            self.status = "4 corners: Enter/space saves, u undoes, Esc drops"

    # --- output ----------------------------------------------------------------------

    def save(self) -> None:
        if self.images:
            source = self.images[self.image_index]
            raw_name = source.name
        else:
            raw_name = f"shot_{len(self.labels['shots']) + 1:03d}.png"
            cv2.imwrite(str(self.out_dir / raw_name), self.frame)
        name = Path(raw_name).stem
        marked = draw_marks(self.frame.copy(), self.points, self.square)
        cv2.imwrite(str(self.out_dir / f"{name}_marked.png"), marked)
        captured = self.captured.get(raw_name, {})
        record = {
            "image": raw_name if not self.images or self.out_dir == source.parent else str(source),
            "marked_image": f"{name}_marked.png",
            "square": self.square,
            "corners_px": [[round(x, 2), round(y, 2)] for x, y in self.points],
            "corner_sources": self.sources,
            "detected_inner_grid": None if self.grid is None else [int(self.grid.shape[1]), int(self.grid.shape[0])],
            "image_size": [int(self.frame.shape[1]), int(self.frame.shape[0])],
            "shot_hint": captured.get("hint") if self.images else self.shots[self.shot_hint % len(self.shots)],
            "plan_view": captured.get("plan_view") if self.images else (
                self.current_plan_view()["view"] if self.current_plan_view() else None),
            "joints_reading_deg": captured.get("joints_reading_deg") if self.images else (
                None if self.joints is None else [round(j, 3) for j in self.joints]),
            "time": datetime.now().isoformat(timespec="seconds"),
        }
        self.labels["shots"].append(record)
        tmp = self.labels_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.labels, indent=2), encoding="utf-8")
        tmp.replace(self.labels_path)
        print(f"saved {name} ({self.square}, corners {record['corner_sources']})")
        self.status = f"saved {name} ({len(self.labels['shots'])} so far)"
        if self.images:
            self.next_image(+1)
        else:
            self.shot_hint += 1
            self.mode = "live"

    def save_raw(self, frame: np.ndarray) -> None:
        """--capture-only: save the frame as it is, and note what the grid detector finds in it."""
        name = f"shot_{len(self.capture['shots']) + 1:03d}.png"
        cv2.imwrite(str(self.out_dir / name), frame)
        grid = detect_grid(frame)
        self.capture["shots"].append({
            "image": name,
            "hint": self.shots[self.shot_hint % len(self.shots)],
            "square": self.square,
            "detected_inner_grid": None if grid is None else [int(grid.shape[1]), int(grid.shape[0])],
            "detected_corners_px": None if grid is None else np.round(grid.reshape(-1, 2), 1).tolist(),
            "time": datetime.now().isoformat(timespec="seconds"),
        })
        tmp = self.capture_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.capture, indent=2), encoding="utf-8")
        tmp.replace(self.capture_path)
        found = "no grid found" if grid is None else f"grid {grid.shape[1]}x{grid.shape[0]} found"
        print(f"saved {name} ({found})")
        self.status = f"saved {name}, {found} ({len(self.capture['shots'])} so far)"
        self.shot_hint += 1

    def current_plan_view(self) -> dict | None:
        if not self.plan_views or self.images:
            return None
        return self.plan_views[self.shot_hint % len(self.plan_views)]

    def plan_lines(self) -> list[str]:
        """Target joints of the planned shot, and how far the arm is from them."""
        view = self.current_plan_view()
        if view is None:
            return []
        target = view["joints_reading_deg"][:5]
        lines = ["", "target joints (reading deg):"]
        if self.joints is None:
            lines.append("  " + "  ".join(f"{t:.0f}" for t in target))
            return lines
        deltas = [t - j for t, j in zip(target, self.joints[:5])]
        names = ("pan", "lift", "elbow", "w_flex", "w_roll")
        for name, t, j, d in zip(names, target, self.joints, deltas):
            mark = "ok" if abs(d) <= ON_TARGET_DEG else f"move {d:+.0f}"
            lines.append(f"  {name:6s} {t:7.1f}  now {j:7.1f}  {mark}")
        if all(abs(d) <= ON_TARGET_DEG for d in deltas):
            lines.append("ON TARGET - press space")
        return lines

    def next_image(self, step: int) -> None:
        self.image_index += step
        if not 0 <= self.image_index < len(self.images):
            self.image_index = min(max(self.image_index, 0), len(self.images) - 1)
            self.status = "no more photos: q to quit"
            return
        self.load_image()

    # --- display ---------------------------------------------------------------------

    def compose(self) -> np.ndarray:
        if self.mode == "live" and self.cap is not None:
            view = self.live_frame()
            if self.args.robot and self.plan_views:
                self.joints = read_joints_robot()
            planned = self.current_plan_view()
            if planned is not None:
                pts = np.round(np.asarray(planned["square_corners_px"])).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(view, [pts], True, MAGENTA, 2, cv2.LINE_AA)
                cv2.circle(view, tuple(int(v) for v in pts[0, 0]), 6, GREEN, -1)
            # Where earlier photos put the square (or found corners), so the shots cover the whole frame.
            for shot in (self.capture or {}).get("shots", []):
                for x, y in shot["detected_corners_px"] or []:
                    cv2.circle(view, (int(round(x)), int(round(y))), 2, (0, 200, 255), -1)
            for shot in self.labels["shots"]:
                pts = np.round(np.asarray(shot["corners_px"])).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(view, [pts], True, (0, 200, 255), 1, cv2.LINE_AA)
        else:
            view = self.frame.copy()
            if self.mode == "label" and not self.manual:
                draw_detected_corners(view, self.grid)
            planned = self.captured.get(self.images[self.image_index].name) if self.images else None
            if planned and planned.get("planned_square_corners_px") and self.mode == "label" and len(self.points) < 4:
                pts = np.round(np.asarray(planned["planned_square_corners_px"])).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(view, [pts], True, MAGENTA, 1, cv2.LINE_AA)
            draw_marks(view, self.points, self.square)

        lines = [
            f"{self.role} camera - square {self.square} "
            f"({len((self.capture or self.labels)['shots'])} {'saved' if self.capture else 'marked'})",
            f"status: {self.status}",
            "",
        ]
        if self.mode == "square":
            lines += [f"square: {self.square_buffer}_", "type e.g. e4, Enter to confirm, Esc to cancel"]
        elif self.mode == "live":
            lines += [
                f"suggested shot {self.shot_hint % len(self.shots) + 1}/{len(self.shots)} (n/b to change):",
                f"  {self.shots[self.shot_hint % len(self.shots)]}",
                f"  keep {self.square} in view; orange = earlier photos' corners"
                + ("; magenta = planned" if self.current_plan_view() else ""),
                "",
                "space: SAVE this frame (no marking)" if self.capture else "space: freeze this frame",
                *(["keep 35-80 cm from the board (focus is locked)", "don't zoom"] if self.role == "stationary" else []),
                "s: change square   q: quit",
            ]
        else:
            lines += [
                "MANUAL corners (f: snap)" if self.manual else "clicks snap to rings / refine (f: manual)",
                f"1st click: GREEN = corner of {self.square} nearest a1",
                "then 3 cyan, counter-clockwise in the image:",
                "toward h, then far corner, then rank 8",
                f"corners: {len(self.points)}/4   u: undo",
                "Enter/space: save   Esc: drop",
            ]
            if self.images:
                lines.append("n / b: skip / previous photo")
        if self.mode == "live":
            lines += self.plan_lines()
        if self.args.robot:
            lines += ["", f"arm: {arm_state_text(self.held)} (h)"]
            if self.joints is not None and not self.plan_views:
                lines.append("joints: " + ", ".join(f"{j:.1f}" for j in self.joints))
        panel = panel_text(lines, width=PANEL_WIDTH, height=view.shape[0])
        return np.hstack([view, panel])

    def handle_key(self, key: int) -> bool:
        """False to quit."""
        if key == 255:
            return True
        if self.mode == "square":
            if key == 27:
                self.mode, self.square_buffer = self.previous_mode, ""
            elif key in (13, 10):
                square = parse_square_name(self.square_buffer)
                if square is None:
                    self.status = f"not a square: {self.square_buffer!r}"
                else:
                    self.square = square
                    self.status = f"square is now {square}"
                self.mode, self.square_buffer = self.previous_mode, ""
            elif key in (8, 127):
                self.square_buffer = self.square_buffer[:-1]
            elif chr(key).lower() in "abcdefgh12345678":
                self.square_buffer = (self.square_buffer + chr(key).lower())[-2:]
            return True
        if key == ord("q"):
            return False
        if key == ord("s"):
            self.previous_mode, self.mode = self.mode, "square"
            self.square_buffer = ""
            return True
        if key == ord("h") and self.args.robot:
            self.held = toggle_arm_hold()
            self.status = f"arm {arm_state_text(self.held)}"
            return True
        if key == ord("f"):
            self.manual = not self.manual
            return True
        if self.mode == "live":
            if key == ord(" "):
                if self.capture is not None:
                    self.save_raw(self.live_frame())
                else:
                    self.freeze(self.live_frame())
            elif key in (ord("n"), ord("b")):
                self.shot_hint += 1 if key == ord("n") else -1
            return True
        # label mode
        if key == ord("u") and self.points:
            self.points.pop()
            self.sources.pop()
        elif key in (13, 10, ord(" ")):
            if len(self.points) == 4:
                self.save()
            else:
                self.status = f"{4 - len(self.points)} corner(s) still to click"
        elif key == 27:
            if self.images:
                self.points, self.sources = [], []
            else:
                self.mode = "live"
                self.status = "photo dropped"
        elif key in (ord("n"), ord("b")) and self.images:
            self.next_image(+1 if key == ord("n") else -1)
        return True

    def run(self) -> None:
        self.open()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        try:
            while True:
                cv2.imshow(WINDOW, self.compose())
                if not self.handle_key(cv2.waitKey(30) & 0xFF):
                    break
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
        print(f"{len(self.labels['shots'])} labelled photos in {self.out_dir}")


def run_smoke() -> None:
    """Headless check of the corner-order test and the corner refinement."""
    # On screen: top-left, bottom-left, bottom-right, top-right is counter-clockwise.
    ccw = np.array([[100, 100], [100, 200], [200, 200], [200, 100]], dtype=np.float64)
    assert visually_counter_clockwise(ccw)
    assert not visually_counter_clockwise(ccw[::-1])
    assert not visually_counter_clockwise(ccw[[0, 2, 1, 3]])  # crossed
    board = np.zeros((480, 640), dtype=np.uint8)
    board[200:280, 300:380] = 255
    board[280:360, 220:300] = 255
    refined = refine_corner(board, (302.0, 282.0))
    assert refined is not None and abs(refined[0] - 299.5) < 1.0 and abs(refined[1] - 279.5) < 1.0, refined
    print("smoke ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", choices=("wrist", "stationary"), help="which camera the photos are from")
    parser.add_argument("--square", default="d4", type=lambda s: parse_square_name(s) or parser.error(f"bad square {s!r}"),
                        help="the square to mark in every photo (default d4; s changes it)")
    parser.add_argument("--device", type=parse_camera, default=None,
                        help="camera index or device path (default: output/session_config.yaml, then paths.py)")
    parser.add_argument("--images", default=None, help="mark photos already taken in this folder instead of a live camera")
    parser.add_argument("--out", default=None, help=f"output folder (default: {PHOTOS_DIR}/<camera>_<time>)")
    parser.add_argument("--plan", default=None, help=f"pose plan from plan_wrist_poses.py (wrist default: {DEFAULT_PLAN})")
    parser.add_argument("--robot", action="store_true", help="log the arm's joint readings with every photo (wrist)")
    parser.add_argument("--capture-only", action="store_true",
                        help="space saves raw frames without marking (capture.json), for marking later with --images")
    parser.add_argument("--manual-corners", action="store_true", help="start with manual corner clicks; f toggles")
    parser.add_argument("--smoke", action="store_true", help="headless self-check, no camera")
    args = parser.parse_args()
    if args.smoke:
        run_smoke()
        return
    if args.camera is None:
        parser.error("--camera wrist|stationary is required")
    if args.capture_only and args.images:
        parser.error("--capture-only takes photos; --images marks ones already taken")
    hide_jpeg_warnings()
    Labeller(args).run()


if __name__ == "__main__":
    main()
