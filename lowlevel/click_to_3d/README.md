# Click-to-3D

Park the SO-101, calibrate both cameras to the chessboard, click a point in the
wrist camera, and get that point in board millimetres and arm-base millimetres.

Assumptions for this handoff: **same SO-101**, **same wrist camera + Logitech C920**,
**same 33.6 cm board** (42.0 mm squares; Oliver tape), frames at **640×480 MJPG**.

Wrist lens files (pick with `m` → fisheye):

- **Default:** `output/camera_mount/wrist_fisheye.yaml` — OpenCV `fisheye.calibrate`
  on all a1-indexed grown corners from the Nov-3 d4 set (`run_standard_wrist_fisheye.py`).
- **Alternate:** `output/camera_mount/wrist_fisheye_status_quo.yaml` — earlier dense-grid
  interim (f≈540, principal point fixed at centre).

The old f=1400 fallback is removed. Bundle mirrors the default OpenCV file only as a last-resort copy.

## Place in OXAI

Copy this folder to:

```text
LeRobot_OXAI/lowlevel/click_to_3d/
```

The parent directory must contain `paths.py` (standard OXAI `lowlevel` layout).
That is how forward kinematics and `--robot` find the arm API.

## Environments

Own `.venv` in this folder (no `helk` required). Light stack for calib / click /
mount; SAM gold mode is optional.

```bash
cd click_to_3d
python3 -m venv .venv
source .venv/bin/activate          # Linux and macOS
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`--smoke` and `--demo-pair` need the packages above (including `placo` and
`draccus` for forward kinematics through OXAI `paths.py`).

**Gold-mask mode (`g` / `p`)** needs the same venv plus the SAM add-on:

```bash
pip install -r requirements-sam.txt
```

For dataset labeling without the robot UI, use sibling `gold_mask_batch/` with
its own heavier `.venv` instead.

`--robot` also needs the OXAI lowlevel lerobot checkout already present beside
this folder (`import paths` / SO-101 follower). Power the arm and use the same
USB port file as any other OXAI hardware script. Allow this program camera
access in the operating system's privacy settings. Shared grasp / SAM helpers
live in `../coord_finder/` when this app sits under OXAI `lowlevel/`.

## Checklist (first run)

1. Place the folder at `LeRobot_OXAI/lowlevel/click_to_3d/` and install (above).
2. Smoke test (no cameras, no arm):

   ```bash
   python run_click_to_3d.py --smoke
   python run_mount_calibration.py --smoke
   ```

3. Live cameras (and arm when ready):

   ```bash
   python run_click_to_3d.py --robot
   ```

4. Confirm **mounted camera on the left**, **stationary on the right**. Press `x`
   to swap indices until that is true. Indices are saved under
   `output/session_config.yaml`.
5. Press `c`. Type the square (e.g. `d` then `6`) and Enter.
6. Rings appear on **both** views as soon as you press `c` — pick a square that
   has a complete cell in both cameras (any square is fine; it need not hold a
   piece). On each view, click the four corners of that square. The **first click
   is green** and must be the corner nearest a1. Then click the other three
   **counter-clockwise in the picture**: next corner is toward the h-file, then
   the far corner, then the rank-8 side. A clockwise tour is the mirror image and
   is rejected. Rings snap to detected corners when available; **Shift+click**
   accepts a free pixel when the snap is wrong. The stationary view gates corners
   with a padded LiveChess2FEN board quad (rectangular OpenCV crop, then
   accept/reject); the wrist / overhead view is ungated.
7. Press `u` for mount calibration. Park the arm, press `space` at least three times at **widely different arm rotations** (same square), then `s`. Each capture uses the same corner circuit: green nearest a1, then the other three counter-clockwise in the wrist image. Small nudges are not enough; hand–eye fails when the motion is too small. The click tool reloads the new mount when that window closes.
8. Press `c` again, then `space`, click a point, `y` (on the board) or `n`
   (above — then click the ray on the stationary view).

Do not zoom the C920. Keep 640×480 MJPG and the locked focus used for lens calibration.
The stationary matrix is a measured pinhole (`stationary_intrinsics.yaml` under
`output/lens_calib/`, `output/camera_mount/`, and `bundle/`). Regenerate with
`python run_stationary_lens_calib.py` from `CALIBPICS/stationary_20261107_001628/`.

A yellow **BACKUP IN USE** banner for the bundled lens files is normal on a correct install.

Until you finish step 7, the tool uses the URDF wrist-camera estimate. Do not
copy another machine’s `output/camera_mount/wrist_camera_mount.yaml` as if it
were yours.

## What a click stores

The board frame has its origin at the a1 corner, +x toward the h-file, +y toward rank 8, and +z up. Squares are 42.0 mm (33.6 cm / 8).

## Stationary lens calibration

```bash
python ../run_stationary_lens_calib.py
```

Reads `../CALIBPICS/stationary_20261107_001628/` (d4-labeled multi-views, 42.0 mm
squares). Writes the same measured `stationary_intrinsics.yaml` to
`../output/lens_calib/`, `output/camera_mount/`, and `bundle/`.

## Wrist lens calibration

```bash
python run_standard_wrist_fisheye.py   # default file (OpenCV full-point)
python run_refit_wrist_fisheye.py      # earlier dense-grid interim path
```

Both read `../sample images/wrist_photos_20261103_225124/` (drop `shot_001`).
The click tool defaults to `wrist_fisheye.yaml`; use `m` → fisheye to select
`wrist_fisheye_status_quo.yaml` if needed.

Corner clicks in step 6, and each mount capture, snap to a detected ring within 12 pixels, solve the camera pose, and are not written to the log.

After `space`, the wrist click is the raw undistorted pixel. `y` intersects that ray with the board plane `z = 0`. `n` takes a second click on the stationary view and uses the midpoint of the two rays.

Each saved point is one line in `output/sessions/session_log.jsonl`:

- `wrist_px` and `side_px` (`side_px` is empty when the point is on the board)
- `board_mm`, `base_mm`, `square`
- `method`: `board_plane` or `two_view`
- `ray_gap_mm`

Pixels are in that view’s undistorted image, origin at the top-left, rounded to 2 decimals.

## Keys (click tool)

| Key | Action |
| --- | --- |
| `c` | Calibrate both cameras to the board (extrinsics) |
| `x` | Swap camera indices (mounted left, stationary right) |
| `m` | Asset menu (fisheye / stationary K / URDF / mount) |
| `u` | Launch mount calibration |
| `v` | Open 3D scene visualiser |
| `space` | Freeze and click on the undistorted wrist view |
| `g` | Gold-mask: type a square name, then SAM FG/BG on prism crops |
| `y` / `n` | After click: on the board / above the board |
| `r` | Live preview |
| `q` | Quit |

### Gold-mask mode (`g`)

Requires finished board calib (`c`) so `T_board_*` poses already exist. Press `g`, then **type only the square name** (e.g. `e4`) and Enter — no corner re-clicks and no new PnP. The buffer starts empty (the calib square is shown as a hint only, not auto-selected). Both frozen views are cropped with the same rectangular-prism window as the two-view grasp pipeline (`side_crop_window`), then you refine with SAM clicks.

| Key / click | Action |
| --- | --- |
| (name entry) | Type file+rank; Backspace edits; Enter accepts; Esc cancels |
| LMB / RMB | Foreground / background point on the panel under the cursor |
| `u` / `r` | Undo / reset points on the **active** view (not mount / live) |
| `t` | Toggle active wrist ↔ side |
| Enter / `s` | Save active view if the mask is `ok` |
| `p` | Save **both** masks, run two-view grasp, return to `ready` like a click |
| Esc | Return to `ready` (warns if unsaved) |

Saves under `output/gold_masks/<session>/<square>_<view>/` as `crop.png`, `mask.png`, `overlay.png`, and `meta.json` (points, labels, window, model id, gold square, calib square). These are human-accepted SAM masks for later comparison—not auto pipeline output.

**`p` (coordinate):** requires wrist and side masks both `ok`. Saves both gold bundles, pastes masks into the frozen full frames, runs `grasp_from_views` with the champion 3D fuse `fuse_mode=overhead_ray` (empty occupancy; see `../two_view_pipeline_champions.md`), logs `method: gold_two_view` and `fuse_mode` to `session_log.jsonl`, then leaves gold-mask UI for the normal wrist|stationary|panel `ready` view. The grasp point is projected as white dots on both cameras and listed as `board […]  base […]` exactly like a space-click result.

## Mount calibration

```bash
python run_mount_calibration.py --robot
```

Or press `u` from the click tool. On every capture, the first click is green and is the corner nearest a1. Then click the other three corners counter-clockwise in the wrist image: toward the h-file, then the far corner, then the rank-8 side. A clockwise tour is the mirror and is rejected.

Writes:

- `output/camera_mount/wrist_camera_mount.yaml`
- `output/camera_mount/so101_new_calib_wrist_cam_calibrated.urdf`
- `output/camera_mount/board_in_base_mm.yaml` (diagnostic; see below)

Forward kinematics gives the gripper in the robot base. The corner pose gives the camera in the board. Hand–eye solves the camera-to-gripper mount from at least three widely rotated parks, and the written `board_in_base_mm.yaml` is only the average from those parks. A later click’s `base_mm` is that saved mount combined with the pose from the current `c`, so it changes when the board or the arm pose changes. Until `u` succeeds, `base_mm` uses the URDF camera estimate.

## Demo without hardware

```bash
python run_click_to_3d.py --demo-pair --use-saved-poses
```

Bundled backups under `bundle/` are used when project files are missing. The yellow
**BACKUP IN USE** banner for those lens files is expected.
