# Click-to-3D

Park the SO-101, calibrate both cameras to the chessboard, click a point in the
wrist camera, and get that point in board millimetres and arm-base millimetres.

Assumptions for this handoff: **same SO-101**, **same wrist camera + Logitech C920**,
**same 33.4 cm board** (41.75 mm squares), frames at **640×480 MJPG**. Bundled
lens files are used; this tool does not refit intrinsics.

## Place in OXAI

Copy this folder to:

```text
LeRobot_OXAI/lowlevel/click_to_3d/
```

The parent directory must contain `paths.py` (standard OXAI `lowlevel` layout).
That is how forward kinematics and `--robot` find the arm API.

## Install

No `helk` env is required. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Linux and macOS
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`--smoke` and `--demo-pair` need the packages above (including `placo` and
`draccus` for forward kinematics through OXAI `paths.py`).

`--robot` also needs the OXAI lowlevel lerobot checkout already present beside
this folder (`import paths` / SO-101 follower). Power the arm and use the same
USB port file as any other OXAI hardware script. Allow this program camera
access in the operating system's privacy settings.

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
6. On each view, click the four corners of that square. The **first click is green** and must be the corner nearest a1. Then click the other three **counter-clockwise in the picture**: next corner is toward the h-file, then the far corner, then the rank-8 side. A clockwise tour is the mirror image and is rejected. Rings snap to detected corners when available.
7. Press `u` for mount calibration. Park the arm, press `space` at least three times at **widely different arm rotations** (same square), then `s`. Each capture uses the same corner circuit: green nearest a1, then the other three counter-clockwise in the wrist image. Small nudges are not enough; hand–eye fails when the motion is too small. The click tool reloads the new mount when that window closes.
8. Press `c` again, then `space`, click a point, `y` (on the board) or `n`
   (above — then click the ray on the stationary view).

Do not zoom the C920. The bundled stationary matrix assumes the full 640×480 view (about 70° horizontal).

A yellow **BACKUP IN USE** banner for the bundled lens files is normal on a correct install.

Until you finish step 7, the tool uses the URDF wrist-camera estimate. Do not
copy another machine’s `output/camera_mount/wrist_camera_mount.yaml` as if it
were yours.

## What a click stores

The board frame has its origin at the a1 corner, +x toward the h-file, +y toward rank 8, and +z up. Squares are 41.75 mm.

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
| `y` / `n` | After click: on the board / above the board |
| `r` | Live preview |
| `q` | Quit |

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
