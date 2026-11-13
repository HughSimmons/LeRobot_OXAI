# Click-to-3D (Oliver's fork)

Oliver's working copy of Asha's `click_to_3d/` (branch `origin/find_piece`, commit
`b8dc58e`, 2026-10-30), so it can change freely while `../click_to_3d/` tracks Asha's.
Asha's `5d5f4b4` (2026-11-06: recalibrated wrist lens, 42.0 mm squares, rings on both
views at `c`, cable-proof corner gating on the tripod view, Shift+click) was merged in on
2026-11-11.
Apart from shared files in `../` (`paths.py`, the URDF, `probe_xy_pose.py`,
`multisim_chess_fast.py`'s physics settings, `real_world_runs/`, `output/lens_calib/`), it reads
and writes only inside this folder.
Changes from Asha's version:

- **`base_mm` comes from the measured board, not the camera's depth.** A click's board
  position (from `c`: the image and the 4 clicked corners) is shifted by the ruler-measured
  board centre, (292.6, 0, 29) mm from the robot origin (z = the board's 2.9 cm thickness) (`BOARD_CENTRE_BASE_MM` in
  `click3d_shared.py`; the board's edges are taken as parallel to the robot's x / y axes).
  So joints, the mount and the wrist lens's depth don't enter it. Re-measure and edit that
  constant if the board moves. The camera-detected board, "cv" (Asha's route: the wrist camera's
  `c` pose + forward kinematics + the camera mount), is worked out too, as `base_mm_cv`; `b` (or
  `--base cv`) makes it the `base_mm` that is shown, that `g` moves to and that `p` picks and
  places at. Both are logged with every click, and each pick's `metadata.json` records which one
  it used (`target_source`). Before 2026-11-13 this route was called fk (`base_mm_fk`, `camera_fk`).
- Forward kinematics adds the joint calibration offsets from
  `../real_world_runs/friction/friction_sag_fit.json` (see "What a click stores"). That file
  is not in git; without it the offsets are zero and clicks land ~5 cm off in `base_mm`.
- Cameras are chosen by `/dev/v4l/by-id/` path (defaults in `../paths.py`, with their
  v4l2 controls, e.g. the wrist camera's `gain=0`), saved in `output/session_config.yaml`.
- `h` holds / releases the arm; the serial port stays open for the session.
- `space` always clicks on the views frozen at `c`, so the arm can move between clicks. You can
  click the live tripod (right) view instead, for a point on the board surface; see below.
- `g` moves the gripper to the last click via `../probe_xy_pose.py` (`move_to_click.py`).
  The board surface (z = 29 mm) is its floor; clicks logged before the board-centre
  change are refused.
- `f` toggles manual corner clicks (no snapping), here and in mount calibration.
- `u` releases the cameras before mount calibration starts, and reports a failure.

Run it with the click3d venv, from this folder:

```bash
~/venvs/click3d/bin/python run_click_to_3d.py --robot
```

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

This folder lives at `LeRobot_OXAI/lowlevel/click_to_3d_oliver/`. The parent directory must contain `paths.py` (standard OXAI `lowlevel` layout).
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

1. Install (above); the folder must sit directly in `LeRobot_OXAI/lowlevel/`.
2. Smoke test (no cameras, no arm):

   ```bash
   python run_click_to_3d.py --smoke
   python run_mount_calibration.py --smoke
   ```

   The mount smoke test writes only to a temporary folder (it checks the solver on synthetic
   runs with a known mount, including a board moved between runs).

3. Live cameras (and arm when ready):

   ```bash
   python run_click_to_3d.py --robot
   ```

4. Confirm **mounted camera on the left**, **stationary on the right**. Press `x`
   to swap them until that is true. Cameras default to `WRIST_CAMERA` /
   `STATIONARY_CAMERA` in `lowlevel/paths.py` (with their v4l2 controls, e.g. the
   Innomaker's `gain=0`); `--wrist-camera` / `--stationary-camera` take an index or
   device path. The choice is saved under `output/session_config.yaml` as
   `/dev/v4l/by-id/` paths, so it survives replugs.
5. Press `c`. Type the square (e.g. `d` then `6`) and Enter.
6. Rings appear on **both** views as soon as you press `c` — pick a square that
   has a complete cell in both cameras (any square is fine; it need not hold a
   piece). On each view, click the four corners of that square. The **first click
   is green** and must be the corner nearest a1. Then click the other three
   **counter-clockwise in the picture**: next corner is toward the h-file, then
   the far corner, then the rank-8 side. A clockwise tour is the mirror image and
   is rejected. Rings snap to detected corners when available; **Shift+click**
   accepts a free pixel when one snap is wrong, and `f` turns snapping off for all
   clicks. The stationary view gates corners with a padded LiveChess2FEN board quad
   (rectangular OpenCV crop, then accept/reject); the wrist / overhead view is ungated.
7. Press `u` for mount calibration. Park the arm, press `space` at least three times at **widely different arm rotations** (same square), then `s`. Each capture uses the same corner circuit: green nearest a1, then the other three counter-clockwise in the wrist image. Small nudges are not enough; hand–eye fails when the motion is too small. Every capture is saved as you take it (`output/camera_mount/runs/<date_time>/`), so quitting loses nothing and runs can be solved together later (see [Mount calibration](#mount-calibration)). The click tool reloads the new mount when that window closes.
8. Press `c` again, then `space`, click a point, `y` (on the board) or `n`
   (above — then click the ray on the stationary view).

Do not zoom the C920. The bundled stationary matrix assumes the full 640×480 view (about 70° horizontal).

A yellow **BACKUP IN USE** banner means a bundled lens file is in use. For the stationary C920
that is expected until its own calibration is in (the assumed 70° lens); for the wrist camera it
means `output/camera_mount/wrist_fisheye.yaml` is missing.

Until you finish step 7, the tool uses the URDF wrist-camera estimate. Do not
copy another machine’s `output/camera_mount/wrist_camera_mount.yaml` as if it
were yours.

## What a click stores

The board frame has its origin at the a1 corner, +x toward the h-file, +y toward rank 8, and +z up. Squares are 42.0 mm (33.6 cm / 8).

## Wrist lens calibration

```bash
python run_standard_wrist_fisheye.py   # default file (OpenCV full-point)
python run_refit_wrist_fisheye.py      # earlier dense-grid interim path
```

Both read `../sample images/wrist_photos_20261103_225124/` (drop `shot_001`) by default,
which is Asha's layout; here the photos are in `../lens_calibration/output/`, so pass
`--photos ../lens_calibration/output/wrist_photos_20261103_225124`. They write into
`output/camera_mount/` (`--out`), replacing the files there.
The click tool defaults to `wrist_fisheye.yaml`; use `m` → fisheye to select
`wrist_fisheye_status_quo.yaml` if needed.

Corner clicks in step 6, and each mount capture, snap to a detected ring within 12 pixels, solve the camera pose, and are not written to the log.

After `space`, the wrist click is the raw undistorted pixel. `y` intersects that ray with the board plane `z = 0`. `n` takes a second click on the stationary view and uses the midpoint of the two rays.

Or, after `space`, click the tripod (right) view: that ray meets the board plane directly
(`method` `side_board_plane`, no `y`/`n`, `wrist_px` empty). The tripod view is live, so it sees pieces
moved since `c`. Click where the piece meets the board, not its top. Accuracy depends on the tripod
pose from `c`, which comes from one square and an assumed 70° lens: checked against the 49
detected inner corners of a saved calibration (square f4), it was within 0.2 mm near f4, 2–6 mm
6–12 cm away and 11–28 mm further out.

Each saved point is one line in `output/sessions/session_log.jsonl`:

- `wrist_px` and `side_px` (`side_px` is empty for a wrist click on the board, `wrist_px` for a tripod click)
- `board_mm`, `base_mm`, `square`
- `base_mm_ruler` (board position + the ruler-measured board centre) and `base_mm_cv` (the
  camera-detected board: wrist camera's `c` pose + forward kinematics + mount); `base_source`
  (`board_centre_ruler` or `camera_cv`) says which one `base_mm` is, as chosen with `b` when the
  click was made. Clicks logged before 2026-11-11 have no `base_mm_ruler`; their `base_mm` is the
  ruler one. Clicks before 2026-11-13 call the camera route `base_mm_fk` / `camera_fk`.
- `T_base_board_cv_mm`: the camera-detected board pose behind `base_mm_cv` (added 2026-11-13; with
  `b` on cv, `p` maps the piece and the destination square's centre through it, x and y only)
- `method`: `board_plane` or `two_view`
- `ray_gap_mm`
- `joint_reading_deg_at_c` (motor readings when `c` froze the wrist image) and `joint_offsets_deg_at_c`
  (renamed 2026-11-04 from `joints_reading_deg` / `joint_offsets_deg`; older log lines keep the old names)

Forward kinematics adds the calibration offsets from
`lowlevel/real_world_runs/friction/friction_sag_fit.json` to the readings
(URDF angle = reading + offset), and prints them at startup. If they change, re-solve the
mount with `run_mount_calibration.py --resolve` (saved runs are redone with the new offsets).

Pixels are in that view’s undistorted image, origin at the top-left, rounded to 2 decimals.

## Keys (click tool)

| Key | Action |
| --- | --- |
| `c` | Calibrate both cameras to the board (extrinsics) |
| `x` | Swap cameras (mounted left, stationary right) |
| `f` | Corner clicks: snap to detected corners / manual (no snapping). Also in mount calibration; `--manual-corners` starts manual |
| `h` | Hold the arm where it is (torque on) / release it (torque off; support it first). `--robot` only, also in mount calibration |
| `m` | Asset menu (fisheye / stationary K / URDF / mount) |
| `u` | Launch mount calibration |
| `v` | Open 3D scene visualiser |
| `space` | Click on the wrist view frozen at `c` (same image and joints every time), or on the live tripod view (board surface only) |
| `y` / `n` | After click: on the board / above the board |
| `g` | Move the gripper to the last click (prompts in the terminal; see below) |
| `b` | `base_mm` (shown, `g`'s target, and both of `p`'s points) from the ruler-measured board centre / the camera-detected board, cv (`--base cv` starts there). The last two clicks switch too, so `g` and `p` can use the same clicks either way. The panel's `p (ruler)` / `p (cv)` line says which `p` will use |
| `o` | Compensation for `g` and `p`: OFF → BASELINE → ON. BASELINE (`p` only, `--baseline` starts it) sends the hand-measured offsets (`DEFAULT_JOINT_OFFSETS_DEG` in `multisim_chess_fast.py`) without the friction/sag model; ON (`--compensate`) sends the fitted offsets plus the model |
| `r` | Live preview |
| `q` | Quit |

## Move the arm to a click

```bash
python run_click_to_3d.py --robot               # g: go exactly to the clicked point
python run_click_to_3d.py --robot --compensate  # same, with offsets + friction/sag compensation
```

After a click, press `g`. The fixed-jaw tip (`gripper_frame_link`) goes to the click's `base_mm`,
pointing straight down. `base_mm` is in the same base frame as `lowlevel/probe_xy_pose.py --xy`,
and `g` runs that script unchanged as a subprocess, so the IK, the up-and-over via waypoint,
`--compensate` and the return to rest are all probe's. The window freezes while it runs; follow
the terminal:

1. **Dry run** (about 30 s, no arm). The move is refused, with the reason, if:
   - the click was logged before `base_mm` came from the measured board centre;
   - the target is more than 5 mm below the board surface (z = 29 mm). Between 0 and 5 mm
     below, it is raised to the surface;
   - the point is out of reach straight down (IK residual over 2 mm or tilt over 2°).
2. **Enter** connects to the arm. Support it: connecting turns torque off for a moment.
3. Probe asks `Move to probe pose? [yes/no]`, moves, and holds. **Enter** returns to rest and releases.

`--above-mm N` stops N mm above the point instead (default 0). Without `--robot`, `g` does the dry
run only. Each move is saved to `lowlevel/real_world_runs/click_moves/<time>_<square>_<mode>/`, mode `off` or `compensate` (`p` runs end in `off`, `compensate` or `baseline` too):
`click.json` (the click and the plan) and `probe_xy_pose.json` (commands and joint readings after the move).

`p` takes both points' x, y from the route `b` is on: with ruler, the piece click's board position and
the destination square's centre on the ruler-measured board (as always); with cv, the same two board
points mapped through each click's camera-detected board pose. The height stays the fixed board
surface, 29 mm. Each pick is saved to
`lowlevel/real_world_runs/click_picks/<time>_<piece>_click_to_<square>_<mode>[_cv]/`; after the arm run
its `metadata.json` also records `target_source` (`board_centre_ruler` or `camera_cv`),
`from_target_base_xy_mm` / `to_target_base_xy_mm` and `lookup_from_board_xy_m` / `lookup_to_board_xy_m`.
Runs before 2026-11-13 always used the ruler, whatever `base_source` their clicks show.

With `--robot`, every `g` and `p` also films the tripod view, with the `1` / `2` click markers, from
the moment the arm starts moving (not during the planning and dry run) until the move ends, and saves it as `tripod.mp4` in that run's folder (`click_moves/` or
`click_picks/`). It keeps real time at 30 fps. A move refused before the arm moved keeps no video; one
that errors keeps it in `output/videos/`.

Clicks are always made on the wrist and stationary images frozen at `c`, with the joints read
then, so the arm can move (with `g` or by hand) and you can click the next point straight away,
with no recalibration. Those images don't update: a piece moved since `c` is still shown where it
was. Press `c` again to get a fresh view. `g` again repeats the same point.

Each finished `c` is saved to `output/sessions/calibration/` (both frozen images, the joints and
both board poses), replacing the previous one, and the next start loads it, so the arm doesn't need
to be back at the `c` pose. It only holds while the board and tripod haven't moved; if they have, press `c`.
A saved calibration made with a different wrist lens file (e.g. before the 6 Nov recalibration) is
not reloaded: the status line asks for a fresh `c`.

## Compare a run with the sim

Each `p` run (from `../run_real_so101_from_xy_lookup.py --execute`) writes two timelines to its
folder: `timeline_hardware.json` (what the arm did, with measured send and read times) and
`timeline_sim.json` (the same commands on the planned schedule, one per `command_delay_s`). Three
scripts compare them. They run in the `IKsim_mj` env (they need pybullet), from anywhere. They only
simulate and plot, and never move the arm.

```bash
# Replay one run's plan in PyBullet on the planned schedule; writes <run>/pybullet_replay.json
python lowlevel/click_to_3d_oliver/compare_hardware_to_pybullet.py lowlevel/real_world_runs/click_picks/<run>
# Plot it: angles over time, or by command (--align index); --velocity; --error for hardware - sim
python lowlevel/click_to_3d_oliver/plot_pybullet_replay.py lowlevel/real_world_runs/click_picks/<run> --align index --velocity
# Tables over every replayed run in click_picks (or name run folders), to see if one run's findings hold
python lowlevel/click_to_3d_oliver/summarise_pybullet_replays.py
```

To replay every run that has timelines but no replay yet:

```bash
for run in lowlevel/real_world_runs/click_picks/*/; do [ -f "$run/timeline_hardware.json" ] && [ ! -f "$run/pybullet_replay.json" ] && python lowlevel/click_to_3d_oliver/compare_hardware_to_pybullet.py "$run"; done
```

Each script's docstring says what it compares and how.

## Mount calibration

```bash
python run_mount_calibration.py --robot
```

Or press `u` from the click tool. On every capture, the first click is green and is the corner nearest a1. Then click the other three corners counter-clockwise in the wrist image: toward the h-file, then the far corner, then the rank-8 side. A clockwise tour is the mirror and is rejected.

With `--robot`, both tools open the arm's serial port once and leave torque as it is
(lerobot's full connect switches torque off for a moment, which drops a loaded arm).
Press `h` to hold the arm before letting go of it, and to release it to re-park.

Each `space` + 4 clicks is saved straight away in `output/camera_mount/runs/<date_time>/`:
`sample_NN.png` (the raw wrist frame) and `samples.json` (square, clicked corners in both
the undistorted and the raw fisheye image, raw joint reading). `s` solves that run and writes:

- `output/camera_mount/wrist_camera_mount.yaml` (read by the click tool)
- `output/camera_mount/so101_new_calib_wrist_cam_calibrated.urdf` (for reference)
- `output/camera_mount/board_in_base_mm.yaml` (diagnostic; see below)

Each `s` or `--resolve` overwrites these three; the runs stay, so any earlier answer can be re-solved.

### Solving several runs together

```bash
python run_mount_calibration.py --resolve all                      # every run taken with --robot
python run_mount_calibration.py --resolve 20261111_163000 20261112_101500
python run_mount_calibration.py --resolve all --board-per-run      # the board moved between runs
python run_mount_calibration.py --resolve all --exclude 20261111_163000:3 --no-write
```

No camera or arm needed. Each sample is redone from its raw corners and raw joint reading with
the *current* lens file, URDF and joint offsets (`friction_sag_fit.json`), so old runs stay
usable after any of those change. `--exclude RUN:N` drops a mis-clicked sample; `--no-write`
only prints the fit. Runs from `--demo-pair` or without `--robot` are skipped by `all`.

Combining assumes the board did not move between runs, and the camera was not taken off its
mount. If the board moved, `--board-per-run` fits one board pose per run around a shared mount;
then each run needs its own 3+ widely rotated parks.

The solve: forward kinematics gives the gripper in the robot base, and each sample's 4 corners
give the camera in the board. OpenCV hand–eye (`calibrateHandEye`) gives a start, then the mount
and board pose are refined to minimise the corners' reprojection error over all samples
(Koide & Menegatti 2019, "General hand–eye calibration based on reprojection error
minimization"). It prints, per run, the reprojection rms (sub-pixel to ~1 px is good; a sample
far above the rest is likely mis-clicked); the mount's 1-sigma uncertainty (if a rotation is
over ~0.5° or a translation over a few mm, add parks with more varied rotation); the change from
the URDF estimate; and the fitted board's tilt and centre against the ruler measurement.

`board_in_base_mm.yaml` is that fitted board (the last run's with `--board-per-run`), a check
only. A click's `base_mm_cv` is the saved mount combined with the current `c` pose, so it follows
the board and the arm. Until a mount is saved, `base_mm_cv` uses the URDF camera estimate.

## Lens calibration photos

Taking and marking the photos for the wrist and stationary lens recalibration lives in
[`../lens_calibration/`](../lens_calibration/README.md).

## Demo without hardware

```bash
python run_click_to_3d.py --demo-pair --use-saved-poses
```

Bundled backups under `bundle/` are used when project files are missing. The yellow
**BACKUP IN USE** banner for those lens files is expected.
