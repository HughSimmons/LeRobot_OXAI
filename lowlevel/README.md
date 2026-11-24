# lowlevel/

PyBullet-based simulation, trajectory-generation, and hardware-calibration
code for driving the SO101 arm through chess moves. This directory has no
packaging (no `setup.py`/`pyproject.toml`) -- everything is run directly as a
script, e.g. `python lowlevel/build_general_xy_lookup.py ...`, from anywhere
in the repo (each script resolves its own paths via `paths.py`, below).

## Environment

Two conda environments cover this code (there is no single env with every
dependency): one with `pybullet`/`pinocchio`/`meshcat`/`numpy` for the
simulation and kinematics side, and one with `opencv`/`scikit-learn` for the
vision side (`coord_finder/`). Whichever env you use for simulation work also
needs the vendored `lerobot` package on `PYTHONPATH`:

```
PYTHONPATH=lowlevel/lerobot/src python lowlevel/<script>.py
```

The simulation/kinematics env used throughout this README and its tests is
`IKsim_mj`; [`realsimenv.yml`](realsimenv.yml) is its full `conda env export`
-- create it with:
```
conda env create -n IKsim_mj -f lowlevel/realsimenv.yml
```
Keep it in sync if you add a dependency: `conda env export -n IKsim_mj > lowlevel/realsimenv.yml`.
The vision env used for `coord_finder/` in this README is `Live2FEN`; see
[`coord_finder/live2fen_env.yml`](coord_finder/live2fen_env.yml) and
[`coord_finder/README.md`](coord_finder/README.md#environment).

## Board layout

> **Board origin changed on 2026-11-03.** `DEFAULT_BOARD_ORIGIN` in `board_coordinates.py`
> and the `SIM_BOARD_ORIGIN_X` / `SIM_BOARD_ORIGIN_Z` defaults in `multisim_chess_fast.py`
> moved from (0.27, 0, 0.03) to x = 0.2926 m, to match the ruler-measured board on Oliver's
> setup (z: 0.029 in `DEFAULT_BOARD_ORIGIN`, 0.024 in the sim, which puts the sim's playing
> surface at the measured 29 mm). What that changes:
>
> - **Lookups built before the change are off by 22.6 mm.** They were built with the board at
>   x = 0.27, but lookups don't record their board origin: `run_real_so101_from_xy_lookup.py`
>   takes it from `DEFAULT_BOARD_ORIGIN`. Replayed now, an old lookup (everything under
>   `rook_kiri_xy_lookup/`, and `real_world_runs/click_picks/` runs from before the change)
>   aims 22.6 mm closer to the robot than it should. To replay one, set
>   `DEFAULT_BOARD_ORIGIN` back to `(0.27, 0.0, 0.03)` first.
> - Square names and `--frame board` coordinates now resolve around x = 0.2926, in the sim,
>   the lookup builders and `chess_traj.py`.
> - `two_camera/camera_rig.py` places its board, and the tripod around it, from the same
>   constant, so its scene moved too.
> - For the old board in the sim alone, set `SIM_BOARD_ORIGIN_X=0.27 SIM_BOARD_ORIGIN_Z=0.03`.
>   The coordinate conversion still follows `DEFAULT_BOARD_ORIGIN`, so the two then disagree.

The robot does **not** sit behind rank 1 like a player. It sits beside the
a-file, with white on its right. `square_center_world_xy()` in
`board_coordinates.py` maps **files to world x** (away from the robot) and
**ranks to world y** (+y is the robot's left). Top-down view, robot at the
bottom:

```
                   +x (away from robot)
                         ^
            h8 . . . . . . . . h1      h-file centres ~43.7 cm out
            .                   .
     +y <-  .   board centre    .      rank 8 on the robot's left,
            .  (x=0.2926, y=0)  .      rank 1 (white) on its right
            .                   .
            a8 . . . . . . . . a1      a-file centres ~14.8 cm out
                     [robot]
```

- Board centre: 29.26 cm straight out from the robot base origin, on its
  centreline (`DEFAULT_BOARD_ORIGIN`; 27 cm before 2026-11-03, see above).
- Squares: 41.25 mm (`DEFAULT_SQUARE_SIZE`).
- Height: the z of the board origin is the centre of the 10 mm board slab, so the
  playing surface is 5 mm higher: 2.9 cm in sim (z = 0.024), 3.0 cm for the real arm by
  default (`DEFAULT_REAL_BOARD_ORIGIN` in `run_real_so101_from_xy_lookup.py`, z = 0.025).

So a "d4→d5" lookup moves the piece one square **sideways**, to the robot's
left. If the board is set up the player's way round (robot behind rank 1),
that same lookup plays e4→d4 instead. Every lookup table and the simulation
use this layout, so rotate the physical board to match it rather than
changing the code.

## Layout

- **Core library** -- shared modules other scripts import, not run directly:
  - `board_coordinates.py` -- world-frame XY / algebraic-square dataclasses
  - `chess_traj.py` -- trajectory/IK generation (uses `pinocchio`)
  - `testkinematics.py` -- wraps `lerobot.model.kinematics.RobotKinematics`
  - `continuous_xy_candidate_db.py` -- sqlite-backed grasp-candidate store
  - `multisim_chess_fast.py` -- the PyBullet simulation engine
  - `paths.py` -- **import this instead of hardcoding paths.** Exposes
    `REPO_ROOT`, `LOWLEVEL_DIR`, `URDF_PATH`, `ASSETS_DIR`, `MESH_DIR`,
    `REAL_WORLD_RUNS_DIR`, all resolved relative to the repo so the code
    works regardless of where it's cloned.

- **Canonical pipeline** -- run these to build and check a lookup table of
  robot moves:
  ```
  python lowlevel/build_general_xy_lookup.py --help
  python lowlevel/verify_xy_lookup_move.py --help
  python lowlevel/build_general_nonh_reverse_lookup.py   # no CLI flags; runs a full build
  python lowlevel/verify_nonh_lookup_moves.py --help
  python lowlevel/verify_continuous_xy_grid.py --help
  python lowlevel/visualize_lookup_success.py --help
  ```
  Plus newer diagnostic probes for a shoulder-lift zero-offset/servo-sag issue
  found on real hardware: `toggle_joint_calibration.py`, `read_current_pose.py`,
  `probe_xy_pose.py`, `probe_right_angle_pose.py`, `fk_from_pose_json.py`,
  `live_calibrate_so101_offsets.py`, `run_lookup_sequence.py`.

- **`friction/`** -- the joint friction and gravity-sag model: fitting it to
  real-arm sweeps, recording the sweeps, and compensating commands so the arm
  stops where it's sent. See [`friction/README.md`](friction/README.md).
  The fit it loads, `real_world_runs/friction/friction_sag_fit.json`, is not
  in git (`real_world_runs/` stays on the lab machine): without it
  `--compensate` fails and the click tools use zero joint offsets.

- **`lens_calibration/`** -- photos for recalibrating the wrist and stationary
  camera lenses: planned, collision-checked wrist-camera arm poses, a guarded
  script that drives the arm through them, and a tool for marking the
  calibration square in each photo. See
  [`lens_calibration/README.md`](lens_calibration/README.md).

- **`click_to_3d/`** -- Asha's click-to-3D tool: calibrate the wrist and
  stationary cameras to the board, click a point in the wrist image, and get
  it in board and arm-base millimetres. See
  [`click_to_3d/README.md`](click_to_3d/README.md).

- **`click_to_3d_oliver/`** -- Oliver's working copy of `click_to_3d/`. It also
  moves the arm to a click (`g`), picks and places between clicks (`p`), and
  holds the scripts that replay a pick run in PyBullet and compare it with
  the hardware. See [`click_to_3d_oliver/README.md`](click_to_3d_oliver/README.md).

  `capture_camera_pair.py` previews the wrist and stationary cameras side by
  side and saves synchronised image pairs (space = pair, c = calibration pair)
  to `outputs/camera_pairs/<session>/` with a `pairs.csv` log. With `--teleop`
  it also runs leader->follower teleop itself (so it can share the serial port)
  and logs the follower's joints per pair as `current_pose_v1` JSON, readable
  by `fk_from_pose_json.py`. **`--teleop` moves the real arm -- a human runs it.**

  **`run_real_so101_from_xy_lookup.py` drives the physical robot arm -- it is
  never run automatically by an assistant; a human runs it deliberately.**

  ### Example: real hardware -- smoketest, then full run

  Always dry-run first (omit `--execute`): this regenerates the trajectory
  and writes a preview JSON without connecting to the arm at all, so it's
  safe to run any time.
  ```bash
  PYTHONPATH=lowlevel/lerobot/src /opt/miniconda3/envs/IKsim_mj/bin/python lowlevel/run_real_so101_from_xy_lookup.py \
    --lookup-json "lowlevel/rook_kiri_xy_lookup/real_world2_x027_no_edge_d4_d5_full_lookup_20260821/lookup/d4_x027_full_to_d5_x027_full.json" \
    --output-dir /tmp/hardware_smoketest_dryrun
  ```
  Only once that output looks right, add `--execute` and the arm's serial
  port to actually move it -- start with `--pause-at-pickup` so you can
  confirm the grip before it lifts, and `--allow-start-bridge` if the arm
  isn't already near the trajectory's start pose:
  ```bash
  PYTHONPATH=lowlevel/lerobot/src /opt/miniconda3/envs/IKsim_mj/bin/python lowlevel/run_real_so101_from_xy_lookup.py \
    --lookup-json "lowlevel/rook_kiri_xy_lookup/real_world2_x027_no_edge_d4_d5_full_lookup_20260821/lookup/d4_x027_full_to_d5_x027_full.json" \
    --output-dir /tmp/hardware_full_run \
    --port /dev/tty.usbmodem5B7B0157051 \
    --pause-at-pickup \
    --allow-start-bridge \
    --execute
  ```
  `--port` is your arm's serial device (`ls /dev/tty.usb*` to find it).
  **This is the only command in this README that moves the physical arm** --
  run it yourself, deliberately, never as part of an automated/assistant flow.

  An `--execute` run writes two timelines to its output folder:
  `timeline_hardware.json` (what the arm did, with measured send/read times)
  and `timeline_sim.json` (the same commands on the planned schedule, one per
  `--command-delay`, with no readings). Each file's `fields` block describes
  its fields. The scripts that replay a run in PyBullet and compare it with
  the hardware (one run, plots, and a summary over many runs) are in
  `click_to_3d_oliver/`; see "Compare a run with the sim" in
  [its README](click_to_3d_oliver/README.md).

  ### Example: build + verify a lookup

  The piece model used by the simulation is chosen via the
  `LOOKUP_PIECE_MODEL` env var (`cylinder`, the default, or `rook_kiri`), and
  when using `rook_kiri`, `ROOK_KIRI_COLLISION_MODEL` picks the collision
  geometry (`cylinder_proxy` default, `mesh_convex`, or `banded_hulls`). Build
  and verify must use the *same* env vars, since the replay scores against
  whatever collision geometry the lookup was built with.

  `SIM_GRIPPER_MODEL` picks how the sim gripper is shaped and driven, and must
  also match between build and verify. `legacy` (default, used by every lookup
  so far) collides with the convex hull of each gripper mesh, which puts a
  phantom fixed-jaw face ~13 mm from the real one, and grips thanks to a
  tuned moving-jaw friction of 2.0. `realistic` loads
  `urdf/so101_with_mount_vhacd_gripper.urdf` (gripper split into convex parts
  that follow the real jaws; regenerate with
  `urdf/build_vhacd_gripper_urdf.py`) and closes the gripper like the real one:
  until the piece stops it, at a torque near the real servo's. See the
  comment above `SIM_GRIPPER_MODEL` in `multisim_chess_fast.py`.

  The simulation includes the real arm's joint calibration offsets x (pan,
  lift, elbow, wrist flex, wrist roll; URDF angle = motor angle + x): every
  commanded arm angle lands x away, as on a real arm whose trajectory doesn't
  correct for them. The defaults are `orange_follower_arm`'s, set as
  `DEFAULT_JOINT_OFFSETS_DEG` (2.6, 2.0, -3.0, -4.6, 0) in
  `multisim_chess_fast.py`. Every run uses exactly these values (there is no
  random draw; until 2026-11-19 each run drew them from a normal distribution),
  and each result records them. Override them with `SIM_JOINT_OFFSETS_DEG`
  (`"0 0 0 0 0"` gives exact tracking). The same build/verify rule applies.

  It also includes joint friction, modelled as a **directional undershoot**:
  each arm joint stops short of its target by its friction h, on the side it's
  coming from. Moving down onto a target it stops slightly high, moving up it
  stops slightly low. This is a consistent bias, not random noise: the arm
  always lowers onto a piece the same way, so friction makes it stop short the
  same way every time (roughly 2 mm per joint at the tip), and a consistent
  miss can be corrected where noise can't.
  - **Measured** on `orange_follower_arm` (lift, elbow and wrist flex 22 Oct
    2026; pan and wrist roll 24 Oct 2026) with `toggle_joint_calibration.py`
    sweeps that reach each angle from above and below (e.g.
    `--values -90 -75 -60 -45 -30 -45 -60 -75 -90`; starting poses in
    `real_world_runs/friction_reference_poses/`, raw files in
    `real_world_runs/friction/`). At each angle reached both ways, h is half
    the gap between the two readings; gravity sag is the same both ways, so it
    cancels.
  - **Defaults** in `multisim_chess_fast.py`: `DEFAULT_JOINT_FRICTION_DEG`
    (0.29, 0.84, 0.76, 0.47, 0.29 for pan, lift, elbow, wrist flex, wrist roll)
    is the average over the angles measured. Friction changes with pose on
    the joints gravity loads (spread 0.46-0.48 deg on lift and elbow); pan and
    wrist roll carry no gravity load and barely vary. Every run uses exactly
    these values (no random draw), and each result records them.
  - **Measured with LeRobot's servo I gain at 0.** A nonzero I gain keeps the
    servo pushing until it reaches its target, which should remove most of
    this; remeasure after changing it.
  - Override with `SIM_JOINT_FRICTION_DEG`; `"0 0 0 0 0"` turns friction off.
  - **A better model** is in [`friction/`](friction/README.md): load-dependent
    friction plus gravity sag, fitted to these and later sweeps, about 0.2-0.6
    deg of held-out error against 2.4-3.9 deg for this fixed shift. It can also
    compensate real-arm commands (`run_real_so101_from_xy_lookup.py
    --compensate`).

  Default cylinder piece:
  ```bash
  rm -rf /tmp/smoketest_lookup
  PYTHONPATH=lowlevel/lerobot/src /opt/miniconda3/envs/IKsim_mj/bin/python lowlevel/build_general_xy_lookup.py \
    --from-xy 0.249375 -0.020625 --to-xy 0.249375 0.020625 --frame world \
    --from-name d4_smoketest --to-name d5_smoketest \
    --output-dir /tmp/smoketest_lookup --grid-radius 0

  PYTHONPATH=lowlevel/lerobot/src /opt/miniconda3/envs/IKsim_mj/bin/python lowlevel/verify_xy_lookup_move.py \
    /tmp/smoketest_lookup/d4_smoketest_to_d5_smoketest.json --output-dir /tmp/smoketest_verify --video
  ```

  Rook piece model, with the higher-fidelity banded-hulls collision geometry:
  ```bash
  rm -rf /tmp/rook_lookup_banded
  LOOKUP_PIECE_MODEL=rook_kiri ROOK_KIRI_COLLISION_MODEL=banded_hulls \
  PYTHONPATH=lowlevel/lerobot/src /opt/miniconda3/envs/IKsim_mj/bin/python lowlevel/build_general_xy_lookup.py \
    --from-xy 0.249375 -0.020625 --to-xy 0.249375 0.020625 --frame world \
    --from-name d4_rook --to-name d5_rook \
    --output-dir /tmp/rook_lookup_banded --grid-radius 0

  LOOKUP_PIECE_MODEL=rook_kiri ROOK_KIRI_COLLISION_MODEL=banded_hulls \
  PYTHONPATH=lowlevel/lerobot/src /opt/miniconda3/envs/IKsim_mj/bin/python lowlevel/verify_xy_lookup_move.py \
    /tmp/rook_lookup_banded/d4_rook_to_d5_rook.json --output-dir /tmp/rook_verify_banded --video
  ```

- **`legacy/`** -- superseded or abandoned scripts, kept for reference, not
  imported by anything above. Includes an abandoned early MuJoCo experiment
  (`egphysics.py`, `create_mjenv.py`, `testmujoco.py`), an abandoned
  two-camera vision prototype (`main*.py`, `vision*.py`, `camfk.py`), square/
  rank-specific one-off lookup builders from before the general pipeline
  existed, and `legacy/drafts/` -- 26 one-off experiment scripts that never
  fed back into the core library.

- **`kinematicscalibration/`** -- PyBullet/URDF gripper-contact-point
  calibration tools (moved out of `coord_finder/`, which they never belonged
  in -- no image processing at all). See its own README.

- **`coord_finder/`** -- the vision pipeline (board + piece detection from
  camera images). See its own README.

- **Everything else** (`archivejson/`, `recordings/`, `recordings_archive/`,
  `rook_kiri_lookup/`, `rook_kiri_xy_lookup/`, `mirror_runs/`,
  `cleanup_cylinder/`, `logs/`, `lerobot/`, `real_world_runs/`, etc.) is
  generated run data or a vendored dependency, intentionally untouched by
  this cleanup pass.

## Tests

```
PYTHONPATH=lowlevel/lerobot/src pytest lowlevel/tests -q
```
Covers: `paths.py` resolving correctly, every `kinematicscalibration/` script
still running cleanly from its new location, and a hardcoded-path regression
guard over `legacy/`.
