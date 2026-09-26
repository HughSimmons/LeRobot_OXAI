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

  ### Example: build + verify a lookup

  The piece model used by the simulation is chosen via the
  `LOOKUP_PIECE_MODEL` env var (`cylinder`, the default, or `rook_kiri`), and
  when using `rook_kiri`, `ROOK_KIRI_COLLISION_MODEL` picks the collision
  geometry (`cylinder_proxy` default, `mesh_convex`, or `banded_hulls`). Build
  and verify must use the *same* env vars, since the replay scores against
  whatever collision geometry the lookup was built with.

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
