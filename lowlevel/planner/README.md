# `planner` — SO-101 online chess trajectory planner (production)

Single deployed algorithm: **`DLSNullspacePlanner`** (`dls_nullspace.py`) — damped-least-squares
Cartesian stepping with null-space repulsion from an analytical capsule-SDF scene model.
It was selected from three candidate families (DLS + null-space, B-spline TrajOpt, RRT-Connect)
by the PyBullet benchmark in [`../planner_benchmarks`](../planner_benchmarks/README.md):
highest strict pass rate (7/16 on crowded boards), deterministic output, ~140-190 ms planning.

## Pipeline

1. `frames.py` — board/world frames; `coord_finder` mm -> world m; off-board graveyard slots.
2. `kinematics_so101.py` — FK / analytic Jacobian and the **coupled pick-place IK**:
   strict 0° vertical gripper where reachable (files a–e), otherwise the minimum feasible tilt,
   with pick and place gripper attitude matched (preemptive tilt on near squares when the
   destination is a far square).
3. `collision_sdf.py` — capsule-SDF collision model: standing pieces, board/table, all robot
   links, wrist-camera mount, both jaws, and the piece rigidly held in the gripper.
4. `dls_nullspace.py` + `base.py` — planner, minimum-jerk timing, jaw open/close, throat-axis
   retreat, and 2-leg captures to off-board graveyard (`plan_move_or_capture`).
5. `compensation.py` — calibration offsets everywhere + friction/sag stopping-error compensation at
   rest waypoints. With a `friction_sag_fit.json` it delegates to Oliver's validated
   `friction.joint_compensation.JointCompensator` when importable, else uses an equivalent
   built-in model. `_paths_compat.py` hides the differences between the two `paths.py` variants.
6. `safety.py` — pre-motion gate (IK error, transit clearance, joint limits) + tilt/reach notices.
7. `hardware.py` — everything that touches the servo bus: connect, retried reads, live-pose start,
   fixed-schedule 60 Hz streaming (trajectory speed = planner timing), hold-torque-on-error.
8. `click_bridge.py` — `pick_piece_planned()`: Oliver's two clicks -> planned pick-and-place.
9. `run_planned_move_real.py` — CLI for squares / grasp JSON; dry-run by default.

## Usage

```bash
# Dry-run (no hardware): near -> far move with matched tilt
python lowlevel/planner/run_planned_move_real.py --from-sq d1 --to-sq h5 --piece Q

# 2-leg capture
python lowlevel/planner/run_planned_move_real.py --from-sq e4 --to-sq d5 --piece P --capture

# Continuous coord_finder grasp, then real execution (asks for confirmation)
python lowlevel/planner/run_planned_move_real.py --from-grasp-json grasp.json --to-sq e4 --execute

# Compensation: --compensation off | baseline (default, offsets only) | on (offsets + friction/sag)
```

```python
from planner import (DLSNullspacePlanner, MoveRequest, SO101Kinematics, SceneCollisionChecker,
                     SIM_BOARD_CONFIG, plan_move_or_capture)
kin = SO101Kinematics()
checker = SceneCollisionChecker(kin, SIM_BOARD_CONFIG, pieces)
plan = plan_move_or_capture(DLSNullspacePlanner(kin, SIM_BOARD_CONFIG),
                            MoveRequest(from_target="e2", to_target="e4"), checker)
```

## Running on Oliver's hardware (`oliver/two-camera-capture`)

Assumes his checkout already has working calibration, `paths.py` (port / robot id / URDF), the
`SO-ARM100` submodule (STL meshes), the `lowlevel/lerobot/src` tree and his venv. This planner needs
only `numpy`, `scipy`, `pybullet` (and `pinocchio` if installed; PyBullet is the fallback) on top.

1. Copy `lowlevel/planner/` (this folder, including `assets/`) into his `LeRobot_OXAI/lowlevel/`.
2. Apply the click-tool patch (adds key **`d`**, leaves `k` untouched), from the repo root:
   ```bash
   git apply lowlevel/planner/integration/click_to_3d_oliver_hotkey.patch
   ```
   Undo with `git apply -R ...`. The patch is against `run_click_to_3d.py` as of his branch head
   `oliver/two-camera-capture`; if it no longer applies, add the `d` branch by hand: it is a copy of
   `pick_piece_at_click` that calls `planner.click_bridge.pick_piece_planned`.
3. Run his tool as usual (with `PYTHONPATH=lowlevel/lerobot/src` if his launcher does not set it):
   calibrate (`c`), click the piece, click the destination square, then **`d`** instead of `k`.
   `b` (ruler / camera-detected board) and `o` (off / baseline / on compensation) apply to `d` too.
   Without `--robot`, `d` is a dry run: it plans, prints the safety report and writes
   `plan_preview.json`, and never opens the serial port.
4. With `--robot`, `d` is the go-ahead (no further prompt, as with `k`): it plans once from the home
   pose (no arm) and refuses on a blocked gate; then connects (**support the arm**: connecting can drop
   torque briefly), reads the live pose, re-plans from it, re-checks the gate, and streams the
   trajectory. On error / Ctrl-C the arm holds until you press Enter.
5. Output goes to `real_world_runs/click_picks/<stamp>_<from>_to_<to>_<mode>[_cv]_planner/` with his
   `metadata.json` keys plus `planner: dls_nullspace`, `plan_preview.json` / `planned_move.json`
   (commands and joint readings). His replay/plot step needs timelines this path does not write.

Scope and assumptions: empty board (only the moving piece is modelled; no neighbours), default piece
height 40 mm / radius 12.5 mm (the sim rook; override with `PLANNER_PIECE_HEIGHT_MM` /
`PLANNER_PIECE_RADIUS_MM`), trajectory timing unchanged (~7 s per move). The destination is always the
square centre. The plan opens the gripper from its first command and leaves it open at the home pose.

Offline checks (no arm): `python lowlevel/planner/tests/test_hardware_fake_follower.py` and
`OLIVER_CLICK_DIR=<folder with his move_to_click.py> python lowlevel/planner/tests/test_click_bridge_synthetic.py`.
**Not validated on the real arm**: everything above was exercised against fake followers only.

## Known limits

- The planner matches pick and place gripper attitude and tilts where the strict-vertical envelope
  ends: dry runs gave ~12° for e2 -> e4, ~16° for c2 -> f5 and ~47° for d1 -> h5, but 0° for d2 -> d4.
  Oliver's `a` / `k` tools refuse > 2°; `d` prints a notice instead of refusing. Read the tilt report
  before trusting a first run.
- Compensated motor commands can land a few degrees outside the URDF joint range when a pose sits at
  a joint limit (printed as a notice); the arm may stop short there.
- Pickups on far `g/h` files (R ≈ 390–440 mm) sit at the 251 mm arm-span ceiling: tilt is forced
  to 23–37° and there is little vertical lift headroom. Moving the board ~40–50 mm closer to
  the base is the highest-leverage hardware fix.
- `assets/urdf/so101_with_mount_vhacd_gripper.urdf` is generated at runtime from the
  `.template.urdf` (see `compensation.resolve_vhacd_urdf`) and is git-ignored.
- Requires `numpy`, `scipy`, `pybullet` (`pinocchio` optional); `--execute` / `d --robot` additionally
  need `lerobot` and `paths.connect_so101_hardware`.
- Without `friction_sag_fit.json`, `--compensation on` falls back to generic M3 defaults (a notice
  is printed); use `baseline` unless a fit exists for this arm.
