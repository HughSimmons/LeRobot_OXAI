# kinematicscalibration/

PyBullet/URDF gripper-geometry calibration tools. Moved out of
`../coord_finder/` (2026) because they do no image processing at all -- they
load the SO101 URDF/STL meshes in PyBullet and compute where the gripper
physically contacts the table vs. what forward kinematics predicts, then
cross-check that against real recorded joint-angle readings
(`../real_world_runs/current_pose*.json`). None of them touch real hardware
(all use PyBullet's headless `DIRECT` mode), so every script here is safe to
run directly.

## Environment

`pip install -r requirements.txt` (pybullet, numpy, Pillow), plus the
vendored `lerobot` package on `PYTHONPATH` for the two scripts that use it:
```
PYTHONPATH=../lerobot/src python compare_fk.py
PYTHONPATH=../lerobot/src python sweep_gripper_contact_point_vertical.py
```
The rest have no lerobot dependency and can be run with a plain
`python <script>.py`.

## Conventions

- `import paths` (from `lowlevel/paths.py`, one directory up) resolves
  `URDF_PATH`/`ASSETS_DIR` -- never hardcode a path to the SO-ARM100 assets.
- Several scripts run their PyBullet setup at module scope (no
  `if __name__ == "__main__":` guard) -- importing one is equivalent to
  running it.

## Representative scripts

- `find_gripper_contact_point.py`, `exact_lowest_point.py`,
  `exact_lowest_point_vertical.py` -- lowest-world-Z-point analysis for saved
  poses, at increasing levels of precision (PyBullet AABB → exact STL-vertex
  transform).
- `exact_base_lowest.py` -- same idea for the fixed base, not the gripper.
- `compare_fk.py`, `check_4_readings_affine.py`, `check_4_readings_offset.py`,
  `check_base_zero.py` -- cross-check forward-kinematics predictions against
  real recorded joint readings.
- `render_*.py` -- PyBullet renders of the arm/gripper from various views,
  written to `output/`. **Known duplication**: `render_origin_isolated.py`,
  `render_origin_level.py`, `render_origin_ortho.py`,
  `render_origin_with_table.py`, and `render_plate_isolated_origin.py` are
  five close variants of the same idea -- flagged for a future consolidation
  pass, not touched in this cleanup.
- `sweep_gripper_contact_point.py` / `_vertical.py` -- broader parameter
  sweeps over the same contact-point question.

## Verification

```
PYTHONPATH=../lerobot/src pytest ../tests/test_kinematicscalibration_imports.py -q
```
runs every script here as a subprocess and asserts a clean exit. To spot-check
by hand, covering each import pattern used across the set:
```
python find_gripper_contact_point.py      # plain URDF + real_world_runs
python exact_base_lowest.py               # ASSETS_DIR-derived
PYTHONPATH=../lerobot/src python compare_fk.py   # imports testkinematics
python render_fk_points.py                # PIL rendering
```
