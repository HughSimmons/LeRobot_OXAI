# Friction and gravity-sag model (`lowlevel/friction/`)

A servo joint commanded to an angle stops short of it: the servo pulls toward the target like
a spring, and gravity and gearbox friction hold it back. This folder fits a model of where each
joint really stops (the load-dependent friction model **M3**, Duclusaud et al., ICRA 2025,
arXiv 2410.08650, taken at rest) to sweeps recorded on the real arm, and uses it to
**compensate** commands so the arm lands where you want.

On poses the fit never saw it predicts the stop to about 0.2° for joints arriving downward
and 0.6° for joints arriving upward. The sim's current fixed shift is off by 2.4–3.9°.

Run everything from `lowlevel/` in the `IKsim_mj` env. Scripts that talk to the arm also need
`PYTHONPATH=lerobot/src`. Each script also works when run from another folder, since it puts
`lowlevel/` on `sys.path` itself.

## Compensating a command

```python
import numpy as np
from friction.joint_compensation import JointCompensator   # import from lowlevel/

comp = JointCompensator()   # loads real_world_runs/friction/friction_sag_fit.json
command, details = comp.compensate(want_deg, previous_command_deg=previous_command)
# send `command` to the servos
```

- **`want_deg`**: where you want the arm to stop, as **URDF angles** in degrees, shape (6,), in the
  order `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`. That's
  what IK and the lookups give you.
- **`previous_command_deg`**: the motor command the arm is moving from. It sets each joint's
  approach direction, which decides the friction case. Or pass `move_dirs=` (+1, −1 or 0 per
  joint) instead.
- **Returns**: the motor command (6,) to send, and one `JointCompensation` per joint with the
  offset, the predicted stopping error `f_deg` (target − reading) and the case used
  (`lifting`, `lowering`, `band`, `offset only`, `gripper`).

Per joint it computes `command = (want − offset) + f(want, direction)`:

- `offset`: the calibration offset (URDF angle = motor angle + offset).
- `f` for the joints that carry gravity (lift, elbow, wrist flex): the M3 stopping error, with
  the gravity torque computed at the wanted pose, which is where the joint will rest.
- `f` for pan and wrist roll (no gravity load): the fitted ±0.3° friction band.
- A joint moving less than 0.3° into the pose gets the offset only, since its direction is
  unknown. The gripper is left alone.

### Other functions in `joint_compensation.py`

| call | what it does |
|---|---|
| `comp.predict_rest_motor_deg(command, previous_command)` | the forward model: where a command will stop, as motor angles |
| `comp.resting_urdf_deg(reading)` | turns a servo reading into URDF angles, with the fitted offsets |
| `approach_dirs(target, previous)` | +1, −1 or 0 per joint for a move from `previous` to `target` |
| `comp.offsets_deg` | the offsets in use, in the joint order above |

Lower level, in `fit_friction_sag_model.py`: `predicted_up_error(tau_g, lifting, JointParams)`
gives the three M3 formulas, and `GravityModel().holding_torque(joint, pose_deg, offsets_deg)`
gives the gravity torque from the URDF.

### Already wired in

- `run_real_so101_from_xy_lookup.py --compensate [--compensation-fit FIT]` compensates every
  waypoint of a lookup trajectory (`compensate_waypoints`). Joints that hold still keep their
  last direction.
- `probe_xy_pose.py --compensate` compensates a single `--square` / `--xy` target, and prints the
  predicted and measured error.

### Things to know before relying on it

- **It's static.** It's exact only where the arm comes to rest (grasp, release). For waypoints
  passed through while moving, it's a quasi-static estimate.
- **Direction matters most.** For the same pose, a joint arriving upward stops about 3× further
  short than one arriving downward. Upward stops at high load also scatter by 0.5–1° between
  repeats, because the joint sometimes sticks, which M3 doesn't model. So the final approach of
  each loaded joint should be downward where possible.
- **Compensation can flip a direction.** Shifting a command can turn a small downward step into
  an upward one, so take directions from the commands you actually send.
- **Match the sim.** The sim's `JOINT_OFFSETS` and `JOINT_FRICTION` in `multisim_chess_fast.py`
  model an *uncompensated* arm. A compensated real run corresponds to a sim with
  `SIM_JOINT_OFFSETS_DEG="0 0 0 0 0"` and `SIM_JOINT_FRICTION_DEG="0 0 0 0 0"`. Build and verify
  lookups to match whichever you run.
- **Servo settings.** The fit assumes the gains `so_follower.py` writes: P=16, I=0, D=32. Change
  them and the model has to be refitted. The fit also assumes the current
  `orange_follower_arm` calibration file.
- **The pan offset is set by hand.** `friction_sag_fit.json` currently has `shoulder_pan: 5.0`.
  The fit doesn't estimate a pan offset; it copies multisim's (0). A refit writes 0 over the
  5.0, so restore it afterwards, or set it in `DEFAULT_JOINT_OFFSETS_DEG` first.

## Files

| file | what it is |
|---|---|
| `joint_compensation.py` | the compensator above |
| `fit_friction_sag_model.py` | the M3 fit: least squares of k_p, K_c, K_l per joint plus the calibration offsets (with the 16 Oct ruler calibration as prior), held-out scoring, and the formulas |
| `fit_servo_model_to_sweeps.py` | sweep loading shared by the other scripts, plus an older fit that replays each sweep in PyBullet with a PD servo and a constant friction torque (slow: ~35 min) |
| `run_friction_coverage_sweeps.py` | records the coverage sweeps on the real arm in one connection, with sim checks and a table guard |
| `probe_upward_stop_repeatability.py` | repeats one heavy upward lift move, logging the joint, to study the sticking |
| `plot_friction_sag_fit.py` | the held-out error and miss-vs-load figures |
| `system_identification.py` | an early draft: loads the sweep data per joint |

## Refitting

```bash
python friction/fit_friction_sag_model.py --train-dirs real_world_runs/friction_coverage/* real_world_runs/friction_validation/new_angles real_world_runs/friction_validation/new_pose --noise-deg 0.33 --held-out-sweeps
```

- `--noise-deg 0.33` is the real misfit. The default, 0.1, over-weights the data against the
  offset prior.
- `--held-out-sweeps` also scores whole poses the fit never saw, which takes about 30 min
  together with the per-angle scoring. Add `--no-held-out` for a quick fit (about 10 s).
- Other options: `--test-dirs` to score held-out folders, `--fix-offsets` / `--no-offset-prior`
  to change how the offsets are treated.
- The output overwrites `real_world_runs/friction/friction_sag_fit.json` unless you pass
  `--output`. Then redraw the figures with `python friction/plot_friction_sag_fit.py`.

Recording more data, without `--execute` (sim check only). Only run `--execute` yourself,
watching the arm:

```bash
PYTHONPATH=lerobot/src python friction/run_friction_coverage_sweeps.py
```

```bash
PYTHONPATH=lerobot/src python friction/probe_upward_stop_repeatability.py
```

## Data

All under `lowlevel/real_world_runs/`:

| folder | contents |
|---|---|
| `friction/` | the original 5 sweeps, the fit (`friction_sag_fit.json` / `.txt`), earlier fits, and `plots/` |
| `friction_coverage/`, `friction_coverage_poses/` | the 10 coverage sweeps and their plan |
| `friction_coverage_drifted/` | the first coverage run, whose held joints crept (kept, not fitted on) |
| `friction_validation/` | the new-angle and new-pose validation sweeps |
| `upward_stop_probe/` | the sticking probe's logs |
