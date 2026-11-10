# e4 → 3×3 neighbourhood: blind robustness and place-offset calibration

Started 2026-11-06 (Fri) on this machine (WSL, i7-1260P, 6 workers).
Status: **done**, including the f file (pivot-first fix, 21:26). Final figure: `e4_neighbourhood_bullseye_blind_FINAL_after.png`; final table: `e4_neighbourhood_FINAL_before_after.md`.

## Question

1. With the best e4→e5 settings, how robust is the rook move from e4 to each square of the
   3×3 block around it (d3–f5, plus e4→e4 pick-and-replace) when the robot is blind to
   placement noise?
2. Does re-calibrating the place offset per target improve the landings, and which way of
   calibrating is best?

## Fixed settings (every run)

| Setting | Value |
|---|---|
| Piece | `rook_kiri` mesh, `banded_hulls` collision |
| Noise | blind mode, 150 Gaussian start offsets, σ = 3.3 mm, **seed 0** (same 150 draws for every run) |
| Grasp offset | `(-20, -1, -3)` mm, from the 16 Oct e4→e5 grasp sweep (5×5 grid, `e4_to_e5_grasp_sweep_xy_20261016_155655`) |
| Placement corrections | 0 (corrections inside a run would leak the noise to the robot) |
| Other | defaults of `run_rook_offset_bullseye.py` (lower steps 10, gripper forces unchanged) |

## Conditions

| Condition | Run tag | Place offset |
|---|---|---|
| **before** | `e4ring` | `(-8.75, 2.23, -2.67)` mm for every target: calibrated on e4→e5 in **known** mode on 16 Oct |
| **after: clean** | `e4cal_clean` | per target, from the **mean** landing error of a **known-mode** 9×9 ±10 mm run (tag `e4ringknown`) |
| **after: blind mean** | `e4cal_mean` | per target, from the **mean** landing error of the 150 blind *before* landings |
| **after: blind median** | `e4cal_median` | per target, from the **median** landing error of the 150 blind *before* landings |

Calibration formula (same as `build_general_xy_lookup.corrected_place_offset`, gain 1):
`new_place = old_place + R_releaseᵀ · [bias_x, bias_y, 0]`, where `R_release` is the gripper
orientation at the release waypoint. The calibration script reproduces the 16 Oct value
`(-8.75, 2.23, -2.67)` exactly from run `e4_to_e5_bullseye_known_corr0_grid9_20261016_164404`.

Targets with fewer than 30 upright blind landings are not calibrated, because there is no
landing cloud to centre.

## Scripts

| Script | Role |
|---|---|
| `lowlevel/drafts/run_rook_offset_bullseye.py` | runs one bullseye (150 samples) |
| `lowlevel/drafts/calibrate_place_offset_from_bullseye.py` | bias → new place offset (median / trimmed-mean) |
| `lowlevel/drafts/run_e4_neighbourhood_calibration.sh` | driver for the clean/mean/median after-runs |
| `lowlevel/drafts/summarise_bullseye_conditions.py` | before/after table (`e4_neighbourhood_before_after.md/.csv`) |
| `lowlevel/drafts/plot_bullseye_neighbourhood.py` | 3×3 bullseye figure per condition |

The *before* sweep was driven by this loop (not in the repo):
`for T in e5 d5 f5 d4 f4 d3 e3 f3 e4: run_rook_offset_bullseye.py --mode blind --from-square e4 --to-square $T --random-samples 150 --noise-sigma 0.0033 --seed 0 --grasp-offset -0.02 -0.001 -0.003 --place-offset -0.00875 0.00223 -0.00267 --workers 6 --tag e4ring`

## Outputs

- Run folders: `e4_to_<target>_bullseye_<mode>_corr0_<random150|grid9>_<tag>_<timestamp>/`
- Calibrated place offsets: `e4_neighbourhood_place_offsets_{clean,mean,median}.json`
- Driver log: `e4_neighbourhood_calibration_log.txt`
- Tables: `e4_neighbourhood_before_after.md` / `.csv`
- Figures: `e4_neighbourhood_bullseye_blind_<tag>.png`

## Log / incidents

- **10:58**: first before-sweep launched with σ = 4 mm (script default) by mistake. Stopped after
  about 4 min, partial folder (logs only) deleted. Relaunched 11:02 with σ = 3.3 mm to match all earlier runs.
- **11:02**: before-sweep (`e4ring`) started.
- **e4→e5 before reproduces 16 Oct exactly** (126/150, `…_20261016_165155`), so the sim is deterministic.
- **e4→d3**: 4,948 s instead of about 320 s. Six samples each took about 4,600 s at the same moment, which means
  the machine paused (sleep). No errors; results valid.
- **Finding: f-file targets fail completely before calibration.** e4→f4 0/150, e4→f5 0/150.
  The rook is lifted fine, but touches down about two waypoints before release and ends up lying on
  its side (tilt ≈ 87°, z ≈ 14.6 mm). f/g/h targets use the separate far-reach pose
  (gripper not pointing straight down), for which these grasp/place offsets were never tuned.
  Not fixable by re-centring the place offset; f targets are excluded from calibration.

- **e4→e4**: 4,398 s, another machine pause (six samples ~4,000 s each). No errors; results valid.
- **e4→f3** fails like f4/f5 (1/150 upright).
- **14:19**: the automatic chain after the before-sweep never started, because the wait loop's
  `pgrep -f` matched its own shell. Stopped, and the driver was launched by hand at 14:20.
- **14:20–14:38**: known-mode runs (9×9 grid, ±10 mm, tag `e4ringknown`) for the six calibratable targets.
- **14:38–16:24**: 18 blind after-runs (clean, mean, median × 6 targets), about 5.5 min each, no pauses, no errors.

## Results

### Before (`e4ring`, figure `e4_neighbourhood_bullseye_blind_e4ring.png`)

| target | ≤5 mm | ≤2 mm | missed | fell | median error mm | median landing (x, y) mm | run |
|---|---|---|---|---|---|---|---|
| d5 | 125/150 | 88/150 | 1 | 6 | 1.49 | (−1.0, +0.2) | `…_e4ring_20261106_110758` |
| e5 | 126/150 | 91/150 | 1 | 5 | 1.28 | (−0.7, +0.2) | `…_e4ring_20261106_110225` |
| f5 | 0/150 | 0/150 | 1 | 148 | – | – | `…_e4ring_20261106_111313` |
| d4 | 128/150 | 80/150 | 1 | 7 | 1.78 | (−0.9, −0.6) | `…_e4ring_20261106_111918` |
| e4 (pick & replace) | 129/150 | 85/150 | 1 | 6 | 1.59 | (−0.7, −0.5) | `…_e4ring_20261106_130544` |
| f4 | 0/150 | 0/150 | 1 | 149 | – | – | `…_e4ring_20261106_112439` |
| d3 | 125/150 | 30/150 | 1 | 3 | 2.65 | (−0.9, −1.7) | `…_e4ring_20261106_113108` |
| e3 | 126/150 | 54/150 | 1 | 3 | 2.25 | (−0.6, −1.4) | `…_e4ring_20261106_125336` |
| f3 | 0/150 | 0/150 | 1 | 148 | – | – | `…_e4ring_20261106_125854` |

The final before/after numbers for every condition are in `e4_neighbourhood_before_after.md`.

### Calibrated place offsets (mm, gripper frame; from the before runs)

| target | blind median | blind mean | clean (known) |
|---|---|---|---|
| e5 | (−9.08, 1.56, −2.81) | (−9.18, 0.76, −2.93) | (−8.74, 2.35, −2.65) |
| d5 | (−9.19, 1.36, −2.85) | (−9.32, 0.70, −2.95) | (−8.78, 1.97, −2.71) |
| d4 | (−9.82, 1.90, −2.83) | (−9.96, 1.21, −2.96) | (−9.52, 2.54, −2.70) |
| d3 | (−10.62, 2.68, −2.73) | (−10.87, 2.05, −2.86) | (−10.38, 2.70, −2.71) |
| e3 | (−10.22, 2.68, −2.72) | (−10.50, 1.86, −2.89) | (−10.06, 2.75, −2.69) |
| e4 | (−9.57, 1.98, −2.80) | (−9.74, 1.16, −2.95) | (−9.41, 2.69, −2.67) |

Clean = from the known-mode runs (`e4ringknown`, 81/81 within 5 mm on every target; mean known landing
error per target, mm: e5 (+0.1, −0.1), d5 (−0.2, +0.1), d4 (−0.3, −0.8), e4 (−0.1, −0.8), d3 (−0.7, −1.6),
e3 (−0.5, −1.3); table in `e4_neighbourhood_known_e4ringknown.md`).

The blind mean is pulled about 0.7 mm further along x than the median by the far-miss tail
(the diagonal streak of red points in every panel).

### After (clean / mean / median)

Figures: `e4_neighbourhood_bullseye_blind_e4cal_{clean,mean,median}.png` (3×3, same layout as before),
`e4_neighbourhood_before_after.png` (one-page comparison). Full table: `e4_neighbourhood_before_after.md/.csv`.

Six calibrated targets pooled (900 samples per condition):

| condition | ≤5 mm | ≤2 mm | missed | fell | median error (median of per-target medians) |
|---|---|---|---|---|---|
| before | 759/900 (84%) | 428/900 (48%) | 6 | 30 | 1.68 mm |
| clean | 763/900 (85%) | 539/900 (60%) | 6 | 35 | 1.35 mm |
| blind mean | 761/900 (85%) | 535/900 (59%) | 6 | 28 | 1.71 mm |
| **blind median** | **770/900 (86%)** | **545/900 (61%)** | 6 | 35 | **1.32 mm** |

Per target, ≤2 mm count (before → clean / mean / median):
d5 88 → 91 / 90 / 93 · e5 91 → 88 / 93 / 96 · d4 80 → 92 / 93 / 92 · e4 85 → 91 / 90 / 89 ·
**d3 30 → 88 / 88 / 88** · **e3 54 → 89 / 81 / 87**.

Median landing point after (mm): clean leaves about (−0.7, 0) on d5/e5/d4/e4 and about (−0.2, 0) on d3/e3;
mean overshoots to about (+0.6, −0.3) everywhere; median lands on (0.0, 0.0) ± 0.1 everywhere.

**Conclusions**

1. **Calibration fixes precision, not robustness.** ≤5 mm barely moves (759 → 761–770, i.e. 0–2
   samples per target, within run-to-run noise). The failures are the diagonal far-miss streak and the
   falls (top-left / bottom-right red points in every panel), which come from large start offsets and
   do not depend on where the place offset is centred.
2. **≤2 mm rises from 48% to about 60% with every method**, almost all of it on d3 and e3, where the
   16 Oct e4→e5 place offset left a −1.5 mm rank bias (≤2 mm 30 → 88 on d3). Calibrating on e4→e5 only
   is good enough for d5/e5/d4/e4 but not for the rank-3 targets.
3. **Blind median is the best of the three**, and blind mean the worst: the far-miss tail pulls the
   mean about 0.7 mm too far along x, so mean-calibrated clouds overshoot (median error 1.71 mm, no
   better than before). Clean is close to median but leaves a −0.7 mm x bias on four targets: in
   blind mode the landings sit about 0.7 mm further back along x than in known mode, and the known runs
   cannot see that.
4. **Caveat: in-sample.** The blind mean/median offsets were fitted on the seed-0 before-landings and
   tested on the same seed-0 draws, so they are evaluated in-sample; the clean offsets are not (fitted
   on known-mode runs). The median's (0.0, 0.0) centring is therefore partly by construction. A
   held-out check (seed 1, before vs clean vs median) would settle it.

### Open follow-ups

- Held-out check of the median and clean offsets on a new noise seed (18 runs, about 1¾ h).
- f-file targets (f3–f5): see **f-file fix** below.
- Single e4→e5 before/after figure (user request, later).
- Optional: per-target grasp sweeps (the grasp offset is still the e4→e5 one, about 4 h).

## f-file fix: bend the arm so the gripper releases near-vertical (started 19:15)

**Diagnosis.** Gripper tilt from vertical at the release waypoint (FK of the planned trajectory):
e3/e4/e5 about 11°, **f3/f4/f5 33–35°**, g4 37°, h4 34°. For f/g/h targets `chess_traj.pickupmove_traj`
uses the far reach pose (`far_rows = ["f","g","h"]`, `downflag=False`, seed
`reach_pose = [0, 70, -90, 40, 90, 5]`). The IK weights orientation 0.01 against position 10, so the
tilt is set by that seed, not chosen. 34° is past the rook's critical tilt (27–33° in the tilt test),
and the leaning rook's bottom edge touches the board two waypoints before release, so it pivots
in the jaws and lies down.

**Reachability** (scratchpad `far_feasibility.py`: ask the IK for the same release point at
decreasing tilt, orientation weight 1). Most upright tilt with < 1 mm reach error:
f3 10.4°, f4 ~7–10°, f5 6.6°, g4 18.5°, h4/h8 no improvement (elbow at its −96.8° limit).
So f can be placed like d/e; g partly; h not without other changes (e.g. lower lift height).

**Change (opt-in, default behaviour unchanged).**
- `lowlevel/chess_traj.py`: new `far_place_max_tilt_deg()` (reads env `CHESS_FAR_PLACE_MAX_TILT_DEG`),
  `gripper_tilt_deg()`, `straighten_toward_down()`. When the env var is set, the far-file
  above-place waypoint and every place-lowering waypoint are re-solved: the gripper is rotated
  toward straight down (same heading) about the same place point, the arm bends to keep that point,
  and it stops at the cap or at the last orientation reaching within 1 mm. Pickup is untouched.
- `run_rook_offset_bullseye.py --far-place-max-tilt DEG`: sets the env var in every worker and stores
  `far_place_max_tilt_deg` in `results.json` config. `calibrate_place_offset_from_bullseye.py` reads it
  back so the release rotation it uses matches the run.
- Check with cap 11: release tilt f3 10.8°, f4 10.5°, f5 10.1°, release point unchanged (0.0 mm);
  e5 trajectory identical to before (11.0°). Tilt now ramps down during lowering like e5.

**Smoke test** e4→f4, known, 3×3 grid, cap 11 (`…_ftilt11_smoke_20261106_191219`): **9/9 upright**
(before: 0/150), xy error 3.0–4.8 mm (place offset not yet calibrated for f).

**Runs** (driver `lowlevel/drafts/run_e4_ffile_tilt_fix.sh`, log `e4_ffile_tilt_fix_log.txt`):
known 9×9 `e4ftilt11known`; blind 150 with 16 Oct place `e4ftilt11`; median-calibrated blind 150
`e4ftilt11_cal_median` (offsets `e4_ffile_tilt11_place_offsets_median.json`).

**Stopped 19:55** before any run finished (user asked for a pivot-first version instead; the laptop
had also slept 19:13–19:51). Only the smoke test above exists for the per-waypoint style.

## f-file fix v2: pivot first, then lower (started 20:01)

User request: pivot the top of the arm (wrist/gripper) first, then lower the rest; not a re-solve at
every waypoint. Constraint found first (`far_feasibility_dz.py`): on f the wrist is at its ~95° limit,
so the most upright reachable tilt depends on height above the release point:
0 cm 7–9°, 2 cm 10–12°, 4 cm ~15°, 6 cm 19–21°, 11 cm (carry height) 31–33°. A near-vertical pivot is
only possible close to the board.

**Change** (`chess_traj.py`, opt-in, `CHESS_FAR_PLACE_STYLE=pivot`, CLI `--far-place-style pivot
--far-place-pivot-height 0.02`, recorded in config): (a) lower in the original far pose to 2 cm
above the release point; (b) pivot once about the place point toward vertical, capped at 11°, about
2° per waypoint (the arm re-solves to keep the point fixed); (c) lower the last 2 cm with the
orientation locked (`solve_locked_orientation`). Planned profile e4→f4: tilt 35° down to 20 mm,
35→12° in place at 20 mm, 12° locked to release; release point unchanged. Non-far squares unchanged.

**Smoke tests** e4→f4, known, 3×3 grid (±16 mm corners), 16 Oct place offset:

| version | upright | mean landing (x, y) mm | run |
|---|---|---|---|
| original far pose | 0/150 (blind) | – | `…_e4ring_20261106_112439` |
| per-waypoint straighten, cap 11° | 9/9 | (+1.7, +3.5) | `…_ftilt11_smoke_20261106_191219` |
| pivot at 2 cm, 4 waypoints | 8/9 | (+3.8, +3.6) | `…_fpivot2cm_smoke_20261106_195722` |
| pivot at 3 cm, 4 waypoints | 8/9 | (+3.8, +3.7) | `…_fpivot3cm_smoke_20261106_195824` |
| **pivot at 2 cm, every step (~8 waypoints)** | **9/9** | (+4.2, +3.1) | `…_fpivot2cm_slow_smoke_20261106_200003` |

The 8/9 failures were the (−16, −16) mm corner slipping out of the jaws mid-pivot (premature
drop at waypoints 38–39), fixed by pivoting more slowly. The 3–4 mm landing bias is the uncalibrated
place offset (see "why the offset is off" below).

**Why the place offset is off on f.** The place offset is a vector in the gripper's own frame,
tuned on e4→e5 where the gripper heading (wrist roll) is −122°. On f the wrist roll is −155°, so
the same vector points about 33° differently on the board: in world terms (−4.3, −8.1) mm on e4/e5
against (−8.0, −4.6) mm on f4, a shift of about 5 mm, the size of the bias seen. Recalibrating per
target (step 3 of the driver) removes it.

**Runs** (driver `lowlevel/drafts/run_e4_ffile_pivot.sh`, log `e4_ffile_pivot_log.txt`): blind 150 with
16 Oct place `e4fpivot`; median-calibrated blind 150 `e4fpivot_cal_median`
(offsets `e4_ffile_pivot_place_offsets_median.json`).

**Results** (blind, 150 starts, σ = 3.3 mm, seed 0; table `e4_ffile_pivot_before_after.md`):

| target | original far pose | pivot, 16 Oct place | pivot + median calibration |
|---|---|---|---|
| f5 | 0/150 ≤5 mm (148 fell) | 75/150 · ≤2 mm 1 · F25 · median 4.78 mm | **115/150** · ≤2 mm 68 · F26 · 1.81 mm |
| f4 | 0/150 (149 fell) | 45/150 · ≤2 mm 2 · F23 · 5.41 mm | **115/150** · ≤2 mm 56 · F22 · 2.31 mm |
| f3 | 0/150 (148 fell) | 51/150 · ≤2 mm 1 · F18 · 5.47 mm | **125/150** · ≤2 mm 66 · F13 · 2.04 mm |
| total | 0/450 | 171/450 | **355/450 (79%)** |

Calibrated f place offsets (mm): f5 (−5.24, −0.30, −2.21), f4 (−4.24, −0.29, −2.01), f3 (−4.28, +0.02, −1.98).
The pivot removes the topple; calibration removes the ~5 mm offset. f still falls more often than
d/e (13–26 vs 3–9 per 150), so it is about 6–10 points below d/e on ≤5 mm.

### Final 3×3 (d/e: blind-median calibration, f: pivot + blind-median calibration)

**1125/1350 (83%) within 5 mm, 735/1350 (54%) within 2 mm, 96 fell**; before (original far pose,
16 Oct place for every target): 759/1350 (56%), 428/1350 (32%), 475 fell.
Figure `e4_neighbourhood_bullseye_blind_FINAL_after.png`, table `e4_neighbourhood_FINAL_before_after.md/.csv`.

## Speed-up for the block queue (21:30)

Profile of one f4 sample: 23 s, of which 19 s `pybullet.stepSimulation` (physics), 1 s planning;
12 s imports per worker, repeated every 8 samples (`maxtasksperchild=8`). Physics is untouched
(changing it would change results), so the gain is in fitting more workers in the 7.7 GB of RAM:

- `testkinematics.py` imports the real-robot driver `SO101Follower` at import time, which loads
  **torch** (~200 MB, ~5 s); placo loads every arm **mesh** when `RobotKinematics` reads the URDF
  (~440 MB; `Flags.ignore_collisions` does not avoid it). Neither is used by the simulation.
- `lowlevel/drafts/sim_no_robot_driver.py` (sim workers only; Hugh's files unchanged) stubs
  `lerobot.robots` and gives placo the URDF with `<visual>/<collision>` blocks removed.
  Worker RSS after imports 727 → 113 MB, about 336 MB while simulating.
- Checks: planned trajectories identical for 5 moves incl. pivot places
  (`check_sim_light_imports.py`, max joint difference 0); 28 stored e4→f4 samples re-run with
  12 and 16 workers are **bit-identical** (`_check_light_f4_12w`, `_check_light_f4_16w`);
  calibration reproduces the stored f4 offset exactly. `--full-imports` restores the old behaviour.
- Throughput (f4 samples): 6 workers 0.16/s, 12 workers 0.26/s, **16 workers 0.30/s (1.85×)**,
  RAM 5.3 GB used / 2.5 GB free at 16.

Queue `lowlevel/drafts/run_block_queue.sh`: sources e3 e5 d3 d4 d5 × targets d3–f5, same protocol
(blind 150 before with 16 Oct place → median calibration → blind 150 after; f targets pivot), 2
pairs at a time × 8 workers, resumable. Log `block_queue_log.txt`; offsets `block_place_offsets/`;
tags `blk` / `blk_cal_median`. See `BLOCK_QUEUE_EXPERIMENT.md`.
