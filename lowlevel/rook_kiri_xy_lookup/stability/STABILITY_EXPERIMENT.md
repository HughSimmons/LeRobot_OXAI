# Pickup and set-down stability (e4 pilot)

Started 2026-11-10 on this machine (WSL, i7-1260P, 16 workers, light sim imports).
Status: **e4 pickup pilot done** (6561 samples, 64 min, 0 errors). Next: f4 (see bottom).

## Question

Does the rook *get there* without dropping, falling over or slipping, wherever it starts in the
square? Which grasp offset (pickup) and which set-down settings give the largest stable region?
Accuracy is not part of this (already improved by place-offset calibration); it is recorded only.

## Pass rules (known mode: the robot is told the true start offset)

A drop is anything that is not upright on the board before and after.

| phase | check | limit |
|---|---|---|
| before grasp | rook tilt / pushed by the approaching jaws | ≤ 3° / ≤ 3 mm |
| pickup | lifted at all | otherwise `missed_grasp` |
| carry | both jaws still touching at the end of the carry | otherwise `dropped_in_carry` |
| carry | rook position change in the jaws, lift-off → end of carry | ≤ 2 mm, otherwise `slipped_in_jaws` |
| lowering | slip in the jaws before release, and both jaws touching at release | ≤ 5 mm, otherwise `slipped_before_release` |
| after | final tilt | ≤ 3° (3–15° `leaning`, > 15° `toppled`) |
| after | movement on the board from first board contact to rest / during retreat | ≤ 1 mm (`slid_on_board` / `dragged_on_retreat`) |
| after | at rest after settling | < 1 mm/s, < 3°/s (`not_settled`) |

In-jaw tilt (angle between the rook axis and the gripper-fixed direction it had just before closing)
is recorded as a **margin** and used to break ties; it is not a failure (the baseline already tilts
9–12° in the jaws on e4→e5 and lands at 0.05°).

**Stable area** for one setting = the 4-connected region of passing start offsets that contains the
square centre, in mm² (start grid 9×9, ±16 mm, 4 mm spacing, 16 mm² per cell, max 1296 mm²).
Ranking: area, pass count, mean in-jaw tilt, mean in-jaw slip.

## Fixed settings

| setting | value |
|---|---|
| piece | `rook_kiri`, `banded_hulls` collision |
| baseline grasp | (−20, −1, −3) mm, 16 Oct e4→e5 grasp sweep (scored on accuracy, blind; −20 was the edge of that grid) |
| place (pickup phase) | (−8.75, 2.23, −2.67) mm, 16 Oct; irrelevant, the pickup phase stops before lowering |
| place (set-down phase) | e4→e5 (−9.08, 1.56, −2.81) mm, e4→f4 (−4.24, −0.29, −2.01) mm (6 Nov median calibrations); x/y fixed, not swept |
| grip forces, frictions | sim defaults |

## Code

| file | role |
|---|---|
| `lowlevel/multisim_chess_fast.py` | new opt-in `run_sim_move(step_callback=...)`: called after every physics step, reads state only; returning True ends the move early. Default `None` = unchanged |
| `lowlevel/chess_traj.py` | new opt-in env `CHESS_PLACE_RELEASE_CLEARANCE` (m): height above the pickup height where lowering ends and the gripper opens. Unset = the original hard-coded 0.02 (`#wiggle room for drop`) |
| `lowlevel/drafts/stability_probe.py` | the per-step probe and the pass/fail classifier |
| `lowlevel/drafts/run_stability_sweep.py` | sweep driver: settings × start grid, `results.jsonl` per sample, resumable, holds back new samples while free RAM < 1 GB |
| `lowlevel/drafts/stability_report.py` | `summary.{csv,md}` and figures |

## Checks

- **The probe does not change the physics**: e4→e5 known r0_c0 / r4_c4 / r8_c8 (grasp (−20, −1, −3),
  16 Oct place) re-run with the probe end at exactly the stored final positions of
  `e4_to_e5_bullseye_known_corr0_grid9_e4ringknown_20261106_142020` (e.g. r8_c8 x = 0.26946062220916317).
- Smoke test, pickup phase, grasp corners ±4 mm × starts r0_c0/r4_c4/r8_c8 at ±16 mm: 27/27 pass, in-jaw
  slip 0.5–1.2 mm, in-jaw tilt 8–13°. A ±4 mm grasp grid would be saturated, so the pilot uses ±8 mm in 2 mm steps.

## Findings so far

- **The rook is dropped on purpose.** `chess_traj.pickupmove_traj` ends the place lowering at
  `to_xyz + [0, 0, 0.02]` ("wiggle room for drop"), i.e. 2 cm above the pickup height. On e4→e5 the rook
  centre is at z ≈ 53 mm when the jaws open (rest height 26 mm), tilted ~16°, and it falls ~27 mm.
  This is the universal `premature_drop` seen in every earlier run. It lands upright on e4→e5, but it is
  the obvious first set-down setting to sweep (release clearance 0–20 mm).

## Log / incidents

- **Pickup pilot start**: the first launch hit `NameError` in 247 samples because workers re-imported
  `chess_traj.py` while I was adding the release-clearance option (`os` not imported at module level).
  Fixed, sweep stopped and resumed; errored samples are re-run on resume, no valid sample affected
  (pickup does not use the release clearance).

## Runs

| run | what | folder |
|---|---|---|
| pickup pilot | e4 (→ e5 carry), grasp x −28…−12, y −9…+7 mm (2 mm steps), z −3; starts 9×9 ±16 mm; 6561 samples | `pickup/e4_to_e5_pickup_pilot/` |

## Results: e4 pickup pilot (`pickup/e4_to_e5_pickup_pilot/`)

Figures `grasp_area_heatmap.png`, `start_maps_small_multiples.png`, `best_vs_baseline.png`; table `summary.md`.

**54 of 81 grasp offsets pick up from every start (81/81, 1296 mm²)**: e4 pickup in known mode has a wide plateau
(x −28…−14, y −5…+5 mm almost entirely). Stable cells per grasp (of 81; columns x −28…−12, rows y):

| y \ x | −28 | −26 | −24 | −22 | −20 | −18 | −16 | −14 | −12 |
|---|---|---|---|---|---|---|---|---|---|
| +7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| +5 | 81 | 81 | 81 | 81 | 81 | 81 | 81 | 81 | 0 |
| +3 | 81 | 81 | 81 | 81 | 81 | 81 | 81 | 81 | 0 |
| +1 | 81 | 81 | 81 | 81 | 81 | 81 | 81 | 81 | 0 |
| −1 | 81 | 81 | 81 | 81 | **81 (baseline)** | 81 | 81 | 81 | 0 |
| −3 | 79 | 77 | 81 | 81 | 81 | 81 | 81 | 81 | 0 |
| −5 | 79 | 71 | 81 | 81 | 81 | 81 | 81 | 81 | 0 |
| −7 | 81 | 69 | 77 | 81 | 81 | 81 | 81 | 81 | 51 |
| −9 | 75 | 73 | 81 | 81 | 81 | 80 | 81 | 79 | 0 |

Edges of the plateau, by failure type:
- **y = +7 mm: missed grasp** from every start (the jaws close beside the rook).
- **x = −12 mm: knocked before grasp** from every start: the closing jaws tip the rook ~4° (limit 3°) and push it
  ~1.5 mm before lift-off. It is still lifted, so this edge is soft (a 5° limit would pass it).
- **x = −26/−28, y ≤ −3 mm: slipped in the jaws** (> 2 mm) from scattered starts.
- In-jaw slip on the plateau 0.5–1.5 mm; in-jaw tilt 8.5–15° (lowest at y = +3, highest at x = −28, y ≤ −7).

**Chosen grasp: the baseline (−20, −1, −3) mm.** Ranked by area, then margin (distance to the nearest grasp with a
smaller area): the baseline is the centre of the plateau, 6.3 mm from the nearest failing grasp, the most of any
setting (next: (−20, +1), (−18, ±1), (−18, −3) at 6.0 mm). So for e4 the accuracy-tuned grasp is also the most
robust one. (Ranking by lowest in-jaw tilt instead picked (−16, −9), a grid corner; rejected, no margin.)

## Next

1. Pickup pilot report: best grasp vs baseline (`grasp_area_heatmap.png`, `start_maps_small_multiples.png`,
   `best_vs_baseline.png`); if the area is saturated over a broad plateau, pick its centre (most margin).
2. Set-down on e4→e5 and e4→f4 (f4 with the pivot place) with the chosen grasp: release clearance
   0 / 2 / 5 / 10 / 15 / 20 mm × lowering steps 10 / 20 / 30, starts 9×9 ±16 mm (2 × 18 × 81 = 2916 full moves).
