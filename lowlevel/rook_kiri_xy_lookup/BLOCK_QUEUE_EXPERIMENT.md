# Block queue: every source in the d3–f5 block → every square of the block

Started 2026-11-06 21:36 (Fri) on this machine (WSL, i7-1260P, 16 workers). Follows
`E4_NEIGHBOURHOOD_EXPERIMENT.md` (source e4, done) with the same protocol.
Status: first queue (e3 e5 d3 d4 d5) **done** 2026-11-07 04:08 (45/45 pairs, no failures);
second queue (f3 f4 f5) **done** 06:38 (27/27 pairs, no failures). All 81 source → target pairs done
(e4 from the earlier sweep). 9×9 summary: `block_results/block_9x9_within5mm.{png,md,csv}`
(`lowlevel/drafts/plot_block_matrix.py`).

## What runs

- Sources: e3, e5, d3, d4, d5 (in that order), then f3, f4, f5 (second queue, added 23:20, starts
  automatically when the first ends). With e4 (done earlier) this covers all 9 → 9.
- Targets: d5 e5 f5 / d4 e4 f4 / d3 e3 f3 (source → itself = pick and replace).
- 45 + 27 = 72 pairs, 2 runs each = 144 runs × 150 samples = 21,600 simulated moves.

Per pair (a "chain"):

| step | what | tag / file |
|---|---|---|
| 1 | blind 150, σ = 3.3 mm, seed 0, grasp (−20, −1, −3) mm, 16 Oct place (−8.75, 2.23, −2.67) mm | `blk` |
| 2 | median place calibration from 1 (skipped if < 30 upright) | `block_place_offsets/<src>_to_<dst>_median.json` |
| 3 | blind 150, same starts, calibrated place offset | `blk_cal_median` |

f targets use the pivot-first place (`--far-place-max-tilt 11 --far-place-style pivot
--far-place-pivot-height 0.02`); the flags do nothing for d/e targets.

## How it is parallelised

- Workers skip the real-robot driver/torch and the IK meshes (`lowlevel/drafts/sim_no_robot_driver.py`):
  ~336 MB instead of ~1 GB each, results bit-identical (checks in `E4_NEIGHBOURHOOD_EXPERIMENT.md`,
  "Speed-up for the block queue"). So 16 workers fit in RAM instead of 6.
- 2 chains at once × 8 workers each, so one run's tail overlaps the next run's start.
- Measured 1.85× the old throughput. Estimate: about 7 h for all 90 runs (about 12.5 h before),
  if the laptop does not sleep.
- Resumable: re-running `lowlevel/drafts/run_block_queue.sh` skips steps whose results exist.
  Change parallelism with `CHAINS=… WORKERS=…`.

## Files

- Driver `lowlevel/drafts/run_block_queue.sh`, log `block_queue_log.txt` (one line per step with ≤5 mm counts).
- Run folders `<src>_to_<dst>_bullseye_blind_corr0_random150_{blk,blk_cal_median}_<stamp>/`.
- Figures per source (to make when a source finishes):
  `plot_bullseye_neighbourhood.py --from-square e4 …` takes `--from-square <src>` but lays out the
  3×3 around the source; for the block layout use the targets above (see results section).

## f-file pickup smoke test (23:11–23:18)

Before queueing f sources: 12 blind starts (σ = 3.3 mm, seed 0), 16 Oct place offset, pivot place.
Folders `f4_to_{e4,f4}_bullseye_blind_corr0_random12_fpick_smoke_20261106_*`.

| move | picked up | upright | ≤5 mm | xy errors (mm) |
|---|---|---|---|---|
| f4 → e4 | 11/12 | 11 | 0 | 6.5–8.3 typical, some 11–12, one 45 |
| f4 → f4 | 11/12 | 11 | 3 | 2.4–10.7, one 14.4 |

Pickup from the far pose works; the rook sits differently in the jaws (tilted grasp), giving a ~7 mm
consistent landing bias. That is what the median place calibration corrects, so f sources run with the
same protocol (no pickup change).

## Incidents

_(none yet)_

## Results

Per-source figures/tables are made by `lowlevel/drafts/report_block_source.sh <src>` into
`block_results/` (`watch_block_reports.sh` runs it as each source's 9 chains finish):
`<src>_to_blocke4_bullseye_blind_{before,after}.png`, `<src>_to_blocke4_before_after.{md,csv,png}`.

Within 5 mm out of 150, before → after (median place calibration):

| source | d5 | e5 | f5 | d4 | e4 | f4 | d3 | e3 | f3 | total ≤5 mm | total ≤2 mm | fell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4 (earlier sweep, for reference) | 128 | 131 | 115 | 128 | 126 | 115 | 128 | 129 | 125 | 759 → 1125 (83%) | 428 → 735 | 475 → 96 |
| e3 | 118 → 118 | 112 → 120 | 33 → 95 | 113 → 115 | 114 → 119 | 26 → 113 | 122 → 129 | 127 → 128 | 30 → 116 | 795 → 1053 (78%) | 345 → 685 | 157 → 157 |
| e5 | 115 → 118 | 118 → 120 | 72 → 117 | 123 → 124 | 125 → 123 | 68 → 115 | 126 → 129 | 126 → 126 | 60 → 116 | 933 → 1088 (81%) | 330 → 686 | 90 → 101 |
| d3 | 120 → 123 | 123 → 125 | 31 → 105 | 114 → 113 | 114 → 114 | 14 → 114 | 126 → 126 | 124 → 124 | 23 → 109 | 789 → 1053 (78%) | 391 → 708 | 160 → 122 |
| d4 | 124 → 126 | 123 → 126 | 40 → 82 | 123 → 121 | 123 → 124 | 9 → 111 | 126 → 127 | 125 → 127 | 27 → 118 | 820 → 1062 (79%) | 449 → 706 | 154 → 127 |
| d5 | 127 → 127 | 128 → 126 | 63 → 106 | 126 → 127 | 126 → 127 | 50 → 123 | 127 → 127 | 125 → 126 | 43 → 119 | 915 → 1108 (82%) | 419 → 687 | 82 → 75 |
| f3 | 1 → 59 | 2 → 58 | 4 → 55 | 5 → 61 | 6 → 62 | 12 → 55 | 14 → 63 | 16 → 62 | 16 → 52 | 76 → 527 (39%) | 0 → 151 | 134 → 129 |
| f4 | 9 → 77 | 11 → 79 | 22 → 68 | 17 → 85 | 14 → 84 | 23 → 70 | 25 → 79 | 25 → 80 | 23 → 65 | 169 → 687 (51%) | 9 → 168 | 62 → 64 |
| f5 | 17 → 77 | 19 → 74 | 20 → 73 | 23 → 79 | 21 → 78 | 25 → 71 | 27 → 77 | 27 → 74 | 37 → 70 | 216 → 673 (50%) | 12 → 212 | 75 → 68 |

(e4 row shows after only per target; its "before" for f used the old far pose, so it is not like-for-like.)

**e3 notes.** Calibration fixes the f-file bias (median error 5.7–6.2 → 2.1–2.3 mm) but cannot fix
falls: e3 → f5 has 42 falls of 150 after calibration (36 before), the worst cell so far. d/e targets are
5–10 lower from e3 than from e4, mostly falls (14–20 per target on rank 4–5 vs 3–9 from e4); the
landing scatter runs along a diagonal streak, i.e. the error is tied to the grasp noise direction.

**First queue overall (e3 e5 d3 d4 d5, 45 pairs).** 4,252 → 5,364 of 6,750 within 5 mm (63% → 79%).
- d/e targets: calibration gains little on ≤5 mm (mostly ±3), but ≤2 mm rises on every source.
- f targets: 9–72 → 82–123 per cell; every source needs the calibration there.
- Falls dominate what is left, and they are worst on f5 (20–55 per 150; d4 → f5 55, the worst cell).
- Best source d5 (1108, 75 falls); worst e3/d3 (1053). Sources on rank 3 (e3, d3) fall more on rank 4–5
  targets than sources on rank 5.

**f sources (f3 f4 f5, 27 pairs).** 461 → 1,887 of 4,050 within 5 mm (11% → 47%); far below d/e sources.
The *pickup* is the problem, not the place:
- Missed grasps: 24–30 of 150 per pair (f3 30, f4 25, f5 24-25), against 1–2 for d/e sources. The
  12-sample smoke test (1/12 missed) under-estimated this. Missed grasps happen for the same noise draws
  for every target; their mean offset is (+3.6 … +4.1, −0.8) mm, i.e. rooks displaced *away* from the robot
  along x, which the tilted far-pose jaws do not reach.
- Upright landings are spread wide even after calibration (median 3.6–4.8 mm, IQR ~2–7 mm vs ~1.3 mm for
  d/e sources): with the tilted grasp, where the rook sits in the jaws depends on the start offset, so a
  single place offset cannot correct it. Calibrated place offsets are large, about (−12 … −16, +5 … +8) mm.
- Next step: a pivot-first *pickup* for f sources (descend in the far pose to ~2 cm, straighten to ~11°,
  descend locked and grasp), mirroring the place fix, then a grasp-offset check for f, then rerun f sources.

**All 81 pairs.** 5,472 → 8,376 of 12,150 within 5 mm (45% → 69%). Without the f sources (54 pairs):
5,011 → 6,489 of 8,100 (62% → 80%).

## f → f in known mode (2026-11-07, 12:14–12:56)

f3/f4/f5 → f3/f4/f5, robot told the true start offset, 9×9 grid ±10 mm (81 starts per pair), grasp
(−20, −1, −3) mm, pivot-first place. Script `lowlevel/drafts/run_f_known.sh`, log `f_known_log.txt`,
figures `plot_f_known.py` → `block_results/f_to_f_known{,_cal2}.{png,md}`. Reference: e4 → e5 known = 81/81.

Within 5 mm of 81:

| source \ target | f5 | f4 | f3 | pass 1 total | pass 2 total |
|---|---|---|---|---|---|
| f5 | 70 → 79 | 69 → 74 | 61 → 77 | 200 | 230/243 (95%) |
| f4 | 70 → 75 | 61 → 76 | 73 → 78 | 204 | 229/243 (94%) |
| f3 | 43 → 63 | 37 → 65 | 43 → 65 | 123 | 193/243 (79%) |
| all | | | | 527/729 (72%) | **652/729 (89%)** |

- Pass 1 (`fknown_cal`): place offset = blind median calibration. Lands +1.3 … +4.5 mm along x for every pair:
  the blind calibration absorbs the blind-mode grasp bias, which known mode does not have.
- Pass 2 (`fknown_cal2`): place offset recalibrated (median) from pass 1 → `block_place_offsets_known/`.
  **In-sample** (calibrated and scored on the same 81 starts).
- Not 100% for any pair. Remaining failures: f3 source 12–14 per pair (4–6 missed grasps + falls), scattered
  over the start grid rather than at its edge, so it is the tilted far-pose grasp, not reach; f4 source 1–5 falls
  per pair; f5 source 0–1. The 2–5 mm landings are also scattered (no offset-dependent pattern).

## f sources with pivot-first pickup, known-calibrated, then blind (started 2026-11-07 14:27)

Why: in the blind block queue the f-source place offsets came from blind median calibration only,
and in known mode they were poor (f→f known 72% pass 1). With the pivot-first pickup
(`--far-pickup-max-tilt 11 --far-pickup-lift direct`, script `run_f_known_pickpivot.sh`, log
`f_known_pickpivot_log.txt`) f3→f3/f4/f5 known is 80–81/81 after one median calibration. That run stopped at
14:17 after f3 sources and part of f4 (f4→f4/f5 calibrated runs incomplete, f5 not started).

Driver `lowlevel/drafts/run_f_pickpivot_block.sh f3 f4 f5` (log `f_pickpivot_block_log.txt`), per pair,
targets d5 e5 f5 d4 e4 f4 d3 e3 f3:
1. known 9×9 ±10 mm, start place offset P0 = f3→f3 calibrated (−21.51, −2.75, −1.73) mm, tag `fpp`
2. median calibration from 1 → `fpickpiv_offsets/<src>_to_<dst>_median.json`
3. known 9×9 with the calibrated offset (check), tag `fpp_cal`
4. blind 150, σ = 3.3 mm, seed 0, calibrated offset, tag `fpp_blind`

f→f steps 1–3 reuse the earlier `fpp`/`fpp_cal` runs. Start offset for d/e targets checked with a 3×3 known
smoke on f3→e4 (`…_fpp_smoke_{P0,oct16}_20261107_142519`): P0 lands 0.9–3.4 mm (6/9 upright, the three
falls at the (−10,−10), (0,−10), (−10,+10) mm corners); the 16 Oct offset lands 11–14 mm off. So P0 for all targets.
Note: `premature_drop` is True on every sample of every run (also the 16 Oct e4→e5 known run), so it is not a fault signal.

**Stopped 14:40: f3 → d/e targets fall in known mode.** Known 9×9 (81 starts):

| pair | start offset P0 | median-calibrated |
|---|---|---|
| f3 → e5 | 46/81 ≤5 mm, **fell 35** | 45/81, **fell 36** |
| f3 → d5 | 42/81 ≤5 mm, **fell 39** | 44/81, **fell 37** |

Calibration centres the upright landings (≤2 mm 30 → 41) but cannot fix the falls. Falls are scattered over
the start grid (no edge pattern). Old blind f3 runs (no pickup pivot) fell only 12–17/150 on d/e (but missed 30).

Diagnosis (per-step rook trace, scratchpad `trace_sample.py`, f3→e5 r0_c0 fell vs r0_c1 upright):
- The rook tilts 6° in the jaws at the grasp, then rotates in the soft hold during the `direct` lift to ~29°;
  it arrives over e5 still ~9° off vertical (the plan, with a rigid grasp, would give 2°).
- About 2 cm above the board (piece centre z ≈ 50 mm) it slips out of the jaws (this is the universal
  `premature_drop`, also on e4→e5), lands at 9–13° with ~8 mm sideways motion, and either rocks back upright or
  topples. r0_c0 and r0_c1 are identical up to that landing, so the outcome is chaotic.
- f3 → f3 does the same (rook ~30° in the jaws, straightened by the place pivot to ~12°, same 2 cm drop and
  wobble) but recovers every time.
- Quick fixes on f3 → e5 known 9×9, calibrated offset: `--far-pickup-lift retrace` **63/81 upright** (62 ≤5 mm,
  48 ≤2 mm); `direct` + `--soft-hold-force 50` 45/81 (no help). Folders `…_fpp_cal_test_{retrace,hold50}_*`.

Not run: any blind `fpp_blind` run (the two started for f3→d5/e5 were killed before finishing; their folders have
logs only). f3 → f targets are unaffected and could be run blind now with the existing known-calibrated offsets.
