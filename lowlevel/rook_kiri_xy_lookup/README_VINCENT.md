# Vincent's rook experiments

All runs are in simulation, with the rook (`rook_kiri`). Each folder has its own notes, figures and tables.

## Grasp and place offsets for single moves (`2026-10_e-file_grasp_and_place/`)

- Swept the grasp offset and calibrated the place offset on e4→e5, then checked e3→e5 and e2→e5. Also tried gripper physics tweaks.
- e4→e5, robot blind to a 3.3 mm start error, 150 starts: within 5 mm went from 79 to **126/150**. That came from moving the grasp from (−14, +2, −3) to (−20, −1, −3) mm and the place offset to (−8.75, +2.23, −2.67) mm.
- e3→e5 reached 117/150 and e2→e5 98/150 within 5 mm.

## From e4 to the 3×3 around it (`2026-11-06_e4_neighbourhood/`)

- Moved from e4 to every square of d3–f5, then calibrated a place offset per target.
- Within 5 mm: 759 → **1125 of 1350 (83%)**. The f targets only worked after adding a pivot-first place.

## Every square of d3–f5 to every other (`2026-11-07_block_d3-f5/`)

- Ran all 81 source→target pairs, before and after per-pair place calibration.
- Within 5 mm: 45% → **69%** overall, and 62% → 80% for d/e sources.
- From f sources the *pickup* fails: 24–30 of 150 grasps missed.

## Pickup stability pilot (`stability/pickup/`)

- Swept the grasp offset on e4 with the robot told the true start position, scoring only whether the rook is picked up and held.
- 54 of 81 grasp offsets picked up from every start, so this setup could not rank grasps.

## Pickup success under the real arm's joint error (`stability/pickup_tolerance/`)

- Drove the simulated arm with joint errors measured on the real arm (Oliver). Searched the grasp offset at e4, then tested c/d/e ranks 2–7.
- At the measured error, the rook is picked up ~100% of the time on all 18 squares, with either grasp.
- At 2× the measured error, gripping 3 mm lower, (−20, −1, −6) mm, holds **98% vs 95%** for the current grasp (paired p ≈ 1e-6). It slips in the jaws more often, though. The same grasp also came out best on d4.
- The real arm fails far more often than this. Most real failures likely come from calibration and timing, not from joint error at rest.
- Slide-ready report: `stability/pickup_tolerance/report/PICKUP_TOLERANCE.md`.
