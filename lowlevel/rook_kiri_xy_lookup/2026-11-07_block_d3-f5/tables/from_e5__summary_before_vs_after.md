# e5 → 3×3 neighbourhood, blind mode: before vs after

Each cell: **≤5 mm · ≤2 mm · missed (M) / fell (F) · median error mm · median landing point (x, y) mm**

| target | before | after (median cal) |
|---|---|---|
| d5 | 115/150 · 74/150 · M1 F14 · 1.73 · (-1.1, +0.3) | 118/150 · 90/150 · M1 F15 · 1.64 · (+0.1, -0.1) |
| e5 | 118/150 · 79/150 · M1 F13 · 1.57 · (-0.8, +0.3) | 120/150 · 85/150 · M1 F13 · 1.62 · (-0.1, -0.0) |
| f5 | 72/150 · 1/150 · M1 F17 · 4.89 · (+2.1, +3.9) | 117/150 · 67/150 · M1 F20 · 1.92 · (-0.1, +0.0) |
| d4 | 123/150 · 58/150 · M1 F5 · 2.32 · (-1.0, -0.9) | 124/150 · 83/150 · M1 F7 · 1.74 · (+0.1, -0.0) |
| e4 | 125/150 · 70/150 · M1 F6 · 2.16 · (-0.6, -0.7) | 123/150 · 85/150 · M1 F7 · 1.80 · (-0.0, +0.0) |
| f4 | 68/150 · 0/150 · M1 F16 · 4.94 · (+2.7, +3.5) | 115/150 · 61/150 · M1 F16 · 2.09 · (+0.1, +0.0) |
| d3 | 126/150 · 20/150 · M1 F3 · 3.07 · (-1.0, -1.9) | 129/150 · 80/150 · M1 F3 · 1.77 · (-0.0, +0.1) |
| e3 | 126/150 · 28/150 · M1 F2 · 2.83 · (-0.9, -1.7) | 126/150 · 81/150 · M1 F4 · 1.80 · (+0.3, +0.1) |
| f3 | 60/150 · 0/150 · M1 F14 · 5.29 · (+3.3, +3.3) | 116/150 · 54/150 · M1 F16 · 2.36 · (-0.1, +0.0) |
| **total (d5, e5, f5, d4, e4, f4, d3, e3, f3)** | **933/1350** · 330/1350 · M9 F90 | **1088/1350** · 686/1350 · M9 F101 |

Place offset used (mm, gripper frame):

| target | before | after (median cal) |
|---|---|---|
| d5 | [-8.75, 2.23, -2.67] | [-9.18, 1.25, -2.86] |
| e5 | [-8.75, 2.23, -2.67] | [-8.99, 1.43, -2.81] |
| f5 | [-8.75, 2.23, -2.67] | [-5.1, -0.28, -2.18] |
| d4 | [-8.75, 2.23, -2.67] | [-10.03, 2.07, -2.83] |
| e4 | [-8.75, 2.23, -2.67] | [-9.7, 2.21, -2.78] |
| f4 | [-8.75, 2.23, -2.67] | [-5.11, -0.27, -2.17] |
| d3 | [-8.75, 2.23, -2.67] | [-10.87, 2.74, -2.75] |
| e3 | [-8.75, 2.23, -2.67] | [-10.57, 2.71, -2.75] |
| f3 | [-8.75, 2.23, -2.67] | [-4.98, -0.4, -2.15] |

Run folders:

- d5 / before: `e5_to_d5_bullseye_blind_corr0_random150_blk_20261106_225121`
- d5 / after (median cal): `e5_to_d5_bullseye_blind_corr0_random150_blk_cal_median_20261106_225930`
- e5 / before: `e5_to_e5_bullseye_blind_corr0_random150_blk_20261106_230313`
- e5 / after (median cal): `e5_to_e5_bullseye_blind_corr0_random150_blk_cal_median_20261106_231233`
- f5 / before: `e5_to_f5_bullseye_blind_corr0_random150_blk_20261106_230758`
- f5 / after (median cal): `e5_to_f5_bullseye_blind_corr0_random150_blk_cal_median_20261106_232031`
- d4 / before: `e5_to_d4_bullseye_blind_corr0_random150_blk_20261106_232159`
- d4 / after (median cal): `e5_to_d4_bullseye_blind_corr0_random150_blk_cal_median_20261106_233011`
- e4 / before: `e5_to_e4_bullseye_blind_corr0_random150_blk_20261106_233109`
- e4 / after (median cal): `e5_to_e4_bullseye_blind_corr0_random150_blk_cal_median_20261106_233913`
- f4 / before: `e5_to_f4_bullseye_blind_corr0_random150_blk_20261106_233811`
- f4 / after (median cal): `e5_to_f4_bullseye_blind_corr0_random150_blk_cal_median_20261106_234858`
- d3 / before: `e5_to_d3_bullseye_blind_corr0_random150_blk_20261106_234716`
- d3 / after (median cal): `e5_to_d3_bullseye_blind_corr0_random150_blk_cal_median_20261106_235441`
- e3 / before: `e5_to_e3_bullseye_blind_corr0_random150_blk_20261106_235913`
- e3 / after (median cal): `e5_to_e3_bullseye_blind_corr0_random150_blk_cal_median_20261107_000630`
- f3 / before: `e5_to_f3_bullseye_blind_corr0_random150_blk_20261107_000224`
- f3 / after (median cal): `e5_to_f3_bullseye_blind_corr0_random150_blk_cal_median_20261107_001247`
