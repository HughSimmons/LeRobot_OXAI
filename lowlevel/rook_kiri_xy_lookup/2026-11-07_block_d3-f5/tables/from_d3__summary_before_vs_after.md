# d3 → 3×3 neighbourhood, blind mode: before vs after

Each cell: **≤5 mm · ≤2 mm · missed (M) / fell (F) · median error mm · median landing point (x, y) mm**

| target | before | after (median cal) |
|---|---|---|
| d5 | 120/150 · 95/150 · M2 F13 · 1.46 · (-1.0, +0.4) | 123/150 · 99/150 · M2 F10 · 1.26 · (-0.0, +0.1) |
| e5 | 123/150 · 98/150 · M2 F9 · 1.41 · (-0.8, +0.3) | 125/150 · 99/150 · M2 F8 · 1.12 · (-0.1, +0.1) |
| f5 | 31/150 · 0/150 · M2 F47 · 5.60 · (+4.4, +2.9) | 105/150 · 48/150 · M2 F28 · 2.28 · (-0.5, +0.3) |
| d4 | 114/150 · 70/150 · M2 F17 · 1.97 · (-0.9, -0.9) | 113/150 · 81/150 · M2 F16 · 1.34 · (+0.0, +0.0) |
| e4 | 114/150 · 79/150 · M2 F16 · 1.67 · (-0.7, -0.8) | 114/150 · 83/150 · M2 F16 · 1.19 · (+0.1, -0.0) |
| f4 | 14/150 · 0/150 · M2 F43 · 6.06 · (+5.0, +2.2) | 114/150 · 60/150 · M2 F21 · 2.08 · (-0.6, +0.4) |
| d3 | 126/150 · 18/150 · M2 F2 · 2.75 · (-1.3, -1.6) | 126/150 · 86/150 · M2 F2 · 1.51 · (+0.0, +0.0) |
| e3 | 124/150 · 31/150 · M2 F3 · 2.37 · (-1.0, -1.5) | 124/150 · 96/150 · M2 F3 · 1.30 · (+0.0, +0.1) |
| f3 | 23/150 · 0/150 · M2 F10 · 6.11 · (+5.1, +2.1) | 109/150 · 56/150 · M2 F18 · 2.31 · (+0.1, -0.1) |
| **total (d5, e5, f5, d4, e4, f4, d3, e3, f3)** | **789/1350** · 391/1350 · M18 F160 | **1053/1350** · 708/1350 · M18 F122 |

Place offset used (mm, gripper frame):

| target | before | after (median cal) |
|---|---|---|
| d5 | [-8.75, 2.23, -2.67] | [-8.99, 1.13, -2.85] |
| e5 | [-8.75, 2.23, -2.67] | [-8.95, 1.38, -2.82] |
| f5 | [-8.75, 2.23, -2.67] | [-3.59, 1.64, -1.75] |
| d4 | [-8.75, 2.23, -2.67] | [-9.99, 2.12, -2.81] |
| e4 | [-8.75, 2.23, -2.67] | [-9.84, 2.19, -2.79] |
| f4 | [-8.75, 2.23, -2.67] | [-3.41, 1.69, -1.69] |
| d3 | [-8.75, 2.23, -2.67] | [-10.75, 2.2, -2.81] |
| e3 | [-8.75, 2.23, -2.67] | [-10.56, 2.36, -2.79] |
| f3 | [-8.75, 2.23, -2.67] | [-3.49, 1.12, -1.74] |

Run folders:

- d5 / before: `d3_to_d5_bullseye_blind_corr0_random150_blk_20261107_001406`
- d5 / after (median cal): `d3_to_d5_bullseye_blind_corr0_random150_blk_cal_median_20261107_002138`
- e5 / before: `d3_to_e5_bullseye_blind_corr0_random150_blk_20261107_002308`
- e5 / after (median cal): `d3_to_e5_bullseye_blind_corr0_random150_blk_cal_median_20261107_003051`
- f5 / before: `d3_to_f5_bullseye_blind_corr0_random150_blk_20261107_002907`
- f5 / after (median cal): `d3_to_f5_bullseye_blind_corr0_random150_blk_cal_median_20261107_003925`
- d4 / before: `d3_to_d4_bullseye_blind_corr0_random150_blk_20261107_003830`
- d4 / after (median cal): `d3_to_d4_bullseye_blind_corr0_random150_blk_cal_median_20261107_004625`
- e4 / before: `d3_to_e4_bullseye_blind_corr0_random150_blk_20261107_004927`
- e4 / after (median cal): `d3_to_e4_bullseye_blind_corr0_random150_blk_cal_median_20261107_005703`
- f4 / before: `d3_to_f4_bullseye_blind_corr0_random150_blk_20261107_005358`
- f4 / after (median cal): `d3_to_f4_bullseye_blind_corr0_random150_blk_cal_median_20261107_010411`
- d3 / before: `d3_to_d3_bullseye_blind_corr0_random150_blk_20261107_010458`
- d3 / after (median cal): `d3_to_d3_bullseye_blind_corr0_random150_blk_cal_median_20261107_011255`
- e3 / before: `d3_to_e3_bullseye_blind_corr0_random150_blk_20261107_011440`
- e3 / after (median cal): `d3_to_e3_bullseye_blind_corr0_random150_blk_cal_median_20261107_012228`
- f3 / before: `d3_to_f3_bullseye_blind_corr0_random150_blk_20261107_012053`
- f3 / after (median cal): `d3_to_f3_bullseye_blind_corr0_random150_blk_cal_median_20261107_013117`
