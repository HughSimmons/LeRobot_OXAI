# d5 → 3×3 neighbourhood, blind mode: before vs after

Each cell: **≤5 mm · ≤2 mm · missed (M) / fell (F) · median error mm · median landing point (x, y) mm**

| target | before | after (median cal) |
|---|---|---|
| d5 | 127/150 · 93/150 · M1 F2 · 1.41 · (-0.5, +0.4) | 127/150 · 90/150 · M1 F2 · 1.56 · (+0.1, -0.1) |
| e5 | 128/150 · 93/150 · M1 F4 · 1.29 · (-0.2, +0.3) | 126/150 · 95/150 · M1 F4 · 1.42 · (+0.0, -0.0) |
| f5 | 63/150 · 0/150 · M1 F32 · 4.83 · (+2.4, +3.6) | 106/150 · 63/150 · M1 F32 · 1.91 · (+0.2, -0.0) |
| d4 | 126/150 · 78/150 · M1 F2 · 1.86 · (-0.4, -0.6) | 127/150 · 88/150 · M1 F2 · 1.52 · (+0.1, -0.0) |
| e4 | 126/150 · 81/150 · M1 F2 · 1.81 · (-0.2, -0.5) | 127/150 · 89/150 · M1 F2 · 1.62 · (+0.0, +0.0) |
| f4 | 50/150 · 2/150 · M1 F25 · 5.31 · (+3.5, +3.4) | 123/150 · 52/150 · M1 F16 · 2.44 · (-0.5, +0.1) |
| d3 | 127/150 · 33/150 · M1 F1 · 2.68 · (-0.3, -1.7) | 127/150 · 75/150 · M1 F1 · 1.95 · (-0.0, +0.0) |
| e3 | 125/150 · 38/150 · M1 F1 · 2.45 · (-0.2, -1.5) | 126/150 · 77/150 · M1 F1 · 1.95 · (+0.0, +0.1) |
| f3 | 43/150 · 1/150 · M1 F13 · 5.60 · (+4.5, +2.9) | 119/150 · 58/150 · M1 F15 · 2.18 · (+0.0, -0.0) |
| **total (d5, e5, f5, d4, e4, f4, d3, e3, f3)** | **915/1350** · 419/1350 · M9 F82 | **1108/1350** · 687/1350 · M9 F75 |

Place offset used (mm, gripper frame):

| target | before | after (median cal) |
|---|---|---|
| d5 | [-8.75, 2.23, -2.67] | [-8.72, 1.6, -2.75] |
| e5 | [-8.75, 2.23, -2.67] | [-8.66, 1.93, -2.7] |
| f5 | [-8.75, 2.23, -2.67] | [-4.97, 0.09, -2.13] |
| d4 | [-8.75, 2.23, -2.67] | [-9.45, 2.28, -2.73] |
| e4 | [-8.75, 2.23, -2.67] | [-9.3, 2.37, -2.71] |
| f4 | [-8.75, 2.23, -2.67] | [-4.42, 0.11, -2.01] |
| d3 | [-8.75, 2.23, -2.67] | [-10.27, 3.1, -2.63] |
| e3 | [-8.75, 2.23, -2.67] | [-10.03, 3.08, -2.63] |
| f3 | [-8.75, 2.23, -2.67] | [-3.83, 0.23, -1.88] |

Run folders:

- d5 / before: `d5_to_d5_bullseye_blind_corr0_random150_blk_20261107_024741`
- d5 / after (median cal): `d5_to_d5_bullseye_blind_corr0_random150_blk_cal_median_20261107_025512`
- e5 / before: `d5_to_e5_bullseye_blind_corr0_random150_blk_20261107_025704`
- e5 / after (median cal): `d5_to_e5_bullseye_blind_corr0_random150_blk_cal_median_20261107_030441`
- f5 / before: `d5_to_f5_bullseye_blind_corr0_random150_blk_20261107_030233`
- f5 / after (median cal): `d5_to_f5_bullseye_blind_corr0_random150_blk_cal_median_20261107_031229`
- d4 / before: `d5_to_d4_bullseye_blind_corr0_random150_blk_20261107_031208`
- d4 / after (median cal): `d5_to_d4_bullseye_blind_corr0_random150_blk_cal_median_20261107_031940`
- e4 / before: `d5_to_e4_bullseye_blind_corr0_random150_blk_20261107_032214`
- e4 / after (median cal): `d5_to_e4_bullseye_blind_corr0_random150_blk_cal_median_20261107_032947`
- f4 / before: `d5_to_f4_bullseye_blind_corr0_random150_blk_20261107_032703`
- f4 / after (median cal): `d5_to_f4_bullseye_blind_corr0_random150_blk_cal_median_20261107_033642`
- d3 / before: `d5_to_d3_bullseye_blind_corr0_random150_blk_20261107_033714`
- d3 / after (median cal): `d5_to_d3_bullseye_blind_corr0_random150_blk_cal_median_20261107_034447`
- e3 / before: `d5_to_e3_bullseye_blind_corr0_random150_blk_20261107_034635`
- e3 / after (median cal): `d5_to_e3_bullseye_blind_corr0_random150_blk_cal_median_20261107_035417`
- f3 / before: `d5_to_f3_bullseye_blind_corr0_random150_blk_20261107_035157`
- f3 / after (median cal): `d5_to_f3_bullseye_blind_corr0_random150_blk_cal_median_20261107_040136`
