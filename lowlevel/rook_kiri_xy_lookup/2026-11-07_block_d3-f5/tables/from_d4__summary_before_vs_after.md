# d4 → 3×3 neighbourhood, blind mode: before vs after

Each cell: **≤5 mm · ≤2 mm · missed (M) / fell (F) · median error mm · median landing point (x, y) mm**

| target | before | after (median cal) |
|---|---|---|
| d5 | 124/150 · 100/150 · M1 F5 · 1.29 · (-0.7, +0.2) | 126/150 · 102/150 · M1 F6 · 1.23 · (+0.0, -0.0) |
| e5 | 123/150 · 101/150 · M1 F4 · 1.24 · (-0.4, +0.1) | 126/150 · 102/150 · M1 F5 · 1.17 · (+0.0, +0.0) |
| f5 | 40/150 · 0/150 · M1 F59 · 5.35 · (+3.7, +2.7) | 82/150 · 47/150 · M1 F55 · 2.00 · (+0.1, +0.1) |
| d4 | 123/150 · 77/150 · M1 F7 · 1.80 · (-0.7, -0.8) | 121/150 · 92/150 · M1 F8 · 1.29 · (+0.0, +0.0) |
| e4 | 123/150 · 87/150 · M1 F8 · 1.59 · (-0.6, -0.7) | 124/150 · 93/150 · M1 F7 · 1.32 · (-0.0, +0.0) |
| f4 | 9/150 · 0/150 · M1 F47 · 6.04 · (+5.1, +2.4) | 111/150 · 52/150 · M1 F24 · 2.28 · (-0.7, +0.4) |
| d3 | 126/150 · 36/150 · M1 F0 · 2.65 · (-0.9, -1.7) | 127/150 · 89/150 · M1 F1 · 1.43 · (+0.1, +0.0) |
| e3 | 125/150 · 48/150 · M1 F2 · 2.38 · (-0.6, -1.5) | 127/150 · 91/150 · M1 F3 · 1.34 · (-0.1, -0.0) |
| f3 | 27/150 · 0/150 · M1 F22 · 6.01 · (+5.1, +2.3) | 118/150 · 38/150 · M1 F18 · 2.37 · (+0.2, +0.2) |
| **total (d5, e5, f5, d4, e4, f4, d3, e3, f3)** | **820/1350** · 449/1350 · M9 F154 | **1062/1350** · 706/1350 · M9 F127 |

Place offset used (mm, gripper frame):

| target | before | after (median cal) |
|---|---|---|
| d5 | [-8.75, 2.23, -2.67] | [-8.96, 1.53, -2.79] |
| e5 | [-8.75, 2.23, -2.67] | [-8.9, 1.88, -2.74] |
| f5 | [-8.75, 2.23, -2.67] | [-4.24, 1.49, -1.88] |
| d4 | [-8.75, 2.23, -2.67] | [-9.86, 2.17, -2.79] |
| e4 | [-8.75, 2.23, -2.67] | [-9.68, 2.24, -2.77] |
| f4 | [-8.75, 2.23, -2.67] | [-3.25, 1.62, -1.67] |
| d3 | [-8.75, 2.23, -2.67] | [-10.65, 2.58, -2.74] |
| e3 | [-8.75, 2.23, -2.67] | [-10.34, 2.72, -2.71] |
| f3 | [-8.75, 2.23, -2.67] | [-3.38, 0.94, -1.73] |

Run folders:

- d5 / before: `d4_to_d5_bullseye_blind_corr0_random150_blk_20261107_013036`
- d5 / after (median cal): `d4_to_d5_bullseye_blind_corr0_random150_blk_cal_median_20261107_013845`
- e5 / before: `d4_to_e5_bullseye_blind_corr0_random150_blk_20261107_014134`
- e5 / after (median cal): `d4_to_e5_bullseye_blind_corr0_random150_blk_cal_median_20261107_014925`
- f5 / before: `d4_to_f5_bullseye_blind_corr0_random150_blk_20261107_014618`
- f5 / after (median cal): `d4_to_f5_bullseye_blind_corr0_random150_blk_cal_median_20261107_015639`
- d4 / before: `d4_to_d4_bullseye_blind_corr0_random150_blk_20261107_015707`
- d4 / after (median cal): `d4_to_d4_bullseye_blind_corr0_random150_blk_cal_median_20261107_020457`
- e4 / before: `d4_to_e4_bullseye_blind_corr0_random150_blk_20261107_020636`
- e4 / after (median cal): `d4_to_e4_bullseye_blind_corr0_random150_blk_cal_median_20261107_021413`
- f4 / before: `d4_to_f4_bullseye_blind_corr0_random150_blk_20261107_021232`
- f4 / after (median cal): `d4_to_f4_bullseye_blind_corr0_random150_blk_cal_median_20261107_022234`
- d3 / before: `d4_to_d3_bullseye_blind_corr0_random150_blk_20261107_022150`
- d3 / after (median cal): `d4_to_d3_bullseye_blind_corr0_random150_blk_cal_median_20261107_022921`
- e3 / before: `d4_to_e3_bullseye_blind_corr0_random150_blk_20261107_023240`
- e3 / after (median cal): `d4_to_e3_bullseye_blind_corr0_random150_blk_cal_median_20261107_024022`
- f3 / before: `d4_to_f3_bullseye_blind_corr0_random150_blk_20261107_023655`
- f3 / after (median cal): `d4_to_f3_bullseye_blind_corr0_random150_blk_cal_median_20261107_024710`
