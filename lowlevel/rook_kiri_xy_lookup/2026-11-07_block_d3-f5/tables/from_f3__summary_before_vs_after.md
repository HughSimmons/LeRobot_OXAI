# f3 → 3×3 neighbourhood, blind mode: before vs after

Each cell: **≤5 mm · ≤2 mm · missed (M) / fell (F) · median error mm · median landing point (x, y) mm**

| target | before | after (median cal) |
|---|---|---|
| d5 | 1/150 · 0/150 · M29 F23 · 9.19 · (-4.6, +7.2) | 59/150 · 18/150 · M30 F17 · 4.30 · (+0.4, +0.0) |
| e5 | 2/150 · 0/150 · M29 F14 · 9.08 · (-4.6, +7.2) | 58/150 · 20/150 · M31 F17 · 4.06 · (-0.4, -0.2) |
| f5 | 4/150 · 0/150 · M30 F14 · 8.55 · (-3.8, +6.1) | 55/150 · 14/150 · M30 F16 · 4.51 · (-0.2, -0.3) |
| d4 | 5/150 · 0/150 · M30 F17 · 8.71 · (-4.1, +6.6) | 61/150 · 17/150 · M30 F17 · 4.20 · (+0.0, -0.0) |
| e4 | 6/150 · 0/150 · M29 F14 · 8.55 · (-4.6, +6.7) | 62/150 · 21/150 · M30 F12 · 3.99 · (-0.2, -0.4) |
| f4 | 12/150 · 0/150 · M30 F15 · 7.88 · (-3.8, +5.6) | 55/150 · 10/150 · M30 F15 · 4.83 · (-0.1, -0.2) |
| d3 | 14/150 · 0/150 · M30 F11 · 8.12 · (-4.8, +5.8) | 63/150 · 20/150 · M30 F14 · 4.38 · (-0.1, -0.1) |
| e3 | 16/150 · 0/150 · M30 F10 · 7.97 · (-4.8, +5.6) | 62/150 · 20/150 · M30 F9 · 4.24 · (-0.4, -0.3) |
| f3 | 16/150 · 0/150 · M30 F16 · 7.65 · (-3.0, +4.9) | 52/150 · 11/150 · M30 F12 · 5.09 · (+0.1, -0.2) |
| **total (d5, e5, f5, d4, e4, f4, d3, e3, f3)** | **76/1350** · 0/1350 · M267 F134 | **527/1350** · 151/1350 · M271 F129 |

Place offset used (mm, gripper frame):

| target | before | after (median cal) |
|---|---|---|
| d5 | [-8.75, 2.23, -2.67] | [-15.59, 7.37, -3.4] |
| e5 | [-8.75, 2.23, -2.67] | [-15.68, 7.1, -3.42] |
| f5 | [-8.75, 2.23, -2.67] | [-14.7, 6.21, -3.41] |
| d4 | [-8.75, 2.23, -2.67] | [-14.9, 6.88, -3.57] |
| e4 | [-8.75, 2.23, -2.67] | [-15.11, 7.12, -3.65] |
| f4 | [-8.75, 2.23, -2.67] | [-13.93, 6.43, -3.59] |
| d3 | [-8.75, 2.23, -2.67] | [-14.0, 7.51, -3.86] |
| e3 | [-8.75, 2.23, -2.67] | [-13.97, 7.24, -3.82] |
| f3 | [-8.75, 2.23, -2.67] | [-13.07, 5.88, -3.55] |

Run folders:

- d5 / before: `f3_to_d5_bullseye_blind_corr0_random150_blk_20261107_040913`
- d5 / after (median cal): `f3_to_d5_bullseye_blind_corr0_random150_blk_cal_median_20261107_041605`
- e5 / before: `f3_to_e5_bullseye_blind_corr0_random150_blk_20261107_040913`
- e5 / after (median cal): `f3_to_e5_bullseye_blind_corr0_random150_blk_cal_median_20261107_041616`
- f5 / before: `f3_to_f5_bullseye_blind_corr0_random150_blk_20261107_042208`
- f5 / after (median cal): `f3_to_f5_bullseye_blind_corr0_random150_blk_cal_median_20261107_042852`
- d4 / before: `f3_to_d4_bullseye_blind_corr0_random150_blk_20261107_042217`
- d4 / after (median cal): `f3_to_d4_bullseye_blind_corr0_random150_blk_cal_median_20261107_042720`
- e4 / before: `f3_to_e4_bullseye_blind_corr0_random150_blk_20261107_043241`
- e4 / after (median cal): `f3_to_e4_bullseye_blind_corr0_random150_blk_cal_median_20261107_043751`
- f4 / before: `f3_to_f4_bullseye_blind_corr0_random150_blk_20261107_043545`
- f4 / after (median cal): `f3_to_f4_bullseye_blind_corr0_random150_blk_cal_median_20261107_044203`
- d3 / before: `f3_to_d3_bullseye_blind_corr0_random150_blk_20261107_044243`
- d3 / after (median cal): `f3_to_d3_bullseye_blind_corr0_random150_blk_cal_median_20261107_044739`
- e3 / before: `f3_to_e3_bullseye_blind_corr0_random150_blk_20261107_044824`
- e3 / after (median cal): `f3_to_e3_bullseye_blind_corr0_random150_blk_cal_median_20261107_045331`
- f3 / before: `f3_to_f3_bullseye_blind_corr0_random150_blk_20261107_045238`
- f3 / after (median cal): `f3_to_f3_bullseye_blind_corr0_random150_blk_cal_median_20261107_045829`
