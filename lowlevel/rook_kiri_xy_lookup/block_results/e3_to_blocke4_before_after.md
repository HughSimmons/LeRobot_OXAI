# e3 → 3×3 neighbourhood, blind mode: before vs after

Each cell: **≤5 mm · ≤2 mm · missed (M) / fell (F) · median error mm · median landing point (x, y) mm**

| target | before | after (median cal) |
|---|---|---|
| d5 | 118/150 · 81/150 · M2 F14 · 1.61 · (-1.2, +0.7) | 118/150 · 92/150 · M2 F17 · 1.30 · (-0.0, +0.0) |
| e5 | 112/150 · 84/150 · M2 F20 · 1.45 · (-1.0, +0.8) | 120/150 · 92/150 · M2 F17 · 1.24 · (-0.0, -0.0) |
| f5 | 33/150 · 1/150 · M2 F36 · 5.72 · (+3.7, +3.6) | 95/150 · 47/150 · M2 F42 · 2.12 · (-0.1, -0.1) |
| d4 | 113/150 · 67/150 · M2 F21 · 1.93 · (-1.3, -0.7) | 115/150 · 86/150 · M2 F18 · 1.36 · (+0.1, +0.0) |
| e4 | 114/150 · 73/150 · M2 F16 · 1.81 · (-1.1, -0.5) | 119/150 · 93/150 · M2 F14 · 1.20 · (-0.0, +0.0) |
| f4 | 26/150 · 1/150 · M2 F21 · 6.22 · (+4.4, +3.6) | 113/150 · 50/150 · M2 F22 · 2.19 · (-0.7, +0.0) |
| d3 | 122/150 · 9/150 · M2 F6 · 2.97 · (-1.7, -1.5) | 129/150 · 86/150 · M2 F3 · 1.70 · (+0.0, +0.0) |
| e3 | 127/150 · 29/150 · M2 F4 · 2.44 · (-1.3, -1.3) | 128/150 · 93/150 · M2 F5 · 1.31 · (-0.2, +0.1) |
| f3 | 30/150 · 0/150 · M2 F19 · 5.80 · (+4.5, +3.1) | 116/150 · 46/150 · M2 F19 · 2.33 · (+0.2, -0.1) |
| **total (d5, e5, f5, d4, e4, f4, d3, e3, f3)** | **795/1350** · 345/1350 · M18 F157 | **1053/1350** · 685/1350 · M18 F157 |

Place offset used (mm, gripper frame):

| target | before | after (median cal) |
|---|---|---|
| d5 | [-8.75, 2.23, -2.67] | [-8.89, 0.81, -2.88] |
| e5 | [-8.75, 2.23, -2.67] | [-8.75, 1.0, -2.84] |
| f5 | [-8.75, 2.23, -2.67] | [-3.86, 0.7, -1.87] |
| d4 | [-8.75, 2.23, -2.67] | [-10.14, 1.64, -2.91] |
| e4 | [-8.75, 2.23, -2.67] | [-9.81, 1.65, -2.88] |
| f4 | [-8.75, 2.23, -2.67] | [-3.5, 0.2, -1.83] |
| d3 | [-8.75, 2.23, -2.67] | [-10.97, 1.91, -2.88] |
| e3 | [-8.75, 2.23, -2.67] | [-10.6, 2.05, -2.86] |
| f3 | [-8.75, 2.23, -2.67] | [-3.85, -0.04, -1.9] |

Run folders:

- d5 / before: `e3_to_d5_bullseye_blind_corr0_random150_blk_20261106_213608`
- d5 / after (median cal): `e3_to_d5_bullseye_blind_corr0_random150_blk_cal_median_20261106_214622`
- e5 / before: `e3_to_e5_bullseye_blind_corr0_random150_blk_20261106_213608`
- e5 / after (median cal): `e3_to_e5_bullseye_blind_corr0_random150_blk_cal_median_20261106_214622`
- f5 / before: `e3_to_f5_bullseye_blind_corr0_random150_blk_20261106_215615`
- f5 / after (median cal): `e3_to_f5_bullseye_blind_corr0_random150_blk_cal_median_20261106_220607`
- d4 / before: `e3_to_d4_bullseye_blind_corr0_random150_blk_20261106_215622`
- d4 / after (median cal): `e3_to_d4_bullseye_blind_corr0_random150_blk_cal_median_20261106_220416`
- e4 / before: `e3_to_e4_bullseye_blind_corr0_random150_blk_20261106_221142`
- e4 / after (median cal): `e3_to_e4_bullseye_blind_corr0_random150_blk_cal_median_20261106_221919`
- f4 / before: `e3_to_f4_bullseye_blind_corr0_random150_blk_20261106_221606`
- f4 / after (median cal): `e3_to_f4_bullseye_blind_corr0_random150_blk_cal_median_20261106_222551`
- d3 / before: `e3_to_d3_bullseye_blind_corr0_random150_blk_20261106_222702`
- d3 / after (median cal): `e3_to_d3_bullseye_blind_corr0_random150_blk_cal_median_20261106_223454`
- e3 / before: `e3_to_e3_bullseye_blind_corr0_random150_blk_20261106_223543`
- e3 / after (median cal): `e3_to_e3_bullseye_blind_corr0_random150_blk_cal_median_20261106_224337`
- f3 / before: `e3_to_f3_bullseye_blind_corr0_random150_blk_20261106_224215`
- f3 / after (median cal): `e3_to_f3_bullseye_blind_corr0_random150_blk_cal_median_20261106_225216`
