# e4 → 3×3 neighbourhood, blind mode: before vs after

Each cell: **≤5 mm · ≤2 mm · missed (M) / fell (F) · median error mm · median landing point (x, y) mm**

| target | before | clean | blind mean | blind median |
|---|---|---|---|---|
| d5 | 125/150 · 88/150 · M1 F6 · 1.49 · (-1.0, +0.2) | 124/150 · 91/150 · M1 F10 · 1.36 · (-0.7, +0.0) | 128/150 · 90/150 · M1 F5 · 1.52 · (+0.6, -0.3) | 128/150 · 93/150 · M1 F9 · 1.24 · (+0.0, +0.0) |
| e5 | 126/150 · 91/150 · M1 F5 · 1.28 · (-0.7, +0.2) | 126/150 · 88/150 · M1 F5 · 1.34 · (-0.9, +0.2) | 129/150 · 93/150 · M1 F5 · 1.61 · (+0.6, -0.4) | 131/150 · 96/150 · M1 F3 · 1.32 · (-0.1, +0.0) |
| f5 | 0/150 · 0/150 · M1 F148 · 14.89 · (+3.3, -14.5) | not run | not run | not run |
| d4 | 128/150 · 80/150 · M1 F7 · 1.78 · (-0.9, -0.6) | 128/150 · 92/150 · M1 F8 · 1.28 · (-0.7, +0.2) | 125/150 · 93/150 · M1 F7 · 1.66 · (+0.7, -0.3) | 128/150 · 92/150 · M1 F7 · 1.24 · (+0.0, +0.0) |
| e4 | 129/150 · 85/150 · M1 F6 · 1.59 · (-0.7, -0.5) | 127/150 · 91/150 · M1 F6 · 1.34 · (-0.7, +0.4) | 126/150 · 90/150 · M1 F5 · 1.76 · (+0.7, -0.3) | 126/150 · 89/150 · M1 F9 · 1.31 · (-0.1, +0.0) |
| f4 | 0/150 · 0/150 · M1 F149 · – · – | not run | not run | not run |
| d3 | 125/150 · 30/150 · M1 F3 · 2.65 · (-0.9, -1.7) | 128/150 · 88/150 · M1 F3 · 1.43 · (-0.2, -0.1) | 126/150 · 88/150 · M1 F3 · 1.79 · (+0.6, -0.2) | 128/150 · 88/150 · M1 F3 · 1.35 · (+0.0, +0.0) |
| e3 | 126/150 · 54/150 · M1 F3 · 2.25 · (-0.6, -1.4) | 130/150 · 89/150 · M1 F3 · 1.46 · (-0.2, -0.0) | 127/150 · 81/150 · M1 F3 · 1.86 · (+0.6, -0.2) | 129/150 · 87/150 · M1 F4 · 1.42 · (-0.1, +0.1) |
| f3 | 0/150 · 0/150 · M1 F148 · 44.49 · (-8.3, -43.7) | not run | not run | not run |
| **total (d5, e5, d4, e4, d3, e3)** | **759/900** · 428/900 · M6 F30 | **763/900** · 539/900 · M6 F35 | **761/900** · 535/900 · M6 F28 | **770/900** · 545/900 · M6 F35 |

Place offset used (mm, gripper frame):

| target | before | clean | blind mean | blind median |
|---|---|---|---|---|
| d5 | [-8.75, 2.23, -2.67] | [-8.78, 1.97, -2.71] | [-9.32, 0.7, -2.95] | [-9.19, 1.36, -2.85] |
| e5 | [-8.75, 2.23, -2.67] | [-8.74, 2.35, -2.65] | [-9.18, 0.76, -2.93] | [-9.08, 1.56, -2.81] |
| f5 | [-8.75, 2.23, -2.67] | – | – | – |
| d4 | [-8.75, 2.23, -2.67] | [-9.52, 2.54, -2.7] | [-9.96, 1.21, -2.96] | [-9.82, 1.9, -2.83] |
| e4 | [-8.75, 2.23, -2.67] | [-9.41, 2.69, -2.67] | [-9.74, 1.16, -2.95] | [-9.57, 1.98, -2.8] |
| f4 | [-8.75, 2.23, -2.67] | – | – | – |
| d3 | [-8.75, 2.23, -2.67] | [-10.38, 2.7, -2.71] | [-10.87, 2.05, -2.86] | [-10.62, 2.68, -2.73] |
| e3 | [-8.75, 2.23, -2.67] | [-10.06, 2.75, -2.69] | [-10.5, 1.86, -2.89] | [-10.22, 2.68, -2.72] |
| f3 | [-8.75, 2.23, -2.67] | – | – | – |

Run folders:

- d5 / before: `e4_to_d5_bullseye_blind_corr0_random150_e4ring_20261106_110758`
- d5 / clean: `e4_to_d5_bullseye_blind_corr0_random150_e4cal_clean_20261106_144323`
- d5 / blind mean: `e4_to_d5_bullseye_blind_corr0_random150_e4cal_mean_20261106_151711`
- d5 / blind median: `e4_to_d5_bullseye_blind_corr0_random150_e4cal_median_20261106_155336`
- e5 / before: `e4_to_e5_bullseye_blind_corr0_random150_e4ring_20261106_110225`
- e5 / clean: `e4_to_e5_bullseye_blind_corr0_random150_e4cal_clean_20261106_143758`
- e5 / blind mean: `e4_to_e5_bullseye_blind_corr0_random150_e4cal_mean_20261106_151124`
- e5 / blind median: `e4_to_e5_bullseye_blind_corr0_random150_e4cal_median_20261106_154706`
- f5 / before: `e4_to_f5_bullseye_blind_corr0_random150_e4ring_20261106_111313`
- d4 / before: `e4_to_d4_bullseye_blind_corr0_random150_e4ring_20261106_111918`
- d4 / clean: `e4_to_d4_bullseye_blind_corr0_random150_e4cal_clean_20261106_144846`
- d4 / blind mean: `e4_to_d4_bullseye_blind_corr0_random150_e4cal_mean_20261106_152254`
- d4 / blind median: `e4_to_d4_bullseye_blind_corr0_random150_e4cal_median_20261106_160005`
- e4 / before: `e4_to_e4_bullseye_blind_corr0_random150_e4ring_20261106_130544`
- e4 / clean: `e4_to_e4_bullseye_blind_corr0_random150_e4cal_clean_20261106_150534`
- e4 / blind mean: `e4_to_e4_bullseye_blind_corr0_random150_e4cal_mean_20261106_154041`
- e4 / blind median: `e4_to_e4_bullseye_blind_corr0_random150_e4cal_median_20261106_161853`
- f4 / before: `e4_to_f4_bullseye_blind_corr0_random150_e4ring_20261106_112439`
- d3 / before: `e4_to_d3_bullseye_blind_corr0_random150_e4ring_20261106_113108`
- d3 / clean: `e4_to_d3_bullseye_blind_corr0_random150_e4cal_clean_20261106_145413`
- d3 / blind mean: `e4_to_d3_bullseye_blind_corr0_random150_e4cal_mean_20261106_152845`
- d3 / blind median: `e4_to_d3_bullseye_blind_corr0_random150_e4cal_median_20261106_160630`
- e3 / before: `e4_to_e3_bullseye_blind_corr0_random150_e4ring_20261106_125336`
- e3 / clean: `e4_to_e3_bullseye_blind_corr0_random150_e4cal_clean_20261106_150001`
- e3 / blind mean: `e4_to_e3_bullseye_blind_corr0_random150_e4cal_mean_20261106_153425`
- e3 / blind median: `e4_to_e3_bullseye_blind_corr0_random150_e4cal_median_20261106_161254`
- f3 / before: `e4_to_f3_bullseye_blind_corr0_random150_e4ring_20261106_125854`
