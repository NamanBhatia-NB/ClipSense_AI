# ClipSense AI — Human Evaluation Pilot Package Validation Report

> [!NOTE]
> **Study Status**: This package represents a prepared pilot protocol and blinded evaluation material; data collection and ratings have not been conducted. No human-rated quality results are claimed.

## 1. Package Summary

- **Total Candidate Clips**: 20 clips across 5 sub-hour test episodes.
- **Primary Systems Evaluated**: System A (Transcript-Only), System B (Simple Aggregation), System C (MTER), System D (MTER Refined).
- **Rater Assignments**: 3 independent raters per clip (60 total assignments across 3 separate rater CSVs).
- **Blinding Status**: Strict double-blinding. All system names, model scores, and blinding codes are excluded from rater CSVs.
- **Score Form State**: All scoring fields are completely blank.
- **Researcher Unblinding Key**: Strictly isolated in `researcher_unblinding_key.json` outside the `rater_distribution/` folder.

## 2. Clip Verification & Decodability Table

| Clip ID | Episode ID | Intended Interval | Intended Dur (s) | Rendered File | Decodable | Decoded Dur (s) | File Size (MB) |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| `clip_01` | `rhapsody_real_brYEgU782YU` | [134.22s -> 150.67s] | 16.45s | `rendered_clips/clip_01.mp4` | YES [OK] | 16.48s | 1.31 MB |
| `clip_02` | `rhapsody_real_brYEgU782YU` | [134.22s -> 150.67s] | 16.45s | `rendered_clips/clip_02.mp4` | YES [OK] | 16.48s | 1.31 MB |
| `clip_03` | `rhapsody_real_brYEgU782YU` | [134.07s -> 150.82s] | 16.75s | `rendered_clips/clip_03.mp4` | YES [OK] | 16.78s | 1.33 MB |
| `clip_04` | `rhapsody_real_brYEgU782YU` | [1416.16s -> 1450.81s] | 34.65s | `rendered_clips/clip_04.mp4` | YES [OK] | 34.70s | 3.86 MB |
| `clip_05` | `rhapsody_real_gEcW5rta3wU` | [1230.43s -> 1280.24s] | 49.81s | `rendered_clips/clip_05.mp4` | YES [OK] | 49.82s | 5.34 MB |
| `clip_06` | `rhapsody_real_gEcW5rta3wU` | [1230.58s -> 1280.09s] | 49.51s | `rendered_clips/clip_06.mp4` | YES [OK] | 49.55s | 5.32 MB |
| `clip_07` | `rhapsody_real_gEcW5rta3wU` | [2338.58s -> 2355.57s] | 16.99s | `rendered_clips/clip_07.mp4` | YES [OK] | 17.05s | 1.73 MB |
| `clip_08` | `rhapsody_real_gEcW5rta3wU` | [1411.80s -> 1455.44s] | 43.64s | `rendered_clips/clip_08.mp4` | YES [OK] | 43.64s | 3.62 MB |
| `clip_09` | `rhapsody_real_h91NpMIgbdI` | [39.03s -> 53.80s] | 14.77s | `rendered_clips/clip_09.mp4` | YES [OK] | 14.78s | 0.78 MB |
| `clip_10` | `rhapsody_real_h91NpMIgbdI` | [102.59s -> 135.49s] | 32.90s | `rendered_clips/clip_10.mp4` | YES [OK] | 32.90s | 1.48 MB |
| `clip_11` | `rhapsody_real_h91NpMIgbdI` | [102.74s -> 135.34s] | 32.60s | `rendered_clips/clip_11.mp4` | YES [OK] | 32.63s | 1.47 MB |
| `clip_12` | `rhapsody_real_h91NpMIgbdI` | [577.47s -> 636.08s] | 58.62s | `rendered_clips/clip_12.mp4` | YES [OK] | 58.62s | 2.81 MB |
| `clip_13` | `rhapsody_real_hSuSutvGdXQ` | [1098.09s -> 1115.15s] | 17.06s | `rendered_clips/clip_13.mp4` | YES [OK] | 17.07s | 1.99 MB |
| `clip_14` | `rhapsody_real_hSuSutvGdXQ` | [1098.09s -> 1115.15s] | 17.06s | `rendered_clips/clip_14.mp4` | YES [OK] | 17.07s | 1.99 MB |
| `clip_15` | `rhapsody_real_hSuSutvGdXQ` | [1098.09s -> 1115.15s] | 17.06s | `rendered_clips/clip_15.mp4` | YES [OK] | 17.07s | 1.99 MB |
| `clip_16` | `rhapsody_real_hSuSutvGdXQ` | [1097.94s -> 1115.30s] | 17.36s | `rendered_clips/clip_16.mp4` | YES [OK] | 17.40s | 2.02 MB |
| `clip_17` | `rhapsody_real_r0VyLTLGcWk` | [2733.22s -> 2752.95s] | 19.73s | `rendered_clips/clip_17.mp4` | YES [OK] | 19.75s | 2.39 MB |
| `clip_18` | `rhapsody_real_r0VyLTLGcWk` | [1040.49s -> 1093.43s] | 52.94s | `rendered_clips/clip_18.mp4` | YES [OK] | 52.95s | 5.85 MB |
| `clip_19` | `rhapsody_real_r0VyLTLGcWk` | [1040.34s -> 1093.59s] | 53.24s | `rendered_clips/clip_19.mp4` | YES [OK] | 53.29s | 5.78 MB |
| `clip_20` | `rhapsody_real_r0VyLTLGcWk` | [2733.22s -> 2752.95s] | 19.73s | `rendered_clips/clip_20.mp4` | YES [OK] | 19.75s | 2.39 MB |

## 3. Invariant C/D Candidate Pairing Verification

System D applies sub-second boundary refinement (-0.15s start, +0.15s end) to the exact base candidate selected by System C:

| Episode ID | System C Clip ID | System C Interval | System D Clip ID | System D Interval | Start Shift | End Shift |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `rhapsody_real_brYEgU782YU` | `clip_01` | [134.22s -> 150.67s] | `clip_03` | [134.07s -> 150.82s] | -0.15s | +0.15s |
| `rhapsody_real_gEcW5rta3wU` | `clip_06` | [1230.58s -> 1280.09s] | `clip_05` | [1230.43s -> 1280.24s] | -0.15s | +0.15s |
| `rhapsody_real_h91NpMIgbdI` | `clip_11` | [102.74s -> 135.34s] | `clip_10` | [102.59s -> 135.49s] | -0.15s | +0.15s |
| `rhapsody_real_hSuSutvGdXQ` | `clip_14` | [1098.09s -> 1115.15s] | `clip_16` | [1097.94s -> 1115.30s] | -0.15s | +0.15s |
| `rhapsody_real_r0VyLTLGcWk` | `clip_18` | [1040.49s -> 1093.43s] | `clip_19` | [1040.34s -> 1093.59s] | -0.15s | +0.15s |

## 4. Distribution Instructions for Raters

To distribute the evaluation package to 3 independent raters:
1. Provide each rater with the `rater_distribution/` directory containing:
   - `HUMAN_EVAL_PROTOCOL.md` (Scoring guide and 1–5 rubric definition)
   - Their respective sheet: `rating_sheet_rater_01.csv`, `rating_sheet_rater_02.csv`, or `rating_sheet_rater_03.csv`
   - The `rendered_clips/` folder containing `clip_01.mp4` through `clip_20.mp4`
2. Raters watch the 20 local MP4 files directly, ensuring exact playback of both start and end boundaries.
3. Keep `researcher_unblinding_key.json` strictly private until all ratings are returned.
