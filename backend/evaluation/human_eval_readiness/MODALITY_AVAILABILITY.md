# ClipSense AI — Stored Evidence Modality Availability Report

## 1. Executive Summary

This report documents the actual evidence availability across all 5 sub-3600s test episodes evaluated in the Rhapsody pilot under `runs/rhapsody_real_pilot/`.

All stored evidence files (`transcript_evidence.json`, `conversation_evidence.json`, `visual_evidence.json`, and `prosody_evidence.json`) were directly inspected.

---

## 2. Actual Proposal Counts per Modality

| Episode Directory | Source Video ID | Video Title | Transcript Proposals | Conversation Proposals | Visual Proposals | Prosody Proposals | Effective Modalities |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `rhapsody_real_brYEgU782YU` | `brYEgU782YU` | NJ Mystery Drones Situation is Crazy, Congressman Claims Secret Iran Mothership Off US Coast, & More | **59** | **1** | **0 (Empty)** | **0 (Empty)** | Spoken Language Only |
| `rhapsody_real_gEcW5rta3wU` | `gEcW5rta3wU` | Two Wigs, a Gun and a Murder Part 2 \| Full Episode | **109** | **42** | **0 (Empty)** | **0 (Empty)** | Spoken Language Only |
| `rhapsody_real_h91NpMIgbdI` | `h91NpMIgbdI` | 'Gutless'! Walmart Tosses DEI After Right-Wing Conservatives TARGET Public Companies \| Roland Martin | **46** | **4** | **0 (Empty)** | **0 (Empty)** | Spoken Language Only |
| `rhapsody_real_hSuSutvGdXQ` | `hSuSutvGdXQ` | Fox News MELTS DOWN Live on AIR as Trump LOSES IT | **50** | **18** | **0 (Empty)** | **0 (Empty)** | Spoken Language Only |
| `rhapsody_real_r0VyLTLGcWk` | `r0VyLTLGcWk` | Getting In Bed w/ Mia Malkova & Gabby Epstein \| First Date with Lauren Compton | **187** | **44** | **0 (Empty)** | **0 (Empty)** | Spoken Language Only |
| **Total Proposals** | — | — | **451** | **109** | **0 (Empty)** | **0 (Empty)** | **Spoken Language Only** |

---

## 3. Modality Analysis & Root Cause

1. **Transcript & Conversation**:
   - `transcript_evidence.json` and `conversation_evidence.json` are fully populated from the Rhapsody WhisperX transcripts.
   - Together they provide 560 candidate segments across lexical importance, key statements, and dialogue question-response turn structures.

2. **Visual & Prosody**:
   - `visual_evidence.json` and `prosody_evidence.json` are present on disk for all 5 episodes, but contain empty proposal arrays (`"proposals": []`).
   - **Reason**: The Rhapsody benchmark distribution provides WhisperX transcripts and raw temporal replay curves, but does not provide pre-extracted visual frame embeddings or continuous pitch/energy audio tracks.

3. **Evaluation Implication**:
   - Systems B (Simple Aggregation), C (MTER), and D (MTER Refined) in this pilot study operated as **spoken-language multimodal fusion** (integrating transcript semantic density and conversation turn dynamics).
   - This evaluation **must not** be described as a full 4-modality audiovisual evaluation. It rigorously benchmarks linguistic and dialogue structure fusion against transcript-only selection. Full 4-modality evaluation requires ingesting raw MP4 video streams.

---

## 4. Associated Evaluation Deliverables

- Detailed Evaluation Report: `evaluation/rhapsody_sub3600_eval_output_corrected/EVALUATION_REPORT.md`
- Predictions CSV: `evaluation/rhapsody_sub3600_eval_output_corrected/predictions_timestamp_only.csv`
- Final Human-Eval Package: `evaluation/human_eval_final_pilot/`
