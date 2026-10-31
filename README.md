---
title: ClipSense AI
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
---

# ClipSense AI

> **Query-Free Long-Form Video Highlight Extraction Framework**  
> *Deterministic Multimodal Temporal Evidence Reasoner (MTER)*

---

## Overview

ClipSense AI is a research prototype that identifies, ranks, and physically renders self-contained highlight clips from long-form conversational videos (interviews, lectures, dialogues).

Instead of relying on prompt-biased text queries or uninspectable end-to-end black boxes, ClipSense AI combines four independent modality experts into a **Common Evidence Bundle** and arbitrates cross-modal agreement using a deterministic **Multimodal Temporal Evidence Reasoner (MTER)**.

---

## Architecture Pipeline

1. **Multimodal Extraction (W2)**: Continuous word timestamps (WhisperX), visual scene cuts and motion energy (PySceneDetect, OpenCV), acoustic prosody pitch ($F_0$) and RMS energy (Librosa), and conversational turn-taking dynamics.
2. **Modality-Specific Experts (W3 & W4)**:
   - **Transcript Expert**: Sliding-window semantic completeness and discourse payoff.
   - **Conversation Expert**: Rapid turn exchanges, question-answer clustering.
   - **Visual Expert**: Grayscale frame-difference motion energy and shot boundary transitions (PySceneDetect).
   - **Prosody Expert**: Standardized acoustic pitch and energy peak deviations.
3. **MTER Reasoner & Top-$K$ Selection (W5 & W8)**:
   - Temporal compatibility graph connecting overlapping proposals into event regions.
   - Intra-region boundary clustering and explicit conflict detection.
   - Deterministic composite scoring bounded to $[0.0, 1.0]$.
   - Configurable **Top-$K$ highlight selection** with **Non-Maximum Suppression (NMS)** to eliminate duplicate or heavily overlapping subclips.
4. **Evidence-Aware Temporal Boundary Refinement (W10)**:
   - Modular, deterministic boundary refiner aligning highlight candidates with speech activity.
   - Prevents mid-word and mid-utterance cuts by snapping to WhisperX word onsets/offsets and sentence boundaries.
   - Acoustic-pause confirmation (energy-based silence detection via Librosa) is implemented and unit-tested, though boundary adjustments in stored end-to-end runs were driven primarily by sentence boundary snapping and visual scene transitions.
   - Optional visual scene transition alignment (avoids abrupt mid-shot cuts without forcing unnatural speech clipping).
   - Strict enforcement of media bounds and configurable duration constraints ($[min\_clip\_duration, max\_clip\_duration]$).
   - Full audit trail recorded in `BoundaryRefinementMetadata` (original vs. refined boundaries, deltas, signals used, rationale).
5. **Physical Video Rendering & Local API Bridge (W8 & W9)**:
   - Frame-accurate FFmpeg subclip rendering (libx264/AAC re-encoding with stream-copy fallback).
   - Generates playable `.mp4` video files and a structured `clips/manifest.json`.
   - FastAPI backend service with async task queue, streaming endpoints, and interactive Next.js studio.

---

## Directory Structure

```
runs/<run_id>/
├── multimodal_features.json        # Raw extracted multimodal features
├── transcript.json                 # Aligned word timestamps
├── conversation.json               # Speaker turn dynamics
├── frames.json                     # Scene cuts and motion energy
├── prosody.json                    # Acoustic F0 pitch & RMS energy
├── common_evidence_bundle.json     # Isolated cross-modal evidence container
├── mter_output.json                # Ranked highlight candidates with boundary refinement metadata
├── mter_ledger.json                # Full inspectable reasoning ledger
├── clips_manifest.json             # Top-level manifest of exported video clips
├── demo_report.html                # Interactive telemetry HTML report with video players & refinement pills
└── clips/                          # Physical exported highlight MP4 files
    ├── manifest.json               # Catalog of exported clips with timestamps, file sizes, & refinement logs
    ├── highlight_01_<cand_id>.mp4  # Playable highlight clip #1 (Rank 1)
    ├── highlight_02_<cand_id>.mp4  # Playable highlight clip #2 (Rank 2)
    └── highlight_03_<cand_id>.mp4  # Playable highlight clip #3 (Rank 3)
```

---

## Setup & Execution

### 1. Environment Setup

```powershell
cd backend
.venv\Scripts\activate
```

Ensure FFmpeg is installed and accessible in `PATH` (`ffmpeg -version`).

### 2. Running Automated Tests

Run the automated test suite (unit tests, mock tests, boundary refiner tests, API bridge tests, benchmark adapters, and evaluation regression tests):

```powershell
cmd.exe /c "set PYTHONPATH=. && .venv\Scripts\python.exe -m unittest discover tests -v"
```

### 3. Running the Pipeline Demo with Video Export (W8/W10)

Execute the end-to-end pipeline with Top-$K$ highlight selection, boundary refinement, and physical MP4 rendering by supplying a local video file:

```powershell
python -m demo.run_demo --video path/to/your_video.mp4 --output-dir runs/my_run --top-k 3 --render-mode accurate
```

> [!NOTE]
> **Media & Run Artifact Availability**: Raw `.mp4` video files, audio waveforms, and stored run directories (`runs/`) are excluded from the repository to prevent version-control bloat. Users should supply their own local `.mp4` video (replacing `path/to/your_video.mp4`) when running the end-to-end pipeline demonstration.

#### CLI Options:
- `--video <path>`: Source video file (`.mp4`, `.mov`).
- `--output-dir <path>`: Output directory for artifacts, clips, and HTML telemetry report.
- `--top-k <int>`: Number of top non-overlapping highlight candidates to extract (default: `3`).
- `--refine-boundaries`: Enable evidence-aware temporal boundary refinement (default: enabled).
- `--no-refine`: Disable boundary refinement stage (preserves raw MTER boundaries).
- `--render-clips`: Render physical MP4 clips using FFmpeg (default: enabled).
- `--no-render`: Skip physical rendering for evaluation/metadata-only benchmarks.
- `--render-mode {accurate,fast}`: 
  - `accurate` (default): Exact frame boundaries using `libx264`/`aac` re-encoding with `+faststart`.
  - `fast`: Stream-copy (`-c copy`) without re-encoding, falling back to accurate mode if copy fails.
- `--use-cache`: Reuse pre-computed extraction artifacts for rapid iteration.

---

---

## Benchmark Evaluation & Ablation Study (W11–W12)

ClipSense AI includes a deterministic evaluation harness to investigate the core research question:

> **Does conflict-sensitive temporal arbitration improve query-free long-form highlight selection compared with transcript-only selection and simple multimodal score aggregation?**

### Evaluated Four-System Framework

| System | Identifier | Technical Specification | Evidence Isolated |
| :--- | :--- | :--- | :--- |
| **System A** | `system_a_transcript_only` | Spoken transcript semantic proposals from `TranscriptExpert` scored by lexical/discourse semantic importance and greedy NMS (with `LegacyTranscriptBaselineSelector` retained as a heuristic fallback when expert proposals are absent). | Lexical and discourse semantics only (blinds video/audio). |
| **System B** | `system_b_simple_aggregation` | Four-expert multimodal baseline using uniform unweighted coverage aggregation without MTER graph or conflict penalties. | Multimodal coverage without structural arbitration. |
| **System C** | `system_c_mter` | Multimodal Temporal Evidence Reasoner with temporal compatibility graph, boundary clustering, and explicit cross-modal conflict penalties. | Cross-modal agreement graph & conflict penalty arbitration. |
| **System D** | `system_d_mter_refined` | System C candidate highlight passed through evidence-aware boundary refinement (sentence boundary snapping & padding). | Boundary refinement impact on identical candidates. |

### Architectural Mechanisms (Verified in Implementation)
- **Modality Support Aggregation:** MTER candidate scoring applies **uniform 1/4 (0.25) weights** across modalities with proposal-level confidence estimates; it does *not* currently implement learned or differentiated modality reliability weights.
- **Cross-Modal Conflict Penalties:** MTER detects start/end boundary discrepancies and interval breadth disparities across modalities, penalizing candidate composite scores via a configured multiplier (`penalty_weight = 0.15`).
- **Boundary Refinement:** Formulated as a deterministic post-processing heuristic snapping candidate boundaries to word onsets and sentence completions. On the development sample, it shifted boundaries closer to turn-0 diarization timestamps, but because that reference interval was derived from diarization rather than independent human ground truth, boundary refinement cannot be claimed to improve ground-truth clipping accuracy prior to collecting human rater responses.

### Software Validation Fixtures vs. Benchmark Evaluations
- **Development Fixture (`dev_fixture.json`):** A single-sample software validation run ($N=1$, 90-second conversational sample) used strictly for integration testing and pipeline determinism. The reference interval ($[0.091\text{s}, 27.270\text{s}]$) was derived from Turn 0 of speaker diarization (`conversation.json`) for pipeline verification, **not** independent human ground truth.
- **Rhapsody Replay-Engagement Evaluation:** Evaluated separately on five official sub-hour test episodes ($N=5$, 46m max duration) from the Rhapsody benchmark (Park et al., COLM 2025). Full methodology, baseline comparisons, and replay-alignment metrics are documented in the [Rhapsody Evaluation Report](backend/evaluation/rhapsody_sub3600_eval_output_final/EVALUATION_REPORT.md).
- **Official TVSum and QVHighlights Status:** Synthetic fixtures (`tvsum_toy_fixture.json`, etc.) are included strictly for software adapter contract validation. **Official full-benchmark evaluations on TVSum (Song et al., CVPR 2015) and QVHighlights (Lei et al., NeurIPS 2021) have not been executed.**
- **Sample Count & Generalizability:** Reported single-sample fixture metrics ($N=1$) demonstrate code execution rather than highlight quality or statistical superiority.

### Running the Evaluation CLI
```powershell
# Run 4-system comparative evaluation on development fixture:
python -m evaluation.run_evaluation --systems A,B,C,D --mock-llm --output-dir runs/w11_system_comparison_run

# Run TVSum adapter interface contract validation:
python -m evaluation.run_evaluation --dataset evaluation/datasets/tvsum_toy_fixture.json --adapter tvsum --systems A,B,C,D --mock-llm
```
Generates structured JSON (`eval_results.json`), academic markdown scorecard (`eval_report.md`), and standalone HTML dashboard (`eval_report.html`).

