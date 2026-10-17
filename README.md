# ClipSense AI

> **Query-Free Long-Form Video Highlight Extraction Framework**  
> *Deterministic Multimodal Temporal Evidence Reasoner (MTER)*

---

## Overview

ClipSense AI is a research prototype that identifies, ranks, and physically renders self-contained highlight clips from long-form conversational videos (podcasts, lectures, interviews).

Instead of relying on prompt-biased text queries or uninspectable end-to-end black boxes, ClipSense AI combines four independent modality experts into a **Common Evidence Bundle** and arbitrates cross-modal agreement using a deterministic **Multimodal Temporal Evidence Reasoner (MTER)**.

---

## Architecture Pipeline

1. **Multimodal Extraction (W2)**: Continuous word timestamps (WhisperX), visual scene cuts and motion energy (PySceneDetect, OpenCV), acoustic prosody pitch ($F_0$) and RMS energy (Librosa), and conversational turn-taking dynamics.
2. **Modality-Specific Experts (W3 & W4)**:
   - **Transcript Expert**: Sliding-window semantic completeness and discourse payoff.
   - **Conversation Expert**: Rapid turn exchanges, question-answer clustering.
   - **Visual Expert**: Optical frame motion density and shot boundary transitions.
   - **Prosody Expert**: Standardized acoustic pitch and energy peak deviations.
24: 3. **MTER Reasoner & Top-$K$ Selection (W5 & W8)**:
   - Temporal compatibility graph connecting overlapping proposals into event regions.
   - Intra-region boundary clustering and explicit conflict detection.
   - Deterministic composite scoring bounded to $[0.0, 1.0]$.
   - Configurable **Top-$K$ highlight selection** with **Non-Maximum Suppression (NMS)** to eliminate duplicate or heavily overlapping subclips.
4. **Evidence-Aware Temporal Boundary Refinement (W10)**:
   - Modular, deterministic boundary refiner aligning highlight candidates with speech activity.
   - Prevents mid-word and mid-utterance cuts by snapping to WhisperX word onsets/offsets and sentence boundaries.
   - Inter-utterance pause alignment verified via acoustic prosody (low voicing fraction, RMS energy dips).
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

Run the complete 115-test verification suite (unit tests, mock tests, boundary refiner tests, API bridge tests, and real FFmpeg integration tests):

```powershell
cmd.exe /c "set PYTHONPATH=. && .venv\Scripts\python.exe -m unittest discover tests -v"
```

### 3. Running the Pipeline Demo with Video Export (W8/W10)

Execute the end-to-end pipeline with Top-$K$ highlight selection, boundary refinement, and physical MP4 rendering:

```powershell
python -m demo.run_demo --video tests/sample_conversational_90s.mp4 --output-dir runs/my_run --top-k 3 --render-mode accurate
```

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
