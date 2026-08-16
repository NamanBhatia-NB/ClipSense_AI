# Walkthrough - ClipSense W1: Architecture & Schemas

## Completed Objectives in W1

The foundational architecture and schema contracts for ClipSense have been implemented, strictly adhering to all architectural constraints without modifying the frontend or compromising the integrity of authoritative continuous timestamps.

---

## Changes Implemented

### 1. Core Data Schemas (`core/schemas.py`)
- **Multimodal Data Extraction**:
  - `TimestampedWord`: Exact continuous start/end timestamps (`float` in seconds) and optional alignment confidence.
  - `TranscriptSegment`, `TranscriptData`: Structured spoken text and word alignments.
  - `SceneBoundary`, `SampledFrameMetadata`, `VisualData`: Visual frames and shot boundaries.
  - `ProsodyWindow`, `ProsodyData`: F0 fundamental frequency, RMS energy, dynamic range, and voicing fraction over continuous windows.
  - `SpeakerTurn`, `ConversationData`: Turn-taking timeline and silence intervals.
- **Independent Expert Evidence & Proposals**:
  - `ProposalSourceMetadata`: Explicitly logs `SourceResolutionType` (`word_timestamp`, `frame_timestamp`, `prosody_window`, `speaker_turn`), temporal resolution in seconds, and alignment anchor.
  - `TemporalProposal`: Preserves continuous floating-point `start_time` and `end_time`, `confidence_estimate` (0.0 to 1.0), measurable `evidence_type`, `explanation`, and `supporting_features`.
  - `ExpertEvidenceBundle`: Isolated container for each expert's findings (`transcript`, `visual`, `prosody`, `conversation`). **No premature score collapse.**
- **Auxiliary Index & Common Evidence Bundle**:
  - `AuxiliaryTemporalGrid`: Discretized 1-second view for query indexing without modifying continuous timestamps.
  - `CommonEvidenceBundle`: Aggregates the four independent bundles while preserving distinct modality dimensions.
- **MTER Contracts**:
  - `SemanticVerificationCriteria`: Explicit checklist covering `natural_semantic_start`, `sufficient_context`, `important_content_retained`, `complete_conversational_payoff`, and `no_mid_sentence_ending`.
  - `MTERCandidate`: Multi-expert temporal region with agreement IoU, conflict notes, semantic verification criteria, and programmatic reasoning trace.
  - `MTEROutput`: Selected candidate bounds for downstream boundary refinement.
- **Frontend Compatibility**:
  - `ProcessVideoRequest`, `ProcessVideoResponse`, and `ClipMetadata` models matching the Next.js Inngest endpoint interface.

### 2. Configurable Baselines & Thresholds (`app/config.py`)
- Configurable baselines rather than arbitrary hard-coded claims:
  - Prosody: `local_baseline_window_sec` (30s), `peak_f0_multiplier` (1.5x), `peak_energy_multiplier` (1.5x), `min_voicing_fraction` (0.3).
  - Visual: `scene_threshold` (27.0), `sample_fps` (1.0 fps), `motion_change_threshold` (0.25).
  - Conversation: `turn_pause_threshold_sec` (0.7s), `rapid_exchange_turn_rate_per_min` (12.0 turns/min).
  - Clip bounds: `min_clip_duration_sec` (25.0s), `max_clip_duration_sec` (90.0s), `dead_air_buffer_sec` (0.35s).

### 3. Base Expert & Pluggable Visual Interface (`pipeline/experts/base.py`)
- `BaseExpert`: Abstract base class enforcing `evaluate(...) -> ExpertEvidenceBundle`.
- `VisualFeatureExtractorInterface`: Protocol enabling pluggable visual representations without premature model commitment.

### 4. Storage & Workspace Management (`core/storage.py`)
- `WorkspaceManager`: Handles scoped directory creation (`media/`, `frames/`, `clips/`, `subtitles/`) and cleanup.
- `S3StorageClient`: S3 download/upload abstraction using `boto3`.

### 5. Staged Pipeline Structure
- Initialized all pipeline packages:
  - `pipeline/extraction/` (W2)
  - `pipeline/experts/` (W3 & W4)
  - `pipeline/aggregation/` (W5)
  - `pipeline/reasoner/` (W5)
  - `pipeline/optimization/` (Post-W5)
  - `pipeline/render/` (Downstream LR-ASD + Subtitles)
- `pipeline/orchestrator.py`: Invariant validation for multi-expert bundles.

---

## Verification Results

A dedicated test suite was executed:
```bash
.venv\Scripts\python.exe tests\test_w1_architecture.py
```
**Output**:
```text
All W1 architecture & schema tests passed successfully!
```
