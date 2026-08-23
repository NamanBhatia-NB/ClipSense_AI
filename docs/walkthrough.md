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

## Verification Results (W1)

A dedicated test suite was executed:
```bash
.venv\Scripts\python.exe tests\test_w1_architecture.py
```
**Output**:
```text
All W1 architecture & schema tests passed successfully!
```

---

# Walkthrough - ClipSense W2: Multimodal Data Extraction Pipeline

## Scope & Separation of Concerns

The W2 multimodal data-extraction stage was implemented and validated on a representative conversational sample. The pipeline generates timestamp-aligned transcript, visual-frame, acoustic-prosodic and conversation-structure artifacts. Higher-level modality-specific expert reasoning and MTER are intentionally deferred to subsequent stages.

- **W2 (Data Extraction)**:
  - Timestamped transcript extraction
  - Visual data extraction (sampled frames and scene boundaries)
  - Audio prosody extraction (physical acoustic metrics)
  - Conversation structure extraction (speaker turns and pauses)
- **W3 / W4 (Modality-Specific Expert Reasoning)**:
  - Transcript Expert (W3)
  - Conversation Expert (W3)
  - Visual Expert (W4)
  - Prosody Expert (W4)
- **W5 (Evidence Integration & Arbitration)**:
  - Common evidence bundle & auxiliary temporal indexing
  - Multimodal Temporal Evidence Reasoner (MTER) prototype
- **Post-W5**:
  - Temporal boundary optimization & downstream render (LR-ASD, 9:16 crop, ASS subtitles)

---

## Modules Implemented in W2

1. **`pipeline/extraction/transcript_extractor.py`**:
   - Decoupled WhisperX transcription with wav2vec2 forced alignment.
   - Produces continuous floating-point word-level timestamps (`[0.091s -> 0.631s]`).
   - Introduces **STRICT MODE** requiring real model inference and failing immediately if model execution cannot run. Fallback mode is strictly isolated to offline infrastructure testing.
   - Model used for representative validation: `tiny.en` (recorded in `manifest.json` as `"whisper_model": "tiny.en"`). The model is configurable so subsequent evaluations can specify another model without code modification.
   - Speaker metadata is preserved where available (`SPEAKER_00`).
2. **`pipeline/extraction/frame_extractor.py`** (*Visual Data Extraction*):
   - Uniformly samples timestamped frames (default 1.0 fps) and detects shot transitions using PySceneDetect ContentDetector.
   - Preserves continuous floating-point timestamps for each sampled frame.
3. **`pipeline/extraction/prosody_extractor.py`** (*Audio Prosody Extraction*):
   - Purely physical acoustic signal measurements: autocorrelation F0 pitch, RMS loudness energy, dynamic range, voicing fraction, and speaking rate over sliding windows.
   - Strictly no highlight classification, emotion detection, or arbitrary thresholds.
4. **`pipeline/extraction/conversation_extractor.py`** (*Conversation Structure Extraction*):
   - Measurable dialogue dynamics: distinguishes multi-speaker conversational turn-taking from single-speaker pause-separated speech segments (>0.7s silence thresholds), while preserving speaker IDs and pause/switch metadata for multi-speaker turn-taking detection.
   - Discourse interpretation is deferred to the W3 Conversation Expert.
5. **`pipeline/extraction/extractor_pipeline.py`** (*Extraction Coordinator*):
   - Coordinates extraction stages, enforces continuous timestamp validity against actual video duration, generates reproducible artifacts (`transcript.json`, `frames/`, `frames.json`, `prosody.json`, `conversation.json`, `manifest.json`), and maintains configuration-aware cache invalidation.

---

## Verification & Validation Results

### 1. W1 Architecture & Schemas Invariant Tests
```bash
.venv\Scripts\python.exe tests/test_w1_architecture.py
```
**Output**:
```text
All W1 architecture & schema tests passed successfully!
```

### 2. W2 Multimodal Extractor Unit Tests
```bash
.venv\Scripts\python.exe tests/test_w2_extractors.py
```
**Output**:
```text
Running W2 unit tests...
  [✓] FrameExtractor tests passed
  [✓] ProsodyExtractor tests passed
  [✓] ConversationExtractor tests passed
  [✓] Conversation pause fallback tests passed
All W2 unit tests passed successfully!
```

### 3. Automated Caching & Timestamp Validation Tests
```bash
.venv\Scripts\python.exe tests/test_w2_caching_and_timestamps.py
```
**Output**:
```text
Running W2 Caching and Timestamp Validation Tests...
  [✓] Automated timestamp validation passed (bounds, finite, continuous float)
  [✓] Caching & automatic cache invalidation on config change passed
  [✓] Strict mode failure semantics passed
All W2 caching and timestamp tests passed successfully!
```

### 4. Representative Conversational Sample Validation (90.0s, STRICT Real WhisperX Mode)
```bash
.venv\Scripts\python.exe tests/run_representative_validation.py
```
**Output**:
```text
====================================================================
ClipSense W2: Official Representative Conversational Validation
Sample Video: sample_conversational_90s.mp4 (90.0s)
Execution Mode: STRICT (Real WhisperX inference required; fallback disabled)
====================================================================
[✓] Transcript extraction
[✓] Visual data extraction
[✓] Prosody extraction
[✓] Conversation structure extraction

Generated extraction artifacts:
  - run_dir: runs\representative_90s_validation
  - transcript_json: runs\representative_90s_validation\transcript.json
  - frames_dir: runs\representative_90s_validation\frames
  - frames_json: runs\representative_90s_validation\frames.json
  - prosody_json: runs\representative_90s_validation\prosody.json
  - conversation_json: runs\representative_90s_validation\conversation.json
  - manifest_json: runs\representative_90s_validation\manifest.json

--- DETAILED INSPECTION OF GENERATED MULTIMODAL EVIDENCE ---
Total Words Extracted: 321
First 5 Words with Exact Timestamps:
  - 'I': [0.091s -> 0.631s] (score: 0.403)
  - 'was': [0.872s -> 0.972s] (score: 0.894)
  - 'shit': [1.152s -> 1.492s] (score: 0.778)
  - 'scared': [1.532s -> 1.812s] (score: 0.876)
  - 'of': [1.852s -> 1.912s] (score: 0.528)
Last 5 Words with Exact Timestamps:
  - 'to': [88.863s -> 88.903s] (score: 0.672)
  - 'have': [88.943s -> 89.104s] (score: 0.91)
  - 'a': [89.144s -> 89.164s] (score: 0.991)
  - 'firm': [89.264s -> 89.584s] (score: 0.878)
  - 'line.': [89.624s -> 89.865s] (score: 0.922)

Sampled Visual Frames Count: 90 at 25.0 fps
Detected Scene Boundaries: 9
Frame 0 Timestamp: 0.00s | Frame Last: 89.00s

Prosody Windows Count: 180 (hop: 0.5s)
Voiced Windows Count: 172/180
Average Fundamental Frequency (F0): 156.6 Hz
Average Acoustic RMS Energy: 0.1034

Speech Segments Extracted: 4 (pause-separated speech segments; single-speaker sample, no multi-speaker turn-taking)
Segment Pacing: 2.67 segments/min
Detected Speaker: 1 (SPEAKER_00) | Multi-speaker Turn Transitions: 0
Interaction Nature: Single-speaker monologue segmented by inter-utterance pauses (>= 0.7s)
  - Speech Segment 0 (SPEAKER_00): [0.09s -> 27.27s], 106 words, pause_after=1.20s
  - Speech Segment 1 (SPEAKER_00): [28.47s -> 71.56s], 163 words, pause_after=0.98s
  - Speech Segment 2 (SPEAKER_00): [72.54s -> 83.50s], 32 words, pause_after=1.12s
  - Speech Segment 3 (SPEAKER_00): [84.62s -> 89.86s], 20 words, pause_after=0.14s
Speaker IDs and pause/switch information preserved for downstream multi-speaker turn-taking detection.

Actual Video Duration: 90.00s
Validation Checks Passed: True
Inference Mode Confirmed: real_whisperx
Whisper Model Recorded: tiny.en
Strict Mode Active: True
Fallback Used: False
Speaker Metadata Preserved: 1 speaker detected (SPEAKER_00)
Prosody Time Range: [0.000s -> 90.000s] (Clamped <= 90.00s)

[✓] ALL CHECKS PASSED: Real WhisperX inference confirmed on representative 90s sample!
```


