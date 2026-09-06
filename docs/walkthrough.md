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

---

# W3: Independent Modality-Specific Temporal Evidence Generation (Transcript Expert + Conversation Expert)

## Overview & Research Goal
Development of the Transcript Expert and Conversation Expert for generating independent, timestamped semantic and conversational evidence proposals.

In strict adherence to the research roadmap:
- **W3 is strictly independent modality-specific temporal evidence generation**, NOT final highlight selection, clipping, or viral classification.
- **No final highlight generation, MTER reasoning, multimodal conflict resolution, or boundary optimization** was implemented.
- The two experts independently evaluate extracted W2 artifacts and output isolated evidence bundles without cross-expert score collapse or temporal merging.

## Architectural Components Implemented
1. **`core/llm.py`** (*Provider-Agnostic Structured LLM Client*):
   - Abstract `LLMClient` with concrete `GeminiLLMClient` (Google GenAI SDK, `gemini-2.5-flash`) and offline `MockLLMClient`.
   - Strict Pydantic structured output, exponential backoff retry for transient network/rate-limit errors, and zero hidden chain-of-thought (`thinking_budget=0`).
2. **`pipeline/experts/transcript_expert.py`** (*Transcript Expert*):
   - Evaluates linguistic and semantic discourse across bounded analysis windows (`min: 15.0s`, `max: 45.0s`).
   - Programmatically grounds candidate boundaries using indexed word representation (`start_word_index`, `end_word_index`) to exact continuous WhisperX timestamps (`SourceResolutionType.WORD_TIMESTAMP`).
3. **`pipeline/experts/conversation_expert.py`** (*Conversation Expert*):
   - Evaluates conversational structure and discourse progression across localized sub-intervals (`min: 15.0s`, `max: 50.0s`).
   - Source & Anchor Consistency:
     - Proposals aligned with speech segment boundaries report `SourceResolutionType.SPEECH_SEGMENT` with anchor `speech_segment_X`.
     - Proposals targeting sub-segment word intervals report `SourceResolutionType.WORD_TIMESTAMP` with anchor `word_anchored_X:Y`.
   - Single-speaker safety: recognizes single-speaker monologues, sets `discourse_unit_complete = True/False`, and strictly enforces `speaker_interaction: false` and `is_single_speaker_monologue: true` without fabricating multi-speaker interactions.

## Verification & Representative Validation Results

### 1. Automated Test Suite
- `tests/test_w1_architecture.py`: Passed.
- `tests/test_w2_extractors.py`: Passed.
- `tests/test_w2_caching_and_timestamps.py`: Passed.
- `tests/test_transcript_expert.py`: Passed (includes deterministic word grounding test and candidate duration constraint checks).
- `tests/test_conversation_expert.py`: Passed (includes single-speaker safety, multi-speaker exchange, source/anchor consistency, artifact provenance loading, and prompt transcript injection tests).

### 2. Representative Validation Output (`tests/run_w3_expert_validation.py`)
```
====================================================================
ClipSense W3: Independent Transcript & Conversation Experts Validation
====================================================================
[PROVENANCE & DATA INTEGRITY]
  - Source Run ID: representative_90s_validation
  - Source Video: .../sample_conversational_90s.mp4
  - Transcript Artifact: .../runs/representative_90s_validation/transcript.json
    SHA256: 9bd06b71c1143051434b77d86e40ee12acb17916b50a6465141cd4cc029a3aa7 (Duration: 89.86s, Words: 321)
  - Conversation Artifact: .../runs/representative_90s_validation/conversation.json
    SHA256: 8a3feefe178e9ac0a5b2c6b0340f893ec7f5c04b842bf8b732fe764edb1bbb89 (Duration: 90.00s, Turns: 4)
  - Consistency Check: Confirmed identical source video and duration bounds.

[✓] W2 transcript loaded
[✓] W2 conversation structure loaded
[✓] Transcript Expert
[✓] Conversation Expert
[✓] Proposal schema validation
[✓] Timestamp validation
[✓] Evidence artifacts generated

--- TRANSCRIPT EXPERT EVIDENCE PROPOSALS ---
Total Proposals Generated: 3
  - [5.975s -> 24.548s] (dur: 18.57s, conf: 0.90)
    Type: explanatory_claim | Anchor: word_index_20:96
    Source: word-level timestamp | Avg Word Duration: 0.2412s
    Excerpt: "I think kids get away with too much and I think there's got an element of discipline. There's got an element of you've got that fear for your dad. I t..."
    Features: {'topic_transition': False, 'contextual_completeness': 0.9, 'semantic_importance': 0.8, 'self_contained': True, 'word_count': 77, 'start_word_index': 20, 'end_word_index': 96, 'start_word': 'I', 'end_word': 'that.', 'avg_word_duration_sec': 0.2412077922077922}
    Explanation: The speaker explains that kids get away with too much and need discipline, suggesting that a healthy fear or respect for parents, particularly fathers, can prevent them from making bad choices.

  - [24.568s -> 48.014s] (dur: 23.45s, conf: 0.85)
    Type: explanatory_claim | Anchor: word_index_97:178
    Source: word-level timestamp | Avg Word Duration: 0.2859s
    Excerpt: "that. I think it's healthy to have some sort of Feel over your parents back in the day you were getting slapped in the head and it's kicked in the ass..."
    Features: {'topic_transition': False, 'contextual_completeness': 0.85, 'semantic_importance': 0.75, 'self_contained': True, 'word_count': 82, 'start_word_index': 97, 'end_word_index': 178, 'start_word': 'I', 'end_word': 'generation.', 'avg_word_duration_sec': 0.2859268292682927}
    Explanation: The speaker claims it's healthy to have some fear of parents, contrasting past disciplinary methods (slapping, kicking) with current 'soft' generations who lack respect.

  - [68.272s -> 89.865s] (dur: 21.59s, conf: 0.90)
    Type: explanatory_claim | Anchor: word_index_253:320
    Source: word-level timestamp | Avg Word Duration: 0.3175s
    Excerpt: "So the first thing is I agree with you that we've got soft on our kids. But my dad was at times I did things like I'd like to him or I'd hide things f..."
    Features: {'topic_transition': False, 'contextual_completeness': 0.9, 'semantic_importance': 0.8, 'self_contained': True, 'word_count': 68, 'start_word_index': 253, 'end_word_index': 320, 'start_word': 'So', 'end_word': 'line.', 'avg_word_duration_sec': 0.31754411764705864}
    Explanation: The speaker agrees that society has become soft on children but also explains that their own father's strictness led to fear and hiding things, suggesting a need for balance in parenting.

--- CONVERSATION EXPERT EVIDENCE PROPOSALS ---
Total Proposals Generated: 2
  - [0.091s -> 27.269s] (dur: 27.18s, conf: 0.90)
    Type: monologue_thematic_unit | Anchor: speech_segment_0
    Source: speech segment | Pause Threshold: 0.70s
    Excerpt: "I was shit scared of my dad, but healthy scared. I think that's important for kids to know it is. I think kids get away with too much and I think ther..."
    Features: {'discourse_unit_complete': True, 'speaker_interaction': False, 'pause_boundary_supported': True, 'discourse_phase': 'setup_development_payoff', 'num_speakers': 1, 'is_single_speaker_monologue': True}
    Explanation: The speaker introduces the concept of 'healthy fear' for a parent and explains its importance for discipline and respect in children's decision-making.

  - [28.471s -> 71.557s] (dur: 43.09s, conf: 0.85)
    Type: monologue_thematic_unit | Anchor: speech_segment_1
    Source: speech segment | Pause Threshold: 0.70s
    Excerpt: "Feel over your parents back in the day you were getting slapped in the head and it's kicked in the ass. Well, I was anyway coming from Glasgow, but......"
    Features: {'discourse_unit_complete': True, 'speaker_interaction': False, 'pause_boundary_supported': True, 'discourse_phase': 'development', 'num_speakers': 1, 'is_single_speaker_monologue': True}
    Explanation: The speaker contrasts past disciplinary methods with current approaches, arguing that the present generation is 'soft' due to a lack of respect for parents, linking it back to the 'fear' element.

Evidence artifacts successfully saved:
  - runs/representative_90s_validation/transcript_evidence.json
  - runs/representative_90s_validation/conversation_evidence.json

[✓] ALL W3 EVIDENCE CHECKS PASSED (Localized candidate proposals verified)!

---

# Walkthrough - ClipSense W4: Visual Expert + Prosody Expert

## Milestone Overview
**W4**: Independent visual and prosodic temporal evidence generation from the W2 multimodal representations.

## Key Principles & Scope Boundaries
- **Strict Modality Independence**:
  - The Visual Expert operates exclusively on visual artifacts (`frames.json`, sampled frames, scene boundaries).
  - The Prosody Expert operates exclusively on physical acoustic prosody artifacts (`prosody.json`).
  - No cross-modal reasoning, score aggregation, or fusion is performed.
- **Contract Adherence**:
  - Both experts inherit `BaseExpert` and return structured `ExpertEvidenceBundle` containers holding `TemporalProposal` objects.
  - Continuous floating-point seconds are strictly preserved ($0.0 \le \text{start} < \text{end} \le \text{duration}$).
  - Non-probabilistic evidence-strength `confidence_estimate` in $[0.0, 1.0]$: strictly documented and treated as relative evidence strength, not a calibrated probability.
  - Explicit source resolution metadata (`SourceResolutionType.FRAME_TIMESTAMP` and `SourceResolutionType.PROSODY_WINDOW`).
- **Deferred Functionality**:
  - Cross-expert aggregation, evidence normalization across experts, agreement analysis, conflict resolution, MTER, boundary optimization, final clip selection, and frontend changes remain deferred to W5 and downstream stages.

## Components Implemented

### 1. Visual Feature Extractor & Visual Expert (`pipeline/experts/visual_expert.py`)
- **`LightweightFrameDiffExtractor`**:
  - Implements `VisualFeatureExtractorInterface`.
  - Downsamples frames to 32x32 grayscale intensity representations and computes normalized L1 inter-frame distance.
  - Fast, deterministic, and avoids loading heavy foundation models at import time.
- **`VisualExpert`**:
  - Confined strictly to visual-activity, visual-change, and scene-transition evidence generation (no semantic visual understanding or gesture recognition).
  - Computes visual activity curves smoothed over temporal windows (`activity_smoothing_sec = 3.0s`).
  - Incorporates scene transition density and detects visual shifts.
  - Clusters candidate active frames into candidate intervals (`min_candidate_duration_sec = 10.0s`, `max_candidate_duration_sec = 45.0s`).
  - Anchors boundaries to exact sampled frame timestamps (`SourceResolutionType.FRAME_TIMESTAMP`), explicitly distinguishing sampled-frame indices from native video frame indices (`sampled_frame_S:E (native_frame_NS:NE)`).
  - Explicitly records effective sampling interval (1.0s) distinct from native video frame rate (25.0 FPS).

### 2. Prosody Expert (`pipeline/experts/prosody_expert.py`)
- **`ProsodyExpert`**:
  - Computes a centered moving local acoustic baseline (median voiced F0 and RMS energy over `local_baseline_window_sec = 30.0s`).
  - Filters unvoiced silence (`voicing_fraction < min_voicing_fraction = 0.3`) from baseline calculations to prevent distortion.
  - Computes relative elevation ratios for F0 and RMS energy, forming a composite acoustic emphasis score without hardcoded thresholds.
  - Generates faithful explanations dynamically from actual measured feature values (distinguishing pitch rise vs. energy fall vs. transient emphasis peaks).
  - Clusters emphasis windows (`merge_gap_sec = 3.0s`) and bounds intervals between `min_candidate_duration_sec = 10.0s` and `max_candidate_duration_sec = 45.0s`.
  - Anchors boundaries to exact acoustic window boundaries (`SourceResolutionType.PROSODY_WINDOW`), distinguishing window hop (0.5s) from audio sample rate (16000 Hz).

### 3. Automated Verification Suites
- `tests/test_visual_expert.py`: 8 unit tests validating schemas, timestamp bounds, duration consistency, determinism, duration clamping, whole-input rejection, and empty/insufficient frame handling.
- `tests/test_prosody_expert.py`: 9 unit tests validating schemas, timestamp bounds, duration consistency, determinism, silent region filtering, configurable baseline parameters, no hardcoded thresholds, and whole-input rejection.
- Complete regression suite verified: W1 architecture, W2 extractors, W2 caching/timestamps, W3 Transcript Expert, W3 Conversation Expert, W4 Visual Expert, and W4 Prosody Expert.

## Representative Validation Results (`tests/run_w4_expert_validation.py`)

Executed against representative 90s validation run (`runs/representative_90s_validation/`):

```text
2026-09-10 21:47:19,904 [INFO] Loaded W2 visual data: 90 frames, SHA256: 1bf6a07d4ff3...
2026-09-10 21:47:19,916 [INFO] Loaded W2 prosody data: 180 windows, SHA256: 3b5ba7f91461...
2026-09-10 21:47:20,849 [INFO] Visual Expert produced 4 localized temporal proposals.
2026-09-10 21:47:20,962 [INFO] Prosody Expert produced 5 localized temporal proposals.
[✓] W2 visual data loaded
[✓] W2 prosody data loaded
[✓] Visual Expert
[✓] Prosody Expert
[✓] Proposal schema validation
[✓] Timestamp validation
[✓] Evidence artifacts generated

--- VISUAL EXPERT EVIDENCE PROPOSALS ---
Total Proposals Generated: 4
  - [0.000s -> 10.000s] (dur: 10.00s, conf: 0.39)
    Type: visual_shift | Anchor: sampled_frame_0:10 (native_frame_0:250)
    Source: frame_timestamp
    Sampling interval: 1.0s | Native FPS: 25.0
    Features: {'visual_change_score': 0.0339, 'peak_visual_change': 0.2892, 'mean_visual_change': 0.0339, 'scene_change_count': 1, 'frame_count': 11, 'start_sampled_frame_index': 0, 'end_sampled_frame_index': 10, 'start_native_frame_index': 0, 'end_native_frame_index': 250, 'sampling_interval_sec': 1.0, 'native_fps': 25.0}
    Explanation: A pronounced visual shift occurs in this interval (peak change: 0.289) with 1 scene transition(s).

  - [10.000s -> 20.000s] (dur: 10.00s, conf: 0.19)
    Type: visual_activity | Anchor: sampled_frame_10:20 (native_frame_250:500)
    Source: frame_timestamp
    Sampling interval: 1.0s | Native FPS: 25.0
    Features: {'visual_change_score': 0.0438, 'peak_visual_change': 0.0749, 'mean_visual_change': 0.0438, 'scene_change_count': 0, 'frame_count': 11, 'start_sampled_frame_index': 10, 'end_sampled_frame_index': 20, 'start_native_frame_index': 250, 'end_native_frame_index': 500, 'sampling_interval_sec': 1.0, 'native_fps': 25.0}
    Explanation: A sustained increase in visual activity occurs around this interval (mean change score: 0.044, frames: 11).

  - [32.000s -> 42.000s] (dur: 10.00s, conf: 0.64)
    Type: scene_transition_density | Anchor: sampled_frame_32:42 (native_frame_800:1050)
    Source: frame_timestamp
    Sampling interval: 1.0s | Native FPS: 25.0
    Features: {'visual_change_score': 0.08, 'peak_visual_change': 0.2883, 'mean_visual_change': 0.08, 'scene_change_count': 2, 'frame_count': 11, 'start_sampled_frame_index': 32, 'end_sampled_frame_index': 42, 'start_native_frame_index': 800, 'end_native_frame_index': 1050, 'sampling_interval_sec': 1.0, 'native_fps': 25.0}
    Explanation: A cluster of 2 scene transitions occurs within this interval with sustained visual changes (mean change score: 0.080).

  - [46.000s -> 63.000s] (dur: 17.00s, conf: 0.77)
    Type: scene_transition_density | Anchor: sampled_frame_46:63 (native_frame_1150:1575)
    Source: frame_timestamp
    Sampling interval: 1.0s | Native FPS: 25.0
    Features: {'visual_change_score': 0.0898, 'peak_visual_change': 0.2917, 'mean_visual_change': 0.0898, 'scene_change_count': 5, 'frame_count': 18, 'start_sampled_frame_index': 46, 'end_sampled_frame_index': 63, 'start_native_frame_index': 1150, 'end_native_frame_index': 1575, 'sampling_interval_sec': 1.0, 'native_fps': 25.0}
    Explanation: A cluster of 5 scene transitions occurs within this interval with sustained visual changes (mean change score: 0.090).

--- PROSODY EXPERT EVIDENCE PROPOSALS ---
Total Proposals Generated: 5
  - [0.000s -> 10.000s] (dur: 10.00s, conf: 0.54)
    Type: vocal_emphasis | Anchor: window_0:18
    Window hop: 0.5000s | Sample Rate: 16000.0 Hz
    Features: {'f0_change': 1.2836, 'energy_change': 0.9224, 'composite_emphasis': 1.103, 'peak_composite_emphasis': 1.4032, 'mean_f0_hz': 180.22, 'mean_energy': 0.9224, 'voicing_fraction': 0.522, 'window_count': 19, 'start_window_index': 0, 'end_window_index': 18}
    Explanation: Vocal pitch increases (+28.4%) while acoustic energy decreases (-7.8%) relative to the local 30s acoustic baseline (mean voicing fraction: 0.52).

  - [16.500s -> 26.500s] (dur: 10.00s, conf: 0.51)
    Type: vocal_emphasis | Anchor: window_33:51
    Window hop: 0.5000s | Sample Rate: 16000.0 Hz
    Features: {'f0_change': 1.0937, 'energy_change': 1.0468, 'composite_emphasis': 1.0703, 'peak_composite_emphasis': 1.2719, 'mean_f0_hz': 153.62, 'mean_energy': 1.0468, 'voicing_fraction': 0.6, 'window_count': 19, 'start_window_index': 33, 'end_window_index': 51}
    Explanation: Both vocal pitch (+9.4%) and acoustic energy (+4.7%) increase relative to the local 30s acoustic baseline (mean voicing fraction: 0.60).

  - [33.500s -> 43.500s] (dur: 10.00s, conf: 0.52)
    Type: vocal_emphasis | Anchor: window_67:85
    Window hop: 0.5000s | Sample Rate: 16000.0 Hz
    Features: {'f0_change': 1.1298, 'energy_change': 1.0182, 'composite_emphasis': 1.074, 'peak_composite_emphasis': 1.3697, 'mean_f0_hz': 163.84, 'mean_energy': 1.0182, 'voicing_fraction': 0.676, 'window_count': 19, 'start_window_index': 67, 'end_window_index': 85}
    Explanation: Both vocal pitch (+13.0%) and acoustic energy (+1.8%) increase relative to the local 30s acoustic baseline (mean voicing fraction: 0.68).

  - [51.000s -> 61.000s] (dur: 10.00s, conf: 0.50)
    Type: vocal_emphasis | Anchor: window_102:120
    Window hop: 0.5000s | Sample Rate: 16000.0 Hz
    Features: {'f0_change': 1.1191, 'energy_change': 0.9968, 'composite_emphasis': 1.0579, 'peak_composite_emphasis': 1.279, 'mean_f0_hz': 170.24, 'mean_energy': 0.9968, 'voicing_fraction': 0.621, 'window_count': 19, 'start_window_index': 102, 'end_window_index': 120}
    Explanation: Vocal pitch increases (+11.9%) with energy near baseline (-0.3%) relative to the local 30s acoustic baseline (mean voicing fraction: 0.62).

  - [70.000s -> 80.000s] (dur: 10.00s, conf: 0.45)
    Type: vocal_emphasis | Anchor: window_140:158
    Window hop: 0.5000s | Sample Rate: 16000.0 Hz
    Features: {'f0_change': 0.997, 'energy_change': 0.9976, 'composite_emphasis': 0.8394, 'peak_composite_emphasis': 1.273, 'mean_f0_hz': 156.29, 'mean_energy': 0.9976, 'voicing_fraction': 0.472, 'window_count': 19, 'start_window_index': 140, 'end_window_index': 158}
    Explanation: A transient vocal emphasis peak (peak composite: 1.27x) occurs in this interval, while span-averaged pitch (-0.3%) and energy (-0.2%) remain near the local 30s baseline (mean voicing fraction: 0.47).

Evidence artifacts successfully saved:
  - runs/representative_90s_validation/visual_evidence.json
  - runs/representative_90s_validation/prosody_evidence.json

[✓] ALL W4 EVIDENCE CHECKS PASSED (Visual & Prosody candidates verified)!
```

