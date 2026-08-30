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
```



