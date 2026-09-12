# ClipSense Weekly Milestones & Research Progress

## W1 — 10 Aug – 16 Aug 2026
**Milestone: Architecture and Schemas**

### Research & Architecture Accomplishments:
- Established modular backend package architecture (`app/`, `core/`, `pipeline/`, `tests/`).
- Designed strict Pydantic data contracts for multimodal extraction (WhisperX word alignments, scene cuts, audio prosody dynamics, speaker turn structures).
- Defined independent expert proposal representations across four distinct domains: Transcript, Visual, Prosody, and Conversation.
- Preserved continuous floating-point timestamps and strictly eliminated premature score collapse.
- Implemented `ProposalSourceMetadata` tracking source resolution, sensor type, and alignment anchors.
- Added configurable baselines in `app/config.py` for acoustic peak detection and dialogue turn segmentation.
- Specified MTER candidate criteria with explicit semantic verification checklist (natural start, sufficient context, content retained, conversational payoff, complete sentence closure).
- Built W1 invariant contract verification suite in `tests/test_w1_architecture.py`.

### Supporting Engineering:
- Aligned API schemas with the frontend Inngest interface (`ProcessVideoRequest`, `ProcessVideoResponse`, `ClipMetadata`).

## W2 — 17 Aug – 23 Aug 2026
**Milestone: Multimodal Data Extraction Pipeline**

### Scope & Architectural Boundary:
- **W2 Scope**: Confined strictly to **Multimodal Data Extraction** (timestamped transcript extraction, visual data extraction, audio prosody extraction, and conversation structure extraction).
- **Explicit Stage Separation**:
  - `W2`: Multimodal Data Extraction (raw and derived acoustic, linguistic, structural, and frame data).
  - `W3 / W4`: Modality-Specific Expert Reasoning (Transcript Expert & Conversation Expert in W3; Visual Expert & Prosody Expert in W4).
  - `W5`: Evidence Normalization, Common Evidence Bundle, and MTER Prototype.
- **Explicit Boundary Adherence**: No expert reasoning, highlight classification, candidate scoring, virality/emotion prediction, multimodal fusion, MTER reasoning, or boundary optimization were implemented in W2.

### Research & Extraction Accomplishments:
- The W2 multimodal data-extraction stage was implemented and validated on a representative conversational sample. The pipeline generates timestamp-aligned transcript, visual-frame, acoustic-prosodic and conversation-structure artifacts. Higher-level modality-specific expert reasoning and MTER are intentionally deferred to subsequent stages.
- **Transcript Extraction** (`transcript_extractor.py`):
  - Decoupled WhisperX transcription and wav2vec2 forced alignment producing continuous floating-point word-level timestamps.
  - Added STRICT mode requiring real model execution and failing immediately if model inference cannot run. Fallback baseline generation is isolated to infrastructure/unit testing only.
  - Model used for representative validation: `tiny.en` (recorded as `"whisper_model": "tiny.en"` in run manifests). The extractor remains configurable to support larger models in subsequent evaluation runs without code changes; no claims are made regarding large-v2 performance.
  - Speaker metadata is preserved where available. (The 90s validation clip contains one detected speaker `SPEAKER_00`; multi-speaker diarization is preserved when present in source alignments).
- **Visual Data Extraction** (`frame_extractor.py`):
  - Uniform frame sampling at configurable FPS (default 1.0 fps) and lightweight scene boundary detection (`scenedetect.detect` ContentDetector) preserving continuous frame timestamps without integer second truncation.
- **Audio Prosody Extraction** (`prosody_extractor.py`):
  - Pure physical acoustic measurements: autocorrelation F0 fundamental frequency, RMS loudness energy, dynamic range, voicing fraction, and speaking rate over continuous sliding windows.
  - Zero highlight detection or emotion recognition; no arbitrary highlight thresholds.
- **Conversation Structure Extraction** (`conversation_extractor.py`):
  - Measurable conversational structure: distinguishes multi-speaker conversational turn-taking from single-speaker pause-separated speech segments (>0.7s silence thresholds), while preserving speaker IDs, pause durations, and switch indicators for multi-speaker turn-taking detection.
  - Qualitative discourse or semantic interpretation is deferred to the W3 Conversation Expert.
- **Pipeline Coordinator & Artifact Manifest** (`extractor_pipeline.py`):
  - Coordinates extraction sequence, reports standard status indicators (`[✓] Transcript extraction`, `[✓] Visual data extraction`, `[✓] Prosody extraction`, `[✓] Conversation structure extraction`), and outputs reproducible artifacts (`transcript.json`, `frames/`, `frames.json`, `prosody.json`, `conversation.json`, `manifest.json`).
  - Extended run manifest with execution configuration metadata: `whisper_model`, `strict_mode`, `fallback_used`, `frame_sampling_fps`, `prosody_window_sec`, `prosody_hop_sec`, `conversation_pause_threshold_sec`, `source_video_duration_sec`.
  - Cache validation: validates existing manifest against current video and configuration; automatically invalidates cache when parameters or input media change.
- **Automated Verification Suite**:
  - Automated timestamp validation tests verifying finite, ordered, continuous floating-point timestamps strictly bounded within video duration across all modalities.
  - Automated caching tests confirming cache reuse on identical runs and automatic cache invalidation when configuration parameters change (`tests/test_w2_caching_and_timestamps.py`).
  - Strict-mode representative validation on 90.0-second conversational sample (`tests/sample_conversational_90s.mp4`) with real WhisperX inference.

### Supporting Engineering:
- Video ingestion, PyTorch 2.6+ checkpoint compatibility patch, and artifact caching lifecycle.

## W3 — 24 Aug – 30 Aug 2026
**Milestone: Independent Modality-Specific Temporal Evidence Generation (Transcript Expert + Conversation Expert)**

### Scope & Architectural Boundary:
- **W3 Scope**: Development of independent modality-specific temporal evidence experts (**Transcript Expert** and **Conversation Expert**) on top of W2 extraction artifacts.
- **Explicit Stage Separation**:
  - W3 is strictly **independent modality-specific temporal evidence generation**, NOT final highlight selection, clipping, or viral classification.
  - The experts generate independent candidate proposals packaged in `ExpertEvidenceBundle` and output to dedicated artifacts (`transcript_evidence.json`, `conversation_evidence.json`).
  - Cross-expert aggregation, MTER reasoning, multimodal conflict resolution, score fusion, boundary optimization, and final clip selection are intentionally deferred to subsequent stages (W4/W5).

### Research & Reasoning Accomplishments:
- **LLM Abstraction (`core/llm.py`)**:
  - Built a provider-agnostic `LLMClient` interface decoupling expert logic from proprietary LLM implementations.
  - Implemented `GeminiLLMClient` using the Google GenAI SDK (`gemini-2.5-flash` by default) with native Pydantic structured output, exponential backoff for transient errors, timeout handling, zero hidden chain-of-thought (`thinking_budget=0`), and strict mode failure semantics.
  - Implemented `MockLLMClient` for deterministic offline testing.
- **Transcript Expert (`pipeline/experts/transcript_expert.py`)**:
  - Evaluates timestamped spoken transcripts across bounded analysis windows for semantic and linguistic evidence (semantic importance, core claims, explanations, and conceptual self-containment).
  - Configurable duration limits (`min_candidate_duration_sec = 15.0s`, `max_candidate_duration_sec = 45.0s`).
  - Grounding: analyzes indexed word representations (`[i] word`) and programmatically maps proposed `start_word_index` and `end_word_index` to exact WhisperX continuous floating-point timestamps (`SourceResolutionType.WORD_TIMESTAMP`).
- **Conversation Expert (`pipeline/experts/conversation_expert.py`)**:
  - Evaluates conversational structure and discourse dynamics: coherent localized conversational sub-intervals (setup -> development -> payoff), pause boundaries, and speech segment pacing.
  - Source & Anchor Consistency:
    - Proposals aligned with speech segment boundaries report `SourceResolutionType.SPEECH_SEGMENT` with anchor `speech_segment_X`.
    - Proposals targeting sub-segment word intervals report `SourceResolutionType.WORD_TIMESTAMP` with anchor `word_anchored_X:Y`.
  - Single-speaker safety: handles monologues with `discourse_unit_complete = True/False`, strictly enforcing `speaker_interaction = False` and `is_single_speaker_monologue = True` without inventing speaker exchanges.
- **Verification & Demonstration**:
  - Automated unit test suites (`tests/test_transcript_expert.py` and `tests/test_conversation_expert.py`) covering schema validation, bounds checks, error rejection, deterministic word grounding, source/anchor consistency, artifact provenance, and prompt text injection.
  - Reproducible validation runner `tests/run_w3_expert_validation.py` executed live against the representative 90s sample in strict mode generating localized candidate evidence artifacts.

### Supporting Engineering:
- Implemented frontend authentication foundation, session management, secure credential validation, and legal routing (`src/server/auth/`, `src/actions/auth.ts`, login/signup/terms views).

## W4 — 31 Aug – 06 Sep 2026
**Milestone: Independent visual and prosodic temporal evidence generation**

### Scope & Architectural Boundary:
- **W4 Scope**: Independent visual and prosodic temporal evidence generation from the W2 multimodal extraction artifacts.
- **Strict Modality Independence**:
  - The Visual Expert consumes ONLY visual extraction data (`frames.json`, sampled frames, scene boundaries, duration). It does not ingest transcript, conversation, or prosody data.
  - The Prosody Expert consumes ONLY acoustic prosody extraction data (`prosody.json`, windowed physical acoustic measurements, duration). It does not ingest transcript, conversation, or visual data.
  - Zero cross-expert reasoning, score fusion, or aggregation.
- **Explicit Boundary Adherence**:
  - No cross-expert aggregation, evidence normalization across experts, agreement analysis, conflict resolution, MTER, boundary optimization, final clip selection, benchmark superiority, viewer retention, emotion recognition, gesture understanding, or semantic visual claims were implemented.
  - The `confidence_estimate` field across all proposals is strictly documented and treated as a normalized evidence-strength estimate, not a calibrated probability.

### Research & Reasoning Accomplishments:
- **Visual Feature Extractor Protocol & Default Extractor (`pipeline/experts/base.py`, `pipeline/experts/visual_expert.py`)**:
  - Established `VisualFeatureExtractorInterface` protocol for pluggable visual encoders.
  - Implemented `LightweightFrameDiffExtractor`: downsamples frames to normalized 32x32 intensity representations and computes normalized L1 distance, providing fast, deterministic, reproducible visual change metrics without committing prematurely to large vision models or loading models at import time.
- **Visual Expert (`pipeline/experts/visual_expert.py`)**:
  - Confined strictly to visual-activity, visual-change, and scene-transition evidence generation (no semantic visual understanding or gesture recognition).
  - Evaluates inter-frame visual change, scene boundary transition density, and continuous visual activity curves smoothed over configurable temporal windows (`activity_smoothing_sec = 3.0s`).
  - Identifies localized candidate visual intervals via percentile thresholding (`min_activity_percentile = 65.0%`) and transition clustering (`merge_gap_sec = 3.0s`).
  - Temporal grounding: boundaries strictly correspond to sampled frame timestamps (`SourceResolutionType.FRAME_TIMESTAMP`) with explicit anchors distinguishing sampled-frame indices from native video frame indices (`sampled_frame_S:E (native_frame_NS:NE)`).
  - Explicitly distinguishes native video FPS from sampled-frame interval (effective temporal evidence resolution: 1.0s).
  - Configurable candidate duration bounds (`min_candidate_duration_sec = 10.0s`, `max_candidate_duration_sec = 45.0s`).
  - Strictly rejects whole-input candidate spans ($> 90\%$ duration).
- **Prosody Expert (`pipeline/experts/prosody_expert.py`)**:
  - Evaluates physical acoustic features (F0 pitch, RMS energy, voicing fraction).
  - No LLM for raw acoustic measurements; no direct classification of acoustic dynamics as emotion.
  - Filters unvoiced / silent windows (`voicing_fraction < min_voicing_fraction`) from baseline calculation to avoid baseline corruption.
  - Computes moving local acoustic baseline (median voiced F0 and RMS energy over configurable `local_baseline_window_sec = 30.0s`).
  - Detects vocal emphasis based on composite pitch and loudness elevation relative to the local baseline.
  - Generates faithful explanations dynamically from actual measured feature values (distinguishing pitch rise vs. energy fall vs. transient emphasis peaks).
  - Merges nearby emphasis windows (`merge_gap_sec = 3.0s`) and bounds candidates between `min_candidate_duration_sec = 10.0s` and `max_candidate_duration_sec = 45.0s`.
  - Temporal grounding: boundaries strictly correspond to acoustic window boundaries (`SourceResolutionType.PROSODY_WINDOW`) with anchor `window_start:end` and resolution reporting window hop (0.5s) separately from sample rate (16000 Hz).
- **Verification & Demonstration**:
  - Comprehensive unit test suites (`tests/test_visual_expert.py` and `tests/test_prosody_expert.py`) verifying schema conformity, finite continuous floating-point timestamps ($0.0 \le \text{start} < \text{end} \le \text{duration}$), duration consistency, source metadata correctness, deterministic output, empty/insufficient input handling, and silent region filtering.
  - Complete regression suite passing: W1 architecture, W2 extractors, W2 caching/timestamps, W3 Transcript Expert, W3 Conversation Expert, W4 Visual Expert, and W4 Prosody Expert.
  - Validation runner `tests/run_w4_expert_validation.py` executed on the representative 90s validation run (`runs/representative_90s_validation/`), producing structured evidence artifacts `visual_evidence.json` and `prosody_evidence.json`.

### Supporting Engineering:
- Implemented frontend dashboard interface, client workspace, clip display component, Polar billing integration, S3 media upload actions, and Inngest background job processing pipelines.

## W5 — 07 Sep – 13 Sep 2026
**Milestone: Integration of independent temporal evidence from the four modality-specific experts and development of an initial Multimodal Temporal Evidence Reasoner (MTER) for explicit boundary agreement and conflict analysis**

### Scope & Architectural Boundary:
- **W5 Scope**: Connecting independent temporal evidence streams from the four modality-specific experts (Transcript, Conversation, Visual, Prosody) into a unified `CommonEvidenceBundle` and implementing an initial programmatic Multimodal Temporal Evidence Reasoner (MTER) prototype.
- **Strict Evidence Preservation**:
  - Maintained total separation of the four expert evidence bundles without premature scalar score averaging or confidence collapse.
  - Authoritative timestamps strictly preserved as continuous floating-point seconds.
  - Auxiliary temporal grid (1.0s bins) implemented solely as an indexing/lookup aid without overwriting continuous coordinates.
- **Explicit Boundary Adherence**:
  - No fine-grained boundary search (±1s/±2s local audio search), subtitle-aware trimming, dead-air trimming, final clip rendering, final highlight ranking, benchmark superiority, or viewer retention claims were made or implemented.
  - MTER operates as a fully deterministic, programmatic reasoner without an opaque LLM making boundary decisions.
  - Intermediate evidence remains completely inspectable via an `EvidenceLedger` to facilitate future ablation experiments comparing individual modalities and reasoner configurations.

### Research & Reasoning Accomplishments:
- **Evidence Bundle & Temporal Normalization (`pipeline/mter/evidence_bundle.py`)**:
  - Implemented `build_common_evidence_bundle` and `load_evidence_bundle_from_run` ingesting `transcript_evidence.json`, `conversation_evidence.json`, `visual_evidence.json`, and `prosody_evidence.json`.
  - Built `AuxiliaryTemporalGrid` for temporal binning while preserving continuous float boundaries.
- **Event-Region Grouping & Temporal Compatibility (`pipeline/mter/clustering.py`)**:
  - Implemented graph connected components with temporal compatibility rules (`event_compatibility_min_iou: 0.15`, `event_compatibility_min_overlap_ratio: 0.45`, `event_compatibility_max_center_gap_sec: 20.0s`).
  - Fixed proposal daisy-chaining: broad proposals no longer collapse all 14 input proposals into one giant region. Instead, 3 coherent event regions are formed (Region 0: 0.0s–27.27s, Region 1: 24.57s–71.56s, Region 2: 68.27s–89.87s).
- **Intra-Region Boundary Clustering & Conflict Scoping (`pipeline/mter/clustering.py`)**:
  - Boundary clustering runs strictly per event region with `boundary_cluster_tolerance_sec = 3.0s`.
  - Conflict detection is strictly scoped per event region: every conflict records `event_region_id`.
  - Candidate conflict filtering guarantees candidates in Region 0 never report conflicts against distant proposals (e.g. at 89.865s in Region 2).
  - Proposal reference integrity verified: proposal IDs and timestamps resolve directly to the referenced proposal object (e.g. `transcript_prop_2` [the 3rd proposal] with timestamp 89.865s).
- **MTER Reasoner Prototype & Metrics Clarification (`pipeline/mter/reasoner.py`)**:
  - Implemented a deterministic, internally consistent MTER prototype with inspectable evidence and configurable heuristic parameters.
  - Documented that all heuristic weights and thresholds in `MTERConfig` are configurable prototype parameters, not learned weights or empirically validated research constants.
  - Explicitly bounded `CompositeScore` to $[0.0, 1.0]$ via $\max(0.0, \min(1.0, \text{RawCompositeScore}(I)))$ for ALL evaluated candidates, justifying "Normalized Evidence Strength" as a normalized relative consensus metric rather than a calibrated probability.
  - Renamed `semantic_boundary_support` to `linguistic_boundary_support`, documented strictly as word timestamp and speech segment onset/offset alignment evidence rather than independent semantic verification.
  - Separated modality presence (`modalities_present`) from support strength tiers (`strong_support_modalities` $\ge 0.50$, `moderate_support_modalities` $0.20 \le s < 0.50$, `weak_support_modalities` $0.05 \le s < 0.20$).
  - Preserved transparent, inspectable decision summary and full `EvidenceLedger`.
- **Diagnostic Result Preservation**:
  - Retained the current selected candidate `[0.030s -> 26.106s]` without artificial weight tuning to force a visually or semantically preferred answer.
  - Recorded it as a representative diagnostic result demonstrating cross-modal boundary disagreement (early acoustic/visual activity convergence at 0.030s vs. delayed spoken sentence onset at 5.975s).
- **Verification & Demonstration**:
  - Comprehensive unit test suites (`tests/test_evidence_bundle.py` and `tests/test_mter.py`) verifying non-chaining event grouping, candidate-scoped conflict isolation, proposal reference integrity, explicit bounding of composite scores to $[0.0, 1.0]$ across all candidates, modality presence vs. strength separation, modality separation, continuous timestamps, auxiliary grid lookup, cross-event pairing prevention, unimodal candidate preservation, multimodal candidate preference, determinism, and rejection logging.
  - Complete regression suite passing: 46 tests across W1, W2, W3, W4, and W5.
  - Representative validation runner `tests/run_w5_mter_validation.py` executed on `runs/representative_90s_validation/`, generating structured inspectable artifacts `mter_output.json` and `mter_ledger.json`.
