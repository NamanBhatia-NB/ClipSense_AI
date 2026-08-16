# Implementation Plan - ClipSense Architecture & W1: Modular Architecture and Schemas

## Architecture Overview

ClipSense is a multimodal video clipping system designed around four independent evidence experts, continuous floating-point temporal reasoning, and explicit code-level evidence arbitration:

```
Input Video
    ↓
Multimodal Data Extraction
    ├── Timestamped Transcript (WhisperX word alignments)
    ├── Video Frames (Scene cuts + sampled frames)
    ├── Audio Prosody (F0 pitch + RMS dynamics via Librosa/SciPy)
    └── Speaker / Conversation Information (Turn structure & pause timeline)
    ↓
Four Independent Evidence Experts
    ├── Transcript Expert (Semantic importance & contextual completeness)
    ├── Visual Expert (Visual activity, shot pacing, pluggable representations)
    ├── Prosody Expert (Acoustic emphasis, dynamic pitch variation)
    └── Conversation Expert (Conversational structure, turn exchanges)
    ↓
Structured Temporal Evidence (Preserved independently; NO score collapse)
    ↓
Evidence Aggregation & Auxiliary Normalization
    (Continuous timestamps preserved; auxiliary temporal grid for indexing)
    ↓
Multimodal Temporal Evidence Reasoner (MTER)
    (Programmatic overlap/agreement, conflict analysis, semantic verification)
    ↓
Final Temporal Boundaries
    ↓
Temporal Boundary Refinement / Rule-Based Boundary Optimization
    (Fine-grained adjustments, pause snapping, dead-air trimming)
    ↓
Downstream Video Render (LR-ASD active speaker, 9:16 reframe, ASS subtitles, FFmpeg)
    ↓
Final Short Clip
```

## Alignment on Architectural Constraints

1. **Single Repository**: Single ClipSense codebase; no baseline/research fragmentation.
2. **Four Independent Experts**: Transcript, Visual, Prosody, and Conversation.
3. **No Premature Score Collapse**: Expert outputs are kept as independent structured temporal proposals. No scalar aggregation before MTER.
4. **Authoritative Continuous Timestamps**: Floating-point timestamps (`float` in seconds) are strictly preserved. The 1-second grid is strictly an auxiliary view.
5. **Confidence Estimates**: Terminology uses "confidence estimate" rather than calibrated probabilities.
6. **Configurable Baselines & Thresholds**: No hard-coded research claims (e.g. pitch z-score > 2.0); all thresholds reside in `app/config.py`.
7. **Overlapping Proposals Allowed**: Proposals from multiple experts can and should overlap for agreement analysis.
8. **Boundary Refinement**: Termed "Temporal Boundary Refinement" or "Rule-based Boundary Optimization" (never "neurosymbolic").
9. **Semantic Completeness Criteria in MTER**: Natural semantic start, sufficient context, key content retained, conversational payoff, no mid-sentence ending.
10. **Role Separation**: MTER selects supported temporal regions; Boundary Optimizer performs fine-grained boundary adjustments.
11. **Programmatic MTER Pipeline**: Overlap/agreement, conflict detection, proposal reconciliation, evidence-strength comparison, semantic verification, boundary selection.
12. **W1–W5 Scope**:
    - W1: Architecture + schemas
    - W2: Multimodal extraction
    - W3: Transcript + Conversation Experts
    - W4: Visual + Prosody Experts
    - W5: Common evidence bundle + timestamp normalization + MTER interface + deterministic prototype
13. **Honest Scope**: No claims of full MTER implementation or performance improvements during W1–W5.
14. **No PyAnnote**: Diarization and turn transitions derived from WhisperX and acoustic silences.
15. **Pluggable Visual Encoder**: Abstract interface allowing interchangeable visual representations.
16. **No Unsupported Concepts**: Avoid claims of virality prediction, emotion recognition, or viewer retention.
17. **Source Metadata**: `TemporalProposal` carries `ProposalSourceMetadata` recording source type, temporal resolution, and anchor.
18. **W1 Implementation First**: Implement only W1 now.

---

## W1 Deliverables

### `ai-podcast-clipper-backend/`

- `core/schemas.py`: Pydantic contracts for extraction, 4 independent experts, proposals, continuous timestamps, evidence bundles, and MTER criteria.
- `app/config.py`: Centralized configurable thresholds and baseline parameters.
- `core/storage.py`: Storage and ephemeral workspace management.
- `pipeline/experts/base.py`: Base expert contract and pluggable visual encoder protocol.
- `pipeline/orchestrator.py`: Pipeline orchestrator stub with invariant validation.
- `tests/test_w1_architecture.py`: Automated verification suite for W1 contracts.
