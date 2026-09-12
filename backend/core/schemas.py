"""
ClipSense Core Schemas (W1)

Strict data contracts for:
1. Multimodal Extracted Features (Transcript, Visual, Prosody, Conversation)
2. Four Independent Evidence Experts (Proposals, Features, Evidence Bundles)
3. Common Evidence Bundle & Auxiliary Temporal Grid
4. MTER (Multimodal Temporal Evidence Reasoner) Interfaces
5. API Request/Response contracts

All timestamps are authoritative continuous floating-point seconds.
Expert outputs remain separate without premature scalar aggregation.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ============================================================================
# 1. Multimodal Data Extraction Schemas
# ============================================================================

class TimestampedWord(BaseModel):
    """Word-level timestamp produced by WhisperX alignment."""
    word: str
    start: float = Field(..., description="Continuous start timestamp in seconds")
    end: float = Field(..., description="Continuous end timestamp in seconds")
    score: Optional[float] = Field(None, description="Alignment confidence estimate if available")


class TranscriptSegment(BaseModel):
    """Utterance or sentence segment containing word-level alignments."""
    id: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = "SPEAKER_00"
    words: List[TimestampedWord] = Field(default_factory=list)


class TranscriptData(BaseModel):
    """Full multimodal extraction output for the spoken transcript."""
    full_text: str
    duration: float
    segments: List[TranscriptSegment] = Field(default_factory=list)
    words: List[TimestampedWord] = Field(default_factory=list)


class SceneBoundary(BaseModel):
    """Visual shot/scene change detected by scene detection."""
    scene_index: int
    start_time: float
    end_time: float
    duration: float


class SampledFrameMetadata(BaseModel):
    """Metadata for a video frame sampled for visual feature analysis."""
    frame_index: int
    timestamp: float = Field(..., description="Continuous timestamp of frame in seconds")
    scene_index: int
    file_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


class VisualData(BaseModel):
    """Full visual extraction output."""
    fps: float
    total_frames: int
    duration: float
    scene_boundaries: List[SceneBoundary] = Field(default_factory=list)
    sampled_frames: List[SampledFrameMetadata] = Field(default_factory=list)


class ProsodyWindow(BaseModel):
    """Acoustic prosody features computed over a continuous window."""
    start_time: float
    end_time: float
    duration: float
    mean_f0: float = Field(..., description="Mean fundamental frequency (pitch) in Hz")
    peak_f0: float = Field(..., description="Peak fundamental frequency in Hz")
    f0_std: float = Field(..., description="Standard deviation of pitch in Hz")
    rms_energy: float = Field(..., description="Root Mean Square energy / acoustic loudness")
    rms_dynamic_range: float = Field(..., description="Difference between max and min energy in window")
    voicing_fraction: float = Field(..., description="Fraction of voiced speech frames in window")
    speech_rate_wps: float = Field(..., description="Estimated words per second in window")


class ProsodyData(BaseModel):
    """Full acoustic prosody extraction output."""
    duration: float
    sample_rate: int
    window_length_sec: float
    hop_length_sec: float
    windows: List[ProsodyWindow] = Field(default_factory=list)


class SpeakerTurn(BaseModel):
    """Conversational speaker turn with duration and boundary context."""
    turn_index: int
    speaker: str
    start_time: float
    end_time: float
    duration: float
    word_count: int
    pause_before: float = Field(..., description="Silence duration before this turn starts in seconds")
    pause_after: float = Field(..., description="Silence duration after this turn ends in seconds")


class ConversationData(BaseModel):
    """Full conversational structure extraction output."""
    duration: float
    num_speakers: int
    turns: List[SpeakerTurn] = Field(default_factory=list)
    turn_frequency_per_minute: float
    total_speech_time: float
    total_pause_time: float


# ============================================================================
# 2. Expert Temporal Proposals & Independent Evidence Schemas
# ============================================================================

class SourceResolutionType(str, Enum):
    """Temporal source resolution backing a proposal."""
    WORD_TIMESTAMP = "word_timestamp"
    FRAME_TIMESTAMP = "frame_timestamp"
    PROSODY_WINDOW = "prosody_window"
    SPEAKER_TURN = "speaker_turn"
    SPEECH_SEGMENT = "speech_segment"
    SCENE_BOUNDARY = "scene_boundary"
    MIXED = "mixed"


class ProposalSourceMetadata(BaseModel):
    """
    Source metadata recording the temporal resolution, source sensor, 
    and alignment anchor of a proposal.
    """
    source_type: SourceResolutionType
    temporal_resolution_sec: float = Field(
        ..., 
        description="Temporal granularity of the raw measurement (e.g. word duration from WhisperX, prosody window hop)"
    )
    alignment_anchor: str = Field(
        ..., 
        description="Anchor reference (e.g., 'word_index_142:215', 'frame_450:900', 'turn_12:15')"
    )
    sample_rate_or_fps: Optional[float] = None


class TemporalProposal(BaseModel):
    """
    An independent temporal proposal generated by one of the four experts.
    Timestamps remain exact continuous floating-point seconds.
    """
    proposal_id: str
    start_time: float = Field(..., description="Authoritative continuous start timestamp in seconds")
    end_time: float = Field(..., description="Authoritative continuous end timestamp in seconds")
    duration: float = Field(..., description="Duration in seconds (end_time - start_time)")
    confidence_estimate: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Non-probabilistic model/heuristic confidence estimate (0.0 to 1.0)"
    )
    evidence_type: str = Field(
        ..., 
        description="Measurable category (e.g., 'semantic_topic_boundary', 'acoustic_emphasis_peak')"
    )
    explanation: str = Field(
        ..., 
        description="Human-interpretable explanation of the proposal rationale"
    )
    supporting_features: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Quantitative feature measurements backing this proposal"
    )
    source_metadata: ProposalSourceMetadata = Field(
        ..., 
        description="Exact source metadata documenting measurement resolution and anchor"
    )


class ExpertEvidenceBundle(BaseModel):
    """
    Isolated container for an individual expert's findings.
    Proposals are preserved independently without score fusion.
    """
    expert_name: Literal["transcript", "visual", "prosody", "conversation"]
    proposals: List[TemporalProposal] = Field(default_factory=list)
    dense_features_summary: Optional[Dict[str, Any]] = Field(
        None, 
        description="High-level statistical summary across the whole media (optional)"
    )


# ============================================================================
# 3. Common Evidence Bundle & Auxiliary Temporal Grid
# ============================================================================

class AuxiliaryGridCell(BaseModel):
    """
    Auxiliary 1-second discretized bucket used ONLY for indexed queries / lookup.
    Never replaces authoritative continuous floating-point timestamps.
    """
    bin_index: int
    bin_start: float
    bin_end: float
    active_proposal_ids: Dict[str, List[str]] = Field(
        default_factory=dict, 
        description="Map of expert_name -> list of proposal_ids spanning this bin"
    )


class AuxiliaryTemporalGrid(BaseModel):
    """Auxiliary temporal index spanning the video duration."""
    bin_size_sec: float = 1.0
    total_bins: int
    cells: List[AuxiliaryGridCell] = Field(default_factory=list)


class CommonEvidenceBundle(BaseModel):
    """
    Unified container aggregating the four independent evidence bundles.
    Maintains complete separation between experts.
    """
    video_id: str
    total_duration: float
    transcript_evidence: ExpertEvidenceBundle
    visual_evidence: ExpertEvidenceBundle
    prosody_evidence: ExpertEvidenceBundle
    conversation_evidence: ExpertEvidenceBundle
    auxiliary_grid: Optional[AuxiliaryTemporalGrid] = None


# ============================================================================
# 4. MTER (Multimodal Temporal Evidence Reasoner) Schemas
# ============================================================================

class BoundaryProposalRef(BaseModel):
    """Reference to a contributing proposal at a specific temporal boundary."""
    expert_name: str
    proposal_id: str
    timestamp: float
    confidence_estimate: float
    evidence_type: str


class BoundaryCluster(BaseModel):
    """
    Cluster of nearby boundary candidates from one or more experts.
    Preserves member proposals and separate modality identities without collapsing confidence.
    """
    cluster_id: str
    boundary_type: Literal["start", "end"]
    cluster_center: float = Field(..., description="Continuous floating-point center timestamp in seconds")
    min_time: float
    max_time: float
    spread_sec: float
    supporting_experts: List[str]
    member_proposals: List[BoundaryProposalRef] = Field(default_factory=list)


class TemporalConflict(BaseModel):
    """
    Explicitly recorded cross-modal boundary disagreement or span discrepancy.
    """
    event_region_id: str = Field(..., description="ID of the event region where conflict was detected")
    boundary_type: Literal["start", "end", "interval_breadth"]
    conflict_type: str = Field(
        ..., 
        description="Type of conflict, e.g. 'early_start_disagreement', 'late_end_disagreement', 'broad_vs_narrow_interval'"
    )
    early_proposals: List[BoundaryProposalRef] = Field(default_factory=list)
    late_proposals: List[BoundaryProposalRef] = Field(default_factory=list)
    temporal_delta_sec: float
    description: str


class EventRegion(BaseModel):
    """
    Coherent temporal region formed by temporally overlapping / compatible proposals across modalities.
    Prevents cross-pairing boundaries of unrelated events.
    """
    region_id: str
    start_time: float
    end_time: float
    member_proposal_ids: List[str] = Field(default_factory=list)
    contributing_experts: List[str] = Field(default_factory=list)


class CandidateSupportMetrics(BaseModel):
    """
    Explicit per-modality support and multi-factor evaluation metrics for a candidate interval.
    Preserves distinct modality contributions without premature uninspectable scalar collapse.
    """
    transcript_support: float = 0.0
    conversation_support: float = 0.0
    visual_support: float = 0.0
    prosody_support: float = 0.0

    # Explicit per-modality temporal overlap & coverage
    transcript_iou: float = 0.0
    conversation_iou: float = 0.0
    visual_iou: float = 0.0
    prosody_iou: float = 0.0
    transcript_coverage: float = 0.0
    conversation_coverage: float = 0.0
    visual_coverage: float = 0.0
    prosody_coverage: float = 0.0

    # Separated boundary evidence types
    boundary_activity_support: float = Field(
        0.0, 
        description="Evidence that physical acoustic/visual activity or scene transitions occur near boundary"
    )
    linguistic_boundary_support: float = Field(
        0.0, 
        description="Evidence that boundary coincides with a spoken word timestamp or speech segment onset/offset (linguistic/temporal alignment only, not independent semantic understanding)"
    )

    # Modality presence vs support strength distinction
    modalities_present: List[str] = Field(default_factory=list, description="Modalities with coverage > 0")
    strong_support_modalities: List[str] = Field(default_factory=list, description="Modalities with support >= 0.50")
    moderate_support_modalities: List[str] = Field(default_factory=list, description="Modalities with 0.20 <= support < 0.50")
    weak_support_modalities: List[str] = Field(default_factory=list, description="Modalities with 0.05 <= support < 0.20")

    modality_diversity: float = 0.0
    boundary_agreement: float = 0.0
    inherited_contextual_evidence: float = Field(
        0.0, 
        description="Normalized score of contextual completeness evidence inherited from W3 proposals"
    )
    conflict_penalty: float = 0.0
    duration_penalty: float = 0.0
    composite_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Composite evidence strength score explicitly bounded to [0.0, 1.0] via max(0.0, min(1.0, raw_composite)) for all evaluated candidates"
    )


class SemanticVerificationCriteria(BaseModel):
    """
    Checklist of contextual completeness evidence inherited from contributing expert proposals
    (not an independent verification).
    """
    natural_semantic_start: bool = Field(
        ..., 
        description="True if start aligns near a spoken word or speech segment onset"
    )
    sufficient_context: bool = Field(
        ..., 
        description="True if overlapping transcript proposal has contextual_completeness >= threshold and self_contained == True"
    )
    important_content_retained: bool = Field(
        ..., 
        description="True if key thematic or conversational content from the region is within the candidate"
    )
    complete_conversational_payoff: bool = Field(
        ..., 
        description="True if discourse_unit_complete or payoff phase is completed"
    )
    no_mid_sentence_ending: bool = Field(
        ..., 
        description="True if candidate avoids cutting off mid-word or mid-utterance"
    )
    inherited_from_proposals: List[str] = Field(
        default_factory=list,
        description="List of proposal IDs from which these contextual features were extracted"
    )


class MTERCandidate(BaseModel):
    """
    A temporal region selected by MTER via multi-expert agreement analysis.
    """
    candidate_id: str
    proposed_start: float = Field(..., description="Continuous start timestamp in seconds")
    proposed_end: float = Field(..., description="Continuous end timestamp in seconds")
    duration: float
    confidence_estimate: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Normalized evidence strength estimate (0.0 to 1.0)"
    )
    contributing_experts: List[Literal["transcript", "visual", "prosody", "conversation"]]
    temporal_agreement_iou: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Mean Intersection-over-Union across overlapping active expert proposals"
    )
    modality_ious: Dict[str, float] = Field(default_factory=dict)
    modality_coverages: Dict[str, float] = Field(default_factory=dict)
    boundary_activity_support: float = 0.0
    linguistic_boundary_support: float = 0.0

    # Modality presence vs support strength
    modalities_present: List[str] = Field(default_factory=list)
    strong_support_modalities: List[str] = Field(default_factory=list)
    moderate_support_modalities: List[str] = Field(default_factory=list)
    weak_support_modalities: List[str] = Field(default_factory=list)

    support_metrics: Optional[CandidateSupportMetrics] = None
    conflict_notes: List[str] = Field(
        default_factory=list, 
        description="Notes on cross-modal disagreements affecting this candidate"
    )
    detected_conflicts_count: int = 0
    conflicts_affecting_candidate_count: int = 0
    unresolved_conflicts_count: int = 0
    semantic_verification: SemanticVerificationCriteria
    semantic_verification_status: Literal["passed", "flagged", "pending"]
    decision_summary: str = Field(
        ..., 
        description="Concise evidence summary explaining the rationale for selecting this candidate"
    )


class EvidenceLedger(BaseModel):
    """
    Inspectable evidence ledger recording all intermediate reasoning, clustering,
    conflicts, rejected candidates, and the selected highlight boundary.
    """
    run_id: str
    video_duration: float
    total_proposals: int
    event_regions: List[EventRegion] = Field(default_factory=list)
    start_clusters: List[BoundaryCluster] = Field(default_factory=list)
    end_clusters: List[BoundaryCluster] = Field(default_factory=list)
    conflicts: List[TemporalConflict] = Field(default_factory=list)
    total_detected_conflicts: int = 0
    conflicts_affecting_candidate: List[TemporalConflict] = Field(default_factory=list)
    unresolved_conflicts_affecting_candidate: List[TemporalConflict] = Field(default_factory=list)
    rejected_intervals: List[Dict[str, Any]] = Field(default_factory=list)
    evaluated_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    selected_candidate: Optional[MTERCandidate] = None
    decision_summary: str = ""


class MTEROutput(BaseModel):
    """Final output of MTER containing selected candidates and reasoning ledger."""
    video_id: str
    candidates: List[MTERCandidate] = Field(default_factory=list)
    ledger: Optional[EvidenceLedger] = None
    execution_time_sec: float


# ============================================================================
# 5. API Request/Response Schemas (Frontend Compatibility)
# ============================================================================

class ProcessVideoRequest(BaseModel):
    """Incoming request payload from Next.js / Inngest workflow."""
    s3_key: str
    mode: str = "auto"  # "auto" or "manual"
    requested_clips: Optional[int] = None
    user_credits: int
    user_prompt: Optional[str] = None


class ClipMetadata(BaseModel):
    """Metadata per generated clip returned to frontend."""
    s3_key: str
    title: str
    description: str


class ProcessVideoResponse(BaseModel):
    """Final response schema expected by Next.js / Inngest workflow."""
    clips_processed: int
    duration: float
    clip_metadata: List[ClipMetadata] = Field(default_factory=list)
