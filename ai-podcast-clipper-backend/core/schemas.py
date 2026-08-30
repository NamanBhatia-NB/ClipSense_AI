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

class SemanticVerificationCriteria(BaseModel):
    """
    Explicit checklist verifying contextual completeness and semantic integrity.
    """
    natural_semantic_start: bool = Field(
        ..., 
        description="True if the segment starts at a natural thought/clause onset"
    )
    sufficient_context: bool = Field(
        ..., 
        description="True if the premise or setup is understandable without preceding context"
    )
    important_content_retained: bool = Field(
        ..., 
        description="True if key thematic or conversational content is preserved"
    )
    complete_conversational_payoff: bool = Field(
        ..., 
        description="True if the resolution, punchline, or core insight is completed"
    )
    no_mid_sentence_ending: bool = Field(
        ..., 
        description="True if the segment avoids cutting off mid-sentence or mid-thought"
    )


class MTERCandidate(BaseModel):
    """
    A temporal region selected by MTER via multi-expert agreement analysis.
    """
    candidate_id: str
    proposed_start: float = Field(..., description="Continuous start timestamp in seconds")
    proposed_end: float = Field(..., description="Continuous end timestamp in seconds")
    duration: float
    contributing_experts: List[Literal["transcript", "visual", "prosody", "conversation"]]
    temporal_agreement_iou: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Intersection-over-Union across overlapping contributing expert proposals"
    )
    conflict_notes: List[str] = Field(
        default_factory=list, 
        description="Notes on cross-modal disagreements or discrepancies"
    )
    semantic_verification: SemanticVerificationCriteria
    semantic_verification_status: Literal["passed", "flagged", "pending"]
    reasoning_trace: str = Field(
        ..., 
        description="Programmatic trace documenting how this candidate was selected"
    )


class MTEROutput(BaseModel):
    """Final output of MTER containing selected candidates prior to boundary optimization."""
    video_id: str
    candidates: List[MTERCandidate] = Field(default_factory=list)
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
