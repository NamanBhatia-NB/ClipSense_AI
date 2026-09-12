"""
ClipSense Application Configuration

Central configuration module holding:
- Prosody baseline and peak detection parameters
- Visual scene detection and motion sensitivity thresholds
- Conversation turn pause thresholds
- Candidate duration constraints
- S3 and API credentials
"""

import os
from typing import List, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class ProsodyConfig(BaseModel):
    """Configurable thresholds for acoustic prosody feature extraction & expert."""
    window_length_sec: float = Field(1.0, description="Sliding window size in seconds")
    hop_length_sec: float = Field(0.5, description="Window hop/step size in seconds")
    sample_rate: int = Field(16000, description="Audio resample target rate in Hz")
    f0_min_hz: float = Field(65.0, description="Minimum human voice fundamental frequency in Hz")
    f0_max_hz: float = Field(400.0, description="Maximum human voice fundamental frequency in Hz")
    local_baseline_window_sec: float = Field(
        30.0, 
        description="Duration of local window over which moving baseline stats are computed"
    )
    peak_f0_multiplier: float = Field(
        1.5, 
        description="Factor above local baseline median to flag an acoustic pitch emphasis peak"
    )
    peak_energy_multiplier: float = Field(
        1.5, 
        description="Factor above local baseline median RMS energy to flag volume emphasis"
    )
    min_voicing_fraction: float = Field(
        0.3, 
        description="Minimum voiced speech fraction required to evaluate prosody window"
    )
    min_candidate_duration_sec: float = Field(
        10.0, 
        description="Minimum localized candidate duration in seconds for prosody proposals"
    )
    max_candidate_duration_sec: float = Field(
        45.0, 
        description="Maximum localized candidate duration in seconds for prosody proposals"
    )
    merge_gap_sec: float = Field(
        3.0, 
        description="Maximum silence/gap duration in seconds across which nearby emphasis windows are clustered"
    )
    emphasis_f0_weight: float = Field(0.5, description="Weight for pitch elevation in composite emphasis score")
    emphasis_energy_weight: float = Field(0.5, description="Weight for RMS energy elevation in composite emphasis score")
    emphasis_threshold_multiplier: float = Field(
        1.25, 
        description="Relative elevation factor above local baseline median to flag an emphasis window"
    )


class VisualConfig(BaseModel):
    """Configurable thresholds for visual extraction & expert."""
    scene_threshold: float = Field(27.0, description="ContentDetector threshold in PySceneDetect")
    sample_fps: float = Field(1.0, description="Uniform sampling rate for frames in fps")
    motion_change_threshold: float = Field(
        0.25, 
        description="Relative visual frame difference threshold to flag visual shift"
    )
    min_face_detection_confidence: float = Field(0.5, description="Confidence for active face detection")
    active_encoder: str = Field(
        "lightweight_frame_diff", 
        description="Pluggable visual representation (e.g. lightweight_frame_diff, mobilenet_v3, resnet18)"
    )
    min_candidate_duration_sec: float = Field(
        10.0, 
        description="Minimum localized candidate duration in seconds for visual proposals"
    )
    max_candidate_duration_sec: float = Field(
        45.0, 
        description="Maximum localized candidate duration in seconds for visual proposals"
    )
    activity_smoothing_sec: float = Field(
        3.0, 
        description="Temporal smoothing window in seconds for computing continuous visual activity"
    )
    min_activity_percentile: float = Field(
        65.0, 
        description="Percentile threshold above which continuous visual activity forms candidate regions"
    )


class ConversationConfig(BaseModel):
    """Configurable thresholds for conversation turn extraction & expert."""
    turn_pause_threshold_sec: float = Field(
        0.7, 
        description="Minimum silence gap in seconds that demarcates a conversational turn boundary"
    )
    rapid_exchange_turn_rate_per_min: float = Field(
        12.0, 
        description="Turns per minute threshold indicating high-density interactive dialogue"
    )
    min_turn_word_count: int = Field(2, description="Minimum words to count as an active turn")
    min_candidate_duration_sec: float = Field(
        15.0, 
        description="Minimum localized candidate duration in seconds for conversation proposals"
    )
    max_candidate_duration_sec: float = Field(
        50.0, 
        description="Maximum localized candidate duration in seconds for conversation proposals"
    )


class TranscriptConfig(BaseModel):
    """Configurable thresholds for transcript extraction & expert."""
    model_name: str = Field("large-v2", description="WhisperX model variant")
    batch_size: int = Field(16, description="WhisperX batch size")
    compute_type: str = Field("float16", description="WhisperX compute precision")
    chunk_window_sec: float = Field(900.0, description="15-minute sliding analysis chunk")
    chunk_overlap_sec: float = Field(120.0, description="2-minute overlap between chunks")
    min_wpm_filter: float = Field(60.0, description="Words-per-minute lower bound to discard silence")
    min_candidate_duration_sec: float = Field(
        15.0, 
        description="Minimum localized candidate duration in seconds for transcript proposals"
    )
    max_candidate_duration_sec: float = Field(
        45.0, 
        description="Maximum localized candidate duration in seconds for transcript proposals"
    )
    window_duration_sec: float = Field(
        60.0, 
        description="Bounded analysis window duration in seconds for localized semantic extraction"
    )
    window_overlap_sec: float = Field(
        15.0, 
        description="Overlap between consecutive bounded analysis windows in seconds"
    )


class ClipBoundaryConfig(BaseModel):
    """Target clip duration and boundary constraints."""
    min_clip_duration_sec: float = Field(25.0, description="Minimum acceptable clip length")
    max_clip_duration_sec: float = Field(90.0, description="Maximum acceptable clip length")
    target_clip_duration_sec: float = Field(50.0, description="Ideal target duration")
    dead_air_buffer_sec: float = Field(0.35, description="Buffer around spoken words when trimming silence")
    max_clips_per_run: int = Field(10, description="Upper bound of clips generated per request")


class LLMConfig(BaseModel):
    """Configurable settings for LLM reasoning in semantic experts."""
    provider: str = Field("gemini", description="LLM provider (e.g., 'gemini', 'mock')")
    model_name: str = Field("gemini-2.5-flash", description="Model variant for structured semantic extraction")
    temperature: float = Field(0.2, description="Sampling temperature for deterministic extraction")
    max_output_tokens: int = Field(8192, description="Upper token limit on structured responses")
    timeout_sec: float = Field(60.0, description="HTTP / RPC request timeout in seconds")
    max_retries: int = Field(3, description="Retry count for transient rate limits or server errors")


class MTERConfig(BaseModel):
    """
    Configurable prototype heuristic parameters for the Multimodal Temporal Evidence Reasoner (MTER).
    
    NOTE: All weights, tolerances, and thresholds defined here are configurable prototype
    heuristic parameters intended for transparent cross-modal evidence reasoning and future
    ablation studies; they are not learned parameters or empirically validated research constants.
    """
    ablation_mode: Literal[
        "mter_full",
        "transcript_only",
        "transcript_conversation",
        "transcript_conversation_visual",
        "transcript_conversation_prosody",
        "no_mter_heuristic",
    ] = Field(
        "mter_full", 
        description="Ablation configuration mode for evaluating individual modalities or heuristics"
    )
    enabled_modalities: List[str] = Field(
        default_factory=lambda: ["transcript", "conversation", "visual", "prosody"],
        description="List of modalities ingested into the reasoning procedure"
    )
    event_merge_gap_sec: float = Field(
        2.0, 
        description="Temporal gap tolerance in seconds for connecting overlapping proposals into coherent event regions"
    )
    event_compatibility_min_iou: float = Field(
        0.15, 
        description="Minimum temporal IoU threshold for considering two proposals temporally compatible as the same event"
    )
    event_compatibility_min_overlap_ratio: float = Field(
        0.45, 
        description="Minimum fraction of the shorter proposal's duration that must overlap with the longer proposal"
    )
    event_compatibility_max_center_gap_sec: float = Field(
        20.0, 
        description="Maximum distance between proposal temporal midpoints to consider them part of the same event"
    )
    boundary_cluster_tolerance_sec: float = Field(
        3.0, 
        description="Temporal tolerance in seconds for grouping adjacent boundary candidates"
    )
    min_candidate_duration_sec: float = Field(
        10.0, 
        description="Minimum acceptable candidate highlight span in seconds"
    )
    max_candidate_duration_sec: float = Field(
        60.0, 
        description="Maximum acceptable candidate highlight span in seconds"
    )
    target_candidate_duration_sec: float = Field(
        30.0, 
        description="Ideal target highlight span in seconds (used as a secondary soft preference)"
    )
    conflict_threshold_sec: float = Field(
        4.0, 
        description="Threshold in seconds above which boundary discrepancies are flagged as major conflicts"
    )
    auxiliary_grid_bin_size_sec: float = Field(
        1.0, 
        description="Resolution in seconds for the auxiliary lookup/indexing temporal grid"
    )
    active_modality_threshold: float = Field(
        0.05, 
        description="Coverage support threshold to consider a modality actively contributing to candidate span"
    )
    semantic_alignment_tolerance_sec: float = Field(
        2.0, 
        description="Temporal tolerance in seconds for verifying boundary alignment against spoken word/pause boundaries"
    )
    min_context_completeness_threshold: float = Field(
        0.70, 
        description="Minimum completeness score in transcript proposal for sufficient_context check"
    )
    weight_modality_support: float = Field(
        0.30, 
        description="Configurable prototype weight for mean modality coverage support (not a learned research constant)"
    )
    weight_modality_diversity: float = Field(
        0.25, 
        description="Configurable prototype weight for cross-modal diversity reward (not a learned research constant)"
    )
    weight_boundary_agreement: float = Field(
        0.20, 
        description="Configurable prototype weight for start/end cluster tightness (not a learned research constant)"
    )
    weight_contextual_completeness: float = Field(
        0.20, 
        description="Configurable prototype weight for inherited contextual completeness (not a learned research constant)"
    )
    weight_conflict_penalty: float = Field(
        0.15, 
        description="Configurable prototype penalty multiplier for unresolved conflicts (not a learned research constant)"
    )
    weight_duration_penalty: float = Field(
        0.05, 
        description="Configurable prototype soft penalty for deviation from target duration (not a learned research constant)"
    )


class AppConfig(BaseModel):
    """Top-level ClipSense application configuration."""
    auth_token: str = Field(default_factory=lambda: os.getenv("AUTH_TOKEN", "test-secret-token"))
    gemini_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    s3_bucket_name: str = Field(
        default_factory=lambda: os.getenv("AWS_S3_BUCKET", "ai-podcast-clipper-saas-buckets")
    )
    aws_region: str = Field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))
    temp_dir_base: str = Field(default_factory=lambda: os.getenv("CLIPSENSE_TEMP_DIR", "/tmp/clipsense"))

    prosody: ProsodyConfig = Field(default_factory=ProsodyConfig)
    visual: VisualConfig = Field(default_factory=VisualConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    transcript: TranscriptConfig = Field(default_factory=TranscriptConfig)
    clip_boundary: ClipBoundaryConfig = Field(default_factory=ClipBoundaryConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    mter: MTERConfig = Field(default_factory=MTERConfig)


# Global default configuration instance
config = AppConfig()
