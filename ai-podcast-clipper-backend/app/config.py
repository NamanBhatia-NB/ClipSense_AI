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
from typing import Optional
from pydantic import BaseModel, Field


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


class TranscriptConfig(BaseModel):
    """Configurable thresholds for transcript extraction & expert."""
    model_name: str = Field("large-v2", description="WhisperX model variant")
    batch_size: int = Field(16, description="WhisperX batch size")
    compute_type: str = Field("float16", description="WhisperX compute precision")
    chunk_window_sec: float = Field(900.0, description="15-minute sliding analysis chunk")
    chunk_overlap_sec: float = Field(120.0, description="2-minute overlap between chunks")
    min_wpm_filter: float = Field(60.0, description="Words-per-minute lower bound to discard silence")


class ClipBoundaryConfig(BaseModel):
    """Target clip duration and boundary constraints."""
    min_clip_duration_sec: float = Field(25.0, description="Minimum acceptable clip length")
    max_clip_duration_sec: float = Field(90.0, description="Maximum acceptable clip length")
    target_clip_duration_sec: float = Field(50.0, description="Ideal target duration")
    dead_air_buffer_sec: float = Field(0.35, description="Buffer around spoken words when trimming silence")
    max_clips_per_run: int = Field(10, description="Upper bound of clips generated per request")


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


# Global default configuration instance
config = AppConfig()
