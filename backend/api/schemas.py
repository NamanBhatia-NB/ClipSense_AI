"""
ClipSense Local API Schemas (Week 9)

Defines data contracts for:
- Backend health and readiness checks
- Asynchronous task status, stages, and progress
- Video processing submissions (upload or path)
- Highlight candidate results with modality evidence summaries
- Clip streaming and download metadata
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Readiness response indicating backend availability and capabilities."""
    status: str = "ok"
    service: str = "clipsense-api"
    version: str = "1.0.0"
    ffmpeg_available: bool = True
    device: str = "cpu"
    timestamp: float


class ProcessTaskResponse(BaseModel):
    """Immediate acknowledgement returned upon queuing a video processing task."""
    task_id: str
    status: Literal["queued", "processing", "completed", "failed"] = "queued"
    message: str = "Processing queued successfully"
    video_filename: str


class TaskStatusResponse(BaseModel):
    """Current execution status and progress for an asynchronous processing task."""
    task_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    stage: Literal[
        "queued", "extraction", "experts", "bundle", "mter", "rendering", "completed", "failed"
    ]
    progress: float = Field(0.0, ge=0.0, le=1.0)
    message: str
    error: Optional[str] = None
    created_at: float
    completed_at: Optional[float] = None


class ModalityEvidenceSummary(BaseModel):
    """Concise modality-level support metrics for UI inspection."""
    transcript_support: float = 0.0
    conversation_support: float = 0.0
    visual_support: float = 0.0
    prosody_support: float = 0.0
    linguistic_boundary_support: float = 0.0
    strong_modalities: List[str] = Field(default_factory=list)
    moderate_modalities: List[str] = Field(default_factory=list)
    weak_modalities: List[str] = Field(default_factory=list)


class HighlightClipInfo(BaseModel):
    """Metadata and safe download/streaming endpoints for an exported MP4 clip."""
    filename: str
    stream_url: str
    download_url: str
    file_size_bytes: Optional[int] = None
    status: Literal["success", "failed", "skipped"] = "success"


class HighlightItem(BaseModel):
    """A ranked highlight proposal with evidence metrics and clip references."""
    candidate_id: str
    rank: int
    start_time: float
    end_time: float
    duration: float
    confidence_estimate: float
    temporal_agreement_iou: float
    decision_summary: str
    modality_evidence: ModalityEvidenceSummary
    clip: Optional[HighlightClipInfo] = None


class TaskHighlightsResponse(BaseModel):
    """Complete ranked highlights, candidate metadata, and clip references for a completed task."""
    task_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    video_filename: str
    total_duration: float
    total_candidates: int
    highlights: List[HighlightItem] = Field(default_factory=list)
    execution_time_sec: Optional[float] = None
