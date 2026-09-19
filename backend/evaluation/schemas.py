"""
Evaluation Data Contracts and Schemas (W6)

Defines canonical data structures for:
1. Benchmark datasets (distinguishing 'development', 'synthetic', 'research' samples)
2. Uniform prediction spans emitted by all experiment modes
3. Pairwise matching and aggregate metric records
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

# Experiment mode names with precise semantics:
# - baseline: legacy transcript-only LLM baseline
# - transcript_conversation: MTER evidence ablation using transcript + conversation
# - transcript_conversation_visual: MTER evidence ablation using transcript + conversation + visual
# - transcript_conversation_prosody: MTER evidence ablation using transcript + conversation + prosody
# - all_experts_no_mter: deterministic multimodal non-MTER baseline
# - mter_full: complete MTER
ExperimentMode = Literal[
    "baseline",
    "transcript_conversation",
    "transcript_conversation_visual",
    "transcript_conversation_prosody",
    "all_experts_no_mter",
    "mter_full",
]

EXPERIMENT_MODE_DESCRIPTIONS: Dict[ExperimentMode, str] = {
    "baseline": "legacy transcript-only LLM baseline",
    "transcript_conversation": "MTER evidence ablation using transcript + conversation",
    "transcript_conversation_visual": "MTER evidence ablation using transcript + conversation + visual",
    "transcript_conversation_prosody": "MTER evidence ablation using transcript + conversation + prosody",
    "all_experts_no_mter": "deterministic multimodal non-MTER baseline",
    "mter_full": "complete MTER",
}

# Explicit sample types to prevent accidental inclusion of dev/synthetic data in research reports
SampleType = Literal["development", "synthetic", "research"]


class ReferenceHighlight(BaseModel):
    """
    Ground truth or reference highlight interval.
    For development samples, annotation_source must indicate 'manual_dev_debug'.
    """
    span_id: str
    start_time: float
    end_time: float
    label: Optional[str] = None
    annotation_source: str = "unknown"  # e.g., "manual_dev_debug", "synthetic_testbed", "independent_annotator"
    description: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class BenchmarkSample(BaseModel):
    """
    A single evaluated video or synthetic scenario.
    """
    sample_id: str
    sample_type: SampleType
    video_path: Optional[str] = None
    duration_sec: float
    run_dir: str  # Path to pre-extracted/cached run directory or artifact folder
    reference_spans: List[ReferenceHighlight]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkDataset(BaseModel):
    """
    Collection of benchmark samples with dataset-level type categorization.
    """
    dataset_name: str
    dataset_type: SampleType
    description: str
    samples: List[BenchmarkSample]


class EvaluationSpan(BaseModel):
    """
    Canonical output span emitted by all experiment modes.
    Ensures a uniform interface for evaluation metrics and matching.
    """
    span_id: str
    start_time: float
    end_time: float
    duration: float
    confidence: float
    mode_name: ExperimentMode
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PairwiseMetricResult(BaseModel):
    """
    Metrics computed between one predicted span and one matched reference span.
    """
    prediction_span_id: str
    reference_span_id: str
    pred_start: float
    pred_end: float
    pred_duration: float
    ref_start: float
    ref_end: float
    ref_duration: float
    temporal_iou: float
    start_error_sec: float
    end_error_sec: float
    duration_error_sec: float
    rel_duration_error: float
    signed_start_bias_sec: float
    signed_end_bias_sec: float
    hit_at_1s: bool
    hit_at_2s: bool
    hit_at_3s: bool


class SampleEvaluationResult(BaseModel):
    """
    Evaluation output for one benchmark sample under one experiment mode.
    """
    sample_id: str
    sample_type: SampleType
    mode_name: ExperimentMode
    mode_description: str
    num_predictions: int
    num_ground_truth: int
    num_matched: int
    num_unmatched_predictions: int  # False alarms
    num_unmatched_ground_truth: int  # Misses
    matched_metrics: List[PairwiseMetricResult]
    unmatched_prediction_ids: List[str]
    unmatched_reference_ids: List[str]


class AggregateModeMetrics(BaseModel):
    """
    Comprehensive multi-metric summary for one experiment mode across a dataset.
    Never reduced to a single score.
    """
    mode_name: ExperimentMode
    mode_description: str
    dataset_name: str
    dataset_type: SampleType
    num_samples: int
    total_predicted_spans: int
    total_ground_truth_spans: int
    total_matched_spans: int
    total_unmatched_predictions: int
    total_unmatched_ground_truth: int

    # Temporal IoU
    mean_tiou: float
    median_tiou: float
    std_tiou: float

    # Start Error (seconds)
    mean_start_error_sec: float
    median_start_error_sec: float
    std_start_error_sec: float

    # End Error (seconds)
    mean_end_error_sec: float
    median_end_error_sec: float
    std_end_error_sec: float

    # Duration Error (seconds)
    mean_duration_error_sec: float
    median_duration_error_sec: float
    std_duration_error_sec: float
    mean_rel_duration_error: float

    # Tolerance Hit Rates (percentage: 0.0 -> 100.0)
    hit_rate_at_1s: float
    hit_rate_at_2s: float
    hit_rate_at_3s: float


class BenchmarkSummaryReport(BaseModel):
    """
    Complete benchmark evaluation report across all evaluated modes.
    """
    dataset_name: str
    dataset_type: SampleType
    is_research_report: bool
    evaluated_modes: List[ExperimentMode]
    mode_summaries: Dict[str, AggregateModeMetrics]
    sample_results: List[SampleEvaluationResult]
