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
# - baseline / system_a_transcript_only: legacy transcript-only LLM baseline (Consumes ONLY transcript words)
# - all_experts_no_mter / system_b_simple_aggregation: deterministic multimodal non-MTER baseline (uniform unweighted coverage sum)
# - mter_full / system_c_mter: complete MTER (multimodal temporal compatibility graph + reliability weights + conflict isolation)
# - system_d_mter_refined: MTER with evidence-aware boundary refinement enabled (W10 refiner)
# - transcript_conversation: MTER evidence ablation using transcript + conversation
# - transcript_conversation_visual: MTER evidence ablation using transcript + conversation + visual
# - transcript_conversation_prosody: MTER evidence ablation using transcript + conversation + prosody
ExperimentMode = Literal[
    "baseline",
    "system_a_transcript_only",
    "all_experts_no_mter",
    "system_b_simple_aggregation",
    "mter_full",
    "system_c_mter",
    "system_d_mter_refined",
    "transcript_conversation",
    "transcript_conversation_visual",
    "transcript_conversation_prosody",
]

EXPERIMENT_MODE_DESCRIPTIONS: Dict[str, str] = {
    "baseline": "System A: Legacy transcript-only LLM baseline (word boundary snapped)",
    "system_a_transcript_only": "System A: Legacy transcript-only baseline (word boundary snapped)",
    "all_experts_no_mter": "System B: Four-expert simple multimodal aggregation (unweighted coverage sum)",
    "system_b_simple_aggregation": "System B: Four-expert simple multimodal aggregation (unweighted coverage sum)",
    "mter_full": "System C: MTER (uniform modality weights, conflict-sensitive temporal arbitration)",
    "system_c_mter": "System C: MTER (uniform modality weights, conflict-sensitive temporal arbitration)",
    "system_d_mter_refined": "System D: MTER with boundary refinement (uniform modality weights, conflict-sensitive temporal arbitration)",
    "transcript_conversation": "MTER evidence ablation: transcript + conversation",
    "transcript_conversation_visual": "MTER evidence ablation: transcript + conversation + visual",
    "transcript_conversation_prosody": "MTER evidence ablation: transcript + conversation + prosody",
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
    predicted_spans: List[EvaluationSpan] = Field(default_factory=list)


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

    # Temporal IoU (None if 0 matched spans)
    mean_tiou: Optional[float] = None
    median_tiou: Optional[float] = None
    std_tiou: Optional[float] = None

    # Start Error (seconds, None if 0 matched spans)
    mean_start_error_sec: Optional[float] = None
    median_start_error_sec: Optional[float] = None
    std_start_error_sec: Optional[float] = None

    # End Error (seconds, None if 0 matched spans)
    mean_end_error_sec: Optional[float] = None
    median_end_error_sec: Optional[float] = None
    std_end_error_sec: Optional[float] = None

    # Duration Error (seconds, None if 0 matched spans)
    mean_duration_error_sec: Optional[float] = None
    median_duration_error_sec: Optional[float] = None
    std_duration_error_sec: Optional[float] = None
    mean_rel_duration_error: Optional[float] = None

    # Tolerance Hit Rates (percentage: 0.0 -> 100.0, None if 0 matched spans)
    hit_rate_at_1s: Optional[float] = None
    hit_rate_at_2s: Optional[float] = None
    hit_rate_at_3s: Optional[float] = None

    # Precision, Recall, F1 at discrete tIoU thresholds
    precision_at_iou_01: float = 0.0
    recall_at_iou_01: float = 0.0
    f1_at_iou_01: float = 0.0
    precision_at_iou_03: float = 0.0
    recall_at_iou_03: float = 0.0
    f1_at_iou_03: float = 0.0
    precision_at_iou_05: float = 0.0
    recall_at_iou_05: float = 0.0
    f1_at_iou_05: float = 0.0

    # Operational metrics
    mean_candidate_duration_sec: float = 0.0
    mean_intra_overlap_iou: float = 0.0
    execution_latency_sec: float = 0.0
    abstention_count: int = 0


class SegmentImportanceAnnotation(BaseModel):
    """
    Continuous segment or frame-level importance annotations (e.g. TVSum, Video2GIF).
    Represents time-series salience/importance rather than discrete boundary intervals.
    """
    sample_id: str
    segment_duration_sec: float = 1.0  # Duration of each temporal bin in seconds
    num_segments: int
    segment_scores: List[float]  # Normalized importance scores on [0.0, 1.0] per segment
    annotator_count: int = 1
    category: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SegmentMetricResult(BaseModel):
    """
    Segment-level ranking and correlation metrics for a single video sample.
    """
    sample_id: str
    mode_name: str
    num_segments: int
    spearman_rho: float
    spearman_pvalue: Optional[float] = None
    kendall_tau: float
    kendall_pvalue: Optional[float] = None
    top_15_hit_rate: float  # Top 15% temporal overlap (standard TVSum evaluation protocol)
    top_20_hit_rate: float  # Top 20% temporal overlap
    # Rhapsody official metrics
    map_score: Optional[float] = None
    hit_rate: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None


class AggregateSegmentMetrics(BaseModel):
    """
    Aggregate segment-level ranking summary across all videos in a benchmark dataset.
    """
    mode_name: str
    mode_description: str
    num_samples: int
    mean_spearman_rho: float
    median_spearman_rho: float
    std_spearman_rho: float
    mean_kendall_tau: float
    median_kendall_tau: float
    std_kendall_tau: float
    mean_top_15_hit_rate: float
    mean_top_20_hit_rate: float
    # Rhapsody official metrics
    mean_map: Optional[float] = None
    mean_hit_rate: Optional[float] = None
    mean_precision: Optional[float] = None
    mean_recall: Optional[float] = None
    mean_f1: Optional[float] = None


class BoundaryRefinementAblationResult(BaseModel):
    """
    Dedicated comparative measurement isolating Boundary Refinement on identical candidates.
    Evaluates boundary drift, reference-relative error reduction, and tIoU delta.
    """
    sample_id: str
    num_candidates: int
    num_changed: int
    mean_start_shift_sec: float
    mean_end_shift_sec: float
    mean_duration_shift_sec: float
    unrefined_mean_tiou: Optional[float] = None
    refined_mean_tiou: Optional[float] = None
    tiou_delta: Optional[float] = None
    start_error_reduction_sec: Optional[float] = None
    end_error_reduction_sec: Optional[float] = None
    hit1_unrefined: Optional[float] = None
    hit1_refined: Optional[float] = None
    hit2_unrefined: Optional[float] = None
    hit2_refined: Optional[float] = None
    hit3_unrefined: Optional[float] = None
    hit3_refined: Optional[float] = None
    rationale_samples: List[str] = Field(default_factory=list)


class SystemComparisonSummary(BaseModel):
    """
    Side-by-side comparison summary across Systems A, B, C, D on an identical sample pool.
    """
    dataset_name: str
    dataset_type: SampleType
    num_samples: int
    systems_evaluated: List[str]
    system_metrics: Dict[str, AggregateModeMetrics]
    refinement_ablation: Optional[BoundaryRefinementAblationResult] = None
    segment_metrics: Optional[Dict[str, AggregateSegmentMetrics]] = None
    research_conclusions: Dict[str, Any] = Field(default_factory=dict)


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
    segment_summaries: Optional[Dict[str, AggregateSegmentMetrics]] = None
    refinement_ablation: Optional[BoundaryRefinementAblationResult] = None
    system_comparison: Optional[SystemComparisonSummary] = None
    evaluation_budget: Optional[str] = "natural_selection"
    budget_k: Optional[int] = None
