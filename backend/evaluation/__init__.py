"""
ClipSense Research Evaluation and Benchmark Framework (W6-W11)

Exposes formal baseline, ablation runner, metrics, adapters, and benchmark orchestrator.
"""

from evaluation.ablation_runner import (
    AblationRunner,
    run_all_experts_no_mter_algorithm,
)
from evaluation.adapters import (
    BaseDatasetAdapter,
    IntervalHighlightAdapter,
    QVHighlightsAdapter,
    TVSumAdapter,
    get_dataset_adapter,
)
from evaluation.baseline import (
    BaselineConfig,
    LegacyTranscriptBaselineSelector,
)
from evaluation.benchmark_orchestrator import BenchmarkOrchestrator
from evaluation.dataset import (
    create_synthetic_regression_fixture,
    load_benchmark_dataset,
    load_benchmark_dataset_with_annotations,
)
from evaluation.metrics import (
    DEFAULT_MIN_MATCH_IOU,
    compute_boundary_hits,
    compute_duration_error,
    compute_end_error,
    compute_intra_candidate_overlap_rate,
    compute_kendall_tau,
    compute_pairwise_metrics,
    compute_precision_recall_f1_at_iou,
    compute_spearman_rank_correlation,
    compute_start_error,
    compute_temporal_iou,
    compute_top_k_overlap,
    evaluate_boundary_refinement_effect,
    match_spans_hungarian,
    spans_to_segment_importance_curve,
)
from evaluation.schemas import (
    EXPERIMENT_MODE_DESCRIPTIONS,
    AggregateModeMetrics,
    AggregateSegmentMetrics,
    BenchmarkDataset,
    BenchmarkSample,
    BenchmarkSummaryReport,
    BoundaryRefinementAblationResult,
    EvaluationSpan,
    ExperimentMode,
    PairwiseMetricResult,
    ReferenceHighlight,
    SampleEvaluationResult,
    SampleType,
    SegmentImportanceAnnotation,
    SegmentMetricResult,
    SystemComparisonSummary,
)

__all__ = [
    "AblationRunner",
    "run_all_experts_no_mter_algorithm",
    "BaselineConfig",
    "LegacyTranscriptBaselineSelector",
    "BenchmarkOrchestrator",
    "create_synthetic_regression_fixture",
    "load_benchmark_dataset",
    "load_benchmark_dataset_with_annotations",
    "compute_temporal_iou",
    "compute_start_error",
    "compute_end_error",
    "compute_duration_error",
    "compute_boundary_hits",
    "compute_pairwise_metrics",
    "match_spans_hungarian",
    "compute_precision_recall_f1_at_iou",
    "compute_intra_candidate_overlap_rate",
    "compute_spearman_rank_correlation",
    "compute_kendall_tau",
    "compute_top_k_overlap",
    "evaluate_boundary_refinement_effect",
    "spans_to_segment_importance_curve",
    "DEFAULT_MIN_MATCH_IOU",
    "ExperimentMode",
    "EXPERIMENT_MODE_DESCRIPTIONS",
    "SampleType",
    "ReferenceHighlight",
    "BenchmarkSample",
    "BenchmarkDataset",
    "EvaluationSpan",
    "PairwiseMetricResult",
    "SampleEvaluationResult",
    "AggregateModeMetrics",
    "AggregateSegmentMetrics",
    "BoundaryRefinementAblationResult",
    "SystemComparisonSummary",
    "BenchmarkSummaryReport",
    "SegmentImportanceAnnotation",
    "SegmentMetricResult",
    "BaseDatasetAdapter",
    "TVSumAdapter",
    "QVHighlightsAdapter",
    "IntervalHighlightAdapter",
    "get_dataset_adapter",
]
