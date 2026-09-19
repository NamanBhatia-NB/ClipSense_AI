"""
ClipSense Research Evaluation Framework (W6)

Exposes formal baseline, ablation runner, metrics, and benchmark orchestrator.
"""

from evaluation.ablation_runner import (
    AblationRunner,
    run_all_experts_no_mter_algorithm,
)
from evaluation.baseline import (
    BaselineConfig,
    LegacyTranscriptBaselineSelector,
)
from evaluation.benchmark_orchestrator import BenchmarkOrchestrator
from evaluation.dataset import (
    create_synthetic_regression_fixture,
    load_benchmark_dataset,
)
from evaluation.metrics import (
    DEFAULT_MIN_MATCH_IOU,
    compute_boundary_hits,
    compute_duration_error,
    compute_end_error,
    compute_pairwise_metrics,
    compute_start_error,
    compute_temporal_iou,
    match_spans_hungarian,
)
from evaluation.schemas import (
    EXPERIMENT_MODE_DESCRIPTIONS,
    AggregateModeMetrics,
    BenchmarkDataset,
    BenchmarkSample,
    BenchmarkSummaryReport,
    EvaluationSpan,
    ExperimentMode,
    PairwiseMetricResult,
    ReferenceHighlight,
    SampleEvaluationResult,
    SampleType,
)

__all__ = [
    "AblationRunner",
    "run_all_experts_no_mter_algorithm",
    "BaselineConfig",
    "LegacyTranscriptBaselineSelector",
    "BenchmarkOrchestrator",
    "create_synthetic_regression_fixture",
    "load_benchmark_dataset",
    "compute_temporal_iou",
    "compute_start_error",
    "compute_end_error",
    "compute_duration_error",
    "compute_boundary_hits",
    "compute_pairwise_metrics",
    "match_spans_hungarian",
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
    "BenchmarkSummaryReport",
]
