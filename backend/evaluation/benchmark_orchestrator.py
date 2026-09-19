"""
Benchmark Orchestrator and Multi-Metric Reporter (W6)

Coordinates multi-mode benchmark execution across dataset samples.
Computes multi-metric aggregations (Mean/Median tIoU, Start/End errors,
Duration errors, Hit@1s/2s/3s, matched/unmatched accounting) and emits
structured JSON and Markdown reports.
"""

import json
import logging
import math
import pathlib
from typing import Dict, List, Optional
import numpy as np

from evaluation.ablation_runner import AblationRunner
from evaluation.metrics import match_spans_hungarian
from evaluation.schemas import (
    EXPERIMENT_MODE_DESCRIPTIONS,
    AggregateModeMetrics,
    BenchmarkDataset,
    BenchmarkSummaryReport,
    ExperimentMode,
    PairwiseMetricResult,
    SampleEvaluationResult,
)

logger = logging.getLogger("clipsense.evaluation.orchestrator")


class BenchmarkOrchestrator:
    """
    Executes and reports benchmark evaluations across experiment modes.
    """

    def __init__(self, ablation_runner: Optional[AblationRunner] = None):
        self.runner = ablation_runner or AblationRunner()

    def evaluate_dataset(
        self,
        dataset: BenchmarkDataset,
        modes: List[ExperimentMode],
        cache_dir: Optional[str] = None,
        min_match_iou: float = 0.10,
    ) -> BenchmarkSummaryReport:
        """
        Runs evaluation for specified modes across all samples in the dataset.
        """
        sample_results: List[SampleEvaluationResult] = []
        mode_pairwise_map: Dict[ExperimentMode, List[PairwiseMetricResult]] = {
            m: [] for m in modes
        }
        mode_pred_counts: Dict[ExperimentMode, int] = {m: 0 for m in modes}
        mode_ref_counts: Dict[ExperimentMode, int] = {m: 0 for m in modes}
        mode_unmatched_preds: Dict[ExperimentMode, int] = {m: 0 for m in modes}
        mode_unmatched_refs: Dict[ExperimentMode, int] = {m: 0 for m in modes}

        for sample in dataset.samples:
            ref_spans = sample.reference_spans
            total_ref = len(ref_spans)

            for mode in modes:
                mode_ref_counts[mode] += total_ref
                predictions = self.runner.run_mode(
                    mode=mode,
                    run_dir=sample.run_dir,
                    video_duration=sample.duration_sec,
                    sample_id=sample.sample_id,
                    cache_dir=cache_dir,
                )
                mode_pred_counts[mode] += len(predictions)

                # Solely Hungarian bipartite matching protocol
                matched_metrics, unmatched_preds, unmatched_refs = match_spans_hungarian(
                    predictions=predictions,
                    references=ref_spans,
                    min_iou=min_match_iou,
                )

                mode_pairwise_map[mode].extend(matched_metrics)
                mode_unmatched_preds[mode] += len(unmatched_preds)
                mode_unmatched_refs[mode] += len(unmatched_refs)

                sample_results.append(
                    SampleEvaluationResult(
                        sample_id=sample.sample_id,
                        sample_type=sample.sample_type,
                        mode_name=mode,
                        mode_description=EXPERIMENT_MODE_DESCRIPTIONS.get(mode, mode),
                        num_predictions=len(predictions),
                        num_ground_truth=total_ref,
                        num_matched=len(matched_metrics),
                        num_unmatched_predictions=len(unmatched_preds),
                        num_unmatched_ground_truth=len(unmatched_refs),
                        matched_metrics=matched_metrics,
                        unmatched_prediction_ids=unmatched_preds,
                        unmatched_reference_ids=unmatched_refs,
                    )
                )

        # Compute aggregate metrics per mode
        mode_summaries: Dict[str, AggregateModeMetrics] = {}
        for mode in modes:
            matched = mode_pairwise_map[mode]
            num_matched = len(matched)

            if num_matched > 0:
                tious = [m.temporal_iou for m in matched]
                start_errs = [m.start_error_sec for m in matched]
                end_errs = [m.end_error_sec for m in matched]
                dur_errs = [m.duration_error_sec for m in matched]
                rel_dur_errs = [m.rel_duration_error for m in matched]

                mean_tiou = float(np.mean(tious))
                median_tiou = float(np.median(tious))
                std_tiou = float(np.std(tious))

                mean_start_err = float(np.mean(start_errs))
                median_start_err = float(np.median(start_errs))
                std_start_err = float(np.std(start_errs))

                mean_end_err = float(np.mean(end_errs))
                median_end_err = float(np.median(end_errs))
                std_end_err = float(np.std(end_errs))

                mean_dur_err = float(np.mean(dur_errs))
                median_dur_err = float(np.median(dur_errs))
                std_dur_err = float(np.std(dur_errs))
                mean_rel_dur_err = float(np.mean(rel_dur_errs))

                hit1 = float(sum(1 for m in matched if m.hit_at_1s) / num_matched * 100.0)
                hit2 = float(sum(1 for m in matched if m.hit_at_2s) / num_matched * 100.0)
                hit3 = float(sum(1 for m in matched if m.hit_at_3s) / num_matched * 100.0)
            else:
                mean_tiou = median_tiou = std_tiou = 0.0
                mean_start_err = median_start_err = std_start_err = 0.0
                mean_end_err = median_end_err = std_end_err = 0.0
                mean_dur_err = median_dur_err = std_dur_err = mean_rel_dur_err = 0.0
                hit1 = hit2 = hit3 = 0.0

            mode_summaries[mode] = AggregateModeMetrics(
                mode_name=mode,
                mode_description=EXPERIMENT_MODE_DESCRIPTIONS.get(mode, mode),
                dataset_name=dataset.dataset_name,
                dataset_type=dataset.dataset_type,
                num_samples=len(dataset.samples),
                total_predicted_spans=mode_pred_counts[mode],
                total_ground_truth_spans=mode_ref_counts[mode],
                total_matched_spans=num_matched,
                total_unmatched_predictions=mode_unmatched_preds[mode],
                total_unmatched_ground_truth=mode_unmatched_refs[mode],
                mean_tiou=round(mean_tiou, 4),
                median_tiou=round(median_tiou, 4),
                std_tiou=round(std_tiou, 4),
                mean_start_error_sec=round(mean_start_err, 4),
                median_start_error_sec=round(median_start_err, 4),
                std_start_error_sec=round(std_start_err, 4),
                mean_end_error_sec=round(mean_end_err, 4),
                median_end_error_sec=round(median_end_err, 4),
                std_end_error_sec=round(std_end_err, 4),
                mean_duration_error_sec=round(mean_dur_err, 4),
                median_duration_error_sec=round(median_dur_err, 4),
                std_duration_error_sec=round(std_dur_err, 4),
                mean_rel_duration_error=round(mean_rel_dur_err, 4),
                hit_rate_at_1s=round(hit1, 2),
                hit_rate_at_2s=round(hit2, 2),
                hit_rate_at_3s=round(hit3, 2),
            )

        return BenchmarkSummaryReport(
            dataset_name=dataset.dataset_name,
            dataset_type=dataset.dataset_type,
            is_research_report=(dataset.dataset_type == "research"),
            evaluated_modes=modes,
            mode_summaries=mode_summaries,
            sample_results=sample_results,
        )

    def generate_markdown_report(self, report: BenchmarkSummaryReport) -> str:
        """
        Formats evaluation metrics into a clean, comprehensive markdown table.
        """
        lines = [
            f"# Benchmark Evaluation Report: {report.dataset_name}",
            "",
            f"- **Dataset Category:** `{report.dataset_type}`",
            f"- **Is Research Report:** `{report.is_research_report}`",
            f"- **Evaluated Modes:** {', '.join(f'`{m}`' for m in report.evaluated_modes)}",
            "",
        ]

        if report.dataset_type != "research":
            lines.extend([
                "> [!NOTE]",
                f"> This report reflects a **{report.dataset_type}** benchmark suite "
                "(development/debug reference or synthetic fixture). "
                "These results serve for code verification and must NOT be interpreted as research benchmark performance.",
                "",
            ])

        lines.extend([
            "## Aggregate Metrics by Experiment Mode",
            "",
            "| Experiment Mode | Semantics | Pred / GT / Match | Mean tIoU | Med tIoU | Mean Start Err | Mean End Err | Mean Dur Err | Hit@1s | Hit@2s | Hit@3s |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for mode_key, m in report.mode_summaries.items():
            counts_str = f"{m.total_predicted_spans} / {m.total_ground_truth_spans} / {m.total_matched_spans}"
            lines.append(
                f"| `{m.mode_name}` | {m.mode_description} | {counts_str} | "
                f"{m.mean_tiou:.3f} | {m.median_tiou:.3f} | "
                f"{m.mean_start_error_sec:.2f}s | {m.mean_end_error_sec:.2f}s | {m.mean_duration_error_sec:.2f}s | "
                f"{m.hit_rate_at_1s:.1f}% | {m.hit_rate_at_2s:.1f}% | {m.hit_rate_at_3s:.1f}% |"
            )

        lines.append("")
        return "\n".join(lines)
