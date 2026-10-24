"""
Benchmark Orchestrator and Multi-Metric Reporter (W11)

Coordinates multi-mode benchmark execution across dataset samples.
Computes multi-metric aggregations:
- Continuous Temporal IoU (Mean, Median, Std)
- Boundary Errors (Start, End, Duration, Signed Bias)
- Tolerance Hit Rates (Hit@1s, Hit@2s, Hit@3s)
- Precision, Recall, F1 at tIoU in {0.1, 0.3, 0.5}
- Operational metrics (Candidate duration, intra-overlap rate, latency, abstention)
- Segment-level importance metrics (Spearman rho, Kendall tau, Top-15% overlap)
- Dedicated Boundary Refinement ablation (isolation of boundary shift vs candidate identity)
- Systems A, B, C, D comparative scorecard
Emits structured JSON, Markdown, and rich interactive HTML reports.
"""

import json
import logging
import math
import pathlib
import time
from typing import Any, Dict, List, Optional
import numpy as np

from evaluation.ablation_runner import AblationRunner
from evaluation.metrics import (
    compute_intra_candidate_overlap_rate,
    compute_kendall_tau,
    compute_precision_recall_f1_at_iou,
    compute_rhapsody_metrics,
    compute_spearman_rank_correlation,
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
    BenchmarkSummaryReport,
    BoundaryRefinementAblationResult,
    EvaluationSpan,
    ExperimentMode,
    PairwiseMetricResult,
    SampleEvaluationResult,
    SegmentImportanceAnnotation,
    SegmentMetricResult,
    SystemComparisonSummary,
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
        segment_annotations: Optional[Dict[str, SegmentImportanceAnnotation]] = None,
        budget_k: Optional[int] = None,
    ) -> BenchmarkSummaryReport:
        """
        Runs comprehensive evaluation for specified modes across all samples in the dataset.
        Computes interval matching, operational metrics, segment importance metrics,
        and boundary refinement ablations.
        Supports fixed-budget evaluation when budget_k is provided.
        """
        sample_results: List[SampleEvaluationResult] = []
        mode_pairwise_map: Dict[str, List[PairwiseMetricResult]] = {m: [] for m in modes}
        mode_pred_counts: Dict[str, int] = {m: 0 for m in modes}
        mode_ref_counts: Dict[str, int] = {m: 0 for m in modes}
        mode_unmatched_preds: Dict[str, int] = {m: 0 for m in modes}
        mode_unmatched_refs: Dict[str, int] = {m: 0 for m in modes}
        mode_candidate_durations: Dict[str, List[float]] = {m: [] for m in modes}
        mode_intra_overlaps: Dict[str, List[float]] = {m: [] for m in modes}
        mode_latencies: Dict[str, List[float]] = {m: [] for m in modes}
        mode_abstentions: Dict[str, int] = {m: 0 for m in modes}

        # Store predictions per sample per mode for boundary ablation & segment ranking
        sample_mode_predictions: Dict[str, Dict[str, List[EvaluationSpan]]] = {
            s.sample_id: {} for s in dataset.samples
        }

        for sample in dataset.samples:
            ref_spans = sample.reference_spans
            total_ref = len(ref_spans)

            for mode in modes:
                mode_ref_counts[mode] += total_ref
                t_start = time.time()
                predictions = self.runner.run_mode(
                    mode=mode,
                    run_dir=sample.run_dir,
                    video_duration=sample.duration_sec,
                    sample_id=sample.sample_id,
                    cache_dir=cache_dir,
                )
                latency = time.time() - t_start

                # Fixed-budget selection: deterministic top-K highest confidence first
                if budget_k is not None and len(predictions) > budget_k:
                    predictions = sorted(predictions, key=lambda p: (p.confidence, -p.start_time), reverse=True)[:budget_k]

                mode_latencies[mode].append(latency)
                mode_pred_counts[mode] += len(predictions)
                sample_mode_predictions[sample.sample_id][mode] = predictions

                if len(predictions) == 0:
                    mode_abstentions[mode] += 1

                for p in predictions:
                    mode_candidate_durations[mode].append(p.duration)

                intra_ov = compute_intra_candidate_overlap_rate(predictions)
                mode_intra_overlaps[mode].append(intra_ov)

                # Hungarian bipartite matching protocol
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
                        predicted_spans=predictions,
                    )
                )

        # Compute aggregate metrics per mode
        mode_summaries: Dict[str, AggregateModeMetrics] = {}
        for mode in modes:
            matched = mode_pairwise_map[mode]
            num_matched = len(matched)
            total_preds = mode_pred_counts[mode]
            total_refs = mode_ref_counts[mode]

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
                mean_tiou = median_tiou = std_tiou = None
                mean_start_err = median_start_err = std_start_err = None
                mean_end_err = median_end_err = std_end_err = None
                mean_dur_err = median_dur_err = std_dur_err = mean_rel_dur_err = None
                hit1 = hit2 = hit3 = None

            # Thresholded Precision / Recall / F1
            p01, r01, f01 = compute_precision_recall_f1_at_iou(matched, total_preds, total_refs, 0.10)
            p03, r03, f03 = compute_precision_recall_f1_at_iou(matched, total_preds, total_refs, 0.30)
            p05, r05, f05 = compute_precision_recall_f1_at_iou(matched, total_preds, total_refs, 0.50)

            dur_list = mode_candidate_durations[mode]
            mean_cand_dur = float(np.mean(dur_list)) if dur_list else 0.0
            mean_intra_ov = float(np.mean(mode_intra_overlaps[mode])) if mode_intra_overlaps[mode] else 0.0
            total_latency = float(sum(mode_latencies[mode]))

            mode_summaries[mode] = AggregateModeMetrics(
                mode_name=mode,
                mode_description=EXPERIMENT_MODE_DESCRIPTIONS.get(mode, mode),
                dataset_name=dataset.dataset_name,
                dataset_type=dataset.dataset_type,
                num_samples=len(dataset.samples),
                total_predicted_spans=total_preds,
                total_ground_truth_spans=total_refs,
                total_matched_spans=num_matched,
                total_unmatched_predictions=mode_unmatched_preds[mode],
                total_unmatched_ground_truth=mode_unmatched_refs[mode],
                mean_tiou=round(mean_tiou, 4) if mean_tiou is not None else None,
                median_tiou=round(median_tiou, 4) if median_tiou is not None else None,
                std_tiou=round(std_tiou, 4) if std_tiou is not None else None,
                mean_start_error_sec=round(mean_start_err, 4) if mean_start_err is not None else None,
                median_start_error_sec=round(median_start_err, 4) if median_start_err is not None else None,
                std_start_error_sec=round(std_start_err, 4) if std_start_err is not None else None,
                mean_end_error_sec=round(mean_end_err, 4) if mean_end_err is not None else None,
                median_end_error_sec=round(median_end_err, 4) if median_end_err is not None else None,
                std_end_error_sec=round(std_end_err, 4) if std_end_err is not None else None,
                mean_duration_error_sec=round(mean_dur_err, 4) if mean_dur_err is not None else None,
                median_duration_error_sec=round(median_dur_err, 4) if median_dur_err is not None else None,
                std_duration_error_sec=round(std_dur_err, 4) if std_dur_err is not None else None,
                mean_rel_duration_error=round(mean_rel_dur_err, 4) if mean_rel_dur_err is not None else None,
                hit_rate_at_1s=round(hit1, 2) if hit1 is not None else None,
                hit_rate_at_2s=round(hit2, 2) if hit2 is not None else None,
                hit_rate_at_3s=round(hit3, 2) if hit3 is not None else None,
                precision_at_iou_01=round(p01 * 100.0, 2),
                recall_at_iou_01=round(r01 * 100.0, 2),
                f1_at_iou_01=round(f01 * 100.0, 2),
                precision_at_iou_03=round(p03 * 100.0, 2),
                recall_at_iou_03=round(r03 * 100.0, 2),
                f1_at_iou_03=round(f03 * 100.0, 2),
                precision_at_iou_05=round(p05 * 100.0, 2),
                recall_at_iou_05=round(r05 * 100.0, 2),
                f1_at_iou_05=round(f05 * 100.0, 2),
                mean_candidate_duration_sec=round(mean_cand_dur, 2),
                mean_intra_overlap_iou=round(mean_intra_ov, 4),
                execution_latency_sec=round(total_latency, 3),
                abstention_count=mode_abstentions[mode],
            )

        # Optional segment-level evaluation (for TVSum / importance-based annotations)
        segment_summaries: Optional[Dict[str, AggregateSegmentMetrics]] = None
        if segment_annotations:
            segment_summaries = self._evaluate_segment_importance(
                dataset=dataset,
                modes=modes,
                sample_mode_predictions=sample_mode_predictions,
                segment_annotations=segment_annotations,
            )

        # Boundary Refinement Isolation Ablation (System C vs System D)
        refinement_ablation = self._evaluate_boundary_refinement_isolation(
            dataset=dataset,
            sample_mode_predictions=sample_mode_predictions,
        )

        # System Comparison Summary (A vs B vs C vs D)
        system_comparison = self._build_system_comparison(
            dataset=dataset,
            mode_summaries=mode_summaries,
            refinement_ablation=refinement_ablation,
            segment_summaries=segment_summaries,
        )

        budget_label = f"fixed_budget_k{budget_k}" if budget_k is not None else "natural_selection"
        return BenchmarkSummaryReport(
            dataset_name=dataset.dataset_name,
            dataset_type=dataset.dataset_type,
            is_research_report=(dataset.dataset_type == "research"),
            evaluated_modes=modes,
            mode_summaries=mode_summaries,
            sample_results=sample_results,
            segment_summaries=segment_summaries,
            refinement_ablation=refinement_ablation,
            system_comparison=system_comparison,
            evaluation_budget=budget_label,
            budget_k=budget_k,
        )

    def _evaluate_segment_importance(
        self,
        dataset: BenchmarkDataset,
        modes: List[ExperimentMode],
        sample_mode_predictions: Dict[str, Dict[str, List[EvaluationSpan]]],
        segment_annotations: Dict[str, SegmentImportanceAnnotation],
    ) -> Dict[str, AggregateSegmentMetrics]:
        """
        Computes segment-level correlation (Spearman rho, Kendall tau) and top-15% overlap.
        """
        seg_summaries: Dict[str, AggregateSegmentMetrics] = {}

        for mode in modes:
            rhos = []
            taus = []
            top15s = []
            top20s = []
            maps = []
            hits = []
            precs = []
            recs = []
            f1s = []

            for sample in dataset.samples:
                if sample.sample_id not in segment_annotations:
                    continue
                seg_ann = segment_annotations[sample.sample_id]
                preds = sample_mode_predictions[sample.sample_id].get(mode, [])

                # Project predicted spans onto continuous segment curve matching annotation bins
                num_segs = seg_ann.num_segments if seg_ann.num_segments == 100 else None
                pred_curve = spans_to_segment_importance_curve(
                    spans=preds,
                    video_duration=sample.duration_sec,
                    segment_duration_sec=seg_ann.segment_duration_sec,
                    num_segments=num_segs,
                )

                ref_curve = np.array(seg_ann.segment_scores, dtype=float)

                if num_segs == 100:
                    if len(pred_curve) != 100 or len(ref_curve) != 100:
                        raise ValueError(
                            f"Rhapsody temporal bin count mismatch for sample '{sample.sample_id}': "
                            f"pred has {len(pred_curve)} bins, ref has {len(ref_curve)} bins. Expected exactly 100."
                        )
                    p_eval = pred_curve
                    r_eval = ref_curve
                else:
                    min_len = min(len(pred_curve), len(ref_curve))
                    if min_len < 3:
                        continue
                    p_eval = pred_curve[:min_len]
                    r_eval = ref_curve[:min_len]

                rho, _ = compute_spearman_rank_correlation(p_eval, r_eval)
                tau, _ = compute_kendall_tau(p_eval, r_eval)
                t15 = compute_top_k_overlap(p_eval, r_eval, top_ratio=0.15)
                t20 = compute_top_k_overlap(p_eval, r_eval, top_ratio=0.20)

                rhos.append(rho)
                taus.append(tau)
                top15s.append(t15 * 100.0)
                top20s.append(t20 * 100.0)

                # Official Rhapsody metrics for 100-segment annotations
                if len(r_eval) == 100:
                    r_res = compute_rhapsody_metrics(p_eval, r_eval)
                    maps.append(r_res["average_precision"] * 100.0)
                    hits.append(r_res["hit_rate"] * 100.0)
                    precs.append(r_res["precision"] * 100.0)
                    recs.append(r_res["recall"] * 100.0)
                    f1s.append(r_res["f1"] * 100.0)

            n_s = len(rhos)
            if n_s > 0:
                seg_summaries[mode] = AggregateSegmentMetrics(
                    mode_name=mode,
                    mode_description=EXPERIMENT_MODE_DESCRIPTIONS.get(mode, mode),
                    num_samples=n_s,
                    mean_spearman_rho=round(float(np.mean(rhos)), 4),
                    median_spearman_rho=round(float(np.median(rhos)), 4),
                    std_spearman_rho=round(float(np.std(rhos)), 4),
                    mean_kendall_tau=round(float(np.mean(taus)), 4),
                    median_kendall_tau=round(float(np.median(taus)), 4),
                    std_kendall_tau=round(float(np.std(taus)), 4),
                    mean_top_15_hit_rate=round(float(np.mean(top15s)), 2),
                    mean_top_20_hit_rate=round(float(np.mean(top20s)), 2),
                    mean_map=round(float(np.mean(maps)), 2) if maps else None,
                    mean_hit_rate=round(float(np.mean(hits)), 2) if hits else None,
                    mean_precision=round(float(np.mean(precs)), 2) if precs else None,
                    mean_recall=round(float(np.mean(recs)), 2) if recs else None,
                    mean_f1=round(float(np.mean(f1s)), 2) if f1s else None,
                )

        return seg_summaries

    def _evaluate_boundary_refinement_isolation(
        self,
        dataset: BenchmarkDataset,
        sample_mode_predictions: Dict[str, Dict[str, List[EvaluationSpan]]],
    ) -> Optional[BoundaryRefinementAblationResult]:
        """
        Isolates boundary refinement by comparing identical candidates from
        unrefined MTER (System C) and refined MTER (System D).
        """
        unrefined_mode = "system_c_mter" if "system_c_mter" in sample_mode_predictions[dataset.samples[0].sample_id] else "mter_full"
        refined_mode = "system_d_mter_refined"

        has_unref = any(unrefined_mode in sample_mode_predictions[s.sample_id] for s in dataset.samples)
        has_ref = any(refined_mode in sample_mode_predictions[s.sample_id] for s in dataset.samples)

        if not (has_unref and has_ref):
            return None

        all_u_spans = []
        all_r_spans = []
        all_refs = []

        for sample in dataset.samples:
            u_spans = sample_mode_predictions[sample.sample_id].get(unrefined_mode, [])
            r_spans = sample_mode_predictions[sample.sample_id].get(refined_mode, [])
            all_u_spans.extend(u_spans)
            all_r_spans.extend(r_spans)
            all_refs.extend(sample.reference_spans)

        return evaluate_boundary_refinement_effect(
            sample_id=dataset.dataset_name,
            unrefined_spans=all_u_spans,
            refined_spans=all_r_spans,
            reference_spans=all_refs,
        )

    def _build_system_comparison(
        self,
        dataset: BenchmarkDataset,
        mode_summaries: Dict[str, AggregateModeMetrics],
        refinement_ablation: Optional[BoundaryRefinementAblationResult] = None,
        segment_summaries: Optional[Dict[str, AggregateSegmentMetrics]] = None,
    ) -> Optional[SystemComparisonSummary]:
        """
        Synthesizes comparative metrics across Systems A, B, C, D to answer the core research question.
        """
        sys_keys = ["system_a_transcript_only", "system_b_simple_aggregation", "system_c_mter", "system_d_mter_refined"]
        active_systems = [k for k in sys_keys if k in mode_summaries]

        # Check backward compatibility mode names
        if not active_systems:
            legacy_keys = ["baseline", "all_experts_no_mter", "mter_full"]
            active_systems = [k for k in legacy_keys if k in mode_summaries]

        if not active_systems:
            return None

        # Determine if MTER outperforms baseline and simple aggregation
        conclusions = {}
        c_mode = "system_c_mter" if "system_c_mter" in mode_summaries else "mter_full"
        a_mode = "system_a_transcript_only" if "system_a_transcript_only" in mode_summaries else "baseline"
        b_mode = "system_b_simple_aggregation" if "system_b_simple_aggregation" in mode_summaries else "all_experts_no_mter"

        if c_mode in mode_summaries and a_mode in mode_summaries:
            t_a = mode_summaries[a_mode].mean_tiou
            t_c = mode_summaries[c_mode].mean_tiou
            conclusions["mter_vs_transcript_tiou_gain"] = (
                round(t_c - t_a, 4) if (t_c is not None and t_a is not None) else None
            )
            s_a = mode_summaries[a_mode].mean_start_error_sec
            s_c = mode_summaries[c_mode].mean_start_error_sec
            conclusions["mter_vs_transcript_start_err_delta"] = (
                round(s_a - s_c, 4) if (s_a is not None and s_c is not None) else None
            )

        if c_mode in mode_summaries and b_mode in mode_summaries:
            t_b = mode_summaries[b_mode].mean_tiou
            t_c = mode_summaries[c_mode].mean_tiou
            conclusions["mter_vs_simple_agg_tiou_gain"] = (
                round(t_c - t_b, 4) if (t_c is not None and t_b is not None) else None
            )
            s_b = mode_summaries[b_mode].mean_start_error_sec
            s_c = mode_summaries[c_mode].mean_start_error_sec
            conclusions["mter_vs_simple_agg_start_err_delta"] = (
                round(s_b - s_c, 4) if (s_b is not None and s_c is not None) else None
            )

        if refinement_ablation:
            conclusions["boundary_refinement_num_changed"] = refinement_ablation.num_changed
            conclusions["boundary_refinement_tiou_delta"] = refinement_ablation.tiou_delta
            conclusions["boundary_refinement_start_err_reduction"] = refinement_ablation.start_error_reduction_sec
            conclusions["boundary_refinement_end_err_reduction"] = refinement_ablation.end_error_reduction_sec

        return SystemComparisonSummary(
            dataset_name=dataset.dataset_name,
            dataset_type=dataset.dataset_type,
            num_samples=len(dataset.samples),
            systems_evaluated=active_systems,
            system_metrics={k: mode_summaries[k] for k in active_systems if k in mode_summaries},
            refinement_ablation=refinement_ablation,
            segment_metrics=segment_summaries,
            research_conclusions=conclusions,
        )

    def generate_markdown_report(self, report: BenchmarkSummaryReport) -> str:
        """
        Formats evaluation metrics into an academic markdown report.
        """
        lines = [
            f"# ClipSense AI Benchmark Evaluation Report: {report.dataset_name}",
            "",
            f"- **Dataset Category:** `{report.dataset_type}`",
            f"- **Research Grade:** `{'YES' if report.is_research_report else 'NO (Verification Suite)'}`",
            f"- **Evaluated Systems / Modes:** {', '.join(f'`{m}`' for m in report.evaluated_modes)}",
            "",
        ]

        if not report.is_research_report:
            lines.extend([
                "> [!NOTE]",
                f"> This report reflects a **{report.dataset_type}** benchmark suite (development debug fixture or synthetic testbed).",
                "> Results serve for architectural verification and protocol regression testing; they must NOT be interpreted as peer-reviewed benchmark claims.",
                "",
            ])

        # Systems Comparison Scorecard
        lines.extend([
            "## 1. System Comparison Scorecard (Research Question Evaluation)",
            "",
            "**Research Question:** *Does conflict-sensitive temporal arbitration improve query-free long-form highlight selection compared with transcript-only selection and simple multimodal score aggregation?*",
            "",
            "| System | Description | Pred/GT/Match | Mean tIoU | Med tIoU | Start Err | End Err | Dur Err | Hit@1s | Hit@2s | F1@0.5 | Latency |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for mode_key, m in report.mode_summaries.items():
            counts_str = f"{m.total_predicted_spans} / {m.total_ground_truth_spans} / {m.total_matched_spans}"
            s_err = f"{m.mean_start_error_sec:.2f}s" if m.mean_start_error_sec is not None else "N/A"
            e_err = f"{m.mean_end_error_sec:.2f}s" if m.mean_end_error_sec is not None else "N/A"
            d_err = f"{m.mean_duration_error_sec:.2f}s" if m.mean_duration_error_sec is not None else "N/A"
            tiou_val = f"**{m.mean_tiou:.3f}**" if m.mean_tiou is not None else "N/A"
            med_tiou = f"{m.median_tiou:.3f}" if m.median_tiou is not None else "N/A"
            hit1 = f"{m.hit_rate_at_1s:.1f}%" if m.hit_rate_at_1s is not None else "N/A"
            hit2 = f"{m.hit_rate_at_2s:.1f}%" if m.hit_rate_at_2s is not None else "N/A"

            lines.append(
                f"| `{m.mode_name}` | {m.mode_description} | {counts_str} | "
                f"{tiou_val} | {med_tiou} | "
                f"{s_err} | {e_err} | {d_err} | "
                f"{hit1} | {hit2} | {m.f1_at_iou_05:.1f}% | {m.execution_latency_sec:.2f}s |"
            )

        lines.append("")

        # Boundary Refinement Isolation Ablation
        if report.refinement_ablation:
            b = report.refinement_ablation
            tiou_str = (
                f"{b.unrefined_mean_tiou:.4f} -> {b.refined_mean_tiou:.4f} (**Delta: {b.tiou_delta:+.4f}**)"
                if b.tiou_delta is not None and b.unrefined_mean_tiou is not None and b.refined_mean_tiou is not None
                else "N/A (no matched reference pairs)"
            )
            s_red = f"{b.start_error_reduction_sec:+.3f}s" if b.start_error_reduction_sec is not None else "N/A"
            e_red = f"{b.end_error_reduction_sec:+.3f}s" if b.end_error_reduction_sec is not None else "N/A"
            hit_str = (
                f"Hit@1s: {b.hit1_unrefined:.1f}% -> {b.hit1_refined:.1f}% | Hit@2s: {b.hit2_unrefined:.1f}% -> {b.hit2_refined:.1f}% | Hit@3s: {b.hit3_unrefined:.1f}% -> {b.hit3_refined:.1f}%"
                if b.hit1_unrefined is not None and b.hit1_refined is not None
                else "N/A"
            )
            lines.extend([
                "## 2. Boundary Refinement Isolation Ablation (System C vs. System D)",
                "",
                "Evaluates boundary adjustments in isolation on identical candidate highlights without ranking confounding:",
                "",
                f"- **Total Candidates Evaluated:** {b.num_candidates}",
                f"- **Boundaries Adjusted:** {b.num_changed} / {b.num_candidates} ({float(b.num_changed / max(1, b.num_candidates) * 100.0):.1f}%)",
                f"- **Mean Start Boundary Shift:** {b.mean_start_shift_sec:.3f}s",
                f"- **Mean End Boundary Shift:** {b.mean_end_shift_sec:.3f}s",
                f"- **Mean Duration Shift:** {b.mean_duration_shift_sec:.3f}s",
                f"- **Ground-Truth Relative tIoU Change:** {tiou_str}",
                f"- **Start Boundary Error Reduction:** {s_red}",
                f"- **End Boundary Error Reduction:** {e_red}",
                f"- **Tolerance Hit Rates (Unrefined vs. Refined):** {hit_str}",
                "",
            ])

        # Segment-Level Importance (if available)
        if report.segment_summaries:
            is_rhapsody = any(
                sm.mean_map is not None for sm in report.segment_summaries.values()
            )
            if is_rhapsody:
                lines.extend([
                    "## 3. Official Rhapsody Benchmark Metrics (COLM 2025 Protocol)",
                    "",
                    "> [!NOTE]",
                    "> Evaluated under the official Rhapsody protocol (100 equal-duration segments per episode, gt_threshold=0.5, max_n_gt=20).",
                    "> Continuous highlight candidate intervals are projected onto the 100 segments via maximum confidence-weighted overlap.",
                    "> Rhapsody replay annotations are viewer engagement curves mapped to 100 buckets, NOT precise human start/end boundaries.",
                    "",
                    "| System Mode | Samples | mAP | Hit Rate | Precision | Recall | F1 Score | Spearman Rho | Kendall Tau |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                ])
                for sm_name, sm in report.segment_summaries.items():
                    m_ap = f"**{sm.mean_map:.2f}%**" if sm.mean_map is not None else "N/A"
                    h_rate = f"{sm.mean_hit_rate:.1f}%" if sm.mean_hit_rate is not None else "N/A"
                    p_val = f"{sm.mean_precision:.1f}%" if sm.mean_precision is not None else "N/A"
                    r_val = f"{sm.mean_recall:.1f}%" if sm.mean_recall is not None else "N/A"
                    f1_val = f"{sm.mean_f1:.1f}%" if sm.mean_f1 is not None else "N/A"
                    lines.append(
                        f"| `{sm.mode_name}` | {sm.num_samples} | "
                        f"{m_ap} | {h_rate} | {p_val} | {r_val} | {f1_val} | "
                        f"{sm.mean_spearman_rho:.3f} | {sm.mean_kendall_tau:.3f} |"
                    )
            else:
                lines.extend([
                    "## 3. Continuous Segment-Level Importance Metrics (TVSum Protocol)",
                    "",
                    "Evaluates frame/segment salience curves against non-binary user importance ratings:",
                    "",
                    "| System Mode | Samples | Spearman Rho | Median Rho | Kendall Tau | Top-15% Overlap | Top-20% Overlap |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
                ])
                for sm_name, sm in report.segment_summaries.items():
                    lines.append(
                        f"| `{sm.mode_name}` | {sm.num_samples} | "
                        f"**{sm.mean_spearman_rho:.3f}** | {sm.median_spearman_rho:.3f} | "
                        f"{sm.mean_kendall_tau:.3f} | {sm.mean_top_15_hit_rate:.1f}% | {sm.mean_top_20_hit_rate:.1f}% |"
                    )
            lines.append("")

        return "\n".join(lines)

    def generate_html_report(self, report: BenchmarkSummaryReport) -> str:
        """
        Emits a standalone HTML report with rich styling and executive scorecard.
        """
        table_rows = []
        for mode_key, m in report.mode_summaries.items():
            tiou_html = f"{m.mean_tiou:.3f}" if m.mean_tiou is not None else "N/A"
            med_tiou_html = f"{m.median_tiou:.3f}" if m.median_tiou is not None else "N/A"
            s_err = f"{m.mean_start_error_sec:.2f}s" if m.mean_start_error_sec is not None else "N/A"
            e_err = f"{m.mean_end_error_sec:.2f}s" if m.mean_end_error_sec is not None else "N/A"
            hit1 = f"{m.hit_rate_at_1s:.1f}%" if m.hit_rate_at_1s is not None else "N/A"
            hit2 = f"{m.hit_rate_at_2s:.1f}%" if m.hit_rate_at_2s is not None else "N/A"
            table_rows.append(f"""
            <tr>
              <td class="font-mono text-xs">{m.mode_name}</td>
              <td class="text-xs text-slate-300">{m.mode_description}</td>
              <td class="text-center font-bold text-emerald-400">{tiou_html}</td>
              <td class="text-center">{med_tiou_html}</td>
              <td class="text-center">{s_err}</td>
              <td class="text-center">{e_err}</td>
              <td class="text-center">{hit1}</td>
              <td class="text-center">{hit2}</td>
              <td class="text-center">{m.f1_at_iou_05:.1f}%</td>
              <td class="text-center text-xs text-slate-400">{m.execution_latency_sec:.2f}s</td>
            </tr>
            """)

        rows_html = "\n".join(table_rows)

        refinement_html = ""
        if report.refinement_ablation:
            b = report.refinement_ablation
            tiou_delta_html = f"{b.tiou_delta:+.4f}" if b.tiou_delta is not None else "N/A"
            tiou_color = "text-emerald-400" if (b.tiou_delta is not None and b.tiou_delta >= 0) else ("text-amber-400" if b.tiou_delta is not None else "text-slate-400")
            refinement_html = f"""
            <div class="mt-8 bg-slate-800/80 border border-slate-700 rounded-xl p-6 shadow-xl">
              <h3 class="text-lg font-semibold text-sky-400 mb-2">Boundary Refinement Isolation Ablation</h3>
              <p class="text-xs text-slate-400 mb-4">Measures boundary shifts on identical candidates (System C vs. System D) without candidate ranking confounding.</p>
              <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-700/60">
                  <div class="text-xs text-slate-400">Candidates Adjusted</div>
                  <div class="text-lg font-bold text-white">{b.num_changed} / {b.num_candidates}</div>
                </div>
                <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-700/60">
                  <div class="text-xs text-slate-400">Mean Start Shift</div>
                  <div class="text-lg font-bold text-sky-300">{b.mean_start_shift_sec:.3f}s</div>
                </div>
                <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-700/60">
                  <div class="text-xs text-slate-400">Mean End Shift</div>
                  <div class="text-lg font-bold text-sky-300">{b.mean_end_shift_sec:.3f}s</div>
                </div>
                <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-700/60">
                  <div class="text-xs text-slate-400">tIoU Delta</div>
                  <div class="text-lg font-bold {tiou_color}">{tiou_delta_html}</div>
                </div>
              </div>
            </div>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ClipSense AI — Benchmark Evaluation Report</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen p-8">
  <div class="max-w-6xl mx-auto space-y-6">
    <div class="border-b border-slate-800 pb-4">
      <div class="flex items-center justify-between">
        <h1 class="text-2xl font-bold text-white tracking-tight">ClipSense AI — Benchmark Evaluation Report</h1>
        <span class="px-3 py-1 text-xs rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 font-mono">
          {report.dataset_name} ({report.dataset_type})
        </span>
      </div>
      <p class="text-sm text-slate-400 mt-1">Research Question: Does conflict-sensitive temporal arbitration improve highlight selection?</p>
    </div>

    <div class="bg-slate-900/80 border border-slate-800 rounded-xl overflow-hidden shadow-2xl">
      <div class="p-4 border-b border-slate-800 bg-slate-900">
        <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400">System Comparison Scorecard</h2>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-left text-sm border-collapse">
          <thead>
            <tr class="border-b border-slate-800 bg-slate-950/60 text-slate-400 text-xs font-semibold">
              <th class="py-3 px-4">System Mode</th>
              <th class="py-3 px-4">Description</th>
              <th class="py-3 px-4 text-center">Mean tIoU</th>
              <th class="py-3 px-4 text-center">Med tIoU</th>
              <th class="py-3 px-4 text-center">Start Err</th>
              <th class="py-3 px-4 text-center">End Err</th>
              <th class="py-3 px-4 text-center">Hit@1s</th>
              <th class="py-3 px-4 text-center">Hit@2s</th>
              <th class="py-3 px-4 text-center">F1@0.5</th>
              <th class="py-3 px-4 text-center">Latency</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-800/60">
            {rows_html}
          </tbody>
        </table>
      </div>
    </div>

    {refinement_html}

    <div class="text-xs text-slate-500 text-center pt-8">
      ClipSense AI B.Tech Minor Project • Academic Evaluation Runner (Week 11) • Generated deterministically
    </div>
  </div>
</body>
</html>
"""
        return html
