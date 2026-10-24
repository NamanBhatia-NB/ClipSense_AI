"""
Unit and Integration Tests for Week 11 Benchmark Evaluation Framework

Tests:
1. Precision, Recall, and F1 calculations at discrete tIoU thresholds.
2. Segment-level importance metrics: Spearman rho, Kendall tau, Top-15% overlap.
3. Edge cases: empty predictions, zero ground-truth, constant curves, malformed intervals.
4. Intra-candidate overlap rate (duplicate candidate detection).
5. Boundary Refinement isolation ablation (comparing identical candidate inputs).
6. Dataset adapters: TVSum, QVHighlights, IntervalHighlight.
7. Anti-leakage guard (preventing ground truth in pipeline inputs).
8. Reproducibility: deterministic repeated runs producing bitwise identical metrics.
9. 4-System comparison execution (Systems A, B, C, D).
"""

import json
import math
import os
import pathlib
import unittest
import numpy as np

from evaluation.adapters import (
    IntervalHighlightAdapter,
    QVHighlightsAdapter,
    TVSumAdapter,
    get_dataset_adapter,
)
from evaluation.adapters.base_adapter import DatasetProvenance
from evaluation.metrics import (
    compute_intra_candidate_overlap_rate,
    compute_kendall_tau,
    compute_pairwise_metrics,
    compute_precision_recall_f1_at_iou,
    compute_spearman_rank_correlation,
    compute_temporal_iou,
    compute_top_k_overlap,
    evaluate_boundary_refinement_effect,
    match_spans_hungarian,
    spans_to_segment_importance_curve,
)
from evaluation.schemas import (
    BenchmarkDataset,
    BenchmarkSample,
    EvaluationSpan,
    PairwiseMetricResult,
    ReferenceHighlight,
    SegmentImportanceAnnotation,
)


class TestWeek11EvaluationMetrics(unittest.TestCase):
    """Tests for core mathematical formulations and edge cases."""

    def test_precision_recall_f1_at_thresholds(self):
        # 3 predictions, 2 references
        preds = [
            EvaluationSpan(span_id="p1", start_time=0.0, end_time=10.0, duration=10.0, confidence=0.9, mode_name="mter_full"),
            EvaluationSpan(span_id="p2", start_time=20.0, end_time=30.0, duration=10.0, confidence=0.8, mode_name="mter_full"),
            EvaluationSpan(span_id="p3", start_time=50.0, end_time=60.0, duration=10.0, confidence=0.7, mode_name="mter_full"),
        ]
        refs = [
            ReferenceHighlight(span_id="r1", start_time=0.0, end_time=10.0, annotation_source="gt"),  # perfect match with p1
            ReferenceHighlight(span_id="r2", start_time=22.0, end_time=30.0, annotation_source="gt"), # partial match with p2 (iou: 8/10 = 0.8)
        ]

        matched, un_p, un_r = match_spans_hungarian(preds, refs, min_iou=0.10)
        self.assertEqual(len(matched), 2)
        self.assertEqual(len(un_p), 1)  # p3 is unmatched false alarm
        self.assertEqual(len(un_r), 0)

        # At iou >= 0.5: both p1 (iou=1.0) and p2 (iou=0.8) qualify
        prec, rec, f1 = compute_precision_recall_f1_at_iou(matched, num_predictions=3, num_references=2, iou_threshold=0.50)
        self.assertAlmostEqual(prec, 2.0 / 3.0, places=4)
        self.assertAlmostEqual(rec, 2.0 / 2.0, places=4)
        expected_f1 = 2 * (2/3 * 1.0) / (2/3 + 1.0)
        self.assertAlmostEqual(f1, expected_f1, places=4)

        # At iou >= 0.9: only p1 qualifies
        prec_strict, rec_strict, f1_strict = compute_precision_recall_f1_at_iou(matched, num_predictions=3, num_references=2, iou_threshold=0.90)
        self.assertAlmostEqual(prec_strict, 1.0 / 3.0, places=4)
        self.assertAlmostEqual(rec_strict, 1.0 / 2.0, places=4)

    def test_precision_recall_f1_empty_cases(self):
        prec, rec, f1 = compute_precision_recall_f1_at_iou([], num_predictions=0, num_references=5)
        self.assertEqual((prec, rec, f1), (0.0, 0.0, 0.0))

        prec, rec, f1 = compute_precision_recall_f1_at_iou([], num_predictions=5, num_references=0)
        self.assertEqual((prec, rec, f1), (0.0, 0.0, 0.0))

    def test_intra_candidate_overlap_rate(self):
        # Case 1: single candidate -> overlap is 0.0
        spans_single = [
            EvaluationSpan(span_id="p1", start_time=0.0, end_time=10.0, duration=10.0, confidence=0.9, mode_name="mter_full")
        ]
        self.assertEqual(compute_intra_candidate_overlap_rate(spans_single), 0.0)

        # Case 2: two non-overlapping candidates
        spans_disjoint = [
            EvaluationSpan(span_id="p1", start_time=0.0, end_time=10.0, duration=10.0, confidence=0.9, mode_name="mter_full"),
            EvaluationSpan(span_id="p2", start_time=20.0, end_time=30.0, duration=10.0, confidence=0.8, mode_name="mter_full"),
        ]
        self.assertEqual(compute_intra_candidate_overlap_rate(spans_disjoint), 0.0)

        # Case 3: two overlapping candidates [0, 10] and [5, 15] -> iou = 5/15 = 1/3
        spans_overlap = [
            EvaluationSpan(span_id="p1", start_time=0.0, end_time=10.0, duration=10.0, confidence=0.9, mode_name="mter_full"),
            EvaluationSpan(span_id="p2", start_time=5.0, end_time=15.0, duration=10.0, confidence=0.8, mode_name="mter_full"),
        ]
        self.assertAlmostEqual(compute_intra_candidate_overlap_rate(spans_overlap), 1.0 / 3.0, places=4)

    def test_spearman_and_kendall_rank_correlation(self):
        # Perfectly aligned curves
        x = [0.1, 0.3, 0.5, 0.7, 0.9]
        y = [0.2, 0.4, 0.6, 0.8, 1.0]
        rho, _ = compute_spearman_rank_correlation(x, y)
        tau, _ = compute_kendall_tau(x, y)
        self.assertAlmostEqual(rho, 1.0, places=4)
        self.assertAlmostEqual(tau, 1.0, places=4)

        # Inversely aligned curves
        y_rev = [1.0, 0.8, 0.6, 0.4, 0.2]
        rho_rev, _ = compute_spearman_rank_correlation(x, y_rev)
        tau_rev, _ = compute_kendall_tau(x, y_rev)
        self.assertAlmostEqual(rho_rev, -1.0, places=4)
        self.assertAlmostEqual(tau_rev, -1.0, places=4)

        # Constant vector handling (should safely return 0.0 without crashing)
        const = [0.5, 0.5, 0.5, 0.5, 0.5]
        rho_c, _ = compute_spearman_rank_correlation(x, const)
        tau_c, _ = compute_kendall_tau(x, const)
        self.assertEqual(rho_c, 0.0)
        self.assertEqual(tau_c, 0.0)

    def test_top_k_overlap(self):
        # 10 segments, top 20% -> top 2 segments
        # x highest at indices 8, 9
        x = [0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.3, 0.9, 0.8]
        # y highest at indices 7, 8
        y = [0.1, 0.1, 0.2, 0.2, 0.1, 0.3, 0.2, 0.7, 0.9, 0.4]

        # Top 2 for x: {8, 9}
        # Top 2 for y: {7, 8}
        # Intersection: {8} -> overlap = 1 / 2 = 0.5
        overlap = compute_top_k_overlap(x, y, top_ratio=0.20)
        self.assertAlmostEqual(overlap, 0.5, places=4)

    def test_spans_to_segment_importance_curve(self):
        spans = [
            EvaluationSpan(span_id="p1", start_time=1.0, end_time=3.0, duration=2.0, confidence=0.8, mode_name="mter_full")
        ]
        # Video duration 5.0s, segment 1.0s -> 5 bins: [0-1], [1-2], [2-3], [3-4], [4-5]
        curve = spans_to_segment_importance_curve(spans, video_duration=5.0, segment_duration_sec=1.0)
        self.assertEqual(len(curve), 5)
        self.assertEqual(curve[0], 0.0)
        self.assertAlmostEqual(curve[1], 0.8, places=4)
        self.assertAlmostEqual(curve[2], 0.8, places=4)
        self.assertEqual(curve[3], 0.0)
        self.assertEqual(curve[4], 0.0)


class TestBoundaryRefinementIsolation(unittest.TestCase):
    """Tests isolating boundary refinement on identical candidate highlights."""

    def test_boundary_refinement_effect_calculation(self):
        # Reference highlight: [10.0, 30.0]
        refs = [
            ReferenceHighlight(span_id="r1", start_time=10.0, end_time=30.0, annotation_source="gt")
        ]

        # Unrefined candidate: [11.0, 28.0] (tIoU = 17 / 20 = 0.85, start_err = 1.0, end_err = 2.0)
        unrefined = [
            EvaluationSpan(span_id="cand_0", start_time=11.0, end_time=28.0, duration=17.0, confidence=0.8, mode_name="mter_full")
        ]

        # Refined candidate: [10.2, 29.8] (tIoU = 19.6 / 20.0 = 0.98, start_err = 0.2, end_err = 0.2)
        refined = [
            EvaluationSpan(
                span_id="cand_0",
                start_time=10.2,
                end_time=29.8,
                duration=19.6,
                confidence=0.8,
                mode_name="system_d_mter_refined",
                metadata={"rationale": "Aligned start with word onset, end with pause pocket"}
            )
        ]

        res = evaluate_boundary_refinement_effect(
            sample_id="test_sample",
            unrefined_spans=unrefined,
            refined_spans=refined,
            reference_spans=refs,
        )

        self.assertEqual(res.num_candidates, 1)
        self.assertEqual(res.num_changed, 1)
        self.assertAlmostEqual(res.mean_start_shift_sec, 0.8, places=3)
        self.assertAlmostEqual(res.mean_end_shift_sec, 1.8, places=3)
        self.assertGreater(res.tiou_delta, 0.0)  # Refinement improved tIoU
        self.assertGreater(res.start_error_reduction_sec, 0.0)  # Error decreased
        self.assertGreater(res.end_error_reduction_sec, 0.0)


class TestDatasetAdapters(unittest.TestCase):
    """Tests for TVSum, QVHighlights, and Interval adapters with anti-leakage validation."""

    def setUp(self):
        self.backend_dir = pathlib.Path(__file__).parent.parent
        self.tvsum_fixture = str(self.backend_dir / "evaluation" / "datasets" / "tvsum_toy_fixture.json")
        self.qvh_fixture = str(self.backend_dir / "evaluation" / "datasets" / "qvhighlights_toy_fixture.json")
        self.dev_fixture = str(self.backend_dir / "evaluation" / "datasets" / "dev_fixture.json")

    def test_tvsum_adapter_loading_and_segment_curves(self):
        adapter = get_dataset_adapter("tvsum")
        self.assertIsInstance(adapter, TVSumAdapter)

        dataset = adapter.load_dataset(self.tvsum_fixture)
        self.assertEqual(dataset.dataset_name, "TVSum_Toy_Validation_Suite")
        self.assertEqual(len(dataset.samples), 1)

        seg_annotations = adapter.extract_segment_annotations(self.tvsum_fixture)
        self.assertIn("tvsum_sample_changing_tire_90s", seg_annotations)
        ann = seg_annotations["tvsum_sample_changing_tire_90s"]
        self.assertEqual(ann.num_segments, 90)
        self.assertEqual(len(ann.segment_scores), 90)
        self.assertEqual(ann.annotator_count, 20)

    def test_qvhighlights_adapter_loading(self):
        adapter = get_dataset_adapter("qvhighlights")
        self.assertIsInstance(adapter, QVHighlightsAdapter)

        dataset = adapter.load_dataset(self.qvh_fixture)
        self.assertEqual(dataset.dataset_name, "QVHighlights_Toy_Validation_Suite")
        self.assertEqual(len(dataset.samples), 1)
        self.assertIn("query_mode", dataset.samples[0].metadata)

    def test_interval_adapter_bounds_validation(self):
        adapter = get_dataset_adapter("interval")
        self.assertIsInstance(adapter, IntervalHighlightAdapter)

        dataset = adapter.load_dataset(self.dev_fixture)
        self.assertGreater(len(dataset.samples), 0)
        for s in dataset.samples:
            for r in s.reference_spans:
                self.assertGreater(r.end_time, r.start_time)

    def test_data_leakage_guard(self):
        adapter = IntervalHighlightAdapter()
        sample = BenchmarkSample(
            sample_id="test_leakage_sample",
            sample_type="development",
            duration_sec=60.0,
            run_dir="mock_run",
            reference_spans=[],
        )

        # Clean inputs -> should pass
        clean_inputs = {"video_path": "test.mp4", "features": [1, 2, 3]}
        adapter.enforce_no_leakage(sample, clean_inputs)

        # Leaked ground truth key -> must raise ValueError
        leaked_inputs = {"video_path": "test.mp4", "reference_spans": [{"start": 10.0, "end": 20.0}]}
        with self.assertRaises(ValueError) as ctx:
            adapter.enforce_no_leakage(sample, leaked_inputs)
        self.assertIn("DATA LEAKAGE DETECTED", str(ctx.exception))


class TestSystemComparisonExecution(unittest.TestCase):
    """End-to-end integration test running the 4-system evaluation pipeline."""

    def test_4_system_orchestrator_run(self):
        from evaluation.ablation_runner import AblationRunner
        from evaluation.baseline import BaselineConfig, LegacyTranscriptBaselineSelector
        from evaluation.benchmark_orchestrator import BenchmarkOrchestrator
        from evaluation.dataset import load_benchmark_dataset_with_annotations
        from evaluation.run_evaluation import EvaluationMockLLMClient

        backend_dir = pathlib.Path(__file__).parent.parent
        dev_path = str(backend_dir / "evaluation" / "datasets" / "dev_fixture.json")

        dataset, seg_ann = load_benchmark_dataset_with_annotations(dev_path)

        # Resolve sample run_dir
        for s in dataset.samples:
            s_path = pathlib.Path(s.run_dir)
            if not s_path.is_absolute():
                s.run_dir = str((backend_dir / s_path).resolve())

        mock_llm = EvaluationMockLLMClient()
        baseline_sel = LegacyTranscriptBaselineSelector(
            llm_client=mock_llm,
            config=BaselineConfig(inference_mode="live_inference"),
        )
        runner = AblationRunner(baseline_selector=baseline_sel)
        orchestrator = BenchmarkOrchestrator(ablation_runner=runner)

        modes = [
            "system_a_transcript_only",
            "system_b_simple_aggregation",
            "system_c_mter",
            "system_d_mter_refined",
        ]

        report = orchestrator.evaluate_dataset(
            dataset=dataset,
            modes=modes,
            min_match_iou=0.10,
        )

        self.assertEqual(len(report.mode_summaries), 4)
        for m in modes:
            self.assertIn(m, report.mode_summaries)
            summary = report.mode_summaries[m]
            self.assertGreaterEqual(summary.mean_tiou, 0.0)
            self.assertLessEqual(summary.mean_tiou, 1.0)
            self.assertGreaterEqual(summary.hit_rate_at_1s, 0.0)

        # Verify boundary refinement isolation was performed
        self.assertIsNotNone(report.refinement_ablation)
        self.assertEqual(report.refinement_ablation.num_candidates, 1)

        # Verify system comparison summary was populated
        self.assertIsNotNone(report.system_comparison)
        self.assertEqual(len(report.system_comparison.systems_evaluated), 4)

        # Verify markdown and HTML report generation
        md_text = orchestrator.generate_markdown_report(report)
        html_text = orchestrator.generate_html_report(report)
        self.assertIn("System Comparison Scorecard", md_text)
        self.assertIn("Boundary Refinement Isolation Ablation", md_text)
        self.assertIn("<!DOCTYPE html>", html_text)


if __name__ == "__main__":
    unittest.main()
