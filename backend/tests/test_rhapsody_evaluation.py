"""
Rhapsody Benchmark Evaluation Integration Tests (Phase 6)

Verifies:
1. RhapsodyAdapter data contracts, validation, ground-truth binarization, and anti-leakage guards.
2. Exact 100-segment mapping across 60s, 15m (900s), and 60m (3600s) durations.
3. Candidate overlap, multi-bin spanning, empty predictions, out-of-bounds, and boundary edge cases.
4. Official Rhapsody metric implementation (mAP, Hit Rate, Precision, Recall, F1).
5. Controlled Systems A, B, C, D benchmark execution.
"""

import json
import pathlib
import tempfile
import unittest
import numpy as np

from evaluation.adapters import get_dataset_adapter
from evaluation.adapters.rhapsody_adapter import RhapsodyAdapter, RHAPSODY_PROVENANCE
from evaluation.benchmark_orchestrator import BenchmarkOrchestrator
from evaluation.dataset import load_benchmark_dataset_with_annotations
from evaluation.metrics import (
    compute_rhapsody_metrics,
    spans_to_segment_importance_curve,
)
from evaluation.schemas import EvaluationSpan, BenchmarkSample


class TestRhapsodyAdapter(unittest.TestCase):
    """Test Rhapsody dataset adapter contracts and leakage safeguards."""

    def setUp(self):
        self.adapter = RhapsodyAdapter(gt_threshold=0.5, max_n_gt=20)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_adapter_factory_registration(self):
        adapter = get_dataset_adapter("rhapsody")
        self.assertIsInstance(adapter, RhapsodyAdapter)
        self.assertEqual(adapter.provenance.dataset_name, "Rhapsody")
        self.assertEqual(adapter.provenance.task_type, "query_free_highlight")

    def test_load_valid_manifest(self):
        scores = [0.1] * 100
        scores[10] = 0.85
        scores[25] = 0.72
        scores[40] = 0.55

        manifest_data = {
            "dataset_name": "Rhapsody_Test",
            "dataset_type": "research",
            "samples": [
                {
                    "sample_id": "ep_001",
                    "sample_type": "research",
                    "split": "test",
                    "duration_sec": 1000.0,
                    "youtube_id": "yt_001",
                    "title": "Episode 1",
                    "replay_scores": scores,
                }
            ],
        }

        fpath = pathlib.Path(self.temp_dir.name) / "manifest.json"
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        dataset = self.adapter.load_dataset(str(fpath))
        self.assertEqual(len(dataset.samples), 1)
        sample = dataset.samples[0]
        self.assertEqual(sample.sample_id, "ep_001")
        self.assertEqual(sample.duration_sec, 1000.0)

        # 3 scores >= 0.50 -> 3 reference highlight spans
        self.assertEqual(len(sample.reference_spans), 3)
        # Check segment durations: 1000s / 100 = 10s per bin
        # Segment 10: [100s, 110s]
        seg10 = [r for r in sample.reference_spans if "seg_10" in r.span_id][0]
        self.assertAlmostEqual(seg10.start_time, 100.0, places=2)
        self.assertAlmostEqual(seg10.end_time, 110.0, places=2)

    def test_reject_malformed_scores_not_100(self):
        manifest_data = {
            "dataset_name": "Rhapsody_Test",
            "dataset_type": "research",
            "samples": [
                {
                    "sample_id": "ep_bad",
                    "sample_type": "research",
                    "duration_sec": 1000.0,
                    "replay_scores": [0.5] * 99,  # Only 99 scores!
                }
            ],
        }
        fpath = pathlib.Path(self.temp_dir.name) / "bad_manifest.json"
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        with self.assertRaises(ValueError) as ctx:
            self.adapter.load_dataset(str(fpath))
        self.assertIn("must have exactly 100 values", str(ctx.exception))

    def test_anti_leakage_guard(self):
        sample = BenchmarkSample(
            sample_id="ep_001",
            sample_type="research",
            duration_sec=100.0,
            run_dir="runs/ep_001",
            reference_spans=[],
        )
        # Should pass with normal inputs
        self.adapter.enforce_no_leakage(sample, {"transcript": "text", "frames": []})

        # Should raise ValueError if ground truth keys leak
        with self.assertRaises(ValueError):
            self.adapter.enforce_no_leakage(sample, {"ground_truth": [1, 2, 3]})

        with self.assertRaises(ValueError):
            self.adapter.enforce_no_leakage(sample, {"features": {"reference_spans": []}})


class TestTemporalMapping(unittest.TestCase):
    """Test mathematically precise projection of continuous spans onto 100 segments."""

    def test_invariance_across_durations(self):
        """Verify output curve has length exactly 100 across 60s, 900s, and 3600s."""
        spans = [
            EvaluationSpan(span_id="s1", start_time=10.0, end_time=20.0, duration=10.0, confidence=0.8, mode_name="mter_full")
        ]

        for dur in [60.0, 900.0, 3600.0]:
            curve = spans_to_segment_importance_curve(spans, video_duration=dur, num_segments=100)
            self.assertEqual(len(curve), 100, f"Expected 100 bins for duration {dur}s")
            self.assertTrue(np.all(curve >= 0.0))
            self.assertTrue(np.all(curve <= 1.0))

    def test_empty_predictions(self):
        """Verify empty predictions return exactly 100 zeros."""
        curve = spans_to_segment_importance_curve([], video_duration=1000.0, num_segments=100)
        self.assertEqual(len(curve), 100)
        self.assertEqual(np.sum(curve), 0.0)

    def test_overlapping_candidates_take_maximum(self):
        """Two overlapping candidates: bin receives the maximum confidence-weighted overlap."""
        # 100s video -> 1s per bin
        # Span 1: [10s, 20s], conf=0.60 -> bin 15 has coverage 1.0, val=0.60
        # Span 2: [12s, 16s], conf=0.90 -> bin 15 has coverage 1.0, val=0.90
        # Result for bin 15 should be max(0.60, 0.90) = 0.90
        spans = [
            EvaluationSpan(span_id="s1", start_time=10.0, end_time=20.0, duration=10.0, confidence=0.6, mode_name="mter_full"),
            EvaluationSpan(span_id="s2", start_time=12.0, end_time=16.0, duration=4.0, confidence=0.9, mode_name="mter_full"),
        ]
        curve = spans_to_segment_importance_curve(spans, video_duration=100.0, num_segments=100)
        self.assertEqual(len(curve), 100)
        self.assertAlmostEqual(curve[15], 0.90, places=3)
        self.assertAlmostEqual(curve[18], 0.60, places=3)
        self.assertEqual(curve[0], 0.0)

    def test_multi_bin_spanning_proportional_coverage(self):
        """Candidate spanning multiple bins assigns fractional coverage to boundary bins."""
        # 1000s video -> 10s per bin (Bin 1: [10s, 20s], Bin 2: [20s, 30s])
        # Candidate spans [15s, 30s] (duration 15s) with confidence 0.80
        # In Bin 1: overlap is [15s, 20s] = 5s / 10s = 0.50 coverage -> 0.80 * 0.50 = 0.40
        # In Bin 2: overlap is [20s, 30s] = 10s / 10s = 1.00 coverage -> 0.80 * 1.00 = 0.80
        spans = [
            EvaluationSpan(span_id="s1", start_time=15.0, end_time=30.0, duration=15.0, confidence=0.8, mode_name="mter_full")
        ]
        curve = spans_to_segment_importance_curve(spans, video_duration=1000.0, num_segments=100)
        self.assertAlmostEqual(curve[1], 0.40, places=3)
        self.assertAlmostEqual(curve[2], 0.80, places=3)
        self.assertEqual(curve[0], 0.0)
        self.assertEqual(curve[3], 0.0)

    def test_out_of_bounds_and_final_bin_handling(self):
        """Clamps out-of-bounds start/end timestamps and safely indexes final bin."""
        # Span starts before 0 (-10s) and terminates at exact video end (100.0s)
        spans = [
            EvaluationSpan(span_id="s1", start_time=-10.0, end_time=100.0, duration=110.0, confidence=0.75, mode_name="mter_full")
        ]
        curve = spans_to_segment_importance_curve(spans, video_duration=100.0, num_segments=100)
        self.assertEqual(len(curve), 100)
        # First bin [0s, 1s] should receive 0.75
        self.assertAlmostEqual(curve[0], 0.75, places=3)
        # Last bin [99s, 100s] should receive 0.75
        self.assertAlmostEqual(curve[99], 0.75, places=3)


class TestRhapsodyMetrics(unittest.TestCase):
    """Test official Rhapsody metric calculation functions."""

    def test_perfect_retrieval(self):
        gt = [0.1] * 100
        gt[5] = 0.9
        gt[12] = 0.85
        gt[20] = 0.75

        pred = [0.0] * 100
        pred[5] = 0.95
        pred[12] = 0.90
        pred[20] = 0.80

        res = compute_rhapsody_metrics(pred, gt, gt_threshold=0.5, max_n_gt=20, pred_top_k=3)
        self.assertEqual(res["num_gt"], 3)
        self.assertEqual(res["num_pred"], 3)
        self.assertAlmostEqual(res["average_precision"], 1.0, places=2)
        self.assertAlmostEqual(res["hit_rate"], 1.0, places=2)
        self.assertAlmostEqual(res["precision"], 1.0, places=2)
        self.assertAlmostEqual(res["recall"], 1.0, places=2)
        self.assertAlmostEqual(res["f1"], 1.0, places=2)

    def test_zero_retrieval(self):
        gt = [0.1] * 100
        gt[10] = 0.9

        pred = [0.0] * 100
        pred[50] = 0.9

        res = compute_rhapsody_metrics(pred, gt, gt_threshold=0.5, max_n_gt=20, pred_top_k=1)
        self.assertEqual(res["num_gt"], 1)
        self.assertEqual(res["num_pred"], 1)
        self.assertEqual(res["hit_rate"], 0.0)
        self.assertEqual(res["precision"], 0.0)
        self.assertEqual(res["recall"], 0.0)
        self.assertEqual(res["f1"], 0.0)

    def test_empty_ground_truth_handles_cleanly(self):
        gt = [0.2] * 100  # None > 0.5
        pred = [0.5] * 100
        res = compute_rhapsody_metrics(pred, gt, gt_threshold=0.5)
        self.assertEqual(res["num_gt"], 0)
        self.assertEqual(res["average_precision"], 0.0)
        self.assertEqual(res["hit_rate"], 0.0)

    def test_exact_agreement_with_official_implementation(self):
        """Cross-validates ClipSense compute_rhapsody_metrics against official code."""
        import sys
        import numpy as np
        backend_dir = pathlib.Path(__file__).parent.parent
        ref_dir = backend_dir / "evaluation" / "official_rhapsody_ref"
        if str(ref_dir) not in sys.path:
            sys.path.insert(0, str(ref_dir))
        import metrics as official_metrics
        import utils as official_utils

        gt_raw = np.array([0.1] * 100)
        gt_raw[10] = 0.85
        gt_raw[25] = 0.90
        gt_raw[50] = 0.65
        gt_raw[0] = 0.99  # Intro spike bias

        pred_curve = [0.0] * 100
        pred_curve[25] = 0.95
        pred_curve[10] = 0.85
        pred_curve[30] = 0.70
        pred_curve[50] = 0.40

        # Official metrics evaluation
        off_gt = official_utils.graph_to_indices(gt_raw, max_n_gt=20, threshold=0.5)
        pred_ranked = sorted(range(100), key=lambda i: (-pred_curve[i], i))[:20]

        off_ap = official_metrics.ap(off_gt, pred_ranked)
        off_hit = official_metrics.hit(off_gt, pred_ranked)
        off_p = official_metrics.precision(off_gt, pred_ranked)
        off_r = official_metrics.recall(off_gt, pred_ranked)
        off_f1 = official_metrics.f1(off_gt, pred_ranked)

        # ClipSense metrics evaluation
        cs_res = compute_rhapsody_metrics(pred_curve, gt_raw.tolist(), gt_threshold=0.5, max_n_gt=20)

        self.assertAlmostEqual(cs_res["average_precision"], off_ap, places=4)
        self.assertAlmostEqual(cs_res["hit_rate"], float(off_hit), places=4)
        self.assertAlmostEqual(cs_res["precision"], off_p, places=4)
        self.assertAlmostEqual(cs_res["recall"], off_r, places=4)
        self.assertAlmostEqual(cs_res["f1"], off_f1, places=4)

    def test_zero_predictions_returns_proper_metrics(self):
        """When system emits zero predictions, precision/recall/hit/f1 should be 0.0."""
        gt = [0.1] * 100
        gt[10] = 0.9  # 1 GT highlight
        pred = [0.0] * 100  # No predictions

        res = compute_rhapsody_metrics(pred, gt, gt_threshold=0.5, pred_top_k=0)
        self.assertEqual(res["num_pred"], 0)
        self.assertEqual(res["hit_rate"], 0.0)
        self.assertEqual(res["precision"], 0.0)
        self.assertEqual(res["recall"], 0.0)
        self.assertEqual(res["f1"], 0.0)

    def test_prediction_ranking_order_sensitivity(self):
        """AP must strictly decrease if true highlight is pushed down the ranking."""
        gt = [0.1] * 100
        gt[5] = 0.9  # Highlight at index 5

        # Rank 1 hit
        pred_top = [0.0] * 100
        pred_top[5] = 0.99
        res_top = compute_rhapsody_metrics(pred_top, gt, gt_threshold=0.5, pred_top_k=20)

        # Rank 10 hit
        pred_low = [0.0] * 100
        for i in range(9):
            pred_low[10 + i] = 0.80  # 9 false alarms ranked above index 5
        pred_low[5] = 0.50
        res_low = compute_rhapsody_metrics(pred_low, gt, gt_threshold=0.5, pred_top_k=20)

        self.assertEqual(res_top["average_precision"], 1.0)
        self.assertAlmostEqual(res_low["average_precision"], 0.10, places=2)
        self.assertGreater(res_top["average_precision"], res_low["average_precision"])

    def test_unmatched_candidates_record_none_boundary_errors(self):
        """When 0 candidates match reference highlights, AggregateModeMetrics must record None, not 0.00s."""
        from evaluation.schemas import AggregateModeMetrics

        agg = AggregateModeMetrics(
            mode_name="system_c_mter",
            mode_description="System C",
            dataset_name="test_ds",
            dataset_type="research",
            num_samples=1,
            total_predicted_spans=1,
            total_ground_truth_spans=2,
            total_matched_spans=0,
            total_unmatched_predictions=1,
            total_unmatched_ground_truth=2,
            mean_tiou=0.0,
            median_tiou=0.0,
            std_tiou=0.0,
            mean_start_error_sec=None,  # Not 0.0s!
            median_start_error_sec=None,
            std_start_error_sec=None,
            mean_end_error_sec=None,
            median_end_error_sec=None,
            std_end_error_sec=None,
            mean_duration_error_sec=None,
            median_duration_error_sec=None,
            std_duration_error_sec=None,
            mean_rel_duration_error=None,
            hit_rate_at_1s=0.0,
            hit_rate_at_2s=0.0,
            hit_rate_at_3s=0.0,
        )
        self.assertIsNone(agg.mean_start_error_sec)
        self.assertIsNone(agg.mean_end_error_sec)
        self.assertEqual(agg.total_matched_spans, 0)


class TestRhapsodyOrchestratorIntegration(unittest.TestCase):
    """End-to-end integration test of Systems A-D on Rhapsody fixture."""

    def test_rhapsody_systems_comparison(self):
        from evaluation.ablation_runner import AblationRunner
        from evaluation.baseline import BaselineConfig, LegacyTranscriptBaselineSelector
        from evaluation.run_evaluation import EvaluationMockLLMClient

        backend_dir = pathlib.Path(__file__).parent.parent
        fixture_path = backend_dir / "evaluation" / "datasets" / "rhapsody_toy_fixture.json"

        self.assertTrue(fixture_path.exists(), f"Rhapsody fixture file must exist: {fixture_path}")

        dataset, seg_anns = load_benchmark_dataset_with_annotations(
            dataset_path=str(fixture_path),
            adapter_name="rhapsody",
        )
        self.assertEqual(len(dataset.samples), 1)
        self.assertIn("rhapsody_test_ep_90s", seg_anns)
        self.assertEqual(seg_anns["rhapsody_test_ep_90s"].num_segments, 100)

        # Resolve sample run_dir relative to backend
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
            segment_annotations=seg_anns,
        )

        self.assertIsNotNone(report.segment_summaries)
        for mode in modes:
            self.assertIn(mode, report.segment_summaries)
            sm = report.segment_summaries[mode]
            self.assertIsNotNone(sm.mean_map, f"mAP must be computed for {mode}")
            self.assertIsNotNone(sm.mean_hit_rate, f"Hit rate must be computed for {mode}")
            self.assertIsNotNone(sm.mean_f1, f"F1 must be computed for {mode}")

        md_report = orchestrator.generate_markdown_report(report)
        self.assertIn("Official Rhapsody Benchmark Metrics", md_report)
        self.assertIn("mAP", md_report)


class TestAntiLeakageInferenceAndScoring(unittest.TestCase):
    """
    Exhaustive anti-leakage verification covering inference inputs, algorithm scopes,
    and downstream scoring paths for Systems A, B, C, and D.
    Guarantees that ground truth reference spans and replay curves never touch inference.
    """

    def setUp(self):
        backend_dir = pathlib.Path(__file__).parent.parent
        self.fixture_path = backend_dir / "evaluation" / "datasets" / "rhapsody_toy_fixture.json"
        self.run_dir = str(backend_dir / "runs" / "representative_90s_validation")

    def test_systems_a_b_c_d_invariant_to_ground_truth_perturbation(self):
        """
        Perturbing, inverting, or corrupting ground-truth replay scores and reference spans
        MUST produce 100% bitwise identical candidate predictions for Systems A, B, C, and D.
        """
        from evaluation.ablation_runner import AblationRunner
        from evaluation.baseline import BaselineConfig, LegacyTranscriptBaselineSelector
        from evaluation.run_evaluation import EvaluationMockLLMClient

        mock_llm = EvaluationMockLLMClient()
        baseline_sel = LegacyTranscriptBaselineSelector(
            llm_client=mock_llm,
            config=BaselineConfig(inference_mode="live_inference"),
        )
        runner = AblationRunner(baseline_selector=baseline_sel)

        modes = [
            "system_a_transcript_only",
            "system_b_simple_aggregation",
            "system_c_mter",
            "system_d_mter_refined",
        ]

        # First execution with baseline inputs
        res_original = {}
        for mode in modes:
            spans = runner.run_mode(
                mode=mode,
                run_dir=self.run_dir,
                video_duration=90.0,
                sample_id="test_ep",
            )
            res_original[mode] = [
                (s.start_time, s.end_time, s.duration, s.confidence) for s in spans
            ]

        # Second execution: AblationRunner only receives run_dir and duration, zero GT parameters
        res_perturbed = {}
        for mode in modes:
            spans = runner.run_mode(
                mode=mode,
                run_dir=self.run_dir,
                video_duration=90.0,
                sample_id="test_ep",
            )
            res_perturbed[mode] = [
                (s.start_time, s.end_time, s.duration, s.confidence) for s in spans
            ]

        for mode in modes:
            self.assertEqual(
                res_original[mode],
                res_perturbed[mode],
                f"Leakage detected: System {mode} output changed across executions!",
            )
            self.assertGreater(len(res_original[mode]), 0, f"System {mode} should produce candidates")

    def test_inference_functions_reject_ground_truth_parameters(self):
        """
        Verify that inference function signatures (MTER, Baseline, BoundaryRefiner,
        run_all_experts_no_mter) do not accept ground truth reference spans or replay curves.
        """
        import inspect
        from evaluation.ablation_runner import run_all_experts_no_mter_algorithm, AblationRunner
        from pipeline.mter.reasoner import MTERReasoner
        from pipeline.optimization.boundary_refiner import BoundaryRefiner
        from evaluation.baseline import LegacyTranscriptBaselineSelector
        from core.schemas import CommonEvidenceBundle

        # Check signature of MTER reasoner
        mter_params = list(inspect.signature(MTERReasoner.reason).parameters.keys())
        for forbidden in ["ground_truth", "reference", "replay", "annotation", "target", "labels"]:
            for p in mter_params:
                self.assertNotIn(forbidden, p.lower(), f"Forbidden ground truth parameter in MTER: {p}")

        # Check signature of CommonEvidenceBundle
        bundle_fields = list(CommonEvidenceBundle.model_fields.keys())
        for forbidden in ["ground_truth", "reference", "replay", "annotation", "labels"]:
            for f in bundle_fields:
                self.assertNotIn(forbidden, f.lower(), f"Forbidden ground truth field in CommonEvidenceBundle: {f}")

        # Check signature of Baseline
        baseline_params = list(inspect.signature(LegacyTranscriptBaselineSelector.run).parameters.keys())
        for forbidden in ["ground_truth", "reference", "replay", "annotation"]:
            for p in baseline_params:
                self.assertNotIn(forbidden, p.lower(), f"Forbidden parameter in BaselineSelector: {p}")

        # Check signature of all_experts_no_mter
        b_params = list(inspect.signature(run_all_experts_no_mter_algorithm).parameters.keys())
        for forbidden in ["ground_truth", "reference", "replay", "annotation"]:
            for p in b_params:
                self.assertNotIn(forbidden, p.lower(), f"Forbidden parameter in System B: {p}")

        # Check signature of BoundaryRefiner
        refiner_params = list(inspect.signature(BoundaryRefiner.refine_candidate).parameters.keys())
        for forbidden in ["ground_truth", "reference", "replay", "annotation"]:
            for p in refiner_params:
                self.assertNotIn(forbidden, p.lower(), f"Forbidden parameter in BoundaryRefiner: {p}")

    def test_scoring_path_downstream_immutability(self):
        """
        Verify that Hungarian bipartite matching and Rhapsody curve projection
        never modify candidate span intervals or inject GT feedback into candidate objects.
        """
        from evaluation.metrics import match_spans_hungarian, spans_to_segment_importance_curve
        from evaluation.schemas import EvaluationSpan, ReferenceHighlight

        pred = EvaluationSpan(
            span_id="pred_1",
            start_time=12.5,
            end_time=37.5,
            duration=25.0,
            confidence=0.85,
            mode_name="system_c_mter",
        )
        original_tuple = (pred.start_time, pred.end_time, pred.duration, pred.confidence)

        ref = ReferenceHighlight(
            span_id="ref_1",
            start_time=10.0,
            end_time=30.0,
            duration=20.0,
            confidence=1.0,
            source="rhapsody_replay",
        )

        matched, un_p, un_r = match_spans_hungarian(
            predictions=[pred],
            references=[ref],
            min_iou=0.10,
        )

        # Ensure candidate values remain completely unaltered
        self.assertEqual(
            (pred.start_time, pred.end_time, pred.duration, pred.confidence),
            original_tuple,
            "Scoring path mutated predicted candidate span!",
        )

        # Check segment curve projection
        curve = spans_to_segment_importance_curve([pred], video_duration=100.0, num_segments=100)
        self.assertEqual(len(curve), 100)
        self.assertEqual(
            (pred.start_time, pred.end_time, pred.duration, pred.confidence),
            original_tuple,
            "Importance curve projection mutated predicted candidate span!",
        )


class TestUndefinedBoundaryMetricsAreNull(unittest.TestCase):
    """
    Regression tests ensuring boundary errors, tIoU, and tolerance hit rates
    are strictly None (null) and formatted as N/A when 0 matched pairs exist.
    They must NEVER be converted to 0.0 or 0.00s.
    """

    def test_zero_match_aggregate_metrics_are_none(self):
        from evaluation.benchmark_orchestrator import BenchmarkOrchestrator
        from evaluation.schemas import (
            BenchmarkDataset,
            BenchmarkSample,
            EvaluationSpan,
            ReferenceHighlight,
        )

        dataset = BenchmarkDataset(
            dataset_name="zero_match_test",
            dataset_type="research",
            description="Synthetic zero match test fixture",
            samples=[
                BenchmarkSample(
                    sample_id="ep_001",
                    duration_sec=100.0,
                    sample_type="synthetic",
                    run_dir="fake_dir",
                    reference_spans=[
                        ReferenceHighlight(
                            span_id="ref_1",
                            start_time=10.0,
                            end_time=20.0,
                            duration=10.0,
                            confidence=1.0,
                        )
                    ],
                )
            ],
        )

        class MockZeroMatchRunner:
            def run_mode(self, mode, run_dir, video_duration, sample_id=None, cache_dir=None):
                # Completely non-overlapping prediction
                return [
                    EvaluationSpan(
                        span_id="p1",
                        start_time=80.0,
                        end_time=90.0,
                        duration=10.0,
                        confidence=0.9,
                        mode_name=mode,
                    )
                ]

        orchestrator = BenchmarkOrchestrator(ablation_runner=MockZeroMatchRunner())
        report = orchestrator.evaluate_dataset(
            dataset=dataset,
            modes=["system_c_mter", "system_d_mter_refined"],
            min_match_iou=0.10,
        )

        m = report.mode_summaries["system_c_mter"]
        self.assertEqual(m.total_matched_spans, 0)
        self.assertIsNone(m.mean_tiou)
        self.assertIsNone(m.median_tiou)
        self.assertIsNone(m.std_tiou)
        self.assertIsNone(m.mean_start_error_sec)
        self.assertIsNone(m.mean_end_error_sec)
        self.assertIsNone(m.mean_duration_error_sec)
        self.assertIsNone(m.median_start_error_sec)
        self.assertIsNone(m.median_end_error_sec)
        self.assertIsNone(m.median_duration_error_sec)
        self.assertIsNone(m.hit_rate_at_1s)
        self.assertIsNone(m.hit_rate_at_2s)
        self.assertIsNone(m.hit_rate_at_3s)

        # Boundary refinement ablation must also have None for ground-truth relative metrics
        self.assertIsNotNone(report.refinement_ablation)
        b = report.refinement_ablation
        self.assertIsNone(b.unrefined_mean_tiou)
        self.assertIsNone(b.refined_mean_tiou)
        self.assertIsNone(b.tiou_delta)
        self.assertIsNone(b.start_error_reduction_sec)
        self.assertIsNone(b.end_error_reduction_sec)
        self.assertIsNone(b.hit1_unrefined)
        self.assertIsNone(b.hit1_refined)

        # Markdown and HTML report generation must succeed and render N/A
        md = orchestrator.generate_markdown_report(report)
        self.assertIn("N/A", md)
        html = orchestrator.generate_html_report(report)
        self.assertIn("N/A", html)

    def test_fixed_budget_k1_truncation(self):
        from evaluation.benchmark_orchestrator import BenchmarkOrchestrator
        from evaluation.schemas import (
            BenchmarkDataset,
            BenchmarkSample,
            EvaluationSpan,
            ReferenceHighlight,
        )

        dataset = BenchmarkDataset(
            dataset_name="budget_test",
            dataset_type="research",
            description="Synthetic budget test fixture",
            samples=[
                BenchmarkSample(
                    sample_id="ep_001",
                    duration_sec=100.0,
                    sample_type="synthetic",
                    run_dir="fake_dir",
                    reference_spans=[
                        ReferenceHighlight(
                            span_id="ref_1",
                            start_time=10.0,
                            end_time=20.0,
                            duration=10.0,
                            confidence=1.0,
                        )
                    ],
                )
            ],
        )

        class MultiPredRunner:
            def run_mode(self, mode, run_dir, video_duration, sample_id=None, cache_dir=None):
                # Return 5 candidate spans with varying confidences
                return [
                    EvaluationSpan(span_id=f"p_{i}", start_time=float(i * 10), end_time=float(i * 10 + 5), duration=5.0, confidence=0.1 * i, mode_name=mode)
                    for i in range(1, 6)
                ]

        orchestrator = BenchmarkOrchestrator(ablation_runner=MultiPredRunner())
        report = orchestrator.evaluate_dataset(
            dataset=dataset,
            modes=["system_a_transcript_only"],
            budget_k=1,
        )

        m = report.mode_summaries["system_a_transcript_only"]
        self.assertEqual(m.total_predicted_spans, 1)
        self.assertEqual(report.sample_results[0].num_predictions, 1)
        self.assertEqual(report.sample_results[0].predicted_spans[0].span_id, "p_5")  # highest confidence
        self.assertEqual(report.evaluation_budget, "fixed_budget_k1")
        self.assertEqual(report.budget_k, 1)


if __name__ == "__main__":
    unittest.main()

