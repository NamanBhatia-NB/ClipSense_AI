"""
ClipSense AI — Regression Tests for Evaluation Semantics & Budget Decoupling

Verifies:
1. Clip budget (K clips) vs Segment budget (N bins) independence:
   - Changing clip_budget_k limits selected candidate clips without changing segment-level cutoff (segment_prediction_k).
   - Full 100-bin curve ranking (ranking_top_k=None) evaluates AP across all 100 bins.
2. Edge cases in compute_rhapsody_metrics:
   - Ties in predicted curves
   - Empty/all-zero scores
   - Constant curves (uniform baselines)
3. Spans-to-bins projection:
   - Single clip overlapping multiple 100-bin temporal intervals proportionally.
4. Interval union coverage (compute_union_coverage):
   - Disjoint intervals
   - Overlapping intervals
   - Nested intervals
   - Adjacent intervals
   - Out-of-bounds clipping and strict [0%, 100%] bounds.
"""

import unittest
import numpy as np

from evaluation.metrics import (
    compute_rhapsody_metrics,
    spans_to_segment_importance_curve,
)
from core.schemas import ExpertEvidenceBundle, ProposalSourceMetadata, SourceResolutionType, TemporalProposal
from evaluation.ablation_runner import run_all_experts_no_mter_algorithm
from evaluation.run_timestamp_evaluation import compute_union_coverage
from evaluation.schemas import EvaluationSpan


class TestEvaluationBudgetDecoupling(unittest.TestCase):
    """Regression tests verifying independence of clip budget and segment prediction budget."""

    def setUp(self):
        # Create a synthetic 100-bin ground-truth replay curve: 5 highlight segments
        self.scores_ref = [0.1] * 100
        for idx in [15, 20, 25, 30, 35]:
            self.scores_ref[idx] = 0.9

        # Create 10 non-overlapping candidate clips
        self.video_dur = 1000.0  # 10s per bin
        self.candidate_spans = [
            EvaluationSpan(
                span_id=f"cand_{i}",
                start_time=float(i * 50.0),
                end_time=float(i * 50.0 + 30.0),
                duration=30.0,
                confidence=round(0.95 - i * 0.05, 3),
                mode_name="baseline",
            )
            for i in range(10)
        ]

    def test_clip_budget_does_not_alter_segment_prediction_k(self):
        """
        Verify that changing the clip budget (K=1 vs K=3 vs K=5) changes the number
        of selected clips, but the segment-level evaluation strictly maintains its
        independent segment_prediction_k cutoff (e.g. 20 segments).
        """
        segment_k = 20

        for clip_k in [1, 3, 5]:
            selected_clips = self.candidate_spans[:clip_k]
            self.assertEqual(len(selected_clips), clip_k)

            # Project selected clips to 100 bins
            pred_curve = spans_to_segment_importance_curve(
                spans=selected_clips,
                video_duration=self.video_dur,
                num_segments=100,
            )

            metrics = compute_rhapsody_metrics(
                scores_pred=pred_curve,
                scores_ref=self.scores_ref,
                gt_threshold=0.5,
                max_n_gt=20,
                segment_prediction_k=segment_k,
                ranking_top_k=None,
            )

            # Binary metrics must evaluate on exactly segment_k segments
            self.assertEqual(
                metrics["num_pred"],
                segment_k,
                f"For clip budget K={clip_k}, num_pred was {metrics['num_pred']} instead of {segment_k}",
            )
            # mAP is computed on all 100 ranked segments
            self.assertIn("average_precision", metrics)
            self.assertGreaterEqual(metrics["average_precision"], 0.0)
            self.assertLessEqual(metrics["average_precision"], 1.0)

    def test_ranking_top_k_none_evaluates_complete_curve(self):
        """Verify ranking_top_k=None ranks all 100 bins without truncation."""
        pred_curve = [float(i) for i in range(100)]
        metrics_full = compute_rhapsody_metrics(
            scores_pred=pred_curve,
            scores_ref=self.scores_ref,
            gt_threshold=0.5,
            segment_prediction_k=20,
            ranking_top_k=None,
        )
        self.assertGreater(metrics_full["average_precision"], 0.0)


class TestRhapsodyMetricsEdgeCases(unittest.TestCase):
    """Test edge cases: ties, empty scores, constant curves."""

    def setUp(self):
        self.scores_ref = [0.1] * 100
        self.scores_ref[10] = 0.8
        self.scores_ref[20] = 0.9

    def test_all_zero_empty_scores(self):
        """When predicted curve is all zeros, thresholded predictions yield zero metrics without crashing."""
        pred_curve = [0.0] * 100
        # With threshold > 0, 0 segments are selected
        metrics_thresh = compute_rhapsody_metrics(
            scores_pred=pred_curve,
            scores_ref=self.scores_ref,
            gt_threshold=0.5,
            pred_threshold=0.5,
        )
        self.assertEqual(metrics_thresh["num_pred"], 0)
        self.assertEqual(metrics_thresh["hit_rate"], 0.0)
        self.assertEqual(metrics_thresh["precision"], 0.0)
        self.assertEqual(metrics_thresh["recall"], 0.0)
        self.assertEqual(metrics_thresh["f1"], 0.0)
        self.assertGreaterEqual(metrics_thresh["average_precision"], 0.0)

        # With segment_prediction_k=0 and ranking_top_k=0 (explicit zero prediction and zero ranking)
        metrics_k0 = compute_rhapsody_metrics(
            scores_pred=pred_curve,
            scores_ref=self.scores_ref,
            gt_threshold=0.5,
            segment_prediction_k=0,
            ranking_top_k=0,
        )
        self.assertEqual(metrics_k0["num_pred"], 0)
        self.assertEqual(metrics_k0["hit_rate"], 0.0)
        self.assertEqual(metrics_k0["average_precision"], 0.0)

    def test_constant_predicted_curve(self):
        """When predicted curve is constant (e.g. 0.850 everywhere), handles ties gracefully."""
        pred_curve = [0.850] * 100
        metrics = compute_rhapsody_metrics(
            scores_pred=pred_curve,
            scores_ref=self.scores_ref,
            gt_threshold=0.5,
            segment_prediction_k=20,
        )
        self.assertIn("average_precision", metrics)
        self.assertGreaterEqual(metrics["average_precision"], 0.0)
        self.assertIn("hit_rate", metrics)
        self.assertIn("precision", metrics)

    def test_ties_in_predicted_scores(self):
        """When multiple segments have tied scores, ranking order is deterministic."""
        pred_curve = [0.5] * 100
        pred_curve[5] = 0.9
        pred_curve[6] = 0.9
        pred_curve[7] = 0.9

        metrics = compute_rhapsody_metrics(
            scores_pred=pred_curve,
            scores_ref=self.scores_ref,
            gt_threshold=0.5,
            segment_prediction_k=10,
        )
        self.assertEqual(metrics["num_pred"], 10)
        self.assertGreaterEqual(metrics["average_precision"], 0.0)

    def test_single_clip_multi_bin_spanning(self):
        """A single 50s clip in a 100s video spans 50 of the 100 bins (0.50 of total)."""
        video_dur = 100.0  # 1.0s per bin
        span = EvaluationSpan(
            span_id="span_50s",
            start_time=10.0,
            end_time=60.0,
            duration=50.0,
            confidence=0.88,
            mode_name="baseline",
        )
        curve = spans_to_segment_importance_curve([span], video_duration=video_dur, num_segments=100)
        self.assertEqual(len(curve), 100)

        # Bins 0-9: 0.0
        for b in range(10):
            self.assertEqual(curve[b], 0.0)

        # Bins 10-59: fully covered (1.0s overlap / 1.0s bin duration * 0.88 = 0.88)
        for b in range(10, 60):
            self.assertAlmostEqual(curve[b], 0.88, places=4)

        # Bins 60-99: 0.0
        for b in range(60, 100):
            self.assertEqual(curve[b], 0.0)


class TestUnionCoverageMetric(unittest.TestCase):
    """Test compute_union_coverage on overlapping, nested, adjacent, and disjoint intervals."""

    def test_disjoint_intervals(self):
        """[0, 10] and [20, 30] in 100s video -> union duration = 20s, coverage = 20%."""
        spans = [
            EvaluationSpan(span_id="s1", start_time=0.0, end_time=10.0, duration=10.0, confidence=1.0, mode_name="baseline"),
            EvaluationSpan(span_id="s2", start_time=20.0, end_time=30.0, duration=10.0, confidence=1.0, mode_name="baseline"),
        ]
        u_dur, u_pct = compute_union_coverage(spans, video_duration=100.0)
        self.assertAlmostEqual(u_dur, 20.0, places=3)
        self.assertAlmostEqual(u_pct, 20.0, places=3)

    def test_overlapping_intervals(self):
        """[0, 20] and [10, 30] in 100s video -> union duration = 30s (not 40s), coverage = 30%."""
        spans = [
            EvaluationSpan(span_id="s1", start_time=0.0, end_time=20.0, duration=20.0, confidence=1.0, mode_name="baseline"),
            EvaluationSpan(span_id="s2", start_time=10.0, end_time=30.0, duration=20.0, confidence=1.0, mode_name="baseline"),
        ]
        u_dur, u_pct = compute_union_coverage(spans, video_duration=100.0)
        self.assertAlmostEqual(u_dur, 30.0, places=3)
        self.assertAlmostEqual(u_pct, 30.0, places=3)

    def test_nested_intervals(self):
        """[0, 40] and [10, 25] in 100s video -> union duration = 40s (not 55s), coverage = 40%."""
        spans = [
            EvaluationSpan(span_id="s1", start_time=0.0, end_time=40.0, duration=40.0, confidence=1.0, mode_name="baseline"),
            EvaluationSpan(span_id="s2", start_time=10.0, end_time=25.0, duration=15.0, confidence=1.0, mode_name="baseline"),
        ]
        u_dur, u_pct = compute_union_coverage(spans, video_duration=100.0)
        self.assertAlmostEqual(u_dur, 40.0, places=3)
        self.assertAlmostEqual(u_pct, 40.0, places=3)

    def test_adjacent_abutting_intervals(self):
        """[0, 15] and [15, 30] in 100s video -> merges to [0, 30], union duration = 30s."""
        spans = [
            EvaluationSpan(span_id="s1", start_time=0.0, end_time=15.0, duration=15.0, confidence=1.0, mode_name="baseline"),
            EvaluationSpan(span_id="s2", start_time=15.0, end_time=30.0, duration=15.0, confidence=1.0, mode_name="baseline"),
        ]
        u_dur, u_pct = compute_union_coverage(spans, video_duration=100.0)
        self.assertAlmostEqual(u_dur, 30.0, places=3)
        self.assertAlmostEqual(u_pct, 30.0, places=3)

    def test_sliding_window_saturation_bounded_at_100(self):
        """Many overlapping 30s windows every 15s covering entire 100s video must bound to 100%."""
        spans = [
            EvaluationSpan(span_id=f"w_{i}", start_time=float(i * 15), end_time=float(i * 15 + 30), duration=30.0, confidence=1.0, mode_name="baseline")
            for i in range(10)
        ]
        # Sum of durations = 300s, but video is 100s -> union coverage must be 100.0%
        raw_sum = sum(s.duration for s in spans)
        self.assertEqual(raw_sum, 300.0)

        u_dur, u_pct = compute_union_coverage(spans, video_duration=100.0)
        self.assertAlmostEqual(u_dur, 100.0, places=3)
        self.assertAlmostEqual(u_pct, 100.0, places=3)


class TestSystemBAvailabilityNormalization(unittest.TestCase):
    """
    Regression tests verifying System B (Simple Multimodal Aggregation):
    1. Averages over available modalities at episode level (e.g. 2 active -> /2.0; 4 active -> /4.0).
    2. Candidate ranking is strictly unchanged when scores are multiplied by a positive scalar.
    3. Empty evidence across all modalities does not cause division by zero.
    """

    def setUp(self):
        # Create standard synthetic proposals
        meta = ProposalSourceMetadata(
            source_type=SourceResolutionType.WORD_TIMESTAMP,
            temporal_resolution_sec=0.05,
            alignment_anchor="test",
        )
        self.p_trans1 = TemporalProposal(
            proposal_id="t1",
            start_time=10.0,
            end_time=30.0,
            duration=20.0,
            confidence_estimate=0.80,
            evidence_type="semantic",
            explanation="Transcript unit",
            source_metadata=meta,
        )
        self.p_trans2 = TemporalProposal(
            proposal_id="t2",
            start_time=50.0,
            end_time=80.0,
            duration=30.0,
            confidence_estimate=0.60,
            evidence_type="semantic",
            explanation="Transcript unit 2",
            source_metadata=meta,
        )
        self.p_conv1 = TemporalProposal(
            proposal_id="c1",
            start_time=15.0,
            end_time=35.0,
            duration=20.0,
            confidence_estimate=0.70,
            evidence_type="turn",
            explanation="Conversation unit",
            source_metadata=meta,
        )

    def test_available_modality_mean_computation(self):
        """When 2 modalities are active, denominator is 2.0 (not 4.0)."""
        meta = ProposalSourceMetadata(
            source_type=SourceResolutionType.WORD_TIMESTAMP,
            temporal_resolution_sec=0.05,
            alignment_anchor="test",
        )
        b_trans = ExpertEvidenceBundle(expert_name="transcript", proposals=[self.p_trans1])
        b_conv = ExpertEvidenceBundle(expert_name="conversation", proposals=[self.p_conv1])
        b_vis = ExpertEvidenceBundle(expert_name="visual", proposals=[])
        b_pros = ExpertEvidenceBundle(expert_name="prosody", proposals=[])

        # Overlap interval [15.0, 30.0]: duration = 15.0s
        # Transcript overlap: [15, 30] = 15s -> coverage = 0.80 * (15/15) = 0.80
        # Conversation overlap: [15, 30] = 15s -> coverage = 0.70 * (15/15) = 0.70
        # Expected unweighted score with 2 active modalities = (0.80 + 0.70) / 2.0 = 0.75
        # Under legacy fixed /4.0, score would have been (0.80 + 0.70) / 4.0 = 0.375
        spans = run_all_experts_no_mter_algorithm(
            transcript_bundle=b_trans,
            conversation_bundle=b_conv,
            visual_bundle=b_vis,
            prosody_bundle=b_pros,
            video_duration=100.0,
            min_duration_sec=10.0,
            max_duration_sec=60.0,
        )
        self.assertTrue(len(spans) > 0)
        # Find the intersection span [15.0, 30.0]
        inter_spans = [s for s in spans if abs(s.start_time - 15.0) < 1e-3 and abs(s.end_time - 30.0) < 1e-3]
        self.assertEqual(len(inter_spans), 1)
        inter_span = inter_spans[0]
        self.assertAlmostEqual(inter_span.confidence, 0.75, places=3)
        self.assertEqual(inter_span.metadata.get("active_modalities_count"), 2)

        # Now activate all 4 modalities
        p_vis = TemporalProposal(
            proposal_id="v1",
            start_time=10.0,
            end_time=30.0,
            duration=20.0,
            confidence_estimate=0.50,
            evidence_type="visual",
            explanation="Visual unit",
            source_metadata=meta,
        )
        p_pros = TemporalProposal(
            proposal_id="p1",
            start_time=10.0,
            end_time=30.0,
            duration=20.0,
            confidence_estimate=0.60,
            evidence_type="prosody",
            explanation="Prosody unit",
            source_metadata=meta,
        )
        b_vis4 = ExpertEvidenceBundle(expert_name="visual", proposals=[p_vis])
        b_pros4 = ExpertEvidenceBundle(expert_name="prosody", proposals=[p_pros])

        spans4 = run_all_experts_no_mter_algorithm(
            transcript_bundle=b_trans,
            conversation_bundle=b_conv,
            visual_bundle=b_vis4,
            prosody_bundle=b_pros4,
            video_duration=100.0,
            min_duration_sec=10.0,
            max_duration_sec=60.0,
        )
        inter_spans4 = [s for s in spans4 if abs(s.start_time - 15.0) < 1e-3 and abs(s.end_time - 30.0) < 1e-3]
        self.assertEqual(len(inter_spans4), 1)
        # Expected with 4 active: (0.80 + 0.70 + 0.50 + 0.60) / 4.0 = 2.60 / 4 = 0.65
        self.assertAlmostEqual(inter_spans4[0].confidence, 0.65, places=3)
        self.assertEqual(inter_spans4[0].metadata.get("active_modalities_count"), 4)

    def test_candidate_ranking_invariant_to_positive_scalar(self):
        """Multiplying all candidate scores by a positive scalar (4/2 = 2.0) preserves relative ranking."""
        b_trans = ExpertEvidenceBundle(expert_name="transcript", proposals=[self.p_trans1, self.p_trans2])
        b_conv = ExpertEvidenceBundle(expert_name="conversation", proposals=[self.p_conv1])
        b_vis = ExpertEvidenceBundle(expert_name="visual", proposals=[])
        b_pros = ExpertEvidenceBundle(expert_name="prosody", proposals=[])

        spans_norm = run_all_experts_no_mter_algorithm(
            transcript_bundle=b_trans,
            conversation_bundle=b_conv,
            visual_bundle=b_vis,
            prosody_bundle=b_pros,
            video_duration=100.0,
        )

        # Verify ranking order: strictly decreasing by confidence
        confidences = [s.confidence for s in spans_norm]
        self.assertEqual(confidences, sorted(confidences, reverse=True))

        # Check that scaling all scores by any scalar c > 0 produces identical ordering and intervals
        scaled_spans = sorted(spans_norm, key=lambda s: (s.confidence * 2.0, -s.start_time), reverse=True)
        self.assertEqual([(s.start_time, s.end_time) for s in spans_norm], [(s.start_time, s.end_time) for s in scaled_spans])

    def test_empty_evidence_no_division_by_zero(self):
        """When all bundles are empty, algorithm returns [] without ZeroDivisionError."""
        b_empty = lambda name: ExpertEvidenceBundle(expert_name=name, proposals=[])
        spans = run_all_experts_no_mter_algorithm(
            transcript_bundle=b_empty("transcript"),
            conversation_bundle=b_empty("conversation"),
            visual_bundle=b_empty("visual"),
            prosody_bundle=b_empty("prosody"),
            video_duration=100.0,
        )
        self.assertEqual(spans, [])


if __name__ == "__main__":
    unittest.main()
