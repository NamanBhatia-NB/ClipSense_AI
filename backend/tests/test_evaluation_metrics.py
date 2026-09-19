"""
Unit Tests for Evaluation Metrics and Hungarian Matching (W6)
"""

import unittest
from evaluation.metrics import (
    compute_boundary_hits,
    compute_duration_error,
    compute_end_error,
    compute_pairwise_metrics,
    compute_start_error,
    compute_temporal_iou,
    match_spans_hungarian,
)
from evaluation.schemas import EvaluationSpan, ReferenceHighlight


class TestEvaluationMetrics(unittest.TestCase):

    def test_temporal_iou_identical(self):
        iou = compute_temporal_iou(10.0, 20.0, 10.0, 20.0)
        self.assertAlmostEqual(iou, 1.0, places=6)

    def test_temporal_iou_disjoint(self):
        iou = compute_temporal_iou(10.0, 20.0, 30.0, 40.0)
        self.assertEqual(iou, 0.0)

    def test_temporal_iou_abutting(self):
        iou = compute_temporal_iou(10.0, 20.0, 20.0, 30.0)
        self.assertEqual(iou, 0.0)

    def test_temporal_iou_subset(self):
        # [10, 20] inside [10, 30] -> inter=10, union=20 -> 0.5
        iou = compute_temporal_iou(10.0, 20.0, 10.0, 30.0)
        self.assertAlmostEqual(iou, 0.5, places=6)

    def test_temporal_iou_partial_overlap(self):
        # [10, 25] and [20, 30] -> inter=5, union=20 -> 0.25
        iou = compute_temporal_iou(10.0, 25.0, 20.0, 30.0)
        self.assertAlmostEqual(iou, 0.25, places=6)

    def test_temporal_iou_invalid_spans(self):
        self.assertEqual(compute_temporal_iou(20.0, 10.0, 10.0, 20.0), 0.0)
        self.assertEqual(compute_temporal_iou(10.0, 10.0, 10.0, 20.0), 0.0)

    def test_start_end_duration_errors(self):
        start_err, start_bias = compute_start_error(12.5, 10.0)
        self.assertAlmostEqual(start_err, 2.5, places=6)
        self.assertAlmostEqual(start_bias, 2.5, places=6)

        start_err_neg, start_bias_neg = compute_start_error(8.0, 10.0)
        self.assertAlmostEqual(start_err_neg, 2.0, places=6)
        self.assertAlmostEqual(start_bias_neg, -2.0, places=6)

        end_err, end_bias = compute_end_error(28.0, 30.0)
        self.assertAlmostEqual(end_err, 2.0, places=6)
        self.assertAlmostEqual(end_bias, -2.0, places=6)

        dur_err, rel_dur_err = compute_duration_error(15.0, 20.0)
        self.assertAlmostEqual(dur_err, 5.0, places=6)
        self.assertAlmostEqual(rel_dur_err, 0.25, places=6)

    def test_boundary_hits(self):
        hits = compute_boundary_hits(start_err=0.8, end_err=1.5, tolerances=(1.0, 2.0, 3.0))
        self.assertFalse(hits[1.0])  # end_err 1.5 > 1.0
        self.assertTrue(hits[2.0])   # both <= 2.0
        self.assertTrue(hits[3.0])   # both <= 3.0

    def test_hungarian_matching_exact(self):
        preds = [
            EvaluationSpan(
                span_id="p1",
                start_time=10.0,
                end_time=20.0,
                duration=10.0,
                confidence=0.9,
                mode_name="baseline",
            ),
            EvaluationSpan(
                span_id="p2",
                start_time=40.0,
                end_time=55.0,
                duration=15.0,
                confidence=0.8,
                mode_name="baseline",
            ),
        ]
        refs = [
            ReferenceHighlight(
                span_id="r1",
                start_time=10.5,
                end_time=20.2,
                annotation_source="manual_dev_debug",
            ),
            ReferenceHighlight(
                span_id="r2",
                start_time=40.0,
                end_time=54.0,
                annotation_source="manual_dev_debug",
            ),
        ]

        matched, unmatched_p, unmatched_r = match_spans_hungarian(preds, refs, min_iou=0.10)
        self.assertEqual(len(matched), 2)
        self.assertEqual(len(unmatched_p), 0)
        self.assertEqual(len(unmatched_r), 0)

        # First match is p1 -> r1
        m1 = next(m for m in matched if m.prediction_span_id == "p1")
        self.assertEqual(m1.reference_span_id, "r1")
        self.assertAlmostEqual(m1.start_error_sec, 0.5, places=5)

    def test_hungarian_matching_rejection_below_iou_threshold(self):
        preds = [
            EvaluationSpan(
                span_id="p_distant",
                start_time=10.0,
                end_time=20.0,
                duration=10.0,
                confidence=0.9,
                mode_name="baseline",
            )
        ]
        refs = [
            ReferenceHighlight(
                span_id="r_distant",
                start_time=50.0,
                end_time=60.0,
                annotation_source="manual_dev_debug",
            )
        ]

        matched, unmatched_p, unmatched_r = match_spans_hungarian(preds, refs, min_iou=0.10)
        self.assertEqual(len(matched), 0)
        self.assertEqual(unmatched_p, ["p_distant"])
        self.assertEqual(unmatched_r, ["r_distant"])


if __name__ == "__main__":
    unittest.main()
