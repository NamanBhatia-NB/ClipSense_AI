"""
ClipSense AI — Candidate Quality, Diversity, and Diagnostics Regression Tests
Phase 6 Verification Suite

Tests:
1. Score decomposition and component consistency (heuristic evidence score).
2. Missing modalities handling (ensuring unavailable modalities are not treated as negative evidence).
3. Selection modes: Natural Selection vs Fixed-Budget Selection.
4. Zero-clip output under natural selection when candidates fall below acceptance threshold.
5. Overlap suppression (NMS) and redundancy metrics (union coverage, redundancy ratio).
6. Interval duration constraints and filtering stages waterfall.
7. Human evaluation export blinding integrity and protocol verification.
"""

import csv
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from app.config import MTERConfig
from core.schemas import (
    CommonEvidenceBundle,
    ExpertEvidenceBundle,
    ProposalSourceMetadata,
    SourceResolutionType,
    TemporalProposal,
)
from evaluation.human_eval_export import (
    generate_human_evaluation_package,
    blind_system_name,
)
from pipeline.mter.evidence_bundle import build_common_evidence_bundle
from pipeline.mter.reasoner import MTERReasoner, compute_union_duration


def make_prop(expert: str, pid: str, start: float, end: float, conf: float = 0.8) -> TemporalProposal:
    return TemporalProposal(
        proposal_id=pid,
        start_time=start,
        end_time=end,
        duration=end - start,
        confidence_estimate=conf,
        evidence_type="highlight_evidence",
        explanation=f"Explanation for {pid}",
        supporting_features={"contextual_completeness": 0.85, "self_contained": True},
        source_metadata=ProposalSourceMetadata(
            source_type=SourceResolutionType.WORD_TIMESTAMP if expert == "transcript" else SourceResolutionType.FRAME_TIMESTAMP,
            temporal_resolution_sec=0.25,
            alignment_anchor=f"{expert}_anchor",
        ),
    )


class TestCandidateQualityAndDiagnostics(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_score_decomposition_and_components_consistency(self):
        """Verifies that score_components breakdown is mathematically consistent with composite_score."""
        p_t = make_prop("transcript", "t1", 10.0, 35.0, 0.90)
        p_c = make_prop("conversation", "c1", 11.0, 34.0, 0.85)
        p_v = make_prop("visual", "v1", 10.5, 33.0, 0.75)
        p_p = make_prop("prosody", "p1", 12.0, 32.0, 0.80)

        bundle = CommonEvidenceBundle(
            video_id="test_decomp",
            total_duration=60.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p_t]),
            conversation_evidence=ExpertEvidenceBundle(expert_name="conversation", proposals=[p_c]),
            visual_evidence=ExpertEvidenceBundle(expert_name="visual", proposals=[p_v]),
            prosody_evidence=ExpertEvidenceBundle(expert_name="prosody", proposals=[p_p]),
        )

        cfg = MTERConfig(top_k=1, selection_mode="fixed_budget")
        reasoner = MTERReasoner(cfg)
        output, ledger = reasoner.reason(bundle)

        self.assertEqual(len(output.candidates), 1)
        cand = output.candidates[0]
        metrics = cand.support_metrics

        # Verify score_type is explicit heuristic score, NOT probability
        self.assertEqual(metrics.score_type, "heuristic_evidence_strength")
        self.assertIn("score_components", dir(metrics))
        components = metrics.score_components
        self.assertIsNotNone(components)

        # Check all expected contribution keys
        expected_keys = [
            "modality_support_contrib",
            "modality_diversity_contrib",
            "boundary_agreement_contrib",
            "contextual_completeness_contrib",
            "conflict_penalty_contrib",
            "duration_penalty_contrib",
            "raw_composite_score",
            "composite_score",
        ]
        for k in expected_keys:
            self.assertIn(k, components)

        # Raw composite should equal sum of contributions
        computed_sum = (
            components["modality_support_contrib"] +
            components["modality_diversity_contrib"] +
            components["boundary_agreement_contrib"] +
            components["contextual_completeness_contrib"] +
            components["conflict_penalty_contrib"] +
            components["duration_penalty_contrib"]
        )
        self.assertAlmostEqual(computed_sum, components["raw_composite_score"], places=3)
        self.assertEqual(round(metrics.composite_score, 4), components["composite_score"])

    def test_missing_modalities_not_treated_as_negative_evidence(self):
        """
        Verifies that when visual and prosody modalities are unavailable (e.g. unrun/failed),
        modality_availability records False, and mean modality support normalizes strictly over available modalities.
        """
        p_t = make_prop("transcript", "t1", 15.0, 40.0, 0.80)
        p_c = make_prop("conversation", "c1", 16.0, 39.0, 0.80)

        # bundle has NO visual and NO prosody
        bundle = CommonEvidenceBundle(
            video_id="test_missing_mod",
            total_duration=60.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p_t]),
            conversation_evidence=ExpertEvidenceBundle(expert_name="conversation", proposals=[p_c]),
            visual_evidence=ExpertEvidenceBundle(expert_name="visual", proposals=[]),
            prosody_evidence=ExpertEvidenceBundle(expert_name="prosody", proposals=[]),
        )

        cfg = MTERConfig(top_k=1, selection_mode="fixed_budget")
        reasoner = MTERReasoner(cfg)
        output, ledger = reasoner.reason(bundle)

        self.assertIn("modality_availability", dir(ledger))
        self.assertTrue(ledger.modality_availability["transcript"])
        self.assertTrue(ledger.modality_availability["conversation"])
        self.assertFalse(ledger.modality_availability["visual"])
        self.assertFalse(ledger.modality_availability["prosody"])

        cand = output.candidates[0]
        # Since transcript & conversation had support ~0.80, mean support across the 2 available modalities
        # should be ~0.80, NOT diluted by 4 (which would have been 0.40)
        mod_contrib = cand.support_metrics.score_components["modality_support_contrib"]
        expected_unweighted = cand.support_metrics.score_components["mean_modality_support_unweighted"]
        self.assertGreaterEqual(expected_unweighted, 0.70)
        self.assertAlmostEqual(mod_contrib, round(cfg.weight_modality_support * expected_unweighted, 4), places=3)

    def test_selection_budgets_natural_vs_fixed_budget(self):
        """Verifies candidate count under Natural Selection vs Fixed-Budget Selection."""
        # 3 distinct non-overlapping events across time
        p1 = make_prop("transcript", "t1", 5.0, 25.0, 0.85)
        p2 = make_prop("transcript", "t2", 40.0, 60.0, 0.82)
        p3 = make_prop("transcript", "t3", 80.0, 100.0, 0.80)

        bundle = build_common_evidence_bundle(
            video_id="test_budgets",
            total_duration=120.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p1, p2, p3]),
        )

        # Mode A: Fixed-budget selection with top_k = 2
        cfg_fixed = MTERConfig(top_k=2, selection_mode="fixed_budget")
        reasoner_fixed = MTERReasoner(cfg_fixed)
        out_fixed, ledger_fixed = reasoner_fixed.reason(bundle)
        self.assertEqual(len(out_fixed.candidates), 2)
        self.assertEqual(ledger_fixed.selection_mode, "fixed_budget")

        # Mode B: Natural selection (variable count, acceptance threshold = 0.40)
        cfg_nat = MTERConfig(selection_mode="natural", natural_acceptance_threshold=0.40)
        reasoner_nat = MTERReasoner(cfg_nat)
        out_nat, ledger_nat = reasoner_nat.reason(bundle)
        self.assertEqual(len(out_nat.candidates), 3)  # all 3 distinct events qualify naturally
        self.assertEqual(ledger_nat.selection_mode, "natural")

    def test_natural_selection_zero_clip_output(self):
        """Verifies that Natural Selection safely returns 0 candidates when none meet acceptance threshold."""
        # Weak proposal with low confidence
        p_weak = make_prop("transcript", "w1", 10.0, 30.0, conf=0.15)
        bundle = build_common_evidence_bundle(
            video_id="test_zero_clip",
            total_duration=50.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p_weak]),
        )

        # High acceptance threshold that cannot be met by weak proposal
        cfg = MTERConfig(selection_mode="natural", natural_acceptance_threshold=0.90)
        reasoner = MTERReasoner(cfg)
        output, ledger = reasoner.reason(bundle)

        self.assertEqual(len(output.candidates), 0)
        self.assertIsNone(output.highlight)
        self.assertIsNone(ledger.selected_candidate)
        self.assertGreater(ledger.rejected_by_rule["below_acceptance_threshold"], 0)
        self.assertEqual(ledger.filtering_stages["final_selected_highlights"], 0)
        self.assertEqual(ledger.redundancy_metrics["total_selected_duration_sec"], 0.0)

    def test_overlap_suppression_and_redundancy_metrics(self):
        """Verifies NMS suppression of near-duplicate intervals and redundancy ratio calculation."""
        # Event 1 with two distinct start boundary clusters (10.0s and 15.0s, delta=5.0s > tolerance 3.0s)
        p1 = make_prop("transcript", "t1", 10.0, 35.0, 0.90)
        p1_alt = make_prop("visual", "v1", 15.0, 35.0, 0.85)

        # Event 2: distinct event at 60.0s
        p2 = make_prop("transcript", "t2", 60.0, 85.0, 0.80)

        bundle = build_common_evidence_bundle(
            video_id="test_redundancy",
            total_duration=100.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p1, p2]),
            visual_evidence=ExpertEvidenceBundle(expert_name="visual", proposals=[p1_alt]),
        )

        cfg = MTERConfig(selection_mode="natural", nms_iou_threshold=0.30)
        reasoner = MTERReasoner(cfg)
        output, ledger = reasoner.reason(bundle)

        # NMS should suppress alternative overlapping boundary combinations within event 1
        self.assertGreater(ledger.rejected_by_rule["nms_overlap_suppressed"], 0)
        # Exactly 2 non-overlapping candidates selected (one for each event)
        self.assertEqual(len(output.candidates), 2)

        # Redundancy metrics must be populated
        r_metrics = ledger.redundancy_metrics
        self.assertGreater(r_metrics["total_selected_duration_sec"], 0.0)
        self.assertGreater(r_metrics["union_selected_duration_sec"], 0.0)
        # Since candidates do not overlap, total == union, redundancy ratio == 1.0
        self.assertAlmostEqual(r_metrics["redundancy_ratio"], 1.0, places=2)
        self.assertGreater(r_metrics["temporal_coverage_ratio"], 0.0)

    def test_compute_union_duration_helper(self):
        """Tests interval merging helper for non-overlapping, contiguous, and overlapping intervals."""
        # Non-overlapping
        self.assertAlmostEqual(compute_union_duration([(0.0, 10.0), (20.0, 30.0)]), 20.0)
        # Partially overlapping
        self.assertAlmostEqual(compute_union_duration([(0.0, 15.0), (10.0, 25.0)]), 25.0)
        # Fully nested
        self.assertAlmostEqual(compute_union_duration([(0.0, 30.0), (5.0, 20.0)]), 30.0)
        # Empty
        self.assertAlmostEqual(compute_union_duration([]), 0.0)

    def test_asymmetric_interval_overlap_suppression(self):
        """
        Directly tests the specific failure case: [126.13, 137.40] and [133.75, 158.20].
        Standard temporal IoU is 3.65 / 32.07 = 0.1138 (< 0.30 NMS threshold).
        However, the 3.65s overlap is 32.4% of the shorter 11.27s clip.
        The asymmetric overlap detector must suppress the lower-confidence candidate.
        """
        # Proposal 1: 126.13 to 137.40 (conf 0.80)
        p1 = make_prop("transcript", "t_short", 126.13, 137.40, 0.80)
        # Proposal 2: 133.75 to 158.20 (conf 0.88, higher confidence)
        p2 = make_prop("transcript", "t_long", 133.75, 158.20, 0.88)

        bundle = build_common_evidence_bundle(
            video_id="test_asym_overlap",
            total_duration=180.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p1, p2]),
        )

        cfg = MTERConfig(
            selection_mode="natural",
            natural_acceptance_threshold=0.40,
            nms_iou_threshold=0.30,
            max_shorter_interval_overlap_ratio=0.30,
        )
        reasoner = MTERReasoner(cfg)
        output, ledger = reasoner.reason(bundle)

        # Exactly 1 candidate selected (the higher-scoring longer clip)
        self.assertEqual(len(output.candidates), 1)
        selected = output.candidates[0]
        self.assertAlmostEqual(selected.proposed_start, 133.75, places=1)
        self.assertAlmostEqual(selected.proposed_end, 158.20, places=1)

        # Check suppression ledger
        self.assertTrue(len(ledger.suppression_records) >= 1)
        sup = ledger.suppression_records[0]
        self.assertIn("cand_", sup["candidate_id"])
        self.assertIn("Asymmetric interval overlap", sup["reason"])
        self.assertGreaterEqual(sup["shorter_overlap_ratio"], 0.30)
        self.assertTrue(any("Asymmetric interval overlap" in r["reason"] for r in ledger.rejected_intervals))

    def test_distinct_clips_with_shared_topic_retained(self):
        """
        Verifies that distinct clips discussing the same broad topic at separated timestamps
        are NOT suppressed merely because they share vocabulary.
        """
        # Event A: [10.0, 30.0] discussing machine learning evaluation
        p_a = make_prop("transcript", "t_a", 10.0, 30.0, 0.85)
        p_a.explanation = "Discussing machine learning model performance and evaluation metrics"

        # Event B: [80.0, 100.0] also discussing machine learning evaluation at a later section
        p_b = make_prop("transcript", "t_b", 80.0, 100.0, 0.82)
        p_b.explanation = "Later discussing machine learning evaluation on benchmark datasets"

        bundle = build_common_evidence_bundle(
            video_id="test_shared_topic",
            total_duration=120.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p_a, p_b]),
        )

        cfg = MTERConfig(
            selection_mode="natural",
            natural_acceptance_threshold=0.40,
            nms_iou_threshold=0.30,
            max_shorter_interval_overlap_ratio=0.30,
            transcript_similarity_threshold=0.50,
        )
        reasoner = MTERReasoner(cfg)
        output, ledger = reasoner.reason(bundle)

        # Both clips must be retained because temporal overlap is 0
        self.assertEqual(len(output.candidates), 2)
        self.assertEqual(len(ledger.suppression_records), 0)

    def test_transcript_token_jaccard_suppression(self):
        """
        Verifies near-duplicate candidate suppression when transcripts share >= 50% vocabulary
        and have a small overlap.
        """
        # Candidate 1: [20.0, 45.0]
        p1 = make_prop("transcript", "t1", 20.0, 45.0, 0.90)
        p1.explanation = "Quantum computing algorithms for molecular optimization and chemistry simulations"

        # Candidate 2: [42.0, 67.0] with overlapping vocabulary, small intersection (3.0s < 0.30 shorter_ratio)
        p2 = make_prop("transcript", "t2", 42.0, 67.0, 0.75)
        p2.explanation = "Quantum computing algorithms for molecular optimization and simulation problem sets"

        bundle = build_common_evidence_bundle(
            video_id="test_token_jaccard",
            total_duration=80.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p1, p2]),
        )

        cfg = MTERConfig(
            selection_mode="natural",
            natural_acceptance_threshold=0.40,
            transcript_similarity_threshold=0.50,
        )
        reasoner = MTERReasoner(cfg)
        output, ledger = reasoner.reason(bundle)

        self.assertEqual(len(output.candidates), 1)
        self.assertTrue(len(ledger.suppression_records) >= 1)
        sup = ledger.suppression_records[0]
        self.assertIn("Near-duplicate transcript similarity", sup["reason"])

    def test_export_matched_pilot_package_integrity(self):
        """
        Verifies export_matched_pilot_package produces blinded codes, rating sheet with 4 primary
        and 2 secondary outcome columns, isolated unblinding key, and limits clips <= 3 per system.
        """
        from evaluation.human_eval_export import export_matched_pilot_package

        # Create a mock timestamp_comparison.csv
        mock_csv = Path(self.temp_dir) / "mock_comparison.csv"
        with open(mock_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["source_video_id", "system_name", "start", "end", "duration", "confidence"])
            # Video 1: 4 clips for sys_a (should be capped at 3), 3 for sys_b, 1 for sys_c, 1 for sys_d
            for i in range(4):
                writer.writerow(["ep_1", "system_a_transcript_only", 10.0 + i * 20, 25.0 + i * 20, 15.0, 0.8])
            for i in range(3):
                writer.writerow(["ep_1", "system_b_simple_aggregation", 12.0 + i * 20, 27.0 + i * 20, 15.0, 0.75])
            writer.writerow(["ep_1", "system_c_mter", 15.0, 30.0, 15.0, 0.85])
            writer.writerow(["ep_1", "system_d_mter_refined", 14.5, 30.5, 16.0, 0.85])

        out_pilot_dir = Path(self.temp_dir) / "pilot_out"
        pkg = export_matched_pilot_package(str(mock_csv), str(out_pilot_dir), max_clips_per_system=3)

        self.assertTrue(os.path.exists(pkg["csv_path"]))
        self.assertTrue(os.path.exists(pkg["unblinding_path"]))
        self.assertTrue(os.path.exists(out_pilot_dir / "pilot_system_comparison.json"))

        # Video 1 should have 3 for sys_a, 3 for sys_b, 1 for sys_c, 1 for sys_d = 8 clips total
        # 8 clips * 3 raters = 24 rows
        self.assertEqual(pkg["total_clips"], 8)
        self.assertEqual(pkg["total_rating_rows"], 24)

        # Verify rating sheet columns
        with open(pkg["csv_path"], "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            self.assertIn("score_interestingness", headers)
            self.assertIn("score_self_containment", headers)
            self.assertIn("score_content_payoff", headers)
            self.assertIn("score_start_end_quality", headers)
            self.assertIn("score_redundancy", headers)
            self.assertIn("score_shareability", headers)

            # Ensure scores are blank
            first_row = next(reader)
            self.assertEqual(first_row["score_interestingness"], "")
            self.assertEqual(first_row["score_self_containment"], "")


if __name__ == "__main__":
    unittest.main()

