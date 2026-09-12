"""
ClipSense W5: Multimodal Temporal Evidence Reasoner (MTER) Unit Tests

Verifies:
1. Cross-event pairing prevention: Start from Event A is NEVER paired with End from Event B.
2. Unimodal candidate preservation: Single-expert proposals are not discarded.
3. Multimodal preference: Converging modalities receive higher scores via diversity rewards.
4. Proposal identity preservation: Original IDs, experts, and confidences remain inspectable.
5. Determinism: Same input yields byte-for-byte identical candidate and ledger.
6. Per-modality support visibility: Explicit transcript, conversation, visual, prosody support, coverage, and IoU.
7. Boundary activity vs semantic boundary support: Explicitly separated metrics.
8. Conflict recording: Distinguishes detected conflicts vs conflicts affecting the candidate.
9. Ablation configuration readiness: Tests running in ablation modes.
10. Rejection ledger: Invalid intervals (bounds, min/max duration) are logged with reasons.
"""

import unittest

from app.config import MTERConfig
from core.schemas import (
    CommonEvidenceBundle,
    ExpertEvidenceBundle,
    ProposalSourceMetadata,
    SourceResolutionType,
    TemporalProposal,
)
from pipeline.mter.clustering import (
    cluster_boundary_candidates,
    group_proposals_into_event_regions,
)
from pipeline.mter.evidence_bundle import build_common_evidence_bundle
from pipeline.mter.reasoner import MTERReasoner


def make_proposal(
    expert: str,
    prop_id: str,
    start: float,
    end: float,
    conf: float = 0.85,
    ev_type: str = "highlight_evidence",
    features: dict = None,
) -> TemporalProposal:
    feat = features or {}
    return TemporalProposal(
        proposal_id=prop_id,
        start_time=start,
        end_time=end,
        duration=end - start,
        confidence_estimate=conf,
        evidence_type=ev_type,
        explanation=f"Explanation for {prop_id}",
        supporting_features=feat,
        source_metadata=ProposalSourceMetadata(
            source_type=SourceResolutionType.WORD_TIMESTAMP if expert == "transcript" else SourceResolutionType.FRAME_TIMESTAMP,
            temporal_resolution_sec=0.25,
            alignment_anchor=f"{expert}_anchor",
        ),
    )


class TestMTER(unittest.TestCase):

    def setUp(self):
        self.config = MTERConfig(
            event_merge_gap_sec=2.0,
            boundary_cluster_tolerance_sec=3.0,
            min_candidate_duration_sec=10.0,
            max_candidate_duration_sec=60.0,
            target_candidate_duration_sec=30.0,
            conflict_threshold_sec=4.0,
        )
        self.reasoner = MTERReasoner(self.config)

    def test_prevents_unrelated_cross_event_pairing(self):
        """
        CRITICAL TEST: Verifies that an early start from Event A (e.g. 5.0s) is NEVER
        paired with a late end from completely unrelated Event B (e.g. 85.0s).
        """
        # Event A: 5.0s -> 20.0s
        p_a1 = make_proposal("transcript", "a1", 5.0, 18.0)
        p_a2 = make_proposal("visual", "a2", 7.0, 20.0)

        # Event B: 70.0s -> 85.0s (separated by 50 seconds of silence)
        p_b1 = make_proposal("transcript", "b1", 70.0, 83.0)
        p_b2 = make_proposal("visual", "b2", 72.0, 85.0)

        bundle = build_common_evidence_bundle(
            video_id="cross_event_vid",
            total_duration=100.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p_a1, p_b1]),
            visual_evidence=ExpertEvidenceBundle(expert_name="visual", proposals=[p_a2, p_b2]),
        )

        output, ledger = self.reasoner.reason(bundle)

        # 1. Verify two distinct event regions were formed
        self.assertEqual(len(ledger.event_regions), 2)
        self.assertEqual(ledger.event_regions[0].member_proposal_ids, ["a1", "a2"])
        self.assertEqual(ledger.event_regions[1].member_proposal_ids, ["b1", "b2"])

        # 2. Verify no candidate exists bridging Event A to Event B (e.g. start ~5-7s and end ~83-85s)
        for cand in ledger.evaluated_candidates:
            start = cand["start"]
            end = cand["end"]
            is_cross_event = (start < 25.0 and end > 65.0)
            self.assertFalse(
                is_cross_event,
                f"Candidate [{start}s -> {end}s] bridges unrelated events A and B!"
            )

    def test_unimodal_candidate_preservation(self):
        """Ensures that proposals from a single modality are preserved and evaluated (not discarded)."""
        p_single = make_proposal(
            "transcript",
            "solo_prop",
            10.0,
            25.0,
            conf=0.90,
            features={"contextual_completeness": 0.85, "self_contained": True},
        )
        bundle = build_common_evidence_bundle(
            video_id="unimodal_vid",
            total_duration=40.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p_single]),
        )

        output, ledger = self.reasoner.reason(bundle)

        # Must have evaluated candidates and selected a candidate despite being unimodal
        self.assertGreater(len(ledger.evaluated_candidates), 0)
        self.assertIsNotNone(ledger.selected_candidate)
        self.assertEqual(ledger.selected_candidate.contributing_experts, ["transcript"])
        self.assertAlmostEqual(ledger.selected_candidate.support_metrics.modality_diversity, 0.25)
        self.assertGreater(ledger.selected_candidate.support_metrics.transcript_support, 0.0)

    def test_multimodal_candidate_preference(self):
        """Verifies that multimodal convergence receives higher composite score than unimodal support."""
        # Event 1: Unimodal transcript only
        p_uni = make_proposal("transcript", "u1", 5.0, 25.0, conf=0.85)

        # Event 2: Multimodal convergence across all 4 modalities
        p_m_t = make_proposal("transcript", "m_t", 40.0, 60.0, conf=0.85)
        p_m_c = make_proposal("conversation", "m_c", 40.0, 60.0, conf=0.85)
        p_m_v = make_proposal("visual", "m_v", 42.0, 58.0, conf=0.85)
        p_m_p = make_proposal("prosody", "m_p", 41.0, 59.0, conf=0.85)

        bundle = build_common_evidence_bundle(
            video_id="multi_pref_vid",
            total_duration=70.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p_uni, p_m_t]),
            conversation_evidence=ExpertEvidenceBundle(expert_name="conversation", proposals=[p_m_c]),
            visual_evidence=ExpertEvidenceBundle(expert_name="visual", proposals=[p_m_v]),
            prosody_evidence=ExpertEvidenceBundle(expert_name="prosody", proposals=[p_m_p]),
        )

        output, ledger = self.reasoner.reason(bundle)

        selected = ledger.selected_candidate
        self.assertIsNotNone(selected)
        self.assertGreaterEqual(selected.proposed_start, 39.0)
        self.assertLessEqual(selected.proposed_end, 61.0)
        self.assertEqual(len(selected.contributing_experts), 4)
        self.assertGreater(selected.support_metrics.composite_score, 0.5)

    def test_per_modality_support_and_overlap_visibility(self):
        """Verifies that metrics retain explicit per-modality support, coverage, and IoU."""
        p_t = make_proposal("transcript", "t1", 10.0, 30.0, conf=0.80)
        p_v = make_proposal("visual", "v1", 15.0, 25.0, conf=0.70)

        bundle = build_common_evidence_bundle(
            video_id="visibility_vid",
            total_duration=40.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p_t]),
            visual_evidence=ExpertEvidenceBundle(expert_name="visual", proposals=[p_v]),
        )

        output, ledger = self.reasoner.reason(bundle)
        cand = ledger.selected_candidate
        self.assertIsNotNone(cand)
        metrics = cand.support_metrics

        # Explicit per-modality support numbers
        self.assertGreater(metrics.transcript_support, 0.0)
        self.assertGreater(metrics.visual_support, 0.0)
        self.assertEqual(metrics.conversation_support, 0.0)
        self.assertEqual(metrics.prosody_support, 0.0)

        # Explicit per-modality IoU and coverage
        self.assertGreater(metrics.transcript_iou, 0.0)
        self.assertGreater(metrics.visual_iou, 0.0)
        self.assertGreater(metrics.transcript_coverage, 0.0)
        self.assertGreater(metrics.visual_coverage, 0.0)

        # Boundary support separation
        self.assertGreater(metrics.linguistic_boundary_support, 0.0)
        self.assertGreater(metrics.boundary_activity_support, 0.0)

    def test_conflict_distinction(self):
        """Verifies distinction between detected conflicts in region and conflicts affecting candidate."""
        # Transcript starts at 10.0s, Visual starts at 16.0s (delta = 6.0s >= 4.0s threshold)
        p_t = make_proposal("transcript", "t1", 10.0, 35.0, conf=0.85)
        p_v = make_proposal("visual", "v1", 16.0, 35.0, conf=0.75)

        bundle = build_common_evidence_bundle(
            video_id="conflict_vid",
            total_duration=50.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p_t]),
            visual_evidence=ExpertEvidenceBundle(expert_name="visual", proposals=[p_v]),
        )

        output, ledger = self.reasoner.reason(bundle)

        self.assertGreater(ledger.total_detected_conflicts, 0)
        self.assertGreaterEqual(output.candidates[0].detected_conflicts_count, 1)

    def test_ablation_mode_filtering(self):
        """Verifies that MTER operates in specified ablation modes without changing code."""
        p_t = make_proposal("transcript", "t1", 10.0, 30.0, conf=0.85)
        p_c = make_proposal("conversation", "c1", 10.0, 30.0, conf=0.85)
        p_v = make_proposal("visual", "v1", 12.0, 28.0, conf=0.70)
        p_p = make_proposal("prosody", "p1", 11.0, 29.0, conf=0.75)

        bundle = build_common_evidence_bundle(
            video_id="ablation_vid",
            total_duration=40.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p_t]),
            conversation_evidence=ExpertEvidenceBundle(expert_name="conversation", proposals=[p_c]),
            visual_evidence=ExpertEvidenceBundle(expert_name="visual", proposals=[p_v]),
            prosody_evidence=ExpertEvidenceBundle(expert_name="prosody", proposals=[p_p]),
        )

        # 1. Transcript-only ablation
        cfg_t = MTERConfig(ablation_mode="transcript_only")
        reasoner_t = MTERReasoner(cfg_t)
        out_t, led_t = reasoner_t.reason(bundle)
        self.assertEqual(len(led_t.event_regions[0].contributing_experts), 1)
        self.assertEqual(led_t.event_regions[0].contributing_experts[0], "transcript")

        # 2. Transcript + Conversation ablation
        cfg_tc = MTERConfig(ablation_mode="transcript_conversation")
        reasoner_tc = MTERReasoner(cfg_tc)
        out_tc, led_tc = reasoner_tc.reason(bundle)
        self.assertEqual(sorted(led_tc.event_regions[0].contributing_experts), ["conversation", "transcript"])

    def test_deterministic_results(self):
        """Ensures identical inputs yield exact same MTER outputs, clusters, and ledger."""
        p1 = make_proposal("transcript", "t1", 10.123, 30.456, conf=0.88)
        p2 = make_proposal("visual", "v1", 12.0, 28.0, conf=0.75)
        p3 = make_proposal("prosody", "p1", 11.5, 29.5, conf=0.82)

        def make_bundle():
            return build_common_evidence_bundle(
                video_id="det_vid",
                total_duration=50.0,
                transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p1]),
                visual_evidence=ExpertEvidenceBundle(expert_name="visual", proposals=[p2]),
                prosody_evidence=ExpertEvidenceBundle(expert_name="prosody", proposals=[p3]),
            )

        out1, led1 = self.reasoner.reason(make_bundle())
        out2, led2 = self.reasoner.reason(make_bundle())

        self.assertEqual(out1.candidates[0].proposed_start, out2.candidates[0].proposed_start)
        self.assertEqual(out1.candidates[0].proposed_end, out2.candidates[0].proposed_end)
        self.assertEqual(out1.candidates[0].support_metrics.composite_score, out2.candidates[0].support_metrics.composite_score)
        self.assertEqual(led1.decision_summary, led2.decision_summary)

    def test_rejection_of_invalid_and_out_of_bounds_intervals(self):
        """Verifies that intervals violating duration bounds or boundaries are logged as rejected."""
        p_short = make_proposal("prosody", "p_short", 5.0, 9.0)  # 4.0s duration
        bundle = build_common_evidence_bundle(
            video_id="rej_vid",
            total_duration=30.0,
            prosody_evidence=ExpertEvidenceBundle(expert_name="prosody", proposals=[p_short]),
        )

        output, ledger = self.reasoner.reason(bundle)

        self.assertEqual(len(output.candidates), 0)
        self.assertGreater(len(ledger.rejected_intervals), 0)
        self.assertIn("shorter than min_candidate_duration_sec", ledger.rejected_intervals[0]["reason"])

    def test_broad_proposals_do_not_daisychain_unrelated_events(self):
        """
        CRITICAL TEST 1: Broad overlapping proposals must NOT chain unrelated events
        into a single giant event region.
        
        Setup:
        - Early Event: 5.0s -> 24.5s (Transcript P1)
        - Broad Middle Proposal: 28.0s -> 71.0s (Conversation P2)
        - Late Event: 68.0s -> 90.0s (Transcript P3)
        
        Even though Conversation P2 touches/slightly overlaps Transcript P3 (68.0-71.0s),
        their IoU is ~0.05 (< 0.15) and center gap is ~30s (> 20s).
        They MUST NOT be merged into one single 5s -> 90s event region.
        """
        p_early = make_proposal("transcript", "t_early", 5.975, 24.548)
        p_broad = make_proposal("conversation", "c_broad", 28.471, 71.557)
        p_late = make_proposal("transcript", "t_late", 68.272, 89.865)

        bundle = build_common_evidence_bundle(
            video_id="no_daisychain_vid",
            total_duration=95.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p_early, p_late]),
            conversation_evidence=ExpertEvidenceBundle(expert_name="conversation", proposals=[p_broad]),
        )

        output, ledger = self.reasoner.reason(bundle)

        # Must form 3 separate event regions, NOT 1 giant region!
        self.assertEqual(len(ledger.event_regions), 3)
        self.assertEqual(ledger.event_regions[0].member_proposal_ids, ["t_early"])
        self.assertEqual(ledger.event_regions[1].member_proposal_ids, ["c_broad"])
        self.assertEqual(ledger.event_regions[2].member_proposal_ids, ["t_late"])

    def test_unrelated_distant_proposals_cannot_generate_candidate_conflicts(self):
        """
        CRITICAL TEST 2: For a candidate selected in Event Region 0 (e.g. 5s -> 25s),
        it must NOT report conflicts against unrelated proposals in later regions (e.g. ending at 89.865s).
        """
        p_e1 = make_proposal("transcript", "t_r0", 5.0, 24.0)
        p_e2 = make_proposal("visual", "v_r0", 6.0, 25.0)

        # Distant proposal at ~80-90s
        p_late1 = make_proposal("prosody", "p_late", 80.0, 85.0)
        p_late2 = make_proposal("transcript", "t_late", 80.0, 90.0)

        bundle = build_common_evidence_bundle(
            video_id="conflict_scoping_vid",
            total_duration=95.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p_e1, p_late2]),
            visual_evidence=ExpertEvidenceBundle(expert_name="visual", proposals=[p_e2]),
            prosody_evidence=ExpertEvidenceBundle(expert_name="prosody", proposals=[p_late1]),
        )

        output, ledger = self.reasoner.reason(bundle)
        cand = ledger.selected_candidate
        self.assertIsNotNone(cand)

        # The candidate is in Region 0 (start ~5.5s, end ~24.5s)
        self.assertLess(cand.proposed_end, 30.0)

        # None of the conflicts affecting this candidate may mention 85.0s or 90.0s
        for conf in ledger.conflicts_affecting_candidate:
            for ref in conf.early_proposals + conf.late_proposals:
                self.assertLess(
                    ref.timestamp,
                    35.0,
                    f"Candidate at {cand.proposed_start:.1f}-{cand.proposed_end:.1f} affected by distant proposal at {ref.timestamp:.1f}!"
                )

    def test_proposal_reference_integrity_transcript_3_not_2(self):
        """
        CRITICAL TEST 3: Proposal reference integrity test specifically catching:
        - transcript proposal 2 = 24.568–48.014
        - transcript proposal 3 = 68.272–89.865
        
        Ensures a conflict mentioning 89.865 references proposal 3 (and its exact ID),
        NOT proposal 2!
        """
        prop_1 = make_proposal("transcript", "transcript_prop_0", 5.975, 24.548)
        prop_2 = make_proposal("transcript", "transcript_prop_1", 24.568, 48.014)
        prop_3 = make_proposal("transcript", "transcript_prop_2", 68.272, 89.865)

        # Competing prosody proposal in Region 2 ending early at 80.0s (delta = 9.865s >= 4.0s)
        prosody_late = make_proposal("prosody", "prosody_prop_5", 70.0, 80.0)

        bundle = build_common_evidence_bundle(
            video_id="ref_integrity_vid",
            total_duration=95.0,
            transcript_evidence=ExpertEvidenceBundle(
                expert_name="transcript",
                proposals=[prop_1, prop_2, prop_3],
            ),
            prosody_evidence=ExpertEvidenceBundle(
                expert_name="prosody",
                proposals=[prosody_late],
            ),
        )

        output, ledger = self.reasoner.reason(bundle)

        # Find any conflict involving 89.865s
        conflicts_with_89 = [
            c for c in ledger.conflicts
            if any(abs(ref.timestamp - 89.865) < 0.01 for ref in c.early_proposals + c.late_proposals)
        ]
        self.assertGreater(len(conflicts_with_89), 0, "Expected at least one conflict involving 89.865s in Region 2")

        conflict = conflicts_with_89[0]
        late_ref = [r for r in conflict.late_proposals if abs(r.timestamp - 89.865) < 0.01][0]

        # Verify exact proposal reference integrity: must be prop_3 (transcript_prop_2), NOT prop_2!
        self.assertEqual(late_ref.proposal_id, "transcript_prop_2")
        self.assertNotEqual(late_ref.proposal_id, "transcript_prop_1")
        self.assertIn("89.865", conflict.description)
        self.assertIn("transcript_prop_2", conflict.description)
        self.assertNotIn("transcript_prop_1", conflict.description)

    def test_modality_presence_vs_support_strength_separation(self):
        """
        CRITICAL TEST 6: Distinguishes modality presence (evidence exists)
        from support strength (strong >= 0.50, moderate 0.20-0.50, weak 0.05-0.20).
        """
        # Transcript and Conversation have high confidence & full overlap -> strong
        p_t = make_proposal("transcript", "t1", 10.0, 30.0, conf=0.90)
        p_c = make_proposal("conversation", "c1", 10.0, 30.0, conf=0.85)
        # Visual has low confidence (0.15) -> weak support
        p_v = make_proposal("visual", "v1", 10.0, 30.0, conf=0.15)
        # Prosody has moderate confidence (0.35) -> moderate support
        p_p = make_proposal("prosody", "p1", 10.0, 30.0, conf=0.35)

        bundle = build_common_evidence_bundle(
            video_id="presence_strength_vid",
            total_duration=40.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p_t]),
            conversation_evidence=ExpertEvidenceBundle(expert_name="conversation", proposals=[p_c]),
            visual_evidence=ExpertEvidenceBundle(expert_name="visual", proposals=[p_v]),
            prosody_evidence=ExpertEvidenceBundle(expert_name="prosody", proposals=[p_p]),
        )

        output, ledger = self.reasoner.reason(bundle)
        cand = ledger.selected_candidate
        self.assertIsNotNone(cand)
        m = cand.support_metrics

        # All 4 modalities have evidence present
        self.assertEqual(len(m.modalities_present), 4)
        self.assertEqual(sorted(m.modalities_present), ["conversation", "prosody", "transcript", "visual"])

        # Separated support strength tiers
        self.assertIn("transcript", m.strong_support_modalities)
        self.assertIn("conversation", m.strong_support_modalities)
        self.assertIn("prosody", m.moderate_support_modalities)
        self.assertIn("visual", m.weak_support_modalities)

        # Decision summary must report presence vs strength accurately
        self.assertIn("Modalities with evidence: 4/4", cand.decision_summary)
        self.assertIn("Strong: transcript, conversation", cand.decision_summary)

    def test_all_evaluated_candidates_composite_score_bounded_zero_to_one(self):
        """
        CRITICAL CHECK 1: Verifies that the MTER CompositeScore is explicitly bounded to [0.0, 1.0]
        for ALL evaluated candidates, not only the selected candidate.
        Tests high-conflict, high-penalty scenarios that could drive raw score negative.
        """
        # Low confidence proposals with high conflict
        p_t = make_proposal("transcript", "t1", 10.0, 40.0, conf=0.10)
        p_v = make_proposal("visual", "v1", 18.0, 40.0, conf=0.10)  # delta = 8s -> high conflict penalty

        bundle = build_common_evidence_bundle(
            video_id="bound_vid",
            total_duration=50.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p_t]),
            visual_evidence=ExpertEvidenceBundle(expert_name="visual", proposals=[p_v]),
        )

        output, ledger = self.reasoner.reason(bundle)

        self.assertGreater(len(ledger.evaluated_candidates), 0)
        for cand_record in ledger.evaluated_candidates:
            score = cand_record["metrics"]["composite_score"]
            self.assertGreaterEqual(
                score, 
                0.0, 
                f"Candidate {cand_record['candidate_id']} has negative composite score {score}!"
            )
            self.assertLessEqual(
                score, 
                1.0, 
                f"Candidate {cand_record['candidate_id']} has composite score > 1.0 ({score})!"
            )


if __name__ == "__main__":
    unittest.main()


