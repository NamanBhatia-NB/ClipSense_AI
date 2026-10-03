"""
Unit Tests for Top-K Candidate Selection and Non-Maximum Suppression (W8)

Verifies:
1. Deterministic Top-K selection preserving existing ranking order.
2. Suppression of candidates overlapping with higher-ranked candidates (IoU > nms_iou_threshold).
3. Preservation of non-overlapping multi-event candidates.
4. Backward compatibility: MTEROutput.highlight property matches candidates[0].
5. Default top_k=1 produces exactly 1 candidate (identical to W5 behavior).
6. Empty candidate list handling (no unhandled exceptions).
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
from pipeline.mter.reasoner import MTERReasoner, compute_temporal_interval_iou


def make_proposal(expert: str, prop_id: str, start: float, end: float, conf: float = 0.8) -> TemporalProposal:
    return TemporalProposal(
        expert_name=expert,
        proposal_id=prop_id,
        start_time=start,
        end_time=end,
        duration=end - start,
        confidence_estimate=conf,
        evidence_type="test_evidence",
        explanation=f"Proposal {prop_id}",
        source_metadata=ProposalSourceMetadata(
            source_type=SourceResolutionType.WORD_TIMESTAMP if expert == "transcript" else SourceResolutionType.FRAME_TIMESTAMP,
            temporal_resolution_sec=0.25,
            alignment_anchor=f"{expert}_anchor",
        ),
    )


class TestTopKAndNMS(unittest.TestCase):
    """Test suite for Top-K highlight selection and Non-Maximum Suppression."""

    def test_temporal_interval_iou_math(self):
        # Disjoint intervals
        self.assertEqual(compute_temporal_interval_iou(0.0, 10.0, 15.0, 25.0), 0.0)
        # Identical intervals
        self.assertEqual(compute_temporal_interval_iou(10.0, 20.0, 10.0, 20.0), 1.0)
        # 50% overlap: [0, 10] and [5, 15] -> inter=5, union=15 -> 1/3
        self.assertAlmostEqual(compute_temporal_interval_iou(0.0, 10.0, 5.0, 15.0), 5.0 / 15.0)
        # Contained interval: [0, 10] and [2, 6] -> inter=4, union=10 -> 0.4
        self.assertAlmostEqual(compute_temporal_interval_iou(0.0, 10.0, 2.0, 6.0), 0.4)

    def test_default_top_k_is_one_preserving_backward_compatibility(self):
        """Ensures that default MTERConfig produces exactly 1 candidate as in W5."""
        cfg = MTERConfig(
            min_candidate_duration_sec=5.0,
            max_candidate_duration_sec=30.0,
        )
        reasoner = MTERReasoner(cfg)

        # Create two distinct events
        p1 = make_proposal("transcript", "p1", 5.0, 20.0, conf=0.9)
        p2 = make_proposal("visual", "p2", 6.0, 21.0, conf=0.85)

        p3 = make_proposal("transcript", "p3", 40.0, 55.0, conf=0.7)
        p4 = make_proposal("visual", "p4", 41.0, 56.0, conf=0.65)

        bundle = CommonEvidenceBundle(
            video_id="test_top1_vid",
            total_duration=70.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p1, p3]),
            visual_evidence=ExpertEvidenceBundle(expert_name="visual", proposals=[p2, p4]),
            prosody_evidence=ExpertEvidenceBundle(expert_name="prosody", proposals=[]),
            conversation_evidence=ExpertEvidenceBundle(expert_name="conversation", proposals=[]),
        )

        output, ledger = reasoner.reason(bundle)

        # Default must return exactly 1 candidate
        self.assertEqual(len(output.candidates), 1)
        self.assertIsNotNone(output.highlight)
        self.assertEqual(output.highlight.candidate_id, output.candidates[0].candidate_id)
        self.assertEqual(ledger.selected_candidate.candidate_id, output.candidates[0].candidate_id)

    def test_top_k_multi_candidate_selection_with_nms(self):
        """Verifies that requesting top_k=2 extracts non-overlapping candidates across events."""
        cfg = MTERConfig(
            min_candidate_duration_sec=5.0,
            max_candidate_duration_sec=30.0,
            top_k=3,
            nms_iou_threshold=0.3,
        )
        reasoner = MTERReasoner(cfg)

        # Event 1: strong evidence around 5s-20s
        e1_t = make_proposal("transcript", "e1_t", 5.0, 20.0, conf=0.95)
        e1_v = make_proposal("visual", "e1_v", 6.0, 21.0, conf=0.90)

        # Event 2: distinct event around 50s-65s
        e2_t = make_proposal("transcript", "e2_t", 50.0, 65.0, conf=0.80)
        e2_v = make_proposal("visual", "e2_v", 51.0, 66.0, conf=0.75)

        bundle = CommonEvidenceBundle(
            video_id="test_multi_topk_vid",
            total_duration=80.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[e1_t, e2_t]),
            visual_evidence=ExpertEvidenceBundle(expert_name="visual", proposals=[e1_v, e2_v]),
            prosody_evidence=ExpertEvidenceBundle(expert_name="prosody", proposals=[]),
            conversation_evidence=ExpertEvidenceBundle(expert_name="conversation", proposals=[]),
        )

        output, ledger = reasoner.reason(bundle, top_k=2)

        # Must return 2 candidates from separate events
        self.assertEqual(len(output.candidates), 2)

        cand1 = output.candidates[0]
        cand2 = output.candidates[1]

        # Highest score should be first
        self.assertGreaterEqual(cand1.confidence_estimate, cand2.confidence_estimate)

        # Check that IoU between the two selected candidates is below nms threshold
        iou = compute_temporal_interval_iou(
            cand1.proposed_start, cand1.proposed_end,
            cand2.proposed_start, cand2.proposed_end,
        )
        self.assertLessEqual(iou, 0.3)

        # Candidate 1 should be around Event 1 (5s-21s)
        self.assertTrue(cand1.proposed_start < 25.0)
        # Candidate 2 should be around Event 2 (50s-66s)
        self.assertTrue(cand2.proposed_start >= 45.0)

    def test_duplicate_suppression_within_same_event(self):
        """Verifies that heavily overlapping alternative boundaries from the same event are suppressed."""
        cfg = MTERConfig(
            min_candidate_duration_sec=5.0,
            max_candidate_duration_sec=30.0,
            top_k=5,
            nms_iou_threshold=0.3,
        )
        reasoner = MTERReasoner(cfg)

        # Only ONE single event with closely overlapping proposals
        p1 = make_proposal("transcript", "p1", 10.0, 25.0, conf=0.9)
        p2 = make_proposal("visual", "p2", 11.0, 26.0, conf=0.85)

        bundle = CommonEvidenceBundle(
            video_id="test_single_event_vid",
            total_duration=35.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p1]),
            visual_evidence=ExpertEvidenceBundle(expert_name="visual", proposals=[p2]),
            prosody_evidence=ExpertEvidenceBundle(expert_name="prosody", proposals=[]),
            conversation_evidence=ExpertEvidenceBundle(expert_name="conversation", proposals=[]),
        )

        output, ledger = reasoner.reason(bundle, top_k=5, nms_iou_threshold=0.3)

        # Even though top_k=5 was requested, because all alternative candidate intervals
        # within this single event overlap with each other with IoU > 0.3, NMS must suppress duplicates!
        self.assertEqual(len(output.candidates), 1)

    def test_empty_candidate_proposals(self):
        """Ensures that empty bundles or discarded proposals do not crash Top-K logic."""
        cfg = MTERConfig()
        reasoner = MTERReasoner(cfg)

        bundle = CommonEvidenceBundle(
            video_id="empty_vid",
            total_duration=30.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[]),
            visual_evidence=ExpertEvidenceBundle(expert_name="visual", proposals=[]),
            prosody_evidence=ExpertEvidenceBundle(expert_name="prosody", proposals=[]),
            conversation_evidence=ExpertEvidenceBundle(expert_name="conversation", proposals=[]),
        )

        output, ledger = reasoner.reason(bundle, top_k=3)

        self.assertEqual(len(output.candidates), 0)
        self.assertIsNone(output.highlight)


if __name__ == "__main__":
    unittest.main()
