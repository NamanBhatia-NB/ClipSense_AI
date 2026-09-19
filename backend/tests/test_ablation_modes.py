"""
Unit Tests for Experiment Modes and all_experts_no_mter Algorithm (W6)
"""

import unittest
from core.schemas import ExpertEvidenceBundle, ProposalSourceMetadata, SourceResolutionType, TemporalProposal
from evaluation.ablation_runner import run_all_experts_no_mter_algorithm
from evaluation.schemas import EvaluationSpan


def make_prop(prop_id: str, start: float, end: float, conf: float = 0.8) -> TemporalProposal:
    return TemporalProposal(
        proposal_id=prop_id,
        start_time=start,
        end_time=end,
        duration=end - start,
        confidence_estimate=conf,
        evidence_type="test_evidence",
        explanation="Synthetic test proposal",
        source_metadata=ProposalSourceMetadata(
            source_type=SourceResolutionType.WORD_TIMESTAMP,
            temporal_resolution_sec=0.1,
            alignment_anchor="test_anchor",
        ),
    )


class TestAblationModes(unittest.TestCase):

    def test_all_experts_no_mter_formula(self):
        # Transcript proposal: [10.0, 20.0], conf=0.8
        trans_bundle = ExpertEvidenceBundle(
            expert_name="transcript",
            proposals=[make_prop("t1", 10.0, 20.0, conf=0.8)],
        )
        # Visual proposal: [10.0, 20.0], conf=0.6
        vis_bundle = ExpertEvidenceBundle(
            expert_name="visual",
            proposals=[make_prop("v1", 10.0, 20.0, conf=0.6)],
        )
        # Conversation: empty
        conv_bundle = ExpertEvidenceBundle(expert_name="conversation", proposals=[])
        # Prosody: empty
        pros_bundle = ExpertEvidenceBundle(expert_name="prosody", proposals=[])

        spans = run_all_experts_no_mter_algorithm(
            transcript_bundle=trans_bundle,
            conversation_bundle=conv_bundle,
            visual_bundle=vis_bundle,
            prosody_bundle=pros_bundle,
            video_duration=30.0,
            min_duration_sec=5.0,
            max_duration_sec=30.0,
        )

        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.mode_name, "all_experts_no_mter")
        self.assertAlmostEqual(span.start_time, 10.0, places=5)
        self.assertAlmostEqual(span.end_time, 20.0, places=5)

        # Expected unweighted score: (0.8 [trans] + 0.0 [conv] + 0.6 [vis] + 0.0 [pros]) / 4 = 1.4 / 4 = 0.35
        self.assertAlmostEqual(span.confidence, 0.35, places=5)

    def test_all_experts_no_mter_nms_suppression(self):
        # Two highly overlapping proposals [10, 25] and [11, 25]
        trans_bundle = ExpertEvidenceBundle(
            expert_name="transcript",
            proposals=[
                make_prop("t1", 10.0, 25.0, conf=0.9),
                make_prop("t2", 11.0, 25.0, conf=0.7),
            ],
        )
        empty_conv = ExpertEvidenceBundle(expert_name="conversation", proposals=[])
        empty_vis = ExpertEvidenceBundle(expert_name="visual", proposals=[])
        empty_pros = ExpertEvidenceBundle(expert_name="prosody", proposals=[])

        spans = run_all_experts_no_mter_algorithm(
            transcript_bundle=trans_bundle,
            conversation_bundle=empty_conv,
            visual_bundle=empty_vis,
            prosody_bundle=empty_pros,
            video_duration=40.0,
            min_duration_sec=5.0,
            nms_iou_threshold=0.30,
        )

        # NMS should suppress the lower scoring overlapping candidate
        self.assertEqual(len(spans), 1)
        self.assertAlmostEqual(spans[0].start_time, 10.0, places=4)


if __name__ == "__main__":
    unittest.main()
