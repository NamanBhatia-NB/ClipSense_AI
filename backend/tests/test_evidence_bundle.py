"""
ClipSense W5: Evidence Bundle Unit Tests

Verifies:
1. Four expert bundles remain separate in CommonEvidenceBundle.
2. Continuous timestamps are preserved (floats, no rounding, no snapping).
3. Auxiliary grid indexing correctly maps proposals without overwriting timestamps.
4. Missing or empty expert evidence is handled cleanly.
5. JSON disk loading and serialization parity.
"""

import json
import os
import tempfile
import unittest

from core.schemas import (
    CommonEvidenceBundle,
    ExpertEvidenceBundle,
    ProposalSourceMetadata,
    SourceResolutionType,
    TemporalProposal,
)
from pipeline.mter.evidence_bundle import (
    build_auxiliary_grid,
    build_common_evidence_bundle,
    load_evidence_bundle_from_run,
)


def create_dummy_proposal(
    expert: str,
    prop_id: str,
    start: float,
    end: float,
    conf: float = 0.85,
    ev_type: str = "test_event",
) -> TemporalProposal:
    return TemporalProposal(
        proposal_id=prop_id,
        start_time=start,
        end_time=end,
        duration=end - start,
        confidence_estimate=conf,
        evidence_type=ev_type,
        explanation=f"Dummy proposal for {expert}",
        supporting_features={"test_key": 123},
        source_metadata=ProposalSourceMetadata(
            source_type=SourceResolutionType.WORD_TIMESTAMP if expert == "transcript" else SourceResolutionType.FRAME_TIMESTAMP,
            temporal_resolution_sec=0.25,
            alignment_anchor=f"{expert}_anchor_0",
        ),
    )


class TestEvidenceBundle(unittest.TestCase):

    def test_four_expert_bundles_remain_separate(self):
        """Validates that all 4 experts remain distinct without scalar score collapse."""
        p_t = create_dummy_proposal("transcript", "t1", 10.1234, 25.5678, conf=0.90)
        p_c = create_dummy_proposal("conversation", "c1", 10.0, 30.0, conf=0.80)
        p_v = create_dummy_proposal("visual", "v1", 12.0, 22.0, conf=0.70)
        p_p = create_dummy_proposal("prosody", "p1", 14.5, 24.5, conf=0.60)

        t_bundle = ExpertEvidenceBundle(expert_name="transcript", proposals=[p_t])
        c_bundle = ExpertEvidenceBundle(expert_name="conversation", proposals=[p_c])
        v_bundle = ExpertEvidenceBundle(expert_name="visual", proposals=[p_v])
        p_bundle = ExpertEvidenceBundle(expert_name="prosody", proposals=[p_p])

        bundle = build_common_evidence_bundle(
            video_id="test_vid",
            total_duration=40.0,
            transcript_evidence=t_bundle,
            conversation_evidence=c_bundle,
            visual_evidence=v_bundle,
            prosody_evidence=p_bundle,
        )

        self.assertEqual(bundle.video_id, "test_vid")
        self.assertEqual(bundle.total_duration, 40.0)
        self.assertEqual(len(bundle.transcript_evidence.proposals), 1)
        self.assertEqual(len(bundle.conversation_evidence.proposals), 1)
        self.assertEqual(len(bundle.visual_evidence.proposals), 1)
        self.assertEqual(len(bundle.prosody_evidence.proposals), 1)

        # Modality identities and confidences are isolated
        self.assertEqual(bundle.transcript_evidence.proposals[0].proposal_id, "t1")
        self.assertEqual(bundle.transcript_evidence.proposals[0].confidence_estimate, 0.90)
        self.assertEqual(bundle.conversation_evidence.proposals[0].proposal_id, "c1")
        self.assertEqual(bundle.visual_evidence.proposals[0].proposal_id, "v1")
        self.assertEqual(bundle.prosody_evidence.proposals[0].proposal_id, "p1")

    def test_continuous_timestamps_preserved(self):
        """Ensures that timestamps are continuous floats and not rounded or altered."""
        start_float = 12.3456789
        end_float = 28.9876543
        p = create_dummy_proposal("transcript", "t_exact", start_float, end_float)
        bundle = build_common_evidence_bundle(
            video_id="exact_ts",
            total_duration=50.0,
            transcript_evidence=ExpertEvidenceBundle(expert_name="transcript", proposals=[p]),
        )

        extracted = bundle.transcript_evidence.proposals[0]
        self.assertAlmostEqual(extracted.start_time, start_float, places=6)
        self.assertAlmostEqual(extracted.end_time, end_float, places=6)
        self.assertAlmostEqual(extracted.duration, end_float - start_float, places=6)

    def test_auxiliary_grid_lookup(self):
        """Verifies that the auxiliary grid indexes proposals into 1s bins without changing timestamps."""
        p1 = create_dummy_proposal("transcript", "t1", 1.2, 3.8)  # spans bin 1, 2, 3
        p2 = create_dummy_proposal("visual", "v1", 2.5, 4.5)      # spans bin 2, 3, 4

        grid = build_auxiliary_grid(
            total_duration=6.0,
            all_proposals={"transcript": [p1], "visual": [p2]},
            bin_size_sec=1.0,
        )

        self.assertEqual(grid.total_bins, 6)
        # Bin 0: [0.0, 1.0) -> neither
        self.assertNotIn("t1", grid.cells[0].active_proposal_ids.get("transcript", []))
        # Bin 2: [2.0, 3.0) -> both active
        self.assertIn("t1", grid.cells[2].active_proposal_ids["transcript"])
        self.assertIn("v1", grid.cells[2].active_proposal_ids["visual"])
        # Bin 4: [4.0, 5.0) -> only visual
        self.assertNotIn("transcript", grid.cells[4].active_proposal_ids)
        self.assertIn("v1", grid.cells[4].active_proposal_ids["visual"])

    def test_missing_or_empty_expert_bundles(self):
        """Checks graceful construction when some expert bundles are None or empty."""
        bundle = build_common_evidence_bundle(
            video_id="partial_vid",
            total_duration=30.0,
            transcript_evidence=None,
            visual_evidence=None,
        )

        self.assertEqual(len(bundle.transcript_evidence.proposals), 0)
        self.assertEqual(len(bundle.visual_evidence.proposals), 0)
        self.assertEqual(bundle.transcript_evidence.expert_name, "transcript")
        self.assertEqual(bundle.visual_evidence.expert_name, "visual")

    def test_load_evidence_bundle_from_run_dir(self):
        """Verifies loading from JSON artifacts in a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p_t = create_dummy_proposal("transcript", "tp1", 5.0, 15.0)
            p_v = create_dummy_proposal("visual", "vp1", 8.0, 18.0)

            t_bundle = ExpertEvidenceBundle(expert_name="transcript", proposals=[p_t])
            v_bundle = ExpertEvidenceBundle(expert_name="visual", proposals=[p_v])

            with open(os.path.join(tmpdir, "transcript_evidence.json"), "w", encoding="utf-8") as f:
                f.write(t_bundle.model_dump_json())
            with open(os.path.join(tmpdir, "visual_evidence.json"), "w", encoding="utf-8") as f:
                f.write(v_bundle.model_dump_json())

            # Load from directory
            loaded = load_evidence_bundle_from_run(tmpdir, total_duration=25.0)

            self.assertEqual(len(loaded.transcript_evidence.proposals), 1)
            self.assertEqual(loaded.transcript_evidence.proposals[0].proposal_id, "tp1")
            self.assertEqual(len(loaded.visual_evidence.proposals), 1)
            self.assertEqual(loaded.visual_evidence.proposals[0].proposal_id, "vp1")
            # Missing conversation and prosody should be empty bundles
            self.assertEqual(len(loaded.conversation_evidence.proposals), 0)
            self.assertEqual(len(loaded.prosody_evidence.proposals), 0)


if __name__ == "__main__":
    unittest.main()
