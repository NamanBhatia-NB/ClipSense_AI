"""
ClipSense Visual Expert Unit Tests (W4)

Validates:
1. Valid proposal schema: produces valid ExpertEvidenceBundle and TemporalProposal instances.
2. Valid timestamps: continuous floating-point seconds, ordered, finite, no NaN/inf.
3. Duration consistency: duration == end_time - start_time within numerical precision.
4. Source metadata correctness: SourceResolutionType.FRAME_TIMESTAMP, valid anchor and resolution.
5. Deterministic behavior: identical visual input and config produce identical proposals.
6. Configurable candidate duration handling: respects min_duration and max_duration bounds.
7. Empty / insufficient frame input handling: graceful degradation to empty proposals without error.
8. Bounds preservation: all proposals satisfy 0.0 <= start_time <= end_time <= video_duration.
9. Whole-input rejection: rejects whole-input candidate spans when duration exceeds max_duration.
"""

import math
import os
import sys
import unittest
from typing import List

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import VisualConfig
from core.schemas import (
    ExpertEvidenceBundle,
    SampledFrameMetadata,
    SceneBoundary,
    SourceResolutionType,
    TemporalProposal,
    VisualData,
)
from pipeline.experts.visual_expert import (
    LightweightFrameDiffExtractor,
    VisualExpert,
)


def create_sample_visual_data(
    num_frames: int = 60,
    fps: float = 1.0,
    duration: float = 60.0,
    include_scenes: bool = True,
) -> VisualData:
    """Creates synthetic sample VisualData for testing."""
    sampled_frames: List[SampledFrameMetadata] = []
    for i in range(num_frames):
        sampled_frames.append(
            SampledFrameMetadata(
                frame_index=int(i * (25.0 / fps)),
                timestamp=float(i / fps),
                scene_index=0 if i < 20 else (1 if i < 40 else 2),
                file_path=None,
                width=640,
                height=360,
            )
        )

    scenes: List[SceneBoundary] = []
    if include_scenes:
        scenes = [
            SceneBoundary(scene_index=0, start_time=0.0, end_time=20.0, duration=20.0),
            SceneBoundary(scene_index=1, start_time=20.0, end_time=40.0, duration=20.0),
            SceneBoundary(scene_index=2, start_time=40.0, end_time=duration, duration=duration - 40.0),
        ]

    return VisualData(
        fps=fps,
        total_frames=int(duration * 25.0),
        duration=duration,
        scene_boundaries=scenes,
        sampled_frames=sampled_frames,
    )


class TestVisualExpert(unittest.TestCase):
    """Unit tests for VisualExpert."""

    def setUp(self):
        self.config = VisualConfig(
            min_candidate_duration_sec=10.0,
            max_candidate_duration_sec=30.0,
            activity_smoothing_sec=3.0,
            min_activity_percentile=60.0,
        )
        self.expert = VisualExpert(config=self.config)

    def test_schema_conformity_and_source_metadata(self):
        """Verify output bundle, proposals, and source metadata conform to schema contracts."""
        v_data = create_sample_visual_data(num_frames=60, duration=60.0)
        bundle = self.expert.evaluate(v_data)

        self.assertIsInstance(bundle, ExpertEvidenceBundle)
        self.assertEqual(bundle.expert_name, "visual")
        self.assertIsInstance(bundle.proposals, list)
        self.assertGreater(len(bundle.proposals), 0)

        for prop in bundle.proposals:
            self.assertIsInstance(prop, TemporalProposal)
            self.assertTrue(prop.proposal_id.startswith("visual_prop_"))
            self.assertIsInstance(prop.start_time, float)
            self.assertIsInstance(prop.end_time, float)
            self.assertIsInstance(prop.duration, float)

            # Metadata contract
            self.assertEqual(prop.source_metadata.source_type, SourceResolutionType.FRAME_TIMESTAMP)
            self.assertTrue(prop.source_metadata.alignment_anchor.startswith("sampled_frame_"))
            self.assertEqual(prop.source_metadata.temporal_resolution_sec, 1.0)
            self.assertEqual(prop.source_metadata.sample_rate_or_fps, 1.0)

            # Supporting features
            self.assertIn("visual_change_score", prop.supporting_features)
            self.assertIn("scene_change_count", prop.supporting_features)
            self.assertIn("frame_count", prop.supporting_features)
            self.assertIn("start_sampled_frame_index", prop.supporting_features)
            self.assertIn("start_native_frame_index", prop.supporting_features)
            self.assertIn("sampling_interval_sec", prop.supporting_features)
            self.assertIn("native_fps", prop.supporting_features)

    def test_timestamps_valid_finite_and_continuous_float(self):
        """Verify timestamps are continuous floating-point numbers, finite, and ordered."""
        v_data = create_sample_visual_data(num_frames=90, duration=90.0)
        bundle = self.expert.evaluate(v_data)

        for prop in bundle.proposals:
            # Finite and no NaN
            self.assertFalse(math.isnan(prop.start_time))
            self.assertFalse(math.isnan(prop.end_time))
            self.assertFalse(math.isinf(prop.start_time))
            self.assertFalse(math.isinf(prop.end_time))

            # Bounds
            self.assertGreaterEqual(prop.start_time, 0.0)
            self.assertLess(prop.start_time, prop.end_time)
            self.assertLessEqual(prop.end_time, v_data.duration)

            # Continuous floating point
            self.assertIsInstance(prop.start_time, float)
            self.assertIsInstance(prop.end_time, float)

    def test_duration_consistency(self):
        """Verify duration == end_time - start_time within numerical precision."""
        v_data = create_sample_visual_data(num_frames=60, duration=60.0)
        bundle = self.expert.evaluate(v_data)

        for prop in bundle.proposals:
            expected_duration = prop.end_time - prop.start_time
            self.assertAlmostEqual(prop.duration, expected_duration, places=5)
            self.assertGreaterEqual(prop.duration, self.config.min_candidate_duration_sec - 1e-4)
            self.assertLessEqual(prop.duration, self.config.max_candidate_duration_sec + 1e-4)

    def test_confidence_estimate_bounded(self):
        """Verify non-probabilistic confidence estimate is bounded in [0.0, 1.0]."""
        v_data = create_sample_visual_data(num_frames=60, duration=60.0)
        bundle = self.expert.evaluate(v_data)

        for prop in bundle.proposals:
            self.assertGreaterEqual(prop.confidence_estimate, 0.0)
            self.assertLessEqual(prop.confidence_estimate, 1.0)
            self.assertFalse(math.isnan(prop.confidence_estimate))

    def test_deterministic_behavior(self):
        """Verify identical inputs and configuration produce identical proposals."""
        v_data = create_sample_visual_data(num_frames=60, duration=60.0)
        bundle_a = self.expert.evaluate(v_data)
        bundle_b = self.expert.evaluate(v_data)

        self.assertEqual(len(bundle_a.proposals), len(bundle_b.proposals))
        for p_a, p_b in zip(bundle_a.proposals, bundle_b.proposals):
            self.assertEqual(p_a.proposal_id, p_b.proposal_id)
            self.assertAlmostEqual(p_a.start_time, p_b.start_time, places=5)
            self.assertAlmostEqual(p_a.end_time, p_b.end_time, places=5)
            self.assertAlmostEqual(p_a.confidence_estimate, p_b.confidence_estimate, places=5)
            self.assertEqual(p_a.evidence_type, p_b.evidence_type)

    def test_empty_and_insufficient_frames_handling(self):
        """Verify graceful handling when frame list is empty or insufficient."""
        empty_data = VisualData(
            fps=1.0,
            total_frames=0,
            duration=0.0,
            scene_boundaries=[],
            sampled_frames=[],
        )
        bundle_empty = self.expert.evaluate(empty_data)
        self.assertEqual(len(bundle_empty.proposals), 0)
        self.assertEqual(bundle_empty.dense_features_summary.get("status"), "insufficient_frames")

        one_frame_data = VisualData(
            fps=1.0,
            total_frames=25,
            duration=1.0,
            scene_boundaries=[],
            sampled_frames=[
                SampledFrameMetadata(frame_index=0, timestamp=0.0, scene_index=0)
            ],
        )
        bundle_one = self.expert.evaluate(one_frame_data)
        self.assertEqual(len(bundle_one.proposals), 0)

    def test_configurable_duration_bounds(self):
        """Verify proposals obey custom min and max candidate duration bounds."""
        custom_cfg = VisualConfig(
            min_candidate_duration_sec=15.0,
            max_candidate_duration_sec=25.0,
            activity_smoothing_sec=2.0,
            min_activity_percentile=50.0,
        )
        expert = VisualExpert(config=custom_cfg)
        v_data = create_sample_visual_data(num_frames=80, duration=80.0)
        bundle = expert.evaluate(v_data)

        for prop in bundle.proposals:
            self.assertGreaterEqual(prop.duration, 15.0 - 1e-4)
            self.assertLessEqual(prop.duration, 25.0 + 1e-4)

    def test_rejection_of_unjustified_whole_input(self):
        """Verify expert does not propose the entire video when total duration > max_duration."""
        v_data = create_sample_visual_data(num_frames=120, duration=120.0)
        bundle = self.expert.evaluate(v_data)

        for prop in bundle.proposals:
            self.assertLess(prop.duration, 0.90 * v_data.duration)


if __name__ == "__main__":
    unittest.main()
