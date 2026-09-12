"""
ClipSense Prosody Expert Unit Tests (W4)

Validates:
1. Valid proposal schema: produces valid ExpertEvidenceBundle and TemporalProposal instances.
2. Valid timestamps: continuous floating-point seconds, ordered, finite, no NaN/inf.
3. Duration consistency: duration == end_time - start_time within numerical precision.
4. Source metadata correctness: SourceResolutionType.PROSODY_WINDOW, valid anchor and hop resolution.
5. Deterministic behavior: identical acoustic input and config produce identical proposals.
6. Handling of silent / low-voicing regions: excludes unvoiced silence from baseline corruption.
7. Configurable local-baseline parameters: respects custom baseline window and emphasis thresholds.
8. No hardcoded highlight thresholds: operates strictly on configurable multipliers.
9. Bounds preservation: all proposals satisfy 0.0 <= start_time <= end_time <= video_duration.
10. Whole-input rejection: rejects whole-input candidate spans when duration exceeds max_duration.
"""

import math
import os
import sys
import unittest
from typing import List

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import ProsodyConfig
from core.schemas import (
    ExpertEvidenceBundle,
    ProsodyData,
    ProsodyWindow,
    SourceResolutionType,
    TemporalProposal,
)
from pipeline.experts.prosody_expert import ProsodyExpert


def create_sample_prosody_data(
    num_windows: int = 120,
    window_length_sec: float = 1.0,
    hop_length_sec: float = 0.5,
    inject_emphasis: bool = True,
    inject_silence: bool = True,
) -> ProsodyData:
    """Creates synthetic ProsodyData with baseline speech, silence, and emphasis peaks."""
    total_duration = float((num_windows - 1) * hop_length_sec + window_length_sec)
    windows: List[ProsodyWindow] = []

    for i in range(num_windows):
        start_t = i * hop_length_sec
        end_t = start_t + window_length_sec

        # Default normal speech baseline
        mean_f0 = 160.0 + 10.0 * math.sin(i * 0.2)
        rms_energy = 0.08 + 0.01 * math.cos(i * 0.15)
        voicing = 0.65

        # Inject silence/unvoiced region in middle
        if inject_silence and 40 <= i < 55:
            mean_f0 = 0.0
            rms_energy = 0.002
            voicing = 0.05

        # Inject clear vocal emphasis peak in later section
        if inject_emphasis and 70 <= i < 90:
            mean_f0 = 240.0 + 20.0 * math.sin(i * 0.3)  # elevated pitch
            rms_energy = 0.18 + 0.02 * math.cos(i * 0.2)  # elevated loudness
            voicing = 0.85

        windows.append(
            ProsodyWindow(
                start_time=float(start_t),
                end_time=float(end_t),
                duration=float(window_length_sec),
                mean_f0=float(mean_f0),
                peak_f0=float(mean_f0 * 1.2),
                f0_std=15.0,
                rms_energy=float(rms_energy),
                rms_dynamic_range=0.15,
                voicing_fraction=float(voicing),
                speech_rate_wps=3.0 if voicing > 0.3 else 0.0,
            )
        )

    return ProsodyData(
        duration=total_duration,
        sample_rate=16000,
        window_length_sec=window_length_sec,
        hop_length_sec=hop_length_sec,
        windows=windows,
    )


class TestProsodyExpert(unittest.TestCase):
    """Unit tests for ProsodyExpert."""

    def setUp(self):
        self.config = ProsodyConfig(
            local_baseline_window_sec=20.0,
            min_voicing_fraction=0.3,
            emphasis_threshold_multiplier=1.20,
            min_candidate_duration_sec=10.0,
            max_candidate_duration_sec=30.0,
            merge_gap_sec=2.0,
        )
        self.expert = ProsodyExpert(config=self.config)

    def test_schema_conformity_and_source_metadata(self):
        """Verify output bundle, proposals, and source metadata conform to schema contracts."""
        p_data = create_sample_prosody_data()
        bundle = self.expert.evaluate(p_data)

        self.assertIsInstance(bundle, ExpertEvidenceBundle)
        self.assertEqual(bundle.expert_name, "prosody")
        self.assertIsInstance(bundle.proposals, list)
        self.assertGreater(len(bundle.proposals), 0)

        for prop in bundle.proposals:
            self.assertIsInstance(prop, TemporalProposal)
            self.assertTrue(prop.proposal_id.startswith("prosody_prop_"))
            self.assertIsInstance(prop.start_time, float)
            self.assertIsInstance(prop.end_time, float)
            self.assertIsInstance(prop.duration, float)

            # Metadata contract
            self.assertEqual(prop.source_metadata.source_type, SourceResolutionType.PROSODY_WINDOW)
            self.assertTrue(prop.source_metadata.alignment_anchor.startswith("window_"))
            self.assertEqual(prop.source_metadata.temporal_resolution_sec, p_data.hop_length_sec)

            # Supporting features
            self.assertIn("f0_change", prop.supporting_features)
            self.assertIn("energy_change", prop.supporting_features)
            self.assertIn("composite_emphasis", prop.supporting_features)
            self.assertIn("voicing_fraction", prop.supporting_features)

    def test_timestamps_valid_finite_and_continuous_float(self):
        """Verify timestamps are continuous floating-point numbers, finite, and ordered."""
        p_data = create_sample_prosody_data()
        bundle = self.expert.evaluate(p_data)

        for prop in bundle.proposals:
            # Finite and no NaN
            self.assertFalse(math.isnan(prop.start_time))
            self.assertFalse(math.isnan(prop.end_time))
            self.assertFalse(math.isinf(prop.start_time))
            self.assertFalse(math.isinf(prop.end_time))

            # Bounds
            self.assertGreaterEqual(prop.start_time, 0.0)
            self.assertLess(prop.start_time, prop.end_time)
            self.assertLessEqual(prop.end_time, p_data.duration)

            # Continuous floating point
            self.assertIsInstance(prop.start_time, float)
            self.assertIsInstance(prop.end_time, float)

    def test_duration_consistency(self):
        """Verify duration == end_time - start_time within numerical precision."""
        p_data = create_sample_prosody_data()
        bundle = self.expert.evaluate(p_data)

        for prop in bundle.proposals:
            expected_duration = prop.end_time - prop.start_time
            self.assertAlmostEqual(prop.duration, expected_duration, places=5)
            self.assertGreaterEqual(prop.duration, self.config.min_candidate_duration_sec - 1e-4)
            self.assertLessEqual(prop.duration, self.config.max_candidate_duration_sec + 1e-4)

    def test_confidence_estimate_bounded(self):
        """Verify non-probabilistic confidence estimate is bounded in [0.0, 1.0]."""
        p_data = create_sample_prosody_data()
        bundle = self.expert.evaluate(p_data)

        for prop in bundle.proposals:
            self.assertGreaterEqual(prop.confidence_estimate, 0.0)
            self.assertLessEqual(prop.confidence_estimate, 1.0)
            self.assertFalse(math.isnan(prop.confidence_estimate))

    def test_deterministic_behavior(self):
        """Verify identical inputs and configuration produce identical proposals."""
        p_data = create_sample_prosody_data()
        bundle_a = self.expert.evaluate(p_data)
        bundle_b = self.expert.evaluate(p_data)

        self.assertEqual(len(bundle_a.proposals), len(bundle_b.proposals))
        for p_a, p_b in zip(bundle_a.proposals, bundle_b.proposals):
            self.assertEqual(p_a.proposal_id, p_b.proposal_id)
            self.assertAlmostEqual(p_a.start_time, p_b.start_time, places=5)
            self.assertAlmostEqual(p_a.end_time, p_b.end_time, places=5)
            self.assertAlmostEqual(p_a.confidence_estimate, p_b.confidence_estimate, places=5)
            self.assertEqual(p_a.evidence_type, p_b.evidence_type)

    def test_silent_and_low_voicing_regions_filtered(self):
        """Verify silent unvoiced regions do not corrupt baseline or produce false highlights."""
        # Dataset with only silence
        silent_windows = [
            ProsodyWindow(
                start_time=float(i * 0.5),
                end_time=float(i * 0.5 + 1.0),
                duration=1.0,
                mean_f0=0.0,
                peak_f0=0.0,
                f0_std=0.0,
                rms_energy=0.001,
                rms_dynamic_range=0.001,
                voicing_fraction=0.05,
                speech_rate_wps=0.0,
            )
            for i in range(40)
        ]
        silent_data = ProsodyData(
            duration=20.5,
            sample_rate=16000,
            window_length_sec=1.0,
            hop_length_sec=0.5,
            windows=silent_windows,
        )
        bundle = self.expert.evaluate(silent_data)
        # Should have 0 proposals because all windows are unvoiced
        self.assertEqual(len(bundle.proposals), 0)

    def test_configurable_baseline_parameters(self):
        """Verify configurable parameters alter baseline computation without hardcoded magic numbers."""
        custom_cfg = ProsodyConfig(
            local_baseline_window_sec=10.0,
            emphasis_threshold_multiplier=1.40,
            min_candidate_duration_sec=12.0,
            max_candidate_duration_sec=25.0,
        )
        expert = ProsodyExpert(config=custom_cfg)
        p_data = create_sample_prosody_data()
        bundle = expert.evaluate(p_data)

        self.assertEqual(bundle.dense_features_summary.get("local_baseline_window_sec"), 10.0)
        for prop in bundle.proposals:
            self.assertGreaterEqual(prop.duration, 12.0 - 1e-4)
            self.assertLessEqual(prop.duration, 25.0 + 1e-4)

    def test_empty_and_insufficient_windows(self):
        """Verify graceful handling when window list is empty or insufficient."""
        empty_data = ProsodyData(
            duration=0.0,
            sample_rate=16000,
            window_length_sec=1.0,
            hop_length_sec=0.5,
            windows=[],
        )
        bundle = self.expert.evaluate(empty_data)
        self.assertEqual(len(bundle.proposals), 0)
        self.assertEqual(bundle.dense_features_summary.get("status"), "insufficient_windows")

    def test_rejection_of_unjustified_whole_input(self):
        """Verify expert does not propose the entire audio when duration > max_duration."""
        p_data = create_sample_prosody_data(num_windows=180, hop_length_sec=0.5)
        bundle = self.expert.evaluate(p_data)

        for prop in bundle.proposals:
            self.assertLess(prop.duration, 0.90 * p_data.duration)


if __name__ == "__main__":
    unittest.main()
