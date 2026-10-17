"""
Unit and Integration Tests for Evidence-Aware Temporal Boundary Refinement (Week 10)

Verifies:
1. Disabled refinement preserves original timestamps and candidate metadata identically.
2. Word boundary alignment and prevention of mid-word clipping.
3. Sentence/utterance boundary snapping.
4. Speech pause alignment supported by acoustic silence from prosody windows.
5. Sparse or completely missing evidence fallback handling.
6. Conflicting evidence resolution (avoiding visual scene cuts in the middle of words).
7. Minimum and maximum duration constraint enforcement (short and long candidate handling).
8. Ordering and media duration boundary clamping.
9. Continuous floating-point precision preservation (no integer quantization).
10. Deterministic identical results across repeated runs.
11. Rejection of invalid, negative, or non-finite timestamps.
12. Real sample video integration using actual multimodal extraction artifacts.
"""

import json
import math
from pathlib import Path
import unittest

from app.config import BoundaryRefinerConfig
from core.schemas import (
    BoundaryRefinementMetadata,
    CandidateSupportMetrics,
    ConversationData,
    MTERCandidate,
    ProsodyData,
    ProsodyWindow,
    SceneBoundary,
    SemanticVerificationCriteria,
    SpeakerTurn,
    TimestampedWord,
    TranscriptData,
    TranscriptSegment,
    VisualData,
)
from pipeline.optimization.boundary_refiner import BoundaryRefiner


def make_dummy_candidate(
    cand_id: str,
    start: float,
    end: float,
    conf: float = 0.85,
    iou: float = 0.70,
) -> MTERCandidate:
    """Helper to construct a fully valid MTERCandidate instance for testing."""
    dur = end - start
    return MTERCandidate(
        candidate_id=cand_id,
        proposed_start=start,
        proposed_end=end,
        duration=dur,
        confidence_estimate=conf,
        contributing_experts=["transcript", "conversation", "visual", "prosody"],
        temporal_agreement_iou=iou,
        modality_ious={"transcript": 0.8, "conversation": 0.7, "visual": 0.6, "prosody": 0.7},
        modality_coverages={"transcript": 0.8, "conversation": 0.7, "visual": 0.6, "prosody": 0.7},
        boundary_activity_support=0.6,
        linguistic_boundary_support=0.8,
        modalities_present=["transcript", "conversation", "visual", "prosody"],
        strong_support_modalities=["transcript", "conversation"],
        moderate_support_modalities=["visual", "prosody"],
        weak_support_modalities=[],
        support_metrics=CandidateSupportMetrics(
            transcript_support=0.8,
            conversation_support=0.7,
            visual_support=0.6,
            prosody_support=0.7,
            composite_score=conf,
        ),
        semantic_verification=SemanticVerificationCriteria(
            natural_semantic_start=True,
            sufficient_context=True,
            important_content_retained=True,
            complete_conversational_payoff=True,
            no_mid_sentence_ending=True,
        ),
        semantic_verification_status="passed",
        decision_summary=f"Test candidate {cand_id} [{start:.2f}s -> {end:.2f}s]",
    )


class TestBoundaryRefiner(unittest.TestCase):
    """Test suite for BoundaryRefiner unit and integration testing."""

    def setUp(self):
        self.refiner = BoundaryRefiner(
            BoundaryRefinerConfig(
                enabled=True,
                search_window_start_sec=1.5,
                search_window_end_sec=1.5,
                min_clip_duration_sec=10.0,
                max_clip_duration_sec=60.0,
                dead_air_buffer_sec=0.15,
                speech_pause_threshold_sec=0.25,
            )
        )

    def test_refiner_disabled_preserves_boundaries_exactly(self):
        """When enabled=False, timestamps are completely untouched and refined=False."""
        disabled_refiner = BoundaryRefiner(BoundaryRefinerConfig(enabled=False))
        cand = make_dummy_candidate("c1", 10.234, 30.567)

        refined = disabled_refiner.refine_candidate(cand)
        self.assertEqual(refined.proposed_start, 10.234)
        self.assertEqual(refined.proposed_end, 30.567)
        self.assertEqual(refined.duration, 30.567 - 10.234)
        self.assertIsNotNone(refined.boundary_refinement)
        self.assertFalse(refined.boundary_refinement.refined)
        self.assertEqual(refined.boundary_refinement.start_delta_sec, 0.0)
        self.assertEqual(refined.boundary_refinement.end_delta_sec, 0.0)

    def test_word_boundary_and_mid_word_cut_prevention(self):
        """
        Verify that if candidate cut starts inside a word, refiner shifts
        start before word onset (prevents mid-word cut).
        """
        words = [
            TimestampedWord(word="Hello", start=5.00, end=5.50),  # Cand starts at 5.25 (inside Hello!)
            TimestampedWord(word="world", start=5.60, end=6.10),
            TimestampedWord(word="final", start=25.00, end=25.80), # Cand ends at 25.40 (inside final!)
            TimestampedWord(word="word", start=26.50, end=27.00),
        ]
        t_data = TranscriptData(full_text="Hello world final word", duration=30.0, words=words)
        cand = make_dummy_candidate("c1", 5.25, 25.40)

        refined = self.refiner.refine_candidate(cand, transcript_data=t_data, video_duration=30.0)

        self.assertIsNotNone(refined.boundary_refinement)
        rf = refined.boundary_refinement
        self.assertTrue(rf.refined)
        # Start moved before 5.00s (e.g. 5.00 - 0.15 = 4.85s)
        self.assertLessEqual(refined.proposed_start, 5.00)
        self.assertIn("mid_word_cut_prevention", rf.signals_used)
        # End moved after 25.80s (e.g. 25.80 + 0.15 = 25.95s)
        self.assertGreaterEqual(refined.proposed_end, 25.80)

    def test_sentence_boundary_alignment(self):
        """Verify that sentence/segment boundaries are prioritized for clean semantic phrasing."""
        segments = [
            TranscriptSegment(id=1, start=6.00, end=15.00, text="First complete thought."),
            TranscriptSegment(id=2, start=15.20, end=28.00, text="Second complete thought."),
        ]
        words = [
            TimestampedWord(word="First", start=6.00, end=6.80),
            TimestampedWord(word="thought", start=14.20, end=15.00),
            TimestampedWord(word="Second", start=15.20, end=16.00),
            TimestampedWord(word="thought", start=27.20, end=28.00),
        ]
        t_data = TranscriptData(full_text="...", duration=40.0, segments=segments, words=words)
        # Candidate starts near sentence 1 (6.20s) and ends near sentence 2 (27.80s)
        cand = make_dummy_candidate("c1", 6.20, 27.80)

        refined = self.refiner.refine_candidate(cand, transcript_data=t_data, video_duration=40.0)
        rf = refined.boundary_refinement
        self.assertTrue(rf.refined)
        self.assertEqual(rf.start_evidence_type, "sentence_boundary")
        self.assertEqual(rf.end_evidence_type, "sentence_boundary")
        # Start aligns with 6.00s (- buffer) and end with 28.00s (+ buffer)
        self.assertAlmostEqual(refined.proposed_start, 5.85, places=2)
        self.assertAlmostEqual(refined.proposed_end, 28.15, places=2)

    def test_speech_pause_alignment_with_acoustic_silence(self):
        """Verify that an inter-word pause is chosen when supported by low acoustic energy."""
        words = [
            TimestampedWord(word="PartA", start=2.00, end=9.50),
            TimestampedWord(word="PartB", start=10.50, end=18.00), # 1.0s pause from 9.50s to 10.50s
        ]
        windows = [
            ProsodyWindow(
                start_time=9.0, end_time=10.0, duration=1.0, mean_f0=100.0, peak_f0=120.0,
                f0_std=10.0, rms_energy=0.01, rms_dynamic_range=0.02, voicing_fraction=0.05, speech_rate_wps=0.0
            ),
            ProsodyWindow(
                start_time=10.0, end_time=11.0, duration=1.0, mean_f0=100.0, peak_f0=120.0,
                f0_std=10.0, rms_energy=0.01, rms_dynamic_range=0.02, voicing_fraction=0.05, speech_rate_wps=0.0
            ),
        ]
        t_data = TranscriptData(full_text="PartA PartB", duration=25.0, words=words)
        p_data = ProsodyData(duration=25.0, sample_rate=16000, window_length_sec=1.0, hop_length_sec=0.5, windows=windows)

        cand = make_dummy_candidate("c1", 10.10, 22.00)
        refined = self.refiner.refine_candidate(cand, transcript_data=t_data, prosody_data=p_data, video_duration=25.0)

        rf = refined.boundary_refinement
        self.assertIn("prosody_silence_confirmation", rf.signals_used)

    def test_missing_and_sparse_evidence_fallback(self):
        """When evidence is empty or None, refiner gracefully retains original boundaries."""
        empty_t = TranscriptData(full_text="", duration=0.0, words=[], segments=[])
        cand = make_dummy_candidate("c1", 12.50, 32.50)

        refined = self.refiner.refine_candidate(cand, transcript_data=empty_t, video_duration=50.0)
        rf = refined.boundary_refinement
        self.assertFalse(rf.refined)
        self.assertEqual(refined.proposed_start, 12.50)
        self.assertEqual(refined.proposed_end, 32.50)
        self.assertIsNotNone(rf.fallback_reason)

    def test_conflicting_evidence_avoids_clipping_speech_for_scene_cut(self):
        """
        Verify that a visual scene cut in the middle of a spoken word is rejected
        in favor of clean speech boundary.
        """
        words = [
            TimestampedWord(word="SpeakingWord", start=9.50, end=11.00), # Spoken from 9.5s to 11.0s
            TimestampedWord(word="NextWord", start=12.00, end=15.00),
        ]
        # Visual scene cut occurs at 10.20s (right in middle of speaking word!)
        scenes = [
            SceneBoundary(scene_index=0, start_time=0.0, end_time=10.20, duration=10.20),
            SceneBoundary(scene_index=1, start_time=10.20, end_time=25.0, duration=14.80),
        ]
        t_data = TranscriptData(full_text="SpeakingWord NextWord", duration=25.0, words=words)
        v_data = VisualData(fps=30.0, total_frames=750, duration=25.0, scene_boundaries=scenes)

        cand = make_dummy_candidate("c1", 10.00, 22.00)
        refined = self.refiner.refine_candidate(cand, transcript_data=t_data, visual_data=v_data, video_duration=25.0)

        # Start should NOT snap to 10.20s (which cuts into SpeakingWord)
        self.assertNotEqual(refined.proposed_start, 10.20)
        # It must snap before 9.50s
        self.assertLessEqual(refined.proposed_start, 9.50)

    def test_duration_bounds_enforcement_short_candidate(self):
        """
        Verify that if snapping would shrink an interval below min_clip_duration_sec,
        refiner falls back or clamps to satisfy minimum duration.
        """
        words = [
            TimestampedWord(word="A", start=10.0, end=11.0),
            TimestampedWord(word="B", start=18.0, end=19.0),
        ]
        t_data = TranscriptData(full_text="A B", duration=30.0, words=words)
        # Candidate is 10.5s duration (satisfies min_clip_duration_sec=10.0s)
        # but if end snaps to 19.0s and start to 10.0s, proposed duration is 9.0s (<10.0s)
        cand = make_dummy_candidate("c1", 10.0, 20.5)

        refined = self.refiner.refine_candidate(cand, transcript_data=t_data, video_duration=30.0)
        self.assertGreaterEqual(refined.duration, 10.0)

    def test_duration_bounds_enforcement_long_candidate(self):
        """
        Verify that refiner does not expand an already long candidate beyond max_clip_duration_sec.
        """
        words = [
            TimestampedWord(word="Start", start=0.0, end=1.0),
            TimestampedWord(word="End", start=62.0, end=63.0),
        ]
        t_data = TranscriptData(full_text="Start End", duration=70.0, words=words)
        # Candidate is 59.5s (within 60.0s max)
        cand = make_dummy_candidate("c1", 1.0, 60.5)

        refined = self.refiner.refine_candidate(cand, transcript_data=t_data, video_duration=70.0)
        self.assertLessEqual(refined.duration, 60.0)

    def test_ordering_and_media_duration_constraints(self):
        """Verify that refined boundaries never exceed video duration or become negative."""
        words = [TimestampedWord(word="test", start=49.0, end=50.0)]
        t_data = TranscriptData(full_text="test", duration=50.0, words=words)
        cand = make_dummy_candidate("c1", 45.0, 49.5)

        refined = self.refiner.refine_candidate(cand, transcript_data=t_data, video_duration=50.0)
        self.assertGreaterEqual(refined.proposed_start, 0.0)
        self.assertLessEqual(refined.proposed_end, 50.0)
        self.assertLess(refined.proposed_start, refined.proposed_end)

    def test_floating_point_precision_preserved(self):
        """Verify timestamps retain continuous float precision rather than integer seconds."""
        words = [
            TimestampedWord(word="precise", start=3.1415, end=3.7892),
            TimestampedWord(word="float", start=24.5678, end=25.1234),
        ]
        t_data = TranscriptData(full_text="precise float", duration=30.0, words=words)
        cand = make_dummy_candidate("c1", 3.200, 25.000)

        refined = self.refiner.refine_candidate(cand, transcript_data=t_data, video_duration=30.0)
        # Check that timestamps have non-zero fractional components
        self.assertNotEqual(refined.proposed_start, float(int(refined.proposed_start)))
        self.assertNotEqual(refined.proposed_end, float(int(refined.proposed_end)))

    def test_deterministic_repeated_runs(self):
        """Verify that 10 repeated runs on the same candidate produce bitwise identical results."""
        words = [
            TimestampedWord(word="Alpha", start=5.10, end=5.90),
            TimestampedWord(word="Beta", start=12.20, end=13.00),
            TimestampedWord(word="Gamma", start=25.40, end=26.20),
        ]
        t_data = TranscriptData(full_text="Alpha Beta Gamma", duration=30.0, words=words)
        cand = make_dummy_candidate("c1", 5.50, 25.80)

        first_run = self.refiner.refine_candidate(cand, transcript_data=t_data, video_duration=30.0)
        for _ in range(10):
            next_run = self.refiner.refine_candidate(cand, transcript_data=t_data, video_duration=30.0)
            self.assertEqual(first_run.proposed_start, next_run.proposed_start)
            self.assertEqual(first_run.proposed_end, next_run.proposed_end)
            self.assertEqual(first_run.duration, next_run.duration)
            self.assertEqual(
                first_run.boundary_refinement.rationale,
                next_run.boundary_refinement.rationale,
            )

    def test_invalid_and_non_finite_timestamps_rejected_safely(self):
        """Reversed, negative, or NaN timestamps are safely rejected without crashing."""
        cand_nan = make_dummy_candidate("nan_cand", float("nan"), 20.0)
        refined_nan = self.refiner.refine_candidate(cand_nan, video_duration=30.0)
        self.assertFalse(refined_nan.boundary_refinement.refined)
        self.assertIn("non-finite", refined_nan.boundary_refinement.rationale)

        cand_rev = make_dummy_candidate("rev_cand", 25.0, 10.0)
        refined_rev = self.refiner.refine_candidate(cand_rev, video_duration=30.0)
        self.assertFalse(refined_rev.boundary_refinement.refined)
        self.assertIn("reversed", refined_rev.boundary_refinement.rationale)

    def test_real_sample_video_refinement_integration(self):
        """
        Integration test verifying refinement on actual extracted features of
        sample_conversational_90s.mp4.
        """
        backend_dir = Path(__file__).parent.parent
        cache_dir = backend_dir / "runs" / "representative_90s_validation"
        t_file = cache_dir / "transcript.json"
        p_file = cache_dir / "prosody.json"
        v_file = cache_dir / "frames.json"

        if not t_file.exists():
            self.skipTest(f"Pre-extracted artifacts not found in {cache_dir}")

        with open(t_file, "r", encoding="utf-8") as f:
            t_data = TranscriptData.model_validate_json(f.read())
        with open(p_file, "r", encoding="utf-8") as f:
            p_data = ProsodyData.model_validate_json(f.read())
        with open(v_file, "r", encoding="utf-8") as f:
            v_data = VisualData.model_validate_json(f.read())

        # Original MTER highlight on 90s sample: [0.03s -> 26.11s]
        cand = make_dummy_candidate("sample_highlight_1", 0.03, 26.11)

        refined = self.refiner.refine_candidate(
            candidate=cand,
            transcript_data=t_data,
            prosody_data=p_data,
            visual_data=v_data,
            video_duration=90.0,
        )

        rf = refined.boundary_refinement
        self.assertIsNotNone(rf)
        self.assertGreaterEqual(refined.proposed_start, 0.0)
        self.assertLessEqual(refined.proposed_end, 90.0)
        self.assertGreaterEqual(refined.duration, 10.0)
        self.assertLessEqual(refined.duration, 60.0)
        self.assertTrue(len(rf.signals_used) > 0)


if __name__ == "__main__":
    unittest.main()
