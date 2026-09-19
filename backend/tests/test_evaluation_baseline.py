"""
Unit Tests for Legacy Transcript Baseline (W6)
"""

import sys
import tempfile
import unittest

from core.schemas import TimestampedWord, TranscriptData
from evaluation.baseline import (
    BaselineConfig,
    BaselineLLMCandidate,
    BaselineLLMResponse,
    LegacyTranscriptBaselineSelector,
)
from evaluation.schemas import EvaluationSpan


class MockBaselineClient:
    """Mock client returning deterministic word-indexed candidates."""

    def generate_structured(self, prompt: str, response_schema: any, system_instruction: str = None):
        return BaselineLLMResponse(
            candidates=[
                BaselineLLMCandidate(
                    start_word_index=1,
                    end_word_index=4,
                    confidence=0.88,
                    title="Mock Topic",
                    summary="Coherent spoken point",
                )
            ]
        )


class TestEvaluationBaseline(unittest.TestCase):

    def setUp(self):
        # 10 words spaced by 1.0s: word i spans [i.0, i.0 + 0.8]
        self.words = [
            TimestampedWord(word=f"word_{i}", start=float(i * 3.0), end=float(i * 3.0 + 2.0))
            for i in range(10)
        ]
        self.transcript = TranscriptData(
            full_text=" ".join(f"word_{i}" for i in range(10)),
            duration=30.0,
            words=self.words,
        )

    def test_window_generation_deterministic(self):
        selector = LegacyTranscriptBaselineSelector(
            config=BaselineConfig(window_duration_sec=60.0, overlap_sec=15.0)
        )
        # Short duration <= 60s -> exactly 1 window
        windows_short = selector.generate_windows(self.transcript, video_duration=30.0)
        self.assertEqual(len(windows_short), 1)
        self.assertEqual(windows_short[0]["start_sec"], 0.0)
        self.assertEqual(windows_short[0]["end_sec"], 30.0)

        # Longer duration -> multiple windows with fixed stride = 45s
        windows_long = selector.generate_windows(self.transcript, video_duration=100.0)
        self.assertEqual(len(windows_long), 2)
        self.assertEqual(windows_long[0]["start_sec"], 0.0)
        self.assertEqual(windows_long[0]["end_sec"], 60.0)
        self.assertEqual(windows_long[1]["start_sec"], 45.0)
        self.assertEqual(windows_long[1]["end_sec"], 100.0)

    def test_baseline_word_snapping_and_run(self):
        client = MockBaselineClient()
        selector = LegacyTranscriptBaselineSelector(
            llm_client=client,
            config=BaselineConfig(min_candidate_duration_sec=5.0, max_candidate_duration_sec=60.0),
        )

        spans = selector.run(self.transcript, video_duration=30.0, sample_id="sample_test")
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.mode_name, "baseline")
        # Word index 1 starts at 3.0s; word index 4 ends at 4 * 3.0 + 2.0 = 14.0s
        self.assertAlmostEqual(span.start_time, 3.0, places=5)
        self.assertAlmostEqual(span.end_time, 14.0, places=5)
        self.assertAlmostEqual(span.duration, 11.0, places=5)

    def test_baseline_caching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MockBaselineClient()
            cfg_live = BaselineConfig(
                inference_mode="live_inference",
                cache_dir=tmpdir,
                min_candidate_duration_sec=5.0,
            )
            selector_live = LegacyTranscriptBaselineSelector(llm_client=client, config=cfg_live)
            spans1 = selector_live.run(self.transcript, video_duration=30.0, sample_id="cached_sample")
            self.assertEqual(len(spans1), 1)

            # Second selector in cached_inference mode WITHOUT an LLM client
            cfg_cached = BaselineConfig(
                inference_mode="cached_inference",
                cache_dir=tmpdir,
            )
            selector_cached = LegacyTranscriptBaselineSelector(llm_client=None, config=cfg_cached)
            spans2 = selector_cached.run(self.transcript, video_duration=30.0, sample_id="cached_sample")
            self.assertEqual(len(spans2), 1)
            self.assertEqual(spans1[0].start_time, spans2[0].start_time)
            self.assertEqual(spans1[0].end_time, spans2[0].end_time)

    def test_baseline_has_no_expert_or_mter_imports(self):
        import evaluation.baseline as base_mod
        with open(base_mod.__file__, "r", encoding="utf-8") as f:
            code = f.read()
        self.assertNotIn("VisualExpert", code)
        self.assertNotIn("ProsodyExpert", code)
        self.assertNotIn("ConversationExpert", code)
        self.assertNotIn("MTERReasoner", code)
        self.assertNotIn("pipeline.mter", code)


if __name__ == "__main__":
    unittest.main()
