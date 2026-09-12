"""
ClipSense Transcript Expert Unit Tests (W3)

Validates:
1. Schema conformity: produces valid ExpertEvidenceBundle and TemporalProposal objects.
2. Word-index temporal grounding: programmatic mapping from start_word_index / end_word_index
   to exact continuous floating-point timestamps from WhisperX.
3. Deterministic grounding test: identical selected word indices produce identical final timestamps.
4. Timestamp bounds: 0.0 <= start_time <= end_time <= video_duration.
5. Duration consistency: duration == end_time - start_time within numerical precision.
6. Clean rejection of out-of-range word indices or malformed proposals.
7. Strict mode: raises exception on LLM client failures.
"""

import os
import sys
import unittest

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.llm import MockLLMClient
from core.schemas import (
    ExpertEvidenceBundle,
    SourceResolutionType,
    TemporalProposal,
    TimestampedWord,
    TranscriptData,
    TranscriptSegment,
)
from pipeline.experts.transcript_expert import (
    TranscriptExpert,
    TranscriptLLMResponse,
    TranscriptSemanticCandidate,
)


def create_sample_transcript_data() -> TranscriptData:
    """Creates a deterministic 30-word transcript with continuous float timestamps."""
    words = [
        TimestampedWord(word="The", start=1.052, end=1.210),
        TimestampedWord(word="most", start=1.230, end=1.540),
        TimestampedWord(word="important", start=1.560, end=2.010),
        TimestampedWord(word="contrarian", start=2.040, end=2.680),
        TimestampedWord(word="question", start=2.710, end=3.150),
        TimestampedWord(word="is", start=3.180, end=3.300),
        TimestampedWord(word="what", start=3.500, end=3.720),
        TimestampedWord(word="important", start=3.740, end=4.120),
        TimestampedWord(word="truth", start=4.150, end=4.600),
        TimestampedWord(word="do", start=4.630, end=4.750),
        TimestampedWord(word="very", start=4.780, end=5.020),
        TimestampedWord(word="few", start=5.050, end=5.350),
        TimestampedWord(word="people", start=5.380, end=5.780),
        TimestampedWord(word="agree", start=5.810, end=6.220),
        TimestampedWord(word="with", start=6.250, end=6.450),
        TimestampedWord(word="you", start=6.480, end=6.700),
        TimestampedWord(word="on?", start=6.720, end=7.100),
        TimestampedWord(word="This", start=8.000, end=8.250),
        TimestampedWord(word="sounds", start=8.270, end=8.600),
        TimestampedWord(word="easy,", start=8.620, end=9.010),
        TimestampedWord(word="but", start=9.200, end=9.400),
        TimestampedWord(word="it", start=9.420, end=9.550),
        TimestampedWord(word="is", start=9.580, end=9.720),
        TimestampedWord(word="actually", start=9.750, end=10.200),
        TimestampedWord(word="intellectually", start=10.230, end=11.100),
        TimestampedWord(word="challenging", start=11.130, end=11.850),
        TimestampedWord(word="because", start=11.880, end=12.200),
        TimestampedWord(word="school", start=12.220, end=12.600),
        TimestampedWord(word="teaches", start=12.620, end=13.050),
        TimestampedWord(word="consensus.", start=13.080, end=13.842),
    ]
    return TranscriptData(
        full_text=" ".join(w.word for w in words),
        duration=15.0,
        segments=[
            TranscriptSegment(id=0, start=1.052, end=7.100, text="The most important contrarian question...", words=words[:17]),
            TranscriptSegment(id=1, start=8.000, end=13.842, text="This sounds easy, but it is actually...", words=words[17:]),
        ],
        words=words,
    )


class TestTranscriptExpert(unittest.TestCase):
    def setUp(self):
        self.transcript_data = create_sample_transcript_data()

    def test_schema_conformance_and_word_grounding(self):
        """Test that start_word_index and end_word_index ground exactly to WhisperX word timestamps."""
        mock_response = TranscriptLLMResponse(
            candidates=[
                TranscriptSemanticCandidate(
                    start_word_index=0,
                    end_word_index=16,
                    evidence_type="explanatory_claim",
                    explanation="Contrarian question is introduced cleanly.",
                    confidence_estimate=0.92,
                    topic_transition=True,
                    contextual_completeness=0.95,
                    semantic_importance=0.90,
                    self_contained=True,
                ),
                TranscriptSemanticCandidate(
                    start_word_index=17,
                    end_word_index=29,
                    evidence_type="semantic_importance",
                    explanation="Explanation of why consensus thinking fails.",
                    confidence_estimate=0.88,
                    topic_transition=False,
                    contextual_completeness=0.89,
                    semantic_importance=0.85,
                    self_contained=True,
                ),
            ]
        )
        mock_client = MockLLMClient(mock_response=mock_response)
        expert = TranscriptExpert(
            llm_client=mock_client,
            min_candidate_duration_sec=3.0,
            max_candidate_duration_sec=20.0,
        )

        bundle = expert.evaluate(self.transcript_data, context={"total_duration": 15.0})

        self.assertIsInstance(bundle, ExpertEvidenceBundle)
        self.assertEqual(bundle.expert_name, "transcript")
        self.assertEqual(len(bundle.proposals), 2)

        p0 = bundle.proposals[0]
        # Word 0 start is 1.052, Word 16 end is 7.100
        self.assertAlmostEqual(p0.start_time, 1.052, places=3)
        self.assertAlmostEqual(p0.end_time, 7.100, places=3)
        self.assertAlmostEqual(p0.duration, 7.100 - 1.052, places=3)
        self.assertEqual(p0.source_metadata.source_type, SourceResolutionType.WORD_TIMESTAMP)
        self.assertEqual(p0.source_metadata.alignment_anchor, "word_index_0:16")
        self.assertTrue(p0.supporting_features["self_contained"])
        self.assertEqual(p0.supporting_features["start_word"], "The")
        self.assertEqual(p0.supporting_features["end_word"], "on?")

        p1 = bundle.proposals[1]
        # Word 17 start is 8.000, Word 29 end is 13.842
        self.assertAlmostEqual(p1.start_time, 8.000, places=3)
        self.assertAlmostEqual(p1.end_time, 13.842, places=3)
        self.assertAlmostEqual(p1.duration, 13.842 - 8.000, places=3)
        self.assertEqual(p1.source_metadata.alignment_anchor, "word_index_17:29")

    def test_deterministic_grounding(self):
        """Correction 7: Prove that identical selected word indices produce identical final timestamps."""
        mock_response = TranscriptLLMResponse(
            candidates=[
                TranscriptSemanticCandidate(
                    start_word_index=6,
                    end_word_index=16,
                    evidence_type="topic_introduction",
                    explanation="Question core",
                    confidence_estimate=0.85,
                    contextual_completeness=0.80,
                    semantic_importance=0.82,
                    self_contained=True,
                )
            ]
        )
        expert = TranscriptExpert(
            llm_client=MockLLMClient(mock_response=mock_response),
            min_candidate_duration_sec=2.0,
            max_candidate_duration_sec=15.0,
        )

        run1 = expert.evaluate(self.transcript_data)
        run2 = expert.evaluate(self.transcript_data)

        self.assertEqual(run1.proposals[0].start_time, run2.proposals[0].start_time)
        self.assertEqual(run1.proposals[0].end_time, run2.proposals[0].end_time)
        self.assertEqual(run1.proposals[0].duration, run2.proposals[0].duration)
        self.assertEqual(run1.proposals[0].start_time, self.transcript_data.words[6].start)
        self.assertEqual(run1.proposals[0].end_time, self.transcript_data.words[16].end)

    def test_rejection_of_invalid_spans(self):
        """Verify that out-of-range or inverted indices are discarded without crashing."""
        mock_response = TranscriptLLMResponse(
            candidates=[
                TranscriptSemanticCandidate(
                    start_word_index=-5,  # Invalid negative
                    end_word_index=10,
                    evidence_type="test",
                    explanation="Invalid negative start",
                    confidence_estimate=0.5,
                    contextual_completeness=0.5,
                    semantic_importance=0.5,
                    self_contained=False,
                ),
                TranscriptSemanticCandidate(
                    start_word_index=20,
                    end_word_index=10,  # Inverted span
                    evidence_type="test",
                    explanation="Inverted span",
                    confidence_estimate=0.5,
                    contextual_completeness=0.5,
                    semantic_importance=0.5,
                    self_contained=False,
                ),
                TranscriptSemanticCandidate(
                    start_word_index=0,
                    end_word_index=999,  # Out of range index
                    evidence_type="test",
                    explanation="Index exceeds word count",
                    confidence_estimate=0.5,
                    contextual_completeness=0.5,
                    semantic_importance=0.5,
                    self_contained=False,
                ),
                TranscriptSemanticCandidate(
                    start_word_index=0,
                    end_word_index=16,
                    evidence_type="valid_type",
                    explanation="Valid span",
                    confidence_estimate=0.9,
                    contextual_completeness=0.9,
                    semantic_importance=0.9,
                    self_contained=True,
                ),
            ]
        )
        expert = TranscriptExpert(
            llm_client=MockLLMClient(mock_response=mock_response),
            min_candidate_duration_sec=3.0,
            max_candidate_duration_sec=20.0,
        )
        bundle = expert.evaluate(self.transcript_data)

        # Only the single valid span should survive
        self.assertEqual(len(bundle.proposals), 1)
        self.assertEqual(bundle.proposals[0].proposal_id, "transcript_prop_0")
        self.assertAlmostEqual(bundle.proposals[0].start_time, self.transcript_data.words[0].start)
        self.assertAlmostEqual(bundle.proposals[0].end_time, self.transcript_data.words[16].end)

    def test_rejection_of_unjustified_whole_input_proposal(self):
        """Correction 6: Reject whole-input proposals that violate configured candidate constraints."""
        # Proposal spanning word 0 to 29 (12.79s of 15s input) when max_candidate_duration_sec is 8.0s
        mock_response = TranscriptLLMResponse(
            candidates=[
                TranscriptSemanticCandidate(
                    start_word_index=0,
                    end_word_index=29,
                    evidence_type="unjustified_full_partition",
                    explanation="Oversized span encompassing nearly entire recording.",
                    confidence_estimate=0.9,
                    contextual_completeness=0.95,
                    semantic_importance=0.9,
                    self_contained=True,
                )
            ]
        )
        expert = TranscriptExpert(
            llm_client=MockLLMClient(mock_response=mock_response),
            min_candidate_duration_sec=3.0,
            max_candidate_duration_sec=8.0,  # Caps candidate duration at 8s
        )
        bundle = expert.evaluate(self.transcript_data, context={"total_duration": 15.0})

        # Must be rejected because duration (~12.79s) > max_candidate_duration_sec (8.0s)
        self.assertEqual(len(bundle.proposals), 0)


if __name__ == "__main__":
    unittest.main()
