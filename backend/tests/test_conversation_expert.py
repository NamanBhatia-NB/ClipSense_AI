"""
ClipSense Conversation Expert Unit Tests (W3)

Validates:
1. Schema conformity: produces valid ExpertEvidenceBundle and TemporalProposal instances.
2. Source metadata: source_type == SourceResolutionType.SPEECH_SEGMENT.
3. Single-speaker safety: explicitly enforces speaker_interaction == False for 1 detected speaker.
4. Multi-speaker exchange: handles multi-speaker dialogue transitions when num_speakers > 1.
5. Boundary temporal grounding: snaps to speech segment and word anchors.
6. Bounds verification: 0.0 <= start_time <= end_time <= video_duration.
7. Rejection of invalid/out-of-range estimates.
"""

import os
import pathlib
import sys
import unittest
from typing import Any, Optional

# Ensure backend root and tests dir are on sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
tests_dir = os.path.abspath(os.path.dirname(__file__))
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

from core.llm import MockLLMClient
from core.schemas import (
    ConversationData,
    ExpertEvidenceBundle,
    SourceResolutionType,
    SpeakerTurn,
    TimestampedWord,
    TranscriptData,
    TranscriptSegment,
)
from pipeline.experts.conversation_expert import (
    ConversationExpert,
    ConversationLLMCandidate,
    ConversationLLMResponse,
)


def create_sample_conversation_data(num_speakers: int = 1) -> ConversationData:
    """Creates a sample conversation data object with 3 speech segments."""
    s1 = "SPEAKER_00"
    s2 = "SPEAKER_01" if num_speakers > 1 else "SPEAKER_00"
    turns = [
        SpeakerTurn(
            turn_index=0,
            speaker=s1,
            start_time=1.0,
            end_time=8.5,
            duration=7.5,
            word_count=25,
            pause_before=1.0,
            pause_after=1.2,
        ),
        SpeakerTurn(
            turn_index=1,
            speaker=s2,
            start_time=9.7,
            end_time=18.2,
            duration=8.5,
            word_count=32,
            pause_before=1.2,
            pause_after=0.9,
        ),
        SpeakerTurn(
            turn_index=2,
            speaker=s1,
            start_time=19.1,
            end_time=28.0,
            duration=8.9,
            word_count=28,
            pause_before=0.9,
            pause_after=2.0,
        ),
    ]
    return ConversationData(
        duration=30.0,
        num_speakers=num_speakers,
        turns=turns,
        turn_frequency_per_minute=6.0,
        total_speech_time=24.9,
        total_pause_time=5.1,
    )


class TestConversationExpert(unittest.TestCase):
    def test_single_speaker_monologue_handling(self):
        """Correction 1 & 9: Verify single-speaker sample produces SPEECH_SEGMENT resolution and speaker_interaction=False."""
        conv_data = create_sample_conversation_data(num_speakers=1)

        # Mock LLM response that erroneously tries to claim speaker_interaction=True on a 1-speaker monologue
        mock_response = ConversationLLMResponse(
            candidates=[
                ConversationLLMCandidate(
                    start_time_estimate=1.0,
                    end_time_estimate=18.2,
                    evidence_type="monologue_thematic_unit",
                    explanation="Speaker presents complete premise and elaboration.",
                    confidence_estimate=0.88,
                    discourse_unit_complete=True,
                    speaker_interaction=True,  # Should be overridden to False
                    pause_boundary_supported=True,
                    discourse_phase="setup_development_payoff",
                )
            ]
        )
        expert = ConversationExpert(llm_client=MockLLMClient(mock_response=mock_response))
        bundle = expert.evaluate(conv_data, context={"total_duration": 30.0})

        self.assertIsInstance(bundle, ExpertEvidenceBundle)
        self.assertEqual(bundle.expert_name, "conversation")
        self.assertEqual(len(bundle.proposals), 1)

        p = bundle.proposals[0]
        # Verify resolution type is SPEECH_SEGMENT and anchor identifies speech segment
        self.assertEqual(p.source_metadata.source_type, SourceResolutionType.SPEECH_SEGMENT)
        self.assertTrue(p.source_metadata.alignment_anchor.startswith("speech_segment_"))
        self.assertNotIn("word", p.source_metadata.alignment_anchor)

        # Verify discourse_unit_complete is present and exchange_complete is eliminated
        self.assertTrue(p.supporting_features["discourse_unit_complete"])
        self.assertNotIn("exchange_complete", p.supporting_features)

        # Verify speaker_interaction is strictly False for single speaker
        self.assertFalse(p.supporting_features["speaker_interaction"])
        self.assertTrue(p.supporting_features["is_single_speaker_monologue"])
        self.assertEqual(p.supporting_features["num_speakers"], 1)

        # Timestamps remain exact continuous floating point
        self.assertAlmostEqual(p.start_time, 1.0, places=2)
        self.assertAlmostEqual(p.end_time, 18.2, places=2)
        self.assertAlmostEqual(p.duration, 17.2, places=2)

    def test_multi_speaker_interaction_handling(self):
        """Verify multi-speaker dialogue retains speaker_interaction=True when num_speakers > 1."""
        conv_data = create_sample_conversation_data(num_speakers=2)

        mock_response = ConversationLLMResponse(
            candidates=[
                ConversationLLMCandidate(
                    start_time_estimate=1.0,
                    end_time_estimate=18.2,
                    evidence_type="dialogue_exchange",
                    explanation="SPEAKER_00 asks a question and SPEAKER_01 provides rebuttal.",
                    confidence_estimate=0.91,
                    discourse_unit_complete=True,
                    speaker_interaction=True,
                    pause_boundary_supported=True,
                    discourse_phase="setup_development_payoff",
                )
            ]
        )
        expert = ConversationExpert(llm_client=MockLLMClient(mock_response=mock_response))
        bundle = expert.evaluate(conv_data, context={"total_duration": 30.0})

        self.assertEqual(len(bundle.proposals), 1)
        p = bundle.proposals[0]
        self.assertTrue(p.supporting_features["speaker_interaction"])
        self.assertFalse(p.supporting_features["is_single_speaker_monologue"])
        self.assertEqual(p.supporting_features["num_speakers"], 2)
        self.assertTrue(p.supporting_features["discourse_unit_complete"])
        self.assertNotIn("exchange_complete", p.supporting_features)

    def test_word_anchored_grounding(self):
        """Ground candidate boundaries to nearest word when proposal is a sub-segment interval."""
        conv_data = create_sample_conversation_data(num_speakers=1)
        # Speech segment 0 is [1.0 -> 8.5]. Candidate is an internal sub-segment [3.0 -> 7.0]
        words = [
            TimestampedWord(word="Hello", start=1.000, end=1.350),
            TimestampedWord(word="crucial", start=3.120, end=3.600),
            TimestampedWord(word="argument", start=3.650, end=4.200),
            TimestampedWord(word="payoff", start=6.800, end=7.150),
            TimestampedWord(word="point.", start=8.180, end=8.500),
        ]
        t_data = TranscriptData(full_text="Hello crucial argument payoff point.", duration=30.0, segments=[], words=words)

        mock_response = ConversationLLMResponse(
            candidates=[
                ConversationLLMCandidate(
                    start_time_estimate=3.0,  # Far from turn start (1.0), near word (3.120)
                    end_time_estimate=7.0,   # Far from turn end (8.5), near word (7.150)
                    evidence_type="monologue_unit",
                    explanation="Coherent sub-segment argument",
                    confidence_estimate=0.85,
                    discourse_unit_complete=True,
                    speaker_interaction=False,
                    pause_boundary_supported=False,
                    discourse_phase="payoff",
                )
            ]
        )
        expert = ConversationExpert(
            llm_client=MockLLMClient(mock_response=mock_response),
            min_candidate_duration_sec=2.0,
            max_candidate_duration_sec=15.0,
        )
        bundle = expert.evaluate(conv_data, context={"transcript_data": t_data, "total_duration": 30.0})

        self.assertEqual(len(bundle.proposals), 1)
        p = bundle.proposals[0]
        # Start snapped to closest word start: 3.120
        self.assertAlmostEqual(p.start_time, 3.120, places=3)
        self.assertAlmostEqual(p.end_time, 7.150, places=3)
        self.assertEqual(p.source_metadata.source_type, SourceResolutionType.WORD_TIMESTAMP)
        self.assertTrue(p.source_metadata.alignment_anchor.startswith("word_anchored_"))
        self.assertNotIn("speech_segment", p.source_metadata.alignment_anchor)

    def test_source_type_and_anchor_consistency(self):
        """Final consistency test: SPEECH_SEGMENT must have speech_segment anchor; WORD_TIMESTAMP must have word anchor."""
        conv_data = create_sample_conversation_data(num_speakers=1)
        words = [
            TimestampedWord(word="Hello", start=1.000, end=1.350),
            TimestampedWord(word="middle", start=4.000, end=4.500),
            TimestampedWord(word="segment", start=7.800, end=8.480),
        ]
        t_data = TranscriptData(full_text="Hello middle segment", duration=30.0, segments=[], words=words)

        # Candidate 1 aligns with Speech Segment 0 [1.0 -> 8.5]
        # Candidate 2 is sub-segment [3.8 -> 4.6]
        mock_response = ConversationLLMResponse(
            candidates=[
                ConversationLLMCandidate(
                    start_time_estimate=1.0,
                    end_time_estimate=8.5,
                    evidence_type="monologue_thematic_unit",
                    explanation="Full speech segment unit",
                    confidence_estimate=0.90,
                    discourse_unit_complete=True,
                    speaker_interaction=False,
                    pause_boundary_supported=True,
                    discourse_phase="setup_development_payoff",
                ),
                ConversationLLMCandidate(
                    start_time_estimate=3.8,
                    end_time_estimate=4.6,
                    evidence_type="monologue_sub_unit",
                    explanation="Sub-segment word excerpt",
                    confidence_estimate=0.80,
                    discourse_unit_complete=False,
                    speaker_interaction=False,
                    pause_boundary_supported=False,
                    discourse_phase="development",
                ),
            ]
        )
        expert = ConversationExpert(
            llm_client=MockLLMClient(mock_response=mock_response),
            min_candidate_duration_sec=0.5,
            max_candidate_duration_sec=15.0,
        )
        bundle = expert.evaluate(conv_data, context={"transcript_data": t_data, "total_duration": 30.0})

        self.assertEqual(len(bundle.proposals), 2)
        prop_segment, prop_word = bundle.proposals[0], bundle.proposals[1]

        # Proposal 1: Speech segment resolution
        self.assertEqual(prop_segment.source_metadata.source_type, SourceResolutionType.SPEECH_SEGMENT)
        self.assertEqual(prop_segment.source_metadata.alignment_anchor, "speech_segment_0")
        self.assertNotIn("word", prop_segment.source_metadata.alignment_anchor)

        # Proposal 2: Word timestamp resolution
        self.assertEqual(prop_word.source_metadata.source_type, SourceResolutionType.WORD_TIMESTAMP)
        self.assertTrue(prop_word.source_metadata.alignment_anchor.startswith("word_anchored_"))
        self.assertNotIn("speech_segment", prop_word.source_metadata.alignment_anchor)

        # Both must preserve monologue consistency
        for p in (prop_segment, prop_word):
            self.assertIn("discourse_unit_complete", p.supporting_features)
            self.assertNotIn("exchange_complete", p.supporting_features)
            self.assertFalse(p.supporting_features["speaker_interaction"])
            self.assertTrue(p.supporting_features["is_single_speaker_monologue"])

    def test_rejection_of_unjustified_whole_input_proposal(self):
        """Correction 6: Reject whole-input proposals that violate configured candidate constraints."""
        conv_data = create_sample_conversation_data(num_speakers=1)
        # Attempt to propose the entire 30s recording when max_candidate_duration_sec is 15.0s
        mock_response = ConversationLLMResponse(
            candidates=[
                ConversationLLMCandidate(
                    start_time_estimate=1.0,
                    end_time_estimate=28.0,  # 27s duration > 15s max
                    evidence_type="unjustified_entire_monologue",
                    explanation="Proposal spanning nearly the entire timeline.",
                    confidence_estimate=0.9,
                    discourse_unit_complete=True,
                    speaker_interaction=False,
                    pause_boundary_supported=True,
                    discourse_phase="setup_development_payoff",
                )
            ]
        )
        expert = ConversationExpert(
            llm_client=MockLLMClient(mock_response=mock_response),
            min_candidate_duration_sec=5.0,
            max_candidate_duration_sec=15.0,
        )
        bundle = expert.evaluate(conv_data, context={"total_duration": 30.0})

        # Must be rejected because 27s > max_candidate_duration_sec (15.0s)
        self.assertEqual(len(bundle.proposals), 0)

    def test_provenance_and_artifact_consistency(self):
        """Verify artifact provenance loading and spoken text availability for ConversationExpert."""
        try:
            from tests.run_w3_expert_validation import load_run_artifacts
        except ImportError:
            from run_w3_expert_validation import load_run_artifacts

        run_dir = pathlib.Path(__file__).parent.parent / "runs" / "representative_90s_validation"
        artifacts = load_run_artifacts(run_dir)

        self.assertEqual(artifacts["run_id"], "representative_90s_validation")
        self.assertIn("sample_conversational_90s.mp4", artifacts["source_video"])
        self.assertEqual(len(artifacts["t_sha256"]), 64)
        self.assertEqual(len(artifacts["c_sha256"]), 64)

        # Verify that both artifacts represent the exact same duration
        t_data: TranscriptData = artifacts["t_data"]
        c_data: ConversationData = artifacts["c_data"]
        self.assertAlmostEqual(t_data.duration, 89.865, places=2)
        self.assertAlmostEqual(c_data.duration, 90.0, places=1)

        # Verify missing directory raises FileNotFoundError
        fake_dir = pathlib.Path(__file__).parent / "non_existent_run"
        with self.assertRaises(FileNotFoundError):
            load_run_artifacts(fake_dir)

    def test_spoken_text_injected_into_conversation_prompt(self):
        """Verify that ConversationExpert injects actual timestamped spoken text into the LLM prompt."""
        conv_data = create_sample_conversation_data(num_speakers=1)
        words = [
            TimestampedWord(word="Parenting", start=1.000, end=1.800),
            TimestampedWord(word="discipline", start=1.900, end=2.800),
            TimestampedWord(word="is", start=2.900, end=3.100),
            TimestampedWord(word="essential.", start=3.200, end=4.200),
        ]
        t_data = TranscriptData(full_text="Parenting discipline is essential.", duration=30.0, segments=[], words=words)

        captured_prompt = []

        class PromptCapturingMockClient(MockLLMClient):
            def generate_structured(self, prompt: str, response_schema: Any, system_instruction: Optional[str] = None):
                captured_prompt.append(prompt)
                return super().generate_structured(prompt, response_schema, system_instruction)

        mock_response = ConversationLLMResponse(
            candidates=[
                ConversationLLMCandidate(
                    start_time_estimate=1.0,
                    end_time_estimate=8.5,
                    evidence_type="monologue_thematic_unit",
                    explanation="Speaker presents core premise on parenting discipline.",
                    confidence_estimate=0.90,
                    discourse_unit_complete=True,
                    speaker_interaction=False,
                    pause_boundary_supported=True,
                    discourse_phase="setup_development_payoff",
                )
            ]
        )
        mock_client = PromptCapturingMockClient(mock_response=mock_response)
        expert = ConversationExpert(llm_client=mock_client)
        expert.evaluate(conv_data, context={"transcript_data": t_data, "total_duration": 30.0})

        self.assertEqual(len(captured_prompt), 1)
        prompt_str = captured_prompt[0]
        # Assert spoken text is present in the prompt passed to the LLM
        self.assertIn("Spoken text:", prompt_str)
        self.assertIn("Parenting discipline is essential.", prompt_str)


if __name__ == "__main__":
    unittest.main()
