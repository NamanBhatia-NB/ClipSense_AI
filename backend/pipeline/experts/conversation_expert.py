"""
ClipSense Conversation Expert (W3)

Evaluates conversational structure and discourse dynamics:
- Conversational / discourse unit coherence (setup -> development -> payoff)
- Completion of explanatory exchanges or monologue arguments
- Inter-utterance pause boundaries and speech-to-pause pacing
- Multi-speaker interaction (when multiple speakers are present) vs.
  single-speaker monologue progression (strictly avoiding fabricated speaker interaction).

Temporal Grounding:
- Anchors candidate regions using speech segments, pause boundaries, and word-level timestamps.
- Ground boundaries to the nearest valid speech segment or word timestamp.
- Uses SourceResolutionType.SPEECH_SEGMENT for source metadata.
- Validates 0.0 <= start_time <= end_time <= video_duration with continuous floating-point precision.
"""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.config import config as app_config, ConversationConfig
from core.llm import LLMClient, get_llm_client
from core.schemas import (
    ConversationData,
    ExpertEvidenceBundle,
    ProposalSourceMetadata,
    SourceResolutionType,
    TemporalProposal,
    TranscriptData,
)
from pipeline.experts.base import BaseExpert

logger = logging.getLogger("clipsense.experts.conversation")


class ConversationLLMCandidate(BaseModel):
    """Raw structured conversational candidate proposal from LLM."""
    start_time_estimate: float = Field(
        ..., 
        description="Target start timestamp in seconds for this conversational discourse unit"
    )
    end_time_estimate: float = Field(
        ..., 
        description="Target end timestamp in seconds for this conversational discourse unit"
    )
    evidence_type: str = Field(
        ..., 
        description="Measurable discourse category (e.g. 'monologue_thematic_unit', 'dialogue_exchange', 'topic_transition')"
    )
    explanation: str = Field(
        ..., 
        description="Concise factual evidence justification of discourse completeness (no chain-of-thought)"
    )
    confidence_estimate: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence estimate of discourse coherence and structural integrity (0.0 to 1.0)"
    )
    discourse_unit_complete: bool = Field(
        True, 
        description="True if the speech segment or discourse unit reaches natural conversational or thematic closure"
    )
    exchange_complete: Optional[bool] = Field(
        None,
        description="Deprecated alias for discourse_unit_complete."
    )
    speaker_interaction: bool = Field(
        False, 
        description="True ONLY if multi-speaker dialogue/interaction occurs. Must be False for single-speaker monologues!"
    )
    pause_boundary_supported: bool = Field(
        True, 
        description="True if the candidate starts or ends near significant natural pause silences"
    )
    discourse_phase: str = Field(
        "setup_development_payoff", 
        description="Structural progression: 'setup', 'development', 'payoff', or 'setup_development_payoff'"
    )


class ConversationLLMResponse(BaseModel):
    """Container schema for structured LLM conversation response."""
    candidates: List[ConversationLLMCandidate] = Field(default_factory=list)


class ConversationExpert(BaseExpert):
    """
    Modality Evidence Expert for Conversational Structure and Discourse Units.
    Evaluates coherence of conversational exchanges or single-speaker monologue progression.
    Generates independent temporal evidence proposals (not final highlight clips).
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[Any] = None,
        min_candidate_duration_sec: Optional[float] = None,
        max_candidate_duration_sec: Optional[float] = None,
    ):
        cfg = config or app_config.conversation
        super().__init__(name="conversation", config=cfg)
        self.llm_client = llm_client or get_llm_client()
        self.min_duration = (
            min_candidate_duration_sec
            if min_candidate_duration_sec is not None
            else cfg.min_candidate_duration_sec
        )
        self.max_duration = (
            max_candidate_duration_sec
            if max_candidate_duration_sec is not None
            else cfg.max_candidate_duration_sec
        )

    def evaluate(
        self,
        extraction_data: ConversationData,
        context: Optional[Dict[str, Any]] = None,
        transcript_data: Optional[TranscriptData] = None,
    ) -> ExpertEvidenceBundle:
        """
        Evaluate ConversationData (with optional TranscriptData) and generate localized proposals.

        Args:
            extraction_data: Validated ConversationData from W2.
            context: Optional dict containing 'transcript_data': TranscriptData and 'total_duration': float.
            transcript_data: Optional direct reference to TranscriptData.

        Returns:
            ExpertEvidenceBundle containing validated, localized TemporalProposal instances.
        """
        conv_data = extraction_data
        t_data: Optional[TranscriptData] = (
            transcript_data
            if transcript_data is not None
            else (context.get("transcript_data") if context else None)
        )
        total_duration = (
            float(context.get("total_duration"))
            if context and context.get("total_duration")
            else conv_data.duration
        )

        turns = conv_data.turns
        if not turns:
            logger.warning("ConversationData contains no speech segments/turns. Returning empty bundle.")
            return ExpertEvidenceBundle(expert_name="conversation", proposals=[])

        is_single_speaker = conv_data.num_speakers <= 1
        words = t_data.words if t_data and t_data.words else []

        # 1. Format conversational structure overview with actual spoken words for LLM
        lines = []
        for t in turns:
            if words:
                seg_words = [
                    w.word for w in words
                    if not (w.end < t.start_time - 0.05 or w.start > t.end_time + 0.05)
                ]
                spoken_text = " ".join(seg_words).strip()
            elif t_data and t_data.segments:
                seg_texts = [
                    s.text for s in t_data.segments
                    if not (s.end < t.start_time - 0.05 or s.start > t.end_time + 0.05)
                ]
                spoken_text = " ".join(seg_texts).strip()
            else:
                spoken_text = "[Spoken transcript text unavailable]"

            lines.append(
                f"- Speech Segment {t.turn_index} ({t.speaker}): "
                f"[{t.start_time:.2f}s -> {t.end_time:.2f}s], duration={t.duration:.2f}s, {t.word_count} words, "
                f"pause_before={t.pause_before:.2f}s, pause_after={t.pause_after:.2f}s\n"
                f"  Spoken text: \"{spoken_text}\""
            )
        segments_overview = "\n".join(lines)

        speaker_context = (
            "SINGLE-SPEAKER MONOLOGUE: Only 1 speaker is present. "
            "Do NOT claim multi-speaker interaction or debate. "
            "Analyze monologue thought progression, rhetorical setup/development/payoff, and pause boundaries. "
            "Set speaker_interaction to FALSE and evaluate discourse_unit_complete."
            if is_single_speaker
            else f"MULTI-SPEAKER DIALOGUE: {conv_data.num_speakers} speakers detected. Analyze speaker transitions and dialogue exchanges."
        )

        prompt = f"""
You are the Conversation Evidence Expert in a multimodal podcast video clipping research system.
Your job is to identify localized, coherent conversational discourse units based on the actual spoken text and speech segment pacing (e.g., setup -> development -> payoff, focused question/answer sequence, or a single cohesive argumentative passage).

CRITICAL LOCALIZATION & FIDELITY RULES:
1. Ground your analysis strictly in the actual spoken text and discourse progression shown below.
2. Do NOT hallucinate or invent topics, arguments, or dialogue not present in the spoken text.
3. Do NOT select the entire recording or monologue as one candidate.
4. Identify localized candidate sub-intervals. A candidate may correspond to a single self-contained speech segment, a dialogue exchange, or a focused setup-development-payoff passage.
5. Target candidate duration MUST be between {self.min_duration:.0f}s and {self.max_duration:.0f}s.
6. You may propose 1 or multiple distinct localized candidate units.
7. Provide estimated start_time_estimate and end_time_estimate matching speech segment or pause boundaries.

{speaker_context}

TOTAL DURATION: {total_duration:.2f} seconds
SPEECH SEGMENTS EXTRACTED WITH SPOKEN TEXT ({len(turns)} segments):
{segments_overview}
"""

        system_instruction = (
            "You are an objective conversational structure and discourse analysis system. "
            "Output only factual, localized conversational candidate units based strictly on the provided spoken text and speech segments. "
            "Never propose the entire recording as a single candidate."
        )

        logger.info(f"Invoking LLM for Conversation Expert ({self.llm_client.provider}/{self.llm_client.model_name})...")
        llm_response: ConversationLLMResponse = self.llm_client.generate_structured(
            prompt=prompt,
            response_schema=ConversationLLMResponse,
            system_instruction=system_instruction,
        )

        # 2. Ground candidate boundaries to nearest valid speech segments or words
        proposals: List[TemporalProposal] = []
        words = t_data.words if t_data and t_data.words else []

        for idx, cand in enumerate(llm_response.candidates):
            raw_start = cand.start_time_estimate
            raw_end = cand.end_time_estimate

            # Find closest speech segment boundaries
            closest_start_turn = min(turns, key=lambda t: abs(t.start_time - raw_start))
            closest_end_turn = min(turns, key=lambda t: abs(t.end_time - raw_end))
            start_diff_turn = abs(raw_start - closest_start_turn.start_time)
            end_diff_turn = abs(raw_end - closest_end_turn.end_time)

            # Grounding decision:
            # If candidate aligns closely with speech segment boundaries (within 1.5s tolerance)
            # or if word-level timestamps are unavailable, ground to SPEECH_SEGMENT.
            # If the candidate is a sub-segment window that specifically targets word-level boundaries,
            # ground to WORD_TIMESTAMP.
            aligns_with_speech_segments = (
                not words or (start_diff_turn <= 1.5 and end_diff_turn <= 1.5)
            )

            if aligns_with_speech_segments:
                grounded_start = float(closest_start_turn.start_time)
                grounded_end = float(closest_end_turn.end_time)
                source_type = SourceResolutionType.SPEECH_SEGMENT
                temporal_res = self.config.turn_pause_threshold_sec
                if closest_start_turn.turn_index == closest_end_turn.turn_index:
                    anchor_desc = f"speech_segment_{closest_start_turn.turn_index}"
                else:
                    anchor_desc = f"speech_segment_{closest_start_turn.turn_index}:{closest_end_turn.turn_index}"
            else:
                # Sub-segment candidate grounded to word-level timestamps
                closest_start_word = min(words, key=lambda w: abs(w.start - raw_start))
                closest_end_word = min(words, key=lambda w: abs(w.end - raw_end))
                grounded_start = float(closest_start_word.start)
                grounded_end = float(closest_end_word.end)
                source_type = SourceResolutionType.WORD_TIMESTAMP
                anchor_desc = f"word_anchored_{closest_start_word.start:.2f}:{closest_end_word.end:.2f}"
                spanned = [w for w in words if w.start >= grounded_start - 1e-3 and w.end <= grounded_end + 1e-3]
                temporal_res = (grounded_end - grounded_start) / max(1, len(spanned))

            # Strict bounds validation: 0 <= start <= end <= total_duration
            if grounded_start < 0.0 or grounded_end > total_duration or grounded_start >= grounded_end:
                logger.warning(
                    f"Discarding invalid grounded conversation interval [{grounded_start:.2f}s -> {grounded_end:.2f}s] "
                    f"against total duration {total_duration:.2f}s"
                )
                continue

            duration = max(0.0, grounded_end - grounded_start)

            # Candidate duration constraints: reject whole-input or oversized proposals
            if duration > self.max_duration:
                logger.warning(
                    f"Discarding oversized conversation candidate [{grounded_start:.2f}s -> {grounded_end:.2f}s] "
                    f"(duration: {duration:.2f}s > max: {self.max_duration:.2f}s)"
                )
                continue

            if duration < self.min_duration:
                logger.warning(
                    f"Discarding undersized conversation candidate [{grounded_start:.2f}s -> {grounded_end:.2f}s] "
                    f"(duration: {duration:.2f}s < min: {self.min_duration:.2f}s)"
                )
                continue

            if total_duration > self.max_duration and duration >= 0.90 * total_duration:
                logger.warning(
                    f"Rejecting unjustified whole-input proposal spanning {duration:.2f}s of {total_duration:.2f}s"
                )
                continue

            # Enforce single-speaker constraint: speaker_interaction must be false if single speaker
            speaker_interaction = False if is_single_speaker else cand.speaker_interaction

            unit_complete = (
                cand.discourse_unit_complete
                if cand.discourse_unit_complete is not None
                else (cand.exchange_complete if cand.exchange_complete is not None else True)
            )

            proposal = TemporalProposal(
                proposal_id=f"conversation_prop_{idx}",
                start_time=grounded_start,
                end_time=grounded_end,
                duration=duration,
                confidence_estimate=cand.confidence_estimate,
                evidence_type=cand.evidence_type,
                explanation=cand.explanation,
                supporting_features={
                    "discourse_unit_complete": unit_complete,
                    "speaker_interaction": speaker_interaction,
                    "pause_boundary_supported": cand.pause_boundary_supported,
                    "discourse_phase": cand.discourse_phase,
                    "num_speakers": conv_data.num_speakers,
                    "is_single_speaker_monologue": is_single_speaker,
                },
                source_metadata=ProposalSourceMetadata(
                    source_type=source_type,
                    temporal_resolution_sec=temporal_res,
                    alignment_anchor=anchor_desc,
                ),
            )
            proposals.append(proposal)

        logger.info(f"Conversation Expert produced {len(proposals)} localized temporal proposals.")
        return ExpertEvidenceBundle(
            expert_name="conversation",
            proposals=proposals,
            dense_features_summary={
                "num_speakers": conv_data.num_speakers,
                "turn_frequency_per_minute": conv_data.turn_frequency_per_minute,
                "is_single_speaker_monologue": is_single_speaker,
                "model_used": f"{self.llm_client.provider}:{self.llm_client.model_name}",
            },
        )
