"""
ClipSense Transcript Expert (W3)

Evaluates timestamped spoken transcripts for semantic and linguistic evidence:
- Semantic importance and informativeness
- Meaningful topic introduction and conceptual shifts
- Core claims and self-contained explanations
- Question-and-answer discourse
- Natural semantic beginnings and endings

Temporal Grounding:
- Formats transcript words with exact 0-based word indices.
- Requests LLM selection of candidate spans via start_word_index and end_word_index.
- Programmatically maps selected indices to exact continuous floating-point timestamps
  derived from WhisperX forced alignment (SourceResolutionType.WORD_TIMESTAMP).
- Enforces strict validation: 0.0 <= start_time <= end_time <= video_duration.
- Emits ExpertEvidenceBundle containing independent TemporalProposal objects.
"""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.config import config as app_config, TranscriptConfig
from core.llm import LLMClient, get_llm_client
from core.schemas import (
    ExpertEvidenceBundle,
    ProposalSourceMetadata,
    SourceResolutionType,
    TemporalProposal,
    TranscriptData,
)
from pipeline.experts.base import BaseExpert

logger = logging.getLogger("clipsense.experts.transcript")


class TranscriptSemanticCandidate(BaseModel):
    """Raw structured semantic candidate output from LLM."""
    start_word_index: int = Field(
        ..., 
        description="0-based index of the first word marking the natural semantic beginning"
    )
    end_word_index: int = Field(
        ..., 
        description="0-based index of the last word marking the natural semantic ending (inclusive)"
    )
    evidence_type: str = Field(
        ..., 
        description="Measurable semantic category (e.g. 'semantic_importance', 'explanatory_claim', 'topic_introduction')"
    )
    explanation: str = Field(
        ..., 
        description="Concise factual evidence justification (no chain-of-thought, no private reasoning)"
    )
    confidence_estimate: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence estimate of semantic coherence and completeness (0.0 to 1.0)"
    )
    topic_transition: bool = Field(
        False, 
        description="True if a distinct topic introduction or conceptual transition occurs"
    )
    contextual_completeness: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Contextual completeness score (is the passage self-contained without missing context?)"
    )
    semantic_importance: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Informational value and clarity of the core claim or explanation"
    )
    self_contained: bool = Field(
        True, 
        description="True if the candidate can be understood as an independent thought unit"
    )


class TranscriptLLMResponse(BaseModel):
    """Container schema for structured LLM response."""
    candidates: List[TranscriptSemanticCandidate] = Field(default_factory=list)


class TranscriptExpert(BaseExpert):
    """
    Modality Evidence Expert for Spoken Transcript Semantics.
    Produces independent temporal proposals grounded in WhisperX word-level timestamps.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[Any] = None,
        min_candidate_duration_sec: Optional[float] = None,
        max_candidate_duration_sec: Optional[float] = None,
        window_duration_sec: Optional[float] = None,
        window_overlap_sec: Optional[float] = None,
    ):
        cfg = config or app_config.transcript
        super().__init__(name="transcript", config=cfg)
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
        self.window_duration = (
            window_duration_sec
            if window_duration_sec is not None
            else cfg.window_duration_sec
        )
        self.window_overlap = (
            window_overlap_sec
            if window_overlap_sec is not None
            else cfg.window_overlap_sec
        )

    def evaluate(
        self,
        extraction_data: TranscriptData,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExpertEvidenceBundle:
        """
        Evaluate TranscriptData and generate localized semantic temporal proposals.

        Args:
            extraction_data: Validated TranscriptData with word-level timestamps.
            context: Optional dictionary containing global metadata (e.g. video_duration).

        Returns:
            ExpertEvidenceBundle containing validated, localized TemporalProposal instances.
        """
        words = extraction_data.words
        total_duration = (
            float(context.get("total_duration"))
            if context and context.get("total_duration")
            else extraction_data.duration
        )

        if not words:
            logger.warning("TranscriptData contains no words. Returning empty evidence bundle.")
            return ExpertEvidenceBundle(expert_name="transcript", proposals=[])

        # 1. Divide transcript into bounded analysis windows to ensure localized candidate extraction
        analysis_windows: List[tuple[float, float]] = []
        if total_duration <= self.window_duration:
            analysis_windows.append((0.0, total_duration))
        else:
            cur_start = 0.0
            while cur_start < total_duration:
                cur_end = min(total_duration, cur_start + self.window_duration)
                analysis_windows.append((cur_start, cur_end))
                if cur_end >= total_duration:
                    break
                cur_start += max(10.0, self.window_duration - self.window_overlap)

        proposals: List[TemporalProposal] = []
        seen_spans = set()
        prop_counter = 0

        for win_idx, (win_start, win_end) in enumerate(analysis_windows):
            # Collect word indices belonging to this bounded analysis window
            window_word_indices = [
                i for i, w in enumerate(words)
                if w.end >= win_start and w.start <= win_end
            ]

            if len(window_word_indices) < 5:
                continue

            # Format indexed words with their global word index
            indexed_lines = []
            for i in window_word_indices:
                w = words[i]
                speaker_tag = f"<{w.speaker}> " if getattr(w, "speaker", None) else ""
                indexed_lines.append(f"[{i}] {speaker_tag}{w.word}")
            indexed_transcript_text = " ".join(indexed_lines)

            prompt = f"""
You are the Transcript Evidence Expert in a multimodal podcast video clipping research system.
Your job is to identify specific, localized candidate highlight spans within this bounded analysis window ({win_start:.1f}s to {win_end:.1f}s).

CRITICAL LOCALIZATION RULES:
1. Do NOT partition the entire window or select the whole transcript.
2. Identify a specific, self-contained claim, explanation, or thematic argument.
3. Target candidate duration MUST be between {self.min_duration:.0f}s and {self.max_duration:.0f}s.
4. Select spans using word indices: start_word_index and end_word_index.

TOTAL DURATION: {total_duration:.2f}s | WINDOW: [{win_start:.1f}s -> {win_end:.1f}s] ({len(window_word_indices)} words)

INDEXED TRANSCRIPT WINDOW:
{indexed_transcript_text}
"""
            system_instruction = (
                "You are an objective linguistic and semantic analysis system. "
                "Output only factual, localized semantic candidates with word index boundaries. "
                "Never select the entire analysis window."
            )

            logger.info(
                f"Invoking LLM for Transcript Expert window {win_idx + 1}/{len(analysis_windows)} "
                f"[{win_start:.1f}s -> {win_end:.1f}s]..."
            )
            llm_response: TranscriptLLMResponse = self.llm_client.generate_structured(
                prompt=prompt,
                response_schema=TranscriptLLMResponse,
                system_instruction=system_instruction,
            )

            for cand in llm_response.candidates:
                start_idx = cand.start_word_index
                end_idx = cand.end_word_index

                # Index validity
                if not (0 <= start_idx <= end_idx < len(words)):
                    logger.warning(
                        f"Discarding invalid word index span [{start_idx}, {end_idx}] (total words: {len(words)})"
                    )
                    continue

                span_key = (start_idx, end_idx)
                if span_key in seen_spans:
                    continue
                seen_spans.add(span_key)

                # Programmatic mapping to WhisperX timestamps
                start_time = float(words[start_idx].start)
                end_time = float(words[end_idx].end)
                duration = max(0.0, end_time - start_time)

                # Bounds validation: 0 <= start <= end <= total_duration
                if start_time < 0.0 or end_time > total_duration or start_time >= end_time:
                    logger.warning(
                        f"Discarding proposal with out-of-bounds timestamps: [{start_time:.3f}s -> {end_time:.3f}s] "
                        f"vs video duration {total_duration:.2f}s"
                    )
                    continue

                # Candidate duration constraints: reject whole-input or oversized proposals
                if duration > self.max_duration:
                    logger.warning(
                        f"Discarding oversized candidate span [{start_time:.2f}s -> {end_time:.2f}s] "
                        f"(duration: {duration:.2f}s > max: {self.max_duration:.2f}s)"
                    )
                    continue

                if duration < self.min_duration:
                    logger.warning(
                        f"Discarding undersized candidate span [{start_time:.2f}s -> {end_time:.2f}s] "
                        f"(duration: {duration:.2f}s < min: {self.min_duration:.2f}s)"
                    )
                    continue

                if total_duration > self.max_duration and duration >= 0.90 * total_duration:
                    logger.warning(
                        f"Rejecting unjustified whole-input proposal spanning {duration:.2f}s of {total_duration:.2f}s"
                    )
                    continue

                num_words = end_idx - start_idx + 1
                word_res_sec = duration / max(1, num_words)

                proposal = TemporalProposal(
                    proposal_id=f"transcript_prop_{prop_counter}",
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    confidence_estimate=cand.confidence_estimate,
                    evidence_type=cand.evidence_type,
                    explanation=cand.explanation,
                    supporting_features={
                        "topic_transition": cand.topic_transition,
                        "contextual_completeness": cand.contextual_completeness,
                        "semantic_importance": cand.semantic_importance,
                        "self_contained": cand.self_contained,
                        "word_count": num_words,
                        "start_word_index": start_idx,
                        "end_word_index": end_idx,
                        "start_word": words[start_idx].word,
                        "end_word": words[end_idx].word,
                        "avg_word_duration_sec": word_res_sec,
                    },
                    source_metadata=ProposalSourceMetadata(
                        source_type=SourceResolutionType.WORD_TIMESTAMP,
                        temporal_resolution_sec=word_res_sec,
                        alignment_anchor=f"word_index_{start_idx}:{end_idx}",
                    ),
                )
                proposals.append(proposal)
                prop_counter += 1

        logger.info(f"Transcript Expert produced {len(proposals)} localized temporal proposals.")
        return ExpertEvidenceBundle(
            expert_name="transcript",
            proposals=proposals,
            dense_features_summary={
                "total_words_analyzed": len(words),
                "num_analysis_windows": len(analysis_windows),
                "model_used": f"{self.llm_client.provider}:{self.llm_client.model_name}",
            },
        )
