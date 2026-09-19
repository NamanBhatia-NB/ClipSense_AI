"""
Legacy Transcript-Only LLM Baseline (W6)

The baseline represents the legacy transcript/LLM highlight-selection approach
used in the previous ClipSense prototype.

Invariant Constraints:
- Consumes ONLY spoken transcript words (TranscriptData).
- Strictly does NOT use Visual Expert, Prosody Expert, Conversation Expert, or MTER.
- Implements one fixed, deterministic transcript chunking policy.
- Supports both 'live_inference' (calling LLM) and 'cached_inference' (reproducible offline evaluation).
- Snaps proposed times deterministically to word boundaries.
"""

import json
import logging
import os
import pathlib
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from core.llm import LLMClient
from core.schemas import TranscriptData
from evaluation.schemas import EvaluationSpan

logger = logging.getLogger("clipsense.evaluation.baseline")


class BaselineConfig(BaseModel):
    """
    Fixed deterministic configuration for the legacy transcript baseline.
    """
    # Deterministic sliding window parameters
    window_duration_sec: float = 60.0
    overlap_sec: float = 15.0
    
    # Candidate duration bounds
    min_candidate_duration_sec: float = 10.0
    max_candidate_duration_sec: float = 60.0
    
    # Execution mode: 'live_inference' (API call) or 'cached_inference' (load from disk)
    inference_mode: Literal["live_inference", "cached_inference"] = "live_inference"
    cache_dir: Optional[str] = None


class BaselineLLMCandidate(BaseModel):
    start_word_index: int = Field(description="Word index marking the start of the highlight")
    end_word_index: int = Field(description="Word index marking the end of the highlight")
    confidence: float = Field(default=0.8, description="Estimated highlight quality on [0.0, 1.0]")
    title: Optional[str] = Field(default=None, description="Brief title of the moment")
    summary: Optional[str] = Field(default=None, description="Reason this moment is a highlight")


class BaselineLLMResponse(BaseModel):
    candidates: List[BaselineLLMCandidate] = Field(default_factory=list)


class LegacyTranscriptBaselineSelector:
    """
    Standalone transcript-only baseline selector.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[BaselineConfig] = None,
    ):
        self.config = config or BaselineConfig()
        self.llm_client = llm_client

    def generate_windows(
        self, transcript: TranscriptData, video_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Deterministic sliding temporal window generation.
        
        Policy:
        - If video_duration <= window_duration_sec: exactly one window [0.0, video_duration]
        - Otherwise: windows at stride = window_duration_sec - overlap_sec
        - Strictly chronological ordering by window index k
        - Word w belongs to window k if w.start >= window_start and w.start < window_end
        """
        cfg = self.config
        win_dur = cfg.window_duration_sec
        overlap = cfg.overlap_sec
        stride = win_dur - overlap

        if video_duration <= win_dur:
            return [{
                "window_idx": 0,
                "start_sec": 0.0,
                "end_sec": video_duration,
                "words": transcript.words,
            }]

        windows = []
        cur_start = 0.0
        win_idx = 0

        while cur_start < video_duration:
            cur_end = min(cur_start + win_dur, video_duration)
            w_words = [
                w for w in transcript.words
                if w.start >= cur_start and w.start < cur_end
            ]
            windows.append({
                "window_idx": win_idx,
                "start_sec": cur_start,
                "end_sec": cur_end,
                "words": w_words,
            })
            win_idx += 1
            cur_start += stride
            if cur_end >= video_duration:
                break

        return windows

    def run(
        self,
        transcript: TranscriptData,
        video_duration: float,
        sample_id: str,
        cache_dir: Optional[str] = None,
    ) -> List[EvaluationSpan]:
        """
        Executes baseline candidate generation.
        Checks cache if configured; otherwise runs live inference and caches output.
        """
        effective_cache_dir = cache_dir or self.config.cache_dir
        cache_file = None
        if effective_cache_dir:
            p = pathlib.Path(effective_cache_dir)
            p.mkdir(parents=True, exist_ok=True)
            cache_file = p / f"baseline_{sample_id}.json"

        # Check cached inference
        if self.config.inference_mode == "cached_inference":
            if cache_file and cache_file.exists():
                logger.info(f"Loading cached baseline predictions from {cache_file}")
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return [EvaluationSpan(**item) for item in data]
            else:
                raise FileNotFoundError(
                    f"Cached baseline requested but cache file not found: {cache_file}"
                )

        # Live inference
        if self.llm_client is None:
            raise ValueError(
                "LLMClient must be provided to LegacyTranscriptBaselineSelector for live_inference."
            )

        windows = self.generate_windows(transcript, video_duration)
        raw_candidates: List[EvaluationSpan] = []
        seen_word_spans = set()

        for win in windows:
            win_words = win["words"]
            if len(win_words) < 5:
                continue

            # Format words with index
            word_lines = [
                f"[{i}] ({w.start:.2f}s - {w.end:.2f}s): {w.word}"
                for i, w in enumerate(win_words)
            ]
            transcript_text = "\n".join(word_lines)

            prompt = (
                f"You are evaluating a transcript window [{win['start_sec']:.1f}s -> {win['end_sec']:.1f}s] "
                f"from a video to identify standalone, engaging highlight moments.\n\n"
                f"TRANSCRIPT WORDS:\n{transcript_text}\n\n"
                f"Identify 1-3 coherent highlight moments. For each moment, return the start_word_index "
                f"and end_word_index corresponding to the word indices shown in brackets [i]. "
                f"The duration should be between {self.config.min_candidate_duration_sec}s and "
                f"{self.config.max_candidate_duration_sec}s."
            )

            try:
                response: BaselineLLMResponse = self.llm_client.generate_structured(
                    prompt=prompt,
                    response_schema=BaselineLLMResponse,
                    system_instruction=(
                        "You are a video highlight selection assistant. Identify coherent discussion moments "
                        "using only the provided transcript words."
                    ),
                )
            except Exception as e:
                logger.warning(f"Baseline LLM inference failed on window {win['window_idx']}: {e}")
                continue

            for c in response.candidates:
                if 0 <= c.start_word_index < c.end_word_index < len(win_words):
                    span_key = (c.start_word_index, c.end_word_index, win["window_idx"])
                    if span_key in seen_word_spans:
                        continue
                    seen_word_spans.add(span_key)

                    # Word boundary snapping
                    start_time = float(win_words[c.start_word_index].start)
                    end_time = float(win_words[c.end_word_index].end)
                    duration = end_time - start_time

                    if (
                        self.config.min_candidate_duration_sec
                        <= duration
                        <= self.config.max_candidate_duration_sec
                    ):
                        span = EvaluationSpan(
                            span_id=f"baseline_span_{len(raw_candidates)}",
                            start_time=start_time,
                            end_time=end_time,
                            duration=duration,
                            confidence=float(c.confidence),
                            mode_name="baseline",
                            metadata={
                                "title": c.title or "",
                                "summary": c.summary or "",
                                "window_idx": win["window_idx"],
                                "word_range": [c.start_word_index, c.end_word_index],
                            },
                        )
                        raw_candidates.append(span)

        # If cache directory is set, save live predictions for future cached_inference runs
        if cache_file:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump([s.model_dump() for s in raw_candidates], f, indent=2)
            logger.info(f"Saved {len(raw_candidates)} baseline predictions to cache: {cache_file}")

        return raw_candidates
