"""
ClipSense Evidence-Aware Temporal Boundary Refinement (Week 10)

Implements modular, deterministic refinement of highlight candidate boundaries:
- Ingests MTER candidate intervals and available multimodal evidence (WhisperX word/sentence
  timestamps, speech-activity pauses, acoustic prosody energy, visual scene transitions).
- Evaluates candidate boundary adjustments within a bounded temporal search window.
- Eliminates mid-word/mid-utterance cuts by aligning start/end points with natural speech pauses
  and sentence boundaries.
- Uses acoustic prosody (low voicing fraction, RMS energy dips) and visual scene cuts as optional
  supporting evidence, never as ungrounded arbitrary cut points.
- Strictly enforces media duration boundaries and configured minimum/maximum clip duration bounds.
- Deterministic tie-breaking ensures identical output across repeated runs without LLM hallucinations.
- Preserves continuous floating-point precision (no coarse second-quantization).
- Produces comprehensive BoundaryRefinementMetadata audit trail for every candidate.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from app.config import BoundaryRefinerConfig
from core.schemas import (
    BoundaryRefinementMetadata,
    CommonEvidenceBundle,
    ConversationData,
    MTERCandidate,
    ProsodyData,
    TranscriptData,
    VisualData,
)

logger = logging.getLogger(__name__)


class BoundarySnapCandidate:
    """Internal representation of a candidate temporal snap point."""

    def __init__(
        self,
        timestamp: float,
        delta: float,
        evidence_type: str,
        score: float,
        signals_used: List[str],
        rationale: str,
    ):
        self.timestamp = timestamp
        self.delta = delta
        self.evidence_type = evidence_type
        self.score = score
        self.signals_used = signals_used
        self.rationale = rationale

    def __repr__(self) -> str:
        return (
            f"BoundarySnapCandidate(t={self.timestamp:.3f}s, delta={self.delta:+.3f}s, "
            f"type={self.evidence_type}, score={self.score:.2f})"
        )


class BoundaryRefiner:
    """
    Evidence-Aware Temporal Boundary Refinement Engine.
    
    Refines MTER candidate intervals using fine-grained temporal alignment evidence.
    """

    def __init__(self, config: Optional[BoundaryRefinerConfig] = None):
        self.config = config or BoundaryRefinerConfig()

    def is_enabled(self) -> bool:
        """Return True if boundary refinement is enabled."""
        return bool(self.config.enabled)

    def refine_candidates(
        self,
        candidates: List[MTERCandidate],
        transcript_data: Optional[TranscriptData] = None,
        conversation_data: Optional[ConversationData] = None,
        visual_data: Optional[VisualData] = None,
        prosody_data: Optional[ProsodyData] = None,
        video_duration: Optional[float] = None,
        bundle: Optional[CommonEvidenceBundle] = None,
    ) -> List[MTERCandidate]:
        """
        Refines a list of MTER highlight candidates sequentially.
        
        If disabled, attaches an unrefined BoundaryRefinementMetadata record and
        preserves the original boundaries identically.
        """
        if not candidates:
            return []

        refined_list: List[MTERCandidate] = []
        for cand in candidates:
            refined_cand = self.refine_candidate(
                candidate=cand,
                transcript_data=transcript_data,
                conversation_data=conversation_data,
                visual_data=visual_data,
                prosody_data=prosody_data,
                video_duration=video_duration,
                bundle=bundle,
            )
            refined_list.append(refined_cand)

        return refined_list

    def refine_candidate(
        self,
        candidate: MTERCandidate,
        transcript_data: Optional[TranscriptData] = None,
        conversation_data: Optional[ConversationData] = None,
        visual_data: Optional[VisualData] = None,
        prosody_data: Optional[ProsodyData] = None,
        video_duration: Optional[float] = None,
        bundle: Optional[CommonEvidenceBundle] = None,
    ) -> MTERCandidate:
        """
        Refines a single highlight candidate's start and end timestamps.
        """
        orig_start = float(candidate.proposed_start)
        orig_end = float(candidate.proposed_end)
        orig_duration = orig_end - orig_start

        # Determine effective video duration
        eff_video_dur = video_duration
        if eff_video_dur is None and bundle is not None and bundle.total_duration > 0:
            eff_video_dur = bundle.total_duration

        if eff_video_dur is None:
            if transcript_data and transcript_data.duration > 0:
                eff_video_dur = transcript_data.duration
            elif visual_data and visual_data.duration > 0:
                eff_video_dur = visual_data.duration
            elif prosody_data and prosody_data.duration > 0:
                eff_video_dur = prosody_data.duration
            else:
                eff_video_dur = orig_end + 10.0

        # 1. Check if refinement is disabled
        if not self.is_enabled():
            meta = BoundaryRefinementMetadata(
                refined=False,
                original_start=orig_start,
                original_end=orig_end,
                original_duration=orig_duration,
                refined_start=orig_start,
                refined_end=orig_end,
                refined_duration=orig_duration,
                start_delta_sec=0.0,
                end_delta_sec=0.0,
                start_evidence_type="original",
                end_evidence_type="original",
                signals_used=[],
                rationale="Refinement disabled via configuration.",
                fallback_reason=None,
            )
            updated = candidate.model_copy(deep=True)
            updated.boundary_refinement = meta
            return updated

        # 2. Validate input candidate bounds
        if not math.isfinite(orig_start) or not math.isfinite(orig_end):
            meta = BoundaryRefinementMetadata(
                refined=False,
                original_start=orig_start,
                original_end=orig_end,
                original_duration=orig_duration,
                refined_start=orig_start,
                refined_end=orig_end,
                refined_duration=orig_duration,
                start_delta_sec=0.0,
                end_delta_sec=0.0,
                start_evidence_type="original",
                end_evidence_type="original",
                signals_used=[],
                rationale="Candidate rejected: non-finite timestamps.",
                fallback_reason="Non-finite start or end timestamp.",
            )
            updated = candidate.model_copy(deep=True)
            updated.boundary_refinement = meta
            return updated

        if orig_start < 0.0 or orig_end <= orig_start:
            meta = BoundaryRefinementMetadata(
                refined=False,
                original_start=orig_start,
                original_end=orig_end,
                original_duration=orig_duration,
                refined_start=orig_start,
                refined_end=orig_end,
                refined_duration=orig_duration,
                start_delta_sec=0.0,
                end_delta_sec=0.0,
                start_evidence_type="original",
                end_evidence_type="original",
                signals_used=[],
                rationale=f"Candidate rejected: reversed or negative bounds [{orig_start:.2f}s -> {orig_end:.2f}s].",
                fallback_reason="Invalid start/end ordering or negative start timestamp.",
            )
            updated = candidate.model_copy(deep=True)
            updated.boundary_refinement = meta
            return updated

        # 3. Check for available temporal evidence
        has_transcript = transcript_data is not None and bool(transcript_data.words or transcript_data.segments)
        has_prosody = prosody_data is not None and bool(prosody_data.windows)
        has_visual = visual_data is not None and bool(visual_data.scene_boundaries)

        if not (has_transcript or has_prosody or has_visual):
            meta = BoundaryRefinementMetadata(
                refined=False,
                original_start=orig_start,
                original_end=orig_end,
                original_duration=orig_duration,
                refined_start=orig_start,
                refined_end=orig_end,
                refined_duration=orig_duration,
                start_delta_sec=0.0,
                end_delta_sec=0.0,
                start_evidence_type="original",
                end_evidence_type="original",
                signals_used=[],
                rationale="No temporal evidence available for refinement. Preserving original boundaries.",
                fallback_reason="No temporal evidence available across modalities.",
            )
            updated = candidate.model_copy(deep=True)
            updated.boundary_refinement = meta
            return updated

        # 4. Find candidate snap points for start and end
        start_candidates = self._find_start_candidates(
            orig_start=orig_start,
            transcript_data=transcript_data,
            prosody_data=prosody_data,
            visual_data=visual_data,
            video_duration=eff_video_dur,
        )

        end_candidates = self._find_end_candidates(
            orig_end=orig_end,
            transcript_data=transcript_data,
            prosody_data=prosody_data,
            visual_data=visual_data,
            video_duration=eff_video_dur,
        )

        # 5. Deterministic selection with tie-breaking
        best_start = self._select_best_snap(start_candidates, orig_start)
        best_end = self._select_best_snap(end_candidates, orig_end)

        refined_start = best_start.timestamp
        refined_end = best_end.timestamp

        # 6. Apply Duration and Ordering Constraints
        signals_used = sorted(list(set(best_start.signals_used + best_end.signals_used)))
        fallback_reason: Optional[str] = None

        # Clamp within media bounds
        refined_start = max(0.0, min(eff_video_dur, refined_start))
        refined_end = max(0.0, min(eff_video_dur, refined_end))

        # Check start/end ordering
        if refined_end <= refined_start:
            fallback_reason = (
                f"Refinement would invert boundaries ([{refined_start:.2f}s -> {refined_end:.2f}s]). "
                "Retaining original boundaries."
            )
            refined_start = orig_start
            refined_end = orig_end
            start_ev_type = "original"
            end_ev_type = "original"
        else:
            proposed_dur = refined_end - refined_start
            min_dur = self.config.min_clip_duration_sec
            max_dur = self.config.max_clip_duration_sec

            # Duration too short
            if proposed_dur < min_dur:
                if orig_duration >= min_dur:
                    # Snapping shrank interval below min duration -> fallback to original
                    fallback_reason = (
                        f"Refined duration ({proposed_dur:.2f}s) fell below minimum limit ({min_dur:.2f}s). "
                        "Retaining original boundaries to satisfy duration constraint."
                    )
                    refined_start = orig_start
                    refined_end = orig_end
                    start_ev_type = "original"
                    end_ev_type = "original"
                else:
                    # Original was already shorter than min_dur; adjust end if possible
                    needed = min_dur - proposed_dur
                    if refined_end + needed <= eff_video_dur:
                        refined_end += needed
                        signals_used.append("duration_constraint_expansion")
                    elif refined_start - needed >= 0.0:
                        refined_start -= needed
                        signals_used.append("duration_constraint_expansion")
                    start_ev_type = best_start.evidence_type
                    end_ev_type = best_end.evidence_type

            # Duration too long
            elif proposed_dur > max_dur:
                if orig_duration <= max_dur:
                    # Snapping expanded interval beyond max duration
                    # Adjust the boundary with lower evidence score
                    excess = proposed_dur - max_dur
                    if best_start.score < best_end.score:
                        refined_start += excess
                        signals_used.append("duration_constraint_clamping")
                    else:
                        refined_end -= excess
                        signals_used.append("duration_constraint_clamping")
                    start_ev_type = best_start.evidence_type
                    end_ev_type = best_end.evidence_type
                else:
                    # Clamp symmetrically
                    excess = proposed_dur - max_dur
                    half_excess = excess / 2.0
                    refined_start = max(0.0, refined_start + half_excess)
                    refined_end = min(eff_video_dur, refined_end - half_excess)
                    signals_used.append("duration_constraint_clamping")
                    start_ev_type = best_start.evidence_type
                    end_ev_type = best_end.evidence_type
            else:
                start_ev_type = best_start.evidence_type
                end_ev_type = best_end.evidence_type

        # Ensure precision: round to 3 decimal places (millisecond precision)
        refined_start = round(refined_start, 3)
        refined_end = round(refined_end, 3)
        refined_duration = round(refined_end - refined_start, 3)

        start_delta = round(refined_start - orig_start, 3)
        end_delta = round(refined_end - orig_end, 3)
        is_changed = (abs(start_delta) > 0.005) or (abs(end_delta) > 0.005)

        # Formulate human-interpretable rationale
        if fallback_reason:
            rationale = fallback_reason
        elif is_changed:
            parts = []
            if abs(start_delta) > 0.005:
                parts.append(
                    f"Start shifted by {start_delta:+.2f}s ({best_start.rationale})"
                )
            else:
                parts.append("Start maintained near original boundary")
            if abs(end_delta) > 0.005:
                parts.append(
                    f"End shifted by {end_delta:+.2f}s ({best_end.rationale})"
                )
            else:
                parts.append("End maintained near original boundary")
            rationale = ". ".join(parts) + "."
        else:
            rationale = "Original boundaries are already closely aligned with evidence."

        meta = BoundaryRefinementMetadata(
            refined=is_changed,
            original_start=orig_start,
            original_end=orig_end,
            original_duration=round(orig_duration, 3),
            refined_start=refined_start,
            refined_end=refined_end,
            refined_duration=refined_duration,
            start_delta_sec=start_delta,
            end_delta_sec=end_delta,
            start_evidence_type=start_ev_type,
            end_evidence_type=end_ev_type,
            signals_used=signals_used,
            rationale=rationale,
            fallback_reason=fallback_reason,
        )

        # Return updated candidate with new authoritative timestamps
        updated = candidate.model_copy(deep=True)
        updated.proposed_start = refined_start
        updated.proposed_end = refined_end
        updated.duration = refined_duration
        updated.boundary_refinement = meta

        # Record decision update in candidate decision summary if refined
        if is_changed:
            updated.decision_summary = (
                f"{candidate.decision_summary} [Refined boundaries: {orig_start:.2f}s->{refined_start:.2f}s, "
                f"{orig_end:.2f}s->{refined_end:.2f}s ({rationale})]"
            )

        return updated

    # -------------------------------------------------------------------------
    # Internal Boundary Search Algorithms
    # -------------------------------------------------------------------------

    def _find_start_candidates(
        self,
        orig_start: float,
        transcript_data: Optional[TranscriptData],
        prosody_data: Optional[ProsodyData],
        visual_data: Optional[VisualData],
        video_duration: float,
    ) -> List[BoundarySnapCandidate]:
        """
        Locates candidate start boundary snap points within search window.
        """
        w_sec = self.config.search_window_start_sec
        buffer_sec = self.config.dead_air_buffer_sec
        max_shift = self.config.max_shift_sec
        candidates: List[BoundarySnapCandidate] = []

        # Baseline candidate: Original timestamp
        candidates.append(
            BoundarySnapCandidate(
                timestamp=orig_start,
                delta=0.0,
                evidence_type="original",
                score=0.50,
                signals_used=["original_boundary"],
                rationale="Retaining original MTER consensus start timestamp",
            )
        )

        search_min = max(0.0, orig_start - w_sec)
        search_max = min(video_duration, orig_start + w_sec)

        # 1. Transcript: Sentence / Utterance Boundaries
        if transcript_data and transcript_data.segments:
            for seg in transcript_data.segments:
                if search_min <= seg.start <= search_max:
                    # Snap to segment onset with lead-in buffer
                    t_snap = max(0.0, seg.start - buffer_sec)
                    delta = t_snap - orig_start
                    if abs(delta) <= max_shift:
                        # Distance penalty: closer to orig_start gets higher score
                        dist_penalty = (abs(delta) / max_shift) * 0.4
                        base_score = 3.2 if self.config.prioritize_sentence_boundaries else 2.4
                        score = base_score - dist_penalty
                        signals = ["transcript_sentence_boundary"]

                        # Check for acoustic silence confirmation
                        if self._is_acoustic_silence(t_snap, prosody_data):
                            score += 0.3
                            signals.append("prosody_silence_confirmation")

                        # Check for visual scene cut alignment
                        scene_cut = self._find_nearby_scene_cut(t_snap, visual_data)
                        if scene_cut is not None and scene_cut <= seg.start:
                            t_snap = scene_cut
                            delta = t_snap - orig_start
                            score += 0.2
                            signals.append("visual_scene_cut_aligned")

                        candidates.append(
                            BoundarySnapCandidate(
                                timestamp=t_snap,
                                delta=delta,
                                evidence_type="sentence_boundary",
                                score=score,
                                signals_used=signals,
                                rationale=f"Aligned with sentence onset at {seg.start:.2f}s",
                            )
                        )

        # 2. Transcript: Word Boundaries and Mid-Word Cut Prevention
        if transcript_data and transcript_data.words:
            # Check if orig_start falls inside a word (mid-word cut!)
            mid_word = None
            for w in transcript_data.words:
                if w.start <= orig_start <= w.end:
                    mid_word = w
                    break

            if mid_word:
                # Urgent priority: move cut before the word starts!
                t_snap = max(0.0, mid_word.start - buffer_sec)
                delta = t_snap - orig_start
                if abs(delta) <= max_shift:
                    candidates.append(
                        BoundarySnapCandidate(
                            timestamp=t_snap,
                            delta=delta,
                            evidence_type="word_onset_prevention_of_mid_word_cut",
                            score=2.8,  # High score to resolve mid-word clipping
                            signals_used=["whisperx_word_timestamp", "mid_word_cut_prevention"],
                            rationale=f"Prevented mid-word cut on '{mid_word.word}' at {mid_word.start:.2f}s",
                        )
                    )

            # Check for words starting in search window
            for w in transcript_data.words:
                if search_min <= w.start <= search_max:
                    t_snap = max(0.0, w.start - buffer_sec)
                    delta = t_snap - orig_start
                    if abs(delta) <= max_shift:
                        dist_penalty = (abs(delta) / max_shift) * 0.5
                        score = 1.6 - dist_penalty
                        signals = ["whisperx_word_timestamp"]

                        if self._is_acoustic_silence(t_snap, prosody_data):
                            score += 0.25
                            signals.append("prosody_silence_confirmation")

                        candidates.append(
                            BoundarySnapCandidate(
                                timestamp=t_snap,
                                delta=delta,
                                evidence_type="word_boundary",
                                score=score,
                                signals_used=signals,
                                rationale=f"Aligned with word '{w.word}' onset at {w.start:.2f}s",
                            )
                        )

            # Check for inter-word pauses
            words = transcript_data.words
            for i in range(len(words) - 1):
                w_curr = words[i]
                w_next = words[i + 1]
                pause_len = w_next.start - w_curr.end
                if pause_len >= self.config.speech_pause_threshold_sec:
                    # Pause interval is [w_curr.end, w_next.start]
                    mid_pause = w_curr.end + (pause_len / 2.0)
                    if search_min <= mid_pause <= search_max:
                        t_snap = max(w_curr.end + 0.02, min(w_next.start - buffer_sec, mid_pause))
                        delta = t_snap - orig_start
                        if abs(delta) <= max_shift:
                            dist_penalty = (abs(delta) / max_shift) * 0.4
                            score = 1.9 - dist_penalty
                            signals = ["inter_word_speech_pause"]

                            if self._is_acoustic_silence(t_snap, prosody_data):
                                score += 0.3
                                signals.append("prosody_silence_confirmation")

                            candidates.append(
                                BoundarySnapCandidate(
                                    timestamp=t_snap,
                                    delta=delta,
                                    evidence_type="speech_pause",
                                    score=score,
                                    signals_used=signals,
                                    rationale=f"Placed in {pause_len:.2f}s speech pause before '{w_next.word}'",
                                )
                            )

        return candidates

    def _find_end_candidates(
        self,
        orig_end: float,
        transcript_data: Optional[TranscriptData],
        prosody_data: Optional[ProsodyData],
        visual_data: Optional[VisualData],
        video_duration: float,
    ) -> List[BoundarySnapCandidate]:
        """
        Locates candidate end boundary snap points within search window.
        """
        w_sec = self.config.search_window_end_sec
        buffer_sec = self.config.dead_air_buffer_sec
        max_shift = self.config.max_shift_sec
        candidates: List[BoundarySnapCandidate] = []

        # Baseline candidate: Original timestamp
        candidates.append(
            BoundarySnapCandidate(
                timestamp=orig_end,
                delta=0.0,
                evidence_type="original",
                score=0.50,
                signals_used=["original_boundary"],
                rationale="Retaining original MTER consensus end timestamp",
            )
        )

        search_min = max(0.0, orig_end - w_sec)
        search_max = min(video_duration, orig_end + w_sec)

        # 1. Transcript: Sentence / Utterance Boundaries
        if transcript_data and transcript_data.segments:
            for seg in transcript_data.segments:
                if search_min <= seg.end <= search_max:
                    t_snap = min(video_duration, seg.end + buffer_sec)
                    delta = t_snap - orig_end
                    if abs(delta) <= max_shift:
                        dist_penalty = (abs(delta) / max_shift) * 0.4
                        base_score = 3.2 if self.config.prioritize_sentence_boundaries else 2.4
                        score = base_score - dist_penalty
                        signals = ["transcript_sentence_boundary"]

                        if self._is_acoustic_silence(t_snap, prosody_data):
                            score += 0.3
                            signals.append("prosody_silence_confirmation")

                        scene_cut = self._find_nearby_scene_cut(t_snap, visual_data)
                        if scene_cut is not None and scene_cut >= seg.end:
                            t_snap = scene_cut
                            delta = t_snap - orig_end
                            score += 0.2
                            signals.append("visual_scene_cut_aligned")

                        candidates.append(
                            BoundarySnapCandidate(
                                timestamp=t_snap,
                                delta=delta,
                                evidence_type="sentence_boundary",
                                score=score,
                                signals_used=signals,
                                rationale=f"Aligned with sentence completion at {seg.end:.2f}s",
                            )
                        )

        # 2. Transcript: Word Boundaries and Mid-Word Cut Prevention
        if transcript_data and transcript_data.words:
            # Check if orig_end falls inside a word (mid-word cut!)
            mid_word = None
            for w in transcript_data.words:
                if w.start <= orig_end <= w.end:
                    mid_word = w
                    break

            if mid_word:
                # Urgent priority: extend cut until word completes
                t_snap = min(video_duration, mid_word.end + buffer_sec)
                delta = t_snap - orig_end
                if abs(delta) <= max_shift:
                    candidates.append(
                        BoundarySnapCandidate(
                            timestamp=t_snap,
                            delta=delta,
                            evidence_type="word_offset_prevention_of_mid_word_cut",
                            score=2.8,
                            signals_used=["whisperx_word_timestamp", "mid_word_cut_prevention"],
                            rationale=f"Prevented mid-word cutoff on '{mid_word.word}' ending at {mid_word.end:.2f}s",
                        )
                    )

            # Check for words ending in search window
            for w in transcript_data.words:
                if search_min <= w.end <= search_max:
                    t_snap = min(video_duration, w.end + buffer_sec)
                    delta = t_snap - orig_end
                    if abs(delta) <= max_shift:
                        dist_penalty = (abs(delta) / max_shift) * 0.5
                        score = 1.6 - dist_penalty
                        signals = ["whisperx_word_timestamp"]

                        if self._is_acoustic_silence(t_snap, prosody_data):
                            score += 0.25
                            signals.append("prosody_silence_confirmation")

                        candidates.append(
                            BoundarySnapCandidate(
                                timestamp=t_snap,
                                delta=delta,
                                evidence_type="word_boundary",
                                score=score,
                                signals_used=signals,
                                rationale=f"Aligned with word '{w.word}' offset at {w.end:.2f}s",
                            )
                        )

            # Check for inter-word pauses
            words = transcript_data.words
            for i in range(len(words) - 1):
                w_curr = words[i]
                w_next = words[i + 1]
                pause_len = w_next.start - w_curr.end
                if pause_len >= self.config.speech_pause_threshold_sec:
                    mid_pause = w_curr.end + (pause_len / 2.0)
                    if search_min <= mid_pause <= search_max:
                        t_snap = min(video_duration, max(w_curr.end + buffer_sec, mid_pause))
                        delta = t_snap - orig_end
                        if abs(delta) <= max_shift:
                            dist_penalty = (abs(delta) / max_shift) * 0.4
                            score = 1.9 - dist_penalty
                            signals = ["inter_word_speech_pause"]

                            if self._is_acoustic_silence(t_snap, prosody_data):
                                score += 0.3
                                signals.append("prosody_silence_confirmation")

                            candidates.append(
                                BoundarySnapCandidate(
                                    timestamp=t_snap,
                                    delta=delta,
                                    evidence_type="speech_pause",
                                    score=score,
                                    signals_used=signals,
                                    rationale=f"Placed in {pause_len:.2f}s speech pause after '{w_curr.word}'",
                                )
                            )

        return candidates

    # -------------------------------------------------------------------------
    # Auxiliary Evidence Checkers
    # -------------------------------------------------------------------------

    def _is_acoustic_silence(self, timestamp: float, prosody_data: Optional[ProsodyData]) -> bool:
        """
        Checks if the acoustic prosody windows near timestamp exhibit low voicing
        fraction or low RMS energy, indicating acoustic pause or silence.
        """
        if not prosody_data or not prosody_data.windows:
            return False

        # Find window containing or nearest to timestamp
        for win in prosody_data.windows:
            if win.start_time <= timestamp <= win.end_time:
                # Low voicing fraction (< 0.25) or low energy (< 0.03) confirms pause/silence
                if win.voicing_fraction < 0.25 or win.rms_energy < 0.03:
                    return True
                break
        return False

    def _find_nearby_scene_cut(
        self, timestamp: float, visual_data: Optional[VisualData]
    ) -> Optional[float]:
        """
        Finds a visual scene cut timestamp within scene_cut_tolerance_sec if one exists.
        """
        if not visual_data or not visual_data.scene_boundaries:
            return None

        tol = self.config.scene_cut_tolerance_sec
        closest_cut: Optional[float] = None
        min_diff = float("inf")

        for sb in visual_data.scene_boundaries:
            # Check scene start_time as a cut point
            diff = abs(sb.start_time - timestamp)
            if diff <= tol and diff < min_diff:
                min_diff = diff
                closest_cut = sb.start_time

        return closest_cut

    def _select_best_snap(
        self, candidates: List[BoundarySnapCandidate], original_timestamp: float
    ) -> BoundarySnapCandidate:
        """
        Deterministically selects the optimal snap point using evidence score,
        distance penalty, and strict timestamp tie-breaking.
        """
        if not candidates:
            return BoundarySnapCandidate(
                timestamp=original_timestamp,
                delta=0.0,
                evidence_type="original",
                score=0.50,
                signals_used=["original_boundary"],
                rationale="Fallback to original timestamp",
            )

        # Sort criteria:
        # 1. Highest score (descending)
        # 2. Smallest absolute delta |t - orig| (ascending)
        # 3. Timestamp value (ascending) for deterministic stability
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (-c.score, abs(c.delta), c.timestamp),
        )
        return sorted_candidates[0]
