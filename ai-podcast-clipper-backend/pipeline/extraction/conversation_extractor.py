"""
ClipSense Conversation Extractor (W2)

Extracts conversational structure and dialogue timeline from timestamped transcripts:
- Distinguishes multi-speaker turn-taking (active speaker transitions) from single-speaker
  pause-separated speech segments (monologue phrasing delineated by silence gaps).
- Preserves speaker IDs, turn/segment durations, and inter-utterance pause boundaries (pause_before, pause_after)
  to ensure actual multi-speaker turn-taking can be detected whenever diarization labels are present.
- Calculates conversational pace metrics (utterance/turn frequency per minute, speech vs pause ratios).

Handles missing or single-speaker tags gracefully via configurable silence pause thresholds.
Produces validated ConversationData / conversation.json artifact.
"""

import json
import logging
import os
import pathlib
from typing import List, Optional

from app.config import config
from core.schemas import SpeakerTurn, ConversationData, TranscriptData

logger = logging.getLogger("clipsense.extraction.conversation")


class ConversationExtractor:
    """Extracts speaker turns, turn transitions, and conversational dynamics."""

    def __init__(
        self,
        turn_pause_threshold_sec: Optional[float] = None,
        min_turn_word_count: Optional[int] = None,
    ):
        self.turn_pause_threshold_sec = (
            turn_pause_threshold_sec or config.conversation.turn_pause_threshold_sec
        )
        self.min_turn_word_count = (
            min_turn_word_count or config.conversation.min_turn_word_count
        )

    def extract(
        self,
        transcript_data: TranscriptData,
        total_duration: Optional[float] = None,
        output_json_path: Optional[str] = None,
        force_rerun: bool = False,
    ) -> ConversationData:
        """
        Extract speaker turns and dialogue dynamics from TranscriptData.
        Uses caching if output_json_path exists and force_rerun is False.
        """
        # Cache check
        if output_json_path and os.path.exists(output_json_path) and not force_rerun:
            logger.info(f"Loading cached conversation data from {output_json_path}")
            with open(output_json_path, "r", encoding="utf-8") as f:
                data_dict = json.load(f)
            return ConversationData.model_validate(data_dict)

        duration = total_duration or transcript_data.duration
        segments = transcript_data.segments

        turns: List[SpeakerTurn] = []

        if not segments:
            # Empty transcript fallback
            conv_data = ConversationData(
                duration=duration,
                num_speakers=0,
                turns=[],
                turn_frequency_per_minute=0.0,
                total_speech_time=0.0,
                total_pause_time=duration,
            )
            if output_json_path:
                with open(output_json_path, "w", encoding="utf-8") as f:
                    f.write(conv_data.model_dump_json(indent=2))
            return conv_data

        # Determine if multiple distinct speakers are labeled
        labeled_speakers = {seg.speaker for seg in segments if seg.speaker}
        has_multi_speakers = len(labeled_speakers) > 1

        current_turn_speaker = segments[0].speaker or "SPEAKER_00"
        current_turn_start = segments[0].start
        current_turn_end = segments[0].end
        current_turn_words = len(segments[0].words) if segments[0].words else max(1, len(segments[0].text.split()))
        last_seg_end = segments[0].end

        turn_index = 0
        pause_before = current_turn_start

        for seg in segments[1:]:
            seg_speaker = seg.speaker or "SPEAKER_00"
            pause_duration = max(0.0, seg.start - last_seg_end)
            seg_word_count = len(seg.words) if seg.words else max(1, len(seg.text.split()))

            # Turn break criteria:
            # 1. If multi-speaker tags exist, switch on speaker label change
            # 2. Or if silence pause between utterances exceeds turn_pause_threshold_sec
            is_speaker_change = has_multi_speakers and (seg_speaker != current_turn_speaker)
            is_pause_boundary = pause_duration >= self.turn_pause_threshold_sec

            if is_speaker_change or is_pause_boundary:
                # Close current turn
                turn_dur = max(0.0, current_turn_end - current_turn_start)
                turns.append(
                    SpeakerTurn(
                        turn_index=turn_index,
                        speaker=current_turn_speaker,
                        start_time=current_turn_start,
                        end_time=current_turn_end,
                        duration=turn_dur,
                        word_count=current_turn_words,
                        pause_before=pause_before,
                        pause_after=pause_duration,
                    )
                )
                turn_index += 1

                # Start new turn
                current_turn_speaker = seg_speaker
                current_turn_start = seg.start
                current_turn_end = seg.end
                current_turn_words = seg_word_count
                pause_before = pause_duration
            else:
                # Merge into current turn
                current_turn_end = seg.end
                current_turn_words += seg_word_count

            last_seg_end = seg.end

        # Append final turn
        final_turn_dur = max(0.0, current_turn_end - current_turn_start)
        final_pause_after = max(0.0, duration - current_turn_end)
        turns.append(
            SpeakerTurn(
                turn_index=turn_index,
                speaker=current_turn_speaker,
                start_time=current_turn_start,
                end_time=current_turn_end,
                duration=final_turn_dur,
                word_count=current_turn_words,
                pause_before=pause_before,
                pause_after=final_pause_after,
            )
        )

        # Calculate macro conversation metrics
        total_speech_time = sum(t.duration for t in turns)
        total_pause_time = max(0.0, duration - total_speech_time)
        distinct_speakers = len({t.speaker for t in turns})
        duration_minutes = duration / 60.0 if duration > 0 else 1.0
        turn_frequency = len(turns) / duration_minutes if duration_minutes > 0 else 0.0

        conv_data = ConversationData(
            duration=duration,
            num_speakers=distinct_speakers,
            turns=turns,
            turn_frequency_per_minute=turn_frequency,
            total_speech_time=total_speech_time,
            total_pause_time=total_pause_time,
        )

        # Save artifact
        if output_json_path:
            out_p = pathlib.Path(output_json_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                f.write(conv_data.model_dump_json(indent=2))
            logger.info(f"Saved conversation data artifact to {output_json_path}")

        return conv_data
