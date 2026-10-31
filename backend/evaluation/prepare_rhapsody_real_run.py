"""
Prepares real test-split episodes from Hugging Face yhpark/rhapsody for ClipSense evaluation.
Preserves ground-truth isolation:
- Replay scores are written ONLY to the evaluation manifest (accessible solely to evaluation adapters).
- Inference run directories contain ONLY genuine extracted transcript and conversational features.
"""

import json
import logging
import os
import pathlib
import sys
import numpy as np
import pyarrow as pa
from datasets.features.features import ArrayExtensionArray

# Ensure backend root is in sys.path
backend_root = str(pathlib.Path(__file__).parent.parent)
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

# Monkey-patch pyarrow 22 compatibility for datasets ArrayExtensionArray
orig_to_pylist = ArrayExtensionArray.to_pylist
ArrayExtensionArray.to_pylist = lambda self, *args, **kwargs: orig_to_pylist(self)

from datasets import load_dataset

from core.schemas import (
    ConversationData,
    ExpertEvidenceBundle,
    ProposalSourceMetadata,
    SourceResolutionType,
    SpeakerTurn,
    TemporalProposal,
    TimestampedWord,
    TranscriptData,
    TranscriptSegment,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_rhapsody_real")


def parse_transcript_data(raw_transcript_json: str, duration: float) -> TranscriptData:
    """Parses WhisperX JSON into ClipSense TranscriptData schema."""
    if isinstance(raw_transcript_json, str):
        data = json.loads(raw_transcript_json)
    else:
        data = raw_transcript_json

    segments_raw = data.get("segments", [])
    segments: list[TranscriptSegment] = []
    all_words: list[TimestampedWord] = []
    full_text_parts = []

    for idx, s in enumerate(segments_raw):
        s_start = float(s.get("start", 0.0))
        s_end = float(s.get("end", s_start))
        s_text = str(s.get("text", "")).strip()
        s_speaker = s.get("speaker", "SPEAKER_00") or "SPEAKER_00"

        seg_words: list[TimestampedWord] = []
        for w in s.get("words", []):
            w_text = str(w.get("word", "")).strip()
            if not w_text:
                continue
            w_start = float(w.get("start", s_start))
            w_end = float(w.get("end", w_start))
            w_score = float(w.get("score", 1.0)) if w.get("score") is not None else None
            t_word = TimestampedWord(word=w_text, start=w_start, end=w_end, score=w_score)
            seg_words.append(t_word)
            all_words.append(t_word)

        full_text_parts.append(s_text)
        segments.append(
            TranscriptSegment(
                id=idx,
                start=s_start,
                end=s_end,
                text=s_text,
                speaker=s_speaker,
                words=seg_words,
            )
        )

    full_text = " ".join(full_text_parts)
    return TranscriptData(
        full_text=full_text,
        duration=duration,
        segments=segments,
        words=all_words,
    )


def derive_conversational_evidence(
    transcript_data: TranscriptData,
    min_cand_duration: float = 15.0,
    max_cand_duration: float = 60.0,
) -> tuple[ConversationData, ExpertEvidenceBundle]:
    """
    Constructs genuine ConversationData and conversational discourse proposals
    derived from speaker transitions and pause boundaries.
    """
    segments = transcript_data.segments
    if not segments:
        empty_conv = ConversationData(
            duration=transcript_data.duration,
            num_speakers=1,
            turns=[],
            turn_frequency_per_minute=0.0,
            total_speech_time=0.0,
            total_pause_time=transcript_data.duration,
        )
        empty_bundle = ExpertEvidenceBundle(expert_name="conversation", proposals=[])
        return empty_conv, empty_bundle

    turns: list[SpeakerTurn] = []
    current_speaker = segments[0].speaker
    turn_start = segments[0].start
    turn_end = segments[0].end
    turn_words = len(segments[0].words)
    last_end = 0.0

    for i in range(1, len(segments)):
        seg = segments[i]
        if seg.speaker == current_speaker:
            turn_end = seg.end
            turn_words += len(seg.words)
        else:
            pause_before = max(0.0, turn_start - last_end)
            pause_after = max(0.0, seg.start - turn_end)
            turns.append(
                SpeakerTurn(
                    turn_index=len(turns),
                    speaker=current_speaker,
                    start_time=turn_start,
                    end_time=turn_end,
                    duration=turn_end - turn_start,
                    word_count=turn_words,
                    pause_before=pause_before,
                    pause_after=pause_after,
                )
            )
            last_end = turn_end
            current_speaker = seg.speaker
            turn_start = seg.start
            turn_end = seg.end
            turn_words = len(seg.words)

    # Final turn
    turns.append(
        SpeakerTurn(
            turn_index=len(turns),
            speaker=current_speaker,
            start_time=turn_start,
            end_time=turn_end,
            duration=turn_end - turn_start,
            word_count=turn_words,
            pause_before=max(0.0, turn_start - last_end),
            pause_after=0.0,
        )
    )

    speakers_set = set(t.speaker for t in turns)
    speech_time = sum(t.duration for t in turns)
    dur = max(transcript_data.duration, 1.0)
    turn_freq = len(turns) / (dur / 60.0)

    conv_data = ConversationData(
        duration=dur,
        num_speakers=len(speakers_set),
        turns=turns,
        turn_frequency_per_minute=round(turn_freq, 2),
        total_speech_time=round(speech_time, 2),
        total_pause_time=round(max(0.0, dur - speech_time), 2),
    )

    # Form conversational proposals from multi-turn dialogue or coherent monologues
    proposals: list[TemporalProposal] = []
    idx = 0
    while idx < len(turns):
        cand_start = turns[idx].start_time
        cand_end = turns[idx].end_time
        cand_turns = 1
        j = idx + 1
        while j < len(turns) and (turns[j].end_time - cand_start) <= max_cand_duration:
            cand_end = turns[j].end_time
            cand_turns += 1
            j += 1

        cand_dur = cand_end - cand_start
        if min_cand_duration <= cand_dur <= max_cand_duration:
            # Score based on turn density and speech fraction
            density = cand_turns / (cand_dur / 60.0)
            confidence = min(0.95, max(0.50, 0.60 + 0.05 * min(cand_turns, 6)))
            proposals.append(
                TemporalProposal(
                    proposal_id=f"conv_prop_{len(proposals)}",
                    start_time=round(cand_start, 3),
                    end_time=round(cand_end, 3),
                    duration=round(cand_dur, 3),
                    evidence_type="dialogue_exchange" if cand_turns > 1 else "monologue_thematic_unit",
                    confidence_estimate=round(confidence, 3),
                    explanation=f"Conversational discourse unit spanning {cand_turns} turn(s) across {cand_dur:.1f}s",
                    source_metadata=ProposalSourceMetadata(
                        source_type=SourceResolutionType.SPEAKER_TURN,
                        temporal_resolution_sec=0.1,
                        alignment_anchor=f"turns_{idx}:{j-1}",
                    ),
                    metadata={"turn_count": cand_turns, "speaker_count": len(speakers_set)},
                )
            )
            idx = j
        else:
            idx += 1

    bundle = ExpertEvidenceBundle(expert_name="conversation", proposals=proposals)
    return conv_data, bundle


def derive_transcript_semantic_proposals(
    transcript_data: TranscriptData,
    min_cand_duration: float = 15.0,
    max_cand_duration: float = 60.0,
) -> ExpertEvidenceBundle:
    """
    Constructs genuine transcript semantic proposals based on thematic paragraph grouping.
    """
    segments = transcript_data.segments
    proposals: list[TemporalProposal] = []
    if not segments:
        return ExpertEvidenceBundle(expert_name="transcript", proposals=[])

    step = 5
    i = 0
    while i < len(segments):
        s_start = segments[i].start
        j = i
        while j < len(segments) and (segments[j].end - s_start) <= max_cand_duration:
            j += 1

        if j > i:
            s_end = segments[j - 1].end
            dur = s_end - s_start
            if min_cand_duration <= dur <= max_cand_duration:
                # Semantic informativeness score based on lexical richness
                chunk_words = [w.word for s in segments[i:j] for w in s.words]
                unique_ratio = len(set(chunk_words)) / max(len(chunk_words), 1)
                conf = min(0.90, max(0.40, 0.50 + 0.35 * unique_ratio))
                proposals.append(
                    TemporalProposal(
                        proposal_id=f"trans_prop_{len(proposals)}",
                        start_time=round(s_start, 3),
                        end_time=round(s_end, 3),
                        duration=round(dur, 3),
                        evidence_type="semantic_importance",
                        confidence_estimate=round(conf, 3),
                        explanation=f"Thematic semantic passage ({len(chunk_words)} words, {dur:.1f}s)",
                        source_metadata=ProposalSourceMetadata(
                            source_type=SourceResolutionType.WORD_TIMESTAMP,
                            temporal_resolution_sec=0.05,
                            alignment_anchor=f"segments_{i}:{j-1}",
                        ),
                        metadata={"word_count": len(chunk_words), "unique_ratio": round(unique_ratio, 3)},
                    )
                )
        i += step

    return ExpertEvidenceBundle(expert_name="transcript", proposals=proposals)


def prepare_real_rhapsody_dataset(num_episodes: int = 5, output_dir: str = "runs/rhapsody_real_pilot"):
    """Fetches real test episodes from Hugging Face and prepares run directories and manifest."""
    backend_dir = pathlib.Path(__file__).parent.parent
    base_run_path = backend_dir / output_dir
    base_run_path.mkdir(parents=True, exist_ok=True)

    manifest_path = backend_dir / "evaluation" / "datasets" / f"rhapsody_real_test_{num_episodes}ep.json"

    logger.info(f"Loading real Rhapsody test split from Hugging Face (streaming, target={num_episodes} episodes)...")
    ds = load_dataset("yhpark/rhapsody", split="test", streaming=True)

    manifest_samples = []
    prepared_count = 0

    for row in ds:
        vid = row.get("vid")
        title = row.get("title", vid)
        meta_raw = row.get("metadata", "{}")
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
        dur = float(meta.get("duration", 0.0))
        gt = row.get("gt")

        if not vid or dur <= 0.0 or gt is None or len(gt) != 100:
            logger.warning(f"Skipping malformed row vid={vid}")
            continue

        raw_tr = row.get("transcript")
        if not raw_tr:
            logger.warning(f"Skipping row with missing transcript vid={vid}")
            continue

        sample_id = f"rhapsody_real_{vid}"
        sample_run_dir = base_run_path / sample_id
        sample_run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Parse WhisperX transcript
        transcript_data = parse_transcript_data(raw_tr, dur)
        with open(sample_run_dir / "transcript.json", "w", encoding="utf-8") as f:
            f.write(transcript_data.model_dump_json(indent=2))

        # 2. Derive Conversation & Expert Bundles (GENUINE, NOT MOCK)
        conv_data, conv_bundle = derive_conversational_evidence(transcript_data)
        with open(sample_run_dir / "conversation.json", "w", encoding="utf-8") as f:
            f.write(conv_data.model_dump_json(indent=2))
        with open(sample_run_dir / "conversation_evidence.json", "w", encoding="utf-8") as f:
            f.write(conv_bundle.model_dump_json(indent=2))

        trans_bundle = derive_transcript_semantic_proposals(transcript_data)
        with open(sample_run_dir / "transcript_evidence.json", "w", encoding="utf-8") as f:
            f.write(trans_bundle.model_dump_json(indent=2))

        # 3. Visual & Prosody are explicitly empty (raw video/audio not stored in HF dataset)
        vis_empty = ExpertEvidenceBundle(expert_name="visual", proposals=[])
        pros_empty = ExpertEvidenceBundle(expert_name="prosody", proposals=[])
        with open(sample_run_dir / "visual_evidence.json", "w", encoding="utf-8") as f:
            f.write(vis_empty.model_dump_json(indent=2))
        with open(sample_run_dir / "prosody_evidence.json", "w", encoding="utf-8") as f:
            f.write(pros_empty.model_dump_json(indent=2))

        # 4. Manifest entry (isolated replay scores for evaluation adapter only)
        rel_run_dir = str(pathlib.Path(output_dir) / sample_id).replace("\\", "/")
        manifest_samples.append({
            "sample_id": sample_id,
            "sample_type": "research",
            "split": "test",
            "duration_sec": dur,
            "run_dir": rel_run_dir,
            "replay_scores": [float(x) for x in gt],
            "metadata": {
                "youtube_vid": vid,
                "title": title,
                "channel_id": meta.get("channel_id", ""),
                "channel_title": meta.get("uploader", ""),
                "view_count": meta.get("view_count", 0),
            }
        })

        prepared_count += 1
        logger.info(
            f"Prepared episode {prepared_count}/{num_episodes}: {vid} - '{title[:40]}' ({dur:.0f}s, {len(transcript_data.words)} words, {len(conv_bundle.proposals)} conv proposals, {len(trans_bundle.proposals)} trans proposals)"
        )

        if prepared_count >= num_episodes:
            break

    manifest_obj = {
        "dataset_name": f"Rhapsody_Official_Test_Real_{num_episodes}ep",
        "dataset_type": "research",
        "created_at": "2026-10-17T20:45:00Z",
        "description": f"Real test-split episodes from younghanstark/rhapsody (COLM 2025) using official WhisperX transcripts and ground-truth replay graphs.",
        "samples": manifest_samples,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_obj, f, indent=2)

    logger.info(f"Successfully created real Rhapsody manifest at: {manifest_path}")
    return str(manifest_path)


if __name__ == "__main__":
    n = 1
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    path = prepare_real_rhapsody_dataset(num_episodes=n)
    print("MANIFEST_READY:", path)
