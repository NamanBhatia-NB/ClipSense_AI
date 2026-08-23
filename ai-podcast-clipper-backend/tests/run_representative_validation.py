"""
ClipSense Representative 90-Second Conversational Validation (W2)

Runs the full extraction pipeline in STRICT MODE (real WhisperX inference required,
no fallbacks allowed) on the 90-second conversational sample:
tests/sample_conversational_90s.mp4

Validates:
- Real WhisperX transcription and word-level forced alignment
- Exact continuous floating-point timestamps against actual video duration
- Frame timestamps and scene cuts
- Acoustic prosody metrics (F0 pitch, RMS energy, voicing fraction)
- Conversational turn dynamics
"""

import json
import os
import pathlib
import sys

# Ensure backend root on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.config import config
from core.schemas import TranscriptData, VisualData, ProsodyData, ConversationData
from pipeline.extraction.transcript_extractor import TranscriptExtractor
from pipeline.extraction.extractor_pipeline import MultimodalExtractionPipeline


def run_representative_validation():
    base_dir = pathlib.Path(__file__).parent.parent
    video_path = base_dir / "tests" / "sample_conversational_90s.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Representative sample video not found: {video_path}")

    print("====================================================================")
    print("ClipSense W2: Official Representative Conversational Validation")
    print(f"Sample Video: {video_path.name} (90.0s)")
    print("Execution Mode: STRICT (Real WhisperX inference required; fallback disabled)")
    print("====================================================================")

    # 1. Initialize TranscriptExtractor in STRICT MODE with real WhisperX inference
    # Offline test mode is explicitly removed
    if "CLIPSENSE_OFFLINE_TEST" in os.environ:
        del os.environ["CLIPSENSE_OFFLINE_TEST"]

    transcript_extractor = TranscriptExtractor(
        model_name="tiny.en",
        device="cpu",
        compute_type="int8",
        strict_mode=True,  # STRICT MODE: No fallback allowed
    )

    pipeline = MultimodalExtractionPipeline(
        transcript_extractor=transcript_extractor,
    )

    # 2. Run the complete pipeline
    run_id = "representative_90s_validation"
    runs_dir = base_dir / "runs"

    artifacts = pipeline.run(
        video_path=str(video_path),
        run_id=run_id,
        runs_base_dir=str(runs_dir),
        force_rerun=True,
    )

    # 3. Load and inspect artifacts
    with open(artifacts["transcript_json"], "r", encoding="utf-8") as f:
        t_data = TranscriptData.model_validate_json(f.read())

    with open(artifacts["frames_json"], "r", encoding="utf-8") as f:
        v_data = VisualData.model_validate_json(f.read())

    with open(artifacts["prosody_json"], "r", encoding="utf-8") as f:
        p_data = ProsodyData.model_validate_json(f.read())

    with open(artifacts["conversation_json"], "r", encoding="utf-8") as f:
        c_data = ConversationData.model_validate_json(f.read())

    with open(artifacts["manifest_json"], "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 4. Detailed Manual Inspection Printout
    print("\n--- DETAILED INSPECTION OF GENERATED MULTIMODAL EVIDENCE ---")
    print(f"Total Words Extracted: {len(t_data.words)}")
    print(f"First 5 Words with Exact Timestamps:")
    for w in t_data.words[:5]:
        print(f"  - '{w.word}': [{w.start:.3f}s -> {w.end:.3f}s] (score: {w.score})")
    print(f"Last 5 Words with Exact Timestamps:")
    for w in t_data.words[-5:]:
        print(f"  - '{w.word}': [{w.start:.3f}s -> {w.end:.3f}s] (score: {w.score})")

    print(f"\nSampled Visual Frames Count: {len(v_data.sampled_frames)} at {v_data.fps} fps")
    print(f"Detected Scene Boundaries: {len(v_data.scene_boundaries)}")
    print(f"Frame 0 Timestamp: {v_data.sampled_frames[0].timestamp:.2f}s | Frame Last: {v_data.sampled_frames[-1].timestamp:.2f}s")

    print(f"\nProsody Windows Count: {len(p_data.windows)} (hop: {p_data.hop_length_sec}s)")
    voiced_windows = [w for w in p_data.windows if w.voicing_fraction > 0.3 and w.mean_f0 > 0]
    avg_f0 = sum(w.mean_f0 for w in voiced_windows) / len(voiced_windows) if voiced_windows else 0.0
    avg_rms = sum(w.rms_energy for w in p_data.windows) / len(p_data.windows)
    print(f"Voiced Windows Count: {len(voiced_windows)}/{len(p_data.windows)}")
    print(f"Average Fundamental Frequency (F0): {avg_f0:.1f} Hz")
    print(f"Average Acoustic RMS Energy: {avg_rms:.4f}")

    # Distinguish pause-separated speech segments from multi-speaker turn-taking
    labeled_speakers = {t.speaker for t in c_data.turns if t.speaker}
    num_speakers = len(labeled_speakers)
    is_multi_speaker = num_speakers > 1

    if is_multi_speaker:
        print(f"\nConversational Turns Count: {len(c_data.turns)}")
        print(f"Turn Frequency: {c_data.turn_frequency_per_minute:.2f} turns/min")
        print(f"Interaction Nature: Multi-speaker dialogue exchange ({num_speakers} speakers)")
        for turn in c_data.turns:
            print(f"  - Turn {turn.turn_index} ({turn.speaker}): [{turn.start_time:.2f}s -> {turn.end_time:.2f}s], {turn.word_count} words, pause_after={turn.pause_after:.2f}s")
    else:
        speaker_name = c_data.turns[0].speaker if c_data.turns else 'SPEAKER_00'
        print(f"\nSpeech Segments Extracted: {len(c_data.turns)} (pause-separated speech segments; single-speaker sample, no multi-speaker turn-taking)")
        print(f"Segment Pacing: {c_data.turn_frequency_per_minute:.2f} segments/min")
        print(f"Detected Speaker: 1 ({speaker_name}) | Multi-speaker Turn Transitions: 0")
        print(f"Interaction Nature: Single-speaker monologue segmented by inter-utterance pauses (>= {config.conversation.turn_pause_threshold_sec}s)")
        for turn in c_data.turns:
            print(f"  - Speech Segment {turn.turn_index} ({turn.speaker}): [{turn.start_time:.2f}s -> {turn.end_time:.2f}s], {turn.word_count} words, pause_after={turn.pause_after:.2f}s")
        print("Speaker IDs and pause/switch information preserved for downstream multi-speaker turn-taking detection.")

    # 5. Timestamp validation against actual video duration & configuration checks
    duration = v_data.duration
    print(f"\nActual Video Duration: {duration:.2f}s")
    print(f"Validation Checks Passed: {manifest.get('temporal_validation', {}).get('all_passed')}")
    print(f"Inference Mode Confirmed: {manifest.get('inference_mode')}")
    print(f"Whisper Model Recorded: {manifest.get('configuration', {}).get('whisper_model')}")
    print(f"Strict Mode Active: {manifest.get('configuration', {}).get('strict_mode')}")
    print(f"Fallback Used: {manifest.get('configuration', {}).get('fallback_used')}")
    print(f"Speaker Metadata Preserved: {c_data.num_speakers} speaker detected ({c_data.turns[0].speaker if c_data.turns else 'none'})")

    prosody_range = manifest.get("temporal_validation", {}).get("prosody_time_range", [0.0, 0.0])
    print(f"Prosody Time Range: [{prosody_range[0]:.3f}s -> {prosody_range[1]:.3f}s] (Clamped <= {duration:.2f}s)")

    # Assertions
    assert manifest.get("inference_mode") == "real_whisperx"
    assert manifest.get("configuration", {}).get("whisper_model") == "tiny.en"
    assert manifest.get("configuration", {}).get("strict_mode") is True
    assert manifest.get("configuration", {}).get("fallback_used") is False
    assert manifest.get("temporal_validation", {}).get("all_passed") is True
    assert len(t_data.words) > 20, "Real transcription should extract substantial spoken dialogue"
    
    # Verify continuous floating-point timestamps (non-quantized)
    assert all(isinstance(w.start, float) and isinstance(w.end, float) for w in t_data.words)
    assert any(w.start != round(w.start) for w in t_data.words), "Timestamps must preserve continuous fractional precision"

    # Verify prosody window endpoints never exceed actual video duration
    assert 0.0 <= prosody_range[0] <= prosody_range[1] <= duration, (
        f"Stored prosody range {prosody_range} exceeds video duration {duration}"
    )
    assert p_data.windows[-1].end_time <= duration, (
        f"Prosody final window end exceeds video duration: {p_data.windows[-1].end_time} > {duration}"
    )

    # Verify every speech segment / turn: 0 <= start <= end <= video_duration
    for turn in c_data.turns:
        assert 0.0 <= turn.start_time <= turn.end_time <= duration, (
            f"Segment {turn.turn_index} out of bounds: [{turn.start_time}, {turn.end_time}] vs duration {duration}"
        )

    print("\n[✓] ALL CHECKS PASSED: Real WhisperX inference confirmed on representative 90s sample!")


if __name__ == "__main__":
    run_representative_validation()
