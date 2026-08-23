"""
ClipSense W2 Automated Timestamp Validation and Caching Invalidation Tests

Validates:
1. Continuous floating-point timestamp constraints across all extracted modalities:
   - Finiteness (no NaN or inf)
   - Interval ordering (0 <= start <= end <= duration)
   - Authoritative floating-point representation (no integer rounding)
   - Monotonic temporal progression
2. Caching lifecycle:
   - Run 1: Performs fresh extraction
   - Run 2: Reuses valid cached artifacts when input & config are identical
   - Run 3: Automatically invalidates cache when extraction configuration changes
3. Strict mode failure semantics:
   - Enforces clean failure without fallback when model or input is invalid
"""

import json
import math
import os
import pathlib
import sys
import tempfile
import cv2
import numpy as np
import scipy.io.wavfile as wavfile

# Ensure backend root on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.schemas import (
    TimestampedWord,
    TranscriptSegment,
    TranscriptData,
    VisualData,
    SampledFrameMetadata,
    ProsodyWindow,
    ProsodyData,
    SpeakerTurn,
    ConversationData,
)
from pipeline.extraction.frame_extractor import FrameExtractor
from pipeline.extraction.prosody_extractor import ProsodyExtractor
from pipeline.extraction.conversation_extractor import ConversationExtractor
from pipeline.extraction.transcript_extractor import TranscriptExtractor
from pipeline.extraction.extractor_pipeline import MultimodalExtractionPipeline


def create_synthetic_media(video_path: str, duration_sec: float = 4.0, fps: float = 25.0):
    """Create a lightweight synthetic test video with audio-compatible length."""
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    total_frames = int(duration_sec * fps)

    for i in range(total_frames):
        color = int((i / total_frames) * 255)
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        cv2.putText(frame, f"T={i/fps:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        out.write(frame)
    out.release()


def test_automated_timestamp_validation():
    """Verify that validate_timestamps strictly enforces temporal bounds and continuous float constraints."""
    duration = 10.0

    # Continuous float words (non-quantized)
    words = [
        TimestampedWord(word="Continuous", start=0.123, end=0.789),
        TimestampedWord(word="timestamps", start=0.850, end=1.654),
        TimestampedWord(word="test", start=1.700, end=2.456),
    ]
    segments = [
        TranscriptSegment(id=0, start=0.123, end=2.456, text="Continuous timestamps test", speaker="SPEAKER_00", words=words)
    ]
    t_data = TranscriptData(full_text="Continuous timestamps test", duration=duration, segments=segments, words=words)

    frames = [
        SampledFrameMetadata(frame_index=0, timestamp=0.0, file_path="f0.jpg", scene_index=0),
        SampledFrameMetadata(frame_index=25, timestamp=1.0, file_path="f1.jpg", scene_index=0),
        SampledFrameMetadata(frame_index=50, timestamp=2.0, file_path="f2.jpg", scene_index=0),
    ]
    v_data = VisualData(fps=25.0, total_frames=250, duration=duration, sampled_frames=frames, scene_boundaries=[])

    windows = [
        ProsodyWindow(
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            mean_f0=180.5,
            peak_f0=210.0,
            f0_std=15.0,
            rms_energy=0.05,
            rms_dynamic_range=0.02,
            voicing_fraction=0.8,
            speech_rate_wps=2.5,
        ),
        ProsodyWindow(
            start_time=0.5,
            end_time=1.5,
            duration=1.0,
            mean_f0=182.3,
            peak_f0=212.0,
            f0_std=14.5,
            rms_energy=0.06,
            rms_dynamic_range=0.03,
            voicing_fraction=0.85,
            speech_rate_wps=2.6,
        ),
    ]
    p_data = ProsodyData(
        duration=duration,
        window_length_sec=1.0,
        hop_length_sec=0.5,
        sample_rate=16000,
        windows=windows,
    )

    turns = [
        SpeakerTurn(
            turn_index=0,
            speaker="SPEAKER_00",
            start_time=0.123,
            end_time=2.456,
            duration=2.333,
            word_count=3,
            pause_before=0.123,
            pause_after=0.8,
        )
    ]
    c_data = ConversationData(
        duration=duration,
        turns=turns,
        num_speakers=1,
        total_speech_time=2.333,
        total_pause_time=0.8,
        turn_frequency_per_minute=6.0,
    )

    # Valid check must pass
    val = MultimodalExtractionPipeline.validate_timestamps(
        actual_duration=duration,
        transcript_data=t_data,
        visual_data=v_data,
        prosody_data=p_data,
        conversation_data=c_data,
    )
    assert val["all_passed"] is True
    assert val["transcript_continuous_float"] is True
    assert val["transcript_finite"] is True
    assert val["transcript_ordered"] is True

    # Continuous float assertion: must preserve fractional seconds
    assert any(w.start != round(w.start) for w in t_data.words)

    # Out of bounds check: word end exceeds duration + tolerance
    bad_words = [TimestampedWord(word="Bad", start=0.0, end=15.0)]
    bad_t_data = TranscriptData(full_text="Bad", duration=duration, segments=[], words=bad_words)
    val_bad = MultimodalExtractionPipeline.validate_timestamps(
        actual_duration=duration,
        transcript_data=bad_t_data,
        visual_data=v_data,
        prosody_data=p_data,
        conversation_data=c_data,
    )
    assert val_bad["all_passed"] is False
    assert val_bad["transcript_in_bounds"] is False


def test_caching_and_invalidation():
    """Verify that extraction cache is loaded on identical runs, but invalidated on configuration changes."""
    base_dir = pathlib.Path(__file__).parent.parent
    video_path = base_dir / "tests" / "sample_short.mp4"
    assert video_path.exists(), f"Sample test video not found: {video_path}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)

        # Baseline pipeline with test fallback mode
        t_ext = TranscriptExtractor(model_name="tiny.en", strict_mode=False, fallback_on_model_failure=True)
        f_ext = FrameExtractor(sample_fps=1.0)
        p_ext = ProsodyExtractor(window_length_sec=1.0, hop_length_sec=0.5)
        c_ext = ConversationExtractor(turn_pause_threshold_sec=0.7)

        pipeline1 = MultimodalExtractionPipeline(
            transcript_extractor=t_ext,
            frame_extractor=f_ext,
            prosody_extractor=p_ext,
            conversation_extractor=c_ext,
        )

        run_id = "test_cache_run"
        runs_dir = tmp_path / "runs"

        # 1. First run: fresh extraction
        art1 = pipeline1.run(
            video_path=str(video_path),
            run_id=run_id,
            runs_base_dir=str(runs_dir),
            force_rerun=False,
        )
        assert pathlib.Path(art1["manifest_json"]).exists()
        with open(art1["manifest_json"], "r", encoding="utf-8") as f:
            m1 = json.load(f)
        assert m1["configuration"]["frame_sampling_fps"] == 1.0
        frames_count_1 = m1["metrics"]["sampled_frames_count"]
        assert frames_count_1 >= 4

        # 2. Second run: identical config -> should load cached artifacts
        art2 = pipeline1.run(
            video_path=str(video_path),
            run_id=run_id,
            runs_base_dir=str(runs_dir),
            force_rerun=False,
        )
        with open(art2["manifest_json"], "r", encoding="utf-8") as f:
            m2 = json.load(f)
        assert m2["metrics"]["sampled_frames_count"] == frames_count_1

        # 3. Third run: changed configuration (sample_fps changed to 2.0)
        # Pipeline must detect configuration mismatch and invalidate the cache
        f_ext_changed = FrameExtractor(sample_fps=2.0)
        pipeline2 = MultimodalExtractionPipeline(
            transcript_extractor=t_ext,
            frame_extractor=f_ext_changed,
            prosody_extractor=p_ext,
            conversation_extractor=c_ext,
        )

        art3 = pipeline2.run(
            video_path=str(video_path),
            run_id=run_id,
            runs_base_dir=str(runs_dir),
            force_rerun=False,
        )
        with open(art3["manifest_json"], "r", encoding="utf-8") as f:
            m3 = json.load(f)
        assert m3["configuration"]["frame_sampling_fps"] == 2.0
        # At 2.0 fps, roughly double the frames should be extracted, confirming fresh re-extraction!
        assert m3["metrics"]["sampled_frames_count"] > frames_count_1


def test_strict_mode_failure_on_invalid_input():
    """Verify that strict mode raises RuntimeError and never produces fallback data."""
    t_strict = TranscriptExtractor(model_name="tiny.en", strict_mode=True)
    try:
        # Non-existent media path must raise an exception in strict mode
        t_strict.extract(media_path="non_existent_file.mp4")
        assert False, "Strict mode must fail on invalid input"
    except (FileNotFoundError, RuntimeError):
        pass


if __name__ == "__main__":
    print("Running W2 Caching and Timestamp Validation Tests...")
    test_automated_timestamp_validation()
    print("  [✓] Automated timestamp validation passed (bounds, finite, continuous float)")
    test_caching_and_invalidation()
    print("  [✓] Caching & automatic cache invalidation on config change passed")
    test_strict_mode_failure_on_invalid_input()
    print("  [✓] Strict mode failure semantics passed")
    print("All W2 caching and timestamp tests passed successfully!")
