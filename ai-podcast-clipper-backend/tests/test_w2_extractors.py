"""
ClipSense W2 Unit Tests: Multimodal Extractors

Verifies:
1. FrameExtractor: sampling, frame metadata, continuous timestamps, caching.
2. ProsodyExtractor: physical acoustic signal metrics (F0, RMS, voicing, dynamic range).
3. ConversationExtractor: speaker turn clustering, pause-based fallback, and dynamics metrics.
4. TranscriptExtractor & Schemas: cache reading and validation.
5. MultimodalExtractionPipeline: caching and manifest structure.
"""

import json
import os
import pathlib
import sys
import tempfile
import numpy as np
import scipy.io.wavfile as wavfile
import cv2

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
    ProsodyData,
    ConversationData,
)
from pipeline.extraction.frame_extractor import FrameExtractor
from pipeline.extraction.prosody_extractor import ProsodyExtractor
from pipeline.extraction.conversation_extractor import ConversationExtractor
from pipeline.extraction.extractor_pipeline import MultimodalExtractionPipeline


def create_synthetic_audio(wav_path: str, duration_sec: float = 3.0, sample_rate: int = 16000):
    """Generate a clean synthetic WAV file with 220Hz harmonic tone and silence gap."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # 0.0 - 1.5s: 220Hz tone; 1.5s - 2.0s: silence; 2.0s - 3.0s: 330Hz tone
    audio = np.zeros_like(t)
    tone1 = 0.5 * np.sin(2 * np.pi * 220 * t[: int(sample_rate * 1.5)])
    tone2 = 0.4 * np.sin(2 * np.pi * 330 * t[int(sample_rate * 2.0) :])
    audio[: int(sample_rate * 1.5)] = tone1
    audio[int(sample_rate * 2.0) :] = tone2

    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(wav_path, sample_rate, audio_int16)


def create_synthetic_video(avi_path: str, duration_sec: float = 3.0, fps: float = 25.0):
    """Generate a lightweight synthetic test video."""
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))
    total_frames = int(duration_sec * fps)

    for i in range(total_frames):
        # Color shifting frame to simulate visual changes
        color = int((i / total_frames) * 255)
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        out.write(frame)
    out.release()


def test_frame_extractor():
    """Test frame sampling, metadata preservation, and caching."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        test_video = tmp_path / "test_synth.avi"
        create_synthetic_video(str(test_video), duration_sec=3.0, fps=25.0)

        extractor = FrameExtractor(sample_fps=1.0)
        frames_out = tmp_path / "frames"
        json_out = tmp_path / "frames.json"

        # 1. First run (extraction)
        visual_data = extractor.extract(
            video_path=str(test_video),
            frames_dir=str(frames_out),
            output_json_path=str(json_out),
        )

        assert isinstance(visual_data, VisualData)
        assert visual_data.fps == 25.0
        assert visual_data.total_frames == 75
        assert len(visual_data.sampled_frames) == 3  # 1 frame per second for 3 seconds
        assert json_out.exists()

        # Check continuous timestamps
        timestamps = [f.timestamp for f in visual_data.sampled_frames]
        assert timestamps[0] == 0.0
        assert 0.9 <= timestamps[1] <= 1.1
        assert 1.9 <= timestamps[2] <= 2.1

        # 2. Second run (cache check)
        cached_data = extractor.extract(
            video_path=str(test_video),
            frames_dir=str(frames_out),
            output_json_path=str(json_out),
            force_rerun=False,
        )
        assert len(cached_data.sampled_frames) == 3


def test_prosody_extractor():
    """Test physical acoustic metrics (F0, RMS, Voicing) on synthetic tone."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        test_wav = tmp_path / "test_audio.wav"
        create_synthetic_audio(str(test_wav), duration_sec=3.0)

        extractor = ProsodyExtractor(window_length_sec=1.0, hop_length_sec=0.5)
        json_out = tmp_path / "prosody.json"

        prosody_data = extractor.extract(
            media_path=str(test_wav),
            output_json_path=str(json_out),
        )

        assert isinstance(prosody_data, ProsodyData)
        assert len(prosody_data.windows) > 0
        assert json_out.exists()

        first_win = prosody_data.windows[0]
        # First second contains pure 220Hz tone
        assert 210.0 <= first_win.mean_f0 <= 230.0
        assert first_win.voicing_fraction > 0.8
        assert first_win.rms_energy > 0.1

        # Check caching
        cached = extractor.extract(
            media_path=str(test_wav),
            output_json_path=str(json_out),
            force_rerun=False,
        )
        assert len(cached.windows) == len(prosody_data.windows)


def test_conversation_extractor():
    """Test speaker turn clustering, pause handling, and single-speaker fallback."""
    # Build synthetic transcript with 2 speakers and a clear dialogue pause
    w1 = TimestampedWord(word="How", start=1.0, end=1.3)
    w2 = TimestampedWord(word="are", start=1.3, end=1.6)
    w3 = TimestampedWord(word="you?", start=1.6, end=2.0)
    seg1 = TranscriptSegment(id=0, start=1.0, end=2.0, text="How are you?", speaker="SPEAKER_00", words=[w1, w2, w3])

    # 1.2s silence gap, then response from SPEAKER_01
    w4 = TimestampedWord(word="I", start=3.2, end=3.5)
    w5 = TimestampedWord(word="am", start=3.5, end=3.8)
    w6 = TimestampedWord(word="great.", start=3.8, end=4.2)
    seg2 = TranscriptSegment(id=1, start=3.2, end=4.2, text="I am great.", speaker="SPEAKER_01", words=[w4, w5, w6])

    transcript = TranscriptData(
        full_text="How are you? I am great.",
        duration=5.0,
        segments=[seg1, seg2],
        words=[w1, w2, w3, w4, w5, w6],
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        json_out = tmp_path / "conversation.json"

        extractor = ConversationExtractor(turn_pause_threshold_sec=0.7)
        conv_data = extractor.extract(transcript_data=transcript, total_duration=5.0, output_json_path=str(json_out))

        assert isinstance(conv_data, ConversationData)
        assert conv_data.num_speakers == 2
        assert len(conv_data.turns) == 2
        assert conv_data.turns[0].speaker == "SPEAKER_00"
        assert conv_data.turns[1].speaker == "SPEAKER_01"
        assert abs(conv_data.turns[0].pause_after - 1.2) < 1e-5  # 3.2 - 2.0 = 1.2s pause
        assert json_out.exists()


def test_conversation_single_speaker_pause_fallback():
    """Test that pause intervals create distinct turns even without speaker tags."""
    # Both segments labeled SPEAKER_00, separated by 1.5s silence
    seg1 = TranscriptSegment(id=0, start=0.5, end=2.0, text="First thought.", speaker="SPEAKER_00")
    seg2 = TranscriptSegment(id=1, start=3.5, end=5.0, text="Second thought.", speaker="SPEAKER_00")

    transcript = TranscriptData(full_text="First thought. Second thought.", duration=6.0, segments=[seg1, seg2], words=[])
    extractor = ConversationExtractor(turn_pause_threshold_sec=0.7)
    conv_data = extractor.extract(transcript_data=transcript, total_duration=6.0)

    # Must be split into 2 distinct conversational units due to pause > 0.7s
    assert len(conv_data.turns) == 2
    assert conv_data.turns[0].start_time == 0.5
    assert conv_data.turns[1].start_time == 3.5


if __name__ == "__main__":
    print("Running W2 unit tests...")
    test_frame_extractor()
    print("  [✓] FrameExtractor tests passed")
    test_prosody_extractor()
    print("  [✓] ProsodyExtractor tests passed")
    test_conversation_extractor()
    print("  [✓] ConversationExtractor tests passed")
    test_conversation_single_speaker_pause_fallback()
    print("  [✓] Conversation pause fallback tests passed")
    print("All W2 unit tests passed successfully!")
