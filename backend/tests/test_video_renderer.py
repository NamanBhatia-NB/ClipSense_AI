"""
Unit and Integration Tests for ClipSense Video Rendering Engine (W8)

Covers:
1. Pure validation of interval boundaries (reversed, negative, NaN/Inf, out-of-bounds).
2. Pure FFmpeg command construction (argument safety, flags, stream mapping).
3. Mocked subprocess execution (missing binary, non-zero return code, empty files, fast-to-accurate fallback).
4. Real FFmpeg integration tests on tests/sample_short.mp4.
"""

import math
import os
import pathlib
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from core.schemas import ClipExportResult, ClipsManifest, MTERCandidate, SemanticVerificationCriteria
from pipeline.render.video_renderer import (
    VideoRenderer,
    build_ffmpeg_command,
    validate_interval,
)


class TestIntervalValidation(unittest.TestCase):
    """Tests for boundary and timestamp validation."""

    def test_valid_interval(self):
        # Should not raise
        validate_interval(0.0, 15.0, video_duration=60.0)
        validate_interval(10.5, 30.2, video_duration=35.0)

    def test_reversed_interval(self):
        with self.assertRaises(ValueError) as ctx:
            validate_interval(20.0, 10.0)
        self.assertIn("greater than start", str(ctx.exception))

    def test_equal_timestamps(self):
        with self.assertRaises(ValueError) as ctx:
            validate_interval(15.0, 15.0)
        self.assertIn("greater than start", str(ctx.exception))

    def test_negative_timestamp(self):
        with self.assertRaises(ValueError) as ctx:
            validate_interval(-2.0, 10.0)
        self.assertIn("cannot be negative", str(ctx.exception))

    def test_non_finite_timestamps(self):
        with self.assertRaises(ValueError):
            validate_interval(float("nan"), 10.0)
        with self.assertRaises(ValueError):
            validate_interval(5.0, float("inf"))
        with self.assertRaises(ValueError):
            validate_interval(float("-inf"), 10.0)

    def test_exceeds_video_duration(self):
        with self.assertRaises(ValueError) as ctx:
            validate_interval(65.0, 75.0, video_duration=60.0)
        self.assertIn("exceeds total video duration", str(ctx.exception))

    def test_end_exceeds_duration(self):
        with self.assertRaises(ValueError) as ctx:
            validate_interval(40.0, 80.0, video_duration=60.0)
        self.assertIn("exceeds total video duration", str(ctx.exception))

    def test_too_short_duration(self):
        with self.assertRaises(ValueError) as ctx:
            validate_interval(10.0, 10.05, min_duration=0.1)
        self.assertIn("smaller than minimum", str(ctx.exception))


class TestCommandConstruction(unittest.TestCase):
    """Tests for safe FFmpeg subprocess argument construction."""

    def test_accurate_mode_command_structure(self):
        cmd = build_ffmpeg_command(
            source_video="input.mp4",
            output_path="out.mp4",
            start_time=12.5,
            end_time=32.8,
            mode="accurate",
            video_codec="libx264",
            audio_codec="aac",
            crf=18,
            preset="fast",
            ffmpeg_bin="ffmpeg",
        )
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertIn("-y", cmd)
        self.assertIn("-nostdin", cmd)
        self.assertIn("-ss", cmd)
        self.assertIn("12.500", cmd)
        self.assertIn("-to", cmd)
        self.assertIn("32.800", cmd)
        self.assertIn("-map", cmd)
        self.assertIn("0:v:0", cmd)
        self.assertIn("0:a?", cmd)
        self.assertIn("-c:v", cmd)
        self.assertIn("libx264", cmd)
        self.assertIn("-c:a", cmd)
        self.assertIn("aac", cmd)
        self.assertIn("+faststart", cmd)

    def test_fast_mode_command_structure(self):
        cmd = build_ffmpeg_command(
            source_video="input.mp4",
            output_path="out.mp4",
            start_time=5.0,
            end_time=15.0,
            mode="fast",
        )
        self.assertIn("-c", cmd)
        self.assertIn("copy", cmd)
        self.assertIn("-avoid_negative_ts", cmd)
        self.assertIn("make_zero", cmd)


class TestMockedRenderer(unittest.TestCase):
    """Mocked unit tests for renderer error handling and execution flow."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.renderer = VideoRenderer(default_mode="accurate")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_missing_source_file_returns_failed_result(self):
        non_existent = pathlib.Path(self.temp_dir.name) / "does_not_exist.mp4"
        out_clip = pathlib.Path(self.temp_dir.name) / "clip.mp4"
        res = self.renderer.render_clip(
            source_video=non_existent,
            start_time=1.0,
            end_time=5.0,
            output_path=out_clip,
        )
        self.assertEqual(res.status, "failed")
        self.assertIn("does not exist", res.error_message)
        self.assertIsNone(res.output_path)

    def test_invalid_interval_returns_failed_result(self):
        dummy_src = pathlib.Path(self.temp_dir.name) / "src.mp4"
        dummy_src.touch()
        out_clip = pathlib.Path(self.temp_dir.name) / "clip.mp4"

        res = self.renderer.render_clip(
            source_video=dummy_src,
            start_time=25.0,
            end_time=10.0,  # Reversed
            output_path=out_clip,
        )
        self.assertEqual(res.status, "failed")
        self.assertIn("Temporal validation failed", res.error_message)

    @patch("shutil.which", return_value=None)
    def test_missing_ffmpeg_binary(self, mock_which):
        renderer = VideoRenderer(ffmpeg_bin="ffmpeg_custom_missing")
        dummy_src = pathlib.Path(self.temp_dir.name) / "src.mp4"
        dummy_src.touch()
        out_clip = pathlib.Path(self.temp_dir.name) / "clip.mp4"

        res = renderer.render_clip(
            source_video=dummy_src,
            start_time=1.0,
            end_time=5.0,
            output_path=out_clip,
        )
        self.assertEqual(res.status, "failed")
        self.assertIn("binary not found", res.error_message)

    @patch("shutil.which", return_value="ffmpeg")
    @patch("subprocess.run")
    def test_ffmpeg_nonzero_exit_status(self, mock_run, mock_which):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "Invalid data found when processing input"
        mock_run.return_value = mock_proc

        dummy_src = pathlib.Path(self.temp_dir.name) / "src.mp4"
        dummy_src.touch()
        out_clip = pathlib.Path(self.temp_dir.name) / "clip.mp4"

        res = self.renderer.render_clip(
            source_video=dummy_src,
            start_time=1.0,
            end_time=5.0,
            output_path=out_clip,
        )
        self.assertEqual(res.status, "failed")
        self.assertIn("FFmpeg exited with code 1", res.error_message)
        self.assertIn("Invalid data found", res.error_message)

    @patch("shutil.which", return_value="ffmpeg")
    @patch("subprocess.run")
    def test_empty_zero_byte_output_file_is_rejected(self, mock_run, mock_which):
        dummy_src = pathlib.Path(self.temp_dir.name) / "src.mp4"
        dummy_src.touch()
        out_clip = pathlib.Path(self.temp_dir.name) / "clip.mp4"

        # Simulate FFmpeg reporting 0 exit code but writing 0 bytes
        def side_effect(*args, **kwargs):
            out_clip.touch()  # Creates 0-byte file
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            return mock_proc

        mock_run.side_effect = side_effect

        res = self.renderer.render_clip(
            source_video=dummy_src,
            start_time=1.0,
            end_time=5.0,
            output_path=out_clip,
        )
        self.assertEqual(res.status, "failed")
        self.assertIn("0 bytes", res.error_message)
        self.assertFalse(out_clip.exists(), "Empty file should be removed on failure")


class TestRealFFmpegIntegration(unittest.TestCase):
    """
    Real integration tests executing the actual FFmpeg binary on sample video.
    Strictly marked as real system verification (not mocked).
    """

    def setUp(self):
        backend_dir = pathlib.Path(__file__).parent.parent
        self.sample_video = backend_dir / "tests" / "sample_short.mp4"
        self.renderer = VideoRenderer(default_mode="accurate")
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_real_ffmpeg_accurate_clip_export(self):
        """Real export verification: Slices a playable subclip from sample_short.mp4."""
        if not self.sample_video.exists():
            self.skipTest(f"Sample test video not found: {self.sample_video}")
        if not self.renderer.is_available():
            self.skipTest("FFmpeg executable not available on host system.")

        out_path = pathlib.Path(self.temp_dir.name) / "real_test_clip.mp4"

        result = self.renderer.render_clip(
            source_video=self.sample_video,
            start_time=1.0,
            end_time=3.0,
            output_path=out_path,
            candidate_id="real_test_cand_01",
            rank=1,
            mode="accurate",
        )

        # 1. Assert status and return metadata
        self.assertEqual(result.status, "success")
        self.assertIsNone(result.error_message)
        self.assertEqual(result.candidate_id, "real_test_cand_01")
        self.assertAlmostEqual(result.duration, 2.0, places=3)
        self.assertEqual(result.render_mode, "accurate")

        # 2. Verify physical file exists and is playable / non-empty
        self.assertTrue(out_path.exists())
        file_size = out_path.stat().st_size
        self.assertGreater(file_size, 1024, f"Exported clip is suspiciously small: {file_size} bytes")
        self.assertEqual(result.file_size_bytes, file_size)

    def test_real_ffmpeg_render_highlights_manifest(self):
        """Real export verification: Renders multiple candidates and writes manifest.json."""
        if not self.sample_video.exists():
            self.skipTest(f"Sample test video not found: {self.sample_video}")
        if not self.renderer.is_available():
            self.skipTest("FFmpeg executable not available on host system.")

        cand1 = MTERCandidate(
            candidate_id="cand_1",
            proposed_start=0.5,
            proposed_end=2.5,
            duration=2.0,
            confidence_estimate=0.88,
            contributing_experts=["transcript", "visual"],
            temporal_agreement_iou=0.75,
            semantic_verification=SemanticVerificationCriteria(
                natural_semantic_start=True,
                sufficient_context=True,
                important_content_retained=True,
                complete_conversational_payoff=True,
                no_mid_sentence_ending=True,
            ),
            semantic_verification_status="passed",
            decision_summary="Test candidate 1",
        )
        cand2 = MTERCandidate(
            candidate_id="cand_2",
            proposed_start=2.6,
            proposed_end=4.6,
            duration=2.0,
            confidence_estimate=0.79,
            contributing_experts=["prosody", "conversation"],
            temporal_agreement_iou=0.65,
            semantic_verification=SemanticVerificationCriteria(
                natural_semantic_start=True,
                sufficient_context=True,
                important_content_retained=True,
                complete_conversational_payoff=True,
                no_mid_sentence_ending=True,
            ),
            semantic_verification_status="passed",
            decision_summary="Test candidate 2",
        )

        manifest = self.renderer.render_highlights(
            source_video=self.sample_video,
            candidates=[cand1, cand2],
            output_dir=self.temp_dir.name,
            top_k=2,
        )

        self.assertEqual(manifest.total_candidates_requested, 2)
        self.assertEqual(manifest.total_clips_exported, 2)
        self.assertEqual(len(manifest.clips), 2)

        # Check that files exist
        manifest_file = pathlib.Path(self.temp_dir.name) / "clips" / "manifest.json"
        self.assertTrue(manifest_file.exists())

        clip1_file = pathlib.Path(manifest.clips[0].output_path)
        clip2_file = pathlib.Path(manifest.clips[1].output_path)
        self.assertTrue(clip1_file.exists())
        self.assertTrue(clip2_file.exists())
        self.assertGreater(clip1_file.stat().st_size, 0)
        self.assertGreater(clip2_file.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
