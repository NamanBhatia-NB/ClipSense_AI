"""
ClipSense Video Rendering Engine (W8)

Provides deterministic, robust FFmpeg-based video slicing for MTER highlight proposals:
- Validates temporal boundaries (finite, positive, ordered, duration-bounded).
- Executes FFmpeg via safely constructed argument lists (no shell interpolation).
- Accurate frame-level re-encoding (libx264/aac) with fallback from fast stream-copy.
- Handles audio-optional video streams gracefully via optional stream mapping (-map 0:a?).
- Returns structured ClipExportResult metadata and writes ClipsManifest catalogs.
"""

import logging
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from core.schemas import ClipExportResult, ClipsManifest, MTERCandidate
from pipeline.render.caption_generator import burn_subtitles_ffmpeg, generate_karaoke_ass
from pipeline.render.vertical_cropper import ActiveSpeakerCropper

logger = logging.getLogger(__name__)


def validate_interval(
    start_time: float,
    end_time: float,
    video_duration: Optional[float] = None,
    min_duration: float = 0.1,
) -> None:
    """
    Validates that temporal interval boundaries are mathematically valid and within bounds.
    
    Raises:
        ValueError: If start/end timestamps are non-finite, reversed, negative,
                    or exceed video_duration.
    """
    if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
        raise ValueError(
            f"Start ({start_time!r}) and end ({end_time!r}) timestamps must be numeric floats or ints."
        )

    if not math.isfinite(start_time) or not math.isfinite(end_time):
        raise ValueError(
            f"Interval boundaries must be finite real numbers. Received start={start_time}, end={end_time}."
        )

    if start_time < 0.0:
        raise ValueError(
            f"Start timestamp cannot be negative. Received start={start_time:.3f}s."
        )

    if end_time <= start_time:
        raise ValueError(
            f"End timestamp ({end_time:.3f}s) must be strictly greater than start timestamp ({start_time:.3f}s)."
        )

    duration = end_time - start_time
    if duration < min_duration:
        raise ValueError(
            f"Clip duration ({duration:.3f}s) is smaller than minimum required ({min_duration:.3f}s)."
        )

    if video_duration is not None:
        if not math.isfinite(video_duration) or video_duration <= 0.0:
            raise ValueError(f"Video duration must be a positive finite number: {video_duration}")
        # Allow tiny numerical tolerance for floating point boundary clamping (0.5s)
        if start_time >= video_duration:
            raise ValueError(
                f"Start timestamp ({start_time:.3f}s) equals or exceeds total video duration ({video_duration:.3f}s)."
            )
        if end_time > video_duration + 0.5:
            raise ValueError(
                f"End timestamp ({end_time:.3f}s) exceeds total video duration ({video_duration:.3f}s)."
            )


def build_ffmpeg_command(
    source_video: Union[str, Path],
    output_path: Union[str, Path],
    start_time: float,
    end_time: float,
    mode: Literal["accurate", "fast"] = "accurate",
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    crf: int = 18,
    preset: str = "fast",
    ffmpeg_bin: str = "ffmpeg",
) -> List[str]:
    """
    Constructs a safe, non-shell-interpolated FFmpeg argument list for subclip extraction.
    
    In 'accurate' mode:
      Fast input seeking (-ss before -i) paired with libx264 re-encoding guarantees
      exact frame boundaries and synchronized audio.
    In 'fast' mode:
      Stream-copy (-c copy) slices at nearest I-frames for maximum speed without transcoding.
    """
    src_str = str(Path(source_video).resolve())
    dst_str = str(Path(output_path).resolve())

    # Fast input seek with exact duration cut
    cmd: List[str] = [
        ffmpeg_bin,
        "-y",               # Overwrite output without prompting
        "-nostdin",          # Disable interactive input on stdin
        "-ss", f"{start_time:.3f}",
        "-to", f"{end_time:.3f}",
        "-i", src_str,
        "-map", "0:v:0",     # Primary video stream
        "-map", "0:a?",      # Primary audio stream if present, optional without error
    ]

    if mode == "fast":
        cmd.extend([
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            dst_str,
        ])
    else:
        cmd.extend([
            "-c:v", video_codec,
            "-c:a", audio_codec,
            "-crf", str(crf),
            "-preset", preset,
            "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero",
            dst_str,
        ])

    return cmd


class VideoRenderer:
    """
    Production-grade video rendering engine for ClipSense highlight extraction.
    """

    def __init__(
        self,
        ffmpeg_bin: Optional[str] = None,
        default_mode: Literal["accurate", "fast"] = "accurate",
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        crf: int = 18,
        preset: str = "fast",
        fallback_to_reencode: bool = True,
        timeout_sec: float = 300.0,
    ):
        self.ffmpeg_bin = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
        self.default_mode = default_mode
        self.video_codec = video_codec
        self.audio_codec = audio_codec
        self.crf = crf
        self.preset = preset
        self.fallback_to_reencode = fallback_to_reencode
        self.timeout_sec = timeout_sec

    def is_available(self) -> bool:
        """Checks if the FFmpeg executable is present and callable on the system."""
        if not shutil.which(self.ffmpeg_bin):
            return False
        try:
            res = subprocess.run(
                [self.ffmpeg_bin, "-version"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            return res.returncode == 0
        except Exception:
            return False

    def render_clip(
        self,
        source_video: Union[str, Path],
        start_time: float,
        end_time: float,
        output_path: Union[str, Path],
        candidate_id: str = "cand_manual",
        rank: int = 1,
        video_duration: Optional[float] = None,
        mode: Optional[Literal["accurate", "fast"]] = None,
        aspect_ratio: str = "16:9",
        add_subtitles: bool = False,
        transcript_words: Optional[List[Dict[str, Any]]] = None,
    ) -> ClipExportResult:
        """
        Extracts and renders a single highlight clip to disk using FFmpeg.
        
        Returns:
            ClipExportResult with complete execution status and output verification.
        """
        t0 = time.time()
        src_path = Path(source_video).resolve()
        dst_path = Path(output_path).resolve()
        active_mode = mode or self.default_mode
        raw_dur = end_time - start_time if (math.isfinite(end_time) and math.isfinite(start_time)) else 0.0
        duration = max(0.0, raw_dur)
        safe_start = start_time if math.isfinite(start_time) and start_time >= 0.0 else 0.0
        safe_end = end_time if math.isfinite(end_time) and end_time >= 0.0 else safe_start

        # 1. Temporal boundary validation
        try:
            validate_interval(
                start_time=start_time,
                end_time=end_time,
                video_duration=video_duration,
            )
        except ValueError as err:
            logger.error("Boundary validation error for candidate %s: %s", candidate_id, err)
            return ClipExportResult(
                candidate_id=candidate_id,
                rank=rank,
                source_video=str(src_path),
                start_time=safe_start,
                end_time=safe_end,
                duration=duration,
                output_path=None,
                status="failed",
                error_message=f"Temporal validation failed: {str(err)}",
                render_mode=active_mode,
                execution_time_sec=time.time() - t0,
            )

        # 2. Source file verification
        if not src_path.exists():
            return ClipExportResult(
                candidate_id=candidate_id,
                rank=rank,
                source_video=str(src_path),
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                output_path=None,
                status="failed",
                error_message=f"Source video file does not exist: {src_path}",
                render_mode=active_mode,
                execution_time_sec=time.time() - t0,
            )

        # 3. FFmpeg executable availability
        if not shutil.which(self.ffmpeg_bin):
            return ClipExportResult(
                candidate_id=candidate_id,
                rank=rank,
                source_video=str(src_path),
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                output_path=None,
                status="failed",
                error_message=f"FFmpeg binary not found in PATH or configured path: '{self.ffmpeg_bin}'",
                render_mode=active_mode,
                execution_time_sec=time.time() - t0,
            )

        # 4. Ensure destination directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # 5. Build and execute command
        cmd = build_ffmpeg_command(
            source_video=src_path,
            output_path=dst_path,
            start_time=start_time,
            end_time=end_time,
            mode=active_mode,
            video_codec=self.video_codec,
            audio_codec=self.audio_codec,
            crf=self.crf,
            preset=self.preset,
            ffmpeg_bin=self.ffmpeg_bin,
        )

        try:
            logger.debug("Executing FFmpeg command: %s", " ".join(cmd))
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return ClipExportResult(
                candidate_id=candidate_id,
                rank=rank,
                source_video=str(src_path),
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                output_path=None,
                status="failed",
                error_message=f"FFmpeg process timed out after {self.timeout_sec:.1f}s",
                render_mode=active_mode,
                execution_time_sec=time.time() - t0,
            )
        except Exception as ex:
            return ClipExportResult(
                candidate_id=candidate_id,
                rank=rank,
                source_video=str(src_path),
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                output_path=None,
                status="failed",
                error_message=f"FFmpeg subprocess execution failed: {str(ex)}",
                render_mode=active_mode,
                execution_time_sec=time.time() - t0,
            )

        # 6. Fallback handling: If fast stream copy fails, attempt accurate re-encoding
        if proc.returncode != 0 and active_mode == "fast" and self.fallback_to_reencode:
            logger.warning(
                "Fast stream-copy failed for candidate %s. Falling back to accurate re-encoding mode.",
                candidate_id,
            )
            active_mode = "accurate"
            fallback_cmd = build_ffmpeg_command(
                source_video=src_path,
                output_path=dst_path,
                start_time=start_time,
                end_time=end_time,
                mode="accurate",
                video_codec=self.video_codec,
                audio_codec=self.audio_codec,
                crf=self.crf,
                preset=self.preset,
                ffmpeg_bin=self.ffmpeg_bin,
            )
            try:
                proc = subprocess.run(
                    fallback_cmd,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                )
            except Exception as ex:
                return ClipExportResult(
                    candidate_id=candidate_id,
                    rank=rank,
                    source_video=str(src_path),
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    output_path=None,
                    status="failed",
                    error_message=f"Re-encode fallback failed: {str(ex)}",
                    render_mode=active_mode,
                    execution_time_sec=time.time() - t0,
                )

        # 7. Check final return code
        if proc.returncode != 0:
            err_snippet = (proc.stderr or "").strip()
            if len(err_snippet) > 400:
                err_snippet = err_snippet[-400:]
            return ClipExportResult(
                candidate_id=candidate_id,
                rank=rank,
                source_video=str(src_path),
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                output_path=None,
                status="failed",
                error_message=f"FFmpeg exited with code {proc.returncode}: {err_snippet}",
                render_mode=active_mode,
                execution_time_sec=time.time() - t0,
            )

        # 8. Verify output file exists and is non-empty
        if not dst_path.exists():
            return ClipExportResult(
                candidate_id=candidate_id,
                rank=rank,
                source_video=str(src_path),
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                output_path=None,
                status="failed",
                error_message="Output clip was not created by FFmpeg despite exit code 0.",
                render_mode=active_mode,
                execution_time_sec=time.time() - t0,
            )

        file_size = dst_path.stat().st_size
        if file_size == 0:
            dst_path.unlink(missing_ok=True)
            return ClipExportResult(
                candidate_id=candidate_id,
                rank=rank,
                source_video=str(src_path),
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                output_path=None,
                status="failed",
                error_message="Output clip is 0 bytes (empty file). Deleted incomplete file.",
                render_mode=active_mode,
                execution_time_sec=time.time() - t0,
            )

        # 9. Optional: Active speaker vertical cropping (9:16)
        if aspect_ratio == "9:16":
            vert_path = dst_path.with_name(f"{dst_path.stem}_vert{dst_path.suffix}")
            logger.info("Starting active speaker vertical cropping (9:16) on %s", dst_path.name)
            cropper = ActiveSpeakerCropper(ffmpeg_bin=self.ffmpeg_bin)
            crop_success = cropper.crop_video_vertical(dst_path, vert_path)
            if crop_success and vert_path.exists() and vert_path.stat().st_size > 0:
                vert_path.replace(dst_path)
                logger.info("Successfully cropped %s to 9:16 vertical active speaker", dst_path.name)
            else:
                logger.warning("Active speaker cropping skipped or failed for %s, keeping original ratio", dst_path.name)

        # 10. Optional: Dynamic karaoke subtitles
        if add_subtitles and transcript_words:
            ass_path = dst_path.with_name(f"{dst_path.stem}_sub.ass")
            sub_video_path = dst_path.with_name(f"{dst_path.stem}_subtitled{dst_path.suffix}")
            logger.info(
                "Burning karaoke subtitles onto %s (clip: %.2fs -> %.2fs, words: %d)",
                dst_path.name, start_time, end_time, len(transcript_words)
            )
            ass_ok = generate_karaoke_ass(
                transcript_words=transcript_words,
                clip_start=start_time,
                clip_end=end_time,
                output_ass_path=ass_path,
                play_res_x=1080 if aspect_ratio == "9:16" else 1920,
                play_res_y=1920 if aspect_ratio == "9:16" else 1080,
            )
            if ass_ok:
                burn_ok = burn_subtitles_ffmpeg(
                    video_path=dst_path,
                    ass_path=ass_path,
                    output_path=sub_video_path,
                    ffmpeg_bin=self.ffmpeg_bin,
                )
                if burn_ok and sub_video_path.exists() and sub_video_path.stat().st_size > 0:
                    sub_video_path.replace(dst_path)
                    logger.info("Karaoke subtitles burned successfully onto %s", dst_path.name)
                else:
                    logger.warning("burn_subtitles_ffmpeg failed for %s", dst_path.name)
            else:
                logger.warning("generate_karaoke_ass generated no subtitles for clip interval [%.2f, %.2f]", start_time, end_time)
            ass_path.unlink(missing_ok=True)
        else:
            logger.info(
                "Subtitle burning skipped: add_subtitles=%s, transcript_words count=%d",
                add_subtitles, len(transcript_words) if transcript_words else 0
            )

        final_file_size = dst_path.stat().st_size if dst_path.exists() else file_size

        return ClipExportResult(
            candidate_id=candidate_id,
            rank=rank,
            source_video=str(src_path),
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            output_path=str(dst_path),
            status="success",
            error_message=None,
            file_size_bytes=final_file_size,
            render_mode=active_mode,
            execution_time_sec=time.time() - t0,
        )

    @staticmethod
    def canonical_candidate_sort_key(cand: MTERCandidate) -> Tuple[float, float, float, float, str]:
        """
        Authoritative canonical ranking rule for MTER highlight candidates:
        1. composite_score / confidence_estimate (descending)
        2. temporal_agreement_iou (descending)
        3. duration proximity to target duration 30s (descending)
        4. proposed_start (deterministic tie-breaker: descending -start)
        5. candidate_id (deterministic tie-breaker: ascending)
        """
        conf = getattr(cand, "confidence_estimate", 0.0) or 0.0
        iou = getattr(cand, "temporal_agreement_iou", 0.0) or 0.0
        dur = getattr(cand, "duration", 0.0) or 0.0
        start = getattr(cand, "proposed_start", 0.0) or 0.0
        cid = getattr(cand, "candidate_id", "") or ""
        return (
            round(float(conf), 5),
            round(float(iou), 5),
            round(-abs(float(dur) - 30.0), 5),
            round(-float(start), 5),
            str(cid),
        )

    def render_highlights(
        self,
        source_video: Union[str, Path],
        candidates: List[MTERCandidate],
        output_dir: Union[str, Path],
        top_k: Optional[int] = None,
        video_duration: Optional[float] = None,
        mode: Optional[Literal["accurate", "fast"]] = None,
        prefix: str = "highlight",
        aspect_ratio: str = "16:9",
        add_subtitles: bool = False,
        transcript_words: Optional[List[Dict[str, Any]]] = None,
    ) -> ClipsManifest:
        """
        Renders physical MP4 highlight clips for a list of MTER candidates.
        
        Writes exported video files to {output_dir}/clips/ and creates {output_dir}/clips/manifest.json.
        """
        src_path = Path(source_video).resolve()
        out_base = Path(output_dir).resolve()
        clips_dir = out_base / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        sorted_candidates = sorted(candidates, key=canonical_candidate_sort_key, reverse=True)
        selected_candidates = sorted_candidates[:top_k] if top_k is not None else sorted_candidates
        total_requested = len(selected_candidates)
        active_mode = mode or self.default_mode

        exported_results: List[ClipExportResult] = []
        success_count = 0

        for idx, cand in enumerate(selected_candidates, start=1):
            clip_filename = f"{prefix}_{idx:02d}_{cand.proposed_start:.2f}_{cand.proposed_end:.2f}.mp4"
            clip_path = clips_dir / clip_filename

            res = self.render_clip(
                source_video=src_path,
                start_time=cand.proposed_start,
                end_time=cand.proposed_end,
                output_path=clip_path,
                candidate_id=cand.candidate_id,
                rank=idx,
                video_duration=video_duration,
                mode=active_mode,
                aspect_ratio=aspect_ratio,
                add_subtitles=add_subtitles,
                transcript_words=transcript_words,
            )
            res.score = round(float(cand.confidence_estimate), 4)
            if cand.boundary_refinement is not None:
                rf = cand.boundary_refinement
                res.boundary_refinement = rf
                res.original_start = round(float(rf.original_start), 3)
                res.original_end = round(float(rf.original_end), 3)
                res.original_duration = round(float(rf.original_duration), 3)
                res.refinement_status = "refined" if rf.refined else "not_refined"
                res.fallback_reason = rf.fallback_reason
            else:
                res.original_start = round(float(cand.proposed_start), 3)
                res.original_end = round(float(cand.proposed_end), 3)
                res.original_duration = round(float(cand.duration), 3)
                res.refinement_status = "not_refined"
                res.fallback_reason = "Boundary refinement was not applied"

            exported_results.append(res)
            if res.status == "success":
                success_count += 1

        manifest = ClipsManifest(
            run_id=out_base.name,
            source_video=str(src_path),
            total_candidates_requested=total_requested,
            total_clips_exported=success_count,
            render_mode=active_mode,
            clips=exported_results,
        )

        # Write manifest.json inside clips directory
        manifest_file = clips_dir / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))

        return manifest


# Module-level export for ranking consistency
canonical_candidate_sort_key = VideoRenderer.canonical_candidate_sort_key
