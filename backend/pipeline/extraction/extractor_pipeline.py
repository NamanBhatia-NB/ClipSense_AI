"""
ClipSense Multimodal Extraction Pipeline Coordinator (W2)

Runs the four extraction stages in sequence with caching and determinism:
1. Transcript Extraction -> transcript.json
2. Frame Extraction -> frames/ and frames.json
3. Prosody Extraction -> prosody.json
4. Conversation Extraction -> conversation.json
5. Manifest Generation -> manifest.json

Prints the required status indicators:
[✓] Transcript extraction
[✓] Frame extraction
[✓] Prosody extraction
[✓] Conversation extraction
And prints the generated artifact paths.
"""

import datetime
import json
import logging
import os
import pathlib
import sys
import uuid
from typing import Any, Dict, Optional

# Ensure backend root on path if invoked directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Support UTF-8 output on Windows console
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.config import config
from pipeline.extraction.transcript_extractor import TranscriptExtractor
from pipeline.extraction.frame_extractor import FrameExtractor
from pipeline.extraction.prosody_extractor import ProsodyExtractor
from pipeline.extraction.conversation_extractor import ConversationExtractor

logger = logging.getLogger("clipsense.pipeline.extraction")


class MultimodalExtractionPipeline:
    """Coordinates and executes the W2 extraction pipeline."""

    def __init__(
        self,
        transcript_extractor: Optional[TranscriptExtractor] = None,
        frame_extractor: Optional[FrameExtractor] = None,
        prosody_extractor: Optional[ProsodyExtractor] = None,
        conversation_extractor: Optional[ConversationExtractor] = None,
    ):
        self.transcript_extractor = transcript_extractor or TranscriptExtractor()
        self.frame_extractor = frame_extractor or FrameExtractor()
        self.prosody_extractor = prosody_extractor or ProsodyExtractor()
        self.conversation_extractor = conversation_extractor or ConversationExtractor()

    def run(
        self,
        video_path: str,
        run_id: Optional[str] = None,
        runs_base_dir: Optional[str] = None,
        force_rerun: bool = False,
    ) -> Dict[str, str]:
        """
        Execute full extraction pipeline on a video file.
        Returns a dictionary of generated artifact file paths.
        """
        video_p = pathlib.Path(video_path).resolve()
        if not video_p.exists():
            raise FileNotFoundError(f"Input video does not exist: {video_path}")

        run_id = run_id or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        base_dir = pathlib.Path(runs_base_dir or "runs").resolve()
        run_dir = base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        frames_dir = run_dir / "frames"
        transcript_json = run_dir / "transcript.json"
        frames_json = run_dir / "frames.json"
        prosody_json = run_dir / "prosody.json"
        conversation_json = run_dir / "conversation.json"
        manifest_json = run_dir / "manifest.json"

        # Check cache validity if force_rerun is False
        effective_force_rerun = force_rerun
        all_artifacts_exist = (
            transcript_json.exists()
            and frames_json.exists()
            and prosody_json.exists()
            and conversation_json.exists()
            and manifest_json.exists()
        )

        if not effective_force_rerun and all_artifacts_exist:
            try:
                with open(manifest_json, "r", encoding="utf-8") as f:
                    old_manifest = json.load(f)
                old_cfg = old_manifest.get("configuration", {})
                current_model = getattr(self.transcript_extractor, "model_name", config.transcript.model_name)
                current_sample_fps = getattr(self.frame_extractor, "sample_fps", config.visual.sample_fps)
                current_win_sec = getattr(self.prosody_extractor, "window_length_sec", config.prosody.window_length_sec)
                current_hop_sec = getattr(self.prosody_extractor, "hop_length_sec", config.prosody.hop_length_sec)
                current_pause_thr = getattr(self.conversation_extractor, "turn_pause_threshold_sec", config.conversation.turn_pause_threshold_sec)

                source_matches = (
                    old_manifest.get("source_video") == str(video_p)
                    or old_manifest.get("video_filename") == video_p.name
                )
                config_matches = (
                    old_cfg.get("whisper_model") == current_model
                    and abs(float(old_cfg.get("frame_sampling_fps", 0)) - float(current_sample_fps)) < 1e-4
                    and abs(float(old_cfg.get("prosody_window_sec", 0)) - float(current_win_sec)) < 1e-4
                    and abs(float(old_cfg.get("prosody_hop_sec", 0)) - float(current_hop_sec)) < 1e-4
                    and abs(float(old_cfg.get("conversation_pause_threshold_sec", 0)) - float(current_pause_thr)) < 1e-4
                )

                if not (source_matches and config_matches):
                    logger.info("Source video or extraction configuration changed; invalidating cache.")
                    effective_force_rerun = True
            except Exception as err:
                logger.warning(f"Could not validate cached manifest ({err}); forcing rerun.")
                effective_force_rerun = True

        # 1. Transcript Extraction
        try:
            transcript_data = self.transcript_extractor.extract(
                media_path=str(video_p),
                output_json_path=str(transcript_json),
                force_rerun=effective_force_rerun,
            )
            print("[✓] Transcript extraction")
        except Exception as e:
            print(f"[✗] Transcript extraction failed: {e}")
            raise e

        # 2. Visual Data Extraction (Frames & Scenes)
        try:
            visual_data = self.frame_extractor.extract(
                video_path=str(video_p),
                frames_dir=str(frames_dir),
                output_json_path=str(frames_json),
                force_rerun=effective_force_rerun,
            )
            print("[✓] Visual data extraction")
        except Exception as e:
            print(f"[✗] Visual data extraction failed: {e}")
            raise e

        # 3. Audio Prosody Extraction
        try:
            video_duration = visual_data.duration if visual_data.duration > 0 else transcript_data.duration
            prosody_data = self.prosody_extractor.extract(
                media_path=str(video_p),
                output_json_path=str(prosody_json),
                transcript_data=transcript_data,
                total_duration=video_duration,
                force_rerun=effective_force_rerun,
            )
            print("[✓] Prosody extraction")
        except Exception as e:
            print(f"[✗] Prosody extraction failed: {e}")
            raise e

        # 4. Conversation Structure Extraction
        try:
            video_duration = visual_data.duration if visual_data.duration > 0 else transcript_data.duration
            conversation_data = self.conversation_extractor.extract(
                transcript_data=transcript_data,
                total_duration=video_duration,
                output_json_path=str(conversation_json),
                force_rerun=effective_force_rerun,
            )
            print("[✓] Conversation structure extraction")
        except Exception as e:
            print(f"[✗] Conversation structure extraction failed: {e}")
            raise e

        # 5. Temporal Range & Consistency Validation
        actual_video_duration = visual_data.duration if visual_data.duration > 0 else transcript_data.duration
        val_checks = self.validate_timestamps(
            actual_duration=actual_video_duration,
            transcript_data=transcript_data,
            visual_data=visual_data,
            prosody_data=prosody_data,
            conversation_data=conversation_data,
        )
        if not val_checks["all_passed"]:
            raise ValueError(f"Temporal consistency validation failed: {val_checks}")

        # Determine inference mode and fallback status
        strict_mode_active = bool(getattr(self.transcript_extractor, "strict_mode", False))
        fallback_used = bool(getattr(self.transcript_extractor, "last_fallback_used", False))
        
        if strict_mode_active and not fallback_used:
            inference_mode = "real_whisperx"
        elif fallback_used:
            inference_mode = "test_fallback"
        else:
            inference_mode = "standard"

        # 6. Deterministic Manifest Generation with Configuration Metadata
        manifest_data = {
            "run_id": run_id,
            "source_video": str(video_p),
            "video_filename": video_p.name,
            "created_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "inference_mode": inference_mode,
            "configuration": {
                "whisper_model": getattr(self.transcript_extractor, "model_name", config.transcript.model_name),
                "strict_mode": strict_mode_active,
                "fallback_used": fallback_used,
                "frame_sampling_fps": float(getattr(self.frame_extractor, "sample_fps", config.visual.sample_fps)),
                "prosody_window_sec": float(getattr(self.prosody_extractor, "window_length_sec", config.prosody.window_length_sec)),
                "prosody_hop_sec": float(getattr(self.prosody_extractor, "hop_length_sec", config.prosody.hop_length_sec)),
                "conversation_pause_threshold_sec": float(getattr(self.conversation_extractor, "turn_pause_threshold_sec", config.conversation.turn_pause_threshold_sec)),
                "source_video_duration_sec": float(actual_video_duration),
            },
            "temporal_validation": val_checks,
            "metrics": {
                "duration_sec": actual_video_duration,
                "word_count": len(transcript_data.words),
                "sampled_frames_count": len(visual_data.sampled_frames),
                "scene_count": len(visual_data.scene_boundaries),
                "prosody_windows_count": len(prosody_data.windows),
                "conversation_turns_count": len(conversation_data.turns),
                "distinct_speakers": conversation_data.num_speakers,
            },
            "artifacts": {
                "transcript": str(transcript_json),
                "frames_dir": str(frames_dir),
                "frames_metadata": str(frames_json),
                "prosody": str(prosody_json),
                "conversation": str(conversation_json),
            },
        }

        with open(manifest_json, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        artifacts = {
            "run_dir": str(run_dir),
            "transcript_json": str(transcript_json),
            "frames_dir": str(frames_dir),
            "frames_json": str(frames_json),
            "prosody_json": str(prosody_json),
            "conversation_json": str(conversation_json),
            "manifest_json": str(manifest_json),
        }

        # Print paths of all generated artifacts
        print("\nGenerated extraction artifacts:")
        for name, path in artifacts.items():
            print(f"  - {name}: {path}")

        return artifacts

    @staticmethod
    def validate_timestamps(
        actual_duration: float,
        transcript_data,
        visual_data,
        prosody_data,
        conversation_data,
    ) -> Dict[str, Any]:
        """
        Validate that all extracted modalities produce continuous timestamps
        strictly consistent with the actual video timeline.
        Enforces:
        - Finite numbers (no NaN/inf)
        - Non-quantized floating-point continuous values
        - 0 <= start <= end <= actual_duration + container_tolerance
        - Monotonic ordering
        """
        import math
        checks: Dict[str, Any] = {}
        tolerance = 1.0  # seconds tolerance for audio/video container padding

        # 1. Transcript checks
        if transcript_data.words:
            t_min = float(transcript_data.words[0].start)
            t_max = float(transcript_data.words[-1].end)
            checks["transcript_finite"] = bool(all(
                math.isfinite(w.start) and math.isfinite(w.end) for w in transcript_data.words
            ))
            checks["transcript_ordered"] = bool(all(
                w.start <= w.end for w in transcript_data.words
            ))
            checks["transcript_continuous_float"] = bool(all(
                isinstance(w.start, float) and isinstance(w.end, float) for w in transcript_data.words
            ))
            checks["transcript_in_bounds"] = bool(
                0.0 <= t_min <= t_max <= actual_duration + tolerance
                and all(0.0 <= w.start <= w.end <= actual_duration + tolerance for w in transcript_data.words)
            )
            checks["transcript_monotonic"] = bool(all(
                transcript_data.words[i].start <= transcript_data.words[i + 1].start
                for i in range(len(transcript_data.words) - 1)
            ))
            checks["transcript_time_range"] = [t_min, t_max]
        else:
            checks["transcript_finite"] = True
            checks["transcript_ordered"] = True
            checks["transcript_continuous_float"] = True
            checks["transcript_in_bounds"] = True
            checks["transcript_monotonic"] = True
            checks["transcript_time_range"] = [0.0, 0.0]

        # 2. Visual frames checks
        if visual_data.sampled_frames:
            f_min = float(visual_data.sampled_frames[0].timestamp)
            f_max = float(visual_data.sampled_frames[-1].timestamp)
            checks["frames_finite"] = bool(all(
                math.isfinite(f.timestamp) for f in visual_data.sampled_frames
            ))
            checks["frames_continuous_float"] = bool(all(
                isinstance(f.timestamp, float) for f in visual_data.sampled_frames
            ))
            checks["frames_in_bounds"] = bool(
                0.0 <= f_min <= f_max <= actual_duration + tolerance
                and all(0.0 <= f.timestamp <= actual_duration + tolerance for f in visual_data.sampled_frames)
            )
            checks["frames_monotonic"] = bool(all(
                visual_data.sampled_frames[i].timestamp <= visual_data.sampled_frames[i + 1].timestamp
                for i in range(len(visual_data.sampled_frames) - 1)
            ))
            checks["frames_time_range"] = [f_min, f_max]
        else:
            checks["frames_finite"] = True
            checks["frames_continuous_float"] = True
            checks["frames_in_bounds"] = True
            checks["frames_monotonic"] = True
            checks["frames_time_range"] = [0.0, 0.0]

        # 3. Prosody windows checks
        if prosody_data.windows:
            p_min = float(prosody_data.windows[0].start_time)
            p_max = float(prosody_data.windows[-1].end_time)
            checks["prosody_finite"] = bool(all(
                math.isfinite(w.start_time) and math.isfinite(w.end_time) for w in prosody_data.windows
            ))
            checks["prosody_ordered"] = bool(all(
                w.start_time <= w.end_time for w in prosody_data.windows
            ))
            checks["prosody_continuous_float"] = bool(all(
                isinstance(w.start_time, float) and isinstance(w.end_time, float) for w in prosody_data.windows
            ))
            checks["prosody_in_bounds"] = bool(
                0.0 <= p_min <= p_max <= actual_duration + tolerance
                and all(0.0 <= w.start_time <= w.end_time <= actual_duration + tolerance for w in prosody_data.windows)
            )
            checks["prosody_monotonic"] = bool(all(
                prosody_data.windows[i].start_time <= prosody_data.windows[i + 1].start_time
                for i in range(len(prosody_data.windows) - 1)
            ))
            checks["prosody_time_range"] = [p_min, p_max]
        else:
            checks["prosody_finite"] = True
            checks["prosody_ordered"] = True
            checks["prosody_continuous_float"] = True
            checks["prosody_in_bounds"] = True
            checks["prosody_monotonic"] = True
            checks["prosody_time_range"] = [0.0, 0.0]

        # 4. Conversation turns checks
        if conversation_data.turns:
            c_min = float(conversation_data.turns[0].start_time)
            c_max = float(conversation_data.turns[-1].end_time)
            checks["conversation_finite"] = bool(all(
                math.isfinite(t.start_time) and math.isfinite(t.end_time) for t in conversation_data.turns
            ))
            checks["conversation_ordered"] = bool(all(
                t.start_time <= t.end_time for t in conversation_data.turns
            ))
            checks["conversation_continuous_float"] = bool(all(
                isinstance(t.start_time, float) and isinstance(t.end_time, float) for t in conversation_data.turns
            ))
            checks["conversation_in_bounds"] = bool(
                0.0 <= c_min <= c_max <= actual_duration + tolerance
                and all(0.0 <= t.start_time <= t.end_time <= actual_duration + tolerance for t in conversation_data.turns)
            )
            checks["conversation_monotonic"] = bool(all(
                conversation_data.turns[i].start_time <= conversation_data.turns[i + 1].start_time
                for i in range(len(conversation_data.turns) - 1)
            ))
            checks["conversation_time_range"] = [c_min, c_max]
        else:
            checks["conversation_finite"] = True
            checks["conversation_ordered"] = True
            checks["conversation_continuous_float"] = True
            checks["conversation_in_bounds"] = True
            checks["conversation_monotonic"] = True
            checks["conversation_time_range"] = [0.0, 0.0]

        all_passed = bool(
            checks["transcript_finite"]
            and checks["transcript_ordered"]
            and checks["transcript_continuous_float"]
            and checks["transcript_in_bounds"]
            and checks["transcript_monotonic"]
            and checks["frames_finite"]
            and checks["frames_continuous_float"]
            and checks["frames_in_bounds"]
            and checks["frames_monotonic"]
            and checks["prosody_finite"]
            and checks["prosody_ordered"]
            and checks["prosody_continuous_float"]
            and checks["prosody_in_bounds"]
            and checks["prosody_monotonic"]
            and checks["conversation_finite"]
            and checks["conversation_ordered"]
            and checks["conversation_continuous_float"]
            and checks["conversation_in_bounds"]
            and checks["conversation_monotonic"]
        )
        checks["all_passed"] = all_passed
        return checks


def main():
    """CLI entrypoint for standalone test execution."""
    import argparse
    parser = argparse.ArgumentParser(description="ClipSense W2 Multimodal Extraction Pipeline")
    parser.add_argument("video_path", type=str, help="Path to input video or audio file")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run identifier")
    parser.add_argument("--output-dir", type=str, default="runs", help="Base directory for runs")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore cache and recompute all extractions")
    args = parser.parse_args()

    pipeline = MultimodalExtractionPipeline()
    pipeline.run(
        video_path=args.video_path,
        run_id=args.run_id,
        runs_base_dir=args.output_dir,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()
