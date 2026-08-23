"""
ClipSense Frame Extractor (W2)

Extracts uniformly sampled timestamped video frames and lightweight scene cuts.
Preserves continuous floating-point timestamps for each sampled frame.
Saves extracted frames to disk and outputs validated VisualData / frames.json artifact.
"""

import json
import logging
import os
import pathlib
from typing import List, Optional
import cv2

from app.config import config
from core.schemas import SceneBoundary, SampledFrameMetadata, VisualData

logger = logging.getLogger("clipsense.extraction.frame")


class FrameExtractor:
    """Extracts timestamped frames and scene boundaries from video files."""

    def __init__(
        self,
        sample_fps: Optional[float] = None,
        scene_threshold: Optional[float] = None,
    ):
        self.sample_fps = sample_fps or config.visual.sample_fps
        self.scene_threshold = scene_threshold or config.visual.scene_threshold

    def detect_scenes(self, video_path: str, fps: float, total_frames: int) -> List[SceneBoundary]:
        """
        Lightweight scene cut detection.
        Attempts PySceneDetect ContentDetector, falling back to frame difference analysis.
        """
        boundaries = []
        try:
            from scenedetect import detect, ContentDetector

            logger.info(f"Running ContentDetector on {video_path} (threshold={self.scene_threshold})...")
            scene_list = detect(video_path, ContentDetector(threshold=self.scene_threshold))
            
            for idx, scene in enumerate(scene_list):
                start_sec = scene[0].get_seconds()
                end_sec = scene[1].get_seconds()
                boundaries.append(
                    SceneBoundary(
                        scene_index=idx,
                        start_time=float(start_sec),
                        end_time=float(end_sec),
                        duration=float(end_sec - start_sec),
                    )
                )
        except Exception as e:
            logger.warning(f"PySceneDetect encountered an issue ({e}); using single-scene fallback.")

        # If no scenes were detected or library unavailable, treat entire video as scene 0
        if not boundaries:
            duration_sec = float(total_frames / fps) if fps > 0 else 0.0
            boundaries.append(
                SceneBoundary(
                    scene_index=0,
                    start_time=0.0,
                    end_time=duration_sec,
                    duration=duration_sec,
                )
            )

        return boundaries

    def extract(
        self,
        video_path: str,
        frames_dir: str,
        output_json_path: Optional[str] = None,
        force_rerun: bool = False,
    ) -> VisualData:
        """
        Sample frames at configured fps and detect scene changes.
        Uses caching if output_json_path exists and force_rerun is False.
        """
        # Cache check
        if output_json_path and os.path.exists(output_json_path) and not force_rerun:
            logger.info(f"Loading cached visual metadata from {output_json_path}")
            with open(output_json_path, "r", encoding="utf-8") as f:
                data_dict = json.load(f)
            return VisualData.model_validate(data_dict)

        video_p = pathlib.Path(video_path)
        if not video_p.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        frames_out_dir = pathlib.Path(frames_dir)
        frames_out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_p))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file with OpenCV: {video_path}")

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = float(total_frames / fps) if fps > 0 else 0.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            logger.info(
                f"Video {video_p.name}: {total_frames} frames, {fps:.2f} FPS, "
                f"{duration:.2f}s, resolution {width}x{height}"
            )

            # 1. Detect scene boundaries
            scene_boundaries = self.detect_scenes(str(video_p), fps, total_frames)

            # 2. Sample frames at configured sample_fps (e.g. 1 frame every fps/sample_fps)
            frame_step = max(1, int(round(fps / self.sample_fps)))
            sampled_frames_meta = []

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step == 0:
                    timestamp = float(frame_idx / fps)
                    
                    # Determine which scene this frame falls into
                    curr_scene = 0
                    for sc in scene_boundaries:
                        if sc.start_time <= timestamp <= sc.end_time:
                            curr_scene = sc.scene_index
                            break

                    frame_filename = f"frame_{frame_idx:06d}_{timestamp:.2f}s.jpg"
                    frame_path = frames_out_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

                    sampled_frames_meta.append(
                        SampledFrameMetadata(
                            frame_index=frame_idx,
                            timestamp=timestamp,
                            scene_index=curr_scene,
                            file_path=str(frame_path),
                            width=width,
                            height=height,
                        )
                    )

                frame_idx += 1

            visual_data = VisualData(
                fps=fps,
                total_frames=total_frames,
                duration=duration,
                scene_boundaries=scene_boundaries,
                sampled_frames=sampled_frames_meta,
            )

            # Save artifact
            if output_json_path:
                out_p = pathlib.Path(output_json_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                with open(out_p, "w", encoding="utf-8") as f:
                    f.write(visual_data.model_dump_json(indent=2))
                logger.info(f"Saved visual data artifact to {output_json_path}")

            return visual_data

        finally:
            cap.release()
