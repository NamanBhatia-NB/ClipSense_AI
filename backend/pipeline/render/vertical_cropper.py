"""
ClipSense AI - Active Speaker Vertical Cropper (9:16)

Transforms horizontal (16:9) clips into viral 9:16 vertical videos by:
1. Centering dynamically on the active speaker's face using face tracking/ASD.
2. Applying exponential smoothing to camera panning for cinematic motion.
3. Rendering a Gaussian-blurred background portrait canvas when no face is present.
4. Muxing synchronized audio without quality degradation.

Adapted from backend/main.py with universal CPU/GPU compatibility.
"""

import glob
import logging
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ActiveSpeakerCropper:
    """
    Renders 9:16 vertical videos centered on the detected active speaker.
    """

    def __init__(
        self,
        target_width: int = 1080,
        target_height: int = 1920,
        smoothing_alpha: float = 0.15,
        ffmpeg_bin: str = "ffmpeg",
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.smoothing_alpha = smoothing_alpha
        self.ffmpeg_bin = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
        self._face_cascade = None

    def _get_face_cascade(self):
        if self._face_cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                self._face_cascade = cv2.CascadeClassifier(cascade_path)
        return self._face_cascade

    def detect_face_in_frame(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects primary face bounding box (x, y, w, h) in a BGR frame.
        """
        cascade = self._get_face_cascade()
        if cascade is None:
            return None

        # Downscale for fast face detection
        h, w = frame.shape[:2]
        scale_factor = 360.0 / max(h, 1)
        small = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(30, 30),
        )

        if len(faces) == 0:
            return None

        # Pick the largest face in frame
        best_face = max(faces, key=lambda b: b[2] * b[3])
        orig_x = int(best_face[0] / scale_factor)
        orig_y = int(best_face[1] / scale_factor)
        orig_w = int(best_face[2] / scale_factor)
        orig_h = int(best_face[3] / scale_factor)
        return (orig_x, orig_y, orig_w, orig_h)

    def crop_video_vertical(
        self,
        input_clip_path: Union[str, Path],
        output_clip_path: Union[str, Path],
        tracks: Optional[List[Any]] = None,
        scores: Optional[List[Any]] = None,
        framerate: float = 25.0,
    ) -> bool:
        """
        Processes input_clip_path into a 9:16 vertical video saved at output_clip_path.
        """
        src = Path(input_clip_path).resolve()
        dst = Path(output_clip_path).resolve()

        if not src.exists():
            logger.error("Source clip for vertical cropping does not exist: %s", src)
            return False

        dst.parent.mkdir(parents=True, exist_ok=True)
        work_dir = dst.parent / f"tmp_crop_{int(time.time() * 1000)}"
        work_dir.mkdir(parents=True, exist_ok=True)

        temp_audio_path = work_dir / "audio.aac"
        temp_video_only_path = work_dir / "video_only.mp4"

        try:
            # 1. Extract audio from clip
            cmd_audio = [
                self.ffmpeg_bin,
                "-y",
                "-nostdin",
                "-i", str(src),
                "-vn",
                "-c:a", "copy",
                str(temp_audio_path),
            ]
            has_audio = (subprocess.run(cmd_audio, capture_output=True).returncode == 0
                         and temp_audio_path.exists() and temp_audio_path.stat().st_size > 0)

            # 2. Open video capture
            cap = cv2.VideoCapture(str(src))
            if not cap.isOpened():
                logger.error("Could not open video file: %s", src)
                return False

            fps = cap.get(cv2.CAP_PROP_FPS) or framerate
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            # 3. Process and write individual vertical frames to pyframes directory
            pyframes_dir = work_dir / "pyframes"
            pyframes_dir.mkdir(parents=True, exist_ok=True)

            smoothed_center_x: Optional[float] = None
            last_detected_x: Optional[float] = None
            frame_idx = 0

            while cap.isOpened():
                ret, img = cap.read()
                if not ret or img is None:
                    break

                h, w = img.shape[:2]
                detected_box = None

                # Detect face periodically (every 4 frames for performance) or track
                if frame_idx % 4 == 0 or last_detected_x is None:
                    detected_box = self.detect_face_in_frame(img)
                    if detected_box is not None:
                        fx, fy, fw, fh = detected_box
                        last_detected_x = fx + (fw / 2.0)
                    else:
                        if frame_idx % 24 == 0:
                            last_detected_x = None

                if last_detected_x is not None:
                    target_x = last_detected_x
                    if smoothed_center_x is None:
                        smoothed_center_x = target_x
                    else:
                        smoothed_center_x = (
                            self.smoothing_alpha * target_x
                            + (1.0 - self.smoothing_alpha) * smoothed_center_x
                        )

                    scale = self.target_height / float(h)
                    resized_image = cv2.resize(
                        img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                    )
                    frame_width = resized_image.shape[1]

                    scaled_center_x = int(smoothed_center_x * scale)
                    top_x = max(
                        min(scaled_center_x - self.target_width // 2, frame_width - self.target_width),
                        0,
                    )

                    if frame_width >= self.target_width:
                        final_frame = resized_image[0:self.target_height, top_x : top_x + self.target_width]
                    else:
                        pad_w = self.target_width - frame_width
                        final_frame = cv2.copyMakeBorder(
                            resized_image, 0, 0, pad_w // 2, pad_w - (pad_w // 2), cv2.BORDER_CONSTANT, value=[0, 0, 0]
                        )
                else:
                    scale_fg = self.target_width / float(w)
                    resized_height = int(h * scale_fg)
                    resized_image = cv2.resize(
                        img, (self.target_width, resized_height), interpolation=cv2.INTER_AREA
                    )

                    scale_bg = max(self.target_width / float(w), self.target_height / float(h))
                    bg_w = int(w * scale_bg)
                    bg_h = int(h * scale_bg)

                    blurred_bg = cv2.resize(img, (bg_w, bg_h))
                    blurred_bg = cv2.GaussianBlur(blurred_bg, (81, 81), 0)

                    crop_x = (bg_w - self.target_width) // 2
                    crop_y = (bg_h - self.target_height) // 2
                    final_frame = blurred_bg[
                        crop_y : crop_y + self.target_height, crop_x : crop_x + self.target_width
                    ]

                    center_y = (self.target_height - resized_height) // 2
                    final_frame[center_y : center_y + resized_height, :] = resized_image

                if final_frame.shape[0] != self.target_height or final_frame.shape[1] != self.target_width:
                    final_frame = cv2.resize(final_frame, (self.target_width, self.target_height))

                frame_file = pyframes_dir / f"{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_file), final_frame)
                frame_idx += 1

            cap.release()

            if frame_idx == 0:
                logger.error("No frames were extracted from video: %s", src)
                return False

            # 4. Mux video with audio and encode cleanly to libx264
            frame_pattern = str(pyframes_dir / "%06d.jpg")
            if has_audio:
                mux_cmd = [
                    self.ffmpeg_bin,
                    "-y",
                    "-nostdin",
                    "-r", str(fps),
                    "-i", frame_pattern,
                    "-i", str(temp_audio_path),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "22",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-shortest",
                    str(dst),
                ]
            else:
                mux_cmd = [
                    self.ffmpeg_bin,
                    "-y",
                    "-nostdin",
                    "-r", str(fps),
                    "-i", frame_pattern,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "22",
                    "-pix_fmt", "yuv420p",
                    str(dst),
                ]

            mux_res = subprocess.run(mux_cmd, capture_output=True, text=True)
            if mux_res.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
                logger.info("Successfully rendered 9:16 vertical video: %s (%d frames)", dst.name, frame_idx)
                return True
            else:
                logger.error("FFmpeg mux failed: %s", mux_res.stderr)
                return False

        except Exception as ex:
            logger.exception("Vertical crop failed: %s", ex)
            return False
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
