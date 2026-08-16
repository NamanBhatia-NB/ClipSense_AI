"""
ClipSense Storage & Workspace Management

Provides clean abstraction for:
- Local temporary run workspace lifecycle (creation & cleanup)
- S3 video download & clip upload via boto3
"""

import os
import pathlib
import shutil
import uuid
from typing import Optional
import boto3
from app.config import config


class WorkspaceManager:
    """Manages ephemeral disk workspaces for a single video processing pipeline run."""

    def __init__(self, run_id: Optional[str] = None, base_dir: Optional[str] = None):
        self.run_id = run_id or str(uuid.uuid4())
        base = pathlib.Path(base_dir or config.temp_dir_base)
        self.workspace_dir = base / self.run_id
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Predefined standard subdirectories
        self.media_dir = self.workspace_dir / "media"
        self.frames_dir = self.workspace_dir / "frames"
        self.clips_dir = self.workspace_dir / "clips"
        self.subtitles_dir = self.workspace_dir / "subtitles"

        for d in [self.media_dir, self.frames_dir, self.clips_dir, self.subtitles_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_input_video_path(self, filename: str = "input.mp4") -> pathlib.Path:
        return self.media_dir / filename

    def get_audio_path(self, filename: str = "audio.wav") -> pathlib.Path:
        return self.media_dir / filename

    def cleanup(self):
        """Remove the entire workspace directory tree safely."""
        if self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class S3StorageClient:
    """S3 storage interface for downloading source media and uploading generated clips."""

    def __init__(self, bucket_name: Optional[str] = None):
        self.bucket_name = bucket_name or config.s3_bucket_name
        self.client = boto3.client("s3")

    def download_video(self, s3_key: str, local_path: pathlib.Path) -> pathlib.Path:
        """Download remote video from S3 to local filesystem."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket_name, s3_key, str(local_path))
        return local_path

    def upload_clip(self, local_path: pathlib.Path, output_s3_key: str) -> str:
        """Upload a rendered clip to S3 and return the S3 key."""
        self.client.upload_file(str(local_path), self.bucket_name, output_s3_key)
        return output_s3_key
