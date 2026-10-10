"""
ClipSense Task Manager (Week 9)

Manages asynchronous video processing tasks, thread-pool execution,
progress tracking, output directories, safe file streaming, and path traversal protection.
"""

from concurrent.futures import ThreadPoolExecutor
import logging
import os
import pathlib
import re
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from api.schemas import (
    BoundaryRefinementSummary,
    HighlightClipInfo,
    HighlightItem,
    ModalityEvidenceSummary,
    TaskHighlightsResponse,
    TaskStatusResponse,
)

logger = logging.getLogger("clipsense.api.tasks")

# Strict identifier and filename regexes to prevent path traversal or injection
TASK_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{8,64}$")
SAFE_FILENAME_REGEX = re.compile(r"^[a-zA-Z0-9_\-\. ]+\.(mp4|MP4)$")

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
MAX_UPLOAD_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB


class TaskManager:
    """Thread-safe background task manager for ClipSense pipeline execution."""

    def __init__(self, base_tasks_dir: Optional[pathlib.Path] = None, max_workers: int = 2):
        backend_dir = pathlib.Path(__file__).parent.parent.resolve()
        self.base_tasks_dir = base_tasks_dir or (backend_dir / "runs" / "tasks")
        self.base_tasks_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="clipsense-worker")
        self._lock = threading.Lock()
        self._tasks: Dict[str, Dict[str, Any]] = {}

    def is_valid_task_id(self, task_id: str) -> bool:
        """Validate task identifier to prevent arbitrary filesystem traversal."""
        if not task_id or not isinstance(task_id, str):
            return False
        return bool(TASK_ID_REGEX.match(task_id))

    def is_valid_clip_filename(self, filename: str) -> bool:
        """Validate clip filename to ensure it is a safe MP4 file without directory components."""
        if not filename or not isinstance(filename, str):
            return False
        if "/" in filename or "\\" in filename or ".." in filename:
            return False
        return bool(SAFE_FILENAME_REGEX.match(filename))

    def create_task(self, video_filename: str) -> str:
        """Initialize a new task record and create its isolated workspace directory."""
        raw_id = f"task_{uuid.uuid4().hex[:16]}"
        task_dir = (self.base_tasks_dir / raw_id).resolve()
        clips_dir = task_dir / "clips"
        task_dir.mkdir(parents=True, exist_ok=True)
        clips_dir.mkdir(parents=True, exist_ok=True)

        now = time.time()
        record = {
            "task_id": raw_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "message": "Task queued for processing",
            "error": None,
            "created_at": now,
            "completed_at": None,
            "video_filename": video_filename,
            "video_path": None,
            "task_dir": str(task_dir),
            "clips_dir": str(clips_dir),
            "total_duration": 0.0,
            "mter_output": None,
            "clips_manifest": None,
            "execution_time_sec": 0.0,
        }

        with self._lock:
            self._tasks[raw_id] = record

        return raw_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task record if it exists and task_id is valid."""
        if not self.is_valid_task_id(task_id):
            return None
        with self._lock:
            return self._tasks.get(task_id)

    def update_progress(self, task_id: str, stage: str, progress: float, message: str) -> None:
        """Update current task status and stage safely."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task["status"] = "processing"
            task["stage"] = stage
            task["progress"] = min(1.0, max(0.0, float(progress)))
            task["message"] = message

    def set_completed(
        self,
        task_id: str,
        total_duration: float,
        mter_output: Any,
        clips_manifest: Any,
        execution_time_sec: float,
    ) -> None:
        """Mark task as successfully completed with final artifacts."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task["status"] = "completed"
            task["stage"] = "completed"
            task["progress"] = 1.0
            task["message"] = "Processing completed successfully"
            task["completed_at"] = time.time()
            task["total_duration"] = total_duration
            task["mter_output"] = mter_output
            task["clips_manifest"] = clips_manifest
            task["execution_time_sec"] = execution_time_sec

    def set_failed(self, task_id: str, error_message: str) -> None:
        """Mark task as failed with a sanitized error message."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task["status"] = "failed"
            task["stage"] = "failed"
            task["error"] = error_message
            task["message"] = f"Processing failed: {error_message}"
            task["completed_at"] = time.time()

    def get_status_response(self, task_id: str) -> Optional[TaskStatusResponse]:
        """Generate structured TaskStatusResponse schema for a task."""
        task = self.get_task(task_id)
        if not task:
            return None
        return TaskStatusResponse(
            task_id=task["task_id"],
            status=task["status"],
            stage=task["stage"],
            progress=task["progress"],
            message=task["message"],
            error=task["error"],
            created_at=task["created_at"],
            completed_at=task["completed_at"],
        )

    def get_highlights_response(self, task_id: str) -> Optional[TaskHighlightsResponse]:
        """
        Generate structured TaskHighlightsResponse with ranked candidates,
        modality evidence metrics, and clip stream/download links.
        """
        task = self.get_task(task_id)
        if not task:
            return None

        mter_output = task.get("mter_output")
        clips_manifest = task.get("clips_manifest")

        candidates = mter_output.candidates if mter_output else []
        manifest_clips = {c.rank: c for c in clips_manifest.clips} if clips_manifest else {}

        highlight_items: List[HighlightItem] = []
        for idx, cand in enumerate(candidates, start=1):
            sm = cand.support_metrics
            ev_summary = ModalityEvidenceSummary(
                transcript_support=round(sm.transcript_support, 3) if sm else 0.0,
                conversation_support=round(sm.conversation_support, 3) if sm else 0.0,
                visual_support=round(sm.visual_support, 3) if sm else 0.0,
                prosody_support=round(sm.prosody_support, 3) if sm else 0.0,
                linguistic_boundary_support=round(cand.linguistic_boundary_support, 3),
                strong_modalities=cand.strong_support_modalities,
                moderate_modalities=cand.moderate_support_modalities,
                weak_modalities=cand.weak_support_modalities,
            )

            clip_meta: Optional[HighlightClipInfo] = None
            if idx in manifest_clips:
                c = manifest_clips[idx]
                fname = pathlib.Path(c.output_path).name if c.output_path else f"highlight_{idx}.mp4"
                clip_meta = HighlightClipInfo(
                    filename=fname,
                    stream_url=f"/api/clips/{task_id}/{fname}",
                    download_url=f"/api/clips/{task_id}/{fname}?download=true",
                    file_size_bytes=c.file_size_bytes,
                    status=c.status,
                )

            refine_summary: Optional[BoundaryRefinementSummary] = None
            if cand.boundary_refinement is not None:
                rf = cand.boundary_refinement
                refine_summary = BoundaryRefinementSummary(
                    refined=rf.refined,
                    original_start=round(rf.original_start, 3),
                    original_end=round(rf.original_end, 3),
                    original_duration=round(rf.original_duration, 3),
                    refined_start=round(rf.refined_start, 3),
                    refined_end=round(rf.refined_end, 3),
                    refined_duration=round(rf.refined_duration, 3),
                    start_delta_sec=round(rf.start_delta_sec, 3),
                    end_delta_sec=round(rf.end_delta_sec, 3),
                    start_evidence_type=rf.start_evidence_type,
                    end_evidence_type=rf.end_evidence_type,
                    signals_used=rf.signals_used,
                    rationale=rf.rationale,
                    fallback_reason=rf.fallback_reason,
                )

            highlight_items.append(
                HighlightItem(
                    candidate_id=cand.candidate_id,
                    rank=idx,
                    start_time=round(cand.proposed_start, 2),
                    end_time=round(cand.proposed_end, 2),
                    duration=round(cand.duration, 2),
                    confidence_estimate=round(cand.confidence_estimate, 3),
                    temporal_agreement_iou=round(cand.temporal_agreement_iou, 3),
                    decision_summary=cand.decision_summary,
                    modality_evidence=ev_summary,
                    clip=clip_meta,
                    boundary_refinement=refine_summary,
                )
            )

        return TaskHighlightsResponse(
            task_id=task_id,
            status=task["status"],
            video_filename=task["video_filename"],
            total_duration=round(task.get("total_duration", 0.0), 2),
            total_candidates=len(highlight_items),
            highlights=highlight_items,
            execution_time_sec=task.get("execution_time_sec"),
        )

    def safe_get_clip_path(self, task_id: str, filename: str) -> Optional[pathlib.Path]:
        """
        Verify that the requested clip file strictly belongs to the task's output directory
        and prevents any directory traversal attacks (.. or slashes).
        """
        if not self.is_valid_task_id(task_id) or not self.is_valid_clip_filename(filename):
            return None

        task = self.get_task(task_id)
        if not task:
            return None

        task_dir = pathlib.Path(task["task_dir"]).resolve()
        clips_dir = (task_dir / "clips").resolve()
        target_path = (clips_dir / filename).resolve()

        # Strict containment check
        try:
            target_path.relative_to(clips_dir)
        except ValueError:
            logger.warning(f"Path traversal attempt rejected: task={task_id}, file={filename}")
            return None

        if not target_path.exists() or not target_path.is_file():
            return None

        return target_path

    def submit_processing(
        self,
        task_id: str,
        video_path: str,
        top_k: int = 3,
        use_cache: bool = True,
        mock_llm: bool = False,
        render_mode: str = "accurate",
    ) -> None:
        """Submit pipeline execution to background worker pool."""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["video_path"] = video_path

        self.executor.submit(
            self._execute_pipeline,
            task_id,
            video_path,
            top_k,
            use_cache,
            mock_llm,
            render_mode,
            refine_boundaries,
        )

    def _execute_pipeline(
        self,
        task_id: str,
        video_path: str,
        top_k: int,
        use_cache: bool,
        mock_llm: bool,
        render_mode: str,
    ) -> None:
        """Execute the real ClipSense research pipeline in worker thread."""
        task = self.get_task(task_id)
        if not task:
            return

        task_dir = pathlib.Path(task["task_dir"])
        t_start = time.time()
        logger.info(f"Starting processing for task {task_id} with video {video_path}")

        try:
            self.update_progress(task_id, "extraction", 0.10, "Extracting multimodal features...")

            from demo.run_demo import run_demonstration

            def progress_cb(stage: str, progress: float, message: str) -> None:
                self.update_progress(task_id, stage, progress, message)

            result = run_demonstration(
                video_path=video_path,
                output_dir=str(task_dir),
                use_cache=use_cache,
                mock_llm=mock_llm,
                top_k=top_k,
                render_clips=True,
                render_mode=render_mode,
                refine_boundaries=refine_boundaries,
                progress_callback=progress_cb,
            )

            total_elapsed = time.time() - t_start
            mter_out = result.get("mter_output") if isinstance(result, dict) else None
            clips_mf = result.get("clips_manifest") if isinstance(result, dict) else None
            tot_dur = result.get("total_duration", 0.0) if isinstance(result, dict) else 0.0

            self.set_completed(
                task_id=task_id,
                total_duration=tot_dur,
                mter_output=mter_out,
                clips_manifest=clips_mf,
                execution_time_sec=total_elapsed,
            )
            logger.info(f"Task {task_id} completed successfully in {total_elapsed:.2f}s")

        except Exception as e:
            logger.exception(f"Processing failed for task {task_id}: {e}")
            # Sanitize error message to avoid leaking internal filepaths or secrets
            clean_err = str(e)
            if "backend" in clean_err.lower() or "\\" in clean_err or "/" in clean_err:
                clean_err = "Internal pipeline execution error during video analysis"
            self.set_failed(task_id, clean_err)
