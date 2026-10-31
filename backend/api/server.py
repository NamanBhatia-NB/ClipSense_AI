"""
ClipSense AI - Local FastAPI Bridge (Week 9)

Integrates the ClipSense multimodal research pipeline with frontend applications.
Provides:
- GET /api/health: Backend readiness and FFmpeg capability check.
- POST /api/process: Multipart video upload and processing initiation.
- GET /api/status/{task_id}: Real-time status, stage, and error reporting.
- GET /api/highlights/{task_id}: Ranked candidates, modality evidence summaries, and clip links.
- GET /api/clips/{task_id}/{filename}: Safe streaming and downloading of exported MP4 clips.
"""

import json
import logging
import os
import pathlib
import sys
import time
from typing import Any, Dict, Optional

# Ensure backend root is on sys.path
backend_dir = pathlib.Path(__file__).parent.parent.resolve()
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from api.schemas import (
    HealthResponse,
    ProcessTaskResponse,
    TaskHighlightsResponse,
    TaskStatusResponse,
)
from api.task_manager import (
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_UPLOAD_SIZE_BYTES,
    TaskManager,
)
from app.config import config
from pipeline.render.video_renderer import VideoRenderer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("clipsense.api")

app = FastAPI(
    title="ClipSense AI Local API",
    description="Local FastAPI bridge connecting Next.js frontend with ClipSense multimodal research pipeline.",
    version="1.0.0",
)

# Configure CORS for local Next.js frontend and production Vercel apps
default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "https://clip-sense-ai.vercel.app",
]
env_origins = os.getenv("CLIPSENSE_CORS_ORIGINS")
cors_origins = [o.strip() for o in env_origins.split(",") if o.strip() and o.strip() != "*"] if env_origins else default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|.*\.vercel\.app)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize TaskManager singleton
tasks_dir = backend_dir / "runs" / "tasks"
task_manager = TaskManager(base_tasks_dir=tasks_dir, max_workers=2)


@app.get("/")
def root_info() -> Dict[str, Any]:
    """Root metadata providing API description and health pointers."""
    return {
        "service": "ClipSense AI Local Bridge",
        "version": "1.0.0",
        "health_check": "/api/health",
        "docs_url": "/docs",
    }


@app.get("/api/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """
    Check backend readiness and verify FFmpeg binary availability on system PATH.
    """
    ffmpeg_ok = VideoRenderer().is_available()
    return HealthResponse(
        status="ok",
        service="clipsense-api",
        version="1.0.0",
        ffmpeg_available=ffmpeg_ok,
        device="cpu",
        timestamp=time.time(),
    )


@app.post("/api/process", response_model=ProcessTaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_video(
    request: Request,
    file: Optional[UploadFile] = File(None),
    video_path: Optional[str] = Form(None),
    top_k: int = Form(3),
    use_cache: bool = Form(True),
    mock_llm: bool = Form(False),
    render_mode: str = Form("accurate"),
    refine_boundaries: bool = Form(True),
    aspect_ratio: str = Form("9:16"),
    add_subtitles: bool = Form(True),
    user_id: Optional[str] = Form(None),
    selection_mode: Optional[str] = Form(None),
    natural_acceptance_threshold: Optional[float] = Form(None),
    max_total_selected_duration_sec: Optional[float] = Form(None),
) -> ProcessTaskResponse:
    """
    Accept a supported video upload or local path, initiate asynchronous processing,
    and return an immediate task identifier.
    """
    content_type = request.headers.get("content-type", "")

    # Check for JSON request body if file and video_path are not in form
    if "application/json" in content_type and not file and not video_path:
        try:
            body = await request.json()
            video_path = body.get("video_path") or body.get("s3_key")
            top_k = int(body.get("top_k", top_k))
            use_cache = bool(body.get("use_cache", use_cache))
            mock_llm = bool(body.get("mock_llm", mock_llm))
            render_mode = str(body.get("render_mode", render_mode))
            refine_boundaries = bool(body.get("refine_boundaries", refine_boundaries))
            aspect_ratio = str(body.get("aspect_ratio", aspect_ratio))
            add_subtitles = bool(body.get("add_subtitles", add_subtitles))
            user_id = body.get("user_id") or user_id
            selection_mode = body.get("selection_mode") or selection_mode
            if "natural_acceptance_threshold" in body:
                natural_acceptance_threshold = float(body["natural_acceptance_threshold"])
            if "max_total_selected_duration_sec" in body:
                max_total_selected_duration_sec = float(body["max_total_selected_duration_sec"])
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Malformed JSON payload: {str(e)}",
            )

    # Validate Top-K bounds (strictly between 1 and 10 inclusive)
    if top_k < 1 or top_k > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Number of clips ('top_k') must be between 1 and 10 (inclusive). Received: {top_k}",
        )
    if render_mode not in ("accurate", "fast"):
        render_mode = "accurate"

    # Scenario A: Multipart file upload
    if file and file.filename:
        filename = pathlib.Path(file.filename).name
        ext = pathlib.Path(filename).suffix.lower()

        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported video format '{ext}'. Allowed: {sorted(list(ALLOWED_VIDEO_EXTENSIONS))}",
            )

        task_id = task_manager.create_task(video_filename=filename)
        task_dir = pathlib.Path(task_manager.get_task(task_id)["task_dir"])
        upload_dir = task_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest_path = upload_dir / filename

        # Stream save file safely while verifying maximum allowed upload size
        bytes_read = 0
        try:
            with open(dest_path, "wb") as f_out:
                while chunk := await file.read(1024 * 1024):  # 1MB chunks
                    bytes_read += len(chunk)
                    if bytes_read > MAX_UPLOAD_SIZE_BYTES:
                        dest_path.unlink(missing_ok=True)
                        task_manager.set_failed(task_id, "Upload size exceeded 500MB limit")
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail="Uploaded video exceeds the maximum allowed size (500 MB).",
                        )
                    f_out.write(chunk)
        except HTTPException:
            raise
        except Exception as e:
            dest_path.unlink(missing_ok=True)
            logger.exception(f"Failed to save uploaded file: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save uploaded file",
            )

        target_video_path = str(dest_path)

    # Scenario B: Local file path reference (e.g. sample video or server-side path)
    elif video_path:
        # Resolve path safely within workspace or absolute path
        clean_path = pathlib.Path(video_path).resolve()
        if not clean_path.exists() or not clean_path.is_file():
            # Also check relative to backend directory
            alt_path = (backend_dir / video_path).resolve()
            if alt_path.exists() and alt_path.is_file():
                clean_path = alt_path
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Input video file not found: {video_path}",
                )

        ext = clean_path.suffix.lower()
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported video format '{ext}'. Allowed: {sorted(list(ALLOWED_VIDEO_EXTENSIONS))}",
            )

        task_id = task_manager.create_task(video_filename=clean_path.name)
        target_video_path = str(clean_path)

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either a video file upload ('file') or a local video path ('video_path') must be provided.",
        )

    # Submit asynchronous execution to task manager
    task_manager.submit_processing(
        task_id=task_id,
        video_path=target_video_path,
        top_k=top_k,
        use_cache=use_cache,
        mock_llm=mock_llm,
        render_mode=render_mode,
        refine_boundaries=refine_boundaries,
        aspect_ratio=aspect_ratio,
        add_subtitles=add_subtitles,
        user_id=user_id,
        selection_mode=selection_mode,
        natural_acceptance_threshold=natural_acceptance_threshold,
        max_total_selected_duration_sec=max_total_selected_duration_sec,
    )

    return ProcessTaskResponse(
        task_id=task_id,
        status="queued",
        message="Video processing initiated successfully",
        video_filename=pathlib.Path(target_video_path).name,
    )


@app.get("/api/status/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    Retrieve real-time processing status, active stage, progress, and error information.
    """
    if not task_manager.is_valid_task_id(task_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid task identifier format.",
        )

    status_resp = task_manager.get_status_response(task_id)
    if not status_resp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )

    return status_resp


@app.get("/api/highlights/{task_id}", response_model=TaskHighlightsResponse)
def get_task_highlights(task_id: str) -> TaskHighlightsResponse:
    """
    Retrieve ranked highlight candidates, source timestamps, modality evidence summaries,
    and clip stream/download links for a task.
    """
    if not task_manager.is_valid_task_id(task_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid task identifier format.",
        )

    highlights_resp = task_manager.get_highlights_response(task_id)
    if not highlights_resp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )

    return highlights_resp


@app.get("/api/clips/{task_id}/{filename}")
def stream_or_download_clip(
    task_id: str,
    filename: str,
    download: bool = Query(False, description="Set to true to force attachment download"),
) -> FileResponse:
    """
    Safely stream or download an exported highlight MP4 clip.
    Protects against path traversal attacks and arbitrary filesystem access.
    """
    if not task_manager.is_valid_task_id(task_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid task identifier format.",
        )

    if not task_manager.is_valid_clip_filename(filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid clip filename. Must be a direct .mp4 file without path separators.",
        )

    clip_path = task_manager.safe_get_clip_path(task_id, filename)
    if not clip_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clip '{filename}' not found for task '{task_id}'.",
        )

    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'

    return FileResponse(
        path=str(clip_path),
        media_type="video/mp4",
        filename=filename if download else None,
        headers=headers,
    )


@app.get("/api/users/{user_id}/clips")
def get_user_clips(user_id: str) -> Dict[str, Any]:
    """
    Retrieve all local ClipSense AI clips saved for the specified user.
    """
    clips = task_manager.get_user_clips(user_id)
    return {
        "user_id": user_id,
        "total_clips": len(clips),
        "clips": clips,
    }


@app.get("/api/users/{user_id}/clips/{filename}")
def stream_user_clip(
    user_id: str,
    filename: str,
    download: bool = Query(False),
) -> FileResponse:
    """
    Stream or download an exported ClipSense AI video clip saved under the user's storage.
    """
    if not task_manager.is_valid_clip_filename(filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid clip filename. Must be a direct .mp4 file.",
        )

    user_dir = task_manager.get_user_clips_dir(user_id)
    clip_path = (user_dir / filename).resolve()

    try:
        clip_path.relative_to(user_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    if not clip_path.exists() or not clip_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clip '{filename}' not found for user.",
        )

    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'

    return FileResponse(
        path=str(clip_path),
        media_type="video/mp4",
        filename=filename if download else None,
        headers=headers,
    )


# Clean error handling avoiding leaked internals
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled error processing request {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred while processing the request."},
    )
