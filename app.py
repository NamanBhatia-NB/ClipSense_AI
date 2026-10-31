import os
import sys
import pathlib
import time

# Add backend directory to sys.path and resolve 'app' package namespace
backend_dir = pathlib.Path(__file__).parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Enable package resolution so 'from app.config import config' resolves to backend/app/config.py
__path__ = [str(backend_dir / "app")]
if "app" not in sys.modules or not hasattr(sys.modules["app"], "__path__"):
    sys.modules["app"] = sys.modules.get(__name__, sys.modules.get("app"))
    if sys.modules["app"] is not None:
        sys.modules["app"].__path__ = [str(backend_dir / "app")]

# Configure CORS origins if not already provided
if "CLIPSENSE_CORS_ORIGINS" not in os.environ:
    os.environ["CLIPSENSE_CORS_ORIGINS"] = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001,https://clip-sense-ai.vercel.app"

# Support Hugging Face ZeroGPU runtime
try:
    import spaces
except ImportError:
    class MockSpaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is not None:
                return func
            return lambda f: f
    spaces = MockSpaces()

# Ensure torchaudio 2.8+ compatibility for pyannote.audio and WhisperX
try:
    import torchaudio
    if not hasattr(torchaudio, "AudioMetaData"):
        class AudioMetaData:
            def __init__(self, sample_rate: int = 16000, num_frames: int = 0, num_channels: int = 1, bits_per_sample: int = 16, encoding: str = "PCM_S"):
                self.sample_rate = sample_rate
                self.num_frames = num_frames
                self.num_channels = num_channels
                self.bits_per_sample = bits_per_sample
                self.encoding = encoding
        torchaudio.AudioMetaData = AudioMetaData

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
except Exception:
    pass

import gradio as gr
from fastapi.middleware.cors import CORSMiddleware
from api.server import app as fastapi_app
from pipeline.render.video_renderer import VideoRenderer


@spaces.GPU(duration=60)
def check_hardware(command: str = "status"):
    """
    ZeroGPU entrypoint function decorated with @spaces.GPU.
    Directly bound to gr.Interface so the Hugging Face ZeroGPU supervisor
    detects the GPU handler during container startup.
    """
    ffmpeg_ok = VideoRenderer().is_available()
    cuda_ok = False
    device_name = "cpu"
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            device_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "status": "online",
        "service": "ClipSense AI Backend Bridge",
        "version": "1.0.0",
        "ffmpeg_available": ffmpeg_ok,
        "cuda_available": cuda_ok,
        "hardware_device": device_name,
        "endpoints": {
            "health": "/api/health",
            "process": "/api/process",
            "status": "/api/status/{task_id}",
            "highlights": "/api/highlights/{task_id}",
            "clips": "/api/clips/{task_id}/{filename}",
            "docs": "/docs",
        },
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }


# Standard Gradio Interface passed directly into ZeroGPU handler
demo = gr.Interface(
    fn=check_hardware,
    inputs=gr.Textbox(label="Command", value="status"),
    outputs=gr.JSON(label="System & ZeroGPU Hardware Status"),
    title="🎬 ClipSense AI Backend",
    description=(
        "**Query-Free Long-Form Video Highlight Extraction Framework**  \n"
        "Deterministic Multimodal Temporal Evidence Reasoner (MTER)\n\n"
        "This Hugging Face Space hosts the FastAPI backend bridge powering ClipSense Studio.\n\n"
        "- **Health Check:** [/api/health](/api/health)\n"
        "- **Interactive API Docs:** [/docs](/docs)\n"
        "- **Process Endpoint:** `POST /api/process`\n"
    ),
)

# Enable queue required by ZeroGPU
demo.queue()

# Remove default '/' route so Gradio serves the root web interface at '/'
fastapi_app.router.routes = [
    r for r in fastapi_app.router.routes if getattr(r, "path", None) != "/"
]

# Mount Gradio onto the FastAPI app so all FastAPI /api/* routes take precedence
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    try:
        import spaces.zero
        spaces.zero.startup()
    except (ImportError, AttributeError):
        pass
    uvicorn.run(app, host="0.0.0.0", port=7860)
