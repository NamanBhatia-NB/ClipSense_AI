"""
ClipSense Local FastAPI Bridge Tests (Week 9)

Verifies:
1. Health endpoint (/api/health) and FFmpeg detection.
2. File type validation and rejection of unsupported uploads (.txt, .exe).
3. Payload validation (missing file and path rejected with 400).
4. Task identifier validation and path traversal rejection.
5. Clip streaming security (path traversal, invalid extensions rejected).
6. Asynchronous task status lifecycle and progress transitions.
7. Graceful handling and reporting of failed tasks (sanitized error messages).
8. Empty highlight results handling.
9. Clip download header configuration (?download=true).
10. End-to-end real integration test using sample_short.mp4 and physical video rendering.
"""

import io
import os
import pathlib
import sys
import tempfile
import time
import unittest
from fastapi.testclient import TestClient

# Ensure backend root on sys.path
backend_dir = pathlib.Path(__file__).parent.parent.resolve()
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from api.server import app, task_manager
from api.schemas import ModalityEvidenceSummary
from core.schemas import (
    CandidateSupportMetrics,
    ClipExportResult,
    ClipsManifest,
    MTERCandidate,
    MTEROutput,
    SemanticVerificationCriteria,
)


class TestClipSenseAPI(unittest.TestCase):
    """Unit tests for FastAPI endpoints, task management, and security guards."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_short = backend_dir / "tests" / "sample_short.mp4"

    def test_health_endpoint(self):
        """Verify GET /api/health returns readiness status and FFmpeg availability."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "clipsense-api")
        self.assertIn("ffmpeg_available", data)
        self.assertIsInstance(data["ffmpeg_available"], bool)
        self.assertEqual(data["device"], "cpu")
        self.assertGreater(data["timestamp"], 0)

    def test_root_info_endpoint(self):
        """Verify root / returns API metadata and docs pointer."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("ClipSense", data["service"])
        self.assertEqual(data["version"], "1.0.0")

    def test_process_missing_payload_rejected(self):
        """Verify POST /api/process rejects requests without video file or path."""
        response = self.client.post("/api/process", data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())

    def test_process_invalid_file_extension_rejected(self):
        """Verify uploads with unsupported extensions (.txt, .exe) are rejected with 400."""
        fake_file = io.BytesIO(b"fake executable payload")
        response = self.client.post(
            "/api/process",
            files={"file": ("malicious.exe", fake_file, "application/octet-stream")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported video format", response.json()["detail"])

        fake_text = io.BytesIO(b"text data")
        response = self.client.post(
            "/api/process",
            files={"file": ("document.txt", fake_text, "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported video format", response.json()["detail"])

    def test_process_nonexistent_local_path_rejected(self):
        """Verify referencing a non-existent local file returns 404."""
        response = self.client.post(
            "/api/process",
            data={"video_path": "non_existent_video_file_xyz.mp4"},
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("Input video file not found", response.json()["detail"])

    def test_status_invalid_task_id_format(self):
        """Verify invalid task IDs (e.g. path traversal characters) are rejected with 400."""
        for invalid_id in ["../escape", "task/slash", "bad@char!", "", "a" * 100]:
            response = self.client.get(f"/api/status/{invalid_id}")
            self.assertIn(response.status_code, (400, 404))

    def test_status_nonexistent_task(self):
        """Verify query for valid-format but non-existent task returns 404."""
        response = self.client.get("/api/status/task_0123456789abcdef")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"].lower())

    def test_clip_streaming_path_traversal_guards(self):
        """Verify clip streaming strictly rejects path traversal attempts."""
        # Malicious filename with traversal
        response = self.client.get("/api/clips/task_0123456789abcdef/..%2F..%2Fsecret.txt")
        self.assertIn(response.status_code, (400, 404))

        # Disallowed file extension
        response = self.client.get("/api/clips/task_0123456789abcdef/secret.key")
        self.assertEqual(response.status_code, 400)

        # Valid name on non-existent task returns 404
        response = self.client.get("/api/clips/task_0123456789abcdef/highlight_1.mp4")
        self.assertEqual(response.status_code, 404)

    def test_task_status_lifecycle_and_transitions(self):
        """Verify task status and stage transitions are tracked accurately."""
        task_id = task_manager.create_task("test_video.mp4")
        self.assertTrue(task_manager.is_valid_task_id(task_id))

        # Initial queued state
        status_resp = task_manager.get_status_response(task_id)
        self.assertIsNotNone(status_resp)
        self.assertEqual(status_resp.status, "queued")
        self.assertEqual(status_resp.stage, "queued")
        self.assertEqual(status_resp.progress, 0.0)

        # Stage update: extraction
        task_manager.update_progress(task_id, "extraction", 0.20, "Extracting features")
        status_resp = task_manager.get_status_response(task_id)
        self.assertEqual(status_resp.status, "processing")
        self.assertEqual(status_resp.stage, "extraction")
        self.assertAlmostEqual(status_resp.progress, 0.20)

        # Stage update: MTER
        task_manager.update_progress(task_id, "mter", 0.75, "Reasoning consensus")
        status_resp = task_manager.get_status_response(task_id)
        self.assertEqual(status_resp.stage, "mter")
        self.assertAlmostEqual(status_resp.progress, 0.75)

        # API query matches
        api_res = self.client.get(f"/api/status/{task_id}")
        self.assertEqual(api_res.status_code, 200)
        self.assertEqual(api_res.json()["stage"], "mter")

    def test_failed_task_reporting_and_error_sanitization(self):
        """Verify failed task reports 'failed' status with a sanitized error message."""
        task_id = task_manager.create_task("fail_sample.mp4")
        task_manager.set_failed(task_id, "Audio extraction failed on corrupt stream")

        status_resp = task_manager.get_status_response(task_id)
        self.assertEqual(status_resp.status, "failed")
        self.assertEqual(status_resp.stage, "failed")
        self.assertIn("Audio extraction failed", status_resp.error)

        api_res = self.client.get(f"/api/status/{task_id}")
        self.assertEqual(api_res.status_code, 200)
        data = api_res.json()
        self.assertEqual(data["status"], "failed")
        self.assertIsNotNone(data["error"])

    def test_empty_highlights_handling(self):
        """Verify highlights endpoint returns empty list gracefully when no candidates are found."""
        task_id = task_manager.create_task("silent_video.mp4")
        empty_mter = MTEROutput(
            video_id="silent_video",
            candidates=[],
            execution_time_sec=1.5,
        )
        task_manager.set_completed(
            task_id=task_id,
            total_duration=60.0,
            mter_output=empty_mter,
            clips_manifest=None,
            execution_time_sec=1.5,
        )

        response = self.client.get(f"/api/highlights/{task_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["total_candidates"], 0)
        self.assertEqual(data["highlights"], [])

    def test_clip_metadata_and_attachment_download_header(self):
        """Verify exported clip streaming and attachment download headers."""
        task_id = task_manager.create_task("demo.mp4")
        task = task_manager.get_task(task_id)
        clips_dir = pathlib.Path(task["clips_dir"])

        # Write a mock MP4 file into the task's isolated clips directory
        clip_file = clips_dir / "highlight_1.mp4"
        clip_content = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42"
        with open(clip_file, "wb") as f:
            f.write(clip_content)

        # Stream access (inline playback)
        stream_res = self.client.get(f"/api/clips/{task_id}/highlight_1.mp4")
        self.assertEqual(stream_res.status_code, 200)
        self.assertEqual(stream_res.headers.get("content-type"), "video/mp4")
        self.assertNotIn("attachment", stream_res.headers.get("content-disposition", ""))

        # Download access (attachment header)
        dl_res = self.client.get(f"/api/clips/{task_id}/highlight_1.mp4?download=true")
        self.assertEqual(dl_res.status_code, 200)
        self.assertIn("attachment", dl_res.headers.get("content-disposition", ""))
        self.assertIn("highlight_1.mp4", dl_res.headers.get("content-disposition", ""))


class TestRealPipelineIntegration(unittest.TestCase):
    """
    Real end-to-end integration test executing the research pipeline
    (extraction, 4 experts, MTER reasoning, and physical FFmpeg video slicing)
    via the local FastAPI service.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_video = backend_dir / "tests" / "sample_conversational_90s.mp4"
        if not cls.sample_video.exists():
            cls.sample_video = backend_dir / "test.mp4"
        if not cls.sample_video.exists():
            raise unittest.SkipTest(f"Sample video not found")

    def test_real_video_process_end_to_end(self):
        """
        Submits sample_conversational_90s.mp4, waits for background execution, verifies:
        1. Task transitions to 'completed'.
        2. Highlights response contains valid MTER candidates with modality evidence.
        3. Real exported MP4 clip is playable and streamable via the API.
        """
        response = self.client.post(
            "/api/process",
            data={
                "video_path": str(self.sample_video),
                "top_k": 1,
                "use_cache": True,
                "mock_llm": True,
                "render_mode": "accurate",
            },
        )
        self.assertEqual(response.status_code, 202)
        resp_data = response.json()
        task_id = resp_data["task_id"]
        self.assertEqual(resp_data["status"], "queued")
        self.assertIn(self.sample_video.name, resp_data["video_filename"])

        # Poll task status until complete or timeout (max 45 seconds)
        t_start = time.time()
        final_status = "queued"
        while time.time() - t_start < 45.0:
            st_res = self.client.get(f"/api/status/{task_id}")
            self.assertEqual(st_res.status_code, 200)
            final_status = st_res.json()["status"]
            if final_status in ("completed", "failed"):
                break
            time.sleep(0.5)

        self.assertEqual(final_status, "completed", f"Pipeline failed or timed out. Last status: {final_status}")

        # Verify Highlights endpoint output
        hl_res = self.client.get(f"/api/highlights/{task_id}")
        self.assertEqual(hl_res.status_code, 200)
        hl_data = hl_res.json()
        self.assertEqual(hl_data["status"], "completed")
        self.assertGreaterEqual(hl_data["total_candidates"], 1)

        # Inspect first candidate
        cand = hl_data["highlights"][0]
        self.assertEqual(cand["rank"], 1)
        self.assertGreater(cand["duration"], 0.0)
        self.assertGreaterEqual(cand["confidence_estimate"], 0.0)
        self.assertLessEqual(cand["confidence_estimate"], 1.0)
        self.assertIsNotNone(cand["decision_summary"])

        # Verify Modality Evidence breakdown
        ev = cand["modality_evidence"]
        self.assertIn("transcript_support", ev)
        self.assertIn("conversation_support", ev)
        self.assertIn("visual_support", ev)
        self.assertIn("prosody_support", ev)

        # Verify real exported clip metadata and streaming access
        self.assertIsNotNone(cand["clip"])
        clip_info = cand["clip"]
        self.assertEqual(clip_info["status"], "success")
        self.assertGreater(clip_info["file_size_bytes"], 0)
        self.assertTrue(clip_info["stream_url"].startswith("/api/clips/"))

        # Verify actual HTTP streaming of the exported clip
        clip_stream_res = self.client.get(clip_info["stream_url"])
        self.assertEqual(clip_stream_res.status_code, 200)
        self.assertEqual(clip_stream_res.headers.get("content-type"), "video/mp4")
        self.assertEqual(len(clip_stream_res.content), clip_info["file_size_bytes"])


if __name__ == "__main__":
    unittest.main()
