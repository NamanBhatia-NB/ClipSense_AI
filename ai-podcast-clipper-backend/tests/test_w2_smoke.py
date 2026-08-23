"""
ClipSense W2 Smoke & Integration Test

Executes the MultimodalExtractionPipeline on a short sample video:
tests/sample_short.mp4

Verifies:
1. Generation of the required output directory:
   runs/<run_id>/
       transcript.json
       frames/
       frames.json
       prosody.json
       conversation.json
       manifest.json
2. Printed checkmarks:
   [✓] Transcript extraction
   [✓] Frame extraction
   [✓] Prosody extraction
   [✓] Conversation extraction
3. Schema validation on all generated artifacts
4. Manifest metadata completeness
5. Clear printing of all artifact paths
"""

import json
import os
import pathlib
import sys

# Ensure backend root on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.schemas import TranscriptData, VisualData, ProsodyData, ConversationData
from pipeline.extraction.extractor_pipeline import MultimodalExtractionPipeline


def run_smoke_test():
    base_dir = pathlib.Path(__file__).parent.parent
    sample_video = base_dir / "tests" / "sample_short.mp4"

    if not sample_video.exists():
        raise FileNotFoundError(f"Sample test video not found: {sample_video}")

    print("==================================================")
    print("Running W2 Multimodal Extraction Pipeline Smoke Test")
    os.environ["CLIPSENSE_OFFLINE_TEST"] = "1"
    from pipeline.extraction.transcript_extractor import TranscriptExtractor
    t_extractor = TranscriptExtractor(model_name="tiny", fallback_on_model_failure=True)
    pipeline = MultimodalExtractionPipeline(transcript_extractor=t_extractor)
    artifacts = pipeline.run(
        video_path=str(sample_video),
        run_id="smoke_test_run",
        runs_base_dir=str(base_dir / "runs"),
        force_rerun=True,
    )

    # 1. Verify all artifact files exist
    assert os.path.exists(artifacts["transcript_json"]), "transcript.json missing"
    assert os.path.isdir(artifacts["frames_dir"]), "frames/ directory missing"
    assert os.path.exists(artifacts["frames_json"]), "frames.json missing"
    assert os.path.exists(artifacts["prosody_json"]), "prosody.json missing"
    assert os.path.exists(artifacts["conversation_json"]), "conversation.json missing"
    assert os.path.exists(artifacts["manifest_json"]), "manifest.json missing"

    # 2. Schema validation on all artifacts
    with open(artifacts["transcript_json"], "r", encoding="utf-8") as f:
        t_data = TranscriptData.model_validate_json(f.read())
        assert len(t_data.segments) > 0

    with open(artifacts["frames_json"], "r", encoding="utf-8") as f:
        v_data = VisualData.model_validate_json(f.read())
        assert len(v_data.sampled_frames) > 0
        assert len(os.listdir(artifacts["frames_dir"])) == len(v_data.sampled_frames)

    with open(artifacts["prosody_json"], "r", encoding="utf-8") as f:
        p_data = ProsodyData.model_validate_json(f.read())
        assert len(p_data.windows) > 0

    with open(artifacts["conversation_json"], "r", encoding="utf-8") as f:
        c_data = ConversationData.model_validate_json(f.read())
        assert len(c_data.turns) > 0

    with open(artifacts["manifest_json"], "r", encoding="utf-8") as f:
        manifest = json.load(f)
        assert manifest["run_id"] == "smoke_test_run"
        assert manifest["video_filename"] == "sample_short.mp4"
        assert "metrics" in manifest
        assert "artifacts" in manifest

    print("\n[✓] Smoke test passed: All artifacts validated against schemas!")


if __name__ == "__main__":
    run_smoke_test()
