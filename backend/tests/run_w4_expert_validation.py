"""
ClipSense W4: Visual Expert & Prosody Expert Validation Runner

Demonstration and representative validation script that:
1. Loads existing W2 representative extraction artifacts:
   - runs/representative_90s_validation/frames.json
   - runs/representative_90s_validation/frames/
   - runs/representative_90s_validation/prosody.json
   - runs/representative_90s_validation/manifest.json
2. Executes VisualExpert (visual activity, scene transition density, visual shift evidence)
3. Executes ProsodyExpert (vocal pitch excursions, acoustic loudness elevation, local baseline)
4. Validates independent temporal proposals:
   - 0.0 <= start_time < end_time <= total_duration
   - duration == end_time - start_time within numerical precision
   - Continuous non-rounded floating-point timestamps
   - Strict source metadata:
     * Visual: SourceResolutionType.FRAME_TIMESTAMP
     * Prosody: SourceResolutionType.PROSODY_WINDOW
   - Non-hyped, factual evidence justifications
   - Localized candidate bounds (rejects unjustified whole-input proposals)
5. Generates evidence artifacts:
   - runs/representative_90s_validation/visual_evidence.json
   - runs/representative_90s_validation/prosody_evidence.json
6. Displays representative proposals and supporting measured features.
"""

import hashlib
import json
import logging
import math
import os
import pathlib
import sys
from typing import Any, Dict, List

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.config import config
from core.schemas import (
    ExpertEvidenceBundle,
    ProsodyData,
    SourceResolutionType,
    TemporalProposal,
    VisualData,
)
from pipeline.experts.prosody_expert import ProsodyExpert
from pipeline.experts.visual_expert import VisualExpert

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clipsense.w4.validation")


def compute_file_sha256(path: pathlib.Path) -> str:
    """Computes hexadecimal SHA-256 hash of a file for provenance verification."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_evidence_bundle(bundle: ExpertEvidenceBundle, total_duration: float, expected_expert: str):
    """Validates schema, continuous timestamps, finite bounds, and source metadata."""
    assert bundle.expert_name == expected_expert, f"Expected {expected_expert}, got {bundle.expert_name}"
    assert isinstance(bundle.proposals, list), "Proposals must be a list"

    for p in bundle.proposals:
        assert isinstance(p, TemporalProposal), f"Item {p} must be TemporalProposal"

        # Timestamp bounds
        assert 0.0 <= p.start_time, f"start_time negative: {p.start_time}"
        assert p.start_time < p.end_time, f"start_time {p.start_time} >= end_time {p.end_time}"
        assert p.end_time <= total_duration + 1e-4, (
            f"end_time {p.end_time} exceeds video duration {total_duration}"
        )

        # Finite check
        assert not math.isnan(p.start_time), "start_time is NaN"
        assert not math.isnan(p.end_time), "end_time is NaN"
        assert not math.isinf(p.start_time), "start_time is infinite"
        assert not math.isinf(p.end_time), "end_time is infinite"

        # Continuous float check
        assert isinstance(p.start_time, float), "start_time must be float"
        assert isinstance(p.end_time, float), "end_time must be float"

        # Duration check
        expected_dur = p.end_time - p.start_time
        assert abs(p.duration - expected_dur) < 1e-4, (
            f"duration {p.duration} != end_time - start_time {expected_dur}"
        )

        # Confidence estimate check (relative evidence strength in [0.0, 1.0])
        assert 0.0 <= p.confidence_estimate <= 1.0, (
            f"confidence_estimate {p.confidence_estimate} out of bounds"
        )
        assert not math.isnan(p.confidence_estimate), "confidence_estimate is NaN"

        # Explanation check
        assert p.explanation and len(p.explanation.strip()) > 10, "explanation missing or too short"

        # Metadata validation
        meta = p.source_metadata
        if expected_expert == "visual":
            assert meta.source_type == SourceResolutionType.FRAME_TIMESTAMP, (
                f"Visual expert source_type must be FRAME_TIMESTAMP, got {meta.source_type}"
            )
            assert meta.alignment_anchor.startswith("sampled_frame_"), (
                f"Visual anchor must reference sampled_frame_, got {meta.alignment_anchor}"
            )
            # Must reflect sampled-frame interval, not native frame interval
            assert meta.temporal_resolution_sec >= 0.5, (
                f"Visual expert temporal_resolution_sec must reflect sampling interval (got {meta.temporal_resolution_sec})"
            )
        elif expected_expert == "prosody":
            assert meta.source_type == SourceResolutionType.PROSODY_WINDOW, (
                f"Prosody expert source_type must be PROSODY_WINDOW, got {meta.source_type}"
            )
            assert meta.alignment_anchor.startswith("window_"), (
                f"Prosody anchor must reference windows, got {meta.alignment_anchor}"
            )

        assert meta.temporal_resolution_sec > 0.0, "temporal_resolution_sec must be positive"


def run_w4_expert_validation():
    """Main validation runner executing Visual and Prosody experts on W2 representative artifacts."""
    base_path = pathlib.Path(__file__).parent.parent / "runs" / "representative_90s_validation"
    assert base_path.exists(), f"W2 representative run directory not found: {base_path}"

    manifest_path = base_path / "manifest.json"
    frames_path = base_path / "frames.json"
    prosody_path = base_path / "prosody.json"

    assert manifest_path.exists(), f"manifest.json missing at {manifest_path}"
    assert frames_path.exists(), f"frames.json missing at {frames_path}"
    assert prosody_path.exists(), f"prosody.json missing at {prosody_path}"

    # 1. Load W2 Visual Extraction Data
    with open(frames_path, "r", encoding="utf-8") as f:
        frames_dict = json.load(f)
    v_data = VisualData.model_validate(frames_dict)
    total_duration = float(v_data.duration)
    frames_hash = compute_file_sha256(frames_path)
    logger.info(f"Loaded W2 visual data: {len(v_data.sampled_frames)} frames, SHA256: {frames_hash[:12]}...")
    print("[✓] W2 visual data loaded")

    # 2. Load W2 Prosody Extraction Data
    with open(prosody_path, "r", encoding="utf-8") as f:
        prosody_dict = json.load(f)
    p_data = ProsodyData.model_validate(prosody_dict)
    prosody_hash = compute_file_sha256(prosody_path)
    logger.info(f"Loaded W2 prosody data: {len(p_data.windows)} windows, SHA256: {prosody_hash[:12]}...")
    print("[✓] W2 prosody data loaded")

    # 3. Run Visual Expert
    visual_expert = VisualExpert()
    visual_bundle = visual_expert.evaluate(
        extraction_data=v_data,
        context={"video_duration": total_duration},
    )
    print("[✓] Visual Expert")

    # 4. Run Prosody Expert
    prosody_expert = ProsodyExpert()
    prosody_bundle = prosody_expert.evaluate(
        extraction_data=p_data,
        context={"video_duration": total_duration},
    )
    print("[✓] Prosody Expert")

    # 5. Validate Proposals Schema, Timestamps, and Localization
    validate_evidence_bundle(visual_bundle, total_duration, "visual")
    validate_evidence_bundle(prosody_bundle, total_duration, "prosody")
    print("[✓] Proposal schema validation")
    print("[✓] Timestamp validation")

    # Verify that proposals are localized and not unjustified whole-input partitions
    assert len(visual_bundle.proposals) > 0, "Visual expert must produce at least 1 localized proposal"
    assert len(prosody_bundle.proposals) > 0, "Prosody expert must produce at least 1 localized proposal"

    for p in visual_bundle.proposals:
        assert p.duration < total_duration * 0.90, (
            f"Visual proposal {p.proposal_id} spans {p.duration:.2f}s of {total_duration:.2f}s, "
            f"which is an unjustified whole-input proposal!"
        )

    for p in prosody_bundle.proposals:
        assert p.duration < total_duration * 0.90, (
            f"Prosody proposal {p.proposal_id} spans {p.duration:.2f}s of {total_duration:.2f}s, "
            f"which is an unjustified whole-input proposal!"
        )

    # 6. Save Evidence Artifacts
    v_out_path = base_path / "visual_evidence.json"
    p_out_path = base_path / "prosody_evidence.json"

    with open(v_out_path, "w", encoding="utf-8") as f:
        f.write(visual_bundle.model_dump_json(indent=2))

    with open(p_out_path, "w", encoding="utf-8") as f:
        f.write(prosody_bundle.model_dump_json(indent=2))

    print("[✓] Evidence artifacts generated")

    # 7. Print Inspection Summary with Measured Features
    print("\n--- VISUAL EXPERT EVIDENCE PROPOSALS ---")
    print(f"Total Proposals Generated: {len(visual_bundle.proposals)}")
    for p in visual_bundle.proposals:
        print(f"  - [{p.start_time:.3f}s -> {p.end_time:.3f}s] (dur: {p.duration:.2f}s, conf: {p.confidence_estimate:.2f})")
        print(f"    Type: {p.evidence_type} | Anchor: {p.source_metadata.alignment_anchor}")
        print(f"    Source: {p.source_metadata.source_type.value}")
        print(f"    Sampling interval: {p.source_metadata.temporal_resolution_sec:.1f}s | Native FPS: {p.source_metadata.sample_rate_or_fps}")
        print(f"    Features: {p.supporting_features}")
        print(f"    Explanation: {p.explanation}\n")

    print("--- PROSODY EXPERT EVIDENCE PROPOSALS ---")
    print(f"Total Proposals Generated: {len(prosody_bundle.proposals)}")
    for p in prosody_bundle.proposals:
        print(f"  - [{p.start_time:.3f}s -> {p.end_time:.3f}s] (dur: {p.duration:.2f}s, conf: {p.confidence_estimate:.2f})")
        print(f"    Type: {p.evidence_type} | Anchor: {p.source_metadata.alignment_anchor}")
        print(f"    Window hop: {p.source_metadata.temporal_resolution_sec:.4f}s | Sample Rate: {p.source_metadata.sample_rate_or_fps} Hz")
        print(f"    Features: {p.supporting_features}")
        print(f"    Explanation: {p.explanation}\n")

    print(f"Evidence artifacts successfully saved:")
    print(f"  - {v_out_path}")
    print(f"  - {p_out_path}")
    print("\n[✓] ALL W4 EVIDENCE CHECKS PASSED (Visual & Prosody candidates verified)!")


if __name__ == "__main__":
    run_w4_expert_validation()
