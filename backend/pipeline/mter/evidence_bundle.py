"""
ClipSense W5: Evidence Bundle Builder & Loader

Connects the four independent modality-specific evidence streams:
1. Transcript Expert (transcript_evidence.json)
2. Conversation Expert (conversation_evidence.json)
3. Visual Expert (visual_evidence.json)
4. Prosody Expert (prosody_evidence.json)

Core invariants:
- Preserves the four sources completely separate in CommonEvidenceBundle.
- Authoritative timestamps remain continuous floating-point seconds.
- No rounding or discretizing of original timestamps.
- Auxiliary temporal grid is constructed solely as an indexing/lookup aid.
- Handles missing or empty expert bundles gracefully.
"""

import json
import logging
import math
import pathlib
from typing import Dict, List, Optional, Union

from core.schemas import (
    AuxiliaryGridCell,
    AuxiliaryTemporalGrid,
    CommonEvidenceBundle,
    ExpertEvidenceBundle,
    TemporalProposal,
)

logger = logging.getLogger("clipsense.mter.bundle")


def build_auxiliary_grid(
    total_duration: float,
    all_proposals: Dict[str, List[TemporalProposal]],
    bin_size_sec: float = 1.0,
) -> AuxiliaryTemporalGrid:
    """
    Constructs an auxiliary 1-second discretized index over the video timeline.
    
    CRITICAL: This grid is purely for fast temporal lookup. It NEVER overwrites
    or rounds the authoritative continuous floating-point proposal timestamps.
    """
    if total_duration <= 0.0:
        return AuxiliaryTemporalGrid(bin_size_sec=bin_size_sec, total_bins=0, cells=[])

    total_bins = max(1, math.ceil(total_duration / bin_size_sec))
    cells: List[AuxiliaryGridCell] = []

    for bin_idx in range(total_bins):
        bin_start = bin_idx * bin_size_sec
        bin_end = min(total_duration, (bin_idx + 1) * bin_size_sec)
        active_map: Dict[str, List[str]] = {}

        for expert_name, proposals in all_proposals.items():
            active_ids = [
                p.proposal_id
                for p in proposals
                if max(bin_start, p.start_time) < min(bin_end, p.end_time)
            ]
            if active_ids:
                active_map[expert_name] = active_ids

        cells.append(
            AuxiliaryGridCell(
                bin_index=bin_idx,
                bin_start=bin_start,
                bin_end=bin_end,
                active_proposal_ids=active_map,
            )
        )

    return AuxiliaryTemporalGrid(
        bin_size_sec=bin_size_sec,
        total_bins=total_bins,
        cells=cells,
    )


def build_common_evidence_bundle(
    video_id: str,
    total_duration: float,
    transcript_evidence: Optional[ExpertEvidenceBundle] = None,
    conversation_evidence: Optional[ExpertEvidenceBundle] = None,
    visual_evidence: Optional[ExpertEvidenceBundle] = None,
    prosody_evidence: Optional[ExpertEvidenceBundle] = None,
    bin_size_sec: float = 1.0,
) -> CommonEvidenceBundle:
    """
    Creates a CommonEvidenceBundle aggregating all 4 independent expert bundles.
    Fills in empty bundles for any missing modality.
    """
    t_bundle = transcript_evidence or ExpertEvidenceBundle(expert_name="transcript", proposals=[])
    c_bundle = conversation_evidence or ExpertEvidenceBundle(expert_name="conversation", proposals=[])
    v_bundle = visual_evidence or ExpertEvidenceBundle(expert_name="visual", proposals=[])
    p_bundle = prosody_evidence or ExpertEvidenceBundle(expert_name="prosody", proposals=[])

    all_proposals: Dict[str, List[TemporalProposal]] = {
        "transcript": t_bundle.proposals,
        "conversation": c_bundle.proposals,
        "visual": v_bundle.proposals,
        "prosody": p_bundle.proposals,
    }

    aux_grid = build_auxiliary_grid(total_duration, all_proposals, bin_size_sec)

    return CommonEvidenceBundle(
        video_id=video_id,
        total_duration=total_duration,
        transcript_evidence=t_bundle,
        conversation_evidence=c_bundle,
        visual_evidence=v_bundle,
        prosody_evidence=p_bundle,
        auxiliary_grid=aux_grid,
    )


def load_evidence_bundle_from_run(
    run_dir: Union[str, pathlib.Path],
    video_id: Optional[str] = None,
    total_duration: Optional[float] = None,
    bin_size_sec: float = 1.0,
) -> CommonEvidenceBundle:
    """
    Loads transcript_evidence.json, conversation_evidence.json, visual_evidence.json,
    and prosody_evidence.json from a run directory.
    """
    path = pathlib.Path(run_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {path}")

    # Discover total duration and video_id if not explicitly provided
    resolved_id = video_id or path.name
    resolved_duration = total_duration

    if resolved_duration is None:
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    resolved_duration = float(data.get("duration", 0.0))
                    if "video_id" in data:
                        resolved_id = data["video_id"]
            except Exception as e:
                logger.warning(f"Could not read duration from manifest.json: {e}")

    if resolved_duration is None:
        # Fallback to checking transcript.json, frames.json, or prosody.json
        for fname in ["transcript.json", "frames.json", "prosody.json"]:
            fpath = path / fname
            if fpath.exists():
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if "duration" in data:
                            resolved_duration = float(data["duration"])
                            break
                except Exception:
                    pass

    if resolved_duration is None:
        resolved_duration = 0.0

    def load_bundle_file(filename: str, expert_name: str) -> ExpertEvidenceBundle:
        file_path = path / filename
        if not file_path.exists():
            logger.info(f"Evidence file {filename} not found in {path}; returning empty bundle.")
            return ExpertEvidenceBundle(expert_name=expert_name, proposals=[])
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                return ExpertEvidenceBundle.model_validate_json(content)
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            raise ValueError(f"Failed to load valid ExpertEvidenceBundle from {file_path}: {e}")

    t_bundle = load_bundle_file("transcript_evidence.json", "transcript")
    c_bundle = load_bundle_file("conversation_evidence.json", "conversation")
    v_bundle = load_bundle_file("visual_evidence.json", "visual")
    p_bundle = load_bundle_file("prosody_evidence.json", "prosody")

    # If duration wasn't in manifest, ensure it covers the maximum proposal end_time
    all_proposals = t_bundle.proposals + c_bundle.proposals + v_bundle.proposals + p_bundle.proposals
    if all_proposals:
        max_prop_end = max(p.end_time for p in all_proposals)
        if resolved_duration < max_prop_end:
            resolved_duration = max_prop_end

    return build_common_evidence_bundle(
        video_id=resolved_id,
        total_duration=resolved_duration,
        transcript_evidence=t_bundle,
        conversation_evidence=c_bundle,
        visual_evidence=v_bundle,
        prosody_evidence=p_bundle,
        bin_size_sec=bin_size_sec,
    )
