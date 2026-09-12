"""
ClipSense MTER (Multimodal Temporal Evidence Reasoner) Module (W5)
"""

from pipeline.mter.clustering import (
    cluster_boundary_candidates,
    detect_evidence_conflicts,
    extract_all_proposals_flat,
    group_proposals_into_event_regions,
)
from pipeline.mter.evidence_bundle import (
    build_auxiliary_grid,
    build_common_evidence_bundle,
    load_evidence_bundle_from_run,
)
from pipeline.mter.reasoner import MTERReasoner

__all__ = [
    "MTERReasoner",
    "build_common_evidence_bundle",
    "load_evidence_bundle_from_run",
    "build_auxiliary_grid",
    "extract_all_proposals_flat",
    "group_proposals_into_event_regions",
    "cluster_boundary_candidates",
    "detect_evidence_conflicts",
]
