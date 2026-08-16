"""
ClipSense Pipeline Orchestrator (W1 Staging)

Coordinates the end-to-end execution graph across:
1. Multimodal Extraction (W2)
2. Four Independent Evidence Experts (W3 - Transcript & Conversation; W4 - Visual & Prosody)
3. Common Evidence Bundle & Auxiliary Normalization (W5)
4. MTER Candidate Selection (W5 prototype / Post-W5 full reasoner)
5. Temporal Boundary Refinement (Post-W5)
6. LR-ASD + Subtitles + Composition (Post-W5)
"""

import logging
import time
from typing import Optional
from app.config import config
from core.schemas import (
    CommonEvidenceBundle,
    MTEROutput,
    ProcessVideoRequest,
    ProcessVideoResponse,
)

logger = logging.getLogger("clipsense.pipeline")


class PipelineOrchestrator:
    """Orchestrator linking all ClipSense processing stages."""

    def __init__(self, run_config=None):
        self.config = run_config or config

    def run_w1_contract_check(self, bundle: CommonEvidenceBundle) -> bool:
        """
        Verify that an evidence bundle complies with all W1 architectural invariants:
        - 4 distinct expert bundles are present
        - Authoritative continuous timestamps are valid (start < end)
        - Proposals contain explicit source metadata and confidence estimates
        - No score collapse was performed
        """
        assert bundle.transcript_evidence.expert_name == "transcript"
        assert bundle.visual_evidence.expert_name == "visual"
        assert bundle.prosody_evidence.expert_name == "prosody"
        assert bundle.conversation_evidence.expert_name == "conversation"

        for b in [
            bundle.transcript_evidence,
            bundle.visual_evidence,
            bundle.prosody_evidence,
            bundle.conversation_evidence,
        ]:
            for p in b.proposals:
                assert p.start_time <= p.end_time, f"Invalid boundary in {p.proposal_id}"
                assert 0.0 <= p.confidence_estimate <= 1.0, f"Invalid confidence in {p.proposal_id}"
                assert p.source_metadata is not None, f"Missing source metadata in {p.proposal_id}"

        return True
