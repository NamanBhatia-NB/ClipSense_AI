"""
ClipSense W1 Verification Suite

Verifies:
1. Schema contracts and continuous floating-point timestamps
2. ProposalSourceMetadata recording source resolution and anchors
3. Four independent evidence bundles without premature score collapse
4. Auxiliary temporal grid acting strictly as an indexed view
5. MTER Candidate schemas with explicit semantic verification criteria
6. Configurable baselines and thresholds
7. BaseExpert and pluggable visual encoder contracts
8. Orchestrator invariant check
"""

import json
import os
import sys

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import config, AppConfig, ProsodyConfig
from core.schemas import (
    TimestampedWord,
    TranscriptSegment,
    TranscriptData,
    SceneBoundary,
    SampledFrameMetadata,
    VisualData,
    ProsodyWindow,
    ProsodyData,
    SpeakerTurn,
    ConversationData,
    SourceResolutionType,
    ProposalSourceMetadata,
    TemporalProposal,
    ExpertEvidenceBundle,
    AuxiliaryGridCell,
    AuxiliaryTemporalGrid,
    CommonEvidenceBundle,
    SemanticVerificationCriteria,
    MTERCandidate,
    MTEROutput,
)
from pipeline.experts.base import BaseExpert, VisualFeatureExtractorInterface
from pipeline.orchestrator import PipelineOrchestrator
import numpy as np


def test_transcript_schemas():
    """Verify transcript word-level timestamps preserve exact floating-point values."""
    w1 = TimestampedWord(word="Hello", start=12.34567, end=12.78912, score=0.98)
    w2 = TimestampedWord(word="world", start=12.78912, end=13.12345, score=0.95)
    seg = TranscriptSegment(id=0, start=12.34567, end=13.12345, text="Hello world", words=[w1, w2])
    data = TranscriptData(full_text="Hello world", duration=13.12345, segments=[seg], words=[w1, w2])

    assert data.words[0].start == 12.34567
    assert data.words[1].end == 13.12345
    assert len(data.segments[0].words) == 2


def test_prosody_and_visual_schemas():
    """Verify prosody windows and visual sampled frame schemas."""
    pw = ProsodyWindow(
        start_time=10.0,
        end_time=11.0,
        duration=1.0,
        mean_f0=145.2,
        peak_f0=210.5,
        f0_std=25.3,
        rms_energy=0.045,
        rms_dynamic_range=0.035,
        voicing_fraction=0.78,
        speech_rate_wps=3.2,
    )
    prosody_data = ProsodyData(
        duration=60.0,
        sample_rate=16000,
        window_length_sec=1.0,
        hop_length_sec=0.5,
        windows=[pw],
    )
    assert prosody_data.windows[0].peak_f0 == 210.5

    scene = SceneBoundary(scene_index=0, start_time=0.0, end_time=4.55, duration=4.55)
    frame = SampledFrameMetadata(frame_index=30, timestamp=1.0, scene_index=0)
    visual_data = VisualData(
        fps=30.0,
        total_frames=1800,
        duration=60.0,
        scene_boundaries=[scene],
        sampled_frames=[frame],
    )
    assert visual_data.scene_boundaries[0].duration == 4.55


def test_independent_expert_bundles_no_score_collapse():
    """
    Verify four independent expert proposals are preserved separately
    with explicit source metadata and continuous timestamps.
    """
    # 1. Transcript Proposal
    meta_t = ProposalSourceMetadata(
        source_type=SourceResolutionType.WORD_TIMESTAMP,
        temporal_resolution_sec=0.02,
        alignment_anchor="word_12:45",
    )
    prop_t = TemporalProposal(
        proposal_id="prop_transcript_001",
        start_time=10.125,
        end_time=45.678,
        duration=35.553,
        confidence_estimate=0.88,
        evidence_type="semantic_topic_boundary",
        explanation="Speaker introduces thesis with high contextual completeness.",
        supporting_features={"semantic_density": 0.82, "thesis_clarity": 0.9},
        source_metadata=meta_t,
    )
    bundle_t = ExpertEvidenceBundle(expert_name="transcript", proposals=[prop_t])

    # 2. Visual Proposal (overlaps with transcript proposal)
    meta_v = ProposalSourceMetadata(
        source_type=SourceResolutionType.FRAME_TIMESTAMP,
        temporal_resolution_sec=0.033,
        alignment_anchor="frame_310:1350",
    )
    prop_v = TemporalProposal(
        proposal_id="prop_visual_001",
        start_time=10.333,
        end_time=45.000,
        duration=34.667,
        confidence_estimate=0.74,
        evidence_type="visual_activity_surge",
        explanation="Elevated facial motion and two camera cuts during key exchange.",
        supporting_features={"motion_magnitude": 0.45, "cut_count": 2},
        source_metadata=meta_v,
    )
    bundle_v = ExpertEvidenceBundle(expert_name="visual", proposals=[prop_v])

    # 3. Prosody Proposal
    meta_p = ProposalSourceMetadata(
        source_type=SourceResolutionType.PROSODY_WINDOW,
        temporal_resolution_sec=0.5,
        alignment_anchor="window_20:90",
    )
    prop_p = TemporalProposal(
        proposal_id="prop_prosody_001",
        start_time=12.000,
        end_time=44.500,
        duration=32.500,
        confidence_estimate=0.81,
        evidence_type="f0_dynamic_inflection",
        explanation="Pitch excursion with elevated RMS energy indicating emphasis.",
        supporting_features={"peak_f0_ratio": 1.65, "mean_energy": 0.052},
        source_metadata=meta_p,
    )
    bundle_p = ExpertEvidenceBundle(expert_name="prosody", proposals=[prop_p])

    # 4. Conversation Proposal
    meta_c = ProposalSourceMetadata(
        source_type=SourceResolutionType.SPEAKER_TURN,
        temporal_resolution_sec=0.7,
        alignment_anchor="turns_3:7",
    )
    prop_c = TemporalProposal(
        proposal_id="prop_conv_001",
        start_time=9.800,
        end_time=46.200,
        duration=36.400,
        confidence_estimate=0.79,
        evidence_type="rapid_turn_exchange",
        explanation="Interactive 4-turn question-and-answer resolution sequence.",
        supporting_features={"turn_count": 4, "mean_turn_duration_sec": 9.1},
        source_metadata=meta_c,
    )
    bundle_c = ExpertEvidenceBundle(expert_name="conversation", proposals=[prop_c])

    # Auxiliary 1s grid
    aux_grid = AuxiliaryTemporalGrid(
        bin_size_sec=1.0,
        total_bins=50,
        cells=[
            AuxiliaryGridCell(
                bin_index=15,
                bin_start=15.0,
                bin_end=16.0,
                active_proposal_ids={
                    "transcript": ["prop_transcript_001"],
                    "visual": ["prop_visual_001"],
                    "prosody": ["prop_prosody_001"],
                    "conversation": ["prop_conv_001"],
                },
            )
        ],
    )

    common_bundle = CommonEvidenceBundle(
        video_id="video_test_01",
        total_duration=60.0,
        transcript_evidence=bundle_t,
        visual_evidence=bundle_v,
        prosody_evidence=bundle_p,
        conversation_evidence=bundle_c,
        auxiliary_grid=aux_grid,
    )

    # Invariants verification
    orchestrator = PipelineOrchestrator()
    assert orchestrator.run_w1_contract_check(common_bundle) is True

    # Check that proposals remain separate and preserved
    assert len(common_bundle.transcript_evidence.proposals) == 1
    assert len(common_bundle.visual_evidence.proposals) == 1
    assert len(common_bundle.prosody_evidence.proposals) == 1
    assert len(common_bundle.conversation_evidence.proposals) == 1

    # Exact floating point start time check
    assert common_bundle.transcript_evidence.proposals[0].start_time == 10.125
    assert common_bundle.conversation_evidence.proposals[0].start_time == 9.800


def test_mter_candidate_and_semantic_criteria():
    """Verify MTER Candidate schemas with explicit semantic verification checklist."""
    criteria = SemanticVerificationCriteria(
        natural_semantic_start=True,
        sufficient_context=True,
        important_content_retained=True,
        complete_conversational_payoff=True,
        no_mid_sentence_ending=True,
    )

    candidate = MTERCandidate(
        candidate_id="mter_cand_001",
        proposed_start=10.125,
        proposed_end=45.678,
        duration=35.553,
        contributing_experts=["transcript", "prosody", "conversation"],
        temporal_agreement_iou=0.84,
        conflict_notes=[],
        semantic_verification=criteria,
        semantic_verification_status="passed",
        reasoning_trace="Transcript proposal corroborated by prosody emphasis and conversational resolution.",
    )

    output = MTEROutput(
        video_id="video_test_01",
        candidates=[candidate],
        execution_time_sec=0.12,
    )

    assert output.candidates[0].semantic_verification.no_mid_sentence_ending is True
    assert output.candidates[0].temporal_agreement_iou == 0.84

    # Test round-trip JSON serialization
    raw_json = output.model_dump_json()
    reloaded = MTEROutput.model_validate_json(raw_json)
    assert reloaded.candidates[0].candidate_id == "mter_cand_001"


def test_pluggable_visual_encoder_interface():
    """Verify pluggable visual encoder interface satisfies Protocol."""
    class DummyVisualEncoder:
        def extract_features(self, frame_bgr: np.ndarray) -> np.ndarray:
            return np.mean(frame_bgr, axis=(0, 1))

        def compute_distance(self, feat_a: np.ndarray, feat_b: np.ndarray) -> float:
            return float(np.linalg.norm(feat_a - feat_b))

    encoder = DummyVisualEncoder()
    assert isinstance(encoder, VisualFeatureExtractorInterface)


def test_config_defaults_and_override():
    """Verify config holds non-hardcoded configurable thresholds."""
    custom_cfg = AppConfig(
        prosody=ProsodyConfig(peak_f0_multiplier=1.8, local_baseline_window_sec=45.0)
    )
    assert custom_cfg.prosody.peak_f0_multiplier == 1.8
    assert custom_cfg.prosody.local_baseline_window_sec == 45.0
    assert config.prosody.peak_f0_multiplier == 1.5


if __name__ == "__main__":
    test_transcript_schemas()
    test_prosody_and_visual_schemas()
    test_independent_expert_bundles_no_score_collapse()
    test_mter_candidate_and_semantic_criteria()
    test_pluggable_visual_encoder_interface()
    test_config_defaults_and_override()
    print("All W1 architecture & schema tests passed successfully!")
