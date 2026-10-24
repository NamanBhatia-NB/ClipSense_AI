"""
Ablation and Experiment Runner (W6)

Dispatches evaluation execution across the six required experiment modes:
1. baseline: legacy transcript-only LLM baseline
2. transcript_conversation: MTER evidence ablation using transcript + conversation
3. transcript_conversation_visual: MTER evidence ablation using transcript + conversation + visual
4. transcript_conversation_prosody: MTER evidence ablation using transcript + conversation + prosody
5. all_experts_no_mter: deterministic multimodal non-MTER baseline
6. mter_full: complete MTER

Guarantees:
- W1-W5 modules remain completely frozen.
- Continuous floating-point timestamps are preserved.
- all_experts_no_mter implements a documented, deterministic non-graph aggregation.
- All modes output the identical EvaluationSpan interface.
"""

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple

from app.config import config
from core.schemas import (
    CommonEvidenceBundle,
    ConversationData,
    ExpertEvidenceBundle,
    ProsodyData,
    TemporalProposal,
    TranscriptData,
    VisualData,
)
from evaluation.baseline import (
    BaselineConfig,
    LegacyTranscriptBaselineSelector,
)
from evaluation.metrics import compute_temporal_iou
from evaluation.schemas import EvaluationSpan, ExperimentMode
from pipeline.mter.evidence_bundle import build_common_evidence_bundle
from pipeline.mter.reasoner import MTERReasoner
from pipeline.optimization.boundary_refiner import BoundaryRefiner

logger = logging.getLogger("clipsense.evaluation.ablation_runner")


def run_all_experts_no_mter_algorithm(
    transcript_bundle: ExpertEvidenceBundle,
    conversation_bundle: ExpertEvidenceBundle,
    visual_bundle: ExpertEvidenceBundle,
    prosody_bundle: ExpertEvidenceBundle,
    video_duration: float,
    min_duration_sec: float = 10.0,
    max_duration_sec: float = 60.0,
    nms_iou_threshold: float = 0.30,
) -> List[EvaluationSpan]:
    """
    Deterministic multimodal non-MTER baseline.
    
    Formal Specification:
    1. Consumes the 4 independent bundles.
    2. Does NOT use MTER compatibility graph, event clustering, or conflict isolation.
    3. Pools raw candidate proposals satisfying duration bounds, plus valid pairwise
       intersections between distinct experts.
    4. Evaluates unweighted mean modality coverage across available modalities:
       Score(I) = (1 / M_available) * sum_{m in available} max_{p in m} (conf(p) * |p cap I| / |I|)
       Missing modalities (bundles with 0 proposals) are treated as unavailable, not negative evidence.
    5. Applies greedy Non-Maximum Suppression (NMS) with threshold 0.30.
    """
    bundles = {
        "transcript": transcript_bundle,
        "conversation": conversation_bundle,
        "visual": visual_bundle,
        "prosody": prosody_bundle,
    }

    # Determine available modalities at episode level (modalities with actual proposal evidence).
    # Missing modalities are treated as unavailable rather than negative evidence.
    active_modalities = [m_name for m_name, b in bundles.items() if len(b.proposals) > 0]
    num_available = len(active_modalities)
    denom = float(num_available) if num_available > 0 else 1.0

    # 1. Collect all proposals satisfying duration bounds
    base_proposals: List[Tuple[str, TemporalProposal]] = []
    for m_name, b in bundles.items():
        for p in b.proposals:
            dur = p.end_time - p.start_time
            if min_duration_sec <= dur <= max_duration_sec:
                base_proposals.append((m_name, p))

    # Candidate intervals: raw proposals + pairwise intersections
    candidate_intervals: List[Tuple[float, float]] = []
    seen_spans = set()

    for _, p in base_proposals:
        span = (round(p.start_time, 4), round(p.end_time, 4))
        if span not in seen_spans:
            seen_spans.add(span)
            candidate_intervals.append((p.start_time, p.end_time))

    # Pairwise intersections across distinct modalities
    for i in range(len(base_proposals)):
        m1, p1 = base_proposals[i]
        for j in range(i + 1, len(base_proposals)):
            m2, p2 = base_proposals[j]
            if m1 == m2:
                continue
            inter_s = max(p1.start_time, p2.start_time)
            inter_e = min(p1.end_time, p2.end_time)
            dur = inter_e - inter_s
            if min_duration_sec <= dur <= max_duration_sec:
                span = (round(inter_s, 4), round(inter_e, 4))
                if span not in seen_spans:
                    seen_spans.add(span)
                    candidate_intervals.append((inter_s, inter_e))

    if not candidate_intervals:
        return []

    # 2. Score candidate intervals via unweighted mean modality coverage across available modalities
    scored_candidates = []
    for s, e in candidate_intervals:
        interval_dur = e - s
        if interval_dur <= 0.0:
            continue

        modality_scores = []
        coverage_meta = {}
        for m_name, b in bundles.items():
            max_cov = 0.0
            for p in b.proposals:
                overlap = max(0.0, min(e, p.end_time) - max(s, p.start_time))
                cov = float(p.confidence_estimate) * (overlap / interval_dur)
                if cov > max_cov:
                    max_cov = cov
            modality_scores.append(max_cov)
            coverage_meta[f"{m_name}_coverage"] = round(max_cov, 4)

        unweighted_score = sum(modality_scores) / denom
        coverage_meta["active_modalities_count"] = num_available
        scored_candidates.append({
            "start": s,
            "end": e,
            "duration": interval_dur,
            "score": unweighted_score,
            "metadata": coverage_meta,
        })

    # Sort descending by unweighted score
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)

    # 3. Greedy Non-Maximum Suppression (NMS)
    selected_spans: List[EvaluationSpan] = []
    kept = []

    for cand in scored_candidates:
        s1, e1 = cand["start"], cand["end"]
        suppressed = False
        for k in kept:
            s2, e2 = k["start"], k["end"]
            iou = compute_temporal_iou(s1, e1, s2, e2)
            if iou > nms_iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(cand)
            selected_spans.append(
                EvaluationSpan(
                    span_id=f"all_experts_no_mter_span_{len(selected_spans)}",
                    start_time=cand["start"],
                    end_time=cand["end"],
                    duration=cand["duration"],
                    confidence=cand["score"],
                    mode_name="all_experts_no_mter",
                    metadata=cand["metadata"],
                )
            )

    return selected_spans


class AblationRunner:
    """
    Executes experiment modes across pre-extracted or live bundles.
    """

    def __init__(
        self,
        baseline_selector: Optional[LegacyTranscriptBaselineSelector] = None,
        reasoner: Optional[MTERReasoner] = None,
        refiner: Optional[BoundaryRefiner] = None,
    ):
        self.baseline_selector = baseline_selector or LegacyTranscriptBaselineSelector()
        self.reasoner = reasoner or MTERReasoner(mter_config=config.mter)
        self.refiner = refiner or BoundaryRefiner(config=config.refiner)

    def run_mode(
        self,
        mode: ExperimentMode,
        run_dir: str,
        video_duration: float,
        sample_id: str,
        cache_dir: Optional[str] = None,
        top_k: Optional[int] = None,
        selection_mode: Optional[str] = None,
    ) -> List[EvaluationSpan]:
        """
        Executes a specific experiment mode on a given run directory.
        Allows specifying top_k and selection_mode to configure candidate pool size.
        """
        run_path = pathlib.Path(run_dir)

        # System A: Transcript-Only Semantic Baseline
        if mode in ("baseline", "system_a_transcript_only"):
            trans_evidence_file = run_path / "transcript_evidence.json"
            if trans_evidence_file.exists():
                try:
                    with open(trans_evidence_file, "r", encoding="utf-8") as f:
                        bundle_data = json.load(f)
                    trans_bundle = ExpertEvidenceBundle(**bundle_data)
                    cands = [p for p in trans_bundle.proposals if 10.0 <= (p.end_time - p.start_time) <= 60.0]
                    cands.sort(
                        key=lambda p: (
                            p.confidence_estimate,
                            -abs((p.end_time - p.start_time) - 30.0),
                            -p.start_time,
                            p.proposal_id,
                        ),
                        reverse=True,
                    )
                    kept: List[TemporalProposal] = []
                    spans: List[EvaluationSpan] = []
                    for p in cands:
                        s1, e1 = p.start_time, p.end_time
                        suppressed = False
                        for k in kept:
                            if compute_temporal_iou(s1, e1, k.start_time, k.end_time) > 0.30:
                                suppressed = True
                                break
                        if not suppressed:
                            kept.append(p)
                            spans.append(
                                EvaluationSpan(
                                    span_id=f"transcript_expert_span_{len(spans)}",
                                    start_time=p.start_time,
                                    end_time=p.end_time,
                                    duration=p.end_time - p.start_time,
                                    confidence=float(p.confidence_estimate),
                                    mode_name=mode,
                                    metadata={
                                        "proposal_id": p.proposal_id,
                                        "evidence_type": p.evidence_type,
                                        "explanation": p.explanation,
                                    },
                                )
                            )
                    return spans
                except Exception as e:
                    logger.warning(
                        f"Failed to load transcript_evidence.json in {run_dir}, falling back to legacy baseline selector: {e}"
                    )

            trans_file = run_path / "transcript.json"
            if not trans_file.exists():
                raise FileNotFoundError(f"Missing transcript.json in {run_dir}")
            with open(trans_file, "r", encoding="utf-8") as f:
                transcript_data = TranscriptData(**json.load(f))
            spans = self.baseline_selector.run(
                transcript=transcript_data,
                video_duration=video_duration,
                sample_id=sample_id,
                cache_dir=cache_dir,
            )
            for s in spans:
                s.mode_name = mode
            return spans

        # Load the four expert bundles
        def load_bundle(name: str) -> ExpertEvidenceBundle:
            fpath = run_path / f"{name}_evidence.json"
            if fpath.exists():
                with open(fpath, "r", encoding="utf-8") as f:
                    return ExpertEvidenceBundle(**json.load(f))
            return ExpertEvidenceBundle(expert_name=name, proposals=[])

        trans_bundle = load_bundle("transcript")
        conv_bundle = load_bundle("conversation")
        vis_bundle = load_bundle("visual")
        pros_bundle = load_bundle("prosody")

        empty_bundle = lambda n: ExpertEvidenceBundle(expert_name=n, proposals=[])

        # System B: Four-expert simple multimodal aggregation (non-MTER)
        if mode in ("all_experts_no_mter", "system_b_simple_aggregation"):
            spans = run_all_experts_no_mter_algorithm(
                transcript_bundle=trans_bundle,
                conversation_bundle=conv_bundle,
                visual_bundle=vis_bundle,
                prosody_bundle=pros_bundle,
                video_duration=video_duration,
            )
            for s in spans:
                s.mode_name = mode
            return spans

        # MTER based modes (with respective modality masking)
        if mode == "transcript_conversation":
            bundle = build_common_evidence_bundle(
                video_id=sample_id,
                total_duration=video_duration,
                transcript_evidence=trans_bundle,
                conversation_evidence=conv_bundle,
                visual_evidence=empty_bundle("visual"),
                prosody_evidence=empty_bundle("prosody"),
            )
        elif mode == "transcript_conversation_visual":
            bundle = build_common_evidence_bundle(
                video_id=sample_id,
                total_duration=video_duration,
                transcript_evidence=trans_bundle,
                conversation_evidence=conv_bundle,
                visual_evidence=vis_bundle,
                prosody_evidence=empty_bundle("prosody"),
            )
        elif mode == "transcript_conversation_prosody":
            bundle = build_common_evidence_bundle(
                video_id=sample_id,
                total_duration=video_duration,
                transcript_evidence=trans_bundle,
                conversation_evidence=conv_bundle,
                visual_evidence=empty_bundle("visual"),
                prosody_evidence=pros_bundle,
            )
        elif mode in ("mter_full", "system_c_mter", "system_d_mter_refined"):
            bundle = build_common_evidence_bundle(
                video_id=sample_id,
                total_duration=video_duration,
                transcript_evidence=trans_bundle,
                conversation_evidence=conv_bundle,
                visual_evidence=vis_bundle,
                prosody_evidence=pros_bundle,
            )
        else:
            raise ValueError(f"Unknown experiment mode: {mode}")

        mter_output, _ = self.reasoner.reason(
            bundle=bundle,
            top_k=top_k,
            selection_mode=selection_mode,
        )

        # Load raw modal data if available in run_path for fine-grained boundary refiner
        trans_data = None
        t_file = run_path / "transcript.json"
        if t_file.exists():
            try:
                with open(t_file, "r", encoding="utf-8") as f:
                    trans_data = TranscriptData(**json.load(f))
            except Exception:
                pass

        conv_data = None
        c_file = run_path / "conversation.json"
        if c_file.exists():
            try:
                with open(c_file, "r", encoding="utf-8") as f:
                    conv_data = ConversationData(**json.load(f))
            except Exception:
                pass

        vis_data = None
        v_file = run_path / "frames.json"
        if v_file.exists():
            try:
                with open(v_file, "r", encoding="utf-8") as f:
                    vis_data = VisualData(**json.load(f))
            except Exception:
                pass

        pros_data = None
        p_file = run_path / "prosody.json"
        if p_file.exists():
            try:
                with open(p_file, "r", encoding="utf-8") as f:
                    pros_data = ProsodyData(**json.load(f))
            except Exception:
                pass

        spans: List[EvaluationSpan] = []
        for idx, cand in enumerate(mter_output.candidates):
            if mode == "system_d_mter_refined":
                # System D: MTER with Boundary Refiner enabled
                refined_cand = self.refiner.refine_candidate(
                    candidate=cand,
                    transcript_data=trans_data,
                    conversation_data=conv_data,
                    visual_data=vis_data,
                    prosody_data=pros_data,
                    video_duration=video_duration,
                    bundle=bundle,
                )
                bref = refined_cand.boundary_refinement
                spans.append(
                    EvaluationSpan(
                        span_id=f"{mode}_span_{idx}",
                        start_time=refined_cand.proposed_start,
                        end_time=refined_cand.proposed_end,
                        duration=refined_cand.duration,
                        confidence=refined_cand.confidence_estimate,
                        mode_name=mode,
                        metadata={
                            "decision_summary": refined_cand.decision_summary,
                            "modality_diversity": refined_cand.support_metrics.modality_diversity if refined_cand.support_metrics else 0.0,
                            "original_start": bref.original_start if bref else cand.proposed_start,
                            "original_end": bref.original_end if bref else cand.proposed_end,
                            "start_delta_sec": bref.start_delta_sec if bref else 0.0,
                            "end_delta_sec": bref.end_delta_sec if bref else 0.0,
                            "signals_used": bref.signals_used if bref else [],
                            "rationale": bref.rationale if bref else "",
                        },
                    )
                )
            else:
                spans.append(
                    EvaluationSpan(
                        span_id=f"{mode}_span_{idx}",
                        start_time=cand.proposed_start,
                        end_time=cand.proposed_end,
                        duration=cand.duration,
                        confidence=cand.confidence_estimate,
                        mode_name=mode,
                        metadata={
                            "decision_summary": cand.decision_summary,
                            "modality_diversity": cand.support_metrics.modality_diversity if cand.support_metrics else 0.0,
                        },
                    )
                )

        return spans
