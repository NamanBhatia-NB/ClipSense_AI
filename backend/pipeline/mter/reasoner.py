"""
ClipSense W5: Multimodal Temporal Evidence Reasoner (MTER) Prototype

Connects independent evidence from:
- Transcript Expert
- Conversation Expert
- Visual Expert
- Prosody Expert

========================================================================================
MTER CANDIDATE EVALUATION FORMULA & WEIGHTING SYSTEM
========================================================================================

Architecture Description:
Deterministic, internally consistent MTER prototype with inspectable evidence and configurable
heuristic parameters.

For any valid candidate interval I = [s, e] of duration D = e - s generated from start cluster
S and end cluster T within an event region, the composite evidence score is defined as:

    RawCompositeScore(I) =
          w_mod * ModalityCoverage(I)
        + w_div * ModalityDiversity(I)
        + w_bnd * BoundaryAgreement(S, T)
        + w_ctx * InheritedContextualEvidence(I)
        - w_cnf * ConflictPenalty(I)
        - w_dur * DurationPenalty(I)

    CompositeScore(I) = max(0.0, min(1.0, RawCompositeScore(I)))

BOUNDING TO [0.0, 1.0] GUARANTEE:
- The positive terms have maximum potential sum:
    w_mod (0.30) + w_div (0.25) + w_bnd (0.20) + w_ctx (0.20) = 0.95 <= 1.0.
- The subtractive penalties can subtract up to:
    w_cnf (0.15) + w_dur (0.05) = 0.20.
- Thus RawCompositeScore(I) falls in [-0.20, 0.95].
- The exact operation max(0.0, min(1.0, RawCompositeScore(I))) is evaluated for EVERY candidate
  interval, explicitly bounding CompositeScore(I) to [0.0, 1.0] for ALL evaluated candidates.
- "Normalized Evidence Strength" is justified because the output score represents a strictly
  normalized relative evidence metric on [0.0, 1.0] (where 0.0 indicates zero support/fully penalized
  and 1.0 indicates maximal multimodal consensus), NOT a calibrated probability.

Where:
1. ModalityCoverage(I) = mean(support_transcript, support_conversation, support_visual, support_prosody)
   Where for each modality m:
       support_m = max_{p in proposals(m)} [ confidence(p) * min(1.0, |p ∩ I| / D) ]
       coverage_m = max_{p in proposals(m)} min(1.0, |p ∩ I| / D)
       iou_m = max_{p in proposals(m)} [ |p ∩ I| / |p ∪ I| ]

2. ModalityDiversity(I) = (count of modalities with support_m >= active_modality_threshold) / 4.0
   Preserves unimodal proposals (0.25) while rewarding multi-expert convergence (up to 1.0).

3. BoundaryAgreement(S, T) = 0.5 * (
       (count(experts in S) / 4.0) * (1.0 / (1.0 + spread(S))) +
       (count(experts in T) / 4.0) * (1.0 / (1.0 + spread(T)))
   )
   Measures the multi-expert backing and temporal tightness of the boundary clusters.

4. Separated Boundary Support Types:
   - boundary_activity_support: Evidence from visual (scene transitions/shifts) or prosody (vocal emphasis)
     occurring within boundary_cluster_tolerance_sec of s or e.
   - linguistic_boundary_support: Evidence from transcript (word boundaries) or conversation (speech segments)
     coinciding within semantic_alignment_tolerance_sec of s or e.
     (Measures linguistic/temporal alignment of speech timestamps only, not independent semantic verification).

5. InheritedContextualEvidence(I):
   Normalized checklist [0.0, 1.0] of contextual features inherited from W3 proposals:
   - natural_semantic_start (spoken word or turn onset near start)
   - sufficient_context (transcript completeness >= threshold and self_contained)
   - important_content_retained (retains key conversational or semantic events)
   - complete_conversational_payoff (discourse_unit_complete or payoff completed)
   - no_mid_sentence_ending (does not cut off mid-word or mid-utterance)
   NOTE: This represents evidence inherited from contributing W3 expert proposals,
   not an independent second-pass semantic verification.

6. ConflictPenalty(I) = min(1.0, sum of temporal deltas of conflicts directly affecting S or T / 10.0)

7. DurationPenalty(I) = |D - target_candidate_duration_sec| / max_candidate_duration_sec
   Operates strictly as a secondary soft tie-breaker subordinate to evidence metrics.

CONFIGURABLE PROTOTYPE PARAMETERS (not learned or empirically validated research constants):
All weights and thresholds are configurable prototype parameters in MTERConfig:
- weight_modality_support: 0.30
- weight_modality_diversity: 0.25
- weight_boundary_agreement: 0.20
- weight_contextual_completeness: 0.20
- weight_conflict_penalty: 0.15
- weight_duration_penalty: 0.05
========================================================================================
"""

import logging
import re
import time
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from app.config import MTERConfig, config
from core.schemas import (
    BoundaryCluster,
    BoundaryProposalRef,
    CandidateSupportMetrics,
    CommonEvidenceBundle,
    EventRegion,
    EvidenceLedger,
    MTERCandidate,
    MTEROutput,
    SemanticVerificationCriteria,
    TemporalConflict,
    TemporalProposal,
)
from pipeline.mter.clustering import (
    cluster_boundary_candidates,
    detect_evidence_conflicts,
    extract_all_proposals_flat,
    group_proposals_into_event_regions,
)

logger = logging.getLogger("clipsense.mter.reasoner")


def compute_temporal_interval_iou(s1: float, e1: float, s2: float, e2: float) -> float:
    """
    Computes Intersection-over-Union (IoU) between two 1D temporal intervals.
    Used for Non-Maximum Suppression to prevent duplicate candidate exports.
    """
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0.0 else 0.0


def compute_union_duration(intervals: List[Tuple[float, float]]) -> float:
    """
    Computes the total continuous duration of the union of 1D intervals,
    correctly merging overlapping and contiguous spans to measure non-redundant coverage.
    """
    if not intervals:
        return 0.0
    valid = [(s, e) for s, e in intervals if e > s]
    if not valid:
        return 0.0
    sorted_intervals = sorted(valid, key=lambda x: (x[0], x[1]))
    merged: List[Tuple[float, float]] = []
    curr_s, curr_e = sorted_intervals[0]
    for s, e in sorted_intervals[1:]:
        if s <= curr_e:
            curr_e = max(curr_e, e)
        else:
            merged.append((curr_s, curr_e))
            curr_s, curr_e = s, e
    merged.append((curr_s, curr_e))
    return float(sum(e - s for s, e in merged))


class MTERReasoner:
    """
    Multimodal Temporal Evidence Reasoner (MTER) Prototype.
    
    Performs event-region grouping, intra-region boundary clustering,
    explicit per-modality support accounting, and transparent agreement/conflict reasoning.
    """

    def __init__(self, mter_config: Optional[MTERConfig] = None):
        self.config = mter_config or config.mter

    def reason(
        self,
        bundle: CommonEvidenceBundle,
        top_k: Optional[int] = None,
        nms_iou_threshold: Optional[float] = None,
        selection_mode: Optional[str] = None,
    ) -> Tuple[MTEROutput, EvidenceLedger]:
        """
        Executes the full W5 MTER reasoning procedure on a CommonEvidenceBundle.
        Supports configurable Top-K highlight selection with Non-Maximum Suppression (W8).
        """
        start_time_proc = time.time()
        video_duration = bundle.total_duration

        # 1. Flatten and deterministically sort all proposals
        all_proposals_flat = extract_all_proposals_flat(bundle)

        # Apply ablation mode filtering if active
        proposals_flat = self._apply_ablation_filtering(all_proposals_flat)
        total_proposals_count = len(proposals_flat)

        # Determine modality availability (modalities that actually emitted proposals for this video)
        available_modalities = [
            m for m in ["transcript", "conversation", "visual", "prosody"]
            if getattr(bundle, f"{m}_evidence", None) and getattr(bundle, f"{m}_evidence").proposals
        ]
        if not available_modalities and proposals_flat:
            available_modalities = sorted(list({p[0] for p in proposals_flat}))
        modality_availability = {
            m: (m in available_modalities) for m in ["transcript", "conversation", "visual", "prosody"]
        }
        n_available = max(1, len(available_modalities))

        ledger = EvidenceLedger(
            run_id=bundle.video_id,
            video_duration=video_duration,
            total_proposals=total_proposals_count,
            modality_availability=modality_availability,
        )

        if not proposals_flat:
            logger.warning(f"No proposals available for reasoning on {bundle.video_id} (ablation: {self.config.ablation_mode})")
            ledger.decision_summary = f"No expert evidence proposals available for reasoning (ablation: {self.config.ablation_mode})."
            ledger.filtering_stages = {
                "initial_candidate_intervals": 0,
                "passed_bounds_and_duration": 0,
                "passed_acceptance_threshold": 0,
                "survived_nms_suppression": 0,
                "final_selected_highlights": 0,
            }
            ledger.redundancy_metrics = {
                "total_selected_duration_sec": 0.0,
                "union_selected_duration_sec": 0.0,
                "redundancy_ratio": 1.0,
                "temporal_coverage_ratio": 0.0,
            }
            output = MTEROutput(
                video_id=bundle.video_id,
                candidates=[],
                ledger=ledger,
                execution_time_sec=time.time() - start_time_proc,
            )
            return output, ledger

        # 2. Group proposals into coherent temporal Event Regions
        event_regions_with_props = group_proposals_into_event_regions(
            proposals_flat,
            event_compatibility_min_iou=self.config.event_compatibility_min_iou,
            event_compatibility_min_overlap_ratio=self.config.event_compatibility_min_overlap_ratio,
            event_compatibility_max_center_gap_sec=self.config.event_compatibility_max_center_gap_sec,
        )
        ledger.event_regions = [r[0] for r in event_regions_with_props]

        all_start_clusters: List[BoundaryCluster] = []
        all_end_clusters: List[BoundaryCluster] = []
        all_conflicts: List[TemporalConflict] = []

        candidate_evaluations: List[Dict[str, Any]] = []
        total_pairs_generated = 0
        rejected_by_rule = {
            "inverted_bounds": 0,
            "out_of_video_bounds": 0,
            "duration_too_short": 0,
            "duration_too_long": 0,
            "below_acceptance_threshold": 0,
            "nms_overlap_suppressed": 0,
            "budget_cap_exceeded": 0,
        }

        # 3. Analyze each event region independently
        for region_idx, (event_region, region_props) in enumerate(event_regions_with_props):
            # Extract start and end boundary references
            start_refs: List[BoundaryProposalRef] = []
            end_refs: List[BoundaryProposalRef] = []

            for exp_name, p in region_props:
                start_refs.append(
                    BoundaryProposalRef(
                        expert_name=exp_name,
                        proposal_id=p.proposal_id,
                        timestamp=p.start_time,
                        confidence_estimate=p.confidence_estimate,
                        evidence_type=p.evidence_type,
                    )
                )
                end_refs.append(
                    BoundaryProposalRef(
                        expert_name=exp_name,
                        proposal_id=p.proposal_id,
                        timestamp=p.end_time,
                        confidence_estimate=p.confidence_estimate,
                        evidence_type=p.evidence_type,
                    )
                )

            # Intra-region boundary clustering
            start_clusters = cluster_boundary_candidates(
                start_refs,
                tolerance_sec=self.config.boundary_cluster_tolerance_sec,
                boundary_type="start",
                cluster_prefix=f"r{region_idx}",
            )
            end_clusters = cluster_boundary_candidates(
                end_refs,
                tolerance_sec=self.config.boundary_cluster_tolerance_sec,
                boundary_type="end",
                cluster_prefix=f"r{region_idx}",
            )

            all_start_clusters.extend(start_clusters)
            all_end_clusters.extend(end_clusters)

            # Conflict detection scoped strictly to this event region
            region_conflicts = detect_evidence_conflicts(
                start_clusters,
                end_clusters,
                region_props,
                event_region_id=event_region.region_id,
                conflict_threshold_sec=self.config.conflict_threshold_sec,
            )
            all_conflicts.extend(region_conflicts)

            # 4. Generate candidate intervals ONLY within this event region
            total_pairs_generated += len(start_clusters) * len(end_clusters)
            for s_cl in start_clusters:
                for e_cl in end_clusters:
                    c_start = s_cl.cluster_center
                    c_end = e_cl.cluster_center
                    dur = c_end - c_start

                    # Validate bounds
                    if c_start >= c_end:
                        rejected_by_rule["inverted_bounds"] += 1
                        ledger.rejected_intervals.append({
                            "region_id": event_region.region_id,
                            "start": c_start,
                            "end": c_end,
                            "duration": dur,
                            "reason": f"Start timestamp ({c_start:.3f}s) is >= end timestamp ({c_end:.3f}s).",
                        })
                        continue

                    if c_start < 0.0 or c_end > video_duration + 1e-4:
                        rejected_by_rule["out_of_video_bounds"] += 1
                        ledger.rejected_intervals.append({
                            "region_id": event_region.region_id,
                            "start": c_start,
                            "end": c_end,
                            "duration": dur,
                            "reason": f"Interval [{c_start:.3f}s -> {c_end:.3f}s] exceeds video bounds [0.0s -> {video_duration:.3f}s].",
                        })
                        continue

                    if dur < self.config.min_candidate_duration_sec:
                        rejected_by_rule["duration_too_short"] += 1
                        ledger.rejected_intervals.append({
                            "region_id": event_region.region_id,
                            "start": c_start,
                            "end": c_end,
                            "duration": dur,
                            "reason": (
                                f"Duration ({dur:.2f}s) is shorter than min_candidate_duration_sec "
                                f"({self.config.min_candidate_duration_sec:.1f}s)."
                            ),
                        })
                        continue

                    if dur > self.config.max_candidate_duration_sec:
                        rejected_by_rule["duration_too_long"] += 1
                        ledger.rejected_intervals.append({
                            "region_id": event_region.region_id,
                            "start": c_start,
                            "end": c_end,
                            "duration": dur,
                            "reason": (
                                f"Duration ({dur:.2f}s) exceeds max_candidate_duration_sec "
                                f"({self.config.max_candidate_duration_sec:.1f}s)."
                            ),
                        })
                        continue

                    # Evaluate candidate interval with normalized available modalities
                    eval_record = self._evaluate_candidate_interval(
                        c_start,
                        c_end,
                        dur,
                        s_cl,
                        e_cl,
                        region_props,
                        region_conflicts,
                        event_region,
                        available_modalities=available_modalities,
                        n_available=n_available,
                    )
                    candidate_evaluations.append(eval_record)

        ledger.start_clusters = all_start_clusters
        ledger.end_clusters = all_end_clusters
        ledger.conflicts = all_conflicts
        ledger.total_detected_conflicts = len(all_conflicts)
        ledger.evaluated_candidates = [
            {
                "candidate_id": ev["candidate_id"],
                "region_id": ev["region_id"],
                "start": ev["start"],
                "end": ev["end"],
                "duration": ev["duration"],
                "contributing_experts": ev["contributing_experts"],
                "metrics": ev["metrics"].model_dump(),
            }
            for ev in candidate_evaluations
        ]

        # 5. Selection Protocol (Supporting Natural Selection vs Fixed-Budget Selection)
        # Sort candidates deterministically: composite_score descending, then diversity, then duration proximity
        candidate_evaluations.sort(
            key=lambda x: (
                x["metrics"].composite_score,
                x["metrics"].modality_diversity,
                -abs(x["duration"] - self.config.target_candidate_duration_sec),
            ),
            reverse=True,
        )

        mode = selection_mode or getattr(self.config, "selection_mode", "fixed_budget")
        if mode == "fixed_budget":
            effective_mode = "fixed_budget"
            effective_top_k = top_k if top_k is not None else getattr(self.config, "top_k", 1)
        else:
            effective_mode = "natural"
            effective_top_k = top_k

        effective_nms_iou = (
            nms_iou_threshold
            if nms_iou_threshold is not None
            else getattr(self.config, "nms_iou_threshold", 0.3)
        )
        acceptance_threshold = getattr(self.config, "natural_acceptance_threshold", 0.40)
        max_duration_budget = getattr(self.config, "max_total_selected_duration_sec", None)

        selected_candidates: List[MTERCandidate] = []
        candidate_tokens_map: Dict[str, Set[str]] = {}
        cumulative_duration = 0.0

        if not candidate_evaluations:
            logger.warning("No candidate intervals satisfied duration and temporal constraints.")
            ledger.decision_summary = (
                "No candidate interval satisfied temporal bounds and duration constraints "
                f"[{self.config.min_candidate_duration_sec:.1f}s - {self.config.max_candidate_duration_sec:.1f}s]."
            )
        elif effective_mode == "natural":
            # Natural selection mode: variable number of candidates (including zero) based on acceptance rule, NMS, and duration budget
            for ev in candidate_evaluations:
                c_score = ev["metrics"].composite_score
                if c_score < acceptance_threshold:
                    rejected_by_rule["below_acceptance_threshold"] += 1
                    continue

                # Check temporal IoU, asymmetric interval overlap, and transcript near-duplicate suppression
                is_suppressed, reason, supp_by, details = self._check_candidate_suppression(
                    ev, selected_candidates, candidate_tokens_map, effective_nms_iou
                )
                if is_suppressed:
                    rejected_by_rule["nms_overlap_suppressed"] += 1
                    ledger.rejected_intervals.append({
                        "region_id": ev["region_id"],
                        "start": ev["start"],
                        "end": ev["end"],
                        "duration": ev["duration"],
                        "reason": reason,
                    })
                    ledger.suppression_records.append({
                        "candidate_id": ev["candidate_id"],
                        "suppressed_by": supp_by,
                        "reason": reason,
                        **details,
                    })
                    continue

                # Check cumulative duration budget if configured
                if max_duration_budget is not None and (cumulative_duration + ev["duration"] > max_duration_budget):
                    rejected_by_rule["budget_cap_exceeded"] += 1
                    continue

                cand_obj = self._build_candidate_from_eval(ev, len(all_conflicts), available_modalities)
                selected_candidates.append(cand_obj)
                candidate_tokens_map[cand_obj.candidate_id] = ev.get("transcript_tokens", set())
                cumulative_duration += ev["duration"]

        else:
            # Fixed-budget selection mode: at most effective_top_k candidates passing NMS and duration budget
            for ev in candidate_evaluations:
                if len(selected_candidates) >= effective_top_k:
                    break

                is_suppressed, reason, supp_by, details = self._check_candidate_suppression(
                    ev, selected_candidates, candidate_tokens_map, effective_nms_iou
                )
                if is_suppressed:
                    rejected_by_rule["nms_overlap_suppressed"] += 1
                    ledger.rejected_intervals.append({
                        "region_id": ev["region_id"],
                        "start": ev["start"],
                        "end": ev["end"],
                        "duration": ev["duration"],
                        "reason": reason,
                    })
                    ledger.suppression_records.append({
                        "candidate_id": ev["candidate_id"],
                        "suppressed_by": supp_by,
                        "reason": reason,
                        **details,
                    })
                    continue

                if max_duration_budget is not None and (cumulative_duration + ev["duration"] > max_duration_budget):
                    rejected_by_rule["budget_cap_exceeded"] += 1
                    continue

                cand_obj = self._build_candidate_from_eval(ev, len(all_conflicts), available_modalities)
                selected_candidates.append(cand_obj)
                candidate_tokens_map[cand_obj.candidate_id] = ev.get("transcript_tokens", set())
                cumulative_duration += ev["duration"]

        # Calculate filtering waterfall and redundancy diagnostics
        passed_acc = len([ev for ev in candidate_evaluations if ev["metrics"].composite_score >= acceptance_threshold])
        filtering_stages = {
            "initial_candidate_intervals": total_pairs_generated,
            "passed_bounds_and_duration": len(candidate_evaluations),
            "passed_acceptance_threshold": passed_acc,
            "survived_nms_suppression": len(selected_candidates),
            "final_selected_highlights": len(selected_candidates),
        }

        union_dur = compute_union_duration([(c.proposed_start, c.proposed_end) for c in selected_candidates])
        tot_dur = sum(c.duration for c in selected_candidates)
        redundancy_metrics = {
            "total_selected_duration_sec": round(tot_dur, 2),
            "union_selected_duration_sec": round(union_dur, 2),
            "redundancy_ratio": round(tot_dur / union_dur, 3) if union_dur > 0.0 else 1.0,
            "temporal_coverage_ratio": round(union_dur / video_duration, 4) if video_duration > 0.0 else 0.0,
        }

        ledger.selection_mode = effective_mode
        ledger.filtering_stages = filtering_stages
        ledger.redundancy_metrics = redundancy_metrics
        ledger.rejected_by_rule = rejected_by_rule
        ledger.selected_candidate = selected_candidates[0] if selected_candidates else None

        if selected_candidates:
            top_eval = [ev for ev in candidate_evaluations if ev["candidate_id"] == selected_candidates[0].candidate_id][0]
            ledger.conflicts_affecting_candidate = top_eval.get("affecting_conflicts", [])
            ledger.unresolved_conflicts_affecting_candidate = top_eval.get("unresolved_conflicts", [])
            ledger.decision_summary = selected_candidates[0].decision_summary
        else:
            ledger.decision_summary = (
                f"No candidate interval satisfied selection criteria in {effective_mode} mode "
                f"(evaluated: {len(candidate_evaluations)}, acceptance threshold: {acceptance_threshold:.2f})."
            )

        output = MTEROutput(
            video_id=bundle.video_id,
            candidates=selected_candidates,
            ledger=ledger,
            execution_time_sec=time.time() - start_time_proc,
        )

        return output, ledger

    def _check_candidate_suppression(
        self,
        candidate_eval: Dict[str, Any],
        selected_candidates: List[MTERCandidate],
        candidate_tokens_map: Dict[str, Set[str]],
        effective_nms_iou: float,
    ) -> Tuple[bool, Optional[str], Optional[str], Dict[str, float]]:
        """
        Evaluates temporal IoU, asymmetric interval overlap, and transcript token similarity
        against already selected candidates to suppress redundant near-duplicate clips.
        Avoids suppressing distinct clips that merely discuss the same broad topic.
        """
        c_start = candidate_eval["start"]
        c_end = candidate_eval["end"]
        c_dur = candidate_eval["duration"]
        c_tokens = candidate_eval.get("transcript_tokens", set())

        max_shorter_ratio = getattr(self.config, "max_shorter_interval_overlap_ratio", 0.30)
        transcript_sim_thresh = getattr(self.config, "transcript_similarity_threshold", 0.50)

        for existing in selected_candidates:
            iou = compute_temporal_interval_iou(c_start, c_end, existing.proposed_start, existing.proposed_end)
            intersection = max(0.0, min(c_end, existing.proposed_end) - max(c_start, existing.proposed_start))
            shorter_dur = min(c_dur, existing.duration)
            shorter_ratio = (intersection / shorter_dur) if shorter_dur > 0 else 0.0

            e_tokens = candidate_tokens_map.get(existing.candidate_id, set())
            union_tokens = c_tokens | e_tokens
            inter_tokens = c_tokens & e_tokens
            token_jaccard = (len(inter_tokens) / len(union_tokens)) if union_tokens else 0.0

            details = {
                "temporal_iou": round(float(iou), 3),
                "shorter_overlap_ratio": round(float(shorter_ratio), 3),
                "transcript_similarity": round(float(token_jaccard), 3),
            }

            # 1. Standard Temporal IoU NMS
            if iou > effective_nms_iou:
                reason = f"Temporal IoU ({iou:.3f} > threshold {effective_nms_iou:.2f}) with selected candidate {existing.candidate_id}"
                return True, reason, existing.candidate_id, details

            # 2. Asymmetric interval overlap on shorter clip
            if shorter_ratio > max_shorter_ratio and (iou > 0.08 or token_jaccard > 0.20):
                reason = (
                    f"Asymmetric interval overlap ({shorter_ratio*100:.1f}% of shorter interval > threshold {max_shorter_ratio*100:.1f}%) "
                    f"with {existing.candidate_id} (tIoU: {iou:.2f}, token similarity: {token_jaccard:.2f})"
                )
                return True, reason, existing.candidate_id, details

            # 3. High transcript token overlap for near-duplicate content
            if token_jaccard >= transcript_sim_thresh and intersection > 0.0:
                reason = (
                    f"Near-duplicate transcript similarity ({token_jaccard:.2f} >= threshold {transcript_sim_thresh:.2f}) "
                    f"with {existing.candidate_id} (temporal intersection: {intersection:.2f}s)"
                )
                return True, reason, existing.candidate_id, details

        return False, None, None, {}

    def _build_candidate_from_eval(
        self,
        eval_data: Dict[str, Any],
        total_conflicts_count: int,
        available_modalities: Optional[List[str]] = None,
    ) -> MTERCandidate:
        """Constructs a validated MTERCandidate from an evaluated candidate dictionary."""
        decision_summary = self._synthesize_decision_summary(eval_data, total_conflicts_count, available_modalities)
        affecting = eval_data.get("affecting_conflicts", [])
        unresolved = eval_data.get("unresolved_conflicts", [])
        metrics = eval_data["metrics"]

        return MTERCandidate(
            candidate_id=eval_data["candidate_id"],
            proposed_start=eval_data["start"],
            proposed_end=eval_data["end"],
            duration=eval_data["duration"],
            confidence_estimate=min(1.0, max(0.0, metrics.composite_score)),
            contributing_experts=eval_data["contributing_experts"],
            temporal_agreement_iou=eval_data["iou"],
            modality_ious=eval_data["modality_ious"],
            modality_coverages=eval_data["modality_coverages"],
            boundary_activity_support=eval_data["boundary_activity_support"],
            linguistic_boundary_support=eval_data["linguistic_boundary_support"],
            modalities_present=eval_data["modalities_present"],
            strong_support_modalities=eval_data["strong_support_modalities"],
            moderate_support_modalities=eval_data["moderate_support_modalities"],
            weak_support_modalities=eval_data["weak_support_modalities"],
            support_metrics=metrics,
            conflict_notes=eval_data["conflict_notes"],
            detected_conflicts_count=total_conflicts_count,
            conflicts_affecting_candidate_count=len(affecting),
            unresolved_conflicts_count=len(unresolved),
            semantic_verification=eval_data["semantic_verification"],
            semantic_verification_status="passed" if metrics.inherited_contextual_evidence >= 0.6 else "flagged",
            decision_summary=decision_summary,
        )

    def _apply_ablation_filtering(
        self,
        proposals_flat: List[Tuple[str, TemporalProposal]],
    ) -> List[Tuple[str, TemporalProposal]]:
        """Filters input proposals based on configured ablation mode."""
        mode = self.config.ablation_mode

        if mode == "transcript_only":
            return [p for p in proposals_flat if p[0] == "transcript"]
        elif mode == "transcript_conversation":
            return [p for p in proposals_flat if p[0] in ("transcript", "conversation")]
        elif mode == "transcript_conversation_visual":
            return [p for p in proposals_flat if p[0] in ("transcript", "conversation", "visual")]
        elif mode == "transcript_conversation_prosody":
            return [p for p in proposals_flat if p[0] in ("transcript", "conversation", "prosody")]
        elif mode in ("mter_full", "no_mter_heuristic"):
            # Check enabled_modalities if customized
            enabled = set(self.config.enabled_modalities)
            return [p for p in proposals_flat if p[0] in enabled]
        return proposals_flat

    def _evaluate_candidate_interval(
        self,
        start: float,
        end: float,
        duration: float,
        start_cluster: BoundaryCluster,
        end_cluster: BoundaryCluster,
        region_props: List[Tuple[str, TemporalProposal]],
        region_conflicts: List[TemporalConflict],
        event_region: EventRegion,
        available_modalities: Optional[List[str]] = None,
        n_available: int = 4,
    ) -> Dict[str, Any]:
        """
        Evaluates support, diversity, boundary agreement, contextual completeness,
        and conflict penalties for a candidate interval [start, end].
        
        Properly normalizes across available modalities so absent/unrun modalities
        are not silently penalized as negative evidence.
        """
        if available_modalities is None:
            available_modalities = ["transcript", "conversation", "visual", "prosody"]
            n_available = 4

        overlapping_props: Dict[str, List[TemporalProposal]] = {
            "transcript": [],
            "conversation": [],
            "visual": [],
            "prosody": [],
        }

        # Calculate per-modality IoU and coverage
        modality_ious: Dict[str, float] = {}
        modality_coverages: Dict[str, float] = {}
        modality_supports: Dict[str, float] = {}

        for exp_name, p in region_props:
            overlap = max(0.0, min(end, p.end_time) - max(start, p.start_time))
            if overlap > 0.0:
                overlapping_props[exp_name].append(p)

        for m in ["transcript", "conversation", "visual", "prosody"]:
            props_m = overlapping_props[m]
            if not props_m:
                modality_ious[m] = 0.0
                modality_coverages[m] = 0.0
                modality_supports[m] = 0.0
            else:
                best_iou = 0.0
                best_cov = 0.0
                best_sup = 0.0
                for p in props_m:
                    overlap = max(0.0, min(end, p.end_time) - max(start, p.start_time))
                    union = max(end, p.end_time) - min(start, p.start_time)
                    iou = (overlap / union) if union > 0.0 else 0.0
                    cov = min(1.0, overlap / duration)
                    sup = p.confidence_estimate * cov

                    if iou > best_iou:
                        best_iou = iou
                    if cov > best_cov:
                        best_cov = cov
                    if sup > best_sup:
                        best_sup = sup

                modality_ious[m] = float(best_iou)
                modality_coverages[m] = float(best_cov)
                modality_supports[m] = float(best_sup)

        # Aggregate mean modality coverage support normalized by available modalities
        mean_mod_support = (
            sum(modality_supports[m] for m in available_modalities) / n_available
            if available_modalities else 0.0
        )

        # Modality presence: available modalities with temporal overlap / coverage > 0
        modalities_present = [
            m for m in available_modalities
            if modality_coverages[m] > 0.0
        ]

        # Support strength tiers
        strong_support_modalities = [
            m for m in available_modalities
            if modality_supports[m] >= 0.50
        ]
        moderate_support_modalities = [
            m for m in available_modalities
            if 0.20 <= modality_supports[m] < 0.50
        ]
        weak_support_modalities = [
            m for m in available_modalities
            if 0.05 <= modality_supports[m] < 0.20
        ]

        # Modality Diversity: fraction of available modalities actively contributing
        active_modalities = [
            m for m in available_modalities
            if modality_supports[m] >= self.config.active_modality_threshold
        ]
        modality_diversity = len(active_modalities) / 4.0

        # Aggregate temporal agreement IoU: mean of active modality IoUs
        active_ious = [modality_ious[m] for m in active_modalities if modality_ious[m] > 0.0]
        temporal_agreement_iou = (sum(active_ious) / len(active_ious)) if active_ious else 0.0

        # 2. Boundary Agreement
        s_avail_count = len([e for e in start_cluster.supporting_experts if e in available_modalities])
        e_avail_count = len([e for e in end_cluster.supporting_experts if e in available_modalities])
        s_tightness = 1.0 / (1.0 + start_cluster.spread_sec)
        e_tightness = 1.0 / (1.0 + end_cluster.spread_sec)

        boundary_agreement = 0.5 * (
            (s_avail_count / n_available) * s_tightness +
            (e_avail_count / n_available) * e_tightness
        )

        # 3. Explicit Boundary Support Separation
        act_support_start = 0.0
        act_support_end = 0.0
        for ref in start_cluster.member_proposals:
            if ref.expert_name in ("visual", "prosody"):
                act_support_start = max(act_support_start, ref.confidence_estimate)
        for ref in end_cluster.member_proposals:
            if ref.expert_name in ("visual", "prosody"):
                act_support_end = max(act_support_end, ref.confidence_estimate)
        boundary_activity_support = 0.5 * (act_support_start + act_support_end)

        ling_support_start = 0.0
        ling_support_end = 0.0
        for ref in start_cluster.member_proposals:
            if ref.expert_name in ("transcript", "conversation"):
                ling_support_start = max(ling_support_start, ref.confidence_estimate)
        for ref in end_cluster.member_proposals:
            if ref.expert_name in ("transcript", "conversation"):
                ling_support_end = max(ling_support_end, ref.confidence_estimate)
        linguistic_boundary_support = 0.5 * (ling_support_start + ling_support_end)

        # 4. Contextual Evidence Inherited from Contributing Proposals
        has_semantic_start = False
        for p in overlapping_props["transcript"] + overlapping_props["conversation"]:
            if abs(p.start_time - start) <= self.config.semantic_alignment_tolerance_sec:
                has_semantic_start = True
                break
        if not has_semantic_start and (overlapping_props["transcript"] or overlapping_props["conversation"]):
            has_semantic_start = True

        has_sufficient_context = False
        for p in overlapping_props["transcript"]:
            feats = p.supporting_features or {}
            if feats.get("contextual_completeness", 0.0) >= self.config.min_context_completeness_threshold or feats.get("self_contained", False):
                has_sufficient_context = True
                break
        if not has_sufficient_context and overlapping_props["conversation"]:
            has_sufficient_context = True

        has_important_content = False
        for p in overlapping_props["prosody"]:
            if p.start_time >= start - 1.0 and p.end_time <= end + 1.0:
                has_important_content = True
                break
        if not has_important_content and overlapping_props["visual"]:
            has_important_content = True
        if not has_important_content and overlapping_props["transcript"]:
            has_important_content = True

        has_conversational_payoff = False
        for p in overlapping_props["conversation"]:
            feats = p.supporting_features or {}
            if feats.get("discourse_unit_complete", False):
                has_conversational_payoff = True
                break
        if not has_conversational_payoff and overlapping_props["transcript"]:
            for p in overlapping_props["transcript"]:
                if p.supporting_features.get("self_contained", False):
                    has_conversational_payoff = True
                    break

        no_mid_sentence = False
        for p in overlapping_props["transcript"] + overlapping_props["conversation"]:
            if abs(p.end_time - end) <= self.config.semantic_alignment_tolerance_sec:
                no_mid_sentence = True
                break
        if not no_mid_sentence and (overlapping_props["transcript"] or overlapping_props["conversation"]):
            if "transcript" in end_cluster.supporting_experts or "conversation" in end_cluster.supporting_experts:
                no_mid_sentence = True

        context_checks = [
            has_semantic_start,
            has_sufficient_context,
            has_important_content,
            has_conversational_payoff,
            no_mid_sentence,
        ]
        inherited_contextual_score = sum(1.0 for c in context_checks if c) / 5.0

        inherited_prop_ids = [
            p.proposal_id
            for props_list in overlapping_props.values()
            for p in props_list
        ]

        semantic_crit = SemanticVerificationCriteria(
            natural_semantic_start=has_semantic_start,
            sufficient_context=has_sufficient_context,
            important_content_retained=has_important_content,
            complete_conversational_payoff=has_conversational_payoff,
            no_mid_sentence_ending=no_mid_sentence,
            inherited_from_proposals=inherited_prop_ids,
        )

        # 5. Distinguish Conflicts: Detected in Region vs Affecting Candidate vs Unresolved
        affecting_conflicts: List[TemporalConflict] = []
        unresolved_conflicts: List[TemporalConflict] = []
        conflict_notes: List[str] = []
        conflict_pen = 0.0

        start_member_ids = {r.proposal_id for r in start_cluster.member_proposals}
        end_member_ids = {r.proposal_id for r in end_cluster.member_proposals}

        for conf in region_conflicts:
            if conf.event_region_id != event_region.region_id:
                continue

            conf_prop_ids = [ref.proposal_id for ref in conf.early_proposals + conf.late_proposals]
            if not all(pid in event_region.member_proposal_ids for pid in conf_prop_ids):
                continue

            affects_start = (conf.boundary_type == "start") and (
                any(abs(ref.timestamp - start) <= self.config.boundary_cluster_tolerance_sec
                    for ref in conf.early_proposals + conf.late_proposals)
                or any(ref.proposal_id in start_member_ids
                       for ref in conf.early_proposals + conf.late_proposals)
            )
            affects_end = (conf.boundary_type == "end") and (
                any(abs(ref.timestamp - end) <= self.config.boundary_cluster_tolerance_sec
                    for ref in conf.early_proposals + conf.late_proposals)
                or any(ref.proposal_id in end_member_ids
                       for ref in conf.early_proposals + conf.late_proposals)
            )
            affects_breadth = (conf.boundary_type == "interval_breadth") and (
                any(p.start_time >= start - 1.0 and p.end_time <= end + 1.0
                    for p in overlapping_props["prosody"] + overlapping_props["visual"])
            )

            if affects_start or affects_end or affects_breadth:
                affecting_conflicts.append(conf)
                if conf.temporal_delta_sec >= self.config.conflict_threshold_sec:
                    unresolved_conflicts.append(conf)
                    conflict_notes.append(conf.description)
                    conflict_pen += min(1.0, conf.temporal_delta_sec / 10.0)

        conflict_pen = min(1.0, conflict_pen)

        # 6. Duration Penalty (soft preference)
        dur_pen = abs(duration - self.config.target_candidate_duration_sec) / self.config.max_candidate_duration_sec

        # 7. Composite Score Calculation (explicitly using documented weights)
        raw_composite = (
            self.config.weight_modality_support * mean_mod_support +
            self.config.weight_modality_diversity * modality_diversity +
            self.config.weight_boundary_agreement * boundary_agreement +
            self.config.weight_contextual_completeness * inherited_contextual_score -
            self.config.weight_conflict_penalty * conflict_pen -
            self.config.weight_duration_penalty * dur_pen
        )
        composite = max(0.0, min(1.0, raw_composite))

        # Detailed component breakdown for inspectable diagnostics
        score_components = {
            "mean_modality_support_unweighted": round(float(mean_mod_support), 4),
            "modality_diversity_unweighted": round(float(modality_diversity), 4),
            "boundary_agreement_unweighted": round(float(boundary_agreement), 4),
            "inherited_contextual_unweighted": round(float(inherited_contextual_score), 4),
            "conflict_penalty_unweighted": round(float(conflict_pen), 4),
            "duration_penalty_unweighted": round(float(dur_pen), 4),
            "modality_support_contrib": round(float(self.config.weight_modality_support * mean_mod_support), 4),
            "modality_diversity_contrib": round(float(self.config.weight_modality_diversity * modality_diversity), 4),
            "boundary_agreement_contrib": round(float(self.config.weight_boundary_agreement * boundary_agreement), 4),
            "contextual_completeness_contrib": round(float(self.config.weight_contextual_completeness * inherited_contextual_score), 4),
            "conflict_penalty_contrib": round(float(-self.config.weight_conflict_penalty * conflict_pen), 4),
            "duration_penalty_contrib": round(float(-self.config.weight_duration_penalty * dur_pen), 4),
            "raw_composite_score": round(float(raw_composite), 4),
            "composite_score": round(float(composite), 4),
        }

        metrics = CandidateSupportMetrics(
            transcript_support=modality_supports["transcript"],
            conversation_support=modality_supports["conversation"],
            visual_support=modality_supports["visual"],
            prosody_support=modality_supports["prosody"],
            transcript_iou=modality_ious["transcript"],
            conversation_iou=modality_ious["conversation"],
            visual_iou=modality_ious["visual"],
            prosody_iou=modality_ious["prosody"],
            transcript_coverage=modality_coverages["transcript"],
            conversation_coverage=modality_coverages["conversation"],
            visual_coverage=modality_coverages["visual"],
            prosody_coverage=modality_coverages["prosody"],
            boundary_activity_support=boundary_activity_support,
            linguistic_boundary_support=linguistic_boundary_support,
            modalities_present=modalities_present,
            strong_support_modalities=strong_support_modalities,
            moderate_support_modalities=moderate_support_modalities,
            weak_support_modalities=weak_support_modalities,
            modality_diversity=modality_diversity,
            boundary_agreement=boundary_agreement,
            inherited_contextual_evidence=inherited_contextual_score,
            conflict_penalty=conflict_pen,
            duration_penalty=dur_pen,
            composite_score=composite,
            score_components=score_components,
            score_type="heuristic_evidence_strength",
        )

        # Extract content tokens for near-duplicate transcript redundancy checking
        cand_tokens: Set[str] = set()
        stop_words = {
            "the", "and", "a", "an", "to", "in", "it", "of", "that", "this", "is", "was",
            "for", "on", "with", "as", "at", "by", "from", "be", "are", "or", "will",
            "my", "you", "they", "we", "he", "she", "me", "him", "her", "us", "them", "but", "so"
        }
        for tp in overlapping_props["transcript"]:
            raw_text = f"{tp.explanation or ''} {tp.supporting_features.get('start_word', '')} {tp.supporting_features.get('end_word', '')}"
            for token in re.findall(r"\b[a-zA-Z]{3,}\b", raw_text.lower()):
                if token not in stop_words:
                    cand_tokens.add(token)

        cand_id = f"mter_cand_{event_region.region_id}_{start:.2f}_{end:.2f}"

        return {
            "candidate_id": cand_id,
            "region_id": event_region.region_id,
            "start": start,
            "end": end,
            "duration": duration,
            "contributing_experts": active_modalities,
            "iou": temporal_agreement_iou,
            "modality_ious": modality_ious,
            "modality_coverages": modality_coverages,
            "boundary_activity_support": boundary_activity_support,
            "linguistic_boundary_support": linguistic_boundary_support,
            "modalities_present": modalities_present,
            "strong_support_modalities": strong_support_modalities,
            "moderate_support_modalities": moderate_support_modalities,
            "weak_support_modalities": weak_support_modalities,
            "metrics": metrics,
            "conflict_notes": conflict_notes,
            "affecting_conflicts": affecting_conflicts,
            "unresolved_conflicts": unresolved_conflicts,
            "semantic_verification": semantic_crit,
            "start_cluster": start_cluster,
            "end_cluster": end_cluster,
            "transcript_tokens": cand_tokens,
        }

    def _synthesize_decision_summary(
        self,
        eval_record: Dict[str, Any],
        total_detected_conflicts: int,
        available_modalities: Optional[List[str]] = None,
    ) -> str:
        """
        Synthesizes a concise, inspectable, factual decision summary distinguishing modality presence from support strength.
        Explicitly frames evidence score as a heuristic ranking metric rather than a calibrated probability.
        """
        metrics: CandidateSupportMetrics = eval_record["metrics"]
        start = eval_record["start"]
        end = eval_record["end"]
        dur = eval_record["duration"]

        if available_modalities is None:
            available_modalities = ["transcript", "conversation", "visual", "prosody"]

        avail_str = ", ".join(available_modalities)
        pres_str = ", ".join(metrics.modalities_present) if metrics.modalities_present else "none"
        strong_str = ", ".join(metrics.strong_support_modalities) if metrics.strong_support_modalities else "none"
        mod_str = ", ".join(metrics.moderate_support_modalities) if metrics.moderate_support_modalities else "none"
        weak_str = ", ".join(metrics.weak_support_modalities) if metrics.weak_support_modalities else "none"

        affecting_count = len(eval_record["affecting_conflicts"])
        unresolved_count = len(eval_record["unresolved_conflicts"])

        lines = [
            f"Interval [{start:.3f}s -> {end:.3f}s] ({dur:.2f}s) in {eval_record['region_id']} selected with heuristic evidence score {metrics.composite_score:.2f} (non-calibrated ranking score).",
            f"Available modalities ({len(available_modalities)}/4): {avail_str} | Modalities with evidence: {len(metrics.modalities_present)}/{len(available_modalities)} ({pres_str}).",
            f"Support strength breakdown: Strong: {strong_str} | Moderate: {mod_str} | Weak/secondary: {weak_str}.",
            f"Coverage: transcript={metrics.transcript_coverage:.2f}, conversation={metrics.conversation_coverage:.2f}, visual={metrics.visual_coverage:.2f}, prosody={metrics.prosody_coverage:.2f}.",
            f"Modality IoU: transcript={metrics.transcript_iou:.2f}, conversation={metrics.conversation_iou:.2f}, visual={metrics.visual_iou:.2f}, prosody={metrics.prosody_iou:.2f} (mean active IoU: {eval_record['iou']:.2f}).",
            f"Boundary support: activity={metrics.boundary_activity_support:.2f}, linguistic={metrics.linguistic_boundary_support:.2f} | Boundary agreement tightness: {metrics.boundary_agreement:.2f}.",
            f"Inherited contextual evidence: {metrics.inherited_contextual_evidence * 100:.0f}%.",
            f"Conflicts: {total_detected_conflicts} detected in region, {affecting_count} affecting candidate, {unresolved_count} unresolved.",
        ]

        if eval_record["conflict_notes"]:
            lines.append(f"Unresolved conflict notes: {'; '.join(eval_record['conflict_notes'][:2])}")

        return " ".join(lines)

