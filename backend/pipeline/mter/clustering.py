"""
ClipSense W5: Boundary Clustering & Event-Region Grouping

Core functionalities:
1. Grouping proposals into coherent temporal Event Regions (overlap graph connected components).
   - Enforces that start and end boundaries are evaluated only within temporally compatible regions.
   - Prevents pairing an unrelated start from one event with an unrelated end from another.
2. 1D Deterministic Boundary Clustering:
   - Groups nearby boundary proposals within a configurable tolerance (boundary_cluster_tolerance_sec).
   - Preserves member proposals, distinct supporting experts, spread, and continuous center timestamps.
   - No aggregate evidence score collapse inside BoundaryCluster.
3. Structured Conflict Detection:
   - Identifies early-start disagreement, late-end disagreement, and broad-vs-narrow interval conflicts.
"""

import logging
from typing import Dict, List, Literal, Optional, Set, Tuple

from core.schemas import (
    BoundaryCluster,
    BoundaryProposalRef,
    CommonEvidenceBundle,
    EventRegion,
    TemporalConflict,
    TemporalProposal,
)

logger = logging.getLogger("clipsense.mter.clustering")


def extract_all_proposals_flat(
    bundle: CommonEvidenceBundle,
) -> List[Tuple[str, TemporalProposal]]:
    """
    Returns a flat list of (expert_name, proposal) preserving modality identity.
    Sorted deterministically by start_time, end_time, proposal_id.
    """
    items: List[Tuple[str, TemporalProposal]] = []
    for p in bundle.transcript_evidence.proposals:
        items.append(("transcript", p))
    for p in bundle.conversation_evidence.proposals:
        items.append(("conversation", p))
    for p in bundle.visual_evidence.proposals:
        items.append(("visual", p))
    for p in bundle.prosody_evidence.proposals:
        items.append(("prosody", p))

    # Deterministic sorting
    items.sort(key=lambda x: (x[1].start_time, x[1].end_time, x[0], x[1].proposal_id))
    return items


def are_proposals_temporally_compatible(
    p1: TemporalProposal,
    p2: TemporalProposal,
    min_iou: float = 0.15,
    min_overlap_ratio: float = 0.45,
    max_center_gap_sec: float = 20.0,
) -> bool:
    """
    Determines if two proposals p1 and p2 are temporally compatible to belong to the same event.
    
    Prevents broad proposals from daisy-chaining unrelated events across the video.
    Two proposals are compatible if:
        1. Temporal IoU >= min_iou (e.g. 0.15)
        OR
        2. (Overlap / min(duration1, duration2) >= min_overlap_ratio (e.g. 0.45))
           AND (|center1 - center2| <= max_center_gap_sec (e.g. 20.0s))
    """
    s1, e1 = p1.start_time, p1.end_time
    s2, e2 = p2.start_time, p2.end_time

    overlap = max(0.0, min(e1, e2) - max(s1, s2))
    if overlap <= 0.0:
        return False

    union = max(e1, e2) - min(s1, s2)
    iou = (overlap / union) if union > 0.0 else 0.0
    if iou >= min_iou:
        return True

    min_dur = min(p1.duration, p2.duration)
    overlap_ratio = (overlap / min_dur) if min_dur > 0.0 else 0.0
    c1 = (s1 + e1) / 2.0
    c2 = (s2 + e2) / 2.0
    center_gap = abs(c1 - c2)

    if overlap_ratio >= min_overlap_ratio and center_gap <= max_center_gap_sec:
        return True

    return False


def group_proposals_into_event_regions(
    proposals_with_expert: List[Tuple[str, TemporalProposal]],
    event_compatibility_min_iou: float = 0.15,
    event_compatibility_min_overlap_ratio: float = 0.45,
    event_compatibility_max_center_gap_sec: float = 20.0,
) -> List[Tuple[EventRegion, List[Tuple[str, TemporalProposal]]]]:
    """
    Groups temporally compatible proposals into coherent Event Regions via graph connected components.
    
    CRITICAL: Does NOT merge proposals based on arbitrary temporal overlap or simple linear daisy-chaining.
    Constructs an undirected temporal compatibility graph where an edge exists between p_i and p_j
    iff `are_proposals_temporally_compatible` returns True.
    
    Connected components of this graph represent distinct, temporally coherent candidate event regions.
    
    Returns:
        List of (EventRegion, list of member (expert_name, proposal) tuples)
    """
    if not proposals_with_expert:
        return []

    n = len(proposals_with_expert)
    # Build adjacency list
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        p_i = proposals_with_expert[i][1]
        for j in range(i + 1, n):
            p_j = proposals_with_expert[j][1]
            if are_proposals_temporally_compatible(
                p_i,
                p_j,
                min_iou=event_compatibility_min_iou,
                min_overlap_ratio=event_compatibility_min_overlap_ratio,
                max_center_gap_sec=event_compatibility_max_center_gap_sec,
            ):
                adj[i].append(j)
                adj[j].append(i)

    # Compute connected components (deterministic traversal)
    visited: Set[int] = set()
    raw_components: List[List[int]] = []

    for i in range(n):
        if i not in visited:
            comp: List[int] = []
            queue: List[int] = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                # Sort neighbors deterministically
                neighbors = sorted(adj[curr])
                for nxt in neighbors:
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            raw_components.append(comp)

    # Build region data
    component_data = []
    for comp in raw_components:
        member_props = [proposals_with_expert[idx] for idx in comp]
        # Sort member proposals deterministically by start, end, expert, proposal_id
        member_props.sort(key=lambda x: (x[1].start_time, x[1].end_time, x[0], x[1].proposal_id))
        region_start = min(p[1].start_time for p in member_props)
        region_end = max(p[1].end_time for p in member_props)
        component_data.append((region_start, region_end, member_props))

    # Sort components deterministically by region_start, then region_end
    component_data.sort(key=lambda x: (x[0], x[1]))

    result: List[Tuple[EventRegion, List[Tuple[str, TemporalProposal]]]] = []
    for idx, (reg_start, reg_end, member_props) in enumerate(component_data):
        member_ids = [p[1].proposal_id for p in member_props]
        experts = sorted(list(set(p[0] for p in member_props)))

        event_region = EventRegion(
            region_id=f"event_region_{idx}",
            start_time=reg_start,
            end_time=reg_end,
            member_proposal_ids=member_ids,
            contributing_experts=experts,
        )
        result.append((event_region, member_props))

    return result


def cluster_boundary_candidates(
    boundary_refs: List[BoundaryProposalRef],
    tolerance_sec: float,
    boundary_type: Literal["start", "end"],
    cluster_prefix: str = "cluster",
) -> List[BoundaryCluster]:
    """
    Performs deterministic 1D temporal clustering on boundary proposals.
    
    Proposals within `tolerance_sec` of the running cluster are grouped together.
    The cluster center is computed as the mean continuous timestamp of its members.
    
    CRITICAL: Does NOT collapse confidence into an aggregate score here;
    preserves all original member proposal references.
    """
    if not boundary_refs:
        return []

    # Deterministic sort by timestamp, then expert_name, then proposal_id
    sorted_refs = sorted(
        boundary_refs,
        key=lambda r: (r.timestamp, r.expert_name, r.proposal_id)
    )

    clusters: List[List[BoundaryProposalRef]] = []
    curr_group: List[BoundaryProposalRef] = [sorted_refs[0]]

    for ref in sorted_refs[1:]:
        # Compare to the first element of current cluster or running min/max
        if ref.timestamp - curr_group[0].timestamp <= tolerance_sec:
            curr_group.append(ref)
        else:
            clusters.append(curr_group)
            curr_group = [ref]

    if curr_group:
        clusters.append(curr_group)

    result_clusters: List[BoundaryCluster] = []
    for idx, grp in enumerate(clusters):
        timestamps = [r.timestamp for r in grp]
        min_t = min(timestamps)
        max_t = max(timestamps)
        spread = max_t - min_t
        center = sum(timestamps) / len(timestamps)
        supporting_exps = sorted(list(set(r.expert_name for r in grp)))

        cluster = BoundaryCluster(
            cluster_id=f"{cluster_prefix}_{boundary_type}_{idx}",
            boundary_type=boundary_type,
            cluster_center=center,
            min_time=min_t,
            max_time=max_t,
            spread_sec=spread,
            supporting_experts=supporting_exps,
            member_proposals=grp,
        )
        result_clusters.append(cluster)

    return result_clusters


def detect_evidence_conflicts(
    start_clusters: List[BoundaryCluster],
    end_clusters: List[BoundaryCluster],
    region_proposals: List[Tuple[str, TemporalProposal]],
    event_region_id: str,
    conflict_threshold_sec: float = 4.0,
) -> List[TemporalConflict]:
    """
    Identifies structured cross-modal disagreements within a specific event region:
    1. Early-start disagreement: distinct expert start proposals differing by >= conflict_threshold_sec.
    2. Late-end disagreement: distinct expert end proposals differing by >= conflict_threshold_sec.
    3. Broad-vs-narrow interval disagreement: proposals in the region where one is >= 2.5x duration of another.

    CRITICAL REFERENCE INTEGRITY:
    - Every conflict records the enclosing `event_region_id`.
    - Every proposal referenced in `early_proposals` and `late_proposals` is the actual proposal object
      with its exact `proposal_id` and exact timestamp.
    """
    conflicts: List[TemporalConflict] = []

    # Map of proposal_id -> TemporalProposal for validation and direct property retrieval
    proposal_map: Dict[str, TemporalProposal] = {
        p.proposal_id: p for _, p in region_proposals
    }

    # 1. Early-start disagreements
    all_start_refs = [
        ref
        for cl in start_clusters
        for ref in cl.member_proposals
    ]
    if len(all_start_refs) >= 2:
        sorted_starts = sorted(all_start_refs, key=lambda r: (r.timestamp, r.expert_name, r.proposal_id))
        earliest = sorted_starts[0]
        latest = sorted_starts[-1]
        delta_start = latest.timestamp - earliest.timestamp

        if delta_start >= conflict_threshold_sec and earliest.expert_name != latest.expert_name:
            early_members = [r for r in sorted_starts if r.timestamp <= earliest.timestamp + (delta_start / 3.0)]
            late_members = [r for r in sorted_starts if r.timestamp >= latest.timestamp - (delta_start / 3.0)]
            conflicts.append(
                TemporalConflict(
                    event_region_id=event_region_id,
                    boundary_type="start",
                    conflict_type="early_start_disagreement",
                    early_proposals=early_members,
                    late_proposals=late_members,
                    temporal_delta_sec=delta_start,
                    description=(
                        f"Start boundary discrepancy of {delta_start:.2f}s in {event_region_id}: "
                        f"{earliest.expert_name} ({earliest.proposal_id}) starts early at {earliest.timestamp:.3f}s while "
                        f"{latest.expert_name} ({latest.proposal_id}) starts later at {latest.timestamp:.3f}s."
                    ),
                )
            )

    # 2. Late-end disagreements
    all_end_refs = [
        ref
        for cl in end_clusters
        for ref in cl.member_proposals
    ]
    if len(all_end_refs) >= 2:
        sorted_ends = sorted(all_end_refs, key=lambda r: (r.timestamp, r.expert_name, r.proposal_id))
        earliest = sorted_ends[0]
        latest = sorted_ends[-1]
        delta_end = latest.timestamp - earliest.timestamp

        if delta_end >= conflict_threshold_sec and earliest.expert_name != latest.expert_name:
            early_members = [r for r in sorted_ends if r.timestamp <= earliest.timestamp + (delta_end / 3.0)]
            late_members = [r for r in sorted_ends if r.timestamp >= latest.timestamp - (delta_end / 3.0)]
            conflicts.append(
                TemporalConflict(
                    event_region_id=event_region_id,
                    boundary_type="end",
                    conflict_type="late_end_disagreement",
                    early_proposals=early_members,
                    late_proposals=late_members,
                    temporal_delta_sec=delta_end,
                    description=(
                        f"End boundary discrepancy of {delta_end:.2f}s in {event_region_id}: "
                        f"{earliest.expert_name} ({earliest.proposal_id}) ends early at {earliest.timestamp:.3f}s while "
                        f"{latest.expert_name} ({latest.proposal_id}) extends later to {latest.timestamp:.3f}s."
                    ),
                )
            )

    # 3. Broad-vs-narrow interval disagreement
    if len(region_proposals) >= 2:
        durations = [(exp, p) for exp, p in region_proposals]
        shortest = min(durations, key=lambda x: (x[1].duration, x[1].proposal_id))
        longest = max(durations, key=lambda x: (x[1].duration, x[1].proposal_id))

        if shortest[0] != longest[0] and longest[1].duration >= 2.5 * shortest[1].duration:
            ref_short = BoundaryProposalRef(
                expert_name=shortest[0],
                proposal_id=shortest[1].proposal_id,
                timestamp=shortest[1].start_time,
                confidence_estimate=shortest[1].confidence_estimate,
                evidence_type=shortest[1].evidence_type,
            )
            ref_long = BoundaryProposalRef(
                expert_name=longest[0],
                proposal_id=longest[1].proposal_id,
                timestamp=longest[1].start_time,
                confidence_estimate=longest[1].confidence_estimate,
                evidence_type=longest[1].evidence_type,
            )
            conflicts.append(
                TemporalConflict(
                    event_region_id=event_region_id,
                    boundary_type="interval_breadth",
                    conflict_type="broad_vs_narrow_interval",
                    early_proposals=[ref_short],
                    late_proposals=[ref_long],
                    temporal_delta_sec=longest[1].duration - shortest[1].duration,
                    description=(
                        f"Interval breadth disparity in {event_region_id}: "
                        f"{longest[0]} ({longest[1].proposal_id}) spans {longest[1].duration:.2f}s [{longest[1].start_time:.3f}s -> {longest[1].end_time:.3f}s] "
                        f"whereas {shortest[0]} ({shortest[1].proposal_id}) proposes a localized {shortest[1].duration:.2f}s interval [{shortest[1].start_time:.3f}s -> {shortest[1].end_time:.3f}s]."
                    ),
                )
            )

    return conflicts

