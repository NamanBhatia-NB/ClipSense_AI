"""
Evaluation Metrics and Bipartite Matching Protocol (W6)

Pure mathematical formulations for:
- Temporal IoU (continuous floating-point)
- Start and End boundary errors (absolute and signed bias)
- Duration error (absolute and relative)
- Tolerance Hit Rates (Hit@1s, Hit@2s, Hit@3s)
- Solely Hungarian bipartite matching protocol with documented cost and minimum acceptance threshold
"""

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau, spearmanr

logger = logging.getLogger("clipsense.evaluation.metrics")

from evaluation.schemas import (
    BoundaryRefinementAblationResult,
    EvaluationSpan,
    PairwiseMetricResult,
    ReferenceHighlight,
)

# Documented constant for research-valid bipartite matching acceptance
DEFAULT_MIN_MATCH_IOU: float = 0.10


def compute_temporal_iou(
    start_p: float, end_p: float, start_g: float, end_g: float
) -> float:
    """
    Computes exact continuous floating-point Temporal Intersection over Union (tIoU).
    """
    if start_p >= end_p or start_g >= end_g:
        return 0.0

    intersection = max(0.0, min(end_p, end_g) - max(start_p, start_g))
    union = max(end_p, end_g) - min(start_p, start_g)

    if union <= 0.0:
        return 0.0

    return float(intersection / union)


def compute_start_error(start_p: float, start_g: float) -> Tuple[float, float]:
    """
    Returns (absolute_error_sec, signed_bias_sec).
    Signed bias > 0 means late start; < 0 means early start.
    """
    signed_bias = float(start_p - start_g)
    return abs(signed_bias), signed_bias


def compute_end_error(end_p: float, end_g: float) -> Tuple[float, float]:
    """
    Returns (absolute_error_sec, signed_bias_sec).
    Signed bias > 0 means late cutoff; < 0 means early cutoff.
    """
    signed_bias = float(end_p - end_g)
    return abs(signed_bias), signed_bias


def compute_duration_error(dur_p: float, dur_g: float) -> Tuple[float, float]:
    """
    Returns (absolute_duration_error_sec, relative_duration_error).
    """
    abs_err = abs(float(dur_p - dur_g))
    rel_err = float(abs_err / dur_g) if dur_g > 0.0 else 0.0
    return abs_err, rel_err


def compute_boundary_hits(
    start_err: float, end_err: float, tolerances: Sequence[float] = (1.0, 2.0, 3.0)
) -> Dict[float, bool]:
    """
    Returns dict mapping tolerance tau to whether BOTH start and end boundaries fall within tau.
    """
    hits = {}
    for tau in tolerances:
        hits[tau] = (start_err <= tau) and (end_err <= tau)
    return hits


def compute_pairwise_metrics(
    pred: EvaluationSpan, ref: ReferenceHighlight
) -> PairwiseMetricResult:
    """
    Computes all pairwise metric attributes between one prediction and one reference span.
    """
    tiou = compute_temporal_iou(pred.start_time, pred.end_time, ref.start_time, ref.end_time)
    start_err, start_bias = compute_start_error(pred.start_time, ref.start_time)
    end_err, end_bias = compute_end_error(pred.end_time, ref.end_time)
    dur_err, rel_dur_err = compute_duration_error(pred.duration, ref.duration)
    hits = compute_boundary_hits(start_err, end_err, tolerances=(1.0, 2.0, 3.0))

    return PairwiseMetricResult(
        prediction_span_id=pred.span_id,
        reference_span_id=ref.span_id,
        pred_start=pred.start_time,
        pred_end=pred.end_time,
        pred_duration=pred.duration,
        ref_start=ref.start_time,
        ref_end=ref.end_time,
        ref_duration=ref.duration,
        temporal_iou=tiou,
        start_error_sec=start_err,
        end_error_sec=end_err,
        duration_error_sec=dur_err,
        rel_duration_error=rel_dur_err,
        signed_start_bias_sec=start_bias,
        signed_end_bias_sec=end_bias,
        hit_at_1s=hits[1.0],
        hit_at_2s=hits[2.0],
        hit_at_3s=hits[3.0],
    )


def match_spans_hungarian(
    predictions: List[EvaluationSpan],
    references: List[ReferenceHighlight],
    min_iou: float = DEFAULT_MIN_MATCH_IOU,
) -> Tuple[List[PairwiseMetricResult], List[str], List[str]]:
    """
    The SOLE research matching protocol for ClipSense W6.
    
    Constructs cost matrix Cost[i, j] = 1.0 - tIoU(P_i, G_j) and solves
    the minimum cost bipartite assignment using scipy.optimize.linear_sum_assignment.
    
    Assignments with tIoU < min_iou are rejected as invalid matches.
    
    Returns:
        (matched_pairwise_metrics, unmatched_prediction_ids, unmatched_reference_ids)
    """
    K = len(predictions)
    M = len(references)

    if K == 0 or M == 0:
        unmatched_preds = [p.span_id for p in predictions]
        unmatched_refs = [r.span_id for r in references]
        return [], unmatched_preds, unmatched_refs

    # Build cost matrix: Cost = 1.0 - tIoU
    cost_matrix = np.zeros((K, M), dtype=float)
    iou_matrix = np.zeros((K, M), dtype=float)

    for i, p in enumerate(predictions):
        for j, r in enumerate(references):
            iou = compute_temporal_iou(p.start_time, p.end_time, r.start_time, r.end_time)
            iou_matrix[i, j] = iou
            cost_matrix[i, j] = 1.0 - iou

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_results: List[PairwiseMetricResult] = []
    matched_pred_indices = set()
    matched_ref_indices = set()

    for i, j in zip(row_ind, col_ind):
        iou = iou_matrix[i, j]
        if iou >= min_iou:
            p = predictions[i]
            r = references[j]
            metric = compute_pairwise_metrics(p, r)
            matched_results.append(metric)
            matched_pred_indices.add(i)
            matched_ref_indices.add(j)

    unmatched_prediction_ids = [
        predictions[i].span_id for i in range(K) if i not in matched_pred_indices
    ]
    unmatched_reference_ids = [
        references[j].span_id for j in range(M) if j not in matched_ref_indices
    ]

    return matched_results, unmatched_prediction_ids, unmatched_reference_ids


def compute_precision_recall_f1_at_iou(
    matched_results: List[PairwiseMetricResult],
    num_predictions: int,
    num_references: int,
    iou_threshold: float = 0.50,
) -> Tuple[float, float, float]:
    """
    Computes Precision, Recall, and F1 at a specified tIoU threshold.
    """
    if num_predictions == 0 or num_references == 0:
        return 0.0, 0.0, 0.0

    true_positives = sum(1 for m in matched_results if m.temporal_iou >= iou_threshold)
    precision = float(true_positives / num_predictions)
    recall = float(true_positives / num_references)
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    return precision, recall, f1


def compute_intra_candidate_overlap_rate(spans: List[EvaluationSpan]) -> float:
    """
    Computes the mean pairwise tIoU among emitted highlight candidates for a single video.
    High intra-overlap indicates redundancy / duplicate candidate export.
    """
    if len(spans) <= 1:
        return 0.0

    total_iou = 0.0
    pairs_count = 0
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            iou = compute_temporal_iou(
                spans[i].start_time, spans[i].end_time,
                spans[j].start_time, spans[j].end_time,
            )
            total_iou += iou
            pairs_count += 1

    return float(total_iou / pairs_count) if pairs_count > 0 else 0.0


def spans_to_segment_importance_curve(
    spans: List[EvaluationSpan],
    video_duration: float,
    segment_duration_sec: float = 1.0,
    num_segments: Optional[int] = None,
) -> np.ndarray:
    """
    Projects continuous predicted highlight spans onto discrete temporal segment bins.
    Maps candidate confidence and overlap into a continuous salience curve on [0.0, 1.0].

    If num_segments is specified (e.g. 100 for Rhapsody), the episode is partitioned
    into exactly num_segments equal bins of duration (video_duration / num_segments).
    Otherwise, bins are formed by segment_duration_sec.
    """
    if num_segments is not None and num_segments > 0:
        total_bins = int(num_segments)
        eff_seg_dur = max(1e-6, video_duration / float(total_bins))
    else:
        total_bins = max(1, int(math.ceil(video_duration / segment_duration_sec)))
        eff_seg_dur = max(1e-6, segment_duration_sec)

    curve = np.zeros(total_bins, dtype=float)
    if video_duration <= 0.0 or not spans:
        return curve

    for span in spans:
        conf = float(span.confidence)
        if conf <= 0.0:
            continue
        # Clamping timestamps to [0.0, video_duration] to avoid out-of-bounds indexing
        s_clamped = max(0.0, min(float(span.start_time), video_duration))
        e_clamped = max(0.0, min(float(span.end_time), video_duration))
        if e_clamped <= s_clamped:
            continue

        start_bin = max(0, min(total_bins - 1, int(math.floor(s_clamped / eff_seg_dur))))
        end_bin = max(0, min(total_bins - 1, int(math.floor(e_clamped / eff_seg_dur))))

        for b in range(start_bin, end_bin + 1):
            bin_start = b * eff_seg_dur
            bin_end = video_duration if b == (total_bins - 1) else (b + 1) * eff_seg_dur
            overlap = max(0.0, min(e_clamped, bin_end) - max(s_clamped, bin_start))
            bin_dur = max(1e-6, bin_end - bin_start)
            segment_val = conf * (overlap / bin_dur)
            curve[b] = max(curve[b], segment_val)

    return curve


def compute_rhapsody_metrics(
    scores_pred: Sequence[float],
    scores_ref: Sequence[float],
    gt_threshold: float = 0.5,
    max_n_gt: int = 20,
    pred_threshold: Optional[float] = None,
    pred_top_k: Optional[int] = None,
    segment_prediction_k: Optional[int] = None,
    ranking_top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Computes official Rhapsody podcast highlight metrics (Park et al. COLM 2025).

    Args:
        scores_pred: Predicted importance scores for the 100 segments.
        scores_ref: Ground-truth continuous replay scores for the 100 segments.
        gt_threshold: Replay score threshold to classify a segment as highlight (default 0.5).
        max_n_gt: Maximum number of ground-truth segments to retain (default 20).
        pred_threshold: Optional threshold for predicted binary highlights.
        pred_top_k: (Deprecated alias for segment_prediction_k) Top predicted segments for highlights.
        segment_prediction_k: Number of top predicted segments to evaluate for binary metrics (Hit, P, R, F1).
        ranking_top_k: Optional cutoff for Average Precision ranking list. If None, evaluates across all 100 bins.

    Returns:
        Dictionary containing average_precision, hit_rate, precision, recall, f1, num_gt, num_pred.
    """
    n = len(scores_pred)
    if n != 100 or len(scores_ref) != 100:
        logger.warning(f"Rhapsody metrics expect 100 segments, got pred={len(scores_pred)}, ref={len(scores_ref)}")

    # 1. First-segment bias correction (Park et al. COLM 2025, Appendix A.3)
    scores_ref_arr = list(scores_ref)
    if len(scores_ref_arr) > 1:
        scores_ref_arr[0] = scores_ref_arr[1]

    # 2. Identify Ground Truth highlights G (strictly > gt_threshold, top max_n_gt)
    gt_candidates = [
        (idx, float(score))
        for idx, score in enumerate(scores_ref_arr)
        if float(score) > gt_threshold
    ]
    gt_candidates.sort(key=lambda x: (-x[1], x[0]))
    selected_gt = [idx for idx, _ in gt_candidates[:max_n_gt]]
    selected_gt_set = set(selected_gt)
    num_gt = len(selected_gt)

    # 3. Determine Ranked Predicted Highlights P
    ranked_pred_indices = sorted(
        range(n),
        key=lambda idx: (-float(scores_pred[idx]), idx),
    )

    # Determine binary prediction segments (for Hit Rate, Precision, Recall, F1)
    if pred_threshold is not None:
        binary_pred_indices = [idx for idx in ranked_pred_indices if float(scores_pred[idx]) >= pred_threshold]
    elif segment_prediction_k is not None:
        k = max(0, min(n, segment_prediction_k))
        binary_pred_indices = ranked_pred_indices[:k]
    elif pred_top_k is not None:
        k = max(0, min(n, pred_top_k))
        binary_pred_indices = ranked_pred_indices[:k]
    else:
        k = max(1, min(n, max_n_gt))
        binary_pred_indices = ranked_pred_indices[:k]

    num_pred = len(binary_pred_indices)
    pred_set = set(binary_pred_indices)
    overlap = len(pred_set.intersection(selected_gt_set))

    # 4. Average Precision (AP) across ranked segments
    if ranking_top_k is not None:
        ap_indices = ranked_pred_indices[: max(0, min(n, ranking_top_k))]
    elif pred_top_k is not None and segment_prediction_k is None:
        # Legacy backward-compatibility for existing tests specifically passing pred_top_k
        ap_indices = binary_pred_indices
    else:
        # Default protocol: evaluate ranking metric across complete ranked 100-bin predicted curve
        ap_indices = ranked_pred_indices

    if num_gt == 0:
        ap = 1.0 if len(ap_indices) == 0 else 0.0
    else:
        num_hits = 0
        sum_precisions = 0.0
        for rank, p_idx in enumerate(ap_indices, start=1):
            if p_idx in selected_gt_set:
                num_hits += 1
                sum_precisions += num_hits / float(rank)
        ap = float(sum_precisions / float(num_gt))

    # 5. Binary Metrics: Hit Rate, Precision, Recall, F1
    if num_gt == 0:
        hit_rate = 1.0 if num_pred == 0 else 0.0
        precision = 1.0 if num_pred == 0 else 0.0
        recall = 1.0 if num_pred == 0 else 0.0
        f1 = 1.0 if num_pred == 0 else 0.0
    elif num_pred == 0:
        hit_rate = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        hit_rate = 1.0 if overlap > 0 else 0.0
        precision = float(overlap / num_pred)
        recall = float(overlap / num_gt)
        if (precision + recall) > 0.0 and overlap > 0:
            f1 = float(2.0 * precision * recall / (precision + recall))
        else:
            f1 = 0.0

    return {
        "average_precision": round(ap, 4),
        "hit_rate": round(hit_rate, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "num_gt": num_gt,
        "num_pred": num_pred,
    }


def compute_spearman_rank_correlation(
    scores_pred: Sequence[float],
    scores_ref: Sequence[float],
) -> Tuple[float, Optional[float]]:
    """
    Computes Spearman's rank correlation coefficient (rho) and p-value.
    Used for evaluating segment-level importance against non-binary ground truth (e.g. TVSum).
    """
    if len(scores_pred) != len(scores_ref) or len(scores_pred) < 3:
        return 0.0, None

    # Handle constant vector edge case
    std_p = float(np.std(scores_pred))
    std_r = float(np.std(scores_ref))
    if std_p < 1e-9 or std_r < 1e-9:
        return 0.0, 1.0

    try:
        res = spearmanr(scores_pred, scores_ref)
        rho = float(res.correlation) if not np.isnan(res.correlation) else 0.0
        pval = float(res.pvalue) if not np.isnan(res.pvalue) else None
        return rho, pval
    except Exception:
        return 0.0, None


def compute_kendall_tau(
    scores_pred: Sequence[float],
    scores_ref: Sequence[float],
) -> Tuple[float, Optional[float]]:
    """
    Computes Kendall's rank correlation tau-b and p-value.
    Measures concordant versus discordant pairs between predicted and ground-truth curves.
    """
    if len(scores_pred) != len(scores_ref) or len(scores_pred) < 3:
        return 0.0, None

    std_p = float(np.std(scores_pred))
    std_r = float(np.std(scores_ref))
    if std_p < 1e-9 or std_r < 1e-9:
        return 0.0, 1.0

    try:
        res = kendalltau(scores_pred, scores_ref)
        tau = float(res.correlation) if not np.isnan(res.correlation) else 0.0
        pval = float(res.pvalue) if not np.isnan(res.pvalue) else None
        return tau, pval
    except Exception:
        return 0.0, None


def compute_top_k_overlap(
    scores_pred: Sequence[float],
    scores_ref: Sequence[float],
    top_ratio: float = 0.15,
) -> float:
    """
    Computes the temporal overlap between the top-k% segment bins of predicted
    and reference salience curves. Standard TVSum protocol specifies top_ratio=0.15 (top 15%).
    """
    n = len(scores_pred)
    if n == 0 or len(scores_ref) != n:
        return 0.0

    k = max(1, int(round(n * top_ratio)))
    pred_top_indices = set(np.argsort(scores_pred)[-k:])
    ref_top_indices = set(np.argsort(scores_ref)[-k:])

    intersection = len(pred_top_indices.intersection(ref_top_indices))
    return float(intersection / k)


def evaluate_boundary_refinement_effect(
    sample_id: str,
    unrefined_spans: List[EvaluationSpan],
    refined_spans: List[EvaluationSpan],
    reference_spans: Optional[List[ReferenceHighlight]] = None,
) -> BoundaryRefinementAblationResult:
    """
    Isolates and evaluates Boundary Refinement on identical candidate highlights.
    Measures boundary shift amounts, duration changes, and reference error reductions.
    """
    num_cands = min(len(unrefined_spans), len(refined_spans))
    if num_cands == 0:
        return BoundaryRefinementAblationResult(
            sample_id=sample_id,
            num_candidates=0,
            num_changed=0,
            mean_start_shift_sec=0.0,
            mean_end_shift_sec=0.0,
            mean_duration_shift_sec=0.0,
            unrefined_mean_tiou=None,
            refined_mean_tiou=None,
            tiou_delta=None,
            start_error_reduction_sec=None,
            end_error_reduction_sec=None,
            hit1_unrefined=None,
            hit1_refined=None,
            hit2_unrefined=None,
            hit2_refined=None,
            hit3_unrefined=None,
            hit3_refined=None,
        )

    start_shifts = []
    end_shifts = []
    dur_shifts = []
    num_changed = 0
    rationales = []

    for i in range(num_cands):
        u = unrefined_spans[i]
        r = refined_spans[i]
        ds = r.start_time - u.start_time
        de = r.end_time - u.end_time
        dd = r.duration - u.duration
        start_shifts.append(abs(ds))
        end_shifts.append(abs(de))
        dur_shifts.append(abs(dd))
        if abs(ds) > 1e-4 or abs(de) > 1e-4:
            num_changed += 1
        if "rationale" in r.metadata:
            rationales.append(str(r.metadata["rationale"]))

    mean_s_shift = float(np.mean(start_shifts)) if start_shifts else 0.0
    mean_e_shift = float(np.mean(end_shifts)) if end_shifts else 0.0
    mean_d_shift = float(np.mean(dur_shifts)) if dur_shifts else 0.0

    # Ground truth relative metrics if references are present and matched
    unref_tiou = ref_tiou = tiou_delta = None
    start_err_red = end_err_red = None
    h1_u = h1_r = h2_u = h2_r = h3_u = h3_r = None

    if reference_spans and len(reference_spans) > 0:
        u_matched, _, _ = match_spans_hungarian(unrefined_spans, reference_spans, min_iou=0.10)
        r_matched, _, _ = match_spans_hungarian(refined_spans, reference_spans, min_iou=0.10)

        if u_matched and r_matched:
            unref_tiou = float(np.mean([m.temporal_iou for m in u_matched]))
            ref_tiou = float(np.mean([m.temporal_iou for m in r_matched]))
            tiou_delta = round(ref_tiou - unref_tiou, 4)

            u_start_err = float(np.mean([m.start_error_sec for m in u_matched]))
            r_start_err = float(np.mean([m.start_error_sec for m in r_matched]))
            start_err_red = round(u_start_err - r_start_err, 4)

            u_end_err = float(np.mean([m.end_error_sec for m in u_matched]))
            r_end_err = float(np.mean([m.end_error_sec for m in r_matched]))
            end_err_red = round(u_end_err - r_end_err, 4)

            h1_u = round(float(sum(1 for m in u_matched if m.hit_at_1s) / len(u_matched) * 100.0), 2)
            h1_r = round(float(sum(1 for m in r_matched if m.hit_at_1s) / len(r_matched) * 100.0), 2)
            h2_u = round(float(sum(1 for m in u_matched if m.hit_at_2s) / len(u_matched) * 100.0), 2)
            h2_r = round(float(sum(1 for m in r_matched if m.hit_at_2s) / len(r_matched) * 100.0), 2)
            h3_u = round(float(sum(1 for m in u_matched if m.hit_at_3s) / len(u_matched) * 100.0), 2)
            h3_r = round(float(sum(1 for m in r_matched if m.hit_at_3s) / len(r_matched) * 100.0), 2)

    return BoundaryRefinementAblationResult(
        sample_id=sample_id,
        num_candidates=num_cands,
        num_changed=num_changed,
        mean_start_shift_sec=round(mean_s_shift, 4),
        mean_end_shift_sec=round(mean_e_shift, 4),
        mean_duration_shift_sec=round(mean_d_shift, 4),
        unrefined_mean_tiou=round(unref_tiou, 4) if unref_tiou is not None else None,
        refined_mean_tiou=round(ref_tiou, 4) if ref_tiou is not None else None,
        tiou_delta=round(tiou_delta, 4) if tiou_delta is not None else None,
        start_error_reduction_sec=round(start_err_red, 4) if start_err_red is not None else None,
        end_error_reduction_sec=round(end_err_red, 4) if end_err_red is not None else None,
        hit1_unrefined=h1_u,
        hit1_refined=h1_r,
        hit2_unrefined=h2_u,
        hit2_refined=h2_r,
        hit3_unrefined=h3_u,
        hit3_refined=h3_r,
        rationale_samples=rationales[:5],
    )

