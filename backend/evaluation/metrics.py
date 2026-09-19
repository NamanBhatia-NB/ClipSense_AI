"""
Evaluation Metrics and Bipartite Matching Protocol (W6)

Pure mathematical formulations for:
- Temporal IoU (continuous floating-point)
- Start and End boundary errors (absolute and signed bias)
- Duration error (absolute and relative)
- Tolerance Hit Rates (Hit@1s, Hit@2s, Hit@3s)
- Solely Hungarian bipartite matching protocol with documented cost and minimum acceptance threshold
"""

import math
from typing import Dict, List, Sequence, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluation.schemas import (
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
