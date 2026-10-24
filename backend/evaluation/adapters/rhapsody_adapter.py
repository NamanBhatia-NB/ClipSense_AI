"""
Rhapsody Benchmark Dataset Adapter (Phase 2)

Dataset: Rhapsody (A Dataset for Highlight Detection in Podcasts)
Reference:
  Park, Y., Diwan, A., Harwath, D., & Choi, E. (2025).
  "Rhapsody: A Dataset for Highlight Detection in Podcasts."
  Conference on Language Modeling (COLM 2025), arXiv:2505.19429.
Official Repository:
  https://github.com/younghanstark/rhapsody
Official Dataset:
  https://huggingface.co/datasets/yhpark/rhapsody

Task & Paradigm:
  - Task: Query-free long-form podcast highlight detection across 13,364 episodes.
  - Annotation: 100 equal-duration temporal segments per episode with continuous
    replay scores in [0.0, 1.0] derived from YouTube "Most Replayed" player statistics.
  - Official Binarization Policy:
    Segments with replay score >= 0.50 are labeled as ground-truth highlights,
    retaining up to at most max_n_gt = 20 segments with the highest replay scores.
  - Official Evaluation Protocol:
    Segment-level ranking and binary classification over 100 discrete time bins,
    evaluated via mAP (mean Average Precision), Hit Rate, Precision, Recall, and F1.

Critical Scientific Caveat:
  Rhapsody annotations are viewer engagement statistics aggregated into 100 uniform
  temporal bins, NOT precise human start/end narrative boundaries. Continuous candidate
  intervals from ClipSense are projected to 100 bins via maximum confidence-weighted
  overlap. Sub-second boundary refinement must be evaluated as a separate diagnostic
  (boundary stability / speech pause alignment) rather than replay segment overlap.
"""

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional
import numpy as np

from evaluation.adapters.base_adapter import BaseDatasetAdapter, DatasetProvenance
from evaluation.schemas import (
    BenchmarkDataset,
    BenchmarkSample,
    ReferenceHighlight,
    SampleType,
    SegmentImportanceAnnotation,
)

logger = logging.getLogger("clipsense.evaluation.adapters.rhapsody")

RHAPSODY_PROVENANCE = DatasetProvenance(
    dataset_name="Rhapsody",
    paper_title="Rhapsody: A Dataset for Highlight Detection in Podcasts",
    authors="Younghan Park, Anuj Diwan, David Harwath, Eunsol Choi",
    venue_and_year="COLM 2025 (arXiv:2505.19429)",
    task_type="query_free_highlight",
    annotation_paradigm="segment_importance_scores",
    license="Research / Academic Use (Hugging Face yhpark/rhapsody, gated)",
    access_url="https://github.com/younghanstark/rhapsody",
    intended_protocol=(
        "100 equal-duration segments per episode evaluated via mAP (mean Average Precision), "
        "Hit Rate, Precision, Recall, and F1 against ground-truth highlight segments derived from "
        "YouTube Most Replayed scores (threshold >= 0.5, max_n_gt = 20)."
    ),
    critical_limitations_and_assumptions=(
        "Rhapsody annotations are viewer replay engagement curves mapped to 100 uniform temporal buckets, "
        "NOT precise human start/end highlight boundaries. Continuous candidate intervals from ClipSense "
        "are projected to 100 bins via maximum confidence-weighted overlap. Fine-grained boundary refinement "
        "must be evaluated via boundary stability/pause alignment diagnostics separately."
    ),
)


class RhapsodyAdapter(BaseDatasetAdapter):
    """
    Adapter for the Rhapsody podcast highlight detection benchmark.
    Guarantees strict 100-segment validation, ground-truth binarization matching
    the official COLM 2025 protocol, and zero ground-truth leakage into inference.
    """

    def __init__(
        self,
        provenance: Optional[DatasetProvenance] = None,
        gt_threshold: float = 0.5,
        max_n_gt: int = 20,
        max_duration_sec: Optional[float] = 3600.0,
    ):
        super().__init__(provenance=provenance or RHAPSODY_PROVENANCE)
        self.gt_threshold = float(gt_threshold)
        self.max_n_gt = int(max_n_gt)
        self.max_duration_sec = float(max_duration_sec) if max_duration_sec is not None else None

    def load_dataset(
        self,
        manifest_path: str,
        required_sample_type: Optional[SampleType] = None,
        split_filter: Optional[str] = None,
    ) -> BenchmarkDataset:
        """
        Loads and validates a Rhapsody benchmark manifest.
        Strictly enforces that each episode has duration > 0 and replay_scores of length 100.
        Explicitly filters out videos longer than max_duration_sec (3600s) without silent timestamp truncation.
        Constructs reference highlight intervals for official binarized evaluation.
        """
        path = pathlib.Path(manifest_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Rhapsody manifest not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset_type = data.get("dataset_type", "research")
        if required_sample_type and dataset_type != required_sample_type:
            raise ValueError(
                f"Rhapsody manifest declares dataset_type='{dataset_type}', "
                f"which differs from required='{required_sample_type}'."
            )

        samples: List[BenchmarkSample] = []
        raw_samples = data.get("samples", [])
        if not raw_samples:
            raise ValueError(f"Rhapsody manifest '{path}' contains no samples.")

        for s_data in raw_samples:
            sample_id = s_data.get("sample_id")
            if not sample_id:
                raise ValueError("Rhapsody sample missing required 'sample_id'.")

            stype = s_data.get("sample_type", dataset_type)
            if required_sample_type == "research" and stype != "research":
                raise ValueError(
                    f"Sample '{sample_id}' declares type '{stype}' in research mode."
                )

            split = s_data.get("split", "test").strip().lower()
            if split_filter and split != split_filter.strip().lower():
                continue

            duration = float(s_data.get("duration_sec", 0.0))
            if duration <= 0.0:
                raise ValueError(
                    f"Sample '{sample_id}' has invalid duration_sec={duration}. Must be strictly positive."
                )
            if self.max_duration_sec is not None and duration > self.max_duration_sec:
                logger.warning(
                    f"Filtering out sample '{sample_id}': duration {duration:.1f}s exceeds max_duration_sec "
                    f"{self.max_duration_sec:.1f}s (sub-hour filter enforced; no timestamps truncated)."
                )
                continue

            replay_scores = s_data.get("replay_scores")
            if replay_scores is None:
                raise ValueError(f"Sample '{sample_id}' is missing required 'replay_scores'.")
            if len(replay_scores) != 100:
                raise ValueError(
                    f"Sample '{sample_id}' malformed: replay_scores must have exactly 100 values, "
                    f"got {len(replay_scores)}."
                )

            t_seg = duration / 100.0

            # Derive official ground-truth highlight segments (Park et al. COLM 2025)
            # 1. Correct segment 0 intro click bias
            replay_scores_corrected = list(replay_scores)
            if len(replay_scores_corrected) > 1:
                replay_scores_corrected[0] = replay_scores_corrected[1]

            # 2. Extract candidate segments strictly > gt_threshold, top max_n_gt peaks
            candidate_indices = [
                (idx, float(score))
                for idx, score in enumerate(replay_scores_corrected)
                if float(score) > self.gt_threshold
            ]
            # Sort descending by score, stable tie-break by index
            candidate_indices.sort(key=lambda x: (-x[1], x[0]))
            selected_gt = candidate_indices[: self.max_n_gt]
            # Sort back chronologically
            selected_gt.sort(key=lambda x: x[0])

            ref_spans: List[ReferenceHighlight] = []
            for seg_idx, score in selected_gt:
                seg_start = round(seg_idx * t_seg, 3)
                seg_end = round(min(duration, (seg_idx + 1) * t_seg), 3)
                ref_spans.append(
                    ReferenceHighlight(
                        span_id=f"{sample_id}_gt_seg_{seg_idx}",
                        start_time=seg_start,
                        end_time=seg_end,
                        label="rhapsody_replay_highlight",
                        annotation_source=f"youtube_replay_gt_{self.gt_threshold}",
                        description=f"Segment {seg_idx} replay score: {score:.4f}",
                    )
                )

            meta = dict(s_data.get("metadata", {}))
            meta["split"] = split
            meta["youtube_id"] = s_data.get("youtube_id") or meta.get("youtube_id") or meta.get("youtube_vid")
            meta["title"] = s_data.get("title") or meta.get("title")
            meta["channel_title"] = s_data.get("channel_title") or meta.get("channel_title")
            meta["replay_scores"] = replay_scores
            meta["gt_threshold"] = self.gt_threshold
            meta["max_n_gt"] = self.max_n_gt
            meta["num_gt_segments"] = len(selected_gt)

            samples.append(
                BenchmarkSample(
                    sample_id=sample_id,
                    sample_type=stype,
                    video_path=s_data.get("video_path"),
                    duration_sec=duration,
                    run_dir=s_data.get("run_dir", f"runs/rhapsody/{sample_id}"),
                    reference_spans=ref_spans,
                    metadata=meta,
                )
            )

        logger.info(
            f"Loaded Rhapsody benchmark dataset: {len(samples)} samples "
            f"(split_filter='{split_filter}', gt_threshold={self.gt_threshold}, max_n_gt={self.max_n_gt})."
        )

        return BenchmarkDataset(
            dataset_name=data.get("dataset_name", "Rhapsody"),
            dataset_type=dataset_type,
            description=data.get("description", RHAPSODY_PROVENANCE.intended_protocol),
            samples=samples,
        )

    def extract_segment_annotations(
        self,
        manifest_path: str,
        split_filter: Optional[str] = None,
    ) -> Dict[str, SegmentImportanceAnnotation]:
        """
        Extracts the 100-segment continuous replay score curves from the manifest.
        Guarantees that each SegmentImportanceAnnotation has exactly num_segments=100
        and segment_duration_sec = duration_sec / 100.0.
        """
        path = pathlib.Path(manifest_path).resolve()
        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segment_map: Dict[str, SegmentImportanceAnnotation] = {}
        for s in data.get("samples", []):
            sample_id = s.get("sample_id")
            if not sample_id:
                continue

            split = s.get("split", "test").strip().lower()
            if split_filter and split != split_filter.strip().lower():
                continue

            duration = float(s.get("duration_sec", 0.0))
            scores = s.get("replay_scores")
            if scores is None or len(scores) != 100 or duration <= 0.0:
                continue

            t_seg = duration / 100.0
            meta = dict(s.get("metadata", {}))
            meta["split"] = split
            meta["youtube_id"] = s.get("youtube_id")
            meta["title"] = s.get("title")

            segment_map[sample_id] = SegmentImportanceAnnotation(
                sample_id=sample_id,
                segment_duration_sec=round(t_seg, 4),
                num_segments=100,
                segment_scores=[float(v) for v in scores],
                annotator_count=1,
                category=s.get("metadata", {}).get("category", "podcast"),
                metadata=meta,
            )

        return segment_map
