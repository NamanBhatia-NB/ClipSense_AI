"""
Interval Highlight Benchmark Dataset Adapter (W11)

Handles benchmark datasets where ground-truth highlights are annotated as
discrete continuous temporal intervals [t_start, t_end] (e.g. YouTube Highlights,
annotator consensus spans, long-form podcast highlight testbeds).
"""

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional

from evaluation.adapters.base_adapter import BaseDatasetAdapter, DatasetProvenance
from evaluation.schemas import (
    BenchmarkDataset,
    BenchmarkSample,
    ReferenceHighlight,
    SampleType,
    SegmentImportanceAnnotation,
)

logger = logging.getLogger("clipsense.evaluation.adapters.interval")

INTERVAL_PROVENANCE = DatasetProvenance(
    dataset_name="CanonicalIntervalHighlights",
    paper_title="Multi-Modal Temporal Video Highlight Detection Framework",
    authors="ClipSense AI Research Team",
    venue_and_year="B.Tech Minor Research Project 2026",
    task_type="query_free_highlight",
    annotation_paradigm="temporal_intervals",
    license="Research & Educational Use",
    intended_protocol=(
        "Bipartite Hungarian matching protocol on continuous intervals with tIoU, "
        "start/end boundary errors, duration errors, and tolerance hit rates (Hit@1s, Hit@2s, Hit@3s)."
    ),
    critical_limitations_and_assumptions=(
        "Discrete interval ground truth requires distinct start and end boundaries. "
        "Evaluated via tIoU and boundary displacement without subjective threshold distortion."
    ),
)


class IntervalHighlightAdapter(BaseDatasetAdapter):
    """
    Adapter for datasets with discrete ground-truth highlight intervals.
    """

    def __init__(self, provenance: Optional[DatasetProvenance] = None):
        super().__init__(provenance=provenance or INTERVAL_PROVENANCE)

    def load_dataset(
        self,
        manifest_path: str,
        required_sample_type: Optional[SampleType] = None,
    ) -> BenchmarkDataset:
        """
        Loads and strictly validates interval manifest.
        Guarantees boundary ordering, positive durations, and bounds checking.
        """
        path = pathlib.Path(manifest_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Interval benchmark manifest not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset_name = data.get("dataset_name", "interval_benchmark")
        dataset_type: SampleType = data.get("dataset_type", "research")

        if required_sample_type and dataset_type != required_sample_type:
            raise ValueError(
                f"Dataset type mismatch: requested '{required_sample_type}', "
                f"manifest declares '{dataset_type}'."
            )

        samples: List[BenchmarkSample] = []
        for s_idx, s in enumerate(data.get("samples", [])):
            sample_id = s.get("sample_id", f"sample_{s_idx}")
            stype: SampleType = s.get("sample_type", dataset_type)
            duration_sec = float(s["duration_sec"])

            if required_sample_type == "research" and stype != "research":
                raise ValueError(
                    f"Sample '{sample_id}' has sample_type='{stype}' in research mode."
                )

            ref_spans: List[ReferenceHighlight] = []
            for r_idx, r in enumerate(s.get("reference_spans", [])):
                start = float(r["start_time"])
                end = float(r["end_time"])

                # Strict validation of ground-truth intervals
                if start < 0.0 or end <= start or end > duration_sec + 0.5:
                    logger.warning(
                        f"Skipping malformed reference interval [{start:.2f}s -> {end:.2f}s] "
                        f"in sample '{sample_id}' (duration: {duration_sec:.2f}s)."
                    )
                    continue

                ref_spans.append(
                    ReferenceHighlight(
                        span_id=r.get("span_id", f"ref_{sample_id}_{r_idx}"),
                        start_time=start,
                        end_time=end,
                        label=r.get("label"),
                        annotation_source=r.get("annotation_source", "independent_annotators"),
                        description=r.get("description"),
                    )
                )

            samples.append(
                BenchmarkSample(
                    sample_id=sample_id,
                    sample_type=stype,
                    video_path=s.get("video_path"),
                    duration_sec=duration_sec,
                    run_dir=s["run_dir"],
                    reference_spans=ref_spans,
                    metadata=s.get("metadata", {}),
                )
            )

        logger.info(f"Loaded {len(samples)} samples from interval manifest '{dataset_name}'.")
        return BenchmarkDataset(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            description=data.get("description", "Discrete highlight interval benchmark dataset."),
            samples=samples,
        )

    def extract_segment_annotations(self, manifest_path: str) -> Dict[str, SegmentImportanceAnnotation]:
        """
        Interval datasets do not have native segment curves; returns empty dict.
        """
        return {}
