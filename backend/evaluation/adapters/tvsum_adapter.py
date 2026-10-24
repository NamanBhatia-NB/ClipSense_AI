"""
TVSum Benchmark Dataset Adapter (W11)

Dataset: TVSum (Title-based Video Summarization)
Reference:
  Song, Y., Vallmitjana, J., Stent, A., & Jaimes, A. (2015).
  "TVSum: Summarizing Web Videos Using Titles as Additional Weak Supervision."
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.

Task & Paradigm:
  - Task: Query-free video summarization on 50 YouTube videos across 10 categories.
  - Annotation: 20 crowd-workers provided shot-level importance ratings (1 to 5).
  - Intended Protocol: Continuous segment-level ranking (Spearman's rho, Kendall's tau)
    and Top-15% duration selection (maximum duration budget = 15% of video length).

Critical Scientific Caveat:
  TVSum annotations are continuous segment-level importance ratings, NOT binary
  ground-truth highlight start/end boundaries. Converting them into artificial binary
  intervals introduces arbitrary binarization thresholds. Therefore, ClipSense AI evaluates
  TVSum using continuous rank correlation (Spearman rho, Kendall tau) and Top-15% overlap.
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

logger = logging.getLogger("clipsense.evaluation.adapters.tvsum")

TVSUM_PROVENANCE = DatasetProvenance(
    dataset_name="TVSum",
    paper_title="TVSum: Summarizing Web Videos Using Titles as Additional Weak Supervision",
    authors="Yale Song, Jordi Vallmitjana, Amanda Stent, Alejandro Jaimes",
    venue_and_year="IEEE CVPR 2015",
    task_type="query_free_summarization",
    annotation_paradigm="segment_importance_scores",
    license="Yahoo Academic Research License / Creative Commons",
    access_url="https://github.com/yalesong/tvsum",
    intended_protocol=(
        "Continuous shot/frame-level importance ranking evaluated via Spearman's rho "
        "and Kendall's tau against crowd-worker consensus curves, plus Top-15% duration overlap."
    ),
    critical_limitations_and_assumptions=(
        "TVSum labels represent relative importance ratings from 20 annotators, not discrete "
        "highlight boundaries. Binarized start/end intervals must not be fabricated; evaluate "
        "via importance ranking metrics or top-k% segment coverage."
    ),
)


class TVSumAdapter(BaseDatasetAdapter):
    """
    Adapter for the TVSum video summarization benchmark.
    """

    def __init__(self, provenance: Optional[DatasetProvenance] = None):
        super().__init__(provenance=provenance or TVSUM_PROVENANCE)

    def load_dataset(
        self,
        manifest_path: str,
        required_sample_type: Optional[SampleType] = None,
    ) -> BenchmarkDataset:
        """
        Loads TVSum manifest. For videos with shot-level segment scores, synthesizes
        consensus top-15% reference intervals for boundary compatibility if needed,
        while maintaining primary segment importance curves.
        """
        path = pathlib.Path(manifest_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"TVSum manifest not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset_type = data.get("dataset_type", "research")
        if required_sample_type and dataset_type != required_sample_type:
            raise ValueError(
                f"TVSum manifest declares dataset_type='{dataset_type}', "
                f"which differs from required='{required_sample_type}'."
            )

        samples: List[BenchmarkSample] = []
        for s_data in data.get("samples", []):
            stype = s_data.get("sample_type", dataset_type)
            if required_sample_type == "research" and stype != "research":
                raise ValueError(
                    f"Sample '{s_data.get('sample_id')}' declares type '{stype}' in research mode."
                )

            # Build reference highlight intervals from top-15% consensus segments if provided
            ref_spans: List[ReferenceHighlight] = []
            for r in s_data.get("reference_spans", []):
                ref_spans.append(
                    ReferenceHighlight(
                        span_id=r["span_id"],
                        start_time=float(r["start_time"]),
                        end_time=float(r["end_time"]),
                        label=r.get("label", "top_15_percent_summary_shot"),
                        annotation_source=r.get("annotation_source", "tvsum_20_annotators_consensus"),
                        description=r.get("description", "Consensus high-importance shot in TVSum"),
                    )
                )

            samples.append(
                BenchmarkSample(
                    sample_id=s_data["sample_id"],
                    sample_type=stype,
                    video_path=s_data.get("video_path"),
                    duration_sec=float(s_data["duration_sec"]),
                    run_dir=s_data["run_dir"],
                    reference_spans=ref_spans,
                    metadata=s_data.get("metadata", {}),
                )
            )

        logger.info(f"Loaded TVSum benchmark dataset with {len(samples)} samples.")
        return BenchmarkDataset(
            dataset_name=data.get("dataset_name", "TVSum_50"),
            dataset_type=dataset_type,
            description=data.get("description", TVSUM_PROVENANCE.intended_protocol),
            samples=samples,
        )

    def extract_segment_annotations(self, manifest_path: str) -> Dict[str, SegmentImportanceAnnotation]:
        """
        Parses continuous 1.0s or shot-level segment importance curves from the manifest.
        """
        path = pathlib.Path(manifest_path).resolve()
        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segment_map: Dict[str, SegmentImportanceAnnotation] = {}
        for s in data.get("samples", []):
            sample_id = s["sample_id"]
            scores = s.get("segment_scores")
            if scores is not None and len(scores) > 0:
                seg_dur = float(s.get("segment_duration_sec", 1.0))
                segment_map[sample_id] = SegmentImportanceAnnotation(
                    sample_id=sample_id,
                    segment_duration_sec=seg_dur,
                    num_segments=len(scores),
                    segment_scores=[float(v) for v in scores],
                    annotator_count=int(s.get("annotator_count", 20)),
                    category=s.get("metadata", {}).get("category"),
                    metadata=s.get("metadata", {}),
                )

        return segment_map
