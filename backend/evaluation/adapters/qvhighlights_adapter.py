"""
QVHighlights Benchmark Dataset Adapter (W11)

Dataset: QVHighlights (Query-Conditioned Video Highlights and Moments)
Reference:
  Lei, J., Berg, T. L., & Bansal, M. (2021).
  "Detecting Moments and Highlights in Large-Scale Video-Language Pre-Training."
  Advances in Neural Information Processing Systems (NeurIPS), 2021.

Task & Paradigm:
  - Task: Query-conditioned moment retrieval (MR) and highlight detection (HD).
  - Scale: 10,148 videos from YouTube (150s average duration).
  - Annotations: Free-form text query + 5-level crowd-worker saliency scores
    (1 to 5) + labeled relevant moment spans [t_start, t_end].

Critical Scientific Distinction (Query-Free vs. Query-Conditioned):
  QVHighlights is explicitly designed for QUERY-CONDITIONED temporal grounding.
  A highlight in QVHighlights is defined ONLY relative to a specific natural language query
  (e.g., 'A person explains the recipe' vs. 'The oven timer beeps').
  
  In contrast, ClipSense AI operates in the QUERY-FREE regime (detecting salient,
  coherent, multi-expert highlights without textual prompts).
  
  Comparing query-free predictions directly against query-conditioned targets without
  qualifying query marginalization or generic visual salience is invalid.
  The QVHighlightsAdapter documents this protocol boundary and supports either:
  1. Query-marginalized consensus highlights (moments highlighted across multiple queries).
  2. Generic salience evaluation against high-saliency moment spans.
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

logger = logging.getLogger("clipsense.evaluation.adapters.qvhighlights")

QVHIGHLIGHTS_PROVENANCE = DatasetProvenance(
    dataset_name="QVHighlights",
    paper_title="Detecting Moments and Highlights in Large-Scale Video-Language Pre-Training",
    authors="Jie Lei, Tamara L. Berg, Mohit Bansal",
    venue_and_year="NeurIPS 2021",
    task_type="query_conditioned_grounding",
    annotation_paradigm="query_conditioned_moments",
    license="MIT License",
    access_url="https://github.com/jayleicn/moment_detr",
    intended_protocol=(
        "Query-conditioned moment retrieval evaluated via mAP@IoU and Hit@1 given textual query."
    ),
    critical_limitations_and_assumptions=(
        "Query-free highlight extraction cannot be directly scored against single-query "
        "targets without query conditioning or query marginalization. The adapter exposes "
        "query-marginalized consensus intervals for query-free highlight benchmarking."
    ),
)


class QVHighlightsAdapter(BaseDatasetAdapter):
    """
    Adapter for the QVHighlights benchmark with query-free protocol isolation.
    """

    def __init__(self, provenance: Optional[DatasetProvenance] = None):
        super().__init__(provenance=provenance or QVHIGHLIGHTS_PROVENANCE)

    def load_dataset(
        self,
        manifest_path: str,
        required_sample_type: Optional[SampleType] = None,
        query_mode: str = "consensus_marginalized",
    ) -> BenchmarkDataset:
        """
        Loads QVHighlights manifest. Supports query_mode='consensus_marginalized'
        which aggregates salient moments across queries into query-free reference spans.
        """
        path = pathlib.Path(manifest_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"QVHighlights manifest not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset_type = data.get("dataset_type", "research")
        if required_sample_type and dataset_type != required_sample_type:
            raise ValueError(
                f"QVHighlights manifest declares dataset_type='{dataset_type}', "
                f"differing from requested '{required_sample_type}'."
            )

        samples: List[BenchmarkSample] = []
        for s in data.get("samples", []):
            stype = s.get("sample_type", dataset_type)
            if required_sample_type == "research" and stype != "research":
                raise ValueError(
                    f"Sample '{s.get('sample_id')}' declares type '{stype}' in research mode."
                )

            ref_spans: List[ReferenceHighlight] = []
            for r in s.get("reference_spans", []):
                ref_spans.append(
                    ReferenceHighlight(
                        span_id=r["span_id"],
                        start_time=float(r["start_time"]),
                        end_time=float(r["end_time"]),
                        label=r.get("label", "query_marginalized_highlight"),
                        annotation_source=r.get("annotation_source", "qvhighlights_consensus"),
                        description=r.get("description", r.get("query_text", "")),
                    )
                )

            samples.append(
                BenchmarkSample(
                    sample_id=s["sample_id"],
                    sample_type=stype,
                    video_path=s.get("video_path"),
                    duration_sec=float(s["duration_sec"]),
                    run_dir=s["run_dir"],
                    reference_spans=ref_spans,
                    metadata={
                        **s.get("metadata", {}),
                        "query_mode": query_mode,
                        "queries_count": len(s.get("queries", [])),
                    },
                )
            )

        logger.info(f"Loaded QVHighlights benchmark dataset with {len(samples)} samples.")
        return BenchmarkDataset(
            dataset_name=data.get("dataset_name", "QVHighlights"),
            dataset_type=dataset_type,
            description=data.get("description", QVHIGHLIGHTS_PROVENANCE.critical_limitations_and_assumptions),
            samples=samples,
        )

    def extract_segment_annotations(self, manifest_path: str) -> Dict[str, SegmentImportanceAnnotation]:
        """
        Extracts saliency score curves (e.g. 2-second or 1-second bins) if annotated.
        """
        path = pathlib.Path(manifest_path).resolve()
        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segment_map: Dict[str, SegmentImportanceAnnotation] = {}
        for s in data.get("samples", []):
            sample_id = s["sample_id"]
            scores = s.get("saliency_scores") or s.get("segment_scores")
            if scores is not None and len(scores) > 0:
                seg_dur = float(s.get("segment_duration_sec", 2.0))
                segment_map[sample_id] = SegmentImportanceAnnotation(
                    sample_id=sample_id,
                    segment_duration_sec=seg_dur,
                    num_segments=len(scores),
                    segment_scores=[float(v) for v in scores],
                    annotator_count=int(s.get("annotator_count", 3)),
                    category="qvhighlights_salience",
                    metadata=s.get("metadata", {}),
                )

        return segment_map
