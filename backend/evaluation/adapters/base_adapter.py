"""
Base Dataset Adapter Architecture (W11)

Defines the contract for benchmark dataset adapters:
1. Clear taxonomy distinguishing query-free highlight selection,
   video summarization, and query-conditioned temporal grounding.
2. Metadata schemas tracking dataset provenance, task definition,
   licensing/access constraints, and intended evaluation protocols.
3. Strict Train/Test leakage guards preventing ground-truth annotations
   from contaminating expert extraction or MTER candidate generation.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

from evaluation.schemas import (
    BenchmarkDataset,
    BenchmarkSample,
    ReferenceHighlight,
    SampleType,
    SegmentImportanceAnnotation,
)

logger = logging.getLogger("clipsense.evaluation.adapters.base")

DatasetTaskType = Literal[
    "query_free_highlight",          # ClipSense primary task: unsupervised long-form highlight extraction
    "query_free_summarization",      # Continuous importance ranking (e.g. TVSum, SumMe)
    "query_conditioned_grounding",   # Natural language moment retrieval (e.g. QVHighlights, Charades-STA)
]

AnnotationParadigm = Literal[
    "temporal_intervals",            # Explicit discrete start and end timestamps [s, e]
    "segment_importance_scores",     # Shot or second-level salience curves S(t) in [0, 1]
    "query_conditioned_moments",     # (Query, Moment Span) pairs + optional saliency
]


class DatasetProvenance(BaseModel):
    """
    Academic documentation of benchmark dataset origin, task, and licensing.
    """
    dataset_name: str
    paper_title: str
    authors: str
    venue_and_year: str
    task_type: DatasetTaskType
    annotation_paradigm: AnnotationParadigm
    license: str
    access_url: Optional[str] = None
    intended_protocol: str
    critical_limitations_and_assumptions: str


class BaseDatasetAdapter(ABC):
    """
    Abstract interface for benchmark dataset adapters.
    """

    def __init__(self, provenance: DatasetProvenance):
        self.provenance = provenance

    @abstractmethod
    def load_dataset(self, manifest_path: str, required_sample_type: Optional[SampleType] = None) -> BenchmarkDataset:
        """
        Loads the benchmark dataset and returns a validated BenchmarkDataset object.
        """
        pass

    @abstractmethod
    def extract_segment_annotations(self, manifest_path: str) -> Dict[str, SegmentImportanceAnnotation]:
        """
        Extracts continuous segment/frame-level importance annotations if supported by the dataset.
        For interval-only datasets, returns an empty dictionary.
        """
        pass

    def enforce_no_leakage(
        self,
        sample: BenchmarkSample,
        pipeline_inputs: Dict[str, Any],
    ) -> None:
        """
        Strictly guards against train/test contamination and annotation leakage.
        Ensures reference annotations are NOT present in the input feature bundle passed
        to evidence experts or MTER reasoning.
        """
        forbidden_keys = {"reference_spans", "ground_truth", "target_spans", "ref_start", "ref_end"}
        for k in forbidden_keys:
            if k in pipeline_inputs:
                raise ValueError(
                    f"DATA LEAKAGE DETECTED: Ground-truth key '{k}' found in pipeline execution inputs "
                    f"for sample '{sample.sample_id}'. Evaluation annotations must remain strictly isolated."
                )

        # Check nested dictionaries
        for key, val in pipeline_inputs.items():
            if isinstance(val, dict):
                for k in forbidden_keys:
                    if k in val:
                        raise ValueError(
                            f"DATA LEAKAGE DETECTED in sub-dictionary '{key}.{k}' for sample '{sample.sample_id}'."
                        )

        logger.debug(f"Leakage check passed for sample '{sample.sample_id}'.")
