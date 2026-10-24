"""
Dataset Adapters Package (W11)
"""

from evaluation.adapters.base_adapter import (
    AnnotationParadigm,
    BaseDatasetAdapter,
    DatasetProvenance,
    DatasetTaskType,
)
from evaluation.adapters.interval_adapter import (
    INTERVAL_PROVENANCE,
    IntervalHighlightAdapter,
)
from evaluation.adapters.qvhighlights_adapter import (
    QVHIGHLIGHTS_PROVENANCE,
    QVHighlightsAdapter,
)
from evaluation.adapters.rhapsody_adapter import (
    RHAPSODY_PROVENANCE,
    RhapsodyAdapter,
)
from evaluation.adapters.tvsum_adapter import (
    TVSUM_PROVENANCE,
    TVSumAdapter,
)

__all__ = [
    "BaseDatasetAdapter",
    "DatasetProvenance",
    "DatasetTaskType",
    "AnnotationParadigm",
    "TVSumAdapter",
    "TVSUM_PROVENANCE",
    "QVHighlightsAdapter",
    "QVHIGHLIGHTS_PROVENANCE",
    "IntervalHighlightAdapter",
    "INTERVAL_PROVENANCE",
    "RhapsodyAdapter",
    "RHAPSODY_PROVENANCE",
    "get_dataset_adapter",
]


def get_dataset_adapter(adapter_name: str) -> BaseDatasetAdapter:
    """
    Factory function to retrieve the appropriate dataset adapter.
    """
    name_clean = adapter_name.strip().lower()
    if name_clean in ("rhapsody", "podcast", "podcasts"):
        return RhapsodyAdapter()
    elif name_clean in ("tvsum", "tvsum_50", "summarization"):
        return TVSumAdapter()
    elif name_clean in ("qvhighlights", "qvh", "grounding"):
        return QVHighlightsAdapter()
    elif name_clean in ("interval", "canonical", "highlights"):
        return IntervalHighlightAdapter()
    else:
        # Default fallback to IntervalHighlightAdapter
        return IntervalHighlightAdapter()
