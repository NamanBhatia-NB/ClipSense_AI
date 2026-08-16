"""
ClipSense Base Expert Contract

Defines:
1. BaseExpert abstract class required for all four independent evidence experts
2. VisualFeatureExtractorInterface protocol for pluggable visual encoders
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, runtime_checkable
import numpy as np
from core.schemas import ExpertEvidenceBundle


class BaseExpert(ABC):
    """
    Abstract Base Class for the Four Independent Evidence Experts:
    - Transcript Expert
    - Visual Expert
    - Prosody Expert
    - Conversation Expert

    Guarantees:
    - Each expert operates independently and returns an ExpertEvidenceBundle.
    - Expert proposals maintain continuous floating-point timestamps and confidence estimates.
    - Overlapping proposals are allowed and encouraged for downstream agreement analysis.
    - No premature scalar fusion happens inside any expert.
    """

    def __init__(self, name: str, config: Optional[Any] = None):
        self.name = name
        self.config = config

    @abstractmethod
    def evaluate(self, extraction_data: Any, context: Optional[Dict[str, Any]] = None) -> ExpertEvidenceBundle:
        """
        Evaluate extracted multimodal data and generate independent temporal proposals.

        Args:
            extraction_data: The modality-specific extracted data 
                             (e.g., TranscriptData, VisualData, ProsodyData, ConversationData).
            context: Optional contextual dictionary (e.g., global topic summary, user prompt).

        Returns:
            ExpertEvidenceBundle containing temporal proposals with confidence estimates,
            measurable supporting features, and explicit source metadata.
        """
        pass


@runtime_checkable
class VisualFeatureExtractorInterface(Protocol):
    """
    Pluggable visual encoder protocol.
    Allows interchangeable visual representations (e.g., frame difference metrics,
    lightweight MobileNetV3 embeddings, ResNet18 spatial features) without committing
    prematurely to a single deep learning architecture.
    """

    def extract_features(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Extract a 1D visual feature vector from an individual BGR frame.

        Args:
            frame_bgr: OpenCV-compatible BGR image array.

        Returns:
            1D numpy array representing visual feature embeddings or spatial statistics.
        """
        ...

    def compute_distance(self, feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        """
        Compute a normalized visual distance or change metric between two feature vectors.

        Args:
            feat_a: Feature vector at time t1.
            feat_b: Feature vector at time t2.

        Returns:
            Float representing magnitude of visual change.
        """
        ...
