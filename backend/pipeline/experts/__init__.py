"""
ClipSense Independent Evidence Experts Package
"""
from pipeline.experts.base import BaseExpert, VisualFeatureExtractorInterface
from pipeline.experts.transcript_expert import TranscriptExpert
from pipeline.experts.conversation_expert import ConversationExpert
from pipeline.experts.visual_expert import VisualExpert, LightweightFrameDiffExtractor
from pipeline.experts.prosody_expert import ProsodyExpert

__all__ = [
    "BaseExpert",
    "VisualFeatureExtractorInterface",
    "TranscriptExpert",
    "ConversationExpert",
    "VisualExpert",
    "LightweightFrameDiffExtractor",
    "ProsodyExpert",
]
