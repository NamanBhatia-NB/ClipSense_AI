"""
ClipSense Multimodal Extraction Package (W2)
"""

from pipeline.extraction.transcript_extractor import TranscriptExtractor
from pipeline.extraction.frame_extractor import FrameExtractor
from pipeline.extraction.prosody_extractor import ProsodyExtractor
from pipeline.extraction.conversation_extractor import ConversationExtractor
from pipeline.extraction.extractor_pipeline import MultimodalExtractionPipeline

__all__ = [
    "TranscriptExtractor",
    "FrameExtractor",
    "ProsodyExtractor",
    "ConversationExtractor",
    "MultimodalExtractionPipeline",
]
