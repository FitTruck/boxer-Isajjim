"""AI module for boxer-Isajjim.

Pipeline (per image):
    Firebase URL → OWLv2 (LVIS+) → Depth → BoxerNet → DB label → JSON

Boxer outputs absolute metric dimensions and volume directly; no relative→absolute
conversion is required.
"""

__version__ = "1.0.0"

from .pipeline import (
    BoxerLifter,
    BoxerObb,
    DepthEstimator,
    DetectedObject,
    FurniturePipeline,
    ImageFetcher,
    Owlv2Detector,
    PipelineResult,
)

__all__ = [
    "FurniturePipeline",
    "PipelineResult",
    "DetectedObject",
    "ImageFetcher",
    "Owlv2Detector",
    "DepthEstimator",
    "BoxerLifter",
    "BoxerObb",
]
