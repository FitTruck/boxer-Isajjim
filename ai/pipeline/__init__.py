"""Boxer-based furniture analysis pipeline.

Stages:
    1. ImageFetcher       (1_images_fetch.py)
    2. Owlv2Detector      (2_owlv2_2d_detection.py)
    3. DepthEstimator     (3_depth_estimation.py)
    4. BoxerLifter        (4_boxer.py)

Numeric file names are imported via importlib (digits can't start a Python identifier).
"""

import importlib

_stage1 = importlib.import_module(".1_images_fetch", package=__name__)
_stage2 = importlib.import_module(".2_owlv2_2d_detection", package=__name__)
_stage3 = importlib.import_module(".3_depth_estimation", package=__name__)
_stage4 = importlib.import_module(".4_boxer", package=__name__)

ImageFetcher = _stage1.ImageFetcher
Owlv2Detector = _stage2.Owlv2Detector
DepthEstimator = _stage3.DepthEstimator
BoxerLifter = _stage4.BoxerLifter
BoxerObb = _stage4.BoxerObb

from .sam3_detection import Sam3Detector
from .furniture_pipeline import DetectedObject, FurniturePipeline, PipelineResult

__all__ = [
    "ImageFetcher",
    "Owlv2Detector",
    "Sam3Detector",
    "DepthEstimator",
    "BoxerLifter",
    "BoxerObb",
    "FurniturePipeline",
    "DetectedObject",
    "PipelineResult",
]
