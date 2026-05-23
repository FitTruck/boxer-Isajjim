"""Stage 3: Monocular metric depth estimation.

Two backends selectable via `DEPTH_BACKEND` env var:

- `depthpro` (default) : Apple Depth Pro — sharp, single-image metric depth +
                         predicted focal length in pixels (huge bonus for boxer,
                         which otherwise needs synthetic intrinsics).
- `da2`               : Depth Anything V2 (HF `depth-estimation` pipeline) —
                         lighter fallback when Depth Pro weights are unavailable.

`estimate()` returns a `DepthResult` containing both the depth map and (when
available) the predicted focal length so `BoxerLifter` can use real intrinsics.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ai.config import Config

logger = logging.getLogger(__name__)


@dataclass
class DepthResult:
    depth: np.ndarray  # (H, W) float32, meters
    focal_length_px: Optional[float] = None  # Depth Pro only


# ---------------------------------------------------------------------------
# Backend: Apple Depth Pro
# ---------------------------------------------------------------------------
class _DepthProBackend:
    """`apple/DepthPro-hf` via transformers."""

    name = "depthpro"

    def __init__(self, model_name: str, device: str):
        from transformers import (
            DepthProForDepthEstimation,
            DepthProImageProcessorFast,
        )

        logger.info(f"Loading Depth Pro on {device}: {model_name}")
        self.processor = DepthProImageProcessorFast.from_pretrained(model_name)
        self.model = DepthProForDepthEstimation.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def estimate(self, image: Image.Image) -> DepthResult:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        post = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0]
        depth_t = post["predicted_depth"]
        if isinstance(depth_t, torch.Tensor):
            depth = depth_t.detach().cpu().float().numpy()
        else:
            depth = np.asarray(depth_t, dtype=np.float32)

        # HF DepthPro post-processing key is `focal_length` (in pixels).
        fl = post.get("focal_length", post.get("focallength_px"))
        if isinstance(fl, torch.Tensor):
            fl = float(fl.item())
        elif fl is not None:
            fl = float(fl)
        return DepthResult(depth=depth.astype(np.float32), focal_length_px=fl)


# ---------------------------------------------------------------------------
# Backend: Depth Anything V2
# Uses AutoImageProcessor / AutoModelForDepthEstimation directly to avoid the
# legacy `transformers.pipelines` module (which can fail to import under some
# torch / transformers dev-build combinations).
# ---------------------------------------------------------------------------
class _DepthAnythingBackend:
    """Light fallback — Depth Anything V2 via Auto* classes."""

    name = "da2"

    def __init__(self, model_name: str, device: str):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        logger.info(f"Loading Depth Anything V2 on {device}: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def estimate(self, image: Image.Image) -> DepthResult:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        depth_t = outputs.predicted_depth
        if depth_t.dim() == 3:
            depth_t = depth_t[0]
        depth = depth_t.detach().cpu().float().numpy()

        # DA V2 output is up-to-scale; rescale to a plausible metric range.
        if depth.max() > 0:
            depth = depth / depth.max() * 10.0

        if depth.shape != (image.height, image.width):
            depth_pil = Image.fromarray(depth).resize(
                (image.width, image.height), Image.BILINEAR
            )
            depth = np.asarray(depth_pil, dtype=np.float32)
        return DepthResult(depth=depth.astype(np.float32), focal_length_px=None)


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------
class DepthEstimator:
    """Backend-agnostic monocular depth estimator (cuda / mps / cpu)."""

    def __init__(
        self,
        backend: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = device or Config.get_default_device()
        self.backend_name = (backend or Config.DEPTH_BACKEND).lower()
        self._backend = self._build_backend(self.backend_name)

    def _build_backend(self, name: str):
        if name == "depthpro":
            try:
                return _DepthProBackend(Config.DEPTH_PRO_MODEL, self.device)
            except Exception as e:
                logger.exception(f"DepthPro failed to load ({e}); falling back to Depth Anything V2")
                return _DepthAnythingBackend(Config.DEPTH_DA2_MODEL, self.device)
        if name == "da2":
            return _DepthAnythingBackend(Config.DEPTH_DA2_MODEL, self.device)
        raise ValueError(f"Unknown DEPTH_BACKEND: {name}")

    def estimate(self, image: Image.Image) -> DepthResult:
        return self._backend.estimate(image)
