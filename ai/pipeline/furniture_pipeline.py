"""Boxer-based furniture analysis pipeline orchestrator.

Flow (per image):
    URL → PIL → OWLv2 (LVIS+ subset) → Depth Pro → BoxerNet (3D OBB) → JSON

Boxer outputs absolute metric dimensions, so no relative→absolute conversion is done.
"""

import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ai.config import Config
from ai.pipeline.dimension_bounds import sanitize_dims

logger = logging.getLogger(__name__)


def _dims_mm(label: str, obb: Any) -> Tuple[float, float, float, float, List[str]]:
    """Convert a BoxerObb to (width_mm, depth_mm, height_mm, volume_m3, corrections).

    Applies per-class sanity clamping when `SANITIZE_DIMENSIONS` is on (read at
    call time so the env toggle works without a re-import). `obb=None` (no 3D
    lift) yields zeros.
    """
    if obb is None:
        return 0.0, 0.0, 0.0, 0.0, []
    w_mm, d_mm, h_mm = obb.width_m * 1000.0, obb.depth_m * 1000.0, obb.height_m * 1000.0
    corrections: List[str] = []
    if Config.SANITIZE_DIMENSIONS:
        w_mm, d_mm, h_mm, corrections = sanitize_dims(
            label,
            w_mm,
            d_mm,
            h_mm,
            prob=getattr(obb, "confidence", None),
            mode=Config.SANITIZE_MODE,
        )
    volume_m3 = (w_mm / 1000.0) * (d_mm / 1000.0) * (h_mm / 1000.0)
    return w_mm, d_mm, h_mm, volume_m3, corrections


def _flush_logs() -> None:
    """Force log handlers + stderr to flush.

    GPU inference can abort the whole process at the C level (e.g.
    'Cannot load symbol cublasLtCreate' / core dump), which bypasses Python
    exception handling. Flushing after each stage guarantees the last printed
    line pinpoints exactly which stage the process died in.
    """
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass
    sys.stderr.flush()


@dataclass
class DetectedObject:
    image_id: Optional[int]
    label: str  # OWLv2 LVIS class name (lowercase, spaces)
    confidence: float
    bbox_xyxy: List[float]  # raw pixel coords
    center_xy: Tuple[float, float]  # image-pixel center
    width_mm: float = 0.0
    depth_mm: float = 0.0
    height_mm: float = 0.0
    volume_m3: float = 0.0


@dataclass
class PipelineResult:
    image_id: Optional[int]
    image_url: Optional[str]
    objects: List[DetectedObject] = field(default_factory=list)
    error: Optional[str] = None


class FurniturePipeline:
    """End-to-end furniture detection + 3D lifting via Boxer."""

    def __init__(
        self,
        device: Optional[str] = None,
        enable_3d: bool = True,
    ) -> None:
        self.device = device  # "cuda:0" / "mps" / "cpu" / None (auto)
        self.enable_3d = enable_3d

        # Lazy import keeps startup quick when only sub-features are exercised.
        from ai.pipeline import (
            BoxerLifter,
            DepthEstimator,
            ImageFetcher,
            Owlv2Detector,
        )

        self.fetcher = ImageFetcher()
        self.detector = Owlv2Detector(device=device)
        self.depth_model: Optional[DepthEstimator] = None
        self.boxer: Optional[BoxerLifter] = None
        if enable_3d:
            self.depth_model = DepthEstimator(device=device)
            self.boxer = BoxerLifter(device=device)

    # ------------------------------------------------------------------
    # Single image
    # ------------------------------------------------------------------
    async def process_single_image(
        self,
        image_url: str,
        image_id: Optional[int] = None,
    ) -> PipelineResult:
        image = await self.fetcher.fetch_async(image_url)
        if image is None:
            return PipelineResult(
                image_id=image_id,
                image_url=image_url,
                error="Failed to fetch image",
            )
        # Offload sync GPU work to a worker thread so concurrent asyncio.gather()
        # callers on different devices actually overlap on the GPU; otherwise the
        # event loop serializes them.
        return await asyncio.to_thread(
            self.process_pil, image, image_id=image_id, image_url=image_url
        )

    def process_pil(
        self,
        image: Image.Image,
        image_id: Optional[int] = None,
        image_url: Optional[str] = None,
        enable_3d: Optional[bool] = None,
    ) -> PipelineResult:
        use_3d = self.enable_3d if enable_3d is None else enable_3d
        tag = f"img_id={image_id} dev={self.device}"
        logger.info(
            "[pipeline] %s start: size=%s use_3d=%s", tag, image.size, use_3d
        )

        logger.info("[pipeline] %s stage=OWLv2.detect ...", tag)
        _flush_logs()
        t0 = time.perf_counter()
        det = self.detector.detect(image)
        logger.info(
            "[pipeline] %s stage=OWLv2.detect done: %d boxes (%.2fs)",
            tag,
            len(det["boxes"]),
            time.perf_counter() - t0,
        )
        _flush_logs()
        if len(det["boxes"]) == 0:
            return PipelineResult(image_id=image_id, image_url=image_url, objects=[])

        labels: List[str] = list(det["labels"])

        obb_by_idx: Dict[int, Any] = {}
        if use_3d and self.boxer is not None and self.depth_model is not None:
            logger.info("[pipeline] %s stage=Depth.estimate ...", tag)
            _flush_logs()
            t0 = time.perf_counter()
            depth_result = self.depth_model.estimate(image)
            logger.info(
                "[pipeline] %s stage=Depth.estimate done: focal_px=%s (%.2fs)",
                tag,
                getattr(depth_result, "focal_length_px", None),
                time.perf_counter() - t0,
            )
            _flush_logs()

            logger.info("[pipeline] %s stage=Boxer.lift (%d boxes) ...", tag, len(det["boxes"]))
            _flush_logs()
            t0 = time.perf_counter()
            obbs = self.boxer.lift(
                image=image,
                bboxes_xyxy=det["boxes"],
                labels=labels,
                depth=depth_result.depth,
                focal_length_px=depth_result.focal_length_px,
            )
            logger.info(
                "[pipeline] %s stage=Boxer.lift done: %d obbs (%.2fs)",
                tag,
                len(obbs),
                time.perf_counter() - t0,
            )
            _flush_logs()
            obb_by_idx = {obb.input_index: obb for obb in obbs}

        objects: List[DetectedObject] = []
        for i, box in enumerate(det["boxes"]):
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            w_mm, d_mm, h_mm, volume_m3, corrections = _dims_mm(labels[i], obb_by_idx.get(i))
            for c in corrections:
                logger.info("[sanitize] %s %s", tag, c)
            objects.append(
                DetectedObject(
                    image_id=image_id,
                    label=labels[i],
                    confidence=float(det["scores"][i]),
                    bbox_xyxy=[float(v) for v in box],
                    center_xy=(float(cx), float(cy)),
                    width_mm=round(w_mm, 1),
                    depth_mm=round(d_mm, 1),
                    height_mm=round(h_mm, 1),
                    volume_m3=round(volume_m3, 6),
                )
            )
        logger.info("[pipeline] %s complete: %d objects", tag, len(objects))
        _flush_logs()
        return PipelineResult(image_id=image_id, image_url=image_url, objects=objects)

    # ------------------------------------------------------------------
    # Batch fallback (used only when no GPU pool is available — single pipeline
    # serializes the work). The GPU pool path in routes/furniture.py dispatches
    # each image to its own device-bound pipeline instead.
    # ------------------------------------------------------------------
    async def process_multiple_images(
        self,
        image_items: List[Tuple[int, str]],
    ) -> List[PipelineResult]:
        async def _one(image_id: int, url: str) -> PipelineResult:
            return await self.process_single_image(url, image_id=image_id)

        tasks = [_one(iid, url) for iid, url in image_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        out: List[PipelineResult] = []
        for item, res in zip(image_items, results):
            if isinstance(res, Exception):
                out.append(
                    PipelineResult(
                        image_id=item[0],
                        image_url=item[1],
                        error=f"{type(res).__name__}: {res}",
                    )
                )
            else:
                out.append(res)
        return out

    # ------------------------------------------------------------------
    # JSON response format (Isajjim-AI TDD, sans ply_url / type fields)
    # ------------------------------------------------------------------
    @staticmethod
    def to_json_response(results: List[PipelineResult]) -> Dict[str, Any]:
        payload = []
        for r in results:
            payload.append(
                {
                    "image_id": r.image_id,
                    "objects": [
                        {
                            "label": o.label,
                            "width": o.width_mm,
                            "depth": o.depth_mm,
                            "height": o.height_mm,
                            "center_x": round(o.center_xy[0], 1),
                            "center_y": round(o.center_xy[1], 1),
                        }
                        for o in r.objects
                    ],
                }
            )
        return {"results": payload}
