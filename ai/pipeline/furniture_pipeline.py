"""Boxer-based furniture analysis pipeline orchestrator.

Flow (per image):
    URL → PIL → OWLv2 (LVIS+ subset) → Depth Pro → BoxerNet (3D OBB) → JSON

Boxer outputs absolute metric dimensions, so no relative→absolute conversion is done.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


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

        # Lazy imports keep startup quick when only sub-features are exercised.
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
        return self.process_pil(image, image_id=image_id, image_url=image_url)

    def process_pil(
        self,
        image: Image.Image,
        image_id: Optional[int] = None,
        image_url: Optional[str] = None,
        enable_3d: Optional[bool] = None,
    ) -> PipelineResult:
        use_3d = self.enable_3d if enable_3d is None else enable_3d

        det = self.detector.detect(image)
        if len(det["boxes"]) == 0:
            return PipelineResult(image_id=image_id, image_url=image_url, objects=[])

        labels: List[str] = list(det["labels"])

        obb_by_idx: Dict[int, Any] = {}
        if use_3d and self.boxer is not None and self.depth_model is not None:
            depth_result = self.depth_model.estimate(image)
            obbs = self.boxer.lift(
                image=image,
                bboxes_xyxy=det["boxes"],
                labels=labels,
                depth=depth_result.depth,
                focal_length_px=depth_result.focal_length_px,
            )
            obb_by_idx = {obb.input_index: obb for obb in obbs}

        objects: List[DetectedObject] = []
        for i, box in enumerate(det["boxes"]):
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            obb = obb_by_idx.get(i)
            objects.append(
                DetectedObject(
                    image_id=image_id,
                    label=labels[i],
                    confidence=float(det["scores"][i]),
                    bbox_xyxy=[float(v) for v in box],
                    center_xy=(float(cx), float(cy)),
                    width_mm=round(obb.width_m * 1000.0, 1) if obb else 0.0,
                    depth_mm=round(obb.depth_m * 1000.0, 1) if obb else 0.0,
                    height_mm=round(obb.height_m * 1000.0, 1) if obb else 0.0,
                    volume_m3=round(obb.volume_m3, 6) if obb else 0.0,
                )
            )
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
                            "volume": o.volume_m3,
                            "center_x": round(o.center_xy[0], 1),
                            "center_y": round(o.center_xy[1], 1),
                        }
                        for o in r.objects
                    ],
                }
            )
        return {"results": payload}
