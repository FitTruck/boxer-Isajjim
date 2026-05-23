"""Furniture analysis routes.

- POST /analyze-furniture          (async + callback to backend, multi-device dispatch)
- POST /analyze-furniture-single   (sync, single URL)
- POST /analyze-furniture-base64   (sync, base64 image)
- POST /detect-furniture           (detection only, no 3D)

All endpoints acquire devices via `pool.pipeline_context()` so concurrent requests
truly fan out across CUDA / MPS slots instead of all hitting GPU 0.
"""

import asyncio
import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Tuple

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from PIL import Image

from ai.pipeline import FurniturePipeline, PipelineResult
from api.models import (
    AnalyzeFurnitureBase64Request,
    AnalyzeFurnitureRequest,
    AnalyzeFurnitureSingleRequest,
)
from api.services import send_callback

logger = logging.getLogger(__name__)
router = APIRouter()


@asynccontextmanager
async def _borrow_pipeline(task_id: str):
    """Acquire a pre-initialized pipeline from the pool, or build one on demand."""
    try:
        from ai.gpu import get_gpu_pool

        pool = get_gpu_pool()
    except Exception:
        pool = None

    if pool and any(pool.has_pipeline(d) for d in pool.devices):
        async with pool.pipeline_context(task_id=task_id) as (device, pipeline):
            yield pipeline, device
        return

    # Fallback: no pool / no pre-initialized pipelines — build ad-hoc.
    pipeline = FurniturePipeline()
    yield pipeline, pipeline.device or "cpu"


async def _analyze_and_callback(estimate_id: int, image_items: List[Tuple[int, str]]):
    """Run pipeline per image with real multi-device parallelism, then callback."""
    try:
        from ai.gpu import get_gpu_pool

        try:
            pool = get_gpu_pool()
        except Exception:
            pool = None

        if pool and any(pool.has_pipeline(d) for d in pool.devices):

            async def _one(image_id: int, url: str) -> PipelineResult:
                tid = f"est{estimate_id}_img{image_id}"
                try:
                    async with pool.pipeline_context(task_id=tid) as (device, pipeline):
                        return await pipeline.process_single_image(url, image_id=image_id)
                except Exception as e:
                    return PipelineResult(
                        image_id=image_id,
                        image_url=url,
                        error=f"{type(e).__name__}: {e}",
                    )

            results = await asyncio.gather(*(_one(iid, url) for iid, url in image_items))
        else:
            pipeline = FurniturePipeline()
            results = await pipeline.process_multiple_images(image_items)

        await send_callback(
            estimate_id, result_data=FurniturePipeline.to_json_response(results)
        )
    except Exception as e:
        logger.exception(f"[analyze-furniture] failed estimate_id={estimate_id}")
        await send_callback(estimate_id, error=f"Furniture analysis failed: {e}")


@router.post("/analyze-furniture")
async def analyze_furniture(
    request: AnalyzeFurnitureRequest,
    background_tasks: BackgroundTasks,
):
    image_items = [(item.id, item.url) for item in request.image_urls]
    background_tasks.add_task(
        _analyze_and_callback,
        estimate_id=request.estimate_id,
        image_items=image_items,
    )
    return {"success": True, "estimate_id": request.estimate_id, "status": "processing"}


@router.post("/analyze-furniture-single")
async def analyze_furniture_single(request: AnalyzeFurnitureSingleRequest):
    try:
        async with _borrow_pipeline(task_id="single") as (pipeline, _device):
            result = await pipeline.process_single_image(request.image_url)
        return JSONResponse(FurniturePipeline.to_json_response([result]))
    except Exception as e:
        logger.exception("analyze-furniture-single failed")
        return JSONResponse(
            status_code=500, content={"error": f"Furniture analysis failed: {e}"}
        )


@router.post("/analyze-furniture-base64")
async def analyze_furniture_base64(request: AnalyzeFurnitureBase64Request):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(request.image))).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid base64 image: {e}"})

    try:
        async with _borrow_pipeline(task_id="b64") as (pipeline, _device):
            result = await asyncio.to_thread(
                pipeline.process_pil, image, enable_3d=request.enable_3d
            )
        return JSONResponse(FurniturePipeline.to_json_response([result]))
    except Exception as e:
        logger.exception("analyze-furniture-base64 failed")
        return JSONResponse(
            status_code=500, content={"error": f"Furniture analysis failed: {e}"}
        )


@router.post("/detect-furniture")
async def detect_furniture(request: AnalyzeFurnitureBase64Request):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(request.image))).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid base64 image: {e}"})

    try:
        start = time.time()
        async with _borrow_pipeline(task_id="detect") as (pipeline, _device):
            result = await asyncio.to_thread(
                pipeline.process_pil, image, enable_3d=False
            )
        elapsed = time.time() - start
        return JSONResponse(
            {
                "success": True,
                "total_objects": len(result.objects),
                "processing_time_seconds": round(elapsed, 3),
                "objects": [
                    {
                        "label": o.label,
                        "bbox": o.bbox_xyxy,
                        "center_point": list(o.center_xy),
                        "confidence": round(o.confidence, 3),
                    }
                    for o in result.objects
                ],
            }
        )
    except Exception as e:
        logger.exception("detect-furniture failed")
        return JSONResponse(status_code=500, content={"error": f"Detection failed: {e}"})
