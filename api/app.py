"""FastAPI application entry point.

Lifespan:
- Initialize GPU pool (one FurniturePipeline = YOLOE + Depth + Boxer per GPU)
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

# Make `ai` importable when running from repo root or `uvicorn api:app`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from api.config import device
from api.routes import furniture_router, health_router

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Boxer AI server starting on device: {device}")
    try:
        from ai.config import Config
        from ai.gpu import initialize_gpu_pool
        from ai.pipeline import FurniturePipeline

        devices = Config.get_available_devices() or ["cpu"]
        pool = initialize_gpu_pool(devices)
        logger.info(f"GPU pool created with devices: {devices}")
        await pool.initialize_pipelines(
            lambda device: FurniturePipeline(device=device),
            skip_on_error=True,
        )
    except Exception as e:
        logger.exception(f"Pipeline pre-initialization failed: {e}")

    yield

    try:
        from ai.gpu import shutdown_gpu_pool

        await shutdown_gpu_pool()
    except Exception as e:
        logger.warning(f"Shutdown cleanup failed: {e}")


app = FastAPI(
    title="Boxer Furniture Analysis API",
    description="YOLOE-26x-seg + Depth + BoxerNet 3D OBB lifting (no PLY/GCS).",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health_router, tags=["Health"])
app.include_router(furniture_router, tags=["Furniture Analysis"])
