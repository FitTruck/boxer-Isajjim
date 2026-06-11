"""FastAPI application entry point.

Lifespan:
- Initialize GPU pool (one FurniturePipeline = OWLv2 + Depth + Boxer per GPU)
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

# Make `ai` importable when running from repo root or `uvicorn api:app`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from api.config import device
from api.routes import furniture_router, health_router

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from api.config import (
        CALLBACK_TIMEOUT_SECONDS,
        CALLBACK_URL_TEMPLATE,
        X_INTERNAL_TOKEN,
    )

    token_status = (
        f"set ({len(X_INTERNAL_TOKEN)} chars)"
        if X_INTERNAL_TOKEN
        else "MISSING — backend will reject callbacks"
    )
    logger.info("=" * 64)
    logger.info("Boxer AI server starting up")
    logger.info("  device                : %s", device)
    logger.info("  log level             : %s", os.environ.get("LOG_LEVEL", "INFO"))
    logger.info("  callback URL template : %s", CALLBACK_URL_TEMPLATE)
    logger.info("  callback timeout (s)  : %s", CALLBACK_TIMEOUT_SECONDS)
    logger.info("  AUTH_TOKEN            : %s", token_status)
    # torch / CUDA build info — cuBLASLt 'cublasLtCreate' aborts usually mean the
    # torch-bundled CUDA libs mismatch the driver/runtime. Print versions to diagnose.
    try:
        import torch

        logger.info("  torch version         : %s", torch.__version__)
        logger.info("  torch CUDA build      : %s", torch.version.cuda)
        logger.info("  cuDNN version         : %s", torch.backends.cudnn.version())
        if torch.cuda.is_available():
            logger.info("  CUDA device           : %s", torch.cuda.get_device_name(0))
            logger.info("  CUDA capability       : %s", torch.cuda.get_device_capability(0))
    except Exception as e:
        logger.warning("  torch/CUDA info unavailable: %s", e)
    logger.info("=" * 64)

    try:
        from ai.config import Config
        from ai.gpu import initialize_gpu_pool
        from ai.pipeline import FurniturePipeline

        devices = Config.get_available_devices() or ["cpu"]
        logger.info("Initializing GPU pool on devices: %s", devices)
        pool = initialize_gpu_pool(devices)
        await pool.initialize_pipelines(
            lambda device: FurniturePipeline(device=device),
            skip_on_error=True,
        )
        ready = [d for d in devices if pool.has_pipeline(d)]
        failed = [d for d in devices if not pool.has_pipeline(d)]
        logger.info("GPU pool ready — pipelines on: %s", ready or "NONE")
        if failed:
            logger.warning("Pipelines failed to initialize on: %s", failed)
    except Exception as e:
        logger.exception("Pipeline pre-initialization failed: %s", e)

    logger.info("Startup complete — ready to accept requests")
    yield

    try:
        from ai.gpu import shutdown_gpu_pool

        await shutdown_gpu_pool()
    except Exception as e:
        logger.warning(f"Shutdown cleanup failed: {e}")


app = FastAPI(
    title="Boxer Furniture Analysis API",
    description="OWLv2 + Depth Pro + BoxerNet 3D OBB lifting (no PLY/GCS).",
    version="1.0.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every inbound request so backend connectivity is visible in the console."""
    client = request.client.host if request.client else "unknown"
    logger.info("--> %s %s (from %s)", request.method, request.url.path, client)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "<-- %s %s raised after %.0fms", request.method, request.url.path, elapsed_ms
        )
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "<-- %s %s %s (%.0fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


app.include_router(health_router, tags=["Health"])
app.include_router(furniture_router, tags=["Furniture Analysis"])
