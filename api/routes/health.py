"""/health, /gpu-status."""

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.config import device

router = APIRouter()


@router.get("/")
async def root():
    """Root liveness response so health/uptime probes hitting `/` get 200, not 404."""
    return {
        "service": "Boxer Furniture Analysis API",
        "status": "ok",
        "endpoints": {"health": "/health", "gpu_status": "/gpu-status", "docs": "/docs"},
    }


@router.get("/health")
async def health_check():
    return {"status": "healthy", "device": str(device)}


@router.get("/gpu-status")
async def gpu_status():
    try:
        from ai.gpu import get_gpu_pool

        return JSONResponse(get_gpu_pool().get_status())
    except Exception as e:
        return JSONResponse(
            {
                "error": str(e),
                "total_gpus": 1 if torch.cuda.is_available() else 0,
                "available_gpus": 1 if torch.cuda.is_available() else 0,
                "pipelines_initialized": 0,
                "gpus": {},
            }
        )
