"""AI module configuration."""

import os
from typing import List, Optional

import torch


class Config:
    # ---------------- OWLv2 detection ----------------
    OWLV2_MODEL: str = os.environ.get(
        "OWLV2_MODEL", "google/owlv2-base-patch16-ensemble"
    )
    OWLV2_CONFIDENCE: float = float(os.environ.get("OWLV2_CONFIDENCE", "0.25"))
    OWLV2_CHUNK_SIZE: int = int(os.environ.get("OWLV2_CHUNK_SIZE", "256"))
    OWLV2_CLASSES_CSV: Optional[str] = os.environ.get("OWLV2_CLASSES_CSV")

    # ---------------- Depth estimation ----------------
    # DEPTH_BACKEND: "depthpro" (default, metric + focal length) | "da2" (lighter fallback)
    DEPTH_BACKEND: str = os.environ.get("DEPTH_BACKEND", "depthpro").lower()
    DEPTH_PRO_MODEL: str = os.environ.get("DEPTH_PRO_MODEL", "apple/DepthPro-hf")
    DEPTH_DA2_MODEL: str = os.environ.get(
        "DEPTH_DA2_MODEL", "depth-anything/Depth-Anything-V2-Small-hf"
    )

    # Clamp BoxerNet output dims to per-class physical ranges (ai/pipeline/dimension_bounds.py).
    SANITIZE_DIMENSIONS: bool = os.environ.get("SANITIZE_DIMENSIONS", "true").lower() == "true"
    # SANITIZE_MODE: "clamp" (legacy binary clamp/replace) | "fused"
    # (prob-weighted continuous prior fusion + aspect-preserving scale branch).
    SANITIZE_MODE: str = os.environ.get("SANITIZE_MODE", "clamp").lower()


    # ---------------- Boxer (3D OBB lifting) ----------------
    BOXER_REPO_PATH: str = os.environ.get(
        "BOXER_REPO_PATH",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "boxer"),
    )
    BOXER_CHECKPOINT: Optional[str] = os.environ.get("BOXER_CHECKPOINT")
    # Inference autocast precision for BoxerNet. "auto" picks per GPU
    # architecture (bf16 on Ampere+/L4, fp16 on Turing/T4, fp32 otherwise);
    # override with "bf16" | "fp16" | "fp32". See ai/gpu/precision.py.
    BOXER_AUTOCAST: str = os.environ.get("BOXER_AUTOCAST", "auto").lower()

    # ---------------- Device pool ----------------
    # Explicit override: DEVICES="cuda:0,cuda:1" / "mps" / "cpu" (highest precedence)
    # Legacy: GPU_IDS="0,1,2,3" → expanded to "cuda:N,..."
    # Otherwise auto-detect CUDA → MPS → CPU.
    DEVICES: Optional[str] = os.environ.get("DEVICES")
    GPU_IDS: Optional[str] = os.environ.get("GPU_IDS")
    ENABLE_MULTI_GPU: bool = os.environ.get("ENABLE_MULTI_GPU", "true").lower() == "true"

    @staticmethod
    def get_available_devices() -> List[str]:
        """Return device strings to allocate to the GPU pool."""
        if Config.DEVICES:
            return [d.strip() for d in Config.DEVICES.split(",") if d.strip()]
        if Config.GPU_IDS:
            return [f"cuda:{i.strip()}" for i in Config.GPU_IDS.split(",") if i.strip()]
        if torch.cuda.is_available():
            count = torch.cuda.device_count() if Config.ENABLE_MULTI_GPU else 1
            return [f"cuda:{i}" for i in range(count)]
        if torch.backends.mps.is_available():
            return ["mps"]
        return ["cpu"]

    @staticmethod
    def get_default_device() -> str:
        """Single best device when caller does not specify (for ad-hoc instantiations)."""
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
