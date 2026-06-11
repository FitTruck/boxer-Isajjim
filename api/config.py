"""API configuration. Must set CUDA-related env vars BEFORE importing torch.

INVARIANT: this module must be the process's first importer of torch (api/app.py
imports it before anything else touches torch). A module that imports torch
before api.config would initialize torch without OMP/MKL thread caps and the
MPS fallback flag.
"""

import os

# ---- Pre-torch env -------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

torch.set_default_dtype(torch.float32)
torch.set_num_threads(4)

from ai.config import Config  # noqa: E402  (must come after the pre-torch env)

# Single source of truth for device auto-detection (ai.config). The split
# drops the ":0" ordinal so the /health payload string stays "cuda" exactly.
device = torch.device(Config.get_default_device().split(":")[0])

# ---- Callback ------------------------------------------------------------
CALLBACK_URL_TEMPLATE: str = os.environ.get(
    "CALLBACK_URL_TEMPLATE",
    "https://api.isajjim.kro.kr/api/v1/estimates/{estimateId}/callback",
)
CALLBACK_TIMEOUT_SECONDS: int = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))
CALLBACK_RETRY_COUNT: int = int(os.environ.get("CALLBACK_RETRY_COUNT", "1"))
# Shared secret with Isajjim-Backend (`auth.internal-token` ← `AUTH_TOKEN`).
# Sent as `X_INTERNAL_TOKEN` header; backend rejects requests without it.
X_INTERNAL_TOKEN: str = os.environ.get("AUTH_TOKEN", "")
