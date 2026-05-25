"""API configuration. Must set CUDA-related env vars BEFORE importing torch."""

import os

# ---- Pre-torch env -------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

torch.set_default_dtype(torch.float32)
torch.set_num_threads(4)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()

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
