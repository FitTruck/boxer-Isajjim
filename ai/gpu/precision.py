"""GPU-architecture-aware autocast precision selection.

Tensor-Core math precision differs by NVIDIA architecture, so the right
inference dtype depends on which GPU we land on:

- Ampere / Ada / Hopper  (compute capability >= 8.0; e.g. **L4**, A10, A100,
  H100): hardware **bfloat16**. bf16 keeps fp32's 8-bit exponent range, so it
  is the safe *and* fast choice for metric 3D geometry.
- Volta / Turing         (7.0 <= cc < 8.0; e.g. V100, **T4**): no hardware
  bf16 — it is emulated and therefore slow. Their fast Tensor-Core path is
  **float16**.
- Pre-Volta / non-CUDA:   no usable half-precision Tensor Cores -> fp32.

``torch.cuda.is_bf16_supported()`` is intentionally NOT used here: on Turing it
can report ``True`` via slow emulation, which is exactly the T4-vs-L4 trap this
module exists to avoid.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Accepts the same precision keys BoxerNet's DINOv3 wrapper uses, plus a few
# aliases. A value of ``None`` means "no autocast" (run in fp32).
_OVERRIDE_DTYPES = {
    "fp32": None,
    "float32": None,
    "none": None,
    "off": None,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}

_DTYPE_LABELS = {
    None: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}


def autocast_label(dtype: Optional[torch.dtype]) -> str:
    """Human-readable tag for logs: 'bf16' / 'fp16' / 'fp32'."""
    return _DTYPE_LABELS.get(dtype, str(dtype))


def _device_index(device: str) -> Optional[int]:
    """Extract the CUDA ordinal from 'cuda:N', or None for plain 'cuda'."""
    if ":" in device:
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return None
    return None


def select_autocast_dtype(
    device: str,
    override: str = "auto",
) -> Optional[torch.dtype]:
    """Pick the autocast dtype appropriate for ``device``'s GPU architecture.

    Returns the dtype to pass to ``torch.autocast``, or ``None`` to run in fp32
    (no autocast).

    Args:
        device: torch device string ("cuda:0" / "cuda" / "mps" / "cpu").
        override: force a precision regardless of hardware —
            ``auto`` (detect from compute capability) | ``bf16`` | ``fp16`` |
            ``fp32``. Unknown values fall back to ``auto`` with a warning.
    """
    # Only CUDA has the bf16/fp16 Tensor-Core split this resolves. MPS/CPU -> fp32.
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None

    choice = (override or "auto").strip().lower()
    if choice and choice != "auto":
        if choice in _OVERRIDE_DTYPES:
            return _OVERRIDE_DTYPES[choice]
        logger.warning(
            "Unknown autocast override %r; auto-detecting "
            "(expected one of: auto, bf16, fp16, fp32).",
            choice,
        )

    try:
        major, _minor = torch.cuda.get_device_capability(_device_index(device))
    except Exception as exc:  # never let a probe failure break inference
        logger.warning("Could not read CUDA capability for %s (%s); using fp32.", device, exc)
        return None

    if major >= 8:
        return torch.bfloat16  # Ampere / Ada / Hopper (L4, A10, A100, H100, ...)
    if major == 7:
        return torch.float16   # Volta / Turing (T4, V100): fp16 Tensor Cores, no hw bf16
    return None                # pre-Volta: fp32
