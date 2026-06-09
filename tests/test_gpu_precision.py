"""Unit tests for GPU-architecture-aware autocast precision selection.

CUDA is mocked (``torch.cuda.is_available`` / ``get_device_capability``) so the
T4-vs-L4 dtype logic is verified without a real GPU.
"""

import pytest
import torch

from ai.gpu.precision import autocast_label, select_autocast_dtype


def _fake_cuda(monkeypatch, capability):
    """Pretend CUDA is present with the given (major, minor) capability."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )


class TestArchitectureAutoDetect:
    def test_l4_ada_selects_bf16(self, monkeypatch):
        _fake_cuda(monkeypatch, (8, 9))  # L4 = Ada Lovelace
        assert select_autocast_dtype("cuda:0") is torch.bfloat16

    def test_a100_ampere_selects_bf16(self, monkeypatch):
        _fake_cuda(monkeypatch, (8, 0))  # A100 = Ampere
        assert select_autocast_dtype("cuda:0") is torch.bfloat16

    def test_hopper_selects_bf16(self, monkeypatch):
        _fake_cuda(monkeypatch, (9, 0))  # H100 = Hopper
        assert select_autocast_dtype("cuda:0") is torch.bfloat16

    def test_t4_turing_selects_fp16(self, monkeypatch):
        _fake_cuda(monkeypatch, (7, 5))  # T4 = Turing -> no hw bf16
        assert select_autocast_dtype("cuda:0") is torch.float16

    def test_v100_volta_selects_fp16(self, monkeypatch):
        _fake_cuda(monkeypatch, (7, 0))  # V100 = Volta
        assert select_autocast_dtype("cuda:0") is torch.float16

    def test_pre_volta_selects_fp32(self, monkeypatch):
        _fake_cuda(monkeypatch, (6, 1))  # Pascal -> fp32 (None)
        assert select_autocast_dtype("cuda:0") is None


class TestNonCuda:
    def test_cpu_is_fp32(self):
        assert select_autocast_dtype("cpu") is None

    def test_mps_is_fp32(self):
        assert select_autocast_dtype("mps") is None

    def test_cuda_string_but_unavailable_is_fp32(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert select_autocast_dtype("cuda:0") is None

    def test_capability_probe_failure_is_fp32(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def _boom(device=None):
            raise RuntimeError("no driver")

        monkeypatch.setattr(torch.cuda, "get_device_capability", _boom)
        assert select_autocast_dtype("cuda:0") is None


class TestOverride:
    def test_override_forces_fp16_on_l4(self, monkeypatch):
        _fake_cuda(monkeypatch, (8, 9))
        assert select_autocast_dtype("cuda:0", override="fp16") is torch.float16

    def test_override_forces_bf16_on_t4(self, monkeypatch):
        _fake_cuda(monkeypatch, (7, 5))
        assert select_autocast_dtype("cuda:0", override="bf16") is torch.bfloat16

    def test_override_fp32_disables_autocast(self, monkeypatch):
        _fake_cuda(monkeypatch, (8, 9))
        assert select_autocast_dtype("cuda:0", override="fp32") is None

    def test_override_is_case_insensitive(self, monkeypatch):
        _fake_cuda(monkeypatch, (7, 5))
        assert select_autocast_dtype("cuda:0", override="BF16") is torch.bfloat16

    def test_unknown_override_falls_back_to_auto(self, monkeypatch):
        _fake_cuda(monkeypatch, (7, 5))  # T4 -> fp16 by auto-detect
        assert select_autocast_dtype("cuda:0", override="garbage") is torch.float16

    def test_empty_override_falls_back_to_auto(self, monkeypatch):
        _fake_cuda(monkeypatch, (8, 0))
        assert select_autocast_dtype("cuda:0", override="") is torch.bfloat16


class TestDeviceIndexParsing:
    def test_plain_cuda_uses_current_device(self, monkeypatch):
        seen = {}

        def _cap(device=None):
            seen["device"] = device
            return (8, 9)

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", _cap)
        assert select_autocast_dtype("cuda") is torch.bfloat16
        assert seen["device"] is None  # None -> torch picks the current device

    def test_indexed_cuda_passes_ordinal(self, monkeypatch):
        seen = {}

        def _cap(device=None):
            seen["device"] = device
            return (7, 5)

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", _cap)
        assert select_autocast_dtype("cuda:3") is torch.float16
        assert seen["device"] == 3


class TestLabel:
    def test_labels(self):
        assert autocast_label(torch.bfloat16) == "bf16"
        assert autocast_label(torch.float16) == "fp16"
        assert autocast_label(None) == "fp32"
