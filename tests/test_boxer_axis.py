"""Guard the OBB axis labeling in BoxerLifter._to_results.

BoxerNet's yaw rotates about gravity, so bb3_diagonal's 3rd component is the
vertical extent (height) while the first two are the horizontal footprint pair
that interchange under the 90-degree yaw symmetry. This pins that mapping so a
future refactor cannot silently re-swap height<->depth (the bug fixed here;
verified end-to-end: the 3rd axis is exactly world-vertical, |R[2,2]|=1.00).
"""

from types import SimpleNamespace

import pytest
import torch

from ai.pipeline import BoxerLifter


def _lifter(conf: float = 0.2) -> BoxerLifter:
    """BoxerLifter without loading the network (only _to_results is exercised)."""
    lifter = BoxerLifter.__new__(BoxerLifter)
    lifter.conf_threshold = conf
    return lifter


def _fake_obb(diag, prob: float = 0.9, center=(0.0, 0.0, 0.0)) -> SimpleNamespace:
    """Minimal stand-in exposing the ObbTW attributes _to_results reads."""
    w, d, h = diag
    return SimpleNamespace(
        prob=torch.tensor([[prob]], dtype=torch.float32),
        bb3_diagonal=torch.tensor([list(diag)], dtype=torch.float32),
        bb3_volumes=torch.tensor([[w * d * h]], dtype=torch.float32),
        bb3_center_world=torch.tensor([list(center)], dtype=torch.float32),
    )


def test_third_diagonal_component_is_height():
    # diag = (footprint_a, footprint_b, vertical): a tall, thin wardrobe.
    (res,) = _lifter()._to_results(_fake_obb((0.6, 0.4, 1.8)), ["wardrobe"])
    assert res.width_m == pytest.approx(0.6)   # 1st -> width  (horizontal)
    assert res.depth_m == pytest.approx(0.4)   # 2nd -> depth  (horizontal)
    assert res.height_m == pytest.approx(1.8)  # 3rd -> height (vertical; must NOT land in depth)
    assert res.label == "wardrobe"


def test_low_confidence_detection_dropped():
    out = _lifter(conf=0.2)._to_results(_fake_obb((0.6, 0.4, 1.8), prob=0.1), ["wardrobe"])
    assert out == []
