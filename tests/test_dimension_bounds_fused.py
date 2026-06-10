"""Unit tests for the prob-weighted "fused" sanitizer mode (no model / GPU).

Properties pinned here:
  1. in-range values are never touched (identical to legacy),
  2. mild violation + confident model lands strictly between the violated
     bound and the class-typical midpoint (no hard clamp cliff),
  3. violations at/over the legacy severe cutoff (2x) collapse to typical,
  4. a coherent all-axes scale violation is undone aspect-preservingly,
  5. prob modulates how far the value is pulled toward the prior,
  6. legacy mode stays the default and bit-identical to sanitize_dims pre-P2.
"""

import math

import pytest

from ai.pipeline.dimension_bounds import sanitize_dims


def test_fused_in_range_untouched():
    w, d, h, corr = sanitize_dims("bed", 2000, 1500, 400, prob=0.9, mode="fused")
    assert (w, d, h) == (2000, 1500, 400)
    assert corr == []


def test_fused_mild_violation_lands_between_bound_and_typical():
    # bed height range (200, 700), typical 450. Raw 800 = mild violation.
    _, _, h, corr = sanitize_dims("bed", 2000, 1500, 800, prob=0.9, mode="fused")
    assert 450.0 < h < 700.0  # strictly inside (typical, bound)
    assert any("fused" in c for c in corr)


def test_fused_severe_violation_collapses_to_typical():
    # bed height 2375 > 2*700 -> lambda floor 0 -> exactly class-typical 450.
    _, _, h, _ = sanitize_dims("bed", 2062, 1372, 2375, prob=0.99, mode="fused")
    assert h == pytest.approx(450.0)


def test_fused_low_prob_pulls_harder_toward_typical():
    _, _, h_conf, _ = sanitize_dims("bed", 2000, 1500, 800, prob=0.95, mode="fused")
    _, _, h_unsure, _ = sanitize_dims("bed", 2000, 1500, 800, prob=0.10, mode="fused")
    assert h_unsure < h_conf  # less confidence -> closer to typical(450)
    assert h_unsure > 450.0


def test_fused_scale_coherent_violation_preserves_aspect():
    # bed all-axes ~2x too big: long 4200 (>2150), short 3000 (>1900),
    # height 1300 (>700) -> one shared factor, aspect ratios preserved.
    w, d, h, corr = sanitize_dims("bed", 4200, 3000, 1300, prob=0.9, mode="fused")
    assert any("scale" in c for c in corr)
    raw_ratio_wd = 4200 / 3000
    raw_ratio_wh = 4200 / 1300
    # After the scale branch (+ possible residual fusion) ratios stay close.
    assert w / d == pytest.approx(raw_ratio_wd, rel=0.15)
    assert w / h == pytest.approx(raw_ratio_wh, rel=0.35)
    assert 1950 <= w <= 2150  # pulled back into (or onto) the range


def test_fused_mixed_direction_violation_no_scale_branch():
    # height too small but width too big -> NOT a coherent scale error.
    _, _, _, corr = sanitize_dims("bed", 4500, 1500, 90, prob=0.9, mode="fused")
    assert not any("scale" in c for c in corr)


def test_fused_degenerate_nonpositive_replaced_with_typical():
    _, _, h, corr = sanitize_dims("bed", 2000, 1500, 0.0, prob=0.9, mode="fused")
    assert h == pytest.approx(450.0)
    assert any("degenerate" in c for c in corr)


def test_fused_prob_none_defaults_to_half():
    _, _, h_none, _ = sanitize_dims("bed", 2000, 1500, 800, mode="fused")
    _, _, h_half, _ = sanitize_dims("bed", 2000, 1500, 800, prob=0.5, mode="fused")
    assert h_none == pytest.approx(h_half)


def test_default_mode_is_legacy_clamp():
    # No mode argument -> exact legacy clamp behavior (700 for bed height 760).
    _, _, h, corr = sanitize_dims("bed", 2000, 1500, 760)
    assert h == 700
    assert any("clamp" in c for c in corr)


def test_fused_skip_and_unknown_classes_passthrough():
    w, d, h, corr = sanitize_dims("plant", 5000, 10, 9000, prob=0.1, mode="fused")
    assert (w, d, h) == (5000, 10, 9000)
    assert corr == []


@pytest.mark.parametrize("prob", [1.0, 0.5, 0.1])
def test_fused_continuity_at_boundary_for_all_prob(prob):
    # Crossing the range boundary must not jump, REGARDLESS of prob: the
    # original lam = prob * alpha formula had a -139mm cliff at prob=0.5
    # (caught in code review).
    _, _, h_in, _ = sanitize_dims("bed", 2000, 1500, 700.0, prob=prob, mode="fused")
    _, _, h_out, _ = sanitize_dims("bed", 2000, 1500, 700.5, prob=prob, mode="fused")
    assert h_in == 700.0
    assert abs(h_out - h_in) < 5.0


def test_fused_footprint_swap_consistent_with_legacy():
    # chair (long/short 400-750): depth passed larger than width; the long/short
    # mapping must round-trip the same way the legacy path does.
    w, d, h, _ = sanitize_dims("chair", 300, 900, 800, prob=1.0, mode="fused")
    assert d > w  # larger raw axis stays the larger corrected axis
    assert h == 800


@pytest.mark.parametrize("mode", ["clamp", "fused"])
def test_footprint_inversion_repaired(mode):
    # recliner: long (700,1100) but short (800,1100) sits ABOVE long's minimum.
    # Raw long=720 (in range) + short=300 (severe -> typical 950) used to yield
    # short > long; the ordering invariant must be restored.
    w, d, h, corr = sanitize_dims("recliner", 720, 300, 900, prob=0.9, mode=mode)
    assert max(w, d) >= min(w, d)
    assert w >= d  # raw width was the larger axis; must stay the larger one
    assert any("reordered" in c for c in corr)
