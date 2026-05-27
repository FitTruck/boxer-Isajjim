"""Unit tests for the per-class dimension sanitizer (no model / GPU needed)."""

import os

import pytest

from ai.pipeline.dimension_bounds import _BOUNDS, _SKIP, sanitize_dims


def test_in_range_values_kept():
    w, d, h, corr = sanitize_dims("bed", 2000, 1500, 400)
    assert (w, d, h) == (2000, 1500, 400)
    assert corr == []


def test_severe_outlier_replaced_with_typical():
    # bed height 2375 > 2*max(700) -> class-typical midpoint (200+700)/2 = 450
    w, d, h, corr = sanitize_dims("bed", 2062, 1372, 2375)
    assert (w, d) == (2062, 1372)  # horizontals in range, untouched
    assert h == pytest.approx(450.0)
    assert any("severe" in c for c in corr)


def test_mild_outlier_clamped_to_nearest_bound():
    # bed height 760 over max 700 but < 2*max -> clamp to 700
    _, _, h, corr = sanitize_dims("bed", 2000, 1500, 760)
    assert h == 700
    assert any("clamp" in c for c in corr)


def test_horizontal_footprint_swap():
    # chair: long/short ranges (400,750). Pass depth>width; larger axis must be
    # bounded by `long`, smaller by `short`, then mapped back to (w, d).
    w, d, h, _ = sanitize_dims("chair", 300, 900, 800)
    assert w == 400  # smaller -> short min
    assert d == 750  # larger  -> long max
    assert h == 800  # in range


def test_unknown_class_passthrough():
    w, d, h, corr = sanitize_dims("totally unknown thing", 9999, 1, 5000)
    assert (w, d, h) == (9999, 1, 5000)
    assert corr == []


def test_noncuboidal_class_passthrough():
    assert "curtain" in _SKIP
    w, d, h, corr = sanitize_dims("curtain", 5000, 10, 9000)
    assert (w, d, h) == (5000, 10, 9000)
    assert corr == []


def test_label_is_case_insensitive():
    _, _, h, _ = sanitize_dims("BED", 2062, 1372, 2375)
    assert h == pytest.approx(450.0)


def test_owlv2_vocabulary_fully_covered():
    """Every OWLv2 class must be bounded or explicitly skipped (guards regressions)."""
    csv = os.path.join(os.path.dirname(__file__), "..", "ai", "pipeline", "lvisplus_classes.csv")
    labels = [ln.strip().replace("_", " ").lower() for ln in open(csv, encoding="utf-8") if ln.strip()]
    missing = [lbl for lbl in labels if lbl not in _BOUNDS and lbl not in _SKIP]
    assert missing == [], f"classes with no bounds and not skipped: {missing}"
