from __future__ import annotations

import pytest

from research.delta_analysis import compute_dimension_delta, compute_state_deltas


def test_absolute_delta():
    result = compute_dimension_delta("stability", 0.55, 0.47)
    assert result["absolute_delta"] == pytest.approx(-0.08)


def test_percent_delta_when_before_nonzero():
    result = compute_dimension_delta("conflict", 0.20, 0.32)
    assert result["percent_delta"] == 60.0


def test_percent_delta_null_when_before_zero():
    result = compute_dimension_delta("novelty", 0.0, 0.15)
    assert result["percent_delta"] is None


def test_compute_state_deltas_all_dimensions():
    before = {"stability": 0.55, "conflict": 0.20}
    after = {"stability": 0.47, "conflict": 0.32}
    results = compute_state_deltas(before, after, ["stability", "conflict"])
    assert len(results) == 2
    assert results[0]["absolute_delta"] == pytest.approx(-0.08)
    assert results[1]["absolute_delta"] == pytest.approx(0.12)
