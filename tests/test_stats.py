from __future__ import annotations

import pytest

from spritelab.training.stats import wilson_ci_from_rate, wilson_confidence_interval


def test_wilson_confidence_interval_returns_sane_bounds_and_contains_point_estimate() -> None:
    lower, upper = wilson_confidence_interval(81, 96)
    assert 0.0 <= lower < 81 / 96 < upper <= 1.0


def test_wilson_confidence_interval_narrows_with_larger_n() -> None:
    small_lower, small_upper = wilson_confidence_interval(81, 96)
    large_lower, large_upper = wilson_confidence_interval(81 * 3, 96 * 3)
    assert (large_upper - large_lower) < (small_upper - small_lower)


def test_wilson_confidence_interval_handles_zero_n() -> None:
    assert wilson_confidence_interval(0, 0) == (0.0, 1.0)


def test_wilson_confidence_interval_handles_extreme_rates() -> None:
    lower_all, upper_all = wilson_confidence_interval(10, 10)
    assert upper_all == pytest.approx(1.0)
    assert lower_all > 0.0

    lower_none, upper_none = wilson_confidence_interval(0, 10)
    assert lower_none == 0.0
    assert upper_none < 1.0


def test_wilson_ci_from_rate_matches_raw_counts() -> None:
    from_counts = wilson_confidence_interval(81, 96)
    from_rate = wilson_ci_from_rate(81 / 96, 96)
    assert from_rate is not None
    assert from_rate[0] == from_counts[0]
    assert from_rate[1] == from_counts[1]


def test_wilson_ci_from_rate_returns_none_for_missing_rate_or_empty_n() -> None:
    assert wilson_ci_from_rate(None, 96) is None
    assert wilson_ci_from_rate(0.5, 0) is None
