"""
tests/test_evaluation.py
Interval arithmetic, coverage calculation, and reproducibility of the saved test-year results.
"""

import json
import os

import numpy as np
import pytest

from src.evaluate import REPORT_PATH, evaluate_city
from src.intervals import coverage_interval_95, coverage_pct, interval_bounds
from src.train import MODELS_DIR

CITIES = ["bengaluru", "delhi", "mumbai"]

with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
    SAVED = json.load(f)


def test_interval_bounds_scalar_and_array():
    assert interval_bounds(30.04, -1.86, 1.21) == (28.2, 31.2)
    lower, upper = interval_bounds(np.array([30.0, 25.55]), -2.0, 1.5)
    assert np.allclose(lower, [28.0, 23.6]) and np.allclose(upper, [31.5, 27.0])


@pytest.mark.parametrize("city", CITIES)
def test_interval_offsets_are_ordered(city):
    p10, p90 = SAVED[city]["interval_80"]["p10"], SAVED[city]["interval_80"]["p90"]
    assert p10 < 0 < p90
    lower, upper = interval_bounds(np.linspace(10, 45, 50), p10, p90)
    assert np.all(lower < upper)


def test_coverage_pct_counts_inclusive_bounds():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert coverage_pct(y, np.array([1.0, 2.5, 0.0, 4.1]), np.array([1.0, 3.0, 2.9, 5.0])) == 25.0


def test_coverage_interval_95_is_symmetric_and_bounded():
    lo, hi = coverage_interval_95(80.0, 365)
    assert lo < 80.0 < hi and round(80.0 - lo, 1) == round(hi - 80.0, 1)


@pytest.mark.parametrize("city", CITIES)
def test_saved_metrics_are_reproducible_from_artifacts(city):
    report = evaluate_city(city, SAVED[city], with_validation=False)
    assert report["max_abs_diff_vs_metrics_json"] == 0.0
    assert report["n_test"] == 365
    assert report["test_period"] == ["2023-01-01", "2023-12-31"]


@pytest.mark.parametrize("city", CITIES)
def test_served_interval_coverage_sanity_range(city):
    """
    Regression guard kept from the original suite (60-90%), now applied to the interval that is
    actually served. It catches a broken interval; it is NOT a claim that coverage equals 80%
    (measured values are reported in evaluation_report.json and the README).
    """
    report = evaluate_city(city, SAVED[city], with_validation=False)
    assert 60.0 <= report["served_interval"]["test_coverage_pct"] <= 90.0


@pytest.mark.parametrize("city", CITIES)
def test_saved_evaluation_report_matches_recomputation(city):
    with open(REPORT_PATH) as f:
        saved_report = json.load(f)[city]
    report = evaluate_city(city, SAVED[city], with_validation=False)
    assert report["served_interval"] == saved_report["served_interval"]
    assert report["main_model_beats"] == saved_report["main_model_beats"]
