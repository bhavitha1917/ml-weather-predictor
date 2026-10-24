"""
src/intervals.py
The 80% prediction interval served by the API, shared with the offline evaluation so that
the reported coverage is measured on exactly the interval users see.

Method (blueprint): p10 and p90 are the 10th and 90th percentiles of the ensemble's
out-of-fold residuals (actual - predicted) from 5-fold TimeSeriesSplit on the training
years only. For a new prediction y_hat the interval is [y_hat + p10, y_hat + p90].
Nominal coverage is 80% if future errors resemble the training-period errors.
"""

import numpy as np

NOMINAL_COVERAGE = 0.80


def interval_bounds(prediction, p10, p90):
    """Lower and upper bound (rounded to 0.1 C, as displayed) for scalar or array predictions."""
    lower = np.round(np.asarray(prediction, dtype=float) + p10, 1)
    upper = np.round(np.asarray(prediction, dtype=float) + p90, 1)
    if lower.ndim == 0:
        return float(lower), float(upper)
    return lower, upper


def coverage_pct(y_true, lower, upper) -> float:
    """Percentage of actual values inside [lower, upper] (bounds inclusive)."""
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((y_true >= lower) & (y_true <= upper)) * 100.0)


def coverage_interval_95(coverage_percent: float, n: int) -> list:
    """
    Approximate 95% binomial (normal-approximation) range for a coverage estimate from n days.
    Daily errors are autocorrelated, so the true uncertainty is larger than this.
    """
    p = coverage_percent / 100.0
    half_width = float(1.96 * np.sqrt(p * (1 - p) / n) * 100.0)
    return [round(coverage_percent - half_width, 1), round(coverage_percent + half_width, 1)]
