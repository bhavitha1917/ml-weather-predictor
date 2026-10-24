"""
tests/test_features.py
Verifies zero data leakage as required by Blueprint Prompt 2 and Section 4 Tests:
"The features for date t must not change when data after t is removed (no leakage)",
plus the target definition, feature schema and baseline rules.
"""

import os
import pandas as pd
import numpy as np
import pytest
from src.features import (
    FEATURE_COLUMNS,
    FORECAST_HORIZON_DAYS,
    REQUIRED_HISTORY_DAYS,
    compute_climatology_map,
    make_features,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "bengaluru.csv")
CITIES = ["bengaluru", "delhi", "mumbai"]


def load(city="bengaluru"):
    df = pd.read_csv(os.path.join(BASE_DIR, "data", f"{city}.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.mark.parametrize("city", CITIES)
@pytest.mark.parametrize("cutoff_idx", [6, 100, 1000, 1825])
def test_no_data_leakage(city, cutoff_idx):
    """Feature values for date t do NOT change when all data after t is removed."""
    df = load(city)
    clim_map = compute_climatology_map(df[df["date"].dt.year < 2023])
    cutoff_date = df.iloc[cutoff_idx]["date"]

    feats_full = make_features(df, climatology_map=clim_map, is_training=False)
    feats_trunc = make_features(df.iloc[:cutoff_idx + 1], climatology_map=clim_map, is_training=False)
    row_full = feats_full[feats_full["date"] == cutoff_date]
    row_trunc = feats_trunc[feats_trunc["date"] == cutoff_date]
    assert len(row_full) == len(row_trunc) == 1
    cols = FEATURE_COLUMNS + ["baseline_persistence", "baseline_climatology"]
    np.testing.assert_allclose(row_full[cols].values, row_trunc[cols].values, atol=1e-9)


def test_features_depend_only_on_last_seven_days():
    """Changing any value before D-6 must not change the features for day D."""
    df = load()
    d_idx = 500
    window = df.iloc[d_idx - REQUIRED_HISTORY_DAYS + 1: d_idx + 1]
    altered = df.iloc[:d_idx + 1].copy()
    altered.iloc[: d_idx - REQUIRED_HISTORY_DAYS + 1, 1:5] = 999.0
    a = make_features(window, is_training=False)
    b = make_features(altered, is_training=False).iloc[[-1]]
    np.testing.assert_allclose(a[FEATURE_COLUMNS].values, b[FEATURE_COLUMNS].values)


def test_lags_use_the_intended_dates():
    df = load()
    feats = make_features(df, is_training=False).set_index("date")
    s = df.set_index("date")["temperature_2m_max"]
    d = pd.Timestamp("2021-05-20")
    for lag in range(7):
        assert feats.loc[d, f"temp_lag_{lag}"] == s.loc[d - pd.Timedelta(days=lag)]
    assert np.isclose(feats.loc[d, "temp_roll_mean_7"], s.loc[d - pd.Timedelta(days=6): d].mean())
    assert np.isclose(feats.loc[d, "precip_sum_3d"],
                      df.set_index("date")["precipitation_sum"].loc[d - pd.Timedelta(days=2): d].sum())


def test_target_is_max_temperature_two_days_after_d():
    df = load()
    feats = make_features(df, climatology_map=compute_climatology_map(df), is_training=True)
    s = df.set_index("date")["temperature_2m_max"]
    assert FORECAST_HORIZON_DAYS == 2
    assert (feats["target_date"] - feats["date"] == pd.Timedelta(days=2)).all()
    assert np.allclose(feats["target_temp_d2"].values, s.loc[feats["target_date"]].values)


def test_cyclical_features_use_target_date():
    df = load()
    feats = make_features(df, is_training=False)
    doy = feats["target_date"].dt.dayofyear
    assert np.allclose(feats["sin_doy"], np.sin(2 * np.pi * doy / 365.25))
    assert np.allclose(feats["cos_doy"], np.cos(2 * np.pi * doy / 365.25))


def test_feature_schema_is_fixed():
    assert len(FEATURE_COLUMNS) == 22 and len(set(FEATURE_COLUMNS)) == 22
    assert FEATURE_COLUMNS[:7] == [f"temp_lag_{i}" for i in range(7)]


@pytest.mark.parametrize("bad", ["gap", "duplicate"])
def test_non_consecutive_dates_are_rejected(bad):
    df = load().iloc[:30]
    df = df.drop(df.index[10]) if bad == "gap" else pd.concat([df, df.iloc[[10]]])
    with pytest.raises(ValueError):
        make_features(df, is_training=False)


def test_feature_columns_present():
    """Verify all required features exist and are numeric."""
    df = pd.read_csv(DATA_PATH)
    clim_map = compute_climatology_map(df.iloc[:365])
    feats = make_features(df.iloc[:30], climatology_map=clim_map, is_training=False)
    for col in FEATURE_COLUMNS:
        assert col in feats.columns
        assert not feats[col].isnull().any()


def test_baselines_correctness():
    """Persistence equals temp on day D; climatology is the training-years day-of-year mean of the TARGET date."""
    df = load()
    train = df[df["date"].dt.year < 2023]
    clim_map = compute_climatology_map(train)
    feats = make_features(df, climatology_map=clim_map, is_training=True)
    assert np.allclose(feats["baseline_persistence"], feats["temp_lag_0"])
    expected = train.groupby(train["date"].dt.dayofyear)["temperature_2m_max"].mean()
    row = feats[feats["date"] == pd.Timestamp("2023-06-14")].iloc[0]
    assert np.isclose(row["baseline_climatology"], expected.loc[row["target_date"].dayofyear])


def test_climatology_ignores_test_year():
    """Changing 2023 values must not change a climatology built from training years."""
    df = load()
    train = df[df["date"].dt.year < 2023]
    altered = df.copy()
    altered.loc[altered["date"].dt.year == 2023, "temperature_2m_max"] += 10.0
    assert compute_climatology_map(train) == compute_climatology_map(altered[altered["date"].dt.year < 2023])
