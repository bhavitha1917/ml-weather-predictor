"""
tests/conftest.py
Shared fixtures. Live Open-Meteo calls are replaced by payloads built from the project's
own historical CSVs, so the ordinary test suite is deterministic and works offline.
"""

import os
import sys
from datetime import datetime, timezone

import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.data_fetch import DAILY_VARIABLES  # noqa: E402

DATA_DIR = os.path.join(BASE_DIR, "data")


def load_city_csv(city_key: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, f"{city_key}.csv"))


def make_payload(city_key, start, end, drop_dates=(), null_values=(), current=None):
    """
    Build an Open-Meteo-shaped response from historical CSV rows between start and end.
    drop_dates: dates removed from the time axis. null_values: (date, variable) pairs set to None.
    """
    df = load_city_csv(city_key)
    df = df[(df["date"] >= start) & (df["date"] <= end) & (~df["date"].isin(drop_dates))]
    nulls = set(null_values)
    daily = {"time": df["date"].tolist()}
    for var in DAILY_VARIABLES:
        daily[var] = [None if (t, var) in nulls else float(v) for t, v in zip(df["date"], df[var])]
    if current is None:
        current = {"time": f"{end}T12:00", "temperature_2m": 25.0, "relative_humidity_2m": 55,
                   "wind_speed_10m": 9.0, "dew_point_2m": 15.2}
    return {"daily": daily, "current": current}


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


@pytest.fixture
def payload_builder():
    return make_payload


@pytest.fixture(autouse=True)
def block_real_http(monkeypatch):
    """Ordinary tests must never reach the network; only RUN_LIVE_TESTS=1 allows it."""
    if os.environ.get("RUN_LIVE_TESTS") == "1":
        return

    def refuse(*args, **kwargs):
        raise AssertionError("Unexpected real HTTP request in an offline test")
    monkeypatch.setattr("requests.get", refuse)


@pytest.fixture(autouse=True)
def fresh_live_weather_cache():
    """Each test starts with an empty live-weather cache and no rate-limit cooldown."""
    import backend.app as api
    api.LIVE_WEATHER_CACHE.clear()
    yield
    api.LIVE_WEATHER_CACHE.clear()
