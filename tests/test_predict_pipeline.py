"""
tests/test_predict_pipeline.py
/predict with the network mocked: genuine model execution vs. labelled fallback,
no invented values, and training-serving consistency.
"""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend.app as api
from backend.live_data import WeatherDataError
from src.features import FEATURE_COLUMNS, make_features
from src.intervals import interval_bounds
from tests.conftest import load_city_csv, make_payload, utc

# 12:00 IST on 15 June 2023 -> D = 14 June, target = 16 June
NOW = utc(2023, 6, 15, 6, 30)
D, TARGET = "2023-06-14", "2023-06-16"


@pytest.fixture(scope="module")
def client():
    api.load_artifacts()
    with TestClient(api.app) as c:
        yield c


@pytest.fixture
def live(monkeypatch):
    """Patch the clock and the weather service; returns a setter for the payload."""
    monkeypatch.setattr(api, "ist_now", lambda: NOW)
    state = {}

    def fake_fetch(coords, *a, **k):
        if isinstance(state["payload"], Exception):
            raise state["payload"]
        return state["payload"]

    monkeypatch.setattr(api, "fetch_live_recent_weather", fake_fetch)

    def set_payload(p):
        state["payload"] = p
    return set_payload


@pytest.mark.parametrize("city", ["bengaluru", "delhi", "mumbai"])
def test_live_prediction_runs_real_models(client, live, city):
    live(make_payload(city, "2023-06-05", "2023-06-18"))
    r = client.get("/predict", params={"city": city})
    assert r.status_code == 200
    data = r.json()
    assert data["is_demo_fallback"] is False and data["data_source"] == "live"
    assert (data["last_completed_date"], data["target_date"]) == (D, TARGET)
    assert data["interval_80"]["lower"] < data["predicted_temp"] < data["interval_80"]["upper"]


@pytest.mark.parametrize("city", ["bengaluru", "delhi", "mumbai"])
def test_serving_matches_training_features_and_models(client, live, city):
    """The API's prediction must equal the saved model applied to the training-time feature row."""
    live(make_payload(city, "2023-06-05", "2023-06-18"))
    data = client.get("/predict", params={"city": city}).json()

    hist = load_city_csv(city)
    hist = hist[hist["date"] <= TARGET]  # full history up to the target, as in training
    feats = make_features(hist, climatology_map=api.CLIMATOLOGY[city], is_training=True)
    row = feats[feats["date"] == pd.Timestamp(D)]
    assert len(row) == 1
    expected = float(api.MODELS[city]["ensemble"].predict(row[FEATURE_COLUMNS].values)[0])
    assert data["predicted_temp"] == round(expected, 1)
    # The served interval is exactly the one whose coverage src/evaluate.py measures
    interval = api.METRICS[city]["interval_80"]
    assert (data["interval_80"]["lower"], data["interval_80"]["upper"]) == \
        interval_bounds(expected, interval["p10"], interval["p90"])
    assert data["interval_80"]["test_coverage_pct"] == api.EVALUATION[city]["served_interval"]["test_coverage_pct"]
    assert data["persistence_baseline"] == round(float(row["baseline_persistence"].iloc[0]), 1)
    assert data["climatology_baseline"] == round(float(row["baseline_climatology"].iloc[0]), 1)


def test_official_forecast_is_exact_target_date_value(client, live):
    live(make_payload("delhi", "2023-06-05", "2023-06-18"))
    data = client.get("/predict", params={"city": "Delhi"}).json()
    actual = load_city_csv("delhi").set_index("date").loc[TARGET, "temperature_2m_max"]
    assert data["official_forecast"]["temperature_max"] == actual
    assert data["official_forecast"]["date"] == TARGET


def test_missing_official_forecast_is_null_not_invented(client, live):
    # Payload ends on today (15 June): the target date is absent
    live(make_payload("delhi", "2023-06-05", "2023-06-15"))
    data = client.get("/predict", params={"city": "Delhi"}).json()
    assert data["is_demo_fallback"] is False
    assert data["official_forecast"]["temperature_max"] is None
    assert data["official_forecast"]["available"] is False


def test_null_official_forecast_value_stays_null(client, live):
    live(make_payload("delhi", "2023-06-05", "2023-06-18", null_values=[(TARGET, "temperature_2m_max")]))
    data = client.get("/predict", params={"city": "Delhi"}).json()
    assert data["is_demo_fallback"] is False
    assert data["official_forecast"]["temperature_max"] is None


def test_missing_current_conditions_are_null_not_defaults(client, live):
    live(make_payload("delhi", "2023-06-05", "2023-06-18", current={"time": "2023-06-15T12:00", "temperature_2m": 31.0}))
    cc = client.get("/predict", params={"city": "Delhi"}).json()["current_conditions"]
    assert cc["temperature"] == 31.0
    assert cc["humidity"] is None and cc["wind_speed"] is None and cc["dew_point"] is None


@pytest.mark.parametrize("error", [
    WeatherDataError("timeout", "Open-Meteo did not respond in time"),
    WeatherDataError("upstream_error", "Open-Meteo request failed (HTTPError)"),
    WeatherDataError("malformed_response", "Open-Meteo response was not valid JSON"),
])
def test_weather_service_failure_returns_labelled_snapshot(client, live, error):
    live(error)
    r = client.get("/predict", params={"city": "Mumbai"})
    assert r.status_code == 200
    data = r.json()
    assert data["is_demo_fallback"] is True and data["data_source"] == "demo_snapshot"
    assert data["fallback_reason"] == error.reason
    snapshot = api.FALLBACK_SAMPLES["mumbai"]
    # Snapshot dates are reported as captured, never re-dated to look current
    assert data["target_date"] == snapshot["target_date"]


def test_invalid_payload_falls_back_with_reason(client, live):
    live(make_payload("delhi", "2023-06-05", "2023-06-18", null_values=[("2023-06-12", "temperature_2m_max")]))
    data = client.get("/predict", params={"city": "Delhi"}).json()
    assert data["is_demo_fallback"] is True and data["fallback_reason"] == "missing_values"


def test_fallback_does_not_mutate_saved_snapshot(client, live):
    before = json.dumps(api.FALLBACK_SAMPLES, sort_keys=True)
    live(WeatherDataError("timeout", "x"))
    client.get("/predict", params={"city": "Delhi"})
    live(WeatherDataError("upstream_error", "y"))
    client.get("/predict", params={"city": "Delhi"})
    assert json.dumps(api.FALLBACK_SAMPLES, sort_keys=True) == before


def test_no_snapshot_means_explicit_503(client, live, monkeypatch):
    monkeypatch.setattr(api, "FALLBACK_SAMPLES", {})
    live(WeatherDataError("timeout", "x"))
    r = client.get("/predict", params={"city": "Delhi"})
    assert r.status_code == 503


def test_missing_models_are_an_error_not_demo_mode(client, live, monkeypatch):
    live(make_payload("delhi", "2023-06-05", "2023-06-18"))
    monkeypatch.setitem(api.ARTIFACT_ERRORS, "delhi", ["missing delhi_rf.joblib"])
    r = client.get("/predict", params={"city": "Delhi"})
    assert r.status_code == 503
    assert client.get("/health").status_code == 503


def test_internal_bug_is_500_not_demo_mode(client, live, monkeypatch):
    live(make_payload("delhi", "2023-06-05", "2023-06-18"))

    class Broken:
        def predict(self, X):
            raise ValueError("boom")
    monkeypatch.setitem(api.MODELS["delhi"], "ensemble", Broken())
    r = client.get("/predict", params={"city": "Delhi"})
    assert r.status_code == 500
    assert "boom" not in r.text  # internal details stay in the logs


def test_simulate_rejects_unknown_city(client):
    assert client.get("/simulate", params={"city": "London"}).status_code == 400
