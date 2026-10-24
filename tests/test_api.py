"""
tests/test_api.py
API contract tests (blueprint section 4): cities, city validation, health, metrics,
interval ordering, frontend-facing fields and the what-if endpoint.
The weather service is mocked, so these tests are deterministic and run offline.
An optional smoke test against the real Open-Meteo service runs only with RUN_LIVE_TESTS=1.
"""

import os

import pytest
from fastapi.testclient import TestClient

import backend.app as api
from src.features import FEATURE_COLUMNS
from tests.conftest import make_payload, utc

CITIES = ["Bengaluru", "Delhi", "Mumbai"]
NOW = utc(2023, 6, 15, 6, 30)  # 12:00 IST on 15 June 2023


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:  # runs the startup lifespan (load_artifacts)
        yield c


@pytest.fixture
def mocked_live(monkeypatch):
    monkeypatch.setattr(api, "ist_now", lambda: NOW)
    monkeypatch.setattr(api, "fetch_live_recent_weather",
                        lambda coords, *a, **k: make_payload(_city_for(coords), "2023-06-05", "2023-06-18"))


def _city_for(coords):
    return next(c.lower() for c, crd in api.CITIES.items() if crd == coords)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert sorted(data["loaded_cities"]) == ["bengaluru", "delhi", "mumbai"]
    assert data["artifact_errors"] == {}


def test_all_model_artifacts_load_with_expected_feature_count(client):
    for city in ["bengaluru", "delhi", "mumbai"]:
        models = api.MODELS[city]
        assert set(models) >= {"ensemble", "rf", "lgb", "lr"}
        for name in ("rf", "lgb", "lr"):
            assert models[name].n_features_in_ == len(FEATURE_COLUMNS)


def test_cities_endpoint(client):
    response = client.get("/cities")
    assert response.status_code == 200
    assert response.json() == CITIES


@pytest.mark.parametrize("endpoint", ["/predict", "/metrics", "/simulate"])
@pytest.mark.parametrize("bad_city", ["London", "", "delhi2"])
def test_wrong_city_returns_400(client, endpoint, bad_city):
    response = client.get(endpoint, params={"city": bad_city})
    assert response.status_code == 400
    assert "not supported" in response.json()["detail"]


@pytest.mark.parametrize("city", ["delhi", " DELHI ", "Delhi"])
def test_city_name_is_case_and_space_insensitive(client, mocked_live, city):
    assert client.get("/predict", params={"city": city}).json()["city"] == "Delhi"


@pytest.mark.parametrize("city", CITIES)
def test_predict_frontend_fields(client, mocked_live, city):
    """Every field the dashboard (frontend/app.js) reads must be present with the right type."""
    data = client.get("/predict", params={"city": city}).json()
    assert data["is_demo_fallback"] is False
    for key in ("city", "target_date", "last_completed_date", "data_source"):
        assert isinstance(data[key], str)
    for key in ("predicted_temp", "persistence_baseline", "climatology_baseline"):
        assert isinstance(data[key], float)
    assert set(data["model_breakdown"]) == {"ensemble", "lightgbm", "random_forest", "linear_regression"}
    interval = data["interval_80"]
    assert interval["lower"] < interval["upper"]
    assert interval["nominal_coverage_pct"] == 80.0
    assert isinstance(interval["test_coverage_pct"], float) and interval["test_days"] == 365
    assert set(data["current_conditions"]) >= {"temperature", "humidity", "wind_speed", "dew_point", "observed_at"}
    assert set(data["official_forecast"]) >= {"temperature_max", "date", "available"}
    shap = data["explainable_ai"]
    assert shap["model"] == "lightgbm" and len(shap["top_drivers"]) > 0
    assert {"feature", "label", "value", "contribution", "direction"} <= set(shap["top_drivers"][0])
    assert "multi_day_projection" not in data  # hand-written trajectory was removed
    assert [d["date"] for d in data["input_window"]] == [f"2023-06-{d:02d}" for d in range(8, 15)]


@pytest.mark.parametrize("city", CITIES)
def test_metrics_endpoint(client, city):
    response = client.get("/metrics", params={"city": city})
    assert response.status_code == 200
    metrics = response.json()
    for model in ("ensemble", "random_forest", "lightgbm", "linear_regression", "persistence", "climatology"):
        assert {"mae", "rmse"} <= set(metrics["overall"][model])
    assert set(metrics["by_season"]) == {"Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"}
    assert metrics["n_test"] == 365
    served = metrics["served_interval"]
    assert served["p10"] == metrics["interval_80"]["p10"] and served["p90"] == metrics["interval_80"]["p90"]
    lo, hi = served["test_coverage_approx_95ci"]
    assert lo < served["test_coverage_pct"] < hi
    assert set(metrics["main_model_beats"]) == {"persistence", "climatology"}
    assert set(metrics["validation"]["mae"]) >= {"ensemble", "persistence"}


def test_metrics_report_baseline_comparison_honestly(client):
    """The comparison flags must agree with the MAE/RMSE numbers they summarise."""
    for city in CITIES:
        m = client.get("/metrics", params={"city": city}).json()
        for base, flags in m["main_model_beats"].items():
            for k in ("mae", "rmse"):
                assert flags[k] == (m["overall"]["ensemble"][k] < m["overall"][base][k])


def test_simulate_endpoint(client):
    response = client.get("/simulate", params={"city": "Bengaluru", "temp_yesterday": 30.0, "humidity": 50.0,
                                               "wind_speed": 15.0, "temp_momentum": 1.0})
    assert response.status_code == 200
    data = response.json()
    assert data["is_illustrative"] is True
    assert set(data["predictions"]) == {"ensemble", "lightgbm", "random_forest"}
    assert data["explainable_ai"]["top_drivers"]


@pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to call the real Open-Meteo API")
@pytest.mark.parametrize("city", CITIES)
def test_live_open_meteo_smoke(client, city):
    data = client.get("/predict", params={"city": city}).json()
    assert data["is_demo_fallback"] is False, data.get("fallback_message")
    assert data["interval_80"]["lower"] < data["interval_80"]["upper"]
