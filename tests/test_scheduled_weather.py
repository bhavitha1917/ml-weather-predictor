"""
tests/test_scheduled_weather.py
The scheduled weather file written by GitHub Actions (scripts/fetch_scheduled_weather.py) and
read by the API (backend.live_data.fetch_scheduled_weather): the file is validated as untrusted
input, outdated or other-day entries are rejected, failures fall back to Open-Meteo without a
cooldown, and /predict labels the data with its real fetch time. All HTTP is mocked.
"""

import importlib.util
import json
import os
import time
from datetime import date, datetime, timedelta, timezone

import pytest
import requests

import backend.app as api
from backend.live_data import (
    IST,
    MAX_STALE_AGE_SECONDS,
    SCHEDULED_CLOCK_SKEW_SECONDS,
    SCHEDULED_MAX_BYTES,
    SCHEDULED_RETRY_SECONDS,
    LiveWeatherCache,
    WeatherDataError,
    build_observation_window,
    fetch_scheduled_weather,
)
from tests.conftest import BASE_DIR, make_payload, utc

NOW = datetime(2023, 6, 15, 12, 0, tzinfo=IST)
D = date(2023, 6, 14)
FETCHED_UTC = "2023-06-15T06:00:00+00:00"  # 11:30 IST, 30 minutes before NOW
FETCHED_IST = datetime(2023, 6, 15, 11, 30, tzinfo=IST)
URL = "https://example.test/weather.json"
COORDS = api.CITIES["Delhi"]


def load_script():
    spec = importlib.util.spec_from_file_location(
        "fetch_scheduled_weather", os.path.join(BASE_DIR, "scripts", "fetch_scheduled_weather.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sfw = load_script()


def city_entry(city="Delhi", **overrides):
    coords = api.CITIES[city]
    entry = {"latitude": coords["latitude"], "longitude": coords["longitude"],
             "last_completed_date": D.isoformat(), "fetched_at_utc": FETCHED_UTC,
             "raw": make_payload(city.lower(), "2023-06-05", "2023-06-18")}
    entry.update(overrides)
    return entry


def document(**overrides):
    return {"schema_version": 1, "source": sfw.SOURCE_NOTE, "generated_at_utc": FETCHED_UTC,
            "cities": {"delhi": city_entry(**overrides)}}


class FakeResponse:
    def __init__(self, status_code=200, body=b""):
        self.status_code = status_code
        self.body = body

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def served(monkeypatch):
    """Serve `state['outcome']` (a dict as JSON, bytes, a FakeResponse or an exception) for any URL."""
    state = {"outcome": document(), "calls": []}

    def fake_get(url, **kwargs):
        state["calls"].append((url, kwargs))
        outcome = state["outcome"]
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, FakeResponse):
            return outcome
        body = outcome if isinstance(outcome, bytes) else json.dumps(outcome).encode()
        return FakeResponse(200, body)
    monkeypatch.setattr("requests.get", fake_get)
    return state


def read(**kwargs):
    return fetch_scheduled_weather("delhi", COORDS, D, kwargs.pop("now", NOW), url=kwargs.pop("url", URL))


# --- Reading and validating the file ---------------------------------------------------

def test_valid_entry_returns_raw_data_and_real_fetch_time(served):
    result = read()
    assert result["raw"] == city_entry()["raw"]
    assert result["fetched_at"] == FETCHED_IST and result["fetched_at"].utcoffset() == timedelta(hours=5, minutes=30)
    assert build_observation_window(result["raw"], D)["date"].iloc[-1].date() == D
    (url, kwargs), = served["calls"]
    assert url == URL and kwargs["timeout"] > 0 and kwargs["stream"] is True


def test_entry_is_accepted_up_to_but_not_at_max_age(served):
    fetched = datetime(2023, 6, 15, 6, 0, tzinfo=timezone.utc)
    read(now=fetched + timedelta(seconds=MAX_STALE_AGE_SECONDS - 1))
    with pytest.raises(WeatherDataError) as exc:
        read(now=fetched + timedelta(seconds=MAX_STALE_AGE_SECONDS))
    assert exc.value.reason == "scheduled_outdated"


def test_entry_for_another_day_is_rejected(served):
    served["outcome"] = document(last_completed_date="2023-06-13")
    with pytest.raises(WeatherDataError) as exc:
        read()
    assert exc.value.reason == "scheduled_outdated"


def test_fetch_time_in_the_future_is_rejected_beyond_clock_skew(served):
    fetched = datetime(2023, 6, 15, 6, 0, tzinfo=timezone.utc)
    read(now=fetched - timedelta(seconds=SCHEDULED_CLOCK_SKEW_SECONDS - 1))
    with pytest.raises(WeatherDataError) as exc:
        read(now=fetched - timedelta(seconds=SCHEDULED_CLOCK_SKEW_SECONDS + 1))
    assert exc.value.reason == "scheduled_invalid"


@pytest.mark.parametrize("outcome, reason", [
    ({**document(), "schema_version": 2}, "scheduled_invalid"),
    ([document()], "scheduled_invalid"),
    ({"schema_version": 1}, "scheduled_missing_city"),
    ({"schema_version": 1, "cities": {"mumbai": city_entry("Mumbai")}}, "scheduled_missing_city"),
    (document(latitude=0.0), "scheduled_invalid"),
    (document(fetched_at_utc="2023-06-15T06:00:00"), "scheduled_invalid"),  # no timezone
    (document(fetched_at_utc="yesterday"), "scheduled_invalid"),
    (document(fetched_at_utc=None), "scheduled_invalid"),
    (document(raw="not an object"), "scheduled_invalid"),
    (b"<html>not json</html>", "scheduled_invalid"),
])
def test_malformed_files_are_rejected(served, outcome, reason):
    served["outcome"] = outcome
    with pytest.raises(WeatherDataError) as exc:
        read()
    assert exc.value.reason == reason


def test_oversized_file_is_rejected(served):
    served["outcome"] = b" " * (SCHEDULED_MAX_BYTES + 1)
    with pytest.raises(WeatherDataError) as exc:
        read()
    assert exc.value.reason == "scheduled_invalid" and "larger than" in str(exc.value)


@pytest.mark.parametrize("status", [404, 429, 500, 503])
def test_http_errors_are_unavailable_without_an_http_status(served, status):
    served["outcome"] = FakeResponse(status)
    with pytest.raises(WeatherDataError) as exc:
        read()
    # No http_status, so a GitHub failure can never start the Open-Meteo rate-limit cooldown
    assert exc.value.reason == "scheduled_unavailable" and exc.value.http_status is None


@pytest.mark.parametrize("error", [requests.Timeout("slow"), requests.ConnectionError("down")])
def test_network_errors_are_unavailable(served, error):
    served["outcome"] = error
    with pytest.raises(WeatherDataError) as exc:
        read()
    assert exc.value.reason == "scheduled_unavailable" and URL not in str(exc.value)


def test_plain_http_is_refused_except_for_a_local_test_server(served):
    with pytest.raises(WeatherDataError) as exc:
        read(url="http://example.test/weather.json")
    assert exc.value.reason == "scheduled_invalid" and served["calls"] == []
    read(url="http://127.0.0.1:8765/weather.json")
    assert len(served["calls"]) == 1


def test_empty_setting_disables_the_source_without_a_request(served, monkeypatch):
    monkeypatch.setenv("SCHEDULED_WEATHER_URL", "")
    with pytest.raises(WeatherDataError) as exc:
        fetch_scheduled_weather("delhi", COORDS, D, NOW)
    assert exc.value.reason == "scheduled_disabled" and served["calls"] == []


# --- Cache: real fetch time ------------------------------------------------------------

def test_cache_keeps_real_fetch_time_and_never_serves_data_past_max_age():
    cache = LiveWeatherCache(ttl_seconds=600)
    old = NOW - timedelta(seconds=MAX_STALE_AGE_SECONDS - 60)  # fetched 2 h 59 min before NOW
    payload = {"raw": make_payload("delhi", "2023-06-05", "2023-06-18"), "fetched_at": old}
    calls = []

    def fetch():
        calls.append(1)
        return payload
    validate = lambda p: build_observation_window(p["raw"], D)  # noqa: E731
    _, fetched_at, _, _ = cache.get("k", fetch, validate, NOW, fetched_at_of=lambda p: p["fetched_at"])
    assert fetched_at == old
    _, _, from_cache, _ = cache.get("k", fetch, validate, NOW + timedelta(seconds=59), fetched_at_of=lambda p: p["fetched_at"])
    assert from_cache is True and len(calls) == 1
    # One minute later the data is 3 h old: it is not served from the cache, even within the TTL
    _, _, from_cache, _ = cache.get("k", fetch, validate, NOW + timedelta(seconds=60), fetched_at_of=lambda p: p["fetched_at"])
    assert from_cache is False and len(calls) == 2


# --- The scheduled-fetch script ---------------------------------------------------------

def fake_open_meteo(failing=()):
    def fetch(coords, past_days, forecast_days):
        city = next(c for c, crd in api.CITIES.items() if crd == coords)
        if city in failing:
            raise WeatherDataError("rate_limited", "Open-Meteo returned HTTP 429", http_status=429)
        return make_payload(city.lower(), "2023-06-05", "2023-06-18")
    return fetch


def fixed_clock():
    return datetime(2023, 6, 15, 6, 0, tzinfo=timezone.utc)


def test_script_writes_only_validated_data_with_fetch_time_and_d():
    doc, updated, failures = sfw.build_snapshot({}, now=NOW, fetch=fake_open_meteo(), clock=fixed_clock)
    assert sorted(updated) == ["bengaluru", "delhi", "mumbai"] and failures == {}
    assert set(doc) == {"schema_version", "source", "generated_at_utc", "cities"}
    for key, entry in doc["cities"].items():
        assert set(entry) == {"latitude", "longitude", "last_completed_date", "fetched_at_utc", "raw"}
        assert entry["last_completed_date"] == "2023-06-14" and entry["fetched_at_utc"] == FETCHED_UTC
        assert entry["latitude"] == api.CITIES[key.capitalize()]["latitude"]


def test_script_keeps_failed_city_previous_entry_unchanged_and_omits_cities_without_one():
    previous = {"delhi": city_entry(fetched_at_utc="2023-06-15T02:00:00+00:00")}
    doc, updated, failures = sfw.build_snapshot(previous, now=NOW, fetch=fake_open_meteo(failing=("Delhi", "Mumbai")),
                                                clock=fixed_clock)
    assert updated == ["bengaluru"] and failures == {"delhi": "rate_limited", "mumbai": "rate_limited"}
    assert doc["cities"]["delhi"] == previous["delhi"]  # its own old fetch time; never re-dated
    assert doc["cities"]["delhi"]["fetched_at_utc"] == "2023-06-15T02:00:00+00:00"
    assert "mumbai" not in doc["cities"]


def test_script_never_writes_data_that_fails_validation():
    def fetch(coords, past_days, forecast_days):
        payload = make_payload("delhi", "2023-06-05", "2023-06-18")
        payload["daily"]["temperature_2m_max"][9] = None  # 2023-06-14 = D
        return payload
    doc, updated, failures = sfw.build_snapshot({}, now=NOW, fetch=fetch, clock=fixed_clock)
    assert updated == [] and doc["cities"] == {} and set(failures.values()) == {"missing_values"}


def test_script_output_is_accepted_by_the_api_reader(served, tmp_path, monkeypatch):
    real_build = sfw.build_snapshot
    monkeypatch.setattr(sfw, "build_snapshot",
                        lambda previous: real_build(previous, now=NOW, fetch=fake_open_meteo(), clock=fixed_clock))
    out = tmp_path / "out" / "weather.json"
    assert sfw.main(["--output", str(out)]) == 0
    served["outcome"] = out.read_bytes()
    for city in ("Bengaluru", "Delhi", "Mumbai"):
        result = fetch_scheduled_weather(city.lower(), api.CITIES[city], D, NOW, url=URL)
        assert result["fetched_at"] == FETCHED_IST
        build_observation_window(result["raw"], D)


def test_script_fails_and_writes_nothing_when_no_city_updates(tmp_path, monkeypatch):
    real_build = sfw.build_snapshot
    monkeypatch.setattr(sfw, "build_snapshot", lambda previous: real_build(
        previous, now=NOW, fetch=fake_open_meteo(failing=("Bengaluru", "Delhi", "Mumbai")), clock=fixed_clock))
    out = tmp_path / "weather.json"
    assert sfw.main(["--output", str(out)]) == 1
    assert not out.exists()


@pytest.mark.parametrize("content", [None, "not json", json.dumps({"schema_version": 99, "cities": {}})])
def test_unusable_previous_file_is_ignored(tmp_path, content):
    path = tmp_path / "previous.json"
    if content is not None:
        path.write_text(content)
    assert sfw.load_previous(str(path)) == {}


# --- /predict -------------------------------------------------------------------------

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    with TestClient(api.app) as c:
        yield c


@pytest.fixture
def sources(monkeypatch):
    """Scheduled file enabled; both sources return `state[...]` (a payload or an exception)."""
    monkeypatch.setenv("SCHEDULED_WEATHER_URL", URL)
    monkeypatch.setattr(api, "ist_now", lambda: utc(2023, 6, 15, 6, 30))  # 12:00 IST, D = 14 June
    state = {"scheduled": None, "open_meteo": None, "scheduled_calls": 0, "open_meteo_calls": 0}

    def fake_scheduled(city_key, coords, last_completed, now):
        state["scheduled_calls"] += 1
        if isinstance(state["scheduled"], Exception):
            raise state["scheduled"]
        return {"raw": make_payload(city_key, "2023-06-05", "2023-06-18"), "fetched_at": FETCHED_IST}

    def fake_open_meteo_fetch(coords, *a, **k):
        state["open_meteo_calls"] += 1
        if isinstance(state["open_meteo"], Exception):
            raise state["open_meteo"]
        city = next(c for c, crd in api.CITIES.items() if crd == coords)
        return make_payload(city.lower(), "2023-06-05", "2023-06-18")
    monkeypatch.setattr(api, "fetch_scheduled_weather", fake_scheduled)
    monkeypatch.setattr(api, "fetch_live_recent_weather", fake_open_meteo_fetch)
    return state


def unavailable():
    return WeatherDataError("scheduled_unavailable", "Scheduled weather file returned HTTP 404")


def test_predict_uses_scheduled_data_labelled_with_its_real_fetch_time(client, sources):
    data = client.get("/predict?city=Delhi").json()
    assert data["data_source"] == "scheduled" and data["weather_data_origin"] == "github_actions"
    assert data["weather_data_fetched_at"] == "2023-06-15T11:30:00+05:30"
    generated = datetime.fromisoformat(data["generated_at"])
    assert generated - datetime.fromisoformat(data["weather_data_fetched_at"]) == timedelta(minutes=30)
    assert data["is_demo_fallback"] is False and data["stale_reason"] is None
    assert sources["open_meteo_calls"] == 0


def test_scheduled_and_direct_data_give_the_same_prediction(client, sources, monkeypatch):
    scheduled = client.get("/predict?city=Mumbai").json()
    monkeypatch.setenv("SCHEDULED_WEATHER_URL", "")
    direct = client.get("/predict?city=Mumbai").json()
    assert direct["data_source"] == "live" and direct["weather_data_origin"] == "open_meteo_direct"
    for field in ("predicted_temp", "model_breakdown", "interval_80", "input_window", "persistence_baseline",
                  "climatology_baseline", "last_completed_date", "target_date", "official_forecast"):
        assert scheduled[field] == direct[field]


def test_scheduled_data_is_used_while_open_meteo_is_cooling_down(client, sources):
    api.LIVE_WEATHER_CACHE._start_cooldown(600)
    data = client.get("/predict?city=Bengaluru").json()
    assert data["data_source"] == "scheduled" and sources["open_meteo_calls"] == 0


def test_scheduled_entry_is_reused_from_cache(client, sources):
    client.get("/predict?city=Delhi")
    client.get("/predict?city=Delhi")
    assert sources["scheduled_calls"] == 1


def test_unusable_scheduled_file_falls_back_to_open_meteo_and_is_not_reread_for_a_while(client, sources):
    sources["scheduled"] = unavailable()
    first = client.get("/predict?city=Delhi").json()
    second = client.get("/predict?city=Delhi").json()
    assert first["data_source"] == "live" and first["weather_data_origin"] == "open_meteo_direct"
    assert second["data_source"] == "live_cached"
    assert sources["scheduled_calls"] == 1 and sources["open_meteo_calls"] == 1
    assert api.LIVE_WEATHER_CACHE.cooldown_remaining() == 0
    # After the retry interval the file is read again
    api.SCHEDULED_FAILURES["delhi"] = time.monotonic() - SCHEDULED_RETRY_SECONDS
    sources["scheduled"] = None
    assert client.get("/predict?city=Delhi").json()["data_source"] == "scheduled"
    assert sources["scheduled_calls"] == 2


def test_no_scheduled_data_and_open_meteo_rate_limited_shows_demo(client, sources):
    sources["scheduled"] = unavailable()
    sources["open_meteo"] = WeatherDataError("rate_limited", "Open-Meteo returned HTTP 429", http_status=429)
    data = client.get("/predict?city=Delhi").json()
    assert data["is_demo_fallback"] is True and data["data_source"] == "demo_snapshot"
    assert data["fallback_reason"] == "rate_limited"


def test_scheduled_data_that_fails_validation_is_not_used(client, sources, monkeypatch):
    def broken(city_key, coords, last_completed, now):
        sources["scheduled_calls"] += 1
        payload = make_payload(city_key, "2023-06-05", "2023-06-18")
        payload["daily"]["temperature_2m_max"][9] = None  # 2023-06-14 = D
        return {"raw": payload, "fetched_at": FETCHED_IST}
    monkeypatch.setattr(api, "fetch_scheduled_weather", broken)
    data = client.get("/predict?city=Delhi").json()
    assert data["data_source"] == "live" and sources["open_meteo_calls"] == 1
