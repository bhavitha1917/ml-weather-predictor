"""
tests/test_live_cache.py
The in-memory cache of validated Open-Meteo responses and the HTTP 429 cooldown:
cache hits and expiry, failures never cached, Retry-After handling, stale same-day
reuse while rate-limited, concurrent requests, and how /predict labels cached,
stale and rate-limited responses.
"""

import threading
import time
from datetime import date, datetime, timezone

import pytest

import backend.app as api
from backend.live_data import (
    DEFAULT_RATE_LIMIT_BACKOFF_SECONDS,
    LIVE_CACHE_TTL_SECONDS,
    MAX_RATE_LIMIT_BACKOFF_SECONDS,
    MAX_STALE_AGE_SECONDS,
    IST,
    LiveWeatherCache,
    WeatherDataError,
    _parse_retry_after,
    build_observation_window,
)
from tests.conftest import make_payload, utc

D = date(2023, 6, 14)
FETCHED_AT = datetime(2023, 6, 15, 12, 0, tzinfo=IST)
KEY = ("delhi", D.isoformat(), 10, 4)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class CountingFetch:
    """Stands in for fetch_live_recent_weather: returns the payload or raises the error, counting calls."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def validate(raw):
    build_observation_window(raw, D)


def rate_limit_error(retry_after=None):
    return WeatherDataError("rate_limited", "Open-Meteo returned HTTP 429 Too Many Requests",
                            http_status=429, retry_after_seconds=retry_after)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def cache(clock):
    return LiveWeatherCache(clock=clock)


@pytest.fixture
def payload():
    return make_payload("delhi", "2023-06-05", "2023-06-18")


# --- Cache hits and expiry ------------------------------------------------------------

def test_second_request_within_ttl_is_served_from_cache(cache, payload):
    fetch = CountingFetch(payload)
    first, first_at, first_cached, _ = cache.get(KEY, fetch, validate, FETCHED_AT)
    later = datetime(2023, 6, 15, 12, 20, tzinfo=IST)
    second, second_at, second_cached, _ = cache.get(KEY, fetch, validate, later)
    assert fetch.calls == 1
    assert (first_cached, second_cached) == (False, True)
    assert second == first == payload
    assert second_at == first_at == FETCHED_AT  # the time the data was really fetched


def test_cached_payload_cannot_be_modified_by_callers(cache, payload):
    fetch = CountingFetch(payload)
    first, _, _, _ = cache.get(KEY, fetch, validate, FETCHED_AT)
    first["daily"]["temperature_2m_max"][-1] = -99.0
    second, _, _, _ = cache.get(KEY, fetch, validate, FETCHED_AT)
    assert second["daily"]["temperature_2m_max"][-1] != -99.0


def test_entry_expires_after_ttl(cache, clock, payload):
    fetch = CountingFetch(payload)
    cache.get(KEY, fetch, validate, FETCHED_AT)
    clock.advance(LIVE_CACHE_TTL_SECONDS - 1)
    cache.get(KEY, fetch, validate, FETCHED_AT)
    assert fetch.calls == 1
    clock.advance(1)
    _, _, from_cache, _ = cache.get(KEY, fetch, validate, FETCHED_AT)
    assert fetch.calls == 2 and from_cache is False


def test_keys_are_separate_per_city_and_day(cache, payload):
    fetch = CountingFetch(payload)
    cache.get(KEY, fetch, validate, FETCHED_AT)
    cache.get(("mumbai", D.isoformat(), 10, 4), fetch, validate, FETCHED_AT)
    cache.get(("delhi", "2023-06-15", 10, 4), fetch, lambda raw: None, FETCHED_AT)
    assert fetch.calls == 3


def test_ttl_is_thirty_minutes():
    assert LIVE_CACHE_TTL_SECONDS == 30 * 60


# --- Failures are never cached --------------------------------------------------------

def test_invalid_data_is_not_cached(cache, payload):
    payload["daily"]["temperature_2m_max"][9] = None  # 2023-06-14 = D
    fetch = CountingFetch(payload)
    for _ in range(2):
        with pytest.raises(WeatherDataError) as exc:
            cache.get(KEY, fetch, validate, FETCHED_AT)
        assert exc.value.reason == "missing_values"
    assert fetch.calls == 2


def test_upstream_error_is_not_cached_and_starts_no_cooldown(cache, payload):
    fetch = CountingFetch(WeatherDataError("upstream_error", "HTTP 503", http_status=503))
    with pytest.raises(WeatherDataError):
        cache.get(KEY, fetch, validate, FETCHED_AT)
    assert cache.cooldown_remaining() == 0
    fetch.outcome = payload
    _, _, from_cache, _ = cache.get(KEY, fetch, validate, FETCHED_AT)
    assert fetch.calls == 2 and from_cache is False


# --- HTTP 429 cooldown ----------------------------------------------------------------

def test_429_respects_retry_after_and_sends_no_requests_meanwhile(cache, clock, payload):
    fetch = CountingFetch(rate_limit_error(retry_after=120))
    with pytest.raises(WeatherDataError) as exc:
        cache.get(KEY, fetch, validate, FETCHED_AT)
    assert exc.value.http_status == 429
    fetch.outcome = payload
    clock.advance(119)
    with pytest.raises(WeatherDataError) as exc:
        cache.get(KEY, fetch, validate, FETCHED_AT)
    assert exc.value.reason == "rate_limited" and "no request sent" in str(exc.value)
    assert fetch.calls == 1
    clock.advance(1)
    _, _, from_cache, _ = cache.get(KEY, fetch, validate, FETCHED_AT)
    assert fetch.calls == 2 and from_cache is False


def test_429_without_retry_after_uses_default_backoff(cache, clock):
    fetch = CountingFetch(rate_limit_error())
    with pytest.raises(WeatherDataError):
        cache.get(KEY, fetch, validate, FETCHED_AT)
    assert cache.cooldown_remaining() == DEFAULT_RATE_LIMIT_BACKOFF_SECONDS
    clock.advance(DEFAULT_RATE_LIMIT_BACKOFF_SECONDS - 1)
    with pytest.raises(WeatherDataError):
        cache.get(KEY, fetch, validate, FETCHED_AT)
    assert fetch.calls == 1


def test_retry_after_is_capped(cache):
    with pytest.raises(WeatherDataError):
        cache.get(KEY, CountingFetch(rate_limit_error(retry_after=10 ** 9)), validate, FETCHED_AT)
    assert cache.cooldown_remaining() == MAX_RATE_LIMIT_BACKOFF_SECONDS


def test_cooldown_applies_to_all_cities_but_not_to_cached_data(cache, payload):
    cache.get(KEY, CountingFetch(payload), validate, FETCHED_AT)
    with pytest.raises(WeatherDataError):
        cache.get(("mumbai", D.isoformat(), 10, 4), CountingFetch(rate_limit_error(60)), validate, FETCHED_AT)
    # Another city is not requested during the cooldown...
    bengaluru = CountingFetch(payload)
    with pytest.raises(WeatherDataError) as exc:
        cache.get(("bengaluru", D.isoformat(), 10, 4), bengaluru, validate, FETCHED_AT)
    assert exc.value.reason == "rate_limited" and bengaluru.calls == 0
    # ...but data already fetched and validated is still served while it is fresh
    _, _, from_cache, stale_error = cache.get(KEY, CountingFetch(payload), validate, FETCHED_AT)
    assert from_cache is True and stale_error is None


# --- Stale same-day reuse while rate-limited ------------------------------------------

LATER = datetime(2023, 6, 15, 12, 40, tzinfo=IST)


def expire_entry(cache, clock, payload, extra_seconds=0):
    """Store a validated entry fetched at FETCHED_AT, then move the clock past its TTL."""
    cache.get(KEY, CountingFetch(payload), validate, FETCHED_AT)
    clock.advance(LIVE_CACHE_TTL_SECONDS + extra_seconds)


def test_max_stale_age_is_three_hours():
    assert MAX_STALE_AGE_SECONDS == 3 * 60 * 60 and MAX_STALE_AGE_SECONDS > LIVE_CACHE_TTL_SECONDS


def test_active_cooldown_reuses_expired_same_day_entry_without_calling_api(cache, clock, payload):
    expire_entry(cache, clock, payload)
    with pytest.raises(WeatherDataError):
        cache.get(("mumbai", D.isoformat(), 10, 4), CountingFetch(rate_limit_error(600)), validate, LATER)
    fetch = CountingFetch(payload)
    data, fetched_at, from_cache, stale_error = cache.get(KEY, fetch, validate, LATER)
    assert fetch.calls == 0
    assert data == payload and fetched_at == FETCHED_AT and from_cache is True
    assert stale_error.reason == "rate_limited" and "no request sent" in str(stale_error)


def test_new_429_reuses_expired_entry_and_starts_cooldown(cache, clock, payload):
    expire_entry(cache, clock, payload)
    fetch = CountingFetch(rate_limit_error(retry_after=300))
    data, fetched_at, from_cache, stale_error = cache.get(KEY, fetch, validate, LATER)
    assert fetch.calls == 1
    assert data == payload and fetched_at == FETCHED_AT and from_cache is True
    assert stale_error.http_status == 429 and stale_error.reason == "rate_limited"
    assert cache.cooldown_remaining() == 300
    # During the cooldown the stale entry is served again without another upstream call
    _, _, _, stale_error = cache.get(KEY, fetch, validate, LATER)
    assert fetch.calls == 1 and stale_error.reason == "rate_limited"


def test_stale_data_from_previous_indian_day_is_never_reused(cache, clock, payload):
    late_evening = datetime(2023, 6, 15, 23, 50, tzinfo=IST)
    cache.get(KEY, CountingFetch(payload), validate, late_evening)
    clock.advance(LIVE_CACHE_TTL_SECONDS)
    next_day = datetime(2023, 6, 16, 0, 20, tzinfo=IST)
    with pytest.raises(WeatherDataError) as exc:
        cache.get(KEY, CountingFetch(rate_limit_error(600)), validate, next_day)
    assert exc.value.reason == "rate_limited"
    # The new day's key (new D) has no entry at all, so it also fails during the cooldown
    with pytest.raises(WeatherDataError):
        cache.get(("delhi", "2023-06-15", 10, 4), CountingFetch(payload), validate, next_day)


def test_stale_entry_is_reused_up_to_but_not_beyond_max_age(cache, clock, payload):
    cache.get(KEY, CountingFetch(payload), validate, FETCHED_AT)
    clock.advance(MAX_STALE_AGE_SECONDS - 1)
    _, _, _, stale_error = cache.get(KEY, CountingFetch(rate_limit_error(600)), validate, LATER)
    assert stale_error is not None
    clock.advance(1)
    fetch = CountingFetch(payload)
    with pytest.raises(WeatherDataError) as exc:
        cache.get(KEY, fetch, validate, LATER)
    assert exc.value.reason == "rate_limited" and fetch.calls == 0


@pytest.mark.parametrize("error", [
    WeatherDataError("timeout", "Open-Meteo did not respond in time"),
    WeatherDataError("upstream_error", "Open-Meteo returned HTTP 503", http_status=503),
    WeatherDataError("upstream_error", "Open-Meteo returned HTTP 500", http_status=500),
])
def test_timeouts_and_5xx_never_reuse_stale_data(cache, clock, payload, error):
    expire_entry(cache, clock, payload)
    with pytest.raises(WeatherDataError) as exc:
        cache.get(KEY, CountingFetch(error), validate, LATER)
    assert exc.value is error
    assert cache.cooldown_remaining() == 0


def test_invalid_refresh_never_falls_back_to_stale_data(cache, clock, payload):
    expire_entry(cache, clock, payload)
    broken = make_payload("delhi", "2023-06-05", "2023-06-18")
    broken["daily"]["temperature_2m_max"][9] = None  # 2023-06-14 = D
    with pytest.raises(WeatherDataError) as exc:
        cache.get(KEY, CountingFetch(broken), validate, LATER)
    assert exc.value.reason == "missing_values"


def test_successful_refresh_replaces_stale_entry(cache, clock, payload):
    expire_entry(cache, clock, payload)
    fresh = make_payload("delhi", "2023-06-05", "2023-06-18")
    fresh["current"]["temperature_2m"] = 31.5
    data, fetched_at, from_cache, stale_error = cache.get(KEY, CountingFetch(fresh), validate, LATER)
    assert (data, fetched_at, from_cache, stale_error) == (fresh, LATER, False, None)
    data, fetched_at, from_cache, stale_error = cache.get(KEY, CountingFetch(payload), validate, LATER)
    assert data["current"]["temperature_2m"] == 31.5
    assert fetched_at == LATER and from_cache is True and stale_error is None


def test_stale_payload_cannot_be_modified_by_callers(cache, clock, payload):
    expire_entry(cache, clock, payload)
    first, _, _, _ = cache.get(KEY, CountingFetch(rate_limit_error(600)), validate, LATER)
    first["daily"]["temperature_2m_max"][-1] = -99.0
    second, _, _, _ = cache.get(KEY, CountingFetch(payload), validate, LATER)
    assert second["daily"]["temperature_2m_max"][-1] != -99.0


@pytest.mark.parametrize("header, expected", [
    ("120", 120.0),
    (" 30 ", 30.0),
    ("Wed, 21 Oct 2026 07:30:00 GMT", 1800.0),
    ("Wed, 21 Oct 2026 06:00:00 GMT", 0.0),  # already in the past
    ("soon", None),
    ("", None),
    (None, None),
])
def test_parse_retry_after(header, expected):
    assert _parse_retry_after(header, now_utc=datetime(2026, 10, 21, 7, 0, tzinfo=timezone.utc)) == expected


# --- Concurrency ----------------------------------------------------------------------

def test_concurrent_requests_for_one_city_share_one_upstream_call(payload):
    cache = LiveWeatherCache()
    calls = []

    def slow_fetch():
        calls.append(1)
        time.sleep(0.2)
        return payload

    results = []
    threads = [threading.Thread(target=lambda: results.append(cache.get(KEY, slow_fetch, validate, FETCHED_AT)))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert len(results) == 8 and sorted(r[2] for r in results) == [False] + [True] * 7


# --- /predict labelling ---------------------------------------------------------------

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    with TestClient(api.app) as c:
        yield c


@pytest.fixture
def counted_live(monkeypatch):
    """Clock fixed at 12:00 IST on 15 June 2023; the weather service returns `state['outcome']`."""
    monkeypatch.setattr(api, "ist_now", lambda: utc(2023, 6, 15, 6, 30))
    state = {"calls": 0, "outcome": make_payload("delhi", "2023-06-05", "2023-06-18")}

    def fake_fetch(coords, *a, **k):
        state["calls"] += 1
        if isinstance(state["outcome"], Exception):
            raise state["outcome"]
        return state["outcome"]
    monkeypatch.setattr(api, "fetch_live_recent_weather", fake_fetch)
    return state


def test_predict_labels_cached_live_data_and_revalidates_it(client, counted_live, monkeypatch):
    first = client.get("/predict?city=Delhi").json()
    validations = []

    def counting_validation(raw, last_completed):
        validations.append(last_completed)
        return build_observation_window(raw, last_completed)
    monkeypatch.setattr(api, "build_observation_window", counting_validation)
    second = client.get("/predict?city=Delhi").json()

    assert counted_live["calls"] == 1
    assert first["data_source"] == "live" and second["data_source"] == "live_cached"
    assert first["is_demo_fallback"] is False and second["is_demo_fallback"] is False
    assert second["weather_data_fetched_at"] == first["weather_data_fetched_at"] == "2023-06-15T12:00:00+05:30"
    assert second["predicted_temp"] == first["predicted_temp"]
    assert second["input_window"] == first["input_window"]
    assert validations == [date(2023, 6, 14)]  # the cached response went through the same validation


def test_predict_during_429_cooldown_shows_demo_without_calling_api(client, counted_live):
    counted_live["outcome"] = rate_limit_error(retry_after=600)
    first = client.get("/predict?city=Delhi").json()
    second = client.get("/predict?city=Mumbai").json()
    assert counted_live["calls"] == 1
    for data in (first, second):
        assert data["is_demo_fallback"] is True
        assert data["data_source"] == "demo_snapshot"
        assert data["fallback_reason"] == "rate_limited"
        assert "not a forecast for today" in data["fallback_message"]


@pytest.fixture
def cache_clock(monkeypatch):
    """Controls the monotonic clock of the app's live-weather cache."""
    clock = FakeClock()
    monkeypatch.setattr(api.LIVE_WEATHER_CACHE, "_clock", clock)
    return clock


def at_ist(monkeypatch, hour, minute):
    """Move the app's Indian clock to hour:minute IST on 15 June 2023 (D stays 14 June)."""
    monkeypatch.setattr(api, "ist_now", lambda: datetime(2023, 6, 15, hour, minute, tzinfo=IST))


def test_predict_labels_stale_same_day_data_during_rate_limit(client, counted_live, cache_clock, monkeypatch):
    first = client.get("/predict?city=Delhi").json()
    assert first["data_source"] == "live" and first["stale_reason"] is None

    cache_clock.advance(LIVE_CACHE_TTL_SECONDS)
    at_ist(monkeypatch, 12, 40)
    counted_live["outcome"] = rate_limit_error(retry_after=600)
    after_429 = client.get("/predict?city=Delhi").json()
    during_cooldown = client.get("/predict?city=Delhi").json()
    other_city = client.get("/predict?city=Mumbai").json()

    assert counted_live["calls"] == 2  # one live fetch, one 429; nothing sent during the cooldown
    for data in (after_429, during_cooldown):
        assert data["data_source"] == "live_stale"
        assert data["stale_reason"] == "rate_limited"
        assert data["is_demo_fallback"] is False
        assert "fallback_message" not in data
        assert data["weather_data_fetched_at"] == first["weather_data_fetched_at"] == "2023-06-15T12:00:00+05:30"
        assert data["generated_at"] == "2023-06-15T12:40:00+05:30"
        assert data["last_completed_date"] == first["last_completed_date"] == "2023-06-14"
        assert data["input_window"] == first["input_window"]
        assert data["predicted_temp"] == first["predicted_temp"]
    # A city with no same-day data still gets the clearly labelled demo snapshot
    assert other_city["is_demo_fallback"] is True and other_city["data_source"] == "demo_snapshot"
    assert other_city["fallback_reason"] == "rate_limited"


def test_predict_beyond_max_stale_age_shows_demo(client, counted_live, cache_clock, monkeypatch):
    client.get("/predict?city=Delhi")
    cache_clock.advance(MAX_STALE_AGE_SECONDS)
    at_ist(monkeypatch, 15, 0)
    counted_live["outcome"] = rate_limit_error(retry_after=600)
    data = client.get("/predict?city=Delhi").json()
    assert data["is_demo_fallback"] is True and data["data_source"] == "demo_snapshot"
    assert data["fallback_reason"] == "rate_limited"


def test_predict_timeout_after_ttl_shows_demo_not_stale(client, counted_live, cache_clock, monkeypatch):
    client.get("/predict?city=Delhi")
    cache_clock.advance(LIVE_CACHE_TTL_SECONDS)
    at_ist(monkeypatch, 12, 40)
    counted_live["outcome"] = WeatherDataError("timeout", "Open-Meteo did not respond in time")
    data = client.get("/predict?city=Delhi").json()
    assert data["is_demo_fallback"] is True and data["data_source"] == "demo_snapshot"
    assert data["fallback_reason"] == "timeout"
