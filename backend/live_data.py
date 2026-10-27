"""
backend/live_data.py
Fetching and validating the recent daily observations used for a live forecast.

Forecast definition (identical to training, see src/features.py):
- D is the last fully completed day in India, i.e. yesterday in Asia/Kolkata.
- The target is the maximum temperature on D + FORECAST_HORIZON_DAYS (tomorrow in India).
- Only the consecutive days D-6 .. D are used as model inputs. Today's partial day and the
  forecast days returned by the API are never treated as observations.

The server clock is never trusted to be in Indian time (cloud hosts usually run in UTC),
so all calendar logic goes through ist_now() / forecast_dates().
"""

import copy
import json
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import pandas as pd
import requests

from src.data_fetch import DAILY_VARIABLES
from src.features import FORECAST_HORIZON_DAYS, REQUIRED_HISTORY_DAYS

# India does not observe daylight saving time, so a fixed UTC+05:30 offset is exact.
IST = timezone(timedelta(hours=5, minutes=30), name="IST")

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CURRENT_VARIABLES = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "dew_point_2m"]
REQUEST_TIMEOUT_SECONDS = 10
# Request window: past days (covers D-6 .. D with margin) and forecast days (covers the target date)
LIVE_PAST_DAYS = 10
LIVE_FORECAST_DAYS = 4
# Longest excerpt of an error response body kept in a log message
MAX_ERROR_DETAIL_CHARS = 200

# Validated Open-Meteo responses are reused for this long (see LiveWeatherCache)
LIVE_CACHE_TTL_SECONDS = 30 * 60
# While rate-limited, an expired same-day response may be reused (labelled stale) up to this age.
# Observations D-6 .. D do not change during the day, but the current conditions and the
# official forecast in the same response do, so the reuse window is kept short.
MAX_STALE_AGE_SECONDS = 3 * 60 * 60
# After HTTP 429, no request is sent for Retry-After seconds, or this long if the header is absent
DEFAULT_RATE_LIMIT_BACKOFF_SECONDS = 15 * 60
MAX_RATE_LIMIT_BACKOFF_SECONDS = 24 * 60 * 60

# Scheduled fetch: a GitHub Actions job (scripts/fetch_scheduled_weather.py) saves validated
# Open-Meteo responses to weather.json on the weather-data branch. SCHEDULED_WEATHER_URL overrides
# the location; an empty value disables the scheduled source.
DEFAULT_SCHEDULED_WEATHER_URL = (
    "https://raw.githubusercontent.com/bhavitha1917/ml-weather-predictor/weather-data/weather.json")
SCHEDULED_SCHEMA_VERSION = 1
SCHEDULED_MAX_BYTES = 1_000_000
# A scheduled file is re-read at most this often per city, and not retried for this long after a failure
SCHEDULED_CACHE_TTL_SECONDS = 10 * 60
SCHEDULED_RETRY_SECONDS = 5 * 60
# Tolerance for a fetch timestamp slightly in the future (clock differences between runner and server)
SCHEDULED_CLOCK_SKEW_SECONDS = 5 * 60


class WeatherDataError(Exception):
    """The weather service failed, or returned data that cannot support a valid forecast."""

    def __init__(self, reason: str, message: str, http_status: int = None, retry_after_seconds: float = None):
        self.reason = reason
        self.http_status = http_status
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)


def _parse_retry_after(value, now_utc: datetime = None):
    """Seconds to wait from a Retry-After header (delta-seconds or HTTP-date), or None if absent/invalid."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if retry_at.tzinfo is None:
        return None
    return max(0.0, (retry_at - (now_utc or datetime.now(timezone.utc))).total_seconds())


def _describe_http_error(resp) -> str:
    """
    One-line, size-limited description of an HTTP error response, safe to log.
    Keeps the status, the Retry-After header and Open-Meteo's own error text
    ({"error": true, "reason": "..."}); never the request URL or other headers.
    """
    message = f"Open-Meteo returned HTTP {resp.status_code}"
    if resp.reason:
        message += f" {resp.reason}"
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        message += f"; Retry-After: {retry_after[:20]}"
    try:
        body = resp.json()
        detail = body.get("reason") if isinstance(body, dict) else None
    except ValueError:
        detail = resp.text
    detail = " ".join(str(detail or "").split())
    if detail:
        if len(detail) > MAX_ERROR_DETAIL_CHARS:
            detail = detail[:MAX_ERROR_DETAIL_CHARS] + "..."
        message += f"; detail: {detail}"
    return message


def ist_now() -> datetime:
    """Current time in India, independent of the server's local timezone."""
    return datetime.now(timezone.utc).astimezone(IST)


def forecast_dates(now: datetime) -> tuple:
    """Return (last completed day D, target date) for a timezone-aware 'now'."""
    if now.tzinfo is None:
        raise ValueError("forecast_dates() needs a timezone-aware datetime")
    today_ist = now.astimezone(IST).date()
    last_completed = today_ist - timedelta(days=1)
    return last_completed, last_completed + timedelta(days=FORECAST_HORIZON_DAYS)


def fetch_live_recent_weather(coords: dict, past_days: int = LIVE_PAST_DAYS,
                              forecast_days: int = LIVE_FORECAST_DAYS) -> dict:
    """
    Fetch recent daily values (past_days), today's partial day, the official forecast
    days and the current conditions from the Open-Meteo Forecast API.
    Raises WeatherDataError for network, HTTP and decoding failures.
    """
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "past_days": past_days,
        "forecast_days": forecast_days,
        "daily": ",".join(DAILY_VARIABLES),
        "current": ",".join(CURRENT_VARIABLES),
        "timezone": "Asia/Kolkata",
    }
    try:
        resp = requests.get(FORECAST_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.Timeout as e:
        raise WeatherDataError("timeout", "Open-Meteo did not respond in time") from e
    except requests.ConnectionError as e:
        raise WeatherDataError("upstream_error", f"Could not connect to Open-Meteo ({type(e).__name__})") from e
    except requests.RequestException as e:
        raise WeatherDataError("upstream_error", f"Open-Meteo request failed ({type(e).__name__})") from e
    if resp.status_code == 429:
        raise WeatherDataError("rate_limited", _describe_http_error(resp), http_status=429,
                               retry_after_seconds=_parse_retry_after(resp.headers.get("Retry-After")))
    if resp.status_code >= 400:
        raise WeatherDataError("upstream_error", _describe_http_error(resp), http_status=resp.status_code)
    try:
        data = resp.json()
    except ValueError as e:
        raise WeatherDataError("malformed_response", "Open-Meteo response was not valid JSON") from e
    if not isinstance(data, dict):
        raise WeatherDataError("malformed_response", "Open-Meteo response was not a JSON object")
    return data


def scheduled_weather_url():
    """URL of the scheduled weather file, or None when the scheduled source is disabled."""
    return os.environ.get("SCHEDULED_WEATHER_URL", DEFAULT_SCHEDULED_WEATHER_URL).strip() or None


def _is_allowed_scheduled_url(url: str) -> bool:
    """HTTPS only; plain HTTP is accepted for a local test server."""
    parsed = urlparse(url)
    return parsed.scheme == "https" or (parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost"))


def fetch_scheduled_weather(city_key: str, coords: dict, last_completed: date, now_ist: datetime,
                            url: str = None) -> dict:
    """
    Read one city's entry from the scheduled weather file and return
    {"raw": <Open-Meteo response>, "fetched_at": <when the runner fetched it, IST>}.

    The file is untrusted input: size, schema, coordinates, D and age are checked here, and the
    caller still runs build_observation_window on "raw". Entries for another D, or older than
    MAX_STALE_AGE_SECONDS, are rejected. Errors never carry an HTTP status, so a failure here
    never starts the Open-Meteo rate-limit cooldown.
    """
    url = url or scheduled_weather_url()
    if not url:
        raise WeatherDataError("scheduled_disabled", "Scheduled weather source is disabled")
    if not _is_allowed_scheduled_url(url):
        raise WeatherDataError("scheduled_invalid", "Scheduled weather URL must use HTTPS")
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS, stream=True)
    except requests.RequestException as e:
        raise WeatherDataError("scheduled_unavailable",
                               f"Scheduled weather file could not be fetched ({type(e).__name__})") from e
    with resp:
        if resp.status_code != 200:
            raise WeatherDataError("scheduled_unavailable", f"Scheduled weather file returned HTTP {resp.status_code}")
        body = bytearray()
        try:
            for chunk in resp.iter_content(chunk_size=65536):
                body += chunk
                if len(body) > SCHEDULED_MAX_BYTES:
                    raise WeatherDataError("scheduled_invalid",
                                           f"Scheduled weather file is larger than {SCHEDULED_MAX_BYTES} bytes")
        except requests.RequestException as e:
            raise WeatherDataError("scheduled_unavailable",
                                   f"Scheduled weather file could not be read ({type(e).__name__})") from e
    try:
        doc = json.loads(bytes(body))
    except ValueError as e:
        raise WeatherDataError("scheduled_invalid", "Scheduled weather file is not valid JSON") from e
    if not isinstance(doc, dict) or doc.get("schema_version") != SCHEDULED_SCHEMA_VERSION:
        raise WeatherDataError("scheduled_invalid", "Scheduled weather file has an unknown schema")
    cities = doc.get("cities")
    entry = cities.get(city_key) if isinstance(cities, dict) else None
    if not isinstance(entry, dict):
        raise WeatherDataError("scheduled_missing_city", f"Scheduled weather file has no entry for {city_key}")
    if entry.get("latitude") != coords["latitude"] or entry.get("longitude") != coords["longitude"]:
        raise WeatherDataError("scheduled_invalid", f"Scheduled entry for {city_key} has unexpected coordinates")
    if entry.get("last_completed_date") != last_completed.isoformat():
        raise WeatherDataError("scheduled_outdated",
                               f"Scheduled entry for {city_key} is for D={entry.get('last_completed_date')!s:.10}, "
                               f"expected {last_completed.isoformat()}")
    try:
        fetched_at = datetime.fromisoformat(str(entry.get("fetched_at_utc")))
    except ValueError as e:
        raise WeatherDataError("scheduled_invalid", f"Scheduled entry for {city_key} has no valid fetch time") from e
    if fetched_at.tzinfo is None:
        raise WeatherDataError("scheduled_invalid", f"Scheduled entry for {city_key} has a fetch time without timezone")
    age = (now_ist - fetched_at).total_seconds()
    if age < -SCHEDULED_CLOCK_SKEW_SECONDS:
        raise WeatherDataError("scheduled_invalid", f"Scheduled entry for {city_key} has a fetch time in the future")
    if age >= MAX_STALE_AGE_SECONDS:
        raise WeatherDataError("scheduled_outdated",
                               f"Scheduled entry for {city_key} was fetched {age / 3600:.1f} h ago")
    if not isinstance(entry.get("raw"), dict):
        raise WeatherDataError("scheduled_invalid", f"Scheduled entry for {city_key} has no weather data")
    return {"raw": entry["raw"], "fetched_at": fetched_at.astimezone(IST)}


def build_observation_window(raw: dict, last_completed: date) -> pd.DataFrame:
    """
    Extract and validate the REQUIRED_HISTORY_DAYS consecutive completed days ending on
    last_completed (D). Rows after D (today's partial day and forecast days) are ignored.

    Raises WeatherDataError instead of silently using an older day, a partial day,
    or forecast rows when the data cannot support a forecast from D.
    """
    daily = raw.get("daily") if isinstance(raw, dict) else None
    if not isinstance(daily, dict) or not isinstance(daily.get("time"), list):
        raise WeatherDataError("malformed_response", "Response has no daily time series")

    times = daily["time"]
    for var in DAILY_VARIABLES:
        if not isinstance(daily.get(var), list):
            raise WeatherDataError("missing_variable", f"Response is missing daily variable '{var}'")
        if len(daily[var]) != len(times):
            raise WeatherDataError("malformed_response", f"Daily variable '{var}' has a different length from 'time'")

    dates = pd.to_datetime(pd.Series(times), format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise WeatherDataError("malformed_response", "Response contains unparseable dates")
    if dates.duplicated().any():
        raise WeatherDataError("malformed_response", "Response contains duplicate dates")

    values = pd.DataFrame({var: pd.to_numeric(pd.Series(daily[var]), errors="coerce") for var in DAILY_VARIABLES})
    values.index = dates

    first_needed = last_completed - timedelta(days=REQUIRED_HISTORY_DAYS - 1)
    needed = pd.date_range(first_needed, last_completed, freq="D")
    absent = [d.date().isoformat() for d in needed if d not in values.index]
    if absent:
        raise WeatherDataError("insufficient_history", f"Observations missing for {', '.join(absent)}")

    window = values.loc[needed]
    if window.isna().any().any():
        gaps = [f"{d.date().isoformat()}:{var}" for d, row in window.iterrows() for var in DAILY_VARIABLES if pd.isna(row[var])]
        raise WeatherDataError("missing_values", f"Missing or non-numeric values for {', '.join(gaps)}")

    return window.rename_axis("date").reset_index()


class LiveWeatherCache:
    """
    In-memory, per-process cache of validated Open-Meteo responses, plus a rate-limit cooldown.

    - Only responses that pass validate() are stored; failures are never cached.
    - Entries expire after ttl_seconds. Keys include D, so a new Indian day never reuses yesterday's data.
    - One lock per key: concurrent requests for the same city wait for a single upstream call.
    - After HTTP 429 no upstream request is sent until the cooldown ends (Retry-After, else a default).
      The quota is per server IP, so the cooldown applies to every city. Requests during the
      cooldown fail with reason "rate_limited", and the caller shows its usual fallback.
    - Stale reuse: if the entry has expired but is younger than max_stale_seconds and was fetched
      on the same Indian day, it is returned (with the rate-limit error) instead of failing, but only
      when the cooldown is active or the fetch gets HTTP 429. Timeouts, 5xx and invalid data never
      trigger stale reuse, and no extra upstream request is ever sent for it.
    This reduces how often the API is called; it does not work around an exhausted quota.
    """

    def __init__(self, ttl_seconds=LIVE_CACHE_TTL_SECONDS,
                 default_backoff_seconds=DEFAULT_RATE_LIMIT_BACKOFF_SECONDS,
                 max_backoff_seconds=MAX_RATE_LIMIT_BACKOFF_SECONDS,
                 max_stale_seconds=MAX_STALE_AGE_SECONDS, clock=time.monotonic):
        self.ttl_seconds = ttl_seconds
        self.max_stale_seconds = max_stale_seconds
        self.default_backoff_seconds = default_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self._clock = clock
        self._entries = {}        # key -> (stored_at monotonic, fetched_at IST datetime, payload)
        self._key_locks = {}
        self._lock = threading.Lock()
        self._blocked_until = None

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._blocked_until = None

    def _lock_for(self, key):
        with self._lock:
            return self._key_locks.setdefault(key, threading.Lock())

    def cooldown_remaining(self) -> float:
        with self._lock:
            if self._blocked_until is None:
                return 0.0
            return max(0.0, self._blocked_until - self._clock())

    def _start_cooldown(self, retry_after_seconds):
        wait = self.default_backoff_seconds if retry_after_seconds is None else retry_after_seconds
        wait = min(max(wait, 0.0), self.max_backoff_seconds)
        with self._lock:
            self._blocked_until = self._clock() + wait

    def _too_old(self, entry, now_ist: datetime) -> bool:
        """True when the data itself (by its real fetch time) is at least max_stale_seconds old."""
        return (now_ist - entry[1]).total_seconds() >= self.max_stale_seconds

    def _stale_entry(self, entry, now_ist: datetime):
        """The expired entry if it may still be reused while rate-limited, else None."""
        if entry is None or self._clock() - entry[0] >= self.max_stale_seconds or self._too_old(entry, now_ist):
            return None
        if entry[1].astimezone(IST).date() != now_ist.astimezone(IST).date():
            return None  # never reuse data fetched on a previous Indian day
        return entry

    def get(self, key, fetch, validate, now_ist: datetime, fetched_at_of=None):
        """
        Return (payload, fetched_at, from_cache, stale_error). fetch() calls the API; validate(payload)
        raises WeatherDataError for unusable data. The returned payload is a copy the caller may modify.
        stale_error is None for fresh or cached data; for a reused stale entry it is the rate-limit
        WeatherDataError that prevented a refresh.
        fetched_at_of(payload) gives the data's real fetch time when it was fetched earlier elsewhere
        (the scheduled file); by default it is now_ist. Data whose real age reaches max_stale_seconds
        is never returned.
        """
        with self._lock_for(key):
            with self._lock:
                entry = self._entries.get(key)
            if entry is not None and self._clock() - entry[0] < self.ttl_seconds and not self._too_old(entry, now_ist):
                _, fetched_at, payload = entry
                return copy.deepcopy(payload), fetched_at, True, None
            stale = self._stale_entry(entry, now_ist)

            remaining = self.cooldown_remaining()
            if remaining > 0:
                error = WeatherDataError(
                    "rate_limited",
                    f"Open-Meteo rate limit: no request sent; cooling down for another {remaining:.0f} s")
                if stale is None:
                    raise error
                return copy.deepcopy(stale[2]), stale[1], True, error

            try:
                payload = fetch()
            except WeatherDataError as e:
                if e.http_status == 429:
                    self._start_cooldown(e.retry_after_seconds)
                    if stale is not None:
                        return copy.deepcopy(stale[2]), stale[1], True, e
                raise
            validate(payload)
            fetched_at = fetched_at_of(payload) if fetched_at_of is not None else now_ist
            stored_at = self._clock()
            with self._lock:
                # Drop entries too old even for stale reuse (e.g. previous days' keys, never read again)
                for old_key in [k for k, (t, _, _) in self._entries.items() if stored_at - t >= self.max_stale_seconds]:
                    del self._entries[old_key]
                self._entries[key] = (stored_at, fetched_at, copy.deepcopy(payload))
            return payload, fetched_at, False, None
