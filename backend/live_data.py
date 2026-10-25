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

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from src.data_fetch import DAILY_VARIABLES
from src.features import FORECAST_HORIZON_DAYS, REQUIRED_HISTORY_DAYS

# India does not observe daylight saving time, so a fixed UTC+05:30 offset is exact.
IST = timezone(timedelta(hours=5, minutes=30), name="IST")

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CURRENT_VARIABLES = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "dew_point_2m"]
REQUEST_TIMEOUT_SECONDS = 10
# Longest excerpt of an error response body kept in a log message
MAX_ERROR_DETAIL_CHARS = 200


class WeatherDataError(Exception):
    """The weather service failed, or returned data that cannot support a valid forecast."""

    def __init__(self, reason: str, message: str, http_status: int = None):
        self.reason = reason
        self.http_status = http_status
        super().__init__(message)


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


def fetch_live_recent_weather(coords: dict, past_days: int = 10, forecast_days: int = 4) -> dict:
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
    if resp.status_code >= 400:
        raise WeatherDataError("upstream_error", _describe_http_error(resp), http_status=resp.status_code)
    try:
        data = resp.json()
    except ValueError as e:
        raise WeatherDataError("malformed_response", "Open-Meteo response was not valid JSON") from e
    if not isinstance(data, dict):
        raise WeatherDataError("malformed_response", "Open-Meteo response was not a JSON object")
    return data


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
