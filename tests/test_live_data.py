"""
tests/test_live_data.py
Forecast-date (timezone) logic and validation of the live observation window.
"""

from datetime import date, datetime

import pytest

from backend.live_data import IST, WeatherDataError, build_observation_window, forecast_dates
from src.features import REQUIRED_HISTORY_DAYS
from tests.conftest import make_payload, utc


# --- Timezone / forecast dates -------------------------------------------------------

def test_utc_server_after_indian_midnight_uses_indian_date():
    # 18:33 UTC on 24 Oct is 00:03 IST on 25 Oct: D must be 24 Oct and the target 26 Oct
    assert forecast_dates(utc(2026, 10, 24, 18, 33)) == (date(2026, 10, 24), date(2026, 10, 26))


def test_utc_server_before_indian_midnight():
    # 18:29 UTC on 24 Oct is 23:59 IST on 24 Oct: D is 23 Oct, target 25 Oct (tomorrow in India)
    assert forecast_dates(utc(2026, 10, 24, 18, 29)) == (date(2026, 10, 23), date(2026, 10, 25))


def test_target_is_always_tomorrow_in_india():
    for hour in range(24):
        now = utc(2026, 3, 1, hour, 0)
        _, target = forecast_dates(now)
        ist_today = now.astimezone(IST).date()
        assert (target - ist_today).days == 1


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError):
        forecast_dates(datetime(2026, 10, 24, 12, 0))


# --- Observation window -------------------------------------------------------------

D = date(2023, 6, 14)


def test_window_is_exactly_seven_completed_days_ending_on_d():
    # Payload also contains "today" (15 Jun) and forecast days up to 18 Jun
    window = build_observation_window(make_payload("delhi", "2023-06-05", "2023-06-18"), D)
    assert len(window) == REQUIRED_HISTORY_DAYS
    assert window["date"].iloc[-1].date() == D
    assert window["date"].iloc[0].date() == date(2023, 6, 8)


def test_partial_today_and_forecast_rows_never_used():
    payload = make_payload("delhi", "2023-06-05", "2023-06-18",
                           null_values=[("2023-06-15", "temperature_2m_max"), ("2023-06-17", "precipitation_sum")])
    window = build_observation_window(payload, D)  # nulls after D must not matter
    assert window["date"].max().date() == D


@pytest.mark.parametrize("null_date", ["2023-06-14", "2023-06-11", "2023-06-08"])
def test_missing_value_inside_window_raises(null_date):
    payload = make_payload("delhi", "2023-06-05", "2023-06-18", null_values=[(null_date, "relative_humidity_2m_mean")])
    with pytest.raises(WeatherDataError) as exc:
        build_observation_window(payload, D)
    assert exc.value.reason == "missing_values"
    assert null_date in str(exc.value)


def test_missing_middle_date_raises():
    payload = make_payload("delhi", "2023-06-05", "2023-06-18", drop_dates=["2023-06-12"])
    with pytest.raises(WeatherDataError) as exc:
        build_observation_window(payload, D)
    assert exc.value.reason == "insufficient_history"


def test_missing_recent_observation_raises():
    # API data ends before D: must not fall back to an older origin
    with pytest.raises(WeatherDataError) as exc:
        build_observation_window(make_payload("delhi", "2023-06-01", "2023-06-13"), D)
    assert exc.value.reason == "insufficient_history"
    assert "2023-06-14" in str(exc.value)


def test_insufficient_history_raises():
    with pytest.raises(WeatherDataError) as exc:
        build_observation_window(make_payload("delhi", "2023-06-10", "2023-06-18"), D)
    assert exc.value.reason == "insufficient_history"


def test_unexpected_date_range_raises():
    # Entirely different dates (e.g. API returned another period)
    with pytest.raises(WeatherDataError):
        build_observation_window(make_payload("delhi", "2022-01-01", "2022-01-20"), D)


def test_missing_variable_raises():
    payload = make_payload("delhi", "2023-06-05", "2023-06-18")
    del payload["daily"]["wind_speed_10m_max"]
    with pytest.raises(WeatherDataError) as exc:
        build_observation_window(payload, D)
    assert exc.value.reason == "missing_variable"


@pytest.mark.parametrize("mutate", [
    lambda p: p.pop("daily"),
    lambda p: p["daily"].update(time="not-a-list"),
    lambda p: p["daily"]["temperature_2m_max"].pop(),
    lambda p: p["daily"]["time"].__setitem__(3, "14/06/2023"),
    lambda p: p["daily"]["time"].__setitem__(3, p["daily"]["time"][2]),
])
def test_malformed_payload_raises(mutate):
    payload = make_payload("delhi", "2023-06-05", "2023-06-18")
    mutate(payload)
    with pytest.raises(WeatherDataError) as exc:
        build_observation_window(payload, D)
    assert exc.value.reason == "malformed_response"


def test_non_numeric_value_raises():
    payload = make_payload("delhi", "2023-06-05", "2023-06-18")
    payload["daily"]["precipitation_sum"][8] = "n/a"  # 2023-06-13
    with pytest.raises(WeatherDataError) as exc:
        build_observation_window(payload, D)
    assert exc.value.reason == "missing_values"
