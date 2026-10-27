"""
scripts/fetch_scheduled_weather.py
Run by the scheduled GitHub Actions workflow (.github/workflows/scheduled-weather.yml).

Fetches the recent Open-Meteo data for every city with the same request the API uses
(backend.live_data.fetch_live_recent_weather), validates it with the same check
(build_observation_window for D = yesterday in India) and writes weather.json, which the
deployed API reads instead of calling Open-Meteo itself.

- Only validated responses are written, each with its real fetch time (UTC) and its D.
- A city that fails keeps its previous entry unchanged, with its original fetch time and D,
  so the API rejects it once it is outdated. Nothing is re-dated.
- The file holds only public Open-Meteo data: no keys, tokens, URLs or logs.
- Exit code 1 if no city could be updated, so the workflow run shows as failed.

Usage:
    python scripts/fetch_scheduled_weather.py --output out/weather.json [--previous previous.json]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.data_fetch import CITIES  # noqa: E402
from backend.live_data import (  # noqa: E402
    LIVE_FORECAST_DAYS,
    LIVE_PAST_DAYS,
    SCHEDULED_SCHEMA_VERSION,
    WeatherDataError,
    build_observation_window,
    fetch_live_recent_weather,
    forecast_dates,
    ist_now,
)

SOURCE_NOTE = "Open-Meteo Forecast API (https://open-meteo.com), CC BY 4.0"


def load_previous(path):
    """Previous file's city entries, or {} if there is none or it is unreadable."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(doc, dict) or doc.get("schema_version") != SCHEDULED_SCHEMA_VERSION:
        return {}
    cities = doc.get("cities")
    return cities if isinstance(cities, dict) else {}


def build_snapshot(previous_cities, now=None, fetch=fetch_live_recent_weather, clock=None):
    """Return (document, updated city keys, failures {city key: reason})."""
    now = now or ist_now()
    clock = clock or (lambda: datetime.now(timezone.utc))
    last_completed, _ = forecast_dates(now)
    cities, updated, failures = {}, [], {}
    for name, coords in CITIES.items():
        key = name.lower()
        try:
            raw = fetch(coords, past_days=LIVE_PAST_DAYS, forecast_days=LIVE_FORECAST_DAYS)
            fetched_at = clock()
            build_observation_window(raw, last_completed)
        except WeatherDataError as e:
            failures[key] = e.reason
            if isinstance(previous_cities.get(key), dict):
                cities[key] = previous_cities[key]  # unchanged: keeps its own fetched_at_utc and D
            continue
        cities[key] = {
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
            "last_completed_date": last_completed.isoformat(),
            "fetched_at_utc": fetched_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "raw": raw,
        }
        updated.append(key)
    document = {
        "schema_version": SCHEDULED_SCHEMA_VERSION,
        "source": SOURCE_NOTE,
        "generated_at_utc": clock().astimezone(timezone.utc).isoformat(timespec="seconds"),
        "cities": cities,
    }
    return document, updated, failures


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fetch and validate Open-Meteo data for the scheduled weather file.")
    parser.add_argument("--output", required=True, help="Path of the weather.json to write")
    parser.add_argument("--previous", help="Previous weather.json (failed cities keep their old entry)")
    args = parser.parse_args(argv)

    document, updated, failures = build_snapshot(load_previous(args.previous))
    for key, reason in failures.items():
        kept = "previous entry kept with its original fetch time" if key in document["cities"] else "no entry"
        print(f"{key}: not updated ({reason}); {kept}")
    for key in updated:
        entry = document["cities"][key]
        print(f"{key}: updated, D={entry['last_completed_date']}, fetched_at_utc={entry['fetched_at_utc']}")
    if not updated:
        print("No city could be updated; weather.json not written.")
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(document, f, separators=(",", ":"), sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
