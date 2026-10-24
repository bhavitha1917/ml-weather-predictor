"""
scripts/capture_demo_snapshots.py
Captures real /predict responses (live Open-Meteo data + saved models) into
backend/sample_responses.json. The API serves these, clearly labelled as a demo snapshot
with their original dates, only when live weather data cannot be used.

Usage (needs internet access):
    python scripts/capture_demo_snapshots.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import backend.app as api


def main():
    api.load_artifacts()
    snapshots = {}
    for city_key in api.CITY_KEYS:
        if not api.city_is_ready(city_key):
            raise SystemExit(f"Artifacts for {city_key} are not loaded: {api.ARTIFACT_ERRORS.get(city_key)}")
        snapshots[city_key] = api.predict_city(city_key)  # raises if live data is unusable
        print(f"{city_key}: D={snapshots[city_key]['last_completed_date']} "
              f"target={snapshots[city_key]['target_date']} pred={snapshots[city_key]['predicted_temp']}")
    out_path = os.path.join(api.BACKEND_DIR, "sample_responses.json")
    with open(out_path, "w") as f:
        json.dump(snapshots, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
