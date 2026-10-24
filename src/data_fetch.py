"""
src/data_fetch.py
Fetches 5 years of daily historical weather data for Bengaluru, Delhi, and Mumbai
from the Open-Meteo Historical Weather API (free, no API key needed).
Saves CSV files to data/{city}.csv.
"""

import os
import time
import requests
import pandas as pd

CITIES = {
    "Bengaluru": {"latitude": 12.9716, "longitude": 77.5946},
    "Delhi": {"latitude": 28.6139, "longitude": 77.2090},
    "Mumbai": {"latitude": 19.0760, "longitude": 72.8777}
}

START_DATE = "2019-01-01"
END_DATE = "2023-12-31"

DAILY_VARIABLES = [
    "temperature_2m_max",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "precipitation_sum"
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def fetch_city_history(city_name: str, coords: dict, start_date=START_DATE, end_date=END_DATE) -> pd.DataFrame:
    """Fetch daily weather history from Open-Meteo Archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Asia/Kolkata"
    }
    
    print(f"Fetching historical data for {city_name} from {start_date} to {end_date}...")
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    daily = data.get("daily", {})
    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "temperature_2m_max": daily["temperature_2m_max"],
        "relative_humidity_2m_mean": daily["relative_humidity_2m_mean"],
        "wind_speed_10m_max": daily["wind_speed_10m_max"],
        "precipitation_sum": daily["precipitation_sum"]
    })
    
    # Sort strictly by date
    df = df.sort_values("date").reset_index(drop=True)
    df["city"] = city_name
    return df

def fetch_all_cities():
    """Fetch and save 5-year data for all 3 cities."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for city, coords in CITIES.items():
        filename = os.path.join(DATA_DIR, f"{city.lower()}.csv")
        try:
            df = fetch_city_history(city, coords)
            df.to_csv(filename, index=False)
            print(f"Successfully saved {len(df)} records to {filename}")
        except Exception as e:
            print(f"Error fetching data for {city}: {e}")
        time.sleep(1)  # Respectful pause

if __name__ == "__main__":
    fetch_all_cities()
