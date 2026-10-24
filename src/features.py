"""
src/features.py
Shared feature engineering module used for BOTH offline training and live serving.
Adheres strictly to Blueprint Prompt 2:
- Features built ONLY from days up to day D.
- Target is max temperature on D+2 (tomorrow's max when run today on completed days).
- 7 lags of max temperature (D, D-1, ... D-6)
- 7-day rolling mean and standard deviation
- Humidity (D, D-1)
- Wind speed (D)
- Precipitation (D, D-1, and 3-day sum)
- Sine and cosine of target day-of-year (cyclical pair)
- Persistence baseline (T_max on D)
- Climatology baseline (computed strictly from training years)
"""

import numpy as np
import pandas as pd

# Target = max temperature FORECAST_HORIZON_DAYS after the last completed day D.
FORECAST_HORIZON_DAYS = 2
# The longest look-back used by any feature (7 temperature lags / 7-day rolling window),
# so the features for day D need the consecutive days D-6 .. D.
REQUIRED_HISTORY_DAYS = 7

FEATURE_COLUMNS = [
    "temp_lag_0",
    "temp_lag_1",
    "temp_lag_2",
    "temp_lag_3",
    "temp_lag_4",
    "temp_lag_5",
    "temp_lag_6",
    "temp_roll_mean_7",
    "temp_roll_std_7",
    "temp_momentum_1",
    "temp_accel_1",
    "humidity_lag_0",
    "humidity_lag_1",
    "dew_point_lag_0",
    "wind_speed_lag_0",
    "precip_lag_0",
    "precip_lag_1",
    "precip_sum_3d",
    "sin_doy",
    "cos_doy",
    "sin_doy_semi",
    "cos_doy_semi"
]

def seasonal_features(target_doy) -> dict:
    """
    Sine/cosine of the target day-of-year (annual and semi-annual harmonics).
    Accepts a scalar or a pandas Series; shared by make_features and the /simulate endpoint.
    """
    return {
        "sin_doy": np.sin(2 * np.pi * target_doy / 365.25),
        "cos_doy": np.cos(2 * np.pi * target_doy / 365.25),
        "sin_doy_semi": np.sin(4 * np.pi * target_doy / 365.25),
        "cos_doy_semi": np.cos(4 * np.pi * target_doy / 365.25),
    }

def compute_climatology_map(train_df: pd.DataFrame) -> dict:
    """
    Computes daily climatological mean of max temperature strictly from training years.
    Returns a dict mapping dayofyear (1-366) to average max temperature.
    """
    df = train_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    df["doy"] = df["date"].dt.dayofyear
    clim = df.groupby("doy")["temperature_2m_max"].mean().to_dict()
    
    # Ensure all days 1..366 are present (e.g., leap day 366)
    if 366 not in clim and 365 in clim:
        clim[366] = clim[365]
    return clim

def make_features(df: pd.DataFrame, climatology_map: dict = None, is_training: bool = True) -> pd.DataFrame:
    """
    Builds features strictly from completed days up to day D.
    
    Parameters:
    - df: DataFrame with columns: date, temperature_2m_max, relative_humidity_2m_mean, 
          wind_speed_10m_max, precipitation_sum.
    - climatology_map: Dict of {doy: mean_max_temp} computed from training data.
    - is_training: If True, computes the target_temp_d2 (T_max at D+2) and drops rows with NaN.
                   If False (serving), only features for the last row (day D) are built.
                   
    Returns:
    - Clean DataFrame with FEATURE_COLUMNS, baseline columns, and target (if training).
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Lags are built with row shifts, which equal calendar-day lags only if the dates
    # are consecutive. A gap or duplicate would silently misalign every lag feature.
    day_steps = df["date"].diff().dropna()
    if not (day_steps == pd.Timedelta(days=1)).all():
        raise ValueError("make_features requires consecutive daily dates with no gaps or duplicates")

    # Max temperature lags for D, D-1, ... D-6 (7 lags)
    # lag 0 is Day D itself
    for lag in range(7):
        df[f"temp_lag_{lag}"] = df["temperature_2m_max"].shift(lag)
        
    # 7-day rolling mean and standard deviation (over D-6 .. D)
    # min_periods=7 ensures all 7 days are observed
    df["temp_roll_mean_7"] = df["temperature_2m_max"].rolling(window=7, min_periods=7).mean()
    df["temp_roll_std_7"] = df["temperature_2m_max"].rolling(window=7, min_periods=7).std().fillna(0.0)
    
    # Physics-Guided Temperature Dynamics: Velocity & Acceleration
    df["temp_momentum_1"] = df["temp_lag_0"] - df["temp_lag_1"]
    df["temp_accel_1"] = (df["temp_lag_0"] - df["temp_lag_1"]) - (df["temp_lag_1"] - df["temp_lag_2"])
    
    # Humidity for D and D-1
    df["humidity_lag_0"] = df["relative_humidity_2m_mean"]
    df["humidity_lag_1"] = df["relative_humidity_2m_mean"].shift(1)
    
    # Physics-Guided Moisture Metric: Dew Point approximation (Magnus formula approximation)
    df["dew_point_lag_0"] = df["temp_lag_0"] - ((100.0 - df["humidity_lag_0"]) / 5.0)
    
    # Wind speed for D
    df["wind_speed_lag_0"] = df["wind_speed_10m_max"]
    
    # Precipitation for D, D-1, and 3-day sum (D-2, D-1, D)
    df["precip_lag_0"] = df["precipitation_sum"]
    df["precip_lag_1"] = df["precipitation_sum"].shift(1)
    df["precip_sum_3d"] = df["precipitation_sum"].rolling(window=3, min_periods=3).sum()
    
    # Target date is Day D + FORECAST_HORIZON_DAYS
    target_dates = df["date"] + pd.Timedelta(days=FORECAST_HORIZON_DAYS)
    df["target_date"] = target_dates
    target_doy = target_dates.dt.dayofyear
    
    # Cyclical day of year: 1st harmonic (annual cycle) & 2nd harmonic (semi-annual cycle)
    for name, values in seasonal_features(target_doy).items():
        df[name] = values
    
    # Baseline 1: Persistence = Max temperature on Day D
    df["baseline_persistence"] = df["temp_lag_0"]
    
    # Baseline 2: Climatology = Historical average for target DOY
    if climatology_map is not None:
        df["baseline_climatology"] = target_doy.map(climatology_map)
        # Fallback for any unmapped
        overall_mean = np.mean(list(climatology_map.values())) if climatology_map else 30.0
        df["baseline_climatology"] = df["baseline_climatology"].fillna(overall_mean)
    else:
        df["baseline_climatology"] = np.nan

    if is_training:
        # Target: Max temperature on D+2
        # In a contiguous daily dataframe, D+2 is shifted by -2
        df["target_temp_d2"] = df["temperature_2m_max"].shift(-FORECAST_HORIZON_DAYS)
        # Drop rows with NaN in features or target
        df = df.dropna(subset=FEATURE_COLUMNS + ["target_temp_d2", "baseline_climatology"]).reset_index(drop=True)
    else:
        # For serving, we only need the latest row (Day D) with valid features
        # Lags require at least 7 days of history
        df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        
    return df
