"""
backend/app.py
FastAPI application for the ML Weather Predictor.
- Endpoints: /health, /cities, /predict, /metrics, /simulate
- Serves the static frontend (/ and /static) and the saved figures (/figures)

Every number in a /predict response is either an Open-Meteo value, a trained-model output,
a baseline, or a saved test-year statistic. Nothing is invented: when real data is missing
the field is null, and when the weather service fails the response is an explicitly
labelled demo snapshot (is_demo_fallback=true) captured earlier from a real run.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import json
import logging
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.features import make_features, seasonal_features, FEATURE_COLUMNS
from src.data_fetch import CITIES
from src.ensemble import StackingEnsemble  # noqa: F401  (needed to unpickle *_ensemble.joblib)
from src.intervals import NOMINAL_COVERAGE, interval_bounds
from backend.live_data import (
    IST,
    LIVE_FORECAST_DAYS,
    LIVE_PAST_DAYS,
    LiveWeatherCache,
    WeatherDataError,
    build_observation_window,
    fetch_live_recent_weather,
    forecast_dates,
    ist_now,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

logger = logging.getLogger("weather_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

CITY_KEYS = ["bengaluru", "delhi", "mumbai"]
REQUIRED_MODELS = ["ensemble", "rf", "lgb", "lr"]

MODELS = {}           # city_key -> {"ensemble": ..., "rf": ..., "lgb": ..., "lr": ...}
CLIMATOLOGY = {}      # city_key -> {doy: mean max temp from training years}
METRICS = {}          # city_key -> saved test-year metrics (models/metrics.json)
EVALUATION = {}       # city_key -> verified re-evaluation (models/evaluation_report.json, src/evaluate.py)
FALLBACK_SAMPLES = {}  # city_key -> demo snapshot captured from a real run
ARTIFACT_ERRORS = {}  # city_key -> list of problems found while loading
LIVE_WEATHER_CACHE = LiveWeatherCache()  # validated Open-Meteo responses (30 min) + HTTP 429 cooldown

FEATURE_DESCRIPTIONS = {
    "temp_lag_0": ("Max temperature on day D", "Most recent completed day (persistence signal)"),
    "temp_lag_1": ("Max temperature on day D-1", "Short-term persistence"),
    "temp_lag_2": ("Max temperature on day D-2", "Short-term persistence"),
    "temp_lag_3": ("Max temperature on day D-3", "Recent temperature history"),
    "temp_lag_4": ("Max temperature on day D-4", "Recent temperature history"),
    "temp_lag_5": ("Max temperature on day D-5", "Recent temperature history"),
    "temp_lag_6": ("Max temperature on day D-6", "Recent temperature history"),
    "temp_roll_mean_7": ("7-day mean max temperature", "Level of the past week"),
    "temp_roll_std_7": ("7-day std of max temperature", "Variability of the past week"),
    "temp_momentum_1": ("Change D-1 to D", "Warming or cooling over the last day"),
    "temp_accel_1": ("Change in daily change", "Whether warming/cooling is speeding up"),
    "humidity_lag_0": ("Mean humidity on day D", "Moist air and clouds tend to lower max temperature"),
    "humidity_lag_1": ("Mean humidity on day D-1", "Persistence of moisture"),
    "dew_point_lag_0": ("Approx. dew point on day D", "Rough moisture index from max temperature and mean humidity"),
    "wind_speed_lag_0": ("Max wind speed on day D", "Mixing and air-mass change"),
    "precip_lag_0": ("Rain on day D", "Rain and cloud tend to lower max temperature"),
    "precip_lag_1": ("Rain on day D-1", "Wet ground and cloud persistence"),
    "precip_sum_3d": ("Rain over D-2 to D", "Recent wet spell"),
    "sin_doy": ("Season (sine of target day-of-year)", "Annual cycle"),
    "cos_doy": ("Season (cosine of target day-of-year)", "Annual cycle"),
    "sin_doy_semi": ("Semi-annual cycle (sine)", "Second harmonic of the annual cycle"),
    "cos_doy_semi": ("Semi-annual cycle (cosine)", "Second harmonic of the annual cycle"),
}


def load_artifacts():
    """Load serialized models, climatology maps, test metrics and demo snapshots once at startup."""
    for store in (MODELS, CLIMATOLOGY, METRICS, EVALUATION, FALLBACK_SAMPLES, ARTIFACT_ERRORS):
        store.clear()

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    try:
        with open(metrics_path, "r") as f:
            METRICS.update(json.load(f))
    except (OSError, ValueError) as e:
        logger.error("Could not load %s: %s", metrics_path, e)

    report_path = os.path.join(MODELS_DIR, "evaluation_report.json")
    try:
        with open(report_path, "r") as f:
            EVALUATION.update(json.load(f))
    except (OSError, ValueError) as e:
        logger.warning("No evaluation report (%s); run `python -m src.evaluate`. Coverage will be reported as null.", e)

    for city_key in CITY_KEYS:
        errors = []
        city_models = {}
        for name in REQUIRED_MODELS:
            path = os.path.join(MODELS_DIR, f"{city_key}_{name}.joblib")
            if not os.path.exists(path):
                errors.append(f"missing {os.path.basename(path)}")
                continue
            try:
                city_models[name] = joblib.load(path)
            except Exception as e:  # e.g. lightgbm not installed, incompatible scikit-learn
                errors.append(f"cannot load {os.path.basename(path)}: {type(e).__name__}: {e}")

        clim_path = os.path.join(MODELS_DIR, f"{city_key}_climatology.json")
        try:
            with open(clim_path, "r") as f:
                CLIMATOLOGY[city_key] = {int(k): v for k, v in json.load(f).items()}
        except (OSError, ValueError) as e:
            errors.append(f"cannot load {os.path.basename(clim_path)}: {e}")

        if "interval_80" not in METRICS.get(city_key, {}):
            errors.append("metrics.json has no interval_80 for this city")

        MODELS[city_key] = city_models
        if errors:
            ARTIFACT_ERRORS[city_key] = errors
            logger.error("Artifacts for %s are incomplete: %s", city_key, "; ".join(errors))
        else:
            logger.info("Loaded models for %s: %s", city_key, ", ".join(sorted(city_models)))

    samples_path = os.path.join(BACKEND_DIR, "sample_responses.json")
    try:
        with open(samples_path, "r") as f:
            FALLBACK_SAMPLES.update(json.load(f))
        logger.info("Loaded demo snapshots for %s", ", ".join(sorted(FALLBACK_SAMPLES)))
    except (OSError, ValueError) as e:
        logger.warning("No demo snapshots available (%s); weather-service failures will return 503", e)


def city_is_ready(city_key: str) -> bool:
    return city_key in MODELS and not ARTIFACT_ERRORS.get(city_key)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield

app = FastAPI(
    title="ML Weather Predictor API",
    description="Predicts tomorrow's maximum temperature (D+2 from the last completed day) "
                "for Bengaluru, Delhi and Mumbai, with baselines and an empirical 80% interval.",
    version="3.6.0",
    lifespan=lifespan
)

# The dashboard is served by this app (same origin), so CORS is off by default.
# If the frontend is ever hosted elsewhere, list its origin(s) in ALLOWED_ORIGINS,
# e.g. ALLOWED_ORIGINS="https://example.github.io". No credentials are used.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )


def resolve_city(city: str) -> str:
    city_key = city.strip().lower()
    if city_key not in CITY_KEYS:
        raise HTTPException(
            status_code=400,
            detail=f"City '{city}' is not supported. Please choose from Bengaluru, Delhi, or Mumbai."
        )
    return city_key


def require_ready(city_key: str):
    if not city_is_ready(city_key):
        raise HTTPException(status_code=503, detail=f"Model artifacts for {city_key.capitalize()} are not loaded on the server.")


@app.get("/health")
def health_check():
    """Service health check. Returns 503 if any city's artifacts failed to load."""
    ready = [c for c in CITY_KEYS if city_is_ready(c)]
    body = {
        "status": "healthy" if len(ready) == len(CITY_KEYS) else "degraded",
        "timestamp_ist": ist_now().isoformat(timespec="seconds"),
        "loaded_cities": ready,
        "artifact_errors": ARTIFACT_ERRORS,
        "demo_snapshots_available": sorted(FALLBACK_SAMPLES),
    }
    if len(ready) != len(CITY_KEYS):
        raise HTTPException(status_code=503, detail=body)
    return body


@app.get("/cities")
def get_cities():
    """Returns the list of supported trained cities."""
    return ["Bengaluru", "Delhi", "Mumbai"]


def compute_feature_contributions(lgb_model, X_live, feature_names, feat_row):
    """Exact TreeSHAP contributions of the LightGBM component model for one input row."""
    try:
        contribs = lgb_model.booster_.predict(X_live, pred_contrib=True)[0]
    except Exception:
        logger.exception("TreeSHAP computation failed")
        return {"base_value": None, "top_drivers": []}
    items = []
    for name, c_val in zip(feature_names, contribs[:-1]):
        label, explanation = FEATURE_DESCRIPTIONS.get(name, (name, ""))
        items.append({
            "feature": name,
            "label": label,
            "value": round(float(feat_row[name].values[0]), 1),
            "contribution": round(float(c_val), 2),
            "direction": "warming" if c_val >= 0 else "cooling",
            "explanation": explanation
        })
    items.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return {"base_value": round(float(contribs[-1]), 1), "top_drivers": items[:6]}


def _value_or_none(value):
    return None if value is None or (isinstance(value, float) and np.isnan(value)) else value


def predict_city(city_key: str, now=None, raw_data=None) -> dict:
    """
    Run the live forecasting pipeline for one city.
    Raises WeatherDataError if the weather service fails or its data cannot support a forecast.
    """
    city_models = MODELS[city_key]
    clim_map = CLIMATOLOGY[city_key]
    city_metrics = METRICS[city_key]
    p10_offset = city_metrics["interval_80"]["p10"]
    p90_offset = city_metrics["interval_80"]["p90"]

    # D and the target date come from the Indian calendar, never the server clock
    now = now or ist_now()
    last_completed, target = forecast_dates(now)
    target_date = target.isoformat()

    weather_fetched_at, from_cache = now, False
    if raw_data is None:
        coords = CITIES[city_key.capitalize()]
        raw_data, weather_fetched_at, from_cache = LIVE_WEATHER_CACHE.get(
            (city_key, last_completed.isoformat(), LIVE_PAST_DAYS, LIVE_FORECAST_DAYS),
            fetch=lambda: fetch_live_recent_weather(coords, past_days=LIVE_PAST_DAYS, forecast_days=LIVE_FORECAST_DAYS),
            validate=lambda raw: build_observation_window(raw, last_completed),
            now_ist=now,
        )

    # Validated, consecutive completed days D-6 .. D (raises WeatherDataError otherwise);
    # cached responses go through exactly the same check
    window_df = build_observation_window(raw_data, last_completed)

    # Build features strictly up to Day D with the same function used in training
    feat_df = make_features(window_df, climatology_map=clim_map, is_training=False)
    latest_feat_row = feat_df.iloc[[-1]]
    # The feature row must describe exactly day D -> target; never a silently older day
    if len(feat_df) != 1 or latest_feat_row["date"].iloc[0].date() != last_completed \
            or latest_feat_row["target_date"].iloc[0].date() != target:
        raise RuntimeError("Feature row does not match the forecast origin D")
    X_live = latest_feat_row[FEATURE_COLUMNS].values

    ens_pred = float(city_models["ensemble"].predict(X_live)[0])
    predicted_val = round(ens_pred, 1)
    # Same function as the offline coverage evaluation (src/evaluate.py)
    lower_bound, upper_bound = interval_bounds(ens_pred, p10_offset, p90_offset)
    served_eval = EVALUATION.get(city_key, {}).get("served_interval", {})

    # Official forecast: Open-Meteo's own forecast for exactly the target date, or null
    daily = raw_data["daily"]
    official_temp = None
    if target_date in daily["time"]:
        official_temp = _value_or_none(daily["temperature_2m_max"][daily["time"].index(target_date)])

    # Current conditions: real API values only; missing values stay null
    current = raw_data.get("current") if isinstance(raw_data.get("current"), dict) else {}
    current_conditions = {
        "observed_at": current.get("time"),
        "temperature": _value_or_none(current.get("temperature_2m")),
        "humidity": _value_or_none(current.get("relative_humidity_2m")),
        "wind_speed": _value_or_none(current.get("wind_speed_10m")),
        "dew_point": _value_or_none(current.get("dew_point_2m")),
    }

    shap_explanation = compute_feature_contributions(city_models["lgb"], X_live, FEATURE_COLUMNS, latest_feat_row)
    shap_explanation["model"] = "lightgbm"
    shap_explanation["ensemble_weight"] = city_metrics.get("ensemble_weights", [None])[0]

    input_window = [
        {"date": row["date"].date().isoformat(),
         "temperature_2m_max": float(row["temperature_2m_max"]),
         "relative_humidity_2m_mean": float(row["relative_humidity_2m_mean"]),
         "wind_speed_10m_max": float(row["wind_speed_10m_max"]),
         "precipitation_sum": float(row["precipitation_sum"])}
        for _, row in window_df.iterrows()
    ]

    return {
        "city": city_key.capitalize(),
        "target_date": target_date,
        "last_completed_date": last_completed.isoformat(),
        "generated_at": now.isoformat(timespec="seconds"),
        # "live_cached": the same validated Open-Meteo data, reused for up to 30 minutes
        "data_source": "live_cached" if from_cache else "live",
        "weather_data_fetched_at": weather_fetched_at.astimezone(IST).isoformat(timespec="seconds"),
        "predicted_temp": predicted_val,
        "model_breakdown": {
            "ensemble": predicted_val,
            "lightgbm": round(float(city_models["lgb"].predict(X_live)[0]), 1),
            "random_forest": round(float(city_models["rf"].predict(X_live)[0]), 1),
            "linear_regression": round(float(city_models["lr"].predict(X_live)[0]), 1)
        },
        "interval_80": {
            "lower": lower_bound,
            "upper": upper_bound,
            "p10_offset": p10_offset,
            "p90_offset": p90_offset,
            "method": "ensemble prediction + 10th/90th percentile of out-of-fold training errors",
            "nominal_coverage_pct": NOMINAL_COVERAGE * 100,
            "test_coverage_pct": served_eval.get("test_coverage_pct"),
            "test_days": EVALUATION.get(city_key, {}).get("n_test")
        },
        "persistence_baseline": round(float(latest_feat_row["baseline_persistence"].values[0]), 1),
        "climatology_baseline": round(float(latest_feat_row["baseline_climatology"].values[0]), 1),
        "current_conditions": current_conditions,
        "official_forecast": {
            "temperature_max": official_temp,
            "date": target_date,
            "available": official_temp is not None,
            "note": "Open-Meteo numerical weather forecast for the target date (reference only)"
        },
        "explainable_ai": shap_explanation,
        "input_window": input_window,
        "is_demo_fallback": False
    }


def demo_fallback(city_key: str, error: WeatherDataError) -> dict:
    """A per-request copy of the saved demo snapshot, explicitly labelled and never re-dated."""
    snapshot = FALLBACK_SAMPLES.get(city_key)
    if not snapshot:
        raise HTTPException(status_code=503, detail=f"Live weather data unavailable ({error.reason}) and no demo snapshot exists.")
    response = copy.deepcopy(snapshot)
    response["is_demo_fallback"] = True
    response["data_source"] = "demo_snapshot"
    response["fallback_reason"] = error.reason
    response["fallback_message"] = (
        "Live weather data could not be used, so this is a saved snapshot from an earlier real run "
        f"(captured {response.get('generated_at', 'at an unknown time')}). It is not a forecast for today."
    )
    return response


@app.get("/predict")
def predict_temperature(city: str = Query(..., description="City name: Bengaluru, Delhi, or Mumbai")):
    """
    Predicts tomorrow's maximum temperature (D+2 from the last completed day in India)
    with the ensemble, its 80% range, both baselines, current conditions and the
    official forecast as a separate reference.
    """
    city_key = resolve_city(city)
    require_ready(city_key)
    try:
        return predict_city(city_key)
    except WeatherDataError as e:
        logger.warning("Live data unusable for %s (%s): %s", city_key, e.reason, e)
        return demo_fallback(city_key, e)
    except Exception:
        logger.exception("Prediction pipeline failed for %s", city_key)
        raise HTTPException(status_code=500, detail="Internal prediction error. See server logs.")


@app.get("/simulate")
def simulate_scenario(
    city: str = Query("Bengaluru"),
    temp_yesterday: float = Query(28.0),
    humidity: float = Query(60.0),
    wind_speed: float = Query(12.0),
    temp_momentum: float = Query(0.0)
):
    """
    Illustrative what-if tool: feeds a synthetic, hand-constructed feature vector to the
    trained models. The inputs are not real observations and the output is not a forecast.
    """
    city_key = resolve_city(city)
    require_ready(city_key)
    city_models = MODELS[city_key]

    # Season features use the same target date (tomorrow in India) as a real forecast
    _, target = forecast_dates(ist_now())
    doy = target.timetuple().tm_yday
    dew_point = temp_yesterday - ((100.0 - humidity) / 5.0)

    feat_dict = {
        "temp_lag_0": temp_yesterday,
        "temp_lag_1": temp_yesterday - temp_momentum,
        "temp_lag_2": temp_yesterday - (2 * temp_momentum),
        "temp_lag_3": temp_yesterday - (2.5 * temp_momentum),
        "temp_lag_4": temp_yesterday - (3 * temp_momentum),
        "temp_lag_5": temp_yesterday - (3.2 * temp_momentum),
        "temp_lag_6": temp_yesterday - (3.5 * temp_momentum),
        "temp_roll_mean_7": temp_yesterday - (1.5 * temp_momentum),
        "temp_roll_std_7": abs(temp_momentum) + 0.8,
        "temp_momentum_1": temp_momentum,
        "temp_accel_1": temp_momentum * 0.2,
        "humidity_lag_0": humidity,
        "humidity_lag_1": humidity,
        "dew_point_lag_0": dew_point,
        "wind_speed_lag_0": wind_speed,
        "precip_lag_0": 0.0,
        "precip_lag_1": 0.0,
        "precip_sum_3d": 0.0,
        **seasonal_features(doy),
    }

    feat_row = pd.DataFrame([feat_dict])
    X_sim = feat_row[FEATURE_COLUMNS].values
    shap_data = compute_feature_contributions(city_models["lgb"], X_sim, FEATURE_COLUMNS, feat_row)
    shap_data["model"] = "lightgbm"

    return {
        "city": city_key.capitalize(),
        "is_illustrative": True,
        "note": "Synthetic inputs: the 7-day history is generated from the sliders with a fixed "
                "pattern and no rain. Use it to explore model sensitivity, not as a forecast.",
        "target_date": target.isoformat(),
        "simulated_inputs": {
            "temp_yesterday": temp_yesterday,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "temp_momentum": temp_momentum,
            "calculated_dew_point": round(dew_point, 1)
        },
        "predictions": {
            "ensemble": round(float(city_models["ensemble"].predict(X_sim)[0]), 1),
            "lightgbm": round(float(city_models["lgb"].predict(X_sim)[0]), 1),
            "random_forest": round(float(city_models["rf"].predict(X_sim)[0]), 1)
        },
        "explainable_ai": shap_data
    }


@app.get("/metrics")
def get_metrics(city: str = Query(..., description="City name")):
    """
    Returns the saved held-out test-year (2023) metrics for one city (models/metrics.json),
    plus the verified re-evaluation from src/evaluate.py: coverage of the interval that is
    actually served, validation (training-years) MAE, and which baselines the model beats.
    """
    city_key = resolve_city(city)
    metric_data = METRICS.get(city_key)
    if not metric_data:
        raise HTTPException(status_code=404, detail="Metrics not found.")
    response = copy.deepcopy(metric_data)
    evaluation = EVALUATION.get(city_key, {})
    for key in ("served_interval", "validation", "main_model_beats", "best_model_by_test_mae", "n_test", "test_period"):
        response[key] = copy.deepcopy(evaluation.get(key))
    return response


if os.path.exists(FIGURES_DIR):
    app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_index():
    """Serves the dashboard HTML."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "FastAPI Weather Predictor is running.",
        "endpoints": ["/health", "/cities", "/predict?city=Delhi", "/metrics?city=Delhi", "/simulate", "/figures/"]
    }
