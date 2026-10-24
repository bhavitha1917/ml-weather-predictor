"""
streamlit_app.py
Alternative pure-Python UI built with Streamlit.
It reuses the FastAPI backend's prediction pipeline (backend.app.predict_city) and what-if
function, so both interfaces show exactly the same numbers from the same code.

Run with:
    streamlit run streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st

import backend.app as api
from backend.live_data import WeatherDataError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

MODEL_LABELS = {
    "ensemble": "Ensemble (main model)",
    "linear_regression": "Ridge regression (linear)",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "persistence": "Persistence baseline",
    "climatology": "Climatology baseline",
}

# Page config
st.set_page_config(
    page_title="ML Weather Predictor",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Glass CSS injecting blueprint palette
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background-color: #0B132B;
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
    }

    header[data-testid="stHeader"] {
        background-color: transparent;
    }

    .glass-card {
        background: rgba(28, 37, 65, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(163, 177, 198, 0.2);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .hero-temp {
        font-size: 4.2rem;
        font-weight: 300;
        color: #FFFFFF;
        line-height: 1;
    }

    .temp-unit {
        font-size: 2rem;
        color: #4FFBDF;
    }

    .accent-cyan {
        color: #4FFBDF;
        font-weight: 600;
    }

    .muted-text {
        color: #A3B1C6;
        font-size: 0.88rem;
    }

    .interval-box {
        background: rgba(11, 19, 43, 0.6);
        border: 1px solid rgba(79, 251, 223, 0.3);
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_backend_artifacts():
    api.load_artifacts()
    return True


load_backend_artifacts()


def fmt(value, unit=""):
    return "unavailable" if value is None else f"{value}{unit}"


def get_prediction(city_key):
    """Live prediction from the shared backend pipeline, or the labelled demo snapshot."""
    if not api.city_is_ready(city_key):
        return None, f"Model artifacts for {city_key.capitalize()} are not loaded: {api.ARTIFACT_ERRORS.get(city_key)}"
    try:
        return api.predict_city(city_key), None
    except WeatherDataError as e:
        try:
            return api.demo_fallback(city_key, e), None
        except Exception:
            return None, f"Live weather data unavailable ({e.reason}) and no demo snapshot exists."


# Sidebar
st.sidebar.markdown("<h2 style='color: #4FFBDF;'>ML Weather Predictor</h2>", unsafe_allow_html=True)
selected_city = st.sidebar.selectbox("Select Target City", ["Bengaluru", "Delhi", "Mumbai"], index=0)
city_key = selected_city.lower()
active_tab = st.sidebar.radio("Navigation", ["Dashboard", "What-If Simulator (illustrative)", "Exploratory Data Analysis"])

st.markdown("<h1 style='color: #FFFFFF; font-weight: 600; margin-bottom: 0;'>Tomorrow's Maximum Temperature Forecast</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #A3B1C6; margin-top: 2px;'>Trained on 2019–2022 daily data • Evaluated on the held-out year 2023 against two baselines</p>", unsafe_allow_html=True)

if active_tab == "Dashboard":
    res, error = get_prediction(city_key)
    if error:
        st.error(error)
        st.stop()
    if res["is_demo_fallback"]:
        st.warning(res["fallback_message"])

    interval = res["interval_80"]
    coverage = interval.get("test_coverage_pct")
    col1, col2 = st.columns([7, 5])
    with col1:
        st.markdown(f"""
        <div class='glass-card'>
            <div style='display:flex; justify-content:space-between;'>
                <span class='accent-cyan' style='font-size:0.75rem; letter-spacing:0.08em;'>ENSEMBLE PREDICTION</span>
                <span class='muted-text'>Target: {res['target_date']}</span>
            </div>
            <div style='margin-top:10px;'>
                <span class='muted-text' style='font-size:1.1rem;'>{selected_city}</span><br>
                <span class='hero-temp'>{res['predicted_temp']:.1f}</span><span class='temp-unit'>°C</span>
            </div>
            <div class='interval-box'>
                <div style='display:flex; justify-content:space-between;'>
                    <strong style='font-size:0.85rem;'>80% prediction interval</strong>
                    <span class='accent-cyan'>[{interval['lower']:.1f}°C, {interval['upper']:.1f}°C]</span>
                </div>
                <div class='muted-text' style='font-size:0.8rem; margin-top:3px;'>
                    Measured coverage on the 2023 test year: {fmt(coverage, '%')} of {fmt(interval.get('test_days'))} days
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        cc = res["current_conditions"]
        st.markdown(f"""
        <div class='glass-card'>
            <div style='display:flex; justify-content:space-between;'>
                <span class='accent-cyan' style='font-size:0.75rem; letter-spacing:0.08em;'>CURRENT CONDITIONS (OPEN-METEO)</span>
                <span class='muted-text'>Last completed day D: {res['last_completed_date']}</span>
            </div>
            <div class='muted-text' style='margin-top:12px; line-height:1.9;'>
                Temperature: <strong>{fmt(cc['temperature'], ' °C')}</strong><br>
                Humidity: <strong>{fmt(cc['humidity'], ' %')}</strong><br>
                Wind speed: <strong>{fmt(cc['wind_speed'], ' km/h')}</strong><br>
                Dew point: <strong>{fmt(cc['dew_point'], ' °C')}</strong><br>
                Official forecast for {res['official_forecast']['date']} (reference only):
                <strong>{fmt(res['official_forecast']['temperature_max'], ' °C')}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    metrics = api.get_metrics(selected_city)
    overall = metrics["overall"]
    today = {**res["model_breakdown"], "persistence": res["persistence_baseline"], "climatology": res["climatology_baseline"]}
    st.markdown("### Test-year (2023) errors, held out from training and model selection")
    st.dataframe(pd.DataFrame([
        {"Model / method": MODEL_LABELS[m], "Prediction for target date (°C)": today[m],
         "Test MAE (°C)": overall[m]["mae"], "Test RMSE (°C)": overall[m]["rmse"]}
        for m in MODEL_LABELS
    ]), width="stretch", hide_index=True)
    beats = metrics.get("main_model_beats") or {}
    if beats:
        st.caption(
            f"Ensemble beats persistence on MAE: {beats['persistence']['mae']}, on RMSE: {beats['persistence']['rmse']}. "
            f"Lowest test MAE in this city: {MODEL_LABELS.get(metrics.get('best_model_by_test_mae'), '—')}."
        )

    st.markdown("### Model inputs: the 7 completed days used")
    st.dataframe(pd.DataFrame(res["input_window"]), width="stretch", hide_index=True)

elif active_tab == "What-If Simulator (illustrative)":
    st.markdown("## What-If Simulator (illustrative, not a forecast)")
    st.caption("The 7-day history is generated from these sliders with a fixed pattern and no rain, "
               "then passed to the trained models to show their sensitivity.")
    sim_col1, sim_col2 = st.columns([6, 6])
    with sim_col1:
        s_temp = st.slider("Max temperature on day D (°C)", 15.0, 46.0, 30.0, 0.5)
        s_rh = st.slider("Mean relative humidity (%)", 10, 100, 55, 1)
        s_wind = st.slider("Max wind speed (km/h)", 0, 45, 12, 1)
        s_mom = st.slider("Change from previous day (°C/day)", -5.0, 5.0, 0.5, 0.5)
    sim = api.simulate_scenario(city=selected_city, temp_yesterday=s_temp, humidity=float(s_rh),
                                wind_speed=float(s_wind), temp_momentum=s_mom)
    with sim_col2:
        st.markdown(f"""
        <div class='glass-card' style='text-align:center;'>
            <div style='font-size:0.85rem; color:#A3B1C6;'>ENSEMBLE OUTPUT FOR SYNTHETIC INPUTS</div>
            <div style='font-size:3.5rem; font-weight:300; color:#FFFFFF;'>{sim['predictions']['ensemble']:.1f}<span class='temp-unit'>°C</span></div>
            <div class='muted-text'>Random Forest {sim['predictions']['random_forest']:.1f} °C • LightGBM {sim['predictions']['lightgbm']:.1f} °C</div>
        </div>
        """, unsafe_allow_html=True)

elif active_tab == "Exploratory Data Analysis":
    st.markdown("## Exploratory Data Analysis (EDA Figures)")
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(os.path.join(FIGURES_DIR, "1_distributions.png"), caption="1. Distributions of each variable")
        st.image(os.path.join(FIGURES_DIR, "3_correlation_heatmaps.png"), caption="3. Same-day Pearson correlations")
        st.image(os.path.join(FIGURES_DIR, "5_city_comparison.png"), caption="5. City comparison (mean, range, hot days)")
    with col_b:
        st.image(os.path.join(FIGURES_DIR, "2_seasonal_trends.png"), caption="2. Max temperature by season")
        st.image(os.path.join(FIGURES_DIR, "4_autocorrelation.png"), caption="4. Raw autocorrelation (dominated by the seasonal cycle)")
        st.image(os.path.join(FIGURES_DIR, "6_memory_diagnostics.png"), caption="6. Anomaly autocorrelation and partial autocorrelation")
    st.image(os.path.join(FIGURES_DIR, f"residuals_{city_key}.png"), caption=f"Ensemble residuals on the 2023 test year ({selected_city})")
