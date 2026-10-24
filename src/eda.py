"""
src/eda.py
Performs Exploratory Data Analysis (EDA) on 5 years of daily weather data
for Bengaluru, Delhi, and Mumbai according to Blueprint Prompt 1.
Saves all figures to figures/ and numerical summaries to figures/eda_summary.json.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# statsmodels not needed; pandas autocorr is used directly


# Configure Matplotlib styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CITIES = ["bengaluru", "delhi", "mumbai"]

def assign_season(month: int) -> str:
    """
    Standard Indian Meteorological Department (IMD) seasons:
    - Winter: Dec, Jan, Feb
    - Pre-monsoon (Summer): Mar, Apr, May
    - Monsoon: Jun, Jul, Aug, Sep
    - Post-monsoon (Autumn): Oct, Nov
    """
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Pre-monsoon"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-monsoon"

def load_data():
    city_dfs = {}
    for city in CITIES:
        csv_path = os.path.join(DATA_DIR, f"{city}.csv")
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        # Missing values and duplicates check
        missing = df.isnull().sum().to_dict()
        duplicates = df.duplicated(subset=["date"]).sum()
        
        # Linear interpolation if any missing (standard time series handling)
        if df.isnull().values.any():
            df = df.interpolate(method="time").ffill().bfill()
            
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        df["season"] = df["month"].apply(assign_season)
        
        city_dfs[city] = {
            "df": df,
            "missing": missing,
            "duplicates": int(duplicates)
        }
    return city_dfs

def run_eda():
    data = load_data()
    summary_report = {}
    
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS (PROMPT 1)")
    print("=" * 60)
    
    # 1. Summary Statistics & Missing Values
    for city, item in data.items():
        df = item["df"]
        print(f"\n--- {city.upper()} ---")
        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()} ({len(df)} days)")
        print(f"Duplicates: {item['duplicates']}")
        print(f"Missing values: {item['missing']}")
        stats = df[["temperature_2m_max", "relative_humidity_2m_mean", "wind_speed_10m_max", "precipitation_sum"]].describe().to_dict()
        summary_report[city] = {
            "missing": item["missing"],
            "duplicates": item["duplicates"],
            "stats": stats,
            "mean_temp": float(df["temperature_2m_max"].mean()),
            "std_temp": float(df["temperature_2m_max"].std()),
            "min_temp": float(df["temperature_2m_max"].min()),
            "max_temp": float(df["temperature_2m_max"].max()),
            "temp_range": float(df["temperature_2m_max"].max() - df["temperature_2m_max"].min()),
            "hot_days_35": int((df["temperature_2m_max"] >= 35).sum() / 5), # avg per year
            "hot_days_40": int((df["temperature_2m_max"] >= 40).sum() / 5)
        }
    
    # --- FIGURE 1: Distributions ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    variables = [
        ("temperature_2m_max", "Max Temperature (°C)", axes[0, 0]),
        ("relative_humidity_2m_mean", "Mean Relative Humidity (%)", axes[0, 1]),
        ("wind_speed_10m_max", "Max Wind Speed (km/h)", axes[1, 0]),
        ("precipitation_sum", "Precipitation Sum (mm)", axes[1, 1])
    ]
    
    colors = {"bengaluru": "#10B981", "delhi": "#EF4444", "mumbai": "#3B82F6"}
    for var, label, ax in variables:
        for city in CITIES:
            df = data[city]["df"]
            if var == "precipitation_sum":
                # Log scale or clip for visibility due to zero-inflation
                sns.kdeplot(df[df[var] > 0.1][var], ax=ax, label=f"{city.capitalize()} (>0.1mm)", color=colors[city], lw=2)
            else:
                sns.kdeplot(df[var], ax=ax, label=city.capitalize(), color=colors[city], lw=2)
        ax.set_title(f"Distribution of {label}", fontsize=13, fontweight="bold")
        ax.set_xlabel(label)
        ax.legend()
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "1_distributions.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    # --- FIGURE 2: Seasonal Temperature Trends ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    season_order = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
    palette = ["#60A5FA", "#F59E0B", "#10B981", "#8B5CF6"]
    
    for idx, city in enumerate(CITIES):
        df = data[city]["df"]
        sns.boxplot(x="season", y="temperature_2m_max", hue="season", data=df, order=season_order, palette=palette, legend=False, ax=axes[idx])
        axes[idx].set_title(f"{city.capitalize()} Seasonal Max Temperature", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Season")
        axes[idx].set_ylabel("Max Temperature (°C)" if idx == 0 else "")
        axes[idx].tick_params(axis='x', rotation=15)
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "2_seasonal_trends.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    # --- FIGURE 3: Correlation Heatmaps ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    features = ["temperature_2m_max", "relative_humidity_2m_mean", "wind_speed_10m_max", "precipitation_sum"]
    rename_cols = {"temperature_2m_max": "Max Temp", "relative_humidity_2m_mean": "Humidity", 
                   "wind_speed_10m_max": "Wind Speed", "precipitation_sum": "Precip"}
    
    for idx, city in enumerate(CITIES):
        df = data[city]["df"][features].rename(columns=rename_cols)
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", ax=axes[idx], cbar=(idx==2))
        axes[idx].set_title(f"{city.capitalize()} Feature Correlation", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "3_correlation_heatmaps.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")

    # --- FIGURE 4: Autocorrelation of Max Temperature ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, city in enumerate(CITIES):
        df = data[city]["df"]
        # Compute autocorrelation up to lag 30
        lags = range(1, 31)
        autocorr = [df["temperature_2m_max"].autocorr(lag=l) for l in lags]
        axes[idx].stem(lags, autocorr, linefmt='b-', markerfmt='bo', basefmt='r-')
        axes[idx].axvline(7, color="orange", linestyle="--", label="Lag 7 cutoff")
        axes[idx].axhline(0, color="gray", linestyle="-", alpha=0.5)
        axes[idx].set_title(f"{city.capitalize()} Max Temp Autocorrelation", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Lag (Days)")
        axes[idx].set_ylabel("Autocorrelation" if idx == 0 else "")
        axes[idx].legend()
    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "4_autocorrelation.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved: {fig4_path}")

    # --- FIGURE 5: City Comparison (Mean, Range, Hot Days) ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    city_names = [c.capitalize() for c in CITIES]
    means = [summary_report[c]["mean_temp"] for c in CITIES]
    ranges = [summary_report[c]["temp_range"] for c in CITIES]
    hot_35 = [summary_report[c]["hot_days_35"] for c in CITIES]
    
    bars1 = axes[0].bar(city_names, means, color=["#10B981", "#EF4444", "#3B82F6"])
    axes[0].set_title("Mean Max Temperature (°C)", fontweight="bold")
    axes[0].set_ylim(0, 40)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f}°C", ha='center', va='bottom', fontweight='bold')

    bars2 = axes[1].bar(city_names, ranges, color=["#10B981", "#EF4444", "#3B82F6"])
    axes[1].set_title("Temperature Range (Max - Min) (°C)", fontweight="bold")
    axes[1].set_ylim(0, 45)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f}°C", ha='center', va='bottom', fontweight='bold')

    bars3 = axes[2].bar(city_names, hot_35, color=["#10B981", "#EF4444", "#3B82F6"])
    axes[2].set_title("Avg Hot Days (≥35°C) Per Year", fontweight="bold")
    axes[2].set_ylim(0, max(hot_35) + 30)
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2, yval + 2, f"{int(yval)} d/yr", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    fig5_path = os.path.join(FIGURES_DIR, "5_city_comparison.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Saved: {fig5_path}")

    # Save summary report JSON
    summary_json_path = os.path.join(FIGURES_DIR, "eda_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"Saved summary report to {summary_json_path}")

    print("\n" + "=" * 60)
    print("PROMPT 1 CHECK QUESTIONS (computed values above; interpretation in the notebook):")
    print("1. Why are 5 years enough or not?")
    print("   -> 5 years give 4 full training years plus 1 test year, enough to learn the annual cycle,")
    print("      but each season is seen only 4 times, so year-to-year variability is poorly sampled.")
    print("2. What does the heatmap tell you?")
    print("   -> Same-day Pearson correlations only: humidity is clearly negatively correlated with max temperature")
    print("      in every city; precipitation is negative in Bengaluru and Mumbai but close to zero in Delhi.")
    print("      Correlation is not causation, and the heatmap does not show next-day predictive power.")
    print("3. What does autocorrelation decide?")
    print("   -> The raw autocorrelation stays high for weeks because of the seasonal cycle, so it cannot pick")
    print("      a lag window by itself. See plot_memory_diagnostics(): after removing the day-of-year mean, the")
    print("      autocorrelation falls to roughly 0.2-0.4 by lag 7, and the partial autocorrelation is dominated")
    print("      by lag 1. A 7-day window is therefore a reasonable choice, not a sharp cutoff.")
    print("=" * 60)


def climatology_anomaly(df: pd.DataFrame) -> pd.Series:
    """Max temperature minus its day-of-year mean (removes the seasonal cycle)."""
    doy = pd.to_datetime(df["date"]).dt.dayofyear
    return df["temperature_2m_max"] - df.groupby(doy)["temperature_2m_max"].transform("mean")


def partial_autocorrelation(series, max_lag: int) -> np.ndarray:
    """
    Partial autocorrelation at lags 1..max_lag: the last coefficient of an ordinary least-squares
    regression of x_t on x_{t-1} .. x_{t-k}, i.e. the extra information lag k adds.
    """
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    values = []
    for k in range(1, max_lag + 1):
        lags = np.column_stack([x[k - j - 1: len(x) - j - 1] for j in range(k)])
        design = np.column_stack([np.ones(len(x) - k), lags])
        values.append(np.linalg.lstsq(design, x[k:], rcond=None)[0][-1])
    return np.array(values)


def plot_memory_diagnostics(max_lag: int = 21) -> str:
    """
    Figure 6: autocorrelation of the seasonal anomaly and partial autocorrelation of max temperature.
    Saved as a separate file so the original five EDA figures are not regenerated.
    """
    data = load_data()
    lags = np.arange(1, max_lag + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    colors = {"bengaluru": "#10B981", "delhi": "#EF4444", "mumbai": "#3B82F6"}
    for city in CITIES:
        df = data[city]["df"]
        anomaly = climatology_anomaly(df)
        axes[0].plot(lags, [anomaly.autocorr(lag=int(l)) for l in lags], "o-", ms=4, color=colors[city], label=city.capitalize())
        axes[1].plot(lags, partial_autocorrelation(df["temperature_2m_max"], max_lag), "o-", ms=4, color=colors[city], label=city.capitalize())
    axes[0].set_title("Autocorrelation of max-temperature anomaly\n(day-of-year mean removed)", fontweight="bold")
    axes[1].set_title("Partial autocorrelation of max temperature", fontweight="bold")
    for ax in axes:
        ax.axvline(7, color="orange", linestyle="--", label="Lag 7")
        ax.axhline(0, color="gray", lw=1)
        ax.set_xlabel("Lag (days)")
        ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "6_memory_diagnostics.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


if __name__ == "__main__":
    run_eda()
