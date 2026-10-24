"""
src/evaluate.py
Re-evaluates the SAVED model artifacts (no retraining, nothing in models/ is overwritten
except evaluation_report.json) to make the reported results reproducible:

1. Rebuilds the features and the chronological split exactly as src/train.py does.
2. Recomputes test-year (2023) MAE/RMSE per model, per city and per season and checks them
   against models/metrics.json.
3. Measures the coverage of the interval the API actually serves (src/intervals.py).
4. Re-runs the 5-fold TimeSeriesSplit on the training years only to report validation MAE
   per model, which is the legitimate basis for choosing the main model (the test year is
   never used for that choice).

Usage:
    python -m src.evaluate
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.features import make_features, FEATURE_COLUMNS
from src.intervals import NOMINAL_COVERAGE, coverage_interval_95, coverage_pct, interval_bounds
from src.train import CITIES, DATA_DIR, MODELS_DIR, TEST_START_DATE, assign_season

REPORT_PATH = os.path.join(MODELS_DIR, "evaluation_report.json")
MODEL_KEYS = {"ensemble": "ensemble", "random_forest": "rf", "lightgbm": "lgb", "linear_regression": "lr"}
SEASONS = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]


def _metrics(y_true, y_pred):
    return {"mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
            "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 3)}


def load_split(city: str):
    """Features and chronological train/test split, identical to src/train.py."""
    df = pd.read_csv(os.path.join(DATA_DIR, f"{city}.csv"))
    with open(os.path.join(MODELS_DIR, f"{city}_climatology.json")) as f:
        clim = {int(k): v for k, v in json.load(f).items()}
    feats = make_features(df, climatology_map=clim, is_training=True)
    feats["season"] = feats["target_date"].dt.month.apply(assign_season)
    test_start = pd.Timestamp(TEST_START_DATE)
    train = feats[feats["target_date"] < test_start].reset_index(drop=True)
    test = feats[feats["target_date"] >= test_start].reset_index(drop=True)
    return train, test


def test_predictions(city: str, test: pd.DataFrame) -> dict:
    X = test[FEATURE_COLUMNS].values
    preds = {name: joblib.load(os.path.join(MODELS_DIR, f"{city}_{key}.joblib")).predict(X)
             for name, key in MODEL_KEYS.items()}
    preds["persistence"] = test["baseline_persistence"].values
    preds["climatology"] = test["baseline_climatology"].values
    return preds


def validation_mae(city: str, train: pd.DataFrame, saved: dict) -> dict:
    """Out-of-fold MAE on the training years (5-fold TimeSeriesSplit, saved hyperparameters)."""
    X, y = train[FEATURE_COLUMNS].values, train["target_temp_d2"].values
    oof = {"rf": [], "lgb": [], "ridge": [], "y": [], "persistence": []}
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=5).split(X):
        oof["rf"] += list(RandomForestRegressor(**saved["best_rf_params"], random_state=42, n_jobs=-1)
                          .fit(X[tr_idx], y[tr_idx]).predict(X[va_idx]))
        oof["lgb"] += list(LGBMRegressor(**saved["best_lgb_params"], random_state=42, verbose=-1)
                           .fit(X[tr_idx], y[tr_idx]).predict(X[va_idx]))
        oof["ridge"] += list(Ridge(alpha=10.0).fit(X[tr_idx], y[tr_idx]).predict(X[va_idx]))
        oof["y"] += list(y[va_idx])
        oof["persistence"] += list(train["baseline_persistence"].values[va_idx])
    o = {k: np.array(v) for k, v in oof.items()}
    w = saved["ensemble_weights"]
    ens = w[0] * o["lgb"] + w[1] * o["rf"] + w[2] * o["ridge"]
    residuals = o["y"] - ens
    return {
        "n_oof": int(len(o["y"])),
        "mae": {
            "ensemble": round(float(np.mean(np.abs(residuals))), 3),
            "random_forest": round(float(np.mean(np.abs(o["y"] - o["rf"]))), 3),
            "lightgbm": round(float(np.mean(np.abs(o["y"] - o["lgb"]))), 3),
            "linear_regression": round(float(np.mean(np.abs(o["y"] - o["ridge"]))), 3),
            "persistence": round(float(np.mean(np.abs(o["y"] - o["persistence"]))), 3),
        },
        "reproduced_p10": round(float(np.percentile(residuals, 10)), 2),
        "reproduced_p90": round(float(np.percentile(residuals, 90)), 2),
    }


def evaluate_city(city: str, saved: dict, with_validation: bool = True) -> dict:
    train, test = load_split(city)
    y = test["target_temp_d2"].values
    preds = test_predictions(city, test)

    overall = {name: _metrics(y, p) for name, p in preds.items()}
    by_season = {}
    for season in SEASONS:
        mask = (test["season"] == season).values
        by_season[season] = {"count": int(mask.sum()), **{name: _metrics(y[mask], p[mask]) for name, p in preds.items()}}

    # Largest difference between recomputed and saved metrics (should be 0)
    diffs = [abs(overall[m][k] - saved["overall"][m][k]) for m in overall for k in ("mae", "rmse")]
    diffs += [abs(by_season[s][m][k] - saved["by_season"][s][m][k]) for s in SEASONS for m in overall for k in ("mae", "rmse")]

    p10, p90 = saved["interval_80"]["p10"], saved["interval_80"]["p90"]
    lower, upper = interval_bounds(preds["ensemble"], p10, p90)
    served_cov = coverage_pct(y, lower, upper)
    season_cov = {s: round(coverage_pct(y[m], lower[m], upper[m]), 1)
                  for s in SEASONS for m in [(test["season"] == s).values]}
    q10 = joblib.load(os.path.join(MODELS_DIR, f"{city}_q10.joblib")).predict(test[FEATURE_COLUMNS].values)
    q90 = joblib.load(os.path.join(MODELS_DIR, f"{city}_q90.joblib")).predict(test[FEATURE_COLUMNS].values)

    main = overall["ensemble"]
    report = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_period": [test["target_date"].min().date().isoformat(), test["target_date"].max().date().isoformat()],
        "overall": overall,
        "by_season": by_season,
        "max_abs_diff_vs_metrics_json": round(float(max(diffs)), 4),
        "main_model": "ensemble",
        "main_model_beats": {
            base: {k: main[k] < overall[base][k] for k in ("mae", "rmse")} for base in ("persistence", "climatology")
        },
        "best_model_by_test_mae": min(overall, key=lambda m: overall[m]["mae"]),
        "served_interval": {
            "method": "ensemble prediction + [p10, p90] of out-of-fold training residuals",
            "p10": p10,
            "p90": p90,
            "nominal_coverage_pct": NOMINAL_COVERAGE * 100,
            "test_coverage_pct": round(served_cov, 1),
            "test_coverage_approx_95ci": coverage_interval_95(served_cov, len(y)),
            "test_coverage_by_season": season_cov,
            "test_mean_residual": round(float(np.mean(y - preds["ensemble"])), 2),
        },
        "quantile_models_reference": {
            "note": "LightGBM quantile models (alpha 0.1/0.9) trained by src/train.py; not served",
            "test_coverage_pct": round(coverage_pct(y, q10, q90), 1),
        },
    }
    if with_validation:
        report["validation"] = validation_mae(city, train, saved)
    return report


def main(with_validation: bool = True):
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        saved_all = json.load(f)
    report = {city: evaluate_city(city, saved_all[city], with_validation) for city in CITIES}
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    for city, r in report.items():
        o, s = r["overall"], r["served_interval"]
        print(f"\n=== {city.capitalize()} (test {r['test_period'][0]} to {r['test_period'][1]}, n={r['n_test']})")
        print(f"  max |recomputed - metrics.json| = {r['max_abs_diff_vs_metrics_json']}")
        for name in ["ensemble", "linear_regression", "random_forest", "lightgbm", "persistence", "climatology"]:
            print(f"  {name:18s} MAE {o[name]['mae']:.3f}  RMSE {o[name]['rmse']:.3f}")
        print(f"  ensemble beats: {r['main_model_beats']}")
        print(f"  served 80% interval coverage {s['test_coverage_pct']}% (approx 95% range {s['test_coverage_approx_95ci']})")
        if "validation" in r:
            print(f"  validation MAE (training years, OOF): {r['validation']['mae']}")
    print(f"\nSaved {REPORT_PATH}")


if __name__ == "__main__":
    main()
