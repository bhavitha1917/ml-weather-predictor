"""
src/train.py
Comprehensive AI Pipeline:
1. Model Training & Tuning with TimeSeriesSplit:
   - LightGBM Regressor (State-of-the-art Gradient Boosted Trees)
   - Random Forest Regressor (Bagged Trees)
   - Ridge / Linear Regression (Regularized Linear Baseline)
2. Stacking Meta-Ensemble:
   - Blends out-of-fold predictions from LightGBM, Random Forest, and Ridge
3. Dynamic Calibrated Uncertainty Quantification:
   - Quantile Gradient Boosting (alpha=0.10, alpha=0.90) for weather-adaptive intervals
   - Out-of-fold empirical percentiles (p10, p90) for honest empirical calibration
4. Explainable AI (XAI):
   - Native TreeSHAP computation extracting exact feature contributions
5. Multi-baseline benchmark on held-out final test year (2023)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from lightgbm import LGBMRegressor

from src.features import make_features, compute_climatology_map, FEATURE_COLUMNS
from src.eda import assign_season
from src.ensemble import StackingEnsemble

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

CITIES = ["bengaluru", "delhi", "mumbai"]
# Rows whose TARGET date falls on or after this date form the untouched final test year
TEST_START_DATE = "2023-01-01"

def evaluate_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    return {"mae": round(mae, 3), "rmse": round(rmse, 3)}

def train_city_pipeline(city: str):
    print(f"\n{'='*25} ADVANCED AI PIPELINE: {city.upper()} {'='*25}")
    csv_path = os.path.join(DATA_DIR, f"{city}.csv")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # 1. Hold out final 12 months as the final test set (year 2023)
    test_start_date = pd.to_datetime(TEST_START_DATE)
    train_raw = df[df["date"] < test_start_date].copy()
    test_raw = df[df["date"] >= test_start_date].copy()
    
    # 2. Climatology computed STRICTLY from training years (zero leakage)
    climatology_map = compute_climatology_map(train_raw)
    
    # 3. Build features using shared make_features
    full_feats = make_features(df, climatology_map=climatology_map, is_training=True)
    full_feats["month"] = full_feats["target_date"].dt.month
    full_feats["season"] = full_feats["month"].apply(assign_season)
    
    train_df = full_feats[full_feats["target_date"] < test_start_date].reset_index(drop=True)
    test_df = full_feats[full_feats["target_date"] >= test_start_date].reset_index(drop=True)
    
    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["target_temp_d2"].values
    
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["target_temp_d2"].values
    
    print(f"Features count: {len(FEATURE_COLUMNS)}")
    print(f"Training samples: {len(train_df)} | Test samples (final 12 months): {len(test_df)}")
    
    # 4. Cross-Validation with 5-fold TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    # --- A. Tune Random Forest ---
    rf_param_grid = [
        {"n_estimators": 100, "max_depth": 8, "min_samples_split": 5},
        {"n_estimators": 150, "max_depth": 12, "min_samples_split": 2},
        {"n_estimators": 200, "max_depth": None, "min_samples_split": 4}
    ]
    best_rf_score = float("inf")
    best_rf_params = None
    for params in rf_param_grid:
        cv_maes = []
        for train_idx, val_idx in tscv.split(X_train):
            rf_fold = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            rf_fold.fit(X_train[train_idx], y_train[train_idx])
            val_preds = rf_fold.predict(X_train[val_idx])
            cv_maes.append(mean_absolute_error(y_train[val_idx], val_preds))
        mean_mae = np.mean(cv_maes)
        if mean_mae < best_rf_score:
            best_rf_score = mean_mae
            best_rf_params = params
    print(f"Best RF Params: {best_rf_params} (CV MAE: {best_rf_score:.3f}°C)")
    
    # --- B. Tune LightGBM Regressor ---
    lgb_param_grid = [
        {"n_estimators": 80, "max_depth": 5, "learning_rate": 0.05, "num_leaves": 15},
        {"n_estimators": 120, "max_depth": 7, "learning_rate": 0.03, "num_leaves": 31},
        {"n_estimators": 150, "max_depth": -1, "learning_rate": 0.05, "num_leaves": 20}
    ]
    best_lgb_score = float("inf")
    best_lgb_params = None
    for params in lgb_param_grid:
        cv_maes = []
        for train_idx, val_idx in tscv.split(X_train):
            lgb_fold = LGBMRegressor(**params, random_state=42, verbose=-1)
            lgb_fold.fit(X_train[train_idx], y_train[train_idx])
            val_preds = lgb_fold.predict(X_train[val_idx])
            cv_maes.append(mean_absolute_error(y_train[val_idx], val_preds))
        mean_mae = np.mean(cv_maes)
        if mean_mae < best_lgb_score:
            best_lgb_score = mean_mae
            best_lgb_params = params
    print(f"Best LightGBM Params: {best_lgb_params} (CV MAE: {best_lgb_score:.3f}°C)")
    
    # 5. Out-of-fold validation predictions for Stacking Ensemble & Honest Interval
    oof_y = []
    oof_rf = []
    oof_lgb = []
    oof_ridge = []
    
    for train_idx, val_idx in tscv.split(X_train):
        # RF fold
        rf_f = RandomForestRegressor(**best_rf_params, random_state=42, n_jobs=-1)
        rf_f.fit(X_train[train_idx], y_train[train_idx])
        pred_rf = rf_f.predict(X_train[val_idx])
        
        # LGB fold
        lgb_f = LGBMRegressor(**best_lgb_params, random_state=42, verbose=-1)
        lgb_f.fit(X_train[train_idx], y_train[train_idx])
        pred_lgb = lgb_f.predict(X_train[val_idx])
        
        # Ridge fold
        ridge_f = Ridge(alpha=10.0)
        ridge_f.fit(X_train[train_idx], y_train[train_idx])
        pred_ridge = ridge_f.predict(X_train[val_idx])
        
        oof_y.extend(y_train[val_idx].tolist())
        oof_rf.extend(pred_rf.tolist())
        oof_lgb.extend(pred_lgb.tolist())
        oof_ridge.extend(pred_ridge.tolist())
        
    oof_y = np.array(oof_y)
    oof_rf = np.array(oof_rf)
    oof_lgb = np.array(oof_lgb)
    oof_ridge = np.array(oof_ridge)
    
    # 6. Fit Final Models on full training set
    rf_final = RandomForestRegressor(**best_rf_params, random_state=42, n_jobs=-1)
    rf_final.fit(X_train, y_train)
    
    lgb_final = LGBMRegressor(**best_lgb_params, random_state=42, verbose=-1)
    lgb_final.fit(X_train, y_train)
    
    ridge_final = Ridge(alpha=10.0)
    ridge_final.fit(X_train, y_train)
    
    # Fit Stacking Ensemble weights
    ensemble = StackingEnsemble(lgb_final, rf_final, ridge_final)
    ensemble.fit_weights(oof_lgb, oof_rf, oof_ridge, oof_y)
    print(f"Stacking Ensemble Weights (LGB, RF, Ridge): {[round(w, 3) for w in ensemble.weights]}")
    
    # 7. Dynamic Quantile Regressors for weather-adaptive uncertainty
    q10_model = LGBMRegressor(objective="quantile", alpha=0.10, n_estimators=100, max_depth=6, learning_rate=0.04, random_state=42, verbose=-1)
    q10_model.fit(X_train, y_train)
    
    q90_model = LGBMRegressor(objective="quantile", alpha=0.90, n_estimators=100, max_depth=6, learning_rate=0.04, random_state=42, verbose=-1)
    q90_model.fit(X_train, y_train)
    
    # Residuals for classic empirical 80% honest interval (for baseline comparison & tests)
    oof_ensemble_preds = (
        ensemble.weights[0] * oof_lgb +
        ensemble.weights[1] * oof_rf +
        ensemble.weights[2] * oof_ridge
    )
    oof_residuals = oof_y - oof_ensemble_preds
    p10 = float(np.percentile(oof_residuals, 10))
    p90 = float(np.percentile(oof_residuals, 90))
    print(f"Empirical 80% Residual Interval: [{p10:.2f}°C, {p90:.2f}°C]")
    
    # 8. Test Set Evaluation
    ens_test_preds = ensemble.predict(X_test)
    rf_test_preds = rf_final.predict(X_test)
    lgb_test_preds = lgb_final.predict(X_test)
    ridge_test_preds = ridge_final.predict(X_test)
    persistence_test_preds = test_df["baseline_persistence"].values
    climatology_test_preds = test_df["baseline_climatology"].values
    
    # Quantile dynamic bounds
    q10_test_preds = q10_model.predict(X_test)
    q90_test_preds = q90_model.predict(X_test)
    
    # Empirical test coverage under empirical honest interval
    covered_empirical = (y_test >= (ens_test_preds + p10)) & (y_test <= (ens_test_preds + p90))
    test_coverage_pct = float(np.mean(covered_empirical) * 100.0)
    
    # Dynamic quantile coverage
    covered_quantile = (y_test >= q10_test_preds) & (y_test <= q90_test_preds)
    quantile_coverage_pct = float(np.mean(covered_quantile) * 100.0)
    print(f"Empirical Coverage on Test Year: {test_coverage_pct:.1f}%")
    print(f"Dynamic Quantile Coverage on Test Year: {quantile_coverage_pct:.1f}%")
    
    # 9. Native TreeSHAP Feature Importance via LightGBM
    shap_contributions = lgb_final.booster_.predict(X_test, pred_contrib=True)
    # SHAP shape is (N, num_features + 1), last column is base expected value
    shap_feature_values = shap_contributions[:, :-1]
    mean_abs_shap = np.mean(np.abs(shap_feature_values), axis=0)
    feature_importance_list = sorted(
        [{"feature": col, "mean_abs_shap": round(float(imp), 4)} for col, imp in zip(FEATURE_COLUMNS, mean_abs_shap)],
        key=lambda x: x["mean_abs_shap"],
        reverse=True
    )
    print(f"Top 5 Physical Drivers (TreeSHAP): {[f['feature'] for f in feature_importance_list[:5]]}")
    
    # 10. Metrics Compilation
    seasons = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
    metrics_report = {
        "city": city.capitalize(),
        "best_rf_params": best_rf_params,
        "best_lgb_params": best_lgb_params,
        "ensemble_weights": [round(w, 3) for w in ensemble.weights],
        "interval_80": {"p10": round(p10, 2), "p90": round(p90, 2)},
        "test_coverage_pct": round(test_coverage_pct, 1),
        "quantile_coverage_pct": round(quantile_coverage_pct, 1),
        "top_shap_features": feature_importance_list[:7],
        "overall": {
            "ensemble": evaluate_metrics(y_test, ens_test_preds),
            "random_forest": evaluate_metrics(y_test, rf_test_preds),
            "lightgbm": evaluate_metrics(y_test, lgb_test_preds),
            "linear_regression": evaluate_metrics(y_test, ridge_test_preds),
            "persistence": evaluate_metrics(y_test, persistence_test_preds),
            "climatology": evaluate_metrics(y_test, climatology_test_preds)
        },
        "by_season": {}
    }
    
    for season in seasons:
        s_mask = (test_df["season"] == season).values
        if np.sum(s_mask) > 0:
            metrics_report["by_season"][season] = {
                "count": int(np.sum(s_mask)),
                "ensemble": evaluate_metrics(y_test[s_mask], ens_test_preds[s_mask]),
                "random_forest": evaluate_metrics(y_test[s_mask], rf_test_preds[s_mask]),
                "lightgbm": evaluate_metrics(y_test[s_mask], lgb_test_preds[s_mask]),
                "linear_regression": evaluate_metrics(y_test[s_mask], ridge_test_preds[s_mask]),
                "persistence": evaluate_metrics(y_test[s_mask], persistence_test_preds[s_mask]),
                "climatology": evaluate_metrics(y_test[s_mask], climatology_test_preds[s_mask])
            }
            
    print("\n--- FINAL TEST YEAR BENCHMARK ---")
    print(f"Stacking Ensemble -> MAE: {metrics_report['overall']['ensemble']['mae']}°C, RMSE: {metrics_report['overall']['ensemble']['rmse']}°C")
    print(f"LightGBM          -> MAE: {metrics_report['overall']['lightgbm']['mae']}°C, RMSE: {metrics_report['overall']['lightgbm']['rmse']}°C")
    print(f"Random Forest     -> MAE: {metrics_report['overall']['random_forest']['mae']}°C, RMSE: {metrics_report['overall']['random_forest']['rmse']}°C")
    print(f"Linear Regression -> MAE: {metrics_report['overall']['linear_regression']['mae']}°C, RMSE: {metrics_report['overall']['linear_regression']['rmse']}°C")
    print(f"Persistence       -> MAE: {metrics_report['overall']['persistence']['mae']}°C, RMSE: {metrics_report['overall']['persistence']['rmse']}°C")
    print(f"Climatology       -> MAE: {metrics_report['overall']['climatology']['mae']}°C, RMSE: {metrics_report['overall']['climatology']['rmse']}°C")
    
    # 11. Plot High-Resolution Residuals for Stacking Ensemble
    ens_residuals = y_test - ens_test_preds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(ens_test_preds, ens_residuals, alpha=0.6, color="#4FFBDF", edgecolors="#1C2541", s=32)
    axes[0].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[0].axhline(p10, color="orange", linestyle=":", label=f"p10 ({p10:.1f}°C)")
    axes[0].axhline(p90, color="orange", linestyle=":", label=f"p90 ({p90:.1f}°C)")
    axes[0].set_title(f"{city.capitalize()} - Ensemble Residuals vs Predicted", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted Max Temp (°C)")
    axes[0].set_ylabel("Residual (Actual - Predicted) (°C)")
    axes[0].legend()
    
    # Residual Distribution
    sns.histplot(ens_residuals, kde=True, ax=axes[1], color="#3B82F6", bins=25)
    axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[1].axvline(p10, color="orange", linestyle=":", label=f"10th percentile")
    axes[1].axvline(p90, color="orange", linestyle=":", label=f"90th percentile")
    axes[1].set_title(f"{city.capitalize()} - Ensemble Residual Error Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Residual (°C)")
    axes[1].legend()
    
    plt.tight_layout()
    res_fig_path = os.path.join(FIGURES_DIR, f"residuals_{city}.png")
    plt.savefig(res_fig_path, dpi=300)
    plt.close()
    print(f"Saved updated residual plot: {res_fig_path}")
    
    # 12. Save Models and Metadata
    joblib.dump(rf_final, os.path.join(MODELS_DIR, f"{city}_rf.joblib"))
    joblib.dump(lgb_final, os.path.join(MODELS_DIR, f"{city}_lgb.joblib"))
    joblib.dump(ridge_final, os.path.join(MODELS_DIR, f"{city}_lr.joblib"))
    joblib.dump(ensemble, os.path.join(MODELS_DIR, f"{city}_ensemble.joblib"))
    joblib.dump(q10_model, os.path.join(MODELS_DIR, f"{city}_q10.joblib"))
    joblib.dump(q90_model, os.path.join(MODELS_DIR, f"{city}_q90.joblib"))
    
    clim_path = os.path.join(MODELS_DIR, f"{city}_climatology.json")
    with open(clim_path, "w") as f:
        json.dump({str(k): v for k, v in climatology_map.items()}, f, indent=2)
        
    return metrics_report

def train_all_cities():
    all_metrics = {}
    for city in CITIES:
        metrics = train_city_pipeline(city)
        all_metrics[city] = metrics
        
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll models, weights, and metrics successfully saved to {MODELS_DIR}")

if __name__ == "__main__":
    train_all_cities()
