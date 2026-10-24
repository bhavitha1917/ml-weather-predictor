"""
src/ensemble.py
Blended Stacking Ensemble for Weather Forecasting.
Combines LightGBM, Random Forest, and Ridge with non-negative optimal weights.
"""

import numpy as np
from sklearn.linear_model import Ridge

class StackingEnsemble:
    """Combines LightGBM, Random Forest, and Ridge regression models with learned weights."""
    def __init__(self, lgb_model, rf_model, ridge_model, weights=None):
        self.lgb_model = lgb_model
        self.rf_model = rf_model
        self.ridge_model = ridge_model
        self.weights = weights if weights is not None else [0.35, 0.35, 0.30]

    def fit_weights(self, oof_lgb, oof_rf, oof_ridge, y_true):
        # Learn optimal non-negative blending weights via bounded linear regression
        X_oof = np.column_stack([oof_lgb, oof_rf, oof_ridge])
        blender = Ridge(alpha=1.0, positive=True, fit_intercept=False)
        blender.fit(X_oof, y_true)
        raw_weights = blender.coef_
        total = np.sum(raw_weights)
        if total > 0:
            self.weights = (raw_weights / total).tolist()
        else:
            self.weights = [0.45, 0.45, 0.10]

    def predict(self, X):
        pred_lgb = self.lgb_model.predict(X)
        pred_rf = self.rf_model.predict(X)
        pred_ridge = self.ridge_model.predict(X)
        return (
            self.weights[0] * pred_lgb +
            self.weights[1] * pred_rf +
            self.weights[2] * pred_ridge
        )
