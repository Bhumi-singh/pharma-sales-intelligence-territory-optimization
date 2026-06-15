"""
Predictive Modeling Module
XGBoost regression to predict transaction-level sales value,
with SHAP-based interpretability to identify key revenue drivers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import shap


CATEGORICAL_FEATURES = [
    "country", "channel", "sub_channel", "product_class",
    "sales_team", "manager",
]
NUMERIC_FEATURES = ["month_num", "year", "quantity"]
TARGET = "sales"


def build_model_dataset(df: pd.DataFrame,
                         categorical_features: list = None,
                         numeric_features: list = None,
                         target: str = TARGET):
    """
    Prepare feature matrix and target for modeling.
    Encodes categorical features with OrdinalEncoder (XGBoost handles ordinal-encoded
    categoricals well and it's simpler than one-hot for high-cardinality columns).
    Returns X, y, encoder, feature_names.
    """
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    numeric_features = numeric_features or NUMERIC_FEATURES

    data = df[categorical_features + numeric_features + [target]].copy()
    data = data.dropna()

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    data[categorical_features] = encoder.fit_transform(data[categorical_features])

    X = data[categorical_features + numeric_features]
    y = data[target]

    return X, y, encoder, categorical_features + numeric_features


def train_xgb_model(X_train, y_train, **kwargs) -> xgb.XGBRegressor:
    """Train an XGBoost regressor with sensible defaults."""
    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    params.update(kwargs)
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute MAE, RMSE, R2, and MAPE on test set."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    # MAPE excluding zero/near-zero actuals
    mask = y_test.abs() > 1
    mape = (np.abs((y_test[mask] - preds[mask]) / y_test[mask])).mean() * 100

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "predictions": preds}


def plot_actual_vs_predicted(y_test, preds, sample_size: int = 2000):
    """Scatter plot of actual vs predicted sales (sampled for readability)."""
    if len(y_test) > sample_size:
        idx = np.random.RandomState(42).choice(len(y_test), sample_size, replace=False)
        y_sample = y_test.values[idx]
        pred_sample = preds[idx]
    else:
        y_sample = y_test.values
        pred_sample = preds

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_sample, pred_sample, alpha=0.3, s=10)
    max_val = max(y_sample.max(), pred_sample.max())
    ax.plot([0, max_val], [0, max_val], "r--", label="Perfect prediction")
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title("Actual vs Predicted Sales")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names):
    """Bar chart of XGBoost feature importances."""
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(feature_names)[order], importances[order])
    ax.set_title("XGBoost Feature Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def compute_shap_values(model, X_sample):
    """Compute SHAP values for a sample of the data (TreeExplainer for XGBoost)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return explainer, shap_values


def plot_shap_summary(shap_values, X_sample):
    """SHAP summary (beeswarm) plot."""
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    return fig


def plot_shap_bar(shap_values, X_sample):
    """SHAP mean absolute value bar plot."""
    fig = plt.figure(figsize=(8, 5))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    return fig


def decode_category(encoder: OrdinalEncoder, categorical_features: list, feature: str, encoded_value: float) -> str:
    """Decode an ordinal-encoded category value back to its original label."""
    idx = categorical_features.index(feature)
    categories = encoder.categories_[idx]
    if encoded_value < 0 or int(encoded_value) >= len(categories):
        return "Unknown"
    return categories[int(encoded_value)]
