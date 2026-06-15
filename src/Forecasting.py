"""
Forecasting Module
Time-series forecasting of pharma sales using Prophet.
Forecasts retail and bulk sales separately, and by country
(given the structural Germany vs Poland difference found in segmentation).
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
 
 
def prepare_monthly_series(df: pd.DataFrame, group_filter: dict = None) -> pd.DataFrame:
    """
    Aggregate to monthly sales, optionally filtered by a dict of column:value
    (e.g. {'country': 'Germany', 'order_type': 'retail'}).
    Returns a dataframe with columns ['ds', 'y'] as required by Prophet.
    """
    data = df.copy()
    if group_filter:
        for col, val in group_filter.items():
            data = data[data[col] == val]
 
    monthly = data.groupby(pd.Grouper(key="date", freq="ME"))["sales"].sum().reset_index()
    monthly["date"] = monthly["date"].values.astype("datetime64[M]")  # normalize to month start
    monthly = monthly.rename(columns={"date": "ds", "sales": "y"})
    monthly = monthly.sort_values("ds").reset_index(drop=True)
    return monthly
 
 
def fit_prophet(series: pd.DataFrame, yearly_seasonality: bool = True,
                 changepoint_prior_scale: float = 0.05) -> Prophet:
    """Fit a Prophet model on a ['ds','y'] dataframe."""
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
    )
    model.fit(series)
    return model
 
 
def forecast_future(model: Prophet, periods: int = 12, freq: str = "MS"):
    """Generate forecast for `periods` future months."""
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast
 
 
def plot_forecast(model: Prophet, forecast: pd.DataFrame, title: str = "Sales Forecast"):
    fig = model.plot(forecast)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    return fig
 
 
def plot_components(model: Prophet, forecast: pd.DataFrame):
    fig = model.plot_components(forecast)
    plt.tight_layout()
    return fig
 
 
def train_test_split_series(series: pd.DataFrame, test_periods: int = 6):
    """Split a ['ds','y'] series into train/test by holding out the last N months."""
    train = series.iloc[:-test_periods].copy()
    test = series.iloc[-test_periods:].copy()
    return train, test
 
 
def evaluate_forecast(model: Prophet, test: pd.DataFrame) -> dict:
    """Compute MAE, RMSE, MAPE, MdAPE on held-out test data.
 
    MAPE can be extreme/misleading when actuals are near zero (common for
    sparse segments). MdAPE (median APE) is more robust to such outliers.
    """
    future = test[["ds"]]
    pred = model.predict(future)
    merged = test.merge(pred[["ds", "yhat"]], on="ds")
 
    mae = (merged["y"] - merged["yhat"]).abs().mean()
    rmse = np.sqrt(((merged["y"] - merged["yhat"]) ** 2).mean())
    ape = (merged["y"] - merged["yhat"]).abs() / merged["y"].replace(0, np.nan) * 100
    mape = ape.mean()
    mdape = ape.median()
 
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "MdAPE": mdape, "comparison": merged}
 
 
def plot_actual_vs_predicted(comparison: pd.DataFrame, title: str = "Actual vs Predicted"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(comparison["ds"], comparison["y"], marker="o", label="Actual")
    ax.plot(comparison["ds"], comparison["yhat"], marker="o", label="Predicted")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    plt.tight_layout()
    return fig
 