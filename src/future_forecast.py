"""Fit forecasting models on all available observations and forecast beyond the data."""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

MODEL_OPTIONS = [
    "Naive",
    "Drift",
    "AR(5)",
    "MA(5)",
    "ARMA(2,2)",
    "ARIMA(5,1,0)",
    "ARIMA(2,1,2)",
    "SARIMA(1,1,1) seasonal (1,0,1)",
]


def forecast_future(series, model_name, steps=30, seasonal_period=5, alpha=0.05):
    """Forecast future observations using the selected model fitted to all history.

    Returns a DataFrame with Forecast and, where available, lower/upper prediction
    interval columns. AR/MA/ARMA are fitted to first differences and reconstructed
    onto the original price level.
    """
    y = pd.Series(series).dropna().astype(float).sort_index()
    # Keep the original date index for labeling output, but fit statsmodels on a
    # simple positional index. This avoids "No supported index is available"
    # errors across statsmodels versions and irregular trading-day calendars.
    original_index = y.index
    values = y.to_numpy(dtype=float)
    y = pd.Series(values, index=pd.RangeIndex(len(values)), dtype=float)
    if len(y) < 30:
        raise ValueError("At least 30 valid historical observations are required.")
    if steps < 1:
        raise ValueError("Forecast horizon must be at least 1.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model_name == "Naive":
            point = np.repeat(float(y.iloc[-1]), steps)
            lower = upper = None
        elif model_name == "Drift":
            drift = (y.iloc[-1] - y.iloc[0]) / (len(y) - 1)
            point = y.iloc[-1] + np.arange(1, steps + 1) * drift
            lower = upper = None
        elif model_name in {"AR(5)", "MA(5)", "ARMA(2,2)"}:
            d = y.diff().dropna()
            orders = {"AR(5)": (5, 0, 0), "MA(5)": (0, 0, 5), "ARMA(2,2)": (2, 0, 2)}
            fitted = ARIMA(d, order=orders[model_name], trend="c").fit()
            diff_fc = fitted.get_forecast(steps=steps)
            point = y.iloc[-1] + np.cumsum(np.asarray(diff_fc.predicted_mean, dtype=float))
            # Approximate cumulative interval for forecasts of changes.
            se = np.asarray(diff_fc.se_mean, dtype=float)
            cumulative_se = np.sqrt(np.cumsum(se ** 2))
            z = float(norm.ppf(1 - alpha / 2))
            lower, upper = point - z * cumulative_se, point + z * cumulative_se
        elif model_name == "ARIMA(5,1,0)":
            fitted = ARIMA(y, order=(5, 1, 0)).fit()
            fc = fitted.get_forecast(steps=steps)
            point = np.asarray(fc.predicted_mean, dtype=float)
            ci = np.asarray(fc.conf_int(alpha=alpha), dtype=float)
            lower, upper = ci[:, 0], ci[:, 1]
        elif model_name == "ARIMA(2,1,2)":
            fitted = ARIMA(y, order=(2, 1, 2)).fit()
            fc = fitted.get_forecast(steps=steps)
            point = np.asarray(fc.predicted_mean, dtype=float)
            ci = np.asarray(fc.conf_int(alpha=alpha), dtype=float)
            lower, upper = ci[:, 0], ci[:, 1]
        elif model_name == "SARIMA(1,1,1) seasonal (1,0,1)":
            fitted = SARIMAX(
                y,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, int(seasonal_period)),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            fc = fitted.get_forecast(steps=steps)
            point = np.asarray(fc.predicted_mean, dtype=float)
            ci = np.asarray(fc.conf_int(alpha=alpha), dtype=float)
            lower, upper = ci[:, 0], ci[:, 1]
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    # Forecast dates represent future weekdays (not an exchange holiday calendar).
    # Generate weekday labels from the last real observation date. A positional
    # index is used for model fitting above, so never derive dates from it.
    last_observed_date = pd.Timestamp(original_index[-1])
    dates = pd.bdate_range(start=last_observed_date + pd.Timedelta(days=1), periods=steps)
    out = pd.DataFrame({"Forecast": np.asarray(point, dtype=float)}, index=dates)
    out.index.name = "Date"
    if lower is not None and upper is not None:
        out["Lower bound"] = np.asarray(lower, dtype=float)
        out["Upper bound"] = np.asarray(upper, dtype=float)
    return out
