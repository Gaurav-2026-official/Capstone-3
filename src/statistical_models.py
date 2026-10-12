"""Statistical model comparison with index-safe chronological forecasts."""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .data_utils import regression_metrics


def evaluate_statistical_models(series, test_size=0.2, seasonal_period=5):
    """Evaluate forecasts on a chronological holdout, in original price units.

    Statsmodels receives a RangeIndex intentionally. The original date index is
    retained only for output/plots, avoiding 'No supported index is available'.
    """
    y_dates = pd.Series(series).dropna().astype(float).sort_index()
    n_test = max(1, int(round(len(y_dates) * test_size)))
    train_dates, test = y_dates.iloc[:-n_test], y_dates.iloc[-n_test:]
    # Models receive a numeric index so forecast horizons work across statsmodels versions.
    train = pd.Series(train_dates.to_numpy(dtype=float), index=pd.RangeIndex(len(train_dates)))
    results, predictions = [], pd.DataFrame(index=test.index)
    predictions["Actual"] = test.to_numpy(dtype=float)

    baseline_preds = [
        ("Naive", np.repeat(float(train.iloc[-1]), len(test))),
        ("Drift", train.iloc[-1] + np.arange(1, len(test) + 1) *
         (train.iloc[-1] - train.iloc[0]) / max(1, len(train) - 1)),
    ]
    for name, pred in baseline_preds:
        predictions[name] = pred
        results.append({"Model": name, **regression_metrics(test, pred), "Error": ""})

    diff = train.diff().dropna()
    candidates = [
        ("AR(5)", lambda: ARIMA(diff, order=(5, 0, 0), trend="c"), True),
        ("MA(5)", lambda: ARIMA(diff, order=(0, 0, 5), trend="c"), True),
        ("ARMA(2,2)", lambda: ARIMA(diff, order=(2, 0, 2), trend="c"), True),
        ("ARIMA(5,1,0)", lambda: ARIMA(train, order=(5, 1, 0)), False),
        ("ARIMA(2,1,2)", lambda: ARIMA(train, order=(2, 1, 2)), False),
        (f"SARIMA(1,1,1)x(1,0,1,{int(seasonal_period)})",
         lambda: SARIMAX(train, order=(1, 1, 1),
                         seasonal_order=(1, 0, 1, int(seasonal_period)),
                         enforce_stationarity=False, enforce_invertibility=False), False),
    ]
    for name, factory, is_diff_model in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = factory().fit(disp=False) if name.startswith("SARIMA") else factory().fit()
                fc = np.asarray(fitted.forecast(steps=len(test)), dtype=float).reshape(-1)
            if is_diff_model:
                fc = float(train.iloc[-1]) + np.cumsum(fc)
            if len(fc) != len(test) or not np.isfinite(fc).all():
                raise ValueError("Model returned invalid forecast values or an unexpected horizon.")
            predictions[name] = fc
            results.append({"Model": name, **regression_metrics(test, fc), "Error": ""})
        except Exception as exc:
            results.append({"Model": name, "MAE": np.nan, "RMSE": np.nan,
                            "MAPE_%": np.nan, "Error": str(exc)[:180]})
    metrics = pd.DataFrame(results).sort_values("RMSE", na_position="last").reset_index(drop=True)
    return train_dates, test, predictions, metrics
