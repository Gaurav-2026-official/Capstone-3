from pathlib import Path
import pandas as pd
import numpy as np

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "gold_price_2013_2023.csv"

def load_gold_data(path=DEFAULT_DATA):
    df = pd.read_csv(path)
    required = {"Date", "Price"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Price", "Open", "High", "Low"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df = df.dropna(subset=["Date", "Price"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last").set_index("Date")
    return df

def regression_metrics(actual, predicted):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[mask], predicted[mask]
    if not len(actual):
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE_%": np.nan}
    denom = np.where(np.abs(actual) < 1e-8, np.nan, np.abs(actual))
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAPE_%": float(np.nanmean(np.abs((actual - predicted) / denom)) * 100),
    }
