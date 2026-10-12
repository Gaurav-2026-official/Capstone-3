import warnings
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

def stationarity_report(series):
    """ADF H0: unit root/non-stationary. KPSS H0: stationary around a constant."""
    s = pd.Series(series).dropna().astype(float)
    rows = []
    for name, values in [("Price level", s), ("First difference", s.diff()), ("Log return", __import__("numpy").log(s).diff())]:
        values = values.dropna()
        adf_stat, adf_p, adf_lags, adf_n, *_ = adfuller(values, autolag="AIC")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k_stat, k_p, k_lags, *_ = kpss(values, regression="c", nlags="auto")
        rows.append({
            "Series": name, "Observations": len(values),
            "ADF statistic": adf_stat, "ADF p-value": adf_p, "ADF lags": adf_lags,
            "KPSS statistic": k_stat, "KPSS p-value (table bound)": k_p, "KPSS lags": k_lags,
            "ADF conclusion (5%)": "stationary" if adf_p < .05 else "non-stationary",
            "KPSS conclusion (5%)": "stationary" if k_p > .05 else "non-stationary",
        })
    return pd.DataFrame(rows)
