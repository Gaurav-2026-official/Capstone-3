import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data_utils import load_gold_data
from src.stationarity import stationarity_report
from src.statistical_models import evaluate_statistical_models
from src.future_forecast import forecast_future, MODEL_OPTIONS

st.set_page_config(page_title="Gold Price Prediction Capstone", page_icon="📈", layout="wide")
st.title("Gold Price Prediction — Time-Series Capstone")
st.caption("Educational forecasting only • Forecasts are uncertain and are not investment advice.")

try:
    df = load_gold_data()
except Exception as exc:
    st.error(f"Could not load the gold-price CSV: {exc}")
    st.stop()

y = df["Price"]
last_date = y.index.max()
c1, c2, c3 = st.columns(3)
c1.metric("Observations", f"{len(y):,}")
c2.metric("Available history", f"{y.index.min():%d %b %Y} – {last_date:%d %b %Y}")
c3.metric("Latest recorded price", f"${y.iloc[-1]:,.2f}", help="Price units follow the supplied CSV; verify the original source's unit definition.")

if last_date < pd.Timestamp.today().normalize() - pd.Timedelta(days=365):
    st.warning(
        f"Your CSV ends on {last_date:%d %b %Y}. Forecasts below extend from that last recorded date, "
        "not from today's live gold price. For a current-market forecast, update the CSV with recent observations first."
    )

st.subheader("Historical gold price")
st.line_chart(y.rename("Historical price"))

with st.expander("Stationarity tests", expanded=True):
    st.dataframe(stationarity_report(y), use_container_width=True, hide_index=True)
    st.caption("ADF null: unit root (non-stationary). KPSS null: level-stationary. Interpret both tests together; KPSS p-values may be table bounds.")

st.divider()
st.header("1. Model comparison")
st.write("Compare historical holdout forecasts. The latest observations are held out in chronological order; they are not randomly shuffled.")
with st.sidebar:
    st.header("Settings")
    test_size = st.slider("Test fraction", 0.10, 0.30, 0.20, 0.05)
    seasonal_period = st.selectbox(
        "SARIMA seasonal period", [5, 7, 21, 252], index=0,
        help="Candidate periods measured in observations. A period of 5 is a rough trading-week pattern, not an exchange calendar."
    )
    run_comparison = st.button("Run model comparison", type="primary")

if run_comparison:
    with st.spinner("Fitting statistical models on the chronological training window…"):
        try:
            train, test, preds, metrics = evaluate_statistical_models(y, test_size, seasonal_period)
            st.session_state["comparison_metrics"] = metrics
            st.session_state["comparison_predictions"] = preds
            st.session_state["comparison_test_size"] = test_size
            st.session_state["comparison_seasonal_period"] = seasonal_period
        except Exception as exc:
            st.error(f"Model comparison failed: {exc}")

if "comparison_metrics" in st.session_state:
    metrics = st.session_state["comparison_metrics"]
    preds = st.session_state["comparison_predictions"]
    st.subheader("Holdout metrics — lower is better")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.subheader("Actual vs predicted — historical test window")
    st.line_chart(preds)
    st.download_button("Download metrics CSV", metrics.to_csv(index=False), "gold_model_metrics.csv", "text/csv")
    st.download_button("Download holdout forecasts CSV", preds.to_csv(), "gold_holdout_forecasts.csv", "text/csv")
else:
    st.info("Choose the settings in the sidebar and click **Run model comparison** to evaluate models on historical data.")

st.divider()
st.header("2. Predict future gold prices")
st.write(
    "Select a model and forecast horizon to generate price estimates after the last date in the CSV. "
    "The model is refitted using all available historical observations before forecasting."
)

with st.form("future_forecast_form"):
    left, middle, right = st.columns([2, 1, 1])
    with left:
        model_choice = st.selectbox("Forecast model", MODEL_OPTIONS, index=5,
            help="Choose a model based on holdout performance. SARIMA uses the seasonal period set in the sidebar.")
    with middle:
        horizon_label = st.selectbox("Forecast horizon", ["7 observations", "30 observations", "90 observations"], index=1)
    with right:
        confidence = st.selectbox("Prediction interval", [90, 95, 99], index=1, help="Available for ARIMA/SARIMA and approximate for AR/MA/ARMA.")
    make_forecast = st.form_submit_button("Predict Gold Prices", type="primary")

if make_forecast:
    steps = int(horizon_label.split()[0])
    alpha = 1 - confidence / 100
    with st.spinner(f"Fitting {model_choice} on all available data and generating {steps} forecasts…"):
        try:
            forecast = forecast_future(y, model_choice, steps=steps,
                seasonal_period=seasonal_period, alpha=alpha)
            st.session_state["future_forecast"] = forecast
            st.session_state["future_model"] = model_choice
            st.session_state["future_horizon"] = steps
            st.session_state["future_confidence"] = confidence
        except Exception as exc:
            st.error(f"Forecast generation failed: {exc}")

if "future_forecast" in st.session_state:
    forecast = st.session_state["future_forecast"]
    selected_model = st.session_state["future_model"]
    confidence = st.session_state["future_confidence"]
    first_price = float(forecast["Forecast"].iloc[0])
    last_price = float(forecast["Forecast"].iloc[-1])
    change = last_price - float(y.iloc[-1])
    change_pct = change / float(y.iloc[-1]) * 100 if float(y.iloc[-1]) else 0
    m1, m2, m3 = st.columns(3)
    m1.metric("Selected model", selected_model)
    m2.metric("First forecast", f"${first_price:,.2f}", help=f"Forecast date: {forecast.index[0]:%d %b %Y}")
    m3.metric("Last forecast", f"${last_price:,.2f}", f"{change_pct:+.2f}% vs last recorded price")

    st.subheader("Historical prices and future forecast")
    display_history = y.tail(180).rename("Historical price").to_frame()
    chart_data = display_history.join(forecast, how="outer")
    st.line_chart(chart_data)
    if {"Lower bound", "Upper bound"}.issubset(forecast.columns):
        st.caption(f"The table includes nominal {confidence}% prediction interval bounds where supported. The chart shows the point forecast, not a shaded interval.")
    st.subheader("Forecast values")
    shown = forecast.copy()
    shown.index = shown.index.strftime("%Y-%m-%d")
    st.dataframe(shown.style.format("${:,.2f}"), use_container_width=True)
    st.download_button(
        "Download future forecast CSV", forecast.to_csv(index_label="Date"),
        f"gold_price_forecast_{st.session_state['future_horizon']}_observations.csv", "text/csv"
    )
    st.caption(
        "Forecast dates use weekdays and do not account for exchange-specific holidays. Prediction intervals depend on model assumptions and are not guarantees. "
        "These are statistical estimates from historical prices only; macroeconomic events and new information can cause actual prices to differ substantially."
    )
