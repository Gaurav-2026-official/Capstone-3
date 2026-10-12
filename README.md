# Gold Price Prediction — Time-Series Forecasting Capstone

A complete, reproducible capstone project using the supplied **Gold Price (2013–2023)** dataset. Target: daily gold `Price` in **USD per troy ounce** (as represented by the source data; verify the original data provider's unit definition before publishing).

## Capstone objectives
- Inspect, clean, and visualize daily gold prices.
- Test stationarity using **ADF** and **KPSS** (price level and transformed series).
- Establish naïve and seasonal-naïve baselines.
- Compare **AR, MA, ARMA, ARIMA, SARIMA** statistical models on a chronological holdout.
- Compare optional deep-learning **RNN, LSTM, GRU** models when TensorFlow is installed.
- Evaluate using MAE, RMSE, MAPE and directional/visual diagnostics.
- Explore forecasts in a small Streamlit dashboard.

## Dataset
`data/gold_price_2013_2023.csv` — 2,583 rows, columns: `Date`, `Price`, `Open`, `High`, `Low`, `Vol.`, `Change %`. Source rows are reverse chronological; the loader sorts ascending and parses comma-separated numeric values. Missing volume values are allowed because volume is not required for univariate price forecasts.

## Stationarity finding (computed on supplied data)
- Gold **Price level**: ADF p-value ≈ **0.761** (fails to reject unit root); KPSS p-value reported as **0.01** (rejects level stationarity). Evidence supports treating the level as **non-stationary**.
- First difference: ADF p-value effectively **0.000**; KPSS p-value reported as **0.10** (upper bound of KPSS lookup table). Evidence supports the first-differenced series as stationary for this sample.
- Log returns show the same broad conclusion. ADF/KPSS results depend on deterministic terms and lag selection; do not interpret a p-value reported as 0.01/0.10 as an exact value when the test reports table bounds.

## Quick start (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```
Open `notebooks/Gold_Price_Prediction_Capstone.ipynb` and run all cells from top to bottom.

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate the environment.

## Run the dashboard
```powershell
streamlit run app.py
```
The dashboard compares baseline and statistical models. Neural-network models are optional and require TensorFlow; they are intentionally not required for the main project.

## Project structure
```text
Gold_Price_Prediction_Capstone/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── gold_price_2013_2023.csv
├── notebooks/
│   └── Gold_Price_Prediction_Capstone.ipynb
├── src/
│   ├── data_utils.py
│   ├── stationarity.py
│   ├── statistical_models.py
│   └── deep_learning.py
└── reports/
    └── stationarity_summary.csv
```

## Modeling methodology
1. Sort by date, parse numeric fields, remove invalid date/price rows, and inspect gaps.
2. Keep the latest 20% of observations as the test set; do not shuffle time-series data.
3. Fit transformations/scalers on training data only.
4. Fit statistical models on the training segment and forecast the entire held-out horizon.
5. Use the same holdout for all models and compare metrics. Some ARMA/MA fits may fail or be unstable depending on library versions; failures are captured and reported rather than hidden.
6. For deep learning, use rolling input windows and chronological validation. Install TensorFlow separately if needed.

## Important interpretation notes
- This is an educational forecasting project, **not investment advice**.
- Daily gold prices are noisy and influenced by macroeconomic variables not included here. A forecast is uncertain; never describe it as guaranteed.
- SARIMA seasonal period is configurable. A 5-trading-day period can capture a weekly pattern in trading-day observations, but markets have holidays and the dataset is not necessarily a complete business-day calendar. Compare it with non-seasonal models rather than assuming seasonality exists.
- The data ends in 2022 despite the filename's “2013–2023” label; use the actual maximum date in the file.
- AR/MA/ARMA models are best used on a stationary series. In the notebook, AR and MA/ARMA are fitted to first differences and predictions are reconstructed to the price level. ARIMA/SARIMA are fitted directly to price levels with differencing.

## Example run (80/20 chronological split, seasonal period = 5)
On the supplied data, one run produced the following statistical-model ranking by RMSE. Re-run the notebook to reproduce; software versions and optimizer details may cause small differences.

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| SARIMA(1,1,1)x(1,0,1,5) | 145.05 | 160.46 | 8.21% |
| ARIMA(5,1,0) | 145.66 | 161.05 | 8.25% |
| Naive persistence | 148.93 | 164.21 | 8.43% |
| ARIMA(2,1,2) | 150.16 | 165.38 | 8.50% |

This example indicates a small improvement over the naive baseline for SARIMA on this particular long holdout, not a guarantee of future performance. Because the test period is long, treat these results as a baseline; add rolling-origin backtesting before drawing strong conclusions.

## Generate future gold-price forecasts
The Streamlit dashboard now includes a **Predict future gold prices** section beneath the historical holdout comparison. Select a statistical model and a horizon of 7, 30, or 90 future observations, then click **Predict Gold Prices**. The app refits the selected model on all available observations and displays the forecast chart, forecast table, and CSV download. Prediction intervals are shown where the model implementation supports them.

**Data freshness warning:** the included CSV's latest observation is 30 December 2022. Forecasts therefore extend from that date, not from today's market price. Update `data/gold_price_2013_2023.csv` with recent observations before treating forecasts as relevant to current market conditions. Future dates use weekdays but do not account for exchange-specific holidays. Forecasts are educational estimates, not investment advice.
