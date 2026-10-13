# Gold Price Prediction Using Time-Series Forecasting

## Submitter
**Gaurav Redkar**

## Final Capstone Project Documentation


**Project type:** Data Science / Time-Series Forecasting\
**Dataset:** Gold Price (2013--2023).csv\
**Price denomination:** US dollars (USD)\
**Primary tools:** Python, Pandas, NumPy, Matplotlib, Statsmodels,
Scikit-learn, TensorFlow/Keras, Streamlit\
**Document purpose:** Project overview, methodology, implementation
guide, evaluation framework, and final submission reference

------------------------------------------------------------------------

## 1. Executive Summary

This capstone project develops a reproducible workflow for analysing
historical gold prices and evaluating time-series forecasting
techniques. It combines exploratory data analysis, statistical
stationarity tests, classical forecasting models, recurrent neural
networks, and a lightweight interactive dashboard.

Gold prices are influenced by many interacting factors, including
interest-rate expectations, inflation, currency movements, central-bank
demand, geopolitical risk, and investor sentiment. A historical price
series contains useful temporal structure, but it does not contain all
information that drives future prices. Forecasts should therefore be
treated as estimates with uncertainty---not as guaranteed future prices
or investment advice.

The project compares simple baselines with classical models such as
Autoregression (AR), Moving Average (MA), Autoregressive Moving Average
(ARMA), Autoregressive Integrated Moving Average (ARIMA), and Seasonal
ARIMA (SARIMA). It also provides implementations for Simple Recurrent
Neural Networks (RNN), Long Short-Term Memory networks (LSTM), and Gated
Recurrent Units (GRU).

The analysis begins by checking the dataset's structure and date
coverage, then investigates whether the gold-price level is stationary.
If the price level is non-stationary, transformations such as first
differences or log returns are evaluated. Statistical models are
selected and fitted only after those diagnostics. Deep-learning models
use chronological splits and training-only scaling to avoid data
leakage.

Model performance is compared using Mean Absolute Error (MAE), Root Mean
Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE), with
the caveat that MAPE can behave poorly when actual values are zero or
close to zero. The same held-out period should be used for all models. A
more robust extension is rolling-origin backtesting, which evaluates
forecasts across several historical cut-off dates.

### Project goals

1.  Inspect and clean historical gold-price data.
2.  Visualise long-term trends, changes, and volatility.
3.  Test stationarity using the Augmented Dickey-Fuller (ADF) and KPSS
    tests.
4.  Build interpretable statistical forecasting models.
5.  Build and compare RNN, LSTM, and GRU models.
6.  Evaluate all models on chronologically later observations.
7.  Present findings through a reproducible notebook and Streamlit
    dashboard.
8.  Document limitations and avoid overstating predictive accuracy.

------------------------------------------------------------------------

## 2. Problem Statement and Objectives

### 2.1 Problem statement

Given historical gold prices quoted in USD, estimate future prices from
prior observations and determine which forecasting approach performs
best on unseen historical data. The project focuses on univariate
forecasting: the models primarily use the price history rather than
external explanatory variables.

### 2.2 Business and analytical motivation

Gold is commonly monitored as a store of value, a portfolio diversifier,
and a market-sensitive commodity. A forecasting workflow can help
analysts explore historical behaviour, compare model assumptions, and
understand forecast error. It can also serve as an end-to-end learning
project covering data engineering, statistics, machine learning, and
deployment.

The model should not be interpreted as a complete gold-market decision
system. A univariate model cannot directly account for new policy
announcements, sudden geopolitical events, changes in real yields, or
movements in the US dollar unless those signals are explicitly added as
features.

### 2.3 Objectives

-   Establish a reliable date-indexed dataset.
-   Measure trend and variation in the price level and its
    transformations.
-   Test whether statistical assumptions are reasonable.
-   Compare simple baselines with more complex models.
-   Prevent look-ahead bias and data leakage.
-   Explain results in a way that is understandable to a technical
    reviewer.
-   Provide a repeatable path to run the analysis locally.

### 2.4 Research questions

-   Is the gold-price level stationary?
-   Does first differencing produce a series that is more consistent
    with stationarity?
-   Do ARIMA-family models improve on a naive forecast?
-   Does a seasonal specification add value for this dataset?
-   Can RNN, LSTM, or GRU improve performance over classical methods?
-   Are the gains consistent across different forecast origins, or do
    they depend on one test window?

------------------------------------------------------------------------

## 3. Dataset Description and Data Preparation

### 3.1 Dataset overview

The supplied file is named `Gold Price (2013-2023).csv`. The preliminary
inspection found **2,583 observations** from **2 January 2013 through 30
December 2022**. The filename suggests coverage through 2023, but the
observed date range ends in 2022. The actual CSV should be treated as
the source of truth.

Before final submission, record the exact column names, data types,
missing-value counts, duplicate dates, and frequency. Do not assume that
a price column is named `Price`, `Close`, or `Gold Price` until the CSV
is inspected.

### 3.2 Recommended preparation steps

1.  Load the CSV with Pandas.
2.  Standardise column names and identify the date and price columns.
3.  Parse dates using `pandas.to_datetime`.
4.  Convert prices to numeric values, coercing invalid strings to
    missing values for inspection.
5.  Sort records by date in ascending order.
6.  Investigate duplicate dates and decide whether aggregation is
    appropriate.
7.  Check missing dates and missing prices.
8.  Set the date column as the index.
9.  Confirm that the price is quoted in USD and identify the quote
    convention, such as price per troy ounce, if the source specifies
    it.
10. Preserve a copy of the raw dataset and document all transformations.

### 3.3 Missing values and frequency

Market data often follows trading days rather than calendar days.
Weekends and market holidays are not necessarily missing observations.
Avoid automatically filling every calendar date because that can create
artificial repeated prices. If a regular business-day index is needed,
document the calendar and imputation policy.

For a time-series model, the ordering of observations matters. Randomly
shuffling rows would destroy temporal structure and create misleading
validation results.

### 3.4 Data quality checklist

-   [ ] Date column parses without unexpected failures.
-   [ ] Price values are numeric and positive.
-   [ ] Records are sorted chronologically.
-   [ ] Duplicate timestamps are explained or removed.
-   [ ] Missing prices are quantified and handled consistently.
-   [ ] Dataset start and end dates are reported accurately.
-   [ ] Units and currency are documented.
-   [ ] Raw data is kept unchanged for reproducibility.

------------------------------------------------------------------------

## 4. Exploratory Data Analysis

Exploratory Data Analysis (EDA) establishes what the series looks like
before modelling. It should include both numerical summaries and plots.

### 4.1 Descriptive statistics

Calculate count, mean, standard deviation, minimum, quartiles, and
maximum for the price column. These statistics help identify unusual
values and provide context for the price range. Because prices may trend
over time, a single mean and standard deviation do not fully describe
the series.

### 4.2 Visual analysis

Recommended charts:

-   Historical gold-price line chart.
-   Histogram or density plot of prices.
-   Daily price-change histogram.
-   Rolling mean and rolling standard deviation.
-   First-difference time series.
-   Log-return time series.
-   ACF and PACF plots for the transformed series.
-   Actual-versus-predicted chart on the held-out period.
-   Residual time series and residual ACF.

### 4.3 Rolling statistics

A rolling mean and rolling standard deviation can reveal whether the
average level or variability changes through time. A moving window is
useful for visual diagnosis, but rolling statistics alone do not
establish stationarity. Formal tests should be considered alongside
plots and domain understanding.

### 4.4 Returns and differences

Let (P_t) represent the price at time (t).

The first difference is:

\[ `\Delta `{=tex}P_t = P_t - P\_{t-1} \]

The log return is:

\[ r_t = `\log`{=tex}(P_t) - `\log`{=tex}(P\_{t-1}) \]

First differences measure absolute price changes. Log returns measure
relative changes and are often useful when the price level changes
substantially over time. They are not interchangeable, so the chosen
transformation should match the modelling goal.

------------------------------------------------------------------------

## 5. Stationarity Testing

### 5.1 What stationarity means

A weakly stationary time series has a stable mean and variance over
time, and its covariance structure depends on the lag rather than on the
absolute date. Many classical time-series methods rely on a stationary
series or use differencing to make the series more stationary.

A price series that trends over a decade often violates stationarity
assumptions. That does not mean the data is unusable; it means the model
specification must account for the changing level.

### 5.2 Augmented Dickey-Fuller (ADF) test

The ADF test's null hypothesis is that the series has a unit root. A
small p-value, commonly below 0.05, provides evidence against the
unit-root null under the selected test specification.

### 5.3 KPSS test

The KPSS test reverses the framing: its null hypothesis is that the
series is stationary around a level or trend, depending on the
specification. A small p-value is evidence against the KPSS null.

Using both tests provides complementary evidence. Results can differ
because the tests use different null hypotheses and assumptions. The
regression option, lag selection, deterministic trend, and sample size
should be documented.

### 5.4 Preliminary findings for this dataset

The preliminary project run reported the following:

  ----------------------------------------------------------------------------
  Series                   ADF p-value         KPSS p-value Preliminary
                                                            interpretation
  --------------- -------------------- -------------------- ------------------
  Price level                   0.7612                 0.01 Evidence
                                                            consistent with
                                                            non-stationarity

  First           approximately 0.0000                 0.10 Evidence
  difference                                                consistent with
                                                            stationarity

  Log returns     approximately 0.0000                 0.10 Evidence
                                                            consistent with
                                                            stationarity
  ----------------------------------------------------------------------------

The reported values should be reproduced by running the notebook in the
final environment. KPSS implementations often return bounded p-values
such as `0.01` or `0.10`; these are not always exact probabilities. A
reported value of `0.0000` for ADF should be read as a very small
p-value rounded for display, not a literal probability of zero.

### 5.5 Modelling implications

If the level is non-stationary and the first difference is stationary,
an ARIMA model with (d=1) is a reasonable starting point. AR, MA, and
ARMA models are ordinarily fitted to a stationary transformed series.
Forecasts made on a differenced scale must be integrated back to the
original price level before they are compared with actual prices.

Stationarity tests do not establish that future price changes are
predictable. They only help assess the statistical properties of the
observed series.

------------------------------------------------------------------------

## 6. Classical Time-Series Models

### 6.1 Baseline forecasts

Complex models should be compared with simple benchmarks.

**Naive forecast:** the next forecast equals the latest observed price.

\[ `\hat `{=tex}P\_{t+1} = P_t \]

**Drift forecast:** the forecast extends the average historical change
from the training period. A baseline is essential because financial
price levels are persistent, and sophisticated models can fail to
outperform a simple carry-forward prediction.

### 6.2 Autoregression (AR)

An AR model predicts a stationary value using its previous values. An
AR((p)) model can be expressed as:

\[ y_t = c + `\phi`{=tex}*1 y*{t-1} + `\cdots `{=tex}+ `\phi`{=tex}*p
y*{t-p} + `\epsilon`{=tex}\_t \]

Here, (y_t) should generally be a stationary series, such as the first
difference of price, when the original price level is non-stationary.
The order (p) controls how many lags are used.

### 6.3 Moving Average (MA)

An MA((q)) model represents the current value using current and previous
forecast errors:

\[ y_t = `\mu `{=tex}+ `\epsilon`{=tex}\_t +
`\theta`{=tex}*1`\epsilon`{=tex}*{t-1} + `\cdots `{=tex}+
`\theta`{=tex}*q`\epsilon`{=tex}*{t-q} \]

The term "moving average" in this model is not the same as a simple
rolling average chart. The MA model is a stochastic time-series model
based on lagged errors.

### 6.4 ARMA

ARMA combines autoregressive and moving-average components. It is
generally applied to stationary data. When the original price level is
non-stationary, ARMA can be applied to differences or returns, with
forecasts transformed back appropriately.

### 6.5 ARIMA

ARIMA((p,d,q)) includes:

-   (p): autoregressive order.
-   (d): number of differences.
-   (q): moving-average order.

For this dataset, (d=1) is a natural candidate based on the preliminary
stationarity tests. Orders should be selected with diagnostics,
information criteria, and time-series validation rather than by assuming
one configuration will always be best.

### 6.6 SARIMA

SARIMA extends ARIMA with seasonal terms:

\[ `\operatorname{SARIMA}`{=tex}(p,d,q)(P,D,Q,s) \]

The seasonal period (s) represents the assumed number of observations
per seasonal cycle. Daily market data has irregular calendar gaps and
may not contain strong, stable annual seasonality. A seasonal
specification should therefore be justified by exploratory analysis and
validation, not added simply because the model is available.

### 6.7 Model diagnostics

For each fitted model, examine:

-   Convergence and warning messages.
-   Residual mean and residual time plot.
-   Residual autocorrelation.
-   Ljung-Box test for remaining serial correlation.
-   Forecast errors on a held-out period.
-   AIC or BIC for candidate models fitted on comparable data and
    likelihood assumptions.

Information criteria are useful for candidate selection, but they do not
replace out-of-sample evaluation.

------------------------------------------------------------------------

## 7. Deep-Learning Models

Deep-learning models can learn nonlinear patterns, but they require
careful validation and may not outperform simpler models on a univariate
financial series.

### 7.1 Input-window design

A common design uses a sliding historical window of length (L). For
example, a model receives the previous 30 observations and predicts the
next price. The window size is a hyperparameter and should be selected
using validation data.

For a one-step forecast, the training examples have the form:

\[ X_t = \[P\_{t-L}, `\ldots`{=tex}, P\_{t-1}\], `\qquad `{=tex}y_t=P_t
\]

Multi-step forecasting requires an explicit strategy, such as recursive
prediction or direct prediction of a vector of future values. These
strategies can have different error characteristics.

### 7.2 Scaling and leakage prevention

Neural networks often train more reliably when inputs are scaled. The
scaler must be fitted on the training period only, then applied
unchanged to validation and test data. Fitting a scaler to the full
dataset leaks information about the future distribution into the
training process.

Split the data chronologically. Do not shuffle the full time series
before creating train, validation, and test segments.

### 7.3 Simple RNN

A simple recurrent neural network maintains a hidden state across
sequence steps. It is useful as a baseline recurrent architecture but
may struggle to learn long-term dependencies because gradients can
vanish or explode.

### 7.4 LSTM

Long Short-Term Memory networks use gating mechanisms to regulate the
information retained in the hidden state. They are designed to make
learning longer dependencies easier than with a basic RNN, but require
more parameters and training time.

### 7.5 GRU

Gated Recurrent Units use update and reset gates. They are often smaller
than comparable LSTMs and can train faster, although performance depends
on the data and configuration.

### 7.6 Fair comparison

For a fair comparison:

-   Use the same final test period for all models.
-   Tune architecture and hyperparameters on training/validation data
    only.
-   Report random seeds and software versions.
-   Keep scaling and inverse scaling explicit.
-   Record training time as well as forecast error.
-   Compare forecasts on the original USD price scale.
-   Avoid selecting a model based on test performance and then reporting
    that same test result as an unbiased final estimate.

------------------------------------------------------------------------

## 8. Model Evaluation and Backtesting

### 8.1 Metrics

**Mean Absolute Error (MAE)** measures average absolute error:

\[ MAE =
`\frac{1}{n}`{=tex}`\sum`{=tex}\_{t=1}\^{n}\|y_t-`\hat `{=tex}y_t\| \]

**Root Mean Squared Error (RMSE)** penalises larger errors more heavily:

\[ RMSE = `\sqrt{\frac{1}{n}\sum_{t=1}^{n}(y_t-\hat y_t)^2}`{=tex} \]

**Mean Absolute Percentage Error (MAPE)** measures average absolute
percentage error:

\[ MAPE =
`\frac{100}{n}`{=tex}`\sum`{=tex}\_{t=1}\^{n}`\left`{=tex}\|`\frac{y_t-\hat y_t}{y_t}`{=tex}`\right`{=tex}\|
\]

MAE and RMSE are expressed in the same units as the target price. MAPE
is scale-free but can be misleading when actual values are zero or near
zero. Gold prices are positive in this dataset, but MAE and RMSE should
still be reported alongside MAPE.

### 8.2 Preliminary statistical-model comparison

The preliminary run used a chronological holdout comprising the latest
20% of observations. The reported metrics were:

  Model                 MAE     RMSE    MAPE
  ---------------- -------- -------- -------
  SARIMA             145.05   160.46   8.21%
  ARIMA (5,1,0)      145.66   161.05   8.25%
  Naive baseline     148.93   164.21   8.43%
  ARIMA (2,1,2)      150.16   165.38   8.50%

SARIMA had the lowest reported error among these four candidates in this
preliminary run. The improvement over the naive baseline was modest, so
it would be premature to claim that SARIMA is definitively superior.
These values must be checked against the reproducible notebook output
and the exact test-window definition before use in a formal report.

The neural-network models are included in the project workflow, but
their relative performance should be reported only after they have been
run with the same test period and evaluation procedure.

### 8.3 Why chronological holdout matters

A random split can train on later prices and test on earlier prices,
which does not reflect the real forecasting task. A chronological
holdout better mimics forecasting from past observations into the
future.

### 8.4 Rolling-origin evaluation

For a more rigorous final evaluation, use multiple forecast origins:

1.  Train on an initial historical window.
2.  Forecast the next horizon.
3.  Move the origin forward.
4.  Refit or update the model according to a pre-defined policy.
5.  Aggregate errors across all forecast windows.

Rolling-origin evaluation helps determine whether a model performs
consistently across different market regimes. It also reduces dependence
on a single unusually easy or difficult test period.

### 8.5 Residual and forecast analysis

The final report should include an actual-versus-predicted chart, a
table of metrics, a residual plot, and a short interpretation of where
errors are largest. A model may have good average metrics but perform
poorly around sudden market moves. Forecast intervals should be included
when supported by the model and evaluated for coverage.

------------------------------------------------------------------------

## 9. Project Architecture and Folder Structure

A clean project separates data, notebooks, reusable code, evaluation
outputs, and the dashboard. The following is a recommended structure;
adapt names to match the supplied ZIP.

``` text
Gold_Price_Prediction_Capstone/
├── data/
│   └── Gold Price (2013-2023).csv
├── notebooks/
│   └── Gold_Price_Prediction_Capstone.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── stationarity.py
│   ├── statistical_models.py
│   ├── deep_learning_models.py
│   └── evaluation.py
├── app.py
├── requirements.txt
├── README.md
└── outputs/
    ├── figures/
    └── metrics/
```

### 9.1 Notebook responsibilities

The notebook should run sequentially and contain these sections:

1.  Imports and configuration.
2.  Data loading and validation.
3.  EDA and visualisations.
4.  Stationarity tests.
5.  Transformations and ACF/PACF.
6.  Baseline forecasts.
7.  AR, MA, and ARMA models.
8.  ARIMA and SARIMA models.
9.  RNN, LSTM, and GRU models.
10. Evaluation and plots.
11. Findings, limitations, and conclusion.

### 9.2 Reproducibility

Use a fixed random seed where supported, record package versions, and
document the Python version. Neural-network training may still exhibit
small hardware- or library-dependent differences. Store model parameters
and evaluation outputs so that the results can be reviewed without
relying on screenshots alone.

------------------------------------------------------------------------

## 10. Streamlit Dashboard

The Streamlit dashboard provides an accessible interface for exploring
the dataset and comparing model results.

### 10.1 Recommended dashboard pages

**Overview** - Dataset date range and number of observations. - Latest
recorded price in the supplied dataset. - Historical price chart. -
Descriptive statistics.

**Stationarity** - Price level, first difference, and log-return
charts. - ADF and KPSS results. - Clear explanation of each null
hypothesis.

**Model comparison** - Model name, MAE, RMSE, and MAPE. -
Actual-versus-predicted chart. - Best model on the selected metric, with
a reminder that rankings depend on the test period.

**Forecast** - User-selected forecast horizon. - Forecast line and,
where available, prediction intervals. - Explicit labelling of forecasts
as model estimates rather than live market prices.

### 10.2 User experience considerations

The dashboard should show the date range of the source data and the
timestamp of any model run. It should not imply that historical CSV data
is a live feed. Errors should be handled clearly, including missing data
files, unavailable optional TensorFlow dependencies, and model-fitting
failures.

------------------------------------------------------------------------

## 11. Installation and Execution

### 11.1 Prerequisites

-   Python version compatible with the pinned project dependencies.
-   Visual Studio Code or another Python IDE.
-   A virtual environment.
-   The supplied CSV file.
-   Internet access for initial package installation, unless packages
    are already available.

### 11.2 Windows setup

Open PowerShell in the project folder:

``` powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks script activation, use a permitted activation
method for your environment or invoke the virtual-environment Python
executable directly. Follow organisational security policy rather than
weakening execution controls unnecessarily.

### 11.3 Run the notebook

``` powershell
jupyter lab
```

Open `notebooks/Gold_Price_Prediction_Capstone.ipynb`, select the
virtual-environment kernel, and execute cells from top to bottom.
Confirm that the dataset path resolves correctly.

### 11.4 Run Streamlit

``` powershell
streamlit run app.py
```

The terminal should display a local URL. Open it in a browser. If a
model dependency is optional, the dashboard should either disable that
model's functionality or explain how to install the dependency.

### 11.5 Troubleshooting

-   **File not found:** check the working directory and dataset path.
-   **CSV parsing issue:** inspect encoding, delimiter, and column
    names.
-   **Date conversion errors:** review invalid date values before
    coercion.
-   **Statsmodels convergence warnings:** try alternative orders,
    inspect the transformed series, and document the warning.
-   **TensorFlow installation issue:** use a Python version supported by
    the chosen TensorFlow release and install it in the project
    environment.
-   **Notebook kernel mismatch:** select the kernel associated with
    `.venv`.
-   **Different metrics:** verify the split boundary, forecast horizon,
    scaling, inverse transformation, and missing-value treatment.

------------------------------------------------------------------------

## 12. Risks, Limitations, and Responsible Interpretation

### 12.1 Historical data limitation

The dataset ends on 30 December 2022 according to the preliminary
inspection. The forecasts are based on that historical sample and should
not be described as current gold prices. A future version of the project
can incorporate a validated, updated source.

### 12.2 Univariate limitation

The models use price history rather than external drivers. Including
variables such as real yields, US dollar indices, inflation measures,
central-bank purchases, and other market indicators may improve context,
but introduces additional data alignment and availability challenges.

### 12.3 Regime changes

Gold-market relationships can change. A model that fits one period may
fail during inflation shocks, rapid monetary tightening, crises, or
structural shifts. Retraining and rolling evaluation are important.

### 12.4 Overfitting

Trying many model orders and neural-network architectures can lead to
choosing a model that happens to fit the test period. Use validation
data for tuning and reserve the final test set for evaluation. If
repeated model selection uses the same test set, its error estimate
becomes optimistic.

### 12.5 Prediction uncertainty

Point forecasts alone do not communicate the range of plausible
outcomes. When supported, prediction intervals should be shown and
evaluated. A narrow interval is not automatically better; intervals need
appropriate empirical coverage.

### 12.6 Financial disclaimer

This is an educational forecasting project, not investment advice.
Forecasts are uncertain and may be wrong. Do not make a financial
decision solely on a model output. Real-world investment analysis should
consider risk tolerance, transaction costs, liquidity, taxes, market
conditions, and independent evidence.

------------------------------------------------------------------------

## 13. Conclusion and Future Enhancements

This capstone brings together statistical time-series analysis and
neural-network forecasting in one reproducible project. Its key
methodological principle is that the data must be understood before a
model is selected. The preliminary stationarity tests indicate that the
price level is non-stationary, while first differences and log returns
appear more stationary under the selected tests. This supports using
differencing for ARIMA-family models and using stationary
transformations where appropriate for AR, MA, and ARMA.

The preliminary statistical comparison shows SARIMA narrowly ahead of
the naive baseline on the reported holdout metrics. Because the
improvement is modest and the dataset contains a single historical
period, the result should be treated as provisional until reproduced and
validated through rolling-origin backtesting. No conclusion about RNN,
LSTM, or GRU should be made until those models are run on the same test
dates and their metrics are available.

### Future enhancements

1.  Add rolling-origin backtesting and compare several forecast
    horizons.
2.  Add prediction intervals and evaluate interval coverage.
3.  Investigate seasonal patterns rather than assuming a seasonal
    period.
4.  Add exogenous predictors and compare SARIMAX or multivariate
    machine-learning models.
5.  Benchmark additional models such as exponential smoothing and
    tree-based regressors with lag features.
6.  Add experiment tracking, model persistence, and automated evaluation
    reports.
7.  Add tests for date parsing, data leakage, and forecast-output shape.
8.  Refresh the dataset from a reliable source and record source,
    retrieval date, units, and licensing.
9.  Package the application for reproducible local or cloud deployment.

### Final project deliverables checklist

-   [ ] Cleaned and validated data pipeline.
-   [ ] EDA charts and descriptive statistics.
-   [ ] ADF and KPSS stationarity tests.
-   [ ] Baseline, AR, MA, ARMA, ARIMA, and SARIMA evaluations.
-   [ ] RNN, LSTM, and GRU evaluations.
-   [ ] Common chronological test period and documented metrics.
-   [ ] Actual-versus-forecast plots and residual diagnostics.
-   [ ] Streamlit dashboard.
-   [ ] README and dependency list.
-   [ ] Reproducible notebook run from top to bottom.
-   [ ] Conclusions that match measured results and acknowledge
    limitations.

------------------------------------------------------------------------

## Appendix A. Key Terminology

  -----------------------------------------------------------------------
  Term                                Meaning
  ----------------------------------- -----------------------------------
  Time series                         Observations ordered over time

  Stationarity                        Statistical properties remain
                                      stable under a defined stationarity
                                      concept

  Differencing                        Subtracting a prior observation to
                                      remove some time-dependent level
                                      behaviour

  Log return                          Difference between consecutive log
                                      prices

  ACF                                 Autocorrelation function;
                                      correlation with lagged values

  PACF                                Partial autocorrelation function;
                                      relationship at a lag after
                                      controlling for intermediate lags

  AR                                  Autoregressive model

  MA                                  Moving-average error model

  ARMA                                Combined AR and MA model for
                                      stationary data

  ARIMA                               ARMA with differencing

  SARIMA                              ARIMA with seasonal components

  RNN                                 Recurrent neural network

  LSTM                                Long Short-Term Memory recurrent
                                      network

  GRU                                 Gated Recurrent Unit network

  MAE                                 Mean absolute error

  RMSE                                Root mean squared error

  MAPE                                Mean absolute percentage error

  Data leakage                        Unintended use of information
                                      unavailable at forecast time

  Backtesting                         Evaluating forecasts across
                                      historical cut-off points
  -----------------------------------------------------------------------

## Appendix B. Suggested Viva / Presentation Questions

1.  Why is stationarity important for ARIMA modelling?
2.  What is the difference between the ADF and KPSS null hypotheses?
3.  Why should AR, MA, and ARMA generally use a stationary series?
4.  What does the differencing parameter (d) mean in ARIMA?
5.  How does SARIMA differ from ARIMA?
6.  Why can a naive forecast be difficult to beat for persistent price
    levels?
7.  What is the difference between RNN, LSTM, and GRU?
8.  Why must the scaler be fitted only on training data?
9.  Why is random train-test splitting unsuitable for time-series
    forecasting?
10. What do MAE, RMSE, and MAPE measure?
11. Why is rolling-origin backtesting more informative than one holdout
    period?
12. What are the limitations of using only historical gold prices?
13. How would external economic variables be introduced without data
    leakage?
14. How should a forecast interval be interpreted?
15. Why should model output not be treated as guaranteed investment
    advice?

## Appendix C. Recommended References

-   *Forecasting: Principles and Practice*, Rob J. Hyndman and George
    Athanasopoulos --- time-series forecasting concepts and evaluation.
-   Statsmodels documentation --- ARIMA, SARIMAX, stationarity tests,
    and residual diagnostics.
-   Scikit-learn documentation --- preprocessing, metrics, and
    time-series splitting utilities.
-   TensorFlow/Keras documentation --- recurrent layers, LSTM, and GRU.
-   Streamlit documentation --- interactive Python data applications.

Always consult the documentation for the exact package versions
installed in the project, as APIs and supported environments can change.
