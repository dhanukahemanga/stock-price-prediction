📈 Forecasting Stock Prices Using Macroeconomic Indicators
A Hybrid SARIMAX–LSTM Modeling Approach for an Emerging Market
📌 Project Overview

This project focuses on forecasting stock prices in an emerging market context by integrating macroeconomic indicators with hybrid time-series and deep learning models.
Using Sri Lanka Telecom (SLT) stock as a case study, the project combines statistical modeling (SARIMAX) and deep learning (LSTM) to capture both linear/seasonal patterns and nonlinear market dynamics across different economic regimes.

The model is evaluated under Pre-COVID, COVID, and Post-COVID market conditions to assess robustness during periods of economic stability and volatility.

🎯 Objectives

Analyze the impact of macroeconomic indicators (inflation, interest rate, exchange rate, GDP) on stock prices

Build and compare statistical, machine learning, and hybrid forecasting models

Improve forecasting accuracy over standalone models

Provide an interpretable and robust framework for emerging market stock prediction

🧠 Methodology
Modeling Framework

SARIMAX

Captures linear trends, seasonality, and macroeconomic effects

Uses lagged macroeconomic variables as exogenous regressors

LSTM

Learns nonlinear temporal dependencies and long-term memory patterns

Applied to standardized residuals and selected macro features

Hybrid SARIMAX–LSTM

SARIMAX models linear components

LSTM models nonlinear residual dynamics

Final prediction = SARIMAX forecast + LSTM-predicted residuals

📊 Data Description

Stock Data

Source: Colombo Stock Exchange (CSE)

Frequency: Daily

Period: July 2009 – June 2025

Variables: Open, High, Low, Close, Volume

Macroeconomic Data

Source: Central Bank of Sri Lanka (CBSL)

Frequency: Monthly (aligned to daily data)

Variables:

Inflation Rate

Interest Rate

Exchange Rate (LKR/USD)

GDP

📆 Market Regime Segmentation
Period	Date Range	Market Characteristics
Pre-COVID	2009 – mid 2022	Relatively stable market
COVID	mid 2022 – 2023	High volatility & structural breaks
Post-COVID	2024 – 2025	Recovery & normalization
⚙️ Data Processing & Feature Engineering

Missing value handling and forward-filling

Log transformation and differencing for stationarity

Lagged macroeconomic features (up to 120 days)

Min–Max scaling and standardization

Variance Inflation Factor (VIF) analysis for multicollinearity

📈 Evaluation Metrics

Model performance is evaluated using error-based metrics:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

MAPE / sMAPE

🏆 Key Results

Hybrid SARIMAX–LSTM achieved:

~22% RMSE reduction compared to standalone SARIMAX

~18% MAE reduction compared to baseline machine learning models

Demonstrated higher robustness during COVID-period volatility

Improved short-term (1-month) forecasting stability in an emerging market setting

🛠️ Tech Stack

Programming Language: Python 3.10

Libraries & Tools:

Pandas, NumPy

Statsmodels (SARIMAX)

Scikit-learn

TensorFlow / Keras (LSTM)

Matplotlib, Seaborn

Environment: Google Colab (GPU-enabled)
