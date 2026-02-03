# 📈 Forecasting Stock Prices Using Macroeconomic Indicators
  
## Hybrid SARIMAX–LSTM Modeling Approach (Emerging Market Case Study)

---

## Description
This repository contains the implementation of a **hybrid SARIMAX–LSTM time series forecasting model** developed to predict stock prices using macroeconomic indicators in an emerging market context.  
The study focuses on **Sri Lanka Telecom (SLT)** stock prices and evaluates model robustness across different economic regimes.

---

## Problem Statement
Traditional statistical models struggle to capture non-linear market dynamics, while machine learning models often lack economic interpretability.  
This project addresses this gap by combining **econometric modeling** and **deep learning** into a single hybrid framework.

---

## Methodology
The modeling pipeline consists of three stages:

- **SARIMAX**
  - Models linear trends and seasonality
  - Incorporates macroeconomic variables as exogenous regressors

- **LSTM**
  - Captures non-linear temporal dependencies
  - Learns long-term memory effects in stock price movements

- **Hybrid SARIMAX–LSTM**
  - SARIMAX forecasts the linear component
  - LSTM models residual (non-linear) patterns
  - Final forecast = SARIMAX output + LSTM residual prediction

---

## Data
### Stock Market Data
- Company: Sri Lanka Telecom PLC (SLT)
- Frequency: Daily
- Period: 2009 – 2025
- Source: Colombo Stock Exchange (CSE)

### Macroeconomic Indicators
- Inflation Rate (%)
- Interest Rate (%)
- Exchange Rate (LKR/USD)
- Gross Domestic Product (GDP)
- Source: Central Bank of Sri Lanka (CBSL) and Department of Census and Statistics

---

## Market Regimes
The dataset is segmented into three economic periods:

- **Pre-COVID**: Stable market conditions  
- **COVID**: High volatility and structural breaks  
- **Post-COVID**: Market recovery and normalization  

Models are trained and evaluated separately for each regime.

---

## Data Processing
- Stationarity testing and differencing
- Difference of stock prices
- Lagged macroeconomic feature engineering
- Feature scaling (Min-Max / Standardization)
- Residual extraction for hybrid modeling

---

## Evaluation Metrics
Model performance is evaluated using:

- Root Mean Square Error (**RMSE**)
- Mean Absolute Error (**MAE**)
- Mean Absolute Percentage Error (**MAPE / sMAPE**)

---

## Results
- Hybrid SARIMAX–LSTM model achieved:
  - ~22% reduction in RMSE compared to standalone SARIMAX
  - ~18% reduction in MAE compared to baseline machine learning models
- Improved robustness during high-volatility (COVID) period
- Demonstrated effectiveness of macroeconomic-driven forecasting

---

## Tools & Technologies
- Python
- pandas, numpy
- statsmodels
- scikit-learn
- TensorFlow / Keras
- matplotlib

---
