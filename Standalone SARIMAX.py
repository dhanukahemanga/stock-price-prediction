#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =========================
# SARIMAX-only — Period-wise Evaluation
# (with rich metrics + styled HTML view + Excel export)
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from IPython.display import display, HTML

import warnings
warnings.filterwarnings("ignore")

# ---------- 0) Load & prep ----------
csv_path = "Final_Stock_prices_and_macroeconomic_data.csv"
df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
df = df.apply(pd.to_numeric, errors="ignore")

price = df["Price"].astype(float)
logp  = np.log(price).rename("log_price")

# exogenous: ONLY 120-day lags of Inflation% & Interest%
def make_exog_lags(frame, cols=("Inflation rate(%)","Interest rate(%)"), lags=(120,)):
    X = pd.DataFrame(index=frame.index)
    for c in cols:
        for L in lags:
            X[f"{c}_lag{L}"] = frame[c].shift(L)
    return X

X_all = make_exog_lags(df, lags=(120,))
full  = pd.concat([logp, X_all], axis=1).dropna()
y_all = full["log_price"]
X_all = full.drop(columns=["log_price"])

# ---------- 1) Periods & fixed orders ----------
periods = {
    "Pre-COVID":  {"range": ("2009-07-08", "2022-06-30"), "order": (1,1,2), "seasonal_order": (0,1,1,5)},
    "COVID":      {"range": ("2022-07-01", "2023-12-31"), "order": (1,1,0), "seasonal_order": (0,1,1,5)},
    "Post-COVID": {"range": ("2024-01-01", "2025-06-30"), "order": (1,1,1), "seasonal_order": (1,0,0,5)},
}

# ---------- 2) Metrics helpers ----------
def _align(a: pd.Series, p: pd.Series):
    common = a.index.intersection(p.index)
    return a.loc[common].astype(float), p.loc[common].astype(float)

def mape(a: pd.Series, p: pd.Series, eps=1e-8):
    a, p = _align(a, p)
    return float(100*np.mean(np.abs((a - p) / (a.replace(0, eps)))))

def smape(a: pd.Series, p: pd.Series, eps=1e-8):
    a, p = _align(a, p)
    return float(100*np.mean(2*np.abs(p - a) / (np.abs(a) + np.abs(p) + eps)))

def mase(a: pd.Series, p: pd.Series, m=5):
    a, p = _align(a, p)
    mae = np.mean(np.abs(a - p))
    if len(a) <= m: return np.nan
    denom = np.mean(np.abs(a.values[m:] - a.values[:-m]))
    return float(mae / (denom + 1e-12))

def theil_u1(a: pd.Series, p: pd.Series):
    a, p = _align(a, p)
    num = np.sqrt(np.mean((p - a) ** 2))
    denom = np.sqrt(np.mean(a ** 2)) + np.sqrt(np.mean(p ** 2))
    return float(num / (denom + 1e-12))

def huber_loss(a: pd.Series, p: pd.Series, delta=1.0):
    a, p = _align(a, p)
    r = (a - p).values
    absr = np.abs(r)
    quad = np.minimum(absr, delta)
    lin = absr - quad
    return float(np.mean(0.5 * quad**2 + delta * lin))

def pinball_loss(a: pd.Series, p: pd.Series, q=0.5):
    a, p = _align(a, p)
    e = (a - p).values
    return float(np.mean(np.maximum(q*e, (q-1)*e)))

def horizon_rmse(a: pd.Series, p: pd.Series, H: int):
    a, p = _align(a, p)
    n = len(a) - H
    if n <= 0: return np.nan
    return float(np.sqrt(np.mean([(a.iloc[i+H]-p.iloc[i+H])**2 for i in range(n)])))

def returns_metrics(price_true: pd.Series, price_pred: pd.Series):
    rt = np.log(price_true).diff().dropna()
    rp = np.log(price_pred).diff().dropna()
    rt, rp = _align(rt, rp)
    rmse = np.sqrt(mean_squared_error(rt, rp))
    mae  = mean_absolute_error(rt, rp)
    r2   = r2_score(rt, rp)
    da   = float((np.sign(rt) == np.sign(rp)).mean())
    return {"Returns RMSE": rmse, "Returns MAE": mae, "Returns R2": r2, "Directional Accuracy": da}

def evaluate_price_metrics(price_true: pd.Series, price_pred: pd.Series, season_m=5):
    a, p = _align(price_true, price_pred)
    metrics = {
        "Price RMSE": np.sqrt(mean_squared_error(a, p)),
        "Price MAE": mean_absolute_error(a, p),
        "Price MedAE": median_absolute_error(a, p),
        "Price R2": r2_score(a, p),
        "Price MAPE (%)": mape(a, p),
        "Price sMAPE (%)": smape(a, p),
        "Price MASE (m=5)": mase(a, p, m=season_m),
        "Price Theil U1": theil_u1(a, p),
        "Price Huber(δ=1)": huber_loss(a, p, delta=1.0),
        "Pinball q=0.5": pinball_loss(a, p, q=0.5),
        "Pinball q=0.9": pinball_loss(a, p, q=0.9),
        "Price RMSE @1": horizon_rmse(a, p, 1),
        "Price RMSE @5": horizon_rmse(a, p, 5),
        "Price RMSE @20": horizon_rmse(a, p, 20),
    }
    return {k: (float(v) if v is not None else np.nan) for k, v in metrics.items()}

# ---------- 3) Split helper ----------
def train_test_split_by_time(y, X, test_size=0.2):
    n_test = int(len(y) * test_size)
    return y.iloc[:-n_test], y.iloc[-n_test:], X.loc[y.index[:-n_test]], X.loc[y.index[-n_test:]]

# ---------- 4) Run periods ----------
all_rows = []

for label, cfg in periods.items():
    start, end = cfg["range"]
    order, seas = cfg["order"], cfg["seasonal_order"]

    y = y_all.loc[start:end].copy()
    X = X_all.loc[start:end].copy()
    P = price.loc[y.index]

    if len(y) < 300:
        print(f"[{label}] Not enough data. Skipping.")
        continue

    y_tr, y_te, X_tr, X_te = train_test_split_by_time(y, X, test_size=0.2)

    mod = SARIMAX(y_tr, exog=X_tr, order=order, seasonal_order=seas,
                  trend="n", enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(disp=False)
    print(f"[{label}] SARIMAX{order}x{seas} fit (AIC={res.aic:.2f})")

    # Forecast on log-scale then exponentiate
    start_pos = len(y_tr)
    end_pos   = len(y_tr) + len(y_te) - 1
    log_fore  = res.get_prediction(start=start_pos, end=end_pos, exog=X_te, dynamic=True).predicted_mean
    log_fore.index = y_te.index
    price_fore = np.exp(log_fore)

    # Plot
    plt.figure(figsize=(12,5))
    plt.plot(P.loc[y_tr.index], label="Train", color="tab:blue")
    plt.plot(P.loc[y_te.index], label="Test - Actual", color="black")
    plt.plot(price_fore, label="Test - Forecast", color="red")
    plt.title(f"{label} — SARIMAX{order}x{seas} — Train/Test Price")
    plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.tight_layout(); plt.show()

    # Metrics
    actual_p = P.loc[y_te.index]
    pred_p   = price_fore

    ret_stats   = returns_metrics(actual_p, pred_p)
    price_stats = evaluate_price_metrics(actual_p, pred_p, season_m=5)

    row = {"Period": label, "Model": "SARIMAX", "SARIMAX Order": f"{order} x {seas}", **ret_stats, **price_stats}
    all_rows.append(row)

# ---------- 5) Summary (styled view + Excel) ----------
summary = pd.DataFrame(all_rows)
pd.options.display.float_format = "{:,.6f}".format
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

print("\n=== SARIMAX-only — Period-wise Evaluation (extended metrics) ===\n")

styled = (
    summary.style
    .set_table_attributes('style="border-collapse:collapse; border:1px solid #ddd; font-family:Arial;"')
    .set_table_styles([
        {'selector':'th','props':[('background-color','#003366'),('color','white'),('font-weight','bold'),
                                  ('text-align','center'),('padding','6px')]},
        {'selector':'td','props':[('border','1px solid #ddd'),('padding','6px'),('text-align','center')]},
        {'selector':'tr:nth-child(even)','props':[('background-color','#f5f9ff')]}
    ])
    .set_caption("SARIMAX-only — Period-wise Evaluation (extended metrics)")
)
display(HTML(styled.to_html()))

# Export full table to Excel
excel_path = "SARIMAX_only_Results.xlsx"
try:
    summary.to_excel(excel_path, index=False)
    print(f"✅ Full results exported to: {excel_path}")
except ModuleNotFoundError:
    print("⚠️ Install 'openpyxl' to enable Excel export:  pip install openpyxl  (then restart kernel)")

