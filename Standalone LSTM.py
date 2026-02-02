#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =========================
# LSTM (mean on returns) — Period-wise
# Contexted TEST sequences + Rich Metrics + Styled/Excel Summary
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from IPython.display import display, HTML

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

import warnings
warnings.filterwarnings("ignore")

# ---------- 0) Utils ----------
def make_sequences(X, y, index, L):
    """Create rolling sequences of length L to predict the next step."""
    Xs, ys, idxs = [], [], []
    for i in range(L, len(X)):
        Xs.append(X[i-L:i, :])
        ys.append(y[i])           # predict the next return
        idxs.append(index[i])     # target index (prediction date)
    return np.array(Xs), np.array(ys), pd.DatetimeIndex(idxs)

def rebuild_price_from_returns(yhat_ret, price_series, idx):
    """Rebuild price path from predicted returns for the test horizon."""
    yhat_ret = pd.Series(yhat_ret, index=idx)
    P0 = price_series.loc[:idx[0]].iloc[-1]      # last price at/before first test date
    return P0 * np.exp(yhat_ret.cumsum())

# ---------- 0b) Rich metrics (same family as other models) ----------
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

# ---------- 1) Load & base prep ----------
csv_path = "Final_Stock_prices_and_macroeconomic_data.csv"
df = pd.read_csv(csv_path, parse_dates=['Date']).sort_values('Date').set_index('Date')
df = df.apply(pd.to_numeric, errors='ignore')

price   = df['Price'].astype(float)
returns = np.log(price).diff().dropna(); returns.name = 'ret'

# Use all non-Price columns as features
feature_cols = [c for c in df.columns if c != 'Price']
X_all = df[feature_cols].loc[returns.index].copy()

# (Optional) lagged macros — keep commented if not needed
# for col in ['Inflation rate(%)', 'Interest rate(%)', 'Exchange rate']:
#     if col in X_all.columns:
#         X_all[f'{col}_lag120'] = X_all[col].shift(120)
# X_all = X_all.dropna()
# returns = returns.loc[X_all.index]
# price   = price.loc[X_all.index]

# ---------- 2) Periods ----------
periods = {
    "Pre-COVID":  ("2009-07-08", "2022-06-30"),
    "COVID":      ("2022-07-01", "2023-12-31"),
    "Post-COVID": ("2024-01-01", "2025-06-30"),
}

# ---------- 3) Train per period ----------
lookback = 60
all_rows = []

for label, (start_date, end_date) in periods.items():
    Xp = X_all.loc[start_date:end_date].copy()
    rp = returns.loc[start_date:end_date].copy()
    pp = price.loc[start_date:end_date].copy()

    if len(rp) < lookback + 200:
        print(f"[{label}] Not enough data after slicing. Skipping.")
        continue

    # Split 80/20 (no shuffle)
    X_train_df, X_test_df, y_train_srs, y_test_srs = train_test_split(
        Xp, rp, test_size=0.2, shuffle=False
    )

    # Keep raw price for plotting
    price_train = pp.loc[X_train_df.index.min():X_train_df.index.max()]
    price_test_actual = pp.loc[X_test_df.index.min():X_test_df.index.max()]

    # Scale X; standardize y
    scalerX = MinMaxScaler()
    X_train = scalerX.fit_transform(X_train_df)
    X_test  = scalerX.transform(X_test_df)

    yscaler = StandardScaler()
    y_train = yscaler.fit_transform(y_train_srs.values.reshape(-1,1)).ravel()
    y_test  = yscaler.transform(y_test_srs.values.reshape(-1,1)).ravel()

    # ---------- (A) TRAIN sequences ----------
    Xtr3, ytr, idx_tr = make_sequences(X_train, y_train, X_train_df.index, lookback)

    # ---------- (B) TEST sequences with boundary context ----------
    Xte_full = np.vstack([X_train[-lookback:], X_test])
    yte_full = np.hstack([y_train[-lookback:], y_test])
    idx_te_full = X_train_df.index[-lookback:].append(X_test_df.index)

    Xte3_all, yte_all, idx_te_all = make_sequences(Xte_full, yte_full, idx_te_full, lookback)
    mask = idx_te_all.isin(X_test_df.index)
    Xte3, yte, idx_te = Xte3_all[mask], yte_all[mask], idx_te_all[mask]

    if len(idx_te) == 0:
        print(f"[{label}] No test sequences after masking. Skipping.")
        continue

    # ---------- (C) Fit LSTM ----------
    K.clear_session()
    model = Sequential([
        LSTM(64, input_shape=(Xtr3.shape[1], Xtr3.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    _ = model.fit(Xtr3, ytr, validation_split=0.1, epochs=50, batch_size=32,
                  callbacks=[es], verbose=0)

    # ---------- (D) Predict standardized returns → raw → clip ----------
    yhat_test_std = model.predict(Xte3, verbose=0).ravel()
    yhat_test_raw = yscaler.inverse_transform(yhat_test_std.reshape(-1,1)).ravel()
    std_tr = y_train_srs.std()
    yhat_test_raw = np.clip(yhat_test_raw, -3.5*std_tr, 3.5*std_tr)

    # ---------- (E) Rebuild predicted price over the test window ----------
    price_pred = rebuild_price_from_returns(yhat_test_raw, pp, idx_te)

    # ---------- (F) Plot ----------
    plt.figure(figsize=(12,5))
    plt.plot(price_train.index, price_train.values, label='Train', color='tab:blue')
    plt.plot(price_test_actual.index, price_test_actual.values, label='Test - Actual', color='black')
    plt.plot(price_pred.index, price_pred.values, label='Test - Forecast', color='red')
    plt.title(f"{label} — Train/Test Forecast Evaluation (LSTM mean)")
    plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.tight_layout(); plt.show()

    # ---------- (G) Rich metrics ----------
    # Returns (from standardized back to raw already handled above)
    y_test_true_raw = yscaler.inverse_transform(yte.reshape(-1,1)).ravel()
    y_true_ret = pd.Series(y_test_true_raw, index=idx_te)
    y_pred_ret = pd.Series(yhat_test_raw,  index=idx_te)

    # Returns metrics
    rmse_ret = np.sqrt(mean_squared_error(y_true_ret, y_pred_ret))
    mae_ret  = mean_absolute_error(y_true_ret, y_pred_ret)
    da       = (np.sign(y_true_ret) == np.sign(y_pred_ret)).mean()

    # Price metrics
    actual_p = price_test_actual.reindex(price_pred.index)
    pred_p   = price_pred.reindex(actual_p.index)

    ret_stats_lib = returns_metrics(actual_p, pred_p)
    price_stats   = evaluate_price_metrics(actual_p, pred_p, season_m=5)

    # single-row summary for this period
    all_rows.append({
        "Period": label,
        "Model": "LSTM-only",
        "Returns RMSE (seq)": rmse_ret,            # direct from returns targets
        "Returns MAE (seq)": mae_ret,
        "Directional Accuracy (seq)": da,
        **ret_stats_lib,                           # returns metrics via price series
        **price_stats                              # price metrics
    })

# ---------- 4) Summary table (styled + Excel) ----------
summary = pd.DataFrame(all_rows)
pd.options.display.float_format = '{:,.6f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

print("\n=== LSTM-only (mean on returns) — Period-wise Evaluation (extended metrics) ===\n")

styled = (
    summary.style
    .set_table_attributes('style="border-collapse:collapse; border:1px solid #ddd; font-family:Arial;"')
    .set_table_styles([
        {'selector':'th','props':[('background-color','#003366'),('color','white'),
                                  ('font-weight','bold'),('text-align','center'),('padding','6px')]},
        {'selector':'td','props':[('border','1px solid #ddd'),('padding','6px'),
                                  ('text-align','center')]},
        {'selector':'tr:nth-child(even)','props':[('background-color','#f5f9ff')]}
    ])
    .set_caption("LSTM-only (returns → price) — Period-wise Evaluation (extended metrics)")
)
display(HTML(styled.to_html()))

# Export to Excel
excel_path = "LSTM_only_Results.xlsx"
try:
    summary.to_excel(excel_path, index=False)
    print(f"✅ Full results exported to: {excel_path}")
except ModuleNotFoundError:
    print("⚠️ Install 'openpyxl' to enable Excel export:  pip install openpyxl  (then restart kernel)")

