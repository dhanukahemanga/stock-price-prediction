#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =========================
# Hybrid SARIMAX + LSTM (Residual) — Period wise Evaluation
# Residual Diagnostics + Rolling-Origin CV + Metrics + Styled/Excel Summary
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from IPython.display import display, HTML

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

import warnings
warnings.filterwarnings("ignore")

# ---------- 0) Load and prepare data ----------
csv_path = "Final_Stock_prices_and_macroeconomic_data.csv"
df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
df = df.apply(pd.to_numeric, errors="ignore")

price = df["Price"].astype(float)
logp = np.log(price).rename("log_price")

# Only inflation and interest rate (120-day lag)
macro_cols = ["Inflation rate(%)", "Interest rate(%)"]
def make_exog_lags(frame, cols, lags=(120,)):
    X = pd.DataFrame(index=frame.index)
    for c in cols:
        for L in lags:
            X[f"{c}_lag{L}"] = frame[c].shift(L)
    return X

X_macros = make_exog_lags(df, macro_cols, lags=(120,))
full = pd.concat([logp, X_macros], axis=1).dropna()
y_all = full["log_price"]
X_all = full.drop(columns=["log_price"])

# ---------- 1) Periods + SARIMAX orders ----------
periods = {
    "Pre-COVID":  {"range": ("2009-07-08", "2022-06-30"), "order": (1,1,2), "seasonal_order": (0,1,1,5)},
    "COVID":      {"range": ("2022-07-01", "2023-12-31"), "order": (1,1,0), "seasonal_order": (0,1,1,5)},
    "Post-COVID": {"range": ("2024-01-01", "2025-06-30"), "order": (1,1,1), "seasonal_order": (1,0,0,5)},
}

# ---------- 2) Helper (split, windows, diag, metrics) ----------
def train_test_split_by_time(y, X, test_size=0.2):
    n_test = int(len(y) * test_size)
    y_tr, y_te = y.iloc[:-n_test], y.iloc[-n_test:]
    X_tr, X_te = X.loc[y_tr.index], X.loc[y_te.index]
    return y_tr, y_te, X_tr, X_te

def make_residual_windows(resid, lookback=60, extra_feats=None):
    idx = resid.index
    R = resid.values.reshape(-1, 1)
    Xs, ys, idxs = [], [], []
    for i in range(lookback, len(resid)):
        r_win = R[i - lookback : i, :]
        if extra_feats is not None:
            ef_win = extra_feats.loc[idx[i - lookback : i]].values
            Xs.append(np.hstack([r_win, ef_win]))
        else:
            Xs.append(r_win)
        ys.append(R[i, 0])
        idxs.append(idx[i])
    return np.array(Xs), np.array(ys), pd.DatetimeIndex(idxs)

def rebuild_price_from_log(log_series):
    return np.exp(log_series)

def horizon_rmse(a: pd.Series, p: pd.Series, H: int):
    common = a.index.intersection(p.index)
    a, p = a.loc[common], p.loc[common]
    n = len(a) - H
    if n <= 0: return np.nan
    return float(np.sqrt(np.mean([(a.iloc[i+H]-p.iloc[i+H])**2 for i in range(n)])))

def residual_diagnostics(residuals, label):
    std_resid = (residuals - residuals.mean()) / residuals.std()
    plt.figure(figsize=(10,3))
    plt.plot(std_resid, color='tab:blue')
    plt.title(f"{label} — SARIMAX Residuals")
    plt.axhline(0, color='black', lw=1)
    plt.tight_layout(); plt.show()
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    plot_acf(std_resid, ax=ax[0], lags=40, title=f"{label} — ACF of Residuals")
    plot_pacf(std_resid, ax=ax[1], lags=40, title=f"{label} — PACF of Residuals", method='ywm')
    plt.tight_layout(); plt.show()
    lb = acorr_ljungbox(std_resid, lags=[10, 20, 30], return_df=True)
    print(f"\n{label} — Ljung–Box Q-test Results")
    print(lb)
    print("✅ Residuals likely white noise." if (lb["lb_pvalue"] > 0.05).all()
          else "⚠️ Residual autocorrelation remains; consider order tweaks.")

# ----- Rich metrics (same as SARIMAX-only script) -----
def _align(a: pd.Series, p: pd.Series):
    common = a.index.intersection(p.index)
    return a.loc[common].astype(float), p.loc[common].astype(float)

def mape(a, p, eps=1e-8):
    a, p = _align(a, p); return float(100*np.mean(np.abs((a - p) / (a.replace(0, eps)))))

def smape(a, p, eps=1e-8):
    a, p = _align(a, p); return float(100*np.mean(2*np.abs(p - a) / (np.abs(a) + np.abs(p) + eps)))

def mase(a, p, m=5):
    a, p = _align(a, p); mae = np.mean(np.abs(a - p))
    if len(a) <= m: return np.nan
    denom = np.mean(np.abs(a.values[m:] - a.values[:-m]))
    return float(mae / (denom + 1e-12))

def theil_u1(a, p):
    a, p = _align(a, p)
    num = np.sqrt(np.mean((p - a) ** 2))
    denom = np.sqrt(np.mean(a ** 2)) + np.sqrt(np.mean(p ** 2))
    return float(num / (denom + 1e-12))

def huber_loss(a, p, delta=1.0):
    a, p = _align(a, p)
    r = (a - p).values; absr = np.abs(r)
    quad = np.minimum(absr, delta); lin = absr - quad
    return float(np.mean(0.5 * quad**2 + delta * lin))

def pinball_loss(a, p, q=0.5):
    a, p = _align(a, p); e = (a - p).values
    return float(np.mean(np.maximum(q*e, (q-1)*e)))

def returns_metrics(price_true: pd.Series, price_pred: pd.Series):
    rt = np.log(price_true).diff().dropna()
    rp = np.log(price_pred).diff().dropna()
    rt, rp = _align(rt, rp)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    return {
        "Returns RMSE": float(np.sqrt(mean_squared_error(rt, rp))),
        "Returns MAE": float(mean_absolute_error(rt, rp)),
        "Returns R2": float(r2_score(rt, rp)),
        "Directional Accuracy": float((np.sign(rt) == np.sign(rp)).mean())
    }

def evaluate_price_metrics(price_true: pd.Series, price_pred: pd.Series, season_m=5):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
    a, p = _align(price_true, price_pred)
    return {
        "Price RMSE": float(np.sqrt(mean_squared_error(a, p))),
        "Price MAE": float(mean_absolute_error(a, p)),
        "Price MedAE": float(median_absolute_error(a, p)),
        "Price R2": float(r2_score(a, p)),
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

# ---------- 2b) Rolling-origin CV for LSTM on residuals ----------
def build_lstm(units, input_shape, dropout):
    model = Sequential([LSTM(units, input_shape=input_shape), Dropout(dropout), Dense(1)])
    model.compile(optimizer="adam", loss="mse"); return model

def make_windows_for_all(resid_std_srs, extra_feats_df, lookback):
    idx = resid_std_srs.index; R = resid_std_srs.values.reshape(-1,1)
    Xs, ys, idxs = [], [], []
    for i in range(lookback, len(resid_std_srs)):
        r_win = R[i-lookback:i, :]; ef_win = extra_feats_df.loc[idx[i-lookback:i]].values
        Xs.append(np.hstack([r_win, ef_win])); ys.append(R[i,0]); idxs.append(idx[i])
    return np.array(Xs), np.array(ys), pd.DatetimeIndex(idxs)

def rolling_origin_folds(index, lookback, val_len=60, min_train_extra=100, max_folds=4):
    n = len(index); start_i = max(lookback + min_train_extra, lookback + val_len)
    folds=[]; i=start_i
    while i + val_len <= n and len(folds) < max_folds:
        folds.append((index[i-1], index[i+val_len-1])); i += val_len
    return folds

from sklearn.metrics import mean_squared_error
def cv_score_lstm(resid_std_srs, extra_feats_df, lookback, units, dropout,
                  batch_size=32, epochs=60, val_len=60, patience=5, max_folds=4, verbose=0):
    X_all, y_all, idx_all = make_windows_for_all(resid_std_srs, extra_feats_df, lookback)
    folds = rolling_origin_folds(idx_all, lookback, val_len=val_len, max_folds=max_folds)
    if len(folds) == 0: return np.inf
    rmses=[]
    for (train_end_ts, val_end_ts) in folds:
        msk_tr = idx_all <= train_end_ts; msk_va = (idx_all > train_end_ts) & (idx_all <= val_end_ts)
        Xtr, ytr, Xva, yva = X_all[msk_tr], y_all[msk_tr], X_all[msk_va], y_all[msk_va]
        mdl = build_lstm(units, (Xtr.shape[1], Xtr.shape[2]), dropout)
        es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        mdl.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch_size,
                callbacks=[es], verbose=verbose)
        yhat = mdl.predict(Xva, verbose=0).ravel()
        rmses.append(float(np.sqrt(mean_squared_error(yva, yhat))))
    return float(np.mean(rmses))

def tune_lstm_hyperparams(resid_std_srs, extra_feats_df,
                          lookbacks=(30,60,90), units_list=(32,64),
                          dropouts=(0.2,0.3), batch_sizes=(32,), epochs_list=(60,),
                          val_len=60, verbose=0):
    best={"score":np.inf}
    for L in lookbacks:
        for units in units_list:
            for dp in dropouts:
                for bs in batch_sizes:
                    for ep in epochs_list:
                        score = cv_score_lstm(resid_std_srs, extra_feats_df, L, units, dp,
                                              batch_size=bs, epochs=ep, val_len=val_len, verbose=verbose)
                        print(f"[CV] lookback={L}, units={units}, dropout={dp}, batch={bs}, epochs={ep} -> RMSE={score:.6f}")
                        if score < best["score"]:
                            best={"score":score,"lookback":L,"units":units,"dropout":dp,"batch_size":bs,"epochs":ep}
    return best["score"], best

def fit_final_lstm(resid_std_srs, extra_feats_df, best_cfg):
    L = best_cfg["lookback"]
    X_all, y_all, _ = make_windows_for_all(resid_std_srs, extra_feats_df, L)
    mdl = build_lstm(best_cfg["units"], (X_all.shape[1], X_all.shape[2]), best_cfg["dropout"])
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    mdl.fit(X_all, y_all, validation_split=0.1, epochs=best_cfg["epochs"],
            batch_size=best_cfg["batch_size"], callbacks=[es], verbose=0)
    return mdl, L

# ---------- 3) Main loop ----------
default_lookback = 60  # just for min-length check; tuned lookback replaces it
all_rows = []

# 🔹 NEW: container for EGARCH plotting in a separate cell
egarch_payloads = []

for label, cfg in periods.items():
    start, end = cfg["range"]
    order, seasonal_order = cfg["order"], cfg["seasonal_order"]

    y = y_all.loc[start:end].copy()
    X = X_all.loc[start:end].copy()
    P = price.loc[y.index]

    if len(y) < default_lookback + 200:
        print(f"[{label}] Not enough data. Skipping."); continue

    # Split train/test
    y_tr, y_te, X_tr, X_te = train_test_split_by_time(y, X, test_size=0.2)

    # ---- (A) Fit SARIMAX ----
    sar_mod = SARIMAX(y_tr, exog=X_tr, order=order, seasonal_order=seasonal_order,
                      trend="n", enforce_stationarity=False, enforce_invertibility=False)
    sar_res = sar_mod.fit(disp=False)
    print(f"[{label}] SARIMAX{order} fit completed (AIC={sar_res.aic:.2f})")

    # ---- (B) Residuals ----
    fitted_tr = sar_res.get_prediction(start=0, end=len(y_tr)-1, exog=X_tr).predicted_mean
    resid_tr = (y_tr - fitted_tr).dropna()

    # ---- Residual diagnostics ----
    residual_diagnostics(resid_tr, label)

    # ---- (C) Prepare residuals & features for LSTM ----
    yscaler = StandardScaler()
    resid_tr_std = pd.Series(yscaler.fit_transform(resid_tr.values.reshape(-1, 1)).ravel(), index=resid_tr.index)

    extra_feats_tr = X_tr.loc[resid_tr_std.index].copy()
    xscaler = MinMaxScaler()
    extra_feats_tr[:] = xscaler.fit_transform(extra_feats_tr)

    # ---- (D) Tune LSTM via rolling-origin CV ----
    cv_rmse, best_cfg = tune_lstm_hyperparams(
        resid_std_srs=resid_tr_std, extra_feats_df=extra_feats_tr,
        lookbacks=(30,60,90), units_list=(32,64),
        dropouts=(0.2,0.3), batch_sizes=(32,), epochs_list=(60,),
        val_len=60, verbose=0
    )
    print(f"\n[{label}] Best CV config: {best_cfg}  (mean RMSE={cv_rmse:.6f})")

    # Fit final LSTM with best hyperparams
    K.clear_session()
    model, lookback = fit_final_lstm(resid_tr_std, extra_feats_tr, best_cfg)

    # ---- (E) Forecast TEST ----
    start_pos = len(y_tr); end_pos = len(y_tr) + len(y_te) - 1
    sar_fore_te = sar_res.get_prediction(start=start_pos, end=end_pos, exog=X_te, dynamic=True).predicted_mean
    sar_fore_te.index = y_te.index

    resid_ctx_std = resid_tr_std.copy()
    extra_feats_te = X_te.copy(); extra_feats_te[:] = xscaler.transform(extra_feats_te)
    extra_full = pd.concat([extra_feats_tr.iloc[-lookback:], extra_feats_te], axis=0)

    preds_resid_std = []
    for t in y_te.index:
        r_win = resid_ctx_std.iloc[-lookback:].values.reshape(lookback, 1)
        ef_win = extra_full.iloc[-lookback:].values
        if ef_win.shape[0] != lookback:
            ef_win = np.pad(ef_win, ((lookback - ef_win.shape[0], 0), (0, 0)), mode='edge')
        X_win = np.hstack([r_win, ef_win]).reshape(1, lookback, -1)
        rhat_std = model.predict(X_win, verbose=0).ravel()[0]
        preds_resid_std.append(rhat_std)
        resid_ctx_std = pd.concat([resid_ctx_std, pd.Series(rhat_std, index=[t])])
        extra_full = pd.concat([extra_full, extra_feats_te.loc[[t]]])

    preds_resid_raw = yscaler.inverse_transform(np.array(preds_resid_std).reshape(-1, 1)).ravel()
    resid_fore_te = pd.Series(preds_resid_raw, index=y_te.index)

    log_fore_hybrid = sar_fore_te + resid_fore_te
    price_fore_hybrid = rebuild_price_from_log(log_fore_hybrid)

    # 🔹 NEW: Save artifacts for separate EGARCH plotting cell
    egarch_payloads.append({
        "label": label,
        "train_logret": np.log(price.loc[y_tr.index]).diff().dropna(),
        "pred_log_price_test": log_fore_hybrid.copy(),
        "test_index": y_te.index,
        "actual_price_test": price.loc[y_te.index].copy()
    })

    # ---- (F) Plot (unchanged)
    plt.figure(figsize=(12,5))
    plt.plot(price.loc[y_tr.index], label="Train", color="tab:blue")
    plt.plot(price.loc[y_te.index], label="Test - Actual", color="black")
    plt.plot(price_fore_hybrid, label="Test - Hybrid Forecast", color="red")
    plt.title(f"{label} — Hybrid SARIMAX{order} + Tuned LSTM Residual — Train/Test Price")
    plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.tight_layout(); plt.show()

    # ---- (G) Rich metrics
    actual_p = price.loc[y_te.index]
    pred_p   = price_fore_hybrid
    ret_stats   = returns_metrics(actual_p, pred_p)
    price_stats = evaluate_price_metrics(actual_p, pred_p, season_m=5)

    all_rows.append({
        "Period": label,
        "Model": "Hybrid",
        "SARIMAX Order": f"{order} x {seasonal_order}",
        "Best LSTM cfg": str(best_cfg),
        **ret_stats, **price_stats
    })

# ---------- 5) Styled summary + Excel ----------
summary = pd.DataFrame(all_rows)
pd.options.display.float_format = "{:,.6f}".format
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

print("\n=== Hybrid SARIMAX + Tuned LSTM (Residual) — Period-wise Evaluation (extended metrics) ===\n")

styled = (
    summary.style
    .set_table_attributes('style="border-collapse:collapse; border:1px solid #ddd; font-family:Arial;"')
    .set_table_styles([
        {'selector':'th','props':[('background-color','#003366'),('color','white'),('font-weight','bold'),
                                  ('text-align','center'),('padding','6px')]},
        {'selector':'td','props':[('border','1px solid #ddd'),('padding','6px'),('text-align','center')]},
        {'selector':'tr:nth-child(even)','props':[('background-color','#f5f9ff')]}
    ])
    .set_caption("Hybrid SARIMAX + Tuned LSTM (Residual) — Period-wise Evaluation (extended metrics)")
)
display(HTML(styled.to_html()))

excel_path = "Hybrid_SARIMAX_LSTM_Results.xlsx"
try:
    summary.to_excel(excel_path, index=False)
    print(f"✅ Full results exported to: {excel_path}")
except ModuleNotFoundError:
    print(" Install 'openpyxl' to enable Excel export:  pip install openpyxl")

