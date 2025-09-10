# rc_2r2c_solve_ivp_benchmark_mlp.py
import io, requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -----------------------------
# 1) Load data
# -----------------------------
RAW_URL = "https://raw.githubusercontent.com/srouchier/buildingenergygeeks/master/data/statespace.csv"
r = requests.get(RAW_URL, timeout=30)
r.raise_for_status()
df = pd.read_csv(io.StringIO(r.text))

cols = ["Time", "T_ext", "P_hea", "I_sol", "T_int"]
missing = set(cols) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# time (hours) for conditioning
t_s  = df["Time"].to_numpy()                 # [s]
t    = t_s / 3600.0                          # [h]
To   = df["T_ext"].to_numpy()
PhiH = df["P_hea"].to_numpy()                # [W]
PhiS = df["I_sol"].to_numpy()                # [W/m^2]
Ti_meas = df["T_int"].to_numpy()

# -----------------------------
# 2) Train/Test windows (same dataset)
# -----------------------------
train_mask = (t >= 10) & (t <= 40)
test_mask  = (t >= 50) & (t <= 90)

# -----------------------------
# 3) Interpolants (continuous inputs, full series)
# -----------------------------
interp_kind = "linear"
f_To   = interp1d(t, To,   kind=interp_kind, fill_value="extrapolate", assume_sorted=True)
f_PhiH = interp1d(t, PhiH, kind=interp_kind, fill_value="extrapolate", assume_sorted=True)
f_PhiS = interp1d(t, PhiS, kind=interp_kind, fill_value="extrapolate", assume_sorted=True)

# -----------------------------
# 4) 2R2C model (Ri, Ro, Ci, Ce) with heater efficiency kH and solar gains Ai, Ae
#    NOTE: time base is hours -> multiply physical RHS (in 1/s) by 3600
# -----------------------------
def rhs(t_h, y, Ri, Ro, Ci, Ce, Ai, Ae, kH, f_To_, f_PhiH_, f_PhiS_):
    Ti, Te = y
    to   = f_To_(t_h)
    phiH = f_PhiH_(t_h) * kH
    phiS = f_PhiS_(t_h)
    dTi_dt_s = ( (Te - Ti)/Ri + phiH + Ai*phiS ) / Ci
    dTe_dt_s = ( (Ti - Te)/Ri + (to - Te)/Ro + Ae*phiS ) / Ce
    return [3600.0 * dTi_dt_s, 3600.0 * dTe_dt_s]

def simulate_full(params, t_eval):
    Ri, Ro, Ci, Ce, Ai, Ae, kH, Ti0, Te0 = params
    sol = solve_ivp(
        fun=lambda tau, y: rhs(tau, y, Ri, Ro, Ci, Ce, Ai, Ae, kH, f_To, f_PhiH, f_PhiS),
        t_span=(t_eval[0], t_eval[-1]),
        y0=[Ti0, Te0],
        t_eval=t_eval,
        method="RK45",
        atol=1e-6, rtol=1e-6,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.y  # [Ti, Te]

# -----------------------------
# 5) Fit RC on TRAIN window only (robust)
# -----------------------------
t_tr   = t[train_mask]
To_tr  = To[train_mask]
PhiH_tr= PhiH[train_mask]
PhiS_tr= PhiS[train_mask]
Ti_tr  = Ti_meas[train_mask]

# local interpolants on train window for fitting
f_To_tr   = interp1d(t_tr, To_tr,   kind="linear", fill_value="extrapolate", assume_sorted=True)
f_PhiH_tr = interp1d(t_tr, PhiH_tr, kind="linear", fill_value="extrapolate", assume_sorted=True)
f_PhiS_tr = interp1d(t_tr, PhiS_tr, kind="linear", fill_value="extrapolate", assume_sorted=True)

def simulate_train(params):
    Ri, Ro, Ci, Ce, Ai, Ae, kH, Ti0, Te0 = params
    sol = solve_ivp(
        fun=lambda tau, y: rhs(tau, y, Ri, Ro, Ci, Ce, Ai, Ae, kH, f_To_tr, f_PhiH_tr, f_PhiS_tr),
        t_span=(t_tr[0], t_tr[-1]),
        y0=[Ti0, Te0],
        t_eval=t_tr,
        method="RK45", atol=1e-6, rtol=1e-6
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp (train) failed: {sol.message}")
    return sol.y  # [Ti, Te]

def residuals_train(p):
    Ti_sim, _ = simulate_train(p)
    w = np.ones_like(Ti_tr)
    w[: max(1, int(0.05*len(w)))] = 1.5  # pin early transient a bit
    return w * (Ti_sim - Ti_tr)

# Initial guesses & bounds
p0 = np.array([
    0.02,   # Ri [K/W]
    0.05,   # Ro [K/W]
    5e6,    # Ci [J/K]
    2e7,    # Ce [J/K]
    0.20,   # Ai
    0.05,   # Ae
    0.90,   # kH
    Ti_tr[0],                            # Ti0
    (To_tr[0]*0.6 + Ti_tr[0]*0.4)        # Te0
], dtype=float)

lb = np.array([1e-4, 1e-4, 1e5, 1e5, 0.0, 0.0, 0.0, Ti_tr.min()-5.0,
               min(To_tr.min(), Ti_tr.min())-10.0])
ub = np.array([1.0,  1.0,  1e9, 1e9, 10.0,10.0, 1.5, Ti_tr.max()+5.0,
               max(To_tr.max(), Ti_tr.max())+10.0])

res = least_squares(
    residuals_train, p0, bounds=(lb, ub),
    loss="huber", f_scale=0.5, max_nfev=400, verbose=0
)
p_hat = res.x
print("\nFitted [Ri, Ro, Ci, Ce, Ai, Ae, kH, Ti0, Te0]:\n", p_hat)

# Simulate RC over FULL timeline
Ti_rc_full, Te_rc_full = simulate_full(p_hat, t)

# -----------------------------
# 6) Stronger black-box baselines
#    (A) Ridge ARX (with lags on output and inputs)
#    (B) MLP ARX (scaled, early stopping)
# -----------------------------
dt_h = np.median(np.diff(t))
lag_steps = max(1, int(round(1.0/dt_h)))  # ~1h worth of steps

def make_supervised_arx(Ti, To, PhiH, PhiS, lags):
    X, y = [], []
    for k in range(lags, len(Ti)):
        feats = [Ti[k-j-1] for j in range(lags)]
        for j in range(lags):
            idx = k - j - 1
            feats.extend([To[idx], PhiH[idx], PhiS[idx]])
        X.append(feats)
        y.append(Ti[k])
    return np.asarray(X), np.asarray(y)

def arx_predict_full(model, Ti, To, PhiH, PhiS, lags, train_mask):
    n = len(Ti)
    y_hist = Ti.astype(float).copy()
    preds = np.full(n, np.nan, dtype=float)
    for k in range(lags, n):
        feats = [y_hist[k-j-1] for j in range(lags)]
        for j in range(lags):
            idx = k - j - 1
            feats.extend([To[idx], PhiH[idx], PhiS[idx]])
        y_hat = float(model.predict(np.asarray(feats)[None, :])[0])
        preds[k] = y_hat
        if not train_mask[k]:  # rollout only outside train
            y_hist[k] = y_hat
    return preds

# Prepare supervised training data in the train window
Ti_tr_arr   = Ti_meas[train_mask]
To_tr_arr   = To[train_mask]
PhiH_tr_arr = PhiH[train_mask]
PhiS_tr_arr = PhiS[train_mask]
X_tr_arx, y_tr_arx = make_supervised_arx(Ti_tr_arr, To_tr_arr, PhiH_tr_arr, PhiS_tr_arr, lag_steps)

# (A) Ridge ARX with scaling and CV over logspace alphas
ridge_arx = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 41)))
])
ridge_arx.fit(X_tr_arx, y_tr_arx)
Ti_ridge_full = arx_predict_full(ridge_arx, Ti_meas, To, PhiH, PhiS, lag_steps, train_mask)

# (B) MLP ARX with scaling and early stopping
mlp_arx = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu",
                          learning_rate_init=1e-3, early_stopping=True,
                          n_iter_no_change=20, validation_fraction=0.15,
                          max_iter=5000, random_state=0))
])
mlp_arx.fit(X_tr_arx, y_tr_arx)
Ti_mlp_full = arx_predict_full(mlp_arx, Ti_meas, To, PhiH, PhiS, lag_steps, train_mask)

# -----------------------------
# 7) Metrics (train / test)
# -----------------------------
def rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))

Ti_rc_tr   = Ti_rc_full[train_mask]
Ti_ridge_tr = Ti_ridge_full[train_mask]
Ti_mlp_tr  = Ti_mlp_full[train_mask]
Ti_rc_te   = Ti_rc_full[test_mask]
Ti_ridge_te  = Ti_ridge_full[test_mask]
Ti_mlp_te  = Ti_mlp_full[test_mask]
Ti_meas_tr = Ti_tr
Ti_meas_te = Ti_meas[test_mask]

rmse_rc_train   = rmse(Ti_rc_tr,  Ti_meas_tr)
rmse_ridge_train = rmse(Ti_ridge_tr[~np.isnan(Ti_ridge_tr)], Ti_meas_tr[~np.isnan(Ti_ridge_tr)])
rmse_mlp_train  = rmse(Ti_mlp_tr[~np.isnan(Ti_mlp_tr)], Ti_meas_tr[~np.isnan(Ti_mlp_tr)])

rmse_rc_test    = rmse(Ti_rc_te,  Ti_meas_te)
rmse_ridge_test = rmse(Ti_ridge_te[~np.isnan(Ti_ridge_te)], Ti_meas_te[~np.isnan(Ti_ridge_te)])
rmse_mlp_test   = rmse(Ti_mlp_te[~np.isnan(Ti_mlp_te)], Ti_meas_te[~np.isnan(Ti_mlp_te)])

print(f"\nRMSE (°C) — TRAIN [10–40 h]:  RC={rmse_rc_train:.3f},  Ridge ARX={rmse_ridge_train:.3f},  MLP ARX={rmse_mlp_train:.3f}")
print(f"RMSE (°C) — TEST  [50–90 h]:  RC={rmse_rc_test:.3f},  Ridge ARX={rmse_ridge_test:.3f},  MLP ARX={rmse_mlp_test:.3f}")

# -----------------------------
# 8) Closed-loop thermostat counterfactual
#    New policy: proportional heater to track setpoint (model-in-the-loop)
# -----------------------------
start_idx = np.where(test_mask)[0][0]
To0_cf = float(np.median(To[test_mask]))
Pmax = 2000.0
T_set = 22.0
kp = 600.0  # W/°C
t_horizon_h = 24.0
N_cf = int(round(t_horizon_h / dt_h)) + 1
t_cf = np.linspace(0.0, t_horizon_h, N_cf)

def const(val):
    return lambda tau, v=val: v

def heater_power(Ti):
    return float(np.clip(kp * (T_set - Ti), 0.0, Pmax))

# RC simulation with custom inputs
Ti0_cf = float(Ti_meas[start_idx])
Te0_cf = float(Te_rc_full[start_idx])
f_To_cf   = const(To0_cf)
f_PhiS_cf = const(0.0)
def simulate_rc_counterfactual(params, t_eval):
    Ri, Ro, Ci, Ce, Ai, Ae, kH, _, _ = params
    def rhs_cl(tau, y):
        Ti, Te = y
        return rhs(tau, [Ti, Te], Ri, Ro, Ci, Ce, Ai, Ae, kH,
                   f_To_cf,
                   lambda _tau: heater_power(Ti),
                   f_PhiS_cf)
    sol = solve_ivp(fun=rhs_cl,
                    t_span=(t_eval[0], t_eval[-1]), y0=[Ti0_cf, Te0_cf], t_eval=t_eval,
                    method="RK45", atol=1e-6, rtol=1e-6)
    if not sol.success:
        raise RuntimeError(f"solve_ivp (counterfactual) failed: {sol.message}")
    return sol.y[0]

Ti_rc_cf = simulate_rc_counterfactual(p_hat, t_cf)

To_arr_cf   = np.full_like(t_cf, To0_cf, dtype=float)
PhiS_arr_cf = np.zeros_like(t_cf)

def simulate_arx_counterfactual(model):
    preds = []
    y_hist = [Ti0_cf] * lag_steps
    for k in range(len(t_cf)):
        feats = []
        feats.extend([y_hist[-j-1] for j in range(lag_steps)])
        for j in range(lag_steps):
            idx = max(0, k - 1 - j)
            phiH_j = heater_power(preds[-1] if len(preds)>0 else Ti0_cf)
            feats.extend([To_arr_cf[idx], phiH_j, PhiS_arr_cf[idx]])
        y_hat = float(model.predict(np.asarray(feats)[None, :])[0])
        preds.append(y_hat)
        y_hist.append(y_hat)
    return np.asarray(preds)

Ti_ridge_cf = simulate_arx_counterfactual(ridge_arx)
Ti_mlp_cf   = simulate_arx_counterfactual(mlp_arx)

# -----------------------------
# 9) ONE combined figure (3 stacked subplots) + shading + step test
# -----------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 8.6), sharex=False)

# A) Inputs (heating & solar)
ax = axes[0]
ax.plot(t, PhiH, label="Heating power (W)")
ax.plot(t, PhiS, label="Solar irradiance (W/m²)")
ax.set_ylabel("Inputs")
ax.axvspan(10, 40, color="grey",   alpha=0.12, label="Train")
ax.axvspan(40, 90, color="orange", alpha=0.08, label="Test")
ax.axvline(50, color="k", linestyle=":", linewidth=1, alpha=0.6)
handles, labels = ax.get_legend_handles_labels()
ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), loc="upper right")

# B) Temperatures + fits
ax = axes[1]
ax.plot(t, Ti_meas, label="Measured $T_i$")
ax.plot(t, Ti_rc_full, "--", label="RC (solve_ivp)")
ax.plot(t, Ti_ridge_full, ":", label=f"Ridge ARX (lags={lag_steps})")
ax.plot(t, Ti_mlp_full, "-", linewidth=1, alpha=0.9, label=f"MLP ARX (lags={lag_steps})")
ax.axvspan(10, 40, color="grey",   alpha=0.12)
ax.axvspan(40, 90, color="orange", alpha=0.08)
ax.axvline(50, color="k", linestyle=":", linewidth=1, alpha=0.6)
title = (f"Train RMSE — RC: {rmse_rc_train:.2f}°C, Ridge ARX: {rmse_ridge_train:.2f}°C, MLP ARX: {rmse_mlp_train:.2f}°C   |   "
         f"Test RMSE — RC: {rmse_rc_test:.2f}°C, Ridge ARX: {rmse_ridge_test:.2f}°C, MLP ARX: {rmse_mlp_test:.2f}°C")
ax.set_title(title)
ax.set_ylabel("Indoor temperature [°C]")
ax.legend(loc="best")

ax = axes[2]
ax.plot(t_cf, Ti_rc_cf, "--", label="RC (thermostat)")
ax.plot(t_cf, Ti_ridge_cf, ":", label="Ridge ARX (thermostat)")
ax.plot(t_cf, Ti_mlp_cf, "-", alpha=0.9, label="MLP ARX (thermostat)")
ax.axhline(T_set, color="k", linestyle="--", linewidth=1, alpha=0.5, label="Setpoint")
ax.set_xlabel("Time [h]")
ax.set_ylabel("Indoor temperature [°C]")
ax.set_title(f"Closed-loop thermostat: T_set={T_set}°C, To={To0_cf:.1f}°C, Φ_S=0, P_max={int(Pmax)}W")
ax.legend(loc="best")

plt.tight_layout()
plt.show()
