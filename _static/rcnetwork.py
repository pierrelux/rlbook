# rc_2r2c_solve_ivp_benchmark_mlp.py
import io, requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

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
# 6) Black-box baselines
#    (A) Linear on inputs (train -> full predict)
#    (B) MLP with lags of Ti (train one-step; test rollout)
# -----------------------------
# (A) Linear
X_tr_lin = np.column_stack([To_tr, PhiH_tr, PhiS_tr])
lin = LinearRegression().fit(X_tr_lin, Ti_tr)
Ti_lin_full = lin.predict(np.column_stack([To, PhiH, PhiS]))

# (B) MLP with autoregressive lags of Ti
# choose ~1 hour of memory based on sampling
dt_h = np.median(np.diff(t))
lag_steps = max(1, int(round(1.0/dt_h)))  # ~1h worth of steps

def make_supervised(Ti, To, PhiH, PhiS, lags):
    X, y = [], []
    for k in range(lags, len(Ti)):
        xk = [Ti[k-j-1] for j in range(lags)] + [To[k-1], PhiH[k-1], PhiS[k-1]]
        X.append(xk); y.append(Ti[k])
    return np.asarray(X), np.asarray(y)

# Training set for MLP: within the train window
Ti_tr_arr  = Ti_meas[train_mask]
To_tr_arr  = To[train_mask]
PhiH_tr_arr= PhiH[train_mask]
PhiS_tr_arr= PhiS[train_mask]
X_tr_mlp, y_tr_mlp = make_supervised(Ti_tr_arr, To_tr_arr, PhiH_tr_arr, PhiS_tr_arr, lag_steps)

mlp = MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu",
                   max_iter=3000, random_state=0)
mlp.fit(X_tr_mlp, y_tr_mlp)

# Teacher-forced one-step predictions on TRAIN (to show tight fit)
Ti_mlp_full = np.full_like(Ti_meas, np.nan, dtype=float)
for k in range(np.where(train_mask)[0][0] + lag_steps, np.where(train_mask)[0][-1] + 1):
    hist = Ti_meas[k-lag_steps:k]  # teacher forcing inside train
    xk = np.r_[hist, To[k-1], PhiH[k-1], PhiS[k-1]]
    Ti_mlp_full[k] = mlp.predict(xk[None, :])[0]

# Rollout on TEST (no teacher forcing → shows poor generalization)
start_idx = np.where(test_mask)[0][0]
# seed with first lag_steps measured points of test
seed = Ti_meas[start_idx : start_idx + lag_steps].tolist()
Ti_roll = seed.copy()
for k in range(start_idx + lag_steps, np.where(test_mask)[0][-1] + 1):
    hist = Ti_roll[-lag_steps:]
    xk = np.r_[hist, To[k-1], PhiH[k-1], PhiS[k-1]]
    pred = mlp.predict(xk[None, :])[0]
    Ti_roll.append(pred)
Ti_mlp_full[start_idx + lag_steps : start_idx + lag_steps + len(Ti_roll) - lag_steps] = Ti_roll[lag_steps:]

# -----------------------------
# 7) Metrics (train / test)
# -----------------------------
def rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))

Ti_rc_tr   = Ti_rc_full[train_mask]
Ti_lin_tr  = Ti_lin_full[train_mask]
Ti_mlp_tr  = Ti_mlp_full[train_mask]
Ti_rc_te   = Ti_rc_full[test_mask]
Ti_lin_te  = Ti_lin_full[test_mask]
Ti_mlp_te  = Ti_mlp_full[test_mask]
Ti_meas_tr = Ti_tr
Ti_meas_te = Ti_meas[test_mask]

rmse_rc_train  = rmse(Ti_rc_tr,  Ti_meas_tr)
rmse_lin_train = rmse(Ti_lin_tr, Ti_meas_tr)
rmse_mlp_train = rmse(Ti_mlp_tr[~np.isnan(Ti_mlp_tr)], Ti_meas_tr[~np.isnan(Ti_mlp_tr)])

rmse_rc_test   = rmse(Ti_rc_te,  Ti_meas_te)
rmse_lin_test  = rmse(Ti_lin_te, Ti_meas_te)
rmse_mlp_test  = rmse(Ti_mlp_te[~np.isnan(Ti_mlp_te)], Ti_meas_te[~np.isnan(Ti_mlp_te)])

print(f"\nRMSE (°C) — TRAIN [10–40 h]:  RC={rmse_rc_train:.3f},  Linear={rmse_lin_train:.3f},  MLP={rmse_mlp_train:.3f}")
print(f"RMSE (°C) — TEST  [50–90 h]:  RC={rmse_rc_test:.3f},  Linear={rmse_lin_test:.3f},  MLP={rmse_mlp_test:.3f}")

# Residuals for bottom subplot
res_rc  = Ti_rc_full - Ti_meas
res_lin = Ti_lin_full - Ti_meas
res_mlp = Ti_mlp_full - Ti_meas

# -----------------------------
# 8) ONE combined figure (3 stacked subplots) + shading
# -----------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

# A) Inputs (heating & solar)
ax = axes[0]
ax.plot(t, PhiH, label="Heating power (W)")
ax.plot(t, PhiS, label="Solar irradiance (W/m²)")
ax.set_ylabel("Inputs")
ax.axvspan(10, 40, color="grey",   alpha=0.12, label="Train")
ax.axvspan(50, 90, color="orange", alpha=0.08, label="Test")
handles, labels = ax.get_legend_handles_labels()
ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), loc="upper right")

# B) Temperatures + fits
ax = axes[1]
ax.plot(t, Ti_meas, label="Measured $T_i$")
ax.plot(t, Ti_rc_full, "--", label="RC (solve_ivp)")
ax.plot(t, Ti_lin_full, ":", label="Linear (inputs only)")
ax.plot(t, Ti_mlp_full, "-", linewidth=1, alpha=0.9, label=f"MLP AR (lags={lag_steps})")
ax.axvspan(10, 40, color="grey",   alpha=0.12)
ax.axvspan(50, 90, color="orange", alpha=0.08)
title = (f"Train RMSE — RC: {rmse_rc_train:.2f}°C, Linear: {rmse_lin_train:.2f}°C, MLP: {rmse_mlp_train:.2f}°C   |   "
         f"Test RMSE — RC: {rmse_rc_test:.2f}°C, Linear: {rmse_lin_test:.2f}°C, MLP: {rmse_mlp_test:.2f}°C")
ax.set_title(title)
ax.set_ylabel("Indoor temperature [°C]")
ax.legend(loc="best")

# C) Residuals over full series
ax = axes[2]
ax.plot(t, res_rc,  "--", label="RC residual")
ax.plot(t, res_lin, ":",  label="Linear residual")
ax.plot(t, res_mlp, "-",  label="MLP residual")
ax.axhline(0, linewidth=1)
ax.axvspan(10, 40, color="grey",   alpha=0.12)
ax.axvspan(50, 90, color="orange", alpha=0.08)
ax.set_xlabel("Time [h]")
ax.set_ylabel("Residual (model - meas) [°C]")
ax.legend(loc="best")

plt.tight_layout()
plt.show()

# -----------------------------
# 9) SECOND figure: full raw data overview (blog-style)
# -----------------------------
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, PhiH, label="Heating power (W)")
plt.plot(t, PhiS, label="Solar irradiance (W/m²)")
plt.legend(); plt.ylabel("Inputs")

plt.subplot(2, 1, 2)
plt.plot(t, Ti_meas, label="Indoor temperature (°C)")
plt.plot(t, To,      label="Outdoor temperature (°C)")
plt.xlabel("Time [h]"); plt.ylabel("Temperature [°C]")
plt.legend()
plt.tight_layout()
plt.show()
