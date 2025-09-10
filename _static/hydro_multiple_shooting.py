# Let's implement the full MSD (Multiple Shooting for Distributed systems) demo as a single Python module.
# We'll keep it compact but complete, with a runnable demo using small sizes so it finishes quickly.
# The module will be saved as /mnt/data/msd_hydro.py and a short demo run at the end will produce a PNG and some console output.

import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Any

# Try SciPy imports
try:
    from scipy.optimize import minimize
    from scipy.integrate import solve_ivp
    from scipy.special import legendre
    from numpy.polynomial.legendre import leggauss
    SCIPY_OK = True
except Exception as e:
    SCIPY_OK = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------
# Core Models and Utilities
# --------------------------

@dataclass
class ReachParameters:
    """Parameters for a single reach/dam subsystem"""
    length: float = 4000.0  # meters
    width: float = 100.0    # meters
    slope: float = 0.0033
    kstr: float = 30.0      # Strickler coefficient
    kt: float = 8.0         # Turbine constant kJ/m^4 (units absorbed in demo)
    kb: float = 0.6         # Bypass constant
    Sb: float = 18.26       # Bypass area m^2
    n_cells: int = 3        # Spatial discretization cells (smaller to keep demo fast)


class SaintVenantReach:
    """Single reach dynamics using a coarse Saint-Venant ODE semi-discretization (toy)."""

    def __init__(self, params: ReachParameters):
        self.p = params
        self.dz = params.length / params.n_cells
        self.g = 9.81

    def _friction_slope(self, Q: float, H: float) -> float:
        p = self.p
        S = p.width * max(H, 0.0)
        if S <= 0.0:
            return 0.0
        U = Q / S
        R = S / (p.width + 2.0 * max(H, 0.0))
        if R <= 0.0:
            return 0.0
        return (U ** 2) / (p.kstr ** 2 * (R ** (4.0 / 3.0)))

    def dynamics(self, t: float, state: np.ndarray, Q_in: float, Q_turb: float) -> np.ndarray:
        """
        Toy Saint-Venant semi-discretization (explicit FD). State layout:
        [Q1, H1, Q2, H2, ..., Qn, Hn]
        """
        n = self.p.n_cells
        Q = state[0::2].copy()  # discharge per cell
        H = state[1::2].copy()  # water level per cell

        dQ_dt = np.zeros(n)
        dH_dt = np.zeros(n)

        # upstream boundary discharge
        Q[0] = max(Q_in, 0.0)

        for i in range(n):
            # Continuity: dH/dt ~ - (1/width) * dQ/dz
            if i < n - 1:
                dQ_dz = (Q[i + 1] - Q[i]) / self.dz
            else:
                # Downstream boundary outflow at turbine + bypass
                H_turb = max(H[-1], 0.0)
                Q_bypass = self.p.kb * self.p.Sb * math.sqrt(max(2.0 * self.g * H_turb, 0.0))
                Q_out = max(Q_turb, 0.0) + max(Q_bypass, 0.0)
                dQ_dz = (Q_out - Q[i]) / self.dz
            dH_dt[i] = -dQ_dz / self.p.width

            # Momentum (highly simplified): dQ/dt ≈ -g * S * (dH/dz + If - slope)
            if i > 0:
                dH_dz = (H[i] - H[i - 1]) / self.dz
            else:
                dH_dz = (H[1] - H[0]) / self.dz if n > 1 else 0.0

            S_wet = self.p.width * max(H[i], 0.0)
            If = self._friction_slope(Q[i], H[i])
            dQ_dt[i] = -self.g * S_wet * (dH_dz + If - self.p.slope)

        dstate_dt = np.zeros(2 * n)
        dstate_dt[0::2] = dQ_dt
        dstate_dt[1::2] = dH_dt
        return dstate_dt

    def power_output(self, H_turb: float, Q_turb: float) -> float:
        """Simple turbine power model P = k_t * Q_turb * H_turb."""
        return self.p.kt * max(Q_turb, 0.0) * max(H_turb, 0.0)


class CouplingBasis:
    """Legendre basis functions for coupling approximation on an interval [t0, t1]."""

    def __init__(self, degree: int = 2):
        self.degree = degree
        self.S = degree + 1  # number of basis functions

    def legendre_basis(self, t: float, t0: float, t1: float) -> np.ndarray:
        tau = -1.0 + 2.0 * (t - t0) / (t1 - t0)
        out = np.zeros(self.S)
        for i in range(self.S):
            poly = legendre(i)  # evaluates P_i(tau)
            # Orthonormal scaling over [-1,1] with weight 1: sqrt((2i+1)/2)
            out[i] = poly(tau) * math.sqrt((2 * i + 1) / 2.0)
        return out

    def reconstruct(self, coeffs: np.ndarray, t: float, t0: float, t1: float) -> float:
        return float(np.dot(coeffs, self.legendre_basis(t, t0, t1)))

    def fit_coeffs(self, samples_t: np.ndarray, samples_y: np.ndarray, t0: float, t1: float) -> np.ndarray:
        """
        Least-squares fit of y(t) over [t0,t1] onto orthonormal Legendre basis.
        """
        Phi = np.vstack([self.legendre_basis(t, t0, t1) for t in samples_t])  # shape (#samples, S)
        # Least squares
        coeffs, *_ = np.linalg.lstsq(Phi, samples_y, rcond=None)
        return coeffs

    def project_by_quadrature(self, y_func: Callable[[float], float], t0: float, t1: float, quad_order: int = 8) -> np.ndarray:
        """
        Project y(t) onto orthonormal Legendre basis using Gauss-Legendre quadrature.
        Returns coefficient vector c such that y ≈ sum_j c_j * phi_j(t).
        """
        xs, ws = leggauss(quad_order)  # nodes on [-1,1]
        coeffs = np.zeros(self.S)
        # Map nodes to [t0, t1]: t = (t1+t0)/2 + (t1-t0)/2 * x
        for k in range(quad_order):
            tau = xs[k]
            w = ws[k]
            t = 0.5 * (t1 + t0) + 0.5 * (t1 - t0) * tau
            phi = self.legendre_basis(t, t0, t1)  # already orthonormal
            coeffs += w * y_func(t) * phi
        # Change of variables factor from dt = (t1-t0)/2 * dτ
        coeffs *= 0.5 * (t1 - t0)
        return coeffs


# --------------------------
# MSD Optimizer
# --------------------------

class MultipleShootingDistributed:
    """MSD optimizer for a hydro power cascade with input-output coupling."""

    def __init__(self, n_reaches: int = 4, n_intervals: int = 12, dt_hours: float = 2.0, cells_per_reach: int = 3, basis_degree: int = 2):
        assert SCIPY_OK, "SciPy is required to run this optimizer."
        self.M = n_reaches
        self.N = n_intervals
        self.dt = dt_hours  # hours
        self.reaches = [SaintVenantReach(ReachParameters(n_cells=cells_per_reach)) for _ in range(self.M)]
        self.coupling = CouplingBasis(degree=basis_degree)

        # Reference state and power profile
        self.H_ref = self._compute_steady_state_levels(level=17.0)
        self.P_ref = self._day_power_reference(self.N)

        # Bounds
        self.Q_turb_min = 50.0
        self.Q_turb_max = 150.0
        self.Q_in_upstream = 300.0  # inflow to reach 1

        # State bounds
        self.H_tol = 1.0
        self.H_min = [H - self.H_tol for H in self.H_ref]
        self.H_max = [H + self.H_tol for H in self.H_ref]

    # ----- Helpers for references -----

    def _compute_steady_state_levels(self, level: float) -> List[np.ndarray]:
        H_ref = []
        for i in range(self.M):
            H_ref.append(np.ones(self.reaches[i].p.n_cells) * level)
        return H_ref

    def _day_power_reference(self, N: int) -> np.ndarray:
        hours = np.linspace(0.0, 24.0, N)
        return 50.0 + 20.0 * np.sin(2.0 * np.pi * (hours - 6.0) / 24.0) ** 2

    # ----- Decision vector packing/unpacking -----

    def _var_counts_one_interval(self) -> Tuple[int, int, int, int, int]:
        """Counts per interval: controls, states, y coeffs, z coeffs, slack"""
        controls = self.M
        states = sum([2 * r.p.n_cells for r in self.reaches])
        ycoeffs = (self.M - 1) * self.coupling.S  # outputs for reaches 0..M-2
        zcoeffs = (self.M - 1) * self.coupling.S  # inputs for reaches 1..M-1
        slack = 1
        return controls, states, ycoeffs, zcoeffs, slack

    def _decision_vector_length(self) -> int:
        c, s, y, z, e = self._var_counts_one_interval()
        return self.N * (c + s + y + z + e)

    def decision_vector_to_struct(self, x: np.ndarray) -> Dict[str, Any]:
        idx = 0
        S = self.coupling.S
        ncells = [r.p.n_cells for r in self.reaches]
        per_interval = []

        for n in range(self.N):
            u_n = []
            x_n = []
            y_n = []
            z_n = []

            # controls
            for i in range(self.M):
                u_n.append(x[idx]); idx += 1

            # states for each reach
            for i in range(self.M):
                nstates = 2 * ncells[i]
                x_n.append(np.array(x[idx:idx + nstates])); idx += nstates

            # y coeffs for outputs for reaches 0..M-2
            for i in range(self.M - 1):
                y_n.append(np.array(x[idx:idx + S])); idx += S

            # z coeffs for inputs for reaches 1..M-1
            for i in range(1, self.M):
                z_n.append(np.array(x[idx:idx + S])); idx += S

            # slack
            eps_n = x[idx]; idx += 1

            per_interval.append({"u": u_n, "x": x_n, "y": y_n, "z": z_n, "eps": eps_n})

        return {"intervals": per_interval}

    # ----- Objective -----

    def objective(self, x: np.ndarray) -> float:
        data = self.decision_vector_to_struct(x)
        alpha = 50.0
        cost = 0.0
        for n, slot in enumerate(data["intervals"]):
            cost += self.dt * slot["eps"]  # L1 via slack
            # L2 level regularization
            for i in range(self.M):
                H = slot["x"][i][1::2]
                cost += alpha * self.dt * float(np.sum((H - self.H_ref[i]) ** 2))
        return float(cost)

    # ----- Constraints -----

    def _integrate_reach(self, i: int, x0: np.ndarray, u: float, Q_in_fn: Callable[[float], float], t0: float, t1: float):
        """Integrate reach i over [t0, t1] with time-varying inflow Q_in_fn(t)."""
        reach = self.reaches[i]
        # Dynamics with time-varying inflow (piecewise smooth) and constant u
        def f(t, y):
            return reach.dynamics(t, y, Q_in_fn(t), u)

        # Solve ODE
        sol = solve_ivp(f, (t0, t1), x0, method="RK45", rtol=1e-6, atol=1e-8, max_step=(t1 - t0) / 10.0)
        if not sol.success:
            # Fall back: at least provide last value
            y_end = sol.y[:, -1]
        else:
            y_end = sol.y[:, -1]

        # Build callable for y(t) to project coupling output (downstream discharge Q_n(t))
        def discharge_out(t: float) -> float:
            # Interpolate using dense output if available
            if sol.sol is not None:
                st = sol.sol(t)
                return float(st[-2])  # last Q
            # Else approximate via nearest time step
            k = np.searchsorted(sol.t, t)
            k = min(max(k, 1), len(sol.t) - 1)
            # linear interpolate
            t0i, t1i = sol.t[k - 1], sol.t[k]
            y0, y1 = sol.y[:, k - 1], sol.y[:, k]
            w = 0.0 if t1i == t0i else (t - t0i) / (t1i - t0i)
            ylin = (1 - w) * y0 + w * y1
            return float(ylin[-2])

        return y_end, discharge_out

    def constraints_eq(self, x: np.ndarray) -> np.ndarray:
        """Equality constraints: shooting continuity and z=y coefficients (coupling)."""
        data = self.decision_vector_to_struct(x)
        eqs: List[np.ndarray] = []
        S = self.coupling.S

        for n in range(self.N - 1):
            t0 = n * self.dt
            t1 = (n + 1) * self.dt
            slot = data["intervals"][n]
            slot_next = data["intervals"][n + 1]

            # Integrate each reach independently over interval n
            y_coeffs_list = []  # to compare with decision variables for y
            outflows_coeffs = []

            # Build inflow functions per reach
            def make_const(val: float) -> Callable[[float], float]:
                return lambda t: val

            inflow_funcs: List[Callable[[float], float]] = []

            for i in range(self.M):
                if i == 0:
                    # Upstream inflow is constant (river inflow)
                    inflow_funcs.append(make_const(self.Q_in_upstream))
                else:
                    # Inflow to reach i is reconstructed from z coefficients of reach i-1
                    z_coeffs = slot["z"][i - 1]  # because z slots are for i=1..M-1 but stored at indices 0..M-2
                    inflow_funcs.append(lambda t, zc=z_coeffs: self.coupling.reconstruct(zc, t, t0, t1))

            # Integrate and accumulate continuity residuals
            for i in range(self.M):
                x0_i = slot["x"][i]
                u_i = slot["u"][i]
                x1_pred, qout_func = self._integrate_reach(i, x0_i, u_i, inflow_funcs[i], t0, t1)

                # Shooting continuity: x_{n+1} - F_n(x_n, u_n, z_n) = 0
                eqs.append(slot_next["x"][i] - x1_pred)

                # Project output discharge to Legendre basis for y coefficients (for reach i if it feeds i+1)
                if i < self.M - 1:
                    coeffs_y = self.coupling.project_by_quadrature(qout_func, t0, t1, quad_order=8)
                    outflows_coeffs.append(coeffs_y)

            # Coupling equalities: for i=0..M-2, z of reach i+1 equals y of reach i
            for i in range(self.M - 1):
                y_dec = slot["y"][i]
                eqs.append(y_dec - outflows_coeffs[i])  # y decision = projected coeffs
                z_dec = slot["z"][i]
                eqs.append(z_dec - y_dec)               # z = y (downstream uses upstream)

        if len(eqs) == 0:
            return np.zeros(1)
        return np.hstack(eqs)

    def constraints_ineq(self, x: np.ndarray) -> np.ndarray:
        """Inequalities: Power tracking via slack |P_ref - P| <= eps -> eps - (P_ref - P) >= 0 and eps + (P_ref - P) >= 0."""
        data = self.decision_vector_to_struct(x)
        ineqs: List[float] = []

        for n, slot in enumerate(data["intervals"]):
            # Compute instantaneous total power using current states and controls at interval start (proxy)
            P_total = 0.0
            for i in range(self.M):
                H_turb = float(slot["x"][i][-1])  # last H
                Q_turb = float(slot["u"][i])
                P_total += self.reaches[i].power_output(H_turb, Q_turb)
            resid = self.P_ref[n] - P_total
            eps_n = float(slot["eps"])
            ineqs.append(eps_n - resid)
            ineqs.append(eps_n + resid)

        return np.array(ineqs)

    # ----- Bounds -----

    def bounds(self) -> List[Tuple[float, float]]:
        """Bounds per variable in the same order as decision vector packing."""
        bnds: List[Tuple[float, float]] = []
        S = self.coupling.S
        for n in range(self.N):
            # controls
            for i in range(self.M):
                bnds.append((self.Q_turb_min, self.Q_turb_max))
            # states [Q, H] per cell
            for i in range(self.M):
                nc = self.reaches[i].p.n_cells
                for j in range(nc):
                    bnds.append((0.0, np.inf))  # Q >= 0
                    bnds.append((self.H_min[i][j], self.H_max[i][j]))
            # y coeffs (unbounded) for reaches 0..M-2
            for i in range(self.M - 1):
                for _ in range(S):
                    bnds.append((-np.inf, np.inf))
            # z coeffs (unbounded) for reaches 1..M-1
            for i in range(1, self.M):
                for _ in range(S):
                    bnds.append((-np.inf, np.inf))
            # slack
            bnds.append((0.0, np.inf))
        return bnds

    # ----- Initial Guess -----

    def initial_guess(self) -> np.ndarray:
        x0: List[float] = []
        S = self.coupling.S
        for n in range(self.N):
            # controls
            for i in range(self.M):
                x0.append(100.0)  # nominal turbine flow
            # states
            for i in range(self.M):
                nc = self.reaches[i].p.n_cells
                Q_guess = np.ones(nc) * 300.0
                H_guess = self.H_ref[i].copy()
                # Pack interleaved [Q1,H1,Q2,H2,...]
                for j in range(nc):
                    x0.append(Q_guess[j])
                    x0.append(H_guess[j])
            # y coeffs and z coeffs (initialize with constant discharge ~300 on average -> first coeff approx)
            const_coeff = np.zeros(S)
            const_coeff[0] = 300.0 / math.sqrt(2.0)  # because phi0 = 1/sqrt(2) on [-1,1]
            for i in range(self.M - 1):
                x0.extend(const_coeff.tolist())
            for i in range(1, self.M):
                x0.extend(const_coeff.tolist())
            # slack
            x0.append(5.0)
        return np.array(x0, dtype=float)

    # ----- Solve -----

    def solve(self, x0: np.ndarray = None, maxiter: int = 50, verbose: bool = True):
        if x0 is None:
            x0 = self.initial_guess()
        bnds = self.bounds()

        # Build separate SLSQP constraints
        cons = [
            {"type": "eq", "fun": lambda v: self.constraints_eq(v)},
            {"type": "ineq", "fun": lambda v: self.constraints_ineq(v)},
        ]

        res = minimize(
            fun=self.objective,
            x0=x0,
            method="SLSQP",
            bounds=bnds,
            constraints=cons,
            options={"maxiter": maxiter, "disp": verbose, "ftol": 1e-4},
        )
        return res


# --------------------------
# Demo / Quick Run
# --------------------------

def run_demo():
    # Small problem to keep runtime modest
    msd = MultipleShootingDistributed(n_reaches=2, n_intervals=6, dt_hours=4.0, cells_per_reach=3, basis_degree=2)

    print("=" * 60)
    print("Multiple Shooting for Distributed Systems (MSD) — Hydro demo")
    print("=" * 60)
    print(f"Reaches: {msd.M} | Intervals: {msd.N} | Cells per reach: {msd.reaches[0].p.n_cells}")
    print(f"Decision variables: {msd._decision_vector_length()}")
    print(f"Basis functions per interval (Legendre): {msd.coupling.S}")
    print(f"Turbine bounds: [{msd.Q_turb_min}, {msd.Q_turb_max}]")
    print("-" * 60)

    # Plot and save the reference power
    hours = np.linspace(0.0, 24.0, msd.N)
    plt.figure(figsize=(8, 3))
    plt.plot(hours, msd.P_ref, lw=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time (h)")
    plt.ylabel("P_ref (arb)")
    plt.title("Daily Power Reference")
    plt.tight_layout()
    plt.savefig("msd_power_reference.png", dpi=120)
    plt.close()

    # Try solving (may be rough but should run a few iterations)
    print("Building initial guess...")
    x0 = msd.initial_guess()
    print("Solving (SLSQP)... this may take ~10–30s for the tiny demo.")
    res = msd.solve(x0=x0, maxiter=100, verbose=True)

    print("\n=== Optimization Status ===")
    print("Success:", res.success)
    print("Status:", res.status)
    print("Message:", res.message)
    print("Objective value:", float(res.fun))

    # Extract the controls over time and plot
    data = msd.decision_vector_to_struct(res.x if res.success else x0)
    u1 = [slot["u"][0] for slot in data["intervals"]]
    u2 = [slot["u"][1] for slot in data["intervals"]]
    plt.figure(figsize=(8, 3))
    plt.step(range(msd.N), u1, where="post", label="Reach 1 turb")
    plt.step(range(msd.N), u2, where="post", label="Reach 2 turb")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel("Interval")
    plt.ylabel("Q_turb (m^3/s)")
    plt.title("Optimized Turbine Flows (demo)")
    plt.tight_layout()
    plt.savefig("msd_opt_controls.png", dpi=120)
    plt.close()

    print("\nSaved figures:")
    print(" - msd_power_reference.png")
    print(" - msd_opt_controls.png")

if __name__ == "__main__":
    run_demo()