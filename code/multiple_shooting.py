"""
Multiple Shooting as a Boundary-Value Problem (BVP) for a Ballistic Trajectory
-----------------------------------------------------------------------------
We solve for the initial velocities (and total flight time) so that the terminal
position hits a target, enforcing continuity between shooting segments.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from IPython.display import HTML, display

# -----------------------------
# Physical parameters
# -----------------------------
g = 9.81          # gravity (m/s^2)
m = 1.0           # mass (kg)
drag_coeff = 0.1  # quadratic drag coefficient


def dynamics(t, state):
    """Ballistic dynamics with quadratic drag. state = [x, y, vx, vy]."""
    x, y, vx, vy = state
    v = np.hypot(vx, vy)
    drag_x = -drag_coeff * v * vx / m if v > 0 else 0.0
    drag_y = -drag_coeff * v * vy / m if v > 0 else 0.0
    dx  = vx
    dy  = vy
    dvx = drag_x
    dvy = drag_y - g
    return np.array([dx, dy, dvx, dvy])


def flow(y0, h):
    """One-segment flow map Φ(y0; h): integrate dynamics over duration h."""
    sol = solve_ivp(dynamics, (0.0, h), y0, method="RK45", rtol=1e-7, atol=1e-9)
    return sol.y[:, -1], sol

# -----------------------------
# Multiple-shooting BVP residuals
# -----------------------------

def residuals(z, K, x_init, x_target):
    """
    Unknowns z = [vx0, vy0, H, y1(4), y2(4), ..., y_{K-1}(4)]  (total len = 3 + 4*(K-1))
    We define y0 from x_init and (vx0, vy0). Each segment has duration h = H/K.
    Residual vector stacks:
      - initial position constraints: y0[:2] - x_init[:2]
      - continuity: y_{k+1} - Φ(y_k; h) for k=0..K-2
      - terminal position constraint at end of last segment: Φ(y_{K-1}; h)[:2] - x_target[:2]
    """
    n = 4
    vx0, vy0, H = z[0], z[1], z[2]
    if H <= 0:
        # Strongly penalize nonpositive durations to keep solver away
        return 1e6 * np.ones(2 + 4*(K-1) + 2)

    h = H / K

    # Build list of segment initial states y_0..y_{K-1}
    ys = []
    y0 = np.array([x_init[0], x_init[1], vx0, vy0], dtype=float)
    ys.append(y0)
    if K > 1:
        rest = z[3:]
        y_internals = rest.reshape(K-1, n)
        ys.extend(list(y_internals))  # y1..y_{K-1}

    res = []

    # Initial position must match exactly
    res.extend(ys[0][:2] - x_init[:2])

    # Continuity across segments
    for k in range(K-1):
        yk = ys[k]
        yk1_pred, _ = flow(yk, h)
        res.extend(ys[k+1] - yk1_pred)

    # Terminal position at the end of last segment equals target
    y_last_end, _ = flow(ys[-1], h)
    res.extend(y_last_end[:2] - x_target[:2])

    # Optional soft "stay above ground" at knots (kept gentle)
    # res.extend(np.minimum(0.0, np.array([y[1] for y in ys])).ravel())

    return np.asarray(res)

# -----------------------------
# Solve BVP via optimization on 0.5*||residuals||^2
# -----------------------------

def solve_bvp_multiple_shooting(K=5, x_init=np.array([0., 0.]), x_target=np.array([10., 0.])):
    """
    K: number of shooting segments.
    x_init: initial position (x0, y0). Initial velocities are unknown.
    x_target: desired terminal position (xT, yT) at time H (unknown).
    """
    # Heuristic initial guesses:
    dx = x_target[0] - x_init[0]
    dy = x_target[1] - x_init[1]
    H0 = max(0.5, dx / 5.0)  # guess ~ 5 m/s horizontal
    vx0_0 = dx / H0
    vy0_0 = (dy + 0.5 * g * H0**2) / H0  # vacuum guess

    # Intentionally disconnected internal knots to visualize defect shrinkage
    internals = []
    for k in range(1, K):  # y1..y_{K-1}
        xk = x_init[0] + (dx * k) / K
        yk = x_init[1] + (dy * k) / K + 2.0  # offset to create mismatch
        internals.append(np.array([xk, yk, 0.0, 0.0]))
    internals = np.array(internals) if K > 1 else np.array([])

    z0 = np.concatenate(([vx0_0, vy0_0, H0], internals.ravel()))

    # Variable bounds: H > 0, keep velocities within a reasonable range
    # Use wide bounds to let the solver work; tune if needed.
    lb = np.full_like(z0, -np.inf, dtype=float)
    ub = np.full_like(z0,  np.inf, dtype=float)
    lb[2] = 1e-2  # H lower bound
    # Optional velocity bounds
    lb[0], ub[0] = -50.0, 50.0
    lb[1], ub[1] = -50.0, 50.0

    # Objective and callback for L-BFGS-B
    def objective(z):
        r = residuals(z, K,
                      np.array([x_init[0], x_init[1], 0., 0.]),
                      np.array([x_target[0], x_target[1], 0., 0.]))
        return 0.5 * np.dot(r, r)

    iterate_history = []
    def cb(z):
        iterate_history.append(z.copy())

    bounds = list(zip(lb.tolist(), ub.tolist()))
    sol = minimize(objective, z0, method='L-BFGS-B', bounds=bounds,
                   callback=cb, options={'maxiter': 300, 'ftol': 1e-12})

    return sol, iterate_history

# -----------------------------
# Reconstruct and plot (optional static figure)
# -----------------------------

def reconstruct_and_plot(sol, K, x_init, x_target):
    n = 4
    vx0, vy0, H = sol.x[0], sol.x[1], sol.x[2]
    h = H / K

    ys = []
    y0 = np.array([x_init[0], x_init[1], vx0, vy0])
    ys.append(y0)
    if K > 1:
        internals = sol.x[3:].reshape(K-1, n)
        ys.extend(list(internals))

    # Integrate each segment and stitch
    traj_x, traj_y = [], []
    for k in range(K):
        yk = ys[k]
        yend, seg = flow(yk, h)
        traj_x.extend(seg.y[0, :].tolist() if k == 0 else seg.y[0, 1:].tolist())
        traj_y.extend(seg.y[1, :].tolist() if k == 0 else seg.y[1, 1:].tolist())

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(traj_x, traj_y, '-', label='Multiple-shooting solution')
    ax.plot([x_init[0]], [x_init[1]], 'go', label='Start')
    ax.plot([x_target[0]], [x_target[1]], 'r*', ms=12, label='Target')
    total_pts = len(traj_x)
    for k in range(1, K):
        idx = int(k * total_pts / K)
        ax.axvline(traj_x[idx], color='k', ls='--', alpha=0.3, lw=1)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Multiple Shooting BVP (K={K})   H={H:.3f}s   v0=({vx0:.2f},{vy0:.2f}) m/s')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # Report residual norms
    res = residuals(sol.x, K, np.array([x_init[0], x_init[1], 0., 0.]), np.array([x_target[0], x_target[1], 0., 0.]))
    print(f"\nFinal residual norm: {np.linalg.norm(res):.3e}")
    print(f"vx0={vx0:.4f} m/s, vy0={vy0:.4f} m/s, H={H:.4f} s")

# -----------------------------
# Create JS animation for notebooks
# -----------------------------

def create_animation_progress(iter_history, K, x_init, x_target):
    """Return a JS animation (to_jshtml) showing defect shrinkage across segments."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    n = 4

    def unpack(z):
        vx0, vy0, H = z[0], z[1], z[2]
        ys = [np.array([x_init[0], x_init[1], vx0, vy0])]
        if K > 1 and len(z) > 3:
            internals = z[3:].reshape(K-1, n)
            ys.extend(list(internals))
        return H, ys

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.set_xlabel('Segment index (normalized time)')
    ax.set_ylabel('y (m)')
    ax.set_title('Multiple Shooting: Defect Shrinkage (Fixed Boundaries)')
    ax.grid(True, alpha=0.3)

    # Start/target markers at fixed indices
    ax.plot([0], [x_init[1]], 'go', label='Start')
    ax.plot([K], [x_target[1]], 'r*', ms=12, label='Target')
    # Vertical dashed lines at boundaries
    for k in range(1, K):
        ax.axvline(k, color='k', ls='--', alpha=0.35, lw=1)
    ax.legend(loc='best')

    # Pre-create line artists
    colors = plt.cm.plasma(np.linspace(0, 1, K))
    segment_lines = [ax.plot([], [], '-', color=colors[k], lw=2, alpha=0.9)[0] for k in range(K)]
    connector_lines = [ax.plot([], [], 'r-', lw=1.4, alpha=0.75)[0] for _ in range(K-1)]

    text_iter = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    def animate(i):
        idx = min(i, len(iter_history)-1)
        z = iter_history[idx]
        H, ys = unpack(z)
        h = H / K

        all_y = [x_init[1], x_target[1]]
        total_defect = 0.0
        for k in range(K):
            yk = ys[k]
            yend, seg = flow(yk, h)
            # Map local time to [k, k+1]
            t_local = seg.t
            x_vals = k + (t_local / t_local[-1])
            y_vals = seg.y[1, :]
            segment_lines[k].set_data(x_vals, y_vals)
            all_y.extend(y_vals.tolist())
            if k < K-1:
                y_next = ys[k+1]
                # Vertical connector at boundary x=k+1
                connector_lines[k].set_data([k+1, k+1], [yend[1], y_next[1]])
                total_defect += abs(y_next[1] - yend[1])

        # Fixed x-limits in index space
        ax.set_xlim(-0.1, K + 0.1)
        ymin, ymax = min(all_y), max(all_y)
        margin_y = 0.10 * max(1.0, ymax - ymin)
        ax.set_ylim(ymin - margin_y, ymax + margin_y)

        text_iter.set_text(f'Iterate {idx+1}/{len(iter_history)}  |  Sum vertical defect: {total_defect:.3e}')
        return segment_lines + connector_lines + [text_iter]

    anim = FuncAnimation(fig, animate, frames=len(iter_history), interval=600, blit=False, repeat=True)
    plt.tight_layout()
    js_anim = anim.to_jshtml()
    plt.close(fig)
    return js_anim


def main():
    # Problem definition
    x_init = np.array([0.0, 0.0])      # start at origin
    x_target = np.array([10.0, 0.0])   # hit ground at x=10 m
    K = 6                               # number of shooting segments

    sol, iter_hist = solve_bvp_multiple_shooting(K=K, x_init=x_init, x_target=x_target)
    # Optionally show static reconstruction (commented for docs cleanliness)
    # reconstruct_and_plot(sol, K, x_init, x_target)

    # Animate progression (defect shrinkage across segments) and display as JS
    js_anim = create_animation_progress(iter_hist, K, x_init, x_target)
    display(HTML(js_anim))


if __name__ == "__main__":
    main()
