"""Generate the six small, browser-compatible teaching notebooks."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).parent / "notebooks"
KERNEL = {
    "kernelspec": {
        "display_name": "Python (XPython)",
        "language": "python",
        "name": "xpython",
    },
    "language_info": {"name": "python", "version": "3.12"},
}


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def write(name: str, cells: list):
    notebook = nbf.v4.new_notebook(cells=cells, metadata=KERNEL)
    nbf.write(notebook, ROOT / name)


ROOT.mkdir(parents=True, exist_ok=True)

write(
    "01_finite_mdp_policy_evaluation.ipynb",
    [
        markdown("""
        # Finite-MDP policy evaluation

        **Learning goals:** construct a fixed-policy transition matrix, solve its Bellman equation directly, and compare the solution with successive approximation.

        **Predict first:** as $\\gamma$ moves from 0.5 to 0.98, what happens to the number of iterations needed for the same tolerance?
        """),
        code("""
        import numpy as np
        import matplotlib.pyplot as plt
        from ipywidgets import interact, FloatSlider

        P_pi = np.array([[0.80, 0.20, 0.00],
                         [0.10, 0.70, 0.20],
                         [0.00, 0.25, 0.75]])
        r_pi = np.array([0.0, 1.0, 2.0])

        def evaluate_policy(gamma=0.90, tolerance=1e-10):
            exact = np.linalg.solve(np.eye(3) - gamma * P_pi, r_pi)
            value = np.zeros(3)
            errors = []
            for _ in range(10_000):
                updated = r_pi + gamma * P_pi @ value
                errors.append(np.max(np.abs(updated - exact)))
                value = updated
                if errors[-1] < tolerance:
                    break
            return exact, value, np.asarray(errors)

        exact, approximate, errors = evaluate_policy()
        print("Exact value:", exact)
        print("Iterations:", len(errors))
        """),
        code("""
        @interact(gamma=FloatSlider(value=0.90, min=0.50, max=0.98, step=0.01))
        def convergence_plot(gamma):
            exact, approximate, errors = evaluate_policy(gamma)
            plt.figure(figsize=(6, 3))
            plt.semilogy(errors)
            plt.xlabel("Iteration")
            plt.ylabel("Sup-norm error")
            plt.title(f"Policy evaluation, gamma={gamma:.2f}")
            plt.grid(alpha=0.25)
            plt.show()
        """),
        markdown("""
        **Experiment:** change one row of `P_pi` so the chain remains longer in state 2. Predict the sign of the change in each value before rerunning.
        """),
        code("""
        assert np.allclose(P_pi.sum(axis=1), 1.0)
        assert np.allclose(approximate, exact, atol=1e-9)
        assert np.all(errors[1:] <= errors[:-1] + 1e-12)
        print("Checks passed.")
        """),
    ],
)

write(
    "02_value_and_policy_iteration.ipynb",
    [
        markdown("""
        # Value iteration and policy iteration

        **Learning goals:** implement both algorithms for one finite MDP, compare their update patterns, and verify the greedy fixed point.

        **Predict first:** which method uses fewer outer iterations, and which method does more work inside an iteration?
        """),
        code("""
        import numpy as np
        import matplotlib.pyplot as plt

        P = np.array([
            [[0.85, 0.15, 0.00], [0.10, 0.75, 0.15], [0.00, 0.20, 0.80]],
            [[0.20, 0.70, 0.10], [0.00, 0.25, 0.75], [0.10, 0.10, 0.80]],
        ])  # action, state, next state
        R = np.array([[0.0, 0.5], [0.3, 1.0], [1.0, 0.2]])
        gamma = 0.93

        def q_values(value):
            return R + gamma * np.einsum("asj,j->sa", P, value)

        def value_iteration(tol=1e-10):
            value, residuals = np.zeros(3), []
            for _ in range(10_000):
                updated = q_values(value).max(axis=1)
                residuals.append(np.max(np.abs(updated - value)))
                value = updated
                if residuals[-1] < tol:
                    break
            return value, q_values(value).argmax(axis=1), np.asarray(residuals)

        def policy_iteration():
            policy, history = np.zeros(3, dtype=int), []
            while True:
                P_pi = P[policy, np.arange(3)]
                r_pi = R[np.arange(3), policy]
                value = np.linalg.solve(np.eye(3) - gamma * P_pi, r_pi)
                improved = q_values(value).argmax(axis=1)
                history.append(value.copy())
                if np.array_equal(improved, policy):
                    return value, policy, np.asarray(history)
                policy = improved

        v_vi, pi_vi, residuals = value_iteration()
        v_pi, pi_pi, pi_history = policy_iteration()
        print("Value-iteration policy:", pi_vi)
        print("Policy-iteration policy:", pi_pi)
        """),
        code("""
        plt.figure(figsize=(6, 3))
        plt.semilogy(residuals)
        plt.xlabel("Value-iteration update")
        plt.ylabel("Bellman residual")
        plt.grid(alpha=0.25)
        plt.show()
        """),
        markdown("""
        **Experiment:** set `gamma` to 0.5 and rerun. Which policy changes, if any? Explain why a faster contraction need not imply a different optimum.
        """),
        code("""
        assert np.allclose(P.sum(axis=2), 1.0)
        assert np.allclose(v_vi, v_pi, atol=1e-8)
        assert np.array_equal(pi_vi, pi_pi)
        assert np.max(np.abs(q_values(v_vi).max(axis=1) - v_vi)) < 1e-8
        print("Checks passed.")
        """),
    ],
)

write(
    "03_monte_carlo_estimators.ipynb",
    [
        markdown("""
        # Monte Carlo error and maximization bias

        **Learning goals:** verify the $1/\\sqrt{N}$ error rate and reproduce maximization bias with independent simulations.

        **Predict first:** if the number of equally good actions doubles, does the maximum estimator become more or less optimistic?
        """),
        code("""
        import numpy as np
        import matplotlib.pyplot as plt
        from ipywidgets import interact, IntSlider

        rng = np.random.default_rng(2026)
        sample_sizes = 2 ** np.arange(0, 11)
        rmse = []
        for n in sample_sizes:
            means = rng.normal(size=(4000, n)).mean(axis=1)
            rmse.append(np.sqrt(np.mean(means**2)))
        rmse = np.asarray(rmse)

        plt.loglog(sample_sizes, rmse, "o-", label="empirical")
        plt.loglog(sample_sizes, 1 / np.sqrt(sample_sizes), "--", label="theory")
        plt.xlabel("Samples N")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.show()
        """),
        code("""
        @interact(actions=IntSlider(value=8, min=2, max=32, step=2))
        def estimator_histogram(actions):
            local_rng = np.random.default_rng(7)
            first = local_rng.normal(size=(5000, actions))
            second = local_rng.normal(size=(5000, actions))
            selected = first.argmax(axis=1)
            maximum = first.max(axis=1)
            double = second[np.arange(len(second)), selected]
            plt.figure(figsize=(7, 3))
            plt.hist(maximum, bins=50, alpha=0.55, density=True, label="maximum")
            plt.hist(double, bins=50, alpha=0.55, density=True, label="double")
            plt.axvline(0, color="black", linestyle="--")
            plt.legend()
            plt.title(f"Biases: maximum={maximum.mean():.3f}, double={double.mean():.3f}")
            plt.show()
        """),
        markdown("""
        **Experiment:** give action 0 a true advantage of 0.5. When does double estimation become negatively biased because the noisy selector chooses the wrong action?
        """),
        code("""
        fitted_slope = np.polyfit(np.log(sample_sizes), np.log(rmse), 1)[0]
        assert -0.58 < fitted_slope < -0.42
        assert rmse[-1] < rmse[0] / 20
        print(f"Checks passed; fitted log-log slope = {fitted_slope:.3f}.")
        """),
    ],
)

write(
    "04_ode_discretization.ipynb",
    [
        markdown("""
        # ODE discretization: Euler, trapezoidal, and RK4

        **Learning goals:** compare numerical trajectories and estimate each method's convergence order.

        **Predict first:** which methods remain accurate when the step size is halved, and by what factor should their terminal errors shrink?
        """),
        code("""
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.integrate import solve_ivp

        rate, horizon, x0 = -2.0, 3.0, 1.0

        def integrate(method, step):
            times = np.arange(0.0, horizon + 0.5 * step, step)
            values = np.empty_like(times)
            values[0] = x0
            for k in range(len(times) - 1):
                x = values[k]
                if method == "Euler":
                    values[k + 1] = x + step * rate * x
                elif method == "Trapezoidal":
                    values[k + 1] = x * (1 + 0.5 * step * rate) / (1 - 0.5 * step * rate)
                elif method == "RK4":
                    z = step * rate
                    values[k + 1] = x * (1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)
            return times, values

        reference = solve_ivp(lambda t, x: rate * x, (0, horizon), [x0], rtol=1e-12, atol=1e-14, dense_output=True)
        for method in ["Euler", "Trapezoidal", "RK4"]:
            t, x = integrate(method, 0.25)
            plt.plot(t, x, "o-", label=method)
        grid = np.linspace(0, horizon, 300)
        plt.plot(grid, reference.sol(grid)[0], "k--", label="reference")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("State")
        plt.show()
        """),
        code("""
        steps = np.array([0.5, 0.25, 0.125, 0.0625])
        errors = {}
        exact_terminal = np.exp(rate * horizon)
        for method in ["Euler", "Trapezoidal", "RK4"]:
            errors[method] = np.array([abs(integrate(method, h)[1][-1] - exact_terminal) for h in steps])
            order = np.polyfit(np.log(steps), np.log(errors[method]), 1)[0]
            print(f"{method:12s} estimated order: {order:.2f}")
        """),
        markdown("""
        **Experiment:** change `rate` to `-10`. Increase the step size until explicit Euler becomes unstable. Why does the trapezoidal update behave differently?
        """),
        code("""
        assert errors["Euler"][-1] < errors["Euler"][0]
        assert errors["Trapezoidal"][-1] < errors["Euler"][-1]
        assert errors["RK4"][-1] < errors["Trapezoidal"][-1]
        print("Checks passed.")
        """),
    ],
)

write(
    "05_finite_horizon_lqr.ipynb",
    [
        markdown("""
        # Finite-horizon LQR and the Riccati recursion

        **Learning goals:** run the backward Riccati recursion, apply the resulting feedback gains, and inspect closed-loop state decay.

        **Predict first:** increasing the control penalty should make the feedback gains larger or smaller?
        """),
        code("""
        import numpy as np
        import matplotlib.pyplot as plt
        from ipywidgets import interact, FloatLogSlider

        A = np.array([[1.0, 0.1], [0.0, 1.0]])
        B = np.array([[0.005], [0.1]])
        Q = np.diag([1.0, 0.1])
        Q_terminal = 10 * Q
        horizon = 60

        def riccati(control_penalty=0.1):
            R = np.array([[control_penalty]])
            P = [None] * (horizon + 1)
            gains = [None] * horizon
            P[-1] = Q_terminal
            for k in range(horizon - 1, -1, -1):
                gains[k] = np.linalg.solve(R + B.T @ P[k + 1] @ B, B.T @ P[k + 1] @ A)
                P[k] = Q + A.T @ P[k + 1] @ (A - B @ gains[k])
            return P, gains

        def rollout(control_penalty=0.1):
            P, gains = riccati(control_penalty)
            states = np.zeros((horizon + 1, 2))
            controls = np.zeros(horizon)
            states[0] = [2.0, 0.0]
            for k in range(horizon):
                controls[k] = (-gains[k] @ states[k]).item()
                states[k + 1] = A @ states[k] + B[:, 0] * controls[k]
            return P, gains, states, controls
        """),
        code("""
        @interact(control_penalty=FloatLogSlider(value=0.1, base=10, min=-2, max=1, step=0.25))
        def lqr_plot(control_penalty):
            P, gains, states, controls = rollout(control_penalty)
            fig, axes = plt.subplots(1, 2, figsize=(9, 3))
            axes[0].plot(states[:, 0], label="position")
            axes[0].plot(states[:, 1], label="velocity")
            axes[0].legend()
            axes[1].step(np.arange(horizon), controls, where="post")
            axes[1].set_title(f"First gain: {gains[0].ravel()}")
            fig.tight_layout()
            plt.show()
        """),
        markdown("""
        **Experiment:** increase the terminal cost by a factor of ten. Which gains change most: those near the beginning or those near the end of the horizon?
        """),
        code("""
        P, gains, states, controls = rollout()
        assert all(np.allclose(matrix, matrix.T, atol=1e-10) for matrix in P)
        assert np.linalg.norm(states[-1]) < np.linalg.norm(states[0])
        assert np.all(np.linalg.eigvalsh(P[0]) > 0)
        print("Checks passed.")
        """),
    ],
)

write(
    "06_scipy_trajectory_optimization.ipynb",
    [
        markdown("""
        # SciPy trajectory optimization

        **Learning goals:** transcribe a bounded double-integrator problem, solve it with SLSQP, and inspect feasibility separately from objective value.

        **Predict first:** which acceleration bounds should become active when the goal is to reach the target with little control effort over a short horizon?
        """),
        code("""
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.optimize import minimize

        intervals, dt, target = 30, 0.1, 1.0

        def unpack(z):
            position = z[: intervals + 1]
            velocity = z[intervals + 1 : 2 * (intervals + 1)]
            acceleration = z[2 * (intervals + 1) :]
            return position, velocity, acceleration

        def objective(z):
            _, _, acceleration = unpack(z)
            return dt * np.sum(acceleration**2)

        def defects(z):
            position, velocity, acceleration = unpack(z)
            dynamics = np.r_[
                position[1:] - position[:-1] - dt * velocity[:-1] - 0.5 * dt**2 * acceleration,
                velocity[1:] - velocity[:-1] - dt * acceleration,
            ]
            boundary = [position[0], velocity[0], position[-1] - target, velocity[-1]]
            return np.r_[dynamics, boundary]

        size = 2 * (intervals + 1) + intervals
        guess = np.zeros(size)
        guess[: intervals + 1] = np.linspace(0, target, intervals + 1)
        bounds = [(None, None)] * (2 * (intervals + 1)) + [(-1.0, 1.0)] * intervals
        result = minimize(objective, guess, method="SLSQP", bounds=bounds,
                          constraints={"type": "eq", "fun": defects},
                          options={"ftol": 1e-10, "maxiter": 1000})
        position, velocity, acceleration = unpack(result.x)
        print(result.message)
        print("Maximum defect:", np.max(np.abs(defects(result.x))))
        """),
        code("""
        time = np.arange(intervals + 1) * dt
        fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
        axes[0].plot(time, position)
        axes[0].set_ylabel("position")
        axes[1].plot(time, velocity)
        axes[1].set_ylabel("velocity")
        axes[2].step(time[:-1], acceleration, where="post")
        axes[2].axhline(1, color="black", linestyle=":")
        axes[2].axhline(-1, color="black", linestyle=":")
        axes[2].set(xlabel="time", ylabel="acceleration")
        fig.tight_layout()
        plt.show()
        """),
        markdown("""
        **Experiment:** reduce the horizon or tighten the acceleration bounds. Predict feasibility before solving, then use the maximum defect—not only `result.success`—to evaluate the answer.
        """),
        code("""
        assert result.success
        assert np.max(np.abs(defects(result.x))) < 1e-6
        assert np.max(np.abs(acceleration)) <= 1.0 + 1e-9
        assert abs(position[-1] - target) < 1e-8 and abs(velocity[-1]) < 1e-8
        print("Checks passed.")
        """),
    ],
)
