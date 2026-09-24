# Browser Labs

The browser lab is a small, install-free companion to the book. Its six notebooks focus on examples that fit comfortably in a WebAssembly Python kernel; the expensive JAX, MPC, and nonlinear solver demonstrations remain precomputed in the main text.

[Launch the browser labs](https://pierrelux.github.io/rlbook/lab/)

Each notebook begins with a working baseline, asks you to predict an outcome before changing parameters, and ends with executable assertions:

1. finite-MDP policy evaluation;
2. value iteration and policy iteration;
3. Monte Carlo error and maximization bias;
4. Euler, trapezoidal, and higher-order ODE discretization;
5. finite-horizon LQR and the Riccati recursion;
6. a small SciPy trajectory-optimization problem.

The notebooks use only NumPy, SciPy, Matplotlib, and ipywidgets. Changes are stored in your browser. Download a notebook from JupyterLab if you want to keep or submit it.

To serve the lab locally after building it, run:

```bash
uv run jupyter lite serve --lite-dir lab --contents notebooks
```
