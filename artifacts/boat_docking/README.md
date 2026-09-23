# Boat docking with iLQR and DDP

These are deterministic local trajectory-optimization experiments for
`iterative-trajectory-optimization.md`. The boat parameters are synthetic;
they are not measurements of a particular vessel.

Regenerate from the repository root with:

```bash
uv run python scripts/build_boat_docking_artifacts.py
```

The builder solves the same two problems with iLQR and DDP, checks every final
trajectory with a four-times-finer integration, and writes:

- `metrics.json`: solver settings, source hashes, convergence, and docking checks;
- `results.md`: the chapter's results table;
- `../../interactive/boat-docking-data.json`: every accepted nonlinear rollout;
- `../../_static/boat_docking/`: matching SVG, PDF, and PNG teaching figures.

The JSON replay uses SI units and unwrapped heading in radians. Controls are
normalized thrust commands in [-1, 1]; multiply by `thrust_limit` for newtons.
Entry zero of each history is the common initial coasting plan. Other entries
are accepted iterates, with their cost, step size, regularization, and projected
gradient residual. These are whole fixed-horizon plans, not successive MPC
predictions. Simulation time indexes one selected plan.

The quay occupies y <= 0. Clearance uses the complete rectangular footprint,
which conservatively contains the pointed boat icon. The penalty in the cost
is smooth and does not itself guarantee collision avoidance. The separate
clearance check samples every 0.025 seconds; it is not a continuous-time proof.

All four final runs must converge and pass position (<0.2 m), heading (<3°),
speed (<0.05 m/s), yaw rate (<1°/s), thrust-bound, and positive-clearance checks.
Single-run timings exclude JAX compilation and vary by machine. Numerical
values and artifact geometry are reproducible; timings are diagnostic only.

Ordinary book builds and browser replay do not run these optimizers or access
external services.
