# Brownian MPPI artifacts

Regenerate with `MPLCONFIGDIR=/private/tmp/mppi-mpl .venv/bin/python scripts/build_brownian_mppi_artifacts.py`.

`results.json` records all settings, seeds, definitions, numerical checks, confidence intervals, and runtime. `trials.npz` preserves every trajectory and trial metric. `replay.json` contains eight paths per method for the offline replay. Static SVG, PDF, and PNG figures are in `_static/brownian_mppi/`. The experiment needs NumPy, SciPy, and Matplotlib and makes no network requests.

These are finite-penalty, finite-width teaching adaptations of Kappen's slit and finite-thickness passage constructions, not a reproduction of his figure parameters. Samples represent physical Brownian uncertainty; the proposal correction accounts for guided sampling. The closed-loop experiment uses actual forward samples at every nonlinear control update and an analytically integrated terminal tail. The convolution solution is an independent numerical reference.
