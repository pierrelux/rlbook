# Thermal-reactor MPPI prototype

The page now opens with a [moving cold-plume demonstration](../thermal_plume/README.md) under prescribed shear flow. The original uniform-flow MPPI results documented below remain available in the other tabs. The new PDE-only scenario does not use this surrogate checkpoint.


The standalone replay shows a heated, reacting flow on a 64 × 32 mesh. Each cell contains temperature and remaining reactant fraction: **4,096 reactor state values**. Four heater inputs are represented by six knots each, so MPPI samples **24 control parameters**. This spatial mesh is distinct from a grid over the entire state space.

## Open the saved replay

From the repository root:

```sh
python -m http.server 8003 --bind 127.0.0.1
```

Open [the replay](http://localhost:8003/interactive/thermal-reactor-mppi.html). It reads local precomputed data; opening it starts no training, simulation, or optimization.

- **Executed fields:** compare controllers under the same realized inlet disturbance. Play, pause, reset, scrub time, change speed, and select a trial.
- **Inside an MPPI update:** inspect six candidate futures, their heating sequences, costs, and full-batch weights. The updated latent parameters produce a new control sequence, which is simulated again to obtain its fields.
- **Surrogate vs PDE:** compare both models under identical controls and inlet histories. Signed error is surrogate minus PDE.

The model is an illustrative flow-through reactor inspired by flash calcination. It uses advection, diffusion, temperature-dependent reaction, heat loss, four heating zones, and endothermic cooling. The coefficients are teaching assumptions, not calibrated industrial parameters. [MODEL.md](MODEL.md) gives the equations, units, boundary conditions, objective, numerical methods, surrogate architecture, and limitations.

## Model and validation

The controller replans every 2 s over 40 s, using 128 candidates, four shared conditional inlet futures, and two weighted updates. Inlet-temperature uncertainty follows an observed Ornstein–Uhlenbeck process with a 20 s correlation time and 15 K stationary standard deviation. Candidate costs are averaged across futures before weighting; Gaussian likelihood corrections are calculated in latent control coordinates.

A JAX/Optax residual CNN is trained on 256 PDE episodes, split by whole episode into 179/38/39 training/validation/test episodes. The supplied checkpoint was selected after three completed epochs. Predicted reactant fractions are explicitly projected onto [0, 1]; temperature is unrestricted. Raw bound errors are also reported.

| Held-out 40 s rollout metric | Result |
|---|---:|
| Average temperature RMSE | 4.59 K |
| Average outlet-conversion MAE | 0.00393 |
| 95th-percentile worst time-aligned peak underprediction | 8.31 K |
| Requested average-error thresholds | Passed: below 10 K and 0.02 |

Every applied control advances the numerical PDE. Proposed updates must pass a conditional-mean PDE check for outlet conversion ≥95% and temperature ≤1070 K. Failed updates retain the validated incumbent; lack of a feasible incumbent stops the episode. Mean-forecast feasibility does not guarantee feasibility under every random disturbance.

All 24 executions completed their 160 s episodes. Every controller maintained both limits on the 2 s reporting grid and the finer 0.1 s audit grid. The table uses 2 s samples.

| Controller | Mean heating energy (s) | Lowest outlet conversion | Highest temperature |
|---|---:|---:|---:|
| Constant heating | 76.800 | 95.09% | 1059.5 K |
| Mean-forecast MPPI | 75.807 | 95.30% | 1051.9 K |
| Stochastic-forecast MPPI | 75.759 | 95.16% | 1052.4 K |

Both MPPI variants used about 1.3% less heating than the baseline. Stochastic minus mean-forecast MPPI was −0.048 ± 0.099 heating-seconds (paired 95% Student-t interval, eight pairs), which does not establish an advantage between them. Detailed results are in `evaluation.json`; finer 0.1 s checks are in `execution_audit.json`. Heating energy is measured in seconds at full average heating, not calibrated joules. Local stochastic planning averaged 4.45 s per decision, longer than the 2 s control interval: these are offline experiments, not a real-time implementation claim.

## Regenerate

The repository environment supplies JAX, Optax, NumPy, SciPy, and Matplotlib. Run each costly stage explicitly:

```sh
uv run python scripts/build_thermal_reactor.py generate
uv run python scripts/build_thermal_reactor.py train --epochs 3
uv run python scripts/build_thermal_reactor.py validate
uv run python scripts/build_thermal_reactor.py evaluate
uv run python scripts/build_thermal_reactor.py audit
uv run python scripts/build_thermal_reactor.py export
```

The training command supports up to 200 epochs. Use a fresh `--output-dir` for a different checkpoint or experiment. Completed evaluation episodes can be resumed when their checkpoint and planner fingerprints match. Large training arrays remain in ignored `outputs/thermal_reactor/training` (about 244 MB).

On Apple silicon, `uv sync --group dev --group thermal-metal` installs the optional GPU inference backend. Then use `evaluate --backend metal`. This evaluates the same JAX-trained weights and is checked against JAX, including active constraint penalties.

```sh
uv run pytest tests/test_thermal_reactor.py tests/test_mppi_control.py
THERMAL_TEST_METAL=1 uv run pytest tests/test_thermal_reactor.py -k metal
```

The checks cover transport conservation, boundaries, constant fields, reaction bounds and cooling, refinement, normalization and data splits, proposal correction, scenario-cost averaging, control bounds, failed-update handling, independent noise streams, and replay-array consistency. `sensitivity.json` compares candidate and forecast counts and direct-PDE versus surrogate planning. Static PNG/PDF/SVG comparisons are in `_static/thermal_reactor`; replay binaries and their manifest are in `interactive/thermal-reactor-data`.

In the fixed-state comparison, direct-PDE planning took 11.25 s and surrogate planning 14.57 s, including validation and snapshot work. Their held-out PDE costs were 0.47250 and 0.47331. The surrogate did not accelerate this inexpensive PDE in that benchmark. Increasing candidate or forecast counts did not monotonically improve the resulting PDE cost across the three planning seeds.
