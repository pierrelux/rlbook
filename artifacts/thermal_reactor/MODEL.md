# MPPI control of a thermal reactor

The opening **PDE dynamics** view is a separate [prescribed shear-flow scenario](../thermal_plume/README.md), with a localized cold pulse and two scripted heating schedules. This document describes the original uniform-flow surrogate and MPPI experiment, retained in the other views. No accuracy or control-performance result below applies to the new plume scenario.


This standalone prototype controls a flowing, reacting material with four heater zones. Each numerical state contains 2,048 temperatures and 2,048 remaining-reactant fractions. The heatmaps display those physical fields directly. The optimizer samples 24 control parameters: six knots for each of four heating inputs. Increasing the spatial resolution increases the state dimension without increasing this control parameter count, although it makes every rollout more expensive. The mesh subdivides physical space; it is not a grid over the 4,096-dimensional reactor state space.

Open `interactive/thermal-reactor-mppi.html` through a local static server. The page loads saved experiments and does no training, simulation, or optimization. It remains separate from the textbook chapter.

## Model and units

The domain is 8 m long and 2 m wide, with 64 × 32 finite-volume cells. Material enters on the left and leaves on the right. The two fields are temperature $T$, measured in kelvin, and remaining reactant fraction $c$, which is dimensionless. Conversion is $1-c$.

$$
\partial_t c+v\partial_xc=D_c\nabla^2c-k(T)c,
\qquad
\partial_tT+v\partial_xT=D_T\nabla^2T-\beta(T-T_{\mathrm{wall}})
-\Delta T_{\mathrm{rxn}}k(T)c+20\sum_{j=1}^4u_jq_j(x,y).
$$

Each $q_j$ is the indicator of one quadrant; its input $u_j\in[0,1]$ is a fraction of maximum heating. Heater order is upstream-bottom, downstream-bottom, upstream-top, downstream-top. The parameters are:

| Quantity | Value |
|---|---:|
| Flow speed $v$ | 0.2 m/s |
| Both diffusivities | $5\times10^{-4}$ m²/s |
| Heat-loss rate $\beta$ | 0.01 s⁻¹ |
| Wall temperature | 700 K |
| Reaction cooling $\Delta T_{\mathrm{rxn}}$ | 60 K per unit reactant fraction |
| Maximum local heating rate | 20 K/s |
| Reaction rate $k(T)$ | $0.1\exp[12000(1/950-1/T)]$ s⁻¹ |
| Inlet temperature | $800+10\cos(\pi y/2)+g_t$ K |
| Inlet reactant fraction | 1 |

These are illustrative teaching assumptions, not calibrated calciner parameters. The model is inspired by the author's [flash-clay-calciner project](https://github.com/pierrelux/flash-clay-calciner) and the physical setting described by [Cantisani et al.](https://arxiv.org/abs/2404.07674). It omits separate solid/gas phases, detailed chemistry, residence-time distributions, and industrial geometry.

Advection uses conservative first-order upwind fluxes. Centered diffusive fluxes vanish at all boundaries; the inlet supplies the advective flux and the outlet uses the interior advective state. The time integrator is explicit SSPRK2 with substeps no larger than 0.1 s and a tighter transport/reaction stability bound when needed. The PDE states are never clipped. The 40 s residence time sets the planning horizon.

The observed inlet disturbance follows the exact 2 s discretization of an Ornstein–Uhlenbeck process with correlation time 20 s and stationary standard deviation 15 K. The inlet is held constant within each control interval. Every forecast begins with the current observed value. Subsequent values either follow its conditional mean or independently sampled conditional futures. Forecast, proposal, training, and realized-disturbance streams are distinct. All three controllers in a paired trial encounter the same realized inlet sequence.

## Surrogate

The residual CNN has four 3 × 3 convolutions with channel widths 6/16/16/16/2 and dilations 1/2/4/1. Its inputs are normalized temperature and reactant fields, the heater-fraction map, normalized inlet temperature, and two coordinate channels. SiLU follows the first three convolutions. Edge padding supplies the convolutional boundary treatment. The output is the field change over one 2 s interval, converted back to physical units.

The training residual is unrestricted. At inference, the predicted reactant fraction is explicitly projected onto [0, 1] after each transition; temperature is unrestricted. This projection prevents impossible fractions, but does not establish physical accuracy. The held-out report also records the unprojected rollout errors and the largest raw fraction-bound excess.

Training data contain 256 independent, 120 s PDE episodes: 179 training, 38 validation, and 39 test episodes. Episodes include warm operation, cold starts, smooth spatial transients, varying heating, and inlet fluctuations. Whole episodes belong to exactly one split. Channel means and standard deviations use only training episodes. Adam uses learning rate $10^{-3}$, batch size 16, and a loss combining one-step field error, five-step rollout error, and outlet error. Checkpoint selection uses a fixed 40 s validation window from 16 validation episodes. The command supports up to 200 epochs and patience 20; the shipped run completed three full epochs within a local CPU training budget and retained the lowest validation-loss checkpoint among them. Partial work in the fourth epoch was discarded.

Held-out evaluation covers three 40 s windows in each of the 39 test episodes. The supplied checkpoint gives average temperature RMSE **4.59 K**, average outlet-conversion MAE **0.00393**, and 95th-percentile worst, time-aligned peak-temperature underprediction **8.31 K**. It passes the specified average-error thresholds of 10 K and 0.02. These thresholds describe this held-out distribution; they do not certify all sampled control sequences. Raw, unprojected fraction predictions exceed [0, 1] by as much as 0.0675 in this test set. Full window results are in `surrogate_validation.json`.

The optional Apple-GPU backend uses MLX to evaluate the same JAX-trained weights. A tested transition differed by at most 0.000123 K from JAX; the tested candidate costs agreed to float32 precision. JAX remains the training framework, and the reference plant remains the JAX PDE solver. CPU JAX inference is also supported.

## Sampled control update

A Gaussian proposal samples six latent knots for each heater. Each knot is transformed by

$$
u=\operatorname{sigmoid}(\operatorname{logit}(0.48)+0.35z),
$$

then linearly interpolated into 20 piecewise-constant 2 s controls. At finite temperature, the fixed reference regularizes the sampled update toward the warm operating controls. The default proposal variance is $0.7^2$. A fixed zero-mean Gaussian reference has the same variance. The weighting temperature 0.025 is a dimensionless cost scale, unrelated to the physical temperature in kelvin. The controller evaluates 128 candidates against four shared conditional inlet futures, averages each candidate's costs over those futures, then computes

$$
w_i\propto\exp[-\widehat C_i/0.025]\frac{p_0(z_i)}{q(z_i)},
\qquad \bar z_{\mathrm{new}}=\sum_iw_iz_i.
$$

The density ratio and weighted mean are calculated in latent coordinates. The nonlinear transform means that the resulting heating sequence need not equal the weighted average of the displayed physical heating sequences. Normalization subtracts the largest log weight for stability. This disturbance-averaged stochastic shooting construction does not require, or claim, the covariance matching of classical path-integral stochastic control.

The per-step cost is mean heater fraction plus penalties:

- $5[(0.95-\text{outlet conversion})_+/0.05]^2$;
- $10[(\max T-1070)_+/20]^2$;
- $0.05\,\mathrm{mean}[((u_t-u_{t-1})/0.1)^2]$.

Costs are averaged over the 20 horizon steps. Reported normalized heating energy is the integral of the mean heater fraction, in seconds at full average heating. It is not a calibrated energy in joules; the model has no density or heat-capacity conversion factor.

Each decision makes two weighted updates. A proposed mean must improve the estimated cost and pass a conditional-mean PDE rollout check for outlet conversion ≥95% and maximum temperature ≤1070 K. Rejected updates retain the checked incumbent. The previous sequence is shifted for the next decision; if it becomes infeasible, constant 0.48 heating is rechecked from the current measured state. If neither is feasible, the episode stops and records a planning failure. Mean-forecast feasibility is not a guarantee under random inlet disturbances.

Planning checks inspect 2 s output states. A separate audit replays the executed controls with 0.1 s output diagnostics and reports constraint residuals and violation durations on that grid. Neither discrete grid establishes a continuous-time guarantee.

## Experiments and replay

Each of eight independent paired trials lasts 160 s and starts from the same PDE warm operating state, obtained by 400 s at constant 0.48 heating. The methods are constant heating, surrogate MPPI with conditional-mean forecasts, and surrogate MPPI with stochastic forecasts. Every applied control advances the PDE solver.

`evaluation.json` reports energy, conversion, maximum temperature, violation durations, failures, accepted/rejected updates, effective sample size, prediction errors, and runtime. `execution_audit.json` reports finer-grid residuals and paired 95% Student-t intervals. Incomplete episodes are reported explicitly and excluded from full-duration paired energy differences. Eight pairs give limited precision; an energy advantage is not a success criterion.

`sensitivity.json` compares 32/128/512 candidates at four futures and 1/4/16 futures at 128 candidates, using three proposal/forecast seeds at the same warm planning state. The chosen complete sequences are assessed against 32 fresh paired PDE disturbance futures. A separate comparison evaluates surrogate and direct-PDE planning from that state. Timings include proposal generation, scoring, PDE acceptance, and initial compilation where required; the two model-comparison calls also capture candidate snapshots; the comparison reports the actual held-out PDE cost of the resulting sequence. This inexpensive PDE need not be slower than the CNN surrogate.

The replay shows synchronized executed temperature and conversion maps, actual heater settings, and sampled futures at saved planning times. Six candidate movies use the same displayed inlet future. Their costs include the entire forecast ensemble, and their weights remain those of the full 128-candidate batch. The proposed weighted control is simulated again in both models. Candidate field images are never averaged. The surrogate/PDE comparison uses identical controls and inlet histories, with signed error defined as surrogate minus PDE.

Temperature scales are fixed at 750–1120 K, conversion at 0–1, and signed-error scales at ±20 K and ±0.05 conversion. Values outside a display scale saturate; scalar extrema and errors remain available. Binary arrays are little-endian float32 with shape and channel order in the JSON manifest. Field files load lazily when their runs or planning snapshots are selected.

## Regeneration and checks

Run commands from the repository root, using its Python environment:

```sh
uv sync --group dev
uv run python scripts/build_thermal_reactor.py generate
uv run python scripts/build_thermal_reactor.py train --epochs 3
uv run python scripts/build_thermal_reactor.py validate
uv run python scripts/build_thermal_reactor.py evaluate
uv run python scripts/build_thermal_reactor.py audit
uv run python scripts/build_thermal_reactor.py export
uv run python -m http.server 8003 --bind 127.0.0.1
```

Then open `http://localhost:8003/interactive/thermal-reactor-mppi.html`. Training is explicit; reopening the page never regenerates data. Large training arrays live in ignored `outputs/thermal_reactor/training`. The compact checkpoint, metadata, summaries, and replay data are retained. Whole completed evaluation episodes can be resumed; checkpoint and planner fingerprints must match. Use a different `--output-dir` for a new model or experiment, and `--data-dir` for an alternate data cache.

On Apple silicon, install the optional backend with `uv sync --group dev --group thermal-metal`, then use `evaluate --backend metal`. It requires access to a local Metal device. To retrain beyond the shipped three-epoch budget, use `train --epochs 200` in a fresh output directory, validate it, and regenerate its experiments.

```sh
uv run pytest tests/test_thermal_reactor.py tests/test_mppi_control.py
THERMAL_TEST_METAL=1 uv run pytest tests/test_thermal_reactor.py -k metal
```

Tests cover finite-volume boundary balance, zero-flux conservation, uniform fields, reaction bounds and cooling, batched transitions, time refinement, proposal correction, bounded latent controls, explicit surrogate projection, failed-update retention, stopping on an infeasible incumbent, and independent random streams. The numerical refinement report compares the same 40 s trajectory with a halved maximum timestep and a doubled mesh: temperature RMSE differences are 0.00292 K and 0.816 K, respectively. This is a sensitivity check against a finer numerical solution, not an analytic PDE error bound.
