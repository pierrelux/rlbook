# Restore the reactor's operating field

Open [Field recovery](http://localhost:8003/interactive/thermal-reactor-mppi.html#recovery). The cold inlet pulse has ended, but its temperature and conversion deficits are still traveling through the reactor. Four heaters must restore the desired fields while keeping maximum temperature at or below 1070 K and outlet conversion at or above 95%.

The replay compares unchanged heating with direct-PDE MPPI. Select **Error relative to the target** to see the remaining cold and hot regions, or compare MPPI with the **Supplied feasible schedule**. The **Sampled control update** view shows candidate trajectories and the last weighted update at three planning times. Every image comes from a numerical PDE rollout.

## Target, initial state, and information

The domain, 64 × 32 mesh, two fields, shear flow, and reaction model are the same as in the [moving-plume demonstration](../thermal_plume/README.md). The target is the warm operating field obtained after 400 s with all four heaters at **0.46**. Its maximum temperature is 1063.54 K, leaving room for corrective heating. This differs from the 0.48 nominal heating in the introductory prescribed-plume view.

Starting from the target, the same localized cold pulse enters during 6–14 s while heating stays at 0.46. Control begins at **14 s**, from the resulting disturbed field, and the recording ends at **80 s**. Every comparison starts from this identical state. The inlet remains at its known normal profile throughout recovery. Both temperature and remaining reactant fraction are fully observed.

This is deterministic field restoration solved with a sampling algorithm. The random numbers generate candidate controls; there is no physical process noise, uncertain forecast, or learned surrogate in this experiment. The earlier surrogate checkpoint and uniform-flow MPPI results remain separate.

## Objective and controls

Let $a=1-c$ denote conversion and let $T_\star,a_\star$ be the target fields. The spatial tracking error is

$$
E(X)=\operatorname{mean}_{\rm cells}\left[
\left(\frac{T-T_\star}{20\,\mathrm K}\right)^2+
\left(\frac{a-a_\star}{0.1}\right)^2\right].
$$

The same target is used for every candidate and every comparison. Squared error penalizes both warmer and colder departures, including heat delivered to material outside the cold patch. The four broad heaters cannot independently set each of the 4,096 state components.

MPPI replans every 2 s over 20 control intervals, a 40 s horizon. With the previously applied control supplying $u_{-1}$, its objective is

$$
J=\frac1{20}\sum_{k=0}^{19}\left[E(X_{k+1})
+0.01\operatorname{mean}_j\left(\frac{u_{k,j}-0.46}{0.05}\right)^2
+0.005\operatorname{mean}_j\left(\frac{u_{k,j}-u_{k-1,j}}{0.05}\right)^2\right]
+E(X_{20}).
$$

The control terms discourage departures from nominal heating and abrupt changes. Heating energy is reported separately as the integral of mean heater fraction; it is not measured in joules. The tracking objective does not require an energy advantage.

Six knots per heater give **24 sampled parameters**. The knots are linearly interpolated at 20 decision points, and the resulting controls are held for 2 s. All four controls can increase or decrease within [0, 1]. Flow remains prescribed: heating affects temperature and reaction, not the plume's transport velocity.

## Initialization and weighted updates

The supplied feasible schedule sets H1's six physical knots to `[0.50, 0.50, 0.46, 0.46, 0.46, 0.46]`, with all other knots at 0.46. After its 40 s horizon, heating returns to 0.46. This is an explicitly supplied recovery schedule, not a result discovered by the optimizer.

At each decision, the exact feasible control tape is retained. Sampling its values at the six knot locations gives a proposal center $u_{\rm seed}$. The latent decoder is

$$
u_{\rm knots}(z)=\operatorname{sigmoid}\bigl(\operatorname{logit}(u_{\rm seed})+0.12z\bigr),
\qquad p_0(z)=\mathcal N(0,0.49I).
$$

The reference and decoder stay fixed through four updates. The proposal starts at $q=\mathcal N(0,0.49I)$ and samples 256 candidates per update. As its mean changes, weights include the Gaussian density ratio:

$$
w_i\propto\exp[-J_i/0.01]\frac{p_0(z_i)}{q(z_i)},\qquad
\mu_{\rm weighted}=\sum_iw_iz_i.
$$

Weights use stable logarithmic normalization. The weighting temperature 0.01 is dimensionless and unrelated to temperature in kelvin. The Gaussian reference regularizes the update around the starting schedule; it is separate from the physical tracking objective. Finite sampling, mean projection, this regularization, and the bounded parameterization limit any optimality claim.

Candidate costs and constraints are checked at 2 s states. Invalid candidates or candidates violating either constraint receive zero weight. The controller simulates the weighted mean as a new sequence, trying update fractions 1, 1/2, 1/4, and 1/8. It accepts the first objective improvement exceeding numerical tolerance that also passes a finer PDE constraint check. Validation records states every 0.25 s over the proposed 40 s sequence plus another 40 s at nominal heating.

Rejected updates retain the exact feasible tape. Its six-knot approximation is only a sampling center; it cannot silently replace that tape. The first action is executed in the PDE, the exact tape is shifted, and nominal heating is appended. Failure to retain a feasible continuation stops execution and is reported.

## Recorded recovery

The shipped recording uses optimizer seed **19744**. All numbers below cover t=14–80 s and use the same 0.25 s output grid.

| Schedule | Integrated field error | Heating energy | Maximum temperature | Minimum outlet conversion |
|---|---:|---:|---:|---:|
| Unchanged heating | 3.709 s | 30.360 s | 1063.54 K | 94.747% |
| Supplied feasible schedule | 2.098 s | 30.484 s | 1066.94 K | 95.271% |
| MPPI | 1.962 s | 30.439 s | 1066.16 K | 95.329% |

MPPI reduces integrated field error by **47.1%** relative to unchanged heating and **6.5%** relative to the supplied schedule. It adds upstream heat early, then reduces downstream heating as the warmed material moves through. This schedule lowers unwanted warming as well as the cold deficit. Residual field error remains because each heater affects an entire quadrant.

Unchanged heating produces 7.5 s below the 95% conversion threshold. The supplied schedule and MPPI have no detected temperature or conversion violations on the output grid. Outlet conversion is weighted by downstream material flux, since the flow speed varies across the reactor width.

The controller accepted 24 updates and rejected 108 over 33 decisions. Many later proposals add error to an already recovering state, so the retained tape supplies the return toward nominal heating. Average measured planning time was **13.9 s per decision** on this local run, including fine validation and first-use compilation where needed. The 2 s control interval is simulated time; this is an offline replay, not a real-time controller.

The opening decision is also evaluated with 128/256/512 candidates and three optimizer seeds. Those results appear in the replay and `sensitivity.json`. They measure sampling variability at a fixed physical state, not uncertainty in the plant. They do not establish global optimality or performance for other disturbances.

## Replay and validation

The absolute field scales are 680–1120 K and 0–1 conversion. Signed target-error scales are ±100 K and ±0.2 conversion. These targets differ from the introductory plume view's matched no-pulse references. Scalar diagnostics always refer to the actual physical state.

Six candidate futures are saved at t=14, 22, and 34 s, from each decision's fourth update. Weights remain those of the complete batch. The raw weighted proposal and the plan retained after damping/validation are simulated separately. The actual replay uses the final plan's first action, not a weighted average of field images. Snapshot movie generation is excluded from reported optimizer runtime.

Tests cover cost normalization, bounded controls, latent importance correction, invalid batches, failed fine validation, exact-tape retention, prescribed inlet timing, and batched PDE execution. Artifact checks compare binary field shapes, hashes, timestamps, target subtraction, scalar diagnostics, and saved candidate initial states. A separate refinement report repeats the executed controls with halved integration steps and a doubled mesh, preserving initial cell averages. Discrete checks do not guarantee continuous-time feasibility. First-order upwind numerical diffusion still contributes substantially to plume spreading.

Halving the integration step changes the temperature field by 0.0014 K RMSE. On the 128 × 64 mesh, temperature differs by 0.78 K RMSE and outlet conversion by 0.00186 MAE. The finer run reaches 1068.42 K and a minimum outlet conversion of 95.356%; neither limit is violated on its output grid. These comparisons use the same recorded controls, with no further optimization.

The thermal and MPPI regression suite passes 41 tests, with one optional Metal-backend test skipped. Browser checks cover playback, pause, reset, scrubbing, comparison and decision selection, view switching, lazy field loading, and the existing surrogate view. Desktop and 390-pixel layouts, both static figures, and candidate movies were inspected. See `browser_validation.json` for the replay checks and source hashes.

## Regenerate

From the repository root:

```sh
uv run python scripts/build_thermal_reactor.py recovery
uv run pytest tests/test_thermal_recovery.py tests/test_thermal_plume.py tests/test_thermal_reactor.py tests/test_mppi_control.py
uv run python -m http.server 8003 --bind 127.0.0.1
```

The explicit stage generates the complete recovery, supplied baselines, sampled futures, sensitivity checks, refinement results, and static figures. Reports live here; a dedicated manifest and field arrays live in `interactive/thermal-recovery-data/`. The page loads arrays lazily and performs no simulation, optimization, or training.
