# Steer and mix unevenly heated material

Open [Stirring control](http://localhost:8003/interactive/thermal-reactor-mppi.html#stirring). Warm and cold material keeps entering the reactor. Three reversible stirring inputs change its path. The task is to produce more uniform temperature and conversion in the final two metres while limiting stirring effort.

Start with **Physical fields** and flow arrows, then switch to **Departure from cross-section mean**. That second view removes the longitudinal heating profile: blue and red show colder/warmer material, or lower/higher conversion, relative to other material at the same downstream position. These are neither fixed-target errors nor surrogate prediction errors. The dashed vertical line marks the start of the region used in the cost.

## Model and information

The domain is 8 m × 2 m, with 64 × 32 cells and two physical fields, temperature $T$ in kelvin and remaining reactant fraction $c$. Conversion is $1-c$. This gives **4,096 physical state values**, **three control inputs**, and **60 sampled sequence parameters**. The fields are fully observed. The inlet history is deterministic and known to the planner; random numbers generate optimizer candidates only.

The reaction and diffusion parameters are retained from the earlier reactor. Heating in this separate scenario comes from a temperature-controlled jacket:

$$
\partial_t c+\nabla\cdot(\mathbf v c)=D\nabla^2c-k(T)c,
$$

$$
\partial_t T+\nabla\cdot(\mathbf v T)=D\nabla^2T
+\beta(T_{\rm jacket}-T)-60k(T)c.
$$

Here $D=5\times10^{-4}$ m²/s, $\beta=0.08$ s⁻¹, $T_{\rm jacket}=1050$ K, and
$k(T)=0.1\exp[12000(1/950-1/T)]$ s⁻¹. The jacket supplies heat to colder material and removes heat from hotter material. It is modeled as a distributed heat-exchange term; no solid wall temperature field is solved. Direct heater inputs are zero. These are illustrative assumptions, not calibrated calciner parameters.

Fresh reactant enters with $c_{\rm in}=1$ and

$$
T_{\rm in}(y,t)=900+[10+60\sin(2\pi t/32)]\cos(\pi y/2)\ {\rm K}.
$$

The warmer and colder sides reverse smoothly every half-cycle. Their flow-weighted mean inlet temperature remains 900 K. Every comparison starts from the same unstirred warm state, generated over 400 s with the oscillatory term disabled. The recording then covers 160 s, five inlet cycles. The inlet continues cycling during every prediction and validation tail.

## What the controller changes

The background velocity remains the earlier shear flow,
$v_{0x}=0.16+0.24\eta(1-\eta)$ m/s with $\eta=y/2$ and $v_{0y}=0$.
Three streamfunctions define the stirring modes:

$$
\psi_j(x,y)=0.15\max\left(1-\left(\frac{x-x_j}{2}\right)^2,0\right)^3
\sin^2(\pi y/2),\qquad x_j\in\{2,4,6\}\ {\rm m}.
$$

Each control multiplies the corresponding incompressible velocity perturbation:

$$
\mathbf v=\mathbf v_0+\sum_{j=1}^3 a_j(t)
\begin{pmatrix}\partial_y\psi_j\\-\partial_x\psi_j\end{pmatrix},
\qquad \sum_j a_j(t)^2\le1.
$$

Positive controls add counterclockwise circulation; negative controls reverse it. The background flow remains present, so an active mode need not produce closed streamlines. The modes overlap and move material across the channel centreline. Their normal velocity perturbations vanish at the inlet, outlet, and side walls. Throughput is unchanged.

This is an idealized stirring model. Velocity responds instantaneously to the inputs; the simulation does not solve momentum equations, pressure, impeller dynamics, or turbulence. The shared capacity and squared effort are normalized actuator-load assumptions, not measured motor power.

Changing the flow changes which material encounters other material and how long it spends in different regions. Stirring can stretch temperature and concentration variations into thinner structures, which diffusion smooths. [Optimal stirring studies](https://arxiv.org/abs/1009.0834) distinguish such spatial rearrangement from molecular homogenization and often use multiscale mixing measures. This demonstration instead measures physical temperature and conversion variation in a specified downstream region. Jacket exchange and reaction also change these fields, so their variance is not a conserved passive-tracer quantity.

## Objective and MPPI update

For each downstream position $x$, let $\bar T(x)$ and $\bar c(x)$ be the means across the width. Define

$$
M(X)=\operatorname{mean}_{6\le x\le8,\ y}
\left[\left(\frac{T-\bar T(x)}{20\ {\rm K}}\right)^2
+\left(\frac{c-\bar c(x)}{0.1}\right)^2\right].
$$

Conversion and remaining-fraction variances are equal. The longitudinal temperature gradient is not penalized. The 40 s planning objective uses 20 controls held for 2 s:

$$
J=\frac1{20}\sum_{k=0}^{19}
\left[M(X_{k+1})+0.01\|a_k\|^2+0.005\|a_k-a_{k-1}\|^2\right]+M(X_{20}),
$$

where $a_{-1}$ is the previously applied control. Reversals or alternating patterns receive no reward. They occur only if a sampled update improves this objective and passes the checks.

The controller begins with zero stirring. At each decision, its exact retained control tape supplies a reference $A_{\rm ref}$, fixed throughout four updates. It samples 256 independent latent arrays $Z_i\sim\mathcal N(\mu,I)$ of shape $20\times3$. Their entries are filtered along time:

$$
B_0=Z_0,\qquad B_k=0.6B_{k-1}+0.8Z_k.
$$

Candidate controls are the Euclidean projection of $A_{\rm ref}+0.7B$ onto the unit ball separately at each time. The reference law is $p_0(Z)=\mathcal N(0,I)$, and the proposal initially has $\mu=0$. The corrected weights are

$$
w_i\propto\exp(-J_i/0.02)\frac{p_0(Z_i)}{q_\mu(Z_i)}.
$$

Likelihoods are calculated in the independent latent coordinates, before filtering or projection. The weighting temperature is dimensionless. Stable logarithmic normalization gives the weights and effective sample size (ESS). Their weighted latent mean generates a new control sequence, whose PDE evolution is simulated separately. A weighted average of candidate field images is never executed.

Candidate constraints are checked at 2 s states. The controller tries update fractions 1, 1/2, 1/4, and 1/8, accepting the first objective improvement greater than $10^{-7}$ that also passes validation every 0.25 s. Validation covers the 40 s proposed sequence and a further 40 s at zero stirring. Every check retains the cycling inlet. Rejected updates leave the exact incumbent unchanged; shifting it appends zero stirring. Execution stops and reports failure if no feasible continuation remains.

The limits are 1070 K everywhere and 95% flow-weighted outlet conversion. Under the stated continuum model, jacket exchange and an endothermic reaction cannot create temperatures above the largest initial, inlet, or jacket temperature. Thus 1070 K is primarily a consistency check here. Discrete checks do not guarantee continuous-time constraint satisfaction.

## Numerical method and comparisons

The new solver uses conservative fluxes in both directions, with sign-aware upwinding, minmod-limited second-order reconstruction, centered diffusion, and SSPRK2. Vertex streamfunction differences give face velocities with cancelling discrete divergence. The step cap is 0.05 s, further limited when necessary by outgoing transport, diffusion, and reaction rates. Inlet forcing is evaluated at both integration-stage times. No clipping of physical fields hides solver errors.

All executed fields and fine validations use JAX. The shipped experiment optionally evaluates large candidate batches with a fused Metal implementation of the same numerical PDE. This is not a learned model. Individual proposals are scored on the CPU, and parity tests compare complete 40 s candidate rollouts. The GPU step cap is checked against a worst-case outgoing-flux bound over the entire admissible control ball.

The comparisons are no stirring, the best finely validated constant vector on the grid with spacing 0.25 inside the unit ball, and three MPPI runs with optimizer seeds 19744–19746. The constant vector minimizes the full-record mean stage cost plus terminal mixing error; it is not claimed to be the best continuous-valued constant control. All comparisons have identical initial states and inlet histories.

Reported mixing error is integrated over the 0.25 s recording grid. Outlet temperature standard deviation and outlet conversion use outgoing material-flux weights. The main results include effort, control variation, constraints, accepted/rejected updates, ESS, runtime, and variability across optimizer seeds. This variability is not uncertainty in the physical feed.

The transport benchmark translates a smooth periodic scalar around a unit square with zero physical diffusion. Any variance lost there is numerical smoothing. The experiment also repeats the recorded controls with halved timesteps and a doubled mesh, prolonging the same initial cell averages conservatively. Compare the controller ordering on the finer grid before attributing an advantage to physical mixing. A visibly smoother image alone does not establish better mixing.

## Recorded results

The constant-control search evaluated 257 grid vectors and selected $(0,0,0)$. Thus the best grid constant and unstirred baselines coincide. The MPPI controls change throughout all five inlet cycles; their direction changes were not prescribed.

| Control | Mean variation | Downstream T RMS (K) | Outlet T std. (K) | Effort (s) | Lowest outlet conversion |
| --- | ---: | ---: | ---: | ---: | ---: |
| No stirring / best grid constant | 0.05380 | 4.58 | 3.29 | 0.00 | 99.865% |
| MPPI, seed 19744 | 0.03554 | 3.71 | 2.62 | 113.77 | 99.843% |
| MPPI, seed 19745 | 0.03651 | 3.76 | 2.59 | 116.49 | 99.838% |
| MPPI, seed 19746 | 0.03527 | 3.69 | 2.55 | 125.72 | 99.774% |

Across the three optimizer seeds, mean downstream variation is 0.03577 ± 0.00066 (sample standard deviation). Relative to the shared baseline, the paired reduction is 33.5 ± 1.2 percentage points. Mean outlet temperature standard deviation is 2.58 ± 0.04 K, versus 3.29 K without stirring. Three seeds give a limited measure of optimizer variability on this one deterministic feed history; they do not establish performance across different physical conditions.

Every recording has zero temperature and conversion violation duration on the 0.25 s observation grid. The largest recorded temperature is 1044.59 K and the lowest outlet conversion is 99.774%. The controller rejected 449 of 960 updates, retaining its incumbent. Mean ESS is 31.3 of 256. Average planning time is 4.18 s per decision with the Metal candidate evaluator. This exceeds the 2 s simulated control interval, so these are offline experiments, not a demonstrated real-time implementation.

Halving the timestep changes seed 19744's temperature field by 0.0017 K RMSE and mean variation by 0.0084%. Doubling the mesh to 128 × 64 cells changes its temperature field by 0.354 K RMSE and increases variation by 2.54%; the baseline variation increases by 0.90%. The same saved controls reduce variation by 32.9% on the fine mesh, compared with 33.9% on the original mesh. All five recorded control sequences were replayed under both refinements. On the fine mesh, seeds 19745 and 19746 reduce variation by 30.4% and 32.2%; their variation increases by 3.43% and 4.33% relative to the original mesh. None of the refined replays violates the checks. This comparison supports the controller ordering while exposing numerical smoothing; it is not a proof of mesh independence or fully resolved molecular mixing.

With zero physical diffusion, the periodic advection benchmark retains 95.6% of scalar variance at 64 × 64 cells with the new scheme, versus 29.1% with first-order upwinding. Retention increases to 99.0% at 128 × 128. The candidate-count study at the opening state shows no monotonic improvement across 128, 256, and 512 candidates in these three seeds; see `sensitivity.json` for costs, ESS, and runtime. More samples alone do not establish optimizer convergence.

The regression suite passed 51 tests, with one older optional backend test skipped. The new Metal parity test passed over complete 40 s rollouts. Export checks verified 29 field arrays, common initial states, timestamps, bounded controls, scalar diagnostics, and retained-plan objective values. `validation.json` records source hashes; `runtime-validation.json` records dependency versions and backend checks.

## Replay, outputs, and regeneration

The browser reads local precomputed float32 field arrays. Absolute scales are 800–1070 K and 0–1 conversion; cross-section deviations use ±60 K and ±0.2 conversion. Flow arrows are derived from the saved numerical velocity bases and current controls, with arrow displacement representing 1 s. They are instantaneous velocities, not particle trajectories.

Three decisions, at 16, 40, and 72 s of seed 19744, retain six candidate futures from the fourth update, plus the raw weighted proposal and the validated plan. The six movies show weight ranks 1, 2, 3, 65, 129, and 256, so they include both strong and weak contributions rather than a random subset. Candidate weights remain those of the full batch. The main replay executes the first action and replans; the saved future movies illustrate a plan held over its prediction horizon.

From the repository root:

```sh
# Portable CPU execution
uv run python scripts/build_thermal_reactor.py stirring

# Optional Apple GPU candidate evaluation, using the installed thermal-metal extra
uv run python scripts/build_thermal_reactor.py stirring --backend metal

uv run pytest tests/test_thermal_stirring.py tests/test_thermal_recovery.py tests/test_thermal_plume.py tests/test_thermal_reactor.py tests/test_mppi_control.py
uv run python -m http.server 8003 --bind 127.0.0.1
```

The stage generates comparisons, candidate movies, sampling sensitivity, transport benchmarks, refinement results, and figures. Resumable checkpoints live under ignored `outputs/thermal_stirring/`; changing the solver or planner invalidates them. Compact reports live alongside this document, replay data in `interactive/thermal-stirring-data/`, and figures under `_static/thermal_reactor/`. The page performs no optimization or training. The earlier plume, recovery, and surrogate artifacts remain separate.
