---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# MPC Variants and Reliable Operation

Receding-horizon control supplies feedback by repeatedly solving one nominal
finite-horizon problem. How must that problem and its surrounding controller
change when the task tracks a reference, values economic output, contains
uncertainty or discrete modes, or misses its solver deadline?

Once the basic idea of receding-horizon control is clear, it is helpful to see how the same backbone accommodates many variations. In every case, we transcribe the continuous-time optimal control problem into a nonlinear program of the form

$$
\begin{aligned}
    \text{minimize} \quad & c(\mathbf{x}_N) + \sum_{k=0}^{N-1} w_k\,c(\mathbf{x}_k, \mathbf{u}_k) \\
    \text{subject to} \quad & \mathbf{x}_{k+1} = \mathbf{F}_k(\mathbf{x}_k, \mathbf{u}_k) \\
                            & \mathbf{g}(\mathbf{x}_k, \mathbf{u}_k) \leq \mathbf{0} \\
                            & \mathbf{x}_{\min} \leq \mathbf{x}_k \leq \mathbf{x}_{\max} \\
                            & \mathbf{u}_{\min} \leq \mathbf{u}_k \leq \mathbf{u}_{\max} \\
    \text{given} \quad & \mathbf{x}_0 = \hat{\mathbf{x}}(t) \enspace .
\end{aligned}
$$

The components in this NLP come from discretizing the continuous-time problem with a fixed horizon $[t, t+T]$ and step size $\Delta t$. The stage weights $w_k$ and discrete dynamics $\mathbf{F}_k$ are determined by the choice of quadrature and integration scheme. With this blueprint in place, the rest is a matter of interpretation: how we define the cost, how we handle uncertainty, how we treat constraints, and what structure we exploit.

## Tracking MPC

How should the finite-horizon objective penalize deviation from a time-varying
reference trajectory?

The most common setup is reference tracking. Here, we are given time-varying target trajectories $(\mathbf{x}_k^{\text{ref}}, \mathbf{u}_k^{\text{ref}})$, and the controller's job is to keep the system close to these. The cost is typically quadratic:

$$
\begin{aligned}
    c(\mathbf{x}_k, \mathbf{u}_k) &= \| \mathbf{x}_k - \mathbf{x}_k^{\text{ref}} \|_{\mathbf{Q}}^2 + \| \mathbf{u}_k - \mathbf{u}_k^{\text{ref}} \|_{\mathbf{R}}^2 \\
    c(\mathbf{x}_N) &= \| \mathbf{x}_N - \mathbf{x}_N^{\text{ref}} \|_{\mathbf{P}}^2 \enspace .
\end{aligned}
$$

When dynamics are linear and constraints are polyhedral, this yields a convex quadratic program at each time step.

## Regulatory MPC

What changes when the target is a fixed equilibrium rather than a moving
reference?

In regulation tasks, we aim to bring the system back to an equilibrium point $(\mathbf{x}^e, \mathbf{u}^e)$, typically in the presence of disturbances. This is simply tracking MPC with constant references:

$$
\begin{aligned}
    c(\mathbf{x}_k, \mathbf{u}_k) &= \| \mathbf{x}_k - \mathbf{x}^e \|_{\mathbf{Q}}^2 + \| \mathbf{u}_k - \mathbf{u}^e \|_{\mathbf{R}}^2 \\
    c(\mathbf{x}_N) &= \| \mathbf{x}_N - \mathbf{x}^e \|_{\mathbf{P}}^2 \enspace .
\end{aligned}
$$

To guarantee stability, it is common to include a terminal constraint $\mathbf{x}_N \in \mathcal{X}_f$, where $\mathcal{X}_f$ is a control-invariant set under a known feedback law.


## Economic MPC

Can the controller optimize production or revenue directly when the best
operation need not remain near a prescribed setpoint?

Tracking and regulatory MPC penalize deviation from a prescribed reference. An economic controller instead optimizes the quantity produced by operating the system. The optimal state may vary continuously because there is no fixed reference trajectory.

### Wave-Energy Capture

A hinged flap extracts energy from oscillating water through a power-take-off (PTO) device. A single rotational mode gives the model

$$
I\ddot q+b\dot q+kq=\tau_{\mathrm{wave}}(t)-\rho(t)\dot q,
$$

where $q$ is flap angle and $\rho$ is the commanded PTO damping. The PTO torque and captured power are

$$
\tau_{\mathrm{PTO}}=-\rho\dot q,
\qquad
P_{\mathrm{capture}}=-\tau_{\mathrm{PTO}}\dot q=\rho\dot q^2.
$$

Restricting $\rho\geq0$ keeps the PTO passive because $\tau_{\mathrm{PTO}}\dot q=-\rho\dot q^2\leq0$. The actuator can remove mechanical energy from the flap, but it cannot inject energy into it.

A frozen-state calculation suggests using as much damping as possible. At $\dot q=0.4$ rad/s, damping values of $900$ and $1900$ N m s/rad yield instantaneous capture rates of $144$ and $304$ W. The calculation holds velocity fixed. In the dynamical system, stronger damping reduces future velocity, so captured energy need not increase monotonically with $\rho$. The constant-damping sweep below measures that delayed effect.

The experiment uses a deterministic sum of three wave-torque components for $45$ s. The controller updates every $h=0.12$ s and predicts $18$ steps, or $2.16$ s, into the future. Its finite-horizon problem is

$$
\begin{aligned}
\max_{\rho_{0:H-1}}\quad &
h\sum_{j=0}^{H-1}\rho_j\omega_j^2
-\lambda\sum_{j=0}^{H-1}
\left(\frac{\rho_j-\rho_{j-1}}{\rho_{\max}}\right)^2 \\
\text{subject to}\quad &
\mathbf{x}_{j+1}=F_h(\mathbf{x}_j,\rho_j,\tau_{\mathrm{wave}}), \\
&0\leq\rho_j\leq2400\ \mathrm{N\,m\,s/rad},\\
&|q_j|\leq0.55\ \mathrm{rad},\\
&|\rho_j\omega_j|\leq2800\ \mathrm{N\,m},\\
&|\rho_j-\rho_{j-1}|/h\leq5000\ \mathrm{N\,m\,s^{-2}/rad}.
\end{aligned}
$$

Here $F_h$ is one Runge-Kutta step of the flap model and $\rho_{-1}$ is the damping applied during the previous control interval. Only $\rho_0$ is applied. The flap state is measured again, the horizon advances, and the nonlinear program is solved from the shifted previous solution.

Two passive controllers provide matched baselines. Constant damping requests $\rho=900$ N m s/rad. A phase-aware heuristic raises damping when the measured product $\tau_{\mathrm{wave}}\dot q$ is positive and lowers it otherwise. All three requests pass through the same damping, torque, and damping-rate projection before reaching the plant. The stroke constraint differs because it depends on predicted state evolution rather than on the current actuator command.

Before inspecting the trajectories, predict whether constant damping should maximize total captured energy. Also predict which controller should use the most actuator variation. The economic objective can favor operation close to the state and torque limits.

```{code-cell} python
:tags: [remove-cell]

import sys
sys.path.insert(0, "code")

import pandas as pd
from IPython.display import HTML, display
import matplotlib.pyplot as plt

from wave_energy import (
    WaveParameters,
    create_animation as create_wave_animation,
    make_closed_loop_figure,
    make_tradeoff_figure,
    metrics_table as wave_metrics_table,
    run_comparison as run_wave_comparison,
    run_damping_sweep,
)

wave_parameters = WaveParameters()
wave_results = run_wave_comparison(wave_parameters, duration=45.0)
wave_sweep = run_damping_sweep(wave_parameters, duration=45.0)
```

```{code-cell} python
:tags: [remove-input]
:label: fig-wave-economic-mpc-trajectories
:caption: Closed-loop response under the same deterministic three-frequency forcing. The economic MPC captures more energy by varying passive damping and operating close to the stroke and PTO-torque limits. Constant damping crosses the shaded stroke-feasible region, while the phase-aware baseline stays farther inside it. Every curve is generated by the nonlinear plant simulation.

wave_summary_figure = make_closed_loop_figure(wave_results, wave_parameters)
display(wave_summary_figure)
plt.close(wave_summary_figure)
```

The economic MPC captures approximately $47.3$ kJ, compared with $39.9$ kJ for phase-aware damping and $39.3$ kJ for constant damping. It reaches the $0.55$ rad stroke limit and the $2.8$ kN m torque limit without exceeding either beyond numerical tolerance. Constant damping exceeds the stroke limit by about $0.084$ rad. The phase-aware law remains feasible and uses less damping variation, but it leaves energy unharvested.

One of the 375 nonlinear programs terminates without satisfying the solver's convergence flag. The implementation then applies the shifted feasible sequence through the common actuator projection. The resulting trajectory remains within the recorded limits, but the fallback is part of the result rather than evidence of recursive feasibility.

```{code-cell} python
:tags: [remove-input]

wave_table = pd.DataFrame(wave_metrics_table(wave_results))
for column in (
    "energy_kj",
    "peak_stroke_deg",
    "peak_torque_knm",
    "damping_variation_per_s",
    "stroke_violation_deg",
    "torque_violation_nm",
):
    wave_table[column] = wave_table[column].map(lambda x: f"{x:.3f}")
wave_table.rename(
    columns={
        "controller": "controller",
        "energy_kj": "captured energy (kJ)",
        "peak_stroke_deg": "peak stroke (deg)",
        "peak_torque_knm": "peak PTO torque (kN m)",
        "damping_variation_per_s": "damping variation per second",
        "stroke_violation_deg": "stroke violation (deg)",
        "torque_violation_nm": "torque violation (N m)",
    }
)
```

The constant-damping sweep places the three controllers on a common energy-motion diagram. A larger constant damping initially improves capture, then suppresses the motion that produces power. The economic MPC lies above this static tradeoff because it changes damping with the predicted phase while respecting the stroke bound.

```{code-cell} python
:tags: [remove-input]
:label: fig-wave-economic-mpc-tradeoff
:caption: Captured energy against peak flap stroke. Gray circles sweep constant passive damping from 100 to 2800 N m s/rad. The vertical dotted line is the 0.55 rad stroke constraint. Economic MPC obtains more energy than the fixed-damping sweep by scheduling damping over the predicted wave cycle. The phase-aware baseline is feasible but does not attain the MPC energy.

wave_tradeoff_figure = make_tradeoff_figure(
    wave_results,
    wave_sweep,
    wave_parameters,
)
display(wave_tradeoff_figure)
plt.close(wave_tradeoff_figure)
```

The animation shows the economic-MPC trajectory and accumulated energy from the same executed simulation. The displayed damping is the action applied after the actuator projection.

```{code-cell} python
:tags: [remove-input]
:label: fig-wave-economic-mpc-animation
:caption: Time-compressed replay of the economic-MPC flap motion and captured energy. The controller re-solves an 18-step nonlinear program every 0.12 seconds and applies the first passive damping action.

wave_animation = create_wave_animation(wave_results, wave_parameters, frame_stride=3)
wave_html = wave_animation.to_jshtml(fps=25)
plt.close(wave_animation._fig)
display(HTML(wave_html))
```

:::{dropdown} Inspect the economic-MPC solve

```{literalinclude} code/wave_energy.py
:language: python
:start-at: def solve_economic_mpc
:end-before: def _metrics
:linenos:
```

:::

{download}`Download the complete wave-energy experiment <code/wave_energy.py>`

This model assumes an accurate short-term wave forecast and one rigid flap mode. Real devices include radiation-memory effects, additional structural modes, losses, measurement error, and forecast uncertainty. Those omissions affect both predicted capture and constraint margins. Robust or stochastic MPC can represent some of this uncertainty, while data can be used to estimate the unmodeled residual dynamics.

## Robust MPC

How can feasibility and performance be protected against every disturbance in
a bounded uncertainty set?

Some systems are exposed to external disturbances or small errors in the model. In those cases, we want the controller to make decisions that will still work no matter what happens, as long as the disturbances stay within some known bounds. This is the idea behind **robust MPC**.

Instead of planning a single trajectory, the controller plans a "nominal" path (what would happen in the absence of any disturbance) and then adds a feedback correction to react to whatever disturbances actually occur. This looks like:

$$
\mathbf{u}_k = \bar{\mathbf{u}}_k + \mathbf{K} (\mathbf{x}_k - \bar{\mathbf{x}}_k) \enspace ,
$$

where $\bar{\mathbf{u}}_k$ is the planned input and $\mathbf{K}$ is a feedback gain that pulls the system back toward the nominal path if it deviates.

Because we know the worst-case size of the disturbance, we can estimate how far the real state might drift from the plan, and "shrink" the constraints accordingly. The result is that the nominal plan is kept safely away from constraint boundaries, so even if the system gets pushed around, it stays inside limits. This is often called **tube MPC** because the true trajectory stays inside a tube around the nominal one.

The main benefit is that we can handle uncertainty without solving a complicated worst-case optimization at every time step. All the uncertainty is accounted for in the design of the feedback $\mathbf{K}$ and the tightened constraints.


## Stochastic MPC

What becomes possible when uncertainty is represented by probabilities rather
than only worst-case bounds?

If disturbances are random rather than adversarial, a natural goal is to optimize expected cost while enforcing constraints probabilistically. This gives rise to **stochastic MPC**, in which:

* The cost becomes an expectation:

  $$
  \mathbb{E} \left[ c(\mathbf{x}_N) + \sum_{k=0}^{N-1} w_k\, c(\mathbf{x}_k, \mathbf{u}_k) \right]
  $$
* Constraints are allowed to be violated with small probability:

  $$
  \mathbb{P}[\mathbf{g}(\mathbf{x}_k, \mathbf{u}_k) \leq \mathbf{0}] \geq 1 - \varepsilon
  $$

In practice, expectations are approximated using a finite set of disturbance scenarios drawn ahead of time. For each scenario, the system dynamics are simulated forward using the same control inputs $\mathbf{u}_k$, which are shared across all scenarios to respect non-anticipativity. The result is a single deterministic optimization problem with multiple parallel copies of the dynamics, one per sampled future. This retains the standard MPC structure, with only moderate growth in problem size.

Despite appearances, this is not dynamic programming. There is no value function or tree of all possible paths. There is only a finite set of futures chosen a priori, and optimized over directly. This scenario-based approach is common in energy systems such as hydro scheduling, where inflows are uncertain but sample trajectories can be generated from forecasts.

Risk constraints are typically enforced across all scenarios or encoded using risk measures like CVaR. For example, one might penalize violations that occur in the worst $(1 - \alpha)\%$ of samples, while still optimizing expected performance overall.

## Hybrid and Mixed-Integer MPC

How can mode switches, on-off devices, and logical decisions enter the same
finite-horizon optimization?

When systems involve discrete switches  (eg. on/off valves, mode selection, or combinatorial logic) the MPC problem must include integer or binary variables. These show up in constraints like

$$
\boldsymbol{\delta}_k \in \{0,1\}^m, \qquad \mathbf{u}_k \in \mathcal{U}(\boldsymbol{\delta}_k)
$$

along with mode-dependent dynamics and costs. The resulting formulation is a **mixed-integer nonlinear program** (MINLP). The receding-horizon idea is the same, but each solve is more expensive due to the combinatorial nature of the decision space.

## Distributed and Decentralized MPC

When one optimization is too large or ownership is distributed, which
information must local controllers exchange to coordinate their plans?

Large-scale systems often consist of interacting subsystems. Distributed MPC decomposes the global NLP into smaller ones that run in parallel, with coordination constraints enforcing consistency across shared variables:

$$
\sum_{i} \mathbf{H}^i \mathbf{z}^i_k = \mathbf{0} \qquad \text{(coupling constraint)}
$$

Each subsystem solves a local problem over its own state and input variables, then exchanges information with neighbors. Coordination can be done via primal–dual methods, ADMM, or consensus schemes, but each local block looks like a standard MPC problem.


## Adaptive and Learning-Based MPC

How can the prediction model improve from data without discarding the explicit
constraints enforced by MPC?

In practice, we may not know the true model $\mathbf{F}_k$ or cost function $c$ precisely. In **adaptive MPC**, these are updated online from data:

$$
\mathbf{x}_{k+1} = \mathbf{F}_k(\mathbf{x}_k, \mathbf{u}_k; \boldsymbol{\theta}_t), \qquad
c(\mathbf{x}_k, \mathbf{u}_k) = c(\mathbf{x}_k, \mathbf{u}_k; \boldsymbol{\phi}_t)
$$

The parameters $\boldsymbol{\theta}_t$ and $\boldsymbol{\phi}_t$ are learned in real time. When combined with policy distillation, value approximation, or trajectory imitation, this leads to overlaps with reinforcement learning where the MPC solutions act as supervision for a reactive policy.


## Robustness and Failure Handling

What should the closed-loop system do when the preferred MPC problem is
infeasible, inaccurate, or unfinished at the control deadline?

The wave-energy example assumes that every optimization finishes before the next control update and that its prediction model remains accurate over the horizon. An operational MPC controller must also handle incompatible constraints, modeling errors, and missed computation deadlines.

A disturbance can move the measured state to a point from which the requested target is unreachable within the horizon. Model mismatch can make a trajectory feasible in prediction and infeasible on the plant. A solver can also stop because of an ill-conditioned local model or a changing active set. These cases require an explicit hierarchy of hard constraints, soft objectives, and fallback actions. The following mechanisms modify the finite-horizon problem or its deployment without changing the receding-horizon principle.
## Softening Constraints Through Slack Variables

Which constraints may be violated at a quantified price so that the optimizer
can still return a usable action?

The first approach to handling infeasibility recognizes that not all constraints carry equal importance. A chemical reactor's temperature must never exceed the runaway threshold: this is a hard constraint that cannot be violated. However, maintaining temperature within an optimal efficiency band is merely desirable. This can be treated as a soft constraint that we prefer to satisfy but can relax when necessary.

This hierarchy motivates reformulating the optimization problem using **slack variables**:

$$
\begin{aligned}
\min_{\mathbf{u}, \boldsymbol{\epsilon}} \quad & \sum_{i=0}^{N-1} \|\mathbf{x}_i - \mathbf{x}_i^{\text{ref}}\|_{\mathbf{Q}}^2 + \|\mathbf{u}_i\|_{\mathbf{R}}^2 + \boldsymbol{\rho}^T \boldsymbol{\epsilon}_i \\
\text{s.t.} \quad & \mathbf{x}_{i+1} = \mathbf{f}(\mathbf{x}_i, \mathbf{u}_i) \\
& \mathbf{g}_{\text{hard}}(\mathbf{x}_i, \mathbf{u}_i) \leq \mathbf{0} \\
& \mathbf{g}_{\text{soft}}(\mathbf{x}_i, \mathbf{u}_i) \leq \boldsymbol{\epsilon}_i \\
& \boldsymbol{\epsilon}_i \geq \mathbf{0}
\end{aligned}
$$

The penalty weights $\boldsymbol{\rho}$ encode our priorities. Safety constraints might use $\rho_j = 10^6$, while comfort constraints use $\rho_j = 1$. This reformulated problem is always feasible as long as the hard constraints alone admit a solution. That is: we can always make the slack variables $\boldsymbol{\epsilon}$ sufficiently large to satisfy the soft constraints.

Rather than treating constraints as binary hard/soft categories, we can establish a **constraint hierarchy** that enables graceful degradation:

$$
\begin{aligned}
\text{Safety:} \quad & T_{\text{reactor}} \leq T_{\text{runaway}} - 10 \quad & \rho = \infty \text{ (hard)} \\
\text{Equipment:} \quad & 0 \leq u_{\text{valve}} \leq 100 \quad & \rho = 10^4 \\
\text{Efficiency:} \quad & T_{\text{optimal}} - 5 \leq T \leq T_{\text{optimal}} + 5 \quad & \rho = 10^2 \\
\text{Comfort:} \quad & |T - T_{\text{setpoint}}| \leq 1 \quad & \rho = 1
\end{aligned}
$$

As conditions deteriorate, the controller abandons objectives in reverse priority order, maintaining safety even when optimality becomes impossible.

## Feasibility Restoration

If the main problem fails, can a secondary optimization recover a state from
which the nominal constraints become satisfiable again?

When even soft constraints prove insufficient (perhaps due to catastrophic solver failure or corrupted problem structure) we need **feasibility restoration** that finds any feasible point regardless of optimality:

$$
\begin{aligned}
\min_{\mathbf{u}, \mathbf{s}} \quad & \|\mathbf{s}\|_1 \\
\text{s.t.} \quad & \mathbf{x}_{i+1} = \mathbf{f}(\mathbf{x}_i, \mathbf{u}_i) + \mathbf{s}_i \\
& \mathbf{x}_{\min} - \mathbf{s}_{x,i} \leq \mathbf{x}_i \leq \mathbf{x}_{\max} + \mathbf{s}_{x,i} \\
& \mathbf{u}_{\min} \leq \mathbf{u}_i \leq \mathbf{u}_{\max} \\
& \mathbf{s} \geq \mathbf{0}
\end{aligned}
$$

This formulation temporarily relaxes even the dynamics constraints, finding the "least infeasible" solution. It answers the question: if we must violate something, what is the minimal violation required? Once feasibility is restored, we can warm-start the original problem from this point.

## Reference Governors

Can an unsafe reference be modified before it reaches an otherwise reliable
inner controller?

Rather than reacting to infeasibility after it occurs, we can prevent it by filtering references through a **reference governor**. Consider an aircraft following waypoints. Instead of passing waypoints directly to the MPC, the governor asks: what is the closest approachable reference from our current state?

$$
\mathbf{r}_{\text{filtered}} = \arg\max_{\kappa \in [0,1]} \kappa \quad \text{s.t. MPC}(\mathbf{x}_{\text{current}}, \kappa \mathbf{r}_{\text{desired}} + (1-\kappa)\mathbf{x}_{\text{current}}) \text{ is feasible}
$$

The governor performs a line search between the current state (always feasible since staying put requires no action) and the desired reference (potentially infeasible). This guarantees the MPC always receives feasible problems while making maximum progress toward the goal.

For computational efficiency, we can pre-compute the **maximal output admissible set**:

$$
\mathcal{O}_\infty = \{\mathbf{r} : \exists \text{ feasible trajectory from } \mathbf{x} \text{ to } \mathbf{r} \text{ respecting all constraints}\}
$$

Online, the governor simply projects the desired reference onto $\mathcal{O}_\infty$.

## Backup Controllers

Which independently validated action should take over when the optimizer or
its model cannot be trusted?

When MPC fails entirely (due to solver crashes, timeouts, or numerical failures) we need backup controllers that require minimal computation while guaranteeing stability and keeping the system away from dangerous regions.

The standard approach uses a pre-computed **local LQR controller** around the equilibrium:

$$
\mathbf{K}_{\text{LQR}}, \mathbf{P} = \text{LQR}(\mathbf{A}, \mathbf{B}, \mathbf{Q}, \mathbf{R})
$$

where $(\mathbf{A}, \mathbf{B})$ are the linearized dynamics at equilibrium. When MPC fails:

$$
\mathbf{u}_{\text{backup}} = \begin{cases}
\mathbf{K}_{\text{LQR}}(\mathbf{x} - \mathbf{x}_{\text{eq}}) & \text{if } \mathbf{x} \in \mathcal{X}_{\text{LQR}} \\
\mathbf{u}_{\text{safe}} & \text{otherwise}
\end{cases}
$$

The region $\mathcal{X}_{\text{LQR}} = \{\mathbf{x} : (\mathbf{x} - \mathbf{x}_{\text{eq}})^T \mathbf{P} (\mathbf{x} - \mathbf{x}_{\text{eq}}) \leq \alpha\}$ represents the largest invariant set where LQR is guaranteed to work.

## Cascade Architectures

How can slower optimization and faster local feedback be layered without
giving both loops conflicting authority?

Production MPC systems rarely rely on a single solver. Instead, they implement a **cascade of increasingly conservative controllers** that trade optimality for reliability:

```python
def get_control(self, x, time_budget):
    """
    Multi-level cascade for robust real-time control
    """
    time_remaining = time_budget
    
    # Level 1: Full nonlinear MPC
    if time_remaining > 5e-3:  # 5ms minimum
        try:
            u, solve_time = self.solve_nmpc(x, time_remaining)
            if converged:
                return u
        except:
            pass
        time_remaining -= solve_time
    
    # Level 2: Simplified linear MPC
    if time_remaining > 1e-3:  # 1ms minimum
        try:
            # Linearize around current state
            A, B = self.linearize_dynamics(x)
            u, solve_time = self.solve_lmpc(x, A, B, time_remaining)
            return u
        except:
            pass
        time_remaining -= solve_time
    
    # Level 3: Explicit MPC lookup
    if time_remaining > 1e-4:  # 0.1ms minimum
        region = self.find_critical_region(x)
        if region is not None:
            return self.explicit_control_law[region](x)
    
    # Level 4: LQR backup
    if self.in_lqr_region(x):
        return self.K_lqr @ (x - self.x_eq)
    
    # Level 5: Emergency safe mode
    return self.emergency_stop(x)
```

Each level trades optimality for reliability: Level 1 provides optimal but computationally expensive control, Level 2 offers suboptimal but faster solutions, Level 3 provides pre-computed instant evaluation, Level 4 ensures stabilizing control without tracking, and Level 5 implements safe shutdown.

Even when using backup controllers, we can maintain solution continuity through **persistent warm-starting**:

$$
\begin{aligned}
\mathbf{z}_{\text{warm}}^{(k+1)} = \begin{cases}
\text{shift}(\mathbf{z}^{(k)}) & \text{if MPC succeeded at time } k \\
\text{lift}(\mathbf{u}_{\text{backup}}^{(k)}) & \text{if backup controller used} \\
\text{propagate}(\mathbf{z}_{\text{warm}}^{(k)}) & \text{if maintaining virtual solution}
\end{cases}
\end{aligned}
$$

The **shift** operation takes a successful MPC solution and moves it forward by one time step, appending a terminal action: $[\mathbf{u}_1^{(k)}, \mathbf{u}_2^{(k)}, \ldots, \mathbf{u}_{N-1}^{(k)}, \kappa_f(\mathbf{x}_N^{(k)})]$. This shifted sequence provides natural temporal continuity for the next optimization.

When MPC fails and backup control is applied, the **lift** operation extends the single backup action $\mathbf{u}_{\text{backup}}^{(k)}$ into a full horizon-length sequence, either by repetition or by simulating the backup controller forward. This creates a reasonable warm-start guess from limited information.

The **propagate** operation maintains a "virtual" trajectory by continuing to evolve the previous solution as if it were still being executed, even when the actual system follows backup control. This forward simulation keeps the warm-start temporally aligned and relevant for when MPC recovers.

## Inference Serving Under a Latency and Power Budget

How do deadlines, thermal limits, forecast error, and solver time interact in a
receding-horizon controller for a real computing workload?

The offline frequency schedule in [Trajectory Optimization](discrete-time-optimal-control.md#example-offline-frequency-planning-for-inference)
cannot react when a burst arrives earlier than forecast. Receding-horizon
control uses the same aggregate model but replaces the initial forecast after
each second. Measured NVIDIA L4 service-rate and phase-power curves calibrate
the model, while the request and queue trajectories remain simulation outputs.
The scheduling rule remains fixed at the reduced 512-token interleaved
chunked-prefill model from the modeling chapter, so the comparison isolates
feedback through the GPU clock.

At control time $t$, the state

$$
x_t=(p_t,d_t,T_t,f_{t-1})
$$

contains queued prefill work, unfinished decode work, temperature, and the last
applied clock. The controller forecasts ten one-second transitions. Arrivals
are predicted from the trailing 30-second rate, while expected output work uses
the empirical output-length distribution from the trace. The request-level
plant receives the realized requests and output lengths instead.

The finite-horizon problem uses the normalized objective

$$
J_t=\sum_{k=t}^{t+9}\left[
\frac{E_k}{E_{\max}}
+20\delta_{k,\mathrm{TTFT}}^2
+10\delta_{k,\mathrm{TPOT}}^2
+0.05\Delta f_k^2
+1000(s_{P,k}^2+s_{T,k}^2)
\right]+20B_{t+10}^2.
$$

The variables $\delta_{k,\mathrm{TTFT}}$ and
$\delta_{k,\mathrm{TPOT}}$ are normalized overruns of two aggregate delay
estimates. The first divides predicted queued prefill tokens by the profiled
prefill rate. The second divides the predicted number of active decode requests
by the profiled decode rate. These estimates enter the optimization but do not
equal realized request-level TTFT and TPOT; the detailed simulation computes
the latter. The frequency difference $\Delta f_k$ is normalized by the profiled
clock range.
The slacks $s_{P,k}$ and $s_{T,k}$ penalize violations of the experimental
power and thermal limits, and $B_{t+10}$ penalizes unfinished work at the end of
the horizon. SLSQP receives at most 50 iterations and a tolerance of $10^{-6}$.
Only the first frequency is applied. The continuous result is rounded downward
to the nearest profiled requested clock before the next state is observed. The
replay reports the corresponding measured median realized clock separately,
since the requested and realized values can differ under the experimental power
cap.

A solver result is rejected if it is infeasible, non-finite, or arrives after
the 0.8-second control budget. The fallback is a hysteretic reactive governor:
it raises the clock by one profile level when weighted queue pressure reaches
eight or the oldest prompt has waited one second, and lowers it by one level
when temperature reaches 72 degrees C or pressure falls to one. Every fallback
appears in the recorded trajectory rather than being removed from the
aggregate metrics.

The comparison asks whether replanning improves the response to the shifted
burst without hiding its energy cost. Four controllers receive identical
requests and initial conditions:

1. maximum clock throughout the experiment;
2. the reactive governor used as the MPC fallback;
3. the open-loop schedule optimized for the nominal trace;
4. receding-horizon MPC.

```{code-cell} python
:tags: [remove-input]
:label: fig-inference-mpc
:caption: Maximum-clock, reactive, open-loop, and MPC frequency control under the same shifted request trace and fixed reduced interleaved chunked-prefill model. Measured NVIDIA L4 curves calibrate each simulated trajectory. At each second, the dashed MPC segment is the current ten-second plan and the solid segment is the action history. Later plans are not revealed before they are computed.

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from inference_replay import render_serving_replay

display(HTML(render_serving_replay(
    Path("artifacts/inference_serving/textbook_results.json"),
    view="mpc",
)))
```

:::{figure} _static/inference_serving/mpc.svg
:label: fig-inference-mpc-fallback
:class: pdf-fallback
:alt: Static comparison of maximum-clock, reactive, open-loop, and receding-horizon frequency control.

Static view of the closed-loop comparison. The online book provides playback
and exposes the planned MPC horizon available at each control time.
:::

The table compares the four frequency controllers under identical shifted
arrivals and the fixed reduced interleaved chunked-prefill model. Fallback count
records every MPC step that exceeded its feasibility, finiteness, or timing
requirement.

```{code-cell} python
:tags: [remove-input]

import pandas as pd

mpc_metrics = pd.read_csv(
    "artifacts/inference_serving/metrics_mpc.csv"
).set_index("controller")
mpc_metrics[
    [
        "mean_ttft_s",
        "p95_ttft_s",
        "mean_tpot_s",
        "energy_j",
        "peak_queued_requests",
        "ttft_violation_rate",
        "tpot_violation_rate",
        "power_violation_w",
        "thermal_violation_c",
        "fallback_count",
    ]
].T.rename_axis("metric").round(3)
```

{download}`Download every closed-loop metric (CSV) <artifacts/inference_serving/metrics_mpc.csv>`

On the shifted trace, maximum clock gives the lowest mean time to first token,
10.93 seconds, at 3,506.0 joules. Its 95th percentile is 18.41 seconds. MPC
reduces energy to 3,341.8 joules, the lowest of the four controllers, while its
mean and 95th-percentile times to first token are 11.53 and 19.01 seconds. The
reactive governor records a mean of 12.51 seconds, a 95th percentile of 20.01
seconds, and 3,363.3 joules. The unchanged offline plan records 23.23 seconds,
a 95th percentile of 31.28 seconds, and 3,646.8 joules.

Maximum clock and MPC both have a mean time per output token of 0.0417 seconds.
The reactive governor records 0.0419 seconds, while the offline plan records
0.0466 seconds. The peak queue is 21 requests at maximum clock and 22 under
MPC. It grows to 23 under the reactive governor and 30 under the offline plan.
At the 60-second reporting horizon, maximum clock and MPC each leave nine
requests unfinished. The reactive governor leaves 11, and the offline plan
leaves 22. All 48 requests eventually complete during the post-horizon drain,
whose energy is included in the reported totals.

Every controller violates the experimental TTFT limit for every request, and
each has a 12.5% TPOT violation rate. MPC accepts 67 solves over the full
rollout and invokes no fallback. Every accepted solve meets the 0.8-second
deadline. On this trace, MPC improves both mean TTFT and energy relative to the
reactive governor and the offline plan. Maximum clock remains 0.60 seconds
faster in mean TTFT but consumes 164.2 joules more than MPC.

The displayed table focuses on latency, energy, queueing, constraint
violations, and fallback count. The downloadable CSV also reports end-to-end
latency, throughput, energy per output token, peak power and temperature, and
unfinished work. The comparison therefore separates a faster response from a
lower-energy response instead of assigning one score to each controller.

The measured profile supplies the service-rate and phase-power calibration;
the controller trajectories are simulations of a single-GPU serving model.
All four simulations reach a modeled phase power of 64.852 W at the highest
requested clock. This value comes from the measured decode-phase mean for that
profile level and is 0.052 W above the configured 64.800 W cap. The cap was an
experimental setting rather than a hard sample-wise guarantee. None of the
four simulations records a thermal or KV-capacity violation.

### Validation of the Thermal Constraint

The controller above constrains a junction temperature predicted by a
one-state thermal model. None of the simulated controllers violates that
constraint, but this establishes feasibility only for the model used inside
the optimizer.

A power-matched pair of workloads exposes the modeling problem. During the
held-out experiment, a 55 W decode pulse and a 55 W prefill pulse consumed
nearly the same electrical energy, with a difference of 0.2 percent. The
measured peak temperature rise during prefill was nevertheless 2.00 degrees C
larger than during decode. A model driven only by board power receives nearly
the same input for both pulses and has no variable with which to represent
this difference.

The clock sweep used to calibrate the serving model was designed to measure
service rate and power. Its one-state thermal fit had $R^2=0.0020$, so a
separate experiment tested whether workload phase was needed in the thermal
input. Both candidate models used

$$
\dot T(t)
=
-\frac{T(t)-T_{\mathrm{amb}}}{\tau}
+\frac{R}{\tau}P_{\mathrm{eff}}(t).
$$

Here $T_{\mathrm{amb}}$ is an effective ambient temperature, $\tau$ is the
thermal time constant, and $R$ is the steady-state thermal resistance. The two
input models differ by one term:

$$
P_{\mathrm{eff}}(t)
=
\begin{cases}
P(t), & \text{power-only model},\\
P(t)\bigl(1+\beta I_{\mathrm{prefill}}(t)\bigr),
& \text{phase-gain model}.
\end{cases}
$$

The indicator $I_{\mathrm{prefill}}$ equals one during prefill and zero during
decode. The phase-gain model can therefore assign different effective thermal
inputs to two workloads with the same measured board power. Its state
dimension and measured input remain unchanged.

The training and validation pulses were independent cold starts:

| Data split | Workloads and duration | Use |
|---|---|---|
| Training | Decode and prefill at 46 W for 75 s and at 61 W for 45 s | Fit both candidate models |
| Validation | Decode and prefill at 55 W for 60 s | Evaluate the fixed models once |

The order of the four training pulses was counterbalanced. Every pulse began
from a verified post-relock temperature within a one-degree-Celsius band. The
acquisition retained a 77 degree C safe-down threshold and a 79 degree C abort
threshold.

:::{figure} _static/inference_serving/thermal-phase-validation.svg
:label: fig-inference-thermal-validation
:alt: Measured decode and prefill temperature rises on an NVIDIA L4 compared with fixed power-only and phase-gain one-state thermal models, followed by three validation errors against a one degree C acceptance boundary.

The power-only model predicts similar responses for the power-matched pulses.
The phase-gain model separates them and lowers held-out RMSE, but its largest
trajectory and phase-contrast errors remain above the fixed one-degree-Celsius
acceptance boundary. The validation data were not used to fit either model.
:::

:::{include} artifacts/inference_serving/thermal_phase_result.md
:::

The fitted gain gives prefill 1.147 times the effective thermal input of decode.
This parameter is reproducible across the ten optimization starts. The training
Jacobian also has full numerical rank, with condition number 384. These checks
reduce concern about an ambiguous parameter fit, but they do not establish that
one thermal state is sufficient. The validation residuals miss the rapid
initial temperature rise and later reverse sign, which is evidence of missing
dynamics in this experiment.

A second thermal state could represent package or heat-sink temperature through
the cooldown, memory-relock, and workload segments. Repeating the same six
pulses would estimate run-to-run variability but would not add this missing
state. Any enlarged model would require another held-out validation test before
its predictions could support a hardware constraint.

For the present MPC comparison, the temperature state and inequality remain a
teaching model. The absence of a simulated thermal violation is not a hardware
safety certificate. The arrival forecast is deliberately simple, output length
is observed only at completion, and the scheduler is held fixed. Network delay,
tokenization, multi-GPU communication, and model-quality effects are omitted.
The shifted burst is one controlled disturbance, so relative performance on
that trace does not establish dominance under other workloads.

{download}`Download the one-second held-out trajectories and fixed-model predictions (CSV) <artifacts/inference_serving/thermal_phase_validation.csv>`

{download}`Download the complete thermal fit and acceptance report (JSON) <data/inference_serving/thermal-phase-identification-20260903T131518Z/thermal_phase_fit_report.json>`

{download}`Download the measured validation telemetry (CSV) <data/inference_serving/thermal-phase-identification-20260903T131518Z/l4_thermal_phase_telemetry.csv>`

:::{dropdown} Inspect the receding-horizon controller
```{literalinclude} code/inference_control.py
:language: python
:start-at: def _solve(
:end-before: def __call__(
:linenos:
```

```{literalinclude} code/inference_control.py
:language: python
:start-at: def run_mpc
:end-before: def shift_largest_burst
:linenos:
```

{download}`Download the complete inference-control implementation <code/inference_control.py>`
:::

## Summary and Outlook

Tracking, regulation, economic control, robustness, stochastic predictions,
and hybrid decisions all retain the same receding-horizon backbone while
changing the objective, uncertainty model, or feasible set. Slack variables,
reference governors, backup controllers, and cascades determine how the
controller behaves when the preferred optimization problem is late or
infeasible.

Reliable execution still requires solving a closely related optimization
problem at every step. Can repeated structure be reused, or even approximated
by a learned map from parameters to actions? [Parametric and approximate
controllers](parametric-controllers.md) develop those alternatives.

## Self-checks

:::{exercise} Solver deadline
:label: ex-mpc-check-4

Why does the inference MPC reject a feasible solution returned after its
0.8-second deadline?
:::

:::{solution} ex-mpc-check-4
:class: dropdown

The decision is computed for the state at the start of a one-second interval.
After the deadline, little time remains to apply it and the request queue may
already have changed. The reactive fallback supplies a timely action with known
bounds.
:::

:::{exercise} Thermal-model validation
:label: ex-mpc-check-5

The phase-gain thermal model fits its four training pulses reproducibly and
predicts the held-out trajectories better than the power-only model. It still
fails one of three prespecified validation criteria. May its predicted
temperature be used as a hard hardware-safety constraint? What remains useful
about the experiment?
:::

:::{solution} ex-mpc-check-5
:class: dropdown

No. A reproducible parameter estimate does not establish that the model class
captures the relevant plant dynamics. The failed held-out criterion prevents
the constraint from serving as a safety certificate. The experiment remains
useful because it rejects the power-only assumption, quantifies the improvement
from a phase-dependent input, and points to missing thermal state as the next
modeling hypothesis.
:::
