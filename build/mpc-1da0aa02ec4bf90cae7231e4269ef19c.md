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
# Model Predictive Control

The trajectory optimization methods presented so far compute a complete control trajectory from an initial state to a final time or state. Once computed, this trajectory is executed without modification, making these methods fundamentally open-loop. The control function, $\mathbf{u}[k]$ in discrete time or $\mathbf{u}(t)$ in continuous time, depends only on the clock, reading off precomputed values from memory or interpolating between them. This approach assumes perfect models and no disturbances. Under these idealized conditions, repeating the same control sequence from the same initial state would always produce identical results.

Real systems face modeling errors, external disturbances, and measurement noise that accumulate over time. A precomputed trajectory becomes increasingly irrelevant as these perturbations push the actual system state away from the predicted path. The solution is to incorporate feedback, making control decisions that respond to the current state rather than blindly following a predetermined schedule. While dynamic programming provides the theoretical framework for deriving feedback policies through value functions and Bellman equations, there exists a more direct approach that leverages the trajectory optimization methods already developed.

## Closing the Loop by Replanning

Model Predictive Control creates a feedback controller by repeatedly solving trajectory optimization problems. Rather than computing a single trajectory for the entire task duration, MPC solves a finite-horizon problem at each time step, starting from the current measured state. The controller then applies only the first control action from this solution before repeating the entire process. This strategy transforms any trajectory optimization method into a feedback controller.

### The Receding Horizon Principle

The defining characteristic of MPC is its receding horizon strategy. At each time step, the controller solves an optimization problem looking a fixed duration into the future, but this prediction window constantly moves forward in time. The horizon "recedes" because it always starts from the current time and extends forward by the same amount.

Consider the discrete-time optimal control problem in Bolza form:

$$
\begin{aligned}
\text{minimize} \quad & c_T(\mathbf{x}_N) + \sum_{k=0}^{N-1} c(\mathbf{x}_k, \mathbf{u}_k) \\
\text{subject to} \quad & \mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k) \\
& \mathbf{g}(\mathbf{x}_k, \mathbf{u}_k) \leq \mathbf{0} \\
& \mathbf{u}_{\text{min}} \leq \mathbf{u}_k \leq \mathbf{u}_{\text{max}} \\
\text{given} \quad & \mathbf{x}_0 = \mathbf{x}_{\text{current}}
\end{aligned}
$$

At time step $t$, this problem optimizes over the interval $[t, t+N]$. At the next time step $t+1$, the horizon shifts to $[t+1, t+N+1]$. What makes this work is that only the first control $\mathbf{u}_0^*$ from each optimization is applied. The remaining controls $\mathbf{u}_1^*, \ldots, \mathbf{u}_{N-1}^*$ are discarded, though they may initialize the next optimization through warm-starting.

This receding horizon principle enables feedback without computing an explicit policy. By constantly updating predictions based on current measurements, MPC naturally corrects for disturbances and model errors. The apparent waste of computing but not using most of the trajectory is actually the mechanism that provides robustness.

### Horizon Selection and Problem Formulation

The choice of prediction horizon depends on the control objective. We distinguish between three cases, each requiring different mathematical formulations.

#### Infinite-Horizon Regulation

For stabilization problems where the system must operate indefinitely around an equilibrium, the true objective is:

$$
J_\infty = \sum_{k=0}^{\infty} c(\mathbf{x}_k, \mathbf{u}_k)
$$

Since this cannot be solved directly, MPC approximates it with:

$$
\begin{aligned}
\text{minimize} \quad & V_f(\mathbf{x}_N) + \sum_{k=0}^{N-1} c(\mathbf{x}_k, \mathbf{u}_k) \\
\text{subject to} \quad & \mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k) \\
& \mathbf{x}_N \in \mathcal{X}_f \\
& \text{other constraints}
\end{aligned}
$$

The terminal cost $V_f(\mathbf{x}_N)$ approximates $\sum_{k=N}^{\infty} c(\mathbf{x}_k, \mathbf{u}_k)$, the cost-to-go beyond the horizon. The terminal constraint $\mathbf{x}_N \in \mathcal{X}_f$ ensures the state reaches a region where a known stabilizing controller exists. Without these terminal ingredients, the finite-horizon approximation may produce unstable behavior, as the controller ignores consequences beyond the horizon.

#### Finite-Duration Tasks

For tasks ending at time $t_f$, the true objective spans from current time $t$ to $t_f$:

$$
J_{[t, t_f]} = c_f(\mathbf{x}(t_f)) + \sum_{k=t}^{t_f-1} c(\mathbf{x}_k, \mathbf{u}_k)
$$

The MPC formulation must adapt as time progresses:

$$
\begin{aligned}
\text{minimize} \quad & c_{T,k}(\mathbf{x}_{N_k}) + \sum_{j=0}^{N_k-1} c(\mathbf{x}_j, \mathbf{u}_j) \\
\text{where} \quad & N_k = \min(N, t_f - t_k) \\
& c_{T,k} = \begin{cases}
c_f & \text{if } t_k + N_k = t_f \\
c_T & \text{otherwise}
\end{cases}
\end{aligned}
$$

As the task approaches completion, the horizon shrinks and the terminal cost switches from the approximation $c_T$ to the true final cost $c_f$. This prevents the controller from optimizing beyond task completion, which would produce meaningless or aggressive control actions.

#### Periodic Tasks

Some systems operate on repeating cycles where the optimal behavior depends on the time of day, week, or season. Consider a commercial building where heating costs are higher at night, electricity prices vary hourly, and occupancy patterns repeat daily. The MPC controller must account for these periodic patterns while planning over a finite horizon.

For tasks with period $T_p$, such as daily building operations, the formulation accounts for transitions across period boundaries:

$$
\begin{aligned}
\text{minimize} \quad & \sum_{k=0}^{N-1} c_k(\mathbf{x}_k, \mathbf{u}_k, \phi_k) \\
\text{where} \quad & \phi_k = (t + k) \mod T_p \\
& c_k(\cdot, \cdot, \phi) = \begin{cases}
c_{\text{day}}(\cdot, \cdot) & \text{if } \phi \in [6\text{am}, 6\text{pm}] \\
c_{\text{night}}(\cdot, \cdot) & \text{otherwise}
\end{cases}
\end{aligned}
$$

The cost function changes based on the phase $\phi$ within the period. Constraints may similarly depend on the phase, reflecting different operational requirements at different times.

### The MPC Algorithm

The complete MPC procedure implements the receding horizon principle through repeated optimization:

````{prf:algorithm} Model Predictive Control with Horizon Management
:label: alg-mpc-complete

**Input:**
- Nominal prediction horizon $N$
- Sampling period $\Delta t$
- Task type: {infinite, finite with duration $t_f$, periodic with period $T_p$}
- Cost functions and dynamics
- Constraints

**Procedure:**

1. Initialize time $t \leftarrow 0$
2. Measure initial state $\mathbf{x}_{\text{current}} \leftarrow \mathbf{x}(t)$

3. **While** task continues:

   4. **Determine effective horizon and costs:**
      - If infinite task: 
        - $N_{\text{eff}} \leftarrow N$
        - Use terminal cost $V_f$ and constraint $\mathcal{X}_f$
      - If finite task:
        - $N_{\text{eff}} \leftarrow \min(N, \lfloor(t_f - t)/\Delta t\rfloor)$
        - If $t + N_{\text{eff}}\Delta t = t_f$: use final cost $c_f$
        - Otherwise: use approximation $c_T$
      - If periodic task:
        - $N_{\text{eff}} \leftarrow N$
        - Adjust costs/constraints based on phase

   5. **Solve optimization:**
      Minimize over $\mathbf{u}_{0:N_{\text{eff}}-1}$ subject to dynamics, constraints, and $\mathbf{x}_0 = \mathbf{x}_{\text{current}}$

   6. **Apply receding horizon control:**
      - Extract $\mathbf{u}^*_0$ from solution
      - Apply to system for duration $\Delta t$
      - Measure new state
      - Advance time: $t \leftarrow t + \Delta t$

7. **End While**
````

### Successive Linearization and Quadratic Approximations

For many regulation and tracking problems, the nonlinear dynamics and costs we encounter can be approximated locally by linear and quadratic functions. The basic idea is to linearize the system around the current operating point and approximate the cost with a quadratic form. This reduces each MPC subproblem to a **quadratic program (QP)**, which can be solved reliably and very quickly using standard solvers.

Suppose the true dynamics are nonlinear,

$$
\mathbf{x}_{k+1} = f(\mathbf{x}_k,\mathbf{u}_k).
$$

Around a nominal trajectory $(\bar{\mathbf{x}}_k,\bar{\mathbf{u}}_k)$, we take a first-order expansion:

$$
\mathbf{x}_{k+1} \approx f(\bar{\mathbf{x}}_k,\bar{\mathbf{u}}_k) 
+ \mathbf{A}_k(\mathbf{x}_k - \bar{\mathbf{x}}_k) 
+ \mathbf{B}_k(\mathbf{u}_k - \bar{\mathbf{u}}_k),
$$

with Jacobians

$$
\mathbf{A}_k = \frac{\partial f}{\partial \mathbf{x}}(\bar{\mathbf{x}}_k,\bar{\mathbf{u}}_k), 
\qquad
\mathbf{B}_k = \frac{\partial f}{\partial \mathbf{u}}(\bar{\mathbf{x}}_k,\bar{\mathbf{u}}_k).
$$

Similarly, if the stage cost is nonlinear,

$$
c(\mathbf{x}_k,\mathbf{u}_k),
$$

we approximate it quadratically near the nominal point:

$$
c(\mathbf{x}_k,\mathbf{u}_k) \;\approx\; 
\|\mathbf{x}_k - \mathbf{x}_k^{\text{ref}}\|_{\mathbf{Q}_k}^2 
+ \|\mathbf{u}_k - \mathbf{u}_k^{\text{ref}}\|_{\mathbf{R}_k}^2,
$$

with positive semidefinite weighting matrices $\mathbf{Q}_k$ and $\mathbf{R}_k$.

The resulting MPC subproblem has the form

$$
\begin{aligned}
\min_{\mathbf{x}_{0:N},\mathbf{u}_{0:N-1}} \quad &
\|\mathbf{x}_N - \mathbf{x}_N^{\text{ref}}\|_{\mathbf{P}}^2
+ \sum_{k=0}^{N-1} 
\left(
\|\mathbf{x}_k - \mathbf{x}_k^{\text{ref}}\|_{\mathbf{Q}_k}^2
+ \|\mathbf{u}_k - \mathbf{u}_k^{\text{ref}}\|_{\mathbf{R}_k}^2
\right) \\
\text{s.t.} \quad &
\mathbf{x}_{k+1} = \mathbf{A}_k \mathbf{x}_k + \mathbf{B}_k \mathbf{u}_k + \mathbf{d}_k, \\
& \mathbf{u}_{\min} \leq \mathbf{u}_k \leq \mathbf{u}_{\max}, \\
& \mathbf{x}_{\min} \leq \mathbf{x}_k \leq \mathbf{x}_{\max}, \\
& \mathbf{x}_0 = \mathbf{x}_{\text{current}} ,
\end{aligned}
$$

where $\mathbf{d}_k = f(\bar{\mathbf{x}}_k,\bar{\mathbf{u}}_k) - \mathbf{A}_k \bar{\mathbf{x}}_k - \mathbf{B}_k \bar{\mathbf{u}}_k$ captures the local affine offset.

Because the dynamics are now linear and the cost quadratic, this optimization problem is a convex quadratic program. Quadratic programs are attractive in practice: they can be solved at kilohertz rates with mature numerical methods, making them the backbone of many real-time MPC implementations.

At each MPC step, the controller updates its linearization around the new operating point, constructs the local QP, and solves it. The process repeats, with the linear model and quadratic cost refreshed at every reoptimization. Despite the approximation, this yields a closed-loop controller that inherits the fast computation of QPs while retaining the ability to track trajectories of the underlying nonlinear system.

## Theoretical Guarantees

The finite-horizon approximation in MPC brings a new challenge: the controller cannot see consequences beyond the horizon. Without proper design, this myopia can destabilize even simple systems. The solution is to carefully encode information about the infinite-horizon problem into the finite-horizon optimization through its terminal conditions.

Before diving into the mathematics, we should first establish what "stability" means and which tasks these theoretical guarantees address, as the notion of stability varies significantly across different control objectives.

### Stability Notions Across Control Tasks

The terminal conditions provide different types of guarantees depending on the control objective. For regulation problems, where the task is to drive the state to a fixed equilibrium $(\mathbf{x}_\mathrm{eq}, \mathbf{u}_\mathrm{eq})$ (often shifted to the origin), the stability guarantee is **asymptotic stability**: starting sufficiently close to the equilibrium, we have $\mathbf{x}_k \to \mathbf{x}_\mathrm{eq}$ while constraints remain satisfied throughout the trajectory (**recursive feasibility**). This requires the stage cost $\ell(\mathbf{x},\mathbf{u})$ to be positive definite in the deviation from equilibrium.

When tracking a constant setpoint, the task becomes following a constant reference $(\mathbf{x}_\mathrm{ref},\mathbf{u}_\mathrm{ref})$ that solves the steady-state equations. This problem is handled by working in **error coordinates** $\tilde{\mathbf{x}}=\mathbf{x}-\mathbf{x}_\mathrm{ref}$ and $\tilde{\mathbf{u}}=\mathbf{u}-\mathbf{u}_\mathrm{ref}$, transforming the tracking problem into a regulation problem for the error system. The stability guarantee becomes asymptotic **tracking**, meaning $\tilde{\mathbf{x}}_k \to 0$, again with recursive feasibility.

The terminal conditions we discuss below primarily address regulation and constant reference tracking. Time-varying tracking and economic MPC require additional techniques such as tube MPC and dissipativity theory.

### MPC with Stability Guarantees

To provide theoretical guarantees, the finite-horizon MPC problem is augmented with three interconnected components. The **terminal cost** $V_f(\mathbf{x})$ approximates the cost-to-go beyond the horizon, providing a surrogate for the infinite-horizon tail that cannot be explicitly optimized. The **terminal constraint set** $\mathcal{X}_f$ defines a region where we have local knowledge of how to stabilize the system. Finally, the **terminal controller** $\kappa_f(\mathbf{x})$ provides a local stabilizing control law that remains valid within $\mathcal{X}_f$.

These components must satisfy specific compatibility conditions to provide theoretical guarantees:

````{prf:theorem} Recursive Feasibility and Asymptotic Stability
:label: thm-mpc-stability

Consider the MPC problem with terminal cost $V_f$, terminal set $\mathcal{X}_f$, and local controller $\kappa_f$. If the following conditions hold:

**Control invariance**: For all $\mathbf{x} \in \mathcal{X}_f$, we have $\mathbf{f}(\mathbf{x}, \kappa_f(\mathbf{x})) \in \mathcal{X}_f$ (the set is invariant) and $\mathbf{g}(\mathbf{x}, \kappa_f(\mathbf{x})) \leq \mathbf{0}$ (constraints remain satisfied).

**Lyapunov decrease**: For all $\mathbf{x} \in \mathcal{X}_f$:

   $$V_f(\mathbf{f}(\mathbf{x}, \kappa_f(\mathbf{x}))) - V_f(\mathbf{x}) \leq -\ell(\mathbf{x}, \kappa_f(\mathbf{x}))$$

   where $\ell$ is the stage cost.

Then the MPC controller achieves recursive feasibility (if the problem is feasible at time $k$, it remains feasible at time $k+1$), asymptotic stability to the target equilibrium for regulation problems, and monotonic cost decrease along trajectories until the target is reached.
````

### Suboptimality Bounds

The finite-horizon MPC value $V_N(\mathbf{x})$ provides an upper bound approximation of the true infinite-horizon value $V_\infty(\mathbf{x})$. Understanding how close this approximation can be tells us about the effectiveness of short-horizon MPC.


The upper bound $V_N(\mathbf{x}) \geq V_\infty(\mathbf{x})$ follows immediately from the fact that MPC considers fewer control choices. The infinite-horizon controller can choose any sequence $(\mathbf{u}_0, \mathbf{u}_1, \mathbf{u}_2, \ldots)$, while the $N$-horizon controller is restricted to sequences of the form $(\mathbf{u}_0, \ldots, \mathbf{u}_{N-1}, \kappa_f(\mathbf{x}_N), \kappa_f(\mathbf{x}_{N+1}), \ldots)$ where the tail follows the fixed terminal controller. Since the infinite-horizon problem optimizes over a larger feasible set, its optimal value cannot exceed that of the finite-horizon problem.

#### Deriving the Approximation Error

The interesting question is bounding the approximation error $\varepsilon_N = V_N(\mathbf{x}) - V_\infty(\mathbf{x})$. This error represents the cost of being forced to use $\kappa_f$ beyond the horizon rather than continuing to optimize.

Let $(\mathbf{u}_0^*, \mathbf{u}_1^*, \ldots)$ denote the infinite-horizon optimal control sequence with corresponding state trajectory $(\mathbf{x}_0^*, \mathbf{x}_1^*, \ldots)$ where $\mathbf{x}_0^* = \mathbf{x}$. The infinite-horizon cost is:

$$V_\infty(\mathbf{x}) = \sum_{k=0}^{\infty} \ell(\mathbf{x}_k^*, \mathbf{u}_k^*)$$

Now consider what happens when we truncate this optimal sequence at horizon $N$ and continue with the terminal controller. The cost becomes:

$$\tilde{V}_N(\mathbf{x}) = \sum_{k=0}^{N-1} \ell(\mathbf{x}_k^*, \mathbf{u}_k^*) + V_f(\mathbf{x}_N^*)$$

where $V_f(\mathbf{x}_N^*)$ approximates the tail cost $\sum_{k=N}^{\infty} \ell(\mathbf{x}_k^*, \mathbf{u}_k^*)$.

Since $V_N(\mathbf{x})$ is the optimal $N$-horizon cost (which may do better than this particular truncated sequence), we have $V_N(\mathbf{x}) \leq \tilde{V}_N(\mathbf{x})$. The approximation error therefore satisfies:

$$\varepsilon_N \leq \tilde{V}_N(\mathbf{x}) - V_\infty(\mathbf{x}) = V_f(\mathbf{x}_N^*) - \sum_{k=N}^{\infty} \ell(\mathbf{x}_k^*, \mathbf{u}_k^*)$$

This bound shows that the approximation error depends on how well the terminal cost $V_f$ approximates the true tail cost along the infinite-horizon optimal trajectory.


# The Landscape of MPC Variants

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

The most common setup is reference tracking. Here, we are given time-varying target trajectories $(\mathbf{x}_k^{\text{ref}}, \mathbf{u}_k^{\text{ref}})$, and the controller's job is to keep the system close to these. The cost is typically quadratic:

$$
\begin{aligned}
    c(\mathbf{x}_k, \mathbf{u}_k) &= \| \mathbf{x}_k - \mathbf{x}_k^{\text{ref}} \|_{\mathbf{Q}}^2 + \| \mathbf{u}_k - \mathbf{u}_k^{\text{ref}} \|_{\mathbf{R}}^2 \\
    c(\mathbf{x}_N) &= \| \mathbf{x}_N - \mathbf{x}_N^{\text{ref}} \|_{\mathbf{P}}^2 \enspace .
\end{aligned}
$$

When dynamics are linear and constraints are polyhedral, this yields a convex quadratic program at each time step.

## Regulatory MPC

In regulation tasks, we aim to bring the system back to an equilibrium point $(\mathbf{x}^e, \mathbf{u}^e)$, typically in the presence of disturbances. This is simply tracking MPC with constant references:

$$
\begin{aligned}
    c(\mathbf{x}_k, \mathbf{u}_k) &= \| \mathbf{x}_k - \mathbf{x}^e \|_{\mathbf{Q}}^2 + \| \mathbf{u}_k - \mathbf{u}^e \|_{\mathbf{R}}^2 \\
    c(\mathbf{x}_N) &= \| \mathbf{x}_N - \mathbf{x}^e \|_{\mathbf{P}}^2 \enspace .
\end{aligned}
$$

To guarantee stability, it is common to include a terminal constraint $\mathbf{x}_N \in \mathcal{X}_f$, where $\mathcal{X}_f$ is a control-invariant set under a known feedback law.


## Economic MPC

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

Some systems are exposed to external disturbances or small errors in the model. In those cases, we want the controller to make decisions that will still work no matter what happens, as long as the disturbances stay within some known bounds. This is the idea behind **robust MPC**.

Instead of planning a single trajectory, the controller plans a "nominal" path (what would happen in the absence of any disturbance) and then adds a feedback correction to react to whatever disturbances actually occur. This looks like:

$$
\mathbf{u}_k = \bar{\mathbf{u}}_k + \mathbf{K} (\mathbf{x}_k - \bar{\mathbf{x}}_k) \enspace ,
$$

where $\bar{\mathbf{u}}_k$ is the planned input and $\mathbf{K}$ is a feedback gain that pulls the system back toward the nominal path if it deviates.

Because we know the worst-case size of the disturbance, we can estimate how far the real state might drift from the plan, and "shrink" the constraints accordingly. The result is that the nominal plan is kept safely away from constraint boundaries, so even if the system gets pushed around, it stays inside limits. This is often called **tube MPC** because the true trajectory stays inside a tube around the nominal one.

The main benefit is that we can handle uncertainty without solving a complicated worst-case optimization at every time step. All the uncertainty is accounted for in the design of the feedback $\mathbf{K}$ and the tightened constraints.


## Stochastic MPC

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

When systems involve discrete switches  (eg. on/off valves, mode selection, or combinatorial logic) the MPC problem must include integer or binary variables. These show up in constraints like

$$
\boldsymbol{\delta}_k \in \{0,1\}^m, \qquad \mathbf{u}_k \in \mathcal{U}(\boldsymbol{\delta}_k)
$$

along with mode-dependent dynamics and costs. The resulting formulation is a **mixed-integer nonlinear program** (MINLP). The receding-horizon idea is the same, but each solve is more expensive due to the combinatorial nature of the decision space.

## Distributed and Decentralized MPC

Large-scale systems often consist of interacting subsystems. Distributed MPC decomposes the global NLP into smaller ones that run in parallel, with coordination constraints enforcing consistency across shared variables:

$$
\sum_{i} \mathbf{H}^i \mathbf{z}^i_k = \mathbf{0} \qquad \text{(coupling constraint)}
$$

Each subsystem solves a local problem over its own state and input variables, then exchanges information with neighbors. Coordination can be done via primal–dual methods, ADMM, or consensus schemes, but each local block looks like a standard MPC problem.


## Adaptive and Learning-Based MPC

In practice, we may not know the true model $\mathbf{F}_k$ or cost function $c$ precisely. In **adaptive MPC**, these are updated online from data:

$$
\mathbf{x}_{k+1} = \mathbf{F}_k(\mathbf{x}_k, \mathbf{u}_k; \boldsymbol{\theta}_t), \qquad
c(\mathbf{x}_k, \mathbf{u}_k) = c(\mathbf{x}_k, \mathbf{u}_k; \boldsymbol{\phi}_t)
$$

The parameters $\boldsymbol{\theta}_t$ and $\boldsymbol{\phi}_t$ are learned in real time. When combined with policy distillation, value approximation, or trajectory imitation, this leads to overlaps with reinforcement learning where the MPC solutions act as supervision for a reactive policy.


## Robustness and Failure Handling

The wave-energy example assumes that every optimization finishes before the next control update and that its prediction model remains accurate over the horizon. An operational MPC controller must also handle incompatible constraints, modeling errors, and missed computation deadlines.

A disturbance can move the measured state to a point from which the requested target is unreachable within the horizon. Model mismatch can make a trajectory feasible in prediction and infeasible on the plant. A solver can also stop because of an ill-conditioned local model or a changing active set. These cases require an explicit hierarchy of hard constraints, soft objectives, and fallback actions. The following mechanisms modify the finite-horizon problem or its deployment without changing the receding-horizon principle.
## Softening Constraints Through Slack Variables

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

The offline frequency schedule in [Trajectory Optimization](trajectories.md#example-offline-frequency-planning-for-inference)
cannot react when a burst arrives earlier than forecast. Receding-horizon
control uses the same aggregate model but replaces the initial forecast after
each second. The scheduling rule remains fixed at the 512-token interleaved
chunked-prefill surrogate from the modeling chapter, so this experiment
isolates feedback through the GPU clock.

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
proxies. The first divides predicted queued prefill tokens by the profiled
prefill rate. The second divides the predicted number of active decode requests
by the profiled decode rate. They are planning surrogates, not realized
request-level TTFT and TPOT; the detailed replay computes the latter. The
frequency difference $\Delta f_k$ is normalized by the profiled clock range.
The slacks $s_{P,k}$ and $s_{T,k}$ penalize violations of the experimental
power and thermal limits, and $B_{t+10}$ penalizes unfinished work at the end of
the horizon. SLSQP receives at most 50 iterations and a tolerance of $10^{-6}$.
Only the first frequency is applied. The continuous result is rounded downward
to the nearest clock in the committed profile before the next state is
observed.

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
:caption: Maximum-clock, reactive, open-loop, and MPC frequency control under the same shifted request trace and fixed interleaved chunked-prefill surrogate. At each second, the dashed MPC segment is the current ten-second plan and the solid segment is the action history. Later plans are not revealed before they are computed.

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

**Recorded metrics.** The table compares the four frequency controllers under
identical shifted arrivals and the fixed interleaved chunked-prefill surrogate.
Fallback count records every MPC step that exceeded its feasibility, finiteness,
or timing requirement.

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
3.91 seconds, at 2,347.9 joules. The unchanged offline plan uses 2,202.6 joules
but raises that latency to 6.77 seconds. MPC lies between them at 4.14 seconds
and 2,291.4 joules. The reactive governor uses slightly less energy than MPC,
2,255.2 joules, but has a higher mean time to first token of 4.82 seconds. MPC
accepts 58 solves and invokes the recorded fallback twice; its maximum solve
time is 0.061 seconds, below the 0.8-second deadline. Thus no controller
dominates the others on latency and energy, and the reported experimental SLO
violation rates remain substantial for every controller.

The table reports time to first token, time per output token, end-to-end
latency, throughput, energy per output token, peak power and temperature,
unfinished work, constraint violations, and fallback count. These quantities
separate three possible outcomes. A lower clock may save energy while violating
latency, maximum clock may meet latency at an unnecessary energy cost, and MPC
may trade between them by responding to the observed queue.

This is a single-GPU simulation driven by the committed profile, not a claim
about every serving stack. The current profile is an engineering surrogate that
tests the controller and fallback logic without supplying measured hardware
evidence. The arrival
forecast is deliberately simple, output length is observed only at completion,
and the scheduler is held fixed. Network delay, tokenization, multi-GPU
communication, and model-quality effects are omitted. The shifted burst is one
controlled disturbance, so relative performance on that trace does not
establish dominance under other workloads.

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




# Computational Efficiency via Parametric Programming

Real-time model predictive control places strict limits on computation. In applications such as adaptive optics, the controller must run at kilohertz rates. A sampling frequency of 1000 Hz allows only one millisecond per step to compute and apply a control input. This makes efficiency a first-class concern.

The structure of MPC lends itself naturally to optimization reuse. Each time step requires solving a problem with the same dynamics and constraints. Only the initial state, forecasts, or reference signals change. Instead of treating each instance as a new problem, we can frame MPC as a *parametric optimization problem* and focus on how the solution evolves with the parameter.

## General Framework: Parametric Optimization

We begin with a general optimization problem indexed by a parameter $\boldsymbol{\theta} \in \Theta \subset \mathbb{R}^p$:

$$
\begin{aligned}
\min_{\mathbf{x} \in \mathbb{R}^n} \quad & f(\mathbf{x}; \boldsymbol{\theta}) \\
\text{s.t.} \quad & \mathbf{g}(\mathbf{x}; \boldsymbol{\theta}) \le \mathbf{0}, \\
& \mathbf{h}(\mathbf{x}; \boldsymbol{\theta}) = \mathbf{0}.
\end{aligned}
$$

For each value of $\boldsymbol{\theta}$, we obtain a concrete optimization problem. The goal is to understand how the optimizer $\mathbf{x}^\star(\boldsymbol{\theta})$ and value function

$$
v(\boldsymbol{\theta}) := \inf\{\, f(\mathbf{x}; \boldsymbol{\theta}) : \mathbf{x} \text{ feasible at } \boldsymbol{\theta}\,\}
$$

depend on $\boldsymbol{\theta}$.

When the problem is smooth and regular, the Karush–Kuhn–Tucker (KKT) conditions characterize optimality:

$$
\begin{aligned}
\nabla_{\mathbf{x}} f(\mathbf{x}; \boldsymbol{\theta})
+ \nabla_{\mathbf{x}} \mathbf{g}(\mathbf{x}; \boldsymbol{\theta})^\top \boldsymbol{\lambda}
+ \nabla_{\mathbf{x}} \mathbf{h}(\mathbf{x}; \boldsymbol{\theta})^\top \boldsymbol{\nu} &= 0, \\
\mathbf{g}(\mathbf{x}; \boldsymbol{\theta}) \le 0, \quad
\boldsymbol{\lambda} \ge 0, \quad
\lambda_i g_i(\mathbf{x}; \boldsymbol{\theta}) &= 0, \\
\mathbf{h}(\mathbf{x}; \boldsymbol{\theta}) &= 0.
\end{aligned}
$$

If the active set remains fixed over changes in $\boldsymbol{\theta}$, the implicit function theorem ensures that the mappings

$$
\boldsymbol{\theta} \mapsto \mathbf{x}^\star(\boldsymbol{\theta}), \quad
\boldsymbol{\theta} \mapsto \boldsymbol{\lambda}^\star(\boldsymbol{\theta}), \quad
\boldsymbol{\theta} \mapsto \boldsymbol{\nu}^\star(\boldsymbol{\theta})
$$

are differentiable.

In linear and quadratic programming, this structure becomes even more tractable. Consider a linear program with affine dependence on $\boldsymbol{\theta}$:

$$
\min_{\mathbf{x}} \ \mathbf{c}(\boldsymbol{\theta})^\top \mathbf{x}
\quad \text{s.t.} \quad \mathbf{A}(\boldsymbol{\theta})\mathbf{x} \le \mathbf{b}(\boldsymbol{\theta}).
$$

Each active set determines a basis and thus a region in $\Theta$ where the solution is affine in $\boldsymbol{\theta}$. The feasible parameter space is partitioned into polyhedral regions, each with its own affine law.

Similarly, in strictly convex quadratic programs

$$
\min_{\mathbf{x}} \ \tfrac{1}{2} \mathbf{x}^\top \mathbf{H} \mathbf{x} + \mathbf{q}(\boldsymbol{\theta})^\top \mathbf{x}
\quad \text{s.t.} \quad \mathbf{A}\mathbf{x} \le \mathbf{b}(\boldsymbol{\theta}), \qquad \mathbf{H} \succ 0,
$$

each active set again leads to an affine optimizer, with piecewise-affine global structure and a piecewise-quadratic value function.

Parametric programming focuses on the structure of the map $\boldsymbol{\theta} \mapsto \mathbf{x}^\star(\boldsymbol{\theta})$, and the regions over which this map takes a simple form.

### Solution Sensitivity via the Implicit Function Theorem 

We often meet equations of the form

$$
F(y,\boldsymbol{\theta})=0,
$$

where $y\in\mathbb{R}^m$ are unknowns and $\boldsymbol{\theta}\in\mathbb{R}^p$ are parameters. The **implicit function theorem** says that, if $F$ is smooth and the Jacobian with respect to $y$,

$$
\frac{\partial F}{\partial y}(y^\star,\boldsymbol{\theta}^\star),
$$

is invertible at a solution $(y^\star,\boldsymbol{\theta}^\star)$, then in a neighborhood of $\boldsymbol{\theta}^\star$ there exists a unique smooth mapping $y(\boldsymbol{\theta})$ with $F(y(\boldsymbol{\theta}),\boldsymbol{\theta})=0$ and $y(\boldsymbol{\theta}^\star)=y^\star$. Moreover, its derivative is

$$
\frac{d y}{d\boldsymbol{\theta}}(\boldsymbol{\theta}^\star)
\;=\;
-\Big(\tfrac{\partial F}{\partial y}(y^\star,\boldsymbol{\theta}^\star)\Big)^{-1}
\;\tfrac{\partial F}{\partial \boldsymbol{\theta}}(y^\star,\boldsymbol{\theta}^\star).
$$

In words: if the square Jacobian in $y$ is nonsingular, the solution varies smoothly with the parameter, and we can differentiate it by solving one linear system.

Return to $(P_{\theta})$ and its KKT system. Collect the primal and dual variables into

$$
y \;:=\; (\mathbf{x},\,\boldsymbol{\lambda},\,\boldsymbol{\nu}),
$$

and write the KKT equations as a single residual

$$
F(y,\boldsymbol{\theta}) \;=\; 
\begin{bmatrix}
\nabla_{\mathbf{x}} f(\mathbf{x};\boldsymbol{\theta})
+ \nabla_{\mathbf{x}} \mathbf{g}(\mathbf{x};\boldsymbol{\theta})^\top \boldsymbol{\lambda}
+ \nabla_{\mathbf{x}} \mathbf{h}(\mathbf{x};\boldsymbol{\theta})^\top \boldsymbol{\nu} \\
\mathbf{h}(\mathbf{x};\boldsymbol{\theta}) \\
\mathbf{g}_\mathcal{A}(\mathbf{x};\boldsymbol{\theta})
\end{bmatrix}
\;=\; \mathbf{0}.
$$

Here $\mathcal{A}$ denotes the set of inequality constraints active at the solution (the complementarity part is encoded by keeping $\mathcal{A}$ fixed; see below).

To invoke IFT, we need the Jacobian $\partial F/\partial y$ to be invertible at $(y^\star,\boldsymbol{\theta}^\star)$. Standard regularity conditions that ensure this are:

* **LICQ (Linear Independence Constraint Qualification)** at $(\mathbf{x}^\star,\boldsymbol{\theta}^\star)$: the gradients of all active constraints are linearly independent.
* **Second-order sufficiency** on the critical cone (the Lagrangian Hessian is positive definite on feasible directions).
* **Strict complementarity** (optional but convenient): each active inequality has strictly positive multiplier.

Under these, the **KKT matrix**,

$$
K \;=\;
\frac{\partial F}{\partial y}(y^\star,\boldsymbol{\theta}^\star)
\;=\;
\begin{bmatrix}
\nabla^2_{\mathbf{x}\mathbf{x}} \mathcal{L}(\mathbf{x}^\star,\boldsymbol{\lambda}^\star,\boldsymbol{\nu}^\star;\boldsymbol{\theta}^\star)
& \nabla_{\mathbf{x}} \mathbf{g}_\mathcal{A}(\mathbf{x}^\star;\boldsymbol{\theta}^\star)^\top
& \nabla_{\mathbf{x}} \mathbf{h}(\mathbf{x}^\star;\boldsymbol{\theta}^\star)^\top \\
\nabla_{\mathbf{x}} \mathbf{g}_\mathcal{A}(\mathbf{x}^\star;\boldsymbol{\theta}^\star) & 0 & 0 \\
\nabla_{\mathbf{x}} \mathbf{h}(\mathbf{x}^\star;\boldsymbol{\theta}^\star) & 0 & 0
\end{bmatrix},
$$

is nonsingular. Here $\mathcal{L}=f+\boldsymbol{\lambda}^\top \mathbf{g}+\boldsymbol{\nu}^\top \mathbf{h}$.

The right-hand side sensitivity to parameters is

$$
G \;=\; \frac{\partial F}{\partial \boldsymbol{\theta}}(y^\star,\boldsymbol{\theta}^\star)
\;=\;
\begin{bmatrix}
\nabla_{\boldsymbol{\theta}}\nabla_{\mathbf{x}} f
+ \sum_{i\in\mathcal{A}} \lambda_i^\star \nabla_{\boldsymbol{\theta}}\nabla_{\mathbf{x}} g_i
+ \sum_j \nu_j^\star \nabla_{\boldsymbol{\theta}}\nabla_{\mathbf{x}} h_j \\
\nabla_{\boldsymbol{\theta}} \mathbf{h} \\
\nabla_{\boldsymbol{\theta}} \mathbf{g}_\mathcal{A}
\end{bmatrix}_{(\mathbf{x}^\star,\boldsymbol{\theta}^\star)} .
$$

IFT then gives **local differentiability of the optimizer and multipliers**:

$$
\frac{d y^\star}{d\boldsymbol{\theta}}(\boldsymbol{\theta}^\star)
\;=\; -\,K^{-1} G.
$$

The formula above is valid **as long as the active set $\mathcal{A}$ does not change**. If a constraint switches between active/inactive, the mapping remains piecewise smooth, but the derivative may jump. In MPC, this is exactly why warm-starts are very effective most of the time and occasionally require a refactorization when the active set flips.

In parametric MPC, $\boldsymbol{\theta}$ gathers the current state, references, and forecasts. The IFT tells us that, under regularity and a stable active set, the optimal trajectory and first input vary smoothly with $\boldsymbol{\theta}$. The linear map $-K^{-1}G$ is exactly the object used in sensitivity-based warm starts and real-time iterations: small changes in $\boldsymbol{\theta}$ can be propagated through a single KKT solve to update the primal–dual guess before taking one or two Newton/SQP steps.

### Predictor-Corrector MPC

We start with a smooth root-finding problem

$$
F(y)=0,\qquad F:\mathbb{R}^m\to\mathbb{R}^m.
$$

**Newton's method** iterates

$$
y^{(t+1)} \;=\; y^{(t)} - \big[\nabla F(y^{(t)})\big]^{-1} F\big(y^{(t)}\big),
$$

or equivalently solves the linearized system

$$
\nabla F(y^{(t)})\,\Delta y^{(t)} = -F\big(y^{(t)}\big),\qquad y^{(t+1)}=y^{(t)}+\Delta y^{(t)}.
$$

Convergence is local and fast when the Jacobian is nonsingular and the initial guess is close.

Now suppose the root depends on a parameter:

$$
F\big(y,\theta\big)=0,\qquad \theta\in\mathbb{R}.
$$

We want the solution path $\theta\mapsto y^\star(\theta)$. **Numerical continuation** advances $\theta$ in small steps and uses the previous solution as a warm start for the next Newton solve. This is the simplest and most effective way to "track" solutions of parametric systems.

At a known solution $(y^\star,\theta^\star)$, differentiate $F(y^\star(\theta),\theta)=0$ with respect to $\theta$:

$$
\nabla_y F(y^\star,\theta^\star)\,\frac{dy^\star}{d\theta}(\theta^\star) \;+\; \nabla_\theta F(y^\star,\theta^\star) \;=\; 0.
$$

If $\nabla_y F$ is invertible (IFT conditions), the **tangent** is

$$
\frac{dy^\star}{d\theta}(\theta^\star) \;=\; -\big[\nabla_y F(y^\star,\theta^\star)\big]^{-1}\,\nabla_\theta F(y^\star,\theta^\star).
$$

This is exactly the **implicit differentiation** formula. Continuation uses it as a **predictor**:

$$
y_{\text{pred}} \;=\; y^\star(\theta^\star) \;+\; \Delta\theta\;\frac{dy^\star}{d\theta}(\theta^\star).
$$

Then a few **corrector** steps apply Newton to $F(\,\cdot\,,\theta^\star+\Delta\theta)=0$ starting from $y_{\text{pred}}$. If Newton converges quickly, the step $\Delta\theta$ was appropriate; otherwise reduce $\Delta\theta$ and retry.

For parametric KKT systems, set $y=(\mathbf{x},\boldsymbol{\lambda},\boldsymbol{\nu})$ where $\mathbf{x}$ stacks the primal decision variables (states and inputs), and $F(y,\theta)=0$ the KKT residual with $\theta$ collecting state, references, forecasts. The **KKT matrix** $K=\partial F/\partial y$ and **parameter sensitivity** $G=\partial F/\partial \theta$ give the tangent

$$
\frac{dy^\star}{d\theta} \;=\; -\,K^{-1}G.
$$

Continuation then becomes:

1. **Predictor**: $y_{\text{pred}} = y^\star + (\Delta\theta)\,(-K^{-1}G)$.
2. **Corrector**: a few Newton/SQP steps on the KKT equations at the new $\theta$.

In MPC, this yields efficient **warm starts** across time. As the parameter $\theta_t$ (current state and references) changes slightly, we predict the new primal-dual point and correct with 1–2 iterations, which is often enough to reach tolerance in real time.



## Amortized Optimization and Neural Approximation of Controllers

The idea of reusing structure across similar optimization problems is not exclusive to parametric programming. In machine learning, a related concept known as **amortized optimization** aims to reduce the cost of repeated inference by replacing explicit optimization with a function that has been *learned* to approximate the solution map. This approach shifts the computational burden from online solving to offline training.

The goal is to construct a function $\hat{\pi}_{\phi}(\boldsymbol{\theta})$, typically parameterized by a neural network, that maps the input $\boldsymbol{\theta}$ to an approximate solution $\hat{z}^\star(\boldsymbol{\theta})$ or control action $\hat{\mathbf{u}}_0^\star(\boldsymbol{\theta})$. Once trained, this map can be evaluated quickly at runtime, with no need to solve an optimization problem explicitly.

Amortized optimization has emerged in several contexts:

* In **probabilistic inference**, where variational autoencoders (VAEs) amortize the computation of posterior distributions across a dataset.
* In **meta-learning**, where the objective is to learn a model that generalizes across tasks by internalizing how to adapt.
* In **hyperparameter optimization**, where learning a surrogate model can guide the search over configuration space efficiently.

This perspective has also begun to influence control. Recent work investigates how to **amortize nonlinear MPC (NMPC)** policies into neural networks. The training data come from solving many instances of the underlying optimal control problem offline. The resulting neural policy $\hat{\pi}_\phi$ acts as a differentiable, low-latency controller that can generalize to new situations within the training distribution.

Compared to explicit MPC, which partitions the parameter space and stores exact solutions region by region, amortized control smooths over the domain by learning an approximate policy globally. It is less precise, but scalable to high-dimensional problems where enumeration of regions is impossible.

Neural network amortization is advantageous due to the expressivity of these models. However, the challenge is ensuring **constraint satisfaction and safety**, which are hard to guarantee with unconstrained neural approximators. Hybrid approaches attempt to address this by combining a neural warm-start policy with a final projection step, or by embedding the network within a constrained optimization layer. Other strategies include learning structured architectures that respect known physics or control symmetries.


## Imitation Learning Framework
Consider a fixed horizon $N$ and parameter vector $\boldsymbol{\theta}$ encoding the current state, references, and forecasts. The oracle MPC controller solves

$$
\begin{aligned}
z^\star(\boldsymbol{\theta}) \in \arg\min_{z=(\mathbf{x}_{0:N},\mathbf{u}_{0:N-1})}
&\; J(z;\boldsymbol{\theta})\\
\text{s.t. }& \mathbf{x}_{k+1}=f(\mathbf{x}_k,\mathbf{u}_k;\boldsymbol{\theta}),\quad k=0..N-1,\\
& g(\mathbf{x}_k,\mathbf{u}_k;\boldsymbol{\theta})\le 0,\; h(\mathbf{x}_N;\boldsymbol{\theta})=0.
\end{aligned}
$$

The applied action is $\pi^\star(\boldsymbol{\theta}) := \mathbf{u}_0^\star(\boldsymbol{\theta})$. Our goal is to learn a fast surrogate mapping $\hat{\pi}_\phi:\boldsymbol{\theta}\mapsto \hat{\mathbf{u}}_0 \approx \pi^\star(\boldsymbol{\theta})$ that can be evaluated in microseconds, optionally followed by a safety projection layer.

**Supervised learning from oracle solutions.**
One first samples parameters $\boldsymbol{\theta}^{(i)}$ from the operational domain and solves the corresponding NMPC problems offline. The resulting dataset

$$
\mathcal{D} = \{ (\boldsymbol{\theta}^{(i)},\, \mathbf{u}_0^\star(\boldsymbol{\theta}^{(i)})) \}_{i=1}^M
$$

is then used to train a neural network $\hat{\pi}_\phi$ by minimizing

$$
\min_\phi \; \frac{1}{M}\sum_{i=1}^M \big\|\hat{\pi}_\phi(\boldsymbol{\theta}^{(i)}) - \mathbf{u}_0^\star(\boldsymbol{\theta}^{(i)})\big\|^2 .
$$

Once trained, the network acts as a surrogate for the optimizer, providing instantaneous evaluations that approximate the MPC law.


# Example: Propofol Infusion Control 

This problem explores the control of propofol infusion in total intravenous anesthesia (TIVA). Our presentation follows the problem formulation developped by {cite:t}`Sawaguchi2008`. The primary objective is to maintain the desired level of unconsciousness while minimizing adverse reactions and ensuring quick recovery after surgery. 

The level of unconsciousness is measured by the Bispectral Index (BIS), which is obtained using an electroencephalography (EEG) device. The BIS ranges from $0$ (complete suppression of brain activity) to $100$ (fully awake), with the target range for general anesthesia typically between $40$ and $60$.

The goal is to design a control system that regulates the infusion rate of propofol to maintain the BIS within the target range. This can be formulated as an optimal control problem:

$$
\begin{align*}
\min_{u(t)} & \int_{0}^{T} \left( BIS(t) - BIS_{\text{target}} \right)^2 + \lambda\, u(t)^2 \, dt \\
\text{subject to:} \\
\dot{x}_1 &= -(k_{10} + k_{12} + k_{13})x_1 + k_{21}x_2 + k_{31}x_3 + \frac{u(t)}{V_1} \\
\dot{x}_2 &= k_{12}x_1 - k_{21}x_2 \\
\dot{x}_3 &= k_{13}x_1 - k_{31}x_3 \\
\dot{x}_e &= k_{e0}(x_1 - x_e) \\
BIS(t) &= E_0 - E_{\text{max}}\frac{x_e^\gamma}{x_e^\gamma + EC_{50}^\gamma}
\end{align*}
$$

Where:
- $u(t)$ is the propofol infusion rate (mg/kg/h)
- $x_1$, $x_2$, and $x_3$ are the drug concentrations in different body compartments
- $x_e$ is the effect-site concentration
- $k_{ij}$ are rate constants for drug transfer between compartments
- $BIS(t)$ is the Bispectral Index
- $\lambda$ is a regularization parameter penalizing excessive drug use
- $E_0$, $E_{\text{max}}$, $EC_{50}$, and $\gamma$ are parameters of the pharmacodynamic model

The specific dynamics model used in this problem is so-called "Pharmacokinetic-Pharmacodynamic Model" and consists of three main components:

1. **Pharmacokinetic Model**, which describes how the drug distributes through the body over time. It's based on a three-compartment model:
   - Central compartment (blood and well-perfused organs)
   - Shallow peripheral compartment (muscle and other tissues)
   - Deep peripheral compartment (fat)

2. **Effect Site Model**, which represents the delay between drug concentration in the blood and its effect on the brain.

3. **Pharmacodynamic Model** that relates the effect-site concentration to the observed BIS.

The propofol infusion control problem presents several interesting challenges from a research perspective. 
First, there is a delay in how fast the drug can reach a different compartments in addition to the BIS measurements which can lag. This could lead to instability if not properly addressed in the control design. 

Furthermore, every patient is different from another. Hence, we cannot simply learn a single controller offline and hope that it will generalize to an entire patient population. We will account for this variability through Model Predictive Control (MPC) and dynamically adapt to the model mismatch through replanning. How a patient will react to a given dose of drug also varies and must be carefully controlled to avoid overdoses. This adds an additional layer of complexity since we have to incorporate safety constraints. Finally, the patient might suddenly change state, for example due to surgical stimuli, and the controller must be able to adapt quickly to compensate for the disturbance to the system.

```{code-cell} python
:tags: [hide-input]


#  label: fig-mpc-propofol
#  caption: Closed-loop MPC for propofol infusion keeps the Bispectral Index near the target (top), regulates infusion rates (middle), and tracks the effect-site concentration (bottom).

%config InlineBackend.figure_format = 'retina'
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Apply book style
try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except (ImportError, OSError):
    pass  # Use matplotlib defaults

class Patient:
    def __init__(self, age, weight):
        self.age = age
        self.weight = weight
        self.set_pk_params()
        self.set_pd_params()

    def set_pk_params(self):
        self.v1 = 4.27 * (self.weight / 70) ** 0.71 * (self.age / 30) ** (-0.39)
        self.v2 = 18.9 * (self.weight / 70) ** 0.64 * (self.age / 30) ** (-0.62)
        self.v3 = 238 * (self.weight / 70) ** 0.95
        self.cl1 = 1.89 * (self.weight / 70) ** 0.75 * (self.age / 30) ** (-0.25)
        self.cl2 = 1.29 * (self.weight / 70) ** 0.62
        self.cl3 = 0.836 * (self.weight / 70) ** 0.77
        self.k10 = self.cl1 / self.v1
        self.k12 = self.cl2 / self.v1
        self.k13 = self.cl3 / self.v1
        self.k21 = self.cl2 / self.v2
        self.k31 = self.cl3 / self.v3
        self.ke0 = 0.456

    def set_pd_params(self):
        self.E0 = 100
        self.Emax = 100
        self.EC50 = 3.4
        self.gamma = 3

def pk_model(x, u, patient):
    x1, x2, x3, xe = x
    dx1 = -(patient.k10 + patient.k12 + patient.k13) * x1 + patient.k21 * x2 + patient.k31 * x3 + u / patient.v1
    dx2 = patient.k12 * x1 - patient.k21 * x2
    dx3 = patient.k13 * x1 - patient.k31 * x3
    dxe = patient.ke0 * (x1 - xe)
    return np.array([dx1, dx2, dx3, dxe])

def pd_model(ce, patient):
    return patient.E0 - patient.Emax * (ce ** patient.gamma) / (ce ** patient.gamma + patient.EC50 ** patient.gamma)

def simulate_step(x, u, patient, dt):
    x_next = x + dt * pk_model(x, u, patient)
    bis = pd_model(x_next[3], patient)
    return x_next, bis

def objective(u, x0, patient, dt, N, target_bis):
    x = x0.copy()
    total_cost = 0
    for i in range(N):
        x, bis = simulate_step(x, u[i], patient, dt)
        total_cost += (bis - target_bis)**2 + 0.1 * u[i]**2
    return total_cost

def mpc_step(x0, patient, dt, N, target_bis):
    u0 = 10 * np.ones(N)  # Initial guess
    bounds = [(0, 20)] * N  # Infusion rate between 0 and 20 mg/kg/h
    
    result = minimize(objective, u0, args=(x0, patient, dt, N, target_bis),
                      method='SLSQP', bounds=bounds)
    
    return result.x[0]  # Return only the first control input

def run_mpc_simulation(patient, T, dt, N, target_bis):
    rng = np.random.default_rng(2026)
    steps = int(T / dt)
    x = np.zeros((steps+1, 4))
    bis = np.zeros(steps+1)
    u = np.zeros(steps)
    
    for i in range(steps):
        # Add noise to the current state to simulate real-world uncertainty
        x_noisy = x[i] + rng.normal(0, 0.01, size=4)
        
        # Use noisy state for MPC planning
        u[i] = mpc_step(x_noisy, patient, dt, N, target_bis)
        
        # Evolve the true state using the deterministic model
        x[i+1], bis[i] = simulate_step(x[i], u[i], patient, dt)
    
    bis[-1] = pd_model(x[-1, 3], patient)
    return x, bis, u

# Set up the problem
patient = Patient(age=40, weight=70)
T = 120  # Total time in minutes
dt = 0.5  # Time step in minutes
N = 20  # Prediction horizon
target_bis = 50  # Target BIS value

# Run MPC simulation
x, bis, u = run_mpc_simulation(patient, T, dt, N, target_bis)

# Plot results
t = np.arange(0, T+dt, dt)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

ax1.plot(t, bis)
ax1.set_ylabel('BIS')
ax1.set_ylim(0, 100)
ax1.axhline(y=target_bis, color='r', linestyle='--')

ax2.plot(t[:-1], u)
ax2.set_ylabel('Infusion Rate (mg/kg/h)')

ax3.plot(t, x[:, 3])
ax3.set_ylabel('Effect-site Concentration (µg/mL)')
ax3.set_xlabel('Time (min)')

plt.tight_layout()

print(f"Initial BIS: {bis[0]:.2f}")
print(f"Final BIS: {bis[-1]:.2f}")
print(f"Mean infusion rate: {np.mean(u):.2f} mg/kg/h")
print(f"Final effect-site concentration: {x[-1, 3]:.2f} µg/mL")
```





## Self-checks

:::{exercise} Receding horizon
:label: ex-mpc-check-1

An MPC solver returns a sequence $(u_0^*,\ldots,u_{N-1}^*)$. Which controls are normally applied before the problem is solved again?
:::

:::{solution} ex-mpc-check-1
:class: dropdown

Only the first control (or first short control block) is applied. The state is measured again and the horizon is shifted before re-optimizing.
:::

:::{exercise} Terminal ingredients
:label: ex-mpc-check-2

What roles do a terminal cost and a terminal constraint play in finite-horizon MPC?
:::

:::{solution} ex-mpc-check-2
:class: dropdown

The terminal cost approximates value beyond the horizon; the terminal constraint can keep the endpoint in a region from which a known controller remains feasible and stable.
:::

:::{exercise} Disturbance response
:label: ex-mpc-check-3

Why does resolving the same finite-horizon optimization after each measurement provide feedback even if the prediction model is deterministic?
:::

:::{solution} ex-mpc-check-3
:class: dropdown

The newly measured state contains the accumulated effect of disturbances and model error. Reinitializing the optimization from that state changes the planned controls accordingly.
:::

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
