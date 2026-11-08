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
# Receding-Horizon Control

An open-loop trajectory specifies controls from an initial state to a final
time. During execution, the control $\mathbf u_t$ depends only on the clock,
reading precomputed values or interpolating between them. The
[MPPI examples](model-predictive-path-integral-control.md#sec-mppi) already used a
different execution rule: apply one action, observe the new state, and update
the remaining plan. Receding-horizon control applies this rule to any suitable
finite-horizon optimizer, including the transcription methods developed in
the preceding chapters.

Real systems face modeling errors, external disturbances, and measurement noise that accumulate over time. A precomputed trajectory becomes increasingly irrelevant as these perturbations push the actual system state away from the predicted path. The solution is to incorporate feedback, making control decisions that respond to the current state rather than blindly following a predetermined schedule. While dynamic programming provides the theoretical framework for deriving feedback policies through value functions and Bellman equations, there exists a more direct approach that leverages the trajectory optimization methods already developed.

Can the finite-horizon planner itself become a feedback controller by solving
again whenever a new state is measured?

## Closing the Loop by Replanning

Which action from each finite-horizon solution should be applied before the
state is measured and the horizon is shifted forward?

Model Predictive Control creates a feedback controller by repeatedly solving trajectory optimization problems. Rather than computing a single trajectory for the entire task duration, MPC solves a finite-horizon problem at each time step, starting from the current measured state. The controller then applies only the first control action from this solution before repeating the entire process. This strategy transforms any trajectory optimization method into a feedback controller.

The battery example in [Model Interfaces](model-interfaces.md#fast-charging-when-resistance-drifts)
used a transparent current governor: it tested a local voltage and thermal
envelope after updating one resistance parameter. MPC retains that structured
model but optimizes an entire future current sequence and repeats the
optimization as measurements arrive.

### The Receding Horizon Principle

The defining characteristic of MPC is its receding horizon strategy. At each time step, the controller solves an optimization problem looking a fixed duration into the future, but this prediction window constantly moves forward in time. The horizon "recedes" because it always starts from the current time and extends forward by the same amount.

Consider the discrete-time optimal control problem in Bolza form:

$$
\begin{aligned}
\text{minimize} \quad & c_N(\mathbf{x}_N) + \sum_{k=0}^{N-1} c_k(\mathbf{x}_k, \mathbf{u}_k) \\
\text{subject to} \quad & \mathbf{x}_{k+1} = \mathbf{f}_k(\mathbf{x}_k, \mathbf{u}_k) \\
& \mathbf{g}_k(\mathbf{x}_k, \mathbf{u}_k) \leq \mathbf{0} \\
& \mathbf{x}_{\text{min}} \leq \mathbf{x}_k \leq \mathbf{x}_{\text{max}} \\
& \mathbf{u}_{\text{min}} \leq \mathbf{u}_k \leq \mathbf{u}_{\text{max}} \\
\text{given} \quad & \mathbf{x}_0 = \mathbf{x}_{\text{current}}
\end{aligned}
$$

This is the Bolza problem of [Finite-Horizon Optimal Control Problems](discrete-time-optimal-control.md),
with its index counted from the current time rather than from the start of
the task. If the state is measured at time $t$, then
$\mathbf x_0=\mathbf x_{\text{current}}$ is that measurement, $\mathbf x_k$ is
the state predicted for time $t+k$, and the $N$ controls
$\mathbf u_0,\ldots,\mathbf u_{N-1}$ span the prediction horizon, as in the
MPPI algorithm. For a time-varying task, $c_k$, $\mathbf f_k$, and
$\mathbf g_k$ are the stage cost, dynamics, and constraints of time $t+k$.
The terminal cost $c_N$ is charged at the end of the prediction horizon, not
at the end of the task.

At time step $t$, this problem optimizes over the interval $[t, t+N]$. At the next time step $t+1$, the horizon shifts to $[t+1, t+N+1]$. What makes this work is that only the first control $\mathbf{u}_0^\star$ from each optimization is applied. The remaining controls $\mathbf{u}_1^\star, \ldots, \mathbf{u}_{N-1}^\star$ are discarded, though they may initialize the next optimization through warm-starting.

This receding horizon principle enables feedback without computing an explicit policy. By constantly updating predictions based on current measurements, MPC naturally corrects for disturbances and model errors. The apparent waste of computing but not using most of the trajectory is actually the mechanism that provides robustness.

### Horizon Selection and Problem Formulation

The choice of prediction horizon depends on the control objective. We distinguish between three cases, each requiring different mathematical formulations.

#### Infinite-Horizon Regulation

For stabilization problems where the system must operate indefinitely around an equilibrium, the dynamics, constraints, and stage cost are usually time-invariant, so we drop their index: $\mathbf f_k=\mathbf f$, $\mathbf g_k=\mathbf g$, and $c_k=c$. The true objective is:

$$
J_\infty = \sum_{k=0}^{\infty} c(\mathbf{x}_k, \mathbf{u}_k)
$$

Since this cannot be solved directly, MPC approximates it with:

$$
\begin{aligned}
\text{minimize} \quad & c_N(\mathbf{x}_N) + \sum_{k=0}^{N-1} c(\mathbf{x}_k, \mathbf{u}_k) \\
\text{subject to} \quad & \mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k) \\
& \mathbf{x}_N \in \mathcal{X}_f \\
& \text{other constraints}
\end{aligned}
$$

The terminal cost $c_N(\mathbf{x}_N)$ approximates $\sum_{k=N}^{\infty} c(\mathbf{x}_k, \mathbf{u}_k)$, the cost-to-go beyond the horizon. The terminal constraint $\mathbf{x}_N \in \mathcal{X}_f$ ensures the state reaches a region where a known stabilizing controller exists. Without these terminal ingredients, the finite-horizon approximation may produce unstable behavior, as the controller ignores consequences beyond the horizon.

#### Finite-Duration Tasks

For a task ending at time $T$, the true objective from the current time $t$ is the remaining part of the Bolza objective, written in absolute time:

$$
J_{[t, T]} = c_T(\mathbf{x}_T) + \sum_{s=t}^{T-1} c_s(\mathbf{x}_s, \mathbf{u}_s)
$$

The MPC formulation must adapt as time progresses. While the end of the task lies beyond the prediction horizon, the subproblem keeps the surrogate terminal cost $c_N$. Once $t+N\geq T$, the horizon shrinks to the $T-t$ remaining steps:

$$
\text{minimize} \quad
\begin{cases}
c_N(\mathbf{x}_N) + \displaystyle\sum_{k=0}^{N-1} c_k(\mathbf{x}_k, \mathbf{u}_k), & t+N < T, \\[2ex]
c_T(\mathbf{x}_{T-t}) + \displaystyle\sum_{k=0}^{T-t-1} c_k(\mathbf{x}_k, \mathbf{u}_k), & t+N \geq T.
\end{cases}
$$

In the second case, $\mathbf{x}_{T-t}$ is the prediction of the final state $\mathbf{x}_T$, so the task's own terminal cost applies. As the task approaches completion, the horizon shrinks and the terminal cost switches from the approximation $c_N$ to the true final cost $c_T$. This prevents the controller from optimizing beyond task completion, which would produce meaningless or aggressive control actions.

#### Periodic Tasks

Some systems operate on repeating cycles where the optimal behavior depends on the time of day, week, or season. Consider a commercial building where heating costs are higher at night, electricity prices vary hourly, and occupancy patterns repeat daily. The MPC controller must account for these periodic patterns while planning over a finite horizon.

For tasks with period $T_p$, such as daily building operations, the formulation accounts for transitions across period boundaries:

$$
\begin{aligned}
\text{minimize} \quad & \sum_{k=0}^{N-1} c_k(\mathbf{x}_k, \mathbf{u}_k) \\
\text{where} \quad & \varphi_k = (t + k) \mod T_p \\
& c_k = \begin{cases}
c_{\text{day}} & \text{if } \varphi_k \in [6\text{am}, 6\text{pm}] \\
c_{\text{night}} & \text{otherwise}
\end{cases}
\end{aligned}
$$

The stage cost $c_k$ changes with the phase $\varphi_k$ of the predicted time $t+k$ within the period. Constraints may similarly depend on the phase, reflecting different operational requirements at different times.

### The MPC Algorithm

The complete MPC procedure implements the receding horizon principle through repeated optimization:

````{prf:algorithm} Model Predictive Control with Horizon Management
:label: alg-mpc-complete

**Input:**
- Nominal prediction horizon $N$
- Sampling period $\Delta t$
- Task type: {infinite, finite ending at time $T$, periodic with period $T_p$}
- Cost functions and dynamics
- Constraints

**Procedure:**

1. Initialize time $t \leftarrow 1$
2. Measure initial state $\mathbf{x}_{\text{current}}$

3. **While** task continues:

   4. **Determine effective horizon and costs:**
      - If infinite task: 
        - $N_t \leftarrow N$
        - Use terminal cost $c_N$ and constraint $\mathcal{X}_f$
      - If finite task:
        - $N_t \leftarrow \min(N, T - t)$
        - If $t + N_t = T$: use final cost $c_T$
        - Otherwise: use approximation $c_N$
      - If periodic task:
        - $N_t \leftarrow N$
        - Adjust costs/constraints based on phase

   5. **Solve optimization:**
      Minimize over $\mathbf{u}_{0:N_t-1}$ subject to dynamics, constraints, and $\mathbf{x}_0 = \mathbf{x}_{\text{current}}$

   6. **Apply receding horizon control:**
      - Extract $\mathbf{u}^\star_0$ from solution
      - Apply to system for duration $\Delta t$
      - Measure new state $\mathbf{x}_{\text{current}}$
      - Advance time: $t \leftarrow t + 1$

7. **End While**
````

### Successive Linearization and Quadratic Approximations

The [iLQR and DDP chapter](iterative-trajectory-optimization.md) develops
successive approximation as a way to improve one finite-horizon plan. Such a
solver can also be used inside MPC: solve from the newly observed state,
execute the first action, and repeat. Its backward-pass feedback gains and
the outer decision to replan are distinct operations.

For many regulation and tracking problems, the nonlinear dynamics and costs we encounter can be approximated locally by linear and quadratic functions. The basic idea is to linearize the system around the current operating point and approximate the cost with a quadratic form. This reduces each MPC subproblem to a **quadratic program (QP)**, which can be solved reliably and very quickly using standard solvers.

Suppose the true dynamics are nonlinear,

$$
\mathbf{x}_{k+1} = \mathbf{f}_k(\mathbf{x}_k,\mathbf{u}_k).
$$

Around a nominal trajectory $(\bar{\mathbf{x}}_k,\bar{\mathbf{u}}_k)$, we take a first-order expansion:

$$
\mathbf{x}_{k+1} \approx \mathbf{f}_k(\bar{\mathbf{x}}_k,\bar{\mathbf{u}}_k) 
+ A_k(\mathbf{x}_k - \bar{\mathbf{x}}_k) 
+ B_k(\mathbf{u}_k - \bar{\mathbf{u}}_k),
$$

with the same Jacobians as in the iLQR derivation,

$$
A_k = \left.\frac{\partial \mathbf{f}_k}{\partial \mathbf{x}}\right|_{(\bar{\mathbf{x}}_k,\bar{\mathbf{u}}_k)}, 
\qquad
B_k = \left.\frac{\partial \mathbf{f}_k}{\partial \mathbf{u}}\right|_{(\bar{\mathbf{x}}_k,\bar{\mathbf{u}}_k)}.
$$

Similarly, if the stage cost is nonlinear,

$$
c_k(\mathbf{x}_k,\mathbf{u}_k),
$$

we approximate it quadratically near the nominal point:

$$
c_k(\mathbf{x}_k,\mathbf{u}_k) \;\approx\; 
\|\mathbf{x}_k - \mathbf{x}_k^{\mathrm{ref}}\|_{Q_k}^2 
+ \|\mathbf{u}_k - \mathbf{u}_k^{\mathrm{ref}}\|_{R_k}^2,
$$

with positive semidefinite weighting matrices $Q_k$ and $R_k$, where $\|\mathbf{v}\|_{Q}^2=\mathbf{v}^\top Q\,\mathbf{v}$.

The resulting MPC subproblem has the form

$$
\begin{aligned}
\min_{\mathbf{x}_{0:N},\mathbf{u}_{0:N-1}} \quad &
\|\mathbf{x}_N - \mathbf{x}_N^{\mathrm{ref}}\|_{P}^2
+ \sum_{k=0}^{N-1} 
\left(
\|\mathbf{x}_k - \mathbf{x}_k^{\mathrm{ref}}\|_{Q_k}^2
+ \|\mathbf{u}_k - \mathbf{u}_k^{\mathrm{ref}}\|_{R_k}^2
\right) \\
\text{s.t.} \quad &
\mathbf{x}_{k+1} = A_k \mathbf{x}_k + B_k \mathbf{u}_k + \mathbf{d}_k, \\
& \mathbf{u}_{\min} \leq \mathbf{u}_k \leq \mathbf{u}_{\max}, \\
& \mathbf{x}_{\min} \leq \mathbf{x}_k \leq \mathbf{x}_{\max}, \\
& \mathbf{x}_0 = \mathbf{x}_{\text{current}} ,
\end{aligned}
$$

where $\mathbf{d}_k = \mathbf{f}_k(\bar{\mathbf{x}}_k,\bar{\mathbf{u}}_k) - A_k \bar{\mathbf{x}}_k - B_k \bar{\mathbf{u}}_k$ captures the local affine offset. The terminal weight $P$ plays the role of the terminal cost $c_N$.

Because the dynamics are now linear and the cost quadratic, this optimization problem is a convex quadratic program. Quadratic programs are attractive in practice: they can be solved at kilohertz rates with mature numerical methods, making them the backbone of many real-time MPC implementations.

At each MPC step, the controller updates its linearization around the new operating point, constructs the local QP, and solves it. The process repeats, with the linear model and quadratic cost refreshed at every reoptimization. Despite the approximation, this yields a closed-loop controller that inherits the fast computation of QPs while retaining the ability to track trajectories of the underlying nonlinear system.

## Theoretical Guarantees

Repeated optimization creates feedback, but which terminal ingredients make
feasibility persist and the closed-loop state converge?

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

## Summary and Outlook

Receding-horizon control turns a finite-horizon optimizer into feedback by
reinitializing it from each measured state and applying only the first planned
action. Terminal costs, terminal sets, and invariant local controllers connect
the truncated problem to recursive feasibility and stability.

The basic loop leaves several design choices unresolved. How should the same
replanning mechanism represent tracking, economic objectives, uncertainty,
hybrid decisions, solver failures, and hard real-time deadlines? [MPC variants
and reliable operation](mpc-variants-reliability.md) organize those choices.

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
