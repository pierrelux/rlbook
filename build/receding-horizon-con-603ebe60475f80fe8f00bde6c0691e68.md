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

The trajectory optimization methods presented so far compute a complete control trajectory from an initial state to a final time or state. Once computed, this trajectory is executed without modification, making these methods fundamentally open-loop. The control function, $\mathbf{u}[k]$ in discrete time or $\mathbf{u}(t)$ in continuous time, depends only on the clock, reading off precomputed values from memory or interpolating between them. This approach assumes perfect models and no disturbances. Under these idealized conditions, repeating the same control sequence from the same initial state would always produce identical results.

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
