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
<!-- 
### Computational Considerations

The receding horizon principle requires solving optimization problems in real-time, placing stringent demands on the solver. Each problem must be solved within the sampling period $\Delta t$. If the solver requires more time, the system operates without new control updates, potentially degrading performance or stability.

Fortunately, successive MPC problems differ only in their initial conditions and possibly their horizons. This similarity enables warm-starting strategies where the previous solution initializes the current optimization. The standard approach shifts the previous trajectory forward by one time step and appends a nominal control at the end. This initialization typically lies close to the new optimum, dramatically reducing iteration counts.

The computational burden also depends on the horizon length $N$. Longer horizons provide better approximations to infinite-horizon problems and enable more sophisticated maneuvers, but increase problem size. The choice of $N$ balances solution quality against computational resources. For linear systems with quadratic costs, horizons of 10-50 steps often suffice. Nonlinear systems may require longer horizons to capture essential dynamics, though move-blocking and other parameterization techniques can reduce the effective number of decision variables. -->

<!-- ### Connection to Dynamic Programming

The receding horizon principle connects MPC to the dynamic programming framework covered in the next chapter. Each MPC optimization implicitly computes the optimal cost-to-go $V_N(\mathbf{x})$ from the current state over the horizon. This finite-horizon value function approximates the true infinite-horizon value function that dynamic programming seeks globally.

The connection becomes clearer when we consider what MPC actually does: it solves a finite-horizon optimization problem and extracts only the first control action. The remaining $N-1$ steps of the optimal trajectory are discarded, but the terminal cost $V_f$ approximates the value function at the horizon boundary. This suggests hybrid approaches where approximate value functions from dynamic programming provide terminal costs for MPC, combining global optimality properties with local constraint handling capabilities.

This idea is what we would refer to as **bootstrapping** when working with temporal difference learning methods in reinforcement learning. In temporal difference methods like Q-learning or SARSA, bootstrapping occurs when we use our current estimate of the value function to update itself—essentially "pulling ourselves up by our bootstraps." Similarly, MPC bootstraps by using its finite-horizon value function approximation (computed through optimization) to make decisions, even though this approximation may not be perfect. The terminal cost $V_f$ acts as a bootstrap target, providing a value estimate for states beyond the horizon that guides the optimization process. 
 -->

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
<!-- 
### Why Terminal Conditions Work

Understanding why the terminal conditions guarantee recursive feasibility and asymptotic stability requires examining what the controller actually does from one step to the next. Suppose at time $k$ the MPC optimizer finds an optimal sequence of controls $(\mathbf{u}_0^*, \ldots, \mathbf{u}_{N-1}^*)$ and states $(\mathbf{x}_0^*, \ldots, \mathbf{x}_N^*)$, where $\mathbf{x}_N^* \in \mathcal{X}_f$. The first control $\mathbf{u}_0^*$ is applied to the system, and the remaining plan is discarded according to the receding horizon principle.

At time $k+1$, we need a new plan starting from the updated state $\mathbf{x}_{\text{new}} = \mathbf{x}_1^*$. A natural fallback strategy is to **shift** the previous plan forward by one step and **append the terminal controller** $\boldsymbol{\kappa}_f$ at the end, yielding controls $(\mathbf{u}_1^*, \ldots, \mathbf{u}_{N-1}^*, \boldsymbol{\kappa}_f(\mathbf{x}_N^*))$ with states recomputed from the dynamics starting from $\mathbf{x}_1^*$.

This shifted plan may no longer be optimal, but the **Lyapunov decrease condition** ensures it remains feasible and leads to progress. The condition

$$
V_f(\mathbf{f}(\mathbf{x}, \kappa_f(\mathbf{x}))) - V_f(\mathbf{x}) \leq -\ell(\mathbf{x}, \kappa_f(\mathbf{x}))
\quad \forall\, \mathbf{x} \in \mathcal{X}_f
$$

requires that the terminal cost $V_f$ decrease faster than the stage cost accumulates when following $\boldsymbol{\kappa}_f$ inside $\mathcal{X}_f$. This means the controller makes progress in both state evolution and predicted cost-to-go.

This "one-step contractiveness" property ensures that applying $\kappa_f$ within the terminal set leads to value decrease. When used at the horizon's tail, this guarantees that even a non-optimal shifted trajectory results in lower overall cost, making $V_N$ behave like a Lyapunov function for regulation and tracking tasks. -->

<!-- ### Computing Terminal Conditions in Practice

For linear systems with quadratic costs, the terminal conditions follow naturally from LQR theory. The process begins by solving the infinite-horizon LQR problem:

$$\mathbf{P} = \mathbf{Q} + \mathbf{A}^T \mathbf{P} \mathbf{A} - \mathbf{A}^T \mathbf{P} \mathbf{B}(\mathbf{R} + \mathbf{B}^T \mathbf{P} \mathbf{B})^{-1} \mathbf{B}^T \mathbf{P} \mathbf{A}$$
$$\mathbf{K} = -(\mathbf{R} + \mathbf{B}^T \mathbf{P} \mathbf{B})^{-1} \mathbf{B}^T \mathbf{P} \mathbf{A}$$

The terminal cost and controller then follow directly: $V_f(\mathbf{x}) = \mathbf{x}^T \mathbf{P} \mathbf{x}$ and $\kappa_f(\mathbf{x}) = \mathbf{K}\mathbf{x}$. 

Constructing the terminal set $\mathcal{X}_f$ presents more options with varying computational complexity. The most powerful but computationally intensive approach computes the **maximal control-invariant set**: the largest set where $\mathbf{u} = \mathbf{K}\mathbf{x}$ keeps the state feasible indefinitely. This involves fixed-point iterations on polytopes. A more tractable alternative uses **ellipsoidal approximations**, finding the largest $\alpha$ such that $\mathcal{X}_f = \{\mathbf{x} : \mathbf{x}^T \mathbf{P} \mathbf{x} \leq \alpha\}$ satisfies all constraints under $\mathbf{u} = \mathbf{K}\mathbf{x}$. The most conservative but always feasible approach starts with a small safe set where constraints are satisfied and grows it until hitting constraint boundaries.

For nonlinear systems, we linearize around the equilibrium to compute $\mathbf{P}$ and $\mathbf{K}$, then verify the decrease condition holds locally. The terminal set becomes a neighborhood where the linear approximation remains valid. -->


<!-- 
### Performance Implications

The terminal conditions create inherent tradeoffs between conservatism and computational burden. Larger terminal sets $\mathcal{X}_f$ provide greater regions of attraction and impose fewer restrictions on trajectories, but require more intensive computation. Smaller terminal sets may necessitate longer horizons to reach from typical initial conditions. Similarly, more accurate terminal costs $V_f$ provide tighter approximations of infinite-horizon costs, enabling effective control with shorter horizons.

The importance of terminal conditions diminishes as the horizon length increases. With $N \to \infty$, any stabilizing terminal controller suffices for stability. In practice, short horizons (N = 10-20) make terminal conditions crucial for stability, medium horizons (N = 20-50) benefit from but don't critically depend on them, while long horizons (N > 50) often omit them entirely, relying solely on horizon length for stability. -->

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
<!-- 
#### Exponential Convergence in the Linear-Quadratic Case

For linear systems $\mathbf{x}_{k+1} = \mathbf{A}\mathbf{x}_k + \mathbf{B}\mathbf{u}_k$ with quadratic costs $\ell(\mathbf{x}, \mathbf{u}) = \mathbf{x}^T\mathbf{Q}\mathbf{x} + \mathbf{u}^T\mathbf{R}\mathbf{u}$, we can compute this error exactly. When the terminal cost is the LQR cost-to-go $V_f(\mathbf{x}) = \mathbf{x}^T\mathbf{P}\mathbf{x}$, the infinite-horizon optimal trajectory satisfies $\mathbf{x}_k^* = \mathbf{A}_{cl}^k \mathbf{x}$ where $\mathbf{A}_{cl} = \mathbf{A} + \mathbf{B}\mathbf{K}$ is the closed-loop matrix.

The tail cost from time $N$ onward becomes:
$\sum_{k=N}^{\infty} \ell(\mathbf{x}_k^*, \mathbf{u}_k^*) = \sum_{k=N}^{\infty} (\mathbf{x}_k^*)^T \mathbf{Q}_{cl} \mathbf{x}_k^* = (\mathbf{x}_N^*)^T \left(\sum_{k=0}^{\infty} (\mathbf{A}_{cl}^T)^k \mathbf{Q}_{cl} \mathbf{A}_{cl}^k\right) \mathbf{x}_N^*$

where $\mathbf{Q}_{cl} = \mathbf{Q} + \mathbf{K}^T\mathbf{R}\mathbf{K}$ captures the quadratic cost under the optimal controller $\mathbf{u} = \mathbf{K}\mathbf{x}$.

The infinite sum equals $\mathbf{P}$ by definition of the LQR solution, so the terminal cost $V_f(\mathbf{x}_N^*) = (\mathbf{x}_N^*)^T \mathbf{P} \mathbf{x}_N^*$ exactly matches the true tail cost when computed along the infinite-horizon optimal trajectory. This gives $\varepsilon_N = 0$ when following the infinite-horizon optimal path exactly!

However, the finite-horizon optimizer typically finds a different trajectory for the first $N$ steps, leading to a different $\mathbf{x}_N$. The approximation error then depends on how much the finite-horizon trajectory deviates from the infinite-horizon one. Since both are optimal for their respective problems and the terminal cost provides the correct tail approximation, this deviation shrinks exponentially with horizon length at a rate determined by the eigenvalues of $\mathbf{A}_{cl}$.

Specifically, if $\rho(\mathbf{A}_{cl})$ denotes the spectral radius of the closed-loop matrix, then:

$\varepsilon_N = O(\rho(\mathbf{A}_{cl})^N)$

For stable systems, $\rho(\mathbf{A}_{cl}) < 1$, yielding exponential convergence. This explains why short horizons (N = 10-30) often achieve near-optimal performance: the approximation error decreases exponentially fast, making even modest horizons highly effective for regulation and constant reference tracking tasks.

#### Implications for Horizon Selection

These bounds provide practical guidance for choosing prediction horizons. The exponential convergence means that beyond a certain horizon length, further increases yield diminishing returns. The optimal horizon balances approximation accuracy against computational cost, with the break-even point typically occurring when $\rho(\mathbf{A}_{cl})^N$ drops below the desired tolerance level.

For systems with slow dynamics (eigenvalues close to one), longer horizons may be necessary, while systems with fast dynamics achieve good approximations with surprisingly short horizons. This analysis also explains why terminal conditions become less critical as horizons increase: the exponential decay ensures that the tail beyond any reasonable horizon contributes negligibly to the total cost. -->
<!-- 
### When Terminal Constraints Cause Infeasibility

The terminal constraint $\mathbf{x}_N \in \mathcal{X}_f$ can make the optimization infeasible, especially for:
- Large disturbances pushing the state far from equilibrium
- Short horizons that cannot reach $\mathcal{X}_f$ in time
- Conservative terminal sets that are unnecessarily small

Common remedies:

1. **Soft terminal constraints**: Replace hard constraint with penalty
   $$\text{minimize} \quad V_f(\mathbf{x}_N) + \rho \cdot d(\mathbf{x}_N, \mathcal{X}_f) + \ldots$$
   where $d(\cdot, \mathcal{X}_f)$ measures distance to the set

2. **Adaptive horizons**: Extend horizon when far from $\mathcal{X}_f$

3. **Backup strategy**: If infeasible, switch to unconstrained MPC or a fallback controller, then re-enable terminal constraints once feasible

The choice depends on whether theoretical guarantees or practical performance takes priority. Many industrial implementations omit terminal constraints entirely, relying on well-tuned horizons and costs to ensure stability.
 -->
<!-- 
# The Landscape of MPC Variants

Once the basic idea of receding horizon control is clear, it is helpful to see how the same backbone accommodates many variations. In every case, we transcribe the continuous problem to a nonlinear program of the form

$$
\begin{aligned}
\text{minimize}\quad & c_T(\mathbf{x}_N)+\sum_{k=0}^{N-1} w_k\,c(\mathbf{x}_k,\mathbf{u}_k) \\
\text{subject to}\quad & \mathbf{x}_{k+1}=\mathbf{F}_k(\mathbf{x}_k,\mathbf{u}_k) \\
& \mathbf{g}(\mathbf{x}_k,\mathbf{u}_k)\leq \mathbf{0}, \\
& \mathbf{x}_{\min}\leq \mathbf{x}_k\leq \mathbf{x}_{\max},\quad \mathbf{u}_{\min}\leq \mathbf{u}_k\leq \mathbf{u}_{\max}, \\
& \mathbf{x}_0=\mathbf{x}_{\text{current}} ,
\end{aligned}
$$

with \$\mathbf{F}\_k\$ the chosen discretization of the dynamics and \$w\_k\$ the quadrature weights. From this skeleton, several families of MPC emerge.

In **tracking MPC**, the stage and terminal costs are quadratic penalties on deviation from a reference trajectory,

$$
c(\mathbf{x}_k,\mathbf{u}_k)=\|\mathbf{x}_k-\mathbf{x}_k^{\text{ref}}\|_{\mathbf{Q}}^2+\|\mathbf{u}_k-\mathbf{u}_k^{\text{ref}}\|_{\mathbf{R}}^2 , \qquad c_T(\mathbf{x}_N)=\|\mathbf{x}_N-\mathbf{x}_N^{\text{ref}}\|_{\mathbf{P}}^2 .
$$

This is the industrial workhorse, ensuring that the system follows a desired profile within limits.

In **regulatory MPC**, the reference is fixed at an equilibrium \$(\mathbf{x}^e,\mathbf{u}^e)\$, and the quadratic penalty encourages return to this point. Terminal constraints are often added so that stability can be guaranteed.

In **economic MPC**, the quadratic structure disappears altogether. Instead, the cost encodes economic performance,

$$
c(\mathbf{x}_k,\mathbf{u}_k)=c_{\text{econ}}(\mathbf{x}_k,\mathbf{u}_k),
$$

for instance energy cost, profit, or resource efficiency. The optimization then steers the system not toward a setpoint but toward economically optimal regimes.

When uncertainty is represented by bounded sets, one arrives at **robust MPC**, which seeks controls that satisfy the constraints for all admissible disturbances. The resulting NLP has a min–max structure. A tractable alternative is tube MPC, where the nominal optimization is carried out with tightened constraints to guarantee feasibility of the true system under a disturbance feedback law.

If uncertainty is stochastic, the formulation turns into **stochastic MPC**, where the objective is the expected cost and the constraints are imposed with high probability,

$$
\mathbb{P}\!\left[\mathbf{g}(\mathbf{x}_k,\mathbf{u}_k)\leq \mathbf{0}\right]\geq 1-\varepsilon .
$$

Scenario-based versions replace expectations by a sampled, deterministic problem.

Some systems require discrete choices, such as switching devices on or off. **Hybrid MPC** introduces integer variables into the transcription, producing a mixed-integer NLP that can handle such logic.

For large networks of subsystems, **distributed MPC** coordinates several local predictive controllers that optimize their own subsystems while communicating through coupling constraints.

Finally, in settings where models are uncertain or slowly drifting, one finds **adaptive or learning-based MPC**, which uses parameter estimation or machine learning to update the model \$\mathbf{F}\_k(\cdot;\theta\_t)\$ and possibly the cost function. The optimization step remains the same, but the model evolves as more data are collected.

These formulations illustrate that MPC is less a single algorithm than a recipe. The scaffolding is always the same: finite horizon prediction, state and input constraints, and receding horizon application of the control. What changes from one variant to another is the cost function, the way uncertainty is treated, the presence of discrete decisions, or the architecture across multiple agents. -->


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

Not all systems operate around a reference. Sometimes the goal is to optimize a true economic objective (eg. energy cost, revenue, efficiency) directly. This gives rise to **economic MPC**, where the cost functions reflect real operational performance:

$$
c(\mathbf{x}_k, \mathbf{u}_k) = c_{\text{op}}(\mathbf{x}_k, \mathbf{u}_k), \qquad
c(\mathbf{x}_N) = c_{\text{op},T}(\mathbf{x}_N) \enspace .
$$

There is no reference trajectory here. The optimal behavior emerges from the cost itself. In this setting, standard stability arguments no longer apply automatically, and one must be careful to add terminal penalties or constraints that ensure the closed-loop system remains well-behaved.

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


# Robustness in Real-Time MPC

The trajectory optimization methods we have studied assume perfect models and deterministic dynamics. In practice, however, MPC controllers must operate in environments where models are approximate, disturbances are unpredictable, and computational resources are limited. The mathematical elegance of optimal control must always yield to the engineering reality of robust operation as **perfect optimality is less important than reliable operation**. This philosophy permeates industrial MPC applications. A controller that achieves 95% performance 100% of the time is superior to one that achieves 100% performance 95% of the time and fails catastrophically the remaining 5%. Airlines accept suboptimal fuel consumption over missed approaches, power grids tolerate efficiency losses to prevent blackouts, and chemical plants sacrifice yield for safety. By designing for failure, we want to to create MPC systems that degrade gracefully rather than fail catastrophically, maintaining safety and stability even when the impossible is asked of them.


## Example: Wind Farm Yield Optimization
Consider a wind farm where MPC controllers coordinate individual turbines to maximize overall power production while minimizing wake interference. Each turbine can adjust both its thrust coefficient (through blade pitch) and yaw angle to redirect its wake away from downstream turbines. At time $t_k$, the MPC controller solves the optimization problem:

$$
\begin{aligned}
\min_{\mathbf{u}_{0:N-1}} \quad & \sum_{i=0}^{N-1} \|\mathbf{x}_i - \mathbf{x}_i^{\text{ref}}\|_{\mathbf{Q}}^2 + \|\mathbf{u}_i\|_{\mathbf{R}}^2 \\
\text{s.t.} \quad & \mathbf{x}_{i+1} = \mathbf{f}(\mathbf{x}_i, \mathbf{u}_i) \\
& \mathbf{x}_i \in \mathcal{X}_{\text{safe}} \\
& \|\mathbf{u}_i\|_\infty \leq u_{\max} \\
& \mathbf{x}_0 = \mathbf{x}_{\text{current}}
\end{aligned}
$$

Now suppose an unexpected wind direction change occurs, shifting the incoming wind vector by 30 degrees. The current state $\mathbf{x}_{\text{current}}$ reflects wake patterns that no longer align with the new wind direction, and the optimizer discovers that no feasible trajectory exists that can redirect all wakes appropriately within the physical limits of yaw rate and thrust adjustment. The solver reports infeasibility.

This scenario reveals the fundamental challenge of real-time MPC: **constraint incompatibility**. When disturbances push the system into states from which recovery appears impossible, or when reference trajectories demand physically impossible maneuvers, the intersection of all constraint sets becomes empty. Model mismatch compounds this problem as prediction errors accumulate over the horizon.

Even when feasible solutions exist, **computational constraints** can prevent their discovery. A control loop running at 100 Hz allows only 10 milliseconds per iteration. If the solver requires 15 milliseconds to converge, we face an impossible choice: delay the control action and risk destabilizing the system, or apply an unconverged iterate that may violate critical constraints.

A third failure mode involves **numerical instabilities**: ill-conditioned matrices, rank deficiency, or division by zero in the linear algebra routines. These failures are particularly problematic because they occur sporadically, triggered by specific state configurations that create near-singular conditions in the optimization problem.

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

## Example: Chemical Reactor Control Under Failure

Consider a continuous stirred tank reactor (CSTR) where an exothermic reaction must be controlled:

$$
\begin{aligned}
\dot{C}_A &= \frac{q}{V}(C_{A,in} - C_A) - k_0 e^{-E/RT} C_A \\
\dot{T} &= \frac{q}{V}(T_{in} - T) + \frac{\Delta H}{\rho c_p} k_0 e^{-E/RT} C_A - \frac{UA}{\rho c_p V}(T - T_c)
\end{aligned}
$$

The MPC must maintain temperature below the runaway threshold $T_{\text{runaway}}$ while maximizing conversion. Under normal operation, it solves:

$$
\begin{aligned}
\min \quad & -C_A(t_f) + \int_0^{t_f} \|T - T_{\text{optimal}}\|^2 dt \\
\text{s.t.} \quad & T \leq T_{\text{runaway}} - \Delta T_{\text{safety}} \\
& q_{\min} \leq q \leq q_{\max}
\end{aligned}
$$

When the cooling system partially fails, $T_c$ suddenly increases. The MPC cannot maintain $T_{\text{optimal}}$ within safety limits. The cascade activates: soft constraints allow $T$ to exceed $T_{\text{optimal}}$ with penalty, the reference governor reduces the production target $C_{A,\text{target}}$, and if still infeasible, the backup controller switches to maximum cooling $q = q_{\max}$. If temperature approaches runaway, emergency shutdown stops the feed with $q = 0$.




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

In MPC, this yields efficient **warm starts** across time: as the parameter $\theta_t$ (current state, references) changes slightly, we predict the new primal–dual point and correct with 1–2 iterations—often enough to hit tolerance in real time.

<!-- 
## Application to MPC

We now specialize this idea to the structure of finite-horizon MPC. Fix a prediction horizon $N$. At each time step, we solve a problem with fixed structure and varying data. Define

$$
\boldsymbol{\theta} := (\mathbf{x}_0,\, \mathbf{r},\, \mathbf{w}),
$$

which includes the current state $\mathbf{x}_0$, reference signals $\mathbf{r}$, and exogenous forecasts $\mathbf{w}$.

The finite-horizon problem becomes

$$
\begin{aligned}
\min_{z} \quad & J(z;\boldsymbol{\theta}) \\
\text{s.t.} \quad & c(z;\boldsymbol{\theta}) = 0 \\
& d(z;\boldsymbol{\theta}) \leq 0,
\end{aligned}
$$

with decision variable $z = (\mathbf{x}_{0:N}, \mathbf{u}_{0:N-1})$. The equality constraints enforce dynamics and terminal conditions. The inequalities encode input and state bounds.

Solving $(P_\theta)$ produces an optimal trajectory $z^\star(\boldsymbol{\theta})$. The control law is the first input:

$$
\pi(\boldsymbol{\theta}) := \mathbf{u}_0^\star(\boldsymbol{\theta}).
$$

This mapping from parameter to input defines the MPC policy. Parametric programming helps us understand and exploit its structure to speed up evaluation.

## Two Approaches to Efficient MPC

Parametric structure can be used in two main ways: either to construct an explicit control law offline, or to warm-start the optimizer online using sensitivity information.

### Explicit MPC

When the problem is a linear or quadratic program with affine dependence on $\boldsymbol{\theta}$, we can work out the solution symbolically. The parameter space is partitioned into regions $\mathcal{R}_1, \dots, \mathcal{R}_M$, each associated with a fixed active set. On each region:

$$
z^\star(\boldsymbol{\theta}) = A_r\,\boldsymbol{\theta} + b_r,
\qquad
\pi(\boldsymbol{\theta}) = K_r\,\boldsymbol{\theta} + k_r.
$$

At runtime, we identify which region contains $\boldsymbol{\theta}$, then apply the corresponding affine formula. This approach avoids optimization entirely during deployment.

It requires storing the region definitions and control laws. The number of regions grows with horizon length and constraint count, which limits this approach to systems with low state dimension and short horizons.

### Sensitivity-Based MPC

When symbolic enumeration is intractable, we can still track how the solution varies locally. Suppose we have a solution $\bar{y} = (\bar{z}, \bar{\lambda}, \bar{\nu})$ at parameter $\bar{\boldsymbol{\theta}}$. The KKT system reads:

$$
F(y; \boldsymbol{\theta}) = 0.
$$

Differentiating with respect to $\boldsymbol{\theta}$,

$$
\frac{\partial F}{\partial y}(\bar{y}; \bar{\boldsymbol{\theta}})\, \mathrm{d}y
= - \frac{\partial F}{\partial \boldsymbol{\theta}}(\bar{y}; \bar{\boldsymbol{\theta}})\, \mathrm{d}\boldsymbol{\theta}.
$$

Let $K$ be the KKT matrix and $G$ the sensitivity of the residual. Then the sensitivity operator $T$ satisfies

$$
K T = -G \quad \Rightarrow \quad \mathrm{d}y = T\, \mathrm{d}\boldsymbol{\theta}.
$$

If $\boldsymbol{\theta}$ changes slightly, we update the primal-dual pair:

$$
y^{(0)} \leftarrow \bar{y} + T\,\Delta\boldsymbol{\theta},
$$

and use it as the starting point for Newton or SQP.

This is the basis of real-time iteration schemes. When the active set is stable, the warm start is accurate to first order. When it changes, we refactorize and repeat, still with far less effort than solving from scratch. -->


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

```{code-cell} ipython3
:tags: [hide-input]

%run code/hypnosis_control_nmpc.py
```

<!-- ### Deployment Patterns

There are several ways to use an amortized controller once it has been trained. The simplest option is **direct amortization**, where the control input is taken to be $u = \hat{\pi}_\phi(\boldsymbol{\theta})$. In this case, the neural network provides the control action directly, with no optimization performed during deployment.

A second option is **amortization with projection**, where the network output $\tilde u = \hat{\pi}_\phi(\boldsymbol{\theta})$ is passed through a small optimization step, such as a quadratic program or barrier-function filter, in order to enforce constraints. This adds a negligible computational overhead but restores guarantees of feasibility and safety.

We could for example integrate a convex approximation of the MPC subproblem directly as a differentiable layer inside the network. The network proposes a candidate action $\tilde u$, which is then corrected through a small quadratic program:

$$
u = \arg\min_v \tfrac12\|v-\tilde u\|^2 \quad \text{s.t. } g(x,v)\le 0.
$$

Gradients are propagated through this correction using implicit differentiation, allowing the network to be trained end-to-end while retaining constraint satisfaction. This hybrid keeps the fast evaluation of a learned map while preserving the structure of MPC.

A third option is **amortized warm-starting**, where the neural network provides an initialization for one or two Newton or SQP iterations of the underlying NMPC problem. In this setting, the learned map delivers an excellent starting point, so the optimizer converges quickly and the cost of re-solving at each time step is greatly reduced. -->



<!-- ## Demo: Batch Bioreactor MPC with do-mpc

We illustrate nonlinear MPC on a fed-batch bioreactor. The process has four states: biomass concentration \(X_s\), substrate \(S_s\), product \(P_s\), and liquid volume \(V_s\). The manipulated feed flow \(u_{\text{inp}}\) augments volume and changes concentrations. The dynamics are

$$
\begin{aligned}
\dot X_s &= \mu(S_s)X_s - \tfrac{u_{\text{inp}}}{V_s} X_s, \\
\dot S_s &= -\tfrac{\mu(S_s)X_s}{Y_x} - \tfrac{v X_s}{Y_p} + \tfrac{u_{\text{inp}}}{V_s}(S_{\text{in}} - S_s), \\
\dot P_s &= v X_s - \tfrac{u_{\text{inp}}}{V_s} P_s, \\
\dot V_s &= u_{\text{inp}},
\end{aligned}
$$

with inhibited Monod kinetics

$$
\mu(S_s) = \frac{\mu_m S_s}{K_m + S_s + S_s^2/K_i}.
$$

We impose bounds on states and input, e.g. \(0 \le X_s \le 3.7\), \(0 \le P_s \le 3.0\), and \(0 \le u_{\text{inp}} \le 0.2\). Two parameters are uncertain: yield \(Y_x\) and inlet concentration \(S_{\text{in}}\). We treat them via a small scenario set with non-anticipativity (a single input sequence is shared across scenarios).

At each MPC step, we solve a finite-horizon problem that encourages product formation while regularizing effort:

$$
\min_{x_{0:N},u_{0:N-1}} \; -\,P_s(N) + \sum_{k=0}^{N-1} \big( -\,P_s(k) + \rho\, u_k^2 \big)
$$

subject to the discretized dynamics and box constraints for all uncertainty scenarios, sharing the inputs across scenarios. The continuous-time ODEs are discretized by orthogonal collocation on finite elements, producing an NLP [orthogonal collocation](https://www.do-mpc.com/en/latest/theory_orthogonal_collocation.html). The resulting NMPC is re-solved in a receding-horizon loop [MPC basics](https://www.do-mpc.com/en/latest/theory_mpc.html).

The cell below runs the closed-loop simulation and plots the states and input. The script is adapted from the do-mpc Batch Bioreactor example.

```{code-cell} ipython3
:tags: [hide-input]

%run _static/do_mpc_batch_bioreactor.py
```

### Interactive Animation

The following cell creates an interactive animation of the batch bioreactor control process, showing the MPC predictions and the evolution of the system states in real-time. The visualization includes a tank representation with liquid level and biomass particles, along with time-series plots of all states and control inputs.

```{code-cell} ipython3
:tags: [hide-input]

%run _static/do_mpc_batch_bioreactor_animated.py
``` -->