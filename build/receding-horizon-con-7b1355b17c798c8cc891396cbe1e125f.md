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

## Learning Goals

After working through the derivations and the balancing example, you should
be able to:

- Turn a finite-horizon optimal-control problem into a state-feedback law by
  replanning, and state the feedback law and its domain precisely.
- Explain why a short horizon without terminal ingredients can leave an
  unstable system uncontrolled, stall it near a constraint, or lose
  feasibility.
- Prove recursive feasibility and closed-loop convergence from a terminal
  cost, a terminal set, and a local controller, using the shifted candidate
  sequence.
- Construct a quadratic terminal cost and an ellipsoidal terminal set for a
  linear system with box constraints, and check both numerically.
- Decide when the finite-horizon value is a lower bound, an upper bound, or
  neither, for the infinite-horizon value.

## Prerequisites

The chapter uses the [finite-horizon optimal-control formulation](discrete-time-optimal-control.md),
the [transcription methods](numerical-trajectory-optimization.md) that turn it
into a nonlinear program, and the receding-horizon execution used by
[MPPI](model-predictive-path-integral-control.md#sec-mppi). The worked example
solves small convex quadratic programs; only the facts that such programs have
a unique solution and can be solved reliably are needed. Dynamic programming
and the linear-quadratic regulator are not required. Where the example uses an
LQR gain, the property it relies on is derived here.

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

### A Balancing Example

Balancing a stick upright on a fingertip shows what replanning buys. The
upright position is an equilibrium, but a small tilt grows on its own, so a
precomputed command sequence that is slightly wrong does not stay slightly
wrong. Linearized about the upright position, the angle $\theta$ obeys

$$
\ddot\theta = \frac{g}{\ell}\,\theta + u ,
$$

where $\ell$ is the stick length and $u$ is the angular acceleration commanded
through the fingertip. With $g=9.81$ m/s$^2$ and $\ell=1$ m, a tilt grows by a
factor $e$ about every $\sqrt{\ell/g}\approx 0.32$ s. We sample the model
exactly under a zero-order hold with period $\Delta t=0.05$ s, which gives a
linear model $\mathbf x_{k+1}=A\mathbf x_k+B u_k$ with
$\mathbf x=(\theta,\dot\theta)$. The command is limited to $|u_k|\le 5$
rad/s$^2$, and the angle must stay within $|\theta_k|\le 0.35$ rad. The stage
cost is quadratic,

$$
c(\mathbf x,u)=\mathbf x^\top Q\mathbf x+R u^2,\qquad Q=\operatorname{diag}(10,1),\quad R=0.05 .
$$

Each MPC subproblem is therefore a convex quadratic program in the $N$
commands and $N+1$ predicted states. The terminal cost $\mathbf x^\top P\mathbf x$
and the terminal set $\{\mathbf x:\mathbf x^\top P\mathbf x\le\alpha\}$ used in
this section are constructed in
[Constructing Terminal Ingredients](#constructing-terminal-ingredients-for-linear-systems).
For now, treat them as given.

The experiment starts from $\mathbf x_0=(0.2,\,0.6)$: tilted and falling
further. A plan with horizon $N=40$ (two seconds) is computed once from the
nominal model. Two executions of that plan are then compared on a plant whose
stick is 15% shorter than the model assumes, and which receives a tap on its
angular rate of $0.5$ rad/s at $t=1$ s. The first execution applies the
40 planned commands open loop. The second solves the same problem again from
every measured state and applies only the first command.

Before looking at the result, predict which effect hurts the open-loop
execution first: the shorter stick or the tap.

:::{figure} _static/receding_horizon/replanning.svg
:label: fig-rhc-replanning
:width: 100%

Executing one plan open loop against replanning at every step, on a plant
whose stick is 15% shorter than the model. The dashed curve is the plan
applied to the nominal model. Dotted lines mark the angle and command limits.
The open-loop execution leaves the angle limit at $t=0.85$ s, before the tap.
Replanning saturates the command during the first $0.4$ s, absorbs the tap,
and keeps $|\theta|\le 0.28$ rad.
:::

The model error alone defeats the open-loop plan. The shorter stick falls
faster than predicted, the planned commands are too small to catch it, and the
angle leaves its limit at $t=0.85$ s, before the tap arrives. The open-loop
execution never measures the state, so it cannot detect the error. The replanned controller solves the same
optimization with the same wrong model, but each new solve starts from the
measured state, so the accumulated prediction error is corrected one step at a
time. Every one of its 60 subproblems is feasible, and the tap appears only as
a second burst of command. Replanning does not repair the model. It limits how
long a model error can act before it is measured.

### The Controller Defined by Replanning

For a time-invariant problem, such as regulation of the stick, the subproblem
depends on the time $t$ only through its initial state. Write
$\mathbf u_0^\star(\mathbf x)$ for the first optimal command when the subproblem
starts from $\mathbf x$. Replanning then executes the state-feedback law

$$
\kappa_N(\mathbf x)=\mathbf u_0^\star(\mathbf x),\qquad
\mathbf x_{t+1}=\mathbf f\big(\mathbf x_t,\kappa_N(\mathbf x_t)\big).
$$

The law is defined only where the subproblem has a solution. Call this set the
**feasible set** $\mathcal X_N$: the states from which some admissible command
sequence of length $N$ satisfies every constraint and ends in the terminal set.
The law $\kappa_N$ is never written down. It is evaluated one state at a time
by an optimizer. The [parametric controllers](parametric-controllers.md)
chapter studies the map $\mathbf x\mapsto\kappa_N(\mathbf x)$ as an object in
its own right. For linear dynamics, quadratic costs, and polyhedral
constraints, it is a continuous piecewise affine function of the state
{cite}`Bemporad2002`.

:::{figure} _static/receding_horizon/feasible-sets.svg
:label: fig-rhc-feasible-sets
:width: 100%

Left: feasible sets of the stick controller with terminal cost and terminal
set, for $N=5$ (darkest), $10$, and $20$ (lightest), on a grid over the
angle limits. The ellipse is the terminal set $\mathcal X_f$ and the star is
the initial state of {numref}`fig-rhc-replanning`. Right: the first action
$\kappa_{20}$ along the line $\dot\theta=0.8$ rad/s, with the saturated LQR
law in grey. The MPC curve stops at $\theta\approx 0.205$ rad, where the
subproblem becomes infeasible.
:::

{numref}`fig-rhc-feasible-sets` shows both objects for the stick. The
feasible sets are nested, $\mathcal X_5\subset\mathcal X_{10}\subset\mathcal X_{20}$:
a longer horizon allows more steps to reach the terminal ellipse, so more
initial states qualify. On the grid they cover 35%, 43%, and 49% of the
plotted rectangle. The initial state $(0.2,0.6)$ lies outside the terminal
ellipse, and nine steps are the fewest that reach it. Outside the shaded
bands the stick is falling too fast in the direction of its tilt to be caught
by commands of at most $5$ rad/s$^2$ without violating the angle limit.

The right panel contains a lesson specific to this small example. Wherever
the subproblem is feasible, $\kappa_{20}$ agrees with the saturated linear law
$\operatorname{sat}(K\mathbf x)$ to within $4\times10^{-5}$ rad/s$^2$, where
$K$ is the LQR gain from which the terminal cost is built. For a
two-dimensional state with one input, the active constraint along optimal
paths is usually the input bound at the first step, and the optimizer
rediscovers saturation. The optimizer adds two things that the saturated law
lacks. First, it reports a state outside $\mathcal X_{20}$ as infeasible
instead of returning a command that cannot succeed. Second, the stability
argument below comes with it. With more states, more inputs, or state
constraints that become active along the way, the optimizer's law departs from
any saturated linear law.

Warm starting fits naturally here. Consecutive states are close, so the
shifted previous solution
$(\mathbf u_1^\star,\ldots,\mathbf u_{N-1}^\star,\mathbf u_{N-1}^\star)$ is a
good initial guess for the next solve. The proof of stability below uses
almost the same shifted sequence, with the last entry chosen more carefully.

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

1. Initialize time $t \leftarrow 0$
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

Any solver that improves one finite-horizon plan by successive approximation
can be used inside MPC: solve from the newly observed state, execute the first
action, and repeat. The [iLQR and DDP chapter](iterative-trajectory-optimization.md),
after dynamic programming, develops one such solver. Its backward-pass feedback
gains and the outer decision to replan are distinct operations.

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

with Jacobians

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

The terminal conditions provide different types of guarantees depending on the control objective. For regulation problems, where the task is to drive the state to a fixed equilibrium $(\mathbf{x}_\mathrm{eq}, \mathbf{u}_\mathrm{eq})$ (often shifted to the origin), the stability guarantee is **asymptotic stability**: starting sufficiently close to the equilibrium, the closed-loop state satisfies $\mathbf{x}_t \to \mathbf{x}_\mathrm{eq}$ as $t\to\infty$ while constraints remain satisfied throughout the trajectory (**recursive feasibility**). This requires the stage cost $c(\mathbf{x},\mathbf{u})$ to be positive definite in the deviation from equilibrium.

When tracking a constant setpoint, the task becomes following a constant reference $(\mathbf{x}^\mathrm{ref},\mathbf{u}^\mathrm{ref})$ that solves the steady-state equations. This problem is handled by working in **error coordinates** $\tilde{\mathbf{x}}=\mathbf{x}-\mathbf{x}^\mathrm{ref}$ and $\tilde{\mathbf{u}}=\mathbf{u}-\mathbf{u}^\mathrm{ref}$, transforming the tracking problem into a regulation problem for the error system. The stability guarantee becomes asymptotic **tracking**, meaning $\tilde{\mathbf{x}}_t \to \mathbf{0}$, again with recursive feasibility.

The terminal conditions we discuss below primarily address regulation and constant reference tracking. Time-varying tracking and economic MPC require additional techniques such as tube MPC and dissipativity theory.

### What Goes Wrong Without Terminal Ingredients

Remove the terminal cost and the terminal set from the stick controller and
keep everything else. {numref}`fig-rhc-horizon` shows the closed loop from the
same initial state for several horizons, on the exact model and without
disturbances.

:::{figure} _static/receding_horizon/horizon.svg
:label: fig-rhc-horizon
:width: 100%

Closed-loop angle without terminal ingredients for $N=1,3,5,10$ (orange
shades), and with terminal cost and set for $N=10$ (blue), on the exact model.
A cross marks the first infeasible subproblem. With $N=5$ the stick settles
at $\theta\approx0.32$ rad and stays there.
:::

The failures differ in kind, and each one has a clear cause.

- With $N=1$, the objective is $c(\mathbf x_0,u_0)$, and the next state does
  not enter it. Since $\mathbf x_0$ is fixed, the minimizer is the cheapest
  command, $u_0=0$, unless a constraint on $\mathbf x_1$ forces more. The
  controller does nothing until the angle limit is about to be crossed, and by
  then no admissible command can prevent it: the subproblem at step 3 is
  infeasible.
- With $N=3$, the horizon covers $0.15$ s, less than half the time the tilt
  needs to grow by a factor $e$. Within that window, pushing the stick back
  costs command effort and returns little angle cost, so the plan does only
  what is needed to keep the predicted angle under its limit. The subproblem
  again becomes infeasible at step 3.
- With $N=5$, every subproblem is feasible, but the closed loop converges to
  $\theta\approx0.32$ rad, not to the upright position. There, each plan holds
  the stick with a constant command and keeps the predicted angle just inside
  the limit at the end of the horizon. The replanning law has acquired a
  second equilibrium. Convergence to the target is therefore not implied by
  feasibility.
- With $N=10$, the horizon is long enough, and the stick returns upright. Its
  closed-loop cost is $20.83$, against $20.15$ with terminal ingredients. The
  remaining gap is the price of a horizon that sees only part of the recovery.

A long enough horizon eventually stabilizes the system without any terminal
ingredients, and there is a theory that quantifies "long enough"
{cite}`GrunePannek2017`. The required length depends on the system and the
initial state, and nothing in the subproblem reports when it is too short. The
terminal ingredients make a different trade. They give a certificate that
holds for every horizon, at the cost of a smaller feasible set: with the
terminal set, the same initial state is infeasible for every $N\le 8$.

### MPC with Stability Guarantees

To provide theoretical guarantees, the finite-horizon MPC problem is augmented with three interconnected components. The **terminal cost** $c_N(\mathbf{x})$ approximates the cost-to-go beyond the horizon, providing a surrogate for the infinite-horizon tail that cannot be explicitly optimized. The **terminal constraint set** $\mathcal{X}_f$ defines a region where we have local knowledge of how to stabilize the system. Finally, the **terminal controller** $\pi_f(\mathbf{x})$ is a local feedback policy that stabilizes the system within $\mathcal{X}_f$.

These components must satisfy specific compatibility conditions to provide
theoretical guarantees. The statement below is for regulation to the origin
with time-invariant dynamics $\mathbf f$, stage cost $c$, and constraints
$\mathbf g(\mathbf x,\mathbf u)\le\mathbf 0$, which include the state and input
bounds. It follows {cite}`Mayne2000`.

````{prf:theorem} Recursive Feasibility and Asymptotic Stability
:label: thm-mpc-stability

Assume that $\mathbf f(\mathbf 0,\mathbf 0)=\mathbf 0$, that the stage cost
satisfies $c(\mathbf x,\mathbf u)\ge\gamma\|\mathbf x\|^2$ for some
$\gamma>0$, and that $c_N\ge 0$. Let the terminal set $\mathcal X_f$ contain
the origin, and let the local controller $\pi_f$ satisfy two conditions.

**Control invariance**: For all $\mathbf{x} \in \mathcal{X}_f$, we have $\mathbf{f}(\mathbf{x}, \pi_f(\mathbf{x})) \in \mathcal{X}_f$ (the set is invariant) and $\mathbf{g}(\mathbf{x}, \pi_f(\mathbf{x})) \leq \mathbf{0}$ (constraints remain satisfied).

**Lyapunov decrease**: For all $\mathbf{x} \in \mathcal{X}_f$:

   $$c_N(\mathbf{f}(\mathbf{x}, \pi_f(\mathbf{x}))) - c_N(\mathbf{x}) \leq -c(\mathbf{x}, \pi_f(\mathbf{x}))$$

   where $c$ is the stage cost.

Suppose that the plant follows the prediction model exactly. Then, for every
initial state $\mathbf x_0\in\mathcal X_N$, the closed loop
$\mathbf x_{t+1}=\mathbf f(\mathbf x_t,\kappa_N(\mathbf x_t))$ has the
following properties:

1. **Recursive feasibility**: $\mathbf x_t\in\mathcal X_N$ for all $t$, so
   every subproblem has a solution.
2. **Value decrease**: $V_N(\mathbf x_{t+1})\le V_N(\mathbf x_t)-c(\mathbf x_t,\kappa_N(\mathbf x_t))$.
3. **Cost bound and convergence**:
   $\sum_{t=0}^{\infty}c(\mathbf x_t,\kappa_N(\mathbf x_t))\le V_N(\mathbf x_0)$,
   and $\mathbf x_t\to\mathbf 0$.
````

Asymptotic stability, in the sense of a Lyapunov function, also needs $V_N$ to
be bounded above by a multiple of $\|\mathbf x\|^2$ near the origin. This
holds, for example, when $c_N$ is quadratic and the origin lies in the
interior of $\mathcal X_f$ {cite}`RawlingsMayneDiehl2017`. The three
properties listed are the ones a practitioner relies on, and they follow from
one construction.

```{prf:proof}
Fix $\mathbf x\in\mathcal X_N$ and let
$(\mathbf u_0^\star,\ldots,\mathbf u_{N-1}^\star)$ be an optimal solution of
the subproblem from $\mathbf x$, with predicted states
$\mathbf x_0^\star=\mathbf x,\mathbf x_1^\star,\ldots,\mathbf x_N^\star$ and
$\mathbf x_N^\star\in\mathcal X_f$. The controller applies
$\mathbf u_0^\star$, and since the model is exact, the next state is
$\mathbf x^+=\mathbf x_1^\star$.

*A feasible candidate from $\mathbf x^+$.* Consider the shifted sequence

$$
\tilde{\mathbf u}=\big(\mathbf u_1^\star,\ldots,\mathbf u_{N-1}^\star,\ \pi_f(\mathbf x_N^\star)\big).
$$

From $\mathbf x^+$, its first $N-1$ commands reproduce the predicted states
$\mathbf x_2^\star,\ldots,\mathbf x_N^\star$, and these satisfy the
constraints because they did in the original solution. The last command
satisfies $\mathbf g(\mathbf x_N^\star,\pi_f(\mathbf x_N^\star))\le\mathbf 0$,
and the new terminal state
$\mathbf f(\mathbf x_N^\star,\pi_f(\mathbf x_N^\star))$ lies in
$\mathcal X_f$, both by control invariance. Hence $\tilde{\mathbf u}$ is
feasible, so $\mathbf x^+\in\mathcal X_N$. Induction on $t$ proves property 1.

*Its cost.* The candidate drops the first stage of the original solution and
adds one stage under $\pi_f$ at the end. Its cost is

$$
\tilde V=V_N(\mathbf x)-c(\mathbf x,\mathbf u_0^\star)
+\Big[c(\mathbf x_N^\star,\pi_f(\mathbf x_N^\star))
+c_N\big(\mathbf f(\mathbf x_N^\star,\pi_f(\mathbf x_N^\star))\big)
-c_N(\mathbf x_N^\star)\Big].
$$

The bracket is non-positive by the Lyapunov decrease condition. Since
$V_N(\mathbf x^+)$ is the optimal cost from $\mathbf x^+$, it does not exceed
the cost of any feasible candidate:

$$
V_N(\mathbf x^+)\le\tilde V\le V_N(\mathbf x)-c(\mathbf x,\kappa_N(\mathbf x)).
$$

This is property 2.

*Summing.* Adding property 2 over $t=0,\ldots,T-1$ telescopes to

$$
\sum_{t=0}^{T-1}c(\mathbf x_t,\kappa_N(\mathbf x_t))\le V_N(\mathbf x_0)-V_N(\mathbf x_T)\le V_N(\mathbf x_0),
$$

using $V_N\ge0$. Letting $T\to\infty$ gives the cost bound. A convergent
series of non-negative terms has terms tending to zero, so
$\gamma\|\mathbf x_t\|^2\le c(\mathbf x_t,\kappa_N(\mathbf x_t))\to0$.
```

The proof never uses optimality of the plan beyond the first step. It compares
the new optimum with one feasible candidate. Two practical consequences follow.
First, the guarantees survive an inexact solver as long as each returned plan
is feasible and costs no more than the shifted candidate. Many real-time
implementations check this explicitly and apply the candidate when the solver
fails to improve on it. Second, the exact-model assumption enters at exactly
one place, $\mathbf x^+=\mathbf x_1^\star$. With model error or disturbances,
the measured state differs from the predicted one, and the candidate need not
be feasible from it. In {numref}`fig-rhc-replanning`, the recorded values
$V_N(\mathbf x_t)$ decrease at every step until the tap, then jump from $0.61$
to $3.23$. Restoring guarantees under such errors is the subject of robust
MPC in the [next chapter](mpc-variants-reliability.md#robust-mpc).

The bound in property 3 also gives a quantitative statement that the stick
experiment can check. With terminal ingredients and $N=10$, the closed-loop
cost from $\mathbf x_0=(0.2,0.6)$ is $20.15$, equal to $V_{10}(\mathbf x_0)$ to
solver precision. The next two subsections explain why the bound is tight here.

### Constructing Terminal Ingredients for Linear Systems

The conditions of {prf:ref}`thm-mpc-stability` look abstract, but for a linear
system $\mathbf x_{k+1}=A\mathbf x_k+B\mathbf u_k$ with quadratic stage cost
$\mathbf x^\top Q\mathbf x+\mathbf u^\top R\mathbf u$, they can be met with
two matrix computations and one scalar.

*Terminal controller and cost.* Choose a linear terminal controller
$\pi_f(\mathbf x)=K\mathbf x$ whose closed-loop matrix $A_K=A+BK$ has all
eigenvalues inside the unit circle, and a quadratic terminal cost
$c_N(\mathbf x)=\mathbf x^\top P\mathbf x$. Substituting into the Lyapunov
decrease condition gives

$$
\mathbf x^\top\big(A_K^\top PA_K-P+Q+K^\top RK\big)\mathbf x\le0\qquad\text{for all }\mathbf x\in\mathcal X_f .
$$

The condition holds with equality for every $\mathbf x$ when $P$ solves the
discrete Lyapunov equation

$$
A_K^\top PA_K-P+Q+K^\top RK=0 .
$$

Because $A_K$ is stable, the solution is unique and equals the series
$P=\sum_{k=0}^{\infty}(A_K^\top)^k(Q+K^\top RK)A_K^k$. Hence
$\mathbf x^\top P\mathbf x$ is *exactly* the cost of following
$\mathbf u=K\mathbf x$ forever from $\mathbf x$, provided no constraint is
violated along the way. With this choice, the terminal cost is no longer a
guess at the tail. It is the cost of one admissible tail.

Any stabilizing $K$ works. The natural choice is the gain that makes this tail
as cheap as possible, the infinite-horizon linear-quadratic regulator. Its
$P$ solves the discrete algebraic Riccati equation and its gain is
$K=-(R+B^\top PB)^{-1}B^\top PA$. These formulas are derived by dynamic
programming in [Finite-Horizon Dynamic Programming](finite-horizon-dp.md).
The argument here needs only the Lyapunov equation, which the LQR pair
satisfies.

*Terminal set.* The tail cost is meaningful only where $K\mathbf x$ respects
the constraints for all future steps. Take an ellipsoid
$\mathcal E_\alpha=\{\mathbf x:\mathbf x^\top P\mathbf x\le\alpha\}$. It is
invariant under $A_K$ for every $\alpha$: by the Lyapunov equation,

$$
(A_K\mathbf x)^\top P(A_K\mathbf x)=\mathbf x^\top P\mathbf x-\mathbf x^\top(Q+K^\top RK)\mathbf x\le\mathbf x^\top P\mathbf x\le\alpha .
$$

It remains to choose $\alpha$ so that every constraint holds inside
$\mathcal E_\alpha$. Write each constraint, after substituting
$\mathbf u=K\mathbf x$, as a pair of half-spaces $\pm\mathbf a^\top\mathbf x\le h$.
For the stick, the rows are $\mathbf a=K^\top$ with $h=u_{\max}$, and
$\mathbf a=(1,0)$ with $h=\theta_{\max}$. The largest value of
$\mathbf a^\top\mathbf x$ on $\mathcal E_\alpha$ is
$\sqrt{\alpha\,\mathbf a^\top P^{-1}\mathbf a}$, a short Lagrange-multiplier
computation that the exercises ask for. Hence the largest admissible level is

$$
\alpha=\min_{\text{rows}}\ \frac{h^2}{\mathbf a^\top P^{-1}\mathbf a}.
$$

With $\mathcal X_f=\mathcal E_\alpha$, both conditions of the theorem hold.
Invariance and constraint satisfaction hold by construction, and the decrease
holds with equality. This is the construction of {cite}`ChenAllgower1998`,
specialized to linear models. For a nonlinear model, the same recipe is
applied to the linearization at the equilibrium, with $\alpha$ reduced until
the neglected nonlinear terms are dominated.

For the stick, the LQR gain is $K=(-23.69,\,-7.55)$, and the input bound is
the binding row, giving $\alpha=4.02$. The initial state of the experiments has
$\mathbf x_0^\top P\mathbf x_0=15.98$, well outside $\mathcal X_f$, which is
why at least nine steps are needed to reach it. Two numerical checks in the
build script confirm the construction. Iterating the finite-horizon Riccati
recursion from $P=Q$ converges to the same $P$ to within $3\times10^{-12}$,
and on 2000 random points of $\mathcal X_f$, the largest value of
$c_N(A_K\mathbf x)-c_N(\mathbf x)+c(\mathbf x,K\mathbf x)$ is of order
$10^{-15}$, which is zero to rounding.

### Suboptimality Bounds

How does the finite-horizon value $V_N(\mathbf x)$ compare with the
infinite-horizon value $V_\infty(\mathbf x)$ of the constrained problem? The
answer depends on the terminal ingredients, and the three choices used on the
stick give three different answers. {numref}`fig-rhc-values` plots them.

:::{figure} _static/receding_horizon/values.svg
:label: fig-rhc-values
:width: 100%

Optimal finite-horizon value from $\mathbf x_0=(0.2,0.6)$ as a function of
the horizon, for three terminal choices. Without terminal ingredients the
value rises toward $V_\infty$ from below. With the LQR terminal cost alone it
can also lie below $V_\infty$. With terminal cost and terminal set the problem
is infeasible for $N\le8$ (shaded), and equals $V_\infty$ to solver precision
from $N=9$ on.
:::

*Without terminal ingredients, $V_N$ is a lower bound.* Take the optimal
infinite sequence and keep its first $N$ commands. They are feasible for the
$N$-step problem, and their cost omits only non-negative tail terms. Hence
$V_N(\mathbf x)\le V_\infty(\mathbf x)$. The same argument shows that $V_N$
is non-decreasing in $N$. On the stick, $V_1=0.76$ and $V_{10}=13.57$, against
$V_\infty=20.15$.

*With a terminal cost alone, $V_N$ is neither.* The LQR terminal cost
$\mathbf x^\top P\mathbf x$ is the cost of the unconstrained tail
$\mathbf u=K\mathbf x$. Outside $\mathcal X_f$, that tail violates a
constraint, and its cost can be lower than that of any admissible tail. With
$N=1$, the stick gives $V_1=17.38<V_\infty$. An upper bound cannot be
obtained without the terminal set.

*With terminal cost and terminal set, $V_N$ is an upper bound.* Every
feasible $N$-step plan, extended by $\pi_f$ from its terminal state, is an
admissible infinite sequence. By the Lyapunov equation, its cost equals the
$N$-step objective, because $c_N$ is exactly the cost of the extension. The
infinite-horizon problem can choose among more sequences, so
$V_\infty(\mathbf x)\le V_N(\mathbf x)$ on $\mathcal X_N$. The bound requires
that $c_N$ be at least the cost of following $\pi_f$ from every point of
$\mathcal X_f$. A terminal cost that underestimates that tail loses the upper bound.

#### Deriving the Approximation Error

The interesting question is bounding the approximation error
$\varepsilon_N=V_N(\mathbf x)-V_\infty(\mathbf x)\ge0$ in the third case. It
represents the cost of being forced to use $\pi_f$ beyond the horizon rather
than continuing to optimize.

Let $(\mathbf{u}_0^\star, \mathbf{u}_1^\star, \ldots)$ denote the infinite-horizon optimal control sequence with corresponding state trajectory $(\mathbf{x}_0^\star, \mathbf{x}_1^\star, \ldots)$ where $\mathbf{x}_0^\star = \mathbf{x}$. The infinite-horizon cost is:

$$V_\infty(\mathbf{x}) = \sum_{k=0}^{\infty} c(\mathbf{x}_k^\star, \mathbf{u}_k^\star)$$

Suppose the horizon is long enough that this optimal trajectory has entered
the terminal set by step $N$, so that $\mathbf x_N^\star\in\mathcal X_f$. Then
its first $N$ commands are feasible for the $N$-step problem, with cost

$$\tilde{V}_N(\mathbf{x}) = \sum_{k=0}^{N-1} c(\mathbf{x}_k^\star, \mathbf{u}_k^\star) + c_N(\mathbf{x}_N^\star)$$

where $c_N(\mathbf{x}_N^\star)$ approximates the tail cost $\sum_{k=N}^{\infty} c(\mathbf{x}_k^\star, \mathbf{u}_k^\star)$.

Since $V_N(\mathbf{x})$ is the optimal $N$-horizon cost (which may do better than this particular truncated sequence), we have $V_N(\mathbf{x}) \leq \tilde{V}_N(\mathbf{x})$. The approximation error therefore satisfies:

$$\varepsilon_N \leq \tilde{V}_N(\mathbf{x}) - V_\infty(\mathbf{x}) = c_N(\mathbf{x}_N^\star) - \sum_{k=N}^{\infty} c(\mathbf{x}_k^\star, \mathbf{u}_k^\star)$$

This bound shows that the approximation error depends on how well the
terminal cost approximates the optimal tail cost along the infinite-horizon
optimal trajectory, once that trajectory has entered $\mathcal X_f$. If
$\mathbf x_N^\star\notin\mathcal X_f$, the truncated sequence is infeasible for
the $N$-step problem and the bound says nothing.

The stick shows the extreme case. Inside $\mathcal X_f$, the LQR law satisfies
every constraint, so it is optimal for the constrained problem too, and
$c_N=V_\infty$ there. The bound then gives $\varepsilon_N=0$ as soon as
$\mathbf x_N^\star\in\mathcal X_f$, which is what {numref}`fig-rhc-values`
shows from $N=9$ on. Combined with property 3 of the theorem,

$$
V_\infty(\mathbf x_0)\le\sum_{t}c(\mathbf x_t,\kappa_N(\mathbf x_t))\le V_N(\mathbf x_0)=V_\infty(\mathbf x_0),
$$

so the closed-loop cost equals $V_\infty$, as measured. On systems where the
optimal tail is not linear, $\varepsilon_N$ is positive and shrinks with $N$
at a rate set by how fast optimal trajectories approach $\mathcal X_f$.

## Summary and Outlook

Receding-horizon control turns a finite-horizon optimizer into feedback by
reinitializing it from each measured state and applying only the first planned
action. The result is a state-feedback law $\kappa_N$, defined on the feasible
set $\mathcal X_N$ and evaluated by optimization. On the balancing stick,
replanning with a wrong model keeps the stick upright, while under the same plan
executed open loop the stick falls within a second.

A short horizon alone guarantees nothing. It can leave an unstable system
uncontrolled, lose feasibility, or create a spurious equilibrium near a
constraint. A terminal cost, a terminal set, and an invariant local controller
connect the truncated problem to the infinite-horizon one. The shifted
candidate sequence then proves recursive feasibility, a decreasing value, and
a closed-loop cost no larger than $V_N(\mathbf x_0)$. For linear systems, the
terminal cost solves a Lyapunov equation, and the terminal set is an
ellipsoid whose size follows from the constraints in closed form.

The basic loop leaves several design choices unresolved. How should the same
replanning mechanism represent tracking, economic objectives, uncertainty,
hybrid decisions, solver failures, and hard real-time deadlines? [MPC variants
and reliable operation](mpc-variants-reliability.md) organize those choices.
The guarantees here also assumed an exact model, and the robust and
stochastic formulations of that chapter address what replaces them.

## Exercises

:::{exercise} A scalar unstable plant
:label: ex-rhc-scalar

Consider $x_{k+1}=2x_k+u_k$ with $|u_k|\le1$ and stage cost $x^2+u^2$.

1. Show that one-step MPC without terminal ingredients applies $u=0$ from
   every state, and conclude that it does not stabilize the origin.
2. Take the terminal controller $u=Kx$ with $K=-3/2$. Solve the scalar
   Lyapunov equation for $P$, and verify the decrease condition.
3. Find the largest $\alpha$ such that $\{x:Px^2\le\alpha\}$ satisfies the
   input bound under $u=Kx$. Express the terminal set as an interval.
:::

:::{solution} ex-rhc-scalar
:class: dropdown

1. With $N=1$, the objective is $x_0^2+u_0^2$, and $x_0$ is fixed, so the
   minimizer is $u_0=0$. The closed loop is $x_{t+1}=2x_t$, which diverges
   from every nonzero state.
2. $A_K=2-\tfrac32=\tfrac12$ and $Q+K^\top RK=1+\tfrac94=\tfrac{13}4$. The
   Lyapunov equation $\tfrac14P-P+\tfrac{13}4=0$ gives $P=\tfrac{13}3$. The
   decrease holds with equality:
   $P(\tfrac12x)^2-Px^2+\tfrac{13}4x^2=(\tfrac{13}{12}-\tfrac{13}3+\tfrac{13}4)x^2=0$.
3. $|Kx|\le1$ means $|x|\le\tfrac23$. The formula gives
   $\alpha=h^2/(a^2/P)=1/(\tfrac94\cdot\tfrac3{13})=\tfrac{52}{27}$, and
   $\{x:\tfrac{13}3x^2\le\tfrac{52}{27}\}=\{|x|\le\tfrac23\}$.
:::

:::{exercise} One step with the LQR terminal cost
:label: ex-rhc-one-step-lqr

Let $P$ solve the discrete algebraic Riccati equation for $(A,B,Q,R)$, and
let $K=-(R+B^\top PB)^{-1}B^\top PA$. Show that one-step MPC with terminal
cost $\mathbf x^\top P\mathbf x$ and no active constraints applies exactly
$\mathbf u=K\mathbf x$. Which curve of {numref}`fig-rhc-values` does this
explain at $N=1$, and why does that value lie below $V_\infty$?
:::

:::{solution} ex-rhc-one-step-lqr
:class: dropdown

The objective is
$\mathbf x^\top Q\mathbf x+\mathbf u^\top R\mathbf u+(A\mathbf x+B\mathbf u)^\top P(A\mathbf x+B\mathbf u)$.
Setting its gradient in $\mathbf u$ to zero gives
$(R+B^\top PB)\mathbf u=-B^\top PA\mathbf x$, which is $\mathbf u=K\mathbf x$.
Without constraints, its value is $\mathbf x^\top P\mathbf x$ by the Riccati
equation, the unconstrained infinite-horizon cost. This is the green curve at
$N=1$. From $\mathbf x_0=(0.2,0.6)$, the unconstrained LQR tail exceeds the
command bound, so it is cheaper than every admissible tail, and the value
$17.38$ lies below the constrained $V_\infty=20.15$.
:::

:::{exercise} Support function of an ellipsoid
:label: ex-rhc-ellipsoid

Let $P$ be symmetric positive definite and $\alpha>0$. Show that
$\max\{\mathbf a^\top\mathbf x:\mathbf x^\top P\mathbf x\le\alpha\}=\sqrt{\alpha\,\mathbf a^\top P^{-1}\mathbf a}$,
and deduce the formula for the terminal level $\alpha$.
:::

:::{solution} ex-rhc-ellipsoid
:class: dropdown

The maximum of a linear function over the ellipsoid is attained on its
boundary. Stationarity of
$\mathbf a^\top\mathbf x-\lambda(\mathbf x^\top P\mathbf x-\alpha)$ gives
$\mathbf x=P^{-1}\mathbf a/(2\lambda)$. Substituting into
$\mathbf x^\top P\mathbf x=\alpha$ gives
$2\lambda=\sqrt{\mathbf a^\top P^{-1}\mathbf a/\alpha}$, and the value is
$\mathbf a^\top P^{-1}\mathbf a/(2\lambda)=\sqrt{\alpha\,\mathbf a^\top P^{-1}\mathbf a}$.
The half-space $\mathbf a^\top\mathbf x\le h$ contains the ellipsoid exactly
when this value is at most $h$, that is, when
$\alpha\le h^2/(\mathbf a^\top P^{-1}\mathbf a)$. By symmetry, the same
condition covers $-\mathbf a^\top\mathbf x\le h$. Taking the minimum over all
rows gives $\alpha$.
:::

:::{exercise} The spurious equilibrium
:label: ex-rhc-parking

Using `code/receding_horizon.py`, run the controller without terminal
ingredients with $N=5$ from $(0.2,0.6)$ for 200 steps. Inspect the final
state, the applied command, and the predicted trajectory of the last
subproblem. Explain why the state is a fixed point of the closed loop, and why
it is not a fixed point of the open-loop dynamics with $u=0$. Find the
smallest horizon for which the stick returns upright from this initial state.
:::

:::{solution} ex-rhc-parking
:class: dropdown

The closed loop settles at $\dot\theta=0$, $\theta\approx0.32$ rad, with a
constant command close to $-(g/\ell)\,\theta\approx-3.1$ rad/s$^2$, which
cancels the gravitational acceleration. The last plan holds the stick nearly
still, and its predicted angle approaches the limit at the end of the horizon.
Within five steps, pushing the stick back costs more command effort than it
saves in angle cost, so the plan that holds the state is optimal, and applying
its first command reproduces the same state. With $u=0$, the same state falls
away. With $N=6$, the stick still parks, near $\theta\approx0.29$ rad. $N=7$
is the smallest horizon that returns it upright within 200 steps, with
closed-loop cost $34.1$, and $N=8$ lowers that cost to $24.1$.
:::

:::{exercise} Where the proof breaks under model error
:label: ex-rhc-mismatch

Repeat the replanning experiment of {numref}`fig-rhc-replanning`, and record
$V_N(\mathbf x_t)$ along the closed loop. Identify each step where
$V_N(\mathbf x_{t+1})>V_N(\mathbf x_t)-c(\mathbf x_t,\mathbf u_t)$. Which line of
the proof of {prf:ref}`thm-mpc-stability` fails at those steps? Then shorten
the plant's stick further and find the length at which the replanned
controller first meets an infeasible subproblem.
:::

:::{solution} ex-rhc-mismatch
:class: dropdown

The proof assumes $\mathbf x^+=\mathbf x_1^\star$. The tap violates that
assumption by $0.5$ rad/s in one step, and $V_N$ jumps from $0.61$ to
$3.23$. The shorter stick violates it by a small amount at every step, so the
decrease inequality can fail by a margin even where $V_N$ keeps decreasing.
The shifted candidate is built from the predicted state, so it need not be
feasible from the measured one. Recursive feasibility is no longer guaranteed,
and for a short enough stick it fails. The threshold depends on the tap and
the initial state, which is why robust MPC tightens the constraints by bounds
on the prediction error.
:::

:::{exercise} A rate limit in the terminal set
:label: ex-rhc-rate

Add the constraint $|\dot\theta_k|\le1$ rad/s to the stick through
`StickParameters(rate_max=1.0)`. Recompute $\alpha$ and state which row is now
binding. Recompute the feasible sets of {numref}`fig-rhc-feasible-sets` and
check whether $\kappa_{20}$ still agrees with $\operatorname{sat}(K\mathbf x)$.
:::

:::{solution} ex-rhc-rate
:class: dropdown

The terminal level becomes the minimum over three rows. For the rate row,
$\mathbf a=(0,1)$ and $h=1$ give $h^2/(\mathbf a^\top P^{-1}\mathbf a)\approx4.39$.
For the angle row the value is $8.99$, and for the input row $4.02$. The input
bound still binds, so $\alpha$ and $\mathcal X_f$ do not change, although the
rate bound comes close. The function `terminal_level` performs the same
computation. The feasible sets are cut by the horizontal lines
$\dot\theta=\pm1$. Where the rate bound becomes active along the optimal
predicted path, the first command can differ from the saturated linear law, so
measure the gap rather than assume it.
:::

## Computational Sources

The model, the MPC subproblem, and the checks are in one module, and the
figures and numbers are produced by one script:

- {download}`Balancing-stick model and linear MPC <code/receding_horizon.py>`
- {download}`Experiment and figure builder <scripts/build_receding_horizon_artifacts.py>`
- {download}`Numbers quoted in this chapter <artifacts/receding_horizon/metrics.json>`

Run `uv run python scripts/build_receding_horizon_artifacts.py` from the
repository root to regenerate the four figures and the metrics, in about one
minute. Normal book builds read the saved figures. The tests in
`tests/test_receding_horizon.py` check the Lyapunov equation, the terminal
level, the shifted-candidate inequality, recursive feasibility, and the
bounds on $V_N$ on short runs.

## Self-checks

:::{exercise} Receding horizon
:label: ex-mpc-check-1

An MPC solver returns a sequence $(\mathbf{u}_0^\star,\ldots,\mathbf{u}_{N-1}^\star)$. Which controls are normally applied before the problem is solved again?
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
