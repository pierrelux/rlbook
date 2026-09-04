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
# Finite-Horizon Optimal Control Problems

The models introduced in the previous chapter predict how actions change a
system's state. Trajectory optimization adds an objective and constraints, then
selects the actions that produce a desirable state sequence. For a fixed
discrete horizon, the states and controls form a finite vector, so the planning
problem can be written as a nonlinear program.

Which finite-dimensional optimization problem selects the best admissible
trajectory from one known initial condition?

A **trajectory** is the time-indexed sequence of states
$(\mathbf{x}_1,\ldots,\mathbf{x}_T)$ and controls
$(\mathbf{u}_1,\ldots,\mathbf{u}_{T-1})$. This chapter first computes the
complete control sequence from a known initial state. The sequence is
**open loop**: once execution begins, the action at time $t$ does not change in
response to the measured state.

The three-satellite example below makes both the value and the limitation of
open-loop planning visible. A linear model produces a sparse, feasible plan,
but the same immutable plan misses its target when replayed through a nonlinear
model. Later chapters will replace the fixed sequence by feedback through
receding-horizon control and policies.

```{admonition} Learning Goals
:class: note

After studying this chapter, you should be able to:

1. Formulate a discrete-time optimal control problem (DOCP) in Bolza, Lagrange, and Mayer forms, and convert between them.
2. State the KKT conditions for a constrained NLP and explain the role of constraint qualifications.
3. Derive the discrete-time Pontryagin principle from the Lagrangian of a DOCP.
4. Implement single shooting and multiple shooting methods for trajectory optimization.
5. Explain the trade-offs between simultaneous (direct transcription) and sequential (shooting) methods.
6. Compute gradients of the objective with respect to controls using the adjoint (costate) recursion.
7. Audit an immutable open-loop plan under a specified model mismatch.
```

```{admonition} Prerequisites
:class: tip

This chapter uses gradients, Jacobians, the chain rule, matrix-vector products,
and linear systems. [](appendix_nlp.md) reviews the nonlinear-programming
objects used in the KKT derivation. The state and transition models come from
[](modeling-controlled-systems.md).
```

## A Motivating Example: Phasing Three Satellites with Differential Drag

What does an open-loop plan gain from a transparent finite-horizon model, and
which part of its guarantee disappears when the replay model changes?

Three small satellites are released into nearly the same circular orbit, but the
mission calls for them to occupy slots separated by $120^\circ$. They have no
propulsion. Each satellite can instead rotate between low- and high-drag
attitudes. High drag lowers its orbit slightly; the lower satellite then moves
faster and accumulates phase relative to the others. The control authority is
weak, slow, and irreversible because every maneuver spends altitude.

Differential drag has been used to phase propulsionless satellite
constellations in orbit {cite:p}`Foster2018ConstellationPhasing`. Linear
programs provide a useful planning model for this control mechanism
{cite:p}`Sin2018DifferentialDrag`. The example below is a teaching-scale
reconstruction inspired by that literature. It is not a reconstruction of a
particular flight campaign.

We plan over $N=180$ daily intervals. All three satellites begin at a circular
altitude of $475$ km, with phase offsets

$$
\varphi_{0}
=
\begin{bmatrix}-0.5&0&0.5\end{bmatrix}^{\mathsf T}\text{ degrees}
$$

and zero relative angular rates. The action

$$
0\leq u_{i,k}\leq1
$$

is the fraction of day $k$ that satellite $i$ spends in its high-drag
attitude. For the nominal daily model, its state is

$$
x_{i,k}
=
\begin{bmatrix}
\varphi_{i,k}\\
\omega_{i,k}\\
\ell_{i,k}
\end{bmatrix},
$$

where $\varphi$ is phase in degrees, $\omega$ is relative angular rate in
degrees per day, and $\ell$ is extra altitude loss in kilometres relative to
remaining in the low-drag attitude.

### From drag physics to a daily linear model

Use SI units in the derivation. With $h_0=475\text{ km}$, set

$$
a_0=(R_\mathrm{E}+h_0)\frac{10^3\text{ m}}{1\text{ km}}
=6{,}853{,}137\text{ m},
\qquad
n_0=\sqrt{\mu/a_0^3}.
$$

At the reference density $\rho_0$, changing the
ballistic coefficient from $B_\mathrm{low}$ to $B_\mathrm{high}$ changes the
area-to-mass factor by

$$
\Delta\sigma
=\frac{1}{B_\mathrm{high}}-\frac{1}{B_\mathrm{low}}.
$$

Linearizing the semimajor-axis and mean-motion changes over one day gives

$$
\begin{aligned}
d
&=\rho_0\Delta\sigma\sqrt{\mu a_0}\,\Delta t
\left(\frac{1\ {\rm km}}{10^3\ {\rm m}}\right),\\
\alpha
&=\frac{3n_0}{2a_0}
\rho_0\Delta\sigma\sqrt{\mu a_0}\,\Delta t^2
\left(\frac{180^\circ}{\pi\ {\rm rad}}\right),
\end{aligned}
$$

where $\Delta t=86400$ s. For

$$
\begin{gathered}
\rho_0=3\times10^{-13}\ {\rm kg/m^3},\qquad
B_\mathrm{low}=60\ {\rm kg/m^2},\qquad
B_\mathrm{high}=20\ {\rm kg/m^2},\\
R_\mathrm{E}=6378.137\ {\rm km},\qquad
\mu=3.986004418\times10^{14}\ {\rm m^3/s^2},
\end{gathered}
$$

the coefficients are

$$
\alpha=0.0544503\ {\rm deg/day^2},
\qquad
d=0.0451572\ {\rm km/day}.
$$

The daily dynamics are therefore

$$
x_{i,k+1}
=
\underbrace{
\begin{bmatrix}
1&1&0\\
0&1&0\\
0&0&1
\end{bmatrix}}_{A}
x_{i,k}
+
\underbrace{
\begin{bmatrix}
\alpha/2\\
\alpha\\
d
\end{bmatrix}}_{B}
u_{i,k}.
$$

The phase receives half of the new daily rate during the interval, the rate
accumulates the drag-induced acceleration, and the extra altitude loss
accumulates monotonically.

### A finite open-loop plan

Define the cyclic difference matrix

$$
G=
\begin{bmatrix}
-1&1&0\\
0&-1&1\\
1&0&-1
\end{bmatrix}.
$$

The terminal target is

$$
G\varphi_N
\approx
\begin{bmatrix}120\\120\\-240\end{bmatrix}
\text{ degrees},
\qquad
G\omega_N\approx0,
$$

with tolerances of $0.1$ degree and $0.002$ degree per day. These are unwrapped
directed differences. Because the rows of $G$ sum to zero,
the third target is $-240^\circ$, which represents the same circular separation
as $+120^\circ$ modulo $360^\circ$ while preserving a consistent unwrapped
coordinate system.

The primary linear program minimizes the worst final extra altitude loss:

$$
\begin{aligned}
\underset{x,u,z}{\operatorname{minimize}}\quad&z\\
\text{subject to}\quad&
x_{i,k+1}=Ax_{i,k}+Bu_{i,k},\\
&\ell_{i,N}\leq z,\qquad 0\leq u_{i,k}\leq1,\\
&\left\lVert G\varphi_N-
\begin{bmatrix}120&120&-240\end{bmatrix}^{\mathsf T}
\right\rVert_\infty\leq0.1,\\
&\lVert G\omega_N\rVert_\infty\leq0.002.
\end{aligned}
$$

A second linear program keeps $z$ at its primary optimum, up to numerical
tolerance, and minimizes

$$
\sum_{i=1}^{3}\sum_{k=0}^{N-2}
\left|u_{i,k+1}-u_{i,k}\right|.
$$

This lexicographic step selects a low-variation member of the primary optimal
set without changing the worst-loss objective beyond numerical tolerance.

### Nominal and nonlinear replay

The linear plan is first rolled out through the model used by the optimizer.
The exact same $u_{i,k}$ is then replayed, without reoptimization, through the
nonlinear orbital model

$$
\dot a_i
=-\rho(a_i,t)\,\sigma(u_i)\sqrt{\mu a_i},
\qquad
\dot\theta_i
=\sqrt{\frac{\mu}{a_i^3}},
$$

where

$$
\begin{aligned}
\sigma(u)
&=\frac{1-u}{B_\mathrm{low}}+\frac{u}{B_\mathrm{high}},\\
\rho(a,t)
&=\rho_0
\left[
0.90+0.15\sin\left(\frac{2\pi t}{26\text{ days}}+\frac{\pi}{6}\right)
\right]
\exp\left(\frac{h_0-h_i}{60\text{ km}}\right),
\qquad
h_i=\frac{a_i-R_\mathrm{E}}{10^3}\text{ km}.
\end{aligned}
$$

Here $a_i$ and $R_\mathrm{E}$ are in metres inside the orbital equations, so
$h_i$ is the corresponding altitude in kilometres. In the display, the nominal
altitude trace is the planning-model proxy $475-\ell_i$ km and therefore omits
the common low-drag decay; the nonlinear trace reports absolute orbital
altitude.

The nonlinear trajectory is integrated by hourly RK4 and checked against a
30-minute replay. This variable-density model is a deterministic teaching
stress test, not a flight-dynamics reconstruction. The complete command
sequence is known from day zero; the state traces below are revealed only up
to the playhead.

```{code-cell} python
:tags: [remove-input]
:label: fig-cubesat-differential-drag
:caption: A single open-loop differential-drag plan is evaluated by two plant models. The nominal linear rollout reaches the cyclic slot and relative-rate tolerances. The nonlinear variable-density replay uses the unchanged plan and exposes the accumulated phase miss. The orbit diagrams use a fixed radius; altitude differences are reported numerically rather than exaggerated geometrically.

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from cubesat_replay import render_cubesat_replay

display(HTML(render_cubesat_replay(
    Path("artifacts/cubesat/textbook_results.json"),
    replay_id="cubesat-differential-drag-replay",
    fallback_id="fig-cubesat-differential-drag-fallback",
)))
```

:::{figure} _static/cubesat/differential-drag.svg
:label: fig-cubesat-differential-drag-fallback
:class: pdf-fallback
:alt: Nominal and nonlinear final constellation gaps, altitude loss, phase-error histories, and the common open-loop drag plan for three satellites.

Static audit of the immutable differential-drag plan. The online book adds
synchronized playback and scrubbing while keeping the full planned command
heatmap visible from the start.
:::

```{include} artifacts/cubesat/results.md
```

{download}`Download the open-loop plan (CSV) <artifacts/cubesat/open_loop_plan.csv>`

{download}`Download the audit metrics (CSV) <artifacts/cubesat/metrics.csv>`

The nominal rollout establishes feasibility for the optimization model. The
nonlinear replay tests the same plan under the declared model change, and the
terminal phase constraints fail. [Closing the loop by
replanning](receding-horizon-control.md#closing-the-loop-by-replanning) will replace the immutable
schedule by controls that can change when new state measurements arrive.

The example already contains the ingredients of a **discrete-time optimal
control problem** (DOCP): a state $x_{i,k}$, a bounded control $u_{i,k}$, a
transition map, terminal constraints, and an objective accumulated over a
finite horizon. We now formalize that structure.

## Discrete-Time Optimal Control Problems (DOCPs)

Which variables, costs, dynamics, and constraints place the satellite planner
inside a reusable finite-horizon control template?

Consider a system described by a **state** $\mathbf{x}_t \in \mathbb{R}^n$, summarizing everything needed to predict its evolution. At each stage $t$, we can influence the system through a **control input** $\mathbf{u}_t \in \mathbb{R}^m$. The dynamics specify how the state evolves:

$$
\mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t),
$$

where $\mathbf{f}_t$ may be nonlinear or time-varying. We assume the initial state $\mathbf{x}_1$ is known.

The goal is to pick a sequence of controls $\mathbf{u}_1,\dots,\mathbf{u}_{T-1}$ that makes the trajectory desirable. But desirable in what sense? That depends on an **objective function**, which often includes two components:

$$
\text{(i) stage cost: } c_t(\mathbf{x}_t,\mathbf{u}_t), \qquad \text{(ii) terminal cost: } c_T(\mathbf{x}_T).
$$

The stage cost reflects ongoing penalties such as energy, delay, or risk. The terminal cost measures the value (or cost) of ending in a particular state. Together, these give a discrete-time Bolza problem with path constraints and bounds:

$$
\begin{aligned}
    \text{minimize} \quad & c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t) \\
    \text{subject to} \quad & \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t) \\
                            & \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t) \leq \mathbf{0} \\
                            & \mathbf{x}_{\text{min}} \leq \mathbf{x}_t \leq \mathbf{x}_{\text{max}} \\
                            & \mathbf{u}_{\text{min}} \leq \mathbf{u}_t \leq \mathbf{u}_{\text{max}} \\
    \text{given} \quad & \mathbf{x}_1 = \mathbf{x}_0 \enspace .
\end{aligned}
$$

In the satellite example, $\mathbf{x}_t$ stacks the three phase, relative-rate,
and altitude-loss states; $\mathbf{u}_t$ stacks the three daily drag fractions;
and $\mathbf{f}_t$ applies the block-diagonal copies of the daily map. The
terminal inequalities impose the cyclic slot and rate tolerances, while the
epigraph variable $z$ represents the worst altitude loss. This mapping also
shows why state values can appear explicitly as decision variables even when a
deterministic rollout could reconstruct them from the controls.

Written this way, it may seem obvious that the decision variables are the controls $\mathbf{u}_t$. After all, in most intuitive descriptions of control, we think of choosing inputs to influence the system. But notice that in the program above, the entire state trajectory also appears as a set of variables, linked to the controls by the dynamics constraints. This is intentional: it reflects one way of writing the problem that makes the constraints explicit.

Why introduce $\mathbf{x}_t$ as decision variables if they can be simulated forward from the controls? Many readers hesitate here, and the question is natural: *If the model is deterministic and $\mathbf{x}_1$ is known, why not pick $\mathbf{u}_{1:T-1}$ and compute $\mathbf{x}_{2:T}$ on the fly?* That instinct leads to **single shooting**, a method we will return to shortly.

Already in this formulation, though, **the structure of the problem matters**. Ignoring it can make our life much harder. The reason is twofold:

* **Dimensionality grows with the horizon.** For a horizon of length $T$, the program has roughly $(T-1)(m+n)$ decision variables.
* **Temporal coupling.** Each control affects all future states and costs. The feasible set is not a simple box but a narrow manifold defined by the dynamics.

Together, these features explain why specialized methods exist and why the way we write the problem influences the algorithms we can use. Whether we keep states explicit or eliminate them through forward simulation determines the problem size, its conditioning, and the trade-offs between robustness and computational effort.

## Existence of Solutions and Optimality Conditions

Writing the problem as a nonlinear program does not ensure that a minimizer
exists or that a candidate is locally optimal. Which conditions supply those
two claims?

Now that we have the optimization problem written down, we can ask: does it always have a solution? And if so, how do we recognize one? These questions lead us to feasibility and optimality conditions.


### Existence of Solutions

Notice first that nothing in the problem statement required the dynamics

$$
\mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)
$$

to be stable. In fact, many problems of interest involve unstable systems; think of balancing a pole or steering a spacecraft. What matters is that the dynamics are **well defined**: given a state–control pair, the rule $\mathbf{f}_t$ produces a valid next state.

In continuous time, one usually requires $\mathbf{f}$ to be continuous (often Lipschitz continuous) in $\mathbf{x}$ so that the ODE has a unique solution on the horizon of interest. In discrete time, the requirement is lighter: we only need the update map to be well posed.

Existence also hinges on **feasibility**. A candidate control sequence must generate a trajectory that respects all constraints: the dynamics, any bounds on state and control, and any terminal requirements. If no such sequence exists, the feasible set is empty and the problem has no solution. This can happen if the constraints are overly strict, or if the system is uncontrollable from the given initial condition.


### Optimality Conditions

Assume the feasible set is nonempty. To characterize a point that is not only feasible but **locally optimal**, we use the Lagrange multiplier machinery from nonlinear programming. For a smooth problem

$$
\begin{aligned}
\min_{\mathbf{z}}\quad & F(\mathbf{z})\\
\text{s.t.}\quad & G(\mathbf{z})=\mathbf{0},\\
& H(\mathbf{z})\ge \mathbf{0},
\end{aligned}
$$

define the **Lagrangian**

$$
\mathcal{L}(\mathbf{z},\boldsymbol{\lambda},\boldsymbol{\mu})
= F(\mathbf{z})+\boldsymbol{\lambda}^{\top}G(\mathbf{z})+\boldsymbol{\mu}^{\top}H(\mathbf{z}),\qquad \boldsymbol{\mu}\ge \mathbf{0}.
$$

For an inequality system $H(\mathbf{z})\ge \mathbf{0}$ and a candidate point $\mathbf{z}$, the **active set** is

$$
\mathcal{A}(\mathbf{z}) \;=\; \{\, i \;:\; H_i(\mathbf{z})=0 \,\},
$$

while indices with $H_i(\mathbf{z})>0$ are **inactive**. Only active inequalities can carry positive multipliers.

We now make a **constraint qualification** assumption. In plain language, it says the constraints near the solution intersect in a regular way so that the feasible set has a well-defined tangent space and the multipliers exist. Algebraically, this amounts to a **full row rank** condition on the Jacobian of the equalities together with the active inequalities:

$$
\text{rows of }\big[\nabla G(\mathbf{z}^\star);\ \nabla H_{\mathcal{A}}(\mathbf{z}^\star)\big]\ \text{are linearly independent.}
$$

This is the **LICQ** (Linear Independence Constraint Qualification). In convex problems, **Slater's condition** (existence of a strictly feasible point) plays a similar role. You can think of these as the assumptions that let the linearized KKT equations be solvable; we do not literally invert that Jacobian, but the full-rank property is what would make such an inversion possible in principle.

Under such a constraint qualification, any local minimizer $\mathbf{z}^\star$ admits multipliers $(\boldsymbol{\lambda}^\star,\boldsymbol{\mu}^\star)$ that satisfy the **Karush–Kuhn–Tucker (KKT) conditions**:

$$
\begin{aligned}
&\text{stationarity:} && \nabla_{\mathbf{z}}\mathcal{L}(\mathbf{z}^\star,\boldsymbol{\lambda}^\star,\boldsymbol{\mu}^\star)=\mathbf{0},\\
&\text{primal feasibility:} && G(\mathbf{z}^\star)=\mathbf{0},\quad H(\mathbf{z}^\star)\ge \mathbf{0},\\
&\text{dual feasibility:} && \boldsymbol{\mu}^\star\ge \mathbf{0},\\
&\text{complementarity:} && \mu_i^\star\,H_i(\mathbf{z}^\star)=0\quad \text{for all } i.
\end{aligned}
$$

Only constraints that are **active** at $\mathbf{z}^\star$ can have $\mu_i^\star>0$; inactive ones have $\mu_i^\star=0$. The multipliers quantify marginal costs: $\lambda_j^\star$ measures how the optimal value changes if the $j$-th equality is relaxed, and $\mu_i^\star$ does the same for the $i$-th inequality. (If you prefer $h(\mathbf{z})\le 0$, signs flip accordingly.)

In our trajectory problems, $\mathbf{z}$ stacks state and control trajectories, $G$ enforces the dynamics, and $H$ collects bounds and path constraints. The equalities' multipliers act as **costates** or **shadow prices** for the dynamics. Writing the KKT system stage by stage yields the discrete-time Pontryagin principle, derived next. For convex programs these conditions are also sufficient.

*What fails without a CQ?* If the active gradients are dependent (for example duplicated or nearly parallel), the Jacobian loses rank; multipliers may then be nonunique or fail to exist, and the linearized equations become ill-posed. In transcribed trajectory problems this shows up as dependent dynamic constraints or redundant path constraints, which leads to fragile solver behavior.

The KKT conditions provide necessary conditions for a point to be a local minimizer of a constrained optimization problem. They consist of four parts: stationarity (the gradient of the Lagrangian vanishes), primal feasibility (constraints are satisfied), dual feasibility (inequality multipliers are nonnegative), and complementarity (inactive constraints have zero multipliers). The multipliers have an economic interpretation as marginal costs: they tell us how much the optimal value would change if we relaxed a constraint slightly. For convex problems, the KKT conditions are also sufficient, meaning any point satisfying them is globally optimal. In trajectory optimization, these conditions will reappear in structured form as the Pontryagin principle.

### From KKT to algorithms

The KKT system can be read as the first-order optimality conditions of a **saddle-point** problem. With equalities $G(\mathbf{z})=\mathbf{0}$ and inequalities $H(\mathbf{z})\ge \mathbf{0}$, define the Lagrangian

$$
\mathcal{L}(\mathbf{z},\boldsymbol{\lambda},\boldsymbol{\mu})
= F(\mathbf{z})+\boldsymbol{\lambda}^{\top}G(\mathbf{z})+\boldsymbol{\mu}^{\top}H(\mathbf{z}),\quad \boldsymbol{\mu}\ge \mathbf{0}.
$$

Optimality corresponds to a saddle: minimize in $\mathbf{z}$, maximize in $(\boldsymbol{\lambda},\boldsymbol{\mu})$ (with $\boldsymbol{\mu}$ constrained to the nonnegative orthant).

#### Primal–dual gradient dynamics (Arrow–Hurwicz)

The simplest algorithm mirrors this saddle structure by descending in the primal variables and ascending in the dual variables, with a projection for the inequalities:

$$
\begin{aligned}
\mathbf{z}^{k+1} &= \mathbf{z}^{k}-\alpha_k\big(\nabla F(\mathbf{z}^{k})+\nabla G(\mathbf{z}^{k})^{\top}\boldsymbol{\lambda}^{k}+\nabla H(\mathbf{z}^{k})^{\top}\boldsymbol{\mu}^{k}\big),\\[2mm]
\boldsymbol{\lambda}^{k+1} &= \boldsymbol{\lambda}^{k}+\beta_k\,G(\mathbf{z}^{k}),\\[1mm]
\boldsymbol{\mu}^{k+1} &= \Pi_{\ge 0}\!\big(\boldsymbol{\mu}^{k}+\beta_k\,H(\mathbf{z}^{k})\big).
\end{aligned}
$$

Here $\Pi_{\ge 0}$ is the projection onto $\{\boldsymbol{\mu}\ge 0\}$. In convex settings and with suitable step sizes, these iterates converge to a saddle point. In nonconvex problems (our trajectory optimizations after transcription), these updates are often used inside **augmented Lagrangian** or **penalty** frameworks to improve robustness, for example by replacing $\mathcal{L}$ with

$$
\mathcal{L}_\rho(\mathbf{z},\boldsymbol{\lambda},\boldsymbol{\mu})
= \mathcal{L}(\mathbf{z},\boldsymbol{\lambda},\boldsymbol{\mu})
+\tfrac{\rho}{2}\|G(\mathbf{z})\|^2
+\tfrac{\rho}{2}\|\min\{0,H(\mathbf{z})\}\|^2,
$$

which stabilizes the dual ascent when constraints are not yet well satisfied.

#### SQP as Newton on the KKT system (equality case)

With **only equality constraints** $G(\mathbf{z})=\mathbf{0}$, write first-order conditions

$$
\nabla_{\mathbf{z}}\mathcal{L}(\mathbf{z},\boldsymbol{\lambda})=\mathbf{0},
\qquad
G(\mathbf{z})=\mathbf{0},
\quad \text{where }\mathcal{L}=F+\boldsymbol{\lambda}^{\top}G.
$$

Applying Newton's method to this system gives the linear KKT solve

$$
\begin{bmatrix}
\nabla_{\mathbf{z}\mathbf{z}}^2\mathcal{L}(\mathbf{z}^k,\boldsymbol{\lambda}^k) & \nabla G(\mathbf{z}^k)^{\top}\\
\nabla G(\mathbf{z}^k) & 0
\end{bmatrix}
\begin{bmatrix}
\Delta \mathbf{z}\\ \Delta \boldsymbol{\lambda}
\end{bmatrix}
=
-
\begin{bmatrix}
\nabla_{\mathbf{z}}\mathcal{L}(\mathbf{z}^k,\boldsymbol{\lambda}^k)\\
G(\mathbf{z}^k)
\end{bmatrix}.
$$

This is exactly the step computed by **Sequential Quadratic Programming (SQP)** in the equality-constrained case: it is Newton's method on the KKT equations. For general problems with inequalities, SQP forms a **quadratic subproblem** by quadratically modeling $F$ with $\nabla_{\mathbf{z}\mathbf{z}}^2\mathcal{L}$ and linearizing the constraints, then solves that QP with line search or trust region. In least-squares-like problems one often uses **Gauss–Newton** (or a Levenberg–Marquardt trust region) as a positive-definite approximation to the Lagrangian Hessian.

In trajectory optimization, the KKT matrix inherits banded/sparse structure from the dynamics. Newton/SQP steps can be computed efficiently by exploiting this structure; in the special case of quadratic models and linearized dynamics, the QP reduces to an LQR solve along the horizon (this is the backbone of iLQR/DDP-style methods). Primal-dual updates provide simpler iterations and are easy to implement; augmented terms are typically needed to obtain stable progress when constraints couple stages.

The choice between methods depends on the context. Primal-dual gradients give lightweight iterations and are suited for warm starts or as inner loops with penalties. SQP/Newton gives rapid local convergence when close to a solution and LICQ holds; trust regions or line search help globalize convergence.


## Further Sources of Discrete-Time Optimal-Control Problems

Beyond sampled physical dynamics, which computations and continuous-time
models produce the same temporally coupled optimization structure?

The satellite planner begins from a deliberately discretized daily model.
Other problems are discrete because decisions naturally occur at stages, while
still others inherit a discrete transition from numerical integration or
program execution. The next two constructions make those latter connections
explicit.

### DOCPs Arising from the Discretization of Continuous-Time OCPs

Although many applications are natively discrete-time, it is also common to obtain a DOCP by discretizing a continuous-time formulation. Consider a system on $[0, T_c]$ given by

$$
\dot{\mathbf{x}}(t) = \mathbf{f}(t, \mathbf{x}(t), \mathbf{u}(t)), \qquad \mathbf{x}(0) = \mathbf{x}_0.
$$

Choose a step size $\Delta > 0$ and grid $t_k = k\,\Delta$. A one-step integration scheme induces a discrete map $\mathbf{F}_\Delta$ so that

$$
\mathbf{x}_{k+1} = \mathbf{F}_\Delta(\mathbf{x}_k, \mathbf{u}_k, t_k),\qquad k=0,\dots, T-1,
$$

where, for example, explicit Euler gives $\mathbf{F}_\Delta(\mathbf{x},\mathbf{u},t) = \mathbf{x} + \Delta\,\mathbf{f}(t,\mathbf{x},\mathbf{u})$. The resulting discrete-time optimal control problem takes the Bolza form with these induced dynamics:

$$
\begin{aligned}
\min_{\{\mathbf{x}_k,\mathbf{u}_k\}}\; & c_T(\mathbf{x}_T) + \sum_{k=0}^{T-1} c_k(\mathbf{x}_k,\mathbf{u}_k) \\
\text{s.t.}\; & \mathbf{x}_{k+1} - \mathbf{F}_\Delta(\mathbf{x}_k,\mathbf{u}_k, t_k) = 0,\quad k=0,\dots,T-1, \\
& \mathbf{x}_0 = \mathbf{x}_\mathrm{init}.
\end{aligned}
$$

### Programs as DOCPs and Differentiable Programming

It is often useful to view a computer program itself as a discrete-time dynamical system. Let the **program state** collect memory, buffers, and intermediate variables, and let the **control** represent inputs or tunable decisions at each step. A single execution step defines a transition map

$$
\mathbf{x}_{k+1}=\Phi_k(\mathbf{x}_k,\mathbf{u}_k),
$$

and a scalar objective (e.g., loss, error, runtime, energy) yields a DOCP:

$$
\min_{\{\mathbf{u}_k\}} \; c_T(\mathbf{x}_T)+\sum_{k=0}^{T-1} c_k(\mathbf{x}_k,\mathbf{u}_k)
\quad\text{s.t.}\quad \mathbf{x}_{k+1}=\Phi_k(\mathbf{x}_k,\mathbf{u}_k).
$$

In differentiable programming (e.g., JAX, PyTorch), the composed map $\Phi_{T-1}\circ\cdots\circ\Phi_0$ is differentiable, enabling reverse-mode automatic differentiation and efficient gradient-based trajectory optimization. When parts of the program are non-differentiable (discrete branches, simulators with events), DOCPs can still be solved using derivative-free or weak-gradient methods (eg. finite differences, SPSA, Nelder–Mead, CMA-ES, or evolutionary strategies) optionally combined with smoothing, relaxations, or stochastic estimators to navigate non-smooth regions.

#### Example: Offline Frequency Planning for Inference

The inference service from the [model-interface chapter](model-interfaces.md#inference-serving-as-a-controlled-system)
is also a program with a controllable execution rate. Its service-rate and
phase-power curves come from a measured NVIDIA L4 profile. The request and
queue trajectories below come from a simulator calibrated with those curves,
not from running each controller on the GPU. The scheduling rule is held fixed.
Each 0.1-second simulator step prioritizes decode or begins with a prefill chunk
capped at 512 tokens. If that chunk finishes early, the remaining service
budget may return to decode. Active decode receives alternating-step or
cache-pressure priority. This reduced interleaving model is not a reproduction
of the vLLM scheduler used for profiling. The control sequence specifies one
normalized GPU frequency for each second of a 60-second horizon,

$$
u_k=\frac{f_k-f_{\min}}{f_{\max}-f_{\min}}\in[0,1].
$$

An aggregate state collects queued prefill work, active decode work,
temperature, and the preceding frequency:

$$
x_k=(p_k,d_k,T_k,f_{k-1}),
\qquad
x_{k+1}=F_k(x_k,u_k,w_k).
$$

The disturbance $w_k$ contains the arrival times and prompt lengths predicted
for second $k$. Future output lengths remain hidden. The planner substitutes
the trace distribution's expected output length, while the request-level replay
uses each realized length only as a disturbance. Service and power in $F_k$
interpolate the committed profile whose provenance is displayed with the
result. The request-level simulator remains outside the optimizer and validates
the resulting schedule after the solve.

The 60-second workload is selected before optimization by a deterministic,
capacity-screened rule. The rule scans ten-second-aligned windows after load
normalization, discards any window whose forecast work exceeds the horizon's
maximum-clock service capacity, and requires an occupied burst that can be
moved twenty seconds earlier. Among the remaining windows, it chooses the
shift-eligible burst with the largest forecast work; total window work and then
earlier source time break ties. The selected source interval is $[890,950)$
seconds in the normalized trace and is rebased to start at zero. Its 48
requests require 51.936 seconds of forecast work at maximum clock, or 86.56%
of the horizon's capacity. The chosen burst occupies $[40,50)$ seconds after
rebasing and contains 31 requests. The shifted replay moves those requests to
$[20,30)$ seconds.

The offline problem uses the complete nominal arrival forecast. It introduces a
normalized service decision $\nu_k\in[\nu_{\min},1]$ and a nonnegative backlog
variable $B_{k+1}$. If $W_k$ is arriving work in seconds of highest-clock
service, their fluid balance is relaxed to

$$
\begin{aligned}
\underset{\nu_0,\ldots,\nu_{59},B_1,\ldots,B_{60}}
{\operatorname{minimize}}\quad
&\sum_{k=0}^{59}\left[\alpha_P\nu_k+20B_{k+1}\right]
+20B_{60}\\
\text{subject to}\quad
&B_{k+1}\geq B_k+W_k-\Delta t\,\nu_k,\qquad B_0=0,\\
&B_{k+1}\geq0,\qquad \nu_{\min}\leq\nu_k\leq1.
\end{aligned}
$$

The coefficient $\alpha_P$ is the slope of a linear interpolation between the
lowest- and highest-clock normalized power values. The backlog term prices
waiting throughout the horizon, while the additional terminal term discourages
postponing work beyond second 60. This is a linear program, solved with HiGHS.
The service decisions are mapped through the profiled service curve to
continuous frequencies. Execution rounds each frequency downward to the
nearest profiled requested clock, so the request-level validation includes the
actuator's finite action set. The replay reports the corresponding measured
median realized clock separately; a requested level and its realized clock need
not coincide under the experimental power cap.

This planning model deliberately omits request identities, phase-specific
queues, clock slew, and the nonlinear thermal state. The detailed replay
restores those variables and reports power, temperature, latency, and memory
violations. The optimizer therefore supplies an offline plan from a tractable
aggregate model, while the replay audits the assumptions used to obtain it.

The experiment asks what a one-shot frequency schedule gains from a perfect
nominal arrival and prompt-length forecast, and what it loses when that forecast
is wrong. Future output lengths remain uncertain in both cases. The same plan
is replayed twice. The nominal replay uses the forecast supplied to the
optimizer. The shifted replay uses the earlier arrival times defined above. No
reoptimization occurs after either replay starts.

```{code-cell} python
:tags: [remove-input]
:label: fig-inference-open-loop
:caption: The optimized clock schedule is computed once from the nominal 60-second request forecast. The shifted replay moves the selected work burst twenty seconds earlier while keeping the planned clocks fixed. Both request-level trajectories are simulations calibrated by measured NVIDIA L4 service-rate and phase-power curves. The playhead reveals only the executed trajectory prefix; the dashed schedule is the plan available at time zero.

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from inference_replay import render_serving_replay

display(HTML(render_serving_replay(
    Path("artifacts/inference_serving/textbook_results.json"),
    view="open_loop",
)))
```

:::{figure} _static/inference_serving/open-loop.svg
:label: fig-inference-open-loop-fallback
:class: pdf-fallback
:alt: Static comparison of the fixed offline frequency plan under nominal and shifted request arrivals.

Static comparison of the nominal and shifted-burst replays. The online book
adds playback and a controller selector.
:::

The table reports request-level results for the fixed offline schedule under
its nominal forecast and the shifted-burst disturbance. Both columns use the
same requests, controller parameters, and measured profile calibration.

```{code-cell} python
:tags: [remove-input]

import pandas as pd

open_loop_metrics = pd.read_csv(
    "artifacts/inference_serving/metrics_open_loop.csv"
).set_index("controller")
open_loop_metrics[
    [
        "mean_ttft_s",
        "p95_ttft_s",
        "matched_moved_burst_mean_ttft_s",
        "matched_moved_burst_p95_ttft_s",
        "energy_j",
        "peak_queued_requests_at_minimum_clock",
        "queued_requests_at_30_s",
        "power_violation_w",
        "thermal_violation_c",
    ]
].T.rename_axis("metric").round(3)
```

{download}`Download every open-loop metric (CSV) <artifacts/inference_serving/metrics_open_loop.csv>`

HiGHS reports an optimal solution for the stated linear program, with objective
11,205.68 in its weighted model units. Across all 48 requests, mean time to
first token rises from 16.35 seconds under the nominal arrival times to 23.23
seconds after the shift. The 95th percentile rises from 28.09 to 31.28 seconds.
For the 31 moved requests, the mean rises from 15.22 to 22.73 seconds. Their
95th percentile rises from 23.74 to 32.04 seconds. These changes are increases
of 7.52 and 8.30 seconds, respectively.

At 30 seconds, the nominal replay has no queued request, while the shifted
replay has 29. The peak queue while the clock is at its minimum also rises from
zero to 29 requests. Energy falls from 3,719.7 to 3,646.8 joules despite the
larger delays. Moving the burst changes which requests overlap and how long the
system remains in each phase, so an energy decrease does not imply an improved
service trajectory.

All requests eventually complete during the post-horizon drain. At the
60-second reporting horizon, 27 nominal requests and 22 shifted requests remain
unfinished. Both simulations reach a modeled phase power of 64.852 W and
exceed the configured 64.800 W power limit by 0.052 W. Neither simulation
records a thermal or KV-capacity violation.

The displayed table focuses on latency, energy, queueing, and constraint
violations. The downloadable CSV also reports energy per output token, time per
output token, unfinished work, and the full set of recorded diagnostics. The
nominal run tests the optimized trajectory under its own assumptions. The
shifted run tests sensitivity to one explicit forecast error. It does not establish
robustness to arbitrary arrivals, model error, or hardware throttling. Closing
that gap requires new information to alter future controls, which is the role
of feedback and receding-horizon optimization. The profile calibration is a
hardware measurement, while the controller comparison is a simulation of the
calibrated model. The reported latency, energy, and constraint outcomes are not
direct measurements from replaying this trace through vLLM.

:::{dropdown} Inspect the offline frequency optimization
```{literalinclude} code/inference_control.py
:language: python
:start-at: def optimize_open_loop
:end-before: def open_loop_clock_controller
:linenos:
```

{download}`Download the complete inference-control implementation <code/inference_control.py>`
:::



#### Example: Gradient Descent with Momentum as DOCP

To connect this lens to familiar practice, including hyperparameter optimization, treat the learning rate and momentum (or their schedules) as controls. Rather than fixing them a priori, we can optimize them as part of a trajectory optimization. The optimizer itself becomes the dynamical system whose execution we shape to minimize final loss.

Program: gradient descent with momentum on a quadratic loss. We fit $\boldsymbol{\theta}\in\mathbb{R}^p$ to data $(\mathbf{A},\mathbf{b})$ by minimizing

$$
\ell(\boldsymbol{\theta})=\tfrac{1}{2}\,\lVert\mathbf{A}\boldsymbol{\theta}-\mathbf{b}\rVert_2^2.
$$

The program maintains parameters $\boldsymbol{\theta}_k$ and momentum $\mathbf{m}_k$. Each iteration does:

1. compute gradient $ \mathbf{g}_k=\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_k)=\mathbf{A}^\top(\mathbf{A}\boldsymbol{\theta}_k-\mathbf{b})$
2. update momentum $ \mathbf{m}_{k+1}=\beta_k \, \mathbf{m}_k + \mathbf{g}_k$
3. update parameters $ \boldsymbol{\theta}_{k+1}=\boldsymbol{\theta}_k - \alpha_k \, \mathbf{m}_{k+1}$

State, control, and transition. Define the state $\mathbf{x}_k=\begin{bmatrix}\boldsymbol{\theta}_k\\ \mathbf{m}_k\end{bmatrix}\in\mathbb{R}^{2p}$ and the control $\mathbf{u}_k=\begin{bmatrix}\alpha_k\\ \beta_k\end{bmatrix}$. One program step is

$$
\Phi_k(\mathbf{x}_k,\mathbf{u}_k)=
\begin{bmatrix}
\boldsymbol{\theta}_k - \alpha_k\!\left(\beta_k \, \mathbf{m}_k + \mathbf{A}^\top(\mathbf{A}\boldsymbol{\theta}_k-\mathbf{b})\right)\\[2mm]
\beta_k \, \mathbf{m}_k + \mathbf{A}^\top(\mathbf{A}\boldsymbol{\theta}_k-\mathbf{b})
\end{bmatrix}.
$$

Executing the program for $T$ iterations gives the trajectory

$$
\mathbf{x}_{k+1}=\Phi_k(\mathbf{x}_k,\mathbf{u}_k),\quad k=0,\dots,T-1,\qquad
\mathbf{x}_0=\begin{bmatrix}\boldsymbol{\theta}_0\\ \mathbf{m}_0\end{bmatrix}.
$$

Objective as a DOCP. Choose terminal cost $c_T(\mathbf{x}_T)=\ell(\boldsymbol{\theta}_T)$ and (optionally) stage costs $c_k(\mathbf{x}_k,\mathbf{u}_k)=\rho_\alpha \, \alpha_k^2+\rho_\beta\,(\beta_k- \bar\beta)^2$. The program-as-control problem is

$$
\min_{\{\alpha_k,\beta_k\}} \; \ell(\boldsymbol{\theta}_T)+\sum_{k=0}^{T-1}\big(\rho_\alpha \, \alpha_k^2+\rho_\beta\,(\beta_k-\bar\beta)^2\big)
\quad\text{s.t.}\quad \mathbf{x}_{k+1}=\Phi_k(\mathbf{x}_k,\mathbf{u}_k).
$$

Backpropagation = reverse-time costate recursion. Because $\Phi_k$ is differentiable, reverse-mode AD computes $\nabla_{\mathbf{u}_{0:T-1}} \big(c_T+\sum c_k\big)$ by propagating a costate $\boldsymbol{\lambda}_k=\partial \mathcal{J}/\partial \mathbf{x}_k$ backward:

$$
\boldsymbol{\lambda}_T=\nabla_{\mathbf{x}_T} c_T,\qquad
\boldsymbol{\lambda}_k=\nabla_{\mathbf{x}_k} c_k + \left(\nabla_{\mathbf{x}_k}\Phi_k\right)^\top \boldsymbol{\lambda}_{k+1},
$$

and the gradients with respect to controls are

$$
\nabla_{\mathbf{u}_k}\mathcal{J}=\nabla_{\mathbf{u}_k} c_k + \left(\nabla_{\mathbf{u}_k}\Phi_k\right)^\top \boldsymbol{\lambda}_{k+1}.
$$

Unrolling a tiny horizon ($T=3$) to see the composition:

$$
\begin{aligned}
\mathbf{x}_1&=\Phi_0(\mathbf{x}_0,\mathbf{u}_0),\\
\mathbf{x}_2&=\Phi_1(\mathbf{x}_1,\mathbf{u}_1),\\
\mathbf{x}_3&=\Phi_2(\mathbf{x}_2,\mathbf{u}_2),\qquad
\mathcal{J}=c_T(\mathbf{x}_3)+\sum_{k=0}^{2} c_k(\mathbf{x}_k,\mathbf{u}_k).
\end{aligned}
$$

What if the program branches? Suppose we insert a "skip-small-gradients" branch

$$
\boldsymbol{\theta}_{k+1}=\boldsymbol{\theta}_k - \alpha_k\,\mathbf{m}_{k+1}\,\mathbf{1}\{ \lVert\mathbf{g}_k\rVert>\tau\},
$$

which is non-differentiable because of the indicator. The DOCP view still applies, but gradients are unreliable. Two practical paths: smooth the branch (e.g., replace $\mathbf{1}\{\cdot\}$ with $\sigma((\lVert\mathbf{g}_k\rVert-\tau)/\epsilon)$ for small $\epsilon$) and use autodiff; or go derivative-free on $\{\alpha_k,\beta_k,\tau\}$ (e.g., SPSA or CMA-ES) while keeping the inner dynamics exact.

## Variants: Lagrange and Mayer Problems

How does moving cost between running and terminal terms change the
representation without changing the underlying control problem?

The Bolza form is general enough to cover most situations, but two common special cases deserve mention:

* **Lagrange problem (no terminal cost)**
  If the objective only accumulates stage costs:

$$
\min_{\mathbf{u}_{1:T-1}} \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t).
$$

Example: *Energy minimization for a delivery drone*. The concern is total battery use, regardless of the final position.

* **Mayer problem (terminal cost only)**
  If the objective depends only on the final state:

$$
\min_{\mathbf{u}_{1:T-1}} c_T(\mathbf{x}_T).
$$

Example: *Satellite orbital transfer*. The only goal is to reach a specified orbit, no matter the fuel spent along the way.

These distinctions matter when deriving optimality conditions, but conceptually they fit in the same framework: the system evolves over time, and we choose controls to shape the trajectory.

### Reducing to Mayer Form by State Augmentation

Although Bolza, Lagrange, and Mayer problems look different, they are equivalent in expressive power. Any problem with running costs can be rewritten as a Mayer problem (one whose objective depends only on the final state) through a simple trick: **augment the state with a running sum of costs**.

The idea is straightforward. Introduce a new variable, $y_t$, that keeps track of the cumulative cost so far. At each step, we update this running sum along with the system state:

$$
\tilde{\mathbf{x}}_{t+1} =
\begin{pmatrix}
\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t) \\
y_t + c_t(\mathbf{x}_t,\mathbf{u}_t)
\end{pmatrix},
$$

where $\tilde{\mathbf{x}}_t = (\mathbf{x}_t, y_t)$. The terminal cost then becomes:

$$
\tilde{c}_T(\tilde{\mathbf{x}}_T) = c_T(\mathbf{x}_T) + y_T.
$$

The overall effect is that the explicit sum $\sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t)$ disappears from the objective and is captured implicitly by the augmented state. This lets us write every optimal control problem in Mayer form.

This reduction serves two purposes. First, it often simplifies mathematical derivations, as we will see later when deriving necessary conditions. Second, it can streamline algorithmic implementation: instead of writing separate code paths for Mayer, Lagrange, and Bolza problems, we can reduce everything to one canonical form. That said, this unified approach is not always best in practice. Specialized formulations can sometimes be more efficient computationally, especially when the running cost has simple structure.


The unifying theme is that a DOCP may look like a generic NLP on paper, but its structure matters. Ignoring that structure often leads to impractical solutions, whereas formulations that expose sparsity and respect temporal coupling allow modern solvers to scale effectively. In the following sections, we will examine how these choices play out in practice through single shooting, multiple shooting, and collocation methods, and why different formulations strike different trade-offs between robustness and computational effort.

## Summary and Outlook

A finite horizon converts a controlled dynamical model into a nonlinear
program over states and actions. The differential-drag and inference examples
also delimit its guarantee: feasibility and performance apply to the stated
initial condition, dynamics, and disturbance forecast. Bolza, Lagrange, and
Mayer forms change the bookkeeping without changing the admissible action
sequences.

The KKT conditions characterize a local solution of the resulting nonlinear
program, but their raw form hides the direction of time. Can the multipliers be
organized into a backward recursion that matches the forward state dynamics?
[Adjoints and the discrete-time Pontryagin principle](discrete-time-pmp.md)
provide that organization.

## Exercises

:::{exercise} KKT conditions for a simple DOCP
:label: ex-trajectories-kkt

Consider a two-stage optimal control problem with scalar state and control:

$$
\min_{u_1, u_2} \quad x_3^2 + u_1^2 + u_2^2
$$

subject to:

$$
x_2 = x_1 + u_1, \quad x_3 = x_2 + u_2, \quad x_1 = 1.
$$

**(a)** Write out the Lagrangian with multipliers $\lambda_2, \lambda_3$ for the dynamics constraints.

**(b)** Derive the KKT conditions (stationarity with respect to $x_2, x_3, u_1, u_2$).

**(c)** Solve the system to find the optimal controls $u_1^\star, u_2^\star$ and the optimal cost.

:::

:::{solution} ex-trajectories-kkt
:class: dropdown

The Lagrangian is $\mathcal{L} = x_3^2 + u_1^2 + u_2^2 + \lambda_2(x_1 + u_1 - x_2) + \lambda_3(x_2 + u_2 - x_3)$.

Stationarity conditions:
- $\partial \mathcal{L}/\partial x_2 = -\lambda_2 + \lambda_3 = 0 \Rightarrow \lambda_2 = \lambda_3$
- $\partial \mathcal{L}/\partial x_3 = 2x_3 - \lambda_3 = 0 \Rightarrow \lambda_3 = 2x_3$
- $\partial \mathcal{L}/\partial u_1 = 2u_1 + \lambda_2 = 0 \Rightarrow u_1 = -\lambda_2/2$
- $\partial \mathcal{L}/\partial u_2 = 2u_2 + \lambda_3 = 0 \Rightarrow u_2 = -\lambda_3/2$

Substituting and using the dynamics: $x_3 = 1 + u_1 + u_2 = 1 - \lambda_2/2 - \lambda_3/2 = 1 - \lambda_3$. Combined with $\lambda_3 = 2x_3$, we get $x_3 = 1 - 2x_3$, so $x_3^\star = 1/3$, $\lambda_3^\star = 2/3$, $u_1^\star = u_2^\star = -1/3$. The optimal cost is $(1/3)^2 + 2(1/3)^2 = 1/3$.
:::

---

:::{exercise} Lagrange to Mayer conversion
:label: ex-trajectories-mayer

Consider the Lagrange problem:

$$
\min_{u_{1:T-1}} \sum_{t=1}^{T-1} c_t(x_t, u_t) \quad \text{s.t.} \quad x_{t+1} = f_t(x_t, u_t), \quad x_1 = x_0.
$$

**(a)** Introduce an auxiliary state $y_t$ that accumulates the running cost. Write the augmented dynamics and the equivalent Mayer problem.

**(b)** What is the initial condition for $y_1$? What is the terminal cost in Mayer form?

**(c)** Verify that the two formulations have the same optimal control sequence.

:::

:::{solution} ex-trajectories-mayer
:class: dropdown

Define the augmented state $\tilde{x}_t = (x_t, y_t)$ with dynamics:

$$
\tilde{x}_{t+1} = \begin{pmatrix} f_t(x_t, u_t) \\ y_t + c_t(x_t, u_t) \end{pmatrix}.
$$

Initial condition: $y_1 = 0$. Terminal cost: $\tilde{c}_T(\tilde{x}_T) = y_T$.

The objective $y_T = \sum_{t=1}^{T-1} c_t(x_t, u_t)$ is identical to the Lagrange objective, so the optimal controls are the same.
:::

---

:::{exercise} When LICQ fails
:label: ex-trajectories-licq

Consider a DOCP where the state must satisfy $x_T = 0$ and also $x_T \leq 0$ at the terminal time.

**(a)** At a feasible point with $x_T = 0$, which constraints are active?

**(b)** Write the gradients of the active constraints with respect to $x_T$. Are they linearly independent?

**(c)** Explain why this violates LICQ and what consequences this might have for the KKT multipliers.

:::

:::{solution} ex-trajectories-licq
:class: dropdown

Both constraints $h(x_T) = x_T = 0$ and $g(x_T) = -x_T \leq 0$ are active when $x_T = 0$. The gradients are $\nabla h = 1$ and $\nabla g = -1$, which are parallel (linearly dependent). LICQ fails because the constraint gradients do not span independent directions. Consequence: the multipliers $\lambda$ (for equality) and $\mu$ (for inequality) may not be unique—any combination satisfying $\lambda - \mu = c$ for a fixed $c$ could work. This leads to numerical difficulties in optimization algorithms.
:::

---

:::{exercise} Differential-drag derivation and model audit
:label: ex-trajectories-differential-drag

Return to the three-satellite planner. Let
$\Delta\sigma=1/B_\mathrm{high}-1/B_\mathrm{low}$ and
$a_0=R_\mathrm{E}+h_0$, with all quantities converted to consistent SI units
before substitution.

**(a)** Starting from

$$
\dot a=-\rho_0\sigma\sqrt{\mu a},
\qquad
n(a)=\sqrt{\frac{\mu}{a^3}},
$$

derive the extra daily altitude loss $d$ and angular acceleration $\alpha$
caused by full-time high drag after linearization at $a_0$.

**(b)** Unroll the daily model and show that

$$
\omega_{i,N}=\alpha\sum_{k=0}^{N-1}u_{i,k},
\qquad
\varphi_{i,N}
=\varphi_{i,0}
+\alpha\sum_{k=0}^{N-1}
\left(N-k-\frac12\right)u_{i,k}.
$$

Explain why equal total high-drag exposure can produce equal terminal rates
while different command timing still produces different terminal phases.

**(c)** The nominal LP residual is below $10^{-10}$, but the nonlinear replay
misses a cyclic gap by more than $10^\circ$. Identify which claim each number
supports. Why would tightening the LP tolerance not repair the nonlinear miss?

**(d)** Propose a validation or feedback experiment that distinguishes
integration error from model mismatch without changing the already executed
part of the command.

:::

:::{solution} ex-trajectories-differential-drag
:class: dropdown

The extra semimajor-axis rate is

$$
\Delta\dot a
=-\rho_0\Delta\sigma\sqrt{\mu a_0}.
$$

Thus $d=-\Delta\dot a\,\Delta t$, with a metre-to-kilometre conversion. Since
$dn/da=-3n/(2a)$,

$$
\dot n
=\frac{3n_0}{2a_0}
\rho_0\Delta\sigma\sqrt{\mu a_0},
$$

and $\alpha=\dot n\,\Delta t^2$ after converting radians to degrees. Repeated
substitution in the rate equation gives the first expression in part (b).
Each input contributes half of its new rate on its own day and its full rate
on every later day, which gives the phase weight $N-k-\tfrac12$.

Equal sums $\sum_k u_{i,k}$ give equal final rate changes and equal nominal
extra altitude losses. Moving an equal amount of high drag earlier gives it a
larger phase weight, so timing can separate the satellites without leaving a
large terminal rate difference.

The small LP residual certifies that the numerical solution satisfies the
linear program. The cyclic-gap miss measures predictive failure under the
declared nonlinear plant. Solver tolerance cannot remove omitted density and
altitude dependence. First halve the RK4 step to audit integration error. To
address model mismatch during operation, observe the current orbital state and
replan only the remaining command, as in receding-horizon control.
:::

## Self-checks

:::{exercise} Predict the active bound
:label: ex-trajectories-check-3

In a minimum-time braking problem with bounded deceleration, which control bound do you expect to be active before the vehicle stops?
:::

:::{solution} ex-trajectories-check-3
:class: dropdown

The maximum braking bound should be active almost everywhere: delaying or reducing braking cannot shorten the stopping time when the terminal position and zero velocity are fixed.
:::

:::{exercise} Forecast shift
:label: ex-trajectories-check-4

The offline inference controller receives the exact total work over 60 seconds,
but the largest burst arrives twenty seconds earlier than planned. Explain why
matching total work does not preserve the planned queue and temperature
trajectory.
:::

:::{solution} ex-trajectories-check-4
:class: dropdown

The state depends on when work arrives. Earlier work increases the queue and
power demand before the clocks chosen for that work are scheduled, and the
thermal state carries this timing difference into later seconds.
:::
