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

# Discrete-Time Trajectory Optimization

The models introduced in the previous chapter predict how actions change a
system's state. Trajectory optimization adds an objective and constraints, then
selects the actions that produce a desirable state sequence. For a fixed
discrete horizon, the states and controls form a finite vector, so the planning
problem can be written as a nonlinear program.

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
[](dynamics.md).
```

## A Motivating Example: Phasing Three Satellites with Differential Drag

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
replanning](mpc.md#closing-the-loop-by-replanning) will replace the immutable
schedule by controls that can change when new state measurements arrive.

The example already contains the ingredients of a **discrete-time optimal
control problem** (DOCP): a state $x_{i,k}$, a bounded control $u_{i,k}$, a
transition map, terminal constraints, and an objective accumulated over a
finite horizon. We now formalize that structure.

## Discrete-Time Optimal Control Problems (DOCPs)

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

The inference service from the [modeling chapter](dynamics.md#inference-serving-as-a-controlled-system)
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

# Numerical Methods for Solving DOCPs

Before we discuss specific algorithms, it is useful to clarify the goal: we want to recast a discrete-time optimal control problem as a standard nonlinear program (NLP). Collect all decision variables (states, controls, and any auxiliary variables) into a single vector $\mathbf{z}\in\mathbb{R}^{n_z}$ and write

$$
\begin{aligned}
\min_{\mathbf{z}\in\mathbb{R}^{n_z}} \quad & F(\mathbf{z}) \\
\text{s.t.} \quad & G(\mathbf{z}) = 0, \\
& H(\mathbf{z}) \ge 0,
\end{aligned}
$$

with maps $F:\mathbb{R}^{n_z}\to\mathbb{R}$, $G:\mathbb{R}^{n_z}\to\mathbb{R}^{r_e}$, and $H:\mathbb{R}^{n_z}\to\mathbb{R}^{r_h}$. In optimal control, $G$ typically encodes dynamics and boundary conditions, while $H$ captures path and box constraints. 

There are multiple ways to arrive at (and benefit from) this NLP:

* Simultaneous (direct transcription / full discretization): keep all states and controls as variables and impose the dynamics as equality constraints. This is straightforward and exposes sparsity, but the problem can be large unless solver-side techniques (e.g., condensing) are exploited.
* Sequential (recursive elimination / single shooting): eliminate states by forward propagation from the initial condition, leaving controls as the main decision variables. This reduces dimension and constraints, but can be sensitive to initialization and longer horizons.
* Multiple shooting: introduce state variables at segment boundaries and enforce continuity between simulated segments. This compromises between size and conditioning and is often more robust than pure single shooting.

The next sections work through these formulations, starting with simultaneous methods, then sequential methods, and finally multiple shooting, before discussing how generic NLP solvers and specialized algorithms leverage the resulting structure in practice.

## Simultaneous Methods

In the simultaneous (also called direct transcription or full discretization) approach, we keep the entire trajectory explicit and enforce the dynamics as equality constraints. Starting from the Bolza DOCP,

$$
\min_{\{\mathbf{x}_t,\mathbf{u}_t\}}\; c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t)
\quad\text{s.t.}\quad \mathbf{x}_{t+1} - \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t) = 0,\; t=1,\dots,T-1,
$$

collect all variables into a single vector

$$
\mathbf{z} := \begin{bmatrix}
\mathbf{x}_1^\top & \cdots & \mathbf{x}_T^\top & \mathbf{u}_1^\top & \cdots & \mathbf{u}_{T-1}^\top
\end{bmatrix}^\top \in \mathbb{R}^{n_z}.
$$

Path constraints typically apply only at selected times. Let $\mathscr{E}$ index additional equality constraints $g_i$ and $\mathscr{I}$ index inequality constraints $h_i$. For each constraint $i$, define the set of time indices $K_i \subseteq \{1,\dots,T\}$ where it is enforced (e.g., terminal constraints use $K_i = \{T\}$). The simultaneous transcription is the NLP

$$
\begin{aligned}
\min_{\mathbf{z}}\quad & F(\mathbf{z}) := c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t) \\
\text{s.t.}\quad & G(\mathbf{z}) = \begin{bmatrix}
\big[\, g_i(\mathbf{x}_k,\mathbf{u}_k) \big]_{i\in\mathscr{E},\, k\in K_i} \\
\big[\, \mathbf{x}_{t+1} - \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t) \big]_{t=1: T-1} \\
\mathbf{x}_1 - \mathbf{x}_\mathrm{init}
\end{bmatrix} = \mathbf{0}, \\
& H(\mathbf{z}) = \big[\, h_i(\mathbf{x}_k,\mathbf{u}_k) \big]_{i\in\mathscr{I},\, k\in K_i} \; \ge \; \mathbf{0},
\end{aligned}
$$

optionally with simple bounds $\mathbf{x}_{\mathrm{lb}} \le \mathbf{x}_t \le \mathbf{x}_{\mathrm{ub}}$ and $\mathbf{u}_{\mathrm{lb}} \le \mathbf{u}_t \le \mathbf{u}_{\mathrm{ub}}$ folded into $H$ or provided to the solver separately. For notational convenience, some constraints may not depend on $\mathbf{u}_k$ at times in $K_i$; the indexing still helps specify when each condition is active.

This direct transcription is attractive because it is faithful to the model and exposes sparsity. The Jacobian of $G$ has a block bi-diagonal structure induced by the dynamics, and the KKT matrix is sparse and structured. These properties are exploited by interior-point and SQP methods. The trade-off is size: with state dimension $n$ and control dimension $m$, the decision vector has $(T\!\cdot\!n) + ((T\!-
1)\cdot m)$ entries, and there are roughly $(T\!-
1)\cdot n$ dynamic equalities plus any path and boundary conditions. Techniques such as partial or full condensing eliminate state variables to reduce the equality set (at the cost of denser matrices), while keeping states explicit preserves sparsity and often improves robustness on long horizons and in the presence of state constraints.

Compared to alternatives, simultaneous methods avoid the long nonlinear dependency chains of single shooting and make it easier to impose state/path constraints. They can, however, demand more memory and per-iteration linear algebra, so practical performance hinges on exploiting sparsity and good initialization.

The same logic applies when selecting an optimizer. For small-scale problems, it is common to rely on general-purpose routines such as those in `scipy.optimize.minimize`. Derivative-free methods like Nelder–Mead require no gradients but scale poorly as dimensionality increases. Quasi-Newton schemes such as BFGS work well for moderate dimensions and can approximate gradients by finite differences, while large-scale trajectory optimization often calls for gradient-based constrained solvers such as interior-point or sequential quadratic programming methods that can exploit sparse Jacobians and benefit from automatic differentiation. Stochastic techniques, including genetic algorithms, simulated annealing, or particle swarm optimization, occasionally appear when gradients are unavailable, but their cost grows rapidly with dimension and they are rarely competitive for structured optimal control problems.

### Example: Nonlinear Cart-Pole Swing-Up

A cart carries a rigid pendulum whose angle is measured from the upright vertical. The cart can accelerate horizontally, but no actuator applies torque directly at the pendulum joint. Starting from the stable downward configuration, the task is to move the base so that the pendulum arrives upright while the cart returns near the center of a finite rail.

Let the state be $\mathbf{x}=(p,v,\theta,\omega)$, where $p$ and $v$ are the cart position and velocity, and $\theta$ and $\omega$ are the pendulum angle and angular velocity. A commanded horizontal acceleration $u$ produces the nonlinear dynamics

$$
\dot p = v, \qquad
\dot v = u, \qquad
\dot\theta = \omega, \qquad
\dot\omega = \frac{g}{\ell}\sin\theta
                 - \frac{u}{\ell}\cos\theta
                 - b\omega.
$$

The factor $-u\cos\theta/\ell$ identifies the action channel. Horizontal base motion couples into angular acceleration, and its sign and magnitude depend on the current configuration. A black-box optimizer could evaluate these equations without inspecting that term, but the term explains why the cart must first move away from its eventual resting position to build pendulum energy.

The numerical experiment uses a $4.5$ s horizon with $N=30$ zero-order-hold controls and a step size $h=0.15$ s. Fourth-order Runge--Kutta integration defines the discrete map $\mathbf{x}_{k+1}=F_h(\mathbf{x}_k,u_k)$. Both numerical formulations solve the same problem:

$$
\begin{aligned}
\min_{\mathbf{x}_{0:N},u_{0:N-1}}\quad
& h\sum_{k=0}^{N-1}\Bigl[
0.05p_k^2+0.01v_k^2+0.25(1-\cos\theta_k)
+0.01\omega_k^2+0.004u_k^2\Bigr] \\
& {}+20p_N^2+5v_N^2+120(1-\cos\theta_N)+12\omega_N^2 \\
\text{subject to}\quad
& \mathbf{x}_{k+1}=F_h(\mathbf{x}_k,u_k), \\
& \mathbf{x}_0=(0,0,\pi,0), \\
& |p_k|\leq 2.4,\quad |v_k|\leq 4,\quad
  |\omega_k|\leq 12,\quad |u_k|\leq 8.
\end{aligned}
$$

The periodic penalty $1-\cos\theta$ assigns the same terminal cost to angles that differ by a full revolution. Position and velocity penalties still require the cart to finish near rest, so rotating the pole through the top is not enough by itself.

Direct transcription retains all $31$ states and $30$ controls. It therefore optimizes over $154$ scalar variables and imposes $124$ scalar equalities, including the initial condition and one four-dimensional dynamics equation per step. The equality Jacobian is block banded because the residual at step $k$ depends only on $(\mathbf{x}_k,u_k,\mathbf{x}_{k+1})$.

The small demonstration below passes that Jacobian to SLSQP as a dense array. A large-scale direct solver would instead store and factor the same block-banded pattern sparsely. The formulation exposes sparsity, but exploiting it is a separate implementation choice.

```{admonition} Prediction before computation
:class: tip

Single shooting will reduce the decision vector from $154$ variables to $30$. Before examining the matched runs below, predict whether the smaller program must be easier to optimize. Identify which constraints become less explicit after the states are eliminated.
```


## Sequential Methods

The previous section showed how a discrete-time optimal control problem can be solved by treating all states and controls as decision variables and enforcing the dynamics as equality constraints. This produces a nonlinear program that can be passed to solvers such as `scipy.optimize.minimize` with the SLSQP method. For short horizons, this approach is straightforward and works well; the code stays close to the mathematical formulation.

It also has a real advantage: by keeping the states explicit and imposing the dynamics through constraints, we anchor the trajectory at multiple points. This extra structure helps stabilize the optimization, especially for long horizons where small deviations in early steps can otherwise propagate and cause the optimizer to drift or diverge. In that sense, this formulation is better conditioned and more robust than approaches that treat the dynamics implicitly.

The drawback is scale. As the horizon grows, the number of variables and constraints grows with it, and all are coupled by the dynamics. Each iteration of a sequential quadratic programming (SQP) or interior-point method requires building and factorizing large Jacobians and Hessians. These methods have been embedded in reinforcement learning and differentiable programming pipelines, through implicit layers or differentiable convex solvers, but the cost is significant. They remain serial, rely on repeated linear algebra factorizations, and are difficult to parallelize efficiently. When thousands of such problems must be solved inside a learning loop, the overhead becomes prohibitive.

This motivates an alternative that aligns with the computational model of machine learning. For deterministic dynamics, the equality constraints can be eliminated by making the states implicit. Instead of solving for both states and controls, we fix the initial state and roll the system forward under a candidate control sequence. State constraints can remain, but they become nonlinear functions of the entire preceding control sequence. This is the essence of **single shooting**.

The term "shooting" comes from the idea of *aiming and firing* a trajectory from the initial state: you pick a control sequence, integrate (or step) the system forward, and see where it lands. If the final state misses the target, you adjust the controls and try again: like adjusting the angle of a shot until it hits the mark. It is called **single** shooting because we compute the entire trajectory in one pass from the starting point, without breaking it into segments. Later, we will contrast this with **multiple shooting**, where the horizon is divided into smaller arcs that are optimized jointly to improve stability and conditioning.

The analogy with deep learning is also immediate: the control sequence plays the role of parameters, the rollout is a forward pass, and the cost is a scalar loss. Gradients can be obtained with reverse-mode automatic differentiation. In the single shooting formulation of the DOCP, the constrained program

$$
\min_{\mathbf{x}_{1:T},\,\mathbf{u}_{1:T-1}} J(\mathbf{x}_{1:T},\mathbf{u}_{1:T-1})
\quad\text{s.t.}\quad 
\mathbf{x}_{t+1}=\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)
$$

collapses to

$$
\min_{\mathbf{u}_{1:T-1}}\;
c_T\!\bigl(\boldsymbol{\phi}_{T}(\mathbf{u}, \mathbf{x}_1)\bigr)
+\sum_{t=1}^{T-1} c_t\!\bigl(\boldsymbol{\phi}_{t}(\mathbf{u}, \mathbf{x}_1), \mathbf{u}_t\bigr),
\quad\text{s.t.}\quad
\mathbf{h}_t\!\bigl(\boldsymbol{\phi}_{t}(\mathbf{u},\mathbf{x}_1),\mathbf{u}_t\bigr)\geq 0,
\quad
\mathbf{u}_{\mathrm{lb}}\le\mathbf{u}_{t}\le\mathbf{u}_{\mathrm{ub}}.
$$

Here $\boldsymbol{\phi}_t$ denotes the state reached at time $t$ by recursively applying the dynamics to the previous state and current control. This recursion can be written as

$$
\boldsymbol{\phi}_{t+1}(\mathbf{u},\mathbf{x}_1)=
\mathbf{f}_{t}\!\bigl(\boldsymbol{\phi}_{t}(\mathbf{u},\mathbf{x}_1),\mathbf{u}_t\bigr),\qquad
\boldsymbol{\phi}_{1}=\mathbf{x}_1.
$$

Concretely, here is JAX-style pseudocode for defining `phi(u, x_0, t)` using `jax.lax.scan` with a zero-based time index:

```python
def phi(u_seq, x0, t):
    """Return \phi_t(u, x0) with 0-based t (\phi_0 = x0).

    u_seq: controls of length T (or T-1); only first t entries are used
    x0: initial state at time 0
    t: integer >= 0
    """
    if t <= 0:
        return x0

    def step(carry, u):
        x, t_idx = carry
        x_next = f(x, u, t_idx)
        return (x_next, t_idx + 1), None

    (x_t, _), _ = lax.scan(step, (x0, 0), u_seq[:t])
    return x_t
```


The pattern mirrors an RNN unroll: starting from an initial state ($\mathbf{x}^\star_1$) and a sequence of controls ($\mathbf{u}^*_{1:T-1}$), we propagate forward through the dynamics, updating the state at each step and accumulating cost along the way. This structural similarity is why single shooting often feels natural to practitioners with a deep learning background: the rollout is a forward pass, and gradients propagate backward through time exactly as in backpropagation through an RNN.

Algorithmically:

```{prf:algorithm} Single Shooting: Forward Unroll
:label: single-shooting-forward-unroll

**Inputs**: Initial state $\mathbf{x}_1$, horizon $T$, control bounds $\mathbf{u}_{\mathrm{lb}}, \mathbf{u}_{\mathrm{ub}}$, dynamics $\mathbf{f}_t$, costs $c_t$

**Output**: Optimal control sequence $\mathbf{u}^*_{1:T-1}$

1. Initialize $\mathbf{u}_{1:T-1}$ within bounds  
2. Define `ComputeTrajectoryAndCost`($\mathbf{u}, \mathbf{x}_1$):
    - $\mathbf{x} \leftarrow \mathbf{x}_1$, $J \leftarrow 0$
    - For $t = 1$ to $T-1$:
        - $J \leftarrow J + c_t(\mathbf{x}, \mathbf{u}_t)$
        - $\mathbf{x} \leftarrow \mathbf{f}_t(\mathbf{x}, \mathbf{u}_t)$
    - $J \leftarrow J + c_T(\mathbf{x})$
    - Return $J$
3. Solve $\min_{\mathbf{u}} J(\mathbf{u})$ subject to the control bounds and any state constraints evaluated along the rollout
4. Return $\mathbf{u}^*_{1:T-1}$
```

In JAX or PyTorch, this loop can be compiled and differentiated automatically. The control sequence plays the role of trainable parameters, while the simulated trajectory is the forward computation. Reverse-mode differentiation of that computation gives $\nabla J(\mathbf{u})$.

Single shooting is attractive for its simplicity and compatibility with differentiable programming, but it has limitations. Early controls influence every later state through a long product of dynamics Jacobians. This can make gradients poorly conditioned over long horizons. State constraints also lose their local sparse representation because each constrained state depends on all earlier controls. Formulations that keep selected states explicit, such as multiple shooting or collocation, shorten these dependency chains.

### Matched Swing-Up Comparison

The direct-transcription and single-shooting implementations below use the cart-pole problem stated above without changing the model, cost, limits, horizon, initial control guess, or nonlinear-programming solver. Only the decision variables and the representation of the dynamics differ.

```{code-cell} python
:tags: [remove-cell]

from pathlib import Path
import sys

code_directory = Path.cwd() / "code"
if str(code_directory) not in sys.path:
    sys.path.insert(0, str(code_directory))

from cartpole_control import (
    SwingUpScenario,
    format_swingup_metrics,
    make_open_loop_perturbation_figure,
    make_swingup_animation,
    make_swingup_figure,
    replay_open_loop_with_disturbance,
    solve_swingup_comparison,
)

swingup_scenario = SwingUpScenario()
swingup_results = solve_swingup_comparison(swingup_scenario)
```

```{code-cell} python
:label: fig-cartpole-formulations
:caption: Direct transcription and single shooting solve the same nonlinear cart-pole problem from the same initialization. Both reach the upright configuration and respect the matched limits, but they converge to different local solutions. Direct transcription retains 154 scalar variables and 124 local dynamics equalities; single shooting retains only the 30 controls and reconstructs every state by forward simulation.
:tags: [remove-input]

print(format_swingup_metrics(swingup_results))
make_swingup_figure(swingup_results, swingup_scenario)
```

Both solvers produce a successful open-loop swing-up. The direct formulation reaches a lower objective in this fixed run, while single shooting uses a much smaller decision vector. This numerical outcome does not establish that direct transcription always finds better solutions. It exposes a concrete trade-off: eliminating variables shortens the program but lengthens the dependency from an early control to the terminal cost and later constraints.

```{code-cell} python
:label: anim-cartpole-formulations
:caption: The two trajectories are generated by the same nonlinear RK4 plant. The pole starts downward and reaches the upright configuration while the cart remains inside the 2.4 m rail limits. Animation frames are computed from the optimized state trajectories; no browser-side simulator is used.
:tags: [remove-input]

from IPython.display import HTML, display
import matplotlib.pyplot as plt

swingup_animation = make_swingup_animation(swingup_results, swingup_scenario)
display(HTML(swingup_animation.to_jshtml()))
plt.close(swingup_animation._fig)
```

The comparison also separates optimization from feedback. Each optimizer returns one fixed control sequence for one assumed initial state. To test what that object can and cannot do, the next replay applies the direct-transcription controls twice. One realization follows the nominal model. The other receives an additional cart acceleration of $1\;\mathrm{m\,s^{-2}}$ for one $0.15$ s step at $t=2.1$ s, after which both realizations receive the same remaining commands.

```{code-cell} python
:label: fig-cartpole-open-loop-perturbation
:caption: A one-step unmodeled acceleration separates two realizations driven by the same open-loop controls. The nominal trajectory reaches normalized pole height $\cos\theta=1$; the disturbed trajectory finishes below the horizontal. The optimizer has produced a plan, not a rule that reacts to the observed state.
:tags: [remove-input]

open_loop_replay = replay_open_loop_with_disturbance(
    swingup_results["direct"],
    swingup_scenario,
)
make_open_loop_perturbation_figure(open_loop_replay)
```

Feedback changes the object being computed. A feedback controller maps the state observed after the disturbance to a new action. Model predictive control will obtain such a map by repeatedly solving trajectory problems, while dynamic programming will construct state-contingent decisions through the value function.

:::{dropdown} Inspect the shared nonlinear dynamics
```{literalinclude} code/cartpole_control.py
:language: python
:start-at: def cartpole_dynamics
:end-before: def rk4_step
:linenos:
```
:::

{download}`Download the complete cart-pole trajectory-optimization and control source <code/cartpole_control.py>`.



## In Between Sequential and Simultaneous

```{admonition} Before reading on
:class: tip
Consider what happens to single shooting when the horizon $T$ is very large. What numerical difficulties might arise? Think about how small errors in early controls could propagate through the dynamics.
```

The two formulations we have seen so far lie at opposite ends. The **full discretization** approach keeps every state explicit and enforces the dynamics through equality constraints, which makes the structure clear but leads to a large optimization problem. At the other end, **single shooting** removes these constraints by simulating forward from the initial state, leaving only the controls as decision variables. That makes the problem smaller, but it also introduces a long and highly nonlinear dependency from the first control to the last state.

**Multiple shooting** sits in between. Instead of simulating the entire horizon in one shot, we divide it into smaller segments. For each segment, we keep its starting state as a decision variable and propagate forward using the dynamics for that segment. At the end, we enforce continuity by requiring that the simulated end state of one segment matches the decision variable for the next.

Formally, suppose the horizon of $T$ steps is divided into $K$ segments of length $L$ (with $T = K \cdot L$ for simplicity). We introduce:

* The controls for each step: $\mathbf{u}_{1:T-1}$.
* The state at the start of each segment: $\mathbf{x}_1,\dots,\mathbf{x}_K$.

Given $\mathbf{x}_k$ and the controls in its segment, we compute the predicted terminal state by simulating forward:

$$
\hat{\mathbf{x}}_{k+1} = \Phi(\mathbf{x}_k,\mathbf{u}_{\text{segment }k}),
$$

where $\Phi$ represents $L$ applications of the dynamics. Continuity constraints enforce:

$$
\mathbf{x}_{k+1} - \hat{\mathbf{x}}_{k+1} = 0, \qquad k=1,\dots,K-1.
$$

The resulting nonlinear program looks like this:

$$
\begin{aligned}
\min_{\{\mathbf{x}_k,\mathbf{u}_t\}} \quad &
c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t) \\
\text{subject to} \quad &
\mathbf{x}_{k+1} - \Phi(\mathbf{x}_k,\mathbf{u}_{\text{segment }k}) = 0,\quad k = 1,\dots,K-1, \\
& \mathbf{u}_{\mathrm{lb}} \le \mathbf{u}_t \le \mathbf{u}_{\mathrm{ub}}, \\
& \text{boundary conditions on } \mathbf{x}_1 \text{ and } \mathbf{x}_K.
\end{aligned}
$$

Compared to the full NLP, we no longer introduce every intermediate state as a variable, only the anchors at segment boundaries. Inside each segment, states are reconstructed by simulation. Compared to single shooting, these anchors break the long dependency chain that makes optimization unstable: gradients only have to travel across $L$ steps before they hit a decision variable, rather than the entire horizon. This is the same reason why exploding or vanishing gradients appear in deep recurrent networks: when the chain is too long, information either dies out or blows up. Multiple shooting shortens the chain and improves conditioning.

By adjusting the number of segments $K$, we can interpolate between the two extremes: $K = 1$ gives single shooting, while $K = T$ recovers the full direct NLP. In practice, a moderate number of segments often strikes a good balance between robustness and complexity.


```{code-cell} python
:tags: [hide-input]

#  label: fig-ocp-multiple-shooting
#  caption: Multiple shooting ballistic BVP: the code produces an animation (and optional static plot) that shows how segment defects shrink while steering the projectile to the target.

%config InlineBackend.figure_format = 'retina'
"""
Multiple Shooting as a Boundary-Value Problem (BVP) for a Ballistic Trajectory
-----------------------------------------------------------------------------
We solve for the initial velocities (and total flight time) so that the terminal
position hits a target, enforcing continuity between shooting segments.
"""

import numpy as np
import matplotlib.pyplot as plt

# Apply book style
try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except (ImportError, OSError):
    pass  # Use matplotlib defaults
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from IPython.display import HTML, display

# -----------------------------
# Physical parameters
# -----------------------------
g = 9.81          # gravity (m/s^2)
m = 1.0           # mass (kg)
drag_coeff = 0.1  # quadratic drag coefficient


def dynamics(t, state):
    """Ballistic dynamics with quadratic drag. state = [x, y, vx, vy]."""
    x, y, vx, vy = state
    v = np.hypot(vx, vy)
    drag_x = -drag_coeff * v * vx / m if v > 0 else 0.0
    drag_y = -drag_coeff * v * vy / m if v > 0 else 0.0
    dx  = vx
    dy  = vy
    dvx = drag_x
    dvy = drag_y - g
    return np.array([dx, dy, dvx, dvy])


def flow(y0, h):
    """One-segment flow map Φ(y0; h): integrate dynamics over duration h."""
    sol = solve_ivp(dynamics, (0.0, h), y0, method="RK45", rtol=1e-7, atol=1e-9)
    return sol.y[:, -1], sol

# -----------------------------
# Multiple-shooting BVP residuals
# -----------------------------

def residuals(z, K, x_init, x_target):
    """
    Unknowns z = [vx0, vy0, H, y1(4), y2(4), ..., y_{K-1}(4)]  (total len = 3 + 4*(K-1))
    We define y0 from x_init and (vx0, vy0). Each segment has duration h = H/K.
    Residual vector stacks:
      - initial position constraints: y0[:2] - x_init[:2]
      - continuity: y_{k+1} - Φ(y_k; h) for k=0..K-2
      - terminal position constraint at end of last segment: Φ(y_{K-1}; h)[:2] - x_target[:2]
    """
    n = 4
    vx0, vy0, H = z[0], z[1], z[2]
    if H <= 0:
        # Strongly penalize nonpositive durations to keep solver away
        return 1e6 * np.ones(2 + 4*(K-1) + 2)

    h = H / K

    # Build list of segment initial states y_0..y_{K-1}
    ys = []
    y0 = np.array([x_init[0], x_init[1], vx0, vy0], dtype=float)
    ys.append(y0)
    if K > 1:
        rest = z[3:]
        y_internals = rest.reshape(K-1, n)
        ys.extend(list(y_internals))  # y1..y_{K-1}

    res = []

    # Initial position must match exactly
    res.extend(ys[0][:2] - x_init[:2])

    # Continuity across segments
    for k in range(K-1):
        yk = ys[k]
        yk1_pred, _ = flow(yk, h)
        res.extend(ys[k+1] - yk1_pred)

    # Terminal position at the end of last segment equals target
    y_last_end, _ = flow(ys[-1], h)
    res.extend(y_last_end[:2] - x_target[:2])

    # Optional soft "stay above ground" at knots (kept gentle)
    # res.extend(np.minimum(0.0, np.array([y[1] for y in ys])).ravel())

    return np.asarray(res)

# -----------------------------
# Solve BVP via optimization on 0.5*||residuals||^2
# -----------------------------

def solve_bvp_multiple_shooting(K=5, x_init=np.array([0., 0.]), x_target=np.array([10., 0.])):
    """
    K: number of shooting segments.
    x_init: initial position (x0, y0). Initial velocities are unknown.
    x_target: desired terminal position (xT, yT) at time H (unknown).
    """
    # Heuristic initial guesses:
    dx = x_target[0] - x_init[0]
    dy = x_target[1] - x_init[1]
    H0 = max(0.5, dx / 5.0)  # guess ~ 5 m/s horizontal
    vx0_0 = dx / H0
    vy0_0 = (dy + 0.5 * g * H0**2) / H0  # vacuum guess

    # Intentionally disconnected internal knots to visualize defect shrinkage
    internals = []
    for k in range(1, K):  # y1..y_{K-1}
        xk = x_init[0] + (dx * k) / K
        yk = x_init[1] + (dy * k) / K + 2.0  # offset to create mismatch
        internals.append(np.array([xk, yk, 0.0, 0.0]))
    internals = np.array(internals) if K > 1 else np.array([])

    z0 = np.concatenate(([vx0_0, vy0_0, H0], internals.ravel()))

    # Variable bounds: H > 0, keep velocities within a reasonable range
    # Use wide bounds to let the solver work; tune if needed.
    lb = np.full_like(z0, -np.inf, dtype=float)
    ub = np.full_like(z0,  np.inf, dtype=float)
    lb[2] = 1e-2  # H lower bound
    # Optional velocity bounds
    lb[0], ub[0] = -50.0, 50.0
    lb[1], ub[1] = -50.0, 50.0

    # Objective and callback for L-BFGS-B
    def objective(z):
        r = residuals(z, K,
                      np.array([x_init[0], x_init[1], 0., 0.]),
                      np.array([x_target[0], x_target[1], 0., 0.]))
        return 0.5 * np.dot(r, r)

    iterate_history = []
    def cb(z):
        iterate_history.append(z.copy())

    bounds = list(zip(lb.tolist(), ub.tolist()))
    sol = minimize(objective, z0, method='L-BFGS-B', bounds=bounds,
                   callback=cb, options={'maxiter': 300, 'ftol': 1e-12})

    return sol, iterate_history

# -----------------------------
# Reconstruct and plot (optional static figure)
# -----------------------------

def reconstruct_and_plot(sol, K, x_init, x_target):
    n = 4
    vx0, vy0, H = sol.x[0], sol.x[1], sol.x[2]
    h = H / K

    ys = []
    y0 = np.array([x_init[0], x_init[1], vx0, vy0])
    ys.append(y0)
    if K > 1:
        internals = sol.x[3:].reshape(K-1, n)
        ys.extend(list(internals))

    # Integrate each segment and stitch
    traj_x, traj_y = [], []
    for k in range(K):
        yk = ys[k]
        yend, seg = flow(yk, h)
        traj_x.extend(seg.y[0, :].tolist() if k == 0 else seg.y[0, 1:].tolist())
        traj_y.extend(seg.y[1, :].tolist() if k == 0 else seg.y[1, 1:].tolist())

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(traj_x, traj_y, '-', label='Multiple-shooting solution')
    ax.plot([x_init[0]], [x_init[1]], 'go', label='Start')
    ax.plot([x_target[0]], [x_target[1]], 'r*', ms=12, label='Target')
    total_pts = len(traj_x)
    for k in range(1, K):
        idx = int(k * total_pts / K)
        ax.axvline(traj_x[idx], color='k', ls='--', alpha=0.3, lw=1)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Multiple Shooting BVP (K={K})   H={H:.3f}s   v0=({vx0:.2f},{vy0:.2f}) m/s')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()

    # Report residual norms
    res = residuals(sol.x, K, np.array([x_init[0], x_init[1], 0., 0.]), np.array([x_target[0], x_target[1], 0., 0.]))
    print(f"\nFinal residual norm: {np.linalg.norm(res):.3e}")
    print(f"vx0={vx0:.4f} m/s, vy0={vy0:.4f} m/s, H={H:.4f} s")

# -----------------------------
# Create JS animation for notebooks
# -----------------------------

def create_animation_progress(iter_history, K, x_init, x_target):
    """Return a JS animation (to_jshtml) showing defect shrinkage across segments."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Apply book style
    try:
        import scienceplots
        plt.style.use(['science', 'notebook'])
    except (ImportError, OSError):
        pass  # Use matplotlib defaults

    n = 4

    def unpack(z):
        vx0, vy0, H = z[0], z[1], z[2]
        ys = [np.array([x_init[0], x_init[1], vx0, vy0])]
        if K > 1 and len(z) > 3:
            internals = z[3:].reshape(K-1, n)
            ys.extend(list(internals))
        return H, ys

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.set_xlabel('Segment index (normalized time)')
    ax.set_ylabel('y (m)')
    ax.set_title('Multiple Shooting: Defect Shrinkage (Fixed Boundaries)')
    ax.grid(True, alpha=0.3)

    # Start/target markers at fixed indices
    ax.plot([0], [x_init[1]], 'go', label='Start')
    ax.plot([K], [x_target[1]], 'r*', ms=12, label='Target')
    # Vertical dashed lines at boundaries
    for k in range(1, K):
        ax.axvline(k, color='k', ls='--', alpha=0.35, lw=1)
    ax.legend(loc='best')

    # Pre-create line artists
    colors = plt.cm.plasma(np.linspace(0, 1, K))
    segment_lines = [ax.plot([], [], '-', color=colors[k], lw=2, alpha=0.9)[0] for k in range(K)]
    connector_lines = [ax.plot([], [], 'r-', lw=1.4, alpha=0.75)[0] for _ in range(K-1)]

    text_iter = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    def animate(i):
        idx = min(i, len(iter_history)-1)
        z = iter_history[idx]
        H, ys = unpack(z)
        h = H / K

        all_y = [x_init[1], x_target[1]]
        total_defect = 0.0
        for k in range(K):
            yk = ys[k]
            yend, seg = flow(yk, h)
            # Map local time to [k, k+1]
            t_local = seg.t
            x_vals = k + (t_local / t_local[-1])
            y_vals = seg.y[1, :]
            segment_lines[k].set_data(x_vals, y_vals)
            all_y.extend(y_vals.tolist())
            if k < K-1:
                y_next = ys[k+1]
                # Vertical connector at boundary x=k+1
                connector_lines[k].set_data([k+1, k+1], [yend[1], y_next[1]])
                total_defect += abs(y_next[1] - yend[1])

        # Fixed x-limits in index space
        ax.set_xlim(-0.1, K + 0.1)
        ymin, ymax = min(all_y), max(all_y)
        margin_y = 0.10 * max(1.0, ymax - ymin)
        ax.set_ylim(ymin - margin_y, ymax + margin_y)

        text_iter.set_text(f'Iterate {idx+1}/{len(iter_history)}  |  Sum vertical defect: {total_defect:.3e}')
        return segment_lines + connector_lines + [text_iter]

    anim = FuncAnimation(fig, animate, frames=len(iter_history), interval=600, blit=False, repeat=True)
    plt.tight_layout()
    js_anim = anim.to_jshtml()
    plt.close(fig)
    return js_anim


def main():
    # Problem definition
    x_init = np.array([0.0, 0.0])      # start at origin
    x_target = np.array([10.0, 0.0])   # hit ground at x=10 m
    K = 6                               # number of shooting segments

    sol, iter_hist = solve_bvp_multiple_shooting(K=K, x_init=x_init, x_target=x_target)
    # Optionally show static reconstruction (commented for docs cleanliness)
    # reconstruct_and_plot(sol, K, x_init, x_target)

    # Animate progression (defect shrinkage across segments) and display as JS
    js_anim = create_animation_progress(iter_hist, K, x_init, x_target)
    display(HTML(js_anim))


if __name__ == "__main__":
    main()
```





### Example: Hydro Cascade Scheduling with Physical Routing

The ballistic boundary-value problem couples consecutive segments of one trajectory. A hydroelectric cascade adds a second form of coupling: actions taken upstream alter the inflows seen downstream after a travel delay. Multiple shooting exposes both forms through local ODE integrations, temporal continuity defects, and inter-reach routing constraints.

The hydro-reservoir model in [](dp.md) uses a discrete-time abstraction in which precipitation enters as a noisy inflow. That abstraction is useful for learning and control design, but it omits much of the physical behavior of rivers and dams. Here we use a more detailed setup inspired by {cite:p}`Savorgnan2011`. We consider a series of dams arranged in a cascade, where the actions taken upstream influence downstream levels with a delay. The amount of power produced depends on the water flow through the turbines and the head (the vertical distance between the reservoir surface and the turbine outlet). The larger the head, the more potential energy is available for conversion into electricity, and the higher the power output.

To capture these effects, we follow a modeling approach inspired by the Saint-Venant equations, which describe how water levels and flows evolve in open channels. Instead of solving the full PDEs, we use a reduced model that approximates each dammed section of river (called a reach) as a lumped system governed by an ordinary differential equation. The main variable of interest is the water level $h_r(t)$, which changes over time depending on how much water enters, how much is discharged through the turbines $q_r(t)$, and how much is spilled $s_r(t)$. The mass balance for reach $r$ is written as:

$$
\frac{d h_r(t)}{dt} = \frac{1}{A_r} \left( z_r(t) - q_r(t) - s_r(t) \right),
$$

where $A_r$ is the surface area of the reservoir, assumed constant. The inflow $z_r(t)$ to a reach either comes from nature (for the first dam), or from the upstream turbine and spill discharge, delayed by a travel time $\tau_{r-1}$:

$$
z_1(t) = \text{inflow}(t), \qquad
z_r(t) = q_{r-1}(t - \tau_{r-1}) + s_{r-1}(t - \tau_{r-1}), \quad \text{for } r > 1.
$$

Power generation at each reach depends on how much water is discharged and the available head:

$$
P_r(t) = \rho g \eta \, q_r(t) \, H_r(h_r(t)),
$$

where $\rho$ is water density, $g$ is gravitational acceleration, $\eta$ is turbine efficiency, and $H_r(h_r(t))$ denotes the head as a function of the water level. In some models, the head is approximated as the difference between the current level and a fixed tailwater height (the water level downstream of the dam, after it has passed through the turbine).

The operator's goal is to meet a target generation profile $P^\text{ref}(t)$, such as one dictated by a market dispatch or load-following constraint. This leads to an objective that minimizes the deviation from the target over the full horizon:

$$
\min_{\{q_r(t), s_r(t)\}} \int_0^T \left( \sum_{r=1}^R P_r(t) - P^\text{ref}(t) \right)^2 dt.
$$

In practice, this is combined with operational constraints: turbine capacity $0 \le q_r(t) \le \bar{q}_r$, spillway limits $0 \le s_r(t) \le \bar{s}_r$, and safe level bounds $h_r^{\min} \le h_r(t) \le h_r^{\max}$. Depending on the use case, one may also penalize spill to encourage water conservation, or penalize fast changes in levels for ecological reasons.

The reaches are coupled across space and time. An upstream reach cannot simply act in isolation: if the operator wants reach $r$ to produce power at a specific time, the water must be released by reach $r-1$ sufficiently in advance. This coordination is further complicated by delays, nonlinearities in head-dependent power, and limited storage capacity.

We solve the problem using **multiple shooting**. Each reach is divided into local simulation segments over short time windows. Within each segment, the dynamics are integrated forward using the ODEs, and continuity constraints are added to ensure that the water levels match across segment boundaries. At the same time, the inflows passed from upstream reaches must arrive at the right time and be consistent with previous decisions. In discrete time, this gives rise to a set of state-update equations:

$$
h_r^{k+1} = h_r^k + \Delta t \cdot \frac{1}{A_r}(z_r^k - q_r^k - s_r^k),
$$

with delays handled by shifting $z_r^k$ according to the appropriate travel time. These constraints are enforced as part of a nonlinear program, alongside the power tracking objective and control bounds.

Compared with a single-reservoir inflow-outflow model, the cascade adds delayed coupling constraints. Upstream reservoirs can store water in anticipation of future needs, while downstream dams adjust their output to match arrivals and avoid overflows. The resulting schedule coordinates the entire system against the demand profile.

```{code-cell} python
:tags: [hide-input]
:label: fig-trajectories-hydro-multiple-shooting
:caption: Multiple shooting coordinates reservoir levels, turbine discharges, routed inflows, and total generation across a three-reach hydroelectric cascade.

%config InlineBackend.figure_format = 'retina'
# Instrumented MSD hydro demo with heterogeneity + diagnostics
# - Breaks symmetry to avoid trivial identical plots
# - Adds rich diagnostics to explain flat levels and equalities
#
# This cell runs end-to-end and shows plots + tables.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Apply book style
try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except (ImportError, OSError):
    pass  # Use matplotlib defaults
from dataclasses import dataclass
from typing import Tuple
from scipy.optimize import minimize
from math import sqrt
import warnings

# ---------- Model ----------

g = 9.81  # m/s^2

@dataclass
class ReachParams:
    L: float
    W: float
    k_b: float
    S_b: float
    k_t: float
    @property
    def A_surf(self) -> float:
        return self.L * self.W

def smooth_relu(x, eps=1e-9):
    return 0.5*(x + np.sqrt(x*x + eps))

def q_bypass(H, rp: ReachParams):
    H_eff = smooth_relu(H)
    return rp.k_b * rp.S_b * np.sqrt(2*g*H_eff)

def muskingum_coeffs(K: float, X: float, dt: float) -> Tuple[float, float, float]:
    D  = 2.0*K*(1.0 - X) + dt
    C0 = (dt - 2.0*K*X) / D
    C1 = (dt + 2.0*K*X) / D
    C2 = (2.0*K*(1.0 - X) - dt) / D
    return C0, C1, C2

def integrate_interval(H0, u, z, dt, nsub, rp: ReachParams):
    """Forward Euler. Returns Hend, avg_qout."""
    h = dt/nsub
    H = H0
    qsum = 0.0
    for _ in range(nsub):
        qb = q_bypass(H, rp)
        qout = u + qb
        dHdt = (z - qout) / rp.A_surf
        H += h*dHdt
        qsum += qout
    return H, qsum/nsub

def shapes(M,N): return (M*(N+1), M*N, M*N)

def unpack(x, M, N):
    nH, nu, nz = shapes(M,N)
    H = x[:nH].reshape(M,N+1)
    u = x[nH:nH+nu].reshape(M,N)
    z = x[nH+nu:nH+nu+nz].reshape(M,N)
    return H,u,z

def pack(H,u,z): return np.concatenate([H.ravel(), u.ravel(), z.ravel()])

# ---------- Problem builder ----------

def make_params_hetero(M):
    """Heterogeneous reaches to break symmetry."""
    # Widths, spillway areas, and power coeffs vary by reach
    W_list = np.linspace(80, 140, M)         # m
    L_list = np.full(M, 4000.0)              # m
    S_b_list = np.linspace(14.0, 20.0, M)    # m^2
    k_t_list = np.linspace(7.5, 8.5, M)      # power coeff
    k_b_list = np.linspace(0.55, 0.65, M)    # spill coeff
    return [ReachParams(L=float(L_list[i]), W=float(W_list[i]),
                        k_b=float(k_b_list[i]), S_b=float(S_b_list[i]),
                        k_t=float(k_t_list[i])) for i in range(M)]

def build_demo(M=3, N=12, dt=900.0, seed=0, hetero=True):
    rng = np.random.default_rng(seed)
    params = make_params_hetero(M) if hetero else [ReachParams(4000.0, 100.0, 0.6, 18.26, 8.0) for _ in range(M)]

    # initial levels (heterogeneous)
    H0 = np.array([17.0, 16.7, 17.3][:M])

    H_ref = np.array([17.0, 16.9, 17.1][:M]) if hetero else np.full(M, 17.0)
    H_bounds = (16.0, 18.5)
    u_bounds = (40.0, 160.0)

    Qin_base = 300.0
    Qin_ext = Qin_base + 30.0*np.sin(2*np.pi*np.arange(N)/N)  # stronger swing

    Pref_raw = 60.0 + 15.0*np.sin(2*np.pi*(np.arange(N)-2)/N)

    # default Muskingum parameters per link (M-1 links)
    if M > 1:
        K_list = list(np.linspace(1800.0, 2700.0, M-1))
        X_list = [0.2]*(M-1)
    else:
        K_list = []
        X_list = []

    return dict(params=params, H0=H0, H_ref=H_ref, H_bounds=H_bounds,
                u_bounds=u_bounds, Qin_ext=Qin_ext, Pref_raw=Pref_raw,
                dt=dt, N=N, M=M, nsub=10,
                muskingum=dict(K=K_list, X=X_list))

# ---------- Objective / constraints / helpers ----------

def compute_total_power(H,u,params):
    M,N = u.shape
    Pn = np.zeros(N)
    for n in range(N):
        for i in range(M):
            Pn[n] += params[i].k_t * u[i,n] * H[i,n]
    return Pn

def decompose_objective(x, data, Pref, wP, wH, wDu):
    H,u,z = unpack(x, data["M"], data["N"])
    params, H_ref = data["params"], data["H_ref"]
    track = np.sum((compute_total_power(H,u,params)-Pref)**2)
    lvl   = np.sum((H[:,:-1]-H_ref[:,None])**2)
    du    = np.sum((u[:,1:]-u[:,:-1])**2)
    return dict(track=wP*track, lvl=wH*lvl, du=wDu*du, raw=dict(track=track,lvl=lvl,du=du))

def make_objective(data, Pref, wP=8.0, wH=0.02, wDu=1e-4):
    params, H_ref, N, M = data["params"], data["H_ref"], data["N"], data["M"]
    def obj(x):
        H,u,z = unpack(x,M,N)
        return (
            wP*np.sum((compute_total_power(H,u,params)-Pref)**2)
            + wH*np.sum((H[:,:-1]-H_ref[:,None])**2)
            + wDu*np.sum((u[:,1:]-u[:,:-1])**2)
        )
    return obj, dict(wP=wP,wH=wH,wDu=wDu)

def make_constraints(data):
    params, H0, Qin_ext, dt, N, M, nsub = (
        data["params"], data["H0"], data["Qin_ext"], data["dt"], data["N"], data["M"], data["nsub"]
    )
    cons = []
    def init_fun(x):
        H,u,z = unpack(x,M,N); return H[:,0]-H0
    cons.append({'type':'eq','fun':init_fun})
    def dyn_fun(x):
        H,u,z = unpack(x,M,N)
        res=[]
        for i in range(M):
            for n in range(N):
                Hend, _ = integrate_interval(H[i,n], u[i,n], z[i,n], dt, nsub, params[i])
                res.append(H[i,n+1]-Hend)
        return np.array(res)
    cons.append({'type':'eq','fun':dyn_fun})
    def coup_fun(x):
        H,u,z = unpack(x,M,N)
        res=[]
        # First reach is exogenous inflow per interval
        for n in range(N):
            res.append(z[0,n]-Qin_ext[n])
        # Downstream links: Muskingum routing
        K_list = data.get("muskingum", {}).get("K", [])
        X_list = data.get("muskingum", {}).get("X", [])
        for i in range(1,M):
            # Seed condition for z[i,0]
            _, I0 = integrate_interval(H[i-1,0], u[i-1,0], z[i-1,0], dt, nsub, params[i-1])
            res.append(z[i,0] - I0)
            # Coefficients
            Ki = K_list[i-1] if i-1 < len(K_list) else 1800.0
            Xi = X_list[i-1] if i-1 < len(X_list) else 0.2
            C0, C1, C2 = muskingum_coeffs(Ki, Xi, dt)
            # Recursion over intervals
            for n in range(N-1):
                # upstream interval-average outflows for n and n+1
                _, I_n   = integrate_interval(H[i-1,n],   u[i-1,n],   z[i-1,n],   dt, nsub, params[i-1])
                _, I_np1 = integrate_interval(H[i-1,n+1], u[i-1,n+1], z[i-1,n+1], dt, nsub, params[i-1])
                res.append(z[i,n+1] - (C0*I_np1 + C1*I_n + C2*z[i,n]))
        return np.array(res)
    cons.append({'type':'eq','fun':coup_fun})
    return cons

def make_bounds(data):
    Hmin,Hmax = data["H_bounds"]
    umin,umax = data["u_bounds"]
    M,N = data["M"], data["N"]
    nH,nu,nz = shapes(M,N)
    lb = np.empty(nH+nu+nz); ub = np.empty_like(lb)
    lb[:nH]=Hmin; ub[:nH]=Hmax
    lb[nH:nH+nu]=umin; ub[nH:nH+nu]=umax
    lb[nH+nu:]=0.0; ub[nH+nu:]=2000.0
    return list(zip(lb,ub))

def residuals(x, data):
    params, H0, Qin_ext, dt, N, M, nsub = (
        data["params"], data["H0"], data["Qin_ext"], data["dt"], data["N"], data["M"], data["nsub"]
    )
    H,u,z = unpack(x, M, N)
    dyn = np.zeros((M,N)); coup = np.zeros((M,N))
    for i in range(M):
        for n in range(N):
            Hend, qavg = integrate_interval(H[i,n], u[i,n], z[i,n], dt, nsub, params[i])
            dyn[i,n] = H[i,n+1] - Hend
            if i == 0:
                coup[i,n] = z[i,n] - Qin_ext[n]
            else:
                # Muskingum residual, align on current index using n and n-1
                Ki = data.get("muskingum", {}).get("K", [1800.0]*(M-1))[i-1]
                Xi = data.get("muskingum", {}).get("X", [0.2]*(M-1))[i-1]
                C0, C1, C2 = muskingum_coeffs(Ki, Xi, dt)
                if n == 0:
                    coup[i,n] = 0.0
                else:
                    _, I_nm1 = integrate_interval(H[i-1,n-1], u[i-1,n-1], z[i-1,n-1], dt, nsub, params[i-1])
                    _, I_n   = integrate_interval(H[i-1,n],   u[i-1,n],   z[i-1,n],   dt, nsub, params[i-1])
                    coup[i,n] = z[i,n] - (C0*I_n + C1*I_nm1 + C2*z[i,n-1])
    return dyn, coup

# ---------- Feasible initial guess with hetero controls ----------

def feasible_initial_guess(data):
    """Feasible x0 with nontrivial u by setting u at mid + per-reach pattern, then integrating to define H,z."""
    M,N,dt,nsub = data["M"], data["N"], data["dt"], data["nsub"]
    params = data["params"]
    umin,umax = data["u_bounds"]
    Qin_ext = data["Qin_ext"]

    # pattern to break symmetry
    base = 0.5*(umin+umax)
    phase = np.linspace(0, np.pi/2, M)
    tgrid = np.arange(N)
    u_pattern = np.array([base + 25*np.sin(2*np.pi*(tgrid/N) + ph) for ph in phase])
    u_pattern = np.clip(u_pattern, umin, umax)

    H = np.zeros((M, N+1)); u = np.zeros((M, N)); z = np.zeros((M, N))
    H[:,0] = data["H0"]
    # Set controls from pattern first
    for i in range(M):
        u[i,:] = u_pattern[i,:]

    # First reach: exogenous inflow, integrate forward and record outflow averages
    qavg_up = np.zeros((M, N))
    for n in range(N):
        z[0,n] = Qin_ext[n]
        Hend, qavg = integrate_interval(H[0,n], u[0,n], z[0,n], dt, nsub, params[0])
        H[0,n+1] = Hend
        qavg_up[0,n] = qavg

    # Downstream reaches with Muskingum routing
    K_list = data.get("muskingum", {}).get("K", [1800.0]*(M-1))
    X_list = data.get("muskingum", {}).get("X", [0.2]*(M-1))
    for i in range(1,M):
        Ki = K_list[i-1] if i-1 < len(K_list) else 1800.0
        Xi = X_list[i-1] if i-1 < len(X_list) else 0.2
        C0, C1, C2 = muskingum_coeffs(Ki, Xi, dt)
        I = qavg_up[i-1,:]
        # seed
        z[i,0] = I[0]
        # propagate recursively over time
        for n in range(N-1):
            z[i,n+1] = C0*I[n+1] + C1*I[n] + C2*z[i,n]
        # integrate levels for reach i using routed inflow
        for n in range(N):
            Hend, qavg = integrate_interval(H[i,n], u[i,n], z[i,n], dt, nsub, params[i])
            H[i,n+1] = Hend
            qavg_up[i,n] = qavg
    return pack(H,u,z)

def scale_pref(Pref_raw, x0, data):
    H,u,z = unpack(x0, data["M"], data["N"])
    P0 = compute_total_power(H,u,data["params"])
    s = max(np.mean(P0),1e-6)/max(np.mean(Pref_raw),1e-6)
    return Pref_raw*s, P0

def run_demo(show: bool = True, save_path: str | None = 'hydro.png', verbose: bool = False):
    """Build, solve, and render the hydro demo.

    Parameters
    ----------
    show : bool
        If True, displays the matplotlib figure via plt.show().
    save_path : str | None
        If provided, saves the figure to this path.
    verbose : bool
        If True, prints diagnostic information.

    Returns
    -------
    matplotlib.figure.Figure | None
        Returns the Figure when show is False; otherwise returns None.
    """
    # ---------- Solve ----------
    data = build_demo(M=3, N=16, dt=900.0, hetero=True)
    x0 = feasible_initial_guess(data)
    Pref, P0 = scale_pref(data["Pref_raw"], x0, data)

    objective, weights = make_objective(data, Pref, wP=8.0, wH=0.02, wDu=5e-4)
    # Suppress noisy SciPy warning about delta_grad during quasi-Newton updates
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"delta_grad == 0.0",
            category=UserWarning,
            module=r"scipy\.optimize\.\_differentiable_functions",
        )
        res = minimize(
            fun=objective,
            x0=x0,
            method='trust-constr',
            bounds=make_bounds(data),
            constraints=make_constraints(data),
            options=dict(maxiter=1000, disp=verbose),
        )

    H,u,z = unpack(res.x, data["M"], data["N"])
    P = compute_total_power(H,u,data["params"])
    dyn_res, coup_res = residuals(res.x, data)

    # ---------- Diagnostics ----------
    if verbose:
        terms = decompose_objective(res.x, data, Pref, **weights)
        print("\n=== Objective decomposition ===")
        print({k: float(v) if not isinstance(v, dict) else {kk: float(vv) for kk,vv in v.items()} for k,v in terms.items()})

        print("\n=== Constraint residuals (max |.|) ===")
        print("dyn:", float(np.max(np.abs(dyn_res)))), print("coup:", float(np.max(np.abs(coup_res))))

        # Muskingum coefficient sanity and residuals
        if data.get("M", 1) > 1:
            K_list = data.get("muskingum", {}).get("K", [])
            X_list = data.get("muskingum", {}).get("X", [])
            coef_checks = []
            mean_abs_res = []
            for i in range(1, data["M"]):
                Ki = K_list[i-1] if i-1 < len(K_list) else 1800.0
                Xi = X_list[i-1] if i-1 < len(X_list) else 0.2
                C0, C1, C2 = muskingum_coeffs(Ki, Xi, data["dt"])
                coef_checks.append(dict(link=i, sum=float(C0+C1+C2), min_coef=float(min(C0,C1,C2))))
                # compute mean abs residual for this link
                res_vals = []
                for n in range(data["N"]-1):
                    _, I_n   = integrate_interval(H[i-1,n],   u[i-1,n],   z[i-1,n],   data["dt"], data["nsub"], data["params"][i-1])
                    _, I_np1 = integrate_interval(H[i-1,n+1], u[i-1,n+1], z[i-1,n+1], data["dt"], data["nsub"], data["params"][i-1])
                    res_vals.append(float(abs(z[i,n+1] - (C0*I_np1 + C1*I_n + C2*z[i,n]))))
                mean_abs_res.append(dict(link=i, mean_abs=float(np.mean(res_vals))))
            print("\n=== Muskingum coeff checks (sum, min_coef) ===")
            print(coef_checks)
            print("=== Muskingum mean |residual| per link ===")
            print(mean_abs_res)

    # Per-interval diagnostic table for each reach (kept for debugging but unused here)
    def interval_table(i):
        rp = data["params"][i]
        rows = []
        for n in range(data["N"]):
            qb = q_bypass(H[i,n], rp)
            net = z[i,n] - (u[i,n] + qb)
            dH = data["dt"]*net/rp.A_surf
            rows.append(dict(interval=n, Hn=H[i,n], Hn1=H[i,n+1], u=u[i,n], z=z[i,n], qb=qb, net_flow=net, dH_pred=dH))
        return pd.DataFrame(rows)

    # summary and tables available to callers if needed
    tables = [interval_table(i) for i in range(data["M"])]
    summary = pd.DataFrame([
        dict(reach=i+1,
             H_mean=float(np.mean(H[i])), H_std=float(np.std(H[i])),
             u_mean=float(np.mean(u[i])), u_std=float(np.std(u[i])),
             z_mean=float(np.mean(z[i])), z_std=float(np.std(z[i])))
        for i in range(data["M"])
    ])

    # ---------- Plots ----------
    M,N = data["M"], data["N"]
    t_nodes = np.arange(N+1)
    t = np.arange(N)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Hydroelectric System Optimization Results', fontsize=16)

    ax1 = axes[0, 0]
    for i in range(M):
        ax1.plot(t_nodes, H[i], marker='o', label=f'Reach {i+1}')
    ax1.set_xlabel("Node n"); ax1.set_ylabel("H [m]"); ax1.set_title("Water Levels")
    ax1.grid(True); ax1.legend()

    ax2 = axes[0, 1]
    for i in range(M):
        ax2.step(t, u[i], where='post', label=f'Reach {i+1}')
    ax2.set_xlabel("Interval n"); ax2.set_ylabel("u [m³/s]"); ax2.set_title("Turbine Discharge")
    ax2.grid(True); ax2.legend()

    ax3 = axes[1, 0]
    for i in range(M):
        ax3.step(t, z[i], where='post', label=f'Reach {i+1}')
    ax3.set_xlabel("Interval n"); ax3.set_ylabel("z [m³/s]"); ax3.set_title("Inflow (Coupling)")
    ax3.grid(True); ax3.legend()

    ax4 = axes[1, 1]
    ax4.plot(t, P0, marker='s', label="Power @ x0")
    ax4.plot(t, P, marker='o', label="Power @ optimum")
    ax4.plot(t, Pref, marker='x', label="Scaled Pref")
    ax4.set_xlabel("Interval n"); ax4.set_ylabel("Power units"); ax4.set_title("Power Tracking")
    ax4.legend(); ax4.grid(True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        return None
    return fig


# Run the demo directly when loaded in a notebook cell
run_demo(show=True, save_path=None, verbose=False)

```


The figure shows the result of a multiple-shooting optimization applied to a three-reach hydroelectric cascade. The time horizon is discretized into 16 intervals, and SciPy's `trust-constr` solver is used to find a feasible control sequence that satisfies mass balance, turbine and spillway limits, and Muskingum-style routing dynamics. Each reach integrates its own local ODE. Shooting defects link reservoir levels across time, while separate Muskingum constraints link routed flows between reaches.

The top-left panel shows the water levels in each reservoir. We observe that upstream reservoirs tend to increase their levels ahead of discharge events, building potential energy before releasing water downstream. The top-right panel shows turbine discharges for each reach. These vary smoothly and are temporally coordinated across the system. The bottom-right panel compares the total generation to a synthetic demand profile, which is generated by a sum of time-shifted sigmoids and normalized to be feasible given turbine capacities. The optimized schedule (orange) tracks this demand closely, while the initial guess (blue) lags behind. The bottom-left panel plots the routed inflows between reaches, which display the expected lag and smoothing effects from Muskingum routing. The interplay between these plots shows how the system anticipates, stores, and routes water to meet time-varying generation targets within physical and operational limits.

The ballistic and hydro examples use the same numerical structure at different scales: integrate locally, expose states at segment boundaries, and drive every continuity defect to zero. We now return to the first-order optimality conditions of the underlying discrete-time program.


# The Discrete-Time Pontryagin Principle

If we take the Bolza formulation of the DOCP and apply the KKT conditions directly, we obtain an optimization system with many multipliers and constraints. Written in raw form, it looks like any other nonlinear program. But in control, this structure has a long history and a name of its own: the **Pontryagin principle**. In fact, the discrete-time version can be seen as the structured KKT system that results from introducing multipliers for the dynamics and collecting terms stage by stage.

We work with the Bolza program

$$
\begin{aligned}
\min_{\{\mathbf{x}_t,\mathbf{u}_t\}} \quad & c_T(\mathbf{x}_T)\;+\;\sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t) \\
\text{s.t.}\quad & \mathbf{x}_{t+1}=\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t),\quad t=1,\dots,T-1,\\
& \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t)\le \mathbf{0},\quad \mathbf{u}_t\in \mathcal{U}_t,\\
& \mathbf{h}(\mathbf{x}_T)=\mathbf{0}\quad\text{(optional terminal equalities)}.
\end{aligned}
$$

Introduce **costates** $\boldsymbol{\lambda}_{t+1}\in\mathbb{R}^n$ for the dynamics, multipliers $\boldsymbol{\mu}_t\ge \mathbf{0}$ for path inequalities, and $\boldsymbol{\nu}$ for terminal equalities. The Lagrangian is

$$
\mathcal{L}
= c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t)
+ \sum_{t=1}^{T-1} \boldsymbol{\lambda}_{t+1}^\top\!\big(\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)-\mathbf{x}_{t+1}\big)
+ \sum_{t=1}^{T-1} \boldsymbol{\mu}_t^\top \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t)
+ \boldsymbol{\nu}^\top \mathbf{h}(\mathbf{x}_T).
$$

It is convenient to package the stagewise terms in a **Hamiltonian**

$$
H_t(\mathbf{x}_t,\mathbf{u}_t,\boldsymbol{\lambda}_{t+1},\boldsymbol{\mu}_t)
:= c_t(\mathbf{x}_t,\mathbf{u}_t)
+ \boldsymbol{\lambda}_{t+1}^\top \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)
+ \boldsymbol{\mu}_t^\top \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t).
$$

```{admonition} Check your understanding
:class: tip
Verify that the Hamiltonian $H_t$ collects exactly the terms from the Lagrangian that involve stage $t$: the stage cost, the dynamics constraint (weighted by the next costate), and the path constraints (weighted by their multipliers). Why does $\boldsymbol{\lambda}_{t+1}$ appear rather than $\boldsymbol{\lambda}_t$?
```

Then

$$
\mathcal{L} = c_T(\mathbf{x}_T)+\boldsymbol{\nu}^\top \mathbf{h}(\mathbf{x}_T)
+ \sum_{t=1}^{T-1}\Big[H_t(\mathbf{x}_t,\mathbf{u}_t,\boldsymbol{\lambda}_{t+1},\boldsymbol{\mu}_t)
- \boldsymbol{\lambda}_{t+1}^\top \mathbf{x}_{t+1}\Big].
$$

## Necessary conditions

```{note} Gradient convention
Throughout this section, we use the **denominator layout** (gradient layout) convention:
- $\nabla_{\mathbf{x}} f(\mathbf{x})$ produces a **column vector** (gradient)
- $\frac{\partial f}{\partial \mathbf{x}}$ produces the Jacobian matrix
- For scalar functions: $\nabla_{\mathbf{x}} f = \left(\frac{\partial f}{\partial \mathbf{x}}\right)^\top$

This is the standard convention in optimization and control theory.
```

Taking first-order variations and collecting terms gives the discrete-time adjoint system, control stationarity, and complementarity. At a local minimum $\{\mathbf{x}_t^\star,\mathbf{u}_t^\star\}$ with multipliers $\{\boldsymbol{\lambda}_t^\star,\boldsymbol{\mu}_t^\star,\boldsymbol{\nu}^\star\}$:

**State dynamics (primal feasibility)**

$$
\mathbf{x}_{t+1}^\star=\mathbf{f}_t(\mathbf{x}_t^\star,\mathbf{u}_t^\star),\quad t=1,\dots,T-1.
$$

**Costate recursion (backward "adjoint" equation)**

$$
\boldsymbol{\lambda}_t^\star
= \nabla_{\mathbf{x}} H_t\big(\mathbf{x}_t^\star,\mathbf{u}_t^\star,\boldsymbol{\lambda}_{t+1}^\star,\boldsymbol{\mu}_t^\star\big)
= \nabla_{\mathbf{x}} c_t(\mathbf{x}_t^\star,\mathbf{u}_t^\star)
+ \big[\nabla_{\mathbf{x}} \mathbf{f}_t(\mathbf{x}_t^\star,\mathbf{u}_t^\star)\big]^\top \boldsymbol{\lambda}_{t+1}^\star
+ \big[\nabla_{\mathbf{x}} \mathbf{g}_t(\mathbf{x}_t^\star,\mathbf{u}_t^\star)\big]^\top \boldsymbol{\mu}_t^\star,
$$

with the **terminal condition**

$$
\boldsymbol{\lambda}_T^\star
= \nabla_{\mathbf{x}} c_T(\mathbf{x}_T^\star) + \big[\nabla_{\mathbf{x}} \mathbf{h}(\mathbf{x}_T^\star)\big]^\top \boldsymbol{\nu}^\star
\quad\text{(and \(\boldsymbol{\nu}^\star=\mathbf{0}\) if there are no terminal equalities).}
$$

**Control stationarity (first-order optimality in $\mathbf{u}_t$)**
If $\mathcal{U}_t=\mathbb{R}^m$ (no explicit set constraint), then

$$
\nabla_{\mathbf{u}} H_t\big(\mathbf{x}_t^\star,\mathbf{u}_t^\star,\boldsymbol{\lambda}_{t+1}^\star,\boldsymbol{\mu}_t^\star\big)=\mathbf{0}.
$$

If $\mathcal{U}_t$ imposes bounds or a convex set, the condition becomes the **variational inequality**

$$
\mathbf{0}\in \nabla_{\mathbf{u}} H_t(\cdot)\;+\;N_{\mathcal{U}_t}(\mathbf{u}_t^\star),
$$

where $N_{\mathcal{U}_t}(\cdot)$ is the normal cone to $\mathcal{U}_t$. For simple box bounds, this reduces to standard KKT sign and complementarity conditions on the components of $\mathbf{u}_t^\star$.

**Path-constraint multipliers (primal/dual feasibility and complementarity)**

$$
\mathbf{g}_t(\mathbf{x}_t^\star,\mathbf{u}_t^\star)\le \mathbf{0},\quad
\boldsymbol{\mu}_t^\star\ge \mathbf{0},\quad
\mu_{t,i}^\star\, g_{t,i}(\mathbf{x}_t^\star,\mathbf{u}_t^\star)=0\quad \text{for all }i,t.
$$

**Terminal equalities (if present)**

$$
\mathbf{h}(\mathbf{x}_T^\star)=\mathbf{0}.
$$

The triplet "forward state, backward costate, control stationarity" is the discrete-time Euler–Lagrange system tailored to control with dynamics. It is the same KKT logic as before, but organized stagewise through the Hamiltonian.

```{prf:proposition} Discrete-time Pontryagin necessary conditions (summary)
At a local minimum of the DOCP

$$
\min_{\{\mathbf{x}_t,\mathbf{u}_t\}}\ c_T(\mathbf{x}_T)+\sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t)
\quad\text{s.t.}\quad \mathbf{x}_{t+1}=\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t),\ \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t)\le \mathbf{0},\ \mathbf{h}(\mathbf{x}_T)=\mathbf{0},
$$

there exist multipliers $\{\boldsymbol{\lambda}_{t+1}\}$, $\{\boldsymbol{\mu}_t\ge\mathbf{0}\}$, and (if present) $\boldsymbol{\nu}$ such that, for $t=1,\dots,T-1$:

- State dynamics: $\ \mathbf{x}_{t+1}=\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)$.
- Backward costate recursion:

  $$
  \boldsymbol{\lambda}_t = \nabla_{\mathbf{x}} c_t(\mathbf{x}_t,\mathbf{u}_t)
  + \big[\nabla_{\mathbf{x}} \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)\big]^\top \boldsymbol{\lambda}_{t+1}
  + \big[\nabla_{\mathbf{x}} \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t)\big]^\top \boldsymbol{\mu}_t.
  $$
  
- Terminal condition: $\ \boldsymbol{\lambda}_T = \nabla_{\mathbf{x}} c_T(\mathbf{x}_T) + \big[\nabla_{\mathbf{x}} \mathbf{h}(\mathbf{x}_T)\big]^\top \boldsymbol{\nu}$.
- Control stationarity (unconstrained control): $\ \nabla_{\mathbf{u}} H_t(\cdot)=\mathbf{0}$; with a convex control set $\mathcal{U}_t$, $\ \mathbf{0}\in \nabla_{\mathbf{u}} H_t(\cdot)+N_{\mathcal{U}_t}(\mathbf{u}_t)$.
- Path inequalities: $\ \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t)\le \mathbf{0}$, $\ \boldsymbol{\mu}_t\ge\mathbf{0}$, and complementarity $\ \mu_{t,i}\,g_{t,i}(\mathbf{x}_t,\mathbf{u}_t)=0$ for all $i$.
- Terminal equalities (if present): $\ \mathbf{h}(\mathbf{x}_T)=\mathbf{0}$.

Here $H_t(\mathbf{x}_t,\mathbf{u}_t,\boldsymbol{\lambda}_{t+1},\boldsymbol{\mu}_t):=c_t(\mathbf{x}_t,\mathbf{u}_t)+\boldsymbol{\lambda}_{t+1}^\top\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)+\boldsymbol{\mu}_t^\top\mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t)$ is the stage Hamiltonian.
```

**Recap.** The discrete-time Pontryagin principle is the KKT system for trajectory optimization, organized to exploit temporal structure. It has a forward-backward decomposition: states propagate forward through the dynamics, while costates propagate backward through the adjoint equation. The Hamiltonian $H_t$ packages together the stage cost, the dynamics (weighted by the next costate), and any path constraints (weighted by their multipliers). Control stationarity says that optimal controls minimize the Hamiltonian at each stage. Complementarity ensures that only binding constraints carry nonzero multipliers. This structure underlies both analytical solution methods (such as LQR) and numerical algorithms (such as the adjoint method for gradient computation).

## The adjoint equation as reverse accumulation

Optimization needs sensitivities. In trajectory problems we adjust decisions (controls or parameters) to reduce an objective while respecting dynamics and constraints. First‑order methods in the unconstrained case (e.g., gradient descent, L‑BFGS, Adam) require the gradient of the objective with respect to all controls, and constrained methods (SQP, interior‑point) require gradients of the Lagrangian, i.e., of costs and constraints. The discrete‑time adjoint equations provide these derivatives in a way that scales to long horizons and many decision variables.

Consider

$$
J = c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t),
\qquad \mathbf{x}_{t+1}=\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t).
$$

A single forward rollout computes and stores the trajectory $\mathbf{x}_{1:T}$. A single backward sweep then applies the reverse‑mode chain rule stage by stage.

```{admonition} Before reading on
:class: tip
Try applying the chain rule to compute how a perturbation to the state at time $t$ affects the total cost $J$. Start from $\partial J / \partial \mathbf{x}_T$ and work backward. What pattern emerges?
```

Defining the costate by

$$
\boldsymbol{\lambda}_T = \nabla_{\mathbf{x}} c_T(\mathbf{x}_T),\qquad
\boldsymbol{\lambda}_t = \nabla_{\mathbf{x}} c_t(\mathbf{x}_t,\mathbf{u}_t) + \big[\nabla_{\mathbf{x}} \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)\big]^\top \boldsymbol{\lambda}_{t+1},\quad t=T-1,\dots,1,
$$

yields exactly the discrete‑time adjoint (PMP) recursion.

```{admonition} Check your understanding
:class: tip
What terms would you expect to appear in the gradient $\nabla_{\mathbf{u}_t} J$? The control $\mathbf{u}_t$ affects $J$ in two ways: directly through the stage cost $c_t$, and indirectly by changing the next state $\mathbf{x}_{t+1}$. How should these contributions combine?
```

The gradient with respect to each control follows from the same reverse pass:

$$
\nabla_{\mathbf{u}_t} J = \nabla_{\mathbf{u}} c_t(\mathbf{x}_t,\mathbf{u}_t) + \big[\nabla_{\mathbf{u}} \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)\big]^\top \boldsymbol{\lambda}_{t+1}.
$$

This reverse accumulation produces every control gradient with one forward
rollout and one backward adjoint pass. The costate
$\boldsymbol{\lambda}_t$ measures the marginal effect of perturbing the state
at time $t$ on the total objective. Each control gradient combines a direct
contribution from $c_t$ with an indirect contribution through the next state.
Backpropagation through an unrolled dynamical system performs the same
calculation.

Finite differences instead perturb one decision at a time and rerun the
system. They require on the order of $p$ rollouts for $p=(T-1)m$ control
variables and introduce a finite-difference step size. Forward-mode
sensitivities propagate a separate Jacobian-vector product for each parameter
direction, so their work also scales with $p$. Reverse mode propagates one
costate vector backward and reads all partial derivatives from that sweep. For
a scalar objective, this replaces one rollout per parameter by one
forward-backward pass, at the cost of storing or checkpointing the state
trajectory.

The adjoint recursion is therefore the reverse-mode derivative of the
trajectory objective. States carry the nominal trajectory forward, costates
carry its sensitivity backward, and the local control derivatives combine the
two at each stage.

## Summary and Outlook

Fixing a finite horizon turns a discrete-time optimal control problem into a
finite-dimensional nonlinear program. Bolza, Lagrange, and Mayer formulations
differ only in where they place running and terminal costs; state augmentation
converts one form into another.

The CubeSat planner used a linear transition model, terminal phasing
constraints, and an altitude-loss objective. Its nominal feasibility
certificate did not transfer to the nonlinear variable-density replay. The
inference-frequency example produced the same distinction in a different
domain: a clock schedule feasible for one arrival forecast accumulated more
backlog when the same requests arrived earlier. A solver certificate applies to
the stated model, initial condition, and disturbance sequence.

Direct transcription keeps the states and controls as decision variables and
enforces every transition as an equality. The resulting NLP is large but
sparse, and state constraints are explicit. Single shooting keeps only the
controls and obtains the states by forward simulation. This reduces the number
of decision variables but creates a long dependency chain. Multiple shooting
introduces selected segment-boundary states, shortening those chains while
retaining sequential integration within each segment.

The ballistic boundary-value problem and the hydroelectric cascade apply this same construction at different scales. In the hydro application, reservoir levels are segment-boundary states, each interval contains a short local simulation, and Muskingum constraints coordinate routed flows between reaches.

The KKT conditions organize local optimality into primal feasibility,
stationarity, dual feasibility, and complementarity. Applied along a trajectory,
they give the discrete-time Pontryagin principle. States propagate forward,
costates propagate backward, and the Hamiltonian supplies the local control
stationarity condition. The same backward recursion computes all control
gradients with one reverse pass.

The next chapter replaces the continuous functions in a continuous-time
control problem by nodal polynomial values and collocation constraints.
[](mpc.md) then solves finite-horizon trajectory problems repeatedly as new
measurements arrive. Dynamic programming and policy optimization will replace
one planned trajectory by decisions defined over many possible states.

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

:::{exercise} Costate recursion as sensitivity
:label: ex-trajectories-costate

Consider an unconstrained DOCP with objective $J = c_T(x_T) + \sum_{t=1}^{T-1} c_t(x_t, u_t)$ and dynamics $x_{t+1} = f_t(x_t, u_t)$.

**(a)** Define the value-to-go from time $t$ as $V_t(x_t) = \min_{u_{t:T-1}} \left[ c_T(x_T) + \sum_{s=t}^{T-1} c_s(x_s, u_s) \right]$. Show that at an optimal trajectory, $\nabla_{x_t} V_t(x_t^\star) = \lambda_t^\star$, where $\lambda_t$ is the costate.

**(b)** Interpret the costate economically: what does $\lambda_t$ measure?

:::

:::{solution} ex-trajectories-costate
:class: dropdown

By definition, $V_t(x_t)$ is the minimum future cost starting from $x_t$. At the optimal trajectory, the envelope theorem gives $\nabla_{x_t} V_t = \lambda_t$, the marginal value of the state. Economically, $\lambda_t$ measures how much the optimal cost would decrease if we could perturb the state $x_t$ by a small amount—it is the "shadow price" of the state at time $t$.
:::

---

:::{exercise} Single shooting implementation
:label: ex-trajectories-single-shooting

Consider the scalar system $x_{t+1} = x_t + u_t$ with $x_1 = 0$ and objective:

$$
J = x_T^2 + \sum_{t=1}^{T-1} u_t^2.
$$

**(a)** Express $x_T$ as a function of the controls $u_1, \ldots, u_{T-1}$.

**(b)** Substitute into $J$ to obtain an unconstrained objective in the controls only.

**(c)** Implement single shooting in Python/JAX to minimize $J$ for $T = 10$. Use gradient descent with a learning rate of 0.1 for 100 iterations. Report the optimal controls and final cost.

:::

:::{solution} ex-trajectories-single-shooting
:class: dropdown

**(a)** $x_T = \sum_{t=1}^{T-1} u_t$.

**(b)** $J(u) = \left(\sum_{t=1}^{T-1} u_t\right)^2 + \sum_{t=1}^{T-1} u_t^2$.

**(c)** Sample code:
```python
import jax.numpy as jnp
from jax import grad

def objective(u):
    x_T = jnp.sum(u)
    return x_T**2 + jnp.sum(u**2)

T = 10
u = jnp.zeros(T - 1)
for _ in range(100):
    u = u - 0.1 * grad(objective)(u)

print(f"Optimal u: {u}, Cost: {objective(u):.4f}")
```
The optimal controls should be approximately equal and negative, with total cost near $0$.
:::

---

:::{exercise} Multiple shooting segments
:label: ex-trajectories-multiple-shooting

Using the same problem as Exercise 5, implement multiple shooting with $K = 3$ segments.

**(a)** Define the segment boundaries and the continuity defects.

**(b)** Set up the NLP with decision variables $[x_1, x_4, x_7, u_1, \ldots, u_9]$ (for $T=10$, with segments of length 3).

**(c)** Compare the convergence behavior to single shooting. Does multiple shooting require fewer iterations to reach the same tolerance?

:::

:::{solution} ex-trajectories-multiple-shooting
:class: dropdown

The defects are $d_k = x_{k+1}^{\text{simulated}} - x_{k+1}^{\text{variable}}$ at segment boundaries. You can minimize $J + \rho \sum_k \|d_k\|^2$ for large $\rho$ (penalty method) or use a constrained solver. Multiple shooting typically converges faster for longer horizons because the optimization landscape is better conditioned.
:::

---

:::{exercise} Adjoint gradient verification
:label: ex-trajectories-adjoint

For the system $x_{t+1} = x_t^2 + u_t$ with $x_1 = 0.5$, $T = 4$, and objective $J = x_4$:

**(a)** Compute the gradient $\nabla_{u} J$ using finite differences (perturb each $u_t$ by $\epsilon = 10^{-5}$).

**(b)** Compute the gradient using the adjoint method: first simulate forward to get $x_{1:4}$, then propagate costates backward using $\lambda_4 = 1$ and $\lambda_t = 2x_t \lambda_{t+1}$.

**(c)** Verify that the two methods give the same answer. Which is more efficient for large $T$?

:::

:::{solution} ex-trajectories-adjoint
:class: dropdown

For controls $u = [0, 0, 0]$: forward simulation gives $x_1 = 0.5$, $x_2 = 0.25$, $x_3 = 0.0625$, $x_4 = 0.00390625$.

Adjoint: $\lambda_4 = 1$, $\lambda_3 = 2(0.0625)(1) = 0.125$, $\lambda_2 = 2(0.25)(0.125) = 0.0625$, $\lambda_1 = 2(0.5)(0.0625) = 0.0625$.

Control gradients: $\nabla_{u_t} J = \lambda_{t+1}$, so $\nabla_u J = [\lambda_2, \lambda_3, \lambda_4] = [0.0625, 0.125, 1.0]$.

Finite differences should match. The adjoint is $O(T)$ work regardless of the number of controls; finite differences require $O(T \cdot m)$ rollouts for $m$-dimensional control.
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

:::{exercise} Count the decisions
:label: ex-trajectories-check-1

A single-shooting problem has horizon $T$ and scalar controls. How many optimization variables remain after eliminating the states, and what computation couples an early control to the terminal cost?
:::

:::{solution} ex-trajectories-check-1
:class: dropdown

There are $T$ control variables. Forward simulation couples every early control to all later states and therefore to the terminal cost.
:::

:::{exercise} Shooting trade-off
:label: ex-trajectories-check-2

Why can multiple shooting be easier to optimize than single shooting even though it introduces more decision variables?
:::

:::{solution} ex-trajectories-check-2
:class: dropdown

Intermediate states break a long sensitive rollout into shorter segments. The resulting continuity constraints are sparse, and derivatives need not propagate through the full horizon in one chain.
:::

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
