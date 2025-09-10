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

In the previous chapter, we examined different ways to represent dynamical systems: continuous versus discrete time, deterministic versus stochastic, fully versus partially observable, and even simulation-based views such as agent-based or programmatic models. Our focus was on the **structure of models**: how they capture evolution, uncertainty, and information.

In this chapter, we turn to what makes these models useful for **decision-making**. The goal is no longer just to describe how a system behaves, but to leverage that description to **compute actions over time**. This doesn’t mean the model prescribes actions on its own. Rather, it provides the scaffolding for optimization: given a model and an objective, we can derive the control inputs that make the modeled system behave well according to a chosen criterion. 

Our entry point will be trajectory optimization. By a **trajectory**, we mean the time-indexed sequence of states and controls that the system follows under a plan: the states $(\mathbf{x}_1, \dots, \mathbf{x}_T)$ together with the controls $(\mathbf{u}_1, \dots, \mathbf{u}_{T-1})$. In this chapter, we focus on an **open-loop** viewpoint: starting from a known initial state, we compute the entire sequence of controls in advance and then apply it as-is. This is appealing because, for discrete-time problems, it yields a finite-dimensional optimization over a vector of decisions and cleanly exposes the structure of the constraints. In continuous time, the base formulation is infinite-dimensional; in this course we will rely on direct methods—time discretization and parameterization—to transform it into a finite-dimensional nonlinear program.

Open loop also has a clear limitation: if reality deviates from the model—due to disturbances, model mismatch, or unanticipated events—the state you actually reach may differ from the predicted one. The precomputed controls that were optimal for the nominal trajectory can then lead you further off course, and errors can compound over time.

Later, we will study **closed-loop (feedback)** strategies, where the choice of action at time $t$ can depend on the state observed at time $t$. Instead of a single sequence, we optimize a policy $\pi_t$ mapping states to controls, $\mathbf{u}_t = \pi_t(\mathbf{x}_t)$. Feedback makes plans resilient to unforeseen situations by adapting on the fly, but it leads to a more challenging problem class. We start with open-loop trajectory optimization to build core concepts and tools before tackling feedback design.



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

The stage cost reflects ongoing penalties—energy, delay, risk. The terminal cost measures the value (or cost) of ending in a particular state. **Together, these give a discrete-time Bolza problem with path constraints and bounds**:

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

Written this way, it may seem obvious that the decision variables are the controls $\mathbf{u}_t$. After all, in most intuitive descriptions of control, we think of choosing inputs to influence the system. But notice that in the program above, the entire state trajectory also appears as a set of variables, linked to the controls by the dynamics constraints. This is intentional: it reflects one way of writing the problem that makes the constraints explicit.

Why introduce $\mathbf{x}_t$ as decision variables if they can be simulated forward from the controls? Many readers hesitate here, and the question is natural: *If the model is deterministic and $\mathbf{x}_1$ is known, why not pick $\mathbf{u}_{1:T-1}$ and compute $\mathbf{x}_{2:T}$ on the fly?* That instinct leads to **single shooting**, a method we will return to shortly.

Already in this formulation, though, we see an important theme: **the structure of the problem matters**. Ignoring it can make our life much harder. The reason is twofold:

* **Dimensionality grows with the horizon.** For a horizon of length $T$, the program has roughly $(T-1)(m+n)$ decision variables.
* **Temporal coupling.** Each control affects all future states and costs. The feasible set is not a simple box but a narrow manifold defined by the dynamics.

Together, these features explain why specialized methods exist and why the way we write the problem influences the algorithms we can use. Whether we keep states explicit or eliminate them through forward simulation determines not just the problem size, but also its conditioning and the trade-offs between robustness and computational effort.

## Existence of Solutions and Optimality Conditions

Now that we have the optimization problem written down, a natural question arises: **does this program always have a solution?** And if it does, how can we recognize one when we see it? These questions bring us into the territory of feasibility and optimality conditions.


### When does a solution exist?

Notice first that nothing in the problem statement required the dynamics

$$
\mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)
$$

to be stable. In fact, many problems of interest involve unstable systems; think of balancing a pole or steering a spacecraft. What matters is that the dynamics are **well defined**: given a state–control pair, the rule $\mathbf{f}_t$ produces a valid next state.

In continuous time, one usually requires $\mathbf{f}$ to be continuous (often Lipschitz continuous) in $\mathbf{x}$ so that the ODE has a unique solution on the horizon of interest. In discrete time, the requirement is lighter—we only need the update map to be well posed.

Existence also hinges on **feasibility**. A candidate control sequence must generate a trajectory that respects all constraints: the dynamics, any bounds on state and control, and any terminal requirements. If no such sequence exists, the feasible set is empty and the problem has no solution. This can happen if the constraints are overly strict, or if the system is uncontrollable from the given initial condition.


### What does optimality look like?

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

This is the **LICQ** (Linear Independence Constraint Qualification). In convex problems, **Slater’s condition** (existence of a strictly feasible point) plays a similar role. You can think of these as the assumptions that let the linearized KKT equations be solvable; we do not literally invert that Jacobian, but the full-rank property is the key ingredient that would make such an inversion possible in principle.

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

In our trajectory problems, $\mathbf{z}$ stacks state and control trajectories, $G$ enforces the dynamics, and $H$ collects bounds and path constraints. The equalities’ multipliers act as **costates** or **shadow prices** for the dynamics. Writing the KKT system stage by stage yields the discrete-time Pontryagin principle, derived next. For convex programs these conditions are also sufficient.

*What fails without a CQ?* If the active gradients are dependent (for example duplicated or nearly parallel), the Jacobian loses rank; multipliers may then be nonunique or fail to exist, and the linearized equations become ill-posed. In transcribed trajectory problems this shows up as dependent dynamic constraints or redundant path constraints, which leads to fragile solver behavior.

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

Applying Newton’s method to this system gives the linear KKT solve

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

This is exactly the step computed by **Sequential Quadratic Programming (SQP)** in the equality-constrained case: it is Newton’s method on the KKT equations. For general problems with inequalities, SQP forms a **quadratic subproblem** by quadratically modeling $F$ with $\nabla_{\mathbf{z}\mathbf{z}}^2\mathcal{L}$ and linearizing the constraints, then solves that QP with line search or trust region. In least-squares-like problems one often uses **Gauss–Newton** (or a Levenberg–Marquardt trust region) as a positive-definite approximation to the Lagrangian Hessian.

**In trajectory optimization.** After transcription, the KKT matrix inherits banded/sparse structure from the dynamics. Newton/SQP steps can be computed efficiently by exploiting this structure; in the special case of quadratic models and linearized dynamics, the QP reduces to an LQR solve along the horizon (this is the backbone of iLQR/DDP-style methods). Primal–dual updates provide simpler iterations and are easy to implement; augmented terms are typically needed to obtain stable progress when constraints couple stages.

**When to use which.** Primal–dual gradients give lightweight iterations and are good for warm starts or as inner loops with penalties. SQP/Newton gives rapid local convergence when you are close to a solution and LICQ holds; use trust regions or line search to globalize.


## Examples of DOCPs
To make things concrete, here are three problems that are naturally posed as discrete-time OCPs. In each case, we seek an optimal trajectory of states and controls over a finite horizon.

### Periodic Inventory Control
Decisions are made once per period: choose order quantity $u_k \ge 0$ to meet forecast demand $d_k$. The state $x_k$ is on-hand inventory with dynamics $x_{k+1} = x_k + u_k - d_k$. A typical stage cost is $c_k(x_k,u_k) = h\,[x_k]_+ + p\,[-x_k]_+ + c\,u_k$, trading off holding, backorder, and ordering costs. The horizon objective is $\min \sum_{k=0}^{T-1} c_k(x_k,u_k)$ subject to bounds.

### End-of-Day Portfolio Rebalancing
At each trading day $k$, choose trades $u_k$ to adjust holdings $h_k$ before next-day returns $r_{k}$ realize. Deterministic planning uses predicted returns $\mu_k$, with dynamics $h_{k+1} = (h_k + u_k) \odot (\mathbf{1} + \mu_k)$ and budget/box constraints. The stage cost can capture transaction costs and risk, e.g., $c_k(h_k,u_k) = \tau\lVert u_k \rVert_1 + \tfrac{\lambda}{2}\,h_k^\top \Sigma_k h_k$, with a terminal utility or wealth objective.

### Daily Ad-Budget Allocation with Carryover
Allocate spend $u_k \in [0, U_{\max}]$ to build awareness $s_k$ with carryover dynamics $s_{k+1} = \alpha s_k + \beta u_k$. Conversions/revenue at day $k$ follow a response curve $g(s_k,u_k)$; the goal is $\max \sum_{k=0}^{T-1} g(s_k,u_k) - c\,u_k$ subject to spend limits. This is naturally discrete because decisions and measurements occur daily.

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

#### Example: HTTP Retrier Optimization

As an example we cast the problem of optimizing a "HTTP retrier with backoff" as a DOCP where the state tracks wall-clock time, attempt index, success, last code, and jitter; the control is the chosen wait time before the next request (the backoff schedule); the transition encapsulates waiting and a probabilistic request outcome; and the objective penalizes latency and failure. We then optimize the schedule either directly (per-step SPSA) or via a two-parameter exponential policy using common random numbers for variance reduction.

```{code-cell} ipython3
:tags: [hide-input]
:load: _static/prog.py
```


#### Example: Gradient Descent with Momentum as DOCP

To connect this lens to familiar practice—and to hyperparameter optimization—treat the learning rate and momentum (or their schedules) as controls. Rather than fixing them a priori, we can optimize them as part of a trajectory optimization. The optimizer itself becomes the dynamical system whose execution we shape to minimize final loss.

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

What if the program branches? Suppose we insert a "skip-small-gradients” branch

$$
\boldsymbol{\theta}_{k+1}=\boldsymbol{\theta}_k - \alpha_k\,\mathbf{m}_{k+1}\,\mathbf{1}\{ \lVert\mathbf{g}_k\rVert>\tau\},
$$

which is non-differentiable because of the indicator. The DOCP view still applies, but gradients are unreliable. Two practical paths: smooth the branch (e.g., replace $\mathbf{1}\{\cdot\}$ with $\sigma((\lVert\mathbf{g}_k\rVert-\tau)/\epsilon)$ for small $\epsilon$) and use autodiff; or go derivative-free on $\{\alpha_k,\beta_k,\tau\}$ (e.g., SPSA or CMA-ES) while keeping the inner dynamics exact.

## Variants: Lagrange and Mayer Problems

The Bolza form is general enough to cover most situations, but two common special cases are worth noting:

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

Why do this? Two reasons. First, it often simplifies **mathematical derivations**, as we will see later when deriving necessary conditions. Second, it can **streamline algorithmic implementation**: instead of writing separate code paths for Mayer, Lagrange, and Bolza problems, we can reduce everything to one canonical form. That said, this "one size fits all” approach isn’t always best in practice—specialized formulations can sometimes be more efficient computationally, especially when the running cost has simple structure.


The unifying theme is that a DOCP may look like a generic NLP on paper, but its structure matters. Ignoring that structure often leads to impractical solutions, whereas formulations that expose sparsity and respect temporal coupling allow modern solvers to scale effectively. In the following sections, we will examine how these choices play out in practice through single shooting, multiple shooting, and collocation methods, and why different formulations strike different trade-offs between robustness and computational effort.

# Numerical Methods for Solving DOCPs

Before we discuss specific algorithms, it is useful to clarify the goal: we want to recast a discrete-time optimal control problem as a standard nonlinear program (NLP). Collect all decision variables—states, controls, and any auxiliary variables—into a single vector $\mathbf{z}\in\mathbb{R}^{n_z}$ and write

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

The next sections work through these formulations—starting with simultaneous methods, then sequential methods, and finally multiple shooting—before discussing how generic NLP solvers and specialized algorithms leverage the resulting structure in practice.

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

This direct transcription is attractive because it is faithful to the model and exposes sparsity. The Jacobian of $G$ has a block bi-diagonal structure induced by the dynamics, and the KKT matrix is sparse and structured—properties exploited by interior-point and SQP methods. The trade-off is size: with state dimension $n$ and control dimension $m$, the decision vector has $(T\!\cdot\!n) + ((T\!-
1)\cdot m)$ entries, and there are roughly $(T\!-
1)\cdot n$ dynamic equalities plus any path and boundary conditions. Techniques such as partial or full condensing eliminate state variables to reduce the equality set (at the cost of denser matrices), while keeping states explicit preserves sparsity and often improves robustness on long horizons and in the presence of state constraints.

Compared to alternatives, simultaneous methods avoid the long nonlinear dependency chains of single shooting and make it easier to impose state/path constraints. They can, however, demand more memory and per-iteration linear algebra, so practical performance hinges on exploiting sparsity and good initialization.

The same logic applies when selecting an optimizer. For small-scale problems, it is common to rely on general-purpose routines such as those in `scipy.optimize.minimize`. Derivative-free methods like Nelder–Mead require no gradients but scale poorly as dimensionality increases. Quasi-Newton schemes such as BFGS work well for moderate dimensions and can approximate gradients by finite differences, while large-scale trajectory optimization often calls for gradient-based constrained solvers such as interior-point or sequential quadratic programming methods that can exploit sparse Jacobians and benefit from automatic differentiation. Stochastic techniques, including genetic algorithms, simulated annealing, or particle swarm optimization, occasionally appear when gradients are unavailable, but their cost grows rapidly with dimension and they are rarely competitive for structured optimal control problems.

<!-- ### On the Choice of Optimizer

Although the code example uses SLSQP, many alternatives exist. `scipy.optimize.minimize` provides a menu of options, and each has implications for speed, robustness, and scalability:

* **Derivative-free methods** such as Nelder–Mead avoid gradients altogether. They are attractive when gradients are unavailable or noisy, but they scale poorly with dimension.
* **Quasi-Newton methods** like BFGS approximate gradients by finite differences. They work well for moderate-scale problems and often outperform derivative-free schemes when the objective is smooth.
* **Gradient-based constrained solvers** such as interior-point or SQP methods exploit derivatives—exact or automatic—and are typically the most efficient for large structured problems like trajectory optimization.

Beyond these, **stochastic optimizers** occasionally appear in practice, especially when gradients are unreliable or the loss landscape is rugged. Random search is the simplest example, while genetic algorithms, simulated annealing, and particle swarm optimization introduce mechanisms for global exploration at the cost of significant computational effort.

Which method to choose depends on the context: problem size, availability of derivatives, and computational resources. When automatic differentiation is accessible, first-order methods like L-BFGS or Adam often dominate, particularly for single-shooting formulations where the objective is smooth and unconstrained except for simple bounds. This is why researchers with a machine learning background tend to gravitate toward these techniques: they integrate seamlessly with existing frameworks and run efficiently on GPUs. -->
<!-- 
### Example: Direct Solution to the Eco-cruise Problem

Many modern vehicles include features that aim to improve energy efficiency without requiring extra effort from the driver. One such feature is Eco-Cruise. Unlike traditional cruise control, which keeps the car at a fixed speed regardless of conditions, Eco-Cruise adjusts speed within small margins to reduce energy consumption. The reasoning is straightforward: holding speed up a hill by applying full throttle uses more energy than allowing the car to slow slightly and regain speed later. Some systems go further by using map data, anticipating slopes and curves to plan ahead. These ideas are no longer experimental; several manufacturers already deploy predictive cruise systems based on navigation input.

The setup we will use is slightly idealized, but not unrealistic. It assumes that the driver provides a destination and an acceptable time target, something that most navigation systems already require. With that information, the controller can decide how fast to go and when to accelerate while ensuring the trip remains on schedule. Framing the problem in this way allows us to cast Eco-Cruise as a trajectory optimization exercise and to explore the structure of a discrete-time optimal control problem.

Consider a 1 km segment of road that must be completed in exactly 60 seconds. We divide this horizon into 60 steps of one second each. At step $t$, the state consists of the cumulative distance $s_t$ and the speed $v_t$. The control input is the longitudinal acceleration $u_t$. With a time step of one second, the dynamics are written as

$$
s_{t+1} = s_t + v_t, \qquad
v_{t+1} = v_t + u_t.
$$

The trip starts from rest, so $s_1 = 0$ and $v_1 = 0$, and it must end at $s_{T+1} = 1000$ m with $v_{T+1} = 0$.

Energy consumption depends on both acceleration and speed. Rather than model the details of rolling resistance, drivetrain losses, and aerodynamics, we adopt a simple quadratic approximation. Each stage incurs a cost

$$
c_t(v_t, u_t) = \tfrac{1}{2}\beta u_t^2 + \tfrac{1}{2}\gamma v_t^2,
$$

where the first term penalizes strong accelerations and the second discourages high cruising speed. Reasonable values are $\beta = 1.0$ and $\gamma = 0.1$. The objective is to minimize the sum of these stage costs across the horizon:

$$
\min \sum_{t=1}^{T} \bigl( \tfrac{\beta}{2}u_t^2 + \tfrac{\gamma}{2}v_t^2 \bigr).
$$

The optimization must also respect physical limits. Speeds must remain between zero and $20\ \text{m/s}$ (about 72 km/h), and accelerations are bounded by $|u_t| \le 3\ \text{m/s}^2$ for comfort and safety.


The complete formulation is

$$
\begin{aligned}
\min_{\{s_t,v_t,u_t\}} \ & \sum_{t=1}^{T} \bigl( \tfrac{\beta}{2}u_t^2 + \tfrac{\gamma}{2}v_t^2 \bigr) \\
\text{subject to}\ & s_{t+1}-s_t-v_t = 0,\ \ v_{t+1}-v_t-u_t = 0,\ t=1,\dots,T, \\
& s_1 = 0,\ v_1 = 0,\ s_{T+1} = 1000,\ v_{T+1} = 0, \\
& 0 \le v_t \le 20,\ \ |u_t|\le 3.
\end{aligned}
$$

#### Solution

Once the objective and constraints are expressed as Python functions, the problem can be passed to a generic optimizer with very little extra work. Here is a direct implementation using `scipy.optimize.minimize` with the SLSQP method:

```{code-cell} ipython3
:load: code/eco-cruise.py
:tags: [remove-input, remove-output]
```

```{glue:figure} eco_cruise_figure
:figwidth: 100%
:name: "fig-eco-cruise"

Eco-Cruise optimization results showing the comparison between energy-efficient and naive trajectory approaches.
```

``````{tab-set}
:tags: [full-width]

`````{tab-item} Visualization
```{raw} html
<script src="_static/iframe-modal.js"></script>
<div id="eco-cruise-container"></div>
<script>
createIframeModal({
  containerId: 'eco-cruise-container',
  iframeSrc: '_static/eco-cruise-demo.html',
  title: 'Eco-Cruise Optimization Visualization',
  aspectRatio: '200%',
  maxWidth: '1400px',
  maxHeight: '900px'
});
</script>
`````

`````{tab-item} Code
```{literalinclude} code/eco-cruise.py
:language: python
```
`````
``````

The function `scipy.optimize.minimize` expects three things: an objective function that returns a scalar cost, a set of constraints grouped as equality or inequality functions, and bounds on individual variables. Everything else is about bookkeeping.

The first step is to gather all decision variables—positions, speeds, and accelerations—into a single vector $\mathbf{z}$. Helper routines like `unpack` then slice this vector back into its components so that the rest of the code reads naturally. The objective function mirrors the analytical form of the cost: it sums quadratic penalties on speeds and accelerations across the horizon.

Dynamics and boundary conditions appear as equality constraints. Each entry in `dynamics` enforces one of the discrete-time equations

$$
s_{t+1} - s_t - v_t = 0,\qquad
v_{t+1} - v_t - u_t = 0,
$$

while `boundary` pins down the start and end conditions. Together, these ensure that any candidate solution corresponds to a physically consistent trajectory.

Bounds serve two purposes: they impose physical limits on speed and acceleration and keep the otherwise unbounded position variables within a large but finite range. This prevents the optimizer from exploring meaningless regions of the search space during intermediate iterations.

Finally, an initial guess is constructed by interpolating a straight line for the position, assigning a constant speed, and setting accelerations to zero. This is not intended to be optimal; it simply gives the solver a feasible starting point close enough to the constraint manifold to converge quickly.

Once these components are in place, the call to `minimize` does the rest. Internally, SLSQP linearizes the constraints, builds a quadratic subproblem, and iterates until both the Karush–Kuhn–Tucker conditions and the stopping tolerances are met. From the user’s perspective, the heavy lifting reduces to providing functions that compute costs and residuals—everything else is handled by the solver. -->



## Sequential Methods

The previous section showed how a discrete-time optimal control problem can be solved by treating all states and controls as decision variables and enforcing the dynamics as equality constraints. This produces a nonlinear program that can be passed to solvers such as `scipy.optimize.minimize` with the SLSQP method. For short horizons, this approach is straightforward and works well; the code stays close to the mathematical formulation.

It also has a real advantage: by keeping the states explicit and imposing the dynamics through constraints, we anchor the trajectory at multiple points. This extra structure helps stabilize the optimization, especially for long horizons where small deviations in early steps can otherwise propagate and cause the optimizer to drift or diverge. In that sense, this formulation is better conditioned and more robust than approaches that treat the dynamics implicitly.

The drawback is scale. As the horizon grows, the number of variables and constraints grows with it, and all are coupled by the dynamics. Each iteration of a sequential quadratic programming (SQP) or interior-point method requires building and factorizing large Jacobians and Hessians. These methods have been embedded in reinforcement learning and differentiable programming pipelines—through implicit layers or differentiable convex solvers—but the cost is significant. They remain serial, rely on repeated linear algebra factorizations, and are difficult to parallelize efficiently. When thousands of such problems must be solved inside a learning loop, the overhead becomes prohibitive.

This motivates an alternative that aligns better with the computational model of machine learning. If the dynamics are deterministic and state constraints are absent (or reducible to simple bounds on controls), we can eliminate the equality constraints altogether by making the states implicit. Instead of solving for both states and controls, we fix the initial state and roll the system forward under a candidate control sequence. This is the essence of **single shooting**.

The term "shooting” comes from the idea of *aiming and firing* a trajectory from the initial state: you pick a control sequence, integrate (or step) the system forward, and see where it lands. If the final state misses the target, you adjust the controls and try again: like adjusting the angle of a shot until it hits the mark. It is called **single** shooting because we compute the entire trajectory in one pass from the starting point, without breaking it into segments. Later, we will contrast this with **multiple shooting**, where the horizon is divided into smaller arcs that are optimized jointly to improve stability and conditioning.

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
\qquad
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
3. Solve $\min_{\mathbf{u}} J(\mathbf{u})$ subject to $\mathbf{u}_{\mathrm{lb}} \le \mathbf{u}_t \le \mathbf{u}_{\mathrm{ub}}$
4. Return $\mathbf{u}^*_{1:T-1}$
```

In JAX or PyTorch, this loop can be JIT-compiled and differentiated automatically. Any gradient-based optimizer—L-BFGS, Adam, even SGD—can be applied, making the pipeline look very much like training a neural network. In effect, we are "backpropagating through the world model” when computing $\nabla J(\mathbf{u})$.

Single shooting is attractive for its simplicity and compatibility with differentiable programming, but it has limitations. The absence of intermediate constraints makes it sensitive to initialization and prone to numerical instability over long horizons. When state constraints or robustness matter, formulations that keep states explicit—such as multiple shooting or collocation—become preferable. These trade-offs are the focus of the next section.

<!-- 
```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show code demonstration"
:  code_prompt_hide: "Hide code demonstration"
:load: code/single_shooting_unrolled.py
``` -->

## In Between Sequential and Simultaneous

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

Compared to the full NLP, we no longer introduce every intermediate state as a variable—only the anchors at segment boundaries. Inside each segment, states are reconstructed by simulation. Compared to single shooting, these anchors break the long dependency chain that makes optimization unstable: gradients only have to travel across $L$ steps before they hit a decision variable, rather than the entire horizon. This is the same reason why exploding or vanishing gradients appear in deep recurrent networks: when the chain is too long, information either dies out or blows up. Multiple shooting shortens the chain and improves conditioning.

By adjusting the number of segments $K$, we can interpolate between the two extremes: $K = 1$ gives single shooting, while $K = T$ recovers the full direct NLP. In practice, a moderate number of segments often strikes a good balance between robustness and complexity.


```{code-cell} ipython3
:tags: [hide-input]
:load: code/multiple_shooting.py
```




# The Discrete-Time Pontryagin Principle

If we take the Bolza formulation of the DOCP and apply the KKT conditions directly, we obtain an optimization system with many multipliers and constraints. Written in raw form, it looks like any other nonlinear program. But in control, this structure has a long history and a name of its own: the **Pontryagin principle**. In fact, the discrete-time version can be seen as the structured KKT system that emerges once we introduce multipliers for the dynamics and collect terms stage by stage.

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

**Costate recursion (backward “adjoint” equation)**

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

The triplet “forward state, backward costate, control stationarity” is the discrete-time Euler–Lagrange system tailored to control with dynamics. It is the same KKT logic as before, but organized stagewise through the Hamiltonian.

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

## The adjoint equation as reverse accumulation

Optimization needs sensitivities. In trajectory problems we adjust decisions—controls or parameters—to reduce an objective while respecting dynamics and constraints. First‑order methods in the unconstrained case (e.g., gradient descent, L‑BFGS, Adam) require the gradient of the objective with respect to all controls, and constrained methods (SQP, interior‑point) require gradients of the Lagrangian, i.e., of costs and constraints. The discrete‑time adjoint equations provide these derivatives in a way that scales to long horizons and many decision variables.

Consider

$$
J = c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t),
\qquad \mathbf{x}_{t+1}=\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t).
$$

A single forward rollout computes and stores the trajectory $\mathbf{x}_{1:T}$. A single backward sweep then applies the reverse‑mode chain rule stage by stage. Defining the costate by

$$
\boldsymbol{\lambda}_T = \nabla_{\mathbf{x}} c_T(\mathbf{x}_T),\qquad
\boldsymbol{\lambda}_t = \nabla_{\mathbf{x}} c_t(\mathbf{x}_t,\mathbf{u}_t) + \big[\nabla_{\mathbf{x}} \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)\big]^\top \boldsymbol{\lambda}_{t+1},\quad t=T-1,\dots,1,
$$

yields exactly the discrete‑time adjoint (PMP) recursion. The gradient with respect to each control follows from the same reverse pass:

$$
\nabla_{\mathbf{u}_t} J = \nabla_{\mathbf{u}} c_t(\mathbf{x}_t,\mathbf{u}_t) + \big[\nabla_{\mathbf{u}} \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)\big]^\top \boldsymbol{\lambda}_{t+1}.
$$

Two points are worth emphasizing. Computationally, this reverse accumulation produces all control gradients with one forward rollout and one backward adjoint pass; its cost is essentially a small constant multiple of simulating the system once. Conceptually, the costate $\boldsymbol{\lambda}_t$ is the marginal effect of perturbing the state at time $t$ on the total objective; the control gradient combines a direct contribution from $c_t$ and an indirect contribution through how $\mathbf{u}_t$ changes the next state. This is the same structure that underlies backpropagation, expressed for dynamical systems.

It is instructive to contrast this with alternatives. Black‑box finite differences perturb one decision at a time and re‑roll the system, requiring on the order of $p$ rollouts for $p$ decision variables and suffering from step‑size and noise issues—prohibitive when $p=(T-1)m$ for an $m$‑dimensional control over $T$ steps. Forward‑mode (tangent) sensitivities propagate Jacobian–vector products for each parameter direction; their work also scales with $p$. Reverse‑mode (the adjoint) instead propagates a single vector $\boldsymbol{\lambda}_t$ backward and then reads off all partial derivatives $\nabla_{\mathbf{u}_t} J$ at once. For a scalar objective, its cost is effectively independent of $p$, at the price of storing (or checkpointing) the forward trajectory. This scalability is why the adjoint is the method of choice for gradient‑based trajectory optimization and for constrained transcriptions via the Hamiltonian. 
