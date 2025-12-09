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

In this chapter, we turn to what makes these models useful for **decision-making**. The goal is no longer just to describe how a system behaves, but to leverage that description to **compute actions over time**. This doesn't mean the model prescribes actions on its own. Rather, it provides the scaffolding for optimization: given a model and an objective, we can derive the control inputs that make the modeled system behave well according to a chosen criterion. 

Our entry point will be trajectory optimization. By a **trajectory**, we mean the time-indexed sequence of states and controls that the system follows under a plan: the states $(\mathbf{x}_1, \dots, \mathbf{x}_T)$ together with the controls $(\mathbf{u}_1, \dots, \mathbf{u}_{T-1})$. In this chapter, we focus on an **open-loop** viewpoint: starting from a known initial state, we compute the entire sequence of controls in advance and then apply it as-is. This is appealing because, for discrete-time problems, it yields a finite-dimensional optimization over a vector of decisions and cleanly exposes the structure of the constraints. In continuous time, the base formulation is infinite-dimensional; in this course we will rely on direct methods (time discretization and parameterization) to transform it into a finite-dimensional nonlinear program.

Open loop also has a clear limitation: if reality deviates from the model, whether due to disturbances, model mismatch, or unanticipated events, the state you actually reach may differ from the predicted one. The precomputed controls that were optimal for the nominal trajectory can then lead you further off course, and errors can compound over time.

Later, we will study **closed-loop (feedback)** strategies, where the choice of action at time $t$ can depend on the state observed at time $t$. Instead of a single sequence, we optimize a policy $\pi_t$ mapping states to controls, $\mathbf{u}_t = \pi_t(\mathbf{x}_t)$. Feedback makes plans resilient to unforeseen situations by adapting on the fly, but it leads to a more challenging problem class. We start with open-loop trajectory optimization to build core concepts and tools before tackling feedback design.

```{admonition} Learning Goals
:class: note

After studying this chapter, you should be able to:

1. Formulate a discrete-time optimal control problem (DOCP) in Bolza, Lagrange, and Mayer forms, and convert between them.
2. State the KKT conditions for a constrained NLP and explain the role of constraint qualifications.
3. Derive the discrete-time Pontryagin principle from the Lagrangian of a DOCP.
4. Implement single shooting and multiple shooting methods for trajectory optimization.
5. Explain the trade-offs between simultaneous (direct transcription) and sequential (shooting) methods.
6. Compute gradients of the objective with respect to controls using the adjoint (costate) recursion.
```

```{admonition} Prerequisites
:class: tip

This chapter assumes familiarity with:
- Multivariable calculus (gradients, Jacobians, chain rule)
- Basic optimization (unconstrained minimization, gradient descent)
- Linear algebra (matrix-vector products, linear systems)
- Dynamical systems from the previous chapter
```

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

This is the **LICQ** (Linear Independence Constraint Qualification). In convex problems, **Slater's condition** (existence of a strictly feasible point) plays a similar role. You can think of these as the assumptions that let the linearized KKT equations be solvable; we do not literally invert that Jacobian, but the full-rank property is the key ingredient that would make such an inversion possible in principle.

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

```{code-cell} python
:tags: [hide-input]

#  label: fig-ocp-http-retrier
#  caption: Console output comparing the baseline backoff schedule with SPSA-optimized schedules for the HTTP retrier DOCP, including costs, attempts, and success codes.

%config InlineBackend.figure_format = 'retina'
from dataclasses import dataclass
import math, random
import matplotlib.pyplot as plt

# Apply book style
try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except (ImportError, OSError):
    pass  # Use matplotlib defaults

# ---------------------------
# PROGRAM = "HTTP retrier with backoff"
# ---------------------------

@dataclass
class State:
    t: float            # wall-clock time (s)
    k: int              # attempt index
    done: bool          # success flag
    code: int | None    # last HTTP code or None
    jitter: float       # per-run jitter (simulates clock/socket noise)

# Controls (decision variables): per-step wait times (backoff schedule)
# u[k] can be optimized; in a fixed policy you'd set u[k] = base * gamma**k
# We'll keep them bounded for realism.
def clamp(x, lo, hi): return max(lo, min(hi, x))

# Simulated environment: availability is time-varying (spiky outage)
def server_success_prob(t: float) -> float:
    # Low availability for the first 2 seconds, then rebounds
    base = 0.15 if t < 2.0 else 0.85
    # Some diurnal-like wobble (toy)
    wobble = 0.1 * math.sin(2 * math.pi * (t / 3.0))
    return clamp(base + wobble, 0.01, 0.99)

def http_request():
    # Just returns a code; success = 200, failure = 503
    return 200 if random.random() < 0.5 else 503

# -------- DOCP ingredients --------
# State x_k = (t, k, done, code, jitter)
# Control u_k = wait time before next attempt (our backoff schedule entry)
# Transition Phi_k: one "program step" = (optional wait) + (one request) + (branch)
def Phi(state: State, u_k: float) -> State:
    if state.done:
        # No-ops after success (absorbing state)
        return State(state.t, state.k, True, state.code, state.jitter)

    # 1) Wait according to control (backoff schedule) + jitter
    wait = clamp(u_k + 0.02 * state.jitter, 0.0, 3.0)
    t = state.t + wait

    # 2) Environment: success probability depends on time t
    p = server_success_prob(t)

    # 3) "Perform request": success with prob p; otherwise 503
    code = 200 if random.random() < p else 503
    done = (code == 200)

    # 4) Advance attempt counter and wall clock
    return State(t=t, k=state.k + 1, done=done, code=code, jitter=state.jitter)

# Stage cost: latency penalty each step; heavy penalty if still failing late
def stage_cost(state: State, u_k: float) -> float:
    # Latency/energy per unit wait + small per-step overhead when not done
    return 0.20 * u_k + (0.00 if state.done else 0.002)

# Terminal cost: if failed after horizon, big penalty; if succeeded, pay total time
def terminal_cost(state: State, max_attempts: int) -> float:
    # Pay for elapsed time; fail late incurs extra penalty
    return 0.3 * state.t + (5.0 if (not state.done and state.k >= max_attempts) else 0.0)

def rollout(u, max_attempts=8, seed=0):
    random.seed(seed)
    s = State(t=0.0, k=0, done=False, code=None, jitter=random.uniform(-1,1))
    J = 0.0
    for k in range(max_attempts):
        J += stage_cost(s, u[k])
        s = Phi(s, u[k])
        if s.done:  # early stop like a real program
            break
    J += terminal_cost(s, max_attempts)
    return J, s  # return final state for debugging if needed

# ---------- helpers for SPSA with common random numbers ----------
def eval_policy(u, seeds, max_attempts=8):
    # Average over a fixed set of seeds (CRN helps SPSA a lot)
    Js = []
    for sd in seeds:
        J, _ = rollout(u, max_attempts=max_attempts, seed=sd)
        Js.append(J)
    return sum(Js) / len(Js)

def project_waits(u):
    # Keep waits in [0, 3] for realism
    return [max(0.0, min(3.0, x)) for x in u]

# ---------- schedule parameterizations ----------
def schedule_exp(base, gamma, K):
    # u[k] = base * gamma**k
    return [base * (gamma ** k) for k in range(K)]

# If you prefer per-step but monotone nonnegative waits, use softplus increments:
def schedule_softplus(z, K):
    # z in R^K -> u monotone via cumulative softplus increments
    def softplus(x):
        return math.log1p(math.exp(-abs(x))) + max(x, 0.0)
    inc = [softplus(zi) for zi in z]
    u = []
    s_accum = 0.0
    for i in range(K):
        s_accum += inc[i]
        u.append(s_accum)
    return u

# ---------------------------
# Black-box optimization (SPSA) of the schedule u[0:K]
# ---------------------------
def spsa_optimize(K=8, iters=200, seed=0):
    random.seed(seed)
    # Initialize a conservative schedule (small linear backoff)
    u = [0.05 + 0.1*k for k in range(K)]
    alpha = 0.2      # learning rate
    c0 = 0.1         # perturbation scale
    for t in range(1, iters+1):
        c = c0 / (t ** 0.101)
        # Rademacher perturbation
        delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(K)]
        u_plus  = [clamp(u[i] + c * delta[i], 0.0, 3.0) for i in range(K)]
        u_minus = [clamp(u[i] - c * delta[i], 0.0, 3.0) for i in range(K)]

        Jp, _ = rollout(u_plus, seed=seed + 10*t + 1)
        Jm, _ = rollout(u_minus, seed=seed + 10*t + 2)

        # SPSA gradient estimate
        g = [(Jp - Jm) / (2.0 * c * delta[i]) for i in range(K)]
        # Update (project back to bounds)
        u = [clamp(u[i] - alpha * g[i], 0.0, 3.0) for i in range(K)]
    return u

# ---------- SPSA over 2 parameters (base, gamma) with CRN ----------
def spsa_optimize_exp(K=8, iters=200, seed=0, Nmc=16):
    random.seed(seed)
    # fixed seeds reused every iteration (CRN)
    seeds = [seed + 1000 + i for i in range(Nmc)]

    # init: small base, mild growth
    base, gamma = 0.05, 1.4
    alpha0, c0 = 0.15, 0.2  # learning rate and perturbation scales

    for t in range(1, iters + 1):
        a_t = alpha0 / (t ** 0.602)   # standard SPSA decay
        c_t = c0 / (t ** 0.101)

        # Rademacher perturbations for 2 params
        d_base = 1.0 if random.random() < 0.5 else -1.0
        d_gamma = 1.0 if random.random() < 0.5 else -1.0

        base_plus  = base  + c_t * d_base
        base_minus = base  - c_t * d_base
        gamma_plus  = gamma + c_t * d_gamma
        gamma_minus = gamma - c_t * d_gamma

        u_plus  = project_waits(schedule_exp(base_plus,  gamma_plus,  K))
        u_minus = project_waits(schedule_exp(base_minus, gamma_minus, K))

        Jp = eval_policy(u_plus, seeds, max_attempts=K)
        Jm = eval_policy(u_minus, seeds, max_attempts=K)

        # SPSA gradient estimate
        g_base  = (Jp - Jm) / (2.0 * c_t * d_base)
        g_gamma = (Jp - Jm) / (2.0 * c_t * d_gamma)

        # Update
        base  = max(0.0, base  - a_t * g_base)
        gamma = max(0.5, gamma - a_t * g_gamma)  # keep reasonable

    return base, gamma

K = 8
# Baseline linear schedule
u0 = [0.05 + 0.1*k for k in range(K)]
J0, s0 = rollout(u0, seed=42)

# Optimize per-step waits (K-dim SPSA)
u_opt = spsa_optimize(K=K, iters=200, seed=123)
J1, s1 = rollout(u_opt, seed=999)

# Optimize exponential schedule parameters (2-dim SPSA with CRN)
base_opt, gamma_opt = spsa_optimize_exp(K=K, iters=200, seed=321, Nmc=16)
u_exp = project_waits(schedule_exp(base_opt, gamma_opt, K))
J2, s2 = rollout(u_exp, seed=777)

print("Initial schedule:", [round(x,3) for x in u0], "  Cost ≈", round(J0,3))
print("Optimized (per-step SPSA):", [round(x,3) for x in u_opt], "  Cost ≈", round(J1,3))
print("Optimized (exp base, gamma): base=", round(base_opt,3), " gamma=", round(gamma_opt,3),
      "  schedule=", [round(x,3) for x in u_exp], "  Cost ≈", round(J2,3))
print("Attempts (init → per-step → exp):", s0.k, "→", s1.k, "→", s2.k,
      "  Success codes:", s0.code, s1.code, s2.code)

strategies = {
    "Baseline": u0,
    "Per-step SPSA": u_opt,
    "Exp SPSA": u_exp,
}
costs = {"Baseline": J0, "Per-step SPSA": J1, "Exp SPSA": J2}

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

k = range(1, K + 1)
for name, waits in strategies.items():
    axes[0].step(k, waits, where="mid", label=name)
axes[0].set_ylabel("Wait time (s)")
axes[0].set_title("Backoff Schedules")
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].bar(list(costs.keys()), [costs[key] for key in costs], color=["#4a90e2", "#f5a623", "#7ed321"])
axes[1].set_ylabel("Mean rollout cost")
axes[1].set_title("Objective Values (lower is better)")
axes[1].grid(axis="y", alpha=0.3)
axes[1].set_xlabel("Strategy")

fig.tight_layout()
```



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

<!-- ### On the Choice of Optimizer

Although the code example uses SLSQP, many alternatives exist. `scipy.optimize.minimize` provides a menu of options, and each has implications for speed, robustness, and scalability:

* **Derivative-free methods** such as Nelder–Mead avoid gradients altogether. They are attractive when gradients are unavailable or noisy, but they scale poorly with dimension.
* **Quasi-Newton methods** like BFGS approximate gradients by finite differences. They work well for moderate-scale problems and often outperform derivative-free schemes when the objective is smooth.
* **Gradient-based constrained solvers** such as interior-point or SQP methods exploit derivatives (exact or automatic) and are typically the most efficient for large structured problems like trajectory optimization.

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

```{code-cell} python
:tags: [remove-input, remove-output]

import numpy as np
import json
import matplotlib.pyplot as plt

# Apply book style
try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except (ImportError, OSError):
    pass  # Use matplotlib defaults
from scipy.optimize import minimize, Bounds

def solve_eco_cruise(beta=1.0, gamma=0.05, T=60, v_max=20.0, a_max=3.0, distance=1000.0):
    """Solve the eco-cruise optimization problem."""
    
    n_state, n_control = T + 1, T
    
    def unpack(z):
        s, v, u = z[:n_state], z[n_state:2*n_state], z[2*n_state:]
        return s, v, u

    def objective(z):
        _, v, u = unpack(z)
        return 0.5 * beta * np.sum(u**2) + 0.5 * gamma * np.sum(v[:-1]**2)

    def dynamics(z):
        s, v, u = unpack(z)
        ceq = np.empty(2*T)
        ceq[0::2] = s[1:] - s[:-1] - v[:-1]  # position dynamics
        ceq[1::2] = v[1:] - v[:-1] - u        # velocity dynamics
        return ceq

    def boundary(z):
        s, v, _ = unpack(z)
        return np.array([s[0], v[0], s[-1]-distance, v[-1]])  # start/end conditions

    # Optimization setup
    cons = [{'type':'eq', 'fun': dynamics}, {'type':'eq', 'fun': boundary}]
    bounds = Bounds(
        lb=np.concatenate([np.full(n_state,-1e4), np.zeros(n_state), np.full(n_control,-a_max)]),
        ub=np.concatenate([np.full(n_state,1e4), v_max*np.ones(n_state), np.full(n_control,a_max)])
    )

    # Initial guess: triangular velocity profile
    accel_time = int(0.3 * T)
    decel_time = int(0.3 * T)
    cruise_time = T - accel_time - decel_time
    peak_v = min(1.2 * distance/T, 0.8 * v_max)
    
    v0 = np.zeros(n_state)
    v0[:accel_time+1] = np.linspace(0, peak_v, accel_time+1)
    v0[accel_time:accel_time+cruise_time+1] = peak_v
    v0[accel_time+cruise_time:] = np.linspace(peak_v, 0, decel_time+1)
    
    s0 = np.cumsum(np.concatenate([[0], v0[:-1]]))
    scale = distance / s0[-1]
    s0, v0 = s0 * scale, v0 * scale
    u0 = np.diff(v0)
    
    z0 = np.concatenate([s0, v0, u0])
    
    # Solve optimization
    print(f"Solving eco-cruise optimization (β={beta}, γ={gamma})...")
    res = minimize(objective, z0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-9})
    
    if not res.success:
        print(f"Optimization failed: {res.message}")
        return None
        
    s_opt, v_opt, u_opt = unpack(res.x)
    
    # Create trajectory data
    eco_trajectory = []
    cumulative_energy = 0
    
    for t in range(T + 1):
        if t < T:
            stage_cost = 0.5 * beta * u_opt[t]**2 + 0.5 * gamma * v_opt[t]**2
            cumulative_energy += stage_cost
        else:
            stage_cost = 0
            
        eco_trajectory.append({
            "time": float(t), "position": float(s_opt[t]), "velocity": float(v_opt[t]),
            "acceleration": float(u_opt[t]) if t < T else 0.0,
            "stageCost": float(stage_cost), "cumulativeEnergy": float(cumulative_energy)
        })
    
    return {
        "eco_trajectory": eco_trajectory,
        "total_energy": float(cumulative_energy),
        "optimization_success": True,
        "parameters": {"beta": beta, "gamma": gamma, "T": T, "v_max": v_max, "a_max": a_max, "distance": distance}
    }

def generate_naive_trajectory(T=60, distance=1000.0, gamma=0.05):
    """Generate naive constant-speed trajectory for comparison."""
    
    # Simple triangular profile: accelerate, cruise, decelerate
    accel_time = decel_time = 4
    cruise_time = T - accel_time - decel_time
    cruise_speed = distance / (0.5 * accel_time + cruise_time + 0.5 * decel_time)
    
    naive_trajectory = []
    cumulative_energy = 0
    
    for t in range(T + 1):
        if t <= accel_time:
            velocity = (cruise_speed / accel_time) * t
            acceleration = cruise_speed / accel_time
        elif t <= accel_time + cruise_time:
            velocity = cruise_speed
            acceleration = 0.0
        else:
            remaining_time = T - t
            velocity = (cruise_speed / decel_time) * remaining_time
            acceleration = -cruise_speed / decel_time
        
        # Calculate position by integration
        position = 0 if t == 0 else naive_trajectory[t-1]['position'] + naive_trajectory[t-1]['velocity']
        
        # Calculate costs
        if t < T:
            stage_cost = 0.5 * 1.0 * acceleration**2 + 0.5 * gamma * velocity**2
            cumulative_energy += stage_cost
        else:
            stage_cost = 0.0
            
        naive_trajectory.append({
            "time": float(t), "position": float(position), "velocity": float(velocity),
            "acceleration": float(acceleration), "stageCost": float(stage_cost),
            "cumulativeEnergy": float(cumulative_energy)
        })
    
    return {"naive_trajectory": naive_trajectory, "total_energy": float(cumulative_energy)}

def plot_comparison(eco_data, naive_data=None, save_plot=True):
    """Create visualization plots comparing eco-cruise and naive trajectories."""
    
    eco_traj = eco_data['eco_trajectory']
    times = [p['time'] for p in eco_traj]
    positions = [p['position'] for p in eco_traj]
    velocities = [p['velocity'] for p in eco_traj]
    accelerations = [p['acceleration'] for p in eco_traj]
    energy_costs = [p['stageCost'] for p in eco_traj]
    cumulative_energy = [p['cumulativeEnergy'] for p in eco_traj]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Eco-Cruise vs Naive Trajectory Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Position vs Time
    axes[0, 0].plot(times, positions, 'b-', linewidth=2, label='Eco-Cruise')
    if naive_data:
        naive_traj = naive_data['naive_trajectory']
        naive_times = [p['time'] for p in naive_traj]
        naive_positions = [p['position'] for p in naive_traj]
        axes[0, 0].plot(naive_times, naive_positions, 'r--', linewidth=2, label='Naive')
    axes[0, 0].set_xlabel('Time (s)'); axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].set_title('Position vs Time'); axes[0, 0].grid(True, alpha=0.3); axes[0, 0].legend()
    
    # Plot 2: Velocity vs Time
    axes[0, 1].plot(times, velocities, 'b-', linewidth=2, label='Eco-Cruise')
    if naive_data:
        naive_velocities = [p['velocity'] for p in naive_traj]
        axes[0, 1].plot(naive_times, naive_velocities, 'r--', linewidth=2, label='Naive')
    axes[0, 1].set_xlabel('Time (s)'); axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Velocity vs Time'); axes[0, 1].grid(True, alpha=0.3); axes[0, 1].legend()
    
    # Plot 3: Acceleration vs Time
    axes[0, 2].plot(times[:-1], accelerations[:-1], 'b-', linewidth=2, label='Eco-Cruise')
    axes[0, 2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 2].set_xlabel('Time (s)'); axes[0, 2].set_ylabel('Acceleration (m/s²)')
    axes[0, 2].set_title('Acceleration vs Time'); axes[0, 2].grid(True, alpha=0.3); axes[0, 2].legend()
    
    # Plot 4: Stage Cost vs Time
    axes[1, 0].plot(times[:-1], energy_costs[:-1], 'b-', linewidth=2, label='Eco-Cruise')
    if naive_data:
        naive_costs = [p['stageCost'] for p in naive_traj[:-1]]
        axes[1, 0].plot(naive_times[:-1], naive_costs, 'r--', linewidth=2, label='Naive')
    axes[1, 0].set_xlabel('Time (s)'); axes[1, 0].set_ylabel('Stage Cost')
    axes[1, 0].set_title('Stage Cost vs Time'); axes[1, 0].grid(True, alpha=0.3); axes[1, 0].legend()
    
    # Plot 5: Cumulative Energy vs Time
    axes[1, 1].plot(times, cumulative_energy, 'b-', linewidth=2, label='Eco-Cruise')
    if naive_data:
        naive_cumulative = [p['cumulativeEnergy'] for p in naive_traj]
        axes[1, 1].plot(naive_times, naive_cumulative, 'r--', linewidth=2, label='Naive')
    axes[1, 1].set_xlabel('Time (s)'); axes[1, 1].set_ylabel('Cumulative Energy')
    axes[1, 1].set_title('Cumulative Energy vs Time'); axes[1, 1].grid(True, alpha=0.3); axes[1, 1].legend()
    
    # Plot 6: Phase Space (Velocity vs Position)
    axes[1, 2].plot(positions, velocities, 'b-', linewidth=2, label='Eco-Cruise')
    if naive_data:
        naive_positions = [p['position'] for p in naive_traj]
        axes[1, 2].plot(naive_positions, naive_velocities, 'r--', linewidth=2, label='Naive')
    axes[1, 2].set_xlabel('Position (m)'); axes[1, 2].set_ylabel('Velocity (m/s)')
    axes[1, 2].set_title('Phase Space: Velocity vs Position'); axes[1, 2].grid(True, alpha=0.3); axes[1, 2].legend()
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('_static/eco_cruise_visualization.png', dpi=300, bbox_inches='tight')
        print("Plot saved to _static/eco_cruise_visualization.png")
    
    plt.show()
    return fig

def demo():
    """Run complete eco-cruise demonstration with visualization."""
    
    # Solve optimization
    eco_data = solve_eco_cruise(beta=1.0, gamma=0.05, T=60, distance=1000.0)
    if eco_data is None:
        # Use Jupyter Book's gluing feature for error message
        try:
            from myst_nb import glue
            glue("eco_cruise_output", "❌ Optimization failed!", display=False)
        except ImportError:
            print("Optimization failed!")
        return None
    
    # Generate naive trajectory
    naive_data = generate_naive_trajectory(T=60, distance=1000.0, gamma=0.05)
    
    # Create visualization
    fig = plot_comparison(eco_data, naive_data, save_plot=True)
    
    # Use Jupyter Book's gluing feature to display the figure
    try:
        from myst_nb import glue
        glue("eco_cruise_figure", fig, display=False)
    except ImportError:
        # Fallback for when not running in Jupyter Book context
        pass
    
    return eco_data, naive_data, fig

if __name__ == "__main__":
    demo()
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

Once these components are in place, the call to `minimize` does the rest. Internally, SLSQP linearizes the constraints, builds a quadratic subproblem, and iterates until both the Karush–Kuhn–Tucker conditions and the stopping tolerances are met. From the user's perspective, the heavy lifting reduces to providing functions that compute costs and residuals—everything else is handled by the solver. -->



## Sequential Methods

The previous section showed how a discrete-time optimal control problem can be solved by treating all states and controls as decision variables and enforcing the dynamics as equality constraints. This produces a nonlinear program that can be passed to solvers such as `scipy.optimize.minimize` with the SLSQP method. For short horizons, this approach is straightforward and works well; the code stays close to the mathematical formulation.

It also has a real advantage: by keeping the states explicit and imposing the dynamics through constraints, we anchor the trajectory at multiple points. This extra structure helps stabilize the optimization, especially for long horizons where small deviations in early steps can otherwise propagate and cause the optimizer to drift or diverge. In that sense, this formulation is better conditioned and more robust than approaches that treat the dynamics implicitly.

The drawback is scale. As the horizon grows, the number of variables and constraints grows with it, and all are coupled by the dynamics. Each iteration of a sequential quadratic programming (SQP) or interior-point method requires building and factorizing large Jacobians and Hessians. These methods have been embedded in reinforcement learning and differentiable programming pipelines, through implicit layers or differentiable convex solvers, but the cost is significant. They remain serial, rely on repeated linear algebra factorizations, and are difficult to parallelize efficiently. When thousands of such problems must be solved inside a learning loop, the overhead becomes prohibitive.

This motivates an alternative that aligns better with the computational model of machine learning. If the dynamics are deterministic and state constraints are absent (or reducible to simple bounds on controls), we can eliminate the equality constraints altogether by making the states implicit. Instead of solving for both states and controls, we fix the initial state and roll the system forward under a candidate control sequence. This is the essence of **single shooting**.

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

In JAX or PyTorch, this loop can be JIT-compiled and differentiated automatically. Any gradient-based optimizer (L-BFGS, Adam, even SGD) can be applied, making the pipeline look very much like training a neural network. In effect, we are "backpropagating through the world model" when computing $\nabla J(\mathbf{u})$.

Single shooting is attractive for its simplicity and compatibility with differentiable programming, but it has limitations. The absence of intermediate constraints makes it sensitive to initialization and prone to numerical instability over long horizons. When state constraints or robustness matter, formulations that keep states explicit, such as multiple shooting or collocation, become preferable. These trade-offs are the focus of the next section.

```{code-cell} python
:tags: [hide-input]

#  label: fig-ocp-single-shooting
#  caption: Single-shooting EV example: the plot shows optimal state trajectories (battery charge and speed) plus the control sequence, while the console reports the optimized control inputs.

%config InlineBackend.figure_format = 'retina'
import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt

# Apply book style
try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except (ImportError, OSError):
    pass  # Use matplotlib defaults

def single_shooting_ev_optimization(T=20, num_iterations=1000, step_size=0.01):
    """
    Implements the single shooting method for the electric vehicle energy optimization problem.
    
    Args:
    T: time horizon
    num_iterations: number of optimization iterations
    step_size: step size for the optimizer
    
    Returns:
    optimal_u: optimal control sequence
    """
    
    def f(x, u, t):
        return jnp.array([
            x[0] + 0.1 * x[1] + 0.05 * u,
            x[1] + 0.1 * u
        ])
    
    def c(x, u, t):
        if t == T:
            return x[0]**2 + x[1]**2
        else:
            return 0.1 * (x[0]**2 + x[1]**2 + u**2)
    
    def compute_trajectory_and_cost(u, x1):
        x = x1
        total_cost = 0
        for t in range(1, T):
            total_cost += c(x, u[t-1], t)
            x = f(x, u[t-1], t)
        total_cost += c(x, 0.0, T)  # No control at final step
        return total_cost
    
    def objective(u):
        return compute_trajectory_and_cost(u, x1)
    
    def clip_controls(u):
        return jnp.clip(u, -1.0, 1.0)
    
    x1 = jnp.array([1.0, 0.0])  # Initial state: full battery, zero speed
    
    # Initialize controls
    u_init = jnp.zeros(T-1)
    
    # Setup optimizer
    optimizer = optimizers.adam(step_size)
    opt_init, opt_update, get_params = optimizer
    opt_state = opt_init(u_init)
    
    @jit
    def step(i, opt_state):
        u = get_params(opt_state)
        value, grads = jax.value_and_grad(objective)(u)
        opt_state = opt_update(i, grads, opt_state)
        u = get_params(opt_state)
        u = clip_controls(u)
        opt_state = opt_init(u)
        return value, opt_state
    
    # Run optimization
    for i in range(num_iterations):
        value, opt_state = step(i, opt_state)
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {value}")
    
    optimal_u = get_params(opt_state)
    return optimal_u

def plot_results(optimal_u, T):
    # Compute state trajectory
    x1 = jnp.array([1.0, 0.0])
    x_trajectory = [x1]
    for t in range(T-1):
        x_next = jnp.array([
            x_trajectory[-1][0] + 0.1 * x_trajectory[-1][1] + 0.05 * optimal_u[t],
            x_trajectory[-1][1] + 0.1 * optimal_u[t]
        ])
        x_trajectory.append(x_next)
    x_trajectory = jnp.array(x_trajectory)

    time = jnp.arange(T)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time, x_trajectory[:, 0], label='Battery State of Charge')
    plt.plot(time, x_trajectory[:, 1], label='Vehicle Speed')
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.title('Optimal State Trajectories')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time[:-1], optimal_u, label='Motor Power Input')
    plt.xlabel('Time Step')
    plt.ylabel('Control Input')
    plt.title('Optimal Control Inputs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()

# Run the optimization
optimal_u = single_shooting_ev_optimization()
print("Optimal control sequence:", optimal_u)

# Plot the results
plot_results(optimal_u, T=20)
```


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

## The adjoint equation as reverse accumulation

Optimization needs sensitivities. In trajectory problems we adjust decisions (controls or parameters) to reduce an objective while respecting dynamics and constraints. First‑order methods in the unconstrained case (e.g., gradient descent, L‑BFGS, Adam) require the gradient of the objective with respect to all controls, and constrained methods (SQP, interior‑point) require gradients of the Lagrangian, i.e., of costs and constraints. The discrete‑time adjoint equations provide these derivatives in a way that scales to long horizons and many decision variables.

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

It is instructive to contrast this with alternatives. Black-box finite differences perturb one decision at a time and re-roll the system, requiring on the order of $p$ rollouts for $p$ decision variables and suffering from step-size and noise issues. This becomes prohibitive when $p=(T-1)m$ for an $m$-dimensional control over $T$ steps. Forward‑mode (tangent) sensitivities propagate Jacobian–vector products for each parameter direction; their work also scales with $p$. Reverse‑mode (the adjoint) instead propagates a single vector $\boldsymbol{\lambda}_t$ backward and then reads off all partial derivatives $\nabla_{\mathbf{u}_t} J$ at once. For a scalar objective, its cost is effectively independent of $p$, at the price of storing (or checkpointing) the forward trajectory. This scalability is why the adjoint is the method of choice for gradient‑based trajectory optimization and for constrained transcriptions via the Hamiltonian. 