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
# Finite-Horizon Dynamic Programming

MPC creates feedback by solving a new trajectory problem from each measured
state. Can the decisions for an entire family of states be computed in advance
instead? Dynamic programming decomposes a finite-horizon control problem into
cost-to-go functions indexed by time and state.

The deterministic discrete-time optimal control problem provides the starting
point. Stochastic successors will enter in the next chapter through conditional
expectations.

Consider a typical DOCP of Bolza type:

$$
\begin{align*}
\text{minimize} \quad & J \triangleq c_\mathrm{T}(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t, \mathbf{u}_t) \\
\text{subject to} \quad 
& \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t), \quad t = 1, \ldots, T-1, \\
& \mathbf{u}_{lb} \leq \mathbf{u}_t \leq \mathbf{u}_{ub}, \quad t = 1, \ldots, T, \\
& \mathbf{x}_{lb} \leq \mathbf{x}_t \leq \mathbf{x}_{ub}, \quad t = 1, \ldots, T, \\
\text{given} \quad & \mathbf{x}_1
\end{align*}
$$

Rather than considering only the total cost from the initial time to the final time, dynamic programming introduces the concept of cost from an arbitrary point in time to the end. This leads to the definition of the "cost-to-go" or "value function" $J_k(\mathbf{x}_k)$:

$$
J_k(\mathbf{x}_k) \triangleq c_\mathrm{T}(\mathbf{x}_T) + \sum_{t=k}^{T-1} c_t(\mathbf{x}_t, \mathbf{u}_t)
$$

This function represents the total cost incurred from stage $k$ onwards to the end of the time horizon, given that the system is initialized in state $\mathbf{x}_k$ at stage $k$. Suppose the problem has been solved from stage $k+1$ to the end, yielding the optimal cost-to-go $J_{k+1}^\star(\mathbf{x}_{k+1})$ for any state $\mathbf{x}_{k+1}$ at stage $k+1$. The question then becomes: how does this information inform the decision at stage $k$?

Given knowledge of the optimal behavior from $k+1$ onwards, the task reduces to determining the optimal action $\mathbf{u}_k$ at stage $k$. This control should minimize the sum of the immediate cost $c_k(\mathbf{x}_k, \mathbf{u}_k)$ and the optimal future cost $J_{k+1}^\star(\mathbf{x}_{k+1})$, where $\mathbf{x}_{k+1}$ is the resulting state after applying action $\mathbf{u}_k$. Mathematically, this is expressed as:

$$
J_k^\star(\mathbf{x}_k) = \min_{\mathbf{u}_k} \left[ c_k(\mathbf{x}_k, \mathbf{u}_k) + J_{k+1}^\star(\mathbf{f}_k(\mathbf{x}_k, \mathbf{u}_k)) \right]
$$

This equation is known as Bellman's equation, named after Richard Bellman, who formulated the principle of optimality:

> An optimal policy has the property that whatever the previous state and decision, the remaining decisions must constitute an optimal policy with regard to the state resulting from the previous decision.

In other words, any sub-path of an optimal path, from any intermediate point to the end, must itself be optimal. This principle is the basis for the backward induction procedure which computes the optimal value function and provides closed-loop control capabilities without having to use an explicit NLP solver. 

Dynamic programming can handle nonlinear systems and non-quadratic cost functions naturally. It provides a global optimal solution, when one exists, and can incorporate state and control constraints with relative ease. These advantages come from a change in what is being computed. [Trajectory optimization](numerical-trajectory-optimization.md) takes the initial state $\mathbf{x}_1$ as given and returns one sequence of states and controls, typically a local optimum. With state dimension $n$ and control dimension $m$, the resulting nonlinear program has on the order of $T(n+m)$ decision variables, so its size grows only linearly with the horizon and the dimension. [Parametric optimization](parametric-controllers.md) widened the question to a family of problems indexed by the initial state, but still answered it locally: sensitivity updates carry the solution of one instance to a nearby one, and learned controllers are fitted only on states sampled from the operating domain. Backward recursion asks for the optimal cost-to-go and the optimal decision from every state at every stage. The answer is a function on the whole state space rather than a vector of decision variables.

Representing such a function is what becomes expensive as the dimension of the state space increases, a difficulty Bellman termed the "curse of dimensionality." If each of the $n$ state coordinates is discretized into $N$ levels, the table storing $J_t^\star$ has $N^n$ entries, and each entry requires its own minimization over the controls. Computation and memory therefore grow exponentially with the state dimension, rendering direct application of dynamic programming intractable for high-dimensional problems. The curse is the price of asking for a solution everywhere. Trajectory optimization gives up the global, closed-loop answer to keep its cost polynomial in the dimension; dynamic programming obtains that answer and pays for it with exponential growth.

Fortunately, learning-based methods offer efficient tools to combat the curse of dimensionality on two fronts: by using function approximation (e.g., neural networks) to avoid explicit discretization, and by leveraging randomization through Monte Carlo methods inherent in the learning paradigm. Most of this course is dedicated to those ideas.

## Backward Recursion

How does the terminal cost propagate backward into an optimal action and value
for every earlier state?

The principle of optimality provides a methodology for solving optimal control problems. Beginning at the final time horizon and working backwards, at each stage the local optimization problem given by Bellman's equation is solved. This process, termed backward recursion or backward induction, constructs the optimal value function stage by stage.

````{prf:algorithm} Backward Recursion for Dynamic Programming
:label: backward-recursion-deterministic

**Input:** Terminal cost function $c_\mathrm{T}(\cdot)$, stage cost functions $c_t(\cdot, \cdot)$, system dynamics $f_t(\cdot, \cdot)$, time horizon $\mathrm{T}$

**Output:** Optimal value functions $J_t^\star(\cdot)$ and optimal control policies $\pi_t^\star(\cdot)$ for $t = 1, \ldots, T$

1. Initialize $J_T^\star(\mathbf{x}) = c_\mathrm{T}(\mathbf{x})$ for all $\mathbf{x}$ in the state space
2. For $t = T-1, T-2, \ldots, 1$:
   1. For each state $\mathbf{x}$ in the state space:
      1. Compute $J_t^\star(\mathbf{x}) = \min_{\mathbf{u}} \left[ c_t(\mathbf{x}, \mathbf{u}) + J_{t+1}^\star(f_t(\mathbf{x}, \mathbf{u})) \right]$
      2. Compute $\pi_t^\star(\mathbf{x}) = \arg\min_{\mathbf{u}} \left[ c_t(\mathbf{x}, \mathbf{u}) + J_{t+1}^\star(f_t(\mathbf{x}, \mathbf{u})) \right]$
   2. End For
3. End For
4. Return $J_t^\star(\cdot)$, $\pi_t^\star(\cdot)$ for $t = 1, \ldots, T$
````


Upon completion of this backward pass, we now have access to the optimal control to take at any stage and in any state. Furthermore, we can simulate optimal trajectories from any initial state and applying the optimal policy at each stage to generate the optimal trajectory.



```{prf:theorem} Backward induction solves deterministic Bolza DOCP
:label: thm-bolza-backward

**Setting.** Let $\mathbf{x}_{t+1}=\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)$ for $t=1,\dots,T-1$, with admissible action sets $\mathcal{U}_t(\mathbf{x})\neq\varnothing$. Let stage costs $c_t(\mathbf{x},\mathbf{u})$ and terminal cost $c_\mathrm{T}(\mathbf{x})$ be real-valued and bounded below. Assume for every $(t,\mathbf{x})$ the one-step problem

$$
\min_{\mathbf{u}\in\mathcal{U}_t(\mathbf{x})}\big\{c_t(\mathbf{x},\mathbf{u})+J_{t+1}^\star(\mathbf{f}_t(\mathbf{x},\mathbf{u}))\big\}
$$
admits a minimizer (e.g., compact $\mathcal{U}_t(\mathbf{x})$ and continuity suffice).

Define $J_T^\star(\mathbf{x}) \equiv c_\mathrm{T}(\mathbf{x})$ and for $t=T-1,\dots,1$

$$
J_t^\star(\mathbf{x}) \;\triangleq\; \min_{\mathbf{u}\in\mathcal{U}_t(\mathbf{x})}
\Big[c_t(\mathbf{x},\mathbf{u})+J_{t+1}^\star\big(\mathbf{f}_t(\mathbf{x},\mathbf{u})\big)\Big],
$$
and let $\boldsymbol{\pi}_t^\star(\mathbf{x})$ be any minimizer. The function $\boldsymbol{\pi}_t^\star$, which returns one minimizing control for each state, is the **decision rule** at stage $t$.

**Claim.** For every initial state $\mathbf{x}_1$, the control sequence
$\boldsymbol{\pi}_1^\star(\mathbf{x}_1),\dots,\boldsymbol{\pi}_{T-1}^\star(\mathbf{x}_{T-1})$
generated by these decision rules is optimal for the Bolza problem, and
$J_1^\star(\mathbf{x}_1)$ equals the optimal cost. Moreover, $J_t^\star(\cdot)$ is the optimal cost-to-go from stage $t$ for every state, i.e., backward induction recovers the entire value function.
```

The proof uses a "cut-and-paste" argument. Any control sequence can be cut after
its first step and its tail replaced by an optimal continuation without
increasing its cost. Comparing first steps, each followed by its optimal
continuation, then gives a sequence that no other sequence can beat.
{numref}`fig-cut-and-paste` follows these two moves with concrete costs.

:::{figure} _static/cut-and-paste.svg
:label: fig-cut-and-paste
:alt: Two panels over stages t to T. On the left, a sequence from x takes a first step of cost 2 to x_{t+1}, then a dashed tail of cost 9; an optimal tail from the same x_{t+1} costs 7. On the right, three first steps from x cost 2, 3 and 1 and lead to states whose optimal tails cost 7, 4 and 8; the middle step, followed by its tail, costs 7.

One inductive step of the proof of {prf:ref}`thm-bolza-backward` with concrete
costs. **Step 1 (cut):** an arbitrary sequence from $\mathbf{x}$ costs
$2 + 9 = 11$. Keeping its first step and replacing its tail by the optimal tail
from $\mathbf{x}_{t+1}$ gives $2 + 7 = 9$, so the sequence costs at least
$c_t + J^\star_{t+1}(\mathbf{x}_{t+1})$. **Step 2 (paste):** the top candidate is
the first step of Step 1. Among the three first steps, each followed by its
optimal tail, the middle one is cheapest, and its total $3 + 4 = 7$ is
$J^\star_t(\mathbf{x})$. The sequence of Step 1 costs at least this much
($11 \geq 9 \geq 7$), and pasting the optimal tail to
$\boldsymbol{\pi}^\star_t(\mathbf{x})$ attains it.
:::

```{prf:proof}
The proof is by backward induction on $t$. The statement carried from one stage
to the previous one is:

> **($H_t$)** For every state $\mathbf{y}$, the lowest cost achievable from
> stage $t$ starting at $\mathbf{y}$ equals $J_t^\star(\mathbf{y})$, and the
> decision rules $\boldsymbol{\pi}_t^\star,\dots,\boldsymbol{\pi}_{T-1}^\star$
> generate a sequence that achieves it.

Here $J_t^\star$ is the function produced by the recursion. That it equals the
optimal cost-to-go is what must be proved.

**Base case ($t=T$).** No control remains to be chosen, so the only cost from
$\mathbf{y}$ is $c_\mathrm{T}(\mathbf{y})$, which is $J_T^\star(\mathbf{y})$ by
definition.

**Inductive step.** Assume $H_{t+1}$ holds and fix a state $\mathbf{x}$ at
stage $t$.

*Step 1 (cut: a lower bound for every sequence).* Take any admissible sequence
$\mathbf{u}_t,\dots,\mathbf{u}_{T-1}$ from $\mathbf{x}_t=\mathbf{x}$, with
induced states $\mathbf{x}_{k+1}=\mathbf{f}_k(\mathbf{x}_k,\mathbf{u}_k)$. Cut
it after the first step. Its cost splits into the cost of that step and the
cost of the remaining controls $\mathbf{u}_{t+1},\dots,\mathbf{u}_{T-1}$, which
form a sequence that starts from $\mathbf{x}_{t+1}$ at stage $t+1$:

$$
\underbrace{c_t(\mathbf{x},\mathbf{u}_t)}_{\text{first step}}
\;+\;
\underbrace{\sum_{k=t+1}^{T-1}c_k(\mathbf{x}_k,\mathbf{u}_k)+c_\mathrm{T}(\mathbf{x}_T)}_{\text{tail from }\mathbf{x}_{t+1}}.
$$

By $H_{t+1}$, no sequence from $\mathbf{x}_{t+1}$ at stage $t+1$ costs less than
$J_{t+1}^\star(\mathbf{x}_{t+1})$, and this holds in particular for the tail.
Replacing the tail cost by $J_{t+1}^\star(\mathbf{x}_{t+1})$ therefore cannot
increase the total. The first step $\mathbf{u}_t$ is one of the candidates in
the minimum that defines $J_t^\star(\mathbf{x})$, so

$$
\begin{aligned}
\text{cost of the sequence}
&\;\ge\; c_t(\mathbf{x},\mathbf{u}_t)+J_{t+1}^\star\big(\mathbf{f}_t(\mathbf{x},\mathbf{u}_t)\big) \\
&\;\ge\; \min_{\mathbf{u}\in\mathcal{U}_t(\mathbf{x})}\Big[c_t(\mathbf{x},\mathbf{u})+J_{t+1}^\star\big(\mathbf{f}_t(\mathbf{x},\mathbf{u})\big)\Big]
\;=\; J_t^\star(\mathbf{x}).
\end{aligned}
$$ (eq:cut-lower-bound)

The sequence was arbitrary, so no admissible sequence from $(t,\mathbf{x})$
costs less than $J_t^\star(\mathbf{x})$. In {numref}`fig-cut-and-paste`, the
two inequalities are $11 \ge 9$ and $9 \ge 7$.

*Step 2 (paste: a sequence that attains the bound).* Take the first step
$\boldsymbol{\pi}_t^\star(\mathbf{x})$, which attains the minimum in {eq}`eq:cut-lower-bound`.
It leads to $\mathbf{x}_{t+1}^\star=\mathbf{f}_t\big(\mathbf{x},\boldsymbol{\pi}_t^\star(\mathbf{x})\big)$,
and $H_{t+1}$ supplies a tail from $\mathbf{x}_{t+1}^\star$, generated by
$\boldsymbol{\pi}_{t+1}^\star,\dots,\boldsymbol{\pi}_{T-1}^\star$, whose cost is
$J_{t+1}^\star(\mathbf{x}_{t+1}^\star)$. Paste this tail after the first step.
The resulting sequence costs

$$
c_t\big(\mathbf{x},\boldsymbol{\pi}_t^\star(\mathbf{x})\big)
+J_{t+1}^\star(\mathbf{x}_{t+1}^\star)
=J_t^\star(\mathbf{x}).
$$

Combine this with Step 1. Every admissible sequence from $(t,\mathbf{x})$
costs at least $J_t^\star(\mathbf{x})$, and the pasted sequence is one of them
and costs exactly $J_t^\star(\mathbf{x})$. No sequence can therefore cost less
than the pasted one, so the pasted sequence is optimal and the lowest cost
achievable from $(t,\mathbf{x})$ is $J_t^\star(\mathbf{x})$. In
{numref}`fig-cut-and-paste`, every sequence costs at least 7 and the pasted
sequence costs 7. The pasted sequence is generated by the decision rules from
stage $t$ onward, so $H_t$ holds.

By induction, $H_t$ holds for every $t$, in particular for $t=1$ and any initial
state $\mathbf{x}_1$. The backward recursion therefore both **certifies**
optimality (Step 1) and **constructs** an optimal policy (Step 2), while
recovering the full family $\{J_t^\star\}_{t=1}^T$.
```

```{prf:remark} No "big NLP" required
The Bolza DOCP over the whole horizon couples all controls through the dynamics and is typically posed as a single large nonlinear program. The proof shows you can solve **$T-1$ sequences of one-step problems** instead: at each $(t,\mathbf{x})$ minimize

$$
\mathbf{u}\mapsto c_t(\mathbf{x},\mathbf{u}) + J_{t+1}^\star(\mathbf{f}_t(\mathbf{x},\mathbf{u})).
$$

In finite state–action spaces this becomes pure table lookup and argmin. In continuous spaces you still solve local one-step minimizations, but you avoid a monolithic horizon-coupled NLP.
```

```{prf:remark} Graph interpretation (optional intuition)
Unroll time to form a DAG whose nodes are $(t,\mathbf{x})$ and whose edges correspond to feasible controls with edge weight $c_t(\mathbf{x},\mathbf{u})$. The terminal node cost is $c_\mathrm{T}(\cdot)$. The Bolza problem is a shortest-path problem on this DAG. The equation

$$
J_t^\star(\mathbf{x})=\min_{\mathbf{u}}\{c_t(\mathbf{x},\mathbf{u})+J_{t+1}^\star(\mathbf{f}_t(\mathbf{x},\mathbf{u}))\}
$$

is exactly the dynamic programming recursion for shortest paths on acyclic graphs, hence backward induction is optimal.
```

<!-- ```{prf:remark} If minimizers may not exist
Replace each "min" by "inf" in the definitions and state that for every $\varepsilon>0$ there exist $\varepsilon$-optimal decision rules $\boldsymbol{\pi}_t^\varepsilon(\cdot)$ achieving cost within $\varepsilon$ of $J_t^\star(\cdot)$. The same cut-and-paste and induction go through.
``` -->



### Example: Seasonal Operation of a Hydroelectric Reservoir

On a northern river, most of the water arrives in a few weeks of spring, when
the snowpack melts. This is the *freshet*. Electricity is worth the most in
winter, when heating demand peaks. A reservoir behind a dam moves water from
one season to the other. Each week the operator decides how much water to send
through the turbines. Water released now is sold at this week's price. Water
kept in the reservoir can be sold later, and it also raises the water level, so
every cubic metre released afterwards falls from a greater height and produces
more energy. Keeping water has a risk as well. When the freshet reaches a full
reservoir, the excess goes over the spillway and earns nothing.

The [hydro cascade](numerical-trajectory-optimization.md) in the trajectory
optimization chapter computed one plan from one initial state. An operator needs
more than that. Inflows never follow the forecast exactly, so the reservoir is
rarely at the level the plan assumed, and the operator needs a release for
whatever level it is at. Dynamic programming provides this: one backward pass
gives a release for every week and every storage level. Reservoir operators call
the resulting curves *rule curves*, and dynamic programming has been used to
compute them for decades {cite:p}`Yeh1985`.

**Model.** The stage is one week, with $T=52$ weeks starting in January. Volumes
are measured in cubic hectometres ($1\,\mathrm{hm}^3 = 10^6\,\mathrm{m}^3$).

| Symbol | Meaning | Value |
|---|---|---|
| $x_t$ | storage at the start of week $t$ (state) | $x_{\min} \le x_t \le \bar x$ |
| $u_t$ | water released through the turbines (control) | $u_{\min} \le u_t \le u_{\max}$ |
| $w_t$ | inflow forecast for week $t$ | 60 to 400 hm³/week |
| $p_t$ | electricity price in week $t$ | 30 to 90 \$/MWh |
| $x_{\min},\ \bar x$ | dead storage (below the intake) and capacity | 200 and 2000 hm³ |
| $u_{\min},\ u_{\max}$ | ecological minimum flow and turbine capacity | 30 and 250 hm³/week |
| $h(x)$ | head: height through which the water falls | 40 to 80 m |
| $\kappa$ | energy per hm³ per metre of head | 2.45 MWh |

The inflows and prices are stylized curves, not measurements: a winter base flow
of 100 m³/s, a freshet that peaks in mid-May at about four times the mean flow,
an autumn rise from rain, and a price that is highest in early February and
lowest in August. Water above capacity spills, so storage evolves as

$$
x_{t+1} = \min\{\,x_t + w_t - u_t,\ \bar x\,\},
\qquad \text{subject to } x_t + w_t - u_t \ge x_{\min}.
$$

Releasing $u$ with head $h(x)$ produces $\kappa\,h(x)\,u$ megawatt-hours, where
$\kappa = \rho g \eta = 1000 \cdot 9.81 \cdot 0.9$ joules per cubic metre per
metre, converted to MWh per hm³. The head grows with storage, which is what
makes keeping water valuable beyond its later sale price. Water left at the end
of the year is credited at $p_T = 50$ \$/MWh, so the terminal value is
$J_T^\star(x) = p_T\,\kappa\,h(x)\,(x - x_{\min})$. Since the objective is a
revenue to maximize rather than a cost, Bellman's equation takes a maximum:

$$
J_t^\star(x) = \max_{u_{\min} \le u \le u_{\max}}
\Big[\, p_t\,\kappa\,h(x)\,u \;+\; J_{t+1}^\star\big(\min\{x + w_t - u,\ \bar x\}\big) \Big],
\qquad x + w_t - u \ge x_{\min}.
$$

Call the expression inside the maximum

$$
Q_t(x, u) = p_t\,\kappa\,h(x)\,u \;+\; J_{t+1}^\star\big(\min\{x + w_t - u,\ \bar x\}\big),
$$

the total revenue from storage $x$ in week $t$ if we release $u$ now and act
optimally afterwards. It weighs the revenue earned by releasing $u$ this week
against the revenue still obtainable from the storage that remains, and
Bellman's equation reads $J_t^\star(x) = \max_u Q_t(x, u)$.

**Code.** The storage grid has a step of 10 hm³, and the inflows are rounded to
multiples of 10 hm³. Every successor $x + w_t - u$ is therefore a grid point,
and {prf:ref}`backward-recursion-deterministic` applies as stated: $J_{t+1}^\star$
is looked up, never interpolated. The next section removes this convenience.
The loop over states and actions of the algorithm is written with NumPy
broadcasting: `x` is a column of all storage levels and `u` a row of all
releases, so each line evaluates every (storage, release) pair at once.

```{code-cell} python
import numpy as np

# Calendar: 52 weekly decisions; week 0 is the first week of January.
T = 52
weeks = np.arange(T)

# Inflow forecast w_t (hm³ per week): winter base flow, spring freshet, autumn rain.
w = 60 + 340 * np.exp(-((weeks - 19) / 3.5) ** 2) + 60 * np.exp(-((weeks - 43) / 4) ** 2)
w = 10 * np.round(w / 10)            # multiples of the grid step

# Price p_t ($/MWh): winter peak, summer trough.
p = 60 + 30 * np.cos(2 * np.pi * (weeks - 5) / 52)

# Reservoir and plant.
x_min, x_max = 200, 2000             # dead storage and capacity (hm³)
u_min, u_max = 30, 250               # ecological minimum and turbine capacity (hm³/week)
kappa = 2.45                         # MWh per hm³ per metre of head
p_T = 50                             # $/MWh credited to water left at the end of the year

def head(x):                         # metres of head; grows with storage
    return 40 + 40 * np.sqrt(x / x_max)

def terminal_value(x):
    return p_T * kappa * head(x) * (x - x_min)

dx = 10
X = np.arange(x_min, x_max + dx, dx)   # storage grid: 181 levels
U = np.arange(u_min, u_max + dx, dx)   # release grid: 23 levels

def index(x):                          # position of an on-grid storage level
    return np.round((x - x_min) / dx).astype(int)

J = np.zeros((T + 1, len(X)))
policy = np.zeros((T, len(X)))
J[T] = terminal_value(X)

for t in reversed(range(T)):
    x, u = X[:, None], U[None, :]                  # all (storage, release) pairs
    x_next = np.minimum(x + w[t] - u, x_max)       # water above capacity spills
    feasible = x + w[t] - u >= x_min               # cannot draw below dead storage
    value = p[t] * kappa * head(x) * u + J[t + 1][index(np.maximum(x_next, x_min))]
    value = np.where(feasible, value, -np.inf)
    J[t] = value.max(axis=1)
    policy[t] = U[value.argmax(axis=1)]

def simulate(x0):
    storage = [x0]
    for t in range(T):
        u = policy[t, index(storage[-1])]
        storage.append(min(storage[-1] + w[t] - u, x_max))
    return np.array(storage)

print(f"optimal revenue from x0 = 1100 hm³: {J[0, index(1100)] / 1e6:.1f} M$")
```

The backward pass evaluates $52 \times 181 \times 23 \approx 2.2 \times 10^5$
(week, storage, release) triples and takes a fraction of a second. Its output is
not one trajectory but two tables, $J_t^\star(x)$ and $u_t^\star(x)$, with one
entry for every week and storage level.

```{code-cell} python
:tags: [remove-cell]

from pathlib import Path
import sys

code_directory = Path.cwd() / "code"
if str(code_directory) not in sys.path:
    sys.path.insert(0, str(code_directory))

from hydro_reservoir import (
    format_comparison,
    make_interpolation_figure,
    make_policy_figure,
    make_threshold_figure,
    simulate_rule,
)

water_value = np.gradient(J, dx, axis=1)       # dJ_t/dx on the grid, $ per hm³
rollouts = [simulate(x0) for x0 in (300, 700, 1100, 1500, 1900)]
reservoir = dict(w=w, p=p, head=head, terminal_value=terminal_value, x_min=x_min,
                 x_max=x_max, u_min=u_min, u_max=u_max, kappa=kappa)
```

```{code-cell} python
:label: fig-hydro-policy
:caption: Output of one backward pass. Left, the optimal release $u_t^\star(x)$ for every week and storage level, below the inflow forecast. Right, the water value $\partial J_t^\star/\partial x$, the revenue that one more hm³ in storage would add, below the price. White curves are the storage paths obtained by following the policy from five initial levels; after a few weeks they coincide.
:tags: [remove-input]

make_policy_figure(w, p, X, policy, water_value[:T], rollouts)
```

The left map is the policy. It has two regions: release at turbine capacity or
release the ecological minimum, with a thin boundary between them. The white
storage paths read the policy as an operator would. Whatever the level in
January, the reservoir is drawn down to dead storage by late March, refilled by
the freshet and then more slowly through the summer until it is full in
mid-September, held full until late November, and drawn down at turbine
capacity in December, when prices are high again. The initial storage affects
only the first few weeks, because every path reaches the same rule curve before
the freshet. A fuller reservoir in January simply means more water to sell
during the winter drawdown.

The right map is the derivative of the value function with respect to storage,
which hydro operators call the *water value*. It is lowest, close to zero, at
full storage during the freshet: the reservoir cannot hold more, so an extra
hm³ would spill and earn nothing. It is highest in winter at low storage, when
every hm³ can be sold at the winter price and none is at risk of spilling.

**Why the policy is all or nothing.** Fix the week $t$ and the storage $x$,
and look at $Q_t(x, u)$ as a function of the release $u$ alone. It has two
terms, and $u$ enters each one once.

The first term, $p_t\,\kappa\,h(x)\,u$, is linear in $u$. The head $h(x)$
depends on the storage at the start of the week, which is fixed, so each
extra hm³ released earns the same amount, $p_t\,\kappa\,h(x)$.

The second term depends on $u$ only through next week's storage
$x_{t+1} = x + w_t - u$. Releasing one more hm³ lowers $x_{t+1}$ by one hm³,
so by the chain rule this term changes at the rate
$-\partial J_{t+1}^\star/\partial x$ evaluated at $x_{t+1}$, minus the
water value. (If the reservoir overflows, $x_{t+1} = \bar x$ does not depend
on $u$, and this rate is zero: the extra hm³ would have spilled anyway.)

Adding the two rates gives the slope of $Q_t$ in $u$:

$$
\frac{\partial Q_t}{\partial u}(x, u)
\;=\; \underbrace{p_t\,\kappa\,h(x)}_{\text{value of releasing one hm³}}
\;-\; \underbrace{\frac{\partial J_{t+1}^\star}{\partial x}(x + w_t - u)}_{\text{value of keeping it}} .
$$

We maximize $Q_t$ over the interval $[u_{\min}, u_{\max}]$, and the sign of
this slope tells us where the maximum is:

- If the slope is positive for every $u$ in the interval, $Q_t$ increases
  from one end to the other, and the maximum is at $u = u_{\max}$.
- If it is negative for every $u$, $Q_t$ decreases, and the maximum is at
  $u = u_{\min}$.
- Only if the slope changes sign inside the interval can the maximum be
  strictly between the bounds, at a release where the two values are equal.

The first term of the slope does not depend on $u$ at all. The second
depends on $u$ only through the water value, and over the storage levels
reachable in one week the water value changes little compared with the gap
between it and the price. For most $(t, x)$ the slope therefore keeps one
sign over the whole interval, and the release is at a bound. This is the
two-region map of {numref}`fig-hydro-policy`. The interior case is the thin
boundary between the regions: there the policy releases exactly enough
water to bring $x_{t+1}$ to the level at which the water value equals the
price.

Two physical limits also produce releases between $u_{\min}$ and
$u_{\max}$, and both show up in the figures. At low storage the constraint
$x + w_t - u \ge x_{\min}$ caps the release below turbine capacity, so
"release as much as possible" means emptying the reservoir to dead storage.
At full storage the slope jumps at the release $u = x + w_t - \bar x$:
below it the extra water would spill, the second term is zero and the slope
is positive; above it the slope is the price minus the water value at
$\bar x$. When that is negative, the best release is the kink itself, which
keeps the reservoir exactly full and releases the inflow. This is what
happens from September to November in {numref}`fig-hydro-threshold`.

The decision rule is therefore a price comparison: **run the turbines when
the price exceeds the value of the water in the reservoir**, both expressed
per hm³. In hydro-dominated power systems such as Norway's, producers compute
water values with dynamic programming and use them to set the prices at which
they offer their energy.

The same structure appears in continuous time, where the slope is called a
*switching function* and a control that sits at its bounds is called
*bang-bang*. Whenever the reward is linear in a bounded control, the optimum
is at a bound except where the switching function is zero, the *singular*
case.

```{code-cell} python
:label: fig-hydro-threshold
:caption: The two terms of the derivative along the optimal path from $x_0 = 1100$ hm³. The turbines run at capacity (shaded weeks) when the revenue from releasing one hm³ now (orange) exceeds the value of keeping it (blue), and at the ecological minimum when it is lower. From September to November the reservoir is full and any water kept would spill, so the release equals the inflow even though keeping water appears more valuable.
:tags: [remove-input]

x0 = 1100
comparison = {
    "dynamic programming": simulate_rule(lambda t, x: policy[t, index(x)], x0, **reservoir),
    "run of river (release = inflow)": simulate_rule(lambda t, x: w[t], x0, **reservoir),
    "full release when price > mean": simulate_rule(
        lambda t, x: u_max if p[t] > p.mean() else u_min, x0, **reservoir),
}
print(format_comparison(comparison))

optimal = comparison["dynamic programming"]
marginal_release = p * kappa * head(optimal["storage"][:-1])
marginal_storage = np.array([np.interp(optimal["storage"][t + 1], X, water_value[t + 1])
                             for t in range(T)])
make_threshold_figure(p, optimal["storage"], optimal["release"], marginal_release,
                      marginal_storage, u_min, u_max)
```

The table above the figure compares the policy with two simpler operating rules
over the same year. Running the river as it comes, releasing each week's inflow,
earns about 10% less because it sells most of the water during the freshet, when
prices are low. Releasing fully whenever the price is above its annual mean
earns even less and spills 900 hm³, because the rule ignores the storage level.
It empties the reservoir by early February and has only the small winter inflow
to sell for the rest of the high-price season. In May, when the price falls
below its mean, it switches to the minimum release in the middle of the
freshet. The reservoir is full by late June, and from July to November every
hm³ above the minimum flow spills. Dynamic programming accounts for the price,
the head and the remaining capacity at the same time, through one function of
storage.

**Explore the solution.** The page below re-solves the same backward recursion
in your browser whenever a parameter changes. *Backward pass* animates the
recursion: the maps fill in from week 52 back to week 1, one column per Bellman
update, and the panel beside the map shows the value function of the column
being computed. *Play the year* then operates the reservoir forward from the
chosen initial storage. The page compares the two terms of the derivative per
MWh rather than per hm³: dividing both by $\kappa\,h(x_t)$ turns the first
into the price $p_t$ and the second into a water value in \$/MWh.

:::{iframe} ../interactive/hydro-reservoir.html
:label: fig-hydro-interactive
:width: 100%
:title: Seasonal hydro reservoir solved by backward recursion
:class: hydro-reservoir-replay

The reservoir problem re-solved in the browser. The map shows the release,
the water value or the value function over (week, storage); the side panel
shows $J_t^\star(x) - J_t^\star(x_{\min})$ for the current week, whose slope is
the water value. The strips compare the price with the water value along the
simulated year and show inflow, release and spill.
:::

> **Question.** Set the price of water left at the end of the year to zero.
> What does the policy do in November and December? Is this what an operator
> who will still run the dam next year should do?

:::{dropdown} Answer
It empties the reservoir to dead storage by the end of December, because the
model gives no value to water that remains after week 52. A real operator would
start January with an empty reservoir and no water to sell during the
high-price weeks. This end-of-horizon effect is an artifact of truncating the
problem at one year. The terminal value $J_T^\star$ stands in for the years that
follow; the infinite-horizon formulations of later chapters remove the need to
choose it.
:::

> **Question.** Starting from 1100 hm³, the policy keeps the minimum release
> through January, although the price is already above 80 \$/MWh, and releases
> at full capacity from mid-February to mid-March. Why does it wait?

:::{dropdown} Answer
The reservoir must be nearly empty before the freshet starts in April, so the
amount of water to sell before then is fixed. Two effects make it better to
sell it late rather than early. The price keeps rising until early February.
And water released later falls from a fuller reservoir: holding it keeps the
head high, so every hm³ produces more energy. The turbines release at most
250 hm³ per week, about 190 hm³ net of the winter inflow, so emptying the
reservoir takes about six weeks, and the policy starts the drawdown just early
enough to finish before the freshet.
:::

> **Question.** Suppose the river had three such reservoirs in series, each
> discretized with the same 181 storage levels and 23 releases. How many
> (storage, release) combinations would one Bellman update have to evaluate at
> each week?

:::{dropdown} Answer
The state is the vector of the three storages, with $181^3 \approx 5.9 \times
10^6$ grid points, and each has $23^3 = 12\,167$ joint releases, so about
$7 \times 10^{10}$ evaluations per week instead of $4\,163$. This is the curse of
dimensionality from the start of the chapter, and the reason the cascade was
solved by trajectory optimization from one initial state.
:::


## Handling Continuous Spaces with Interpolation

When a successor falls between the states stored in the table, which value
should the recursion use for it, and what problem does the recursion then
solve?

{prf:ref}`thm-bolza-backward` places no restriction on the state space. A
program, however, has to store $J_{t+1}^\star$ as finitely many numbers. The
reservoir code stores one value per 10 hm³ of storage, and it avoided the
question by construction: its inflows were rounded to multiples of the grid
step, so every successor $x + w_t - u$ landed on a grid point. Measured inflows
are not multiples of anything. A week with 63.7 hm³ of inflow and a 30 hm³
release sends a reservoir at 1000 hm³ to 1033.7 hm³, between two grid points,
and Bellman's equation then asks for $J_{t+1}^\star(1033.7)$, which the table
does not contain. Choosing a value for it is a representation problem that the
[collocation chapter](continuous-time-collocation.md) already treated for
functions of time.

### Nodal Representation of the Cost-to-Go

In collocation, a trajectory piece was stored by its values $y_j$ at support
nodes $\sigma_j$ and reconstructed between them by cardinal functions,
$p(\tau)=\sum_j y_j\ell_j(\tau)$ with $\ell_j(\sigma_i)=\delta_{ij}$. A
cost-to-go table is the same kind of object, with the nodes placed in the state
space instead of on the time axis. Choose grid states
$\mathbf{x}^{(1)},\dots,\mathbf{x}^{(N)}$ and let $v_{t,i}$ denote the value
stored for $\mathbf{x}^{(i)}$ at stage $t$. A family of cardinal functions on
the grid turns these numbers into a function of the state:

$$
\hat J_t(\mathbf{x})=\sum_{i=1}^{N}v_{t,i}\,\ell_i(\mathbf{x}),
\qquad
\ell_i\big(\mathbf{x}^{(j)}\big)=\delta_{ij}.
$$ (eq:nodal-cost-to-go)

The array `J` in the reservoir code holds the nodal coordinates $v_{t,i}$. An
interpolation rule is a choice of cardinal functions, which the array alone
does not specify.

When the collocation chapter
[chose how to represent a control](continuous-time-collocation.md#choosing-a-polynomial-for-the-control)
between its support values, it listed three options: a constant command on
each interval, a line between endpoint commands, or a higher-degree
polynomial. The same three options are available for the cost-to-go. On a
one-dimensional grid $x^{(1)}<\dots<x^{(N)}$, such as the storage levels of the
reservoir, they are:

1. **Rounding** (nearest-neighbor interpolation). The cardinal function
   $\ell_i$ equals one on the states closer to $x^{(i)}$ than to any other
   node and zero elsewhere, so $\hat J_t(x)$ is the value stored at the
   nearest node. It is constant on each cell, like a zero-order-hold control,
   except that the cell is centered on its node.
2. **Linear interpolation.** Each $\ell_i$ is a "hat" that rises from zero at
   $x^{(i-1)}$ to one at $x^{(i)}$ and returns to zero at $x^{(i+1)}$. Between
   two adjacent nodes only two hats are nonzero, and
   $\hat J_t(x)=(1-\lambda)\,v_{t,i}+\lambda\,v_{t,i+1}$ with
   $\lambda=(x-x^{(i)})/(x^{(i+1)}-x^{(i)})\in[0,1]$. These are the continuous
   piecewise-linear controls with shared endpoint values of the
   [trapezoidal transcription](continuous-time-collocation.md#endpoint-slope-values-trapezoidal-transcription).
3. **Cubic spline.** Cubic pieces are joined with continuous first and second
   derivatives. Each cardinal function is then nonzero over the whole grid,
   with lobes of alternating sign that decay away from its node.

In all three cases the weights $\ell_i(x)$ sum to one, because each rule
reproduces a constant function exactly. They differ in sign. Rounding and
linear interpolation have nonnegative weights, so $\hat J_t(x)$ is a convex
combination of stored values and stays within their range. The collocation
chapter used the same property to bound a piecewise-linear control by bounding
its support values. The spline has negative weights, like the quadratic
Lagrange basis of the
[classroom exercise](#ex-classroom-lagrange-quadratic) whose weights at
$\tau=1/4$ were $(3/8,3/4,-1/8)$. It can therefore leave the range of the
stored values, as the bottom row of {numref}`fig-dp-cardinal-functions` shows
on data with a jump. In exchange, it is far more accurate than linear
interpolation on smooth functions.

```{code-cell} python
:label: fig-dp-cardinal-functions
:caption: Three interpolation rules on ten grid states. Top row, the cardinal function of one node (blue) and of its two neighbors (gray); each equals one at its own node and zero at every other node. Bottom row, the same stored values reconstructed by each rule. The values jump from zero to one between two nodes, as an all-or-nothing release table does at its switching storage. The shaded band is the range of the stored values. Rounding and linear interpolation stay inside it because their weights are nonnegative; the negative lobes of the spline's cardinal functions make it overshoot on both sides of the jump.
:tags: [remove-input]

from finite_horizon_dp_figures import make_cardinal_functions_figure

make_cardinal_functions_figure()
```

Substituting {eq}`eq:nodal-cost-to-go` into Bellman's equation and evaluating
it only at the grid states gives a recursion on $N$ numbers per stage:

$$
v_{T,i}=c_\mathrm{T}\big(\mathbf{x}^{(i)}\big),
\qquad
v_{t,i}=\min_{\mathbf{u}\in\mathcal{U}_t(\mathbf{x}^{(i)})}
\Big[c_t\big(\mathbf{x}^{(i)},\mathbf{u}\big)
+\sum_{j=1}^{N}\ell_j\big(\mathbf{f}_t(\mathbf{x}^{(i)},\mathbf{u})\big)\,v_{t+1,j}\Big],
\qquad i=1,\dots,N.
$$ (eq:interpolated-bellman)

Compared with {prf:ref}`backward-recursion-deterministic`, the loop over states
becomes a loop over grid states, and the lookup of $J_{t+1}^\star$ at the
successor becomes an evaluation of $\hat J_{t+1}$. Each stage still consists of
independent one-step problems, one per grid state. For the reservoir, which
maximizes revenue, the minimum becomes a maximum. The numbers $v_{t,i}$ are
approximations of $J_t^\star(\mathbf{x}^{(i)})$, and the cardinal functions
determine how good they are.

````{prf:algorithm} Backward Recursion with Interpolation
:label: backward-recursion-interp

**Input:** Terminal cost $c_\mathrm{T}$, stage costs $c_t$, dynamics
$\mathbf{f}_t$, horizon $T$, grid states $\mathbf{x}^{(1)},\dots,\mathbf{x}^{(N)}$,
cardinal functions $\ell_1,\dots,\ell_N$

**Output:** Nodal values $v_{t,i}$ for $t=1,\ldots,T$ and nodal decisions
$\boldsymbol{\pi}_{t,i}$ for $t=1,\ldots,T-1$

1. Set $v_{T,i}=c_\mathrm{T}(\mathbf{x}^{(i)})$ for $i=1,\ldots,N$
2. For $t=T-1,T-2,\ldots,1$:
   1. For each grid state $\mathbf{x}^{(i)}$:
      1. For each candidate $\mathbf{u}$, compute the successor
         $\mathbf{y}=\mathbf{f}_t(\mathbf{x}^{(i)},\mathbf{u})$ and its
         interpolated value $\hat J_{t+1}(\mathbf{y})=\sum_j \ell_j(\mathbf{y})\,v_{t+1,j}$
      2. Set $v_{t,i}=\min_{\mathbf{u}}\big[c_t(\mathbf{x}^{(i)},\mathbf{u})+\hat J_{t+1}(\mathbf{f}_t(\mathbf{x}^{(i)},\mathbf{u}))\big]$
         and let $\boldsymbol{\pi}_{t,i}$ be a minimizer
   2. End For
3. End For
4. Return $\{v_{t,i}\}$ and $\{\boldsymbol{\pi}_{t,i}\}$
````

### Grid Models Behind the Interpolated Recursion

The weights $\ell_j(\mathbf{f}_t(\mathbf{x}^{(i)},\mathbf{u}))$ in
{eq}`eq:interpolated-bellman` depend only on where the successor lands relative
to the grid. Read as a description of where the system goes next, they show
which problem the interpolated recursion solves.

With rounding, exactly one weight equals one, the weight of the node nearest to
the successor. The recursion is then the backward recursion of
{prf:ref}`thm-bolza-backward` for a different deterministic system, which
lives on the grid and moves to the node nearest to
$\mathbf{f}_t(\mathbf{x},\mathbf{u})$. The theorem applies to this rounded
system without change, so the recursion returns its optimal values and
decisions. The approximation error lies entirely in the replacement of the
model, and that replacement can be qualitatively wrong. Any change smaller
than half a grid step disappears in the rounded system. In the example below,
the weekly gain of storage in summer is such a change, and the rounded
reservoir cannot fill.

With linear interpolation, the successor's weight is split between the two
nodes that bracket it, as $1-\lambda$ and $\lambda$. These numbers are
nonnegative and sum to one, so they can be read as probabilities. The bracket
in {eq}`eq:interpolated-bellman` becomes the stage cost plus the expected
stored value at a random next node, and the interpolated recursion is the exact
backward recursion of a finite-state system with random successors: from
$x^{(i)}$ under $u$, the system moves to $x^{(j)}$ with probability
$\ell_j(f_t(x^{(i)},u))$. In the reservoir, a successor of 1033.7 hm³ on the
10 hm³ grid becomes 1030 hm³ with probability $0.63$ and 1040 hm³ with
probability $0.37$. Hat functions reproduce linear functions,
$\sum_j\ell_j(x)\,x^{(j)}=x$, so the random next node equals the true
successor on average. Linear interpolation thus replaces the deterministic
model by a Markov chain on the grid. Matching the mean motion of such a chain
to the continuous dynamics is the local-consistency condition at the base of
the Markov chain approximation method of {cite:t}`KushnerDupuis2001` for
continuous-time stochastic control. In $n$ dimensions, multilinear
interpolation on a rectangular grid multiplies one-dimensional hat weights.
Its $2^n$ weights on the corners of the enclosing cell are again nonnegative
and sum to one, so the same reading holds. The [next chapter](stochastic-dp.md)
makes the inflow itself random and interpolates each of its successors
linearly. The inflow probabilities and the interpolation weights then
multiply, and the result is again a distribution over grid states.

A cubic spline places negative weights on some nodes, and negative numbers
cannot be read as probabilities. No finite-state problem has
{eq}`eq:interpolated-bellman` as its exact recursion. The interpolated value
$\hat J_{t+1}$ can also leave the range of the stored values, and the
minimization over $\mathbf{u}$ is free to steer toward successors where it
does.

### Error Propagation Across Stages

Each stage of {eq}`eq:interpolated-bellman` uses stored values that already
contain the error of later stages. The sign of the weights decides whether
that error is passed on unchanged or amplified. To make this precise, define
the error of the nodal values and the interpolation error of the exact
cost-to-go,

$$
e_t=\max_i\big|v_{t,i}-J_t^\star\big(\mathbf{x}^{(i)}\big)\big|,
\qquad
\delta_t=\sup_{\mathbf{y}}\Big|\sum_j\ell_j(\mathbf{y})\,J_t^\star\big(\mathbf{x}^{(j)}\big)-J_t^\star(\mathbf{y})\Big|,
$$

where the supremum ranges over the successors reached from the grid. The
constant $\Lambda=\sup_{\mathbf{y}}\sum_j|\ell_j(\mathbf{y})|$ bounds how much
the interpolant can magnify errors in the stored values. Because the weights
sum to one, $\Lambda\ge 1$, with equality exactly when they are nonnegative.

```{prf:proposition} Error accumulation in interpolated backward recursion
:label: prop-interp-error-accumulation

Under the assumptions of {prf:ref}`thm-bolza-backward`, the nodal values
computed by {eq}`eq:interpolated-bellman` satisfy $e_T=0$ and

$$
e_t\le\Lambda\,e_{t+1}+\delta_{t+1},\qquad t=T-1,\ldots,1.
$$

With nonnegative weights, $\Lambda=1$ and
$e_1\le\delta_2+\delta_3+\cdots+\delta_T$.
```

```{prf:proof}
The base case holds because $v_{T,i}=c_\mathrm{T}(\mathbf{x}^{(i)})=J_T^\star(\mathbf{x}^{(i)})$.
For $t<T$, fix a grid state $\mathbf{x}^{(i)}$ and a control $\mathbf{u}$, and
let $\mathbf{y}=\mathbf{f}_t(\mathbf{x}^{(i)},\mathbf{u})$. The bracket in
{eq}`eq:interpolated-bellman` and the bracket in Bellman's equation share the
stage cost, so they differ by

$$
\sum_j\ell_j(\mathbf{y})\big(v_{t+1,j}-J_{t+1}^\star(\mathbf{x}^{(j)})\big)
+\Big(\sum_j\ell_j(\mathbf{y})\,J_{t+1}^\star(\mathbf{x}^{(j)})-J_{t+1}^\star(\mathbf{y})\Big).
$$

The first term is at most $\sum_j|\ell_j(\mathbf{y})|\,e_{t+1}\le\Lambda e_{t+1}$
in absolute value and the second at most $\delta_{t+1}$. The numbers $v_{t,i}$
and $J_t^\star(\mathbf{x}^{(i)})$ are the minima of the two brackets over the
same admissible controls. If two functions $a$ and $b$ differ by at most
$\varepsilon$ everywhere, then $\min a\le a(\mathbf{u}_b)\le b(\mathbf{u}_b)+\varepsilon=\min b+\varepsilon$,
where $\mathbf{u}_b$ minimizes $b$, and symmetrically. Hence
$|v_{t,i}-J_t^\star(\mathbf{x}^{(i)})|\le\Lambda e_{t+1}+\delta_{t+1}$ for every
$i$. Unrolling the inequality with $\Lambda=1$ gives the sum.
```

With nonnegative weights, every stage adds its own interpolation error and no
earlier error is amplified, so the error grows at most linearly with the
horizon. With negative weights, the bound allows each earlier error to be
multiplied by $\Lambda>1$ at every subsequent stage. The size of $\delta_t$
depends on the rule and the grid spacing $\Delta$. For a cost-to-go with
bounded derivatives on a one-dimensional grid, rounding gives
$\delta\le\tfrac{\Delta}{2}\max|J'|$ and linear interpolation gives
$\delta\le\tfrac{\Delta^2}{8}\max|J''|$, so halving the spacing halves the
first bound and quarters the second. A cubic spline converges faster still on
smooth functions, which is why it can be the most accurate of the three in
practice even though its $\Lambda$ exceeds one.

The proof compares the two minima through the largest difference between their
brackets, not an average one, and the example below shows why that is the
right comparison. An interpolation error that looks unbiased across controls
becomes biased once a control is selected, because the minimization prefers
controls whose successors happen to be evaluated favorably. On the coarse
reservoir grid, rounding moves the feasible successors by $-0.3$ hm³ on
average, over all weeks, grid states and releases. The successors that the
maximization selects are moved by $+45$ hm³ on average, out of a possible
$50$, and $94\%$ of them are rounded up. The same selection effect reappears as
overestimation bias in
[fitted Q-iteration](fitted-q-iteration.md#double-deep-q-network-double-dqn).

These bounds do not remove the curse of dimensionality from the opening of
this chapter. A grid with $N$ levels per coordinate still has $N^n$ nodes, each
with its own one-step problem, and multilinear interpolation touches $2^n$
stored values per evaluation. A more accurate rule reduces the number of
levels needed for a given error, not the exponential dependence on $n$. The
[approximate value methods](approximate-bellman-equations.md) replace the grid
by parametric approximators and face the same question about weights.
Interpolation with nonnegative weights that sum to one is an
{prf:ref}`averager <gordon-averager>` in Gordon's sense and keeps value
iteration stable, while least-squares fits and neural networks carry no such
guarantee.

### Decisions Between Grid Points

The recursion returns decisions at the grid states only, but the reservoir will
be measured between them. One option is to interpolate the table of decisions
$\boldsymbol{\pi}_{t,i}$. The reservoir's release table is all or nothing, as
{numref}`fig-hydro-policy` shows, so in each week it jumps from the ecological
minimum to turbine capacity at a switching storage. Linear interpolation of
that table stays within the release limits but returns intermediate releases
that the recursion never evaluated. A cubic spline through it overshoots at
the switch, as in the bottom row of {numref}`fig-dp-cardinal-functions`, and
commands releases below the ecological minimum and above turbine capacity.

The other option keeps the decision inside an optimization. At the measured
state, solve the one-step problem with the interpolant standing in for the
future:

$$
\hat{\boldsymbol{\pi}}_t(\mathbf{x})\in\arg\min_{\mathbf{u}\in\mathcal{U}_t(\mathbf{x})}
\Big[c_t(\mathbf{x},\mathbf{u})+\hat J_{t+1}\big(\mathbf{f}_t(\mathbf{x},\mathbf{u})\big)\Big].
$$ (eq:one-step-lookahead)

At a grid state this reproduces the stored decision. Between grid states it
returns an admissible control that is consistent with the values the recursion
computed. The rule {eq}`eq:one-step-lookahead` is a receding-horizon controller
with a prediction horizon of one step and terminal cost $\hat J_{t+1}$. In the
[receding-horizon chapter](receding-horizon-control.md), the terminal cost
stood in for the cost-to-go beyond the prediction horizon and had to be
supplied by the designer. Here the interpolated backward recursion computes it.
This sharpens the answer to the question that opened the chapter. What can be
computed in advance for a whole family of states is the cost-to-go, stored at
grid states. The decision at a particular state still comes from an online
minimization, now over a single step. A longer online horizon makes the
decision depend less on $\hat J$, at the price of a larger online problem.

### Example: Reservoir Operation with Unrounded Inflows

We return to the reservoir with the inflow forecast left unrounded, so that
successors fall between grid points, and solve it on a coarse grid of 19
storage levels spaced 100 hm³ apart. Two versions of
{prf:ref}`backward-recursion-interp` differ only in the interpolation rule
passed to them: nearest neighbor or linear (`np.interp`). A reference solution
uses linear interpolation on a grid twenty times finer. Each value function is
then used the way an operator would use it: the reservoir is simulated with the
exact, unrounded dynamics, and each week's release is chosen by the one-step
lookahead {eq}`eq:one-step-lookahead` with the interpolated $\hat J_{t+1}$.

```{code-cell} python
# Unrounded inflows: successors now fall between grid points.
w_raw = 60 + 340 * np.exp(-((weeks - 19) / 3.5) ** 2) + 60 * np.exp(-((weeks - 43) / 4) ** 2)
U_fine = np.linspace(u_min, u_max, 45)

def nearest(x_query, grid, values):
    step = grid[1] - grid[0]
    i = np.clip(np.round((x_query - grid[0]) / step).astype(int), 0, len(grid) - 1)
    return values[i]

def linear(x_query, grid, values):
    return np.interp(x_query, grid, values)

def backward_recursion(grid, interpolate):
    J = np.zeros((T + 1, len(grid)))
    J[T] = terminal_value(grid)
    for t in reversed(range(T)):
        x, u = grid[:, None], U_fine[None, :]
        x_next = np.clip(x + w_raw[t] - u, x_min, x_max)
        value = p[t] * kappa * head(x) * u + interpolate(x_next, grid, J[t + 1])
        J[t] = np.where(x + w_raw[t] - u >= x_min, value, -np.inf).max(axis=1)
    return J

def operate(grid, J, interpolate, x0):
    """Simulate the true dynamics, choosing each release by one Bellman step."""
    storage, revenue = [x0], 0.0
    for t in range(T):
        x = storage[-1]
        x_next = np.clip(x + w_raw[t] - U_fine, x_min, x_max)
        value = p[t] * kappa * head(x) * U_fine + interpolate(x_next, grid, J[t + 1])
        u = U_fine[np.argmax(np.where(x + w_raw[t] - U_fine >= x_min, value, -np.inf))]
        revenue += p[t] * kappa * head(x) * u
        storage.append(min(x + w_raw[t] - u, x_max))
    return np.array(storage), revenue + terminal_value(storage[-1])

coarse = np.arange(x_min, x_max + 100, 100)   # 19 levels
fine = np.arange(x_min, x_max + 5, 5)         # 361 levels
schemes = {
    "fine grid, linear (reference)": (fine, linear),
    "coarse grid, nearest neighbor": (coarse, nearest),
    "coarse grid, linear": (coarse, linear),
}
outcomes = {}
for name, (grid, interpolate) in schemes.items():
    J_scheme = backward_recursion(grid, interpolate)
    storage, revenue = operate(grid, J_scheme, interpolate, 1100)
    outcomes[name] = (grid, J_scheme[0], storage)
    print(f"{name:<32} predicted {linear(1100, grid, J_scheme[0]) / 1e6:6.2f} M$"
          f"   realized {revenue / 1e6:6.2f} M$")
```

```{code-cell} python
:label: fig-hydro-interpolation
:caption: Coarse-grid solutions of the reservoir problem with unrounded inflows. Left, the value function at the start of the year. Nearest-neighbor rounding (dashed) predicts far more revenue than the reference; linear interpolation on the same 19 points (blue) lies on the reference (grey). Right, storage when each value function is used to operate the reservoir from 1100 hm³. The nearest-neighbor policy never fills the reservoir in summer.
:tags: [remove-input]

make_interpolation_figure(T, outcomes, "fine grid, linear (reference)")
```

Linear interpolation on 19 points predicts a revenue within 0.2% of the
361-point reference and realizes one within 0.03%. Rounding on the same grid
fails in the two ways anticipated above, and both come from the size of a
typical weekly change. Its grid model is wrong in summer. The reservoir gains
about 30 hm³ per week at the minimum release, less than half the grid step, so
the rounded successor equals the current grid point. In the rounded model,
storing water gains nothing, so the policy releases the summer inflow at summer
prices instead of keeping it for winter, and the reservoir never fills. Its
value function is also too optimistic, by about a third, because the
maximization selects the releases whose successors round up and so adds water
that does not exist. Linear interpolation keeps the fractional part of every
step, so small changes accumulate correctly over the year.

> **Question.** Replace `linear` by a cubic spline,
> `lambda x_query, grid, values: CubicSpline(grid, values)(x_query)` with
> `CubicSpline` from `scipy.interpolate`, and rerun the coarse-grid recursion.
> Is the value function more or less accurate than with linear interpolation?
> Does the answer contradict {prf:ref}`prop-interp-error-accumulation`?

:::{dropdown} Answer
It is more accurate. On the 19 grid states, the largest error in $v_0$ falls
from 0.17 M\$ with linear interpolation to 0.02 M\$, and the spline predicts
and realizes 74.16 M\$ from 1100 hm³, like the reference. The proposition is
not contradicted. It bounds the error by interpolation errors $\delta_t$ that
are much smaller for the spline on this value function, and its factor
$\Lambda>1$ allows earlier errors to be amplified without forcing it. Negative
weights cause trouble where the interpolated function jumps, which the next
question examines.
:::

> **Question.** Instead of solving the one-step problem at the measured
> storage, an operator interpolates the coarse-grid release table between grid
> states. Record the maximizing release in `backward_recursion` for the linear
> rule, then operate the reservoir from 1100 hm³ by interpolating that table,
> first linearly and then with a cubic spline. What goes wrong in each case?

:::{dropdown} Answer
The table is all or nothing, so in every week that switches it jumps from 30 to
250 hm³/week between two grid states. A spline through those jumps overshoots,
as in the bottom row of {numref}`fig-dp-cardinal-functions`. Along the year it
commands releases from 22 hm³/week, below the ecological minimum, to
262 hm³/week, above turbine capacity, and over all weeks and storage levels its
range is $-38$ to 279 hm³/week. Linear interpolation of the table stays within
the limits because its weights are nonnegative. Its intermediate releases are
admissible but were never evaluated by the recursion, and it realizes
74.05 M\$, against 74.15 M\$ for the one-step lookahead with the same value
function.
:::


## Linear Quadratic Regulator via Dynamic Programming

What form do the value function and feedback law take when the dynamics are
linear and every cost is quadratic?

The [successive-approximation chapter](iterative-trajectory-optimization.md)
obtained a quadratic tail by eliminating controls backward. In the
linear-quadratic case, that tail is the optimal cost-to-go. Bellman's equation
therefore gives another derivation of the same Riccati recursion.

Linear dynamics and quadratic costs give a backward recursion that can be solved in closed form. The value function remains quadratic at every stage, and the optimal policy is a linear feedback law. No state grid, interpolation scheme, or general function approximator is needed. The recursion tracks a finite sequence of matrices.

Consider a discrete-time linear system:

$$
\mathbf{x}_{t+1} = A_t\mathbf{x}_t + B_t\mathbf{u}_t
$$

where $\mathbf{x}_t \in \mathbb{R}^n$ is the state and $\mathbf{u}_t \in \mathbb{R}^m$ is the control input. The matrices $A_t \in \mathbb{R}^{n \times n}$ and $B_t \in \mathbb{R}^{n \times m}$ describe the system dynamics at time $t$.

The cost function to be minimized is quadratic:

$$
J = \frac{1}{2}\mathbf{x}_T^\top Q_T \mathbf{x}_T + \frac{1}{2}\sum_{t=0}^{T-1} \left(\mathbf{x}_t^\top Q_t \mathbf{x}_t + \mathbf{u}_t^\top R_t \mathbf{u}_t\right)
$$

where $Q_T \succeq 0$ (positive semidefinite), $Q_t \succeq 0$, and $R_t \succ 0$ (positive definite) are symmetric matrices of appropriate dimensions. The positive definiteness of $R_t$ ensures the minimization problem is well-posed.

A quadratic terminal cost implies a quadratic value function at every earlier stage. Suppose the optimal cost-to-go at stage $t$ has the form

$$
J_t^\star(\mathbf{x}_t) = \frac{1}{2}\mathbf{x}_t^\top P_t \mathbf{x}_t
$$

for some positive semidefinite matrix $P_t$. At the terminal time, this is true by definition: $P_T = Q_T$.

Backward induction verifies the hypothesis. Assume $J_{t+1}^\star(\mathbf{x}_{t+1}) = \frac{1}{2}\mathbf{x}_{t+1}^\top P_{t+1} \mathbf{x}_{t+1}$. Bellman's equation at stage $t$ is

$$
J_t^\star(\mathbf{x}_t) = \min_{\mathbf{u}_t} \left[ \frac{1}{2}\mathbf{x}_t^\top Q_t \mathbf{x}_t + \frac{1}{2}\mathbf{u}_t^\top R_t \mathbf{u}_t + J_{t+1}^\star(\mathbf{x}_{t+1}) \right]
$$

Substituting the dynamics $\mathbf{x}_{t+1} = A_t\mathbf{x}_t + B_t\mathbf{u}_t$ and the quadratic form for $J_{t+1}^\star$:

$$
J_t^\star(\mathbf{x}_t) = \min_{\mathbf{u}_t} \left[ \frac{1}{2}\mathbf{x}_t^\top Q_t \mathbf{x}_t + \frac{1}{2}\mathbf{u}_t^\top R_t \mathbf{u}_t + \frac{1}{2}(A_t\mathbf{x}_t + B_t\mathbf{u}_t)^\top P_{t+1} (A_t\mathbf{x}_t + B_t\mathbf{u}_t) \right]
$$

Expanding the last term:

$$
(A_t\mathbf{x}_t + B_t\mathbf{u}_t)^\top P_{t+1} (A_t\mathbf{x}_t + B_t\mathbf{u}_t) = \mathbf{x}_t^\top A_t^\top P_{t+1} A_t \mathbf{x}_t + 2\mathbf{x}_t^\top A_t^\top P_{t+1} B_t \mathbf{u}_t + \mathbf{u}_t^\top B_t^\top P_{t+1} B_t \mathbf{u}_t
$$

The expression inside the minimization becomes:

$$
\frac{1}{2}\mathbf{x}_t^\top Q_t \mathbf{x}_t + \frac{1}{2}\mathbf{u}_t^\top R_t \mathbf{u}_t + \frac{1}{2}\mathbf{x}_t^\top A_t^\top P_{t+1} A_t \mathbf{x}_t + \mathbf{x}_t^\top A_t^\top P_{t+1} B_t \mathbf{u}_t + \frac{1}{2}\mathbf{u}_t^\top B_t^\top P_{t+1} B_t \mathbf{u}_t
$$

Collecting terms involving $\mathbf{u}_t$:

$$
= \frac{1}{2}\mathbf{x}_t^\top (Q_t + A_t^\top P_{t+1} A_t) \mathbf{x}_t + \mathbf{x}_t^\top A_t^\top P_{t+1} B_t \mathbf{u}_t + \frac{1}{2}\mathbf{u}_t^\top (R_t + B_t^\top P_{t+1} B_t) \mathbf{u}_t
$$

This is a quadratic function of $\mathbf{u}_t$. To find the minimizer, we take the gradient with respect to $\mathbf{u}_t$ and set it to zero:

$$
\frac{\partial}{\partial \mathbf{u}_t} = (R_t + B_t^\top P_{t+1} B_t) \mathbf{u}_t + B_t^\top P_{t+1} A_t \mathbf{x}_t = 0
$$

Since $R_t + B_t^\top P_{t+1} B_t$ is positive definite (both $R_t$ and $P_{t+1}$ are positive semidefinite with $R_t$ strictly positive), we can solve for the optimal control:

$$
\mathbf{u}_t^\star = -(R_t + B_t^\top P_{t+1} B_t)^{-1} B_t^\top P_{t+1} A_t \mathbf{x}_t
$$

Define the gain matrix:

$$
K_t = (R_t + B_t^\top P_{t+1} B_t)^{-1} B_t^\top P_{t+1} A_t
$$

so that $\mathbf{u}_t^\star = -K_t\mathbf{x}_t$. This is a **linear feedback policy**: the optimal control is simply a linear function of the current state.

Substituting $\mathbf{u}_t^\star$ back into the cost-to-go expression and simplifying (by completing the square), we obtain:

$$
J_t^\star(\mathbf{x}_t) = \frac{1}{2}\mathbf{x}_t^\top P_t \mathbf{x}_t
$$

where $P_t$ satisfies the **discrete-time Riccati equation**:

$$
P_t = Q_t + A_t^\top P_{t+1} A_t - A_t^\top P_{t+1} B_t (R_t + B_t^\top P_{t+1} B_t)^{-1} B_t^\top P_{t+1} A_t
$$


The resulting backward recursion is:


````{prf:algorithm} Backward Recursion for LQR
:label: backward-recursion-lqr

**Input:** System matrices $A_t, B_t$, cost matrices $Q_t, R_t, Q_T$, time horizon $T$

**Output:** Cost matrices $P_t$ and gain matrices $K_t$ for $t = 0, \ldots, T-1$

1. **Initialize:** $P_T = Q_T$

2. **For** $t = T-1, T-2, \ldots, 0$:
   1. Compute the gain matrix:

      $$K_t = (R_t + B_t^\top P_{t+1} B_t)^{-1} B_t^\top P_{t+1} A_t$$

   2. Compute the cost matrix via the Riccati equation:

      $$P_t = Q_t + A_t^\top P_{t+1} A_t - A_t^\top P_{t+1} B_t (R_t + B_t^\top P_{t+1} B_t)^{-1} B_t^\top P_{t+1} A_t$$

3. **End For**

4. **Return:** $\{P_0, \ldots, P_T\}$ and $\{K_0, \ldots, K_{T-1}\}$

**Optimal policy:** $\mathbf{u}_t^\star = -K_t\mathbf{x}_t$

**Optimal cost-to-go:** $J_t^\star(\mathbf{x}_t) = \frac{1}{2}\mathbf{x}_t^\top P_t \mathbf{x}_t$
````

### Local Stabilization of the Cart-Pole

The cart-pole in the trajectory-optimization chapter started from the downward configuration and required a large nonlinear maneuver to reach the top. Once it is near the upright equilibrium, a smaller problem remains: reject local deviations by moving the cart in response to the measured state. The nonlinear state and input are the same as before,

$$
\mathbf{x}=(p,v,\theta,\omega), \qquad
u=\text{horizontal cart acceleration},
$$

with $\theta=0$ at upright. Linearizing the discrete RK4 update $F_h$ at $(\mathbf{x}^\star,u^\star)=(0,0)$ gives

$$
\delta\mathbf{x}_{k+1}
\approx A\,\delta\mathbf{x}_k+B\,\delta u_k,
\qquad
A=\left.\frac{\partial F_h}{\partial\mathbf{x}}\right|_{(0,0)},
\quad
B=\left.\frac{\partial F_h}{\partial u}\right|_{(0,0)}.
$$

The experiment uses $h=0.02$ s, $Q=\operatorname{diag}(2,0.2,80,3)$, and $R=0.15$. For this time-invariant infinite-horizon case, the Riccati recursion converges to a fixed matrix $P$ that satisfies the discrete algebraic Riccati equation. The corresponding policy is $u_k=-K\delta\mathbf{x}_k$.

The unconstrained linear closed loop is asymptotically stable when every eigenvalue of $A-BK$ lies inside the unit disk. The physical implementation adds two constraints that are absent from that eigenvalue calculation: acceleration is clipped to $|u|\leq 8\;\mathrm{m\,s^{-2}}$, and the cart must remain inside a $2.4$ m rail. Three deterministic nonlinear rollouts distinguish the claims supported by the linearization:

1. An uncontrolled pole begins $5^\circ$ from upright.
2. The LQR controller begins from the same $5^\circ$ displacement.
3. The same controller begins $45^\circ$ from upright with the same actuator and rail limits.

```{code-cell} python
:tags: [remove-cell]

from pathlib import Path
import sys

code_directory = Path.cwd() / "code"
if str(code_directory) not in sys.path:
    sys.path.insert(0, str(code_directory))

from cartpole_control import (
    CartPoleParameters,
    design_lqr,
    format_lqr_metrics,
    make_lqr_animation,
    make_lqr_figure,
    run_lqr_cases,
)

cartpole_parameters = CartPoleParameters()
lqr_design = design_lqr(cartpole_parameters)
lqr_cases = run_lqr_cases(cartpole_parameters)
```

```{code-cell} python
:label: fig-cartpole-lqr-limits
:caption: The discrete linear closed loop is stable, and the nonlinear controller recovers from a 5 degree perturbation. Without control, the same initial displacement grows. From 45 degrees, the commanded acceleration saturates and the cart reaches its rail limit, so the local controller does not complete the recovery. All curves use the same nonlinear plant; only the initial state and controller differ.
:tags: [remove-input]

print(format_lqr_metrics(lqr_cases, lqr_design))
make_lqr_figure(lqr_cases, cartpole_parameters)
```

The closed-loop eigenvalues certify asymptotic stability of the unconstrained linearized model, not every trajectory of the nonlinear constrained plant. The $5^\circ$ rollout remains in a region where the linear approximation supplies useful actions. The $45^\circ$ rollout immediately asks for more acceleration than the actuator can provide, then exhausts the available rail. Increasing the entries of $Q$ cannot remove those physical limits.

```{code-cell} python
:label: anim-cartpole-lqr
:caption: Nonlinear validation of the local LQR controller. The left panel shows the uncontrolled fall from 5 degrees, the center panel shows recovery from the same state, and the right panel shows the 45 degree rollout ending at the rail limit. Python generates each frame from the recorded trajectories.
:tags: [remove-input]

from IPython.display import HTML, display
import matplotlib.pyplot as plt

lqr_animation = make_lqr_animation(lqr_cases, cartpole_parameters)
display(HTML(lqr_animation.to_jshtml()))
plt.close(lqr_animation._fig)
```

Balancing a pen on a finger motivates the action channel because the finger stabilizes the object by moving its base. The cart-pole model replaces the contact by a planar frictionless hinge. A real pen can slip, detach, flex, and rotate out of the plane, while sensing and hand motion introduce delays. The calculation establishes local stabilization for the stated rigid-body model; the classroom demonstration shares its instability and feedback mechanism, not all of its equations.

:::{dropdown} Inspect the linearization and LQR design
```{literalinclude} code/cartpole_control.py
:language: python
:start-at: def linearize_upright
:end-before: def simulate_lqr
:linenos:
```
:::

{download}`Download the shared nonlinear cart-pole and LQR source <code/cartpole_control.py>`.

## Summary and Outlook

Backward recursion replaces one trajectory optimization with a sequence of
state-indexed subproblems. Storing the cost-to-go at grid states and choosing
cardinal functions to interpolate between them makes the recursion a finite
computation. With nonnegative weights, the interpolated recursion is the exact
recursion of a grid model, deterministic for rounding and stochastic for linear
interpolation, and its errors add across stages instead of multiplying. A
decision between grid states comes from a one-step lookahead against the
interpolated cost-to-go. The linear-quadratic case needs no grid: the value
function stays quadratic, the recursion becomes the Riccati recursion, and the
policy is linear state feedback.

Linear interpolation already replaced each deterministic successor by a
distribution over two grid states. How do the value and policy change when the
model itself prescribes a distribution of possible successors?
[Stochastic dynamic programming](stochastic-dp.md) replaces the deterministic
continuation value by a conditional expectation.
