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
# Stochastic Dynamic Programming and Markov Decision Processes

Finite-horizon dynamic programming chooses an action from each state and time,
then follows one deterministic successor. How should the recursion change when
an action induces a distribution of successors and the decision rule itself may
be randomized or depend on history? Stochastic dynamic programming answers by
averaging continuation values under a transition kernel.

## Decision Rules and Policies

Which combinations of state, history, and randomization may a decision rule use
when selecting an action?

Before diving into stochastic systems, we need to establish terminology for the different types of strategies a decision maker might employ. In the deterministic setting, we implicitly used feedback controllers of the form $u(\mathbf{x}, t)$. In the stochastic setting, we must be more precise about what information policies can use and how they select actions.

A **decision rule** is a prescription for action selection in each state at a specified decision epoch. These rules can vary in their complexity based on two main criteria:

1. **Dependence on history**: Markovian or History-dependent
2. **Action selection method**: Deterministic or Randomized

**Markovian decision rules** depend only on the current state, while **history-dependent rules** consider the entire sequence of past states and actions. Formally, a history $h_t$ at time $t$ is:

$$h_t = (s_1, a_1, \ldots, s_{t-1}, a_{t-1}, s_t)$$

The set of all possible histories at time $t$, denoted $H_t$, grows exponentially with $t$:
- $H_1 = \mathcal{S}$ (just the initial state)
- $H_2 = \mathcal{S} \times \mathcal{A} \times \mathcal{S}$
- $H_t = \mathcal{S} \times (\mathcal{A} \times \mathcal{S})^{t-1}$

**Deterministic rules** select an action with certainty, while **randomized rules** specify a probability distribution over the action space.

These classifications lead to four types of decision rules:
1. **Markovian Deterministic (MD)**: $\pi_t: \mathcal{S} \rightarrow \mathcal{A}_s$
2. **Markovian Randomized (MR)**: $\pi_t: \mathcal{S} \rightarrow \mathcal{P}(\mathcal{A}_s)$
3. **History-dependent Deterministic (HD)**: $\pi_t: H_t \rightarrow \mathcal{A}_s$
4. **History-dependent Randomized (HR)**: $\pi_t: H_t \rightarrow \mathcal{P}(\mathcal{A}_s)$

where $\mathcal{P}(\mathcal{A}_s)$ denotes the set of probability distributions over $\mathcal{A}_s$.

A **policy** $\boldsymbol{\pi}$ is a sequence of decision rules, one for each decision epoch:

$$\boldsymbol{\pi} = (\pi_1, \pi_2, ..., \pi_{N-1})$$

The set of all policies of class $K$ (where $K \in \{HR, HD, MR, MD\}$) is denoted as $\Pi^K$. These policy classes form a hierarchy:

$$\Pi^{MD} \subset \Pi^{MR} \subset \Pi^{HR}, \quad \Pi^{MD} \subset \Pi^{HD} \subset \Pi^{HR}$$

The largest set $\Pi^{HR}$ contains all possible policies. We ask: under what conditions can we restrict attention to the much simpler set $\Pi^{MD}$ without loss of optimality?

```{admonition} Notation: rules vs. policies
:class: tip
- **Decision rule (kernel).** A map from information to action distributions:
  - Markov, deterministic:  $\pi_t:\mathcal{S}\to\mathcal{A}_s$
  - Markov, randomized:     $\pi_t(\cdot\mid s)\in\Delta(\mathcal{A}_s)$
  - History-dependent:       $\pi_t(\cdot\mid h_t)\in\Delta(\mathcal{A}_{s_t})$
- **Policy (sequence).** $\boldsymbol{\pi}=(\pi_1,\pi_2,\ldots)$.
- **Stationary policy.** $\boldsymbol{\pi}=\mathrm{const}(\pi)$ with $\pi_t\equiv\pi \ \forall t$.  
  By convention, we identify $\pi$ with its stationary policy $\mathrm{const}(\pi)$ when no confusion arises.
```

## Stochastic System Dynamics

How does a transition kernel replace the single deterministic successor in the
finite-horizon model?

In the stochastic setting, our system evolution takes the form:

$$ \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t) $$

Here, $\mathbf{w}_t$ represents a random disturbance or noise term at time $t$ due to the inherent uncertainty in the system's behavior. The stage cost function may also incorporate stochastic influences:

$$ c_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t) $$

In this context, our objective shifts from minimizing a deterministic cost to minimizing the expected total cost:

$$ \mathbb{E}\left[c_\mathrm{T}(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t)\right] $$

where the expectation is taken over the distributions of the random variables $\mathbf{w}_t$. The principle of optimality still holds in the stochastic case, but Bellman's optimality equation now involves an expectation:

$$ J_k^\star(\mathbf{x}_k) = \min_{\mathbf{u}_k} \mathbb{E}_{\mathbf{w}_k}\left[c_k(\mathbf{x}_k, \mathbf{u}_k, \mathbf{w}_k) + J_{k+1}^\star(\mathbf{f}_k(\mathbf{x}_k, \mathbf{u}_k, \mathbf{w}_k))\right] $$

In practice, this expectation is often computed by discretizing the distribution of $\mathbf{w}_k$ when the set of possible disturbances is very large or even continuous. Let's say we approximate the distribution with $K$ discrete values $\mathbf{w}_k^i$, each occurring with probability $p_k^i$. Then our Bellman equation becomes:

$$ J_k^\star(\mathbf{x}_k) = \min_{\mathbf{u}_k} \sum_{i=1}^K p_k^i \left(c_k(\mathbf{x}_k, \mathbf{u}_k, \mathbf{w}_k^i) + J_{k+1}^\star(\mathbf{f}_k(\mathbf{x}_k, \mathbf{u}_k, \mathbf{w}_k^i))\right) $$

## Optimality Equations in the Stochastic Setting

How is the deterministic continuation value replaced by an expectation over
all possible next states?

When dealing with stochastic systems, a central question arises: what information should our control policy use? In the most general case, a policy might use the entire history of observations and actions. However, as we'll see, the Markovian structure of our problems allows for dramatic simplifications.

Let $h_t = (s_1, a_1, s_2, a_2, \ldots, s_{t-1}, a_{t-1}, s_t)$ denote the complete history up to time $t$. In the stochastic setting, the history-based optimality equations become:

$$
u_t(h_t) = \sup_{a\in A_{s_t}}\left\{ r_t(s_t,a) + \sum_{j\in S} p_t(j\mid s_t,a)\, u_{t+1}(h_t,a,j) \right\},\quad u_N(h_N)=r_N(s_N)
$$

where we now explicitly use the transition probabilities $p_t(j|s_t,a)$ rather than a deterministic dynamics function.

````{prf:theorem} Principle of optimality for stochastic systems
:label: stoch-principle-opt

Let $u_t^*$ be the optimal expected return from epoch $t$ onward. Then:

**a.** $u_t^*$ satisfies the optimality equations:

$$u_t^*(h_t) = \sup_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S} p_t(j|s_t, a) u_{t+1}^*(h_t, a, j) \right\}$$

with boundary condition $u_N^*(h_N) = r_N(s_N)$.

**b.** Any policy $\pi^*$ that selects actions attaining the supremum (or maximum) in the above equation at each history is optimal.
````

**Intuition:** This formalizes Bellman's principle of optimality: "An optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision." The recursive structure means that optimal local decisions (choosing the best action at each step) lead to global optimality, even with uncertainty captured by the transition probabilities.

A simplification occurs when we examine these history-based equations more closely. The Markov property of our system dynamics and rewards means that the optimal return actually depends on the history only through the current state:

````{prf:proposition} State sufficiency for stochastic MDPs
:label: stoch-state-suff

In finite-horizon stochastic MDPs with Markovian dynamics and rewards, the optimal return $u_t^*(h_t)$ depends on the history only through the current state $s_t$. Thus we can write $u_t^*(h_t) = v_t^*(s_t)$ for some function $v_t^*$ that depends only on state and time.
````

````{prf:proof}
Following {cite:t}`Puterman1994` Theorem 4.4.2. We proceed by backward induction.

**Base case:** At the terminal time $N$, we have $u_N^*(h_N) = r_N(s_N)$ by the boundary condition. Since the terminal reward depends only on the final state $s_N$ and not on how we arrived there, $u_N^*(h_N) = u_N^*(s_N)$.

**Inductive step:** Assume $u_{t+1}^*(h_{t+1})$ depends on $h_{t+1}$ only through $s_{t+1}$ for all $t+1, \ldots, N$. Then from the optimality equation:

$$u_t^*(h_t) = \sup_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S} p_t(j|s_t, a) u_{t+1}^*(h_t, a, j) \right\}$$

By the induction hypothesis, $u_{t+1}^*(h_t, a, j)$ depends only on the next state $j$, so:

$$u_t^*(h_t) = \sup_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S} p_t(j|s_t, a) u_{t+1}^*(j) \right\}$$

Since the expression in brackets depends on $h_t$ only through the current state $s_t$ (the rewards and transition probabilities are Markovian), we conclude that $u_t^*(h_t) = u_t^*(s_t)$.
````

**Intuition:** The Markov property means that the current state contains all information needed to predict future evolution. The past provides no additional value for decision-making. This result allows us to work with value functions $v_t^*(s)$ indexed only by state and time, dramatically simplifying both theory and computation.

This state-sufficiency result, combined with the fact that randomization never helps when maximizing expected returns, leads to a dramatic simplification of the policy space:

````{prf:theorem} Policy reduction for stochastic MDPs
:label: stoch-policy-reduction

For finite-horizon stochastic MDPs with finite state and action sets:

$$
\sup_{\pi \in \Pi^{\mathrm{HR}}} v_\pi(s,t) = \max_{\pi \in \Pi^{\mathrm{MD}}} v_\pi(s,t)
$$

That is, there exists an optimal policy that is both deterministic and Markovian.
````

````{prf:proof}
Sketch following {cite:t}`Puterman1994` Lemma 4.3.1 and Theorem 4.4.2. First, Lemma 4.3.1 shows that for any function $w$ and any distribution $q$ over actions, $\sup_a w(a) \ge \sum_a q(a) w(a)$. Thus randomization cannot improve the expected value over choosing a single maximizing action. Second, by state sufficiency (Proposition {ref}`stoch-state-suff` and {cite:t}`Puterman1994` Thm. 4.4.2(a)), the optimal return depends on the history only through $(s_t,t)$. Therefore, selecting at each $(s_t,t)$ an action that attains the maximum yields a deterministic Markov decision rule which is optimal whenever the maximum is attained. If only a supremum exists, $\varepsilon$-optimal selectors exist by choosing actions within $\varepsilon$ of the supremum (see {cite:t}`Puterman1994` Thm. 4.3.4).
````

**Intuition:** Even in stochastic systems, randomization in the policy doesn't help when maximizing expected returns: you should always choose the action with the highest expected value. Combined with state sufficiency, this means simple state-to-action mappings are optimal.

These results justify focusing on deterministic Markov policies and lead to the backward recursion algorithm for stochastic systems: 

````{prf:algorithm} Backward Recursion for Stochastic Dynamic Programming
:label: backward-recursion-stochastic

**Input:** Terminal cost function $c_\mathrm{T}(\cdot)$, stage cost functions $c_t(\cdot, \cdot, \cdot)$, system dynamics $\mathbf{f}_t(\cdot, \cdot, \cdot)$, time horizon $\mathrm{T}$, disturbance distributions

**Output:** Optimal value functions $J_t^\star(\cdot)$ and optimal control policies $\pi_t^\star(\cdot)$ for $t = 1, \ldots, T$

1. Initialize $J_T^\star(\mathbf{x}) = c_\mathrm{T}(\mathbf{x})$ for all $\mathbf{x}$ in the state space
2. For $t = T-1, T-2, \ldots, 1$:
   1. For each state $\mathbf{x}$ in the state space:
      1. Compute $J_t^\star(\mathbf{x}) = \min_{\mathbf{u}} \mathbb{E}_{\mathbf{w}_t}\left[c_t(\mathbf{x}, \mathbf{u}, \mathbf{w}_t) + J_{t+1}^\star(\mathbf{f}_t(\mathbf{x}, \mathbf{u}, \mathbf{w}_t))\right]$
      2. Compute $\pi_t^\star(\mathbf{x}) = \arg\min_{\mathbf{u}} \mathbb{E}_{\mathbf{w}_t}\left[c_t(\mathbf{x}, \mathbf{u}, \mathbf{w}_t) + J_{t+1}^\star(\mathbf{f}_t(\mathbf{x}, \mathbf{u}, \mathbf{w}_t))\right]$
   2. End For
3. End For
4. Return $J_t^\star(\cdot)$, $\pi_t^\star(\cdot)$ for $t = 1, \ldots, T$
````

While SDP provides us with a framework for handling uncertainty, it makes the curse of dimensionality even more difficult to handle in practice. Both the state space and the disturbance space must be discretized. This can lead to a combinatorial explosion in the number of scenarios to be evaluated at each stage.

However, just as we tackled the challenges of continuous state spaces with discretization and interpolation, we can devise efficient methods to handle the additional complexity of evaluating expectations. This problem essentially becomes one of numerical integration. When the set of disturbances is continuous (as is often the case with continuous state spaces), we enter a domain where numerical quadrature methods could be applied. But these methods tend to scale poorly as the number of dimensions grows. This is where more efficient techniques, often rooted in Monte Carlo methods, come into play. Two ingredients tackle the curse of dimensionality:

1. Function approximation (through discretization, interpolation, neural networks, etc.)
2. Monte Carlo integration (simulation)

These two elements essentially distill the key ingredients of machine learning, which is the direction we'll be exploring in this course. 

## Example: Reservoir Operation with Random Inflows

How does the reservoir policy change when each week's release must be chosen
before that week's inflow is known?

The previous chapter operated a [seasonal hydroelectric
reservoir](finite-horizon-dp.md#example-seasonal-operation-of-a-hydroelectric-reservoir)
against an inflow forecast and treated the forecast as exact. Its policy
refills the reservoir after the spring freshet, holds it at capacity from
September to late November, and releases exactly the forecast inflow in each of
those weeks. That plan is only as good as the forecast: in a week wetter than
forecast, the reservoir is already full and the extra water goes over the
spillway.
Stochastic dynamic programming has long been used to compute reservoir
operating policies that account for this risk {cite:p}`Stedinger1984`. We make
the inflow random and keep every other element of the deterministic model: 52
weekly stages, the storage and release limits, the head $h(x)$, the price $p_t$
and the terminal value.

Let $\bar w_t$ denote the forecast of the previous chapter. The inflow in week
$t$ is

$$
w_t = \xi_t\,\bar w_t,
\qquad
\xi_t = \begin{cases}
0.6 & \text{with probability } 1/4, \\
1.0 & \text{with probability } 1/2, \\
1.4 & \text{with probability } 1/4,
\end{cases}
$$

where the factors $\xi_0, \ldots, \xi_{51}$ are independent. The mean inflow
equals the forecast, and one week in four is 40% wetter than forecast. The
inflow plays the role of the disturbance $\mathbf{w}_t$ in the stochastic
dynamics above. Because the factors are independent, past inflows carry no
information about the next one, and storage alone remains a sufficient state,
as in {prf:ref}`stoch-state-suff`.

The operator sets the release $u_t$ at the start of the week, before the inflow
arrives. In a dry week the requested release can exceed the water available
above dead storage. The turbines then stop when the level reaches dead storage,
so the volume actually released is

$$
\tilde u_t = \min\{u_t,\ x_t + w_t - x_{\min}\}.
$$

Storage evolves as $x_{t+1} = \min\{x_t + w_t - \tilde u_t,\ \bar x\}$, and the
week earns $p_t\,\kappa\,h(x_t)\,\tilde u_t$. Both the successor and the revenue
now depend on the disturbance, as $\mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t,
\mathbf{w}_t)$ and $c_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t)$ do in the
general formulation. Writing $\xi^k$ for the three factors and $q_k$ for their
probabilities, Bellman's equation reads

$$
J_t^\star(x) = \max_{u_{\min} \le u \le u_{\max}}
\sum_{k=1}^{3} q_k \Big[\, p_t\,\kappa\,h(x)\,\tilde u^k
+ J_{t+1}^\star\big(\min\{x + \xi^k \bar w_t - \tilde u^k,\ \bar x\}\big) \Big],
\qquad
\tilde u^k = \min\{u,\ x + \xi^k \bar w_t - x_{\min}\},
$$

with the terminal value $J_T^\star(x) = p_T\,\kappa\,h(x)\,(x - x_{\min})$ as
before. The expectation sits inside the maximum because a single release has to
serve all three inflows.

The code below changes the deterministic solver in two places. The bracket is
averaged over the three inflows before the maximum over releases is taken. And
the successors $x + 0.6\,\bar w_t - u$ and $x + 1.4\,\bar w_t - u$ are in
general no longer multiples of the 10 hm³ grid step, so $J_{t+1}^\star$ is
evaluated between grid points by linear interpolation, as in [the interpolation
section](finite-horizon-dp.md#handling-continuous-spaces-with-interpolation)
of the previous chapter. Given the single factor 1 with probability 1, the same
function solves the deterministic problem and returns its 74.1 M\$ from
1100 hm³; with a known inflow, stopping a release at dead storage is the same
as forbidding it. This is the *certainty-equivalent* problem: the random inflow
is replaced by its mean, and the resulting policy is applied as if the forecast
were exact. The last lines operate both policies for 2000 simulated years, on
the same inflow sequences.

```{code-cell} python
import numpy as np

# The deterministic model of the previous chapter.
T = 52
weeks = np.arange(T)
w_bar = 60 + 340 * np.exp(-((weeks - 19) / 3.5) ** 2) + 60 * np.exp(-((weeks - 43) / 4) ** 2)
w_bar = 10 * np.round(w_bar / 10)                    # inflow forecast (hm³/week)
p = 60 + 30 * np.cos(2 * np.pi * (weeks - 5) / 52)   # price ($/MWh)
x_min, x_max = 200, 2000
u_min, u_max = 30, 250
kappa, p_T = 2.45, 50

def head(x):
    return 40 + 40 * np.sqrt(x / x_max)

def terminal_value(x):
    return p_T * kappa * head(x) * (x - x_min)

dx = 10
X = np.arange(x_min, x_max + dx, dx)
U = np.arange(u_min, u_max + dx, dx)

# Random inflow w_t = xi_t * w_bar_t, with xi_t drawn independently each week.
factors = np.array([0.6, 1.0, 1.4])
probs = np.array([0.25, 0.5, 0.25])

def solve(factors, probs):
    J = np.zeros((T + 1, len(X)))
    policy = np.zeros((T, len(X)))
    J[T] = terminal_value(X)
    x, u = X[:, None], U[None, :]                     # all (storage, release) pairs
    for t in reversed(range(T)):
        value = np.zeros((len(X), len(U)))
        for factor, prob in zip(factors, probs):      # expectation over the inflow...
            inflow = factor * w_bar[t]
            release = np.minimum(u, x + inflow - x_min)       # turbines stop at dead storage
            x_next = np.minimum(x + inflow - release, x_max)  # water above capacity spills
            value += prob * (p[t] * kappa * head(x) * release
                             + np.interp(x_next, X, J[t + 1]))
        J[t] = value.max(axis=1)                      # ...inside the maximum over releases
        policy[t] = U[value.argmax(axis=1)]
    return J, policy

def simulate(policy, years, x0):
    """Operate the reservoir for many years at once: one row of factors per year."""
    x = np.full(len(years), float(x0))
    storage, revenue, spill = [x], 0.0, 0.0
    for t in range(T):
        inflow = years[:, t] * w_bar[t]
        requested = policy[t, np.round((x - x_min) / dx).astype(int)]
        release = np.minimum(requested, x + inflow - x_min)
        revenue = revenue + p[t] * kappa * head(x) * release
        spill = spill + np.maximum(x + inflow - release - x_max, 0)
        x = np.minimum(x + inflow - release, x_max)
        storage.append(x)
    return np.array(storage), revenue + terminal_value(x), spill

J_ce, policy_ce = solve([1.0], [1.0])         # certainty equivalent: inflow = forecast
J_sto, policy_sto = solve(factors, probs)     # stochastic dynamic programming

rng = np.random.default_rng(0)
years = rng.choice(factors, size=(2000, T), p=probs)   # the same inflows for both policies
i0 = np.searchsorted(X, 1100)
results = {}
for name, J, policy in (("certainty equivalent", J_ce, policy_ce),
                        ("stochastic", J_sto, policy_sto)):
    storage, revenue, spill = simulate(policy, years, x0=1100)
    results[name] = (storage, revenue, spill)
    print(f"{name:<21} predicted {J[0, i0] / 1e6:5.2f} M$   simulated {revenue.mean() / 1e6:5.2f} M$"
          f"   spill {spill.mean():3.0f} hm³/year")
```

The two backward passes predict different things. The certainty-equivalent
pass predicts 74.06 M\$, the revenue its policy would earn if every week's
inflow matched the forecast, which never happens. On the simulated inflows the
same policy earns 73.05 M\$ on average, about 1 M\$ less, and spills water in
96% of the years. At a water value of about 11 k\$ per hm³, the 85 hm³ it
spills each year account for most of the shortfall. The stochastic pass predicts
73.75 M\$, and the simulation agrees within its standard error of 0.06 M\$.
This agreement is expected: $J_0^\star(1100)$ is the expected revenue of the
stochastic policy under the inflow model, and the average over simulated years
estimates the same expectation.

On the same inflows, the stochastic policy earns 0.68 M\$ more per year than
the certainty-equivalent policy. Stochastic programming calls this difference
the *value of the stochastic solution* {cite:p}`Birge2011`. It is small next to
the total revenue, but most of that revenue comes from water that any rule would
sell. In the deterministic model, running the river without storing anything
earned 66.8 M\$, and managing the storage added 7.3 M\$. The loss from
ignoring the inflow uncertainty is about a tenth of that amount.

```{code-cell} python
:tags: [remove-cell]

from pathlib import Path
import sys

code_directory = Path.cwd() / "code"
if str(code_directory) not in sys.path:
    sys.path.insert(0, str(code_directory))

from hydro_reservoir_stochastic import (
    make_stochastic_comparison_figure,
    make_stochastic_policy_figure,
)

water_value_ce = np.gradient(J_ce, dx, axis=1)    # dJ_t/dx on the grid, $ per hm³
water_value_sto = np.gradient(J_sto, dx, axis=1)
storage_ce, revenue_ce, spill_ce = results["certainty equivalent"]
storage_sto, revenue_sto, spill_sto = results["stochastic"]
```

```{code-cell} python
:label: fig-hydro-stochastic-comparison
:caption: Both policies operated on the same 2000 simulated years from $x_0 = 1100$ hm³. Left, storage from August to December: median (solid) and 5th and 95th percentiles (dashed). The certainty-equivalent policy (orange) holds the reservoir at capacity and spills whenever a week is wetter than forecast; the stochastic policy (blue) holds it about 50 hm³ below. Right, the revenue of the stochastic policy minus that of the certainty-equivalent policy in each simulated year.
:tags: [remove-input]

make_stochastic_comparison_figure(storage_ce, storage_sto, revenue_ce,
                                  revenue_sto, x_max, first_week=30)
```

Because both policies face the same inflows, the gain can be measured year by
year, as in the right panel of {numref}`fig-hydro-stochastic-comparison`. Its
mean has a standard error of 0.01 M\$, against 0.06 M\$ for either mean revenue:
a wet or dry year raises or lowers the revenue of both policies alike, and this
common variation cancels in the difference. Evaluating competing policies on the
same random inputs is a variance-reduction technique known as *common random
numbers*. The stochastic policy earns more in 87% of the years. The other years
have dry autumns, in which the certainty-equivalent policy spills little and
the margin kept by the stochastic policy turns out to be unnecessary.

The gain comes almost entirely from the autumn, shown in the left panel. The
certainty-equivalent policy refills the reservoir to capacity in September and
holds it there, releasing the forecast inflow each week, so every week wetter
than forecast spills. The stochastic policy stops about 50 hm³ below capacity.
In late October the forecast inflow is 120 hm³ per week, and the wettest week
brings $0.4 \times 120 = 48$ hm³ more than forecast. The policy keeps exactly
enough room to absorb it.

The size of this margin follows from the comparison that explained the
deterministic policy in the previous chapter: the turbines run when releasing
one more hm³, worth $p_t\,\kappa\,h(x)$, earns more than keeping it, worth the
water value $\partial J_{t+1}^\star/\partial x$ at the next storage. With random
inflow, the second term becomes an average over the three successors
$x_{t+1}^k$. For an inflow that fills the reservoir, an hm³ kept this week
spills and adds nothing next week, so those successors drop out of the
average. As long as the release is not cut short at dead storage, the
slope $\partial Q_t/\partial u$ from the previous chapter becomes

$$
p_t\,\kappa\,h(x) \;-\; \sum_{k:\ x_{t+1}^k < \bar x} q_k\,
\frac{\partial J_{t+1}^\star}{\partial x}\big(x_{t+1}^k\big).
$$

The rule becomes: run the turbines when the price exceeds the expected value of
the water kept, counting water that would spill as worth nothing. In late
October, near capacity, one hm³ earns about 10.3 k\$ if released and has a
water value of about 11.2 k\$ if kept. Below the margin, the kept hm³ never
spills, so keeping it is worth more. Above the margin, it spills in the wettest
week, with probability 1/4, and its expected value falls to
$0.75 \times 11.2 = 8.4$ k\$, less than the 10.3 k\$ from releasing it. The
policy therefore holds the storage level from which the wettest week just fills
the reservoir. In the deterministic model the inflow is known, so the policy can
fill the reservoir to capacity without spilling.

```{code-cell} python
:label: fig-hydro-stochastic-policy
:caption: Solution of the stochastic problem. Left, the optimal release $u_t^\star(x)$ below the inflow forecast (dark marks) and the range of the random inflow (light bars). White curves are 40 simulated years of operation from 1100 hm³. Right, the water value at the start of the freshet. With the inflow known (orange), it drops abruptly at the storage level from which the freshet fills the reservoir; with random inflow (blue), the drop is spread from about 1100 hm³ to capacity.
:tags: [remove-input]

make_stochastic_policy_figure(w_bar, factors, X, policy_sto, storage_sto[:, :40],
                              14, water_value_ce[14], water_value_sto[14])
```

The same averaging changes the water value. The right panel of
{numref}`fig-hydro-stochastic-policy` shows it in week 14, early April, when the
freshet begins. In the certainty-equivalent model it drops abruptly at
1770 hm³: from that level on, the forecast freshet fills the reservoir, and any
additional hm³ is certain to spill. In the stochastic model the drop is spread
from about 1100 hm³ to capacity. An additional hm³ spills only if the freshet
is wet enough, and the probability of such a freshet rises gradually with
storage. More generally, each $J_t^\star$ averages $J_{t+1}^\star$ over three
successors, and this averaging smooths the kinks of the deterministic value
function at every step of the recursion.

The white curves in the left panel are 40 simulated years. In the deterministic
model a single path described the operation of the policy; here each year
follows its own. The paths spread out in January and February, while the policy
releases the minimum and storage follows the random inflow. They come back
together at the end of the winter drawdown, when every year reaches low storage
before the freshet, spread again during the freshet, whose volume differs from
year to year, and converge in autumn toward the level below capacity where the
policy holds them. The outcome of a policy under uncertainty is a distribution,
and estimating it takes many simulated trajectories. The same holds for the
reinforcement learning methods of later chapters: one good trajectory does not
show that a policy is good.

> **Question.** If the operator could see each week's inflow before choosing
> the release, would the expected revenue from 1100 hm³ be above or below
> 73.75 M\$? Which lines of `solve` would change?

:::{dropdown} Answer
Above, or at least equal. Seeing the inflow first allows a different release
for each of the three inflows, and choosing the same release for all three
remains possible, so the maximum cannot decrease. In Bellman's equation the
maximum moves inside the expectation,
$J_t^\star(x) = \sum_k q_k \max_u [\,\cdots]$. In `solve`, each term is
maximized over releases before it is added, `value += prob * (...).max(axis=1)`
with `value` a vector over storage levels, and `J[t] = value`. The policy then
depends on the inflow as well as on the storage. The expected revenue becomes
74.01 M\$, so knowing each week's inflow in advance is worth 0.26 M\$ per year.
:::

> **Question.** In this model the inflow factor is drawn independently each
> week, so a wet April says nothing about May. Real freshets tend to be wet or
> dry for weeks at a time, because they melt from a single snowpack. What would
> the state have to contain for the backward recursion to remain valid, and
> what would that cost?

:::{dropdown} Answer
With correlated inflows, storage is no longer a sufficient state: the
distribution of next week's inflow depends on past inflows, and the argument of
{prf:ref}`stoch-state-suff` no longer applies to storage alone. If the factor
follows a Markov chain, adding last week's factor to the state restores it. The
state becomes $(x_t, \xi_{t-1})$, with $181 \times 3 = 543$ grid points instead
of 181, and the expectation uses the transition probabilities out of
$\xi_{t-1}$. A measured snowpack would be a continuous second state variable
with its own grid. Each such variable multiplies the size of the grid, which is
the curse of dimensionality again. A common alternative is to re-solve a
deterministic problem every week with an updated forecast, as in
[receding-horizon control](receding-horizon-control.md).
:::

## Markov Decision Process Formulation

Which state, action, transition, reward, and horizon objects define the common
finite MDP representation?

Rather than expressing the stochasticity in our system through a disturbance term as a parameter to a deterministic difference equation, we often work with an alternative representation (more common in operations research) which uses the Markov Decision Process formulation. The idea is that when we model our system in this way with the disturbance term being drawn independently of the previous stages, the induced trajectories are those of a Markov chain. Hence, we can re-cast our control problem in that language, leading to the so-called Markov Decision Process framework in which we express the system dynamics in terms of transition probabilities rather than explicit state equations. In this framework, we express the probability that the system is in a given state using the transition probability function:

$$ p_t(\mathbf{x}_{t+1} | \mathbf{x}_t, \mathbf{u}_t) $$

This function gives the probability of transitioning to state $\mathbf{x}_{t+1}$ at time $t+1$, given that the system is in state $\mathbf{x}_t$ and action $\mathbf{u}_t$ is taken at time $t$. Therefore, $p_t$ specifies a conditional probability distribution over the next states: namely, the sum (for discrete state spaces) or integral over the next state should be 1.

Given the control theory formulation of our problem via a deterministic dynamics function and a noise term, we can derive the corresponding transition probability function through the following relationship:

$$
\begin{aligned}
p_t(\mathbf{x}_{t+1} | \mathbf{x}_t, \mathbf{u}_t) &= \mathbb{P}(\mathbf{W}_t \in \left\{\mathbf{w} \in \mathbf{W}: \mathbf{x}_{t+1} = f_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w})\right\}) \\
&= \sum_{\left\{\mathbf{w} \in \mathbf{W}: \mathbf{x}_{t+1} = f_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w})\right\}} q_t(\mathbf{w})
\end{aligned}
$$

Here, $q_t(\mathbf{w})$ represents the probability density or mass function of the disturbance $\mathbf{W}_t$ (assuming discrete state spaces). When dealing with continuous spaces, the above expression simply contains an integral rather than a summation. 


For a system with deterministic dynamics and no disturbance, the transition probabilities become much simpler and can be expressed using the indicator function. Given a deterministic system with dynamics:

$$ \mathbf{x}_{t+1} = f_t(\mathbf{x}_t, \mathbf{u}_t) $$

The transition probability function can be expressed as:

$$ p_t(\mathbf{x}_{t+1} | \mathbf{x}_t, \mathbf{u}_t) = \begin{cases}
1 & \text{if } \mathbf{x}_{t+1} = f_t(\mathbf{x}_t, \mathbf{u}_t) \\
0 & \text{otherwise}
\end{cases} $$

With this transition probability function, we can recast our Bellman optimality equation:

$$ J_t^\star(\mathbf{x}_t) = \max_{\mathbf{u}_t \in \mathbf{U}} \left\{ c_t(\mathbf{x}_t, \mathbf{u}_t) + \sum_{\mathbf{x}_{t+1}} p_t(\mathbf{x}_{t+1} | \mathbf{x}_t, \mathbf{u}_t) J_{t+1}^\star(\mathbf{x}_{t+1}) \right\} $$

Here, ${c}(\mathbf{x}_t, \mathbf{u}_t)$ represents the expected immediate reward (or negative cost) when in state $\mathbf{x}_t$ and taking action $\mathbf{u}_t$ at time $t$. The summation term computes the expected optimal value for the future states, weighted by their transition probabilities.

This formulation offers several advantages:

1. It makes the Markovian nature of the problem explicit: the future state depends only on the current state and action, not on the history of states and actions.

2. For discrete-state problems, the entire system dynamics can be specified by a set of transition matrices, one for each possible action.

3. It allows us to bridge the gap with the wealth of methods in the field of probabilistic graphical models and statistical machine learning techniques for modelling and analysis. 

## Notation in Operations Reseach

How do equivalent reward, cost, transition, and policy conventions map across
reinforcement learning and operations research?

The presentation above was intended to bridge the gap between the control-theoretic perspective and the world of closed-loop control through the idea of determining the value function of a parametric optimal control problem. We then saw how the backward induction procedure was applicable to both the deterministic and stochastic cases by taking the expectation over the disturbance variable. We then said that we can alternatively work with a representation of our system where instead of writing our model as a deterministic dynamics function taking a disturbance as an input, we would rather work directly via its transition probability function, which gives rise to the Markov chain interpretation of our system in simulation.

Note that the notation used in control theory tends to differ from that found in operations research communities, in which the field of dynamic programming flourished. We summarize those (purely notational) differences in this section.

In operations research, the system state at each decision epoch is typically denoted by $s \in \mathcal{S}$, where $S$ is the set of possible system states. When the system is in state $s$, the decision maker may choose an action $a$ from the set of allowable actions $\mathcal{A}_s$. The union of all action sets is denoted as $\mathcal{A} = \bigcup_{s \in \mathcal{S}} \mathcal{A}_s$.

The dynamics of the system are described by a transition probability function $p_t(j | s, a)$, which represents the probability of transitioning to state $j \in \mathcal{S}$ at time $t+1$, given that the system is in state $s$ at time $t$ and action $a \in \mathcal{A}_s$ is chosen. This transition probability function satisfies:

$$\sum_{j \in \mathcal{S}} p_t(j | s, a) = 1$$

It's worth noting that in operations research, we typically work with reward maximization rather than cost minimization, which is more common in control theory. However, we can easily switch between these perspectives by simply negating the quantity. That is, maximizing a reward function is equivalent to minimizing its negative, which we would then call a cost function.

The reward function is denoted by $r_t(s, a)$, representing the reward received at time $t$ when the system is in state $s$ and action $a$ is taken. In some cases, the reward may also depend on the next state, in which case it is denoted as $r_t(s, a, j)$. The expected reward can then be computed as:

$$r_t(s, a) = \sum_{j \in \mathcal{S}} r_t(s, a, j) p_t(j | s, a)$$

Combined together, these elements specify a Markov decision process, which is fully described by the tuple:

$$\{T, S, \mathcal{A}_s, p_t(\cdot | s, a), r_t(s, a)\}$$

where $\mathrm{T}$ represents the set of decision epochs (the horizon). 

## What is an Optimal Policy?

Against which competing policy class and initial states must a policy be
compared before it can be called optimal?

Let's go back to the starting point and define what it means for a policy to be optimal in a Markov Decision Problem. For this, we will be considering different possible search spaces (policy classes) and comparing policies based on the ordering of their value from any possible start state. The value of a policy $\boldsymbol{\pi}$ (optimal or not) is defined as the expected total reward obtained by following that policy from a given starting state. Formally, for a finite-horizon MDP with $N$ decision epochs, we define the value function $v^{\boldsymbol{\pi}}(s, t)$ as:

$$
v^{\boldsymbol{\pi}}(s, t) \triangleq \mathbb{E}\left[\sum_{k=t}^{N-1} r_t(S_k, A_k) + r_N(S_N) \mid S_t = s\right]
$$

where $S_t$ is the state at time $t$, $A_t$ is the action taken at time $t$, and $r_t$ is the reward function. For simplicity, we write $v^{\boldsymbol{\pi}}(s)$ to denote $v^{\boldsymbol{\pi}}(s, 1)$, the value of following policy $\boldsymbol{\pi}$ from state $s$ at the first stage over the entire horizon $N$.

In finite-horizon MDPs, our goal is to identify an optimal policy, denoted by $\boldsymbol{\pi}^*$, that maximizes total expected reward over the horizon $N$. Specifically:

$$
v^{\boldsymbol{\pi}^*}(s) \geq v^{\boldsymbol{\pi}}(s), \quad \forall s \in \mathcal{S}, \quad \forall \boldsymbol{\pi} \in \Pi^{\text{HR}}
$$

We call $\boldsymbol{\pi}^*$ an **optimal policy** because it yields the highest possible value across all states and all policies within the policy class $\Pi^{\text{HR}}$. We denote by $v^*$ the maximum value achievable by any policy:

$$
v^*(s) = \max_{\boldsymbol{\pi} \in \Pi^{\text{HR}}} v^{\boldsymbol{\pi}}(s), \quad \forall s \in \mathcal{S}
$$

In reinforcement learning literature, $v^*$ is typically referred to as the "optimal value function," while in some operations research references, it might be called the "value of an MDP." An optimal policy $\boldsymbol{\pi}^*$ is one for which its value function equals the optimal value function:

$$
v^{\boldsymbol{\pi}^*}(s) = v^*(s), \quad \forall s \in \mathcal{S}
$$

This notion of optimality applies to every state. Policies optimal in this sense are sometimes called "uniformly optimal policies." A weaker notion of optimality, often encountered in reinforcement learning practice, is optimality with respect to an initial distribution of states. In this case, we seek a policy $\boldsymbol{\pi} \in \Pi^{\text{HR}}$ that maximizes:

$$
\sum_{s \in \mathcal{S}} v^{\boldsymbol{\pi}}(s) P_1(S_1 = s)
$$

where $P_1(S_1 = s)$ is the probability of starting in state $s$.

Both definitions rank policies, and the ranking has two ingredients that are
modeling choices rather than consequences of the dynamics.

The first is the order itself. Uniform optimality compares value functions
state by state: $\boldsymbol{\pi}$ is at least as good as $\boldsymbol{\pi}'$
when $v^{\boldsymbol{\pi}}(s) \ge v^{\boldsymbol{\pi}'}(s)$ for every
$s \in \mathcal{S}$. This is a *partial* order. Two policies can be
incomparable, each better than the other from some initial state. The existence
of a policy that dominates all others in this order is therefore a result to
be proved, not an assumption, and the backward induction algorithm below
supplies the proof for finite horizons.

The second is the expectation. With noise, a policy induces a distribution over
total rewards from each state, and a ranking needs one number per
distribution. Taking the mean is the classical answer of expected utility
theory, going back to von Neumann and Morgenstern: preferences over uncertain
outcomes that satisfy a few consistency axioms can be represented by the
expectation of a utility. Using the reward itself as the utility makes the
decision maker *risk neutral*. Two policies with the same mean are equally good,
however different their spreads or worst cases. A reservoir operator who must
avoid spilling in a wet year may not accept that trade. Risk-sensitive
formulations replace the mean by another functional of the distribution, for
example the conditional value at risk (CVaR), the mean of the worst
$\alpha$-fraction of outcomes. Such criteria lead to different Bellman
equations, and some of them no longer decompose stage by stage. This book keeps
the expectation throughout.

The maximum value can be achieved by searching over the space of deterministic Markovian Policies. Consequently:

$$ v^*(s) = \max_{\boldsymbol{\pi} \in \Pi^{\mathrm{HR}}} v^{\boldsymbol{\pi}}(s) = \max _{\boldsymbol{\pi} \in \Pi^{M D}} v^{\boldsymbol{\pi}}(s), \quad s \in S$$

This equality significantly simplifies the computational complexity of our algorithms, as the search problem can now be decomposed into $N$ sub-problems in which we only have to search over the set of possible actions. This is the backward induction algorithm, which we present a second time, but departing this time from the control-theoretic notation and using the MDP formalism:  

````{prf:algorithm} Backward Induction
:label: backward-induction

**Input:** State space $S$, Action space $A$, Transition probabilities $p_t$, Reward function $r_t$, Time horizon $N$

**Output:** Optimal value functions $v^*$ and optimal action sets $A^*_{s,t}$

1. Initialize:
   - Set $t = N$
   - For all $s \in S$:

     $$v^*(s, N) = r_N(s)$$

2. For $t = N-1$ to $1$:
   - For each $s \in S$:
     a. Compute the optimal value function:

        $$v^*(s, t) = \max_{a \in A_s} \left\{r_t(s, a) + \mathbb{E}_{s' \sim p_t(\cdot | s, a)}\left[v^*(s', t+1)\right]\right\}$$
     
     b. Determine the set of optimal actions:

        $$A_{s,t}^* = \arg\max_{a \in A_s} \left\{r_t(s, a) + \mathbb{E}_{s' \sim p_t(\cdot | s, a)}\left[v^*(s', t+1)\right]\right\}$$

3. Return the optimal value functions $v^*(s, t)$ and optimal action sets $A_{s,t}^*$ for all $t$ and $s$
````

When $S$ is finite, the expectation is the sum $\sum_{s' \in S} p_t(s' | s, a)\, v^*(s', t+1)$. Writing it as an expectation makes the connection with the stochastic dynamic program from earlier in this chapter explicit: $\mathbb{E}_{s' \sim p_t(\cdot|s,a)}$ plays the role of $\mathbb{E}_{\mathbf{w}_t}$, with the noise absorbed into the transition kernel.

Note that the same procedure can also be used for finding the value of a policy with minor changes; 

````{prf:algorithm} Policy Evaluation
:label: backward-policy-evaluation

**Input:** 
- State space $S$
- Action space $A$
- Transition probabilities $p_t$
- Reward function $r_t$
- Time horizon $N$
- A markovian deterministic policy $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_{N-1})$

**Output:** Value function $v^{\boldsymbol{\pi}}$ for policy $\boldsymbol{\pi}$

1. Initialize:
   - Set $t = N$
   - For all $s \in S$:

     $$v^{\boldsymbol{\pi}}(s, N) = r_N(s)$$

2. For $t = N-1$ to $1$:
   - For each $s \in S$:
     a. Compute the value function for the given policy:

        $$v^{\boldsymbol{\pi}}(s, t) = r_t(s, \pi_t(s)) + \mathbb{E}_{s' \sim p_t(\cdot | s, \pi_t(s))}\left[v^{\boldsymbol{\pi}}(s', t+1)\right]$$

3. Return the value function $v^{\boldsymbol{\pi}}(s, t)$ for all $t$ and $s$
````

This code could also finally be adapted to support randomized policies using:

$$v^{\boldsymbol{\pi}}(s_t, t) = \sum_{a_t \in \mathcal{A}_{s_t}} \pi_t(a_t \mid s_t) \left( r_t(s_t, a_t) + \sum_{j \in S} p_t(j | s_t, a_t) v^{\boldsymbol{\pi}}(j, t+1) \right)$$

(sec-antibiotic-steering)=
## Example: Steering Antibiotic Resistance

Can a sequence of drugs push a bacterial population away from its most
resistant form, and how much does it help to look at the population before
choosing each drug?

Many bacteria resist penicillin-like drugs (the β-lactams) by producing an
enzyme, β-lactamase, that breaks the drug down. A few point mutations in the
enzyme's gene change which drugs it can break down.
{cite:t}`Mira2015` built all 16 combinations of four such mutations
(M69L, E104K, G238S and N276D) in the TEM enzyme of *E. coli* and measured the
growth rate of each strain under 15 β-lactams. Each drug therefore defines a
*fitness landscape*: a growth rate for every genotype. The landscapes disagree.
A genotype that grows fastest under one drug can be among the slowest under
another, so a drug given now changes where evolution goes under the next one.
{cite:t}`Nichol2015` used this to *steer* evolution: give one to three
steering drugs before the final treatment, chosen so that the population is
unlikely to end at the final drug's most resistant genotype. The problem is a
finite-horizon MDP small enough to solve exactly by backward induction.

**States.** A genotype is a string $s\in\{0,1\}^4$, where a 1 marks the
presence of a mutation, so $|\mathcal S|=16$. The wild type is $0000$. The
model assumes that the population is dominated by a single genotype at all
times, which is the usual regime when mutations are rare and selection is
strong.

**Actions.** At each of the steering epochs $t=1,\ldots,N-1$ the action is
one of the 15 drugs. Epoch $N$ applies a fixed final drug $d^\star$.

**Transitions.** Let $f_a(s)$ be the measured growth rate of genotype $s$
under drug $a$, and let $U_a(s)$ be the set of genotypes that differ from $s$
in one mutation and grow faster. Under drug $a$, a single fitter mutant takes
over the population with each substitution, and every fitter neighbour is
equally likely to be the one that does:

$$
M_a(j\mid s)=
\begin{cases}
1/|U_a(s)| & j\in U_a(s),\\
1 & j=s \text{ and } U_a(s)=\emptyset,\\
0 & \text{otherwise.}
\end{cases}
$$

A genotype with no fitter neighbour is a local peak of the landscape and
absorbs the chain. Because each substitution strictly increases the growth
rate, every path reaches a peak within 15 substitutions. A drug is given long
enough for this to happen, so the transition kernel of one decision epoch is
the limit of the chain,
$p(j\mid s,a)=\big[\lim_{m\to\infty}M_a^m\big]_{sj}=[M_a^{16}]_{sj}$.

**Objective.** Let $s^\star$ be the genotype with the highest growth rate
under $d^\star$. There is no stage cost. The terminal cost is the probability
that the final drug carries genotype $s$ to $s^\star$,
$c_N(s)=p(s^\star\mid s,d^\star)$, and the goal is to minimize its expected
value. Backward induction ({prf:ref}`backward-induction`, with a minimum in
place of the maximum) reads

$$
v_N(s)=c_N(s),\qquad
v_t(s)=\min_{a}\sum_{j}p(j\mid s,a)\,v_{t+1}(j),\qquad t=N-1,\ldots,1 .
$$

The resulting policy is Markov deterministic: one drug for each pair of epoch
and genotype. Using it assumes that the dominant genotype is sequenced before
each drug. {cite:t}`Nichol2015` assume instead that nothing is observed and
that the initial genotype is uniformly distributed, the worst case in which
any resistant strain may already be present. Their decision rules ignore the
state, so a policy reduces to a fixed sequence of drugs. Both are policies in
the sense of the [decision rules section](#decision-rules-and-policies), and
since a fixed sequence is a Markov policy that happens to be constant in $s$,
backward induction can only do as well or better. The two classes also differ
in cost. The best fixed sequence of $k$ steering drugs is found by checking
all $15^k$ sequences, while backward induction costs $k$ products of 15
transition matrices with a 16-vector.

:::{iframe} ../interactive/antibiotic-steering.html
:label: fig-antibiotic-steering
:width: 100%
:title: Steering antibiotic resistance: fixed sequences and the backward-induction policy
:class: antibiotic-steering-replay
:placeholder: _static/antibiotic_steering/steering.svg

Steering evolution before a final course of ampicillin (AMP), with one
steering drug and every starting genotype equally likely. Nodes are the 16
genotypes, from the wild type on the left to the quadruple mutant on the
right, and edges join genotypes one mutation apart. Node color is the growth
rate under the drug acting on that genotype, and the blue disc has area
proportional to the probability of the genotype. The red ring marks $1111$,
the most resistant genotype under ampicillin. Top: the best fixed steering
drug, cefprozil (CPR). Bottom: the backward-induction policy, which gives each
genotype the drug written next to it. In the online book the figure replays
each substitution, and the reader can change the final drug, the number of
steering drugs, the starting population and the strategy.
:::

Without steering, ampicillin carries 62% of the starting genotypes to $1111$.
The best single steering drug, cefprozil, reduces this to 25%, the value
reported by {cite:t}`Nichol2015`. If the genotype is observed first, the
backward-induction policy reaches 11% with the same number of drugs
({numref}`fig-antibiotic-steering`). It uses four different drugs, depending
on the genotype. Cefprozil can leave a population at $0101$, one of its
peaks, from which ampicillin climbs straight to $1111$. Most genotypes from
which that can happen receive cefotaxime (CTX) or ceftazidime (CAZ) instead.

With two steering drugs, the fixed sequence "ampicillin, then cefprozil"
already reaches zero, and feedback has nothing left to add. Giving the final
drug first sends every genotype to one of ampicillin's three peaks, $0011$,
$0110$ and $1111$. Cefprozil then moves $1111$ to $0110$ and leaves the other
two in place. When ampicillin is given again, the population sits on one of
its two lower peaks and stays there. The first drug has made the state
predictable without measuring it, which is why a sequence that ignores the
state can do as well as one that observes it. The table below shows that this
is the exception. For most final drugs a gap remains after three steering
drugs; for cefepime (FEP) the best fixed sequence leaves a probability of
0.33 and the feedback policy 0.14.

:::{include} artifacts/antibiotic_steering/results.md
:::

Two rows show the limits of steering. Ampicillin with sulbactam (SAM) has a
single peak at $1111$, so every population ends there whatever comes before.
For ceftizoxime (ZOX) and piperacillin with tazobactam (TZP), three steering
drugs remove only a fraction of the risk, with or without feedback. Another difference between the two classes does not
show in the table. The backward-induction policy is computed once and applies
to every starting distribution, while the best fixed sequence must be searched
again whenever the prior over the starting genotype changes. Choosing the wild
type as the starting population in the replay makes cefotaxime the best single
steering drug. With one steering drug and a known starting genotype, there is
nothing to observe before the only decision, so the fixed sequence and the
policy give the same probability, 0.15.

The model is a deliberate idealization. Real populations hold several
genotypes at once, drugs are given for fixed durations rather than until
evolution stops, and growth rate in a laboratory culture is only a proxy for
fitness in a patient. The transition matrices and the measured landscapes are
nevertheless real data, and the example shows the two features that matter
for the rest of the chapter: the optimal decision depends on the state, and
backward induction finds it at a cost that grows linearly with the horizon.

- {download}`Antibiotic-steering model and solver <code/antibiotic_steering.py>`
- {download}`Antibiotic-steering experiment builder <scripts/build_antibiotic_steering_artifacts.py>`
- {download}`Antibiotic-steering results for every final drug <artifacts/antibiotic_steering/metrics.json>`

(sec-storm-surge-barrier)=
## Example: Closing a Storm-Surge Barrier

How long should the operator of a storm-surge barrier wait for a better
forecast before ordering the barrier closed?

The Maeslant barrier (Maeslantkering) protects Rotterdam from storm surges
coming up the Nieuwe Waterweg, the shipping channel that connects the port to
the North Sea. It consists of two gates, each 210 m wide, that rest on the banks.
When the forecast water level at Rotterdam exceeds 300 cm above NAP, the
Dutch reference level, the gates are moved into the channel, filled with
water and sunk to the bottom, which takes about two hours
{cite:p}`RWSMaeslant,TraceKleeberg2025`. While the barrier is closed, the
main shipping route to the port is blocked. Venice faces the same
trade-off with its MOSE barriers, where false closures delay ships and missed
closures let the city flood {cite:p}`Umgiesser2021`. The operator decides on
the basis of a forecast of the peak water level, and this forecast becomes
more accurate as the peak approaches: the uncertainty of forecast surge maxima
grows with the forecast range {cite:p}`Umgiesser2021`. Waiting therefore buys
information. It also uses up the time that the closing procedure needs. The
only decision is when to stop waiting, which makes this an *optimal stopping*
problem, a finite-horizon MDP in which one action ends the decision process.

We build a small model of this decision. Two of its ingredients are
published: the closing criterion of 300 cm above NAP, which we also use as the
level above which the city floods, and the failure probability of 1 in 100
per closure that the Port of Rotterdam gives for the barrier
{cite:p}`PortRotterdamFloodRisk`. Every other number below is a teaching value.

**Decision epochs.** We count time by the number of hours $h$ left before the
forecast peak. A storm is announced at $h=H=24$, the operator decides once per
hour at $h=24,23,\ldots,1$, and the peak occurs at $h=0$.

**States.** The state is $s=(f,b)$. The first component is the current
forecast $f$ of the peak water level at Rotterdam, in cm above NAP, on a grid
of 1 cm from 150 cm to 420 cm. The second is the status $b$ of the barrier:
open; ships warned for 1, 2 or 3 hours; gates moving in the first or second
hour of the closing operation; delayed by a fault; or closed. This gives 271
forecasts and 9 statuses.

**Actions.** When the barrier is open, the action set is
$\mathcal A_{(f,\text{open})}=\{\text{wait},\text{close}\}$. Once the closure is
ordered it cannot be cancelled, so every other status has the single action
wait.

**Transitions.** The forecast and the barrier evolve independently,
$p(j\mid s,a)=K(f'\mid f)\,P(b'\mid b,a)$ for $j=(f',b')$. The forecast moves
by a Gaussian increment each hour,

$$
f' = f + w,\qquad w\sim\mathcal N(0,\sigma^2),\qquad \sigma = 4\ \text{cm},
$$

rounded to the grid, so that $K$ is a $271\times 271$ stochastic matrix. The
forecast at $h=0$ is the realized peak $L$. Because the increments have mean
zero, the forecast is a martingale: the best estimate of the peak at lead $h$
is the current forecast, and the remaining error has standard deviation
$\sigma\sqrt h$, which is 20 cm a day ahead and 10 cm six hours ahead. The
status follows a fixed procedure once the closure is ordered. Ships are warned
and traffic stops; four hours after the order the gates start to move; two
hours later they are closed with probability $1-q$, and with probability
$q=0.1$ a fault delays them. A delayed closure is completed with probability
$r=0.7$ in each following hour. A closure ordered at lead $h$ is therefore
not complete at the peak with probability

$$
\rho(h)=\begin{cases}1 & h<6,\\ q\,(1-r)^{h-6} & h\ge 6,\end{cases}
$$

which is 0.1 for an order 6 hours ahead and 0.009 for an order 8 hours ahead,
close to the published 1 in 100.

**Costs.** The unit of cost is one hour of halted shipping. Every hour after
the order costs 1, whether the gates are moving or closed, and shipping stays
halted for $R=6$ more hours after the peak while the water falls and the gates
are reopened. If the barrier is not closed when the peak $L$ exceeds 300 cm,
the city floods at a cost $C$, which we set to 1000 hours of halted shipping.
The stage and terminal costs are

$$
c(s,a)=\mathbb 1\{b\ne\text{open or }a=\text{close}\},\qquad
c_0(f,b)=R\,\mathbb 1\{b\ne\text{open}\}+C\,\mathbb 1\{b\ne\text{closed}\}\,\mathbb 1\{f>300\}.
$$

**Solution.** Backward induction ({prf:ref}`backward-induction`, with a
minimum in place of the maximum) starts from the peak and moves away from it:

$$
v_0(s)=c_0(s),\qquad
v_h(s)=\min_{a\in\mathcal A_s}\Big\{c(s,a)+\sum_{j}p(j\mid s,a)\,v_{h-1}(j)\Big\},\qquad h=1,\ldots,H .
$$

The solver applies this recursion to all $271\times 9$ states at each of the
24 epochs, which takes a fraction of a second. Only the open status involves a
choice, and there the recursion takes the classical form of optimal stopping.
After the order, the cost no longer depends on anything the operator does. It
is $h+R$ hours of halted shipping, plus the flood cost if the closure is late
and the peak is high. Writing $\pi_h(f)$ for the probability that the peak
exceeds 300 cm given the forecast $f$ at lead $h$, the cost of stopping and
the value of the open barrier are

$$
g_h(f)=h+R+C\,\rho(h)\,\pi_h(f),\qquad
v_h(f,\text{open})=\min\Big\{g_h(f),\ \sum_{f'}K(f'\mid f)\,v_{h-1}(f',\text{open})\Big\}.
$$

The policy orders the closure at the first hour at which stopping costs less
than continuing. The set of states where this happens is the *stopping region*,
and for this model it has a simple shape at every lead: the closure is ordered
when the forecast is at or above a level that depends on $h$. We compare the
policy with two rules of the kind an operating manual might contain. The first
applies the closing criterion as soon as it is reached: order the closure at
the first hour, between 24 and 6 hours ahead, at which the forecast is at
least 300 cm. The second is the best rule of the form "order the closure once
the forecast reaches a level $\bar f$, between 6 and $\bar h$ hours ahead",
found by trying every level from 260 cm to 310 cm and every latest lead
$\bar h$. All policies are evaluated exactly with
{prf:ref}`backward-policy-evaluation` over storms whose forecast 24 hours ahead
is normal with mean 280 cm and standard deviation 20 cm; 15.3% of these storms
exceed 300 cm.

:::{iframe} ../interactive/storm-surge-barrier.html
:label: fig-storm-surge-barrier
:width: 100%
:title: Closing a storm-surge barrier: the optimal stopping region and threshold rules
:class: storm-surge-barrier-replay
:placeholder: _static/storm_surge_barrier/stopping_region.svg

Closing a storm-surge barrier with a flood cost of 1000 hours of halted
shipping. (a) The shaded cells are the stopping region of the
backward-induction policy: at each lead, the forecasts for which it orders the
closure. The dashed box is the best threshold rule, which orders the closure
when the forecast reaches 282 cm between 9 and 6 hours before the peak. The
dotted line is the 300 cm closing criterion, and the gray band marks the last
6 hours, when a closure can no longer be completed. Three sampled storms are
drawn with their forecasts; a dot marks the hour at which the policy orders
the closure, and a square the realized peak. (b) Probability of a false
closure against probability of a missed flood, for flood costs from 200 to
5000 hours of shipping, for the policy and for the best threshold rule at each
cost. The diamond applies the 300 cm criterion as soon as it is reached. In
the online book the figure replays sampled storms with a cross-section of the
channel, the forecast with its narrowing uncertainty, and the current state
moving across the stopping region.
:::

The policy never orders the closure more than 10 hours before the peak
({numref}`fig-storm-surge-barrier`). Ten hours ahead, a closure fails to
finish in time with probability $\rho(10)=0.0008$, and waiting one more hour
raises this to $\rho(9)=0.0027$. With a flood cost of 1000, that extra risk is
worth one hour of shipping only when the flood is more likely than not, and
the policy indeed closes at 10 hours only when the forecast is at least
302 cm, where the probability of exceeding 300 cm is 0.55. As the deadline
approaches, each hour of waiting adds more risk, and the closing level falls:
290 cm at 9 hours, 285 cm at 8 hours, 282 cm at 7 hours and 279 cm at
6 hours. Six hours ahead is the last chance. Closing then costs 12 hours of
shipping and removes 90% of the expected flood cost, so the policy closes as
soon as the probability of exceeding 300 cm reaches about 1.3%. At 279 cm it
is 1.4%.

:::{include} artifacts/storm_surge_barrier/results.md
:::

The closing criterion used as soon as it is reached does badly in this model,
with an expected cost of 30.9 hours against 9.1 for the optimal policy. It
closes less often than the optimal policy, but it orders closures up to
24 hours ahead and so halts shipping for 9.1 hours on average against 7.9.
It also misses floods. A storm whose forecast first reaches 300 cm in the last
6 hours comes too late for it, and a closure ordered at the last moment fails
whenever a fault delays it. Together these misses occur in 2.17% of storms, or
14% of the storms that exceed 300 cm. The optimal
policy closes for forecasts well below 300 cm near the deadline, which costs
false closures in 30.3% of storms but leaves a missed flood in only 0.12%.
Most of this gain comes from the shape of the rule rather than from the slope
of the boundary. The best threshold rule, a level of 282 cm applied between
9 and 6 hours ahead, costs 9.4, about 4% more than the optimum. The flood cost
sets the balance between the two kinds of error. Raising it from 200 to 5000
moves the policy from 19.7% false closures and 0.60% missed floods to 38.7%
and 0.03%, and the stopping region grows: at a flood cost of 5000 the policy
closes from 11 hours ahead, at 296 cm. In panel (b) the two curves nearly
coincide: once a rule has the right window and level, letting the level
change from hour to hour gains little.

The model leaves out much of what matters at the real barrier. The forecast
error is a Gaussian random walk with a constant variance per hour. Real
forecast errors are larger for larger surges; the probabilistic model used by
Rijkswaterstaat for maintenance planning assumes a standard deviation of
10 cm plus a tenth of the surge above 50 cm {cite:p}`TraceKleeberg2025`. The time of the peak is assumed
known, the river discharge that the 290 cm criterion at Dordrecht protects
against is ignored, the closure cannot be cancelled, and the costs, the
closing delays and the distribution of storms are teaching values rather than
data. The real barrier closes automatically when the forecast reaches the
criterion, and it was first closed for a storm in 2007
{cite:p}`RWSMaeslant`. What the example does show is the
structure of an optimal stopping problem. Information keeps improving, the
deadline is fixed, and the right threshold depends on how much time is left.
Backward induction finds that threshold for every lead at once.

- {download}`Storm-surge-barrier model and solver <code/storm_surge_barrier.py>`
- {download}`Storm-surge-barrier experiment builder <scripts/build_storm_surge_barrier_artifacts.py>`
- {download}`Storm-surge-barrier results for every flood cost <artifacts/storm_surge_barrier/metrics.json>`

## Summary and Outlook

Stochastic dynamic programming separates decision rules by the information and
randomization they use, then evaluates each action by averaging its continuation
value over the transition kernel. Finite-horizon optimality remains a backward
recursion because the terminal date supplies the boundary condition.

For continuous-time systems driven by Gaussian disturbances, the same
short-time recursion gives the Hamilton--Jacobi--Bellman equation.
[Path-integral stochastic control](path-integral-stochastic-control.md)
examines a class of such systems for which optimal feedback can be
estimated from weighted trajectories.

What replaces that boundary when decisions continue indefinitely? [The
infinite-horizon formulation](infinite-horizon-mdps.md) uses discounting to
obtain bounded value functions and fixed-point equations whose solutions no
longer depend on a terminal time.
