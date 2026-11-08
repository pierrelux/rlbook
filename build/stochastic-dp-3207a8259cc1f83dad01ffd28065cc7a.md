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

While SDP provides us with a framework to for handling uncertainty, it makes the curse of dimensionality even more difficult to handle in practice. Both the state space and the disturbance space must be discretized. This can lead to a combinatorial explosion in the number of scenarios to be evaluated at each stage.

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
those weeks. That plan is only as good as the forecast: a week wetter than
forecast finds the reservoir full and sends the difference over the spillway.
Stochastic dynamic programming has long been used to compute reservoir
operating policies that account for this risk {cite:p}`Stedinger1984`. We keep
every element of the deterministic model (52 weekly stages, the storage and
release limits, the head $h(x)$, the price $p_t$ and the terminal value) and
make the inflow random.

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
the successors $x + 0.6\,\bar w_t - u$ and $x + 1.4\,\bar w_t - u$ are in general
no longer multiples of the 10 hm³ grid step, so $J_{t+1}^\star$ is evaluated between grid
points by linear interpolation, as in [the interpolation
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

On the same inflows, the stochastic policy earns 0.69 M\$ more per year than
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
:caption: Both policies operated on the same 2000 simulated years from $x_0 = 1100$ hm³. Left, storage from August to December. The certainty-equivalent policy (orange) holds the reservoir at capacity and spills whenever a week is wetter than forecast; the stochastic policy (blue) holds it about 50 hm³ below. Right, the revenue of the stochastic policy minus that of the certainty-equivalent policy in each simulated year.
:tags: [remove-input]

make_stochastic_comparison_figure(storage_ce, storage_sto, revenue_ce,
                                  revenue_sto, x_max)
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
derivative of the bracket with respect to $u$ is

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
the reservoir. The deterministic policy knows the inflow and can fill the
reservoir to capacity without spilling.

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

Rather than expressing the stochasticity in our system through a disturbance term as a parameter to a deterministic difference equation, we often work with an alternative representation (more common in operations research) which uses the Markov Decision Process formulation. The idea is that when we model our system in this way with the disturbance term being drawn indepently of the previous stages, the induced trajectory are those of a Markov chain. Hence, we can re-cast our control problem in that language, leading to the so-called Markov Decision Process framework in which we express the system dynamics in terms of transition probabilities rather than explicit state equations. In this framework, we express the probability that the system is in a given state using the transition probability function:

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


For a system with deterministic dynamics and no disturbance, the transition probabilities become much simpler and be expressed using the indicator function. Given a deterministic system with dynamics:

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

Combined together, these elemetns specify a Markov decision process, which is fully described by the tuple:

$$\{T, S, \mathcal{A}_s, p_t(\cdot | s, a), r_t(s, a)\}$$

where $\mathrm{T}$ represents the set of decision epochs (the horizon). 

## What is an Optimal Policy?

Against which competing policy class and initial states must a policy be
compared before it can be called optimal?

Let's go back to the starting point and define what it means for a policy to be optimal in a Markov Decision Problem. For this, we will be considering different possible search spaces (policy classes) and compare policies based on the ordering of their value from any possible start state. The value of a policy $\boldsymbol{\pi}$ (optimal or not) is defined as the expected total reward obtained by following that policy from a given starting state. Formally, for a finite-horizon MDP with $N$ decision epochs, we define the value function $v^{\boldsymbol{\pi}}(s, t)$ as:

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

The maximum value can be achieved by searching over the space of deterministic Markovian Policies. Consequently:

$$ v^*(s) = \max_{\boldsymbol{\pi} \in \Pi^{\mathrm{HR}}} v^{\boldsymbol{\pi}}(s) = \max _{\boldsymbol{\pi} \in \Pi^{M D}} v^{\boldsymbol{\pi}}(s), \quad s \in S$$

This equality significantly simplifies the computational complexity of our algorithms, as the search problem can now be decomposed into $N$ sub-problems in which we only have to search over the set of possible actions. This is the backward induction algorithm, which we present a second time, but departing this time from the control-theoretic notation and using the MDP formalism:  

````{prf:algorithm} Backward Induction
:label: backward-induction

**Input:** State space $S$, Action space $A$, Transition probabilities $p_t$, Reward function $r_t$, Time horizon $N$

**Output:** Optimal value functions $v^*$

1. Initialize:
   - Set $t = N$
   - For all $s_N \in S$:

     $$v^*(s_N, N) = r_N(s_N)$$

2. For $t = N-1$ to $1$:
   - For each $s_t \in S$:
     a. Compute the optimal value function:

        $$v^*(s_t, t) = \max_{a \in A_{s_t}} \left\{r_t(s_t, a) + \sum_{j \in S} p_t(j | s_t, a) v^*(j, t+1)\right\}$$
     
     b. Determine the set of optimal actions:

        $$A_{s_t,t}^* = \arg\max_{a \in A_{s_t}} \left\{r_t(s_t, a) + \sum_{j \in S} p_t(j | s_t, a) v^*(j, t+1)\right\}$$

3. Return the optimal value functions $u_t^*$ and optimal action sets $A_{s_t,t}^*$ for all $t$ and $s_t$
````

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
   - For all $s_N \in S$:

     $$v^{\boldsymbol{\pi}}(s_N, N) = r_N(s_N)$$

2. For $t = N-1$ to $1$:
   - For each $s_t \in S$:
     a. Compute the value function for the given policy:

        $$v^{\boldsymbol{\pi}}(s_t, t) = r_t(s_t, \pi_t(s_t)) + \sum_{j \in S} p_t(j | s_t, \pi_t(s_t)) v^{\boldsymbol{\pi}}(j, t+1)$$

3. Return the value function $v^{\boldsymbol{\pi}}(s_t, t)$ for all $t$ and $s_t$
````

This code could also finally be adapted to support randomized policies using:

$$v^{\boldsymbol{\pi}}(s_t, t) = \sum_{a_t \in \mathcal{A}_{s_t}} \pi_t(a_t \mid s_t) \left( r_t(s_t, a_t) + \sum_{j \in S} p_t(j | s_t, a_t) v^{\boldsymbol{\pi}}(j, t+1) \right)$$

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
