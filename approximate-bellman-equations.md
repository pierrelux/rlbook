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
# Approximate Bellman Equations

The [previous chapter](weighted-residual-methods.md) replaced the value
function by a parametric function $v(\cdot;\theta)$ and asked which finite set
of conditions should determine $\theta$. This chapter asks how to compute
$\theta$, and when the result can be trusted. The answer that scales is also
the simplest. Run value iteration, but after each application of the Bellman
operator, fit the parametric function to the result. Each iteration is then an
ordinary regression problem whose targets are Bellman backups, and any
regressor can do the fitting: a grid interpolant, a tree ensemble, a neural
network. This is **fitted value iteration**, and fitted Q-iteration, NFQ, DQN
and NFQCA in the chapters ahead are elaborations of it.

The fit is not free. Value iteration converges because $\Bellman$ is a
contraction in the sup norm ({prf:ref}`value-iteration-convergence`), and the
fit can undo that. The chapter therefore follows the algorithm with three
results. If every fit is accurate to $\varepsilon$, the errors do not
accumulate beyond $\varepsilon/(1-\gamma)$, which is what licenses an
arbitrary regressor. Least squares can nonetheless make the errors grow
without bound, on a two-state problem whose value function it represents
exactly. And two kinds of fit are provably safe: averagers, which include grid
interpolation and the tree ensembles of the first fitted Q-iteration
algorithm, and least squares weighted by the state distribution of the policy
being evaluated. Neural networks belong to neither kind. What keeps neural
fitted Q-iteration stable in practice is a set of design choices that the
[fitted Q-iteration chapter](fitted-q-iteration.md) examines. We then turn to
the other solver of the previous chapter, Newton's method, which becomes
approximate policy iteration and, for linear features, the LSTD and LSPI
algorithms. The chapter closes with the steps that separate fitted value
iteration from NFQCA, and the chapter in which each is taken.

Two examples use real data. The model with which the U.S. Fish and Wildlife
Service sets duck-hunting regulations is small enough to solve exactly, so the
error bounds can be checked against what actually happens. The collision
warnings that ESA logged for its satellites come with no model at all, and
fitted Q-iteration learns from them when to maneuver.

::::{admonition} Learning Goals
:class: note

After completing this chapter, you should be able to:
- Write fitted value iteration for an arbitrary regressor, and recognize collocation and Galerkin projection as two choices of the fitting rule
- Bound the error of fitted value iteration, and of its greedy policy, by the error of each fit
- Construct a fitted value iteration that diverges, and state two conditions that prevent divergence: averagers in the sup norm, and on-policy weighting for policy evaluation
- Explain approximate policy iteration as Newton's method on the approximate Bellman equation, and LSTD and LSPI as its linear case
- Name the steps from fitted value iteration to FQI, NFQ and NFQCA, and the chapter in which each is taken
- Run fitted Q-iteration on a log of transitions, and scale its targets so that a few rare, costly events do not dominate the fit

**Prerequisites:** the weighted-residual template ([previous chapter](weighted-residual-methods.md)), the contraction and Newton-Kantorovich material of [infinite-horizon MDPs](infinite-horizon-mdps.md), interpolation with cardinal functions in [finite-horizon DP](finite-horizon-dp.md).
::::

(sec-abe-fvi)=
## Fitted Value Iteration

What does value iteration become when the value function is stored as a
parametric function rather than as a table?

Value iteration computes $\Bellman v_k$ at every state and stores the result.
With a parametric function we can do neither, but we can do two things that
together play the same role. We can evaluate the Bellman operator at finitely
many states $s_1, \ldots, s_m$,

$$
y_i = (\Bellman v_k)(s_i) = \max_{a \in \mathcal{A}_{s_i}} \Big\{ r(s_i,a) + \gamma \sum_{s' \in \mathcal{S}} p(s'|s_i,a)\, v(s';\theta_k) \Big\},
$$

and we can choose $\theta_{k+1}$ so that $v(\cdot;\theta_{k+1})$ fits the pairs
$(s_i, y_i)$. The first step needs the model and the current function. The
second is a supervised learning problem that knows nothing about Bellman
equations.

```{prf:algorithm} Fitted Value Iteration
:label: fitted-value-iteration

**Inputs:** States $\{s_i\}_{i=1}^m$ at which to compute targets, discount factor $\gamma$, function class $\{v(\cdot;\theta)\}$, fitting procedure $\mathtt{fit}$, tolerance $\epsilon$

**Output:** Approximate value function $v(\cdot;\theta) \approx v^*$

1. Initialize $\theta_0$ arbitrarily
2. Set $k \leftarrow 0$
3. **repeat**
4. $\quad$ **for** each state $s_i$ **do**
5. $\quad\quad$ Compute target: $y_i \leftarrow \displaystyle\max_{a \in \mathcal{A}_{s_i}} \Big\{ r(s_i, a) + \gamma \sum_{s'} p(s' \mid s_i, a)\, v(s';\theta_k) \Big\}$
6. $\quad$ **end for**
7. $\quad$ Fit: $\theta_{k+1} \leftarrow \mathtt{fit}\big(\{(s_i, y_i)\}_{i=1}^m\big)$
8. $\quad$ $k \leftarrow k+1$
9. **until** $\|v(\cdot;\theta_k) - v(\cdot;\theta_{k-1})\| < \epsilon$ (or maximum iterations reached)
10. **return** $\theta_k$
```

Write $\Proj$ for the map that takes a function, through its values at
$s_1, \ldots, s_m$, to the fitted function. Then fitted value iteration is

$$
v_{k+1} = \Proj \Bellman v_k,
$$ (eq:projected-vi)

value iteration with a fit after each sweep. Two choices define the algorithm:
where the targets are computed, and how the function is fitted to them.
Nothing depends on the max. Replacing $\Bellman$ by the policy evaluation
operator $\BellmanPi$ gives fitted policy evaluation, and replacing it by the
smooth operator $\mathrm{L}_\beta$ of {ref}`sec-regularized-operators` gives
fitted smooth value iteration. The weighted-residual methods of the previous
chapter are two particular fitting rules.

### Collocation and Galerkin as Fitting Rules

**Collocation** computes targets at exactly $n$ states, as many as there are
basis functions, and interpolates them: with $\boldsymbol{\Phi}$ the $n \times n$
matrix with entries $\Phi_{ij} = \varphi_j(s_i)$, the fit solves
$\boldsymbol{\Phi}\theta_{k+1} = y$. Here $\Proj$ is the interpolation
operator. With piecewise-linear cardinal functions on a grid, this is the
stationary version of the interpolated backward recursion
{eq}`eq:interpolated-bellman` of the finite-horizon chapter.

```{prf:algorithm} Collocation with Function Iteration
:label: collocation-function-iteration

**Input** Collocation points $\{s_1, \ldots, s_n\}$, basis functions $\{\varphi_1, \ldots, \varphi_n\}$, initial $\theta^{(0)}$, tolerance $\varepsilon > 0$

**Output** Converged coefficients $\theta^*$

1. Form collocation matrix $\boldsymbol{\Phi}$ with $\Phi_{ij} = \varphi_j(s_i)$
2. $k \leftarrow 0$
3. **repeat**
    1. For each $i = 1, \ldots, n$:
        1. $t_i^{(k)} \leftarrow \max_{a \in \mathcal{A}_{s_i}} \left\{ r(s_i, a) + \gamma \sum_{s' \in \mathcal{S}} p(s'|s_i, a) \sum_{\ell=1}^n \theta_\ell^{(k)} \varphi_\ell(s') \right\}$
    2. Solve $\boldsymbol{\Phi} \theta^{(k+1)} = t^{(k)}$
    3. $k \leftarrow k + 1$
4. **until** $\|\theta^{(k)} - \theta^{(k-1)}\| < \varepsilon$
5. **return** $\theta^{(k)}$
```

The fishery of {ref}`sec-wr-fishery` runs this algorithm with the hat
functions and with Chebyshev polynomials. The maximization over the
escapement and the quadrature over the shock are done by the module that
produced the figures of the previous chapter. The cell prints how much the
fitted values at the nodes change from one iteration to the next.

```{code-cell} python
:tags: [hide-input]

import sys
from pathlib import Path

import numpy as np

code_directory = Path.cwd() / "code"
if str(code_directory) not in sys.path:
    sys.path.insert(0, str(code_directory))
from fishery_tipping_point import Bellman, ChebyshevBasis, HatBasis, solve_exact

bellman = Bellman()                  # backups on a fine grid of escapements, 21 quadrature nodes
exact = solve_exact(bellman)         # reference solution


def collocation(basis, max_iter=2000):
    theta = np.zeros(basis.n)
    steps = []
    for k in range(max_iter):
        v = basis.function(theta)
        targets = bellman.apply(v, basis.nodes)        # t_i = (L v_theta)(s_i)
        new = basis.fit(targets)                       # solve Phi theta = t
        steps.append(np.abs(basis.function(new)(basis.nodes) - v(basis.nodes)).max())
        theta = new
        if steps[-1] < 1e-10 or steps[-1] > 1e8:
            break
    return basis.function(theta), np.array(steps)


print(f"{'basis':<30}{'step 10':>10}{'step 100':>11}{'last step':>11}"
      f"{'iterations':>12}{'rate':>7}{'max error':>11}")
for basis in (HatBasis(12), ChebyshevBasis(12), ChebyshevBasis(10)):
    v, steps = collocation(basis)
    rate = np.median(steps[50:150] / steps[49:149])
    error = np.abs(v(bellman.x) - exact.v).max()
    label = f"{basis.name}, n = {basis.n}"
    print(f"{label:<30}{steps[9]:>10.2e}{steps[99]:>11.2e}{steps[-1]:>11.2e}"
          f"{len(steps):>12}{rate:>7.3f}{error:>11.3g}")
```

With 12 coefficients, both bases converge, and the change per iteration
shrinks by the factor $\gamma = 0.95$, as in value iteration. They converge to
different fixed points, with errors of about $0.5$ and $1.7$. With 10
Chebyshev coefficients, the change instead grows by about 5% per iteration,
and the coefficients run off to infinity. The Bellman backup still
contracts by $\gamma$. It is the interpolation step that expands, by a
factor of at least $1.049/0.95 \approx 1.10$, enough to undo the contraction.
{ref}`sec-abe-convergence` explains which fitting rules can do this and which cannot.

**Galerkin projection** fits by weighted least squares. Given a weight $w(s)$
over states, $\theta_{k+1}$ minimizes
$\int (v(s;\theta) - (\Bellman v_k)(s))^2 w(s)\,ds$, whose first-order
conditions are the Galerkin conditions
$\langle v(\cdot;\theta_{k+1}) - \Bellman v_k, \varphi_i \rangle_w = 0$, a
linear system $M\theta_{k+1} = b^{(k)}$ with mass matrix
$M_{ij} = \langle \varphi_i, \varphi_j \rangle_w$ and
$b_i^{(k)} = \langle \Bellman v_k, \varphi_i \rangle_w$. The integrals need
quadrature. If the states $s_i$ are drawn from $w$, the sample average
replaces the integral and the fit becomes an ordinary least-squares
regression on the pairs $(s_i, y_i)$. Here $\Proj$ is the $w$-orthogonal
projection onto the span of the basis.

The two rules mark the ends of a range. Collocation computes few targets and
matches them exactly; regression computes many and matches them on average.
Regression is what scales, because it decouples the number of targets from
the number of parameters and it applies unchanged to any model that can be
fitted to data.

### Fitting a Neural Network: Fixed Targets and the Semi-Gradient

For a model $v(\cdot;\boldsymbol\theta)$ that is nonlinear in its parameters,
such as a neural network, the fit is

$$
\boldsymbol{\theta}_{k+1} \in \arg\min_{\boldsymbol{\theta}} \sum_{i=1}^m \big(v(s_i;\boldsymbol{\theta}) - y_i\big)^2, \qquad y_i = (\Bellman v_k)(s_i),
$$

solved by gradient descent. Its first-order condition is again an
orthogonality condition, with test functions that move with the parameters:

$$
\sum_{i=1}^m \big(v(s_i;\boldsymbol{\theta}) - y_i\big) \frac{\partial v(s_i;\boldsymbol{\theta})}{\partial \theta_j} = 0 \quad \text{for all } j.
$$

The derivatives $\partial v/\partial\theta_j$ span the tangent space of the
model at the current parameters. For a linear model they are the basis
functions, and the condition is the sampled Galerkin condition. The targets
$y_i$ are held fixed during the fit, so the gradient differentiates only
$v(s_i;\boldsymbol\theta)$ and never the operator. This is the
**semi-gradient** of temporal difference learning. The least-squares
criterion of {ref}`sec-wr-conditions`, Bellman residual minimization, would
instead minimize $\sum_i \big(v(s_i;\boldsymbol\theta) - (\Bellman v(\cdot;\boldsymbol\theta))(s_i)\big)^2$
with the same parameters on both sides. Its test functions,
$\partial v/\partial\theta_j - \partial(\Bellman v)/\partial\theta_j$, form the
**full gradient**, and computing them requires differentiating through the
max, by the envelope theorem, or through the log-sum-exp of the
[smooth Bellman operator](regularized-dp.md). The two criteria have different
solutions. Every algorithm in the chapters ahead, from NFQ to NFQCA, fits to
fixed targets, and the [fitted Q-iteration chapter](fitted-q-iteration.md)
studies what happens when the fit is warm-started and stopped after a few
gradient steps instead of run to convergence.

(sec-abe-high-dim)=
### Example: A Stopping Problem in Eight Dimensions

The fishery has one state variable, and a grid of a few thousand nodes
solves it exactly enough to serve as a reference. To see why the choice of
fit matters, consider a state with $D$ coordinates. An agent watches an index $s \in [0,1]^D$, for instance the
prices of $D$ assets. Stopping pays the average of the coordinates,
$\bar{s} = \frac{1}{D}\sum_\ell s_\ell$, and ends the problem. Continuing pays
nothing, and each coordinate then moves toward a fresh uniform draw,
$s'_\ell = \rho s_\ell + (1-\rho)\xi_\ell$ with $\xi_\ell \sim U[0,1]$
independent and $\rho = 0.7$. The Bellman equation is

$$
v^*(s) = \max\Big\{ \bar{s},\; \gamma\, \mathbb{E}\big[v^*(\rho s + (1-\rho)\xi)\big] \Big\}, \qquad \gamma = 0.9.
$$

The average $\bar s$ is itself a Markov chain, $\bar{s}' = \rho\bar{s} + (1-\rho)\bar{\xi}$,
so $v^*$ depends on $s$ only through $\bar s$ and can be computed exactly by
one-dimensional value iteration. The algorithm does not know this. A grid
with 11 nodes per axis has $11^D$ nodes: 121 for $D = 2$, 14,641 for $D = 4$,
and $2.1 \times 10^8$ for $D = 8$, each needing an expectation over a
$D$-dimensional successor at every iteration. Fitted value iteration instead
computes targets at 2,000 states drawn uniformly from the cube, whatever $D$
is, and fits them with an ensemble of extremely randomized trees
{cite}`ernst2005tree` or with a neural network of two hidden layers. All three
methods approximate the expectation by the average over the same 32 draws of
$\xi$, so they solve the same sample-average problem; the
[next chapter](monte-carlo-bellman-estimation.md) studies this approximation.
The errors are measured against the exact $v^*$ at 5,000 fresh states.

```{code-cell} python
:label: fig-fvi-eight-dimensions
:caption: Fitted value iteration on the stopping problem with an eight-dimensional state. Each dot is a test state, placed by the average of its coordinates and the value the network assigns to it; the black curve is the exact value function, which depends on the state only through that average. The network was never told so. It was fitted at 2,000 sampled states, while a grid with 11 nodes per axis would have $2.1 \times 10^8$.
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor

gamma, rho = 0.9, 0.7       # discount, persistence of each coordinate
N, M, K = 2000, 32, 40      # sampled states, successor draws, iterations

def exact_value(D, rng):
    """v* depends on s only through the mean of its coordinates: 1-D value iteration."""
    m = np.linspace(0, 1, 1001)
    eta = rng.uniform(size=(400, D)).mean(axis=1)          # mean of the shocks
    m_next = rho * m[:, None] + (1 - rho) * eta[None, :]
    w = m.copy()
    for _ in range(500):
        w = np.maximum(m, gamma * np.interp(m_next, m, w).mean(axis=1))
    return lambda S: np.interp(S.mean(axis=1), m, w)

def fitted_value_iteration(fit, S, xi):
    """Targets max{stop, gamma * E v(s')} at the states S, then a fit."""
    D = S.shape[1]
    S_next = (rho * S[:, None, :] + (1 - rho) * xi[None, :, :]).reshape(-1, D)
    v = lambda X: np.zeros(len(X))
    for _ in range(K):
        continuation = v(S_next).reshape(len(S), -1).mean(axis=1)
        y = np.maximum(S.mean(axis=1), gamma * continuation)
        v = fit(S, y)
    return v

def fit_grid(nodes):
    def fit(S, y):            # S is the grid itself, y the values at its nodes
        D = S.shape[1]
        table = RegularGridInterpolator([nodes] * D, y.reshape([len(nodes)] * D))
        return lambda X: table(np.clip(X, 0, 1))
    return fit

def fit_trees(S, y):
    model = ExtraTreesRegressor(n_estimators=20, min_samples_leaf=3, random_state=0)
    return model.fit(S, y).predict

def fit_network(S, y):
    model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300, random_state=0)
    return model.fit(S, y).predict

rng = np.random.default_rng(0)
nodes = np.linspace(0, 1, 11)
rows = []
for D in [2, 4, 8]:
    v_star = exact_value(D, rng)
    test = rng.uniform(size=(5000, D))
    xi = rng.uniform(size=(M, D))
    S = rng.uniform(size=(N, D))
    errors = {}
    if len(nodes)**D <= 20000:
        grid = np.stack(np.meshgrid(*[nodes] * D, indexing="ij"), axis=-1).reshape(-1, D)
        v = fitted_value_iteration(fit_grid(nodes), grid, xi)
        errors["grid"] = np.abs(v(test) - v_star(test))
    for name, fit in [("trees", fit_trees), ("network", fit_network)]:
        v = fitted_value_iteration(fit, S, xi)
        errors[name] = np.abs(v(test) - v_star(test))
        if D == 8 and name == "network":
            scatter = (test.mean(axis=1), v(test), v_star(test))
    rows.append((D, len(nodes)**D, errors))

print("Mean absolute error (maximum in parentheses) against the exact value")
print(f"{'D':>2} {'grid nodes':>12}  {'grid':^16} {'trees, 2000 states':^20} {'network, 2000 states':^20}")
for D, n_nodes, e in rows:
    cell = lambda key: f"{e[key].mean():.3f} ({e[key].max():.3f})" if key in e else "out of reach"
    print(f"{D:>2} {n_nodes:>12,}  {cell('grid'):^16} {cell('trees'):^20} {cell('network'):^20}")

mean_s, v_hat, v_true = scatter
order = np.argsort(mean_s)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.scatter(mean_s, v_hat, s=3, alpha=0.3, label="network, D = 8")
ax.plot(mean_s[order], v_true[order], color="black", lw=1.5, label="exact $v^*$")
ax.set_xlabel("average of the coordinates of $s$")
ax.set_ylabel("value")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
```

The grid is the most accurate method while it fits in memory, and it is out of
reach at $D = 8$. The trees and the network use the same 2,000 states in every
dimension. The average error of the trees roughly doubles each time $D$
doubles, while that of the network stays near 0.01; the figure shows that the network
has found the one direction along which the value changes. Both have maximum errors several
times their average, at the corners of the cube, where few of the sampled
states lie. That gap is a first glimpse of the issue that dominates the
analysis below and the batch methods of the
[fitted Q-iteration chapter](fitted-q-iteration.md): a fit is controlled
where there are data, and only there.

(sec-abe-error)=
## How Fitting Errors Propagate

If every fit is only approximately right, how far from $v^*$ does fitted value
iteration end, and how good is the policy it returns?

The finite-horizon chapter answered this for backward recursion with
interpolation: with nonnegative weights, the error of the stored values is
at most the sum of the interpolation errors of the later stages
({prf:ref}`prop-interp-error-accumulation`). Over an infinite horizon the sum
becomes a geometric series.

```{prf:theorem} Error of approximate value iteration
:label: thm-approximate-vi

Let $v_0$ be bounded and let the sequence $v_k$ satisfy $\|v_{k+1} - \Bellman v_k\|_\infty \leq \varepsilon$ for every $k$. Then

$$
\|v_k - v^*\|_\infty \leq \gamma^k \|v_0 - v^*\|_\infty + \frac{\varepsilon}{1-\gamma}.
$$
```

```{prf:proof}
Since $v^* = \Bellman v^*$ and $\Bellman$ is a $\gamma$-contraction,

$$
\|v_{k+1} - v^*\|_\infty \leq \|v_{k+1} - \Bellman v_k\|_\infty + \|\Bellman v_k - \Bellman v^*\|_\infty \leq \varepsilon + \gamma \|v_k - v^*\|_\infty.
$$

Unrolling the recursion gives $\|v_k - v^*\|_\infty \leq \gamma^k\|v_0 - v^*\|_\infty + \varepsilon(1 + \gamma + \cdots + \gamma^{k-1})$.
```

The value function is an instrument. What we use is the greedy policy, and its
loss is bounded by the error of the value function it is greedy for
{cite}`SinghYee1994`.

```{prf:proposition} Loss of a greedy policy
:label: prop-greedy-loss

If $\pi$ is greedy with respect to $v$, that is $\BellmanPi v = \Bellman v$, then

$$
\|v^* - v_\pi\|_\infty \leq \frac{2\gamma}{1-\gamma}\|v - v^*\|_\infty.
$$
```

```{prf:proof}
Let $\pi^*$ be an optimal policy. Then $v^* - v_\pi = \mathrm{L}_{\pi^*} v^* - \BellmanPi v_\pi$, which we split as

$$
v^* - v_\pi = (\mathrm{L}_{\pi^*} v^* - \mathrm{L}_{\pi^*} v) + (\mathrm{L}_{\pi^*} v - \BellmanPi v) + (\BellmanPi v - \BellmanPi v^*) + (\BellmanPi v^* - \BellmanPi v_\pi).
$$

The second term is nonpositive because $\pi$ is greedy for $v$. The others are $\gamma\mathbf{P}_{\pi^*}(v^* - v)$, $\gamma\mathbf{P}_\pi(v - v^*)$ and $\gamma\mathbf{P}_\pi(v^* - v_\pi)$, and stochastic matrices do not increase the sup norm. Since $v^* - v_\pi \geq 0$, taking the maximum over states gives $\|v^* - v_\pi\|_\infty \leq 2\gamma\|v - v^*\|_\infty + \gamma\|v^* - v_\pi\|_\infty$.
```

Combining the two, the greedy policies of fitted value iteration are
eventually within $2\gamma\varepsilon/(1-\gamma)^2$ of optimal. The bound
asks nothing of the fit except accuracy: no linearity, no particular basis,
no structure at all. This is what licenses a tree ensemble or a neural
network in place of a table. It also says where the difficulty lies, in
three places that the rest of this chapter and the next ones take up.

1. **The error is measured in the sup norm, at every state.** A regression
   controls an average error over the states where targets were computed,
   and nothing at the states it never saw. The analysis of fitted value
   iteration by {cite:t}`Munos2007` and {cite:t}`MunosSzepesvari2008`
   replaces the sup norm by weighted norms. The price is a concentrability
   coefficient, which measures how far the state distributions that policies
   reach can be from the distribution of the data. It is the theoretical form
   of the coverage problem of batch and offline reinforcement learning.
2. **The fit must be achievable.** The bound needs some $\theta$ with
   $v(\cdot;\theta)$ close to $\Bellman v_k$ for every $v_k$ the algorithm
   produces. The function class must therefore be nearly closed under the
   Bellman operator, not merely able to represent $v^*$.
3. **The error must stay bounded.** The theorem assumes one $\varepsilon$ for
   all iterations. If the fit error grows with $v_k$, the bound says nothing,
   and the next section shows that this happens.

(sec-abe-mallard)=
### Example: Setting Duck-Hunting Regulations

How loose are these bounds on a problem that a government agency solves every
year? Each summer, the U.S. Fish and Wildlife Service chooses the duck-hunting
regulations for the Mississippi and Central Flyways from four packages: a
closed season, or a restrictive, moderate or liberal one, which differ in
season length and bag limit. Since 1995 the choice for mallards has come from
a policy computed by stochastic dynamic programming, a program called adaptive
harvest management {cite}`Johnson1997`. The state is two numbers from the May
survey: the breeding population $N$ of midcontinent mallards and the number
$P$ of ponds in Prairie Canada, both in millions. Ponds follow an
autoregressive process driven by the weather. The population responds to
ponds and to the harvest under four models. They disagree on whether hunting
deaths add to natural deaths or replace them, and on how strongly reproduction
falls as the population grows, and the Service weights them by how well each
has predicted past surveys. The reward is the harvest, devalued in proportion
when the population is expected to fall below the goal of 8.5 million birds,
and the season must close below 4.75 million. We take these equations and
numbers from the Service's report for the 2012 season {cite}`USFWS2012AHM`
and discount future harvests by $\gamma = 0.98$. The model module lists the
few details that the report leaves open.

The Service solves this problem the way the finite-horizon chapter solved the
reservoir: on a grid of states, with interpolation between the nodes. We do
the same with steps of 0.1 million birds and ponds, 12,231 nodes, and bilinear
interpolation. Bilinear interpolation is an averager, so value iteration on
the grid converges ({prf:ref}`santos-vigo-aguiar-stability` below), and
doubling the step changes the values by less than 0.05.
{numref}`fig-mallard-policies` compares the result with the table that the
Service published for 2012. The two policies prescribe the same package in
126 of the 150 cells of the table and differ by one package in the
others. The published policy reacts more strongly to ponds than ours.

:::{figure} _static/mallard_harvest/policies.svg
:label: fig-mallard-policies

Regulation packages as a function of May ponds and breeding population.
(a) The table published by the U.S. Fish and Wildlife Service for 2012.
(b) The greedy policy of our reference solution, on a grid with steps of 0.1.
(c) The greedy policy of fitted value iteration with a neural network
trained on 2,000 sampled states. Below the dotted line the season must close.
:::

The reference solution lets us check {prf:ref}`thm-approximate-vi` and
{prf:ref}`prop-greedy-loss` against what actually happens. We run fitted
value iteration for 400 iterations with five fits. Three are bilinear
interpolation on coarser grids, with steps of 2, 1 and 0.5. The other two are
regressors trained, at each iteration, on targets at the same 2,000 states
drawn uniformly at random: extra trees, which are averagers, and a network
with two hidden layers of 64 units, warm-started from the previous iteration.
For each fit we measure $\varepsilon$, the largest fitting error
$|v_{k+1}(s) - (\Bellman v_k)(s)|$ over the reference grid in the last
iterations, and compare both bounds with the value error and the policy loss
that we compute against the reference.

:::{include} artifacts/mallard_harvest/results.md
:::

The table supports three observations.

1. **Both bounds hold, and both are loose.** The value error is 18 to 63
   times smaller than $\varepsilon/(1-\gamma)$, and the policy loss is four to
   five orders of magnitude below $2\gamma\varepsilon/(1-\gamma)^2$. The bounds
   charge every iteration's error at its worst state and let the errors add up
   in the same direction for $1/(1-\gamma) = 50$ years. Neither happens here.
2. **The policy is much more accurate than the value function.** The
   coarsest grid has 45 nodes and a value error of 2.2
   million birds, yet it picks the reference package in 96.5% of the
   states, and its worst policy loss is 0.10 million birds, about
   a quarter of a percent of the value. A decision depends on the differences between
   the values of the packages, and much of the fitting error is shared by all
   four packages, because their successor states are close.
3. **The regressors do about as well as the coarsest grid.** With 2,000
   targets per iteration, the trees and the network have fitting errors of
   $0.73$ and $1.32$, close to the $0.79$ of the 45-node grid. The network's
   greedy policy reproduces the reference map (panel (c)): it picks the
   reference package in 97% of the states and loses at most $0.06$. The trees
   lose more, $0.30$ at worst. Their fit is piecewise constant, so it is flat
   where the value has a slope, and the greedy choice between two packages
   depends on that slope.

With two state variables the Service's grid is the right tool, and the
regressors bring nothing. They become necessary when the state has many
dimensions, as in the eight-dimensional stopping problem, or when there is no
model at all to compute targets from, as in the last example of this chapter.

- {download}`Mallard harvest model and solvers <code/mallard_harvest.py>`
- {download}`Mallard harvest experiment builder <scripts/build_mallard_harvest_artifacts.py>`
- {download}`Mallard harvest results <artifacts/mallard_harvest/metrics.json>`

(sec-abe-convergence)=
## When Fitting Diverges, and Which Fits Cannot

Can a fit that represents the solution exactly still make value iteration
diverge, and which fitting rules rule this out?

### A Two-State Counterexample

The following example is due to {cite:t}`TsitsiklisVanRoy1996`. There are two
states and one action. State 1 moves to state 2, state 2 stays where it is,
and all rewards are zero, so $v^* = 0$. The value function is represented
with a single feature, $v(1;\theta) = \theta$ and $v(2;\theta) = 2\theta$,
which contains $v^*$ at $\theta = 0$. Fitted value iteration computes the
targets at both states and fits them by least squares with equal weights.
Both targets equal $\gamma v(2;\theta_k) = 2\gamma\theta_k$, and the fit
minimizes $(\theta - 2\gamma\theta_k)^2 + (2\theta - 2\gamma\theta_k)^2$,
which gives

$$
\theta_{k+1} = \frac{6\gamma}{5}\theta_k.
$$

For $\gamma > 5/6$ the iterates diverge from every $\theta_0 \neq 0$. The
Bellman operator shrinks the difference between $v_k$ and $v^*$ by $\gamma$,
and the fit then stretches it by $6/5$: the target is the constant
$2\gamma\theta_k$, and the best line through the origin overshoots it at
state 2, where it predicts $\frac{12}{5}\gamma\theta_k$. The fit error grows in
proportion to $\theta_k$, which is why {prf:ref}`thm-approximate-vi` does not
apply. {cite:t}`BoyanMoore1995` observed the same failure with polynomial
regression and neural networks on small control problems, at a time when
TD-Gammon had made neural value functions look easy.

### Averagers: Fits That Cannot Expand

The counterexample suggests asking which fitting rules cannot stretch
differences between functions. The answer was found independently in
computational economics {cite}`Judd1992,Judd1996,McGrattan1997,SantosVigoAguiar1998`,
economic dynamics {cite}`Stachurski2009`, and reinforcement learning
{cite}`Gordon1995,Gordon1999`, and the communities arrived at essentially the
same conditions.

```{prf:proposition} Monotone operators are nonexpansive (Stachurski)
:label: monotone-nonexpansive

Let $\Proj: \mathcal{V} \to \mathcal{V}$ be a linear operator on the space $\mathcal{V}$ of bounded real-valued functions on $\mathcal{S}$. If $\Proj$ satisfies:

1. **Monotonicity**: $f \leq g$ pointwise implies $\Proj f \leq \Proj g$
2. **Constant preservation**: $\Proj\mathbf{1} = \mathbf{1}$ where $\mathbf{1}$ is the constant function equal to $1$

Then $\Proj$ is nonexpansive in the sup norm: $\|\Proj f - \Proj g\|_\infty \leq \|f - g\|_\infty$ for all $f, g \in \mathcal{V}$.
```

```{prf:proof}
Let $M = \|f - g\|_\infty$. Then $-M \leq f(s) - g(s) \leq M$ for all $s$, which can be written as $g - M\mathbf{1} \leq f \leq g + M\mathbf{1}$. By monotonicity, $\Proj(g - M\mathbf{1}) \leq \Proj f \leq \Proj(g + M\mathbf{1})$. By linearity and constant preservation, $\Proj g - M\mathbf{1} \leq \Proj f \leq \Proj g + M\mathbf{1}$, which means $|\Proj f(s) - \Proj g(s)| \leq M$ for all $s$. Therefore $\|\Proj f - \Proj g\|_\infty \leq \|f - g\|_\infty$.
```

A monotone, constant-preserving operator acts like a weighted average: it
respects order and cannot amplify differences. For an interpolation operator
$(\Proj f)(s) = \sum_j \ell_j(s) f(s_j)$ with cardinal functions $\ell_j$, the
two conditions say that the weights are nonnegative and sum to one, which is
the condition $\Lambda = 1$ of the finite-horizon chapter. The least-squares
fit of the counterexample violates them: it maps the constant function
$(1, 1)$ to $(3/5, 6/5)$. Polynomial interpolation at Chebyshev nodes
preserves constants but is not monotone: its cardinal functions dip below
zero between the nodes, and its Lebesgue constant is about $2.4$ for ten
nodes. That is how the ten-coefficient polynomial of the fishery
({ref}`sec-abe-fvi`) could undo the contraction of the Bellman backup, while
the hat functions, whose cardinal functions are nonnegative, could not.

```{prf:theorem} Stability of projected value iteration (Santos-Vigo-Aguiar)
:label: santos-vigo-aguiar-stability

Let $\Bellman: \mathcal{V} \to \mathcal{V}$ be a $\gamma$-contraction on the space $\mathcal{V}$ of bounded functions with respect to the sup norm. Let $\Proj: \mathcal{V} \to \mathcal{V}$ be a linear approximation operator satisfying monotonicity and constant preservation.

Then the composed operator $\Proj \Bellman$ is a $\gamma$-contraction, and projected value iteration $v_{k+1} = \Proj \Bellman v_k$ converges globally to a unique fixed point $v_\Proj \in \text{Range}(\Proj)$ with approximation error:

$$
\|v_\Proj - v^*\|_\infty \leq \frac{1}{1-\gamma} \|\Proj v^* - v^*\|_\infty,
$$

where $v^*$ is the true value function.
```

```{prf:proof}
By {prf:ref}`monotone-nonexpansive`, $\Proj$ is nonexpansive. Since $\Bellman$ is a $\gamma$-contraction,

$$
\|\Proj \Bellman f - \Proj \Bellman g\|_\infty \leq \|\Bellman f - \Bellman g\|_\infty \leq \gamma\|f-g\|_\infty,
$$

so $\Proj \Bellman$ is a $\gamma$-contraction and {prf:ref}`banach-fixed-point` gives the unique fixed point and global convergence. For the error bound, use $v_\Proj = \Proj\Bellman v_\Proj$ and $v^* = \Bellman v^*$:

$$
\|v_\Proj - v^*\|_\infty \leq \|\Proj\Bellman v_\Proj - \Proj\Bellman v^*\|_\infty + \|\Proj \Bellman v^* - v^*\|_\infty \leq \gamma \|v_\Proj - v^*\|_\infty + \|\Proj v^* - v^*\|_\infty,
$$

and rearranging gives the claim.
```

The fixed-point error is the best-approximation error
$\|\Proj v^* - v^*\|_\infty$ amplified by $(1-\gamma)^{-1}$. For a monotone
fit the per-iteration error stays bounded, and the result is the special case
of {prf:ref}`thm-approximate-vi` in which the third difficulty above cannot
arise. For a finite state space, the monotonicity conditions describe
**averaging with nonnegative weights**, a characterization due to Gordon.

```{prf:definition} Averager (Gordon)
:label: gordon-averager

An operator $\Proj: \mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}^{|\mathcal{S}|}$ is an **averager** if $\Proj v = Wv$ where $W$ is a $|\mathcal{S}| \times |\mathcal{S}|$ stochastic matrix: $w_{ij} \geq 0$ and $\sum_j w_{ij} = 1$ for all $i$.
```

Averagers satisfy the monotonicity conditions: linearity follows from matrix
multiplication, monotonicity from nonnegativity of the entries, and constant
preservation from the row sums.

```{prf:theorem} Stability with averagers (Gordon)
:label: gordon-stability

If $\Proj$ is an averager and $\Bellman$ is the Bellman operator (a $\gamma$-contraction), then $\Proj \Bellman$ is a $\gamma$-contraction, and value iteration $v_{k+1} = \Proj \Bellman v_k$ converges to a unique fixed point.
```

The stochastic matrix gives the averaging a Markov-chain reading: $\Proj v$ is
an expected value after one random transition, and expectations preserve
order. The finite-horizon chapter made the same observation about linear
interpolation, which replaces a deterministic successor by a random grid node
with the interpolation weights as probabilities. Composing that chain with
the model's own transitions makes $\Proj\Bellman$ the Bellman operator of a
finite-state MDP, which is why it contracts.

**Which regressors are averagers?** State aggregation, $k$-nearest neighbors,
kernel smoothing with nonnegative kernels normalized to sum to one, and
multilinear interpolation on a grid all predict a convex combination of
training targets whose weights depend only on the states. So does a
regression tree, which predicts the average of the targets in a leaf, as long
as its partition of the state space does not depend on the targets.
{cite:t}`ernst2005tree` used this to give tree-based fitted Q-iteration a
convergence guarantee. Their extremely randomized trees choose splits with
the help of the targets, and freezing the tree structure after some
iterations restores the guarantee. Linear least squares, high-order
polynomial interpolation, and neural networks are not averagers.

| **Method** | **Averager?** | **Notes** |
|:-----------|:--------------|:----------|
| Piecewise linear interpolation | Yes | Barycentric weights are nonnegative and sum to one |
| Multilinear interpolation (grid) | Yes | Weights are products of one-dimensional hat functions |
| Shape-preserving splines (Schumaker) | Yes | Designed to maintain monotonicity |
| State aggregation, $k$-nearest neighbors | Yes | Equal weights within a group or neighborhood |
| Kernel smoothing (nonnegative kernels) | Yes | If the weights are normalized to sum to one |
| Regression trees and forests, fixed partition | Yes | Leaf averages; guarantee lost if splits depend on the targets |
| High-order polynomial interpolation | No | Negative weights (Runge phenomenon) |
| Least squares on an arbitrary basis | No | The counterexample above |
| Neural networks | No | No sign or normalization constraint on the fit |

The table puts the methods of this chapter in their place. Collocation with
piecewise-linear interpolation is an averager, and the folk wisdom that
linear interpolation is safe for value iteration is a theorem. Collocation
with the cubic through four Chebyshev nodes is not, and its convergence in
the worked example was not guaranteed. Galerkin projection and least squares
are not averagers. The tree ensemble in the eight-dimensional example sits on
the boundary, and the network is outside.

The solver matters too. Fitted value iteration needs $\Proj\Bellman$ to
contract. Newton's method does not; its convergence depends on the
conditioning of the Jacobian and on the starting point. This is the reason for
the hybrid strategy of {ref}`sec-wr-solve`: a few iterations with a monotone,
low-order approximation to get a starting guess, then Newton's method with a
smooth, high-order one.

```{admonition} Exercises: Divergence and Averagers
:class: hint dropdown

1. **The counterexample with an averager.** Replace the least-squares fit in the two-state example by state aggregation into a single group. Write $\Proj$ as a matrix, check that it is an averager, and show that fitted value iteration converges for every $\gamma < 1$. What is its fixed point?

2. **A non-monotone example.** Using Lagrange interpolation with 4 equally spaced nodes on $[0, 1]$, compute the interpolation weights for the point $x = 0.1$. Show that some weights are negative.

3. **Verifying monotonicity.** Consider piecewise linear interpolation on a 5-point grid. Write out the interpolation weights for a point between grid points 2 and 3. Verify that all weights are nonnegative and sum to one.

4. **Trees.** Show that a regression tree with a fixed partition is an averager. Explain why a tree that chooses its splits by the targets can change its partition from one iteration to the next, and which step of the proof of {prf:ref}`santos-vigo-aguiar-stability` this breaks.
```

### On-Policy Weighting Rescues Policy Evaluation

The counterexample has a second reading. The chain moves from state 1 to
state 2 and stays there, so its stationary distribution puts all its mass on
state 2. If the fit weights each state by that distribution, it minimizes
$(2\theta - 2\gamma\theta_k)^2$ alone, and $\theta_{k+1} = \gamma\theta_k$
converges. The equal weighting was the problem: it fits the value at a state
the process does not occupy as much as the value at the state it does.

This observation is a theorem for policy evaluation {cite}`TsitsiklisVanRoy1997`,
proved to explain the convergence of TD(0) with linear features. Take a finite
state space $\mathcal{S} = \{s_1,\ldots,s_m\}$, a feature matrix
$\boldsymbol{\Phi} \in \mathbb{R}^{m\times n}$ with rows $\boldsymbol\varphi(s)^\top$,
and a distribution $\xi$ over states with $\boldsymbol{\Xi} = \text{diag}(\xi)$.
The $\xi$-weighted inner product, norm, and least-squares projection onto the
span of the features are

$$
\langle u, v \rangle_\xi = u^\top \boldsymbol{\Xi} v, \qquad \|v\|_\xi = \sqrt{v^\top \boldsymbol{\Xi} v}, \qquad \Proj = \boldsymbol{\Phi}(\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \boldsymbol{\Xi}.
$$ (eq:xi-projection)

Fitted policy evaluation is $v_{k+1} = \Proj\BellmanPi v_k$ with
$\BellmanPi v = \mathbf{r}_\pi + \gamma\mathbf{P}_\pi v$. In reinforcement
learning the weighting is not a free choice: the states in the data are those
the process visits, and if the chain under $\pi$ is ergodic their
distribution approaches the stationary distribution $\xi$ with
$\xi^\top\mathbf{P}_\pi = \xi^\top$. Suppose that is the weighting. We show
that $\Proj$ is nonexpansive and $\BellmanPi$ is a $\gamma$-contraction, both
in $\|\cdot\|_\xi$, so that

$$
\|\Proj \BellmanPi v - \Proj \BellmanPi w\|_\xi \leq \|\BellmanPi v - \BellmanPi w\|_\xi \leq \gamma \|v - w\|_\xi.
$$

**Orthogonal projection is nonexpansive.** For any vector $v$, decompose
$v = \Proj v + (v - \Proj v)$, where $v - \Proj v$ is $\xi$-orthogonal to
the span of $\boldsymbol\Phi$. By the Pythagorean theorem,

$$
\|v\|_\xi^2 = \|\Proj v\|_\xi^2 + \|v - \Proj v\|_\xi^2 \geq \|\Proj v\|_\xi^2.
$$

No sign condition on the entries of $\Proj$ was needed. The sup-norm argument
fails for this projection; the $\xi$-norm argument does not see the negative
weights at all.

**$\BellmanPi$ contracts in $\|\cdot\|_\xi$.** Since
$\BellmanPi v - \BellmanPi w = \gamma \mathbf{P}_\pi (v - w)$, it suffices to
show $\|\mathbf{P}_\pi z\|_\xi \leq \|z\|_\xi$ for every $z$.

*Before reading on, try to show this yourself. Hint: what property of $\xi$
relates it to $\mathbf{P}_\pi$?*

The $s$-th component of $\mathbf{P}_\pi z$ is $\sum_{s'} p(s'|s,\pi(s)) z(s')$,
an average of the values $z(s')$. Since $x \mapsto x^2$ is convex, Jensen's
inequality gives
$\big[\sum_{s'} p(s'|s,\pi(s)) z(s')\big]^2 \leq \sum_{s'} p(s'|s,\pi(s)) z(s')^2$.
Weighting by $\xi$ and summing over $s$,

$$
\|\mathbf{P}_\pi z\|_\xi^2 \leq \sum_s \xi(s) \sum_{s'} p(s'|s,\pi(s)) z(s')^2 = \sum_{s'} z(s')^2 \sum_s \xi(s) p(s'|s,\pi(s)) = \sum_{s'} \xi(s') z(s')^2 = \|z\|_\xi^2,
$$

where the second-to-last step is stationarity. So $\Proj\BellmanPi$ is a
$\gamma$-contraction in $\|\cdot\|_\xi$, and by {prf:ref}`banach-fixed-point`
fitted policy evaluation converges from any start to the unique solution of
the **projected Bellman equation**

$$
v = \Proj \BellmanPi v.
$$ (eq:projected-bellman-pi)

Its solution is the fixed point to which TD(0) with linear features converges.
The argument of {prf:ref}`santos-vigo-aguiar-stability` bounds its distance to
$v_\pi$ by $(1-\gamma)^{-1}\|\Proj v_\pi - v_\pi\|_\xi$, and the inner-product
structure improves the constant to $(1-\gamma^2)^{-1/2}$
{cite}`TsitsiklisVanRoy1997`.

**The on-policy condition.** Convergence came from matching the weighting to
the operator, not from any property of the fit beyond orthogonality: $\xi$
must be the stationary distribution of the very $\mathbf{P}_\pi$ that appears
in $\BellmanPi$. When the data are collected by following $\pi$, this holds
automatically, which is **on-policy** learning. When they come from another
behavior policy, as in the counterexample's equal weighting or in learning
from a replay buffer filled by older policies, the Jensen step fails,
$\mathbf{P}_\pi$ need not be nonexpansive in $\|\cdot\|_\xi$, and
$\Proj\BellmanPi$ may not contract. Baird's counterexample {cite}`Baird1995` is
the classical demonstration.

```{admonition} Exercises: The On-Policy Condition
:class: hint dropdown

1. **Weighting the counterexample.** In the two-state example, fit with weights $(\xi_1, \xi_2)$. Show that $\theta_{k+1} = 2\gamma\theta_k(\xi_1 + 2\xi_2)/(\xi_1 + 4\xi_2)$ and find the set of weightings for which the iteration converges for every $\gamma < 1$.

2. **Non-expansiveness fails off-policy.** For the 3-state chain of {ref}`sec-abe-lstd` and the weighting $\xi' = (0.9, 0.05, 0.05)$, find a vector $z$ such that $\|\mathbf{P}_\pi z\|_{\xi'} > \|z\|_{\xi'}$.

3. **Losing the argument is not losing convergence.** Implement fitted policy evaluation for the same chain, with features $\varphi(s) = (1, s)$ and weighting $\xi'$. It still converges: compute the spectral radius of $\gamma \Proj \mathbf{P}_\pi$ and check that it equals $\gamma$ for this feature set whatever the weighting. Then keep only the feature $\varphi(s) = s$ and find a weighting concentrated on state 1 under which the spectral radius exceeds one.
```

### The Optimality Operator and the Deadly Triad

Can the weighted argument be extended to the optimality operator
$\Bellman v = \max_a [\mathbf{r}_a + \gamma \mathbf{P}_a v]$? Not this way.
The obstacle is the Jensen step. For the optimality operator we would need to
bound $\big[\max_{a} \sum_{s'} p(s'|s,a) z(s')\big]^2$, and a maximum of
averages is not an average. Nor is there a single stationary distribution to
match, since the greedy policy, and with it the state distribution, changes
from iteration to iteration.

{cite:t}`SuttonBarto2018` call the combination of function approximation,
bootstrapping (targets built from the current estimate), and off-policy data
the **deadly triad**: any two are manageable, all three can diverge. Fitted
Q-iteration with a neural network has all three. There is no general
convergence theorem for it, and divergence can be observed on small examples.
Yet NFQ and DQN solve many problems, because of design choices that limit the
damage: each fit is run against targets held fixed, the fit is warm-started
from the previous parameters, a separate target network changes slowly, and a
large replay buffer keeps the data distribution from tracking every change of
policy. The [fitted Q-iteration chapter](fitted-q-iteration.md) examines these
choices one by one.

(sec-abe-api)=
## Approximate Policy Iteration

What does Newton's method, the other solver of the previous chapter, become
when the value function is parametric, and what does the linear case add?

In the exact setting, Newton's method on the Bellman equation is policy
iteration ({ref}`sec-newton-kantorovich-pi`): by the envelope theorem the
Jacobian at $v_k$ involves the transition matrix of the greedy policy
$\pi_k$, and the Newton step solves the evaluation equation of $\pi_k$. With a
parametric value function the same holds, except that the evaluation is done
within the function class. The algorithm alternates **approximate
evaluation**, finding $\theta$ with $v(\cdot;\theta) \approx v_{\pi_k}$, and
**greedy improvement**. This is approximate policy iteration. Its guarantee
has the same form as for fitted value iteration: if every evaluation is
accurate to $\varepsilon$ in the sup norm, the policies are eventually within
$2\gamma\varepsilon/(1-\gamma)^2$ of optimal {cite}`BertsekasTsitsiklis1996`.
Unlike exact policy iteration, the sequence of policies need not improve
monotonically, and it can cycle among a few policies.

Two remarks carry over from the exact case. With the smooth Bellman operator,
the Jacobian is $\gamma\mathbf{P}_{\pi_k}$ for the softmax policy, with no
ties to resolve ({ref}`sec-smooth-pi-newton`). And the recommendation at the
end of {ref}`sec-wr-solve`, a few sweeps of function iteration before
switching to Newton, is how Rust solved his bus-engine model
({ref}`sec-rust-inverse`).

For collocation, the Newton step reads as follows. Let $a_i^*$ be the greedy
action at collocation point $s_i$ for the current coefficients. The envelope
theorem gives the Jacobian of the targets,
$[J]_{ij} = \gamma \sum_{s'} p(s'|s_i, a_i^*) \varphi_j(s')$, and the Newton
step solves $(\boldsymbol\Phi - J)\theta^{(k+1)} = \mathbf{r}_{\pi_k}$, the
evaluation equation of the greedy policy restricted to the collocation points
and solved within the span of the basis.

(sec-abe-lstd)=
### The Linear Case: LSTD and LSPI

When the features enter linearly and the state space is finite, the
approximate evaluation step has a closed form. The fixed point of fitted
policy evaluation solves the projected Bellman equation
{eq}`eq:projected-bellman-pi`. Its residual $\BellmanPi v - v$ is
$\xi$-orthogonal to every feature, which in matrix form reads

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \boldsymbol{\Phi} \theta - \boldsymbol{\Phi} \theta) = \mathbf{0},
\qquad\text{that is}\qquad
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_\pi \boldsymbol{\Phi}) \theta = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{r}_\pi.
$$ (eq:lstd-galerkin)

This is the **LSTD (least-squares temporal difference)** solution of
{cite:t}`BradtkeBarto1996`, an $n\times n$ linear system
$\mathbf{A}\theta = \mathbf{b}$. The practical algorithm never forms
$\boldsymbol\Phi$ over all states or $\mathbf{P}_\pi$ explicitly. It
accumulates $\mathbf{A} = \mathbb{E}_{s \sim \xi}[\boldsymbol{\varphi}(s)(\boldsymbol{\varphi}(s) - \gamma \boldsymbol{\varphi}(s'))^\top]$
and $\mathbf{b} = \mathbb{E}_{s \sim \xi}[\boldsymbol{\varphi}(s) r]$ from
sampled transitions $(s, r, s')$, which is possible because the system is
linear in the unknown.

*Check that the dimensions work out: with $m$ states and $n$ features, what
are the dimensions of $\boldsymbol{\Phi}$, $\boldsymbol{\Xi}$,
$\mathbf{P}_\pi$, and $\mathbf{A}$?*

On a 3-state chain under a fixed policy, with
$\mathbf{P}_\pi = \begin{pmatrix} 0.7 & 0.2 & 0.1 \\ 0.3 & 0.4 & 0.3 \\ 0.1 & 0.3 & 0.6 \end{pmatrix}$,
$\mathbf{r}_\pi = (1, 2, 0)$, features $\varphi(s) = (1, s)$ and uniform weights:

```{code-cell} python
:tags: [hide-input]
import numpy as np

P_pi = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]])
r_pi = np.array([1.0, 2.0, 0.0])
gamma = 0.9

# Feature matrix: phi_1(s) = 1, phi_2(s) = s
states = np.array([1, 2, 3])
Phi = np.column_stack([np.ones(3), states])

# Uniform weighting
xi = np.ones(3) / 3
Xi = np.diag(xi)

# LSTD matrices
A = Phi.T @ Xi @ (Phi - gamma * P_pi @ Phi)
b = Phi.T @ Xi @ r_pi

theta_lstd = np.linalg.solve(A, b)
v_lstd = Phi @ theta_lstd
v_exact = np.linalg.solve(np.eye(3) - gamma * P_pi, r_pi)

print(f"LSTD solution: theta = ({theta_lstd[0]:.4f}, {theta_lstd[1]:.4f})")
print(f"\n{'State':<8} {'Exact':<12} {'LSTD':<12} {'Error':<12}")
print("-" * 44)
for s in range(3):
    print(f"{s+1:<8} {v_exact[s]:<12.4f} {v_lstd[s]:<12.4f} {v_lstd[s] - v_exact[s]:<12.4f}")

# Verify orthogonality
residual = r_pi + gamma * P_pi @ v_lstd - v_lstd
print(f"\nGalerkin orthogonality: <residual, phi_1> = {np.sum(xi * residual * Phi[:,0]):.6f}")
```

With the max back in, Newton's method on the projected optimality equation
$\boldsymbol{\Phi}^\top \boldsymbol{\Xi}\big(\Bellman(\boldsymbol\Phi\theta) - \boldsymbol{\Phi} \theta\big) = \mathbf{0}$
uses the envelope-theorem Jacobian
$\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\gamma \mathbf{P}_{\pi_k} \boldsymbol{\Phi} - \boldsymbol{\Phi})$
at the greedy policy $\pi_k$. After the terms in $\theta^{(k)}$ cancel, the
Newton step is

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_{\pi_k} \boldsymbol{\Phi})\, \theta^{(k+1)} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{r}_{\pi_k},
$$ (eq:lspi)

the LSTD system for $\pi_k$. Alternating greedy improvement with LSTD
evaluation is **LSPI (least-squares policy iteration)**
{cite}`LagoudakisParr2003`. Compare it with fitted value iteration in the same
coordinates,
$\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \boldsymbol{\Phi}\, \theta^{(k+1)} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\mathbf{r}_{\pi_k} + \gamma \mathbf{P}_{\pi_k} \boldsymbol{\Phi} \theta^{(k)})$:
both use the greedy policy, but fitted value iteration applies its Bellman
operator once and regresses, while LSPI solves for its value.

The closed forms need the features to enter linearly. With a neural network,
the evaluation step has no closed form, and the natural replacement is a few
sweeps of fitted policy evaluation, $q \leftarrow \Proj\BellmanPi q$, which is
what the critic of NFQCA does.

```{admonition} Exercises: LSTD and LSPI
:class: hint dropdown

1. **Computing the stationary distribution.** For the 3-state chain above, compute the stationary distribution $\xi$ satisfying $\xi^\top \mathbf{P}_\pi = \xi^\top$.

2. **LSTD with stationary weighting.** Recompute the LSTD solution using the stationary distribution instead of uniform weighting. Compare the approximation error.

3. **Two updates.** On a small MDP with two actions and linear features, implement fitted value iteration and LSPI {eq}`eq:lspi`. Compare the number of iterations, and find features under which LSPI cycles between two policies.
```

(sec-abe-roadmap)=
## From Fitted Value Iteration to FQI and NFQCA

Which steps separate the algorithm of this chapter from the fitted Q-iteration
methods used in practice, and where is each taken?

Fitted value iteration, as stated here, assumes four things that NFQCA does
not.

**It learns values of states, not of actions.** To act greedily with respect
to $v(\cdot;\theta)$ requires, at every decision, the expectation
$r(s,a) + \gamma\,\mathbb{E}[v(s';\theta) \mid s,a]$ for every action: the
model and an integral at decision time. Storing an action-value function
$q(s,a;\theta)$ instead moves that work into training, so that acting is a
maximization over $q(s,\cdot;\theta)$. The Bellman operator on
$q$-functions, $(\Bellman q)(s,a) = r(s,a) + \gamma\,\mathbb{E}[\max_{a'} q(s',a') \mid s,a]$,
is again a $\gamma$-contraction, and everything in this chapter applies to it
unchanged. Fitted value iteration on $q$ is fitted Q-iteration.

**It computes expectations exactly.** The targets contain
$\mathbb{E}[\,\cdot \mid s,a]$, which the eight-dimensional example already
replaced by an average over draws. With a simulator, or with logged data, a
target typically uses a single sampled successor. The
[Monte Carlo chapter](monte-carlo-bellman-estimation.md) studies what this
costs: variance, and a bias that comes from maximizing over noisy estimates.

**It chooses the states at which targets are computed.** Batch reinforcement
learning receives a fixed set of transitions instead, and online
reinforcement learning a buffer that changes as the agent acts. The error
analysis of this chapter said that a fit is controlled only where there are
data. The [fitted Q-iteration chapter](fitted-q-iteration.md) builds the
batch algorithms on this template, the tree-based FQI of {cite:t}`ernst2005tree`
(an averager) and NFQ {cite}`riedmiller2005neural` (a neural network), and
then DQN, whose design choices respond to the deadly triad.

**It maximizes over a finite set of actions.** Every target contains
$\max_{a'} q(s',a';\theta)$. When the action is continuous, a torque or a
dose, that maximum is a nonlinear program, to be solved once for every
target at every iteration. NFQCA {cite}`Hafner2011` trains a second network,
an actor $\pi_{\boldsymbol w}(s) \approx \arg\max_a q(s,a;\theta)$, and uses it
in the target in place of the maximum. The critic $q$ is fitted by regression
against fixed targets, exactly as in this chapter, and the actor by gradient
ascent on the critic. In the terms of {ref}`sec-abe-api`, NFQCA is
approximate policy iteration with one fitted evaluation sweep per
improvement, that is, a fitted version of modified policy iteration
({prf:ref}`alg-modified-policy-iteration`) in which the improvement step is
learned as well. The [amortized action optimization chapter](amortized-action-optimization.md)
develops it, together with its online descendants DDPG and TD3. The smooth
Bellman operator meets the same obstacle in the partition function over a
continuous action set ({ref}`sec-continuous-actions`), and soft actor-critic
resolves it in the same way.

| What this chapter assumes | What replaces it | Where |
|:--------------------------|:-----------------|:------|
| State values $v(s)$ | Action values $q(s,a)$, so acting needs no model | [Monte Carlo](monte-carlo-bellman-estimation.md), [FQI](fitted-q-iteration.md) |
| Exact expectation in the target | One sampled successor per transition | [Monte Carlo](monte-carlo-bellman-estimation.md) |
| Targets at chosen states | A fixed batch (FQI, NFQ) or a replay buffer (DQN) | [FQI](fitted-q-iteration.md) |
| Maximum over a finite action set | A learned actor in the target (NFQCA, DDPG, TD3) | [Amortized action optimization](amortized-action-optimization.md) |

(sec-abe-conjunctions)=
### Example: Collision Avoidance Learned from Logged Warnings

What does fitted Q-iteration look like when there is no model, only a log?
When a satellite and another object, most often a piece of debris, are
predicted to pass close to each other, the operator receives about three
conjunction data messages per day over the week before the time of closest
approach. Each message gives a collision probability estimated from the latest
tracking. The operator can plan an avoidance maneuver or wait for the next
message. Waiting buys information, because most alarms go away as the
tracking improves. It also makes the maneuver more expensive: closer to the
encounter, a larger change of velocity is needed for the same miss distance,
and there is less time to plan. ESA plans its maneuvers at least two days
before closest approach, and its missions in low orbit react to collision
probabilities between $10^{-5}$ and $10^{-4}$ {cite}`Uriot2022`. ESA released
the messages of 13,154 conjunctions of its satellites for a machine-learning
competition {cite}`Uriot2022`, and we use them as the log.

We pose an optimal stopping problem on these sequences. A decision is taken at
each message received at least two days before closest approach, and once
more at the two-day deadline with the latest message. A maneuver $t$ days
before closest approach costs $c(t) = (1 + 2/t)/2$, in units of a maneuver two
days out. An event without a maneuver costs $C = 10^5$ times the collision
probability of its last message, the best estimate available, so that a
maneuver at the deadline breaks even at a probability of $10^{-5}$. The state
is what the operator knows at a decision: the time $t$ to closest approach,
the latest log-probability $\rho$, and the age of that message. The costs are
teaching choices, and ESA's analysts look at much more than one number.

Two features of the problem let fitted Q-iteration work from the log alone.
A maneuver ends the problem at a known cost, so $q(s, \text{maneuver}) = c(t)$
needs no learning. And waiting does not change which messages arrive: the
released sequences contain no maneuvers, so each one shows what follows
"wait". Fitted Q-iteration only has to learn $W(s) = q(s, \text{wait})$, by
regression on the logged transitions $(s_i, s_i')$ with targets

$$
y_i = \begin{cases} C\, p_{\text{final}} & \text{if } s_i \text{ is the deadline,} \\
\min\{c(t_i'),\, W_k(s_i')\} & \text{otherwise,} \end{cases}
\qquad W_{k+1} = \operatorname{fit}\big(\{(s_i, y_i)\}\big).
$$

The minimum over the two actions is the $\max$ of the Bellman operator for
costs. There is no expectation to compute: each logged successor is one draw
of $s'$, as in the second row of the table above. We fit extra trees, the
averager of tree-based FQI {cite}`ernst2005tree`, on the logged decision
epochs, and judge every policy out of sample: each event is decided by a
policy fitted without it.

:::{figure} _static/conjunction_wait_or_burn/wait_or_burn.svg
:label: fig-conjunction-wait-or-burn

Collision avoidance from ESA's conjunction messages. (a) The collision
probability of the 742 events whose estimate crossed the break-even
probability, about $10^{-5}$ (dashed), before the deadline. The 59 in color
ended above it. (b) Total cost of each policy over all events, split into the
cost of maneuvers and of expected collisions. Never maneuvering costs 2,668.
(c) The policy learned by fitted Q-iteration on all events, at a message that
has just arrived. Above the blue boundary it maneuvers. The dashed line is the
break-even probability of the message taken at face value.
:::

:::{include} artifacts/conjunction_wait_or_burn/results.md
:::

Maneuvering at the first alarm is safe but wasteful: it makes 742
maneuvers for 62 events that needed one, because 92 of every 100 alarms
resolve ({numref}`fig-conjunction-wait-or-burn`, panel (a)). Waiting until the
deadline and then acting on the latest message halves the cost. This rule is
what an operator would write by hand. Fitted Q-iteration was told neither
that waiting pays nor at what probability to act, and it comes within about 6% of
the hand rule, 309 against 291. Its learned rule (panel (c)) waits through a
new alarm more than three days out unless the probability is twenty to forty
times the break-even value, in which case it maneuvers early, while a
maneuver is still cheap. The bar comes down as the deadline nears, but at the
deadline it still asks for about three times the break-even probability, a
bet that later messages will come down, and that bet misses more of the
events that needed a maneuver, 18 against 6 for the hand rule. Two choices
decide whether fitted Q-iteration works here at all, and both are lessons of
this chapter.

1. **The fit must fit the right thing.** The raw target $C\, p_{\text{final}}$
   is heavy-tailed. One event in the log ends at a probability of 2% and
   costs about 2,070 if nothing is done, more than three times the cost of
   all other events together. Extra trees predict the average of the targets
   in a leaf, so this one cost is shared by every state in the leaves where
   its epochs fall, and those states look dangerous. Fitted on all events,
   the raw-target policy makes 391 maneuvers; fitted without that single
   event, 141. Out of sample it makes 381 maneuvers and costs 444, half again
   as much as the hand rule. The second version fits the ratio
   $y_i / (C\, 10^{\rho_i})$, the cost relative to what the current message
   alone predicts, and multiplies the fit back. The regressor then learns a
   correction factor to the operator's own estimate. For the dangerous event
   this factor stays between 0.7 and 7, an ordinary value, and the policy
   makes 171 maneuvers with the event in the data and 162 without it.
2. **The fit is only as good as the data where decisions are made.** Five
   in six of the 123,882 decision epochs have probabilities below $10^{-7}$,
   where waiting is obviously right, and we leave them out of the fit. Only
   62 events needed a maneuver. This is the coverage problem behind the
   concentrability coefficients of {ref}`sec-abe-error`: the regression
   error that matters is the one at the few states where the two actions are
   close, and no amount of data elsewhere reduces it. The hindsight policy,
   at 147, is not a target: it knows the estimate of the messages that arrive
   after the deadline, which no policy can use.

- {download}`Conjunction model and fitted Q-iteration <code/conjunction_wait_or_burn.py>`
- {download}`Conjunction experiment builder <scripts/build_conjunction_wait_or_burn_artifacts.py>`
- {download}`Conjunction results <artifacts/conjunction_wait_or_burn/metrics.json>`
- {download}`Conjunction messages used (columns of the ESA training set, CC BY 4.0) <data/conjunctions/kelvins_cdm_extract.csv.gz>`

## Summary and Outlook

Replacing the table by a parametric function turns value iteration into
fitted value iteration: compute Bellman targets at chosen states, fit the
function to them, repeat. Collocation and Galerkin projection are two fitting
rules, and regression with trees or neural networks is the one that scales,
as the eight-dimensional stopping problem showed. If each fit is accurate to
$\varepsilon$ in the sup norm, the value error stays below
$\varepsilon/(1-\gamma)$ and the greedy policy loses at most
$2\gamma\varepsilon/(1-\gamma)^2$. On the duck-harvest model both bounds
held with one to five orders of magnitude to spare, and the greedy policies
were far more accurate than the values. The fit can, however, stretch differences
and make the iteration diverge, even when it represents the solution exactly.
Averagers cannot, and for policy evaluation, least squares weighted by the
stationary distribution of the policy cannot either. Neither argument covers
the optimality operator with a neural network, which is the setting of
practical fitted Q-iteration. Newton's method becomes approximate policy
iteration, with LSTD and LSPI as its linear case. On ESA's collision
warnings, fitted Q-iteration learned from the log alone a maneuver rule
within about 6% of the one an operator would write by hand, once its
targets were scaled so that one rare event could not dominate the fit.

Exact target evaluation still assumes access to the transition probabilities
or an exact expectation. How can the same Bellman update be estimated when the
model supplies only samples? [Monte Carlo Bellman estimation](monte-carlo-bellman-estimation.md)
replaces the exact integral by sampled averages and makes their variance and
maximization bias explicit.

## Self-checks

:::{exercise} Weighting shift
:label: ex-projection-check-3

If the fitting distribution puts almost no mass on an important region of the state space, what failure should you expect?
:::

:::{solution} ex-projection-check-3
:class: dropdown

The fit may be accurate under the weighted norm yet poor in that neglected region, leading to bad values or decisions there. In the error bound of fitted value iteration, this shows up as a large sup-norm error, or, in the weighted analysis, as a large concentrability coefficient.
:::

:::{exercise} Accurate fits, bad iteration
:label: ex-projection-check-4

In the two-state counterexample, the function class contains $v^*$ exactly. Why does fitted value iteration still diverge, and why does {prf:ref}`thm-approximate-vi` not prevent it?
:::

:::{solution} ex-projection-check-4
:class: dropdown

The least-squares fit is not an averager: it stretches the constant target into a line that overshoots at state 2, by a factor $6/5$, which beats the contraction $\gamma$ when $\gamma > 5/6$. The fit error at iteration $k$ is proportional to $\theta_k$, so there is no single $\varepsilon$ that bounds it for all $k$, and the theorem's assumption fails.
:::

:::{exercise} Where the actor enters
:label: ex-projection-check-5

Which part of fitted Q-iteration does the actor of NFQCA replace, and why is it needed only for continuous actions?
:::

:::{solution} ex-projection-check-5
:class: dropdown

It replaces the maximization over next actions inside each Bellman target. With a finite action set the maximum is computed by enumeration; with a continuous action set it is a nonlinear program for every target, and the actor amortizes those programs into one trained network.
:::
