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
# Fitted Value Methods

The [previous chapter](weighted-residual-methods.md) replaced the value
function by a parametric function $v(\cdot;\theta)$ and asked which finite
set of conditions should determine $\theta$. Its Step 4 solved those
conditions by function iteration, a Bellman backup at chosen states followed
by a fit, with two particular fits: interpolation at the nodes of a basis, and
least squares on a basis. This chapter drops the basis. Any regressor can do
the fit, a grid interpolant, a tree ensemble or a neural network, and the
result is **fitted value iteration**. Each iteration is an ordinary regression
problem whose targets are Bellman backups, and fitted Q-iteration, NFQ, DQN and
NFQCA in the chapters ahead are elaborations of it.

The fit is not free. Value iteration converges because $\Bellman$ is a
contraction in the sup norm ({prf:ref}`value-iteration-convergence`), and the
fishery of the previous chapter showed a fit undoing that. The chapter
therefore follows the algorithm with three results. If every fit is accurate
to $\varepsilon$, the errors do not accumulate beyond $\varepsilon/(1-\gamma)$,
which is what licenses an arbitrary regressor. Least squares can nonetheless
make the errors grow without bound, on a two-state problem whose value
function it represents exactly. And two kinds of fit are provably safe:
averagers, which include grid interpolation and the tree ensembles of the
first fitted Q-iteration algorithm, and least squares weighted by the state
distribution of the policy being evaluated. Neural networks belong to neither
kind. What keeps neural fitted Q-iteration stable in practice is a set of
design choices that the [chapter on logged and online data](fitted-q-iteration.md)
examines. We then turn to the other solver of the previous chapter, Newton's
method, which becomes approximate policy iteration and, for linear features,
the LSTD and LSPI algorithms. The chapter closes with the steps that separate
fitted value iteration from NFQCA, and the chapter in which each is taken.

One example uses real data. The model with which the U.S. Fish and Wildlife
Service sets duck-hunting regulations is small enough to solve exactly, so the
error bounds can be checked against what actually happens.

::::{admonition} Learning Goals
:class: note

After completing this chapter, you should be able to:
- Write fitted value iteration for an arbitrary regressor, and recognize collocation and Galerkin projection as two choices of the fitting rule
- Bound the error of fitted value iteration, and of its greedy policy, by the error of each fit
- Construct a fitted value iteration that diverges, and state two conditions that prevent divergence: averagers in the sup norm, and on-policy weighting for policy evaluation
- Explain approximate policy iteration as Newton's method on the approximate Bellman equation, and LSTD and LSPI as its linear case
- Name the steps from fitted value iteration to FQI, NFQ and NFQCA, and the chapter in which each is taken

**Prerequisites:** the weighted-residual template ([previous chapter](weighted-residual-methods.md)), the contraction and Newton-Kantorovich material of [infinite-horizon MDPs](infinite-horizon-mdps.md), interpolation with cardinal functions in [finite-horizon DP](finite-horizon-dp.md).
::::

(sec-abe-fvi)=
## Fitted Value Iteration

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
$$ (eq:fitted-vi)

the projected value iteration {eq}`eq:projected-vi` of the previous chapter,
with $\Proj$ now any fitting rule. Two choices define the algorithm: where
the targets are computed, and how the function is fitted to them. Nothing
depends on the max. Replacing $\Bellman$ by the policy evaluation operator
$\BellmanPi$ gives fitted policy evaluation, and replacing it by the smooth
operator $\mathrm{L}_\beta$ of {ref}`sec-regularized-operators` gives fitted
smooth value iteration.

The algorithm is two nested loops. The outer loop is successive
approximation: it computes Bellman targets from the current function and
holds them fixed. The inner loop is an optimization: it fits the function to
those targets, in closed form for a linear basis, by building trees for a
tree ensemble, or by gradient steps for a neural network. We call every
algorithm with this structure a **fitted value method**.
{cite:t}`Gordon1995` gave the first analysis of when it converges. The
fitted Q-iteration of {cite:t}`ernst2005tree` is one instance, with
extremely randomized trees as the fit and action values in place of state
values; NFQ {cite}`riedmiller2005neural` is another, with a neural network.
The algorithms of the coming chapters differ mainly in how far the inner loop
is run before the targets are recomputed: DQN runs it for a fixed number of
gradient steps, the period of its target network, and that choice, not a new principle, is what separates it from NFQ.

The two fits of the previous chapter mark the ends of a range. Collocation
computes targets at exactly $n$ states, as many as there are basis functions,
and interpolates them: $\Proj$ is the interpolation operator, and with
piecewise-linear cardinal functions on a grid the algorithm is the stationary
version of the interpolated backward recursion {eq}`eq:interpolated-bellman`
of the finite-horizon chapter. Galerkin projection computes targets at as
many states as the quadrature needs and fits the basis to them by weighted
least squares: $\Proj$ is the $w$-orthogonal projection onto the span of the
basis, and when the states are sampled from $w$ the fit is an ordinary
regression. Collocation computes few targets and matches them exactly;
regression computes many and matches them on average. Regression is what
scales, because it decouples the number of targets from the number of
parameters, and because it applies unchanged to any model that can be fitted
to data.

The fishery of {ref}`sec-wr-fishery` ran the collocation version with the
hat functions and with Chebyshev polynomials, and its sweep over $n$ showed
the polynomial diverging for some $n$. The cell below measures what happens
to the fitted values at the nodes from one iteration to the next. The
maximization over the escapement and the quadrature over the shock are done
by the module that produced the figures of the previous chapter.

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
max, by the envelope theorem ({prf:ref}`thm-envelope`), or through the
log-sum-exp of the [smooth Bellman operator](regularized-dp.md). The two
criteria have different solutions. Every algorithm in the chapters ahead,
from NFQ to NFQCA, fits to fixed targets, and the
[chapter on logged and online data](fitted-q-iteration.md) studies what happens
when the fit is warm-started and stopped after a few gradient steps instead
of run to convergence.

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

from book_figure_theme import INK, PALETTE, figure_style, show

mean_s, v_hat, v_true = scatter
order = np.argsort(mean_s)
with figure_style():
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.scatter(mean_s, v_hat, s=4, alpha=0.25, color=PALETTE["blue"], edgecolor="none")
    ax.plot(mean_s[order], v_true[order], color=INK, lw=1.4)
    ax.text(0.2, 0.74, "network, $D = 8$", color=PALETTE["blue"], fontsize=8.5)
    ax.text(0.72, 0.6, "exact $v^*$", color=INK, fontsize=8.5)
    ax.set_xlabel(r"average of the coordinates, $\bar s$")
    ax.set_ylabel("value")
show(fig)
```

The grid is the most accurate method while it fits in memory, and it is out of
reach at $D = 8$. The trees and the network use the same 2,000 states in every
dimension. The average error of the trees roughly doubles each time $D$
doubles, while that of the network stays near 0.01; the figure shows that the network
has found the one direction along which the value changes. Both have maximum errors several
times their average, at the corners of the cube, where few of the sampled
states lie. That gap is a first glimpse of the issue that dominates the
analysis below and the batch methods of the
[chapter on logged and online data](fitted-q-iteration.md): a fit is controlled
where there are data, and only there.

(sec-abe-error)=
## Propagation of Fitting Errors

Every fit is only approximately right. This section bounds how far from $v^*$
fitted value iteration ends, and how good the returned policy is.

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

1. The error is measured in the sup norm, at every state. A regression
   controls an average error over the states where targets were computed,
   and nothing at the states it never saw. The analysis of fitted value
   iteration by {cite:t}`Munos2007` and {cite:t}`MunosSzepesvari2008`
   replaces the sup norm by weighted norms. The price is a concentrability
   coefficient, which measures how far the state distributions that policies
   reach can be from the distribution of the data. It is the theoretical form
   of the coverage problem of batch and offline reinforcement learning.
2. The fit must be achievable. The bound needs some $\theta$ with
   $v(\cdot;\theta)$ close to $\Bellman v_k$ for every $v_k$ the algorithm
   produces. The function class must therefore be nearly closed under the
   Bellman operator, not merely able to represent $v^*$.
3. The error must stay bounded. The theorem assumes one $\varepsilon$ for
   all iterations. If the fit error grows with $v_k$, the bound says nothing,
   and the next section shows that this happens.

(sec-abe-mallard)=
### Example: Setting Duck-Hunting Regulations

The bounds can be tested on a problem that a government agency solves every
year. Each summer, the U.S. Fish and Wildlife Service chooses the duck-hunting
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

1. Both bounds hold, and both are loose. The value error is 18 to 118
   times smaller than $\varepsilon/(1-\gamma)$, and the policy loss is four to
   five orders of magnitude below $2\gamma\varepsilon/(1-\gamma)^2$. The bounds
   charge every iteration's error at its worst state and let the errors add up
   in the same direction for $1/(1-\gamma) = 50$ years. Neither happens here.
2. The policy is much more accurate than the value function. The
   coarsest grid has 45 nodes and a value error of 2.2
   million birds, yet it picks the reference package in 96.5% of the
   states, and its worst policy loss is 0.10 million birds, about
   a quarter of a percent of the value. A decision depends on the differences between
   the values of the packages, and much of the fitting error is shared by all
   four packages, because their successor states are close.
3. The regressors do about as well as the coarsest grid. With 2,000
   targets per iteration, the trees and the network have fitting errors of
   $0.76$ and $1.47$, of the same order as the $0.79$ of the 45-node grid. The network's
   greedy policy reproduces the reference map (panel (c)): it picks the
   reference package in 98% of the states and loses at most $0.03$. The trees
   lose more, $0.26$ at worst. Their fit is piecewise constant, so it is flat
   where the value has a slope, and the greedy choice between two packages
   depends on that slope.

With two state variables the Service's grid is the right tool, and the
regressors bring nothing. They become necessary when the state has many
dimensions, as in the eight-dimensional stopping problem, or when there is no
model at all to compute targets from, as in the collision-avoidance example
of the [chapter on logged and online data](fitted-q-iteration.md).

- {download}`Mallard harvest model and solvers <code/mallard_harvest.py>`
- {download}`Mallard harvest experiment builder <scripts/build_mallard_harvest_artifacts.py>`
- {download}`Mallard harvest results <artifacts/mallard_harvest/metrics.json>`

(sec-abe-convergence)=
## Divergence and Stable Fits

A function class that contains the solution can still make fitted value
iteration diverge. This section shows how this happens on a two-state
example, then sorts the fits that cannot diverge into two families, one for
each of the two norms in which a convergence proof can be carried out.

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

(sec-abe-two-norms)=
### One Projected Equation, Two Norms

The counterexample is an instance of the projected Bellman equation
{eq}`eq:projected-bellman` of the previous chapter, $v = \Proj\Bellman v$.
The Bellman operator in it is fixed by the problem; the projection $\Proj$ is
chosen by the method. Every stability result in this section is a condition
under which $\Proj$ cannot undo the contraction of $\Bellman$. On a finite state space
$\mathcal{S} = \{s_1, \ldots, s_m\}$, a function is a vector of
$\mathbb{R}^m$, the features are the columns of an $m \times n$ matrix
$\boldsymbol\Phi$ with rows $\boldsymbol\phi(s)^\top$, and the $n$ test
functions of Step 3 of the [previous chapter](weighted-residual-methods.md#sec-wr-conditions)
are the columns of an $m \times n$ matrix $\boldsymbol\Psi$. With a weighting
$\xi$ over the states and $\boldsymbol\Xi = \text{diag}(\xi)$, the conditions
$\boldsymbol\Psi^\top \boldsymbol\Xi (\Bellman v_\theta - v_\theta) = \mathbf{0}$
with $v_\theta = \boldsymbol\Phi\theta$ say that $v_\theta = \Proj\Bellman v_\theta$, where

$$
\Proj = \boldsymbol\Phi\,(\boldsymbol\Psi^\top \boldsymbol\Xi \boldsymbol\Phi)^{-1} \boldsymbol\Psi^\top \boldsymbol\Xi .
$$ (eq:abe-oblique-projection)

This matrix satisfies $\Proj^2 = \Proj$ and returns every element of the span
of the features unchanged, so it is a projection onto that span. What it
discards is the part of a function that the tests do not see. The methods of
the previous chapter differ only in $\boldsymbol\Psi$. Galerkin takes
$\boldsymbol\Psi = \boldsymbol\Phi$, and $\Proj$ is the $\xi$-orthogonal
projection, the weighted least-squares fit. Collocation replaces the row
vectors $\boldsymbol\Psi^\top\boldsymbol\Xi$ by the $n$ rows of the identity
that select the nodes, so that $\Proj$ is interpolation through the nodes; for
a nodal basis such as the hat functions, entry $(s, s_j)$ of $\Proj$ is the
cardinal function $\phi_j(s)$. Other tests give oblique
projections. Bellman residual minimization, for instance, uses the tests
$\boldsymbol\Psi = (\mathbf{I} - \gamma\mathbf{P}_\pi)\boldsymbol\Phi$,
which depend on the operator {cite}`Scherrer2010`. The counterexample above is Galerkin with
uniform weights, and its projection is the matrix
$\Proj = \frac{1}{5}\begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$.

Whether the iteration $v_{k+1} = \Proj\Bellman v_k$ converges is a question
about norms. The composition $\Proj\Bellman$ is a $\gamma$-contraction in a
norm whenever $\Bellman$ is a $\gamma$-contraction in that norm and $\Proj$
does not expand distances in the same norm, since then
$\|\Proj\Bellman v - \Proj\Bellman w\| \leq \|\Bellman v - \Bellman w\| \leq \gamma\|v - w\|$.
Two norms make this argument work, and each gives one family of stability
results.

The first is the sup norm, in which $\Bellman$ is a $\gamma$-contraction for
every MDP, with or without the maximum. The sup-norm operator norm of a matrix
is its largest absolute row sum. When the constant function is in the span of
the features, as it is for hat functions, polynomials and aggregation,
$\Proj\mathbf{1} = \mathbf{1}$, so every row of $\Proj$ sums to one, and its
absolute row sums are at most one exactly when all its entries are
nonnegative. The projection is then a stochastic matrix. This is Gordon's
condition, which the next subsection develops. For collocation it is a
condition on the cardinal functions, which must be nonnegative, and it holds
for hat functions and fails for polynomials. For Galerkin it is a condition on
the entries of $\boldsymbol\Phi(\boldsymbol\Phi^\top\boldsymbol\Xi\boldsymbol\Phi)^{-1}\boldsymbol\Phi^\top\boldsymbol\Xi$,
which fails for the counterexample, whose projection maps $(1, 0)$ to
$(1/5, 2/5)$ and $(0, 1)$ to $(2/5, 4/5)$. Its rows sum to $3/5$ and $6/5$,
so its sup-norm operator norm is $6/5$, the factor by which the fit stretched
the backup in the iteration above.

The second is the $\xi$-weighted norm
$\|v\|_\xi = (\sum_s \xi(s) v(s)^2)^{1/2}$, in which every Galerkin
projection with weighting $\xi$ is nonexpansive, whatever the features, because
it is an orthogonal projection for the inner product that defines the norm.
The difficulty moves to the operator, because the Bellman operator need not
contract in $\|\cdot\|_\xi$. The subsection on on-policy weighting shows
that the evaluation operator $\BellmanPi$ contracts when $\xi$ is the
stationary distribution of the policy, and that the optimality operator does
not.

| | **Sup norm** | **$\xi$-weighted norm** |
|:--|:--|:--|
| Projection | Stochastic matrix (averager) | $\xi$-orthogonal (Galerkin) |
| Features | Nonnegative cardinal functions, local | Any |
| Operator | $\Bellman$ or $\BellmanPi$ | $\BellmanPi$ only |
| Data | Any weighting | $\xi$ stationary for $\mathbf{P}_\pi$ (on-policy) |
| Typical methods | Collocation with hats, aggregation, nearest neighbors, kernel averaging | TD(0), TD($\lambda$), LSTD |
| Main references | {cite:t}`Gordon1995`, {cite:t}`TsitsiklisVanRoy1996` | {cite:t}`TsitsiklisVanRoy1997`, {cite:t}`BertsekasTsitsiklis1996` |

The two columns trade restrictions on the fit for restrictions on the
problem. The sup-norm route accepts the maximum and any data distribution, and
asks for fits that only average their targets. The $\xi$-norm route accepts
any linear features, and asks that the operator be the evaluation operator of
the policy that generated the data. Neither covers deep reinforcement
learning, where a network is fitted from off-policy data to targets that
contain the maximum.

State aggregation sits in both columns. Its features are the indicators of $n$
groups of states, and the Galerkin projection with any weighting $\xi$
replaces each value by the $\xi$-weighted average over its group,
$(\Proj v)(s) = \sum_{s' \in G(s)} \xi(s') v(s') / \sum_{s' \in G(s)} \xi(s')$,
where $G(s)$ is the group of $s$. The entries are nonnegative and the rows sum
to one, so this Galerkin projection is also an averager, for every
weighting. This is why aggregation was among the first approximation schemes with
convergence proofs for value iteration and Q-learning
{cite}`TsitsiklisVanRoy1996,SinghJaakkolaJordan1995`, and why
{cite:t}`VanRoy2006` could bound the loss of its greedy policy with the
optimality operator in place.

### Averagers: Fits That Cannot Expand

The sup-norm route asks which fitting rules cannot stretch differences
between functions in the sup norm. The answer was found independently in
computational economics {cite}`Judd1992,Judd1996,McGrattan1997,SantosVigoAguiar1998`,
economic dynamics {cite}`Stachurski2009`, and reinforcement learning
{cite}`Gordon1995,Gordon1999`, and the communities arrived at the same
condition under different names.

```{prf:definition} Averager
:label: gordon-averager

A linear operator $\Proj$ on the space of bounded real-valued functions on $\mathcal{S}$ is an **averager** if it is monotone, that is $f \leq g$ pointwise implies $\Proj f \leq \Proj g$, and preserves constants, $\Proj\mathbf{1} = \mathbf{1}$. On a finite state space, $\Proj$ is an averager exactly when $\Proj v = Wv$ for a stochastic matrix $W$: $w_{ij} \geq 0$ and $\sum_j w_{ij} = 1$ for every row $i$.
```

The name is Gordon's. The finite-state characterization follows because a
linear map is monotone exactly when its matrix has nonnegative entries, and
preserves constants exactly when its rows sum to one. An interpolation
operator $(\Proj f)(s) = \sum_j \ell_j(s) f(s_j)$ with cardinal functions
$\ell_j$ is an averager exactly when the weights are nonnegative and sum to
one, which is the condition $\Lambda = 1$ of the finite-horizon chapter and
the property that separated the hat functions from the polynomial in the
fishery.

Is an averager the same thing as collocation? In one direction, yes:
collocation with a nodal basis whose cardinal functions are nonnegative and sum
to one, such as the hat functions or multilinear interpolation on a grid, is an
averager, and its $\Proj$ is a projection. The converse fails, because an
averager need not reproduce its targets at the nodes. A kernel smoother
$(\Proj f)(s) = \sum_j k(s, s_j) f(s_j) / \sum_j k(s, s_j)$ with a nonnegative
kernel, or $k$-nearest neighbors, assigns to a node a weighted average of the
targets at nearby nodes, not the target at that node, so $\Proj^2 \neq \Proj$.
Gordon's condition therefore covers collocation with a monotone interpolant
and also a class of smoothers that are not projections at all. What all of
them share is that $\Proj f$ at every state is a convex combination of the
values of $f$, with weights that depend on the states and not on $f$.

```{prf:proposition} Averagers are nonexpansive
:label: monotone-nonexpansive

If $\Proj$ is an averager, then $\|\Proj f - \Proj g\|_\infty \leq \|f - g\|_\infty$ for all bounded $f, g$.
```

```{prf:proof}
Let $M = \|f - g\|_\infty$. Then $-M \leq f(s) - g(s) \leq M$ for all $s$, which can be written as $g - M\mathbf{1} \leq f \leq g + M\mathbf{1}$. By monotonicity, $\Proj(g - M\mathbf{1}) \leq \Proj f \leq \Proj(g + M\mathbf{1})$. By linearity and constant preservation, $\Proj g - M\mathbf{1} \leq \Proj f \leq \Proj g + M\mathbf{1}$, which means $|\Proj f(s) - \Proj g(s)| \leq M$ for all $s$.
```

An averager acts like a weighted average: it respects order and cannot
amplify differences. The least-squares fit of the counterexample is not one.
It maps the constant function $(1, 1)$ to $(3/5, 6/5)$, so it does not
preserve constants, and it does not respect order either. Polynomial
interpolation at Chebyshev nodes preserves constants but is not monotone: its
cardinal functions dip below zero between the nodes, and its Lebesgue
constant is about $2.4$ for ten nodes. That is how the ten-coefficient
polynomial of the fishery ({ref}`sec-abe-fvi`) could undo the contraction of
the Bellman backup, while the hat functions, whose cardinal functions are
nonnegative, could not.

```{prf:theorem} Fitted value iteration with an averager converges
:label: santos-vigo-aguiar-stability

Let $\Bellman$ be a $\gamma$-contraction in the sup norm on the bounded functions on $\mathcal{S}$, and let $\Proj$ be an averager. Then $\Proj \Bellman$ is a $\gamma$-contraction, so fitted value iteration $v_{k+1} = \Proj \Bellman v_k$ converges from any $v_0$ to the unique fixed point $v_\Proj \in \text{Range}(\Proj)$, and

$$
\|v_\Proj - v^*\|_\infty \leq \frac{1}{1-\gamma} \|\Proj v^* - v^*\|_\infty .
$$
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

The theorem is due to {cite:t}`SantosVigoAguiar1998` for monotone
approximation operators and to {cite:t}`Gordon1995` for averagers; the proof
of the proposition follows {cite:t}`Stachurski2009`. The fixed-point error is
the best-approximation error $\|\Proj v^* - v^*\|_\infty$ amplified by
$(1-\gamma)^{-1}$. For an averager the per-iteration error stays bounded, so
the result is the special case of {prf:ref}`thm-approximate-vi` in which the
third difficulty above cannot arise.

The stochastic matrix gives the averaging a Markov-chain reading: $\Proj v$ is
an expected value after one random transition, and expectations preserve
order. The finite-horizon chapter made the same observation about linear
interpolation, which replaces a deterministic successor by a random grid node
with the interpolation weights as probabilities. Composing that chain with
the model's own transitions makes $\Proj\Bellman$ the Bellman operator of a
finite-state MDP, which is why it contracts.

The same condition appears under several names in the literature on
continuous-state control, often without reference to Gordon.
{cite:t}`KushnerDupuis2001` build their Markov chain approximations of
controlled diffusions from interpolation weights that are transition
probabilities, the Markov-chain reading above used as a design principle.
{cite:t}`Rust1997` solves value iteration on $N$ uniformly drawn states, with
the transition density at those states, normalized to sum to one, as the
averaging weights. The random grid is an averager, and for a finite set of
actions the number of draws needed for a given accuracy grows only
polynomially with the dimension of the state. The kernel-based reinforcement learning of
{cite:t}`Ormoneit2002` applies the same normalized-kernel averaging to sampled
transitions, without a model, and proves consistency as the number of samples
grows and the bandwidth shrinks. {cite:t}`MunosMoore2002` refine a
triangulation of the state space where the value function or the policy
varies most, and interpolate within each simplex with barycentric weights,
again an averager. {cite:t}`SzepesvariSmart2004` prove that Q-learning
converges when the action values are interpolated from a finite set of
states, under the same sign condition on the interpolation.

Which regressors are averagers? State aggregation, $k$-nearest neighbors,
kernel smoothing with nonnegative kernels normalized to sum to one, and
multilinear interpolation on a grid all predict a convex combination of
training targets whose weights depend only on the states. So does a
regression tree, which predicts the average of the targets in a leaf, as long
as its partition of the state space does not depend on the targets.
{cite:t}`ernst2005tree` used this to give tree-based fitted Q-iteration a
convergence guarantee. Their extremely randomized trees choose splits with
the help of the targets, and freezing the tree structure after some
iterations restores the guarantee. Linear least squares, high-order
polynomial interpolation, and neural networks are not averagers. The table
lists the common fits, with a paper that used each one for value functions.

| **Method** | **Averager?** | **Notes** | **Used for value functions in** |
|:-----------|:--------------|:----------|:------------|
| Piecewise linear interpolation | Yes | Barycentric weights are nonnegative and sum to one | {cite:t}`MunosMoore2002` |
| Multilinear interpolation (grid) | Yes | Weights are products of one-dimensional hat functions | {cite:t}`Johnson1993` |
| Variation-diminishing splines (Schoenberg) | Yes | B-spline weights are nonnegative and sum to one | {cite:t}`Stachurski2008` |
| Shape-preserving splines (Schumaker, rational Hermite) | Not linear | Preserve monotonicity and concavity of the data, not covered by the averager theorem | {cite:t}`CaiJudd2012` |
| State aggregation | Yes | Equal weights within a group | {cite:t}`TsitsiklisVanRoy1996`, {cite:t}`SinghJaakkolaJordan1995` |
| $k$-nearest neighbors | Yes | Equal weights within a neighborhood | {cite:t}`ShahXie2018` |
| Kernel smoothing (nonnegative kernels) | Yes | If the weights are normalized to sum to one | {cite:t}`Ormoneit2002`, {cite:t}`Rust1997` |
| Tile coding (CMAC) | Depends on the fit | Averaging the tilings' separate fits is an averager; the joint least-squares fit is not in two or more dimensions ({ref}`sec-abe-tile-coding`) | {cite:t}`Sutton1996` |
| Regression trees and forests, fixed partition | Yes | Leaf averages; guarantee lost if splits depend on the targets | {cite:t}`ernst2005tree` |
| High-order polynomial interpolation | No | Negative weights (Runge phenomenon) | {cite:t}`Bellman1963` |
| Least squares on an arbitrary basis | No | The counterexample above | {cite:t}`SchweitzerSeidmann1985`, {cite:t}`LagoudakisParr2003` |
| Neural networks | No | No sign or normalization constraint on the fit | {cite:t}`Tesauro1995`, {cite:t}`riedmiller2005neural` |

The table places the fits of the two chapters. Collocation with
piecewise-linear interpolation is an averager, so the folk wisdom that linear
interpolation is safe for value iteration is a theorem. Collocation
with a high-order polynomial is not, and its convergence in the fishery, when
it happened, was not guaranteed. Galerkin projection and least squares are
not averagers. The tree ensemble of the eight-dimensional example sits on the
boundary, and the network is outside. The solver matters too: Newton's method
never iterates $\Proj\Bellman$, so it does not need the contraction, which is
why the hybrid strategy of {ref}`sec-wr-solve` starts with a monotone,
low-order fit and switches to Newton with the accurate one.

Averagers pay for their stability in accuracy. The error bound of
{prf:ref}`santos-vigo-aguiar-stability` is the best-approximation error
$\|\Proj v^* - v^*\|_\infty$, and an averager can only represent functions
that are smooth on the scale of its weights. That error is the error of a
local method, and in high dimension it decreases slowly with the number of
nodes.

```{admonition} Exercises: Divergence and Averagers
:class: hint dropdown

1. **The counterexample with an averager.** Replace the least-squares fit in the two-state example by state aggregation into a single group. Write $\Proj$ as a matrix, check that it is an averager, and show that fitted value iteration converges for every $\gamma < 1$. What is its fixed point?

2. **A non-monotone example.** Using Lagrange interpolation with 4 equally spaced nodes on $[0, 1]$, compute the interpolation weights for the state $s = 0.1$. Show that some weights are negative.

3. **Verifying monotonicity.** Consider piecewise linear interpolation on a 5-point grid. Write out the interpolation weights for a point between grid points 2 and 3. Verify that all weights are nonnegative and sum to one.

4. **Trees.** Show that a regression tree with a fixed partition is an averager. Explain why a tree that chooses its splits by the targets can change its partition from one iteration to the next, and which step of the proof of {prf:ref}`santos-vigo-aguiar-stability` this breaks.
```

(sec-abe-tile-coding)=
### Tile Coding: Between Aggregation and Galerkin

Tile coding is the linear representation that {cite:t}`Sutton1996` used to
make early reinforcement learning work, and it descends from the cerebellar
model articulation controller (CMAC) of {cite:t}`Albus1975`. It gets no simple
yes or no in the averager column of the table above, and the figure below
shows why.

Each row of the top panel of {numref}`fig-tile-coding-kernel` is one
*tiling*: a grid that cuts $[0, 1]$ into cells of width $h$, called *tiles*.
There are $K$ tilings, and tiling $k$ is shifted by an offset $o_k$, so the
rows are staggered. The vertical line is a state $s$. It lies in exactly one
tile of each row, shaded in orange. Each tile carries one weight, and the
value at $s$ is the sum of the $K$ weights of the orange tiles. In vector
form, the feature vector $\boldsymbol\phi(s)$ has one binary entry per
tile, with exactly $K$ ones, and $v(s; \theta) = \boldsymbol\phi(s)^\top\theta$.

For example, take $K = 2$ tilings of width $h = 1/2$, the second shifted by
$h/2 = 1/4$. The first tiling has two tiles, $[0, 1/2)$ and $[1/2, 1]$, and the
second has three, $[0, 1/4)$, $[1/4, 3/4)$ and $[3/4, 1]$, so there are five
features, one per tile, listed tiling by tiling. The state $s = 0.6$ lies in
the second tile of both tilings, and $s' = 0.8$ lies in the second tile of the
first tiling and the third tile of the second, so

$$
\boldsymbol\phi(0.6) = (0, 1 \mid 0, 1, 0)^\top, \qquad
\boldsymbol\phi(0.8) = (0, 1 \mid 0, 0, 1)^\top,
$$

$$
v(0.6; \theta) = \theta_{1,2} + \theta_{2,2}, \qquad
v(0.8; \theta) = \theta_{1,2} + \theta_{2,3},
$$

where $\theta_{k,j}$ is the weight of tile $j$ of tiling $k$. The two states
share one of their two tiles, so their values share the weight $\theta_{1,2}$.
The middle panel of the figure writes $\boldsymbol\phi(s)$ out in the same way,
one group of boxes per tiling, and lists the $K$ weights that add up to
$v(s; \theta)$ as $s$ moves.

Set $K = 1$ with the slider and a single row remains: that is state
aggregation, which sits in both columns of the two-norm table. With $K > 1$
the tiles overlap, and whether the fit is an averager depends on how the
weights are fitted. Two questions settle it, and the top and bottom panels of
the figure answer them: which functions the features can represent, and how an update at
one state changes the value at another.

:::{iframe} ../interactive/tile-coding.html#view=kernel
:label: fig-tile-coding-kernel
:width: 100%
:title: Tilings and the shared-tile kernel
:class: tile-coding-kernel

Top: $K$ tilings of $[0, 1]$ with tiles of width $h = 1/4$, the tiles that
contain $s$ (orange), and the fine cells where all the tilings agree. Middle:
the feature vector $\boldsymbol\phi(s)$, one box per tile, grouped by tiling,
and the $K$ weights whose sum is $v(s; \theta)$. Bottom:
the fraction of tiles that $s$ shares with each $s'$, against the hat
$(1 - |s - s'|/h)_+$. Drag $s$ or press play. With random offsets, play
redraws the offsets again and again and averages the kernels.
:::

**What tile coding represents.** Move $s$ inside one cell of the bottom row of
the top panel, labeled "fine": the orange tiles do not change. Two states in
the same fine cell lie in the same tile of every tiling, so they have the
same features, and every $v(\cdot\,; \theta)$ is constant on the fine cells.
With offsets spaced $h/K$ apart, the edges of the $K$ tilings interleave and
the fine cells have width $h/K$. In one dimension the code below finds that the
feature matrix has rank equal to the number of fine cells, so the span of the
tile features is exactly the set of functions that are constant on the fine
cells. In one dimension, then, $K$ tilings of width $h$ represent the same
functions as one tiling of width $h/K$, with about the same number of weights:
the text under the figure reports 16 fine cells and 19 weights for $K = 4$.
The saving appears in higher dimension. In $d$ dimensions there are about
$(K/h)^d$ fine cells when the offsets are spaced $h/K$ apart in each
coordinate, but only about $K(1/h + 1)^d$ weights, so tile coding resolves the
fine grid with roughly $K^{d-1}$ times fewer parameters than a fine tiling
would need. The price is that its span is now a strict subspace of the
functions that are constant on the fine cells. For $K = 4$ and $h = 1/4$ in
the plane, the code below finds 91 weights, a rank of 88, and 256 fine cells.

**How tile coding generalizes.** The bottom panel answers the second
question. Sutton used tile coding with incremental updates of the form
$\theta \leftarrow \theta + \frac{\alpha}{K}\boldsymbol\phi(s)\,(y - v(s; \theta))$,
which move each of the $K$ orange weights by the same amount. The value at any
other state $s'$ then changes by

$$
v(s'; \theta_{\text{new}}) - v(s'; \theta) = \alpha\,\kappa(s, s')\,\big(y - v(s; \theta)\big),
\qquad
\kappa(s, s') = \frac{\boldsymbol\phi(s)^\top\boldsymbol\phi(s')}{K},
$$

where $\kappa(s, s')$ is the fraction of tiles that $s$ and $s'$ share, the
blue staircase in the bottom panel. The update at $s$ spreads to its neighbors
in proportion to $\kappa$.

Compare the staircase with the dashed orange curve in the same panel. That
curve is the *hat function* centered at $s$,

$$
\Big(1 - \frac{|s - s'|}{h}\Big)_+ = \max\Big(0,\; 1 - \frac{|s - s'|}{h}\Big),
$$

a triangle of height one and half-width $h$: it equals $1$ at $s' = s$,
decreases linearly with the distance $|s - s'|$, and is $0$ once $s'$ is a
full tile width away from $s$. It is the hat function of piecewise-linear
interpolation, centered at $s$, with the tile width as its half-width.

The top panel shows why the staircase follows the hat. A state $s'$ shares
the tile of row $k$ with $s$ unless an edge of row $k$ falls between them, so
$\kappa(s, s') = 1 - (\text{number of rows with an edge between } s \text{ and } s')/K$.
Each row has one edge every $h$, so when $\delta = |s - s'| < h$ at most one
edge of each row lies between the two states, and when $\delta \geq h$ every
row has one. When the offsets are spaced $h/K$ apart, the edges of the $K$
rows interleave evenly, one every $h/K$, so an interval of length $\delta < h$
contains $K\delta/h$ edges, rounded up or down. Hence $\kappa(s, s')$ is
$1 - \delta/h$ up to one step of $1/K$:

$$
\Big|\,\kappa(s, s') - \Big(1 - \frac{|s - s'|}{h}\Big)_+\Big| < \frac{1}{K}.
$$

The staircase therefore stays within one step of the hat, and the steps shrink
as $K$ grows. Switch to random offsets and press play: for each draw the
staircase is irregular, but a row with a random offset puts an edge between
two states at distance $\delta$ with probability $\min(\delta/h, 1)$, so the
expected fraction of shared tiles is exactly the hat, and the running average,
in green, converges to it.

This answers the question of what kind of function approximator tile coding
is. As a function of the displacement $\delta = s' - s$, the hat is the box
of width $h$ convolved with itself and divided by $h$, which is the linear
(degree-one) B-spline.
The two questions of this subsection have different answers. What tile coding
fits is piecewise constant: each feature is the indicator of one tile, a box,
which is a degree-zero B-spline, and every tile is a union of fine cells, so
every combination $\boldsymbol\phi(s)^\top\theta$ is constant on each fine
cell. How tile coding generalizes is piecewise linear: an update at $s$ changes
the value at $s'$ in proportion to $\kappa(s, s')$, which is the hat up to
$1/K$, a degree-one B-spline. The CMAC literature drew the same conclusion and generalized it:
{cite:t}`LaneHandelmanGelfand1992` replace the box receptive fields by
B-splines of higher order, which makes the fitted function itself a smooth
spline whose derivatives can be learned, and {cite:t}`BrownHarris1994` treat the
CMAC and B-spline networks as members of one family of lattice-based networks
that are linear in their weights. Kernel methods reached tile coding from the
other direction. The random binning features of {cite:t}`RahimiRecht2007` are
tile coding with random offsets: they compute that two points fall in the same
bin with probability given by the hat, and that drawing the tile width $h$ at
random as well, with density $h\,\ddot\kappa(h)$, turns the expected kernel into
a kernel $\kappa(|s - s'|)$ of the distance, for any $\kappa$ whose second
derivative is nonnegative, and into products of such kernels over the
coordinates in several dimensions. The Laplacian kernel $\exp(-|s - s'|)$, for
example, comes from a tile width with a Gamma distribution. Tile coding with
random offsets is thus a finite random-feature approximation of kernel
regression.

**Two dimensions and hashing.** In the plane, each tiling is a grid of
$h \times h$ squares, and {numref}`fig-tile-coding-plane` shades
$\kappa(s, \cdot)$ around a point $s$, with the $K$ squares that contain $s$
outlined. With offsets spaced along the diagonal, every tiling moves along the
line $y = x$, and the shading is stretched along that line: states displaced
along the diagonal share more tiles than states displaced across it. With
random offsets drawn independently in each coordinate, the expected kernel is
the product of one-dimensional hats, a tensor-product linear B-spline. The
displacement $(1, 3)$ comes close to that symmetric shape with evenly spaced
offsets, which is why {cite:t}`SuttonBarto2018` recommend displacement vectors
made of the first odd integers.

:::{iframe} ../interactive/tile-coding.html#view=plane
:label: fig-tile-coding-plane
:width: 100%
:title: Tile coding in the plane
:class: tile-coding-plane

The shared-tile kernel $\kappa(s, \cdot)$ in the plane for $K$ tilings of
$h \times h$ squares, $h = 1/4$, with diagonal, asymmetric $(1, 3)$ or random
offsets. The hash buttons send all the tiles into a table of 256, 64 or 16
weights, and the shading then shows the false similarity that collisions add
far from $s$. Drag the point or press play.
:::

When $K(1/h + 1)^d$ weights are too many, tile coding implementations map
the tiles into a table of $M$ weights with a hash function
{cite}`SuttonBarto2018`. Press "64 weights" in the figure: speckles appear far
from $s$. Within one tiling, hashing only merges tiles into groups, so a hashed
tiling is still a partition of the state space, into groups of cells that need
not be adjacent. Across tilings, two tiles of different tilings can share a
weight, and two distant states then share weights by accident. For a hash that
sends each tile to a bucket uniformly and independently, the expected inner
product of the hashed features is the number of shared tiles plus
$(K^2 - \text{shared})/M$, a bias toward false similarity that grows as the
table shrinks. Feature hashing in machine learning {cite}`Weinberger2009`
removes this bias by multiplying each hashed feature by a random sign. The
inner product is then unbiased, but the features are no longer nonnegative.

**Where tile coding sits.** The answer depends on how the weights are fitted,
and the three common ways give three different operators.

1. *Fit each tiling separately and average.* Each tiling alone is aggregation,
   whose projection $\Proj_k$ is a stochastic matrix. The average
   $\bar\Proj = \frac{1}{K}\sum_k \Proj_k$ is a convex combination of
   stochastic matrices, hence stochastic, so it is an averager and
   {prf:ref}`santos-vigo-aguiar-stability` applies. With uniform weighting and
   away from the boundary, row $s$ of $\bar\Proj$ is
   $\kappa(s, \cdot) / \sum_{s'}\kappa(s, s')$, so this fit is the kernel
   smoother of the table, with the staircase of {numref}`fig-tile-coding-kernel`
   as its kernel. It is not a projection, since $\bar\Proj^2 \neq \bar\Proj$.
   A hashed tiling is still a partition, so hashing within each tiling keeps
   this fit an averager; a signed hash does not.
2. *Fit all the weights jointly by least squares.* This is the Galerkin
   projection onto the span of the tile features. In one dimension, with
   every fine cell weighted, the span is that of the fine tiling, so the
   projection is aggregation onto the fine cells and is again an averager. In
   two dimensions it is not: for $K = 4$ and $h = 1/4$, the projection has
   negative entries and its sup-norm operator norm exceeds $4$. Gordon's
   certificate is then lost, and only the $\xi$-norm results of the next
   subsection remain.
3. *Fit the weights incrementally by TD, as Sutton did.* The sample-based
   update is a stochastic approximation of the joint fit, and its limit is
   the solution of the projected Bellman equation with the Galerkin projection
   of item 2. Each step spreads the correction with the nonnegative kernel
   $\kappa$, but this does not make the fixed point an averager. On-policy, the
   convergence result of {cite:t}`TsitsiklisVanRoy1997` applies, as it does to
   any linear features.

```{code-cell} python
:tags: [hide-input]

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "code"))
import tile_coding as tc

z = lambda x: 0.0 if abs(x) < 1e-9 else x

k = tc.kernel_report_1d(m=8, h=0.25)
print("Shared-tile kernel, K = 8, h = 0.25")
for key, val in k.items():
    print(f"  {key:<42}{val:8.4f}")

a = tc.kernel_2d_anisotropy(m=8, h=0.25, r=0.1)
print("\nTwo dimensions, shared fraction at distance 0.1")
for key, (along, across) in a.items():
    print(f"  {key:<20} along the diagonal {along:.3f}, across it {across:.3f}")

print("\nProjections (K = 4, h = 0.25, uniform weighting on a grid of states)")
print(f"  {'':<4}{'states':>7}{'weights':>8}{'rank':>6}{'fine':>6}"
      f"{'joint min':>11}{'joint sup':>10}{'avg min':>9}{'avg sup':>9}")
for d, n in ((1, 64), (2, 32)):
    r = tc.projection_report(m=4, h=0.25, n=n, d=d)
    print(f"  {d}D  {r['states']:>7}{r['weights']:>8}{r['rank']:>6}{r['fine cells']:>6}"
          f"{z(r['joint LS: min entry']):>11.3f}{r['joint LS: sup norm']:>10.3f}"
          f"{r['averaged: min entry']:>9.3f}{r['averaged: sup norm']:>9.3f}")
print(f"  averaged smoother vs normalized shared-tile kernel (interior rows): "
      f"{tc.smoother_matches_kernel():.1e}")

hs = tc.hashing_report(m=8, h=0.25, M=64)
print("\nHashing 8 tilings into M = 64 weights")
for key, val in hs.items():
    print(f"  {key:<42}{val:8.3f}")

curves = tc.learning_curves()
print("\nRMSE after 30 / 100 / 300 / 1000 / 3000 samples (K = 8, h = 0.2, alpha = 0.1)")
for name, c in curves.items():
    print(f"  {name:<36}" + "".join(f"{e:8.3f}" for e in c))
```

The first block confirms the staircase bound, $0.124 < 1/K = 0.125$, and
shows the random-offset average within $0.006$ of the hat after 4000 draws.
In the plane, diagonal offsets let states share three quarters of their tiles
along the diagonal but only three eighths across it, and the displacement
$(1, 3)$ makes the two equal. In the projection table, the joint least-squares
fit is an averager in one dimension (smallest entry $0$, sup norm $1$) and not
in two (smallest entry $-0.054$, sup norm $4.44$), while the averaged
single-tiling fits are averagers in both and coincide with the normalized
shared-tile kernel. The hashing block matches the collision bias
$(K^2 - \text{shared})/M$ and shows the random sign removing it.

The last block is the reason tile coding was popular, and
{numref}`fig-tile-coding-learn` animates it. Three learners receive the same
noisy samples of a function with a jump. One coarse tiling of width $h$, in
blue, generalizes broadly and learns fast, but its error stops at $0.26$, the
error of its best approximation. One fine tiling of width $h/K$, in purple,
can represent the function well, but each sample teaches it about one narrow
cell, so after 100 samples its error is still $0.43$. Tile coding, in orange,
generalizes like the coarse tiling, with error $0.18$ after 100 samples, and
represents like the fine one, with error $0.08$ after 3000. Watch the shaded
bumps: each is the change made by the last sample, a wide box for the coarse
tiling, a narrow box for the fine one, and the staircase hat of
{numref}`fig-tile-coding-kernel` for tile coding. Overlapping tilings buy fast
generalization at fine resolution, and in several dimensions they also buy
memory.

:::{iframe} ../interactive/tile-coding.html#view=learn
:label: fig-tile-coding-learn
:width: 100%
:title: Learning with tile coding
:class: tile-coding-learn

A coarse tiling, a fine tiling and $K$ tilings learn from the same stream of
noisy samples of $v(s) = \sin(2\pi s) + 0.8\,\mathbb{1}[s > 0.6]$, with step
size $\alpha = 0.1$. The shaded bump under each fit is the change made by the
last sample, magnified eight times. Bottom: root-mean-square error against the
number of samples.
:::

### On-Policy Weighting Rescues Policy Evaluation

The $\xi$-norm route keeps the Galerkin projection, which is nonexpansive in
$\|\cdot\|_\xi$ for any features, and asks when the Bellman operator
contracts in the same norm. The counterexample suggests the answer. The chain moves from state 1 to
state 2 and stays there, so its stationary distribution puts all its mass on
state 2. If the fit weights each state by that distribution, it minimizes
$(2\theta - 2\gamma\theta_k)^2$ alone, and $\theta_{k+1} = \gamma\theta_k$
converges. The equal weighting was the problem: it gives the state that the
process leaves at once the same weight as the state where it stays.

For policy evaluation this observation is a theorem, which
{cite:t}`TsitsiklisVanRoy1997` proved to explain why TD(0) converges with
linear features. With the notation of {ref}`sec-abe-two-norms`, the
$\xi$-weighted inner product, the norm, and the Galerkin projection
($\boldsymbol\Psi = \boldsymbol\Phi$) are

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

Orthogonal projection is nonexpansive. For any vector $v$, decompose
$v = \Proj v + (v - \Proj v)$, where $v - \Proj v$ is $\xi$-orthogonal to
the span of $\boldsymbol\Phi$. By the Pythagorean theorem,

$$
\|v\|_\xi^2 = \|\Proj v\|_\xi^2 + \|v - \Proj v\|_\xi^2 \geq \|\Proj v\|_\xi^2.
$$

No sign condition on the entries of $\Proj$ was needed. The sup-norm argument
fails for this projection, while the $\xi$-norm argument does not depend on
the signs of its entries.

$\BellmanPi$ contracts in $\|\cdot\|_\xi$. Since
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

Convergence came from matching the weighting to the operator, not from any
property of the fit beyond orthogonality: $\xi$ must be the stationary
distribution of the very $\mathbf{P}_\pi$ that appears in $\BellmanPi$. When
the data are collected by following $\pi$, this holds automatically, which is
**on-policy** learning. When they come from another behavior policy, as in
the counterexample's equal weighting or in learning from a replay buffer
filled by older policies, the Jensen step fails, $\mathbf{P}_\pi$ need not be
nonexpansive in $\|\cdot\|_\xi$, and $\Proj\BellmanPi$ may not contract.
Baird's counterexample {cite}`Baird1995` is the classical demonstration.

```{admonition} Exercises: The On-Policy Condition
:class: hint dropdown

1. **Weighting the counterexample.** In the two-state example, fit with weights $(\xi_1, \xi_2)$. Show that $\theta_{k+1} = 2\gamma\theta_k(\xi_1 + 2\xi_2)/(\xi_1 + 4\xi_2)$ and find the set of weightings for which the iteration converges for every $\gamma < 1$.

2. **Non-expansiveness fails off-policy.** For the 3-state chain of {ref}`sec-abe-lstd` and the weighting $\xi' = (0.9, 0.05, 0.05)$, find a vector $z$ such that $\|\mathbf{P}_\pi z\|_{\xi'} > \|z\|_{\xi'}$.

3. **Losing the argument is not losing convergence.** Implement fitted policy evaluation for the same chain, with features $\phi(s) = (1, s)$ and weighting $\xi'$. It still converges: compute the spectral radius of $\gamma \Proj \mathbf{P}_\pi$ and check that it equals $\gamma$ for this feature set whatever the weighting. Then keep only the feature $\phi(s) = s$ and find a weighting concentrated on state 1 under which the spectral radius exceeds one.
```

(sec-abe-galerkin-system)=
### The Galerkin System in Coordinates

The projected Bellman equation {eq}`eq:projected-bellman-pi` is the Galerkin
method of the previous chapter, with the stationary distribution as weight.
Its $n$ conditions
$\boldsymbol\Phi^\top\boldsymbol\Xi(\mathbf{r}_\pi + \gamma\mathbf{P}_\pi\boldsymbol\Phi\theta - \boldsymbol\Phi\theta) = \mathbf{0}$
form the linear system

$$
\mathbf{A}\theta = \mathbf{b}, \qquad
\mathbf{A} = \boldsymbol\Phi^\top\boldsymbol\Xi(\mathbf{I} - \gamma\mathbf{P}_\pi)\boldsymbol\Phi, \qquad
\mathbf{b} = \boldsymbol\Phi^\top\boldsymbol\Xi\,\mathbf{r}_\pi .
$$ (eq:abe-td-system)

Fitted policy evaluation solves it by function iteration,
$\theta_{k+1} = \theta_k + (\boldsymbol\Phi^\top\boldsymbol\Xi\boldsymbol\Phi)^{-1}(\mathbf{b} - \mathbf{A}\theta_k)$,
and the LSTD method of {ref}`sec-abe-lstd` solves it directly. Temporal-difference
learning, TD(0), solves the same system from single sampled transitions; the
[next chapter](monte-carlo-bellman-estimation.md#sec-mc-td) treats it as one
more case of replacing expectations by samples. Whatever the solver, the
system has to be well posed, and on-policy the inequality that made
$\Proj\BellmanPi$ a contraction also makes $\mathbf{A}$ positive definite.
This is the step at the center of the convergence proof of TD by
{cite:t}`TsitsiklisVanRoy1997`.

```{prf:proposition} The Galerkin matrix is positive definite on-policy
:label: prop-td-positive-definite

If $\xi$ is a stationary distribution of $\mathbf{P}_\pi$ with $\xi(s) > 0$ for every $s$, and $\boldsymbol\Phi$ has linearly independent columns, then for every $x \neq 0$

$$
x^\top\mathbf{A}x \geq (1-\gamma)\|\boldsymbol\Phi x\|_\xi^2 > 0 .
$$
```

```{prf:proof}
Write $z = \boldsymbol\Phi x$. Then $x^\top\mathbf{A}x = \|z\|_\xi^2 - \gamma\langle z, \mathbf{P}_\pi z\rangle_\xi$. By the Cauchy–Schwarz inequality and the bound $\|\mathbf{P}_\pi z\|_\xi \leq \|z\|_\xi$ proved above, $\langle z, \mathbf{P}_\pi z\rangle_\xi \leq \|z\|_\xi^2$, which gives the claim. Since $\boldsymbol\Phi$ has independent columns, $z \neq 0$.
```

A matrix with $x^\top\mathbf{A}x > 0$ for every $x \neq 0$ need not be
symmetric, but each of its eigenvalues has a positive real part: if
$\mathbf{A}u = \lambda u$ with $u \neq 0$, the real part of
$\bar u^\top\mathbf{A}u = \lambda\|u\|^2$ is the quadratic form of the
symmetric part of $\mathbf{A}$ at the real and imaginary parts of $u$, which is
positive. In particular $\mathbf{A}$ is invertible, so on-policy the Galerkin
system has exactly one solution for any independent features.

When the weighting $\xi$ is not stationary for $\mathbf{P}_\pi$, which is the
situation of evaluating one policy from the data of another, the Jensen step
fails, and the proposition fails with it. The projection is still orthogonal, but
$\mathbf{A}$ can be singular or have eigenvalues in the left half plane, and
the solution, when there is one, can be far from $v_\pi$:
{cite:t}`Kolter2011` shows that its error can be arbitrarily large relative to
$\|\Proj v_\pi - v_\pi\|_\xi$, and describes a convex set of weightings, close
to the behavior distribution, for which a bound of the on-policy form still
holds. What off-policy weighting does to the sample-based solvers, and how
importance sampling, gradient TD and emphatic TD repair it, is taken up in
the [next chapter](monte-carlo-bellman-estimation.md#sec-mc-off-policy).

(sec-abe-two-routes-chain)=
### Both Routes on One Chain

The two families of results certify different projections. To see them side
by side, the next cell takes the policy-evaluation chain of the third tab of
{numref}`fig-test-functions`, discretizes it on 201 grid points, and adds a
restart to a uniform state with probability $0.2$ at each step (without it the
chain never returns below $s = 0.64$, and its stationary distribution ignores
the low states). For six projections with $n = 6$, it reports the sup-norm and
$\xi$-norm operator norms of $\Proj$, with $\xi$ stationary, the smallest
entry of $\Proj$, the spectral radius of $\gamma\Proj\mathbf{P}_\pi$, the
smallest real part of an eigenvalue of $\mathbf{A}$ for the Galerkin methods,
and the sup-norm error of the fixed point.

```{code-cell} python
:tags: [hide-input]

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "code"))
import projection_stability as ps

rows = ps.chain_table(n=6)
cols = ("sup |P|", "xi |P|", "min entry", "rho(gamma P L)", "min Re eig A", "sup error")
print(f"{'projection':<34}" + "".join(f"{c:>15}" for c in cols))
for row in rows:
    print(f"{row['method']:<34}" + "".join(f"{row[c]:>15.3f}" for c in cols))
```

Two rows have sup-norm operator norm one: the averagers, collocation with
hats and aggregation, whose smallest entry is zero. Two rows have $\xi$-norm
one: the Galerkin projections with the stationary weighting. Each certificate
holds where the theory says it should, and nowhere else. The Galerkin
projections stretch the sup norm, since their smallest entries are negative,
and the averagers stretch the $\xi$-norm. The spectral radius is $0.9$ in
every row, so on this chain the iteration converges even for polynomial
collocation, whose sup-norm operator norm is about $2$. The certificates are
sufficient conditions, not necessary ones. A norm above one signals a risk,
and the spectral radius decides whether the iteration diverges; the fishery
showed that the same polynomial can make it diverge on another problem.
Every Galerkin row, including the one with uniform weights, has $\mathbf{A}$
with eigenvalues in the right half plane. The errors differ more than the
stability does: Galerkin with hats is about three times more accurate than
collocation with the same hats, and aggregation, the most stable fit, is the
least accurate one.

```{admonition} Exercises: Projections and Norms
:class: hint dropdown

1. **Aggregation is an averager.** With indicator features of $n$ groups and any weighting $\xi > 0$, compute the Galerkin projection {eq}`eq:abe-oblique-projection` entry by entry and show that $(\Proj v)(s)$ is the $\xi$-weighted average of $v$ over the group of $s$. Conclude that fitted Q-iteration with aggregation converges for any sampling distribution.

2. **Residual minimization as an oblique projection.** Show that the tests $\boldsymbol\Psi = (\mathbf{I} - \gamma\mathbf{P}_\pi)\boldsymbol\Phi$ turn the conditions $\boldsymbol\Psi^\top\boldsymbol\Xi(\BellmanPi\boldsymbol\Phi\theta - \boldsymbol\Phi\theta) = \mathbf{0}$ into the normal equations of $\min_\theta \|\BellmanPi\boldsymbol\Phi\theta - \boldsymbol\Phi\theta\|_\xi^2$. Is the resulting $\Proj$ of {eq}`eq:abe-oblique-projection` fixed in advance, as it is for Galerkin and collocation?

3. **One feature, one condition.** In the two-state counterexample with weights $(\xi_1, \xi_2)$, compute $\mathbf{A}$ and show that the fitted factor equals $1 - \mathbf{A}/(\xi_1 + 4\xi_2)$, so that fitted evaluation converges exactly when $\mathbf{A} > 0$.
```

### The Optimality Operator

The weighted argument does not extend to the optimality operator
$\Bellman v = \max_a [\mathbf{r}_a + \gamma \mathbf{P}_a v]$, and the
obstacle is the Jensen step. For the optimality operator we would need to
bound $\big[\max_{a} \sum_{s'} p(s'|s,a) z(s')\big]^2$, and a maximum of
averages is not an average. Nor is there a single stationary distribution to
match, since the greedy policy, and with it the state distribution, changes
from iteration to iteration. The failure goes beyond the proof:
{cite:t}`deFariasVanRoy2000` give examples in which the projected optimality
equation $v = \Proj\Bellman v$ with a $\xi$-weighted Galerkin projection
has no solution at all, so that no algorithm can converge to one.

Optimal stopping is the exception that confirms the diagnosis. With a
stopping reward $g$ and a continuation chain $\mathbf{P}$ that the decision
does not affect, the operator is
$\Bellman v = \max\{g, \mathbf{r} + \gamma\mathbf{P}v\}$, the maximum taken
state by state. A maximum with a fixed function does not expand differences,
$|\max\{g, a\} - \max\{g, b\}| \leq |a - b|$, so the Jensen step goes
through with the stationary distribution of $\mathbf{P}$, and $\Proj\Bellman$
is a $\gamma$-contraction in $\|\cdot\|_\xi$.
{cite:t}`TsitsiklisVanRoy1999` use this to prove that a TD method with linear
features converges for optimal stopping, with the pricing of high-dimensional
financial derivatives as their application. The eight-dimensional problem of
{ref}`sec-abe-high-dim` has this structure: the coordinates move the same way
whether the agent stops or not.

(sec-abe-control-results)=
### What Is Known for the Optimality Operator

The convergence results for control are scattered across several
literatures. They are easier to compare when sorted into four groups: the two
routes of {ref}`sec-abe-two-norms`, and two approaches that avoid the
question of contraction.

The first group follows the sup-norm route, which covers control completely. The
proof of {prf:ref}`santos-vigo-aguiar-stability` used only that $\Bellman$ is a
$\gamma$-contraction in the sup norm, which holds with the maximum, so fitted
value iteration and fitted Q-iteration with an averager converge for the
optimality operator, from any data, to a fixed point within
$(1-\gamma)^{-1}\|\Proj v^* - v^*\|_\infty$ of $v^*$, and
{prf:ref}`prop-greedy-loss` bounds the loss of its greedy policy. This is
Gordon's theorem, and it is a control result, not an evaluation result. Its
online counterparts are Q-learning with soft state aggregation
{cite}`SinghJaakkolaJordan1995` and Q-learning on interpolated action values
{cite}`SzepesvariSmart2004`, both convergent under the same sign condition,
and tree-based fitted Q-iteration with a frozen partition
{cite}`ernst2005tree`. For aggregation, {cite:t}`VanRoy2006` bounds the loss of
the greedy policy in terms of the variation of $v^*$ within each group,
weighted by the state distribution of an optimal policy. Approximate policy
iteration is also well behaved here. With aggregation, the approximate
evaluation is the exact evaluation of an aggregate MDP whose states are the
groups, so the policies are those of exact policy iteration on that MDP and do
not oscillate, a property that {cite:t}`Bertsekas2011` uses to recommend
aggregation over projected equations for approximate policy iteration.

The second group follows the $\xi$-norm route with a Galerkin projection.
There the optimality operator gives no contraction, and, as shown above, the
projected equation may have no solution at all; {cite:t}`deFariasVanRoy2000`
also show that a variant with exploration in the weighting always has one.
The positive results are special cases or weaker conclusions.

- **Optimal stopping** converges, by the argument above {cite}`TsitsiklisVanRoy1999`.
- **Q-learning with linear features** converges when the behavior policy is
  close enough to the greedy policy of every parameter it visits, a condition
  on feature covariances that {cite:t}`Melo2008` make precise. It holds
  near on-policy learning and fails in general off-policy.
- **SARSA(0)**, which evaluates the policy it follows and improves it at
  every step, cannot diverge with linear features: its weights stay in a
  bounded region, inside which they may chatter without converging
  {cite}`Gordon2000`.
- **Approximate policy iteration** with SARSA evaluation converges to a unique
  solution when the improvement step produces soft policies that are
  Lipschitz functions of the action values, with a small enough constant
  {cite}`PerkinsPrecup2002`. A softmax with a high enough temperature
  satisfies this condition and the greedy policy does not. The continuity of
  the improvement step is what removes the oscillation.
- **Gradient methods for control**, such as Greedy-GQ, minimize the projected
  Bellman error of the greedy policy by stochastic gradient and converge to a
  stationary point of that objective, not necessarily to a solution of the
  projected equation {cite}`Maei2010`.
- **Optimistic exploration** with a Gibbs policy guarantees that a solution of
  the projected equation for Q-learning exists and that the iterates stay
  bounded; convergence remains open {cite}`Meyn2023`.
- **Target networks with a ridge penalty** give linear off-policy Q-learning
  methods that converge under mild conditions on the behavior policy
  {cite}`ZhangYaoWhiteson2021`, discussed below.

The third group does not require $\Proj\Bellman$ to contract. It
assumes that each fit is accurate, $\|v_{k+1} - \Bellman v_k\| \leq \varepsilon$,
and bounds the result, as {prf:ref}`thm-approximate-vi` did for the sup norm.
The weighted-norm versions of {cite:t}`Munos2007` and
{cite:t}`MunosSzepesvari2008` cover fitted value iteration with the maximum,
for any regressor, at the price of the concentrability coefficient. The
stability question is now hidden in $\varepsilon$. The assumption is
reasonable when the function class is closed, or nearly closed, under the
Bellman operator, a condition called Bellman completeness
{cite}`ChenJiang2019`. If the class is closed, $\Bellman v_k$ is in it, the
projection returns it unchanged, and no norm needs to be preserved: the only
errors are statistical. The approximate policy iteration bound
$2\gamma\varepsilon/(1-\gamma)^2$ of {ref}`sec-abe-api`
{cite}`BertsekasTsitsiklis1996` is of the same kind, and it allows the policies
to oscillate within that distance of optimal.

The fourth group replaces the fixed-point equation by an optimization.
The linear programming approach to approximate dynamic programming
{cite}`deFariasVanRoy2003` restricts the linear program whose solution is
$v^*$ to the span of the features. The restricted program always has a
solution, no iteration can diverge, and its error is bounded by the best
approximation of $v^*$ in a norm weighted by a Lyapunov-like function. The
maximum enters as one linear constraint per state and action, which is why
the approach handles control as easily as evaluation.

| **Route** | **Covers the maximum?** | **Assumption** | **Conclusion** |
|:--|:--|:--|:--|
| Averager (sup norm) | Yes | Nonnegative fit weights | Contraction, unique fixed point, error bound |
| Galerkin ($\xi$-norm) | Only special cases | Stopping; behavior near greedy; soft Lipschitz improvement | Convergence, boundedness, or existence only |
| Error propagation | Yes | Each fit accurate (completeness), concentrability | Error bound, no convergence claim |
| Linear programming | Yes | Feasible restricted program | Error bound, no iteration |

The table summarizes what a practitioner of fitted Q-iteration can rely on.
With trees or aggregation, Gordon's theorem applies directly. With linear
features and a maximum, there are no general guarantees, only the special
cases of the second group. With a neural network, the third group is the
operative theory: the guarantee is conditional on the fits being accurate,
and the design choices discussed next try to keep them accurate.

### The Deadly Triad

{cite:t}`SuttonBarto2018` call the combination of function approximation,
bootstrapping (targets built from the current estimate), and off-policy data
the **deadly triad**: any two are manageable, all three can diverge. Fitted
Q-iteration with a neural network has all three. There is no general
convergence theorem for it, and divergence can be observed on small examples.
Yet NFQ and DQN solve many problems, thanks to design choices that limit the
damage. Each fit is run against targets held fixed, the fit is warm-started
from the previous parameters, a separate target network changes slowly, and a
large replay buffer keeps the data distribution from tracking every change of
policy. Recent analyses read these choices in the language of this section.
{cite:t}`ZhangYaoWhiteson2021` show that a target network, combined with a
ridge penalty on the parameters, gives a linear off-policy Q-learning method
that provably converges, and {cite:t}`Fellows2023` treat the target network as
a partially fitted iteration and identify conditions on the Jacobian of the TD
update under which it restores stability. {cite:t}`Gallici2025` show that
layer normalization in the value network makes TD stable without a target
network, even off-policy. The [chapter on logged and online data](fitted-q-iteration.md)
examines these choices one by one.

(sec-abe-api)=
## Approximate Policy Iteration

In the exact setting, Newton's method on the Bellman equation is policy
iteration ({ref}`sec-newton-kantorovich-pi`): by the envelope theorem the
Jacobian at $v_k$ involves the transition matrix of the greedy policy
$\pi_k$, and the Newton step solves the evaluation equation of $\pi_k$. The
previous chapter showed the same for collocation, where the Newton step
{eq}`eq:collocation-newton` is the evaluation equation of $\pi_k$ imposed at
the nodes and solved within the span of the basis. With an arbitrary
parametric value function the pattern is the same, except that the evaluation
is done within the function class by whatever means are available. The
algorithm alternates **approximate evaluation**, finding $\theta$ with
$v(\cdot;\theta) \approx v_{\pi_k}$, and **greedy improvement**. This is
approximate policy iteration. Its guarantee has the same form as for fitted
value iteration: if every evaluation is accurate to $\varepsilon$ in the sup
norm, the policies are eventually within $2\gamma\varepsilon/(1-\gamma)^2$ of
optimal {cite}`BertsekasTsitsiklis1996`. Unlike exact policy iteration, the
sequence of policies need not improve monotonically, and it can cycle among a
few policies. With the smooth Bellman operator, the Jacobian is
$\gamma\mathbf{P}_{\pi_k}$ for the softmax policy, with no ties to resolve
({ref}`sec-smooth-pi-newton`).

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
$\mathbf{A}\theta = \mathbf{b}$, and the Galerkin counterpart of the
collocation step {eq}`eq:collocation-newton`. Temporal-difference learning
with linear features, TD(0), converges to the same fixed point
{cite}`TsitsiklisVanRoy1997`, so TD and LSTD are two solvers of one Galerkin
equation, the first by stochastic approximation and the second by solving the
linear system. Neither was designed that way; the reading as projected
equations is due to Bertsekas {cite}`BertsekasTsitsiklis1996,Bertsekas2011`,
and the Galerkin method itself is standard in computational economics
{cite}`ndp`. The practical algorithm never
forms $\boldsymbol\Phi$ over all states or $\mathbf{P}_\pi$ explicitly. It
accumulates $\mathbf{A} = \mathbb{E}_{s \sim \xi}[\boldsymbol{\phi}(s)(\boldsymbol{\phi}(s) - \gamma \boldsymbol{\phi}(s'))^\top]$
and $\mathbf{b} = \mathbb{E}_{s \sim \xi}[\boldsymbol{\phi}(s) r]$ from
sampled transitions $(s, r, s')$, which is possible because the system is
linear in the unknown. On a 3-state chain under a fixed policy, with
$\mathbf{P}_\pi = \begin{pmatrix} 0.7 & 0.2 & 0.1 \\ 0.3 & 0.4 & 0.3 \\ 0.1 & 0.3 & 0.6 \end{pmatrix}$,
$\mathbf{r}_\pi = (1, 2, 0)$, features $\phi(s) = (1, s)$ and uniform weights:

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
at the greedy policy $\pi_k$, and the Newton step is

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

The closed forms need the features to enter linearly, and linear features
are rarely what practice uses. With a neural network the evaluation step has
no closed form, and the natural replacement is a few sweeps of fitted policy
evaluation, $q \leftarrow \Proj\BellmanPi q$, which is what the critic of
NFQCA does.

```{admonition} Exercises: LSTD and LSPI
:class: hint dropdown

1. **Computing the stationary distribution.** For the 3-state chain above, compute the stationary distribution $\xi$ satisfying $\xi^\top \mathbf{P}_\pi = \xi^\top$.

2. **LSTD with stationary weighting.** Recompute the LSTD solution using the stationary distribution instead of uniform weighting. Compare the approximation error.

3. **Two updates.** On a small MDP with two actions and linear features, implement fitted value iteration and LSPI {eq}`eq:lspi`. Compare the number of iterations, and find features under which LSPI cycles between two policies.
```

(sec-abe-roadmap)=
## From Fitted Value Iteration to FQI and NFQCA

Fitted value iteration, as stated here, assumes four things that NFQCA does
not, and each of the next chapters removes one of them.

First, it learns values of states. To act greedily with respect to
$v(\cdot;\theta)$ requires, at every decision, the expectation
$r(s,a) + \gamma\,\mathbb{E}[v(s';\theta) \mid s,a]$ for every action: the
model and an integral at decision time. The
[chapter on sampled Bellman updates](monte-carlo-bellman-estimation.md) moves that work into
training by learning the value of each action, $q(s,a;\theta)$, so that
acting is a maximization over the stored values and needs no model. The
Bellman operator on action values is again a $\gamma$-contraction, and
everything in this chapter applies to it unchanged. Fitted value iteration on
$q$ is fitted Q-iteration.

Second, it computes expectations exactly. The targets contain
$\mathbb{E}[\,\cdot \mid s,a]$, which the eight-dimensional example already
replaced by an average over draws. With a simulator, or with logged data, a
target typically uses a single sampled successor. The
[chapter on sampled Bellman updates](monte-carlo-bellman-estimation.md) studies what this
costs: variance, and a bias that comes from maximizing over noisy estimates.

Third, it chooses the states at which targets are computed. Batch reinforcement
learning receives a fixed set of transitions instead, and online
reinforcement learning a buffer that changes as the agent acts. The error
analysis of this chapter said that a fit is controlled only where there are
data. The [chapter on logged and online data](fitted-q-iteration.md) builds the
batch algorithms on this template, the tree-based FQI of {cite:t}`ernst2005tree`
(an averager) and NFQ {cite}`riedmiller2005neural` (a neural network), runs
them on a scheduling log and on ESA's collision warnings, and then develops
DQN, whose design choices respond to the deadly triad.

Fourth, it maximizes over a finite set of actions. Every target contains
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
| State values $v(s)$ | Action values $q(s,a)$, so acting needs no model | [Samples](monte-carlo-bellman-estimation.md), [Logged and online data](fitted-q-iteration.md) |
| Exact expectation in the target | One sampled successor per transition | [Samples](monte-carlo-bellman-estimation.md) |
| Targets at chosen states | A fixed batch (FQI, NFQ) or a replay buffer (DQN) | [Logged and online data](fitted-q-iteration.md) |
| Maximum over a finite action set | A learned actor in the target (NFQCA, DDPG, TD3) | [Amortized action optimization](amortized-action-optimization.md) |

## Exercises

These exercises complement the shorter ones inside the sections. The first
ones are short checks; {ref}`ex-abe-knn` reuses the code of the
eight-dimensional stopping problem.

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

:::{exercise} Reading the bounds
:label: ex-abe-bounds

Fitted value iteration with $\gamma = 0.9$ starts from $v_0 = 0$, and
$\|v^*\|_\infty = 10$. Every fit is accurate to $\varepsilon = 0.1$ in the sup
norm.

1. Bound $\|v_{50} - v^*\|_\infty$ with {prf:ref}`thm-approximate-vi`, and
   separate the part that more iterations remove from the part that they do
   not.
2. Bound the loss of the greedy policy of $v_{50}$ with
   {prf:ref}`prop-greedy-loss`.
3. The duck-harvest table reports policy losses four to five orders of
   magnitude below the second bound. Name the two places in the proofs where
   the worst case is charged, and say why neither happens there.
:::

:::{solution} ex-abe-bounds
:class: dropdown

1. $\|v_{50} - v^*\|_\infty \le 0.9^{50} \cdot 10 + 0.1/0.1 = 0.052 + 1$. Further iterations shrink the first term only; the second, $\varepsilon/(1-\gamma) = 1$, is the price of the fits and stays.
2. The greedy loss is at most $\tfrac{2\gamma}{1-\gamma} \cdot 1.052 \approx 18.9$, nearly twice $\|v^*\|_\infty$, so the bound says nothing useful here.
3. The value bound charges every fitting error at its worst state and lets the errors add in the same direction at every iteration. The greedy bound charges the worst value error twice, once for each of the two actions being compared. In the duck problem the fitting errors are much smaller than $\varepsilon$ at most states, they do not align over the 50-year horizon, and much of the error is common to all four packages, so it cancels in the comparison that decides the action.
:::

:::{exercise} Which fits are averagers
:label: ex-abe-which-averagers

For each fit, say whether it is an averager in the sense of
{prf:ref}`gordon-averager`, and justify your answer with the weights it puts
on the targets.

1. $k$-nearest neighbors with $k = 5$ and equal weights.
2. Kernel regression $\hat v(s) = \sum_i K(s, s_i) y_i / \sum_i K(s, s_i)$ with a Gaussian kernel.
3. Ordinary least squares on the features $(1, s)$ in one dimension.
4. A regression tree whose splits are chosen once, from the targets of the first iteration, and then frozen.
5. A neural network trained to convergence on squared error.
:::

:::{solution} ex-abe-which-averagers
:class: dropdown

1. Yes: weights $1/5$ on the five nearest training states, zero elsewhere.
2. Yes: the weights are positive and normalized to sum to one, and they depend only on the states.
3. No. The prediction at $s$ is $\sum_i w_i(s) y_i$ with $w_i(s) = \tfrac1m + (s - \bar s)(s_i - \bar s)/\sum_j (s_j - \bar s)^2$, which is negative for training states on the far side of the mean when $s$ is far from $\bar s$. The weights do sum to one, so constants are preserved, but monotonicity fails.
4. Yes, from the iteration at which the splits are frozen: each prediction is the average of the targets in a fixed leaf.
5. No. The prediction is not a fixed linear function of the targets at all, and nothing constrains its sign or normalization.
:::

:::{exercise} Fitting the backup or minimizing the residual
:label: ex-abe-brm

Return to the two-state counterexample: $v(1;\theta) = \theta$,
$v(2;\theta) = 2\theta$, every reward zero, state 1 moves to state 2 and state
2 stays put.

1. Bellman residual minimization chooses $\theta$ to minimize
   $\sum_s \big(v(s;\theta) - (\Bellman v(\cdot;\theta))(s)\big)^2$ with equal
   weights. Write this objective and show that its minimizer is $\theta = 0$
   for every $\gamma$.
2. Gradient descent on that objective uses the full gradient. Show that it
   converges for every $\gamma < 1$ with a small enough step, while fitted
   value iteration diverges for $\gamma > 5/6$.
3. Why is Bellman residual minimization rarely used despite this? Consider
   what an unbiased estimate of $\big(v(s;\theta) - \mathbb{E}[r + \gamma v(s';\theta)]\big)^2$
   requires when the successor is random, and connect it to the single-sample
   paradigm of the next chapter.
:::

:::{solution} ex-abe-brm
:class: dropdown

1. The objective is $(\theta - 2\gamma\theta)^2 + (2\theta - 2\gamma\theta)^2 = \theta^2\big[(1-2\gamma)^2 + 4(1-\gamma)^2\big]$, a parabola whose coefficient is positive for $\gamma < 1$. Its minimizer is $\theta = 0$.
2. Gradient descent on a positive-definite quadratic converges for any step size below $2/c$, where $c$ is its curvature. Fitted value iteration does not descend this objective: it holds the target fixed and fits it, which is the semi-gradient, and the semi-gradient map $\theta \mapsto \tfrac65\gamma\theta$ expands for $\gamma > 5/6$.
3. Squaring the residual of a single sampled successor gives $\mathbb{E}\big[(v(s) - r - \gamma v(s'))^2\big]$, which equals the squared residual plus the variance of $r + \gamma v(s')$. The minimizer is biased toward functions with small variance of the successor value. An unbiased estimate needs two independent successors of the same state, the double-sampling problem, which a log with one successor per transition cannot provide.
:::

:::{exercise} An averager on the eight-dimensional problem
:label: ex-abe-knn

In the code of the eight-dimensional stopping problem, add a fit by
$k$-nearest neighbors (`sklearn.neighbors.KNeighborsRegressor`) with
$k = 1$, $10$ and $50$, and record the largest change
$\max_i |v_{k+1}(s_i) - v_k(s_i)|$ at each iteration.

1. Check that the changes shrink by at least the factor $\gamma = 0.9$ per
   iteration, as {prf:ref}`santos-vigo-aguiar-stability` guarantees. Do they
   for the trees and the network?
2. Compare the mean and largest errors with those of the trees and the
   network at $D = 8$. Which value of $k$ is best, and why does a large $k$
   hurt most at the corners of the cube?
:::

:::{solution} ex-abe-knn
:class: dropdown

The changes shrink geometrically, by a factor of about $0.43$ to $0.55$ per
iteration, faster than $\gamma$, because the stopping reward caps the values,
and they reach $10^{-14}$ within the 40 iterations. With $k = 10$ at $D = 8$
the mean error is about $0.022$ and the largest about $0.155$, between the
trees ($0.031$) and the network ($0.013$) on average. Larger $k$ averages
targets over a larger neighborhood; at a corner of the cube all neighbors lie
on one side, so the prediction is pulled toward the interior, and the largest
error grows to about $0.2$ at $k = 50$. For the trees and the network the
measurement is worth making, because no theorem predicts its outcome: the
splits of the trees depend on the targets, and the network has no sign or
normalization constraint.
:::

:::{exercise} From value iteration to policy iteration, approximately
:label: ex-abe-mpi

Modified policy iteration ({prf:ref}`alg-modified-policy-iteration`) applies
$\BellmanPi$ $m$ times after each greedy improvement.

1. Write fitted modified policy iteration, in which each application of the
   operator is followed by a fit. Which values of $m$ give fitted value
   iteration and approximate policy iteration?
2. NFQCA performs one fitted evaluation sweep per improvement, with a learned
   actor in place of the greedy step. Which $m$ is this?
3. In the exact setting, larger $m$ means fewer improvements. Give one reason
   why a large $m$ may be a poor choice when every sweep includes a fit.
:::

:::{solution} ex-abe-mpi
:class: dropdown

1. For $j = 1, \ldots, m$: $q \leftarrow \Proj\mathrm{L}_{\pi_k} q$, then $\pi_{k+1}$ greedy for $q$. With $m = 1$ each sweep uses the operator of the policy that is greedy for the current values, which is fitted value iteration; $m \to \infty$ is approximate policy iteration with fitted policy evaluation as the evaluation step.
2. $m = 1$: NFQCA is the fitted value iteration end of the range, with the maximization in the target done by the actor.
3. Every fit adds an error, and for a nonlinear fit nothing guarantees that repeated fitted evaluation of a fixed policy converges; with off-policy data it can diverge, which is the deadly triad again. Many sweeps on a policy that is about to change also spend computation on accuracy that the next improvement discards.
:::

:::{exercise} Designing a convergent fitted iteration
:label: ex-abe-open

Pick a problem from an earlier chapter that has a continuous state: the
reservoir of the finite-horizon chapter, the storm-surge barrier, or the
fishery. Design a fitted value iteration for it that is guaranteed to
converge, run it, and compare its greedy policy with the exact one. Then
replace the fit by a neural network and report what changes: the number of
iterations, the stability of the iterates, and the policy loss. Explain the
differences with the results of this chapter.
:::

:::{solution} ex-abe-open
:class: dropdown

Any averager gives the guarantee: multilinear interpolation on a grid, state
aggregation, or $k$-nearest neighbors on sampled states. A report should state
the fit, the states where targets are computed, the tolerance, and the policy
loss measured against an exact or fine-grid solution. With the network,
expect the iterates to wander by an amount comparable to the fit error rather
than converge, as in the duck-harvest experiment, and a policy whose errors
concentrate where the data are sparse or two actions are close.
:::

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
iteration, with LSTD and LSPI as its linear case.

Exact target evaluation still assumes access to the transition probabilities
or an exact expectation. How can the same Bellman update be estimated when the
model supplies only samples? [Bellman updates from samples](monte-carlo-bellman-estimation.md)
replaces the exact integral by sampled averages and makes their variance and
maximization bias explicit.
