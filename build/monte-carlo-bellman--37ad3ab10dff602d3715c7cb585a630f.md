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

The [previous chapter](approximate-bellman-equations.md) studied function
iteration with a known model: when a fit preserves the contraction of the
Bellman operator, and how far the iteration ends from $v^*$ when each fit is
accurate. This chapter turns that iteration into a learning algorithm. Three
things change. The fit becomes any regressor, a tree ensemble or a neural
network, which turns function iteration into **fitted value iteration**, a
sequence of regression problems whose targets are Bellman backups. The
expectations in those targets become averages over sampled successors,
because a simulator or a log of transitions is often all there is. And the
values of states become values of actions, so that acting needs no model.
Run on a fixed log of transitions, the result is fitted Q-iteration, the
batch algorithm behind tree-based FQI and NFQ, and the starting point of DQN
in the [next chapter](fitted-q-iteration.md). Replacing exact expectations by
samples is what turns approximate dynamic programming into what the
operations research community calls simulation-based dynamic programming and
the machine learning community calls reinforcement learning.

Sampling has two costs that the chapter makes explicit. The error of a sample
average is random rather than systematic; its rate does not depend on the
dimension of the state, and it adds to the per-iteration error that the
previous chapter bounded. And the maximum over noisy estimates is biased
upward, even though each estimate is unbiased, so the targets of every
sampled Bellman update overestimate. Between the two, temporal-difference
learning solves the Galerkin system of the previous chapter one transition at
a time, and shows how sampling with the wrong weighting can make a well-posed
equation diverge. The chapter ends with fitted Q-iteration on two logs, one
from an inference-serving scheduler and one of the collision warnings that
ESA received for its satellites.

::::{admonition} Learning Goals
:class: note

After completing this chapter, you should be able to:
- Write fitted value iteration for an arbitrary regressor, and recognize collocation and Galerkin projection as two choices of the fitting rule
- Bound the error of fitted value iteration, and the loss of its greedy policy, by the error of each fit
- Locate the two expectations in a fitted Bellman update, inside the operator and over the states, and bound the effect of estimating either one on the fixed point
- Replace an expectation by a sample average, state the bias and variance of the result, and explain why its rate is independent of the dimension
- Define the action-value function and its Bellman operator, and explain what it saves at decision time
- Derive TD(0) as a stochastic solver of the Galerkin system, and explain why it is stable on-policy and can diverge off-policy
- Derive the upward bias of a maximum over noisy estimates, and explain how decoupling selection from evaluation removes part of it
- Write fitted Q-iteration on a fixed set of transitions, and attribute the errors of a fitted policy to approximation, where actions are nearly tied, or to coverage, where the data never tried an action

**Prerequisites:** the convergence results for approximate value iteration ([previous chapter](approximate-bellman-equations.md)), the weighted-residual template ([parametric value functions](weighted-residual-methods.md)), basic probability (expectation, variance, the central limit theorem, Jensen's inequality).
::::

(sec-abe-fvi)=
## Fitted Value Iteration with an Arbitrary Regressor

The averager condition is a property of how a fit maps targets to a
function, not of a basis: kernel smoothers and fixed-partition trees satisfy
it without one. The error bound of {ref}`sec-abe-error` below needs even less,
only that each fit be accurate. Nothing ties function iteration to a basis, then,
and the fit can be any regression procedure, including the tree ensembles and
neural networks of the chapters ahead.

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

the projected value iteration {eq}`eq:projected-vi` of the
[chapter on parametric value functions](weighted-residual-methods.md),
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
is run before the targets are recomputed.

The two fits of the chapter on parametric value functions mark the ends of a range. Collocation
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
[next chapter](fitted-q-iteration.md) studies what happens
when the fit is warm-started and stopped after a few gradient steps instead
of run to convergence.

(sec-abe-error)=
## Error Bounds When Every Fit Is Accurate

Every fit is only approximately right. The averagers of the
[previous chapter](approximate-bellman-equations.md#sec-abe-averagers) keep the
iteration a contraction, but a tree ensemble whose splits follow the targets,
or a neural network, is not an averager. This section bounds how far from
$v^*$ fitted value iteration ends with any fit, and how good the returned
policy is, assuming only that each fit is accurate.

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
   and the two-state example of the
   [previous chapter](approximate-bellman-equations.md#sec-abe-convergence) shows that this happens.

(sec-abe-mallard)=
### Example: Checking the Bounds on Duck-Hunting Regulations

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
the grid converges ({prf:ref}`santos-vigo-aguiar-stability`), and
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
dimensions, or when there is no model at all to compute targets from. This
chapter has an example of each: the eight-dimensional stopping problem of
{ref}`sec-abe-high-dim` and the satellite collision warnings of
{ref}`sec-fqi-conjunctions`.

- {download}`Mallard harvest model and solvers <code/mallard_harvest.py>`
- {download}`Mallard harvest experiment builder <scripts/build_mallard_harvest_artifacts.py>`
- {download}`Mallard harvest results <artifacts/mallard_harvest/metrics.json>`

(sec-mc-two-expectations)=
## Two Expectations in Every Fitted Update

Integrals appear in two places in a fitted Bellman update. The first is inside the operator: the transition
expectation $\int v(s')\,p(ds'|s,a)$ that every target contains. The second
is over the states: Galerkin and least-squares conditions integrate the
residual against test functions, and a regression averages the fitting error
over the states where targets were computed. Collocation has only the first
integral, which is one reason it was the simplest case.

Write $\BellmanQuad$ for the Bellman operator with the transition
expectation replaced by a numerical rule, deterministic or random, and let
fitted value iteration run with it, $v_{k+1} = \Proj\,\BellmanQuad v_k$. The
error after one iteration splits into three pieces by the triangle
inequality. With $v^\star = \Bellman v^\star$,

$$
\begin{aligned}
\|v^\star - v_{k+1}\| &= \|v^\star - \Proj \BellmanQuad v_k\| \\
&\le \underbrace{\|v^\star - \Proj \Bellman v^\star\|}_{\text{best approximation}} + \underbrace{\|\Proj \Bellman v^\star - \Proj \Bellman v_k\|}_{\text{contraction of the current error}} + \underbrace{\|\Proj \Bellman v_k - \Proj \BellmanQuad v_k\|}_{\text{integration error}} .
\end{aligned}
$$

The first term is the best the basis can do, the second shrinks with the
iterations when $\Proj\Bellman$ is a contraction, and the third is the error
of the integration rule, which does not shrink as the iterations proceed. In
the language of {prf:ref}`thm-approximate-vi`, the integration error is one
more contribution to the per-iteration error $\varepsilon$, and it is
amplified by the same $1/(1-\gamma)$. This is why the accuracy of the
integration rule should match the accuracy of the basis. A degree-10
Chebyshev basis represents a smooth value function to about $10^{-8}$, but
with a rectangle rule for the expectation the integration error, of order
$h^2$, dominates the sum, and the basis is wasted.

For low-dimensional problems with a smooth, explicit transition density,
deterministic quadrature is the right tool, and the rule is chosen to match
the basis: Gauss–Chebyshev nodes with Chebyshev polynomials, Gauss–Legendre
with Legendre, Gauss–Hermite with Hermite polynomials and Gaussian shocks,
midpoint or trapezoidal rules with piecewise-constant or piecewise-linear
elements. With orthogonal collocation the nodes of the basis double as the
nodes of the quadrature, and the expectations of the basis functions at the
successors of each node can be precomputed once into a matrix, so that each
iteration costs a matrix product and a maximization. The fishery used
Gauss–Hermite quadrature over its log-normal shock in this way. The same
pairing principle governed continuous-time trajectory optimization, where the
cost integral and the dynamics were discretized on the same nodes.

Deterministic quadrature needs three things: a state of low dimension, a
density that can be evaluated at the nodes, and a cheap model. The problems
this part of the book is heading for have none of them. The state of the
eight-dimensional stopping problem later in this section needs $N^8$ quadrature nodes for a product
rule with $N$ nodes per axis. A simulator provides draws of $s'$ and no
density. And a logged transition provides exactly one draw, which cannot be
chosen. Monte Carlo integration handles all three.

## Estimating Bellman Targets by Monte Carlo

Monte Carlo integration approximates an expectation by the average of
independent draws. For $J = \mathbb{E}[f(X)]$ with $X \sim p$, draw
$X^{(1)}, \ldots, X^{(N)} \sim p$ and form

$$
\hat{J}_N = \frac{1}{N}\sum_{n=1}^N f\big(X^{(n)}\big).
$$

Two properties of this estimator carry the whole chapter. It is unbiased,
$\mathbb{E}[\hat{J}_N] = J$, by linearity of expectation. And its variance is
$\sigma^2/N$ with $\sigma^2 = \mathrm{Var}[f(X)]$, by independence of the
draws, so by the central limit theorem

$$
\sqrt{N}\,(\hat{J}_N - J) \Longrightarrow \mathcal{N}(0,\sigma^2),
$$

and the error decays as $N^{-1/2}$. The rate is slow next to high-order
quadrature in one dimension, and it is the same in every dimension: Monte
Carlo pays in variance, never in an exponential number of nodes. It is also a
quadrature rule, with random nodes and equal weights $1/N$. Importance
sampling changes the distribution of the nodes and compensates with unequal
weights, and the view of Monte Carlo as randomized quadrature covers it too.

The cell below checks both properties on a Gaussian with mean $2$ and
standard deviation $3$, with 4,000 repetitions for each sample size. Before
looking at the figure, predict what multiplying $N$ by four does to the root
mean squared error.

```{code-cell} python
:tags: [hide-input]
:label: fig-mc-rate
:caption: The sample mean of $N$ draws from a Gaussian with mean 2 and standard deviation 3, over 4,000 repetitions for each $N$. (a) The average estimate stays at the true mean whatever $N$: the estimator is unbiased. (b) Its root mean squared error (dots) follows $\sigma/\sqrt{N}$ (line), a slope of $-1/2$ in log-log axes, so four times as many draws halve the error.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

code_directory = Path.cwd() / "code"
if str(code_directory) not in sys.path:
    sys.path.insert(0, str(code_directory))
from book_figure_theme import INK, MUTED, PALETTE, figure_style, show

rng = np.random.default_rng(2026)
true_mean, sigma, trials = 2.0, 3.0, 4000
sample_sizes = 2 ** np.arange(0, 11)
means, rmse = [], []
for n in sample_sizes:
    estimates = rng.normal(true_mean, sigma, size=(trials, n)).mean(axis=1)
    means.append(estimates.mean())
    rmse.append(np.sqrt(np.mean((estimates - true_mean) ** 2)))

with figure_style():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax.axhline(true_mean, color=MUTED, lw=.8, ls="--")
    ax.plot(sample_sizes, means, "o-", color=PALETTE["blue"], ms=3.5, lw=1)
    ax.text(1.2, true_mean + .045, "true mean", color=MUTED, fontsize=8.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 256, 1024], ["1", "4", "16", "64", "256", "1024"])
    ax.set_ylim(true_mean - .15, true_mean + .15)
    ax.set_xlabel("draws $N$")
    ax.set_ylabel("average estimate")
    ax.set_title("(a) No bias at any $N$")

    bx.plot(sample_sizes, sigma / np.sqrt(sample_sizes), color=INK, lw=1)
    bx.plot(sample_sizes, rmse, "o", color=PALETTE["blue"], ms=3.5)
    bx.text(40, 1.0, r"$\sigma/\sqrt{N}$", color=INK, fontsize=9)
    bx.set_xscale("log", base=2)
    bx.set_yscale("log")
    bx.set_xticks([1, 4, 16, 64, 256, 1024], ["1", "4", "16", "64", "256", "1024"])
    bx.set_xlabel("draws $N$")
    bx.set_ylabel("root mean squared error")
    bx.set_title(r"(b) Error falls as $N^{-1/2}$")
    fig.tight_layout(w_pad=2)
show(fig)
```

### The Sampled Bellman Operator

For a fixed value function $v$ and a state-action pair $(s,a)$, the
transition part of the Bellman operator is the expectation
$\mathbb{E}[v(S') \mid S = s, A = a]$. If next states
$S'^{(1)}, \ldots, S'^{(N)}$ can be drawn from $p(\cdot \mid s,a)$, by calling
a simulator or by acting in the real system, the sample average estimates it.
When the reward is random as well, the draws are pairs
$(r^{(n)}, S'^{(n)})$, and the one-step return is estimated by

$$
\widehat{G}_N(s,a) = \frac{1}{N}\sum_{n=1}^N \Big[r^{(n)} + \gamma\, v\big(S'^{(n)}\big)\Big],
$$

which is unbiased for $\mathbb{E}[r + \gamma v(S') \mid s, a]$ as long as
$v$ is fixed. Substituting it into the operator gives the **Monte Carlo
Bellman operator**

$$
(\widehat{\Bellman}_N v)(s) = \max_{a \in \mathcal{A}_s} \widehat{G}_N(s,a).
$$

Nothing about approximation or projection has entered yet. For a fixed $v$,
Monte Carlo provides unbiased, noisy evaluations of $(\Bellman v)(s)$ at any
state, from draws alone. The maximum over the noisy estimates, however, is
not unbiased, and the last section of the chapter is about that.

### Choosing the States at Which Targets Are Computed

The second expectation, over the states, is estimated the same way. Galerkin
and least-squares conditions have the form
$\int R(s;\theta)\, p_i(s)\, \xi(ds) = 0$ for a distribution $\xi$ over
states, and the weighted norm of the residual is $\int R(s;\theta)^2\,\xi(ds)$.
The weight $w$ of the previous chapters could be any positive function; for
sampling it must be a probability distribution that can be drawn from, which
is $\xi$ after normalization, or any other distribution with importance
weights $w/\xi$. If states $S^{(1)}, \ldots, S^{(M)}$ are drawn from $\xi$,
each condition becomes a sample average,

$$
\int R(s;\theta)\, p_i(s)\, \xi(ds) \approx \frac{1}{M}\sum_{m=1}^M R\big(S^{(m)};\theta\big)\, p_i\big(S^{(m)}\big),
$$

and the least-squares objective becomes the empirical risk
$\frac{1}{M}\sum_m R(S^{(m)};\theta)^2$. This is the regression of
{ref}`sec-abe-fvi`: targets at sampled states, and a fit that minimizes the
average error over them. In reinforcement learning $\xi$ is rarely chosen; it
is the distribution of the states that the data happen to contain, which the
previous chapter showed to matter for convergence and the batch experiments of
{ref}`sec-fqi-batch` show to matter for coverage.

With both expectations sampled, the fitted update is

$$
v_{k+1} = \Proj_M\, \widehat{\Bellman}_N\, v_k,
$$

a fit at $M$ sampled states of targets that each average $N$ sampled
successors. It needs no transition density and no state distribution in
closed form, only the ability to draw from them. In the typical
reinforcement-learning setting $N = 1$: each recorded transition
$(s_i, a_i, r_i, s'_i)$ supplies one successor, and the target
$r_i + \gamma v(s'_i)$ has the full variance of a single draw. The variance
is then reduced by the fit, which pools information across neighboring
states, and by the number of transitions rather than by averaging at a fixed
$(s,a)$. The section after next returns to this choice.

(sec-abe-high-dim)=
### Example: A Stopping Problem in Eight Dimensions

The fishery of the earlier chapters has one state variable, and a grid of a few thousand nodes
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
sampled Bellman operator above is this approximation.
The errors are measured against the exact $v^*$ at 5,000 fresh states.

```{code-cell} python
:label: fig-fvi-eight-dimensions
:caption: Fitted value iteration on the stopping problem with an eight-dimensional state. Each dot is a test state, placed by the average of its coordinates and the value the network assigns to it; the black curve is the exact value function, which depends on the state only through that average. The network was never told so. It was fitted at 2,000 sampled states, while a grid with 11 nodes per axis would have $2.1 \times 10^8$.
:tags: [hide-input]
import sys
from pathlib import Path

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

if str(Path.cwd() / "code") not in sys.path:
    sys.path.insert(0, str(Path.cwd() / "code"))
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
analysis below and the batch methods of
{ref}`sec-fqi-batch`: a fit is controlled
where there are data, and only there.

(sec-mc-q-functions)=
## Action Values: Acting Without a Model

The greedy policy of a value function $v$ is

$$
\pi(s) = \arg\max_{a \in \mathcal{A}_s} \Big\{ r(s,a) + \gamma \int v(s')\,p(ds'|s,a) \Big\},
$$

one transition expectation per action at every decision. With a model and a
quadrature rule this is a computation; with a simulator it is a Monte Carlo
estimate per action, repeated each time the state is visited; with a log it
is impossible, because no new successors can be drawn at decision time. The
expectation can instead be computed once, during training, and stored. The
**action-value function** of $v$,

$$
q(s,a) = r(s,a) + \gamma \int v(s')\,p(ds'|s,a),
$$

caches the backup of every action, and the greedy policy becomes
$\pi(s) = \arg\max_{a} q(s,a)$, a maximization over stored values with no
integral and no model. The optimal action-value function $q^\star$ satisfies
its own Bellman equation, obtained by substituting $v^\star(s') = \max_{a'}
q^\star(s',a')$:

$$
q^\star(s,a) = r(s,a) + \gamma \int p(ds'|s,a)\, \max_{a' \in \mathcal{A}_{s'}} q^\star(s',a').
$$

The right-hand side defines the Bellman operator on action-value functions,
$(\Bellman q)(s,a) = r(s,a) + \gamma\,\mathbb{E}[\max_{a'} q(s',a') \mid s,a]$,
and it is a $\gamma$-contraction in the sup norm by the same argument as for
$v$: the maximum over $a'$ does not increase sup-norm distances, and the
expectation does not either. Everything in the previous chapter transfers.
Fitted value iteration on $q$ is **fitted Q-iteration**, its error bounds
hold with $(s,a)$ in place of $s$, averagers remain safe, and the function
class now takes the action as an input or has one output per action. What
changes is the cost accounting: the expectations are evaluated during the
iterations, at the state-action pairs where targets are computed, and acting
afterwards is cheap.

```{prf:algorithm} Fitted Q-Iteration with a Model
:label: simadp-parametric-q-value-iteration

**Input** Given an MDP $(S, A, P, R, \gamma)$, base points $\mathcal{B} \subset S$, function approximator class $q(s,a; \boldsymbol{\theta})$, maximum iterations $N$, tolerance $\varepsilon > 0$, initial parameters $\boldsymbol{\theta}_0$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ (e.g., random initialization)
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D} \leftarrow \emptyset$
    2. For each $(s,a) \in \mathcal{B} \times A$:
        1. $y_{s,a} \leftarrow r(s,a) + \gamma \int p(ds'|s,a)\max_{a' \in A} q(s',a'; \boldsymbol{\theta}_n)$
        2. $\mathcal{D} \leftarrow \mathcal{D} \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}, \boldsymbol{\theta}_{\text{init}}=\boldsymbol{\theta}_0, K=\infty)$
    4. $\delta \leftarrow \frac{1}{|\mathcal{D}||A|}\sum_{(s,a) \in \mathcal{D} \times A} (q(s,a; \boldsymbol{\theta}_{n+1}) - q(s,a; \boldsymbol{\theta}_n))^2$
    5. $n \leftarrow n + 1$
4. **until** ($\delta < \varepsilon$ or $n \geq N$)
5. **return** $\boldsymbol{\theta}_n$
```

The expectation in the target can be computed by quadrature when the model
allows it, or estimated by $N$ draws of the successor from $p(\cdot|s,a)$,
averaging $\max_{a'} q(s'_i, a')$ over them. The fitting step is the one of
{ref}`sec-abe-fvi`, with its initialization $\boldsymbol\theta_{\text{init}}$
and its optimization budget $K$ written out because the
[next chapter](fitted-q-iteration.md) varies both.

### One Sampled Successor per Target

Most reinforcement learning algorithms do not draw $N$ successors per
state-action pair. They work with a data set of **transition tuples**
$\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^M$, each recording the one
successor that was observed when $a_i$ was executed in $s_i$, and the target
of a tuple is $y_i = r_i + \gamma \max_{a'} q(s'_i, a')$, with $N = 1$.

This design has two origins. In some settings it is a constraint. A robot
that executes an action observes one outcome and cannot rewind to the same
state to sample another; the same holds for a clinical trial, a
manufacturing process, or any interaction that costs something. Many settings
in which reinforcement learning is applied have no such constraint. Game
emulators save and restore states, physics simulators reset to any
configuration, and software environments are cloned for free. The second
origin is historical. Reinforcement learning grew out of models of animal
learning, in which an organism experiences one consequence per action and
learns from a stream of experience, and temporal-difference learning,
Q-learning and the actor-critic methods were designed for that stream.

The convention persisted where the constraint did not, and it shaped the
algorithms. They store and manipulate tuples rather than state-action pairs
with several successors. They reduce variance through the function
approximator, which pools targets from nearby states, and through the number
of transitions, rather than by averaging at a fixed $(s,a)$. And they treat
the two expectations asymmetrically: the outer one, over states, is sampled
with whatever budget is convenient, a whole data set, a replay buffer of a
million transitions, or one transition per step, while the inner one is
sampled once. In a domain with a cheap, resettable simulator and
high-variance targets, averaging several successors per $(s,a)$ is a direct
variance reduction that current practice rarely uses.

(sec-mc-td)=
## Temporal-Difference Learning: One Transition per Update

The sampled fitted update above draws a batch of
transitions, computes targets, and fits. Temporal-difference learning takes
the opposite extreme, one transition at a time, with a gradient step in place
of the fit. For policy evaluation with linear features
$v(s;\theta) = \boldsymbol\phi(s)^\top\theta$, its target is the
Galerkin system {eq}`eq:abe-td-system` of the
[previous chapter](approximate-bellman-equations.md#sec-abe-galerkin-system),
$\mathbf{A}\theta = \mathbf{b}$ with
$\mathbf{A} = \boldsymbol\Phi^\top\boldsymbol\Xi(\mathbf{I} - \gamma\mathbf{P}_\pi)\boldsymbol\Phi$
and $\mathbf{b} = \boldsymbol\Phi^\top\boldsymbol\Xi\,\mathbf{r}_\pi$, whose
solution is the fixed point of the projected Bellman equation
$v = \Proj\BellmanPi v$. Both $\mathbf{A}$ and $\mathbf{b}$ are expectations
over a state drawn from $\xi$ and a successor drawn from $\mathbf{P}_\pi$, the
two expectations of {ref}`sec-mc-two-expectations`, so each transition gives an
unbiased estimate of them.

TD(0) draws a state $s$ from $\xi$, a successor $s'$ from $\mathbf{P}_\pi$ and
the reward $r$, and moves the parameters along the temporal difference,

$$
\theta \leftarrow \theta + \alpha\, \boldsymbol\phi(s)\big(r + \gamma\boldsymbol\phi(s')^\top\theta - \boldsymbol\phi(s)^\top\theta\big).
$$

The expected increment is $\alpha(\mathbf{b} - \mathbf{A}\theta)$, the
residual of {eq}`eq:abe-td-system`. The theory of stochastic approximation
{cite}`kushner2003stochastic` says that, with step sizes that decrease
suitably, the iterates follow the ordinary differential equation
$\dot\theta = \mathbf{b} - \mathbf{A}\theta$, which converges to the solution
of {eq}`eq:abe-td-system` from every start exactly when every eigenvalue of
$\mathbf{A}$ has a positive real part. Fitted policy evaluation in the same
coordinates is
$\theta_{k+1} = \theta_k + (\boldsymbol\Phi^\top\boldsymbol\Xi\boldsymbol\Phi)^{-1}(\mathbf{b} - \mathbf{A}\theta_k)$,
one step of the same expected update with the step size replaced by the
inverse of the mass matrix. The two algorithms share their fixed point and
their matrix, and differ in how they step toward it. LSTD
({ref}`sec-abe-lstd`) does not step at all: it estimates $\mathbf{A}$ and
$\mathbf{b}$ from the transitions and solves the system.

{prf:ref}`prop-td-positive-definite` of the previous chapter showed that
on-policy, with $\xi$ the stationary distribution of $\mathbf{P}_\pi$,
$\mathbf{A}$ is positive definite. Its eigenvalues then have positive real
parts, and TD(0) converges to the solution of the projected Bellman equation
{cite}`TsitsiklisVanRoy1997`. The same inequality makes fitted evaluation a
contraction and the LSTD system uniquely solvable: on-policy, the three
solvers are stable for the same reason.

The analysis extends to TD($\lambda$), which bootstraps from a geometric
mixture of multi-step returns. Its fixed point solves
$v = \Proj\BellmanPi^{(\lambda)}v$ with the multi-step operator
$\BellmanPi^{(\lambda)} = (1-\lambda)\sum_{k \geq 0}\lambda^k\BellmanPi^{k+1}$,
which is a contraction in $\|\cdot\|_\xi$ with modulus
$\alpha_\lambda = \gamma(1-\lambda)/(1-\gamma\lambda)$, and the fixed point
satisfies
$\|v_\lambda - v_\pi\|_\xi \leq (1 - \alpha_\lambda^2)^{-1/2}\|\Proj v_\pi - v_\pi\|_\xi$
{cite}`BertsekasTsitsiklis1996,Bertsekas2011`. At $\lambda = 1$ the
modulus is zero, the targets are Monte Carlo returns, and the fixed point is
the best approximation $\Proj v_\pi$; at $\lambda = 0$ the constant is the
$(1-\gamma^2)^{-1/2}$ of the previous chapter. Bertsekas and Yu developed this projected-equation view into a
general method for large linear systems, with error bounds that are sharper
than the contraction argument because they use the data to estimate how
$\Proj$ and $\mathbf{P}_\pi$ interact {cite}`BertsekasYu2009,YuBertsekas2010,Bertsekas2011`.

The two conditions, $\mathbf{A}$ with eigenvalues in the right half plane for
TD and a spectral radius of $\gamma\Proj\mathbf{P}_\pi$ below one for fitted
evaluation, are different conditions on the same matrices. In the two-state
counterexample of the previous chapter they coincide. There is one feature, so $\mathbf{A}$ is the
scalar $\xi_1(1 - 2\gamma) + 4\xi_2(1-\gamma)$, and the fitted iteration
multiplies $\theta$ by $1 - \mathbf{A}/(\xi_1 + 4\xi_2)$, which is the factor
$6\gamma/5$ of the previous chapter when $\xi_1 = \xi_2 = 1/2$. Both fail when
$\mathbf{A} < 0$. With uniform weights that happens for $\gamma > 5/6$, and
with the stationary weighting $\xi = (0, 1)$ it never happens. The code below
runs sampled TD(0) on the example with $\gamma = 0.9$ and three weightings.

```{code-cell} python
:tags: [hide-input]

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "code"))
import projection_stability as ps

print(f"{'weight of state 1':>18} {'fitted factor':>14} {'A':>8} {'TD theta after 400 steps':>26}")
for xi1 in (0.5, 0.2, 0.0):
    factor, A = ps.two_state(xi1, gamma=0.9)
    print(f"{xi1:>18.1f} {factor:>14.3f} {A:>8.3f} {ps.two_state_td(xi1, gamma=0.9):>26.3g}")
```

TD started at $\theta = 1$. Under uniform weighting it has grown by a factor
of about sixteen after 400 updates, and with the weight of state 1 reduced to
$0.2$ or to the stationary value $0$ it has shrunk toward the solution
$\theta = 0$. The sign of $\mathbf{A}$ predicts both outcomes.

(sec-mc-off-policy)=
### Off-Policy Evaluation and Its Repairs

When the states are weighted by a distribution $\xi$ that is not stationary
for $\mathbf{P}_\pi$, which is the situation of learning about one policy from
the data of another, the Jensen step of the on-policy argument fails, and with
it {prf:ref}`prop-td-positive-definite`. The previous chapter noted that the
fixed point itself can then be far from $v_\pi$ {cite}`Kolter2011`. The
solver can fail as well: TD(0) can diverge even when the projected equation
is perfectly well posed.

The classical example is Baird's star {cite}`Baird1995`, in the form given by
{cite:t}`SuttonBarto2018`. There are seven states and eight weights: state $i
\leq 6$ has value $2\theta_i + \theta_8$, and state 7 has value
$\theta_7 + 2\theta_8$. The target policy moves every state to state 7, the
behavior policy moves to a uniformly chosen state, all rewards are zero, and
$\gamma = 0.99$. The features span all of $\mathbb{R}^7$, so $\Proj$ is the
identity, $\Proj\BellmanPi = \BellmanPi$ is a contraction, and fitted policy
evaluation converges in any norm. The difficulty lies entirely in the solver.
The cell below iterates the expected updates of three methods from Sutton and
Barto's starting point, with the states weighted uniformly as the behavior
policy visits them, and reports the largest $|v(s;\theta)|$ along the way; the
true value is zero.

```{code-cell} python
:tags: [hide-input]

out = ps.baird_table()
print(f"rank of the features: {out['rank of Phi']}")
print(f"xi-norm of gamma P_pi under uniform weights: {out['xi-norm of gamma P_pi']:.2f}")
print(f"smallest real part of an eigenvalue, TD matrix:       {out['td']['min Re eig']:.3f}")
print(f"smallest real part of an eigenvalue, emphatic matrix: {abs(out['emphatic']['min Re eig']):.3f}")
print(f"{'method':<10} {'max |v| at 0':>14} {'at 500':>12} {'at 2000':>12}")
for m in ("td", "tdc", "emphatic"):
    v0, v500, v2000 = out[m]["max |v| at 0, 500, 2000"]
    print(f"{m:<10} {v0:>14.3g} {v500:>12.3g} {v2000:>12.3g}")
```

Under the uniform weighting, $\gamma\mathbf{P}_\pi$ has $\xi$-norm about
$2.6$, so it is not a contraction in that norm, and the TD matrix has an eigenvalue with negative real
part. Expected TD(0) diverges, and its values reach the thousands after 2000
updates although every quantity in the problem is bounded. This is the first
leg of the deadly triad, with no maximum and no neural network: linear
features, bootstrapping, and off-policy weighting suffice.

Three families of repairs keep the projected equation and change the
solver or the weighting.

1. **Importance sampling** reweights each update by the ratio of the
   target and behavior probabilities of the action taken
   {cite}`Precup2001`. This makes the expected update use the target
   policy's transitions, but not the target policy's state distribution, so
   the matrix $\mathbf{A}$ above, which already uses $\mathbf{P}_\pi$, is
   unchanged; ratios over whole trajectories correct the distribution too, at
   the price of a variance that can grow exponentially with the horizon.
2. **Gradient TD** methods minimize the mean squared projected Bellman error
   $\|\Proj(\BellmanPi v_\theta - v_\theta)\|_\xi^2$, which is zero exactly
   when the Galerkin conditions hold, by true stochastic gradient descent with
   a second set of weights that estimates $\Proj(\BellmanPi v_\theta - v_\theta)$
   {cite}`Sutton2009`. The objective has the TD fixed point as its minimizer
   for any weighting, and gradient descent on it is stable. In the cell, TDC,
   the variant called TD with gradient correction in that paper, brings the
   values down from 12 to about 2 within 500 updates and then stalls there:
   its stability is bought with slow progress, a common observation with
   these methods.
3. **Emphatic TD** reweights the update at each state by a follow-on trace,
   which on average replaces $\xi$ in $\mathbf{A}$ by the emphasis
   $f = (\mathbf{I} - \gamma\mathbf{P}_\pi^\top)^{-1}\xi$
   {cite}`SuttonMahmoodWhite2016`. The resulting matrix
   $\boldsymbol\Phi^\top\mathbf{F}(\mathbf{I} - \gamma\mathbf{P}_\pi)\boldsymbol\Phi$
   with $\mathbf{F} = \text{diag}(f)$ is positive definite for any
   behavior when the features are independent, by an argument close to that
   of {prf:ref}`prop-td-positive-definite`, and the method converges
   {cite}`Yu2015`. Baird's features are not independent, and the matrix is
   only semidefinite, which the zero printed in the cell reflects. In the cell, its values go to zero. The fixed point is the
   solution of a projected equation with the emphasis weighting, not with
   $\xi$, so it is a different approximation of $v_\pi$.

LSTD avoids the stability question altogether, because it solves
$\mathbf{A}\theta = \mathbf{b}$ directly. Off-policy, it computes the same
fixed point as TD, with the same possible loss of accuracy, whenever
$\mathbf{A}$ is invertible {cite}`BertsekasYu2009`.


(sec-mc-deadly-triad)=
### The Deadly Triad, and What Is Known with the Maximum

{cite:t}`SuttonBarto2018` call the combination of function approximation,
bootstrapping (targets built from the current estimate) and off-policy data
the **deadly triad**: any two are manageable, all three can diverge. Baird's
star has all three, without a maximum. Q-learning adds the maximum, and with
it a target policy that changes with the parameters. What is known for
sampled updates with the maximum follows the two routes of the
[previous chapter](approximate-bellman-equations.md#sec-abe-two-norms).

On the sup-norm route, the averager argument survives sampling. Q-learning
with soft state aggregation {cite}`SinghJaakkolaJordan1995` and Q-learning on
action values interpolated from a finite set of states
{cite}`SzepesvariSmart2004` converge, under the same sign condition on the
weights as in the deterministic case.

On the weighted route, the results are special cases or weaker conclusions.

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
  satisfies this condition and the greedy policy does not.
- **Gradient methods for control**, such as Greedy-GQ, minimize the projected
  Bellman error of the greedy policy by stochastic gradient and converge to a
  stationary point of that objective, not necessarily to a solution of the
  projected equation {cite}`Maei2010`.
- **Optimistic exploration** with a Gibbs policy guarantees that a solution of
  the projected equation for Q-learning exists and that the iterates stay
  bounded; convergence remains open {cite}`Meyn2023`.

Error bounds for fitted iterations on sampled data add a statistical error to
the bounds of {ref}`sec-abe-error`,
as in the finite-sample analysis of fitted value iteration by
{cite:t}`MunosSzepesvari2008`. They need the function class to be closed, or
nearly closed, under the Bellman operator, a condition called Bellman
completeness {cite}`ChenJiang2019`. The design choices that keep neural
fitted Q-iteration stable in practice, target networks among them, are the
subject of the [next chapter](fitted-q-iteration.md).

```{admonition} Exercises: Temporal-Difference Learning
:class: hint dropdown

1. **One feature, one condition.** In the two-state counterexample of the [previous chapter](approximate-bellman-equations.md#sec-abe-convergence), show that fitted evaluation converges exactly when expected TD(0) is stable. Then construct a two-feature example in which one is stable and the other is not. *Hint: the fitted iteration is a preconditioned TD step of size one.*

2. **How far off-policy?** In Baird's star, compute the eigenvalues of $\mathbf{A}$ as a function of $\gamma$ and find the discount below which none has a negative real part. Then mix the uniform weighting with the stationary distribution of the target policy, $\xi_\beta = (1-\beta)\,\text{uniform} + \beta\,\mathbf{e}_7$, and find the smallest $\beta$ for which none has a negative real part at $\gamma = 0.99$.
```

(sec-mc-overestimation)=
## Maximization Bias in Sampled Targets

Each estimate $\widehat{G}_N(s,a)$ is unbiased, and the maximum over them is
not. Fix a state $s$ and a value function $v$, and write the true continuation
value and its estimate as

$$
\mu(s,a) = \int v(s')\,p(ds' \mid s,a), \qquad \hat{\mu}_N(s,a) = \frac{1}{N}\sum_{i=1}^N v(s'_i), \quad s'_i \sim p(\cdot|s,a),
$$

so that $(\widehat{\Bellman}_N v)(s) = \max_a \{r(s,a) + \gamma \hat{\mu}_N(s,a)\}$.
Let $a^\star$ be the action that maximizes the true backup. The maximum of a
collection is at least as large as any of its members, so

$$
\mathbb{E}\big[\max_a \{r(s,a) + \gamma \hat{\mu}_N(s,a)\}\big] \ge \mathbb{E}\big[r(s,a^\star) + \gamma \hat{\mu}_N(s,a^\star)\big] = r(s,a^\star) + \gamma \mu(s,a^\star) = (\Bellman v)(s).
$$

The inequality is strict whenever more than one action has a nonzero
variance: the maximization picks whichever action received a positive noise,
and that inflated estimate becomes the target. This is Jensen's inequality
for the maximum, $\mathbb{E}[\max_a Y_a] \ge \max_a \mathbb{E}[Y_a]$, since
the maximum is convex. The same inequality is the reason that deterministic
policies are optimal in MDPs ({prf:ref}`stoch-policy-reduction`), where it
says that randomizing over actions cannot beat the best one; here it says
that maximizing over noisy estimates cannot be unbiased.

The bias does not average out with more draws of the same estimate, and it
compounds. Monte Carlo value iteration $v_{k+1} = \widehat{\Bellman}_N v_k$
builds its targets at iteration $k+1$ from a $v_k$ that is already biased
upward, so $\mathbb{E}[v_k] \ge \mathbb{E}[v_{k-1}] \ge \cdots \ge v^\star$,
and the inflated values feed back into the next maximization. The cell below
holds every true action value at zero, draws five noisy samples per action,
and measures the bias of the maximum as the number of actions grows, next to
the estimator of the next subsection.

```{code-cell} python
:tags: [hide-input]
:label: fig-mc-max-bias
:caption: Bias of two estimators of $\max_a \mu_a$ when every true action value $\mu_a$ is zero and each is estimated from five standard Gaussian draws, over 6,000 repetitions. The maximum of the estimates (vermilion) is biased upward, and more so with more actions. Selecting the action with one set of draws and evaluating it with an independent set (blue) removes the bias in this case, where all actions are equally good. Two standard errors of each estimate are smaller than the dots.

rng = np.random.default_rng(2026)
replications, samples_per_action = 6000, 5
action_counts = [2, 4, 8, 16, 32]
bias = {"maximum": [], "double": []}
se = {"maximum": [], "double": []}
for actions in action_counts:
    selection = rng.normal(size=(replications, actions, samples_per_action)).mean(axis=2)
    evaluation = rng.normal(size=(replications, actions, samples_per_action)).mean(axis=2)
    chosen = selection.argmax(axis=1)
    for name, values in (("maximum", selection.max(axis=1)),
                         ("double", evaluation[np.arange(replications), chosen])):
        bias[name].append(values.mean())
        se[name].append(values.std(ddof=1) / np.sqrt(replications))

with figure_style():
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.axhline(0, color=MUTED, lw=.8)
    for name, color in (("maximum", PALETTE["vermilion"]), ("double", PALETTE["blue"])):
        ax.errorbar(action_counts, bias[name], yerr=2 * np.array(se[name]), fmt="o-", color=color,
                    ms=3.5, lw=1, capsize=0, elinewidth=1)
    ax.text(32, bias["maximum"][-1] - .06, "maximum of the\nestimates", color=PALETTE["vermilion"],
            fontsize=8.5, ha="right", va="top")
    ax.text(32, .06, "select with one estimate,\nevaluate with the other", color=PALETTE["blue"],
            fontsize=8.5, ha="right", va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks(action_counts, [str(a) for a in action_counts])
    ax.set_xlabel("number of actions")
    ax.set_ylabel("bias")
show(fig)
```

### Double Estimators: Separating Selection from Evaluation

The bias comes from using one noisy estimate for two jobs. Decompose the
estimate into its mean and its noise, $\hat{\mu}_N(s,a) = \mu(s,a) + \varepsilon_a$
with $\mathbb{E}[\varepsilon_a] = 0$. The standard target is

$$
Y = r(s,\hat a) + \gamma \mu(s,\hat a) + \gamma \varepsilon_{\hat a}, \qquad \hat a = \arg\max_a \{r(s,a) + \gamma \mu(s,a) + \gamma \varepsilon_a\},
$$

and the same $\varepsilon_{\hat a}$ that made $\hat a$ win is added to its
value. Double Q-learning {cite}`van2016deep` breaks this coupling with two
independent estimates of every action, $\hat{\mu}^{(1)}_N(s,a) = \mu(s,a) +
\varepsilon^{(1)}_a$ and $\hat{\mu}^{(2)}_N(s,a) = \mu(s,a) + \varepsilon^{(2)}_a$,
with $\varepsilon^{(1)}$ and $\varepsilon^{(2)}$ independent. The first
selects and the second evaluates:

$$
\hat a = \arg\max_{a} \big\{r(s,a) + \gamma \hat{\mu}^{(1)}_N(s,a)\big\}, \qquad Y = r(s,\hat a) + \gamma \hat{\mu}^{(2)}_N(s,\hat a).
$$

The selected action $\hat a$ is random, a function of $\varepsilon^{(1)}$
alone. Conditioning on it,

$$
\mathbb{E}[Y] = \mathbb{E}_{\varepsilon^{(1)}}\Big[\mathbb{E}_{\varepsilon^{(2)}}\big[r(s,\hat a) + \gamma \mu(s,\hat a) + \gamma \varepsilon^{(2)}_{\hat a} \mid \hat a\big]\Big] = \mathbb{E}_{\varepsilon^{(1)}}\big[r(s,\hat a) + \gamma \mu(s,\hat a)\big],
$$

because $\varepsilon^{(2)}$ is independent of $\hat a$, so that
$\mathbb{E}[\varepsilon^{(2)}_{\hat a} \mid \hat a] = \mathbb{E}[\varepsilon^{(2)}_{\hat a}] = 0$.
The evaluation noise contributes nothing on average. What remains is the
**selection bias**: $\hat a$ is more often an action whose $\varepsilon^{(1)}$
was positive than one whose true value is highest, so
$\mathbb{E}[\mu(s,\hat a)] \le \max_a \mu(s,a)$, and the double estimator is
biased downward. In {numref}`fig-mc-max-bias` the maximum estimator grows
more optimistic as the number of actions grows, while the double estimator
has no bias: every action has the same true value there, so selecting the
wrong one costs nothing and the selection bias is zero. When the true values
differ, the double estimator is biased downward instead, by much less than
the maximum is biased upward. Removing the **evaluation bias**, the reuse of
the winning noise, removes most of the overestimation.

```{prf:remark} Two Q-functions instead of two sets of draws
:class: dropdown

The argument assumes two independent sets of draws from $p(\cdot|s,a)$ for every state-action pair, which a log never provides and a simulator provides only by resetting. The practical implementation keeps two Q-functions instead, trained on different data or updated at different times, for instance an online network and a slowly updated copy of it. Their errors are then less correlated than the errors of a single network used twice, which gives part of the independence that the argument needs. This is Double DQN, in the [next chapter](fitted-q-iteration.md) ({ref}`sec-fqi-double-dqn`).
```

The algorithm below writes the principle out with Monte Carlo targets. Two
Q-functions alternate roles: at line 9 the first selects the action and at
line 12 the second evaluates it, and lines 10 and 13 do the same with the
roles exchanged. The mechanism is the same whether the Q-functions are tables
or fitted functions, and with single-sample data ($N = 1$) the two
Q-functions are trained on different subsets or at different frequencies to
approximate the independence.

```{prf:algorithm} Double Q Value Iteration with Monte Carlo
:label: double-q-value-iteration

**Input:** MDP, state sample $\mathcal{S}$, Monte Carlo sample size $N$, maximum iterations $K$

**Output:** Two Q-functions $q^{(1)}$, $q^{(2)}$

1. Initialize $q^{(1)}_0(s,a)$ and $q^{(2)}_0(s,a)$ for all $s,a$
2. $k \leftarrow 0$
3. **repeat**
4. $\quad$ **for** each $s \in \mathcal{S}$ **do**
5. $\quad\quad$ **for** each $a \in \mathcal{A}(s)$ **do**
6. $\quad\quad\quad$ Draw $N$ next states: $s'_1, \ldots, s'_N \sim p(\cdot \mid s,a)$
7. $\quad\quad\quad$ **// Compute targets using decoupled selection-evaluation**
8. $\quad\quad\quad$ **for** $i = 1, \ldots, N$ **do**
9. $\quad\quad\quad\quad$ $a^{(1)}_i \leftarrow \arg\max_{a'} q^{(1)}_k(s'_i, a')$
10. $\quad\quad\quad\quad$ $a^{(2)}_i \leftarrow \arg\max_{a'} q^{(2)}_k(s'_i, a')$
11. $\quad\quad\quad$ **end for**
12. $\quad\quad\quad$ $q^{(1)}_{k+1}(s,a) \leftarrow r(s,a) + \frac{\gamma}{N}\sum_{i=1}^N q^{(2)}_k(s'_i, a^{(1)}_i)$
13. $\quad\quad\quad$ $q^{(2)}_{k+1}(s,a) \leftarrow r(s,a) + \frac{\gamma}{N}\sum_{i=1}^N q^{(1)}_k(s'_i, a^{(2)}_i)$
14. $\quad\quad$ **end for**
15. $\quad$ **end for**
16. $\quad$ $k \leftarrow k+1$
17. **until** convergence or $k \geq K$
18. **return** $q^{(1)}_k$, $q^{(2)}_k$
```

### Other Corrections

Three other approaches keep a single estimate and correct the maximum
itself. They differ in how much they assume about the noise.

The correction can be learned. {cite:t}`Keane1994` treated the bias
$\delta(s) = \mathbb{E}[(\widehat{\Bellman}_N v)(s)] - (\Bellman v)(s) \ge 0$
as a quantity to estimate and subtract. It cannot be computed directly, but
it depends on observable features of the state: the number of actions, the
sample size, and the spread of the action values, since the bias is large
when several actions are close and small when one dominates. At a small set
of states, they ran both a high-fidelity evaluation of the backup, with many
draws, and the $N$-draw estimate used by the iteration, and regressed the gap
on those features; during value iteration the predicted bias is subtracted
from the raw maximum. The template, simulate on a subset and interpolate
everywhere, is common in econometric dynamic programming. It has seen little
use in reinforcement learning, because the high-fidelity evaluations are
expensive and a misspecified regression feeds its own error back into the
values.

The correction can be derived, under a distributional assumption.
{cite:t}`lee2013bias,lee2019biascorrected` assumed Gaussian noise of
standard deviation $\sigma$ on the estimates of $M$ actions, in which case
the expected maximum of the noises follows from extreme value theory,

$$
\mathbb{E}\Big[\max_{a} \varepsilon_a\Big] \approx \sigma \left(\frac{C_{\mathrm{E}}}{b_M} + b_M\right), \qquad b_M = \sqrt{2\log M - \log\log M - \log 4\pi},
$$

with $C_{\mathrm{E}} \approx 0.5772$ the Euler–Mascheroni constant, and the
target is corrected by subtracting this amount with $\sigma$ estimated from
repeated rewards at the same $(s,a)$. The formula is exact for a
single-state problem whose only noise is in the rewards, and it is a
first-order correction otherwise; with heavier tails than Gaussian it
undercorrects. It replaces the regression of the learned correction by a
closed form, at the price of the Gaussian assumption.

The maximum can be replaced. {cite:t}`deramo2016estimating` weight each
action by the probability that it is the best one under a Gaussian model of
the estimates,
$w_a = \mathbb{P}\big(a \in \arg\max_{a'} \hat{\mu}_{a'}\big)$, and use
$\sum_a w_a \hat{\mu}_a$ in place of $\max_a \hat{\mu}_a$. The weighted
estimator sits between the optimism of the maximum and the pessimism of the
double estimator, and adapts to the gaps between actions: when one action
dominates its weight is near one and the estimator is the maximum, and when
several are close their estimates are averaged. It needs a variance estimate
per action and one integral per action, which limits it to small action sets.
The smooth Bellman operator of the [regularized chapter](regularized-dp.md)
also replaces the hard maximum by an average, with weights set by a
temperature rather than by estimated uncertainty. A last option changes
neither the target nor the operator but the loss of the fit, so that
overestimates are penalized more than underestimates; the
[next chapter](fitted-q-iteration.md) takes it up with Gumbel regression.

(sec-fqi-batch)=
## Fitted Q-Iteration on a Fixed Batch of Transitions

The Bellman operator on action values contains an expectation under the
transition law, which we write $P$ over transitions $\tau = (s, a, r, s')$:

$$
(\Bellman q)(s,a) = r(s,a) + \gamma \int \max_{a'} q(s',a')\, P(ds' \mid s,a).
$$

A batch algorithm never sees $P$. It receives a data set
$\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$ and works with the
**empirical distribution**

$$
\hat{P}_{\mathcal{D}} = \frac{1}{N} \sum_{\tau \in \mathcal{D}} \delta_\tau,
$$

where $\delta_\tau$ is a point mass at the transition $\tau$. Sampling from
$\hat{P}_{\mathcal{D}}$ means picking a recorded transition uniformly at
random, and the expectation of any function of a transition under $P$ is
estimated by its average over the data set, the sample-average estimator of
this chapter applied to transitions. Fitted Q-iteration on a batch is value iteration under this
empirical law. The target of a transition is $y_i = r_i + \gamma \max_{a'}
q(s'_i, a')$, with the one successor that was recorded, and the fit averages
the error of the targets over the transitions in the data set. In the
language of statistics this is a plug-in estimate: where the algorithm needs
the unknown $P$, it substitutes $\hat{P}_{\mathcal{D}}$.

```{prf:remark} The empirical distribution as a probability measure
:class: dropdown

The empirical distribution $\hat{P}_{\mathcal{D}} = \frac{1}{N}\sum_{\tau \in \mathcal{D}} \delta_{\tau}$ is a discrete probability measure over $N$ points, whether the state and action spaces are continuous or discrete. For any measurable set $A \subseteq \mathcal{S} \times \mathcal{A} \times \mathbb{R} \times \mathcal{S}$,

$$
\hat{P}_{\mathcal{D}}(A) = \frac{1}{N}\sum_{\tau \in \mathcal{D}} \mathbb{1}[\tau \in A] = \frac{|\{\tau \in \mathcal{D} : \tau \in A\}|}{N},
$$

so it assigns probability $1/N$ to each recorded tuple and zero elsewhere. A transition that never occurred has no weight, which is the formal statement of the coverage problem that the two experiments below exhibit.
```

```{prf:algorithm} Fitted Q-Iteration on a Batch
:label: fitted-q-iteration-batch

**Input:** Dataset $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$, function approximator $q(s,a; \boldsymbol{\theta})$, discount factor $\gamma$, maximum iterations $n_{\max}$

**Output:** Learned Q-function parameters $\boldsymbol{\theta}$

1. Initialize $\boldsymbol{\theta}_0$
2. $n \leftarrow 0$
3. **repeat**  $\quad$ // **Outer loop: value iteration**
4. $\quad$ **// Construct the regression data set with Bellman targets**
5. $\quad$ $X^{(n)} \leftarrow \{(s_i, a_i) : (s_i, a_i, r_i, s'_i) \in \mathcal{D}\}$
6. $\quad$ **for** each $(s_i, a_i, r_i, s'_i) \in \mathcal{D}$ **do**
7. $\quad\quad$ $y_i^{(n)} \leftarrow r_i + \gamma \max_{a' \in \mathcal{A}} q(s'_i, a'; \boldsymbol{\theta}_n)$
8. $\quad$ **end for**
9. $\quad$ **// Inner loop: fit the Q-function to the targets**
10. $\quad$ $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(X^{(n)}, y^{(n)}, \boldsymbol{\theta}_{\text{init}}, K)$
11. $\quad$ $n \leftarrow n+1$
12. **until** convergence or $n \geq n_{\max}$
13. **return** $\boldsymbol{\theta}_n$
```

The state-action pairs $X^{(n)}$ are the same at every iteration; only the
targets change, because they are computed with the current Q-function. The
`fit` operation in line 10 minimizes $\sum_i \ell(q(s_i, a_i; \boldsymbol{\theta}),
y_i^{(n)})$ for a loss $\ell$, by whatever procedure the function class
calls for, from an initialization $\boldsymbol{\theta}_{\text{init}}$ and
with an optimization budget $K$. Supervised learning would initialize at
random and run to convergence. The two original batch algorithms differ in
exactly these choices.

{cite:t}`ernst2005tree` instantiated the template with extremely randomized
trees, an ensemble whose prediction is the average of the targets in a leaf.
Each fit grows the trees from scratch, a cold start, and runs to completion,
since tree construction has no incremental form. A tree with a fixed
partition is an averager, so the convergence theorem of the
[previous chapter](approximate-bellman-equations.md#sec-abe-averagers) applies once
the partitions stop changing, and trees take high-dimensional inputs without
feature design. {cite:t}`riedmiller2005neural` replaced the ensemble by a
neural network $q(s,a;\boldsymbol\theta)$, which interpolates smoothly across
the state-action space, and fitted it by gradient descent (RProp, chosen for
its insensitivity to hyperparameters) until the loss stopped decreasing. The
network is warm-started: each fit begins from the parameters of the previous
iteration, which is natural for a gradient method and is the first of the
choices that the [next chapter](fitted-q-iteration.md) turns into DQN.

```{prf:remark} Goal states in neural fitted Q-iteration
:class: dropdown

For episodic tasks with goal states $S^+$ and forbidden states $S^-$, Riedmiller modified the targets:

$$
y_i^{(n)} = \begin{cases}
c(s_i, a_i, s'_i) & \text{if } s'_i \in S^+ \text{ (goal reached)} \\
C^- & \text{if } s'_i \in S^- \text{ (forbidden state, typically } C^- = 1\text{)} \\
c(s_i, a_i, s'_i) + \gamma \max_{a'} q(s'_i, a'; \boldsymbol{\theta}_n) & \text{otherwise,}
\end{cases}
$$

where $c(s, a, s')$ is the immediate cost. A goal state has no future cost and no bootstrapped term, a forbidden state carries a fixed penalty, and every other state uses the Bellman backup. The hint-to-goal heuristic adds synthetic transitions from goal states with target zero, which pins the Q-function to zero in the goal region and encodes the boundary condition explicitly.
```

### Experiment: Scheduling from Logged Transitions

The reduced inference-scheduling MDP in [Infinite-Horizon MDPs](infinite-horizon-mdps.md#exact-scheduling-mdp-for-inference-serving)
has an exact value-iteration solution because its transition matrix is known.
Suppose instead that only a fixed table of transitions is retained. Each row
contains a reduced state $(p,d)$, a scheduling action, the cost $c$, and the
next reduced state. The generic reward-maximizing form above becomes a
cost-minimizing form after a sign change. FQI can reuse those rows, but it
receives no direct evidence for a state-action region that the behavior policy
never visits. Any prediction there comes from the regressor's inductive bias.

Each transition row is sampled from the finite MDP, not logged directly from
vLLM. Its arrival model comes from the normalized Azure
trace, while its phase rates and powers come from one NVIDIA L4 measurement run
of Qwen/Qwen2.5-7B-Instruct served by vLLM 0.28.0. The data sets therefore
inherit a measured hardware calibration and all of the reduced model's
aggregation assumptions.

Two sampling procedures produce matched data sets of 50,000 transitions. The
**broad** data set draws a reduced state uniformly and then draws uniformly
among its feasible actions. The **narrow** data set follows trajectories from
the empty system under a decode-priority heuristic and selects a random
feasible action with probability 0.05. Both use seed 0 and the same MDP. Their
different state-action sampling distributions are the controlled change.

Each fitted-Q sweep constructs targets

$$
y_i^{(n)}=c_i+0.99\min_{u'\in\mathcal A(s_i')}
q_n(s_i',u')
$$

and fits a joint state-action Extra-Trees regressor. The ensemble contains 200
trees with maximum depth 12 and minimum leaf size 2. The random seed is 0;
successive tree fits use the deterministic seeds 0 through 49 across 50
fitted-Q sweeps. The three scheduling actions enter as a one-hot input,
matching the joint state-action representation in the generic batch algorithm
above. Before each fit, rows with the same state and action are merged into one
point whose target is their average and whose weight is their count. The
minimum leaf size of 1 therefore counts distinct state-action pairs, not rows:
a larger value would force every leaf to average the targets of at least two
different pairs, which blurs the small differences between actions in the same
state.

Changing only the sampling distribution tests how coverage affects an
otherwise identical FQI procedure. The exact dynamic-programming policy and
both fitted policies are evaluated on the same 2,000 seeded episodes, each 300
steps long. Return and policy disagreement measure agreement with the reduced
MDP objective and its exact solution. Mean waiting time, dropped arrivals, and
observed state-action coverage locate operational differences. Decode stalls
count periods in which decode jobs were active but the policy did something
else. In this model they are not errors: the exact policy stalls decode on
purpose while it prefills to grow the batch.

```{code-cell} python
:tags: [remove-input]
:label: fig-inference-scheduling-fqi
:caption: Extra-Trees FQI on two fixed 50,000-transition data sets sampled from the same measured-L4-calibrated scheduling MDP. These are model transitions, not vLLM logs. The broad data set samples states and feasible actions uniformly; the narrow data set follows decode priority with 5 percent random actions. Each policy table covers all 49 states and marks whether the displayed action occurs in that policy's training buffer.

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from inference_replay import render_serving_replay

display(HTML(render_serving_replay(
    Path("artifacts/inference_serving/textbook_results.json"),
    view="fqi",
)))
```

:::{figure} _static/inference_serving/fqi.svg
:label: fig-inference-scheduling-fqi-fallback
:class: pdf-fallback
:alt: Static comparison of exact value iteration with fitted Q policies learned from broad and narrow transition data.

Policy tables over all 49 states. At each state, a white circle marks that the
buffer contains visits to the action selected by the displayed policy. The
interactive version adds playback, stepping, and scrubbing through a queue
trajectory sampled from the reduced transition model rather than observed
directly from vLLM.
:::

Exact dynamic programming and the two fitted-Q policies use the same 2,000
seeded episodes of 300 steps. Coverage is computed from the feasible
state-action pairs present in each fixed transition buffer.

```{code-cell} python
:tags: [remove-input]

import pandas as pd

fqi_metrics = pd.read_csv(
    "artifacts/inference_serving/metrics_fqi.csv"
).set_index("controller")
fqi_metrics[
    [
        "mean_discounted_return",
        "mean_queue_length",
        "mean_waiting_time_s",
        "mean_decode_stalls",
        "mean_dropped_arrivals",
        "coverage_fraction",
        "policy_disagreement_fraction",
    ]
].T.rename_axis("metric").round(3)
```

{download}`Download every fitted-Q metric (CSV) <artifacts/inference_serving/metrics_fqi.csv>`

The broad buffer covers every feasible state-action pair, and its fitted policy
disagrees with the exact policy in 6 of the 49 states (12.24 percent). All six
lie next to the switching curve, where the exact policy prefills and the fitted
policy decodes. Each of these state-action pairs appears about 340 times in the
buffer, so the errors do not come from missing data. In those states the exact
action values of prefill and decode differ by at most 1.24, while the values
themselves range from 254 to 891. A regressor that is accurate to a fraction of
a percent can still rank two actions this close in the wrong order. The broad
policy's mean discounted return is $-242.25$, compared with $-230.04$ for the
exact policy, and its mean waiting time is 5.76 seconds rather than 5.24.

The narrow buffer covers 41.73 percent of feasible pairs and disagrees with the
exact policy in 19 states (38.78 percent). In 15 of those states the buffer
contains no visit to the action that the fitted policy selects. The log follows
decode priority, which never prefills while a request is decoding, so the
regressor has no data on the batching decisions that the exact policy makes.
Its choices there are extrapolations: it idles in several states with six
requests decoding, and it prefills past the point where the exact policy
decodes. Its mean discounted return is $-286.08$, its mean waiting time is
11.20 seconds, and it drops 1.40 arrivals per episode against 0.20 for the
exact policy. Return is the negative of discounted cost, so higher values are
better.

The two data sets have equal size, so the difference between their policies
cannot be attributed to the number of transition rows. The broad buffer's
errors come from approximation where the action gap is small; the narrow
buffer's errors come from coverage, where the behavior policy never took the
action in question. These are the first and the third difficulty named after
the error bound of {ref}`sec-abe-error`:
a fit is accurate on average, not at the states where two actions are nearly
tied, and it is controlled only where there are data.

The hardware calibration is measured, while training and evaluation remain
inside the stated 49-state MDP. These results concern offline coverage under
that abstraction. They are not a benchmark of vLLM scheduling or of NVIDIA L4
hardware across deployments.

:::{dropdown} Inspect the fitted-Q experiment
```{literalinclude} code/inference_control.py
:language: python
:start-at: def fit_fqi
:end-before: def _reduced_policy_trajectory
:linenos:
```
{download}`Download the complete inference-control implementation <code/inference_control.py>`
:::

(sec-fqi-conjunctions)=
### Example: Collision Avoidance Learned from Logged Warnings

This example has no model, only a log. When a satellite and another object, most often a piece of debris, are
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
of $s'$, the one-successor targets of
{ref}`sec-mc-q-functions`. We fit extra
trees, the averager of tree-based FQI {cite}`ernst2005tree`, on the logged
decision epochs, and judge every policy out of sample: each event is decided
by a policy fitted without it.

:::{figure} _static/conjunction_wait_or_burn/wait_or_burn.svg
:label: fig-conjunction-wait-or-burn

Collision avoidance from ESA's conjunction messages. (a) The collision
probability of the 742 events whose estimate crossed the break-even
probability, about $10^{-5}$ (dashed), before the deadline. The 59 in color
ended above it. Messages below $10^{-8}$, half of which report a probability of $10^{-30}$, leave the panel. (b) Total cost of each policy over all events, split into the
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
that waiting pays nor at what probability to act, and it comes within about 7% of
the hand rule, 313 against 291. Its learned rule (panel (c)) waits through a
new alarm more than three days out unless the probability is twenty to forty-five
times the break-even value, in which case it maneuvers early, while a
maneuver is still cheap. The bar comes down as the deadline nears, but at the
deadline it still asks for about three times the break-even probability, a
bet that later messages will come down, and that bet misses more of the
events that needed a maneuver, 20 against 6 for the hand rule. Two choices
decide whether fitted Q-iteration works here at all, and both follow from
the error analysis of the previous chapters.

1. The fit must fit the right thing. The raw target $C\, p_{\text{final}}$
   is heavy-tailed. One event in the log ends at a probability of 2% and
   costs about 2,070 if nothing is done, more than three times the cost of
   all other events together. Extra trees predict the average of the targets
   in a leaf, so this one cost is shared by every state in the leaves where
   its epochs fall, and those states look dangerous. Fitted on all events,
   the raw-target policy makes 342 maneuvers; fitted without that single
   event, 141. Out of sample it makes 374 maneuvers and costs 441, half again
   as much as the hand rule. The second version fits the ratio
   $y_i / (C\, 10^{\rho_i})$, the cost relative to what the current message
   alone predicts, and multiplies the fit back. The regressor then learns a
   correction factor to the operator's own estimate. For the dangerous event
   this factor stays between 0.7 and 7, an ordinary value, and the policy
   makes 166 maneuvers with the event in the data and 175 without it.
2. The fit is only as good as the data where decisions are made. Five
   in six of the 123,882 decision epochs have probabilities below $10^{-7}$,
   where waiting is plainly right, and we leave them out of the fit. Only
   62 events needed a maneuver. This is the coverage problem behind the
   concentrability coefficients of
   {ref}`sec-abe-error`:
   the regression error that matters is the one at the few states where the
   two actions are close, and no amount of data elsewhere reduces it. The
   hindsight policy, at 147, is not a target: it knows the estimate of the
   messages that arrive after the deadline, which no policy can use.

- {download}`Conjunction model and fitted Q-iteration <code/conjunction_wait_or_burn.py>`
- {download}`Conjunction experiment builder <scripts/build_conjunction_wait_or_burn_artifacts.py>`
- {download}`Conjunction results <artifacts/conjunction_wait_or_burn/metrics.json>`
- {download}`Conjunction messages used (columns of the ESA training set, CC BY 4.0) <data/conjunctions/kelvins_cdm_extract.csv.gz>`

(sec-abe-roadmap)=
## What Remains: Online Data and Continuous Actions

Fitted Q-iteration on a fixed log, as this chapter leaves it, still differs
from NFQCA in two ways.

First, its data are fixed, and each fit runs to completion before the targets
change. Online reinforcement learning works from a buffer that changes as the
agent acts, and a neural fit is warm-started and stopped after a few gradient
steps. The [next chapter](fitted-q-iteration.md) merges the fit with the
outer loop, which produces the target network and the replay buffer of DQN.

Second, it maximizes over a finite set of actions. Every target contains
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

| What fitted value iteration assumed | What replaces it | Where |
|:--------------------------|:-----------------|:------|
| State values $v(s)$ | Action values $q(s,a)$, so acting needs no model | {ref}`sec-mc-q-functions` |
| Exact expectation in the target | One sampled successor per transition | {ref}`sec-mc-q-functions` |
| Targets at chosen states | A fixed batch of logged transitions (FQI, NFQ) | {ref}`sec-fqi-batch` |
| A fixed batch, each fit run to completion | A replay buffer and a target network (DQN) | [Deep Q-networks](fitted-q-iteration.md) |
| Maximum over a finite action set | A learned actor in the target (NFQCA, DDPG, TD3) | [Amortized action optimization](amortized-action-optimization.md) |

## Exercises

The exercises follow the chapter: the fitted iteration, its regressor and its
error bounds, single estimators and their effect inside the iteration, then logged data.
{ref}`ex-abe-knn` reuses the code of the eight-dimensional stopping problem,
{ref}`ex-mc-compounding` is a short simulation and {ref}`ex-fqi-chain` is a
short program.

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
   requires when the successor is random, and connect it to the one-successor
   targets of {ref}`sec-mc-q-functions`.
:::

:::{solution} ex-abe-brm
:class: dropdown

1. The objective is $(\theta - 2\gamma\theta)^2 + (2\theta - 2\gamma\theta)^2 = \theta^2\big[(1-2\gamma)^2 + 4(1-\gamma)^2\big]$, a parabola whose coefficient is positive for $\gamma < 1$. Its minimizer is $\theta = 0$.
2. Gradient descent on a positive-definite quadratic converges for any step size below $2/c$, where $c$ is its curvature. Fitted value iteration does not descend this objective: it holds the target fixed and fits it, which is the semi-gradient, and the semi-gradient map $\theta \mapsto \tfrac65\gamma\theta$ has modulus $\tfrac65\gamma > 1$ for $\gamma > 5/6$, so it is not a contraction.
3. Squaring the residual of a single sampled successor gives $\mathbb{E}\big[(v(s) - r - \gamma v(s'))^2\big]$, which equals the squared residual plus the variance of $r + \gamma v(s')$. The minimizer is biased toward functions with small variance of the successor value. An unbiased estimate needs two independent successors of the same state, the double-sampling problem, which a log with one successor per transition cannot provide.
:::

:::{exercise} Accurate fits, bad iteration
:label: ex-projection-check-4

In the two-state counterexample, the function class contains $v^*$ exactly. Why does fitted value iteration still diverge, and why does {prf:ref}`thm-approximate-vi` not prevent it?
:::

:::{solution} ex-projection-check-4
:class: dropdown

The least-squares fit is not an averager: it maps the constant target to a line that overshoots at state 2. Its sup-norm operator norm is $6/5$, so the modulus $6\gamma/5$ of the composition exceeds one when $\gamma > 5/6$. The fit error at iteration $k$ is proportional to $\theta_k$, so there is no single $\varepsilon$ that bounds it for all $k$, and the theorem's assumption fails.
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

:::{exercise} Weighting shift
:label: ex-projection-check-3

If the fitting distribution puts almost no mass on an important region of the state space, what failure should you expect?
:::

:::{solution} ex-projection-check-3
:class: dropdown

The fit may be accurate under the weighted norm yet poor in that neglected region, leading to bad values or decisions there. In the error bound of fitted value iteration, this shows up as a large sup-norm error, or, in the weighted analysis, as a large concentrability coefficient.
:::

:::{exercise} Designing a convergent fitted iteration
:label: ex-abe-open

Pick a problem from an earlier chapter that has a continuous state: the
reservoir of the finite-horizon chapter, the storm-surge barrier, or the
fishery. Design a fitted value iteration for it that is guaranteed to
converge, run it, and compare its greedy policy with the exact one. Then
replace the fit by a neural network and report what changes: the number of
iterations, the stability of the iterates, and the policy loss. Explain the
differences with the results of this chapter and the previous one.
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

:::{exercise} Sign before simulation
:label: ex-montecarlo-check-2

Several actions have equal true values and independent noisy estimates. What sign do you expect for the bias of their maximum?
:::

:::{solution} ex-montecarlo-check-2
:class: dropdown

Positive. Maximization preferentially selects estimates with favorable noise, so the expected maximum exceeds the common true value.
:::

:::{exercise} Double estimation
:label: ex-montecarlo-check-3

Which dependence is broken when one estimate selects the maximizing action and an independent estimate evaluates it, and which bias remains?
:::

:::{solution} ex-montecarlo-check-3
:class: dropdown

The evaluation noise is made independent of the lucky noise that caused the selection, which removes the reuse of the same positive error for both jobs. The selection bias remains: the chosen action is still more often one whose first estimate was lucky, so the double estimator is biased slightly downward.
:::

:::{exercise} Sample sizes
:label: ex-mc-sample-size

A simulator returns successor values with standard deviation $\sigma = 3$.

1. How many draws give a root mean squared error of $0.01$ for one
   expectation?
2. A fitted iteration needs one such expectation for each of 2,000 states and
   4 actions. How many simulator calls does one iteration cost, and how would
   the cost change if the state had twice as many coordinates?
3. By {prf:ref}`thm-approximate-vi`, what does an integration error of $0.01$
   per iteration contribute to the final value error with $\gamma = 0.95$?
:::

:::{solution} ex-mc-sample-size
:class: dropdown

1. $\sigma/\sqrt N = 0.01$ gives $N = (3/0.01)^2 = 90{,}000$.
2. $2{,}000 \times 4 \times 90{,}000 = 7.2 \times 10^8$ calls per iteration. The Monte Carlo rate does not depend on the dimension, so the count does not change unless $\sigma$ does; a product quadrature rule would need $N^D$ nodes instead.
3. At most $0.01/(1 - 0.95) = 0.2$, added to the error of the fit.
:::

:::{exercise} The bias of a maximum
:label: ex-mc-max-two

Two actions have the same true value $0$, and each is estimated by the mean
of five draws of standard deviation $1$.

1. Show that the expected maximum of the two estimates is
   $\sigma_{\text{mean}}/\sqrt\pi$, where $\sigma_{\text{mean}} = 1/\sqrt5$ is
   the standard deviation of each mean, and evaluate it. Hint: the maximum of
   two numbers is their average plus half their absolute difference.
2. Compare with the first point of {numref}`fig-mc-max-bias`.
3. How does the bias change with $N$, the number of draws per action?
:::

:::{solution} ex-mc-max-two
:class: dropdown

1. $\max(X_1, X_2) = \tfrac12(X_1 + X_2) + \tfrac12|X_1 - X_2|$. The first term has mean $0$. The difference is Gaussian with standard deviation $\sqrt2\,\sigma_{\text{mean}}$, and $\mathbb{E}|Z| = \sqrt{2/\pi}$ for a standard Gaussian, so the expectation is $\tfrac12\sqrt2\,\sigma_{\text{mean}}\sqrt{2/\pi} = \sigma_{\text{mean}}/\sqrt\pi = 1/\sqrt{5\pi} \approx 0.252$.
2. The figure shows $0.25$ at two actions.
3. It scales with $\sigma_{\text{mean}} = 1/\sqrt N$: four times as many draws halve it. It never vanishes for finite $N$.
:::

:::{exercise} When the double estimator does worse
:label: ex-mc-double-worse

Two actions have true values $\mu_1 = 0$ and $\mu_2 = -1$, and each estimate
has Gaussian noise of standard deviation $1$.

1. For the maximum estimator, use
   $\mathbb{E}[\max(X_1, X_2)] = \mu_1\Phi(\delta/\vartheta) + \mu_2\Phi(-\delta/\vartheta) + \vartheta\,\phi(\delta/\vartheta)$
   with $\delta = \mu_1 - \mu_2$ and $\vartheta = \sqrt2$, where $\Phi$ and
   $\phi$ are the standard Gaussian distribution and density, to compute its
   bias.
2. For the double estimator, the selected action is wrong with probability
   $\Phi(-\delta/\vartheta)$. Compute its bias.
3. Which estimator is closer to the truth here? Relate the answer to the two
   sources of bias in {ref}`sec-mc-overestimation`, and to the case of equal
   values in {numref}`fig-mc-max-bias`.
:::

:::{solution} ex-mc-double-worse
:class: dropdown

1. $\Phi(-1/\sqrt2) = 0.240$ and $\phi(1/\sqrt2) = 0.311$, so the expected maximum is $-0.240 + \sqrt2 \times 0.311 = 0.200$, a bias of $+0.20$.
2. The double estimator is unbiased for the value of the action it selects, so its expectation is $-1 \times 0.240 = -0.240$, a bias of $-0.24$. A simulation with $2 \times 10^6$ draws gives $0.200$ and $-0.238$.
3. The maximum estimator is closer here, by a little. With equal values the selection costs nothing and the double estimator is unbiased; with a gap of the size of the noise, selection bias dominates, and it is negative. Removing evaluation bias does not remove selection bias, which is why the double estimator is pessimistic rather than unbiased in general.
:::

:::{exercise} Bias that compounds
:label: ex-mc-compounding

Consider one state and $M$ actions, each with reward $0$ observed through
Gaussian noise of standard deviation $1$, and $\gamma = 0.9$. Monte Carlo
value iteration sets $v_{k+1} = \max_a \hat r_a + \gamma v_k$, where each
$\hat r_a$ is a fresh mean of $N$ draws.

1. Show that $\mathbb{E}[v_k] \to b/(1-\gamma)$, where $b$ is the expected
   maximum of $M$ independent means, while the true value is $0$.
2. Simulate it for $M = 4$ and $N = 5$, and compare with
   $b \approx 0.46$ read from {numref}`fig-mc-max-bias`.
3. Replace the maximum by the double estimator, with two independent sets of
   draws, and simulate again.
:::

:::{solution} ex-mc-compounding
:class: dropdown

1. Taking expectations, $\mathbb{E}[v_{k+1}] = b + \gamma\,\mathbb{E}[v_k]$, a contraction with fixed point $b/(1-\gamma)$.
2. With $b \approx 0.460$ the limit is $4.6$, ten times the per-step bias. A simulation such as

```python
import numpy as np
rng = np.random.default_rng(0)
M, N, gamma, runs, iterations = 4, 5, 0.9, 20000, 200
v = np.zeros(runs)
for _ in range(iterations):
    v = rng.normal(size=(runs, M, N)).mean(axis=2).max(axis=1) + gamma * v
print(v.mean())
```

prints a value close to $4.6$.
3. Selecting with one set of means and evaluating with the other gives an expectation of $0$ at each step, so $\mathbb{E}[v_k] \to 0$: with equal true values the double estimator removes the compounding entirely.
:::

:::{exercise} Q-functions from the Bellman operator on values
:label: ex-mc-q-contraction

1. Show that $(\Bellman q)(s,a) = r(s,a) + \gamma\,\mathbb{E}[\max_{a'} q(s',a') \mid s,a]$
   is a $\gamma$-contraction in the sup norm. Use the inequality
   $|\max_a f(a) - \max_a g(a)| \le \max_a |f(a) - g(a)|$, which the
   infinite-horizon chapter used for the operator on values.
2. Given $q^\star$, write the optimal policy and $v^\star$. Which of the two
   needs the transition model?
3. Fitted value iteration and fitted Q-iteration both converge with an
   averager. Which one needs more targets per iteration, and which one is
   cheaper to act with?
:::

:::{solution} ex-mc-q-contraction
:class: dropdown

1. For each $(s,a)$, $|(\Bellman q)(s,a) - (\Bellman q')(s,a)| \le \gamma\,\mathbb{E}\big[|\max_{a'} q(s',a') - \max_{a'} q'(s',a')|\big] \le \gamma\|q - q'\|_\infty$.
2. $\pi^\star(s) \in \arg\max_a q^\star(s,a)$ and $v^\star(s) = \max_a q^\star(s,a)$. Neither needs the model, which is the point of storing $q$.
3. Fitted Q-iteration computes targets for every state-action pair, $|\mathcal{A}|$ times as many as fitted value iteration with a model, but acting with it is a maximization over stored values, with no expectation to estimate.
:::

:::{exercise} Many successors or many transitions
:label: ex-mc-budget

A resettable simulator can return several successors of the same state and
action. With a fixed budget of simulator calls per iteration, compare two
uses on the eight-dimensional stopping problem of {ref}`sec-abe-high-dim`: many
states with one successor each, or a quarter as many states with four
successors each. Measure the error of the fitted value function and of its
greedy policy, and explain the result with the two expectations of
{ref}`sec-mc-two-expectations`: which error does each choice reduce?
:::

:::{solution} ex-mc-budget
:class: dropdown

This exercise is open-ended. More successors per state reduce the variance of
each target, which a regression would otherwise have to average out across
nearby states. More states reduce the error of the fit itself and extend
coverage. Which wins depends on the noise of the targets relative to the
smoothness of the value: a problem with very noisy transitions and a smooth
value favors more successors, and a problem whose value varies a lot across
the state space favors more states.
:::

:::{exercise} Offline support
:label: ex-fqi-check-2

Why is a highly accurate regression fit still dangerous for actions poorly represented in an offline dataset?
:::

:::{solution} ex-fqi-check-2
:class: dropdown

The fit is unconstrained off the data support. Maximization can select extrapolated action values that are numerically large but unsupported by transitions in the dataset.
:::

:::{exercise} Matched coverage comparison
:label: ex-fqi-check-4

The two inference-scheduling buffers contain the same number of transitions.
Why can the buffer with fewer distinct state-action pairs still attain lower
training error while producing a worse policy?
:::

:::{solution} ex-fqi-check-4
:class: dropdown

Repeated transitions make the empirical regression problem easier on the
visited region, but they supply no Bellman targets for omitted actions and
states. Greedy action selection can therefore depend on unsupported
extrapolation even when the in-buffer fit is accurate.
:::

:::{exercise} Scaling the targets
:label: ex-fqi-check-5

In the collision-avoidance example, why did one event with a 2% collision probability change the maneuvers of the fitted policy at hundreds of other states, and what did fitting a ratio instead of the raw cost change?
:::

:::{solution} ex-fqi-check-5
:class: dropdown

Extra trees predict the average of the targets in a leaf, so a single cost three times larger than all the others combined raises the prediction of every state that shares a leaf with that event's epochs. Fitting the cost relative to the current message's own estimate turns the target into a correction factor of ordinary size, so the event no longer dominates the leaves that contain it.
:::

:::{exercise} Coverage on a chain
:label: ex-fqi-chain

Build a chain of ten states $0, \ldots, 9$ with two actions. Action "right"
moves one state to the right with probability $0.9$ and stays with
probability $0.1$; "left" moves one state to the left in the same way. The
reward is $1$ on entering or staying in state 9, and $\gamma = 0.9$.

1. Log 5,000 transitions from uniformly random states with uniformly random
   actions, and run fitted Q-iteration with a table as the fit: each target is
   averaged over the transitions of its state-action pair, and pairs with no
   data keep the value $0$. Evaluate the greedy policy exactly from state $0$.
2. Repeat with a log in which the behavior policy always moves left in states
   5 to 8.
3. Explain why the second policy fails in states 0 to 4, where the log has
   data for both actions.
:::

:::{solution} ex-fqi-chain
:class: dropdown

With the broad log the greedy policy moves right everywhere and its value
from state 0 is $3.90$, the optimal value. With the narrow log, "right" has
no data in states 5 to 8, so its value there stays at $0$, which ties with
"left" and loses the tie. The policy moves left in states 5 to 8, and the
value it attains from state 0 is $0$. In states 0 to 4 both actions have data,
but the targets of "right" are built from the values of states 1 to 5, which
are wrong: the error in the uncovered states propagates backward through the
bootstrapped targets. A fit that generalizes, such as trees on the state
index, could extrapolate "right" from state 4 or 9, and that extrapolation
would decide the result, which is the situation of the narrow
inference-scheduling buffer.
:::

:::{exercise} Where the actor enters
:label: ex-projection-check-5

Which part of fitted Q-iteration does the actor of NFQCA replace, and why is it needed only for continuous actions?
:::

:::{solution} ex-projection-check-5
:class: dropdown

It replaces the maximization over next actions inside each Bellman target. With a finite action set the maximum is computed by enumeration; with a continuous action set it is a nonlinear program for every target, and the actor amortizes those programs into one trained network.
:::

## Summary and Forward Look

Fitted value iteration computes Bellman targets at chosen states and fits a
regressor to them. Collocation and Galerkin projection are two fitting rules,
and regression with trees or neural networks is the one that scales, as the
eight-dimensional stopping problem showed. Fitted Bellman updates contain two expectations, inside the operator and over
the states, and either can be estimated from samples. The sample average is
unbiased, its error falls as $N^{-1/2}$ whatever the dimension, and it adds a
per-iteration error that the bound of the previous chapter absorbs. Sampling
makes the model optional: a simulator or a log of transitions suffices, which
is the setting of reinforcement learning. Acting on a sampled value function
would require sampling at decision time, so the action-value function
$q(s,a)$ stores the backup of each action instead, and fitted Q-iteration is
fitted value iteration on $q$. Temporal-difference learning solves the Galerkin system of the previous
chapter one transition at a time; it is stable on-policy for the same reason
that the projected equation contracts, and off-policy it can diverge even
when the equation is well posed, which importance sampling, gradient TD and
emphatic TD repair in different ways. The maximum over noisy estimates is biased
upward, and the bias compounds through the iterations; using one estimate to
select the action and an independent one to value it removes the larger part
of the bias, and learned, analytic or weighted corrections are the
alternatives.

Fitted Q-iteration on data is fitted value iteration with the model replaced
by the empirical distribution of a buffer: targets from recorded
transitions, one successor each, and a fit that averages its error over
them. The two experiments located the errors of a fitted policy. In the
scheduling log with full coverage, the policy erred only where two actions
were nearly tied, an approximation error that more data would not remove;
in the log that followed one heuristic, it erred where the heuristic had
never tried the exact policy's action, a coverage error that no regressor
can repair. On ESA's collision warnings, fitted Q-iteration learned from
the log alone a rule within about 7% of the one an operator would write by
hand, once its targets were scaled so that one rare event could not dominate
the fit.

Every fit in this chapter ran to completion on data that did not change.
[Deep Q-networks](fitted-q-iteration.md) lets the data grow as the agent acts
and merges the fit with the outer loop, which produces the target network,
the replay buffer and DQN.
