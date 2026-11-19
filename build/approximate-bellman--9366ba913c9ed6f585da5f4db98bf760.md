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
# Convergence and Error Bounds for Approximate Value Iteration

The [previous chapter](weighted-residual-methods.md) ended with function
iteration, $v_{k+1} = \Proj\Bellman v_k$: value iteration with a fit after
each backup. On the fishery it converged with one basis and diverged with
another, and the explanation, that the fit must be a nonexpansion in the sup
norm, was asserted without proof. This chapter supplies the proofs. It asks when the
fit preserves the contraction of the Bellman operator, maximum included, how
far from $v^*$ the iteration ends, and what is known when the contraction is
lost. Throughout, the model is known and every expectation is computed
exactly; the [next chapter](monte-carlo-bellman-estimation.md) replaces both
by data.

The object of study is the one the previous chapter left: the iteration
$v_{k+1} = \Proj\Bellman v_k$, in which $\Bellman$ is applied exactly and
$\Proj$ is the approximation step chosen by the method. For collocation,
$\Proj$ interpolates the backups at the nodes; for Galerkin and least squares,
it projects them onto the span of the basis in a weighted norm. We call
$\Proj$ the fit, and ask which properties of $\Proj$ keep the iteration
convergent.

The answers come in three kinds. Fits that only average their targets, the
averagers, keep the iteration a contraction for the optimality equation; they
include grid interpolation, state aggregation and fixed-partition trees.
Least squares on a basis is not an averager, and it diverges on a two-state
problem whose value function it represents exactly. For evaluating a fixed
policy with linear features, least squares weighted by the stationary
distribution of the policy is safe as well, a special case that explains LSTD
and, in the next chapter, temporal-difference learning. Newton's method, the
other solver of the previous chapter, becomes approximate policy iteration.
The chapter closes with what operations research, control, numerical analysis
and economics know about approximate solutions of the optimality equation.

::::{admonition} Learning Goals
:class: note

After completing this chapter, you should be able to:
- Construct a function iteration that diverges although the function class contains the solution, and explain the divergence by a fit that is not a nonexpansion in the sup norm
- Explain why an averager cannot make the iteration diverge, with or without the maximum, and recognize which common fits are averagers
- For linear features and policy evaluation, write the fit as one projection matrix, explain why on-policy weighting makes the iteration converge, and solve the Galerkin system directly with LSTD
- Explain approximate policy iteration as Newton's method on the approximate Bellman equation
- Place the convergence results for the optimality equation from operations research, control, numerical analysis and economics in one frame

**Prerequisites:** the weighted-residual template ([previous chapter](weighted-residual-methods.md)), the contraction and Newton-Kantorovich material of [infinite-horizon MDPs](infinite-horizon-mdps.md), interpolation with cardinal functions in [finite-horizon DP](finite-horizon-dp.md).
::::

(sec-abe-convergence)=
## Function Iteration Can Diverge

Value iteration converges because $\Bellman$ is a $\gamma$-contraction in the
sup norm ({prf:ref}`value-iteration-convergence`), and the fit can undo that.
The fishery of {ref}`sec-wr-fishery` ran function iteration with
collocation, once with hat functions and once with Chebyshev polynomials,
and its sweep over $n$ showed the polynomial diverging for some $n$. The cell below measures what happens to the fitted
values at the nodes from one iteration to the next. The maximization over the
escapement and the quadrature over the shock are done by the module that
produced the figures of the previous chapter.

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
and the coefficients run off to infinity. The Bellman backup is still a
$\gamma$-contraction in the sup norm, so the growth comes from the
interpolation step: on these iterates it multiplies sup-norm distances by at
least $1.049/0.95 \approx 1.10$, so it is not a nonexpansion, and
$\Proj\Bellman$ is not a contraction. A two-state example isolates the
mechanism, and averagers are the fits that are always sup-norm
nonexpansions.

### A Two-State Example Where Least Squares Diverges

The following example is due to {cite:t}`TsitsiklisVanRoy1996`. There are two
states and one action. State 1 moves to state 2, state 2 stays where it is,
and all rewards are zero, so $v^* = 0$. The value function is represented
with a single feature, $v(1;\theta) = \theta$ and $v(2;\theta) = 2\theta$,
which contains $v^*$ at $\theta = 0$. Function iteration computes the
targets at both states and fits them by least squares with equal weights.
Both targets equal $\gamma v(2;\theta_k) = 2\gamma\theta_k$, and the fit
minimizes $(\theta - 2\gamma\theta_k)^2 + (2\theta - 2\gamma\theta_k)^2$,
which gives

$$
\theta_{k+1} = \frac{6\gamma}{5}\theta_k.
$$

For $\gamma > 5/6$ the iterates diverge from every $\theta_0 \neq 0$. The
Bellman operator multiplies the sup-norm distance between $v_k$ and $v^*$ by
$\gamma$, and the fit then multiplies it by $6/5$, so the fit is not a
nonexpansion: the target is the constant $2\gamma\theta_k$, and the best line through the origin overshoots it at
state 2, where it predicts $\frac{12}{5}\gamma\theta_k$. The fit error grows in
proportion to $\theta_k$, so no bound that assumes a fixed accuracy per
iteration can help here. {cite:t}`BoyanMoore1995` observed the same failure with polynomial
regression and neural networks on small control problems, at a time when
TD-Gammon had made neural value functions look easy.

(sec-abe-averagers)=
## Averagers Preserve the Sup-Norm Contraction

The composition $\Proj\Bellman$ keeps the sup-norm contraction of $\Bellman$
whenever $\Proj$ is a nonexpansion in the sup norm, since then
$\|\Proj\Bellman v - \Proj\Bellman w\|_\infty \leq \|\Bellman v - \Bellman w\|_\infty \leq \gamma\|v - w\|_\infty$.
The argument uses only the contraction, so it holds with the maximum in
$\Bellman$. Which fitting rules are sup-norm nonexpansions? The answer was found independently in
computational economics {cite}`Judd1992,Judd1996,McGrattan1997,SantosVigoAguiar1998`,
economic dynamics {cite}`Stachurski2009`, and reinforcement learning
{cite}`Gordon1995,Gordon1999`, and the communities arrived at the same
condition under different names.

### Definition and Convergence Theorem

```{prf:definition} Averager
:label: gordon-averager

A linear operator $\Proj$ on the space of bounded real-valued functions on $\mathcal{S}$ is an **averager** if it is monotone, that is $f \leq g$ pointwise implies $\Proj f \leq \Proj g$, and preserves constants, $\Proj\mathbf{1} = \mathbf{1}$. On a finite state space, $\Proj$ is an averager exactly when $\Proj v = Wv$ for a stochastic matrix $W$: $w_{ij} \geq 0$ and $\sum_j w_{ij} = 1$ for every row $i$.
```

{cite:t}`Gordon1995` introduced the term averager for such operators. The
finite-state characterization follows because a
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
The averager condition therefore covers collocation with a monotone interpolant
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

An averager acts like a weighted average: it respects order and is a
nonexpansion in the sup norm. The least-squares fit of the counterexample is not one.
It maps the constant function $(1, 1)$ to $(3/5, 6/5)$, so it does not
preserve constants, and it does not respect order either. Polynomial
interpolation at Chebyshev nodes preserves constants but is not monotone: its
cardinal functions dip below zero between the nodes, and its Lebesgue
constant is about $2.4$ for ten nodes. That is how the ten-coefficient
polynomial of the fishery in {ref}`sec-abe-convergence` could make
$\Proj\Bellman$ fail to be a contraction, while the hat functions, whose
cardinal functions are nonnegative, could not.

```{prf:theorem} Function iteration with an averager converges
:label: santos-vigo-aguiar-stability

Let $\Bellman$ be a $\gamma$-contraction in the sup norm on the bounded functions on $\mathcal{S}$, and let $\Proj$ be an averager. Then $\Proj \Bellman$ is a $\gamma$-contraction, so function iteration $v_{k+1} = \Proj \Bellman v_k$ converges from any $v_0$ to the unique fixed point $v_\Proj \in \text{Range}(\Proj)$, and

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
the best-approximation error $\|\Proj v^* - v^*\|_\infty$ multiplied by
$(1-\gamma)^{-1}$. For an averager the per-iteration error stays bounded,
which is the condition that the error bound for arbitrary fits, in the
[chapter on fitted value methods](monte-carlo-bellman-estimation.md#sec-abe-error),
requires.

The stochastic matrix gives the averaging a Markov-chain reading: $\Proj v$ is
an expected value after one random transition, and expectations preserve
order. The finite-horizon chapter made the same observation about linear
interpolation, which replaces a deterministic successor by a random grid node
with the interpolation weights as probabilities. Composing that chain with
the model's own transitions makes $\Proj\Bellman$ the Bellman operator of a
finite-state MDP, which is why it contracts.

### Which Common Fits Are Averagers

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
again an averager.

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
| State aggregation | Yes | Equal weights within a group | {cite:t}`TsitsiklisVanRoy1996`, {cite:t}`Whitt1978` |
| $k$-nearest neighbors | Yes | Equal weights within a neighborhood | {cite:t}`ShahXie2018` |
| Kernel smoothing (nonnegative kernels) | Yes | If the weights are normalized to sum to one | {cite:t}`Ormoneit2002`, {cite:t}`Rust1997` |
| Tile coding (CMAC) | Depends on the fit | Averaging the tilings' separate fits is an averager; the joint least-squares fit is not in two or more dimensions ({ref}`sec-abe-tile-coding`) | {cite:t}`Sutton1996` |
| Regression trees and forests, fixed partition | Yes | Leaf averages; guarantee lost if splits depend on the targets | {cite:t}`ernst2005tree` |
| High-order polynomial interpolation | No | Negative weights (Runge phenomenon) | {cite:t}`Bellman1963` |
| Least squares on an arbitrary basis | No | The counterexample above | {cite:t}`TsitsiklisVanRoy1996` |
| Neural networks | No | No sign or normalization constraint on the fit | {cite:t}`Tesauro1995`, {cite:t}`riedmiller2005neural` |

The table places the fits of the two chapters. Collocation with
piecewise-linear interpolation is an averager, so the folk wisdom that linear
interpolation is safe for value iteration is a theorem. Collocation
with a high-order polynomial is not, and its convergence in the fishery, when
it happened, was not guaranteed. Galerkin projection and least squares are
not averagers. Tree ensembles sit on the boundary, as the eight-dimensional example of the
[next chapter](monte-carlo-bellman-estimation.md#sec-abe-high-dim) shows, and networks are outside. The solver matters too: Newton's method
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
example, comes from a tile width with a Gamma distribution. Because the expected fraction of shared tiles is a kernel, a linear fit on
$K$ random tilings is kernel regression with that kernel estimated from $K$
random draws: tile coding with random offsets is a finite random-feature
approximation of kernel regression.

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
   negative entries and its sup-norm operator norm exceeds $4$. The averager condition needs nonnegative entries, so it fails, and only the
$\xi$-norm results of the next subsection remain.
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

(sec-abe-linear-case)=
## Linear Policy Evaluation: Convergence in the Stationary-Distribution Norm

Least squares is not an averager, yet linear features fitted by least squares
are the setting of temporal-difference learning and LSTD, which converge when
they evaluate a fixed policy. The reason is a second norm in which a stability
proof can be carried out. This section develops it for the case where it
applies, a fixed policy and linear features, and ends with why it does not
extend to the maximum. The neural algorithms of the chapters ahead are outside
this case, but the classical convergence results for temporal-difference
learning come from it.

(sec-abe-two-norms)=
### One Projection Matrix for Every Weighted-Residual Method

The two-state counterexample is an instance of the projected Bellman equation
{eq}`eq:projected-bellman` of the previous chapter, $v = \Proj\Bellman v$.
The Bellman operator in it is fixed by the problem; the projection $\Proj$ is
chosen by the method. Both families of stability results, averagers and on-policy weighting, are
conditions under which $\Proj\Bellman$ remains a contraction. On a finite state space
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
is a nonexpansion in the same norm, since then
$\|\Proj\Bellman v - \Proj\Bellman w\| \leq \|\Bellman v - \Bellman w\| \leq \gamma\|v - w\|$.
Two norms make this argument work, and each gives one family of stability
results.

The first is the sup norm, in which $\Bellman$ is a $\gamma$-contraction for
every MDP, with or without the maximum. The sup-norm operator norm of a matrix
is its largest absolute row sum. When the constant function is in the span of
the features, as it is for hat functions, polynomials and aggregation,
$\Proj\mathbf{1} = \mathbf{1}$, so every row of $\Proj$ sums to one, and its
absolute row sums are at most one exactly when all its entries are
nonnegative. The projection is then a stochastic matrix. This is the
averager condition of {ref}`sec-abe-averagers`. For collocation it is a
condition on the cardinal functions, which must be nonnegative, and it holds
for hat functions and fails for polynomials. For Galerkin it is a condition on
the entries of $\boldsymbol\Phi(\boldsymbol\Phi^\top\boldsymbol\Xi\boldsymbol\Phi)^{-1}\boldsymbol\Phi^\top\boldsymbol\Xi$,
which fails for the counterexample, whose projection maps $(1, 0)$ to
$(1/5, 2/5)$ and $(0, 1)$ to $(2/5, 4/5)$. Its rows sum to $3/5$ and $6/5$,
so its sup-norm operator norm is $6/5$, the factor by which the fit
multiplied the sup-norm distance to $v^*$ in the iteration above.

The second is the $\xi$-weighted norm
$\|v\|_\xi = (\sum_s \xi(s) v(s)^2)^{1/2}$, in which every Galerkin
projection with weighting $\xi$ is nonexpansive, whatever the features, because
it is an orthogonal projection for the inner product that defines the norm.
The difficulty moves to the operator, because the Bellman operator need not
contract in $\|\cdot\|_\xi$. {ref}`sec-abe-on-policy` shows
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
convergence proofs for function iteration
{cite}`Whitt1978,TsitsiklisVanRoy1996`, and why
{cite:t}`VanRoy2006` could bound the loss of its greedy policy with the
optimality operator in place.

(sec-abe-on-policy)=
### On-Policy Weighting Makes Projected Evaluation a Contraction

The $\xi$-norm route keeps the Galerkin projection, which is nonexpansive in
$\|\cdot\|_\xi$ for any features, and asks when the Bellman operator
contracts in the same norm. The counterexample shows which weighting works. The chain moves from state 1 to
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
$\xi^\top\mathbf{P}_\pi = \xi^\top$. Suppose the fit is weighted by this
stationary distribution. We show
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
the counterexample's equal weighting, the Jensen step fails, $\mathbf{P}_\pi$ need not be
nonexpansive in $\|\cdot\|_\xi$, and $\Proj\BellmanPi$ may not contract.
Baird's counterexample {cite}`Baird1995` is the classical demonstration.

```{admonition} Exercises: The On-Policy Condition
:class: hint dropdown

1. **Weighting the counterexample.** In the two-state example, fit with weights $(\xi_1, \xi_2)$. Show that $\theta_{k+1} = 2\gamma\theta_k(\xi_1 + 2\xi_2)/(\xi_1 + 4\xi_2)$ and find the set of weightings for which the iteration converges for every $\gamma < 1$.

2. **Non-expansiveness fails off-policy.** For the 3-state chain of {ref}`sec-abe-lstd` below and the weighting $\xi' = (0.9, 0.05, 0.05)$, find a vector $z$ such that $\|\mathbf{P}_\pi z\|_{\xi'} > \|z\|_{\xi'}$.

3. **Losing the argument is not losing convergence.** Implement fitted policy evaluation for the same chain, with features $\phi(s) = (1, s)$ and weighting $\xi'$. It still converges: compute the spectral radius of $\gamma \Proj \mathbf{P}_\pi$ and check that it equals $\gamma$ for this feature set whatever the weighting. Then keep only the feature $\phi(s) = s$ and find a weighting concentrated on state 1 under which the spectral radius exceeds one.
```

(sec-abe-galerkin-system)=
### The Galerkin Linear System Is Positive Definite On-Policy

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
and the LSTD method of the next subsection solves it directly. Temporal-difference
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

(sec-abe-lstd)=
### Solving the Galerkin System Directly: LSTD

Function iteration on {eq}`eq:abe-td-system` needs many sweeps, but the
system has only $n$ unknowns and can be solved directly. Written out, its
conditions say that the residual $\BellmanPi v - v$ is $\xi$-orthogonal to
every feature:

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \boldsymbol{\Phi} \theta - \boldsymbol{\Phi} \theta) = \mathbf{0},
\qquad\text{that is}\qquad
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_\pi \boldsymbol{\Phi}) \theta = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{r}_\pi.
$$ (eq:lstd-galerkin)

Solving it is the **LSTD (least-squares temporal difference)** method of
{cite:t}`BradtkeBarto1996`, and the system is the Galerkin counterpart of the
collocation step {eq}`eq:collocation-newton`. Temporal-difference learning
with linear features, TD(0), converges to the same fixed point
{cite}`TsitsiklisVanRoy1997`, so TD and LSTD are two solvers of one Galerkin
equation, the first by stochastic approximation and the second by solving the
linear system. Neither was designed that way; the reading as projected
equations is due to Bertsekas {cite}`BertsekasTsitsiklis1996,Bertsekas2011`,
and the Galerkin method itself is standard in computational economics
{cite}`ndp`. Because the system is linear in the unknown, $\mathbf{A}$ and
$\mathbf{b}$ can also be estimated from sampled transitions, which the
[next chapter](monte-carlo-bellman-estimation.md#sec-mc-td) does. On a 3-state chain under a fixed policy, with
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

```{admonition} Exercises: LSTD
:class: hint dropdown

1. **Computing the stationary distribution.** For the 3-state chain above, compute the stationary distribution $\xi$ satisfying $\xi^\top \mathbf{P}_\pi = \xi^\top$.

2. **LSTD with stationary weighting.** Recompute the LSTD solution using the stationary distribution instead of uniform weighting. Compare the approximation error.
```

(sec-abe-two-routes-chain)=
### Comparing the Two Conditions on One Chain

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
projections have sup-norm operator norm above one, since their smallest
entries are negative, and the averagers have $\xi$-norm operator norm above
one. The spectral radius is $0.9$ in
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

(sec-abe-optimality)=
### Why the Maximum Breaks the Weighted Argument

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
state by state. A maximum with a fixed function is a nonexpansion,
$|\max\{g, a\} - \max\{g, b\}| \leq |a - b|$, so the Jensen step goes
through with the stationary distribution of $\mathbf{P}$, and $\Proj\Bellman$
is a $\gamma$-contraction in $\|\cdot\|_\xi$.
{cite:t}`TsitsiklisVanRoy1999` use this to prove that a TD method with linear
features converges for optimal stopping, with the pricing of high-dimensional
financial derivatives as their application. The eight-dimensional problem of the
next chapter ({ref}`sec-abe-high-dim`) has this structure: the coordinates move the same
way whether the agent stops or not.

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
approximate policy iteration. Its guarantee is an error bound rather than a
contraction: if every evaluation is accurate to $\varepsilon$ in the sup
norm, the policies are eventually within $2\gamma\varepsilon/(1-\gamma)^2$ of
optimal {cite}`BertsekasTsitsiklis1996`. Unlike exact policy iteration, the
sequence of policies need not improve monotonically, and it can cycle among a
few policies. With the smooth Bellman operator, the Jacobian is
$\gamma\mathbf{P}_{\pi_k}$ for the softmax policy, with no ties to resolve
({ref}`sec-smooth-pi-newton`).

When the evaluation has no closed form, as with a neural network, the
natural replacement is a few sweeps of fitted policy evaluation,
$q \leftarrow \Proj\BellmanPi q$, which is what the critic of NFQCA does.

(sec-abe-control-results)=
## Convergence Results for the Optimality Equation Across Fields

Reinforcement learning came late to this question. Operations researchers,
control theorists, numerical analysts and economists solved Bellman
optimality equations on continuous state spaces long before fitted
Q-iteration, and each community proved convergence results for approximate
solutions. Read side by side, they agree on one point. The guarantees that
hold with the maximum are sup-norm guarantees for monotone approximations, as
in {ref}`sec-abe-averagers`; the others are special cases, or error bounds
that assume the approximation is accurate.

**Discretized dynamic programs.** The oldest results replace a
continuous-state problem by one on a finite grid and ask whether the solution
converges as the grid is refined. {cite:t}`Fox1973` gave conditions for
convergence on compact state spaces, and {cite:t}`Bertsekas1975` proved
convergence of discretized value functions and policies under Lipschitz-type
continuity conditions. {cite:t}`Whitt1978` bounded the error of replacing a
problem by a smaller one built on a partition of the states and actions: the
value of the small problem, extended to the original states, is within
$(1-\gamma)^{-1}K$ of $v^*$ in the sup norm, where $K$ measures how much the
rewards and transitions vary within a cell. This is the bound of
{prf:ref}`santos-vigo-aguiar-stability` for the averager that aggregation is.
{cite:t}`ChowTsitsiklis1991` showed that refining the grid from coarse to
fine, a one-way multigrid method, reaches a given accuracy with an amount of
work that matches the complexity of the problem up to constants. For
aggregation, {cite:t}`VanRoy2006` bounds the loss of the greedy policy by the
variation of $v^*$ within each group, and {cite:t}`Bertsekas2011` points out
that approximate policy iteration with aggregation evaluates each policy
exactly on an aggregate MDP, so its policies do not oscillate.

**Monotone schemes for Hamilton–Jacobi–Bellman equations.** In continuous
time the Bellman equation is a partial differential equation, and its
finite-difference and semi-Lagrangian discretizations are again fixed-point
equations with a maximum. The Markov chain approximation method of
{cite:t}`KushnerDupuis2001` builds the discretization as a controlled Markov
chain on the grid. Its transition probabilities are the weights of the
scheme, nonnegative and summing to one, so the scheme is an averager by
construction, and value iteration on it converges. {cite:t}`Falcone1987`
analyzed a grid-based discrete-time approximation of a discounted
deterministic control problem, in which the value at a node is computed by
interpolating the current values at the successor point, and proved that it
converges to the value function as the time step and the mesh go to zero.
{cite:t}`BarlesSouganidis1991` gave the general principle: a scheme that is
monotone, stable and consistent converges to the viscosity solution whenever
the limiting equation satisfies a comparison principle. Their monotonicity
is the monotonicity in the definition of an averager: the computed value at a node must not
decrease when a neighboring value increases. {cite:t}`MunosMoore2002` brought
these variable-resolution triangulations into reinforcement learning.

**Numerical dynamic programming in economics.** Economists met the same
question when solving growth and investment models
{cite}`Rust1996`. {cite:t}`SantosVigoAguiar1998` proved that value iteration
with piecewise-linear interpolation converges and bounded its error in terms
of the mesh size, the theorem of {ref}`sec-abe-averagers`.
{cite:t}`Stachurski2008` extended the argument to any nonexpansive
approximation on a continuous state space, which covers kernel averagers and
variation-diminishing splines. On the Newton side, {cite:t}`SantosRust2004`
proved that policy iteration with piecewise-linear interpolation converges
quadratically under regularity conditions, and superlinearly under weaker
ones. Projection methods with global bases, Chebyshev collocation and
Galerkin {cite}`Judd1992`, are accurate and fast when they work, and they
come with no convergence guarantee for the iteration, which is the fishery's
experience. The response of this literature was shape preservation:
interpolants that keep the monotonicity and concavity of the data
{cite}`CaiJudd2012`. They are nonlinear in the data and not covered by the
averager theorem, but they remove the oscillations that give polynomial
interpolation a Lebesgue constant above one.

**Policy evaluation and optimal stopping.** The weighted route of
{ref}`sec-abe-linear-case` gives a contraction for the evaluation operator of
the policy whose stationary distribution weights the fit, and for optimal
stopping {cite}`TsitsiklisVanRoy1999`. It gives none for the general
optimality operator, where the projected equation may have no solution
{cite}`deFariasVanRoy2000`.

| **Approach** | **Covers the maximum?** | **Assumption** | **Conclusion** |
|:--|:--|:--|:--|
| Monotone discretization, averagers | Yes | Nonnegative weights that sum to one | Contraction, unique fixed point, convergence as the grid is refined |
| Aggregation and partition bounds | Yes | Small variation of the data within each cell | Error bound $(1-\gamma)^{-1}K$ |
| Policy iteration on a monotone discretization | Yes | Regularity of the model | Quadratic or superlinear convergence |
| Global bases (Chebyshev, Galerkin) | Only special cases | Policy evaluation on-policy, optimal stopping | No general guarantee for the iteration |
| Error propagation | Yes | Each fit accurate | Error bound, no convergence claim |

Function iteration can therefore rely on two kinds of guarantee. A monotone
fit inherits the contraction. Any other fit has only error bounds, which hold
for any fit but assume that each fit is accurate; the
[chapter on fitted value methods](monte-carlo-bellman-estimation.md#sec-abe-error)
states them for an arbitrary regressor.

## Exercises

These exercises complement the shorter ones inside the sections and follow
the order of the chapter.

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
3. Every fit adds an error, and for a nonlinear fit nothing guarantees that repeated fitted evaluation of a fixed policy converges; with off-policy data it can diverge, as the [next chapter](monte-carlo-bellman-estimation.md#sec-mc-deadly-triad) shows. Many sweeps on a policy that is about to change also spend computation on accuracy that the next improvement discards.
:::

## Summary and Outlook

Function iteration on the Bellman optimality equation,
$v_{k+1} = \Proj\Bellman v_k$, is value iteration with a fit after each
backup, and the fit can make the composition fail to be a contraction. It
did so on the fishery with
ten Chebyshev coefficients, and on a two-state problem whose value function
the features represent exactly. Fits that only average their targets cannot:
grid interpolation with hat functions, aggregation, nearest neighbors,
normalized kernels and fixed-partition trees keep the iteration a
contraction, with the maximum and from any set of states. For policy evaluation with
linear features, least squares weighted by the stationary distribution of the
policy is also safe, and LSTD solves the resulting Galerkin system directly,
but that argument breaks with the maximum. Newton's method becomes
approximate policy iteration. The older
literatures on discretized dynamic programs, monotone schemes for
Hamilton–Jacobi–Bellman equations and numerical dynamic programming in
economics reach the same conclusion: with the maximum, the guarantees are for
monotone approximations, and the rest are special cases or error bounds that
assume each fit accurate.

Every result of this chapter assumed a known model, exact expectations and
states chosen by the designer.
[Fitted value methods](monte-carlo-bellman-estimation.md) drops all three:
the expectations become sample averages, the states become the data at hand,
and action values remove the model from decision time. The fit, meanwhile,
becomes any regressor, and its error is controlled by bounds that ask only
that each fit be accurate.
