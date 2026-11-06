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
# Weighted-Residual Methods

The dynamic programming chapters solved Bellman equations by storing one
number per state, or per grid state. That works for the reservoir, the
antibiotic regimen and a river of eight reaches, and it fails soon after. This
part of the book replaces the table by a function with a fixed number of
parameters, and it builds up to the algorithms used in practice: fitted
Q-iteration with trees or neural networks, its online form DQN, and its
continuous-action form NFQCA. This chapter starts with why the replacement is
needed and what it must provide. It then asks the first question the
replacement raises: a function with finitely many parameters cannot satisfy
the Bellman equation at every state, so which equations should determine its
parameters?

::::{admonition} Learning Goals
:class: note

After completing this chapter, you should be able to:
- Explain why tabular dynamic programming fails for large or continuous state spaces, and what a function approximator must provide in its place
- Explain why functional equations require finite-dimensional approximation and how weighted residual methods provide a systematic framework
- Distinguish between Galerkin, collocation, and least-squares methods by their choice of test functions
- Set up function iteration and Newton's method for the resulting finite-dimensional system, and explain when each applies

**Prerequisites:** Bellman operator and contraction property ([infinite-horizon MDPs](infinite-horizon-mdps.md)), interpolation with cardinal functions ([finite-horizon DP](finite-horizon-dp.md)), basic numerical quadrature, linear algebra (matrix inverses, orthogonal projection).
::::

(sec-why-approximate)=
## Why Approximate the Value Function?

Why can't the tables of the previous chapters simply be made larger, and what
does a replacement for them have to provide?

There are three reasons, and each is already visible in an example we have
solved.

**The number of states grows exponentially with the number of state
variables.** The tamarisk river of {ref}`sec-tamarisk-river` has one site per
reach and three conditions per site, so a river of $n$ reaches has $3^n$
states. Value iteration needed about 150 sweeps whatever the size of the
river, but each sweep touches every state, and a dense transition table for a
seven-reach river without the treatment budget would take 84 GB. The rivers that
land managers face have hundreds of reaches. Continuous states hit the same
wall through the grid. The reservoir of the
[finite-horizon chapter](finite-horizon-dp.md) stored its value on 181 storage
levels; five reservoirs in one valley, managed jointly on the same grid, would
need $181^5 \approx 2 \times 10^{11}$ grid states, each with its own one-step
optimization at every stage. A grid with $N$ nodes per axis in $D$ dimensions
has $N^D$ nodes. This is the curse of dimensionality that Bellman named, and
it belongs to the representation rather than to the problem: the value
function of a large system often has structure, such as smoothness or a
dependence on a few aggregate quantities, that a grid cannot exploit and a
more flexible representation can.

**Successors fall between the stored states.** Even in one dimension, the
[finite-horizon chapter](finite-horizon-dp.md#handling-continuous-spaces-with-interpolation)
had to decide which value to use for a successor that lands between two grid
states. Its answer, a family of cardinal functions, turned the table into a
function defined everywhere, and its analysis of how the reconstruction error
propagates through the recursion was a first instance of what this part of the
book studies in general.

**Learning from data visits only some states.** In the chapters ahead the
model is replaced by a simulator or by recorded transitions $(s, a, r, s')$. A
batch of transitions covers a vanishing fraction of a large state space, and
each Bellman target $r + \gamma v(s')$ needs a value at a successor $s'$ that
may never have been visited. A table has nothing to say there. A function
fitted to the visited states does: it generalizes from them, and how well it
generalizes decides how well the method works.

The replacement for the table is therefore a function $v(s; \theta)$ with a
parameter vector $\theta$ of fixed size. It must be cheap to evaluate at any
state. It must be possible to fit it to a finite set of pairs (state, target
value), which is a supervised learning problem. And it must extrapolate
sensibly between and beyond those states. Grid interpolation is cheap and
easy to fit, but it extrapolates only to the cells of its grid and its size
grows as $N^D$. A polynomial or another fixed basis is linear in $\theta$,
which gives closed-form fits and a complete theory, but its features must be
chosen in advance. Tree ensembles and neural networks choose their features
from the data. The fitted Q-iteration methods used in practice, from the
tree-based FQI of {cite:t}`ernst2005tree` to NFQ {cite}`riedmiller2005neural`
and NFQCA {cite}`Hafner2011`, use these.

Replacing $v$ by $v(\cdot;\theta)$ raises two questions. The Bellman equation
$v = \Bellman v$ imposes one condition per state, infinitely many when the
state space is continuous, while $\theta$ has finitely many components. Which
finite set of conditions should determine $\theta$? This chapter answers it
with the weighted-residual framework of numerical analysis
{cite}`Chakraverty2019,AtkinsonPotra1987`, using linear bases, where the
conditions can be written out and solved in closed form. Once the conditions
are fixed, which algorithm finds $\theta$, and does the contraction that made
value iteration converge survive the approximation? The
[next chapter](approximate-bellman-equations.md) answers the second question
for an arbitrary fitting rule, neural networks included, and ends with the
steps that lead from it to FQI and NFQCA.

(sec-wr-fishery)=
## A Motivating Example: Harvesting a Fishery Near a Tipping Point

What goes wrong when a value function on a continuum must be stored in a few
numbers, and does the way we store it change the decisions it produces?

A regulator sets the annual quota for a fish stock. Before the season, a survey
gives the stock $x$, measured as a fraction of the carrying capacity of the
habitat. The regulator chooses the *escapement* $s \in [0, x]$, the part of the
stock left in the water to spawn. The rest, $x - s$, is caught and sold at a
unit price. Over the following year the escaped stock grows to

$$
x' = z\, f(s), \qquad f(s) = s \exp\!\Big(r \Big(1 - \frac{s}{K}\Big) \frac{s - C}{K}\Big),
$$

where $z$ is a log-normal shock with mean one that stands for the environment:
how many young fish survive varies a lot from year to year. This growth law,
which {cite:t}`Boettiger2015` use to study tipping points in fisheries, has an
Allee effect. Below the threshold $C$, the fish are too sparse to replace
themselves, and the stock shrinks even when nobody fishes it (panel (a) of
{numref}`fig-fishery-problem`). We take $r = 2$, $K = 1$, $C = 0.3$, a shock
whose logarithm has standard deviation $0.15$, and $\gamma = 0.95$. These are
teaching values, not the assessment of a real stock. The Bellman optimality
equation is

$$
v^\star(x) = \max_{0 \le s \le x} \Big\{ x - s + \gamma\, \mathbb{E}_z\big[ v^\star\big(z f(s)\big) \big] \Big\}, \qquad x \in [0, 1.6].
$$

Three things separate it from the finite MDPs of the earlier chapters. The
state is a real number, so $v^\star$ is a function on an interval and no table
holds it. The action is a real number too, so the maximization runs over an
interval. And the expectation is an integral over $z$, which we compute with
21-point Gauss–Hermite quadrature. The successor stock $z f(s)$ almost never
falls on a point where a value was stored, which is the difficulty that
[interpolation](finite-horizon-dp.md#handling-continuous-spaces-with-interpolation)
handled in the finite-horizon chapter.

This problem has one state variable, so we can still afford a reference
solution: value iteration on 4,001 grid points, with linear interpolation
between them and escapements restricted to the grid. Because the harvest value
$x - s$ is linear in $s$, the maximization over $s \le x$ reduces to a running
maximum along the grid, and one backup costs a single pass. Halving the number
of grid points changes the solution by less than $0.001$.

:::{figure} _static/fishery_tipping_point/problem.svg
:label: fig-fishery-problem

The fishery. (a) Expected stock next year against the escapement, with the
band that holds 80% of the years. Below the tipping point $C$ the curve lies
under the diagonal: the stock shrinks even without fishing. (b) The optimal
value $v^\star$, computed on a fine grid. (c) The optimal escapement. The
shading marks the three regions of the policy.
:::

The optimal policy has three regions. Above $S^\star = 0.712$, the regulator
harvests down to the constant escapement $S^\star$, the policy that
{cite:t}`Reed1979` proved optimal for this class of models. His condition
$\gamma f'(S^\star) = 1$ gives $0.710$. The small gap comes from bad years,
which leave the stock below $S^\star$, where a fish left to spawn is worth more
than its price. Between $\hat x = 0.23$ and $S^\star$, the fishery is closed
and the stock rebuilds. Below $\hat x$, the policy fishes the stock out: a
stock this far below the tipping point will most likely decline anyway, and its
catch today is worth more than its small chance of recovery.
{cite:t}`Clark1973` showed that maximizing the value of the harvest can make
extinction optimal in this way. Fisheries law adds conservation limits for that
reason, and this model leaves them out. For the methods of this chapter, what
matters is the shape of $v^\star$. It is linear on $[0, \hat x]$, has a kink at
$\hat x$, rises more than tenfold across the tipping point, and is smooth
elsewhere.

Now suppose we may store only $n$ numbers. Two choices are natural.

- A Chebyshev polynomial, $\hat v(x; \theta) = \sum_{j=0}^{n-1} \theta_j T_j(2x/1.6 - 1)$.
  The basis is smooth and global: every coefficient changes the value at every
  stock.
- Piecewise-linear interpolation on $n$ uniform nodes, the hat functions of the
  finite-horizon chapter. Each coefficient is the value at one node, and it
  changes the value only between the two neighboring nodes.

Neither can satisfy the Bellman equation at every stock. The residual

$$
R(x; \theta) = (\Bellman \hat v)(x) - \hat v(x; \theta)
$$

measures by how much it fails at $x$. Collocation picks $n$ points
$x_1, \ldots, x_n$ and asks the residual to vanish at them:

$$
R(x_i; \theta) = 0, \qquad i = 1, \ldots, n.
$$

We use the Chebyshev nodes for the polynomial and the interpolation nodes for
the hat functions, and we solve the $n$ nonlinear equations by function
iteration. Each iteration computes the Bellman backup at the nodes with the
current $\hat v$, then chooses the $\theta$ that reproduces those targets at
the nodes. This is the simplest case of the fitted value iteration of the
[next chapter](approximate-bellman-equations.md).
{numref}`fig-fishery-collocation` shows both approximations with $n = 12$.

:::{figure} _static/fishery_tipping_point/collocation.svg
:label: fig-fishery-collocation

Collocation with $n = 12$ coefficients. (a) The two approximations against
the exact value, with their collocation nodes. (b) The Bellman residual
$R(x;\theta)$, which vanishes at the nodes. (c) The escapement chosen by the
greedy policy of each approximation, against the optimal escapement.
:::

The residuals in panel (b) vanish at the nodes, as collocation requires, and
both peak at about $0.2$ between them. The approximations themselves differ
much more. A polynomial of degree 11 cannot bend sharply enough to follow the
cliff, so it oscillates around it. Near the left end, its value at an empty
stock is $1.6$ instead of zero. The greedy policy inherits these oscillations
(panel (c)). It fishes out every stock below $0.31$, past the tipping point,
where the optimal policy closes the fishery and lets the stock rebuild. A
stock at $0.31$ is worth $1.49$ under the optimal policy and $0.31$, its
immediate catch, under this one. Between $0.53$ and $0.67$ the policy also
harvests stocks that should be left to grow. The piecewise-linear approximation
is coarse between its nodes, but it follows the cliff, and its greedy policy
loses at most $0.064$ anywhere. That worst case is near $x = 0.21$, where the
policy closes the fishery a little too early.

:::{figure} _static/fishery_tipping_point/sweep.svg
:label: fig-fishery-sweep

The same experiment for $n$ from 4 to 64. (a) Largest error of the
approximate value. (b) Largest loss of its greedy policy, evaluated exactly.
Crosses mark the $n$ for which function iteration with the polynomial diverged.
:::

{numref}`fig-fishery-sweep` repeats the experiment for $n$ from 4 to 64.
Piecewise-linear interpolation improves steadily as nodes are added. The
polynomial is erratic. Adding one coefficient can multiply its error by five,
and for $n = 4$ and $n = 9, 10, 11$ function iteration does not converge: the
coefficients grow without bound, although every iteration fits its targets
exactly. Once $n$ is large enough to resolve the cliff, from about 40
coefficients on, the polynomial overtakes the hat functions and converges much
faster. The table summarizes a few cases.

:::{include} artifacts/fishery_tipping_point/results.md
:::

The example raises the three questions that organize this chapter and the
next.

1. **Where should the residual be small?** With finitely many coefficients
   the residual cannot vanish everywhere. Collocation makes it vanish at
   chosen points. The rest of this chapter treats collocation, Galerkin and
   least squares as three ways of testing the residual, and the choice of test
   is a choice of where the approximation spends its accuracy.
2. **Which basis?** A global polynomial spreads the difficulty of the cliff
   over the whole interval, while local hat functions confine it to the nodes
   next to the cliff. The basis also decides whether the iteration converges.
   The [next chapter](approximate-bellman-equations.md) shows that
   piecewise-linear interpolation belongs to a class of fits, the averagers,
   for which function iteration is always a contraction, and that polynomial
   interpolation does not.
3. **What should be measured?** The residual is what the method controls,
   the value error is what the residual bounds, and the policy loss is what
   the regulator pays. At $n = 12$, the polynomial has the smaller residual of
   the two approximations and a policy loss eighteen times larger. The next
   chapter bounds the policy loss by the fitting error.

- {download}`Fishery model, reference solver and collocation <code/fishery_tipping_point.py>`
- {download}`Fishery experiment builder <scripts/build_fishery_tipping_point_artifacts.py>`
- {download}`Fishery results for every basis size <artifacts/fishery_tipping_point/metrics.json>`

## Testing Whether a Residual Vanishes

Which scalar conditions can certify that a functional residual is zero, small,
or orthogonal to selected directions?

Consider a functional equation $\Residual(f) = 0$, where $\Residual$ is an operator and the unknown $f$ is an entire function (in our case, the Bellman optimality equation $\Bellman v = v$, which we can write as $\Residual(v) \equiv \Bellman v - v = 0$). Suppose we have found a candidate approximate solution $\hat{f}$. To verify it satisfies $\Residual(\hat{f}) = 0$, we compute the **residual function** $R(s) = \Residual(\hat{f})(s)$. For a true solution, this residual should be the **zero function**: $R(s) = 0$ for every state $s$. 

How might we test whether a function is zero? One approach: sample many input points $\{s_1, s_2, \ldots, s_m\}$, check whether $R(s_i) = 0$ at each, and summarize the results into a single scalar test by computing a weighted sum $\sum_{i=1}^m w_i R(s_i)$ with weights $w_i > 0$. If $R$ is zero everywhere, this sum is zero. If $R$ is nonzero somewhere, we can choose points and weights to make the sum nonzero. For vectors in finite dimensions, the inner product $\langle \mathbf{r}, \mathbf{y} \rangle = \sum_{i=1}^n r_i y_i$ implements exactly this idea: it tests $\mathbf{r}$ by weighting and summing. Indeed, a vector $\mathbf{r} \in \mathbb{R}^n$ equals zero if and only if $\langle \mathbf{r}, \mathbf{y} \rangle = 0$ for every vector $\mathbf{y} \in \mathbb{R}^n$. To see why, suppose $\mathbf{r} \neq \mathbf{0}$. Choosing $\mathbf{y} = \mathbf{r}$ gives $\langle \mathbf{r}, \mathbf{r} \rangle = \|\mathbf{r}\|^2 > 0$, contradicting the claim that all inner products vanish.

The same principle extends to functions. A function $R$ equals the zero function if and only if its "inner product" with every "test function" $p$ vanishes:

$$
R = 0 \quad \text{if and only if} \quad \langle R, p \rangle_w = \int_{\mathcal{S}} R(s) p(s) w(s) ds = 0 \quad \text{for all test functions } p,
$$

where $w(s) > 0$ is a weight function that is part of the inner product definition. Why does this work? For the same reason as in finite dimensions: if $R$ is not the zero function, there must be some region where $R(s) \neq 0$. We can then choose a test function $p$ that is nonzero in that same region (for instance, $p(s) = R(s)$ itself), which will produce $\langle R, p \rangle_w = \int R(s) p(s) w(s) ds > 0$, witnessing that $R$ is nonzero. Conversely, if $R$ is the zero function, then $\langle R, p \rangle_w = 0$ for any test function $p$.

This ability to **distinguish between different functions using inner products** is a fundamental principle from functional analysis. Just as we can test a vector by taking inner products with other vectors, we can test a function by taking inner products with other functions.

```{admonition} Connection to Functional Analysis
:class: dropdown

The principle that "a function equals zero if and only if it has zero inner product with all test functions" is a consequence of the **Hahn-Banach theorem**, one of the cornerstones of functional analysis. The theorem guarantees that for any nonzero function $R$ in a suitable function space, there exists a continuous linear functional (which can be represented as an inner product with some test function $p$) that produces a nonzero value when applied to $R$. This is often phrased as "the dual space separates points."

While you don't need to know the Hahn-Banach theorem to use weighted residual methods, it provides the rigorous mathematical foundation ensuring that our inner product tests are theoretically sound. The constructive argument we gave above (choosing $p = R$) works in simple cases with well-behaved functions, but the Hahn-Banach theorem extends this guarantee to much more general settings.
```

This transforms the pointwise condition "$R(s) = 0$ for all $s$" (infinitely many conditions, one per state) into an equivalent condition about inner products. We still cannot test against *all* possible test functions, since there are infinitely many of those too. But the inner product perspective suggests a natural computational strategy: choose a finite collection of test functions $\{p_1, \ldots, p_n\}$ and use them to construct $n$ conditions that we can actually compute.

### From Variational Conditions to Optimization

Making a residual "small" is an optimization problem. We want to find $\theta$ that minimizes $\lVert R(\cdot; \theta) \rVert$ for some norm. Different methods correspond to different choices of norm:

- Minimize the weighted $L^2$ norm $\lVert R \rVert_w^2 = \int R(s)^2 w(s) ds$
- Minimize a discrete norm $\sum_j \omega_j R(s_j)^2$ at selected points
- Minimize $\lVert R \rVert$ in a dual norm induced by the approximation space

The first-order conditions for these optimization problems take the form $\langle R, p_j \rangle_w = 0$ for appropriate "test functions" $p_j$. The variational formulation is useful for analysis, but we are simply minimizing the residual in a chosen norm.

The rest of this chapter develops the computational framework: how to parameterize the unknown function, define the residual, choose a norm, and solve the resulting finite-dimensional problem. 

## The General Framework

How do an approximation space, residual, test space, and solver combine into a
reusable finite-dimensional method?

Consider an operator equation of the form

$$
\Residual(f) = 0,
$$

where $\Residual: B_1 \to B_2$ is a continuous operator between complete normed vector spaces $B_1$ and $B_2$. For the Bellman equation, we have $\Residual(v) = \Bellman v - v$, so that solving $\Residual(v) = 0$ is equivalent to finding the fixed point $v = \Bellman v$.

Just as we transcribed infinite-dimensional continuous optimal control problems into finite-dimensional discrete optimal control problems in earlier chapters, we seek a finite-dimensional approximation to this infinite-dimensional functional equation. Recall that for continuous optimal control, we adopted control parameterization: we represented the control trajectory using a finite set of basis functions (piecewise constants, polynomials, splines) and searched over the finite-dimensional coefficient space instead of the infinite-dimensional function space. For integrals in the objective and constraints, we used numerical quadrature to approximate them with finite sums.

We follow the same strategy here. We parameterize the value function using a finite set of basis functions $\{\varphi_1, \ldots, \varphi_n\}$, commonly polynomials (Chebyshev, Legendre), though other function classes (splines, radial basis functions, neural networks) are possible, and search for coefficients $\theta = (\theta_1, \ldots, \theta_n)$ in $\mathbb{R}^n$. When integrals appear in the Bellman operator or projection conditions, we approximate them using numerical quadrature. The projection method approach consists of several conceptual steps that accomplish this transcription.

### Step 1: Choose a Finite-Dimensional Approximation Space

We begin by selecting a basis $\Phi = \{\varphi_1, \varphi_2, \ldots, \varphi_n\}$ and approximating the unknown function as a linear combination:

$$
\hat{f}(x) = \sum_{i=1}^n \theta_i \varphi_i(x).
$$

The choice of basis functions $\varphi_i$ is problem-dependent. Common choices include:
- **Polynomials**: For smooth problems, we might use Chebyshev polynomials or other orthogonal polynomial families
- **Splines**: For problems where we expect the solution to have regions of different smoothness
- **Radial basis functions**: For high-dimensional problems where tensor product methods become intractable

The number of basis functions $n$ determines the flexibility of our approximation. In practice, we start with small $n$ and increase it until the approximation quality is satisfactory. The only unknowns now are the coefficients $\theta = (\theta_1, \ldots, \theta_n)$.

Nothing in the framework requires $\theta$ to enter linearly. A neural network $\hat{f}(x; \theta)$ defines a function class just as a basis does, and each residual condition below makes sense for it. What changes is how the conditions are solved: linear coefficients give linear systems with closed-form solutions, while adaptive features give nonlinear systems solved by gradient descent. This chapter uses linear bases because they make the conditions explicit. The [next chapter](approximate-bellman-equations.md) shows that the algorithm that carries over to neural networks, fitted value iteration, needs nothing from the representation beyond the ability to fit it to data.


### Step 2: Define the Residual Function

Since we are approximating $f$ with $\hat{f}$, the operator $\Residual$ will generally not vanish exactly. Instead, we obtain a **residual function**:

$$
R(x; \theta) = \Residual(\hat{f}(\cdot; \theta))(x).
$$

This residual measures how far our candidate solution is from satisfying the equation at each point $x$. As we discussed in the introduction, we want to make this residual small—an optimization problem whose formulation depends on how we measure "small."

(sec-wr-conditions)=
### Step 3: Impose Conditions on the Residual

The basis and residual reduce the functional equation to $n$ scalar conditions
on $\theta$. The choice of conditions determines the method:

| **Method** | **Residual criterion** | **Conditions** ($n$ equations) |
|:-----------|:------------------------|:-------------------------------|
| Least squares | $\displaystyle\lVert R \rVert_w^2 = \int R(x; \theta)^2 w(x) dx$ | $\displaystyle\int R \cdot \frac{\partial R}{\partial \theta_j} \, w \, dx = 0$, $j = 1, \ldots, n$ |
| Galerkin | $\lVert R \rVert_{\mathcal{V}^*}$ (dual norm of approx. space) | $\displaystyle\int R(x; \theta) \varphi_j(x) w(x) dx = 0$, $j = 1, \ldots, n$ |
| Collocation | Exact pointwise satisfaction | $R(x_i; \theta) = 0$, $i = 1, \ldots, n$ |

Each criterion yields $n$ equations in the $n$ unknowns
$\theta_1,\ldots,\theta_n$.

#### Collocation: Make the Residual Zero at Selected Points

The simplest approach is to choose $n$ points $\{x_1, \ldots, x_n\}$ and require the residual to vanish exactly at each:

$$
R(x_i; \theta) = 0, \quad i = 1, \ldots, n.
$$

This gives $n$ equations for $n$ unknowns. Collocation is computationally attractive because it avoids integration entirely—we only evaluate $R$ at discrete points. The resulting system is:

$$
\Residual(\hat{f}(\cdot; \theta))(x_i) = 0, \quad i = 1, \ldots, n.
$$

For a linear operator, this is a linear system; for the Bellman equation, it is nonlinear due to the max.

*Verify for yourself: with $n = 2$ collocation points and $n = 2$ basis functions, the system $\boldsymbol{\Phi}\theta = t$ is a $2 \times 2$ linear system. What must be true about the collocation matrix $\boldsymbol{\Phi}$ for this system to have a unique solution?*

The choice of collocation points matters. **Orthogonal collocation** (or **spectral collocation**) places points at the zeros of the $n$-th orthogonal polynomial in a family (Chebyshev, Legendre, etc.). For Chebyshev polynomials $T_0, T_1, \ldots, T_{n-1}$, we place collocation points at the zeros of $T_n(x)$. These points are also optimal nodes for Gauss quadrature, so:

- We get the computational simplicity of pointwise evaluation $R(x_i) = 0$
- When we need integrals (inside the Bellman operator), the collocation points double as quadrature nodes with exactness for polynomials up to degree $2n-1$
- For smooth problems, spectral approximations achieve **exponential convergence**: the error decreases like $O(e^{-cn})$ as we add basis functions, compared to $O(h^{p+1})$ for piecewise polynomials

The Chebyshev interpolation theorem guarantees that forcing $R(x_i; \theta) = 0$ at these carefully chosen points makes $R(x; \theta)$ small everywhere, with well-conditioned systems and near-optimal interpolation error.

#### Galerkin: Make the Residual Orthogonal to the Approximation Space

The Galerkin method requires the residual to be orthogonal to each basis function:

$$
\int_{\mathcal{S}} R(x; \theta) \varphi_i(x) w(x) dx = 0, \quad i = 1, \ldots, n.
$$

To understand why this is optimal, consider the approximation space $\mathcal{V} = \text{span}\{\varphi_1, \ldots, \varphi_n\}$ as an $n$-dimensional subspace. If the residual $R$ is orthogonal to all basis functions, then by linearity, $R$ is orthogonal to every function in $\mathcal{V}$:

$$
\langle R, g \rangle_w = \left\langle R, \sum_{i=1}^n c_i \varphi_i \right\rangle_w = \sum_{i=1}^n c_i \langle R, \varphi_i \rangle_w = 0 \quad \text{for all } g \in \mathcal{V}.
$$

The residual has "zero overlap" with our approximation space—it is as "invisible" to our basis as possible. This is the defining property of orthogonal projection.

In what sense is Galerkin minimizing a norm? The **dual norm** of $R$ with respect to $\mathcal{V}$ measures $R$ by its largest inner product with functions in $\mathcal{V}$:

$$
\lVert R \rVert_{\mathcal{V}^*} = \sup_{\substack{g \in \mathcal{V} \\ \lVert g \rVert_w = 1}} \lvert \langle R, g \rangle_w \rvert.
$$

The Galerkin conditions $\langle R, \varphi_j \rangle_w = 0$ for all $j$ imply $\langle R, g \rangle_w = 0$ for all $g \in \mathcal{V}$, so $\lVert R \rVert_{\mathcal{V}^*} = 0$. Galerkin makes the residual "invisible" when measured against the approximation space—it minimizes the dual norm to zero.

A finite-dimensional analogy: to approximate a vector $\mathbf{v} \in \mathbb{R}^3$ using only the $xy$-plane, the best approximation is $\hat{\mathbf{v}} = (v_1, v_2, 0)$. The error $\mathbf{r} = \mathbf{v} - \hat{\mathbf{v}} = (0, 0, v_3)$ points purely in the $z$-direction, orthogonal to the plane. The Galerkin condition generalizes this: the residual is orthogonal to the approximation space.

Galerkin requires integration to compute the conditions, making it more expensive per iteration than collocation. However, when using orthogonal polynomial bases with matching weight functions, the integrals simplify and the resulting systems are well-conditioned.

Collocation fits the same pattern if its test functions are taken to be Dirac deltas $\delta(x - x_i)$, since $\langle R, \delta(x - x_i) \rangle = R(x_i)$. Deltas are not functions but distributions, and the **dual pairing** formulation {cite}`LegrandJunca2025` makes this precise: test objects act on residuals through a pairing rather than an integral, and the induced dual norm $\|\Residual(v)\|_* = \sup_{\|w\|=1} |\langle \Residual(v), w \rangle|$ measures a residual by its worst-case effect on the test objects. The same perspective has inspired adversarial formulations {cite}`Zang2020` in which both trial and test functions are learned.

#### Least Squares: Minimize the $L^2$ Norm of the Residual

The most direct approach is to minimize the weighted $L^2$ norm of the residual:

$$
\min_\theta \int_{\mathcal{S}} R(x; \theta)^2 w(x) dx.
$$

The first-order conditions are:

$$
\int_{\mathcal{S}} R(x; \theta) \frac{\partial R(x; \theta)}{\partial \theta_j} w(x) dx = 0, \quad j = 1, \ldots, n.
$$

This directly minimizes how far our approximation is from satisfying the equation. For the Bellman equation $R = \Bellman\hat{v} - \hat{v}$, this is **Bellman residual minimization**: we minimize $\lVert \Bellman\hat{v} - \hat{v} \rVert_w^2$.

The gradient $\frac{\partial R}{\partial \theta_j}$ involves differentiating the operator $\Residual$. For the Bellman operator with its max, this requires the Envelope Theorem (discussed in Step 4). The need to differentiate through the operator distinguishes least squares from Galerkin and collocation.

(sec-wr-solve)=
### Step 4: Solve the Finite-Dimensional Problem

The conditions from Step 3 give us a finite-dimensional problem to solve:

- **Collocation**: $n$ equations $R(x_i; \theta) = 0$
- **Galerkin**: $n$ equations $\int R(x; \theta) \varphi_i(x) w(x) dx = 0$
- **Least squares**: minimize $\int R(x; \theta)^2 w(x) dx$

In each case, we have $n$ equations (or first-order conditions) in $n$ unknowns $\theta_1, \ldots, \theta_n$. For the Bellman equation, these systems are nonlinear due to the max operator.

#### Computational Cost and Conditioning

The **computational cost per iteration** varies significantly across methods:
- **Collocation**: Cheapest to evaluate since $P_i(\theta) = R(x_i; \theta)$ requires only pointwise evaluation (no integration). The Jacobian is also cheap: $J_{ij} = \frac{\partial R(x_i; \theta)}{\partial \theta_j}$.
- **Galerkin and moments**: More expensive due to integration. Computing $P_i(\theta) = \int R(x; \theta) p_i(x) w(x) dx$ requires numerical quadrature. Each Jacobian entry requires integrating $\frac{\partial R}{\partial \theta_j} p_i w$.
- **Least squares**: Most expensive when done via the objective function, which requires integrating $R^2 w$. However, the first-order conditions reduce it to a system like Galerkin, with test functions $p_i = \partial R / \partial \theta_i$.

For methods requiring integration, the choice of quadrature rule should match the basis. Gaussian quadrature with nodes at orthogonal polynomial zeros is efficient. When combined with collocation at those same points, the quadrature is exact for polynomials up to a certain degree. This coordination between quadrature and collocation makes **orthogonal collocation** effective.

The **conditioning** of the system depends on the choice of test functions. The Jacobian matrix has entries:

$$
J_{ij} = \frac{\partial P_i}{\partial \theta_j} = \left\langle \frac{\partial R(\cdot; \theta)}{\partial \theta_j}, p_i \right\rangle_w.
$$

When test functions are orthogonal (or nearly so), the Jacobian tends to be well-conditioned. This is why orthogonal polynomial bases are preferred in Galerkin methods: they produce Jacobians with controlled condition numbers. Poorly chosen basis functions or collocation points can lead to nearly singular Jacobians, causing numerical instability. Orthogonal bases and carefully chosen collocation points (like Chebyshev nodes) help maintain good conditioning.

#### Two Main Solution Approaches

We have two fundamentally different ways to solve the projection equations: **function iteration** (exploiting fixed-point structure) and **Newton's method** (exploiting smoothness). The choice depends on whether the original operator equation has contraction properties and how well those properties are preserved by the finite-dimensional approximation.

(sec-wr-function-iteration)=
##### Method 1: Function Iteration (Successive Approximation)

When the operator equation has the form $f = \Contraction f$ where $\Contraction$ is a contraction, the most natural approach is to iterate the operator directly:

$$
\hat{f}^{(k+1)} = \Contraction \hat{f}^{(k)}.
$$

The infinite-dimensional iteration becomes a finite-dimensional iteration in coefficient space once we choose our weighted residual method. Given a current approximation $\hat{f}^{(k)}(x; \theta^{(k)})$, how do we find the coefficients $\theta^{(k+1)}$ for the next iterate $\hat{f}^{(k+1)}$?

Different weighted residual methods answer this differently. For **collocation**, we proceed in two steps:

1. **Evaluate the operator**: At each collocation point $x_i$, compute what the next iterate should be: $t_i^{(k)} = (\Contraction \hat{f}^{(k)})(x_i)$. These $n$ target values tell us what $\hat{f}^{(k+1)}$ should equal at the collocation points.

2. **Find matching coefficients**: Determine $\theta^{(k+1)}$ so that $\hat{f}^{(k+1)}(x_i; \theta^{(k+1)}) = t_i^{(k)}$ for all $i$. This is a linear system: $\sum_j \theta_j^{(k+1)} \varphi_j(x_i) = t_i^{(k)}$.

In matrix form: $\boldsymbol{\Phi} \theta^{(k+1)} = t^{(k)}$, where $\boldsymbol{\Phi}$ is the collocation matrix with entries $\Phi_{ij} = \varphi_j(x_i)$. Solving this system gives $\theta^{(k+1)} = \boldsymbol{\Phi}^{-1} t^{(k)}$.

For **Galerkin**, the target $\Contraction \hat{f}^{(k)}$ is a function rather than $n$ numbers, and the next iterate is its projection: $\theta^{(k+1)}$ solves $\langle \hat{f}^{(k+1)} - \Contraction \hat{f}^{(k)}, \varphi_i \rangle_w = 0$ for $i = 1, \ldots, n$. Because the target is held fixed at the previous iterate, this is a linear system whatever $\Contraction$ is: $M \theta^{(k+1)} = b^{(k)}$ with the mass matrix $M_{ij} = \langle \varphi_i, \varphi_j \rangle_w$ and $b_i^{(k)} = \langle \Contraction \hat{f}^{(k)}, \varphi_i \rangle_w$. It is the weighted least-squares fit of the basis to the target. Least squares in the sense of Step 3 would instead fit $\Contraction \hat{f}^{(k+1)}$, which depends on the unknown coefficients through $\Contraction$; the two differ exactly in whether the operator is differentiated.

Both cases have the same shape. Write $\Proj$ for the map that takes a function to its representative in $\text{span}\{\varphi_1, \ldots, \varphi_n\}$: the interpolant through the collocation points, or the $w$-orthogonal projection for Galerkin. Then function iteration is

$$
\hat{f}^{(k+1)} = \Proj\, \Contraction \hat{f}^{(k)},
$$

the original iteration with a projection inserted after each application of the operator. When $\Contraction$ is a contraction in the infinite-dimensional space with constant $\gamma < 1$, iterating it pulls any starting function toward the unique fixed point. The hope is that the composed operator $\Proj\,\Contraction$ inherits this contraction property. When it does, function iteration converges globally from any initial guess, with each iteration reducing the error by a factor of roughly $\gamma$. This is computationally attractive: we only evaluate the operator and solve a linear system. Whether $\Proj$ preserves the contraction is the main question of the [next chapter](approximate-bellman-equations.md).

However, the finite-dimensional approximation doesn't always preserve contraction. High-order polynomial bases, in particular, can create oscillations between basis functions that amplify rather than contract errors. Even when contraction is preserved, convergence can be painfully slow when $\gamma$ is close to 1, the "weak contraction" regime common in economic problems with patient agents ($\gamma \approx 0.95$ or higher). Finally, not all operator equations naturally present themselves as contractions; some require reformulation (like $f = f - \alpha \Residual(f)$), and finding a good $\alpha$ can be problem-specific.

(sec-wr-newton)=
##### Method 2: Newton's Method

Alternatively, we can treat the projection equations as a rootfinding problem $G(\theta) = 0$ where $G_i(\theta) = P_i(\theta)$ for test function methods, or solve the first-order conditions for least squares. **Newton's method** uses the update:

$$
\theta^{(k+1)} = \theta^{(k)} - J_G(\theta^{(k)})^{-1} G(\theta^{(k)}),
$$

where $J_G(\theta)$ is the Jacobian of $G$ at $\theta$.

To apply this update, we must compute the Jacobian entries $J_{ij} = \frac{\partial G_i}{\partial \theta_j}$. For collocation, $G_i(\theta) = \hat{f}(x_i; \theta) - (\Contraction \hat{f}(\cdot; \theta))(x_i)$, so:

$$
\frac{\partial G_i}{\partial \theta_j} = \frac{\partial \hat{f}(x_i; \theta)}{\partial \theta_j} - \frac{\partial (\Contraction \hat{f}(\cdot; \theta))(x_i)}{\partial \theta_j}.
$$

The first term is straightforward (it's just $\varphi_j(x_i)$ for a linear approximation). The second term requires differentiating the operator $\Contraction$ with respect to the parameters.

When $\Contraction$ involves optimization (as in the Bellman operator $\Bellman v = \max_a \{r(s,a) + \gamma \mathbb{E}[v(s')]\}$), computing this derivative appears problematic because the max operator is not differentiable. However, the **Envelope Theorem** resolves this difficulty.

*Before reading the box below, try differentiating $v(\theta) = \max_x f(x, \theta)$ using the chain rule. What term involving $\partial x^*/\partial \theta$ appears? Why might this term vanish at an optimum?*

```{admonition} The Envelope Theorem
:class: important

**Setup:** Consider a smooth objective function $f(\mathbf{x}, \boldsymbol{\theta})$ and define the optimal value:

$$
v(\boldsymbol{\theta}) = \max_{\mathbf{x}} f(\mathbf{x}, \boldsymbol{\theta}).
$$

Let $\mathbf{x}^*(\boldsymbol{\theta})$ denote the maximizer, satisfying the first-order condition $\nabla_{\mathbf{x}} f(\mathbf{x}^*(\boldsymbol{\theta}), \boldsymbol{\theta}) = \mathbf{0}$.

**The Result:** To find how the optimal value changes with $\boldsymbol{\theta}$, we can compute:

$$
\nabla_{\boldsymbol{\theta}} v(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} f(\mathbf{x}^*(\boldsymbol{\theta}), \boldsymbol{\theta}).
$$

That is, differentiate the objective with respect to $\boldsymbol{\theta}$ while treating the maximizer $\mathbf{x}^*$ as constant. We don't need to compute $\frac{\partial \mathbf{x}^*}{\partial \boldsymbol{\theta}}$ because at the optimum, small changes in $\mathbf{x}$ don't affect the value (first-order condition), so the direct effect dominates.

**Why it works:** By the chain rule, $\nabla_{\boldsymbol{\theta}} v = \nabla_{\boldsymbol{\theta}} f + \underbrace{(\nabla_{\mathbf{x}} f)^{\top}}_{\mathbf{0} \text{ at optimum}} \frac{\partial \mathbf{x}^*}{\partial \boldsymbol{\theta}}$.

**Application to Bellman equations:** For $[\Bellman v](s) = \max_a \{r(s,a) + \gamma \mathbb{E}[v(s')]\}$, the derivative with respect to parameters in $v$ can be computed by treating the optimal action as constant. For example, if $v(s; \theta) = \sum_j \theta_j \varphi_j(s)$:

$$
\frac{\partial [\Bellman v](s)}{\partial \theta_j} = \gamma \mathbb{E}[\varphi_j(s') \mid s, a^*(s; \theta)],
$$

where $a^*(s; \theta)$ is the optimal action given parameters $\theta$.

**Important assumptions:** The objective $f$ is smooth, the maximizer is unique and in the interior (or constraints are smooth with stable active sets), and the first-order condition holds.
```

With the Envelope Theorem providing a tractable way to compute Jacobians for problems involving optimization, Newton's method becomes practical for weighted residual methods applied to Bellman equations and similar problems. The method offers **quadratic convergence** near the solution. Once in the neighborhood of the true fixed point, Newton's method typically converges in just a few iterations. Unlike function iteration, it doesn't rely on the finite-dimensional approximation preserving any contraction property, making it applicable to a broader range of problems, particularly those with high-order polynomial bases or large discount factors where function iteration struggles.

However, Newton's method demands more from both the algorithm and the user. Each iteration requires computing and solving a full Jacobian system, making the per-iteration cost significantly higher than function iteration. The method is also sensitive to initialization: started far from the solution, Newton's method may diverge or converge to spurious fixed points that the finite-dimensional problem introduces but the original infinite-dimensional problem lacks. When we apply the Envelope Theorem, implementation becomes more complex. We must track the optimal action at each evaluation point and compute the Jacobian entries using the formula above (expected basis function values at next states under optimal actions), though the economic interpretation (tracking how value propagates through optimal decisions) often makes the computation conceptually clearer than explicit derivative calculations would be.

##### Comparison and Practical Recommendations

| **Method** | **Convergence** | **Per-iteration cost** | **Initial guess sensitivity** |
|:-----------|:----------------|:-----------------------|:------------------------------|
| **Function iteration** | Linear (when contraction holds) | Low | Robust |
| **Newton's method** | Quadratic (near solution) | Moderate (Jacobian + solve) | Requires good initial guess |

Which method to use? When the problem has strong contraction (small $\gamma$, well-conditioned bases, shape-preserving approximations like linear interpolation or splines), function iteration is simple and robust. For weak contraction (large $\gamma$, high-order polynomials), a hybrid approach works well: run function iteration for several iterations to enter the basin of attraction, then switch to Newton's method for rapid final convergence. When the finite-dimensional approximation destroys contraction entirely (common with non-monotone bases), Newton's method may be necessary from the start, though careful initialization (from a coarser approximation or perturbation methods) is required.

Quasi-Newton methods like BFGS or Broyden offer a middle ground. They approximate the Jacobian using function evaluations only, avoiding explicit derivative computations while maintaining superlinear convergence. This can be useful when computing the exact Jacobian via the Envelope Theorem is expensive or when the approximation quality is acceptable.

### Step 5: Verify the Solution

Once we have computed a candidate solution $\hat{f}$, we must verify its quality. Projection methods optimize $\hat{f}$ with respect to specific criteria (specific test functions or collocation points), but we should check that the residual is small everywhere, including directions or points we did not optimize over.

Typical diagnostic checks include:
- Computing $\|R(\cdot; \theta)\|$ using a more accurate quadrature rule than was used in the optimization
- Evaluating $R(x; \theta)$ at many points not used in the fitting process
- If using Galerkin with the first $n$ basis functions, checking orthogonality against higher-order basis functions

In summary, we have established a template: parameterize the unknown function using basis functions, define a residual measuring how far from a solution we are, and impose conditions via inner products with test functions. Different test functions yield different methods: Galerkin uses the basis itself, collocation uses delta functions at chosen points, and least squares uses residual gradients. We now apply this framework to the Bellman equation.

## Summary and Outlook

Weighted-residual methods replace an unknown function by finitely many basis
coefficients and determine them by testing the residual. Collocation tests at
selected points, Galerkin methods test against basis functions, and least
squares minimizes an aggregate residual. Each choice specifies what it means
for an approximate function to satisfy the original equation.

The framework is independent of the equation being solved, and Step 4 left
two things open. Function iteration becomes $\hat{f}^{(k+1)} = \Proj\,\Contraction \hat{f}^{(k)}$,
and we hoped, without proof, that $\Proj$ keeps the contraction. Newton's
method needs the Jacobian of the operator, and for a Bellman operator the
envelope theorem gave it in terms of the greedy action. [Approximate Bellman
equations](approximate-bellman-equations.md) takes up both for an arbitrary
fitting rule. Function iteration becomes fitted value iteration, a sequence of
regression problems with Bellman targets that any regressor can solve, and
the pattern behind fitted Q-iteration, NFQ and NFQCA. Newton's method becomes
approximate policy iteration. The chapter then asks when the fit preserves the
contraction, and shows what can go wrong when it does not.

## Self-checks

:::{exercise} Orthogonality condition
:label: ex-projection-check-1

In a Galerkin method with basis functions $\phi_i$, against which functions is the Bellman residual required to be orthogonal?
:::

:::{solution} ex-projection-check-1
:class: dropdown

Against the same basis functions: $\langle \phi_i,\Bellman v-v\rangle=0$ for every $i$.
:::

:::{exercise} Collocation versus least squares
:label: ex-projection-check-2

What is the main distinction between collocation and least-squares residual fitting?
:::

:::{solution} ex-projection-check-2
:class: dropdown

Collocation forces the residual to vanish at selected points. Least squares minimizes an aggregate squared residual over a sampling or weighting distribution.
:::
