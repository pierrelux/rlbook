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
# Parametric Value Functions

The dynamic programming chapters solved Bellman equations by storing one
number per state, or per grid state. That works for the reservoir, the
antibiotic regimen and a river of eight reaches, and it fails soon after. This
part of the book replaces the table by a function with a fixed number of
parameters, and it builds up to the algorithms used in practice: fitted
Q-iteration with trees or neural networks, its online form DQN, and its
continuous-action form NFQCA. This chapter starts with why the replacement is
needed and what it must provide. It then asks the first question that the
replacement raises. A function with finitely many parameters cannot satisfy
the Bellman equation at every state, so which equations should determine its
parameters? Numerical analysis answers this question for operator equations in
general with the method of weighted residuals, a template of five steps. We
present the template directly on the Bellman equation. One problem, a fishery
near a tipping point, motivates it and then follows it through every step, in
boxes that the method does not depend on.

::::{admonition} Learning Goals
:class: note

After completing this chapter, you should be able to:
- Explain why tabular dynamic programming fails for large or continuous state spaces, and what a function approximator must provide in its place
- Write the Bellman residual of a parametric value function, and derive the collocation, Galerkin and least-squares conditions that determine its parameters
- Solve the resulting equations by function iteration or by Newton's method, and state what each requires of the problem
- Judge an approximate value function by its residual, its value error and the loss of its greedy policy, and explain why the three can disagree

**Prerequisites:** Bellman operator and contraction property ([infinite-horizon MDPs](infinite-horizon-mdps.md)), interpolation with cardinal functions ([finite-horizon DP](finite-horizon-dp.md)), basic numerical quadrature, linear algebra (matrix inverses, orthogonal projection).
::::

(sec-why-approximate)=
## Limits of Tabular Dynamic Programming

The tables of the previous chapters cannot simply be made larger, for three
reasons, each already visible in an example we have solved.

The first is that the number of states grows exponentially with the number
of state variables. The tamarisk river of {ref}`sec-tamarisk-river` has one site per
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

The second is that successors fall between the stored states. Even in one
dimension, the
[finite-horizon chapter](finite-horizon-dp.md#handling-continuous-spaces-with-interpolation)
had to decide which value to use for a successor that lands between two grid
states. Its answer, a family of cardinal functions, turned the table into a
function defined everywhere, and its analysis of how the reconstruction error
propagates through the recursion was a first instance of what this part of the
book studies in general.

The third is that learning from data visits only some states. In the
chapters ahead the
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
with the weighted-residual template of numerical analysis
{cite}`Chakraverty2019,AtkinsonPotra1987`, using linear bases, for which the
conditions can be written out and solved in closed form. Once the conditions
are fixed, which algorithm finds $\theta$, and does the contraction that made
value iteration converge survive the approximation? The
[next chapter](approximate-bellman-equations.md) answers the second question
for an arbitrary fitting rule, neural networks included, and ends with the
steps that lead from it to FQI and NFQCA.

(sec-wr-fishery)=
## A Motivating Example: Harvesting a Fishery Near a Tipping Point

This example shows what goes wrong when a value function on a continuum is
stored in a few numbers, and how the way it is stored changes the decisions
it produces. A regulator sets the annual quota for a fish stock. Before the season, a survey
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
the nodes. {numref}`fig-fishery-collocation` shows both approximations with
$n = 12$.

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

The example leaves three questions open, and the rest of the chapter takes
them in order. Collocation asked for a zero residual at twelve points; where
else could the residual be made small, and what does that choice decide? The
twelve coefficients were found by repeating a backup and a fit; when does that
repetition converge, and what else could find them? And the polynomial had the
smaller residual of the two approximations and a policy loss eighteen times
larger; which quantity should judge an approximation?

- {download}`Fishery model, reference solver and collocation <code/fishery_tipping_point.py>`
- {download}`Fishery experiment builder <scripts/build_fishery_tipping_point_artifacts.py>`
- {download}`Fishery results for every basis size <artifacts/fishery_tipping_point/metrics.json>`

(sec-wr-framework)=
## From One Equation per State to $n$ Equations

The Bellman optimality equation $v = \Bellman v$ holds at every state. For
the fishery, that is one equation for every stock in $[0, 1.6]$, and the
unknown is the whole function $v^\star$. Replacing $v$ by $v(\cdot;\theta)$
leaves $n$ unknowns, so only $n$ conditions can be imposed, and the choice of
those conditions is the method.

The earlier chapters made the same move for trajectories. A continuous-time
trajectory cannot be chosen one instant at a time, so
[direct collocation](continuous-time-collocation.md) represented the state
and the control by polynomial pieces with finitely many coefficients,
enforced the differential equation at selected points of each interval, and
replaced the integrals of the cost by quadrature. The value function is
treated in the same way: a basis for the unknown function, finitely many
conditions in place of the equation, and quadrature for the expectations.
The method of weighted residuals {cite}`Chakraverty2019,AtkinsonPotra1987`
organizes these choices into five steps. Each step below is stated for a
general Bellman equation and names the alternatives at that fork. The fishery
took one path through the steps, and a box after each step picks the example
up where it went through that step; the steps themselves need nothing from
the boxes.

### Step 1: Choose a Parametric Value Function

The unknown value function is replaced by a family $v(s;\theta)$ indexed by
a parameter vector $\theta \in \mathbb{R}^n$. Nothing in the five steps
requires $\theta$ to enter linearly: a neural network $v(s;\theta)$ with $n$
weights is such a family, and every condition below applies to it. Linearity
matters for solving. When the family is a linear combination of $n$ basis
functions,

$$
v(s;\theta) = \sum_{j=1}^n \theta_j \varphi_j(s),
$$

each condition reduces to linear algebra and can be written out, so this
chapter uses linear families to make the steps explicit. The
[next chapter](approximate-bellman-equations.md) shows that the solver that
carries over to neural networks needs nothing from the representation beyond
the ability to fit it to data.

Two kinds of basis are in common use. A global basis, such as Chebyshev
polynomials or Fourier modes, gives every coefficient influence over the whole
state space, and its error decreases exponentially in $n$ when the function
is smooth. A local basis, such as the hat functions of piecewise-linear
interpolation, splines with small support, or radial basis functions, lets
each coefficient act near one node, and its error decreases at a rate set by
the node spacing, whatever the smoothness of the function.

::::{admonition} The fishery at Step 1
:class: note

The fishery used one basis of each kind, with $n = 12$: the Chebyshev
polynomials $\varphi_j(x) = T_{j-1}(2x/1.6 - 1)$, and the hat functions of
twelve uniform nodes, for which $\theta_j$ is the value at node $j$. Its value
function has a kink at $\hat x$ and a cliff at the tipping point, which is why
the two bases behaved so differently with the same $n$: the polynomial spread
the cliff over the whole interval, and the hat functions confined it to the
two nodes around it.
::::

### Step 2: Define the Bellman Residual

Substituting the approximation into the Bellman equation leaves a function of
the state,

$$
R(s;\theta) = (\Bellman v_\theta)(s) - v(s;\theta),
$$ (eq:bellman-residual)

the **Bellman residual**, where $v_\theta$ abbreviates $v(\cdot;\theta)$. It
measures by how much the approximation fails the equation at $s$, and it is
the zero function exactly when $v_\theta$ is a fixed point of $\Bellman$. One
evaluation of it costs one evaluation of the Bellman operator, a maximization
over the actions and an expectation over the successor, with the current
coefficients inside both. The coefficients enter $R$ twice, directly through
$v(s;\theta)$ and again through the expectation inside the maximum, and the
maximum makes $R$ nonlinear in $\theta$ even though $v_\theta$ is linear in
it.

::::{admonition} The fishery at Step 2
:class: note

With the stock as the state and the escapement as the action, the residual is

$$
R(x;\theta) = \max_{0 \le s \le x}\Big\{x - s + \gamma\,\mathbb{E}_z\big[v(z f(s);\theta)\big]\Big\} - v(x;\theta),
$$

and one evaluation costs a maximization over the escapement and a 21-node
quadrature over the shock. Panel (b) of {numref}`fig-fishery-collocation`
plots this function for the two fits with $n = 12$.
::::

(sec-wr-conditions)=
### Step 3: Impose $n$ Conditions on the Residual

The residual is a function and $\theta$ has $n$ components, so the condition
$R(\cdot;\theta) = 0$ must be replaced by $n$ scalar conditions. The choices
come from how a function can be tested for being zero. A vector
$\mathbf{r} \in \mathbb{R}^m$ is zero if and only if
$\langle \mathbf{r}, \mathbf{y} \rangle = 0$ for every $\mathbf{y}$, since
$\mathbf{y} = \mathbf{r}$ gives $\|\mathbf{r}\|^2$. The same holds for
functions with the weighted inner product

$$
\langle R, p \rangle_w = \int_{\mathcal{S}} R(s)\, p(s)\, w(s)\, ds,
$$

where $w > 0$ is a weight over states: $R$ is the zero function if and only
if $\langle R, p \rangle_w = 0$ for every **test function** $p$. Testing
against every $p$ is as impossible as testing at every state, but testing
against $n$ chosen test functions $p_1, \ldots, p_n$ gives exactly $n$
equations,

$$
\langle R(\cdot;\theta), p_i \rangle_w = 0, \qquad i = 1, \ldots, n.
$$ (eq:weighted-residual-conditions)

The choice of test functions is the method, and three choices account for
almost everything in use.

| **Method** | **Test functions** $p_i$ | **Conditions** ($n$ equations) |
|:-----------|:--------------------------|:-------------------------------|
| Collocation | Point masses at chosen states $s_1, \ldots, s_n$ | $R(s_i; \theta) = 0$ |
| Galerkin | The basis functions $\varphi_i$ | $\langle R(\cdot;\theta), \varphi_i \rangle_w = 0$ |
| Least squares | The derivatives $\partial R / \partial \theta_i$ of the residual | $\langle R(\cdot;\theta), \partial R/\partial\theta_i \rangle_w = 0$ |

#### Collocation: a Zero Residual at Chosen States

Collocation picks $n$ states and asks the residual to vanish there,
$R(s_i;\theta) = 0$ for $i = 1, \ldots, n$. A point mass at $s_i$ picks out
the value $R(s_i)$, so this is {eq}`eq:weighted-residual-conditions` with
point masses as test functions, as the dual-pairing formulation of
{cite:t}`LegrandJunca2025` makes precise. No integral over states appears, so
each condition costs one evaluation of the Bellman operator. What the
conditions buy is a residual that vanishes at the nodes and is unconstrained
between them.

Where to collocate is a choice of its own. For a polynomial of degree
$n - 1$, the standard choice is the zeros of the $n$-th Chebyshev
polynomial, the **Chebyshev nodes**, and the combination is called
**orthogonal collocation**. The same nodes serve as the nodes of
Gauss–Chebyshev quadrature, the interpolation matrix
$\Phi_{ij} = \varphi_j(s_i)$ built on them is well conditioned, and the
polynomial through them is within a small factor of the best polynomial
approximation of a smooth function, so a residual that vanishes at them stays
small between them. Equally spaced nodes have none of these properties once
the degree is high. For a local basis the natural collocation points are the
nodes of the basis itself, where one hat function equals one and the others
vanish, so that the interpolation matrix is the identity and the coefficients
are the values at the nodes.

Write $\Proj$ for interpolation through the collocation points, the map that
sends a function $g$ to the element of the span with $(\Proj g)(s_i) = g(s_i)$
at every node. The collocation conditions say that $v_\theta$ and
$\Bellman v_\theta$ agree at the nodes, that is,

$$
v_\theta = \Proj\, \Bellman v_\theta .
$$ (eq:projected-bellman)

The approximate solution is a fixed point of the Bellman operator followed by
a projection, which is the form that Step 4 exploits.

::::{admonition} The fishery at Step 3
:class: note

The fishery used collocation, with the two standard choices of nodes: the
Chebyshev nodes for the polynomial, and the interpolation nodes for the hat
functions. Panel (b) of {numref}`fig-fishery-collocation` shows the residuals
of the two fits: zero at the twelve nodes of each basis, and about $0.2$ at
their peaks between the nodes, for both bases.
::::

#### Galerkin: a Residual Orthogonal to the Basis

Galerkin takes the basis functions themselves as test functions,
$\langle R(\cdot;\theta), \varphi_i \rangle_w = 0$ for $i = 1, \ldots, n$.
By linearity the residual is then orthogonal to every function in the
approximation space $\mathcal{V} = \text{span}\{\varphi_1, \ldots, \varphi_n\}$,

$$
\Big\langle R, \sum_{i=1}^n c_i \varphi_i \Big\rangle_w = \sum_{i=1}^n c_i \langle R, \varphi_i \rangle_w = 0 \quad \text{for every } c \in \mathbb{R}^n,
$$

which is the defining property of an orthogonal projection: the component of
$\Bellman v_\theta$ that lies in $\mathcal{V}$ is $v_\theta$ itself. To
approximate a vector of $\mathbb{R}^3$ by a vector in the $xy$-plane, one
keeps $(v_1, v_2, 0)$, and the error $(0, 0, v_3)$ is orthogonal to the plane.
With $\Proj$ now the $w$-orthogonal projection onto $\mathcal{V}$, the
Galerkin conditions read again $v_\theta = \Proj\,\Bellman v_\theta$, the
projected Bellman equation {eq}`eq:projected-bellman` with a different
$\Proj$.

The conditions are integrals over the state space, so Galerkin needs a
quadrature rule for them on top of the quadrature inside $\Bellman$. With an
orthogonal polynomial basis and its matching weight, the inner products among
basis functions are known in closed form and the systems are well
conditioned. When the states are instead sampled from $w$, the integrals
become sample averages and the Galerkin conditions become the normal
equations of a least-squares regression: $v_\theta$ is the least-squares fit
of the basis to the backups $(\Bellman v_\theta)(s_m)$ at the sampled states.
The targets of that regression depend on its own solution, and Step 4 breaks
the circularity.

#### Least Squares: the Smallest Residual

The most direct choice minimizes the weighted norm of the residual,

$$
\min_\theta \int_{\mathcal{S}} R(s; \theta)^2\, w(s)\, ds,
$$

whose first-order conditions are {eq}`eq:weighted-residual-conditions` with
test functions $p_i = \partial R / \partial \theta_i$. For the Bellman
residual this is **Bellman residual minimization**. Its test functions,

$$
\frac{\partial R(s;\theta)}{\partial \theta_i} = \frac{\partial (\Bellman v_\theta)(s)}{\partial \theta_i} - \varphi_i(s),
$$

contain the derivative of the Bellman operator with respect to the
coefficients, which Galerkin and collocation never need. Computing it means
differentiating through the maximum, which Step 4 does with the envelope
theorem. The solution also differs from the Galerkin solution, because the
residual is made orthogonal to a different set of functions, and it is not a
projected Bellman equation. The [next chapter](approximate-bellman-equations.md)
returns to the difference between fitting a backup and minimizing the
residual, which separates the algorithms used in practice from this one.

```{admonition} Why testing against functions is enough
:class: dropdown

The constructive argument above, choosing $p = R$, needs $R$ to be a function
with a finite norm. The general statement, that a nonzero element of a normed
space is detected by some continuous linear functional, is a consequence of
the **Hahn–Banach theorem**, often phrased as "the dual space separates
points". It is also what makes Galerkin a minimization in its own right. The
dual norm of $R$ with respect to $\mathcal{V}$,

$$
\lVert R \rVert_{\mathcal{V}^*} = \sup_{g \in \mathcal{V},\, \lVert g \rVert_w = 1} \lvert \langle R, g \rangle_w \rvert,
$$

measures the residual by its largest inner product with the approximation
space, and the Galerkin conditions make it zero. Point masses are not
functions but distributions, and the dual-pairing formulation of
{cite:t}`LegrandJunca2025` treats collocation and Galerkin uniformly: test
objects act on the residual through a pairing, and the induced norm measures
a residual by its worst case over the test objects. The same view motivates
adversarial formulations {cite}`Zang2020` in which both the approximation and
the test functions are learned.
```

(sec-wr-solve)=
### Step 4: Solve the $n$ Equations

The conditions of Step 3 are $n$ equations in the $n$ unknowns, $G(\theta) = 0$
with $G_i(\theta) = \langle R(\cdot;\theta), p_i \rangle_w$, or $G_i(\theta) =
R(s_i;\theta)$ for collocation. They are nonlinear, because the maximum in the
Bellman operator is. The [infinite-horizon chapter](infinite-horizon-mdps.md)
solved the exact Bellman equation with two general methods for operator
equations, successive approximation, which gave value iteration, and the
Newton–Kantorovich method, which gave policy iteration. Each has a counterpart
for the $n$ equations.

(sec-wr-function-iteration)=
#### Function Iteration

For collocation and Galerkin, the conditions have the fixed-point form
$v_\theta = \Proj\,\Bellman v_\theta$ of {eq}`eq:projected-bellman`, and the
natural solver repeats the map,

$$
v_{k+1} = \Proj\, \Bellman v_k,
$$ (eq:projected-vi)

value iteration with a projection after each application of the operator. In
coefficients, one iteration of the collocation version has two steps. The
first evaluates the operator at the nodes with the current function,
$t_i = (\Bellman v_k)(s_i)$ for $i = 1, \ldots, n$. The second chooses the
coefficients that reproduce those $n$ targets, which is the linear system
$\boldsymbol{\Phi}\theta_{k+1} = t$ with the interpolation matrix
$\Phi_{ij} = \varphi_j(s_i)$. With hat functions collocated at their own
nodes the solve is trivial, because $\boldsymbol\Phi$ is the identity. The
Galerkin version replaces the second step by the projection of the target
function, the linear system $M\theta_{k+1} = b_k$ with the mass matrix
$M_{ij} = \langle \varphi_i, \varphi_j \rangle_w$ and
$b_i = \langle \Bellman v_k, \varphi_i \rangle_w$. The system is linear
whatever the operator is, because the target $\Bellman v_k$ is held fixed
while $\theta_{k+1}$ is chosen. It is the weighted least-squares fit of the
basis to the target, and with sampled states it is an ordinary regression of
the backups on the basis. The circularity of Step 3 is gone: the targets come
from the previous iterate. Fitting a parametric function to Bellman backups,
then recomputing the backups, is the algorithm that the
[next chapter](approximate-bellman-equations.md) calls fitted value iteration
and generalizes to any regressor.

Whether the iteration converges is a separate question. The Bellman operator
is a $\gamma$-contraction in the sup norm, so value iteration converges from
any start ({prf:ref}`value-iteration-convergence`). If $\Proj$ does not
increase sup-norm distances, then neither does the composition, and
{eq}`eq:projected-vi` converges to the unique solution of the projected
Bellman equation, with the error shrinking by $\gamma$ per iteration. Whether
$\Proj$ stretches sup-norm distances depends on the sign of its weights. An
interpolation operator with nonnegative cardinal functions returns a convex
combination of the targets, which cannot exceed them; this is the condition
$\Lambda = 1$ of {prf:ref}`prop-interp-error-accumulation`, and such a
$\Proj$ cannot stretch. Polynomial interpolation through Chebyshev nodes has
cardinal functions that dip below zero between the nodes, so it can amplify a
difference between two target vectors, and the iteration can then diverge
although every iteration reproduces its targets exactly. The
[next chapter](approximate-bellman-equations.md) proves that the first kind
of fit always preserves the contraction and shows what happens with the
second. Even when it converges, function iteration is slow when $\gamma$ is
near one: the error shrinks by $\gamma$ per iteration, so reducing it by
$10^{-10}$ with $\gamma = 0.95$ takes about 450 iterations.

::::{admonition} The fishery at Step 4
:class: note

{numref}`fig-fishery-sweep` repeats the collocation experiment for $n$ from 4
to 64, with function iteration as the solver.

:::{figure} _static/fishery_tipping_point/sweep.svg
:label: fig-fishery-sweep

Collocation by function iteration for $n$ from 4 to 64. (a) Largest error of
the approximate value. (b) Largest loss of its greedy policy, evaluated
exactly. Crosses mark the $n$ for which the iteration with the polynomial
diverged.
:::

With the hat functions the iteration converged for every $n$, in about 410
iterations. With the polynomial it diverged for $n = 4$ and for
$n = 9, 10, 11$: the coefficients grew without bound, although every
iteration reproduced its targets exactly. The backup still shrank the
distance between two value functions by $\gamma$; the interpolation step
stretched it, by enough to undo the contraction.

The sweep also shows how the two bases spend their accuracy. Piecewise-linear
interpolation improves steadily as nodes are added, because each node resolves
the value locally. The polynomial is erratic: adding one coefficient can
multiply its error by five, and the cliff at the tipping point is spread over
the whole interval until the degree is high enough to resolve it. From about
40 coefficients on, it overtakes the hat functions, and its error then falls
much faster with $n$. Which basis is better therefore depends on $n$ and on
the shape of the solution.
::::

(sec-wr-newton)=
#### Newton's Method

The alternative treats $G(\theta) = 0$ as a rootfinding problem and
linearizes it,

$$
\theta_{k+1} = \theta_k - J_G(\theta_k)^{-1} G(\theta_k),
$$

where $J_G$ is the Jacobian of $G$. For collocation,
$G_i(\theta) = v(s_i;\theta) - (\Bellman v_\theta)(s_i)$, and the Jacobian
entries are

$$
\frac{\partial G_i}{\partial \theta_j} = \varphi_j(s_i) - \frac{\partial (\Bellman v_\theta)(s_i)}{\partial \theta_j}.
$$

The first term is a basis function at a node. The second is the derivative of
the Bellman operator with respect to a coefficient, the quantity that the
least-squares conditions of Step 3 also needed, and it contains a maximum.
The maximum is not differentiable where two actions tie, but wherever the
maximizer is unique its derivative is given by the envelope theorem.

*Before reading on, differentiate $h(\theta) = \max_a g(a, \theta)$ with the
chain rule, writing $a^*(\theta)$ for the maximizer. Which term involves
$\partial a^*/\partial\theta$, and why does it vanish?*

```{prf:theorem} Envelope theorem
:label: thm-envelope

Let $g(a, \theta)$ be continuously differentiable in both arguments and let
$h(\theta) = \max_a g(a, \theta)$ be attained at a unique interior maximizer
$a^*(\theta)$ that depends differentiably on $\theta$. Then

$$
\nabla_\theta h(\theta) = \nabla_\theta g(a, \theta)\big|_{a = a^*(\theta)} .
$$

The derivative of the maximum is the partial derivative of the objective at
the maximizer, with the maximizer held fixed.
```

```{prf:proof}
By the chain rule,
$\nabla_\theta h = \nabla_\theta g(a^*(\theta), \theta) + \big(\nabla_a g(a^*(\theta), \theta)\big)^{\!\top} \partial a^*/\partial\theta$,
and the second term vanishes because $\nabla_a g = 0$ at an interior maximizer.
```

For the Bellman operator,
$(\Bellman v_\theta)(s) = \max_a \{ r(s,a) + \gamma\, \mathbb{E}[v(s';\theta) \mid s, a] \}$,
the objective is linear in $\theta$ with coefficients
$\gamma\,\mathbb{E}[\varphi_j(s') \mid s, a]$, so

$$
\frac{\partial (\Bellman v_\theta)(s)}{\partial \theta_j} = \gamma\, \mathbb{E}\big[\varphi_j(s') \mid s, a^*(s;\theta)\big],
$$ (eq:envelope-bellman)

where $a^*(s;\theta)$ is the greedy action at $s$ for the current
coefficients. The derivative is the expected value of the basis function at
the successor under the greedy action, and computing it costs one more
quadrature per node and basis function. The same formula holds for the
discrete-action case, and on a finite set of actions the maximizer is unique
except at ties.

Substituting {eq}`eq:envelope-bellman` into the Newton step gives it a
familiar form. Let $\pi_k$ be the greedy policy for $v_{\theta_k}$, let
$\mathbf{r}_{\pi_k}$ collect its rewards at the nodes, and let
$\mathbf{P}_{\pi_k}\boldsymbol{\Phi}$ be the matrix with entries
$\mathbb{E}[\varphi_j(s') \mid s_i, \pi_k(s_i)]$. Then
$J_G = \boldsymbol{\Phi} - \gamma \mathbf{P}_{\pi_k}\boldsymbol{\Phi}$ and
$G(\theta_k) = \boldsymbol\Phi\theta_k - \mathbf{r}_{\pi_k} - \gamma \mathbf{P}_{\pi_k}\boldsymbol\Phi\theta_k$,
and the Newton update $J_G \theta_{k+1} = J_G\theta_k - G(\theta_k)$
simplifies to

$$
(\boldsymbol{\Phi} - \gamma \mathbf{P}_{\pi_k}\boldsymbol{\Phi})\, \theta_{k+1} = \mathbf{r}_{\pi_k}.
$$ (eq:collocation-newton)

This is the policy evaluation equation of $\pi_k$, imposed at the collocation
nodes and solved within the span of the basis. Newton's method on the
collocation equations alternates a greedy improvement with an approximate
evaluation, exactly as Newton's method on the exact Bellman equation is
policy iteration ({ref}`sec-newton-kantorovich-pi`). With the smooth Bellman
operator of the [regularized chapter](regularized-dp.md), the maximum is
replaced by a log-sum-exp, the operator is differentiable everywhere, and the
same step holds with the softmax policy in place of the greedy one
({ref}`sec-smooth-pi-newton`).

Newton's method converges quadratically near the solution, a few iterations
instead of the hundreds of function iteration, and it does not need
$\Proj\,\Bellman$ to be a contraction, since it never iterates that map. It
pays for this in two ways. Each iteration forms and solves the Jacobian
system, whose entries need the expectations of every basis function at every
node. And it depends on the starting point: started far from the solution it
can diverge, or converge to a fixed point that the $n$ equations have and the
Bellman equation does not.

| **Method** | **Convergence** | **Per-iteration cost** | **Needs** |
|:-----------|:----------------|:-----------------------|:----------|
| Function iteration | Linear, factor $\gamma$ | $n$ backups and one linear solve with a fixed matrix | $\Proj\,\Bellman$ to be a contraction |
| Newton's method | Quadratic near the solution | $n$ backups, $n^2$ expectations and a fresh linear solve | A good starting point |

The two are complementary. When the fit is monotone and $\gamma$ is moderate,
function iteration is simple and robust. When $\gamma$ is near one or the
basis is a high-order polynomial, a hybrid works well: a few function
iterations with a coarse, monotone approximation to reach the basin of
attraction, then Newton's method with the accurate basis. This is how Rust
solved his bus-engine model ({ref}`sec-rust-inverse`), and the
[next chapter](approximate-bellman-equations.md) develops the Newton side as
approximate policy iteration.

### Step 5: Check the Solution

The conditions of Step 3 control the residual only where they were imposed,
at the collocation nodes or against the chosen test functions. The first
check is therefore to evaluate the residual where it was not controlled: at
many states between the nodes, with a finer quadrature inside the operator
than the one used in the fit, and against higher-order basis functions in the
Galerkin case.

A small residual everywhere does bound the value error. Since
$v_\theta - v^\star = (v_\theta - \Bellman v_\theta) + (\Bellman v_\theta - \Bellman v^\star)$
and $\Bellman$ is a $\gamma$-contraction,

$$
\|v_\theta - v^\star\|_\infty \le \frac{\|R(\cdot;\theta)\|_\infty}{1 - \gamma},
$$

where the norm of $R$ is taken over all states, not over the nodes. The
bound is loose in general, and neither the residual nor the value error is
what is used. The greedy policy is, and its loss is the difference between
the optimal value and the value of acting on the approximation. The three
quantities answer different questions. The residual is what the method
controls. The value error is what the residual bounds, through
$1/(1-\gamma)$. The policy loss is what matters, and it depends on the value
error only through the differences between actions, which an approximation
can get wrong while its values are accurate and right while they are not.
The [next chapter](approximate-bellman-equations.md) bounds the policy loss
by the value error and tests both bounds on a problem that a government
agency solves every year.

::::{admonition} The fishery at Step 5
:class: note

Panel (b) of {numref}`fig-fishery-collocation` is the residual check, and it
finds residuals of about $0.2$ between the nodes for both bases. The bound
on the value error is then $0.2 / 0.05 = 4$, against the full range of
$v^\star$, which is $4.5$. The actual value errors are $1.66$ for the
polynomial and $0.52$ for the hat functions, so the bound is loose, and it
ranks the two approximations in the wrong order: the polynomial has the
smaller residual.

:::{include} artifacts/fishery_tipping_point/results.md
:::

The table adds the policy loss. At $n = 12$ the polynomial has a residual of
$0.18$ against $0.21$ for the hat functions and a policy loss eighteen times
larger, because its oscillations move the decision boundary past the tipping
point (panel (c) of {numref}`fig-fishery-collocation`).
::::

## Exercises

The exercises use the notation of the five steps. The first ones are short
checks of the definitions; most of the others can be done by hand, and
{ref}`ex-wr-fishery-newton` needs the fishery module that produced the
figures.

:::{exercise} Orthogonality condition
:label: ex-projection-check-1

In a Galerkin method with basis functions $\varphi_i$, against which functions is the Bellman residual required to be orthogonal, and what does that say about $\Proj\,\Bellman v_\theta$?
:::

:::{solution} ex-projection-check-1
:class: dropdown

Against the basis functions themselves: $\langle \Bellman v_\theta - v_\theta, \varphi_i\rangle_w = 0$ for every $i$. The residual is then orthogonal to the whole span, so the $w$-orthogonal projection of $\Bellman v_\theta$ onto the span is $v_\theta$ itself.
:::

:::{exercise} Collocation versus least squares
:label: ex-projection-check-2

What is the main distinction between collocation and least-squares residual fitting, and which of the two needs the envelope theorem?
:::

:::{solution} ex-projection-check-2
:class: dropdown

Collocation forces the residual to vanish at selected states. Least squares minimizes the weighted squared residual over the state space. Least squares needs the envelope theorem, because its first-order conditions contain the derivative of the Bellman operator with respect to the coefficients; collocation needs it only if the collocation equations are solved by Newton's method.
:::

:::{exercise} Three measures of quality
:label: ex-projection-check-6

In the fishery with $n = 12$, the polynomial has the smaller residual and the larger policy loss. Which of residual, value error and policy loss does collocation control, which bounds which, and why can the ranking differ?
:::

:::{solution} ex-projection-check-6
:class: dropdown

Collocation controls the residual at the nodes only. The sup norm of the residual over all states bounds the value error through $1/(1-\gamma)$, and the value error bounds the policy loss (next chapter). Both bounds are loose, and the policy depends on differences between the values of actions rather than on the values themselves, so an approximation with a smaller residual can place a decision boundary in the wrong region, as the polynomial does across the tipping point.
:::

:::{exercise} Three methods on one equation
:label: ex-wr-three-methods

Consider policy evaluation on $\mathcal{S} = [0, 1]$ with reward $r(x) = x$,
deterministic transition $x' = x/2$ and $\gamma = 1/2$, so that the Bellman
equation is $v(x) = x + \tfrac12 v(x/2)$.

1. Check that $v(x) = \tfrac43 x$ solves it.
2. Approximate $v$ by $\hat v(x;\theta) = \theta_1 + \theta_2 x^2$, a basis that
   cannot represent the solution. Write the residual $R(x;\theta)$.
3. Collocate at $x = 0$ and $x = 1$. Find $\theta$.
4. Impose the Galerkin conditions with the uniform weight on $[0, 1]$. Find
   $\theta$.
5. Compare the two approximations by their largest residual and their
   largest value error on $[0, 1]$. Which one is better by each measure?
:::

:::{solution} ex-wr-three-methods
:class: dropdown

1. $\tfrac43 x = x + \tfrac12 \cdot \tfrac43 \cdot \tfrac x2 = x + \tfrac13 x$.
2. $R(x;\theta) = x + \tfrac12(\theta_1 + \theta_2 x^2/4) - \theta_1 - \theta_2 x^2 = x - \tfrac12\theta_1 - \tfrac78\theta_2 x^2$.
3. $R(0) = -\tfrac12\theta_1 = 0$ gives $\theta_1 = 0$, and $R(1) = 1 - \tfrac78\theta_2 = 0$ gives $\theta_2 = \tfrac87$.
4. The conditions $\int_0^1 R\,dx = \tfrac12 - \tfrac12\theta_1 - \tfrac{7}{24}\theta_2 = 0$ and $\int_0^1 R\,x^2\,dx = \tfrac14 - \tfrac16\theta_1 - \tfrac{7}{40}\theta_2 = 0$ give $\theta_1 = \tfrac38$ and $\theta_2 = \tfrac{15}{14}$.
5. Collocation: largest residual $0.25$, largest value error $0.389$ (near $x = 0.58$). Galerkin: largest residual $0.1875$, largest value error $0.375$ (at $x = 0$). Galerkin is slightly better by both measures here. Both are poor, because a parabola cannot follow a line through the origin and the ends: no choice of conditions repairs a basis that cannot represent the solution. Least squares happens to give the Galerkin answer in this problem, because the residual is linear in $\theta$ and its derivatives $\partial R/\partial\theta_i$ are $-\tfrac12$ and $-\tfrac78 x^2$, which span the same space as the basis.
:::

:::{exercise} Which collocation iterations contract
:label: ex-wr-cardinal

Take the basis of {ref}`ex-wr-three-methods` and the collocation points $0$
and $1$.

1. Write the cardinal functions $\ell_0, \ell_1$ of interpolation in
   $\text{span}\{1, x^2\}$ through these points, so that
   $(\Proj g)(x) = \ell_0(x) g(0) + \ell_1(x) g(1)$.
2. Show that they are nonnegative and sum to one on $[0, 1]$. What does this
   imply for function iteration $v_{k+1} = \Proj\Bellman v_k$ on this problem,
   and for any Bellman operator that is a sup-norm contraction?
3. Now interpolate with the quadratic through $0$, $\tfrac12$ and $1$. Compute
   the Lebesgue constant $\Lambda = \max_x \sum_j |\ell_j(x)|$ and explain why
   the argument of part 2 no longer applies. This is the constant of
   {prf:ref}`prop-interp-error-accumulation` in the finite-horizon chapter.
:::

:::{solution} ex-wr-cardinal
:class: dropdown

1. $\ell_0(x) = 1 - x^2$ and $\ell_1(x) = x^2$.
2. Both lie in $[0, 1]$ on $[0, 1]$ and sum to one, so $\Proj g$ is a convex combination of $g(0)$ and $g(1)$ at every $x$ and $\|\Proj f - \Proj g\|_\infty \le \|f - g\|_\infty$. The composition $\Proj\Bellman$ is then a $\gamma$-contraction for any $\gamma$-contraction $\Bellman$, and function iteration converges from any start. On this problem it converges to the collocation solution of {ref}`ex-wr-three-methods`.
3. The cardinal functions are $2(x - \tfrac12)(x-1)$, $-4x(x-1)$ and $2x(x-\tfrac12)$. The first and last are negative on parts of $[0, 1]$, so $\Lambda = 1.25 > 1$. A difference between two target vectors can be amplified by up to $1.25$, which beats a contraction factor $\gamma > 0.8$; nonexpansiveness, the step the argument needs, fails.
:::

:::{exercise} Newton's method at a tie
:label: ex-wr-envelope-tie

Let $h(\theta) = \max\{\theta, 1 - \theta\}$, the maximum over two actions
whose values depend on one coefficient.

1. Where is $h$ differentiable, and what does the envelope theorem give
   there?
2. At $\theta = \tfrac12$ the two actions tie. Which assumption of
   {prf:ref}`thm-envelope` fails? Any slope between $-1$ and $1$ is a valid
   linearization there; show that each choice corresponds to evaluating one of
   the two actions, or a randomization between them.
3. Explain why Newton's method on the collocation equations, which is policy
   iteration in coefficient space ({eq}`eq:collocation-newton`), still works
   when the greedy policy has ties at some nodes, and what the smooth Bellman
   operator of the regularized chapter changes.
:::

:::{solution} ex-wr-envelope-tie
:class: dropdown

1. Everywhere except $\theta = \tfrac12$. The envelope theorem gives the derivative of the active action: $1$ for $\theta > \tfrac12$ and $-1$ for $\theta < \tfrac12$.
2. The maximizer is not unique. The one-sided derivatives are $-1$ and $1$, and every slope $\lambda \cdot 1 + (1-\lambda)(-1)$ with $\lambda \in [0, 1]$ is a subgradient: it is the derivative of the value of the randomized choice that takes the first action with probability $\lambda$.
3. A tie means two actions have the same value at that node, so evaluating either one gives a valid Newton step: the step solves the evaluation equation of some greedy policy, and policy iteration converges whichever greedy action it picks. The smooth Bellman operator replaces the maximum by a log-sum-exp, which is differentiable everywhere; its Jacobian uses the softmax policy, which splits the probability between tied actions, and no tie-breaking rule is needed.
:::

:::{exercise} Newton's method on the fishery
:label: ex-wr-fishery-newton

Function iteration with the twelve hat functions needs about 410 iterations
on the fishery. Implement the Newton step {eq}`eq:collocation-newton` for the
same basis, using `Bellman`, `HatBasis` and `growth` from
`code/fishery_tipping_point.py`, and count the iterations.

1. For the current coefficients, find the greedy escapement at each node.
   `Bellman.greedy` returns it on the fine grid; read off the value at the
   grid point where each node falls.
2. Build the matrix $\mathbf{P}_{\pi}\boldsymbol{\Phi}$ with entries
   $\mathbb{E}_z[\varphi_j(z f(s_i))]$, using the quadrature nodes
   `bellman.z` and weights `bellman.w`. For hat functions, $\boldsymbol\Phi$ is
   the identity.
3. Solve $(\mathbf{I} - \gamma \mathbf{P}_{\pi}\boldsymbol{\Phi})\theta = \mathbf{r}_\pi$
   with $\mathbf{r}_\pi$ the harvests $x_i - s_i$ at the nodes, and repeat.
4. Check that the result agrees with the fixed point of function iteration.
   Why must it?
:::

:::{solution} ex-wr-fishery-newton
:class: dropdown

```python
import numpy as np
from fishery_tipping_point import Bellman, HatBasis, growth, X_MAX

bellman, basis = Bellman(), HatBasis(12)
nodes, gamma = basis.nodes, bellman.gamma

def hats(points):                                   # phi_j(points), one column per node
    return np.stack([np.interp(points, nodes, np.eye(basis.n)[j]) for j in range(basis.n)], axis=-1)

theta = np.zeros(basis.n)
for k in range(30):
    escapement = bellman.greedy(basis.function(theta))
    index = np.clip(np.searchsorted(bellman.x, nodes, side="right") - 1, 0, bellman.x.size - 1)
    s = escapement[index]                           # greedy escapement at each node
    successors = np.clip(np.outer(growth(s), bellman.z), 0, X_MAX)
    P_Phi = np.einsum("q,iqj->ij", bellman.w, hats(successors))
    new = np.linalg.solve(np.eye(basis.n) - gamma * P_Phi, nodes - s)
    step, theta = np.abs(new - theta).max(), new
    if step < 1e-10:
        break
print(k + 1, "Newton iterations")
```

The steps are $1.6$, $2.9$, $0.86$, $3 \times 10^{-7}$ and then zero: five
iterations, against about 410 for function iteration, and the coefficients
agree with the fixed point of function iteration to $10^{-14}$. They must
agree, because both methods solve the same collocation equations
$\theta = \Proj\Bellman v_\theta$, which have a unique solution here since
$\Proj\Bellman$ is a contraction; Newton's method only finds it faster.
:::

:::{exercise} Choosing nodes for the polynomial
:label: ex-wr-nodes

With 10 Chebyshev coefficients, function iteration on the fishery diverges.
Explore what could be changed while keeping a polynomial of degree 9: more
collocation points than coefficients with a least-squares fit, a different
placement of the nodes, or Newton's method started from the converged
piecewise-linear solution. For each option, report whether the iteration
converges and the policy loss of the result, and relate what you see to the
sign of the cardinal functions and to the hybrid strategy of
{ref}`sec-wr-solve`.
:::

:::{solution} ex-wr-nodes
:class: dropdown

This exercise is open-ended. A least-squares fit to more nodes than
coefficients is no longer an averager, so it carries no guarantee either,
although oversampling reduces the size of the weights. Newton's method does
not iterate $\Proj\Bellman$ and needs no contraction; started from the
piecewise-linear solution, which is close, it is the hybrid strategy of the
chapter. Whatever converges, judge it by the policy loss, as in Step 5: a
polynomial that converges can still place the decision boundary on the wrong
side of the tipping point.
:::

## Summary and Outlook

Replacing the value function by $v(\cdot;\theta)$ turns the Bellman equation
into a residual $R(\cdot;\theta) = \Bellman v_\theta - v_\theta$ that no
choice of $\theta$ makes zero everywhere. The method of weighted residuals
imposes $n$ conditions on it instead: collocation makes it vanish at chosen
states, Galerkin makes it orthogonal to the basis, and least squares
minimizes its norm. The first two are projected Bellman equations
$v_\theta = \Proj\,\Bellman v_\theta$, and their solvers are the two solvers
of the exact equation with a projection inserted: function iteration, which
is value iteration with a fit after each sweep, and Newton's method, which is
policy iteration with each evaluation done within the span of the basis. The
fishery showed that the fit decides both the accuracy of the solution and
whether function iteration converges, and that a smaller residual does not
mean a better policy.

Nothing in the five steps is specific to the Bellman equation. Any operator
equation $\Residual(f) = 0$ for an unknown function $f$ is treated the same
way, with a basis for $f$, a residual $\Residual(\hat f)$, test functions, and
one of the two solvers, and the collocation chapter did exactly this for
differential equations, with polynomial pieces for the trajectory and the
equation of motion enforced at the collocation points. What the Bellman
equation adds is the maximum, which makes the equations nonlinear and the
Jacobian a matter for the envelope theorem, and the contraction, which is
what makes function iteration an option at all.

Step 4 left two things open. Function iteration converged for one basis and
not the other, and the explanation, that the fit must not stretch sup-norm
distances, was asserted and not proved. Newton's method was shown to be
policy iteration in coefficient space for collocation, without saying how
accurate each evaluation must be for the policies to improve. The
[next chapter](approximate-bellman-equations.md) takes up both for an
arbitrary fitting rule. Function iteration becomes fitted value iteration, a
sequence of regression problems with Bellman targets that any regressor can
solve, and the pattern behind fitted Q-iteration, NFQ and NFQCA. The chapter
bounds its error, shows when the fit makes it diverge and which fits cannot,
and turns Newton's method into approximate policy iteration.
