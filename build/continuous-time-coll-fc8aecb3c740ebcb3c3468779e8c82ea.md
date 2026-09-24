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

# Continuous-Time Transcription and Collocation

The preceding chapter formulated trajectory optimization for systems that are
already discrete in time. Physical models are often given instead by ordinary
differential equations, so their states and controls are functions of time.
A finite-dimensional optimizer cannot choose an entire function directly.
Continuous-time trajectory optimization therefore begins by replacing those
functions and their differential equations with finitely many variables and
constraints on those variables.

How can that replacement preserve enough of the differential equation to make
the resulting trajectory meaningful between the stored time points?

This replacement is called **transcription**, and the resulting finite
optimization problem is a nonlinear program (NLP).

The direct-collocation formulation used here stores state and control
values at selected times and interpolates between them with low-degree
polynomials. The differential equation is enforced at selected points on each
time interval. This chapter develops that construction from one simple example
through Euler, trapezoidal, and Hermite--Simpson transcriptions, then applies it
to the motion of an overhead crane.

:::{admonition} Learning goals
:class: note

After studying this chapter, you should be able to:

1. Explain why continuous-time optimal control requires transcription before it can be sent to an NLP solver.
2. Represent one polynomial either by basis coefficients or by values at distinct nodes.
3. Distinguish exact polynomial interpolation from least-squares regression.
4. Construct differentiation and quadrature operators from Lagrange cardinal functions.
5. Assemble the complete Euler, trapezoidal, and Hermite--Simpson optimization problems.
6. Choose state and control representations separately and identify their decision variables.
7. Generate collocation coefficients from nodes and use them to assemble a sparse NLP.
:::

:::{admonition} Prerequisites
:class: tip

The chapter uses ordinary differential equations, definite integrals, and the
basic form of a nonlinear program with equality and inequality constraints.
[](discrete-time-optimal-control.md) introduces that optimization problem,
and [](numerical-trajectory-optimization.md) develops shooting and simultaneous
formulations. [](appendix_ivps.md)
reviews the sequential integrators used inside shooting.
:::

## A One-Interval Example

How can a constraint on four endpoint values ensure that the chosen control
produces the required displacement?

Consider a point that must move from $x(0)=0$ to $x(1)=1$. Its velocity is the
control, so

$$
\dot x(t)=u(t),
$$

and the objective penalizes squared control effort,

$$
\underset{x(\cdot),u(\cdot)}{\operatorname{minimize}}
\quad \int_0^1 u(t)^2\,dt.
$$

Both $x$ and $u$ are unknown functions. As a first finite approximation, retain
only their endpoint values $x_0,x_1,u_0,u_1$ and let the control vary linearly
between $u_0$ and $u_1$. Integrating $\dot x=u$ means that the change in state
equals the area under this control. The shaded region below is a unit-width
trapezoid. Its area is the average of its two endpoint heights, $u_0$ and
$u_1$, multiplied by the width.

```{code-cell} python
:tags: [remove-input]
:label: fig-linear-control-trapezoid
:caption: Drag the endpoint controls or play the accumulation from left to right. The shaded rectangle and triangle sum to the state change implied by $\dot x=u$. Setting the interval width to $h=1$ gives the state-change relation $x_1-x_0=(u_0+u_1)/2$.

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from collocation_widgets import render_linear_control_area

control_area = render_linear_control_area()
display(HTML(control_area))
```

:::{figure} _static/collocation/linear-control-area.svg
:label: fig-linear-control-trapezoid-fallback
:class: pdf-fallback
:alt: A linear control between two endpoint values forms a trapezoid whose rectangle and triangle areas sum to the state change.

A linear control connects $u_0$ and $u_1$ across an interval of width $h$.
The rectangle $hu_0$ and triangle $h(u_1-u_0)/2$ sum to
$h(u_0+u_1)/2$. Because $\dot x=u$, this area equals $x_1-x_0$.
:::

The optimizer must choose endpoint states and controls that agree with this
area calculation. Otherwise, it could choose $x_0=0$ and $x_1=1$ while setting
both controls to zero: the endpoint conditions would hold, but zero velocity
could not produce the displacement. To exclude such choices, the stored state
change $x_1-x_0$ must equal the change $(u_0+u_1)/2$ predicted by integrating
the control. Moving both quantities to the left gives the constraint

$$
x_1-x_0-\frac{1}{2}(u_0+u_1)=0.
$$

The left-hand side measures the mismatch between these two changes and is
called the **defect**. Requiring that mismatch to vanish gives a **defect
constraint**, which enforces the dynamics within the chosen approximation.
The equation is **algebraic** because it relates the four numbers
$x_0,x_1,u_0,u_1$ through arithmetic operations; there is no unknown function
to differentiate or integrate when evaluating it. An optimizer can therefore
check the constraint directly for each candidate set of endpoint values.

Approximating the running-cost integral by the average of its endpoint values
gives the finite nonlinear program

$$
\begin{aligned}
\underset{x_0,x_1,u_0,u_1}{\operatorname{minimize}}
\quad&\frac{1}{2}(u_0^2+u_1^2)\\
\text{subject to}\quad&x_0=0,\qquad x_1=1,\\
&x_1-x_0-\frac{1}{2}(u_0+u_1)=0.
\end{aligned}
$$

The defect integrates the linear control exactly. The objective uses another
approximation: $u(t)^2$ is generally quadratic, so averaging its endpoint
values need not give its exact integral. Both calculations become exact for
a constant control.

The boundary conditions reduce the defect to $u_0+u_1=2$. Substituting
$u_1=2-u_0$ into the objective gives

$$
\frac12\left(u_0^2+(2-u_0)^2\right)
=(u_0-1)^2+1.
$$

The squared term is minimized at $u_0=1$, which also gives $u_1=1$. The
resulting interpolation is the exact solution $u(t)=1$ and $x(t)=t$.

This example already contains the main ingredients of direct collocation. The
function values became optimization variables, integration became a weighted
sum, and the differential equation became an equality constraint. The
remaining sections construct these operations systematically for nonlinear
vector dynamics and higher-degree polynomials.

## From a Continuous Problem to a Finite NLP

Which quantities and constraints must a transcription retain from the
continuous-time control problem?

We retain the earlier notation $\mathbf x\in\mathbb R^n$ for the state,
$\mathbf u\in\mathbb R^m$ for the control, and $c$ for cost. Here $t$ is
continuous physical time, while $k$ will index mesh intervals. The running
cost $c(\mathbf x,\mathbf u,t)$ is a cost per unit time; integrating it over
an interval produces the counterpart of the discrete stage cost $c_k$.
The terminal cost is written $c_f$ because the final physical time is $t_f$.
Scalar examples omit boldface.

With inequalities $\mathbf g\leq\mathbf0$ and equalities
$\mathbf h=\mathbf0$, the continuous-time Bolza problem is

$$
\begin{aligned}
\underset{\mathbf{x}(\cdot),\mathbf{u}(\cdot),t_f}{\operatorname{minimize}}
\quad&
c_f(\mathbf{x}(t_f),t_f)
+\int_{t_0}^{t_f} c(\mathbf{x}(t),\mathbf{u}(t),t)\,dt\\
\text{subject to}\quad&
\dot{\mathbf{x}}(t)=\mathbf{f}(\mathbf{x}(t),\mathbf{u}(t),t),\\
&\mathbf{h}(\mathbf{x}(t_0),\mathbf{x}(t_f),t_f)=\mathbf0,\\
&\mathbf{g}(\mathbf{x}(t),\mathbf{u}(t),t)\leq\mathbf0.
\end{aligned}
$$

The equality $\mathbf h=\mathbf0$ imposes endpoint conditions, while
$\mathbf g\leq\mathbf0$ imposes constraints along the path, such as actuator
limits or a bound on the state. A fixed final time is supplied as data and
omitted from the decision variables. Setting $c=0$ gives the Mayer special
case, while setting $c_f=0$ gives the Lagrange special case. All three forms
use the same transcription.

Two transcription strategies differ in which values become decision variables
and in how they impose the differential equation:

| Strategy | Finite decision variables | Treatment of the ODE |
|---|---|---|
| Shooting | Control parameters and, in multiple shooting, selected boundary states | A time integrator advances the state sequentially inside each shooting interval |
| Direct collocation | State and control values at selected nodes | Algebraic defect equations enforce the ODE simultaneously |

Shooting and collocation are therefore distinct transcription strategies. A
collocation method may reproduce a familiar integration formula, but it exposes
the state values to the NLP rather than hiding every state update inside a
simulation. The term **direct** indicates that the continuous problem is
converted directly into an optimization problem, without first deriving
necessary conditions such as the Pontryagin equations.

To apply a local approximation repeatedly over the horizon, divide time into
intervals with endpoints

$$
t_0<t_1<\cdots<t_N=t_f,\qquad h_k=t_{k+1}-t_k.
$$

This sequence is a **mesh**, with $k=0,\ldots,N-1$ indexing its intervals.
We write $\mathbf x_k$ and
$\mathbf u_k$ for values at mesh time $t_k$, as in the earlier discrete-time
chapters; the subscript is an index, not a physical time. On each interval,
the normalized coordinate is
$\tau=(t-t_k)/h_k$, or equivalently

$$
t=t_k+h_k\tau,\qquad 0\leq\tau\leq1.
$$

Every physical interval is thereby mapped to the same reference interval
$[0,1]$. Differentiation and integration formulas can be constructed once on
this reference interval; the length $h_k$ then supplies the physical-time
scaling. If the final time is also optimized, the interval lengths become
variables or fixed fractions of the variable horizon; the reference-interval
operators remain unchanged.

### Polynomial pieces on a time mesh

A single polynomial can represent a trajectory over the entire horizon; the
opening example's solution $x(t)=t$ already does so. Why introduce several
pieces? Consider the same scalar dynamics $\dot x=u$, starting from $x(0)=0$,
but now moving right at unit speed until $t=1/2$ and then left at unit speed:

$$
u(t)=\begin{cases}
1,&0\leq t<\tfrac12,\\
-1,&\tfrac12<t\leq1,
\end{cases}
\qquad
x(t)=\begin{cases}
t,&0\leq t\leq\tfrac12,\\
1-t,&\tfrac12\leq t\leq1.
\end{cases}
$$

The state is continuous, but its slope changes abruptly at the switching
time. Two line segments represent it exactly. A single polynomial has a
continuous derivative, so it cannot reproduce this corner exactly, although
it can approximate the trajectory. The ODE here holds on either side of the
switch; the control's value at that one instant does not affect its integral.

The comparison below uses the quadratic $q(t)=2t(1-t)$, which passes through
the same states at $t=0,1/2,1$. Matching these three values does not reproduce
the motion between them: the quadratic has a continuously varying slope,
while the prescribed velocity switches from $+1$ to $-1$.

:::{figure} _static/collocation/switching-trajectory.svg
:label: fig-switching-trajectory
:alt: The triangular state trajectory consists of two straight segments meeting at time one half. A dashed quadratic passes through the same three points but curves above the segments. Below, the exact slope jumps from plus one to minus one, whereas the quadratic's slope decreases continuously from plus two to minus two.

Two linear pieces represent the prescribed motion exactly. The dashed orange
quadratic $q(t)=2t(1-t)$ matches the three black sample points but differs
between them. The lower panel compares their physical-time derivatives.
Open circles indicate the two one-sided slopes at the switch, where the
exact state has no derivative. Higher-degree global polynomials can improve
the approximation, but every single polynomial has a continuous derivative.
:::

Even when the trajectory is smooth, some portions may change much faster than
others. Separate polynomial pieces let us shorten the mesh intervals near a
rapid change while retaining longer intervals elsewhere. Increasing the
degree of one global polynomial adds flexibility across the whole horizon.
The local representation also gives the optimizer useful structure: each
interval's dynamics constraints involve only values on that interval and
neighboring endpoint states. A global polynomial couples values across the
horizon through its differentiation matrix.

For a trajectory that is smooth throughout the horizon, a single polynomial
of sufficiently high degree can be an efficient choice. Piecewise polynomials
give us local control over resolution, allow changes in slope at joins, and
keep the constraint derivatives sparse. They require us to connect adjacent
pieces explicitly so that the state remains continuous.

We therefore represent the trajectory by a separate polynomial on each mesh
interval $[t_k,t_{k+1}]$. The full approximation $\mathbf{x}_h$ is **piecewise
polynomial**. To construct the piece on interval $k$, use its local coordinate
$\tau=(t-t_k)/h_k$, where $h_k=t_{k+1}-t_k$, and write

$$
\mathbf{p}_k(\tau):=\mathbf{x}_h(t_k+h_k\tau),\qquad 0\le\tau\le1.
$$

Thus $\mathbf{p}_k(0)$ is the state at the start of this interval and $\mathbf{p}_k(1)$ is the
state at its end. On the next interval, the local coordinate starts again at
zero, and a different polynomial $\mathbf{p}_{k+1}$ describes the trajectory. The
reference interval and its integration weights can be reused even when the
physical intervals have different lengths.

:::{figure} _static/collocation/local-time-pieces.svg
:label: fig-local-time-pieces
:alt: A continuous scalar trajectory has three polynomial pieces on the physical intervals zero to one, one to three, and three to four. Beneath it, the same pieces each occupy a local time axis from zero to one. Adjacent pieces meet at the shared endpoint states x1 and x2.

A scalar example of the piecewise representation. The upper panel joins
three quadratic pieces on physical intervals of lengths $1$, $2$, and $1$.
The lower panels show those same pieces in their own coordinates
$0\leq\tau\leq1$, preserving their colors and line styles. The middle piece
uses $t=1+2\tau$, so two units of physical time occupy one unit of local
time. Black dots mark endpoint states; matching the right value of one piece
to the left value of the next keeps the trajectory continuous. The slopes
need not match across joins.
:::

## One Polynomial, Two Coordinate Systems

Should a polynomial trajectory be stored by its coefficients or by its values
at selected points, and how are the two descriptions related?

The opening example stored a linear control by its two endpoint values.
Each piece of a collocation trajectory needs a similar reconstruction:
given the values the optimizer stores, what curve do they define between
nodes? Start with a scalar polynomial $p$ on the reference interval
$[0,1]$. The construction applies to each state and control component.

Suppose $p(0)=y_0$ and $p(1)=y_1$, and restrict $p$ to degree at most one.
A line has the form $p(\tau)=a_0+a_1\tau$. The first endpoint condition gives
$a_0=y_0$, and the second gives $a_0+a_1=y_1$, so $a_1=y_1-y_0$.
Substitution and regrouping give

$$
p(\tau)=y_0(1-\tau)+y_1\tau.
$$

The weights $1-\tau$ and $\tau$ enforce the two endpoint conditions: at $0$,
the formula returns $y_0$, and at $1$, it returns $y_1$. Between the endpoints,
the weights vary continuously; at $\tau=1/4$, for example, the value is
$3y_0/4+y_1/4$. This is the unique line through the two prescribed values.
Without the degree restriction, other curves could pass through them. Even
for this same line, $y_0+(y_1-y_0)\tau$ is an equivalent representation. The
weighted form is convenient because each stored value has its own function
that determines its contribution throughout the interval. The functions
$1-\tau$ and $\tau$ form a basis for linear polynomials, just as $1$ and
$\tau$ do. In the first basis, the coefficients are the endpoint values
$y_0,y_1$; in the monomial basis, they are $a_0=y_0$ and $a_1=y_1-y_0$.

Higher-degree pieces use the same idea with more stored values. To specify
which curves those values may define, restrict the degree to at most $r$.
The resulting polynomial space is

$$
\mathcal P_r=\{p:\deg p\leq r\}.
$$

This space, denoted by $\mathcal P_r$, has dimension $r+1$. Choosing a basis
$\{\phi_0,\ldots,\phi_r\}$ gives coefficient coordinates

$$
p(\tau)=\sum_{j=0}^{r}a_j\phi_j(\tau).
$$

The monomial choice $\phi_j(\tau)=\tau^j$ gives the familiar expression
$a_0+a_1\tau+\cdots+a_r\tau^r$. A different basis changes the coefficients
used to describe the same function.

The same polynomial can instead be identified by its values. Choose $r+1$
distinct points

$$
\sigma_0,\ldots,\sigma_r\in[0,1],
$$

and record

$$
y_i=p(\sigma_i).
$$

These points are called **support nodes**, or **interpolation nodes**: once
the degree is restricted to at most $r$, the $r+1$ stored values determine
the whole polynomial. Storing values is useful for trajectory optimization
because a bound on the state at a node
then becomes a bound on a decision variable. To evaluate the dynamics between
nodes, however, we need a formula that reconstructs the polynomial from those
values.

With more nodes, the same construction needs one weight function $\ell_j$
per stored value $y_j$. To recover $y_j$ at its own node without altering the
values at the other nodes, $\ell_j$ must equal one at $\sigma_j$ and zero at
every other support node. A polynomial with those zeros contains the factors
$(\tau-\sigma_m)$ for all $m\ne j$. Dividing their product by its value at
$\sigma_j$ makes the value there equal to one. This constructs the
**Lagrange basis function**, also called a **cardinal function**,

$$
\ell_j(\tau)
=\prod_{\substack{m=0\\m\neq j}}^r
\frac{\tau-\sigma_m}{\sigma_j-\sigma_m}.
$$

At a support node $\sigma_i$, one factor in the numerator is zero unless
$i=j$. When $i=j$, every numerator equals its corresponding denominator.
Consequently,

$$
\ell_j(\sigma_i)=\delta_{ij},
$$

where the Kronecker delta $\delta_{ij}$ is shorthand for one when $i=j$ and
zero otherwise. This identity specifies the values at the nodes only. At any
other $\tau$, the product formula gives a smoothly varying polynomial weight,
which can be negative or exceed one for higher degrees. Thus, unlike a line
between two endpoint values, a higher-degree interpolant can take values
outside the range of its stored values. The cardinal functions reconstruct
the polynomial throughout the interval by

$$
\boxed{
p(\tau)=\sum_{j=0}^{r}y_j\ell_j(\tau),
\qquad y_j=p(\sigma_j).
}
$$

At a node $\sigma_i$, all terms except $y_i\ell_i(\sigma_i)=y_i$ vanish.
Between nodes, the weighted sum supplies the intervening values. This expression
is the **Lagrange form** of $p$. When the values $y_j$ are prescribed and $p$
is constructed to pass through them, $p$ is called the **Lagrange interpolating
polynomial**. Each $\ell_j$ is one basis function; $p$ is their weighted sum.
Changing one
stored value $y_j$ by $\Delta y_j$ changes the curve by
$\Delta y_j\ell_j(\tau)$, so the same function describes how that variable
affects the entire polynomial.

The uniqueness of this reconstruction follows from a basic root-counting
argument. If two polynomials in $\mathcal P_r$ have the same $r+1$ nodal values,
their difference has $r+1$ distinct roots. A nonzero polynomial of degree at
most $r$ cannot have that many roots, so the two polynomials must be identical.

Consequently, every polynomial in $\mathcal P_r$ has a unique expansion in
$\ell_0,\ldots,\ell_r$. These functions form the **Lagrange basis** associated
with the chosen support nodes. Its coefficients are exactly the nodal values
$y_j=p(\sigma_j)$. Storing a polynomial by its values is therefore also a
basis representation, with the basis chosen so that its coefficients have a
direct interpretation as function values.

For a scalar polynomial, collect the coefficients and values into column vectors
$\mathbf a=(a_0,\ldots,a_r)^\top$ and $\mathbf y=(y_0,\ldots,y_r)^\top$.
The coefficient and nodal descriptions are related by the evaluation matrix

$$
V_{ij}=\phi_j(\sigma_i),
\qquad \mathbf{y}=V\mathbf{a}.
$$

Distinct support nodes make $V$ invertible by the same root-counting argument.
Thus $\mathbf{a}$ and $\mathbf{y}$ are two coordinate vectors for one polynomial, rather than
two different approximations.

For example, take $p(\tau)=1+2\tau-\tau^2$ and the nodes
$0,\tfrac12,1$. In monomial coordinates,

$$
\mathbf{a}=
\begin{bmatrix}1\\2\\-1\end{bmatrix},
\qquad
V=
\begin{bmatrix}
1&0&0\\
1&\tfrac12&\tfrac14\\
1&1&1
\end{bmatrix},
$$

whereas the nodal coordinates are

$$
\mathbf{y}=V\mathbf{a}=
\begin{bmatrix}1\\\tfrac74\\2\end{bmatrix}.
$$

In the Lagrange basis for these three nodes, the same quadratic is

$$
p(\tau)=\ell_0(\tau)+\tfrac74\ell_1(\tau)+2\ell_2(\tau).
$$

Here $\ell_0,\ell_1,\ell_2$ select the nodes $0,\tfrac12,1$, respectively.
Both coefficient vectors describe the same quadratic. The transcription used here chooses
coordinates like $\mathbf y$ for the state and control: values that can be
inserted directly into the dynamics, cost, and bounds. Monomial coefficients
remain useful for deriving formulas that act on those values.

(polynomial-space-basis-and-nodes-are-different-choices)=
### Polynomial spaces, bases, and nodes

The construction separates three decisions that are easy to conflate:

- The **polynomial space** $\mathcal P_r$ specifies which functions are available.
- The **basis** specifies coordinates for a member of that space. Monomial and Lagrange bases span the same $\mathcal P_r$.
- The **nodes** specify where values, residuals, or integrals are evaluated.

Changing the basis does not change the exact polynomial space, although it can
change numerical conditioning. Changing the nodes changes the interpolation
and the operators built from it. The node families introduced later can still
use Lagrange nodal coordinates in the NLP.

## Polynomial Interpolation and Least-Squares Regression

When should a polynomial pass through every supplied value, and when should
it approximate those values with a smaller number of coefficients?

Interpolation and polynomial regression impose different requirements on the
same data. Six values at six distinct nodes determine one polynomial of degree
at most five that passes through every point. If the values are noisy
observations and the aim is to estimate a quadratic trend, a quadratic will
generally be unable to pass through all six. Least-squares regression then
chooses its three coefficients to minimize the sum of squared discrepancies.
The figure below applies both choices to the same six points. Write $V$ for
the basis evaluation matrix, as in $\mathbf y=V\mathbf a$. It has six rows
in both cases, but six columns for the degree-five interpolant and three for
the quadratic fit.

| | Polynomial interpolation | Least-squares regression |
|---|---|---|
| Input | Exact value conditions | Usually noisy or overdetermined observations |
| Algebraic problem | Satisfy $V\mathbf{a}=\mathbf{y}$ exactly | Minimize $\lVert V\mathbf{a}-\mathbf{y}\rVert_2^2$ |
| Residual | Zero when the value conditions uniquely determine a polynomial | Generally nonzero |
| Typical purpose | Represent a function from exact nodal data | Estimate a trend or conditional mean |

If $V$ is square and invertible, minimizing the squared residual also returns
the exact interpolant. The two procedures coincide in that case because
the chosen polynomial space can satisfy every value condition.

```{code-cell} python
:tags: [remove-input]
:label: fig-interpolation-versus-regression
:caption: Two polynomials fitted to the same six points (black dots). The solid blue degree-five interpolant passes through every point. The dashed orange quadratic regression minimizes the sum of squared residuals, shown as gray vertical segments. The polynomial degrees differ because a quadratic generally cannot satisfy all six value conditions exactly.

import numpy as np
import matplotlib.pyplot as plt

observation_nodes = np.linspace(0.0, 1.0, 6)
observation_values = (
    0.25
    + 0.95 * observation_nodes
    - 0.25 * observation_nodes**2
    + np.array([0.00, 0.10, -0.07, 0.08, -0.09, 0.02])
)
plot_nodes = np.linspace(0.0, 1.0, 401)

def evaluate_lagrange(nodes, values, points):
    result = np.zeros_like(points)
    for j, node in enumerate(nodes):
        cardinal = np.ones_like(points)
        for m, other_node in enumerate(nodes):
            if m != j:
                cardinal *= (points - other_node) / (node - other_node)
        result += values[j] * cardinal
    return result

interpolated_values = evaluate_lagrange(
    observation_nodes, observation_values, plot_nodes
)
regression_matrix = np.vander(observation_nodes, 3, increasing=True)
regression_coefficients, *_ = np.linalg.lstsq(
    regression_matrix, observation_values, rcond=None
)
regression_curve = np.vander(plot_nodes, 3, increasing=True) @ regression_coefficients
regression_at_nodes = regression_matrix @ regression_coefficients

with plt.rc_context({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}):
    figure, axis = plt.subplots(figsize=(5.5, 2.8), constrained_layout=True)
    blue = "#0072B2"
    orange = "#D55E00"
    gray = "#4D4D4D"

    axis.plot(plot_nodes, interpolated_values, color=blue, linewidth=1.8,
              label="Degree-five interpolation")
    axis.plot(plot_nodes, regression_curve, color=orange, linewidth=1.8,
              linestyle="--", label="Quadratic least squares")
    axis.vlines(observation_nodes, regression_at_nodes, observation_values,
                color="#777777", linewidth=1.2, zorder=2)
    axis.scatter(observation_nodes, observation_values, color=gray,
                 edgecolor="white", linewidth=0.6, s=30, zorder=3,
                 label="Same six points")
    axis.set_xlabel(r"$\tau$")
    axis.set_ylabel("value")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.18, linewidth=0.6)
    axis.set_xlim(-0.025, 1.025)
    axis.set_ylim(0.15, 1.25)
    axis.legend(loc="upper left", frameon=False, fontsize=8)

display(figure)
plt.close(figure)
```

In a nodal state representation of direct collocation, the stored states are
unknown decision variables rather than observations. Each candidate vector
of support values defines one interpolating polynomial exactly. The
optimizer selects a candidate whose ODE
residual vanishes at the collocation nodes. A numerical solver may temporarily
work with a scalar measure of constraint violation, but the collocation
conditions remain equality constraints rather than a statistical regression
loss.

## Fixed Operators from Nodal Values

How can the stored values give us both the slopes and the integral of every
candidate polynomial?

The nodal representation also turns differentiation and integration into
matrix and vector products. Once the nodes are chosen, the required arrays
can be computed once and reused for every candidate trajectory. The optimizer
can then evaluate slopes and integrals as it changes the nodal values, without
recomputing those arrays. Each cardinal function is a known polynomial, so
its derivative and antiderivative can be computed in closed form by changing
its coefficients. Constructing the arrays therefore does not require
finite-difference approximations or numerical quadrature of the basis
functions. The resulting operators are exact for the represented polynomial,
apart from floating-point error in their construction and application.

This is possible because the cardinal functions stay fixed while only their
coefficients $y_j$ change. In the representation

$$
p(\tau)=\sum_{j=0}^r y_j\ell_j(\tau),
$$

each $y_j$ is constant with respect to $\tau$, so differentiation acts only
on $\ell_j$. To obtain the slopes needed by the ODE, differentiate and then
evaluate at a node $\sigma_i$:

$$
p'(\sigma_i)=\sum_{j=0}^r
\underbrace{\ell_j'(\sigma_i)}_{D_{ij}}y_j,
$$

while integration over the reference interval gives

$$
\int_0^1p(\tau)\,d\tau
=\sum_{j=0}^r
\underbrace{\left(\int_0^1\ell_j(\tau)\,d\tau\right)}_{w_j}y_j.
$$

The resulting differentiation matrix $D$ and integration weights $w$ depend
only on the chosen nodes. Their entries record the derivatives and areas of
the fixed cardinal functions. Multiplication by the current nodal values
combines these contributions into the derivative or integral of the current
polynomial. This is how the transcription supplies the calculus needed by the
dynamics and cost through arithmetic on the decision variables.

For a concrete construction, suppose a quadratic is stored by its values
$y_0,y_m,y_1$ at $0,\tfrac12,1$, where $m$ denotes the midpoint. The desired
outputs are its slope at each of these nodes and its integral over $[0,1]$.
To build the arrays that return those outputs for any choice of the three
values, first construct the three cardinal functions multiplying them:

$$
\begin{aligned}
\ell_0(\tau)
&=\frac{(\tau-\frac12)(\tau-1)}{(0-\frac12)(0-1)}
=2\tau^2-3\tau+1,\\
\ell_m(\tau)
&=\frac{\tau(\tau-1)}{(\frac12-0)(\frac12-1)}
=4\tau(1-\tau),\\
\ell_1(\tau)
&=\frac{\tau(\tau-\frac12)}{(1-0)(1-\frac12)}
=2\tau^2-\tau.
\end{aligned}
$$

These are the three contributions in
$p(\tau)=y_0\ell_0(\tau)+y_m\ell_m(\tau)+y_1\ell_1(\tau)$.
Their derivatives are $4\tau-3$, $4-8\tau$, and $4\tau-1$. At the left
endpoint they give the weights $-3,4,-1$, so
$p'(0)=-3y_0+4y_m-y_1$. Evaluation at the midpoint and right endpoint supplies
the other two rows of the differentiation matrix:

$$
\begin{bmatrix}
p'(0)\\
p'(\tfrac12)\\
p'(1)
\end{bmatrix}
=
\underbrace{
\begin{bmatrix}
-3&4&-1\\
-1&0&1\\
1&-4&3
\end{bmatrix}}_{D}
\begin{bmatrix}
y_0\\y_m\\y_1
\end{bmatrix}.
$$

Thus a matrix-vector product turns the three stored values into the three
nodal derivatives. Integrating the same cardinal functions gives

$$
\int_0^1 p(\tau)\,d\tau
=
\underbrace{
\begin{bmatrix}
\tfrac16&\tfrac46&\tfrac16
\end{bmatrix}}_{w^\mathsf T}
\begin{bmatrix}
y_0\\y_m\\y_1
\end{bmatrix}.
$$

The row $w^\mathsf T$ turns the same three values into the exact integral of
their quadratic interpolant. The matrix $D$ differentiates every polynomial in
$\mathcal P_2$ exactly, and $w$ integrates every polynomial in that space
exactly. "Fixed operator" means that these arrays remain constant while the
NLP changes the nodal values.

For the earlier polynomial $p(\tau)=1+2\tau-\tau^2$, the stored values are
$\mathbf y=(1,\tfrac74,2)^\top$. Applying the arrays gives

$$
D\mathbf y=\begin{bmatrix}2\\1\\0\end{bmatrix},
\qquad
w^\top\mathbf y
=\frac16+\frac46\frac74+\frac16\,2=\frac53.
$$

The first result agrees with $p'(\tau)=2-2\tau$ at the three nodes; the
second agrees with the integral of $1+2\tau-\tau^2$ over $[0,1]$. The figure
collects the coordinate change and these two operations for that same
polynomial.

:::{figure} _static/collocation/polynomial-coordinate-operators.svg
:label: fig-polynomial-coordinate-operators
:alt: The quadratic p of tau equals one plus two tau minus tau squared is represented by its monomial coefficient vector and by its values at support nodes zero, one half, and one. Fixed differentiation and quadrature operators act on the nodal vector.

One quadratic, two coordinate systems. Evaluating the monomial coefficients
$\mathbf{a}$ at the support nodes gives $\mathbf{y}=V\mathbf{a}$. Direct
collocation stores the nodal vector $\mathbf{y}$; the fixed operators $D$ and
$w$ then return its nodal derivatives and exact integral without adding
optimization variables.
:::

A transcription may use nodes for three distinct purposes:

| Node role | What it does |
|---|---|
| Support node | Supplies a value used to define a particular polynomial by interpolation |
| Collocation node | Specifies where the reconstructed state must satisfy the ODE |
| Quadrature node | Supplies a point used to approximate an integral |

A method often reuses one set of points for two or three roles. Support
nodes belong to a particular polynomial: the state, control, and slope
polynomials need not use the same interpolation nodes. For example, a state
polynomial can be supported at one set of nodes and differentiated at
different collocation nodes. In the integration construction below, the
collocation nodes also serve as support nodes for the slope polynomial.
Both integration and differentiation forms have collocation nodes. Their
algebraic residuals differ: one matches state values after integrating
slopes, while the other matches slopes after interpolating state values.

## From Nodal Slopes to Collocation Constraints

How do those fixed maps convert differential equations into algebraic
constraints at the chosen collocation nodes?

The preceding operators apply to any polynomial. A trajectory must also
satisfy the dynamics. On each interval, the construction will interpolate
the slopes supplied by the ODE and integrate them from the stored left
state. Constraints will then require the stored states to agree with the
resulting polynomial.

Interval $k$ extends from physical time $t_k$ to $t_{k+1}$ and contains one
polynomial piece. Its local time coordinate runs from $\tau=0$ at the left
endpoint to $\tau=1$ at the right endpoint. Choose $s$ distinct collocation
nodes $\tau_1,\ldots,\tau_s$ on $[0,1]$. A node $\tau_j$ specifies one time
within this interval, $t_k+h_k\tau_j$; for example, $\tau_j=1/2$ specifies
its midpoint. The state and control values at that time are
$\mathbf{x}_{k,j}$ and $\mathbf{u}_{k,j}$.

This evaluation point, together with its state and control values, is called
a **stage**. A stage is associated with a single time, so it has no separate
beginning or end. The index $k$ identifies the interval and $j$ identifies
a stage in that interval. Depending on the method, stages may be interior
points or may coincide with an endpoint: $\tau_j=0$ places a stage at the
start and $\tau_j=1$ places one at the end. At each stage, the differential
equation prescribes the physical-time slope

$$
\mathbf{f}_{k,j}
=\mathbf{f}(\mathbf{x}_{k,j},\mathbf{u}_{k,j},t_k+h_k\tau_j)
$$

for $j=1,\ldots,s$. The figure below shows one such evaluation for the scalar
dynamics $\dot x=u$. The state and control are read at the same physical time;
the ODE then gives the slope that the state polynomial must match there.

:::{figure} _static/collocation/collocation-stage.svg
:label: fig-collocation-stage
:alt: State and control curves on the physical interval from time two to four. An orange vertical line at time three marks the collocation node tau j equals one half. The stage state is two and the stage control is three halves. A dashed tangent at the state point has physical-time slope three halves, as required by x dot equals u.

A stage combines a time, a state value, and a control value. Here $t_k=2$,
$h_k=2$, and $\tau_j=1/2$, so the stage occurs at physical time $3$.
The orange dots mark $x_{k,j}=2$ and $u_{k,j}=3/2$. For $\dot x=u$, the
prescribed slope is $f_{k,j}=3/2$, drawn as a dashed tangent in the upper
panel. The curves are $u_h(t)=1/2+(t-2)$ and
$x_h(t)=1+(t-2)/2+(t-2)^2/2$. Both panels use physical time; in local
coordinates the derivative is $p_k'(\tau_j)=h_kf_{k,j}=3$.
:::

As before,
$\mathbf p_k(\tau)=\mathbf x_h(t_k+h_k\tau)$ denotes the state piece in
normalized time. Here $\ell_j$ is the Lagrange cardinal function for
the slope nodes $\tau_1,\ldots,\tau_s$; the $s$ slope values determine a
polynomial of degree at most $s-1$. Interpolating them gives the
physical-time derivative on this one interval:

$$
\dot{\mathbf{x}}_h(t_k+h_k\tau)
=\sum_{j=1}^{s}\mathbf{f}_{k,j}\ell_j(\tau).
$$

This polynomial agrees with each sampled slope $\mathbf{f}_{k,j}$ at its
collocation node. Each sample was evaluated at a stored stage state
$\mathbf x_{k,j}$; the state curve recovered by integration need not yet
pass through those stored states. The stage equations below will enforce
that agreement. To recover state values from these slopes,
we need to reverse differentiation. For example, both $\tau+2\tau^2$ and
$1+\tau+2\tau^2$ have derivative $1+4\tau$. Each is an **antiderivative** of
$1+4\tau$: differentiating it returns that function. Adding any constant
gives another antiderivative, because the derivative of a constant is zero.
The slopes therefore determine changes in state, but a starting value is
needed to determine the state itself. This starting value fixes the
**integration constant**.

The **fundamental theorem of calculus** connects antiderivatives to definite
integrals. For a scalar function $p$ with a continuous derivative, it states
that integrating the derivative between two points gives the difference
between the function values:

$$
\int_0^\tau p'(\eta)\,d\eta=p(\tau)-p(0),
\qquad
p(\tau)=p(0)+\int_0^\tau p'(\eta)\,d\eta.
$$

Here $\eta$ is a dummy variable that runs over the integration interval;
$\tau$ specifies where that interval ends. Over a short interval of width
$\Delta\eta$, the change in $p$ is approximately $p'(\eta)\Delta\eta$.
The definite integral is the limit of sums of these signed changes as the
interval widths approach zero. Adding $p(0)$ to this accumulated
change recovers the value at $\tau$. Polynomial derivatives are continuous,
so the theorem applies to the pieces used here. For a vector state, the
same identity applies to each component.

The sampled slopes $\mathbf f_{k,j}$ are derivatives with respect to
physical time, whereas $\mathbf p_k$ is expressed in local time.
Since $t=t_k+h_k\tau$, the chain rule gives

$$
\mathbf p_k'(\tau)
=h_k\dot{\mathbf x}_h(t_k+h_k\tau)
=h_k\sum_{j=1}^{s}\mathbf f_{k,j}\ell_j(\tau).
$$

The factor $h_k$ accounts for the duration of the physical interval:
a change of $\Delta\tau$ in local time corresponds to $h_k\Delta\tau$
units of physical time. For example, a constant physical slope
$\mathbf f$ acting from $0$ to $\tau_i$ changes the state by
$h_k\tau_i\mathbf f$.

Applying the fundamental theorem with the left-endpoint condition
$\mathbf p_k(0)=\mathbf x_k$ now gives the state piece:

$$
\begin{aligned}
\mathbf{p}_k(\tau)
&=\mathbf{x}_k+\int_0^\tau\mathbf p_k'(\eta)\,d\eta\\
&=\mathbf{x}_k+h_k\sum_{j=1}^{s}
\left(\int_0^\tau\ell_j(\eta)\,d\eta\right)\mathbf{f}_{k,j}.
\end{aligned}
$$

The final expression follows by substituting the slope polynomial and
integrating its terms separately; the stage slopes $\mathbf f_{k,j}$ are
constant with respect to $\eta$. At $\tau=0$, every integral vanishes,
leaving $\mathbf p_k(0)=\mathbf x_k$. Differentiating the expression returns
the prescribed slope polynomial. Thus the stored left state selects the
unique antiderivative with that starting value. Integrating a polynomial
of degree at most $s-1$ gives a state piece of degree at most $s$.

For the scalar example above, $h_k=2$ and the physical-time slope is
$1/2+2\tau$, so $p_k'(\tau)=1+4\tau$. Starting from $x_k=1$ gives

$$
p_k(\tau)=1+\int_0^\tau(1+4\eta)\,d\eta
=1+\left[\eta+2\eta^2\right]_0^\tau
=1+\tau+2\tau^2.
$$

The brackets mean evaluating the antiderivative at the upper limit and
subtracting its value at the lower limit. At $\tau_i=1/2$, the accumulated
change is $1$, so $p_k(1/2)=2$. At $\tau=1$, the accumulated change is $3$,
so $p_k(1)=4$. These are two evaluations of the same polynomial: both
integrals begin at the left endpoint, but one stops at a stage and the other
at the right endpoint.

This integration is exact for the interpolated slope polynomial. It does
not generally give the exact solution of the original ODE: between
collocation nodes, the interpolated slope can differ from
$\mathbf f(\mathbf p_k(\tau),\mathbf u_h(t_k+h_k\tau),t_k+h_k\tau)$.

:::{figure} _static/collocation/stage-and-endpoint.svg
:label: fig-stage-and-endpoint
:alt: One quadratic state piece spans local time zero to one, corresponding to physical time two to four. A circular stage point at local time one half has state two; square endpoints have states one and four. A blue arrow from zero to one half shows the integration span for a stage equation. An orange arrow from zero to one shows the integration span for the endpoint equation. Both arrows start at the same left endpoint.

Two integration limits on one polynomial piece. Here $t_k=2$, $h_k=2$,
and $p_k(\tau)=1+\tau+2\tau^2$. The circle marks stage $i$ at
$\tau_i=1/2$, corresponding to physical time $3$; the squares mark the
interval endpoints. The blue span accumulates a state change of $1$ from
$\tau=0$ to $\tau_i$, giving $x_{k,i}=2$. The orange span accumulates a
state change of $3$ from $\tau=0$ to $1$, giving $x_{k+1}=4$. The arrows
mark integration spans, not successive steps of a numerical solver.
The changes come from integrating the derivative of the state curve shown.
Only stage $i$ is highlighted; each sum uses all the slope stages on the
interval.
:::

The integration construction already determines the derivative at each
collocation node. Because $\ell_j(\tau_i)=\delta_{ij}$, any candidate
stage values give

$$
\begin{gathered}
\frac{1}{h_k}\mathbf p_k'(\tau_i)
=\mathbf f_{k,i}
=\mathbf f(\mathbf x_{k,i},\mathbf u_{k,i},t_k+h_k\tau_i),\\
\forall\,k=0,\ldots,N-1,\quad i=1,\ldots,s.
\end{gathered}
$$

This identity holds by construction, before the NLP constraints have been
satisfied. It does not yet guarantee that the reconstructed curve satisfies
the ODE: the dynamics on the right are evaluated at the stored state
$\mathbf x_{k,i}$, which may differ from the curve's value
$\mathbf p_k(\tau_i)$.

The remaining condition is agreement between these two state values.
The point $(\tau_i,\mathbf{x}_{k,i})$ must lie on the graph of
$\mathbf{p}_k$.
Imposing $\mathbf{x}_{k,i}=\mathbf{p}_k(\tau_i)$ at every stage of every
interval gives the **stage equations**

$$
\mathbf{x}_{k,i}
=\mathbf{x}_k+h_k\sum_{j=1}^{s}A_{ij}\mathbf{f}_{k,j},
\qquad \forall\,k=0,\ldots,N-1,\quad i=1,\ldots,s.
$$

Their coefficients are

$$
A_{ij}=\int_0^{\tau_i}\ell_j(\tau)\,d\tau,
\qquad i,j=1,\ldots,s.
$$

Each stage equation therefore sets a state-matching residual to zero:

$$
\mathbf r^{\mathrm{int}}_{k,i}
=\mathbf x_{k,i}-\mathbf p_k(\tau_i)=\mathbf0,
\qquad \forall\,k=0,\ldots,N-1,\quad i=1,\ldots,s.
$$

Once this residual vanishes, the stored state used to evaluate
$\mathbf f_{k,i}$ is the state on the reconstructed curve. Substituting
$\mathbf x_{k,i}=\mathbf p_k(\tau_i)$ into the derivative identity above
then gives the ODE at that node. Thus the integration form enforces the
ODE at collocation nodes even though its explicit stage residuals compare
state values rather than slopes.

For example, consider $\dot x=x$ on an interval of length $h_k=1$, starting
from $x_k=1$, with one collocation node at $\tau_1=1/2$. A trial stage
value $x_{k,1}=3$ supplies the slope $f_{k,1}=3$ and hence the state piece
$p_k(\tau)=1+3\tau$. Its midpoint value is $5/2$, not the stored value $3$.
The polynomial's slope is $3$, but the ODE evaluated on the curve at the
midpoint asks for $5/2$. The state equation
$x_{k,1}=1+\tfrac12 x_{k,1}$ instead gives $x_{k,1}=2$.
The reconstructed polynomial is then $p_k(\tau)=1+2\tau$, whose midpoint
state and slope both equal $2$. The state-matching constraint has made
the ODE hold at the collocation node.

Row $i$ of $A$ integrates each slope basis function from $0$ to $\tau_i$.
Here $i$ selects the destination stage and $j$ runs over all slope stages
that contribute to the polynomial. All these integrals start at $0$,
including when $i>1$; they do not start at the preceding stage. The stage
equations are simultaneous constraints, since the slopes on the right
depend on the state and control variables being chosen.

For example, take three slope stages at
$(\tau_1,\tau_2,\tau_3)=(0,\tfrac12,1)$ and fix the destination $i=2$.
Every integral in row 2 of $A$ now ends at $\tau_2=\tfrac12$.
Varying $j$ selects the basis function being integrated:
$\ell_1$, $\ell_2$, or $\ell_3$.
@fig-collocation-destination-and-slopes shows these three signed areas.
Although the third slope is evaluated at the right endpoint $\tau_3=1$,
its basis function $\ell_3$ is nonzero between $0$ and $\tfrac12$.
That slope therefore contributes to the midpoint state as well. The sum
runs over all $j=1,\ldots,s$, not only stages with $j\leq i$.

:::{figure} _static/collocation/destination-and-slope-stages.svg
:label: fig-collocation-destination-and-slopes
:alt: Three panels show the cardinal slope basis functions for nodes zero, one half, and one. The destination is fixed at stage i equals two, marked by the same vertical dashed line at one half in each panel. Shaded signed areas from zero to one half give A21 equals five twenty-fourths, A22 equals one third, and A23 equals minus one twenty-fourth. All three slopes enter the midpoint state equation.

Fixing $i$ chooses the integration endpoint; varying $j$ includes every
slope basis function. For nodes $(0,\tfrac12,1)$, the blue curves are
$\ell_1(\tau)=2\tau^2-3\tau+1$, $\ell_2(\tau)=4\tau(1-\tau)$, and
$\ell_3(\tau)=2\tau^2-\tau$. Each blue point marks the node where its
basis function equals one. The orange dashed line marks the common upper
limit $\tau_i=\tau_2=\tfrac12$. The shaded signed areas form row 2 of $A$:
$(A_{21},A_{22},A_{23})=(\tfrac5{24},\tfrac13,-\tfrac1{24})$.
The third area is negative because $\ell_3$ lies below zero on this span;
it is not absent just because stage 3 lies after the destination. The bottom
equation expands the sum for a scalar state; the same coefficients multiply
vector-valued slopes in the general case.
:::

The right endpoint has local coordinate $1$. Evaluating the same polynomial
there uses the full integration span $[0,1]$, in place of the span
$[0,\tau_i]$ used for stage $i$. Requiring this value $\mathbf{p}_k(1)$
to equal the stored endpoint state $\mathbf{x}_{k+1}$ gives

$$
\boxed{
\mathbf{x}_{k+1}
=\mathbf{x}_k+h_k\sum_{j=1}^{s}b_j\mathbf{f}_{k,j},
\qquad
b_j=\int_0^1\ell_j(\tau)\,d\tau.
}
$$

The coefficients $b_j$ integrate the cardinal functions over the full
reference interval. In code, both $A_{ij}$ and $b_j$ are obtained by forming
an antiderivative polynomial and evaluating it at the integration limits.
No repeated integration of the nonlinear dynamics is involved in computing
these coefficients. The worked calculation in
@sec-collocation-coefficient-generator shows the coefficient arithmetic.
If a stage lies at $\tau_i=1$, then $A_{ij}=b_j$ for
every $j$, so its stage equation gives the same polynomial value as the
endpoint equation. Its state must therefore equal $\mathbf{x}_{k+1}$;
an implementation can share that variable.

The endpoint equation is a defect constraint: it requires
the stored endpoint to match the endpoint obtained by integrating the slopes
of piece $k$. The next piece starts from that same stored value, so the two
pieces meet:

$$
\mathbf{p}_k(1)=\mathbf{x}_{k+1}=\mathbf{p}_{k+1}(0).
$$

This shared variable enforces state continuity across the join. If an
implementation stores separate endpoint variables for the two pieces, it
must impose an equality between them. Choosing an endpoint as a collocation
node does not by itself join the pieces. Continuity of the state also does
not require continuity of its derivative; neighboring pieces can meet with
different slopes.

:::{figure} _static/collocation/piecewise-trajectory.svg
:label: fig-piecewise-collocation-trajectory
:alt: Three quadratic state segments meet at shared endpoint states in physical time. The middle segment is repeated on a normalized time axis from zero to one, with two interior stage values and their slopes marked.

A scalar trajectory consists of polynomial pieces joined at shared endpoint
states (squares). The middle piece $p_k$ is shown in blue in both panels; only
its time coordinate changes. The stage values (orange dots) lie at
$\tau_1=1/3$ and $\tau_2=2/3$. Tangent marks in the lower panel have slopes
$p_k'(\tau_j)=h_kf_{k,j}$ because that axis uses normalized time. The stage
equations place the dots on the piece, and the endpoint defect makes it reach
$x_{k+1}$, where the next piece begins. This schematic uses quadratic state
pieces and linear slope interpolants.
:::

The running-cost integral can be treated by exactly the same construction.
As in the earlier Bolza-to-Mayer reduction, introduce a scalar state $y(t)$
that records the cost accumulated since
$t_0$:

$$
y(t)=\int_{t_0}^{t}c(\mathbf{x}(s),\mathbf{u}(s),s)\,ds,
\qquad
\dot y(t)=c(\mathbf{x}(t),\mathbf{u}(t),t),\qquad y(t_0)=0.
$$

The original objective is now $c_f(\mathbf{x}(t_f),t_f)+y(t_f)$. The augmented state
$(\mathbf{x}^{\top},y)^{\top}$ has two rates of change: $\mathbf{f}$ supplies the physical state rate and $c$
supplies the cost rate. Applying the same collocation method to this augmented
ODE evaluates both rates at each stage's state, control, and time. In
particular, the cost rate at stage $j$ of interval $k$ is

$$
c_{k,j}:=c(\mathbf{x}_{k,j},\mathbf{u}_{k,j},t_k+h_k\tau_j).
$$

Let $y_k$ and $y_{k,i}$ denote the accumulated-cost values at the interval's
left endpoint and at stage $i$. Integrating the cost-rate interpolant to a
stage and to the right endpoint gives

$$
y_{k,i}=y_k+h_k\sum_{j=1}^{s}A_{ij}c_{k,j},
\qquad
y_{k+1}=y_k+h_k\sum_{j=1}^{s}b_jc_{k,j}.
$$

These are the same stage and endpoint equations as for $\mathbf{x}$, with $c_{k,j}$
in place of $\mathbf{f}_{k,j}$. The increase in accumulated cost across interval $k$
therefore supplies its contribution to the discrete objective:

$$
c_k:=y_{k+1}-y_k
=h_k\sum_{j=1}^{s}b_jc_{k,j}
\approx\int_{t_k}^{t_{k+1}}c(\mathbf{x}_h(t),\mathbf{u}_h(t),t)\,dt.
$$

Here $\mathbf{u}_h$ is the piecewise control approximation. The single-index
$c_k$ is an interval cost, whereas $c_{k,j}$ is a sampled cost rate; the factor
$h_k$ gives them different units. The weighted sum integrates
the polynomial interpolating the sampled cost rates exactly. It approximates
the running-cost integral because the composed function
$c(\mathbf{x}_h(t),\mathbf{u}_h(t),t)$ need not itself be that polynomial. Such a weighted-sum
approximation is called a **quadrature rule**.

Since $y_0=0$, summing the interval increments gives
$y_N=\sum_{k=0}^{N-1}c_k$. The accumulated-cost variables can therefore be
eliminated and the sum used directly in the objective. Introducing $y$ explains
why applying one collocation scheme to both rates uses the same nodes and
weights; it does not require adding cost variables to the implementation.
A different quadrature rule is also possible, with the state and control
polynomials evaluated at that rule's nodes.

### Choosing a polynomial for the control

The state and control play different roles. The ODE constrains the state's
derivative, so integrating an interpolant of $s$ slopes gives a state
polynomial of degree at most $s$. There is no corresponding ODE for the
control in this problem. Its representation is a separate choice: a
constant command on each interval, a line between endpoint commands, or a
higher-degree polynomial.

For example, with two endpoint slopes the state polynomial is quadratic.
The control can still be the line

$$
\mathbf u_h(t_k+h_k\tau)
=(1-\tau)\mathbf u_k+\tau\mathbf u_{k+1}.
$$

These degrees need not match. Also, substituting polynomial state and control
curves into a nonlinear $\mathbf f$ need not produce a polynomial. Collocation
matches the state derivative to $\mathbf f$ at the chosen nodes, rather than
requiring the ODE to hold identically between them.

To make the control choice explicit, select distinct control support nodes
$\rho_0,\ldots,\rho_{d_u}\in[0,1]$ and construct their cardinal functions
$\psi_0,\ldots,\psi_{d_u}$. The decision variables
$\widehat{\mathbf u}_{k,r}$ are control values at these support nodes:

$$
\mathbf u_h(t_k+h_k\tau)
=\sum_{r=0}^{d_u}\widehat{\mathbf u}_{k,r}\psi_r(\tau),
\qquad
\mathbf u_{k,j}
=\sum_{r=0}^{d_u}B_{jr}\widehat{\mathbf u}_{k,r},
\quad B_{jr}:=\psi_r(\tau_j).
$$

Thus the stage controls $\mathbf u_{k,j}$ are computed from the control
variables; they are independent variables only when the representation
allows independent values at all those stages. For a constant control,
$d_u=0$ and every entry of $B$ is one. For a linear control supported at
$0,1$, row $j$ of $B$ is $(1-\tau_j,\tau_j)$.

Using more control support values than collocation stages leaves some control
variations invisible to those stage samples. The examples below use
$d_u+1\leq s$; a richer control representation needs additional sampling or
other constraints to account for those variations.

The ODE requires continuous states, but it can admit controls that jump
between intervals. Independent control polynomials therefore need no
continuity constraint unless the model or chosen parameterization requires
one. Sharing endpoint controls between neighboring linear pieces enforces
continuity. With separate local variables, the equivalent equality is

$$
\sum_{r=0}^{d_u}\psi_r(1)\widehat{\mathbf u}_{k,r}
=\sum_{r=0}^{d_u}\psi_r(0)\widehat{\mathbf u}_{k+1,r}.
$$

At a jump, endpoint stages use the control belonging to their own interval.
Finally, bounds on support values guarantee bounds throughout a constant or
linear control segment, because its values are convex combinations of its
endpoints. A higher-degree polynomial can overshoot those values, so support
bounds alone no longer give that guarantee.

### Assembling the nonlinear program

The state and control representations now specify which quantities the
optimizer chooses and which it evaluates from those choices. The mesh states
$\mathbf x_k,\mathbf x_{k+1}$ are shared decision variables at the ends of
interval $k$. The stage states $\mathbf x_{k,j}$ are decision variables at
the collocation nodes; a stage at an endpoint can use the mesh state itself.
For the control, the optimizer chooses the support values
$\widehat{\mathbf u}_{k,r}$ at $\rho_r$. Evaluating the control polynomial
at $\tau_j$ gives $\mathbf u_{k,j}$. Together with $\mathbf x_{k,j}$ and
the time $t_k+h_k\tau_j$, this evaluated control supplies the inputs to
$\mathbf f$ and $c$.

:::{figure} _static/collocation/nlp-quantities.svg
:label: fig-collocation-nlp-quantities
:alt: State and control curves share one interval with collocation nodes at one third and two thirds. Filled blue squares mark shared mesh state variables, filled blue circles mark stage state variables, and filled blue diamonds mark control support variables at zero and one. An open orange circle marks the stage control evaluated using B. The selected stage state, evaluated control, and time determine the dynamics and cost rates on the right.

Decision variables and evaluated quantities on one interval. The curves
show a scalar example with two collocation nodes, $\tau_1=1/3$ and
$\tau_2=2/3$; the highlighted stage is $j=2$. Squares mark mesh states
shared with neighboring intervals, and filled circles mark stage states.
Diamonds mark the independent control values at $\rho_0=0$ and $\rho_1=1$.
For this linear control, $B_{j,:}=(1-\tau_j,\tau_j)$ evaluates the open
circle $\mathbf u_{k,j}$ from the two diamonds. The quantities
$\mathbf f_{k,j}$ and $c_{k,j}$ are then evaluated at the stage state,
control, and physical time. Filled blue markers are decision variables;
the orange control value and rates are computed from them. The curves
depict a feasible choice: the stage and endpoint constraints place the
state variables on the integrated polynomial. If a collocation node is an
endpoint, its stage state can be identified with the shared mesh variable.
:::

Collect the independent state and control values across all intervals into
the decision vector

$$
\mathbf z=\operatorname{col}\!\left(
\{\mathbf x_k\}_{k=0}^{N},
\{\mathbf x_{k,j}\}_{k=0,\,j=1}^{N-1,\,s},
\{\widehat{\mathbf u}_{k,r}\}_{k=0,\,r=0}^{N-1,\,d_u}
\right).
$$

Here $\operatorname{col}$ stacks the listed vectors into one column, omitting
duplicate variables when a stage shares a mesh endpoint. Stage controls are
evaluated using $B$, and any chosen control-continuity equalities are added
to the constraints. If $t_f$ is optimized,
it is included in $\mathbf z$ as well. The resulting finite optimization
problem has the form

$$
\begin{aligned}
\underset{\mathbf{z}}{\operatorname{minimize}}\quad&
c_f(\mathbf{x}_N,t_f)+\sum_{k=0}^{N-1}c_k\\
\text{subject to}\quad&
\text{stage equations},\\
&\text{endpoint defects and continuity},\\
&\text{boundary, path, and bound constraints}.
\end{aligned}
$$

In the preceding chapter's NLP notation, $F(\mathbf z)$ is this scalar
objective, $H(\mathbf z)=\mathbf0$ stacks the stage, endpoint, and boundary
equalities, and $G(\mathbf z)\leq\mathbf0$ stacks the sampled path and bound
inequalities. The slope $\mathbf f_{k,j}$ is a vector, distinct from $F$.

The original path and bound constraints apply at every continuous time.
This NLP imposes them at selected points, usually its support or collocation
nodes. Feasibility at those nodes does not rule out a violation between them.
Checking the polynomial between nodes and simulating the chosen control
provide complementary diagnostics, developed below.

Every interval constraint touches only local stage variables and neighboring
endpoint states. Consequently, most derivatives of the constraints with
respect to the decision variables are zero. With variables ordered by time,
the nonzero blocks lie near the diagonal of the Jacobian, the matrix of
constraint derivatives. NLP solvers can exploit this sparse, block-banded
pattern. The arrays $A$ and $b$ are numerical constants computed before
optimization.

### Equivalent differentiation form

The differentiation form uses the same collocation nodes $\tau_i$ as the
integration form. It changes which consistency condition is built into the
representation and which must be enforced by a residual. The integration
form constructs a curve whose nodal derivatives match the evaluated stage
slopes, then constrains its values to match the stored stage states. The
differentiation form constructs a curve from state-support values, evaluates
its stage states directly on that curve, and constrains its derivatives
to match the ODE.

To represent the same state polynomials, retain the same degree. The integral
of the degree-$(s-1)$ slope polynomial has
degree at most $s$, so take $d=s$ and choose $d+1$ support nodes
$\sigma_0,\ldots,\sigma_d$ for the state polynomial.
Write $\ell_r^{\mathrm{state}}$ for their cardinal
functions to distinguish this support-node basis from the slope-node basis
$\ell_j$. Denote the support values by $\widehat{\mathbf x}_{k,r}$;
these need not be the stage values $\mathbf x_{k,j}$ at $\tau_j$:

$$
\mathbf{x}_h(t_k+h_k\tau)
=\sum_{r=0}^{d}\widehat{\mathbf x}_{k,r}\ell_r^{\mathrm{state}}(\tau).
$$

At a collocation node $\tau_i$, the state is a weighted sum of its support values,
and its derivative with respect to $\tau$ is another weighted sum. The fixed
arrays containing these weights are

$$
E_{ir}=\ell_r^{\mathrm{state}}(\tau_i),
\qquad
D_{ir}=(\ell_r^{\mathrm{state}})'(\tau_i).
$$

The $i$th row of $E$ computes $\mathbf p_k(\tau_i)$, the point on the state
polynomial that we supply to the dynamics function. The $i$th row of $D$
computes $\mathbf p_k'(\tau_i)$, the tangent slope measured per unit of local
time $\tau$. Both calculations use the same stored support states
$\widehat{\mathbf x}_{k,r}$; they do not introduce new decision variables.
In particular, the stage state is defined as
$\mathbf x_{k,i}=\mathbf p_k(\tau_i)=\sum_r E_{ir}\widehat{\mathbf x}_{k,r}$.
State-value consistency therefore holds for every candidate support array;
there is no separate state-matching residual at a stage.

To compare this tangent slope with the rate returned by $\mathbf f$, both
must be measured in physical time. A local-time increment $\Delta\tau$
corresponds to a physical-time increment $\Delta t=h_k\Delta\tau$. We
therefore divide the local-time slope by $h_k$ to obtain the physical-time
slope: this is the chain rule $d/dt=(1/h_k)d/d\tau$ used earlier.
The remaining constraint sets the physical-time slope residual to zero:

$$
\begin{gathered}
\mathbf r^{\mathrm{diff}}_{k,i}
=\frac{1}{h_k}\mathbf p_k'(\tau_i)
-\mathbf f\!\left(\mathbf p_k(\tau_i),\mathbf u_{k,i},t_k+h_k\tau_i\right)
=\mathbf0,\\
\forall\,k=0,\ldots,N-1,\quad i=1,\ldots,s.
\end{gathered}
$$

Both slopes are evaluated at the same point on the state curve. Unlike
state-value consistency, their equality is not built into the state
interpolant. Writing these evaluations using $E,D$ and multiplying by
$h_k$ gives

$$
\begin{gathered}
\boxed{
\sum_{r=0}^{d}D_{ir}\widehat{\mathbf x}_{k,r}
=h_k \mathbf{f}\left(
\sum_{r=0}^{d}E_{ir}\widehat{\mathbf x}_{k,r},
\mathbf{u}_{k,i},
t_k+h_k\tau_i
\right).
}\\
\forall\,k=0,\ldots,N-1,\quad i=1,\ldots,s.
\end{gathered}
$$

The left side is the polynomial's slope with respect to $\tau$; the right
side is the ODE's physical-time rate multiplied by $h_k$.
The scalar quadratic in @fig-collocation-evaluation-differentiation shows
both calculations at a collocation node that is not a support node.

:::{figure} _static/collocation/evaluation-and-differentiation.svg
:label: fig-collocation-evaluation-differentiation
:alt: Three square support states define a quadratic. At a collocation node between the support nodes, E computes the value shown by an open circle and D computes the dashed tangent's slope. Dividing this slope by the interval length gives the physical-time rate that must match the ODE.

Evaluation and differentiation of the same state polynomial. The support
states $(1,2,4)$ at $\sigma=(0,\tfrac12,1)$ define
$p_k(\tau)=1+\tau+2\tau^2$. At the highlighted collocation node
$\tau_i=\tfrac14$, $E$ computes the state $p_k(\tau_i)=\tfrac{11}{8}$,
while $D$ computes the tangent slope $p_k'(\tau_i)=2$. With $h_k=2$,
the physical-time slope is $2/2=1$. For the dynamics $\dot x=u$ and stage
control $u_{k,i}=1$, the ODE also returns $1$, so the collocation constraint
holds at this node. Taking $t_k=2$ places the node at physical time
$t_k+h_k\tau_i=\tfrac52$.
:::

In matrix shorthand, all nodal constraints are $D\mathsf X_k=h_k\mathsf F_k$.
Here $\mathsf X_k\in\mathbb R^{(d+1)\times n}$
has support states as rows, and $\mathsf F_k\in\mathbb R^{s\times n}$ has
stage slopes as rows; neither is a single state vector. Evaluation at the two
ends connects this polynomial to the shared mesh states:

$$
\mathbf{x}_{k}=\sum_{r=0}^{d}\ell_r^{\mathrm{state}}(0)\widehat{\mathbf x}_{k,r},
\qquad
\mathbf{x}_{k+1}=\sum_{r=0}^{d}\ell_r^{\mathrm{state}}(1)\widehat{\mathbf x}_{k,r}.
$$

When support and collocation nodes coincide, the cardinal property makes $E$
select the corresponding stored state directly. The slope-value and
state-value forms describe the same polynomial construction; one integrates
nodal slopes, while the other differentiates nodal states.

This implementation pattern is standard in direct collocation {cite:p}`Kelly2017DirectCollocation,Andersson2019CasADi`. The official [CasADi direct-collocation example](https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_collocation.py) constructs fixed differentiation, endpoint, and quadrature arrays from Lagrange polynomials, while the NLP variables remain state and control values.

In the integration form, matching the evaluated stage slopes is built into
the polynomial construction, while matching the stage states requires
constraints. In the differentiation form, matching stage states to the
curve is built into evaluation, while matching ODE slopes requires
constraints. Both forms also connect the polynomial to the shared mesh
endpoints. With the same collocation nodes, state degree, and control
representation, their zero residuals describe the same collocation
trajectories, although their residual values need not agree for an
infeasible candidate. The node-to-NLP recipe for the
differentiation form is developed in
@sec-collocation-differentiation-from-nodes.

## Low-Order Transcriptions

Which familiar integration rules appear when the polynomial and node sets are
reduced to their lowest-order choices?

The general construction becomes concrete when only one or two slope values
are retained on each interval. These cases recover familiar integration
formulas, but the formulas now appear as constraints inside an NLP.
For the concrete problems below, fix the mesh and final time. The boundary
map $\mathbf h(\mathbf x_0,\mathbf x_N,t_N)=\mathbf0$ includes the
prescribed initial state and any terminal conditions; $\mathbf g\leq\mathbf0$
includes path constraints and state/control bounds. Each displayed program
specifies the points where those inequalities are imposed. Additional check
points can be included, and a continuous replay still checks the result
between nodes.

### One slope value: explicit and implicit Euler

A single slope value determines a constant derivative on each interval.
Choose a constant control $\mathbf u_{k,1}$ on that interval as well:
$\mathbf u_h(t_k+h_k\tau)=\mathbf u_{k,1}$. Different intervals may use
different constants. The state is linear even though the control is constant.
The following exercise applies the preceding construction to recover the two
Euler methods. The solution gives a recipe that also extends to multiple
slope nodes.

````{exercise} Derive a transcription from one slope
:label: ex-collocation-one-slope-recipe

Consider $\dot{\mathbf x}=\mathbf{f}(\mathbf x,\mathbf u,t)$ on $[t_k,t_{k+1}]$, with duration $h_k$ and
normalized time $\tau=(t-t_k)/h_k$. Approximate the physical-time derivative
by a constant that matches the ODE at a single collocation node $\tau_1$.

Carry out the construction for $\tau_1=0$ and then for $\tau_1=1$. For each choice:

1. Construct the cardinal function and identify the ODE slope it multiplies.
2. Integrate from $\mathbf{x}_k$ to obtain the state polynomial $\mathbf{p}_k(\tau)$, then
   impose its agreement with the stored right endpoint $\mathbf{x}_{k+1}$.
3. Derive the interval cost contribution by applying the same construction
   to $\dot y=c(\mathbf x,\mathbf u,t)$.
4. Identify the method, state the degree of its state polynomial, and explain
   whether its endpoint equation can be evaluated directly in a forward
   simulation with prescribed controls.
````

````{solution} ex-collocation-one-slope-recipe
:class: dropdown

The derivation follows five steps: choose where the ODE is evaluated,
interpolate its slope, integrate that interpolant, match the stored endpoint,
and integrate the cost rate with the same weights.

1. **Choose the stage and evaluate its slope.** The left node uses the
   state, control, and time at the start of the interval; the right node uses
   those at its end:

   $$
   \mathbf{f}_{k,1}=\begin{cases}
   \mathbf{f}(\mathbf{x}_k,\mathbf u_{k,1},t_k),&\tau_1=0,\\
   \mathbf{f}(\mathbf{x}_{k+1},\mathbf u_{k,1},t_{k+1}),&\tau_1=1.
   \end{cases}
   $$

2. **Interpolate the physical-time derivative.** With one node, the cardinal
   function is the constant $\ell_1(\tau)=1$, since it must equal one at
   that node. Hence $\dot{\mathbf{x}}_h(t_k+h_k\tau)=\mathbf{f}_{k,1}$ throughout the interval.
   Its full-interval integration weight is
   $b_1=\int_0^1\ell_1(\tau)\,d\tau=1$.

3. **Integrate from the left state.** The reference-time derivative is
   $\mathbf{p}_k'(\tau)=h_k\mathbf{f}_{k,1}$. Integrating and imposing $\mathbf{p}_k(0)=\mathbf{x}_k$ gives

   $$
   \mathbf{p}_k(\tau)=\mathbf{x}_k+h_k\tau \mathbf{f}_{k,1}.
   $$

   Thus the state polynomial has degree at most one. The factor $h_k$
   accounts for the physical duration of the interval.

4. **Match the stored right endpoint.** Setting $\mathbf{p}_k(1)=\mathbf{x}_{k+1}$ gives the
   defect $\mathbf{x}_{k+1}-\mathbf{x}_k-h_k\mathbf{f}_{k,1}=\mathbf0$. Substituting the two slope choices yields

   $$
   \begin{aligned}
   \mathbf{x}_{k+1}-\mathbf{x}_k-h_k\mathbf{f}(\mathbf{x}_k,\mathbf u_{k,1},t_k)&=\mathbf0
   &&\text{(explicit Euler)},\\
   \mathbf{x}_{k+1}-\mathbf{x}_k-h_k\mathbf{f}(\mathbf{x}_{k+1},\mathbf u_{k,1},t_{k+1})&=\mathbf0
   &&\text{(implicit Euler)}.
   \end{aligned}
   $$

   In forward simulation, explicit Euler computes the next state from known
   left-endpoint data. Implicit Euler places the unknown next state inside
   $\mathbf{f}$, so it generally requires solving an equation, which is nonlinear
   when the dynamics depend nonlinearly on that state. In direct
   transcription, both endpoints are optimization variables and either
   defect is imposed as an equality constraint.

5. **Apply the same integration rule to the cost rate.** The single sampled
   cost rate is constant in the approximation, so its integral is that rate
   multiplied by $h_k$:

   $$
   c_k=\begin{cases}
   h_kc(\mathbf{x}_k,\mathbf u_{k,1},t_k),&\tau_1=0,\\
   h_kc(\mathbf{x}_{k+1},\mathbf u_{k,1},t_{k+1}),&\tau_1=1.
   \end{cases}
   $$

   These are the left- and right-endpoint approximations to the interval
   running-cost integral. They use the same stage as the corresponding
   dynamics constraint.

With additional slope nodes, the sequence stays the same. The constant
cardinal function is replaced by several cardinal polynomials, and their
integrals supply the weights in the stage equations, endpoint defect, and
cost contribution.
````

### The complete Euler NLP

Let $\theta=0$ for explicit Euler and $\theta=1$ for implicit Euler. Both
methods optimize the mesh states $\mathbf x_0,\ldots,\mathbf x_N$ and the
$N$ interval controls $\mathbf u_{k,1}$. There are no additional stage-state
variables after identifying the sole stage with its endpoint:

$$
\begin{aligned}
\min_{\{\mathbf x_k\}_{k=0}^N,\,\{\mathbf u_{k,1}\}_{k=0}^{N-1}}
\quad &c_f(\mathbf x_N,t_N)
+\sum_{k=0}^{N-1}h_kc(\mathbf x_{k+\theta},\mathbf u_{k,1},t_{k+\theta})\\
\text{subject to}\quad
&\mathbf x_{k+1}-\mathbf x_k
-h_k\mathbf f(\mathbf x_{k+\theta},\mathbf u_{k,1},t_{k+\theta})=\mathbf0,\\
&\mathbf g(\mathbf x_{k+\theta},\mathbf u_{k,1},t_{k+\theta})\leq\mathbf0,
\quad k=0,\ldots,N-1,\\
&\mathbf h(\mathbf x_0,\mathbf x_N,t_N)=\mathbf0.
\end{aligned}
$$

Here $k+\theta$ selects an endpoint, not an intermediate time. The path
constraints use the same endpoint as the dynamics; bounds required at the
other endpoint can be imposed there too. The two NLPs differ only in where
the slope and cost rate are evaluated. Both are solved for the entire
trajectory simultaneously.

### Endpoint slope values: trapezoidal transcription

Use continuous piecewise-linear controls, with shared endpoint variables
$\mathbf u_0,\ldots,\mathbf u_N$. Choose the endpoint collocation nodes
$\tau_1=0$ and $\tau_2=1$, and abbreviate the
two ODE slopes by

$$
\mathbf{f}_k=\mathbf{f}(\mathbf{x}_k,\mathbf{u}_k,t_k),\qquad
\mathbf{f}_{k+1}=\mathbf{f}(\mathbf{x}_{k+1},\mathbf{u}_{k+1},t_{k+1}).
$$

For this endpoint formula, label the cardinal functions by their locations, $0$ and $1$, rather than by the stage numbers $1$ and $2$:

$$
\ell_0(\tau)=1-\tau,\qquad
\ell_1(\tau)=\tau.
$$

The derivative interpolant is the line joining the two slopes:

$$
\dot{\mathbf{x}}_h(t_k+h_k\tau)
=(1-\tau)\mathbf{f}_k+\tau \mathbf{f}_{k+1}.
$$

It equals $\mathbf{f}_k$ at $\tau=0$ and $\mathbf{f}_{k+1}$ at $\tau=1$. Integrating from the
stored left state gives the continuous state approximation

$$
\mathbf{x}_h(t_k+h_k\tau)
=\mathbf{x}_k+h_k\left[
\left(\tau-\frac{\tau^2}{2}\right)\mathbf{f}_k
+\frac{\tau^2}{2}\mathbf{f}_{k+1}
\right].
$$

The state approximation is quadratic, even though an implementation may store
only its endpoint states and slopes. Evaluating this polynomial at $\tau=1$
and equating it to the stored endpoint produces

$$
\boxed{
\mathbf{x}_{k+1}-\mathbf{x}_k
-\frac{h_k}{2}
\left[
\mathbf{f}(\mathbf{x}_k,\mathbf{u}_k,t_k)
+\mathbf{f}(\mathbf{x}_{k+1},\mathbf{u}_{k+1},t_{k+1})
\right]
=\mathbf0.
}
$$

The coefficients $1/2$ are the integrals of the two linear cardinal functions.
This is the trapezoidal defect derived in the opening example. Applying the
same endpoint weights to the running cost gives

$$
c_k
=
\frac{h_k}{2}
\left[
c(\mathbf{x}_k,\mathbf{u}_k,t_k)
+c(\mathbf{x}_{k+1},\mathbf{u}_{k+1},t_{k+1})
\right].
$$

The two endpoint costs are averaged and multiplied by the interval length.
Integration increases the degree: a linear derivative interpolant produces
a quadratic state interpolant. This construction also appears in
{cite:t}`Kelly2017DirectCollocation`.

### The complete trapezoidal NLP

The decision variables are the mesh states and controls. Write
$\mathbf f_k=\mathbf f(\mathbf x_k,\mathbf u_k,t_k)$ and
$c_k^{\mathrm{rate}}=c(\mathbf x_k,\mathbf u_k,t_k)$ for evaluated quantities,
not additional optimization variables. Then the transcribed problem is

$$
\begin{aligned}
\min_{\{\mathbf x_k,\mathbf u_k\}_{k=0}^{N}}\quad
&c_f(\mathbf x_N,t_N)
+\sum_{k=0}^{N-1}\frac{h_k}{2}
\left(c_k^{\mathrm{rate}}+c_{k+1}^{\mathrm{rate}}\right)\\
\text{subject to}\quad
&\mathbf x_{k+1}-\mathbf x_k
-\frac{h_k}{2}(\mathbf f_k+\mathbf f_{k+1})=\mathbf0,
\quad k=0,\ldots,N-1,\\
&\mathbf g(\mathbf x_k,\mathbf u_k,t_k)\leq\mathbf0,
\quad k=0,\ldots,N,\\
&\mathbf h(\mathbf x_0,\mathbf x_N,t_N)=\mathbf0.
\end{aligned}
$$

The state between endpoints is the quadratic obtained above, while the
control is linear. Shared mesh states and controls connect adjacent pieces;
no separate continuity equations are needed in this representation.

## Hermite--Simpson Transcription

Can midpoint state and slope information raise the transcription order without
requiring a high-degree polynomial over the whole horizon?

Hermite--Simpson extends the trapezoidal construction by adding the midpoint
state $\mathbf{x}_{k+\frac12}$. Retain the same piecewise-linear control, so
its midpoint value is already determined:

$$
\mathbf u_{k+\frac12}=\frac{\mathbf u_k+\mathbf u_{k+1}}{2}.
$$

Evaluate the midpoint ODE slope using that control:

$$
\mathbf{f}_{k+\frac12}=\mathbf{f}\left(
\mathbf{x}_{k+\frac12},\mathbf{u}_{k+\frac12},t_k+\frac{h_k}{2}
\right).
$$

The notation $\mathbf x_{k+\frac12}$ and $\mathbf u_{k+\frac12}$ denotes
values at physical time $t_k+h_k/2$, not another mesh endpoint.

The three stage nodes are $\tau_1=0$, $\tau_2=\tfrac12$, and $\tau_3=1$. We label their cardinal functions $0,m,1$ for left endpoint, midpoint, and right endpoint. The derivative interpolant is the
quadratic built from the following three cardinal functions:

$$
\ell_0(\tau)=2\tau^2-3\tau+1,\qquad
\ell_m(\tau)=4\tau(1-\tau),\qquad
\ell_1(\tau)=2\tau^2-\tau.
$$

Integrating each function over the full reference interval gives

$$
b_0=\frac16,\qquad
b_m=\frac46,\qquad
b_1=\frac16.
$$

These are the familiar Simpson quadrature weights. Substituting them into the
general endpoint equation produces the Simpson defect:

$$
\boxed{
\mathbf{x}_{k+1}-\mathbf{x}_k
-\frac{h_k}{6}
\left(\mathbf{f}_k+4\mathbf{f}_{k+\frac12}+\mathbf{f}_{k+1}\right)
=\mathbf0.
}
$$

The midpoint state must also lie on the cubic obtained by integrating the
quadratic derivative. Integration only to $\tau=\tfrac12$ gives

$$
\mathbf{x}_{k+\frac12}
=\mathbf{x}_k+\frac{h_k}{24}
\left(5\mathbf{f}_k+8\mathbf{f}_{k+\frac12}-\mathbf{f}_{k+1}\right).
$$

This form still contains the midpoint slope. The endpoint defect gives

$$
4\mathbf{f}_{k+\frac12}
=\frac{6}{h_k}(\mathbf{x}_{k+1}-\mathbf{x}_k)-\mathbf{f}_k-\mathbf{f}_{k+1}.
$$

Substituting this expression into the midpoint equation and collecting the
endpoint states and slopes yields

$$
\boxed{
\mathbf{x}_{k+\frac12}
=\frac{\mathbf{x}_k+\mathbf{x}_{k+1}}{2}
+\frac{h_k}{8}\left(\mathbf{f}_k-\mathbf{f}_{k+1}\right).
}
$$

The midpoint state relation and the definition of $\mathbf{f}_{k+\frac12}$ together
enforce the ODE at the midpoint. The resulting state interpolant is cubic
because it integrates a quadratic derivative. The name reflects its two ingredients: the
state is a cubic Hermite interpolant determined by state and slope information,
and its endpoint defect uses Simpson weights.

### The complete Hermite--Simpson NLP

With linear controls, optimize mesh states, mesh controls, and midpoint
states. For $q=k$ or $k+\tfrac12$, abbreviate
$\mathbf f_q=\mathbf f(\mathbf x_q,\mathbf u_q,t_q)$ and
$c_q^{\mathrm{rate}}=c(\mathbf x_q,\mathbf u_q,t_q)$, with
$t_{k+\frac12}=t_k+h_k/2$ and the midpoint control given by the endpoint
average. The complete program is

$$
\begin{aligned}
\min_{\{\mathbf x_k,\mathbf u_k\}_{k=0}^{N},\,
\{\mathbf x_{k+\frac12}\}_{k=0}^{N-1}}\quad
&c_f(\mathbf x_N,t_N)
+\sum_{k=0}^{N-1}\frac{h_k}{6}
\left(c_k^{\mathrm{rate}}+4c_{k+\frac12}^{\mathrm{rate}}+c_{k+1}^{\mathrm{rate}}\right)\\
\text{subject to}\quad
&\mathbf x_{k+1}-\mathbf x_k
-\frac{h_k}{6}(\mathbf f_k+4\mathbf f_{k+\frac12}+\mathbf f_{k+1})=\mathbf0,\\
&\mathbf x_{k+\frac12}-\frac{\mathbf x_k+\mathbf x_{k+1}}{2}
-\frac{h_k}{8}(\mathbf f_k-\mathbf f_{k+1})=\mathbf0,
\quad k=0,\ldots,N-1,\\
&\mathbf g(\mathbf x_q,\mathbf u_q,t_q)\leq\mathbf0,
\quad q\in\{0,\tfrac12,1,\tfrac32,\ldots,N\},\\
&\mathbf h(\mathbf x_0,\mathbf x_N,t_N)=\mathbf0.
\end{aligned}
$$

An alternative is to optimize the midpoint control independently. The three
control values then define a quadratic on each interval; the midpoint-average
relation is removed, and $\mathbf u_{k+\frac12}$ joins the decision vector.
The two state equations and Simpson cost weights remain the same. This
changes the admissible controls, so it can change the optimizer's solution.

The resulting representations can be compared directly. All degrees are
upper bounds on each interval; they are not claims about convergence order.

| Scheme used here | State degree | Control degree | Independent values beyond mesh states |
|---|---:|---:|---|
| Explicit or implicit Euler | 1 | 0 | One control per interval |
| Trapezoidal | 2 | 1 | Shared endpoint controls |
| Hermite--Simpson, linear control | 3 | 1 | Shared endpoint controls and midpoint states |
| Hermite--Simpson, quadratic control | 3 | 2 | Shared endpoint controls, midpoint states, and midpoint controls |

## Constructing a Scheme from Its Nodes

How can a program generate these transcriptions from node choices without
hand-deriving a new set of defect equations each time?

The same construction generates every scheme in the preceding table. Its
inputs are the collocation nodes, the control support nodes, and a choice
about control continuity. The number of collocation nodes determines the
degree of the slope interpolant and hence the state polynomial. The control
support nodes determine the control polynomial independently.

This recipe covers polynomial collocation obtained by interpolating slopes
at distinct nodes and integrating them. It includes Gauss, Radau, and Lobatto
collocation as well as the low-order examples. An arbitrary Runge--Kutta
tableau need not come from such nodes, so specifying nodes does not generate
every possible integration method.

The first recipe uses the integration form. Its differentiation-form
counterpart in @sec-collocation-differentiation-from-nodes also takes
state-support nodes as input, because it stores values of the state
polynomial rather than constructing that polynomial from its slopes.

````{prf:algorithm} Assemble an integration-form collocation NLP
:label: alg-collocation-from-nodes

**Input:** A mesh $t_0<\cdots<t_N$; dynamics $\mathbf f$, running cost $c$,
terminal cost $c_f$, boundary equalities $\mathbf h$, and path inequalities
$\mathbf g$; distinct collocation nodes $\tau_1,\ldots,\tau_s$ and control
support nodes $\rho_0,\ldots,\rho_{d_u}$ in $[0,1]$.
Specify whether controls must be continuous across intervals.

**Output:** A finite decision vector $\mathbf z$, objective $F(\mathbf z)$,
inequalities $G(\mathbf z)\leq\mathbf0$, and equalities $H(\mathbf z)=\mathbf0$.

1. **Compute the rule once.** Construct the cardinal polynomials $\ell_j$
   for the collocation nodes and $\psi_r$ for the control support nodes.
   Integrate and evaluate them to obtain

   $$
   A_{ij}=\int_0^{\tau_i}\ell_j(\tau)\,d\tau,\qquad
   b_j=\int_0^1\ell_j(\tau)\,d\tau,\qquad
   B_{jr}=\psi_r(\tau_j).
   $$

2. **Allocate variables.** Create mesh states $\mathbf x_k$, stage states
   $\mathbf x_{k,j}$, and control support values $\widehat{\mathbf u}_{k,r}$.
   Initialize $F=c_f(\mathbf x_N,t_N)$, append the boundary residual
   $\mathbf h(\mathbf x_0,\mathbf x_N,t_N)$ to $H$, and initialize $G$ empty.

3. **Assemble each interval.** For $k=0,\ldots,N-1$, set
   $h_k=t_{k+1}-t_k$ and compute the expressions

   $$
   \begin{aligned}
   \mathbf u_{k,j}&=\sum_rB_{jr}\widehat{\mathbf u}_{k,r},\\
   \mathbf f_{k,j}&=\mathbf f(\mathbf x_{k,j},\mathbf u_{k,j},t_k+h_k\tau_j),\\
   c_{k,j}&=c(\mathbf x_{k,j},\mathbf u_{k,j},t_k+h_k\tau_j).
   \end{aligned}
   $$

   Add $h_k\sum_jb_jc_{k,j}$ to $F$. Append to $H$ the stage and endpoint
   residuals

   $$
   \mathbf x_{k,i}-\mathbf x_k-h_k\sum_jA_{ij}\mathbf f_{k,j},
   \qquad
   \mathbf x_{k+1}-\mathbf x_k-h_k\sum_jb_j\mathbf f_{k,j}.
   $$

   Append $\mathbf g(\mathbf x_{k,j},\mathbf u_{k,j},t_k+h_k\tau_j)$ to $G$
   at every stage. Append any additional sampled constraints or variable
   bounds required by the model.

4. **Connect controls if required.** Either share endpoint control variables
   or append the equality between the right control value of piece $k$ and
   the left control value of piece $k+1$. Mesh states are already shared.

5. **Solve and reconstruct.** Pass $F,G,H$ and their derivatives to an NLP
   solver. Recover each state piece by integrating its slope interpolant
   from $\mathbf x_k$, and recover each control piece from its support values.
   Check the ODE residual and constraints between nodes; refine the mesh or
   degree where needed and solve again.
````

The stage controls, slopes, and cost rates in step 3 are expressions in the
decision variables. They are recomputed as the optimizer changes those
variables; $A,b,B$ remain fixed. Automatic differentiation can supply the
derivatives of the assembled expressions. If endpoint stages are retained
as separate variables, the stage and endpoint equations connect them to
the mesh states. Eliminating those copies produces the smaller low-order
programs above; keeping both the copies and their defining equations is also
valid. Identifying a copy with its endpoint requires removing the resulting
duplicate or identically zero equation.

(sec-collocation-coefficient-generator)=
### A reusable coefficient generator and residual evaluator

The [full Python source on GitHub](https://github.com/pierrelux/rlbook/blob/main/code/collocation_transcription.py)
contains the coefficient generator and NLP residual evaluator. Its
integration step uses closed-form polynomial calculus, rather than a
numerical quadrature routine that samples a function to estimate its
integral. To see what the code computes, return to the third slope basis
function for nodes $(0,\tfrac12,1)$ in
@fig-collocation-destination-and-slopes:

$$
\ell_3(\tau)=2\tau^2-\tau.
$$

Integrating a term $a\tau^m$ increases its exponent by one and divides its
coefficient by that new exponent. Applying this rule to both terms gives
an antiderivative

$$
L_3(\tau)=\frac23\tau^3-\frac12\tau^2.
$$

Differentiating $L_3$ returns $\ell_3$. We have chosen the integration
constant to be zero; any other constant would cancel when evaluating a
definite integral. The fundamental theorem of calculus now supplies both
the full-interval weight and the midpoint weight:

$$
\begin{aligned}
b_3
&=L_3(1)-L_3(0)=\frac23-\frac12=\frac16,\\
A_{23}
&=L_3(\tfrac12)-L_3(0)=\frac1{12}-\frac18=-\frac1{24}.
\end{aligned}
$$

Only the upper limit changes between these calculations. The negative
midpoint weight agrees with the signed area under $\ell_3$ in the figure.
Thus computing either weight requires an antiderivative and two polynomial
evaluations, without approximating the area by small time steps.

The implementation represents a polynomial by its coefficients in increasing
order of power. For $\ell_3$, the array is $[0,-1,2]$: the entries multiply
$1,\tau,\tau^2$. Integration and differentiation transform this array into
the coefficient arrays of new polynomials:

$$
\begin{aligned}
[0,-1,2]&\xrightarrow{\text{integrate}}
[0,0,-\tfrac12,\tfrac23],\\
[0,-1,2]&\xrightarrow{\text{differentiate}}[-1,4].
\end{aligned}
$$

The first result represents $L_3$; the second represents
$\ell_3'(\tau)=4\tau-1$. More generally, integrating a coefficient array
$[a_0,a_1,\ldots,a_d]$ produces
$[C,a_0,a_1/2,\ldots,a_d/(d+1)]$, where $C$ is the integration constant.
Differentiating it produces $[a_1,2a_2,\ldots,d a_d]$.
These operations require only arithmetic on the stored coefficients, not
a symbolic algebra system. NumPy's
[polynomial integration](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.integ.html)
method implements the antiderivative operation:

```python
from numpy.polynomial import Polynomial

ell3 = Polynomial([0, -1, 2])  # 2*tau**2 - tau
L3 = ell3.integ()             # (2/3)*tau**3 - (1/2)*tau**2
dell3 = ell3.deriv()          # 4*tau - 1

b3 = L3(1.0) - L3(0.0)       # 1/6, up to floating-point error
A23 = L3(0.5) - L3(0.0)      # -1/24, up to floating-point error
```

The generator below applies this procedure to every slope basis function.
It first constructs each cardinal polynomial by multiplying its linear
factors. Calling `.integ()` then constructs its antiderivative;
evaluating that polynomial at the stage nodes and subtracting its value at
zero supplies a column of $A$. Evaluation at $1$ supplies the corresponding
entry of $b$. Direct evaluation of the control basis functions supplies $B$.
The same coefficient differentiation rule, followed by evaluation at the
collocation nodes, constructs $D$ from the state basis in the differentiation
form. It does not estimate a derivative using a small finite-difference step.

These calculations occur once for the chosen node sets. During an NLP
evaluation, the current states and controls supply new slopes
$\mathbf f_{k,j}$, but the coefficients stay fixed. At that point,
"integrating the interpolated slopes" means computing the weighted sums
$h_k\sum_j A_{ij}\mathbf f_{k,j}$ or
$h_k\sum_j b_j\mathbf f_{k,j}$. The polynomial integral is exact in exact
arithmetic; floating-point computation introduces roundoff, and replacing
the nonlinear ODE by a slope interpolant still introduces a separate
discretization error. Closed-form integration of the interpolant does not
give a closed-form solution of the original ODE.

The following function computes $A,b,B$ for any distinct collocation and
control support nodes:

```{literalinclude} code/collocation_transcription.py
:language: python
:start-at: def make_rule
:end-before: def transcribed_problem
```

For example, the choices below recover the four low-order constructions:

```python
from collocation_transcription import make_rule

explicit_euler = make_rule([0.0], [0.0])
implicit_euler = make_rule([1.0], [0.0])
trapezoidal = make_rule([0.0, 1.0], [0.0, 1.0])
hermite_simpson = make_rule([0.0, 0.5, 1.0], [0.0, 1.0])
# An independent midpoint control changes B, not A or b:
hermite_simpson_quadratic_u = make_rule([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
```

The implicit-Euler control support node can be $0$ even though its
collocation node is $1$: a constant polynomial has the same value at both.
For Hermite--Simpson with linear controls, the generated arrays are

$$
A=\begin{bmatrix}
0&0&0\\
5/24&1/3&-1/24\\
1/6&2/3&1/6
\end{bmatrix},\qquad
b=\begin{bmatrix}1/6\\2/3\\1/6\end{bmatrix},\qquad
B=\begin{bmatrix}1&0\\1/2&1/2\\0&1\end{bmatrix}.
$$

The middle row of $A$ gives the midpoint stage equation before elimination;
the middle row of $B$ gives the endpoint-average control. Supplying three
control support nodes instead makes $B$ the identity, allowing an independent
midpoint control while leaving the state construction unchanged.

The same file's [residual evaluator](https://github.com/pierrelux/rlbook/blob/main/code/collocation_transcription.py#L68),
`transcribed_problem`, evaluates
$F,G,H$ from arrays of mesh states, stage states, and control support values.
Its default allows control jumps; `continuous_control=True` adds the
endpoint-matching equations used by the linear-control examples. A solver
wrapper flattens these arrays into $\mathbf z$ and reshapes each candidate
before evaluation. The NumPy evaluator can be used with finite differences;
an automatic-differentiation implementation uses the same array operations
in its chosen backend. The returned inequality residuals use $G\leq0$;
negate them for a solver interface that expects nonnegative residuals.

{download}`Download the coefficient generator and NLP evaluator <code/collocation_transcription.py>`

Explicit polynomial coefficients keep this implementation readable at modest
degrees. At high degrees, use numerically stable basis evaluation and
integration routines; the NLP assembly remains the same.

(sec-collocation-differentiation-from-nodes)=
### Constructing the differentiation form from its nodes

Both recipes retain collocation nodes where the reconstructed state must
satisfy the ODE. The integration-form recipe interpolates evaluated stage
slopes and integrates them; its residuals then match the resulting state
values to the stored stage states. The differentiation form interpolates
stored state-support values and evaluates the stage states on that curve;
its residuals then match the curve's derivatives to the ODE slopes.
Choosing a form therefore changes the representation and residuals, not
whether the scheme has collocation nodes.

For a concrete example, retain the trapezoidal rule's two collocation nodes
$\tau_1=0$ and $\tau_2=1$. Its slope interpolant is linear, so its state
polynomial is quadratic. Storing only the two endpoint states and linearly
interpolating between them would give a constant derivative. Enforcing
that derivative at both endpoints would require the two ODE slopes to be
equal, which the trapezoidal rule does not require. To represent the same
quadratic as the integration form, choose three state-support nodes, for
example $\sigma=(0,\tfrac12,1)$.

The cardinal polynomials for these support nodes are
$2\tau^2-3\tau+1$, $4\tau-4\tau^2$, and $2\tau^2-\tau$.
Evaluating them and their derivatives at the two collocation nodes gives

$$
E=\begin{bmatrix}1&0&0\\0&0&1\end{bmatrix},
\qquad
D=\begin{bmatrix}-3&4&-1\\1&-4&3\end{bmatrix}.
$$

There are three columns because each polynomial operation uses three
support values, but only two rows because the ODE is enforced at two
collocation nodes. Identify the endpoint support values with
$\mathbf x_k$ and $\mathbf x_{k+1}$, and call the midpoint support value
$\widehat{\mathbf x}_{k,1}$. The two differentiation constraints are

$$
\begin{aligned}
-3\mathbf x_k+4\widehat{\mathbf x}_{k,1}-\mathbf x_{k+1}
&=h_k\mathbf f_{k,1},\\
\mathbf x_k-4\widehat{\mathbf x}_{k,1}+3\mathbf x_{k+1}
&=h_k\mathbf f_{k,2}.
\end{aligned}
$$

The slopes on the right are evaluated at the left and right endpoints,
respectively. Adding the equations eliminates the midpoint support value.
Dividing by two recovers the trapezoidal endpoint constraint:

$$
\mathbf x_{k+1}-\mathbf x_k
=\frac{h_k}{2}(\mathbf f_{k,1}+\mathbf f_{k,2}).
$$

Substituting this relation into either differentiation constraint also
determines the midpoint support value:

$$
\widehat{\mathbf x}_{k,1}
=\frac{\mathbf x_k+\mathbf x_{k+1}}{2}
+\frac{h_k}{8}(\mathbf f_{k,1}-\mathbf f_{k,2}).
$$

The midpoint value records the curvature of the quadratic. Storing it
does not add an ODE constraint at the midpoint. A support node specifies
where a polynomial value is stored; a collocation node specifies where
its derivative must match the dynamics.

The same degree count applies to any number of stages. With $s$ distinct
collocation nodes, the integration form constructs a state polynomial
of degree at most $s$. Its differentiation-form counterpart therefore
uses $s+1$ distinct state-support nodes $\sigma_0,\ldots,\sigma_s$.
They need not coincide with the collocation nodes or include the interval
endpoints. Changing these support locations changes the coordinates used
to describe the polynomial, not the polynomial space. Control support
nodes still specify a separate polynomial and can be chosen independently.

````{prf:algorithm} Assemble a differentiation-form collocation NLP
:label: alg-collocation-differentiation-from-nodes

**Input:** A mesh $t_0<\cdots<t_N$; dynamics $\mathbf f$, costs $c,c_f$,
boundary equalities $\mathbf h$, and path inequalities $\mathbf g$;
collocation nodes $\tau_1,\ldots,\tau_s$, state-support nodes
$\sigma_0,\ldots,\sigma_s$, and control-support nodes
$\rho_0,\ldots,\rho_{d_u}$ in $[0,1]$, distinct within each set.
Specify whether controls must be continuous across intervals.

**Output:** A decision vector $\mathbf z$ and functions
$F(\mathbf z)$, $G(\mathbf z)\leq\mathbf0$, and $H(\mathbf z)=\mathbf0$.

1. **Compute the operators once.** Construct the cardinal polynomials
   $\ell_r^{\mathrm{state}}$ for the state-support nodes, $\ell_i$ for the
   collocation nodes, and $\psi_q$ for the control-support nodes. Form

   $$
   \begin{aligned}
   E_{ir}&=\ell_r^{\mathrm{state}}(\tau_i),&
   D_{ir}&=(\ell_r^{\mathrm{state}})'(\tau_i),\\
   e^L_r&=\ell_r^{\mathrm{state}}(0),&
   e^R_r&=\ell_r^{\mathrm{state}}(1),\\
   B_{iq}&=\psi_q(\tau_i),&
   b_i&=\int_0^1\ell_i(\tau)\,d\tau.
   \end{aligned}
   $$

   Here $i=1,\ldots,s$, $r=0,\ldots,s$, and $q=0,\ldots,d_u$.
   The vectors $e^L,e^R$ evaluate the state polynomial at its left and
   right endpoints. Both $E$ and $D$ have shape $s\times(s+1)$.

2. **Allocate variables.** Create shared mesh states $\mathbf x_k$,
   state-support values $\widehat{\mathbf x}_{k,r}$, and control-support
   values $\widehat{\mathbf u}_{k,q}$. Do not allocate separate stage
   states: they will be evaluated from the support values.
   Initialize $F=c_f(\mathbf x_N,t_N)$, append
   $\mathbf h(\mathbf x_0,\mathbf x_N,t_N)$ to $H$, and initialize $G$ empty.

3. **Assemble each interval.** For every $k=0,\ldots,N-1$, set
   $h_k=t_{k+1}-t_k$. At every stage $i=1,\ldots,s$, compute

   $$
   \begin{aligned}
   \mathbf x_{k,i}&=\sum_{r=0}^{s}E_{ir}\widehat{\mathbf x}_{k,r},\\
   \mathbf u_{k,i}&=\sum_{q=0}^{d_u}B_{iq}\widehat{\mathbf u}_{k,q},\\
   \mathbf f_{k,i}&=\mathbf f(\mathbf x_{k,i},\mathbf u_{k,i},t_k+h_k\tau_i),\\
   c_{k,i}&=c(\mathbf x_{k,i},\mathbf u_{k,i},t_k+h_k\tau_i).
   \end{aligned}
   $$

   Append the slope residuals to $H$:

   $$
   \sum_{r=0}^{s}D_{ir}\widehat{\mathbf x}_{k,r}
   -h_k\mathbf f_{k,i},\qquad i=1,\ldots,s.
   $$

   Append the two endpoint residuals to connect the polynomial to the mesh:

   $$
   \sum_{r=0}^{s}e^L_r\widehat{\mathbf x}_{k,r}-\mathbf x_k,
   \qquad
   \sum_{r=0}^{s}e^R_r\widehat{\mathbf x}_{k,r}-\mathbf x_{k+1}.
   $$

   Add $h_k\sum_{i=1}^{s}b_i c_{k,i}$ to $F$. Append
   $\mathbf g(\mathbf x_{k,i},\mathbf u_{k,i},t_k+h_k\tau_i)$ to $G$
   at every stage, together with any additional sampled constraints or
   variable bounds required by the model.

4. **Connect controls if required.** Share endpoint control variables,
   or equate the right endpoint evaluation of each control polynomial
   with the left endpoint evaluation of the next. The shared mesh
   states and the endpoint equations in step 3 already connect the
   state polynomials across intervals.

5. **Solve and reconstruct.** Pass $F,G,H$ and their derivatives to an
   NLP solver. Reconstruct the state directly as
   $\mathbf p_k(\tau)=\sum_{r=0}^{s}\widehat{\mathbf x}_{k,r}
   \ell_r^{\mathrm{state}}(\tau)$, and reconstruct the control from its
   support values. Check between-node residuals and a separate continuous
   replay as in @sec-collocation-validation; refine and solve again if needed.
````

The factor $h_k$ in the slope residual converts the ODE's physical-time
rate to a local-time derivative, as in
@fig-collocation-evaluation-differentiation. The arrays are computed for
$\tau\in[0,1]$, so the same arrays work on intervals of different lengths.
Mesh refinement changes the interval lengths and variables, but does not
require new arrays unless the local node sets change.

This algorithm keeps mesh states and support states as separate variables
and connects them by equations. If a support node is an endpoint, its
value can instead be identified with the corresponding mesh state, as in
the quadratic example. Then omit its endpoint equality: after identification,
that equality would only say $\mathbf x_k-\mathbf x_k=\mathbf0$.
If neither endpoint is a support node, retain both endpoint evaluations.

Differentiating the state does not remove integration from the objective.
The running cost is still sampled at the $s$ collocation nodes and
integrated with the same weights $b$ as in the integration form. Those
weights integrate the collocation-node basis $\ell_i$, not the
state-support basis $\ell_r^{\mathrm{state}}$. For the quadratic example,
the cost contribution is the trapezoidal expression
$h_k(c_{k,1}+c_{k,2})/2$, even though three values represent the state.

#### Equivalence with the integration construction

At a feasible point, the differentiation constraints specify the values
of $\mathbf p_k'$ at all $s$ collocation nodes. Because this derivative
has degree at most $s-1$, its values at those nodes determine it everywhere:

$$
\mathbf p_k'(\tau)=h_k\sum_{j=1}^{s}\mathbf f_{k,j}\ell_j(\tau),
\qquad \forall\tau\in[0,1].
$$

The left endpoint equation supplies $\mathbf p_k(0)=\mathbf x_k$.
Applying the fundamental theorem of calculus therefore gives

$$
\mathbf p_k(\tau)=\mathbf x_k
+h_k\sum_{j=1}^{s}\mathbf f_{k,j}\int_0^\tau\ell_j(\xi)\,d\xi,
\qquad \forall\tau\in[0,1].
$$

Evaluation at $\tau_i$ recovers the integration form's stage equation;
evaluation at $1$, together with the right endpoint equation, recovers
its endpoint defect. No additional integral defect is needed in the
differentiation-form NLP. Conversely, the state polynomial constructed
by the integration form has these nodal derivatives and can be represented
by its values at any $s+1$ distinct state-support nodes.

Thus the two formulations describe the same feasible polynomial
trajectories when they use the same collocation nodes, degree, and control
space. Using the same cost weights and constraint-sampling points also
gives the same discrete objective and path constraints. Their decision
arrays and residuals differ, so their numerical conditioning need not be
identical.

#### A differentiation-form coefficient generator and residual evaluator

Constructing $D$ uses the coefficient differentiation introduced in
@sec-collocation-coefficient-generator. For example, the first state basis
function in the quadratic example is
$\ell_0^{\mathrm{state}}(\tau)=1-3\tau+2\tau^2$. Differentiation transforms
its coefficient array $[1,-3,2]$ into $[-3,4]$. Evaluating the resulting
polynomial $-3+4\tau$ at $(0,1)$ gives $(-3,1)$, the first column of $D$.
NumPy's [polynomial derivative method](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.deriv.html)
performs this coefficient transformation. No small time increment or
finite-difference estimate is involved.

The following generator uses the same `cardinal_polynomials` helper as
`make_rule`. It evaluates each state basis function to form a column of
$E$, and evaluates its derivative to form the corresponding column of $D$.
It integrates the collocation-node basis only to obtain the cost weights:

```{literalinclude} code/collocation_transcription.py
:language: python
:start-at: def make_differentiation_rule
:end-before: def differentiation_problem
```

The four low-order schemes can now be generated in differentiation form:

```python
from collocation_transcription import make_differentiation_rule

explicit_euler = make_differentiation_rule([0], [0, 1], [0])
implicit_euler = make_differentiation_rule([1], [0, 1], [0])
trapezoidal = make_differentiation_rule([0, 1], [0, 0.5, 1], [0, 1])
hermite_simpson = make_differentiation_rule(
    [0, 0.5, 1], [0, 1/3, 2/3, 1], [0, 1]
)
```

The argument order is collocation nodes, state-support nodes, and
control-support nodes. In the Hermite--Simpson call, four state-support
values represent a cubic. The midpoint stage state is evaluated by $E$
from those values, since $1/2$ is not a state-support node in this choice.
Using only the three collocation nodes as state supports would restrict
the state to a quadratic and would not reproduce the same method.

The fundamental theorem of calculus also supplies checks on the fixed
arrays, before any dynamics function or NLP solver is involved. Subtracting
the left endpoint evaluation from each row of $E$ must give the same
operator as integrating the derivative with $A$. Similarly, the difference
of endpoint evaluations must equal full-interval integration with $b$:

```python
import numpy as np
from collocation_transcription import make_rule

rule = hermite_simpson
integral = make_rule(rule.nodes, rule.control_nodes)
np.testing.assert_allclose(
    rule.E - rule.state_left, integral.A @ rule.D, atol=1e-12
)
np.testing.assert_allclose(
    rule.state_right - rule.state_left, rule.b @ rule.D, atol=1e-12
)
```

In the first check, NumPy subtracts `state_left` from every row of `E`.
These identities hold for any degree-$s$ state polynomial, not just one
that satisfies the ODE. The tolerances allow for floating-point roundoff
in constructing the coefficients.

The residual evaluator `differentiation_problem` takes mesh states with
shape $(N+1,n)$, state-support values with shape $(N,s+1,n)$, and
control-support values with shape $(N,d_u+1,m)$. For each candidate
decision vector, `E @ x_support[k]` supplies the stage states and
`D @ x_support[k] - width * slopes` supplies the slope residuals.
The two endpoint evaluations connect the polynomial to the mesh states.
The cost, boundary, path, and optional control-continuity conventions
match `transcribed_problem`.

:::{dropdown} Inspect the complete module and both residual evaluators

```{literalinclude} code/collocation_transcription.py
:language: python
```

:::

The generator's `.deriv()` differentiates a known basis polynomial with
respect to local time $\tau$. An NLP solver also
needs derivatives, but with respect to the decision vector $\mathbf z$:
those differentiate the assembled objective and constraints, including
the dynamics function. Automatic differentiation can compute the latter
in a compatible implementation; it does not replace or change the fixed
matrix $D$.

(sec-collocation-validation)=
### Checking the continuous trajectory

A small NLP residual confirms that the stored values satisfy the finite
constraints. To check the approximation between nodes, first evaluate the
state polynomial and its derivative at additional times. On interval $k$,
their ODE residual is

$$
\mathbf r_k(\tau)=
\frac{1}{h_k}\mathbf p_k'(\tau)
-\mathbf f\!\left(
\mathbf p_k(\tau),\mathbf u_h(t_k+h_k\tau),t_k+h_k\tau
\right).
$$

This compares the slope of the reconstructed curve with the slope prescribed
by the model at the same state, control, and time. The collocation equations
make it vanish at the stages, up to solver tolerance. A large value elsewhere
identifies an interval whose polynomial does not resolve the dynamics.
Evaluate path constraints on this curve as well. This use of the
method's own interpolant for error assessment is described by
{cite:t}`Kelly2017DirectCollocation`.

A separate check fixes the reconstructed control $\mathbf u_h(t)$ and
integrates the original ODE from the prescribed initial state with an
accurate time integrator. Compare this simulated state with $\mathbf x_h(t)$,
and check its terminal conditions and path constraints. The simulation
measures what the chosen control produces in the model; $\mathbf r_k$
measures how well the polynomial itself satisfies that model.

If the two trajectories disagree or the polynomial residual is large,
subdivide the affected intervals or increase their polynomial degree, then
solve again. Compare the objective and trajectories across refinements.
A dense check supplies evidence about resolution but does not prove that
every point satisfies a bound. Tightening an already small NLP tolerance
alone cannot remove the error introduced by a coarse transcription.

(optional-orientation-gauss-radau-and-lobatto-nodes)=
### Gauss--Legendre and Radau collocation

The coefficient generator accepts any distinct nodes, but their placement
affects the accuracy of the resulting integration rule. To see what node
selection can accomplish, retain two slope values per interval, as in the
trapezoidal construction, and allow their locations to move. The slope
interpolant remains linear and its integral remains a quadratic state piece.
The node choice changes where the ODE is enforced and the weights used to
combine its slopes.

For **two-stage Gauss--Legendre collocation**, choose the interior nodes

$$
\tau_1=\frac12-\frac{\sqrt3}{6},\qquad
\tau_2=\frac12+\frac{\sqrt3}{6}.
$$

These are approximately $0.2113$ and $0.7887$. Constructing the two Lagrange
basis functions at these nodes and integrating them with `make_rule` gives

$$
A=\begin{bmatrix}
\tfrac14&\tfrac14-\tfrac{\sqrt3}{6}\\
\tfrac14+\tfrac{\sqrt3}{6}&\tfrac14
\end{bmatrix},\qquad
b=\begin{bmatrix}\tfrac12\\\tfrac12\end{bmatrix}.
$$

The rows of $A$ determine the two interior stage states. The weights $b$
then give the endpoint constraint

$$
\mathbf x_{k+1}-\mathbf x_k
-\frac{h_k}{2}(\mathbf f_{k,1}+\mathbf f_{k,2})=0.
$$

The weights resemble the trapezoidal rule, but the slopes are evaluated at
the two interior stages. This relocation improves quadrature accuracy:
for a scalar integrand $q(\tau)=\tau^2$, averaging endpoint values gives
$1/2$, whereas these nodes give

$$
\frac{\tau_1^2+\tau_2^2}{2}=\frac13
=\int_0^1\tau^2\,d\tau.
$$

The same two-node Gauss rule integrates every polynomial of degree at most
three exactly. This does not make the state piece cubic: its degree is still
at most two. Quadrature exactness describes which integrands a weighted sum
integrates exactly; the state degree describes the curve represented on an
interval.

The name comes from the **Legendre polynomials** $P_s$ on $[-1,1]$, which are
orthogonal to all lower-degree polynomials under integration on that
interval and are normalized by $P_s(1)=1$. For example,
$P_0(\xi)=1$, $P_1(\xi)=\xi$, and $P_2(\xi)=(3\xi^2-1)/2$. The roots of $P_2$ are
$\pm1/\sqrt3$; mapping them to $[0,1]$ by $\tau=(1+\xi)/2$ gives the two
nodes above. With $s$ stages, the same construction uses the $s$ roots of
$P_s$ and gives a quadrature rule exact through degree $2s-1$.
[Gauss--Legendre quadrature](https://dlmf.nist.gov/3.5#v) supplies these
node and exactness properties. The trajectory is still represented in a
Lagrange basis: Legendre polynomials determine the nodes, and the Lagrange
basis functions reconstruct values from those nodes.

For **right Radau collocation**, require the last node to lie at $1$.
With two stages, the nodes are $\tau_1=1/3$ and $\tau_2=1$. Their Lagrange
basis functions are $\ell_1(\tau)=\tfrac32(1-\tau)$ and
$\ell_2(\tau)=\tfrac32\tau-\tfrac12$. Integrating them gives

$$
A=\begin{bmatrix}
\tfrac5{12}&-\tfrac1{12}\\
\tfrac34&\tfrac14
\end{bmatrix},\qquad
b=\begin{bmatrix}\tfrac34\\\tfrac14\end{bmatrix}.
$$

The last row of $A$ equals $b^\top$ because that stage is the right endpoint.
Thus the stage and endpoint equations imply
$\mathbf x_{k,2}=\mathbf x_{k+1}$; these variables can be identified and the
duplicate equation removed. The rule integrates quadratics exactly, since
$\tfrac34(\tfrac13)^2+\tfrac14=\tfrac13$, but it does not integrate every
cubic exactly. It trades some quadrature exactness for including the endpoint.
For $s$ stages, the right Radau nodes are the shifted roots of
$P_s(\xi)-P_{s-1}(\xi)$, including $\xi=1$. With one stage, Gauss--Legendre
recovers implicit midpoint, and right Radau recovers implicit Euler.

CasADi supplies both node families through
[`collocation_points`](https://web.casadi.org/python-api/#collocation_points).
They can be passed directly to the slope-based coefficient generator:

```python
import casadi as ca
from collocation_transcription import make_rule

s = 2
control_nodes = [0.0, 1.0]  # Linear control, chosen separately.
gauss_rule = make_rule(ca.collocation_points(s, "legendre"), control_nodes)
radau_rule = make_rule(ca.collocation_points(s, "radau"), control_nodes)
```

Here $s$ counts slope stages, giving a state degree of at most $s$.
Both calls retain a linear control, with stage values
$\mathbf u_{k,j}=(1-\tau_j)\widehat{\mathbf u}_{k,0}
+\tau_j\widehat{\mathbf u}_{k,1}$. The same `transcribed_problem` function
then assembles the stage constraints, endpoint constraints, and cost using
either rule. No extra node at zero is needed by this generator: the left
state already enters as the integration constant $\mathbf x_k$.
CasADi's [direct-collocation example](https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_collocation.py)
instead interpolates state values, so it includes the left endpoint among
the state support nodes before differentiating that polynomial.

Gauss nodes include neither endpoint, right Radau includes the right one,
and Lobatto includes both, as in the trapezoidal and Hermite--Simpson examples.
For every family, adjacent state pieces must still share an endpoint state
or be connected by an equality constraint. Constraints required at endpoints
must also be imposed there even when those endpoints are not collocation
stages. More accurate quadrature alone does not guarantee that the full
trajectory optimization is well resolved; the control representation, mesh,
and checks between nodes remain part of the construction.

## Worked Example: Moving an Overhead Crane While Limiting Residual Sway

How do the nodal variables and trapezoidal defects behave in a constrained
motion problem whose terminal state must suppress residual oscillation?

An overhead crane moves a trolley while a payload hangs from a cable. A
precomputed trolley command can complete the move and still leave the payload
swinging. The comparison uses three precomputed acceleration commands. The
unshaped baseline moves the trolley without accounting for the payload. A
zero-vibration input shaper modifies that baseline to cancel the nominal
oscillation. Direct collocation instead chooses a command using the nonlinear
payload dynamics and the motion constraints. All three commands are open loop:
they are fixed before the move and do not respond to measurements during
execution.

The state is $\mathbf{x}=(p,v,\theta,\omega)^\top$, where $p$ and $v$ are trolley
position and velocity, and $\theta$ and $\omega$ are payload angle and angular
velocity. The scalar control is the commanded trolley acceleration $u=a$. It enters the nonlinear dynamics
as

$$
\dot p=v,\qquad
\dot v=a,\qquad
\dot\theta=\omega,\qquad
\dot\omega=-\frac{g}{\ell}\sin\theta-\frac{a}{\ell}\cos\theta-\gamma\omega.
$$

Here $g$ is gravitational acceleration, $\ell$ is cable length, and $\gamma$ is the damping coefficient. The first two equations describe trolley motion, while the last two describe a
damped pendulum driven at its suspension point. Positive trolley acceleration
makes the load lag behind, which accounts for the minus sign multiplying $a$.
The model treats the cable as a rigid, massless link and assumes that the
trolley acceleration can be commanded directly.

The trolley and payload start at rest. The goal is to move the trolley $4$ m,
stop it, and leave the payload hanging vertically without swinging. The nominal
cable length is $\ell=1.20$ m, with $g=9.81$ m/s$^2$ and
$\gamma=0.035$ s$^{-1}$. Every command is limited to
$|a|\leq 1.60$ m/s$^2$, and every command is replayed on the same nonlinear
continuous-time plant with a $0.02$ s sampling interval. A second replay
increases the cable length by $10\%$ without redesigning any command. This
second plant tests sensitivity to a simple model mismatch.

### Two open-loop baselines

The unshaped baseline uses a symmetric trapezoidal velocity profile, produced
by acceleration at $0.80$ m/s$^2$ up to $1.00$ m/s, cruising, and symmetric
deceleration. These phases last $1.25$, $2.75$, and $1.25$ s, respectively.
Its acceleration
and deceleration phases excite the payload oscillation because their timing
ignores the pendulum period.

An **input shaper** filters a command into weighted, delayed copies. The delays
are chosen so that vibrations excited by the copies cancel one another. For a
zero-vibration (ZV) shaper, begin by approximating $\sin\theta\approx\theta$
and $\cos\theta\approx1$ near the hanging equilibrium. The payload equation
then becomes

$$
\ddot\theta+2\zeta\omega_n\dot\theta+\omega_n^2\theta=-\frac{a}{\ell},
\qquad
\omega_n=\sqrt{\frac{g}{\ell}},
\qquad
\zeta=\frac{\gamma}{2\omega_n}.
$$

Here $\omega_n$ is the undamped natural frequency and $\zeta$ is the damping
ratio. For the nominal parameters, $\omega_n=2.86$ rad/s and
$\zeta=0.0061$. The shaper splits the baseline command $a_0$ into two copies
separated by half of the damped oscillation period:

$$
a_{\mathrm{ZV}}(t)=A_1a_0(t)+A_2a_0(t-T_d),
\qquad
T_d=\frac{\pi}{\omega_n\sqrt{1-\zeta^2}}\,.
$$

The delay makes the vibration caused by the second copy oppose the vibration
remaining from the first. Accounting for decay over the delay gives
$A_1=1/(1+K)$, $A_2=K/(1+K)$, and
$K=\exp[-\zeta\pi/\sqrt{1-\zeta^2}]$. Here $T_d=1.10$ s and the two weights are
approximately $0.505$ and $0.495$.

### Nodal decision variables and trapezoidal defects

The shaped command finishes after $5.25+T_d\approx6.35$ s. Give the
collocation command the same fixed horizon, and evaluate residual sway over
the following two seconds with zero commanded acceleration.

The third command is found by solving the trajectory-optimization problem.
Divide $[0,5.25+T_d]$ into $N=28$ equal intervals, so
$h=(5.25+T_d)/28$. The NLP decision vector contains the state

$$
\mathbf{x}_k=(p_k,v_k,\theta_k,\omega_k)^\top
$$

and acceleration $a_k$ at every mesh node. These are polynomial values, not
monomial coefficients. Each interval contributes the trapezoidal defect

$$
\mathbf{x}_{k+1}-\mathbf{x}_k
-\frac{h}{2}\left[
\mathbf{f}(\mathbf{x}_k,a_k)+\mathbf{f}(\mathbf{x}_{k+1},a_{k+1})
\right]=\mathbf0.
$$

The objective trades payload motion against acceleration magnitude and rapid
changes in acceleration. The state and acceleration penalties are collected in the nodal
quantity

$$
c_k^{\mathrm{rate}}=6\theta_k^2+0.15\omega_k^2+0.035a_k^2.
$$

The coefficient on $\theta_k^2$ penalizes sway most strongly, while the smaller
coefficients penalize angular velocity and acceleration. The smooth
state-and-control cost uses trapezoidal endpoint weights. The final term
penalizes **acceleration slew**, the rate at which the acceleration command
changes. Because the control is piecewise linear, this rate is constant on
each interval, giving

$$
\mathcal J
=h\left(\frac12c_0^{\mathrm{rate}}+\sum_{k=1}^{N-1}c_k^{\mathrm{rate}}+\frac12c_N^{\mathrm{rate}}\right)
+0.002h\sum_{k=0}^{N-1}
\left(\frac{a_{k+1}-a_k}{h}\right)^2.
$$

The boundary conditions impose $\mathbf{x}_0=(0,0,0,0)^\top$ and $\mathbf{x}_N=(4,0,0,0)^\top$, so both
the trolley and payload finish at rest. The nodal bounds impose
$|a_k|\leq1.60$ m/s$^2$, $|v_k|\leq1.50$ m/s, and
$|\theta_k|\leq15^\circ$. Linear interpolation preserves the acceleration
bound between its endpoint values. The state bounds still require checks
between nodes and on the independently simulated trajectory.

The ZV shaper is tailored to one frequency, so it should leave the least
residual sway when the cable length matches its design model. The collocation
solution uses the full nonlinear nominal model and handles all constraints at
once, but it is also open loop and has no explicit robustness guarantee. The
cable-length test therefore measures sensitivity to one model mismatch; it
does not establish that either method is robust in general.

```{code-cell} python
:tags: [remove-cell]

import sys
sys.path.insert(0, "code")

import pandas as pd
from IPython.display import HTML, display
import matplotlib.pyplot as plt

from crane_control import (
    CraneParameters,
    create_animation as create_crane_animation,
    make_summary_figure as make_crane_summary_figure,
    metrics_table as crane_metrics_table,
    run_comparison as run_crane_comparison,
)

crane_parameters = CraneParameters()
crane_comparison = run_crane_comparison(
    crane_parameters,
    intervals=28,
    sample_period=0.02,
)
```

```{code-cell} python
:tags: [remove-input]
:label: fig-crane-collocation-comparison
:caption: All three commands move the trolley through the same four-metre task on the nonlinear plant. The unshaped command leaves a large oscillation. The ZV shaper nearly cancels the nominal mode, while direct collocation reaches the terminal state with a smoother, lower-effort command. The lower panel replays the unchanged commands after increasing cable length by 10 percent. Hatched bars denote the mismatched plant.

crane_summary_figure = make_crane_summary_figure(crane_comparison)
display(crane_summary_figure)
plt.close(crane_summary_figure)
```

The ZV shaper produces the smallest nominal residual sway because the simulated
plant closely matches the single oscillatory mode used to design it. The
collocation command uses less squared acceleration than either baseline and
keeps residual sway below one degree in both replays. The unshaped command
completes the trolley move but leaves several degrees of oscillation.
Cable-length mismatch increases the ZV residual substantially, while the
collocation command degrades more gradually for this particular perturbation.

The table reports continuous-plant measurements rather than node values from
the nonlinear program. Residual sway is the largest absolute angle after the
common command horizon.

```{code-cell} python
:tags: [remove-input]

crane_table = pd.DataFrame(crane_metrics_table(crane_comparison))
crane_table["residual_sway_deg"] = crane_table["residual_sway_deg"].map(lambda x: f"{x:.3f}")
crane_table["peak_sway_deg"] = crane_table["peak_sway_deg"].map(lambda x: f"{x:.2f}")
crane_table["position_error_mm"] = crane_table["position_error_mm"].map(lambda x: f"{x:.2f}")
crane_table["effort"] = crane_table["effort"].map(lambda x: f"{x:.3f}")
crane_table.rename(
    columns={
        "scenario": "plant",
        "controller": "command",
        "residual_sway_deg": "residual sway (deg)",
        "peak_sway_deg": "peak sway (deg)",
        "position_error_mm": "final position error (mm)",
        "effort": "integral a^2 dt",
    }
)
```

The animation uses the same high-accuracy nonlinear validation trajectories as
the figure. It does not replay the collocation polynomial itself.

```{code-cell} python
:tags: [remove-input]
:label: fig-crane-collocation-animation
:caption: Continuous nonlinear replay of the unshaped, zero-vibration-shaped, and direct-collocation commands. All panels use the same spatial and temporal scales.

crane_animation = create_crane_animation(crane_comparison, frame_stride=5)
crane_html = crane_animation.to_jshtml(fps=25)
plt.close(crane_animation._fig)
display(HTML(crane_html))
```

:::{dropdown} Inspect the direct-transcription implementation

```{literalinclude} code/crane_control.py
:language: python
:start-at: def solve_direct_collocation
:end-before: def _compute_metrics
:linenos:
```

:::

{download}`Download the complete crane experiment <code/crane_control.py>`

The optimization checks algebraic defects, bounds, and endpoint conditions at
the nodes. The separate nonlinear replay checks what happens between those
nodes. A small nodal defect is evidence that the discrete NLP was solved
accurately; it is not, by itself, evidence that the mesh resolves the continuous
dynamics.

The comparison does not establish that collocation always outperforms input
shaping. The ZV calculation is inexpensive and can nearly cancel residual
vibration when a lightly damped mode is accurately known. Direct collocation
becomes useful when several state and actuator constraints must be handled
together. All three commands remain open loop here. Feedback or receding-horizon
replanning would be needed to react to unmeasured disturbances during the move.

## Exercises

(classroom-lagrange-practice)=
### Lagrange interpolation practice

The worked example and first four exercises form a classroom sequence of
about 30--40 minutes. They progress from constructing a polynomial through
prescribed values to integrating a control on one physical time interval.
The final exercise introduces barycentric evaluation as an optional extension.
Each solution is collapsed so that the problems can be attempted first.

#### Worked example: recovering a quadratic

Suppose we know the values of $f(t)=t^2$ only at $t=1,2,3$ and want a
polynomial of degree at most two passing through them. These are physical
coordinates for this example; they need not lie in the normalized interval
$[0,1]$. The nodes and values are

$$
(\sigma_0,\sigma_1,\sigma_2)=(1,2,3),\qquad
(y_0,y_1,y_2)=(1,4,9).
$$

To attach a basis function to $y_0$, put zeros at the other two nodes and
normalize its value at $\sigma_0=1$. Repeat for the remaining nodes:

$$
\begin{aligned}
\ell_0(t)&=\frac{(t-2)(t-3)}{(1-2)(1-3)}
=\frac{(t-2)(t-3)}2,\\
\ell_1(t)&=\frac{(t-1)(t-3)}{(2-1)(2-3)}
=-(t-1)(t-3),\\
\ell_2(t)&=\frac{(t-1)(t-2)}{(3-1)(3-2)}
=\frac{(t-1)(t-2)}2.
\end{aligned}
$$

For example, at $t=2$ the three basis values are $(0,1,0)$, so their
weighted sum returns $y_1=4$. At any other time, evaluate the same functions
and combine them with the same stored values. Expanding gives

$$
\begin{aligned}
p(t)&=\ell_0(t)+4\ell_1(t)+9\ell_2(t)\\
&=\left(\tfrac12t^2-\tfrac52t+3\right)
+\left(-4t^2+16t-12\right)
+\left(\tfrac92t^2-\tfrac{27}2t+9\right)\\
&=t^2.
\end{aligned}
$$

The reconstruction agrees with $f$ everywhere because $f$ itself is a
polynomial of degree two. Three samples of a general function would determine
its quadratic interpolant, but would not determine that function between
samples. Here, for instance, $t^2+(t-1)(t-2)(t-3)$ has the same three sampled
values and differs between them.

````{exercise} A line from two stored values
:label: ex-classroom-lagrange-line

On $[0,1]$, prescribe $p(0)=2$ and $p(1)=-1$ and restrict $p$ to degree
at most one.

1. Construct the two Lagrange basis functions using the product formula.
2. Write $p$ in Lagrange form, then expand it in the monomial basis.
3. Evaluate $p(1/4)$. Which pair of numbers are the coefficients in each basis?
4. If the degree restriction is removed, give a different polynomial with
   the same endpoint values.
````

````{solution} ex-classroom-lagrange-line
:class: dropdown

The basis functions are $\ell_0(\tau)=(\tau-1)/(0-1)=1-\tau$ and
$\ell_1(\tau)=(\tau-0)/(1-0)=\tau$. Combining the stored values gives

$$
p(\tau)=2(1-\tau)-\tau=2-3\tau,\qquad p(1/4)=5/4.
$$

Its Lagrange coefficients are $(2,-1)$, the endpoint values. Its monomial
coefficients are $(2,-3)$, the intercept and slope. The polynomial
$2-3\tau+\tau(\tau-1)$ has the same endpoint values because the added term
vanishes at both endpoints. It is quadratic, so it lies outside the
original degree restriction.
````

````{exercise} A quadratic from three stored values
:label: ex-classroom-lagrange-quadratic

Use the nodes $0,\tfrac12,1$ and values $y_0=1$, $y_m=2$, $y_1=0$,
where $m$ denotes the midpoint. Seek a polynomial of degree at most two.

1. Construct $\ell_0,\ell_m,\ell_1$ in factored form, then expand them.
2. Check their values at all three nodes and verify
   $\ell_0(\tau)+\ell_m(\tau)+\ell_1(\tau)=1$.
3. Form $p(\tau)=y_0\ell_0(\tau)+y_m\ell_m(\tau)+y_1\ell_1(\tau)$.
   Simplify it and evaluate $p(1/4)$.
4. Evaluate all three basis functions at $1/4$. Are these weights all
   nonnegative, as they were for linear interpolation on $[0,1]$?
````

````{solution} ex-classroom-lagrange-quadratic
:class: dropdown

Putting zeros at the other two nodes and normalizing at the selected node gives

$$
\begin{aligned}
\ell_0(\tau)&=2(\tau-\tfrac12)(\tau-1)=2\tau^2-3\tau+1,\\
\ell_m(\tau)&=-4\tau(\tau-1)=4\tau-4\tau^2,\\
\ell_1(\tau)&=2\tau(\tau-\tfrac12)=2\tau^2-\tau.
\end{aligned}
$$

The rows of their values at $0,\tfrac12,1$ are $(1,0,0)$, $(0,1,0)$,
and $(0,0,1)$. Adding the expanded expressions cancels the quadratic and
linear terms and leaves one. The interpolant is

$$
p(\tau)=\ell_0(\tau)+2\ell_m(\tau)
=1+5\tau-6\tau^2,\qquad p(1/4)=15/8.
$$

At $1/4$, the basis values are $(3/8,3/4,-1/8)$. They sum to one but
include a negative weight. The interpolant therefore need not stay within
the range of its nodal values.
````

````{exercise} Changing one value and checking a bound
:label: ex-classroom-lagrange-perturbation

Keep the nodes and polynomial from the preceding exercise.

1. Change only the midpoint value from $2$ to $3$. Write the new polynomial
   $\widetilde p$ using the old $p$ and a single basis function. How much does
   its value at $1/4$ change? What happens at the endpoints?
2. For the original polynomial $p$, the three stored values all satisfy
   $p\leq2$. Find its maximum on $[0,1]$. Does the bound hold everywhere?
3. Explain what these calculations imply when nodal values are decision
   variables representing a state trajectory.
````

````{solution} ex-classroom-lagrange-perturbation
:class: dropdown

Only the coefficient of $\ell_m$ changes, so

$$
\widetilde p(\tau)=p(\tau)+\ell_m(\tau)=1+9\tau-10\tau^2.
$$

The change at $1/4$ is $\ell_m(1/4)=3/4$, giving
$\widetilde p(1/4)=21/8$. The endpoints are unchanged because $\ell_m$
vanishes there.

For the original polynomial, $p'(\tau)=5-12\tau$ vanishes at $5/12$.
Since $p''=-12<0$, this point is its maximum, with

$$
p(5/12)=\frac{49}{24}=2+\frac1{24}>2.
$$

Changing a nodal decision variable changes the curve between nodes through
its basis function. Likewise, bounds on the nodal variables alone do not
ensure bounds on a quadratic state trajectory between nodes. The
transcription needs additional checks there, as described in
[Checking the continuous trajectory](#sec-collocation-validation).
````

````{exercise} Integrating a control into a state trajectory
:label: ex-classroom-lagrange-state-change

On the physical interval $2\leq t\leq4$, consider $\dot x(t)=u(t)$ and
$x(2)=3$. Represent the control by the quadratic through
$u(2)=0$, $u(3)=1$, and $u(4)=0$. Let $\tau=(t-2)/2$.

1. Express $u(2+2\tau)$ using the basis from the three-node exercise.
2. Integrate each basis function on $[0,1]$ to obtain the weights
   $w_0,w_m,w_1$. Use them to compute $x(4)-x(2)$, including the
   physical interval length.
3. Find the whole state piece $p(\tau)=x(2+2\tau)$ and state its degree.
   Check $p'(\tau)/2=u(2+2\tau)$.
4. If $x_k$ and $x_{k+1}$ are stored endpoint states, write the endpoint
   defect constraint for arbitrary control values $u_0,u_m,u_1$ at the
   same three times. What endpoint state is required for the given data?
````

````{solution} ex-classroom-lagrange-state-change
:class: dropdown

The only nonzero control value is at the midpoint, so
$u(2+2\tau)=\ell_m(\tau)=4\tau(1-\tau)$. Integrating the expanded basis
functions gives

$$
\begin{aligned}
w_0&=\tfrac23-\tfrac32+1=\tfrac16,\\
w_m&=2-\tfrac43=\tfrac23,\\
w_1&=\tfrac23-\tfrac12=\tfrac16.
\end{aligned}
$$

Because $dt=2\,d\tau$, the displacement is
$2(w_0\cdot0+w_m\cdot1+w_1\cdot0)=4/3$.
Integrating only as far as the current $\tau$ gives the state piece

$$
p(\tau)=3+2\int_0^\tau4\eta(1-\eta)\,d\eta
=3+4\tau^2-\frac83\tau^3.
$$

It is cubic, with $p'(\tau)/2=4\tau-4\tau^2$, as required. The endpoint
constraint for arbitrary stored values is

$$
x_{k+1}-x_k-2\left(\frac16u_0+\frac23u_m+\frac16u_1\right)=0.
$$

For $x_k=3$ and the prescribed controls, it requires $x_{k+1}=13/3$.
This integration is exact for the chosen quadratic control and the dynamics
$\dot x=u$; no claim of exactness for general nonlinear dynamics is needed.
````

````{exercise} Optional: barycentric evaluation
:label: ex-classroom-lagrange-barycentric

Return to the worked example with nodes $\sigma_j=1,2,3$ and values
$y_j=1,4,9$. A polynomial can be evaluated without expanding its basis
functions by precomputing the **barycentric weights**

$$
\beta_j=\frac{1}{\prod_{m\ne j}(\sigma_j-\sigma_m)}.
$$

These weights describe an evaluation formula; they are distinct from the
integration weights $w_j$ used above. Away from the nodes, the same
interpolating polynomial is given by

$$
p(t)=\frac{\displaystyle\sum_{j=0}^2\frac{\beta_jy_j}{t-\sigma_j}}
{\displaystyle\sum_{j=0}^2\frac{\beta_j}{t-\sigma_j}}.
$$

1. Compute $\beta_0,\beta_1,\beta_2$ and use the formula at $t=3/2$.
2. Multiply all three weights by $2$. Does the result change?
3. What should an implementation return when the query is exactly $t=2$?
   Why should it avoid directly evaluating the displayed quotients there?
````

````{solution} ex-classroom-lagrange-barycentric
:class: dropdown

The products of node differences give $(\beta_0,\beta_1,\beta_2)
=(1/2,-1,1/2)$. At $t=3/2$, the numerator and denominator are

$$
\frac{1/2}{1/2}+\frac{-4}{-1/2}+\frac{9/2}{-3/2}=6,
\qquad
\frac{1/2}{1/2}+\frac{-1}{-1/2}+\frac{1/2}{-3/2}=\frac83.
$$

Their ratio is $9/4$, agreeing with $(3/2)^2$. Multiplying every weight by
the same nonzero number scales both sums equally and leaves their ratio
unchanged. At a query that equals a node, return the stored value directly:
$p(2)=4$. The polynomial is defined there, but the individual quotients
in this evaluation formula divide by zero.
````

### Further exercises

The following exercises revisit polynomial coordinates and connect them to
the complete collocation schemes and trajectory checks developed in the chapter.

````{exercise}
:label: ex-collocation-coordinates

Let $p(\tau)=2-\tau+2\tau^2$ and choose the support nodes $0,\tfrac12,1$.

1. Compute its nodal coordinate vector $\mathbf{y}$.
2. Write the evaluation matrix $V$ for the monomial basis.
3. Recover the monomial coefficient vector from $V\mathbf{a}=\mathbf{y}$.
````

````{solution} ex-collocation-coordinates
:class: dropdown

The nodal values are

$$
\mathbf{y}=
\begin{bmatrix}
p(0)\\p(\tfrac12)\\p(1)
\end{bmatrix}
=
\begin{bmatrix}
2\\2\\3
\end{bmatrix}.
$$

For the monomial basis,

$$
V=
\begin{bmatrix}
1&0&0\\
1&\tfrac12&\tfrac14\\
1&1&1
\end{bmatrix}.
$$

Solving $V\mathbf{a}=\mathbf{y}$ returns $\mathbf{a}=(2,-1,2)^\mathsf T$. The solve changes coordinates; it does not construct a different polynomial.
````

````{exercise}
:label: ex-collocation-interpolation-regression

Classify each problem as interpolation, least-squares regression, or neither.

1. Find a cubic that passes through four distinct exact values.
2. Fit a cubic trend to twenty noisy temperature measurements.
3. Choose unknown nodal states so that an ODE residual equals zero at four collocation nodes.
4. Fit a line through two distinct exact values by minimizing squared error.
````

````{solution} ex-collocation-interpolation-regression
:class: dropdown

Problems 1 and 4 are interpolation problems; the least-squares formulation in problem 4 has a zero-residual interpolating solution. Problem 2 is regression. Problem 3 is neither statistical regression nor interpolation of observations: the unknown nodal values define an interpolating polynomial, and collocation adds equality constraints that select those values.
````

````{exercise}
:label: ex-collocation-operators

For the nodes $0,\tfrac12,1$:

1. Construct the three Lagrange cardinal functions.
2. Evaluate their derivatives at all three nodes to obtain $D$.
3. Integrate them on $[0,1]$ to obtain $w$.
4. Verify that $D$ differentiates $p(\tau)=1+2\tau-\tau^2$ exactly at the nodes.
````

````{solution} ex-collocation-operators
:class: dropdown

The cardinal functions are

$$
\ell_0=2\tau^2-3\tau+1,\qquad
\ell_m=4\tau(1-\tau),\qquad
\ell_1=2\tau^2-\tau.
$$

They give

$$
D=
\begin{bmatrix}
-3&4&-1\\
-1&0&1\\
1&-4&3
\end{bmatrix},
\qquad
w=\frac16
\begin{bmatrix}1\\4\\1\end{bmatrix}.
$$

For $\mathbf{y}=(1,\tfrac74,2)^\mathsf T$, $D\mathbf{y}=(2,1,0)^\mathsf T$, which equals $p'(\tau)=2-2\tau$ at the three nodes.
````

````{exercise}
:label: ex-collocation-euler-trapezoid

Starting from

$$
\dot{\mathbf{x}}_h(t_k+h_k\tau)=\sum_j\mathbf{f}_{k,j}\ell_j(\tau),
$$

derive:

1. explicit Euler from the single slope node $\tau_1=0$;
2. implicit Euler from the single slope node $\tau_1=1$;
3. the trapezoidal defect from the slope nodes $\tau_1=0$ and $\tau_2=1$.

State the degree of the resulting state approximation in each case.
````

````{solution} ex-collocation-euler-trapezoid
:class: dropdown

One node gives the constant derivative interpolant $\mathbf{f}_k$ or $\mathbf{f}_{k+1}$. Integration yields the explicit or implicit Euler defect and a degree-one state. With endpoint nodes, $\dot{\mathbf{x}}_h=(1-\tau)\mathbf{f}_k+\tau \mathbf{f}_{k+1}$. Its integral at $\tau=1$ is $\tfrac12(\mathbf{f}_k+\mathbf{f}_{k+1})$, which gives the trapezoidal defect. The linear derivative integrates to a degree-two state.
````

````{exercise}
:label: ex-collocation-hermite-simpson-degree

Explain why Hermite--Simpson has a cubic state interpolant even though its derivative is represented at only three nodes. Then derive the Simpson endpoint weights and the midpoint relation.
````

````{solution} ex-collocation-hermite-simpson-degree
:class: dropdown

Three distinct slope values define a quadratic derivative interpolant. Integrating that quadratic adds one degree, so the state is cubic. Integrating the cardinal functions over $[0,1]$ gives $(1,4,1)/6$ and hence the Simpson defect. Integrating to $\tau=\tfrac12$ gives $(5,8,-1)/24$; eliminating the midpoint slope with the endpoint defect gives

$$
\mathbf{x}_{k+\frac12}
=\frac12(\mathbf{x}_k+\mathbf{x}_{k+1})
+\frac{h_k}{8}(\mathbf{f}_k-\mathbf{f}_{k+1}).
$$
````

````{exercise} Generate a midpoint transcription
:label: ex-collocation-generated-midpoint

Choose the single collocation node $\tau_1=1/2$ and a constant control on
each interval.

1. Use the cardinal function to compute $A$, $b$, and $B$, then verify the
   arrays with `make_rule([0.5], [0.0])`.
2. Eliminate the stage state from the stage and endpoint equations. State
   the resulting defect and explain why this is an implicit midpoint rule.
3. For $\dot x=u$, minimize $\int_0^1u(t)^2\,dt$ with $x(0)=0$, $x(1)=1$,
   and $|u|\leq2$. Write the complete NLP on the mesh $0,1/2,1$, including
   its decision variables, and solve it.
````

````{solution} ex-collocation-generated-midpoint
:class: dropdown

The cardinal function is $\ell_1=1$, so
$A=[1/2]$, $b=[1]$, and $B=[1]$. The stage equation gives
$\mathbf x_{k,1}=\mathbf x_k+(h_k/2)\mathbf f_{k,1}$, while the endpoint
equation gives $\mathbf x_{k+1}=\mathbf x_k+h_k\mathbf f_{k,1}$. Eliminating
the slope between these equations gives
$\mathbf x_{k,1}=(\mathbf x_k+\mathbf x_{k+1})/2$ and hence

$$
\mathbf x_{k+1}-\mathbf x_k
-h_k\mathbf f\!\left(
\frac{\mathbf x_k+\mathbf x_{k+1}}2,\mathbf u_{k,1},t_k+\frac{h_k}2
\right)=\mathbf0.
$$

The unknown right state appears inside the dynamics evaluation, which makes
the rule implicit. The state polynomial is linear and the control constant.
For the scalar problem with two half-unit intervals, the complete NLP is

$$
\begin{aligned}
\min_{x_0,x_1,x_2,u_{0,1},u_{1,1}}\quad&
\frac12(u_{0,1}^2+u_{1,1}^2)\\
\text{subject to}\quad&
x_{k+1}-x_k-\frac12u_{k,1}=0,\quad k=0,1,\\
&-2\leq u_{k,1}\leq2,\quad k=0,1,\\
&x_0=0,\qquad x_2=1.
\end{aligned}
$$

The defects imply $u_{0,1}+u_{1,1}=2$. Substitution into the objective
gives $(u_{0,1}-1)^2+1$, so both controls are one, $x_1=1/2$, and the
minimum cost is one. This recovers the opening trajectory on a finer mesh.
````

````{exercise} Control variation between sampled stages
:label: ex-collocation-control-sampling

On $[0,1]$, consider $\dot x=u$ with $x(0)=0$ and collocation stages only
at the endpoints. Compare the controls $u(t)=0$ and $u(t)=4t(1-t)$.

1. What control values does the endpoint transcription see? Can its
   trapezoidal defect distinguish the commands?
2. Integrate each control to find the state at $t=1$. Explain why a
   quadratic control can cause a problem when only endpoint stages are used.
3. Propose a change to the control representation or stage nodes that
   distinguishes these two commands.
````

````{solution} ex-collocation-control-sampling
:class: dropdown

Both controls are zero at both endpoints. The trapezoidal defect therefore
predicts $x(1)-x(0)=0$ for either command. Their actual displacements are
zero and

$$
\int_0^1 4t(1-t)\,dt
=4\left(\frac12-\frac13\right)=\frac23.
$$

Endpoint samples cannot detect the quadratic control's interior variation.
Restricting the control to a line makes two zero endpoint values determine
the zero control. Alternatively, a midpoint stage samples the second
control at value one, so the two commands produce different stage slopes.
With all three stages, Simpson weights integrate this quadratic control
exactly.
````

````{exercise}
:label: ex-collocation-between-node-residual

An NLP reports a maximum collocation defect of $10^{-10}$. The reconstructed
polynomial has a large ODE residual halfway between two nodes, and an
independent simulation of the chosen control violates a state bound.

1. Why are these observations compatible?
2. What diagnostic should be computed?
3. What change to the transcription is the natural first response?
````

````{solution} ex-collocation-between-node-residual
:class: dropdown

The NLP defect measures its equality constraints at the selected nodes. A
coarse polynomial can satisfy those constraints while failing to resolve
behavior between them. Evaluate $\mathbf r_k$ and path constraints on the
polynomial at additional times. Separately, integrate the original ODE using
the chosen control and compare the simulated states with the polynomial.
Refine the offending intervals or raise the local degree, then solve and
validate again. Tightening an already small NLP tolerance does not fix
representation error.
````

````{exercise}
:label: ex-collocation-crane-mismatch

Download `code/crane_control.py`. Replay the fixed ZV and collocation commands for cable lengths between $0.9\ell$ and $1.1\ell$, without redesigning either command. Plot residual sway against cable length and explain the curve using the natural frequency $\sqrt{g/\ell}$.
````

## Summary and Outlook

Direct collocation turns an optimization over whole functions into an
optimization over finitely many state and control values. These nodal values
define polynomial segments. Algebraic defect constraints then require each
segment to agree with the ODE at selected points.

Each polynomial has two equivalent descriptions: basis coefficients or values
at nodes,

$$
p(\tau)=\sum_j a_j\phi_j(\tau)
=\sum_j y_j\ell_j(\tau),
\qquad \mathbf y=V\mathbf a.
$$

Direct collocation uses the nodal description because the stored values have an
immediate physical meaning. Lagrange cardinal functions turn them into fixed
operators for computing derivatives, endpoints, and integrals. Although this
construction resembles polynomial fitting, the nodal values are optimization
variables rather than noisy observations, and the ODE residual is an equality
constraint rather than a regression loss.

Both integration and differentiation forms impose collocation at selected
nodes. The integration form interpolates evaluated ODE slopes and uses
state-matching residuals to make their integral agree with the stored
states. The differentiation form interpolates state values and uses
slope-matching residuals to make their derivative agree with the ODE.
With matching node choices, degree, and control space, both describe the
same feasible polynomial trajectories.

Interpolating one left or right slope gives explicit or implicit Euler and a
linear state approximation. Interpolating both endpoint slopes gives the
trapezoidal defect and a quadratic state approximation. Adding the midpoint
slope gives Simpson weights, the Hermite--Simpson midpoint relation, and a
cubic state approximation.

The control degree is chosen separately. Its support values determine the
stage controls through the evaluation matrix $B$. A program can construct
$A$, $b$, and $B$ from the selected nodes, then assemble the same stage,
endpoint, and cost expressions for all these schemes. Changing the nodes or
control representation changes the finite approximation without requiring
a new derivation of those expressions.

Across many intervals, each defect involves only neighboring endpoints and
local stage values. Most entries in the constraint Jacobian are therefore zero,
which allows an NLP solver to exploit a sparse, block-banded structure. The
overhead-crane example also shows why solving the NLP is not the final check:
constraints that hold at the nodes may still be violated between them.
Polynomial residuals assess the local approximation, and independent
simulation checks the trajectory produced by the chosen control.

How can we improve a control sequence when a simulator evaluates its cost
but does not provide derivatives?
[Model Predictive Path Integral Control](model-predictive-path-integral-control.md)
samples candidate control sequences and combines them using cost-dependent
weights. When physical disturbances make the rollout uncertain, several
simulated futures estimate each candidate's expected cost. The following
[receding-horizon control chapter](receding-horizon-control.md) studies how
replanning from measurements supplies feedback to a finite-horizon optimizer.
