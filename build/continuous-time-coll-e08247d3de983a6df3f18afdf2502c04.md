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

Direct collocation performs this replacement by storing state and control
values at selected times and interpolating between them with low-degree
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
basic form of an equality-constrained nonlinear program. The preceding
trajectory-optimization chapter, [](discrete-time-optimal-control.md), supplies additional
context on shooting and sparse simultaneous formulations. [](appendix_ivps.md)
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

display(HTML(render_linear_control_area()))
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

How do nodal state and control values, quadrature weights, and defect equations
assemble into one nonlinear program?

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

Here $\mathbf{x}(t)\in\mathbb R^{n}$ is the state, $\mathbf{u}(t)\in\mathbb R^{m}$ is the
control, $c_f$ is a terminal cost, and $c$ is a running cost. The equality
$\mathbf{h}=\mathbf0$ imposes endpoint conditions, while $\mathbf{g}\leq\mathbf0$ represents constraints that
must hold along the path. Setting $c=0$ gives the Mayer special case, while
setting $c_f=0$ gives the Lagrange special case. All three forms use the same
transcription machinery.

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

Let

$$
t_0<t_1<\cdots<t_N=t_f,\qquad h_k=t_{k+1}-t_k,
$$

be a mesh with $k=0,\ldots,N-1$. We write $\mathbf x_k$ and
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

## One Polynomial, Two Coordinate Systems

Should a polynomial trajectory be stored by its coefficients or by its values
at support nodes, and how are the two descriptions related?

The opening example represented a line by its two endpoint values. Higher-order
collocation uses the same idea with more values. The polynomials of degree at
most $r$ form the space

$$
\mathcal P_r=\{p:\deg p\leq r\}
$$

This space, denoted by $\mathcal P_r$, has dimension $r+1$. Choosing a basis
$\{\phi_0,\ldots,\phi_r\}$ gives coefficient coordinates

$$
p(\tau)=\sum_{j=0}^{r}a_j\phi_j(\tau).
$$

The monomial choice $\phi_j(\tau)=\tau^j$ is familiar, but it is only a
coordinate system. The polynomial is the function $p$, not its particular list
of coefficients.

The same polynomial can instead be identified by its values. Choose $r+1$
distinct points

$$
\sigma_0,\ldots,\sigma_r\in[0,1],
$$

and record

$$
y_i=p(\sigma_i).
$$

These points are called **support nodes**: once the degree is restricted to at
most $r$, the $r+1$ stored values determine the whole polynomial. Storing values
is useful for trajectory optimization because a bound on the state at a node
then becomes a bound on a decision variable. To evaluate the dynamics between
nodes, however, we need a formula that reconstructs the polynomial from those
values.

For degree one, take the support nodes $0$ and $1$. A line has the form
$p(\tau)=a+b\tau$. The first endpoint condition gives $a=y_0$, and the second
gives $a+b=y_1$, so $b=y_1-y_0$. Substitution and regrouping give

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
that determines its contribution throughout the interval.

With more nodes, the same construction needs one weight function $\ell_j$
per stored value $y_j$. To recover $y_j$ at its own node without altering the
values at the other nodes, $\ell_j$ must equal one at $\sigma_j$ and zero at
every other support node. A polynomial with those zeros contains the factors
$(\tau-\sigma_m)$ for all $m\ne j$. Dividing their product by its value at
$\sigma_j$ makes the value there equal to one. This constructs the
**Lagrange cardinal function**

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
Between nodes, the weighted sum supplies the intervening values. Changing one
stored value $y_j$ by $\Delta y_j$ changes the curve by
$\Delta y_j\ell_j(\tau)$, so the same function describes how that variable
affects the entire polynomial.

The uniqueness of this reconstruction follows from a basic root-counting
argument. If two polynomials in $\mathcal P_r$ have the same $r+1$ nodal values,
their difference has $r+1$ distinct roots. A nonzero polynomial of degree at
most $r$ cannot have that many roots, so the two polynomials must be identical.

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

Both vectors describe exactly the same quadratic. Direct collocation uses
coordinates like $\mathbf{y}$: state values and control values at meaningful points. It
does not ask the NLP solver to choose monomial coefficients. Software may use
coefficient calculations when it constructs fixed operators, but those
calculations remain outside the NLP.

```{code-cell} python
:tags: [remove-input]
:label: fig-polynomial-coordinate-operators
:caption: One quadratic, two coordinate systems. Evaluating the monomial coefficients $\mathbf{a}$ at the support nodes gives $\mathbf{y}=V\mathbf{a}$. Direct collocation stores the nodal vector $\mathbf{y}$; the fixed operators $D$ and $w$ then return its nodal derivatives and exact integral without adding optimization variables.

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

with plt.rc_context({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}):
    figure = plt.figure(figsize=(7.4, 3.7), facecolor="white")
    canvas = figure.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")

    ink = "#20242A"
    muted = "#5B6470"
    line = "#CAD1D8"
    pale = "#F5F7F9"
    blue = "#0072B2"
    pale_blue = "#E8F3F9"
    orange = "#E69F00"
    pale_orange = "#FFF6DD"

    def add_card(x, y, width, height, facecolor=pale, edgecolor=line, linewidth=0.9):
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=linewidth,
            facecolor=facecolor,
            edgecolor=edgecolor,
            transform=canvas.transAxes,
            clip_on=False,
        )
        canvas.add_patch(patch)
        return patch

    def add_arrow(start, end, color=muted, linewidth=1.1):
        canvas.add_patch(FancyArrowPatch(
            start, end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=linewidth,
            color=color,
            transform=canvas.transAxes,
            clip_on=False,
        ))

    # Monomial coordinates: useful for algebra, but not stored by the NLP.
    add_card(0.02, 0.43, 0.23, 0.48)
    canvas.text(0.045, 0.865, "COEFFICIENT VIEW", color=muted,
                fontsize=7.5, fontweight="bold", transform=canvas.transAxes)
    canvas.text(0.045, 0.80, "monomial basis", color=ink,
                fontsize=10, fontweight="bold", transform=canvas.transAxes)
    canvas.text(0.135, 0.69,
                r"$p(\tau)=a_0+a_1\tau+a_2\tau^2$",
                color=ink, fontsize=10, ha="center", transform=canvas.transAxes)
    canvas.text(0.135, 0.565,
                r"$\mathbf{a}=(1,\;2,\;-1)^{\mathsf{T}}$",
                color=ink, fontsize=13, ha="center", transform=canvas.transAxes)
    canvas.text(0.135, 0.485, "three coefficients", color=muted,
                fontsize=8, ha="center", transform=canvas.transAxes)

    # The concrete polynomial makes the coordinate change visible.
    plot_axis = figure.add_axes([0.31, 0.49, 0.34, 0.34])
    tau = np.linspace(0.0, 1.0, 301)
    values = 1.0 + 2.0 * tau - tau**2
    support_nodes = np.array([0.0, 0.5, 1.0])
    nodal_values = 1.0 + 2.0 * support_nodes - support_nodes**2
    plot_axis.plot(tau, values, color=ink, linewidth=1.8, zorder=2)
    plot_axis.scatter(support_nodes, nodal_values, s=38, color=blue,
                      edgecolor="white", linewidth=1.0, zorder=3)
    plot_axis.set_xlim(-0.04, 1.04)
    plot_axis.set_ylim(0.82, 2.12)
    plot_axis.set_xticks(support_nodes, [r"$0$", r"$\frac{1}{2}$", r"$1$"])
    plot_axis.set_yticks([])
    plot_axis.set_xlabel(r"support node $\sigma_i$", color=muted, labelpad=1)
    plot_axis.set_title(r"one polynomial $p\in\mathcal{P}_2$", color=ink, pad=5)
    plot_axis.spines[["top", "right", "left"]].set_visible(False)
    plot_axis.spines["bottom"].set_color(line)
    plot_axis.tick_params(axis="x", colors=muted, length=3)
    plot_axis.grid(axis="x", color=line, linewidth=0.6, linestyle=(0, (2, 3)))
    for x_value, y_value, label, offset in zip(
        support_nodes,
        nodal_values,
        [r"$y_0=1$", r"$y_m=\frac{7}{4}$", r"$y_1=2$"],
        [(6, 8), (0, -19), (-7, 8)],
    ):
        plot_axis.annotate(
            label, (x_value, y_value), xytext=offset,
            textcoords="offset points", color=blue, fontsize=8.5,
            ha="left" if x_value == 0 else ("right" if x_value == 1 else "center"),
            va="bottom",
        )

    add_arrow((0.255, 0.68), (0.30, 0.68))
    add_arrow((0.66, 0.68), (0.705, 0.68), color=blue)
    canvas.text(0.682, 0.715, r"$\mathbf{y}=V\mathbf{a}$", color=blue,
                fontsize=9, ha="center", transform=canvas.transAxes)

    # Nodal coordinates are the representation exposed to direct collocation.
    add_card(0.72, 0.43, 0.26, 0.48,
             facecolor=pale_blue, edgecolor=blue, linewidth=1.25)
    canvas.text(0.745, 0.865, "NODAL VIEW", color=blue,
                fontsize=7.5, fontweight="bold", transform=canvas.transAxes)
    canvas.text(0.745, 0.80, "values at support nodes", color=ink,
                fontsize=10, fontweight="bold", transform=canvas.transAxes)
    canvas.text(0.85, 0.705,
                r"$\sigma=(0,\;\frac{1}{2},\;1)$",
                color=ink, fontsize=10.5, ha="center", transform=canvas.transAxes)
    canvas.text(0.85, 0.61,
                r"$\mathbf{y}=(1,\;\frac{7}{4},\;2)^{\mathsf{T}}$",
                color=blue, fontsize=13, ha="center", transform=canvas.transAxes)
    canvas.text(0.85, 0.515,
                r"$p(\tau)=\sum_j y_j\ell_j(\tau)$",
                color=ink, fontsize=10, ha="center", transform=canvas.transAxes)
    canvas.text(0.85, 0.455, "variables stored by the NLP", color=blue,
                fontsize=8, fontweight="bold", ha="center",
                transform=canvas.transAxes)

    # Once the nodes are chosen, fixed linear maps reuse the same y.
    add_arrow((0.79, 0.42), (0.475, 0.315), color=blue)
    add_arrow((0.91, 0.42), (0.81, 0.315), color=blue)
    add_card(0.31, 0.09, 0.31, 0.22,
             facecolor=pale_orange, edgecolor=orange)
    add_card(0.66, 0.09, 0.32, 0.22,
             facecolor=pale_orange, edgecolor=orange)
    canvas.text(0.465, 0.255, "DIFFERENTIATE AT THE NODES", color=muted,
                fontsize=7.2, fontweight="bold", ha="center",
                transform=canvas.transAxes)
    canvas.text(0.465, 0.17,
                r"$p'(\sigma_i)=(D\mathbf{y})_i,\qquad D\mathbf{y}=(2,\;1,\;0)^{\mathsf{T}}$",
                color=ink, fontsize=10, ha="center", transform=canvas.transAxes)
    canvas.text(0.82, 0.255, "INTEGRATE THE QUADRATIC", color=muted,
                fontsize=7.2, fontweight="bold", ha="center",
                transform=canvas.transAxes)
    canvas.text(0.82, 0.17,
                r"$w^{\mathsf{T}}\mathbf{y}=\int_0^1 p(\tau)\,d\tau=\frac{5}{3}$",
                color=ink, fontsize=10.5, ha="center", transform=canvas.transAxes)
    canvas.text(0.65, 0.025,
                r"$D$ and $w$ are fixed by the nodes; the optimizer changes only $\mathbf{y}$.",
                color=muted, fontsize=8.2, ha="center", transform=canvas.transAxes)

display(figure)
plt.close(figure)
```

### Polynomial space, basis, and nodes are different choices

The construction separates three decisions that are easy to conflate:

- The **polynomial space** $\mathcal P_r$ specifies which functions are available.
- The **basis** specifies coordinates for a member of that space. Monomial and Lagrange bases span the same $\mathcal P_r$.
- The **nodes** specify where values, residuals, or integrals are evaluated.

Changing the basis does not change the exact polynomial space, although it can
change numerical conditioning. Changing the nodes changes the interpolation
and the operators built from it. The higher-order node families introduced
later can still use Lagrange nodal coordinates in the NLP; choosing those nodes
does not require optimizing orthogonal-polynomial coefficients.

## Polynomial Interpolation and Least-Squares Regression

When nodal values do not determine an exact interpolant, which projection
recovers a polynomial that best matches the available samples?

Interpolation and polynomial regression impose different requirements on the
same data. Six values at six distinct nodes determine one polynomial of degree
at most five that passes through every point. If the values are noisy
observations and the aim is to estimate a quadratic trend, a quadratic will
generally be unable to pass through all six. Least-squares regression then
chooses its three coefficients to minimize the sum of squared discrepancies.
The figure below applies both choices to the same six points. Write $A$ for
the matrix obtained by evaluating the chosen polynomial basis at the supplied
input points. The two algebraic
problems are then compared below.

| | Polynomial interpolation | Least-squares regression |
|---|---|---|
| Input | Exact value conditions | Usually noisy or overdetermined observations |
| Algebraic problem | Satisfy $A\mathbf{a}=\mathbf{y}$ exactly | Minimize $\lVert A\mathbf{a}-\mathbf{y}\rVert_2^2$ |
| Residual | Zero when the value conditions uniquely determine a polynomial | Generally nonzero |
| Typical purpose | Represent a function from exact nodal data | Estimate a trend or conditional mean |

If $A$ is square and invertible, least squares happens to return the exact
interpolant with zero residual. That special overlap does not erase the
conceptual distinction.

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

In direct collocation, the nodal states are unknown decision variables rather
than observations. Each candidate vector of nodal values defines one
interpolating polynomial exactly. The optimizer selects a candidate whose ODE
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
repeating symbolic differentiation or numerical integration. These operations
are exact for the represented polynomial.

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

A transcription may use nodes for three distinct purposes:

| Node role | What it does |
|---|---|
| Support node | Supplies coordinates that define a polynomial |
| Collocation node | Supplies a point where the ODE residual is constrained |
| Quadrature node | Supplies a point used to approximate an integral |

A method often reuses one set of points for two or three roles. That is a design
choice, not a definition. For example, a state polynomial can be supported at
one set of nodes and differentiated at different collocation nodes. The next
section begins with slope values at collocation nodes and uses integration to
recover state values.

## From Nodal Slopes to Collocation Constraints

How do those fixed maps convert differential equations into algebraic
constraints at the chosen collocation nodes?

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

Even when the trajectory is smooth, some portions may change much faster than
others. Separate polynomial pieces let us shorten the mesh intervals near a
rapid change while retaining longer intervals elsewhere. Increasing the
degree of one global polynomial adds flexibility across the whole horizon.
The local representation also gives the optimizer useful structure: each
interval's dynamics constraints involve only its own stage values and
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

Choose $s$ collocation nodes $\tau_1,\ldots,\tau_s$ on $[0,1]$. On interval $k$,
node $\tau_j$ corresponds to the physical time $t_k+h_k\tau_j$. The state and
control values there are $\mathbf{x}_{k,j}$ and $\mathbf{u}_{k,j}$. This local evaluation point,
together with its state and control values, is called a **stage**. At each
stage, the differential equation prescribes the physical-time slope

$$
\mathbf{f}_{k,j}
=\mathbf{f}(\mathbf{x}_{k,j},\mathbf{u}_{k,j},t_k+h_k\tau_j)
$$

for $j=1,\ldots,s$. Let $\ell_j$ be the Lagrange cardinal function associated
with the nodes $\tau_1,\ldots,\tau_s$. Interpolating these slopes gives the
physical-time derivative on this one interval:

$$
\dot{\mathbf{x}}_h(t_k+h_k\tau)
=\sum_{j=1}^{s}\mathbf{f}_{k,j}\ell_j(\tau).
$$

This polynomial agrees with the ODE slope $\mathbf{f}_{k,j}$ at every collocation node
of interval $k$. Integrating from its left endpoint, whose stored state is
$\mathbf{x}_k$, constructs the entire state piece:

$$
\mathbf{p}_k(\tau)
=\mathbf{x}_k+h_k\sum_{j=1}^{s}
\left(\int_0^\tau\ell_j(\eta)\,d\eta\right)\mathbf{f}_{k,j}.
$$

The factor $h_k$ converts integration in normalized time into integration in
physical time: $dt=h_k\,d\tau$. For example, a constant physical slope $\mathbf{f}$
acting from $\tau=0$ to $\tau=\tau_i$ acts for $h_k\tau_i$ units of time and
changes the state by $h_k\tau_i\mathbf{f}$. Without $h_k$, the formula would treat every
physical interval as having unit duration. Equivalently, the chain rule gives
$\mathbf{p}_k'(\tau)=h_k\dot{\mathbf{x}}_h(t_k+h_k\tau)$.

The stored stage state $\mathbf{x}_{k,i}$ must lie on this polynomial at $\tau_i$.
Imposing $\mathbf{x}_{k,i}=\mathbf{p}_k(\tau_i)$ gives the **stage equations**

$$
\mathbf{x}_{k,i}
=\mathbf{x}_k+h_k\sum_{j=1}^{s}A_{ij}\mathbf{f}_{k,j},
\qquad
A_{ij}=\int_0^{\tau_i}\ell_j(\tau)\,d\tau.
$$

Each row of $A$ integrates only as far as one stage within interval $k$.
These are simultaneous constraints on the stage values: the slopes on the
right depend on the state and control variables being chosen. To reach the
end of the interval instead, set $\tau=1$ and require the resulting value
$\mathbf{p}_k(1)$ to equal the stored right endpoint $\mathbf{x}_{k+1}$:

$$
\boxed{
\mathbf{x}_{k+1}
=\mathbf{x}_k+h_k\sum_{j=1}^{s}b_j\mathbf{f}_{k,j},
\qquad
b_j=\int_0^1\ell_j(\tau)\,d\tau.
}
$$

The coefficients $b_j$ integrate the cardinal functions over the full
reference interval. The endpoint equation is a defect constraint: it requires
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

After all intervals are assembled, collect the nodal states and controls into
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

The original path and bound constraints apply at every continuous time. The
finite NLP can impose them only at selected points, usually its support or
collocation nodes. Feasibility at those nodes does not rule out a violation
between them, so a continuous replay or dense residual check must follow the
optimization.

Every interval constraint touches only local stage variables and neighboring
endpoint states. Consequently, most derivatives of the constraints with
respect to the decision variables are zero. With variables ordered by time,
the nonzero blocks lie near the diagonal of the Jacobian, the matrix of
constraint derivatives. NLP solvers can exploit this sparse, block-banded
pattern. The arrays $A$ and $b$ are numerical constants computed before
optimization.

### Equivalent differentiation form

The slope-value construction starts from $\mathbf{f}_{k,j}$ and integrates. Many
implementations take the equivalent route of starting from nodal state values
and differentiating. The integral of the degree-$(s-1)$ slope polynomial has
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

Thus $E$ evaluates the state polynomial and $D$ differentiates it. Because
$d/dt=(1/h_k)d/d\tau$, enforcing the ODE at node $\tau_i$ gives

$$
\boxed{
\sum_{r=0}^{d}D_{ir}\widehat{\mathbf x}_{k,r}
=h_k \mathbf{f}\left(
\sum_{r=0}^{d}E_{ir}\widehat{\mathbf x}_{k,r},
\mathbf{u}_{k,i},
t_k+h_k\tau_i
\right).
}
$$

The left side is the derivative with respect to normalized time, and the factor
$h_k$ on the right converts the physical-time derivative accordingly. In
matrix shorthand, all nodal constraints are $D\mathsf X_k=h_k\mathsf F_k$.
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

Both algebraic forms follow the same sequence. Nodal values define a local
polynomial, fixed arrays differentiate or integrate it, and equality
constraints match the resulting slopes and endpoints to the dynamics. The
choice between slope values and state values changes the implementation, not
the represented collocation method.

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
known left state gives the continuous state approximation

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
The degree bookkeeping follows directly from integration: a linear derivative
interpolant produces a quadratic state interpolant. This distinction is
emphasized in the direct-collocation derivation of
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
enforce the ODE at the midpoint. A quadratic derivative interpolant integrates
to a **cubic state interpolant**, so Hermite--Simpson is not based on a
quadratic state approximation. The name reflects its two ingredients: the
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

````{prf:algorithm} Assemble a polynomial collocation NLP
:label: alg-collocation-from-nodes

**Input:** A mesh $t_0<\cdots<t_N$; dynamics $\mathbf f$, running cost $c$,
terminal cost $c_f$, boundary equalities $\mathbf h$, and path inequalities
$\mathbf g$; distinct collocation nodes $\tau_1,\ldots,\tau_s$ and control
support nodes $\rho_0,\ldots,\rho_{d_u}$ in $[0,1]$.

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

### A reusable coefficient generator and residual evaluator

The following function computes $A,b,B$ by polynomial arithmetic. The same
function accepts any distinct collocation and control support nodes:

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

The complete file also supplies `transcribed_problem`, which evaluates
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

### Optional orientation: Gauss, Radau, and Lobatto nodes

Higher-order schemes often place their nodes at roots of Legendre polynomials
or at roots of closely related equations that include prescribed endpoints.
These placements produce accurate quadrature rules for a given number of
function evaluations. Three names indicate which endpoints are included:

| Family | Endpoints included |
|---|---|
| Gauss | Neither endpoint |
| Radau | One endpoint |
| Lobatto | Both endpoints |

Gauss nodes exclude both endpoints, Radau nodes include one, and Lobatto nodes
include both. This vocabulary describes node placement, not the coordinate
basis used by the NLP. It also does not determine continuity. Adjacent state
polynomials are continuous only when they share an endpoint state or are
connected by an equality constraint. Endpoint inclusion can make that linkage
convenient, but it does not automatically provide state continuity or slope
continuity.

The detailed comparison of node families, convergence rates, and adaptive
degree selection is deferred. Each family enters the present construction in
the same way: its nodes define Lagrange functions, which in turn define fixed
differentiation and quadrature operators.

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
cable length is $\ell=1.20$ m. Every command is limited to
$|a|\leq 1.60$ m/s$^2$, and every command is replayed on the same nonlinear
continuous-time plant with a $0.02$ s sampling interval. A second replay
increases the cable length by $10\%$ without redesigning any command. This
second plant tests sensitivity to a simple model mismatch.

### Two open-loop baselines

The unshaped baseline uses a symmetric trapezoidal velocity profile, produced
by constant acceleration, cruising, and constant deceleration. Its acceleration
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

The third command is found by solving the trajectory-optimization problem. On
$N=28$ intervals with step $h$, the NLP decision vector contains the state

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
$|\theta_k|\leq15^\circ$. Because these bounds are imposed only at nodes, a
dense replay remains necessary to check the path between nodes.

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

````{exercise}
:label: ex-collocation-between-node-residual

An NLP reports a maximum collocation defect of $10^{-10}$, but a dense continuous replay violates a state bound and has a large ODE residual halfway between two nodes.

1. Why are these observations compatible?
2. What diagnostic should be computed?
3. What change to the transcription is the natural first response?
````

````{solution} ex-collocation-between-node-residual
:class: dropdown

The NLP defect measures its equality constraints only at the selected nodes. A coarse polynomial can satisfy those constraints while failing to resolve rapid behavior between them. Evaluate the ODE residual and path constraints on a dense grid, preferably using an independent high-accuracy replay. Refine the offending intervals or raise the local approximation degree, then solve and validate again. Tightening an already small NLP tolerance does not fix representation error.
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

Interpolating one left or right slope gives explicit or implicit Euler and a
linear state approximation. Interpolating both endpoint slopes gives the
trapezoidal defect and a quadratic state approximation. Adding the midpoint
slope gives Simpson weights, the Hermite--Simpson midpoint relation, and a
cubic state approximation.

Across many intervals, each defect involves only neighboring endpoints and
local stage values. Most entries in the constraint Jacobian are therefore zero,
which allows an NLP solver to exploit a sparse, block-banded structure. The
overhead-crane example also shows why solving the NLP is not the final check:
constraints that hold at the nodes may still be violated between them. A dense,
independent simulation should therefore follow the optimization. Can the same
finite-horizon problem respond when that replay reveals a state different from
the prediction? [Receding-horizon control](receding-horizon-control.md) turns
the open-loop transcription into feedback by replanning from each measurement.
