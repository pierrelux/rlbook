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

# Trajectory Optimization in Continuous Time

Consider one interval, written in normalized time as

$$
t=t_k+h_k\tau,\qquad 0\leq \tau\leq 1,
$$

and a scalar differential equation

$$
\dot x(t)=f(x(t),u(t),t),\qquad x(t_k)=X_k.
$$

The unknown state and control are functions, but a nonlinear-programming solver accepts only finitely many numbers. We therefore need to choose finite coordinates for those functions and turn the differential equation into algebraic equality constraints. This conversion is the central problem of the chapter.

We will use polynomial values at selected points as the coordinates. The main construction is

$$
\text{nodal values}
\ \longrightarrow\
\text{polynomial trajectory}
\ \longrightarrow\
\text{dynamics and quadrature constraints}
\ \longrightarrow\
\text{finite NLP}.
$$

Euler and trapezoidal transcription will make each part of this construction visible. Hermite--Simpson will then show how the same construction extends to higher order.

:::{admonition} Learning goals
:class: note

After studying this chapter, you should be able to:

1. Explain why continuous-time optimal control requires transcription before it can be sent to an NLP solver.
2. Represent one polynomial either by basis coefficients or by values at distinct nodes.
3. Distinguish exact polynomial interpolation from least-squares regression.
4. Construct differentiation and quadrature operators from Lagrange cardinal functions.
5. Derive explicit Euler, implicit Euler, trapezoidal, and Hermite--Simpson defects from nodal slope values.
6. Identify the actual decision variables and the sparse constraint structure in a direct-collocation implementation.
:::

:::{admonition} Prerequisites
:class: tip

The chapter uses ordinary differential equations, definite integrals, and the basic form of an equality-constrained nonlinear program. The preceding trajectory-optimization chapter, [](trajectories.md), supplies additional context on shooting and sparse simultaneous formulations. [](appendix_ivps.md) reviews the sequential integrators used inside shooting.
:::

## From a Continuous Problem to a Finite NLP

A continuous-time optimal-control problem can be written in Bolza form:

$$
\begin{aligned}
\underset{x(\cdot),u(\cdot),t_f}{\operatorname{minimize}}
\quad&
\Phi(x(t_f),t_f)
+\int_{t_0}^{t_f} L(x(t),u(t),t)\,dt\\
\text{subject to}\quad&
\dot x(t)=f(x(t),u(t),t),\\
&r(x(t_0),x(t_f),t_f)=0,\\
&g(x(t),u(t),t)\leq 0.
\end{aligned}
$$

Here $x(t)\in\mathbb R^{n_x}$ and $u(t)\in\mathbb R^{n_u}$. The terminal term $\Phi$ and running term $L$ may both be present. Setting $L=0$ gives the Mayer special case, while setting $\Phi=0$ gives the Lagrange special case. We do not need three separate derivations because all three lead to the same transcription machinery.

**Transcription** is the umbrella operation that replaces this infinite-dimensional problem by a finite-dimensional optimization problem. Two important strategies make different choices:

| Strategy | Finite decision variables | Treatment of the ODE |
|---|---|---|
| Shooting | Control parameters and, in multiple shooting, selected boundary states | A time integrator advances the state sequentially inside each shooting interval |
| Direct collocation | State and control values at selected nodes | Algebraic defect equations enforce the ODE simultaneously |

Shooting and collocation are therefore distinct transcription strategies. A collocation method may reproduce a familiar integration formula, but it exposes the state values to the NLP rather than hiding every state update inside a simulation.

Let

$$
t_0<t_1<\cdots<t_N=t_f,\qquad h_k=t_{k+1}-t_k,
$$

be a mesh. On each interval we use the normalized coordinate $\tau=(t-t_k)/h_k$. The normalization lets us build differentiation and integration operators once on $[0,1]$ and then scale them by $h_k$.

## One Polynomial, Two Coordinate Systems

The space

$$
\mathcal P_r=\{p:\deg p\leq r\}
$$

has dimension $r+1$. Choosing a basis $\{\phi_0,\ldots,\phi_r\}$ gives coefficient coordinates

$$
p(\tau)=\sum_{j=0}^{r}a_j\phi_j(\tau).
$$

The monomial choice $\phi_j(\tau)=\tau^j$ is familiar, but it is only a coordinate system. The polynomial is the function $p$, not its particular list of coefficients.

Now choose $r+1$ distinct support nodes

$$
\sigma_0,\ldots,\sigma_r\in[0,1],
$$

and record the values

$$
y_i=p(\sigma_i).
$$

With the evaluation matrix

$$
V_{ij}=\phi_j(\sigma_i),
$$

the two coordinate vectors satisfy

$$
y=Va.
$$

Distinct nodes make this map invertible. To see why, suppose $Va=0$. The associated polynomial has $r+1$ distinct roots while its degree is at most $r$, so it must be the zero polynomial. Hence $a=0$, the null space is trivial, and the coordinates determine a unique polynomial. Merely counting $r+1$ equations is not enough; the distinct-node condition is what makes the value conditions independent.

The same result is expressed more directly with the Lagrange cardinal functions

$$
\ell_j(\tau)
=\prod_{\substack{m=0\\m\neq j}}^r
\frac{\tau-\sigma_m}{\sigma_j-\sigma_m}.
$$

They satisfy

$$
\ell_j(\sigma_i)=\delta_{ij},
$$

so the nodal values themselves are the coefficients in this basis:

$$
\boxed{
p(\tau)=\sum_{j=0}^{r}y_j\ell_j(\tau),
\qquad y_j=p(\sigma_j).
}
$$

For example, take $p(\tau)=1+2\tau-\tau^2$ and the nodes $0,\tfrac12,1$. In monomial coordinates,

$$
a=
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
y=Va=
\begin{bmatrix}1\\\tfrac74\\2\end{bmatrix}.
$$

Both vectors describe exactly the same quadratic. Direct collocation uses coordinates like $y$: state values and control values at meaningful points. It does not ask the NLP solver to choose monomial coefficients. Coefficient calculations may still be useful once, when software constructs fixed operators, but those calculations remain outside the NLP.

```{code-cell} python
:tags: [remove-input]
:label: fig-polynomial-coordinate-operators
:caption: The same quadratic can be described by monomial coefficients or by its values at three support nodes. Direct collocation keeps the nodal coordinates. Fixed linear operators map those values to nodal derivatives and an integral; they are precomputed rather than optimized.

import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

with plt.rc_context({"font.size": 8.5, "axes.titlesize": 9}):
    figure, axis = plt.subplots(figsize=(7.0, 2.55))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    def add_box(x, y, width, height, text, facecolor, edgecolor):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012",
            linewidth=1.2,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            linespacing=1.35,
        )

    blue = "#0072B2"
    orange = "#D55E00"
    gray = "#4D4D4D"
    pale_blue = "#E8F2F8"
    pale_orange = "#FBEDE8"
    pale_gray = "#F2F2F2"

    add_box(
        0.02,
        0.52,
        0.34,
        0.36,
        "monomial coordinates\n"
        r"$a=(1,\,2,\,-1)^\mathsf{T}$" "\n"
        r"$p(\tau)=1+2\tau-\tau^2$",
        pale_blue,
        blue,
    )
    add_box(
        0.56,
        0.52,
        0.42,
        0.36,
        "nodal coordinates at "
        r"$0,\frac{1}{2},1$" "\n"
        r"$y=(1,\,\frac{7}{4},\,2)^\mathsf{T}$" "\n"
        r"$p(\tau)=\sum_j y_j\ell_j(\tau)$",
        pale_orange,
        orange,
    )

    axis.add_patch(
        FancyArrowPatch(
            (0.37, 0.70),
            (0.55, 0.70),
            arrowstyle="<->",
            mutation_scale=12,
            linewidth=1.2,
            color=gray,
        )
    )
    axis.text(0.46, 0.77, r"$y=Va$", ha="center", va="center", color=gray)
    axis.text(0.46, 0.62, r"same $p\in\mathcal{P}_2$", ha="center", va="center", color=gray)

    add_box(
        0.48,
        0.08,
        0.23,
        0.25,
        "nodal slopes\n"
        r"$p'(\sigma)=Dy$" "\n"
        r"$D_{ij}=\ell_j'(\sigma_i)$",
        pale_gray,
        gray,
    )
    add_box(
        0.75,
        0.08,
        0.23,
        0.25,
        "integral\n"
        r"$\int_0^1p=w^\mathsf{T}y$" "\n"
        r"$w=\frac{1}{6}(1,4,1)$",
        pale_gray,
        gray,
    )
    axis.add_patch(
        FancyArrowPatch(
            (0.70, 0.51),
            (0.60, 0.34),
            arrowstyle="->",
            mutation_scale=11,
            linewidth=1.1,
            color=orange,
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            (0.83, 0.51),
            (0.86, 0.34),
            arrowstyle="->",
            mutation_scale=11,
            linewidth=1.1,
            color=orange,
        )
    )
    figure.tight_layout(pad=0.25)

display(figure)
plt.close(figure)
```

### Polynomial space, basis, and nodes are different choices

Three decisions are easy to conflate:

- The **polynomial space** $\mathcal P_r$ specifies which functions are available.
- The **basis** specifies coordinates for a member of that space. Monomial and Lagrange bases span the same $\mathcal P_r$.
- The **nodes** specify where values, residuals, or integrals are evaluated.

Changing the basis does not change the exact polynomial space, although it can change numerical conditioning. Changing the nodes changes the interpolation and the operators built from it. Later we will mention nodes derived from Legendre polynomials, but the NLP can still use Lagrange nodal coordinates at those nodes. No recurrence for orthogonal-polynomial coefficients is needed here.

## Interpolation Is Not Regression

Interpolation and polynomial regression both produce polynomials, but they answer different questions.

| | Polynomial interpolation | Least-squares regression |
|---|---|---|
| Input | Exact value conditions | Usually noisy or overdetermined observations |
| Algebraic problem | Satisfy $Aa=y$ exactly | Minimize $\lVert Aa-y\rVert_2^2$ |
| Residual | Zero when the conditions are unisolvent | Generally nonzero |
| Typical purpose | Represent a function from exact nodal data | Estimate a trend or conditional mean |

If $A$ is square and invertible, least squares happens to return the exact interpolant with zero residual. That special overlap does not erase the conceptual distinction.

```{code-cell} python
:tags: [remove-input]
:label: fig-interpolation-versus-regression
:caption: The panels use the same six values. On the left they are treated as six exact conditions for a degree-five interpolant, so every residual is zero. On the right they are treated as six observations for a three-parameter quadratic regression, so the fit trades errors across observations.

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

with plt.rc_context({"font.size": 8.5, "axes.titlesize": 9}):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.45),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    blue = "#0072B2"
    orange = "#D55E00"
    gray = "#4D4D4D"

    axes[0].plot(plot_nodes, interpolated_values, color=blue, linewidth=2)
    axes[0].scatter(
        observation_nodes,
        observation_values,
        color=gray,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    axes[0].set_title("Interpolation: exact conditions")
    axes[0].text(0.04, 0.94, "6 conditions, 6 coefficients\nzero residual", transform=axes[0].transAxes, va="top")

    axes[1].plot(plot_nodes, regression_curve, color=orange, linewidth=2)
    axes[1].vlines(
        observation_nodes,
        regression_at_nodes,
        observation_values,
        color="#999999",
        linewidth=1,
    )
    axes[1].scatter(
        observation_nodes,
        observation_values,
        color=gray,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    axes[1].set_title("Regression: aggregate error")
    axes[1].text(0.04, 0.94, "6 observations, 3 coefficients\nnonzero residuals", transform=axes[1].transAxes, va="top")

    for axis in axes:
        axis.set_xlabel(r"$\tau$")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.18, linewidth=0.6)
    axes[0].set_ylabel("value")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.1, 1.25)

display(figure)
plt.close(figure)
```

Direct collocation is not regression. The state values at the nodes are unknown decision variables, not observed targets. Those values define a polynomial exactly. The optimizer then chooses them so that the ODE residual satisfies equality constraints at selected points. An NLP solver may internally minimize a merit function involving squared constraint violations, but that algorithmic detail does not turn collocation into statistical least squares.

## Fixed Operators from Nodal Values

For the three nodes $0,\tfrac12,1$, the cardinal functions are

$$
\ell_0(\tau)=2\tau^2-3\tau+1,\qquad
\ell_m(\tau)=4\tau(1-\tau),\qquad
\ell_1(\tau)=2\tau^2-\tau.
$$

Differentiating them at the nodes gives

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

Integrating them gives

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

The matrix $D$ differentiates every polynomial in $\mathcal P_2$ exactly, and $w$ integrates every polynomial in that space exactly. Both are fixed once the nodes are chosen.

Three kinds of nodes can appear in a transcription:

| Node role | What it does |
|---|---|
| Support node | Supplies coordinates that define a polynomial |
| Collocation node | Supplies a point where the ODE residual is constrained |
| Quadrature node | Supplies a point used to approximate an integral |

A method often reuses one set of points for two or three roles. That is a design choice, not a definition. For example, a state polynomial can be supported at one set of nodes and differentiated at different collocation nodes.

## From Nodal Slopes to Collocation Constraints

We first derive collocation by representing the state derivative. Choose $s$ collocation nodes $c_1,\ldots,c_s$ on $[0,1]$ and form their Lagrange cardinal functions $\ell_j$. Let

$$
F_{k,j}
=f(X_{k,j},U_{k,j},t_k+h_kc_j)
$$

denote the vector field at node $j$ of interval $k$. Interpolate the physical-time derivative:

$$
\dot x_h(t_k+h_k\tau)
=\sum_{j=1}^{s}F_{k,j}\ell_j(\tau).
$$

Integrating from the left endpoint to a collocation node gives

$$
X_{k,i}
=X_k+h_k\sum_{j=1}^{s}A_{ij}F_{k,j},
\qquad
A_{ij}=\int_0^{c_i}\ell_j(\tau)\,d\tau.
$$

These are the stage equations. Integrating to the right endpoint gives

$$
\boxed{
X_{k+1}
=X_k+h_k\sum_{j=1}^{s}b_jF_{k,j},
\qquad
b_j=\int_0^1\ell_j(\tau)\,d\tau.
}
$$

This endpoint equation is both an integration formula and a defect constraint. If $X_{k+1}$ is shared with the next interval, it enforces state continuity. Otherwise an explicit equality must connect the two interval representations. Merely including an endpoint among the collocation nodes does not create continuity by itself.

When the same nodes and weights are used for the running cost,

$$
J_k
\approx
h_k\sum_{j=1}^{s}
b_jL(X_{k,j},U_{k,j},t_k+h_kc_j).
$$

Different quadrature nodes could be used instead; the state and control polynomials would simply be evaluated there.

After all intervals are assembled, a typical NLP has nodal states and controls

$$
z=\left(X_0,\{X_{k,j},U_{k,j}\}_{k,j},X_N\right)
$$

and the form

$$
\begin{aligned}
\underset{z}{\operatorname{minimize}}\quad&
\Phi(X_N,t_f)+\sum_{k=0}^{N-1}J_k\\
\text{subject to}\quad&
\text{stage equations},\\
&\text{endpoint defects and continuity},\\
&\text{boundary, path, and bound constraints}.
\end{aligned}
$$

Every interval constraint touches only local stage variables and neighboring endpoint states. The Jacobian is therefore sparse and block-banded. The arrays $A$ and $b$ are numerical constants computed before optimization.

### Equivalent differentiation form

Many implementations begin from nodal state values rather than nodal slopes. Let $\sigma_0,\ldots,\sigma_d$ be support nodes for a state polynomial

$$
x_h(t_k+h_k\tau)
=\sum_{r=0}^{d}X_{k,r}L_r(\tau).
$$

At collocation nodes $c_i$, define fixed evaluation and differentiation arrays

$$
E_{ir}=L_r(c_i),
\qquad
D_{ir}=L_r'(c_i).
$$

Because $d/dt=(1/h_k)d/d\tau$, the ODE constraints are

$$
\boxed{
\sum_{r=0}^{d}D_{ir}X_{k,r}
=h_k f\left(
\sum_{r=0}^{d}E_{ir}X_{k,r},
U_{k,i},
t_k+h_kc_i
\right).
}
$$

In matrix shorthand this is $DX_k=h_kF_k$. Endpoint evaluation uses another fixed row,

$$
X_{k+1}=\sum_{r=0}^{d}L_r(1)X_{k,r}.
$$

When support and collocation nodes are reused, $E$ simply selects the corresponding nodal state. The slope-value and state-value forms describe the same polynomial construction; one integrates nodal slopes, while the other differentiates nodal states.

This implementation pattern is standard in direct collocation {cite:p}`Kelly2017DirectCollocation,Andersson2019CasADi`. The official [CasADi direct-collocation example](https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_collocation.py) constructs fixed differentiation, endpoint, and quadrature arrays from Lagrange polynomials, while the NLP variables remain state and control values.

## Low-Order Transcriptions

The general construction contains familiar schemes as small cases.

### One slope value: explicit and implicit Euler

With the single left collocation node $c_1=0$, the only cardinal function is $\ell_1(\tau)=1$. Thus $b_1=1$, and the endpoint defect is

$$
X_{k+1}-X_k-h_k f(X_k,U_k,t_k)=0.
$$

This is explicit Euler. The derivative approximation is constant, so integrating it produces a linear state approximation on the interval.

With the single right collocation node $c_1=1$, the same integration weight gives

$$
X_{k+1}-X_k-h_k
f(X_{k+1},U_{k+1},t_{k+1})=0,
$$

which is implicit Euler. In sequential simulation the second equation requires a nonlinear solve at each step. In direct transcription it is simply one more simultaneous equality constraint.

### Endpoint slope values: trapezoidal transcription

Choose the endpoint collocation nodes $c_0=0$ and $c_1=1$. The cardinal
functions used to interpolate the derivative values are

$$
\ell_0(\tau)=1-\tau,\qquad
\ell_1(\tau)=\tau.
$$

The derivative is linear:

$$
\dot x_h(t_k+h_k\tau)
=(1-\tau)F_k+\tau F_{k+1}.
$$

Integrating gives the continuous state approximation

$$
x_h(t_k+h_k\tau)
=X_k+h_k\left[
\left(\tau-\frac{\tau^2}{2}\right)F_k
+\frac{\tau^2}{2}F_{k+1}
\right].
$$

The state approximation is quadratic, even though an implementation may store only its endpoint states and slopes. At $\tau=1$,

$$
\boxed{
X_{k+1}-X_k
-\frac{h_k}{2}
\left[
f(X_k,U_k,t_k)
+f(X_{k+1},U_{k+1},t_{k+1})
\right]
=0.
}
$$

This is the trapezoidal defect. The same weights approximate the running cost:

$$
J_k
\approx
\frac{h_k}{2}
\left[
L(X_k,U_k,t_k)
+L(X_{k+1},U_{k+1},t_{k+1})
\right].
$$

The degree bookkeeping follows directly from integration: a linear derivative interpolant produces a quadratic state interpolant. This distinction is emphasized in the direct-collocation derivation of {cite:t}`Kelly2017DirectCollocation`.

## A Higher-Order Instance: Hermite--Simpson

Add a midpoint slope value at $c_m=\tfrac12$. The derivative now uses the quadratic cardinal functions

$$
\ell_0(\tau)=2\tau^2-3\tau+1,\qquad
\ell_m(\tau)=4\tau(1-\tau),\qquad
\ell_1(\tau)=2\tau^2-\tau.
$$

Their integrals over the full interval are

$$
b_0=\frac16,\qquad
b_m=\frac46,\qquad
b_1=\frac16.
$$

The endpoint defect is therefore Simpson's formula:

$$
\boxed{
X_{k+1}-X_k
-\frac{h_k}{6}
\left(F_k+4F_{k+\frac12}+F_{k+1}\right)
=0.
}
$$

Integration only to the midpoint gives

$$
X_{k+\frac12}
=X_k+\frac{h_k}{24}
\left(5F_k+8F_{k+\frac12}-F_{k+1}\right).
$$

Combining this equation with the endpoint defect yields the equivalent midpoint relation

$$
\boxed{
X_{k+\frac12}
=\frac{X_k+X_{k+1}}{2}
+\frac{h_k}{8}\left(F_k-F_{k+1}\right).
}
$$

The NLP also imposes

$$
F_{k+\frac12}
=f\left(
X_{k+\frac12},
U_{k+\frac12},
t_k+\frac{h_k}{2}
\right).
$$

A quadratic derivative interpolant integrates to a **cubic state interpolant**. Hermite--Simpson is not based on a quadratic state approximation. Its midpoint relation and Simpson defect are two consequences of the same three-node Lagrange construction.

### Optional orientation: Gauss, Radau, and Lobatto nodes

Higher-order schemes often choose collocation nodes derived from Legendre polynomials:

| Family | Endpoints included |
|---|---|
| Gauss | Neither endpoint |
| Radau | One endpoint |
| Lobatto | Both endpoints |

This vocabulary describes node placement, not the coordinate basis used by the NLP. It also does not determine continuity. Adjacent state polynomials are continuous only when they share an endpoint state or are connected by an equality constraint. Endpoint inclusion can make that linkage convenient, but it does not automatically provide state continuity or slope continuity.

The detailed comparison of node families, convergence rates, and adaptive degree selection is deferred. The operational lesson here is simpler: choose nodes, construct their Lagrange functions, and precompute the resulting differentiation and quadrature operators.

## Overhead-Crane Point-to-Point Motion

An overhead crane moves a trolley while a payload hangs from a cable. A direct trolley command can complete the move and still leave the payload swinging. The oscillation is predictable when the cable length is known, so this example compares two uses of that structure: cancel the known mode with an input shaper, or include the mode in a constrained trajectory optimization.

The state is $\mathbf{x}=(p,v,\theta,\omega)$, where $p$ and $v$ are trolley position and velocity, and $\theta$ and $\omega$ are payload angle and angular velocity. The commanded trolley acceleration $a$ enters the nonlinear dynamics as

$$
\dot p=v,\qquad
\dot v=a,\qquad
\dot\theta=\omega,\qquad
\dot\omega=-\frac{g}{\ell}\sin\theta-\frac{a}{\ell}\cos\theta-c\omega.
$$

Positive trolley acceleration makes the load lag behind, which accounts for the minus sign multiplying $a$. The model treats the cable as a rigid, massless link and assumes that the trolley acceleration can be commanded directly.

The deterministic task starts from rest and moves the trolley $4$ m. The nominal cable length is $\ell=1.20$ m. Every controller is limited to $|a|\leq 1.60$ m/s$^2$, and every command is replayed on the same nonlinear continuous-time plant with a $0.02$ s sampling interval. A second replay increases the cable length by $10\%$ without redesigning any controller.

### Two open-loop baselines

The direct baseline uses a symmetric trapezoidal acceleration profile. Its acceleration and deceleration phases excite the payload mode because their timing ignores the pendulum period.

A zero-vibration shaper first linearizes the payload equation around $\theta=0$:

$$
\ddot\theta+2\zeta\omega_n\dot\theta+\omega_n^2\theta=-\frac{a}{\ell},
\qquad
\omega_n=\sqrt{\frac{g}{\ell}},
\qquad
\zeta=\frac{c}{2\omega_n}.
$$

For the nominal parameters, $\omega_n=2.86$ rad/s and $\zeta=0.0061$. The shaper splits the direct command into two copies separated by half a damped period:

$$
a_{\mathrm{ZV}}(t)=A_1a_0(t)+A_2a_0(t-T_d),
\qquad
T_d=\frac{\pi}{\omega_n\sqrt{1-\zeta^2}},
$$

where $A_1=1/(1+K)$, $A_2=K/(1+K)$, and $K=\exp[-\zeta\pi/\sqrt{1-\zeta^2}]$. Here $T_d=1.10$ s and the two weights are approximately $0.505$ and $0.495$.

### Nodal decision variables and trapezoidal defects

The third command is computed rather than shaped. On $N=28$ intervals with step $h$, the NLP decision vector contains the state

$$
X_k=(p_k,v_k,\theta_k,\omega_k)
$$

and acceleration $a_k$ at every mesh node. These are polynomial values, not monomial coefficients. Each interval contributes the trapezoidal defect

$$
X_{k+1}-X_k
-\frac{h}{2}\left[
f(X_k,a_k)+f(X_{k+1},a_{k+1})
\right]=0.
$$

Define

$$
q_k=6\theta_k^2+0.15\omega_k^2+0.035a_k^2.
$$

The smooth state-and-control cost uses trapezoidal endpoint weights, while the acceleration-slew term integrates the constant slope of a piecewise-linear control:

$$
J
=h\left(\frac12q_0+\sum_{k=1}^{N-1}q_k+\frac12q_N\right)
+0.002h\sum_{k=0}^{N-1}
\left(\frac{a_{k+1}-a_k}{h}\right)^2.
$$

The boundary conditions impose $X_0=(0,0,0,0)$ and $X_N=(4,0,0,0)$. The nodal bounds impose $|a_k|\leq1.60$ m/s$^2$, $|v_k|\leq1.50$ m/s, and $|\theta_k|\leq15^\circ$. Because these bounds are imposed only at nodes, a dense replay remains necessary to check the path between nodes.

Before inspecting the result, predict which method should have the smallest nominal residual sway and which should be least affected by the cable-length change. The ZV shaper is tailored to one frequency. The collocation solution uses the full nonlinear nominal model but is also open loop.

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
:caption: All three commands move the trolley through the same four-metre task on the nonlinear plant. The direct command leaves a large oscillation. The ZV shaper nearly cancels the nominal mode, while direct collocation reaches the terminal state with a smoother, lower-effort command. The lower panel replays the unchanged commands after increasing cable length by 10 percent. Hatched bars denote the mismatched plant.

crane_summary_figure = make_crane_summary_figure(crane_comparison)
display(crane_summary_figure)
plt.close(crane_summary_figure)
```

The ZV shaper produces the smallest nominal residual sway because the simulated plant closely matches the single mode used to design it. This is a favorable case for a direct structural calculation. The collocation command uses less squared acceleration than either baseline and keeps residual sway below one degree in both replays. The direct command completes the trolley move but leaves several degrees of oscillation. Cable-length mismatch increases the ZV residual substantially, while the collocation command degrades more gradually for this particular perturbation.

The table reports continuous-plant measurements rather than node values from the nonlinear program. Residual sway is the largest absolute angle after the common command horizon.

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

The animation uses the same high-accuracy nonlinear validation trajectories as the figure. It does not replay the collocation polynomial itself.

```{code-cell} python
:tags: [remove-input]
:label: fig-crane-collocation-animation
:caption: Continuous nonlinear replay of the direct, zero-vibration-shaped, and direct-collocation commands. All panels use the same spatial and temporal scales.

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

The optimization checks algebraic defects, bounds, and endpoint conditions at the nodes. The separate nonlinear replay checks what happens between those nodes. A small nodal defect is evidence that the discrete NLP was solved accurately; it is not, by itself, evidence that the mesh resolves the continuous dynamics.

The comparison does not establish that collocation always outperforms input shaping. The ZV calculation is direct, inexpensive, and exceptionally effective when a lightly damped mode is accurately known. Direct collocation becomes useful when several state and actuator constraints must be handled together. Both commands remain open loop here. Feedback or receding-horizon replanning would be needed to react to unmeasured disturbances during the move.

## Exercises

````{exercise}
:label: ex-collocation-coordinates

Let $p(\tau)=2-\tau+2\tau^2$ and choose the support nodes $0,\tfrac12,1$.

1. Compute its nodal coordinate vector $y$.
2. Write the evaluation matrix $V$ for the monomial basis.
3. Recover the monomial coefficient vector from $Va=y$.
````

````{solution} ex-collocation-coordinates
:class: dropdown

The nodal values are

$$
y=
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

Solving $Va=y$ returns $a=(2,-1,2)^\mathsf T$. The solve changes coordinates; it does not construct a different polynomial.
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

For $y=(1,\tfrac74,2)^\mathsf T$, $Dy=(2,1,0)^\mathsf T$, which equals $p'(\tau)=2-2\tau$ at the three nodes.
````

````{exercise}
:label: ex-collocation-euler-trapezoid

Starting from

$$
\dot x_h(t_k+h_k\tau)=\sum_jF_{k,j}\ell_j(\tau),
$$

derive:

1. explicit Euler from the single slope node $c=0$;
2. implicit Euler from the single slope node $c=1$;
3. the trapezoidal defect from the slope nodes $c=0,1$.

State the degree of the resulting state approximation in each case.
````

````{solution} ex-collocation-euler-trapezoid
:class: dropdown

One node gives the constant derivative interpolant $F_k$ or $F_{k+1}$. Integration yields the explicit or implicit Euler defect and a degree-one state. With endpoint nodes, $\dot x_h=(1-\tau)F_k+\tau F_{k+1}$. Its integral at $\tau=1$ is $\tfrac12(F_k+F_{k+1})$, which gives the trapezoidal defect. The linear derivative integrates to a degree-two state.
````

````{exercise}
:label: ex-collocation-hermite-simpson-degree

Explain why Hermite--Simpson has a cubic state interpolant even though its derivative is represented at only three nodes. Then derive the Simpson endpoint weights and the midpoint relation.
````

````{solution} ex-collocation-hermite-simpson-degree
:class: dropdown

Three distinct slope values define a quadratic derivative interpolant. Integrating that quadratic adds one degree, so the state is cubic. Integrating the cardinal functions over $[0,1]$ gives $(1,4,1)/6$ and hence the Simpson defect. Integrating to $\tau=\tfrac12$ gives $(5,8,-1)/24$; eliminating the midpoint slope with the endpoint defect gives

$$
X_{k+\frac12}
=\frac12(X_k+X_{k+1})
+\frac{h_k}{8}(F_k-F_{k+1}).
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

Continuous-time transcription begins with a representation problem: functions must become finite coordinate vectors. A polynomial can be written in coefficient coordinates or in nodal coordinates,

$$
p(\tau)=\sum_j a_j\phi_j(\tau)
=\sum_j y_j\ell_j(\tau),
\qquad y=Va.
$$

Direct collocation keeps the nodal state and control values as decision variables. Lagrange cardinal functions turn those values into fixed differentiation, endpoint, and quadrature operators. The ODE residual then becomes an equality constraint, not a regression loss.

Interpolating one left or right slope gives explicit or implicit Euler and a linear state approximation. Interpolating both endpoint slopes gives the trapezoidal defect and a quadratic state approximation. Adding the midpoint slope gives Simpson weights, the Hermite--Simpson midpoint relation, and a cubic state approximation.

Across many intervals, each defect couples only neighboring endpoints and local stages, producing the sparse NLP structure used by practical direct-collocation software. The overhead-crane example also shows the limit of the discrete certificate: nodal feasibility must be followed by a continuous replay to reveal between-node error. The next chapter, [](mpc.md), repeatedly solves finite-horizon trajectory problems from the measured state to construct feedback.
