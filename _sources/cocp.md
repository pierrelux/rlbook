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

As in the discrete-time setting, we work with three continuous-time variants that differ only in how the objective is written while sharing the same dynamics, path constraints, and bounds. The path constraints $\mathbf{g}(\mathbf{x}(t),\mathbf{u}(t))\le \mathbf{0}$ are pointwise in time, and the bounds $\mathbf{x}_{\min}\le \mathbf{x}(t)\le \mathbf{x}_{\max}$ and $\mathbf{u}_{\min}\le \mathbf{u}(t)\le \mathbf{u}_{\max}$ are understood in the same pointwise sense.

::::{grid}
:gutter: 1

:::{grid-item}

```{prf:definition} Mayer Problem
$$
\begin{aligned}
    \text{minimize} \quad & c(\mathbf{x}(t_f)) \\
    \text{subject to} \quad & \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) \\
                            & \mathbf{g}(\mathbf{x}(t), \mathbf{u}(t)) \leq \mathbf{0} \\
                            & \mathbf{x}_{\text{min}} \leq \mathbf{x}(t) \leq \mathbf{x}_{\text{max}} \\
                            & \mathbf{u}_{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}_{\text{max}} \\
    \text{given} \quad & \mathbf{x}(t_0) = \mathbf{x}_0 \enspace .
\end{aligned}
$$
```

:::

:::{grid-item}

```{prf:definition} Lagrange Problem
$$
\begin{aligned}
    \text{minimize} \quad & \int_{t_0}^{t_f} c(\mathbf{x}(t), \mathbf{u}(t)) \, dt \\
    \text{subject to} \quad & \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) \\
                            & \mathbf{g}(\mathbf{x}(t), \mathbf{u}(t)) \leq \mathbf{0} \\
                            & \mathbf{x}_{\text{min}} \leq \mathbf{x}(t) \leq \mathbf{x}_{\text{max}} \\
                            & \mathbf{u}_{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}_{\text{max}} \\
    \text{given} \quad & \mathbf{x}(t_0) = \mathbf{x}_0 \enspace .
\end{aligned}
$$
```

:::

:::{grid-item}

```{prf:definition} Bolza Problem
$$
\begin{aligned}
    \text{minimize} \quad & c(\mathbf{x}(t_f)) + \int_{t_0}^{t_f} c(\mathbf{x}(t), \mathbf{u}(t)) \, dt \\
    \text{subject to} \quad & \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) \\
                            & \mathbf{g}(\mathbf{x}(t), \mathbf{u}(t)) \leq \mathbf{0} \\
                            & \mathbf{x}_{\text{min}} \leq \mathbf{x}(t) \leq \mathbf{x}_{\text{max}} \\
                            & \mathbf{u}_{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}_{\text{max}} \\
    \text{given} \quad & \mathbf{x}(t_0) = \mathbf{x}_0 \enspace .
\end{aligned}
$$
```

:::
::::

These three forms are different lenses on the same task. Bolza contains both terminal and running terms. Lagrange can be turned into Mayer by augmenting the state with an accumulator:

$$
\dot{z}(t)=c(\mathbf{x}(t),\mathbf{u}(t)),\quad z(t_0)=0,\quad \text{minimize } z(t_f),
$$

with the original dynamics left unchanged. Mayer is a special case of Bolza with zero running cost. We will use these equivalences freely, since a numerical method cares only about what must be evaluated and where those evaluations are taken.

With this catalog in place, we now pass from functions to finite representations.

# Direct Transcription Methods

The discrete-time problems of the previous chapter already suggested how to proceed: we convert a continuous problem into one over finitely many numbers by deciding where to look at the trajectories and how to interpolate between those looks. We place a mesh $t_0<t_1<\cdots<t_N=t_f$ and, inside each window $[t_k,t_{k+1}]$, select a small set of interior fractions $\{\tau_j\}$ on the reference interval $[0,1]$. The running cost is additive over windows, so we write it as a sum of local integrals, map each window to $[0,1]$, and approximate each local integral by a quadrature rule with nodes $\tau_j$ and weights $w_j$. This produces

$$
\int_{t_0}^{t_f} c(\mathbf{x}(t),\mathbf{u}(t))\,dt
\;\approx\;
\sum_{k=0}^{N-1} h_k \sum_{j=1}^q w_j\, c\!\big(\mathbf{x}(t_k+h_k\tau_j),\,\mathbf{u}(t_k+h_k\tau_j)\big),
$$

with $h_k=t_{k+1}-t_k$. The dynamics are treated in the same way by the fundamental theorem of calculus,

$$
\mathbf{x}(t_{k+1})-\mathbf{x}(t_k)=\int_{t_k}^{t_{k+1}} \mathbf{f}(\mathbf{x}(t),\mathbf{u}(t))\,dt
\;\approx\;
h_k \sum_{j=1}^q b_j\, \mathbf{f}\!\big(\mathbf{x}(t_k+h_k\tau_j),\,\mathbf{u}(t_k+h_k\tau_j)\big),
$$

so the places where we "pay" running cost are the same places where we "account" for state changes. Path constraints and bounds are then enforced at the same interior times. In the infinite-horizon discounted case, the same formulas apply with an extra factor $e^{-\rho(t_k+h_k\tau_j)}$ multiplying the weights in the cost.

The values $\mathbf{x}(t_k+h_k\tau_j)$ and $\mathbf{u}(t_k+h_k\tau_j)$ do not exist a priori. We create them by a finite representation. One option is shooting: parameterize $\mathbf{u}$ on the mesh, integrate the ODE across each window with a chosen numerical step, and read interior values from that step. Another is collocation: represent $\mathbf{x}$ inside each window by a local polynomial and choose its coefficients so that the ODE holds at the interior nodes. Both constructions lead to the same structure: a nonlinear program whose objective is a composite quadrature of the running term (plus any terminal term in the Bolza case) and whose constraints are algebraic relations that encode the ODE and the pointwise inequalities at the selected nodes.

Specific choices recover familiar schemes. If we use the left endpoint as the single interior node, we obtain the forward Euler transcription. If we use both endpoints with equal weights, we obtain the trapezoidal transcription. Higher-order rules arise when we include interior nodes and richer polynomials for $\mathbf{x}$. What matters here is the unifying picture: choose nodes, translate integrals into weighted sums, and couple those evaluations to a finite trajectory representation so that cost and physics are enforced at the same places. This is the organizing idea that will guide the rest of the chapter.



## Discretizing cost and dynamics together

In a continuous-time OCP, integrals appear twice: in the objective, which accumulates running cost over time, and implicitly in the dynamics, since state changes over any interval are the integral of the vector field. To compute, we must approximate both the integrals and the unknown functions $\mathbf{x}(t)$ and $\mathbf{u}(t)$ with finitely many numbers that an optimizer can manipulate.

A natural way to do this is to lay down a finite set of time points (a mesh) over the horizon. You can think of the mesh as a grid we overlay on the "true" trajectories that exist as mathematical objects but are not directly accessible. Our aim is to approximate those trajectories and their integrals using values and simple models tied to the mesh. Using the same mesh for both the cost and the dynamics keeps the representation coherent: we evaluate what we pay and how the state changes at consistent times.

Concretely, we begin by choosing a mesh

$$
t_0<t_1<\cdots<t_N=t_f,\qquad h_k:=t_{k+1}-t_k .
$$

The running cost is additive over disjoint intervals. When the horizon $[t_0,t_f]$ is partitioned by the mesh, additivity (linearity) of the integral gives

$$
\int_{t_0}^{t_f} c(\mathbf{x}(t),\mathbf{u}(t))\,dt
\;=\; \sum_{k=0}^{N-1} \int_{t_k}^{t_{k+1}} c(\mathbf{x}(t),\mathbf{u}(t))\,dt .
$$

This identity is exact: it is just the additivity (linearity) of the Lebesgue/Riemann integral over a partition. No approximation has been made yet. Approximations enter only when we later replace each window integral by a quadrature rule: a finite set of nodes and positive weights prescribing an integral approximation. This sets the table for three important ingredients that we will use throughout the chapter.

First, it turns a global object into **local contributions** that live on each $[t_k,t_{k+1}]$. Numerical integration is most effective when it is composite: we approximate each small interval integral and then sum the results. Doing so controls error uniformly, because the global quadrature error is the accumulation of local errors that shrink with the step size. It also allows non-uniform steps $h_k=t_{k+1}-t_k$, which we will use later for mesh refinement.

Second, the split aligns the cost with the **local dynamics constraints**. On each interval the ODE can be written, by the fundamental theorem of calculus, as

$$
\mathbf{x}(t_{k+1})-\mathbf{x}(t_k)=\int_{t_k}^{t_{k+1}} \mathbf{f}(\mathbf{x}(t),\mathbf{u}(t))\,dt.
$$

When we approximate this integral, we introduce interior evaluation points $t_{k,j}\in[t_k,t_{k+1}]$. Using the **same points** in the cost and in the dynamics ties $\mathbf{x}$ and $\mathbf{u}$ together coherently: the places where we "pay" for running cost are also the places where we enforce the ODE. This avoids a mismatch between where we approximate the objective and where we impose feasibility.

Third, the decomposition yields a nonlinear program with **sparse structure**. Each interval contributes a small block to the objective and constraints that depends only on variables from that interval (and its endpoints). Modern solvers exploit this banded sparsity to scale to long horizons. 

With the split justified, we standardize the approximation. Map each interval to a reference domain via $t=t_k+h_k\tau$ with $\tau\in[0,1]$ and $dt=h_k\,d\tau$. A **quadrature rule on $[0,1]$** is specified by evaluation points $\{\tau_j\}_{j=1}^q \subset [0,1]$ and positive weights $\{w_j\}_{j=1}^q$ such that, for a smooth $\phi$,

$$
\int_0^1 \phi(\tau)\,d\tau \;\approx\; \sum_{j=1}^q w_j\,\phi(\tau_j).
$$ 

Applying it on each interval gives

$$
\int_{t_k}^{t_{k+1}} c(\mathbf{x}(t),\mathbf{u}(t))\,dt
\;\approx\;
h_k\sum_{j=1}^q w_j\, c\!\big(\mathbf{x}(t_k+h_k\tau_j),\,\mathbf{u}(t_k+h_k\tau_j)\big).
$$


Summing these window contributions gives a composite approximation of the integral over $[t_0,t_f]$:

$$
\int_{t_0}^{t_f} c(\mathbf{x}(t),\mathbf{u}(t))\,dt
\;\approx\;
\sum_{k=0}^{N-1} h_k \sum_{j=1}^q w_j\, c\!\big(\mathbf{x}(t_{k,j}),\mathbf{u}(t_{k,j})\big).
$$

The outer index $k$ indicates the window; the inner index $i$ indicates the samples within that window; the factor $h_k$ appears from the change of variables.

The dynamics admit the same treatment. By the fundamental theorem of calculus,

$$
\mathbf{x}(t_{k+1})-\mathbf{x}(t_k)
=\int_{t_k}^{t_{k+1}} \dot{\mathbf{x}}(t)\,dt
=\int_{t_k}^{t_{k+1}} \mathbf{f}(\mathbf{x}(t),\mathbf{u}(t))\,dt .
$$

Replacing this integral by a quadrature rule that uses the **same** nodes produces the window defect relation

$$
\mathbf{x}_{k+1}-\mathbf{x}_k
\;\approx\;
h_k\sum_{j=1}^q b_j\, \mathbf{f}\!\big(\mathbf{x}(t_{k,j}),\mathbf{u}(t_{k,j})\big),
$$

where $\{b_j\}$ are the weights used for the ODE. Path constraints $\mathbf{g}(\mathbf{x}(t),\mathbf{u}(t))\le 0$ are imposed at selected nodes $t_{k,j}$ in the same spirit. Using the same evaluation points for cost and dynamics keeps the representation coherent: we "pay" running cost and "account" for state changes at the same times.



### On the choice of interior points

Once we select a mesh $t_0<\cdots<t_N$ and interior fractions $\{\tau_j\}_{j=1}^q$ per window $[t_k,t_{k+1}]$, we need $\mathbf{x}(t_{k,j})$ and $\mathbf{u}(t_{k,j})$ at the evaluation times $t_{k,j} := t_k + h_k\tau_j$. These values do not preexist. They come from one of two constructions that align with the standard quadrature taxonomy: **step-function based** and **interpolating-function based** rules.

#### Step-function based construction (piecewise constants; rectangle or midpoint)

Here we approximate the relevant time functions by step functions on each window. For controls, a common choice is piecewise-constant:

$$
\mathbf{u}(t)=\mathbf{u}_k\quad\text{for }t\in[t_k,t_{k+1}].
$$

For the running cost and the vector field, the corresponding quadrature is a rectangle rule on $[t_k,t_{k+1}]$. Using the left endpoint gives

$$
\int_{t_k}^{t_{k+1}} c(\mathbf{x}(t),\mathbf{u}(t))\,dt
\;\approx\; h_k\,c(\mathbf{x}_k,\mathbf{u}_k),
$$

and replacing the dynamics integral by the same step-function idea yields the forward Euler relation

$$
\mathbf{x}_{k+1}=\mathbf{x}_k+h_k\,\mathbf{f}(\mathbf{x}_k,\mathbf{u}_k,t_k).
$$

If we prefer the midpoint rectangle rule, we sample at $t_{k+\frac12}=t_k+\tfrac{h_k}{2}$. In practice we then generate $\mathbf{x}_{k+\frac12}$ by a half-step of the chosen integrator, and set $\mathbf{u}_{k+\frac12}=\mathbf{u}_k$ (piecewise-constant) or an average if we allow a short linear segment. Either way, interior values come from **integrating forward** given a step-function model for $\mathbf{u}$ and a rectangle-rule view of the integrals. This is the shooting viewpoint. Single shooting keeps only control parameters as decision variables; multiple shooting adds the window-start states and enforces step consistency.

#### Interpolating-function based construction (low-order polynomials; trapezoid, Simpson, Gauss/Radau/Lobatto)

Here we approximate time functions by **polynomials** on each window. If we interpolate $\mathbf{x}(t)$ linearly between endpoints, the cost naturally uses the trapezoidal rule

$$
\int_{t_k}^{t_{k+1}} c\,dt\;\approx\;\tfrac{h_k}{2}\big[c(\mathbf{x}_k,\mathbf{u}_k)+c(\mathbf{x}_{k+1},\mathbf{u}_{k+1})\big],
$$

and the dynamics use the matched trapezoidal defect

$$
\mathbf{x}_{k+1}=\mathbf{x}_k+\tfrac{h_k}{2}\Big[\mathbf{f}(\mathbf{x}_k,\mathbf{u}_k,t_k)+\mathbf{f}(\mathbf{x}_{k+1},\mathbf{u}_{k+1},t_{k+1})\Big].
$$

With a quadratic interpolation that includes the midpoint, Simpson's rule appears in the cost and the Hermite–Simpson relations tie $\mathbf{x}_{k+\frac12}$ to endpoint values and slopes. More generally, **collocation** chooses interior nodes on $[t_k,t_{k+1}]$ (equally spaced gives Newton–Cotes like trapezoid or Simpson; Gaussian points give Gauss, Radau, or Lobatto schemes) and enforces the ODE at those nodes:

$$
\frac{d}{dt}\mathbf{x}(t_{k,j})=\mathbf{f}\!\big(\mathbf{x}(t_{k,j}),\mathbf{u}(t_{k,j}),t_{k,j}\big),
$$

with continuity at endpoints. The interior values $\mathbf{x}(t_{k,j})$ are **evaluations of the decision polynomials**; $\mathbf{u}(t_{k,j})$ follows from the chosen control interpolation (constant, linear, or quadratic). The running cost is evaluated by the same interpolatory quadrature at the same nodes, which keeps "where we pay" aligned with "where we enforce."

# Polynomial Interpolation

We often want to construct a function that passes through a given set of points. For example, suppose we know a function should satisfy:

$$
f(x_0) = y_0, \quad f(x_1) = y_1, \quad \dots, \quad f(x_m) = y_m.
$$

These are called **interpolation constraints**. Our goal is to find a function $f(x)$ that satisfies all of them exactly.

To make the problem tractable, we restrict ourselves to a class of functions. In polynomial interpolation, we assume that $f(x)$ is a polynomial of degree at most $N$. That means we are trying to find coefficients $c_0, \dots, c_N$ such that

$$
f(x) = \sum_{n=0}^N c_n \, \phi_n(x),
$$

where the functions $\phi_n(x)$ form a basis for the space of polynomials. The most common choice is the **monomial basis**, where $\phi_n(x) = x^n$. This gives:

$$
f(x) = c_0 + c_1 x + c_2 x^2 + \dots + c_N x^N.
$$

Other valid bases include Legendre, Chebyshev, and Lagrange polynomials, each chosen for specific numerical properties. But all span the same function space.


To find a unique solution, we need the number of unknowns (the $c_n$) to match the number of constraints. Since a degree-$N$ polynomial has $N+1$ coefficients, we need:

$$
N + 1 = m + 1 \quad \Rightarrow \quad N = m.
$$

So if we want a function that passes through 4 points, we need a cubic polynomial ($N = 3$). Choosing a higher degree than necessary would give us infinitely many solutions; a lower degree may make the problem unsolvable.

### Solving for the Coefficients (Monomial Basis)

If we fix the basis functions to be monomials, we can build a system of equations by plugging in each $x_i$ into $f(x)$. This gives:

$$
\begin{aligned}
f(x_0) &= c_0 + c_1 x_0 + c_2 x_0^2 + \dots + c_N x_0^N = y_0 \\
f(x_1) &= c_0 + c_1 x_1 + c_2 x_1^2 + \dots + c_N x_1^N = y_1 \\
&\vdots \\
f(x_m) &= c_0 + c_1 x_m + c_2 x_m^2 + \dots + c_N x_m^N = y_m
\end{aligned}
$$

This system can be written in matrix form as:

$$
\begin{bmatrix}
1 & x_0 & x_0^2 & \cdots & x_0^N \\
1 & x_1 & x_1^2 & \cdots & x_1^N \\
\vdots & \vdots & \vdots & & \vdots \\
1 & x_m & x_m^2 & \cdots & x_m^N
\end{bmatrix}
\begin{bmatrix}
c_0 \\ c_1 \\ \vdots \\ c_N
\end{bmatrix}
=
\begin{bmatrix}
y_0 \\ y_1 \\ \vdots \\ y_m
\end{bmatrix}
$$

The matrix on the left is called the **Vandermonde matrix**. Solving this system gives the coefficients $c_n$ that define the interpolating polynomial.

### Using a Different Basis

We don't have to use monomials. We can pick any set of basis functions $\phi_n(x)$, such as Chebyshev or Fourier modes, and follow the same steps. The interpolating function becomes:

$$
f(x) = \sum_{n=0}^N c_n \, \phi_n(x),
$$

and each interpolation constraint becomes:

$$
f(x_i) = \sum_{n=0}^N c_n \, \phi_n(x_i) = y_i.
$$

Assembling these into a system gives:

$$
\begin{bmatrix}
\phi_0(x_0) & \phi_1(x_0) & \dots & \phi_N(x_0) \\
\phi_0(x_1) & \phi_1(x_1) & \dots & \phi_N(x_1) \\
\vdots & \vdots & & \vdots \\
\phi_0(x_m) & \phi_1(x_m) & \dots & \phi_N(x_m)
\end{bmatrix}
\begin{bmatrix}
c_0 \\ c_1 \\ \vdots \\ c_N
\end{bmatrix}
=
\begin{bmatrix}
y_0 \\ y_1 \\ \vdots \\ y_m
\end{bmatrix}
$$

From here, we solve as before and reconstruct $f(x)$.

### Derivative Constraints

Sometimes, instead of a value constraint $f(x_i) = y_i$, we want to impose a slope constraint $f'(x_i) = s_i$. This is common in applications like spline interpolation or collocation methods, where derivative information is available from an ODE.

Since

$$
f(x) = \sum_{n=0}^N c_n \phi_n(x) \quad \Rightarrow \quad f'(x) = \sum_{n=0}^N c_n \phi_n'(x),
$$

we can directly write the slope constraint:

$$
f'(x_i) = \sum_{n=0}^N c_n \phi_n'(x_i) = s_i.
$$

To enforce this, we replace one of the interpolation equations in our system with this slope constraint. The resulting system still has $m+1$ equations and $N+1 = m+1$ unknowns.

Concretely, suppose we have $k+1$ value constraints at nodes $X=\{x_0,\ldots,x_k\}$ with values $Y=\{y_0,\ldots,y_k\}$ and $r$ slope constraints at nodes $Z=\{z_1,\ldots,z_r\}$ with slopes $S=\{s_1,\ldots,s_r\}$, with $k+1+r=N+1$. The linear system for the coefficients $\mathbf{c}=[c_0,\ldots,c_N]^\top$ is

$$
\begin{bmatrix}
\phi_0(x_0) & \phi_1(x_0) & \cdots & \phi_N(x_0) \\
\vdots & \vdots & & \vdots \\
\phi_0(x_k) & \phi_1(x_k) & \cdots & \phi_N(x_k) \\
\phi_0'(z_1) & \phi_1'(z_1) & \cdots & \phi_N'(z_1) \\
\vdots & \vdots & & \vdots \\
\phi_0'(z_r) & \phi_1'(z_r) & \cdots & \phi_N'(z_r)
\end{bmatrix}
\begin{bmatrix}
c_0 \\ c_1 \\ \vdots \\ c_N
\end{bmatrix}
=
\begin{bmatrix}
y_0 \\ \vdots \\ y_k \\ s_1 \\ \vdots \\ s_r
\end{bmatrix}.
$$

If a value and a slope are imposed at the same node, take $z_j=x_i$ and include both the value row and the derivative row; the system remains square. In the monomial basis, the top block is the Vandermonde matrix and the derivative block has entries $n\,x^{\,n-1}$. Once solved for $\mathbf{c}$, reconstruct

$$
f(x)=\sum_{n=0}^N c_n\,\phi_n(x).
$$

### Interpolating ODE Trajectories (Collocation)

Having established the general framework of interpolation, we now apply these concepts to the specific context of approximating trajectories governed by ordinary differential equations.  The idea of applying polynomial interpolation with derivative constraints yields a method known as "collocation". More precisely, a **degree-$s$ collocation method** is a way to discretize an ordinary differential equation (ODE) by approximating the solution on each time interval with a polynomial of degree $s$, and then enforcing that this polynomial satisfies the ODE exactly at $s$ carefully chosen points (the *collocation nodes*).

Consider a dynamical system described by the ordinary differential equation:

$$
\dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t),\mathbf{u}(t),t)
$$

where $\mathbf{x}(t)$ represents the state trajectory and $\mathbf{u}(t)$ denotes the control input. 
Let us focus on a single mesh interval $[t_k,t_{k+1}]$ with step size $h_k := t_{k+1}-t_k$. To work with a standardized domain, we introduce the transformation $t = t_k + h_k\tau$ that maps the physical interval $[t_k,t_{k+1}]$ to the reference interval $[0,1]$. On this reference interval, we select a set of collocation nodes $\{\tau_j\}_{j=0}^K \subset [0,1]$.

Our goal is now to approximate the unknown trajectory using a polynomial of degree $K$. Using a monomial basis, we represent (parameterize) our trajectory as:

$$
\mathbf{x}_h(\tau) := \sum_{n=0}^K \mathbf{a}_n\,\tau^n
$$

where $\mathbf{a}_n \in \mathbb{R}^d$ are coefficient vectors to be determined.
Collocation enforces the differential equation at a chosen set of nodes on $[0,1]$. Depending on the node family, these nodes may be interior-only or may include one or both endpoints. With the polynomial state model, we can differentiate analytically. Using the change of variables $t=t_k+h_k\,\tau$, we obtain:

$$
\dot{\mathbf{x}}_h(t_k+h_k\,\tau_j) = \frac{1}{h_k} \sum_{n=1}^K n\,\mathbf{a}_n\,\tau_j^{n-1}
$$

The collocation condition requires that this polynomial derivative equals the right-hand side of the ODE at each collocation node $\tau_j$:

$$
\frac{1}{h_k} \sum_{n=1}^K n\,\mathbf{a}_n\,\tau_j^{n-1} = \mathbf{f}\left( \sum_{n=0}^K \mathbf{a}_n\,\tau_j^{n},\ \mathbf{u}_j,\ t_k+h_k\,\tau_j \right), \quad \text{for each collocation node } \tau_j.
$$

where $\mathbf{u}_j$ represents the control value at node $\tau_j$.

#### Boundary Conditions and Node Families

```{code-cell} ipython3
:tags: [remove-input, remove-output]
from importlib import reload
import sys
sys.path.append('_static')
import _static.collocation_illustration as _colloc
reload(_colloc)

```

```{glue:figure} collocation_figure
:name: fig-collocation-illustration
:figwidth: 90%
:align: center


Collocation node families and where slope and continuity constraints are enforced.
```


The choice of collocation nodes determines how boundary conditions are handled and affects the resulting discretization properties. Three standard families are commonly used: Labatto, Randau and and Gauss. 

Let's consider these three setup with more generality over any given basis. We start again by taking a mesh interval $[t_k,t_{k+1}]$ of length $h_k=t_{k+1}-t_k$ and reparametrize time by

$$
t = t_k + h_k\,\tau,\qquad \tau\in[0,1].
$$

We then choose to represent the (unknown) state by a degree–$K$ polynomial

$$
\mathbf{x}_h(\tau) \;=\; \sum_{n=0}^{K}\mathbf{a}_n\,\phi_n(\tau),
$$

where $\{\phi_n\}_{n=0}^K$ is any linearly independent basis of polynomials of degree $\le K$, and $\mathbf{a}_0,\dots,\mathbf{a}_K$ are vector coefficients to be determined.

The **collocation condition**  mean that we require for the chosen $K$ collocation points that:
$$
\frac{d}{dt}\mathbf{x}_h\bigl(\tau_j\bigr) \;=\; \mathbf{f}\bigl(\mathbf{x}_h(\tau_j),\mathbf{u}_h(\tau_j),t_k+h_k\tau_j\bigr),
\qquad j=0,1,\dots,K,
$$

which, using $\tfrac{d}{dt}=(1/h_k)\tfrac{d}{d\tau}$, becomes

$$
\underbrace{\tfrac{1}{h_k}\mathbf{x}_h'(\tau_j)}_{\text{polynomial slope at node}}
\;=\;
\underbrace{\mathbf{f}\!\bigl(\mathbf{x}_h(\tau_j),\mathbf{u}_h(\tau_j),t_k+h_k\tau_j\bigr)}_{\text{ODE slope at same point}},
\qquad j=0,\dots,K.
$$

Put simply: choose some nodes inside the interval, and at each of those nodes force the slope of the polynomial approximation to match the slope prescribed by the ODE. What we mean by the expression "collocation conditions" is simply to say that we want to satisfy a set of "slope-matching equations" at the chosen nodes. 

By **definition of the mesh variables**,

$$
\mathbf{x}_k := \mathbf{x}_h(0),\qquad \mathbf{x}_{k+1} := \mathbf{x}_h(1),
$$

and (if you sample the control at endpoints)

$$
\mathbf{u}_k := \mathbf{u}_h(0),\qquad \mathbf{u}_{k+1} := \mathbf{u}_h(1).
$$

With the monomial basis,

$$
\phi_n(0)=\delta_{n0}\ \Rightarrow\ \mathbf{x}_h(0)=\sum_{n=0}^K \mathbf{a}_n \phi_n(0)=\mathbf{a}_0=\mathbf{x}_k,
$$

$$
\phi_n(1)=1\ \Rightarrow\ \mathbf{x}_h(1)=\sum_{n=0}^K \mathbf{a}_n=\mathbf{x}_{k+1}.
$$

For the derivative, $\phi_n'(\tau)=n\,\tau^{n-1}$, so

$$
\mathbf{x}_h'(0)=\sum_{n=0}^K \mathbf{a}_n\,\phi_n'(0)=\mathbf{a}_1,
\qquad
\mathbf{x}_h'(1)=\sum_{n=1}^K n\,\mathbf{a}_n.
$$

When chaining intervals into a global trajectory, **direct collocation enforces state continuity by construction**: the variable $\mathbf{x}_{k+1}$ at the end of one interval is the same as the starting variable of the next. What is *not* enforced automatically is **slope continuity**; the derivative at the end of one interval generally does not match the derivative at the start of the next. Different collocation methods may have different slope continuity properties depending on the chosen collocation nodes. 


**Lobatto Nodes (endpoints included):**

The family of Labotto methods correspond to any set of so-called **Lobatto nodes** $\{\tau_j\}_{j=0}^K$ with the specificity that we require $\tau_0=0$ and $\tau_K=1$. Let's assume that we work with the **power (monomial) basis** $\phi_n(\tau)=\tau^n$, so that

$$
\mathbf{x}_h(\tau)=\sum_{n=0}^{K}\mathbf{a}_n\,\tau^n,
$$

Differentiating $\mathbf{x}_h$ with respect to $\tau$ gives

$$
\frac{d\mathbf{x}_h}{d\tau}(\tau)=\sum_{n=0}^{K}\mathbf{a}_n\,\phi_n'(\tau),
$$

Since we have the chain rule $\frac{d}{dt} = \frac{1}{h_k}\frac{d}{d\tau}$ from the time transformation $t = t_k + h_k\tau$, the time derivative becomes

$$
\frac{d\mathbf{x}_h}{dt}(t_k + h_k\tau) = \frac{1}{h_k}\frac{d\mathbf{x}_h}{d\tau}(\tau) = \frac{1}{h_k}\sum_{n=0}^{K}\mathbf{a}_n\,\phi_n'(\tau).
$$

so the **collocation equations** at Lobatto nodes are

$$
\frac{1}{h_k}\sum_{n=0}^{K}\mathbf{a}_n\,\phi_n'(\tau_j)
\;=\;
\mathbf{f}\!\Bigl(\sum_{n=0}^{K}\mathbf{a}_n\,\phi_n(\tau_j),\ \mathbf{u}_h(\tau_j),\ t_k+h_k\tau_j\Bigr),
\qquad j=0,1,\dots,K.
$$

For $j=0$ and $j=K$, these conditions become: 

$$
\frac{1}{h_k}\sum_{n=0}^{K}\mathbf{a}_n\,\phi_n'(0)
\;=\;
\mathbf{f}\!\Bigl(\sum_{n=0}^{K}\mathbf{a}_n\,\phi_n(0),\ \mathbf{u}_h(0),\ t_k\Bigr),
\qquad \text{(left endpoint)}
$$

$$
\frac{1}{h_k}\sum_{n=0}^{K}\mathbf{a}_n\,\phi_n'(1)
\;=\;
\mathbf{f}\!\Bigl(\sum_{n=0}^{K}\mathbf{a}_n\,\phi_n(1),\ \mathbf{u}_h(1),\ t_{k+1}\Bigr),
\qquad \text{(right endpoint)}
$$

With the monomial basis $\phi_n(\tau)=\tau^n$, we have $\phi_n'(0)=n\delta_{n,1}$ (only $\phi_1'=1$, others vanish) and $\phi_n'(1)=n$. Also, $\phi_n(0)=\delta_{n,0}$ and $\phi_n(1)=1$ for all $n$. This simplifies the endpoint conditions to:

$$
\frac{\mathbf{a}_1}{h_k} = \mathbf{f}(\mathbf{a}_0, \mathbf{u}_h(0), t_k) = \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k, t_k),
\qquad \text{(left endpoint slope)}
$$

$$
\frac{1}{h_k}\sum_{n=1}^{K}n\,\mathbf{a}_n = \mathbf{f}\!\Bigl(\sum_{n=0}^{K}\mathbf{a}_n, \mathbf{u}_h(1), t_{k+1}\Bigr) = \mathbf{f}(\mathbf{x}_{k+1}, \mathbf{u}_{k+1}, t_{k+1}),
\qquad \text{(right endpoint slope)}
$$

These equations enforce that the polynomial's slope at both endpoints matches the ODE's prescribed slope, which is why the figure shows red tangent lines at both endpoints for Lobatto methods.


**Radau Nodes (one endpoint included):**
Radau points include only one endpoint. Radau-I includes the left endpoint ($\tau_0 = 0$) while Radau-II includes the right endpoint ($\tau_K = 1$). This means that a radau collocation is defined by any set of collocation nodes such that $\tau_K = 1$. This translates into requireing that we match the ODE over the mesh only at the right endpoiunt in addition to the interior nodes.  As a consequence, we leave the solution unconstrained to take any value on the left endpoint. When chaining up multiple intervals across a global solution, this may pose some complication as we will no longer be able to ensure continuity as the slope at one endpoitn need not match that of the next endpoint. (But could you have a situation where slopes match but the states don't line up?)


 At the included endpoint the ODE is enforced (slope shown in the figure), while at the other endpoint continuity links adjacent intervals. For Radau-I with $K+1$ points:

$$
\mathbf{x}_k = \mathbf{x}_h(0) = \mathbf{a}_0
$$

The endpoint $\mathbf{x}_{k+1} = \mathbf{x}_h(1) = \sum_{n=0}^K \mathbf{a}_n$ is not directly constrained by a collocation condition, requiring separate continuity enforcement between intervals.

**Gauss Nodes (endpoints excluded):**
Gauss points exclude both endpoints, using only interior points $\tau_j \in (0,1)$ for $j = 1,\ldots,K$. The ODE is enforced only at interior nodes; both endpoints are handled through separate continuity constraints:

$$
\mathbf{x}_k = \mathbf{x}_h(0) = \mathbf{a}_0
$$
$$
\mathbf{x}_{k+1} = \mathbf{x}_h(1) = \sum_{n=0}^K \mathbf{a}_n
$$

**Origins and Selection Criteria:**
These node families derive from orthogonal polynomial theory. Gauss nodes correspond to roots of Legendre polynomials and provide optimal quadrature accuracy for smooth integrands. Radau nodes are roots of modified Legendre polynomials with one endpoint constraint, while Lobatto nodes include both endpoints and correspond to roots of derivatives of Legendre polynomials.

For optimal control applications, Radau-II nodes are often preferred because they provide implicit time-stepping behavior and good stability properties. Lobatto nodes simplify boundary condition handling but may require smaller time steps. Gauss nodes offer highest quadrature accuracy but complicate endpoint treatment.

#### Control Parameterization and Cost Integration

The control inputs can be handled with similar polynomial approximations. We may use piecewise-constant controls, piecewise-linear controls, or higher-order polynomial parameterizations of the form:

$$
\mathbf{u}_h(\tau) = \sum_{n=0}^{K_u} \mathbf{b}_n\,\tau^n
$$

where $\mathbf{u}_j = \mathbf{u}_h(\tau_j)$ represents the control values at each collocation point. This polynomial framework extends to cost function evaluation, where running costs are integrated using the same quadrature nodes and weights:

$$
\int_{t_k}^{t_{k+1}} c\,dt \approx h_k\sum_{j=0}^K w_j\, c\big(\mathbf{x}_h(\tau_j),\mathbf{u}_h(\tau_j), t_k+h_k\,\tau_j\big)
$$



# Applying the recipe: concrete transcriptions

The mesh and interior nodes are the common scaffold. What distinguishes one transcription from another is how we obtain values at those nodes and how we approximate the two integrals that appear implicitly and explicitly: the integral of the running cost and the integral that carries the state forward. In other words, we now commit to two design choices that mirror the previous section: a finite representation for $\mathbf{x}(t)$ and $\mathbf{u}(t)$ over each interval $[t_i,t_{i+1}]$, and a quadrature rule whose nodes and weights are used consistently for both cost and dynamics. The result is always a sparse nonlinear program; the differences are in where we sample and how we tie samples together.

Below, each transcription should be read as "same grid, same interior points, same evaluations for cost and physics," with only the local representation changing.

## Euler Collocation

Work on one interval $[t_k,t_{k+1}]$ of length $h_k$ with the reparametrization $t=t_k+h_k\,\tau$, $\tau\in[0,1]$. Assume a degree 1 polynomial:

$$
\mathbf{x}_h(\tau)=\sum_{n=0}^{1}\mathbf{a}_n\,\phi_n(\tau),
$$

for any basis $\{\phi_0,\phi_1\}$ of linear polynomials. 
Endpoint conditions give

$$
\mathbf{x}_h(0)=\mathbf{x}_k\Rightarrow \mathbf{a}_0=\mathbf{x}_k,\qquad
\mathbf{x}_h(1)=\mathbf{x}_{k+1}\Rightarrow \mathbf{a}_1=\mathbf{x}_{k+1}-\mathbf{x}_k.
$$

by backsubstitution and because 

$$
\mathbf{x}_h(\tau)=\mathbf{a}_0+\mathbf{a}_1\,\tau.
$$

Furthermore, the derivative with respect to $\tau$ is:
$$
\frac{d}{d\tau}\mathbf{x}_h(\tau)=\mathbf{a}_1=\mathbf{x}_{k+1}-\mathbf{x}_k,
\qquad
\frac{d}{dt}=\frac{1}{h_k}\frac{d}{d\tau}
\Rightarrow
\frac{d}{dt}\mathbf{x}_h=\frac{1}{h_k}\,(\mathbf{x}_{k+1}-\mathbf{x}_k).
$$

Because from the mapping $t(\tau)=t_k+h_k\tau$ we can invert:
$\tau(t)=\frac{t-t_k}{h_k}$ and differentiating gives

$$
\frac{d\tau}{dt}=\frac{1}{h_k}.
$$


The **collocation condition** at a single **Radau-II node** $\tau=1$:

$$
\frac{1}{h_k}\,\mathbf{x}_h'(\tau)\Big|_{\tau=1}
\;=\;
\mathbf{f}\!\big(\mathbf{x}_h(1),\mathbf{u}_h(1),t_{k+1}\big)
\;=\;
\mathbf{f}\!\big(\mathbf{x}_{k+1},\mathbf{u}_{k+1},t_{k+1}\big).
$$

Because $\mathbf{x}_h$ is linear, $\mathbf{x}_h'(\tau)$ is constant in $\tau$, so $\mathbf{x}_h'(1)=\mathbf{x}_h'(\tau)$ for all $\tau$. Moreover, linear interpolation between the two endpoints gives

$$
\mathbf{x}_h'(\tau)=\mathbf{x}_{k+1}-\mathbf{x}_k.
$$

Substitute this into the collocation condition:

$$
\frac{1}{h_k}\big(\mathbf{x}_{k+1}-\mathbf{x}_k\big) \;=\; \mathbf{f}\!\big(\mathbf{x}_{k+1},\mathbf{u}_{k+1},t_{k+1}\big),
$$

which is exactly the **implicit Euler** step

$$
{\ \mathbf{x}_{k+1}=\mathbf{x}_k + h_k\,\mathbf{f}\!\big(\mathbf{x}_{k+1},\mathbf{u}_{k+1},t_{k+1}\big)\ }.
$$



The overall direct transcription is then: 

```{prf:definition} Implicit–Euler Collocation (Radau-II, degree 1)
Let $t_0<\cdots<t_N$ with $h_i:=t_{i+1}-t_i$. Decision variables are $\{\mathbf{x}_i\}_{i=0}^N$, $\{\mathbf{u}_i\}_{i=0}^N$. Solve

$$
\begin{aligned}
\min\ & c_T(\mathbf{x}_N)\;+\;\sum_{i=0}^{N-1} h_i\,c(\mathbf{x}_{i+1},\mathbf{u}_{i+1})\\
\text{s.t.}\ & \mathbf{x}_{i+1}-\mathbf{x}_i - h_i\,\mathbf{f}(\mathbf{x}_{i+1},\mathbf{u}_{i+1})=\mathbf{0},\quad i=0,\ldots,N-1,\\
& \mathbf{g}(\mathbf{x}_{i+1},\mathbf{u}_{i+1})\le \mathbf{0},\\
& \mathbf{x}_{\min}\le \mathbf{x}_i\le \mathbf{x}_{\max},\quad \mathbf{u}_{\min}\le \mathbf{u}_i\le \mathbf{u}_{\max},\\
& \mathbf{x}_0=\mathbf{x}(t_0).
\end{aligned}
$$

```

Note that:

* The running cost and path constraints are evaluated at the **same** right-endpoint where the dynamics are enforced, keeping "where we pay" aligned with "where we enforce."
* State continuity is automatic because $\mathbf{x}_{i+1}$ is a shared variable between adjacent intervals; slope continuity is not enforced unless you add it.
Here's an updated subsection that explicitly says **what collocation nodes are chosen** and why the trapezoidal defect uses them the way it does.


> Side remark. If you instead collocate at the **left** endpoint (Radau-I with $\tau=0$) with the same linear model, you obtain $\frac{1}{h_k}(\mathbf{x}_{k+1}-\mathbf{x}_k)=\mathbf{f}(\mathbf{x}_k,\mathbf{u}_k,t_k)$, i.e., the **explicit Euler** step. In that very precise sense, explicit Euler can be viewed as a (left-endpoint) degree-1 collocation scheme.

```{prf:definition} Explicit–Euler Collocation (Radau-I, degree 1)
Let $t_0<\cdots<t_N$ with $h_i:=t_{i+1}-t_i$. Decision variables are $\{\mathbf{x}_i\}_{i=0}^N$ and $\{\mathbf{u}_i\}_{i=0}^N$. Solve
$$
\begin{aligned}
\min\ & c_T(\mathbf{x}_N)\;+\;\sum_{i=0}^{N-1} h_i\,c(\mathbf{x}_i,\mathbf{u}_i)\\
\text{s.t.}\ & \mathbf{x}_{i+1}-\mathbf{x}_i - h_i\,\mathbf{f}(\mathbf{x}_i,\mathbf{u}_i)=\mathbf{0},\quad i=0,\ldots,N-1,\\
& \mathbf{g}(\mathbf{x}_i,\mathbf{u}_i)\le \mathbf{0},\quad i=0,\ldots,N-1,\\
& \mathbf{x}_{\min}\le \mathbf{x}_i\le \mathbf{x}_{\max},\quad \mathbf{u}_{\min}\le \mathbf{u}_i\le \mathbf{u}_{\max},\\
& \mathbf{x}_0=\mathbf{x}(t_0).
\end{aligned}
$$
```

## Trapezoidal collocation

In this scheme we take the **two endpoints as the nodes** on each interval:

$$
\tau_0=0,\qquad \tau_1=1\quad(\text{"Lobatto with }K=1\text{").
$$

We approximate $\mathbf{x}$ **linearly** over $[t_i,t_{i+1}]$, and we evaluate both the running cost and the dynamics at these two nodes with **equal weights**. Because a linear polynomial has a **constant** derivative, we do **not** try to match the ODE's slope at both endpoints (that would overconstrain a linear function). Instead, we enforce the ODE in its **integral form** over the interval and approximate the integral of $\mathbf{f}$ by the **trapezoid rule** using those two nodes. This makes the cost quadrature and the state-update ("defect") use the **same nodes and weights**.

```{prf:definition} Trapezoidal Collocation 
Let $t_0<\cdots<t_N$ with $h_i:=t_{i+1}-t_i$. Decision variables are $\{\mathbf{x}_i\}_{i=0}^N$, $\{\mathbf{u}_i\}_{i=0}^N$. Solve

$$
\begin{aligned}
\min_{\{\mathbf{x}_i,\mathbf{u}_i\}}\ & c_T(\mathbf{x}_N)\; +\; \sum_{i=0}^{N-1} \tfrac{h_i}{2}\,\Big[c(\mathbf{x}_i,\mathbf{u}_i)+c(\mathbf{x}_{i+1},\mathbf{u}_{i+1})\Big]\\
\text{s.t.}\ & \mathbf{x}_{i+1}-\mathbf{x}_i \;-\; \tfrac{h_i}{2}\Big[\mathbf{f}(\mathbf{x}_i,\mathbf{u}_i)+\mathbf{f}(\mathbf{x}_{i+1},\mathbf{u}_{i+1})\Big] \;=\; \mathbf{0},\quad i=0,\ldots,N-1,\\
& \mathbf{g}(\mathbf{x}_i,\mathbf{u}_i) \le \mathbf{0},\ \ \mathbf{g}(\mathbf{x}_{i+1},\mathbf{u}_{i+1}) \le \mathbf{0},\\
& \mathbf{x}_{\min} \le \mathbf{x}_i \le \mathbf{x}_{\max},\ \ \mathbf{u}_{\min} \le \mathbf{u}_i \le \mathbf{u}_{\max},\\
& \mathbf{x}_0 = \mathbf{x}(t_0).
\end{aligned}
$$

```

*Summary:* the **collocation nodes** for trapezoidal are the **two endpoints**; the state is **linear** on each interval; and the dynamics are enforced via the **integrated** ODE with the **trapezoid rule** at those two nodes, yielding the familiar trapezoidal defect.

## Hermite–Simpson (quadratic interpolation; midpoint included)

On each interval $[t_i,t_{i+1}]$ we pick **three collocation nodes** on the reference domain $\tau\in[0,1]$:

$$
\tau_0=0,\qquad \tau_{1/2}=\tfrac12,\qquad \tau_1=1.
$$

So we evaluate at **left, midpoint, right**. These are the same three nodes used by Simpson's rule (weights $1{:}4{:}1$) for numerical quadrature. We let $\mathbf{x}_h$ be **quadratic** in $\tau$. Two things then happen:

1. **Integral (defect) enforcement with Simpson's rule.**
   We enforce the ODE in integral form over the interval and approximate the integral of $\mathbf{f}$ with Simpson's rule using the three nodes above. This yields the first constraint (the "Simpson defect"), which uses $\mathbf{f}$ evaluated at left, midpoint, and right.

2. **Slope matching at the midpoint (collocation).**
   Because a quadratic has limited shape, we don't try to match slopes at both endpoints. Instead, we **introduce midpoint variables** $(\mathbf{x}_{i+\frac12},\mathbf{u}_{i+\frac12})$ and **match the ODE at the midpoint**. The second constraint below is exactly the midpoint collocation condition written in an equivalent Hermite form: it pins the midpoint state to the average of the endpoints plus a correction based on endpoint slopes, ensuring that the polynomial's derivative is consistent with the ODE **at $\tau=\tfrac12$**.

This way, **where we pay** (Simpson quadrature) and **where we enforce** (midpoint collocation + Simpson defect) are aligned at the same three nodes, which is why the method is both accurate and well conditioned on smooth problems.

```{prf:definition} Hermite–Simpson Transcription 
Let $t_0<\cdots<t_N$ with $h_i:=t_{i+1}-t_i$ and midpoints $t_{i+\frac12}$. Decision variables are $\{\mathbf{x}_i\}_{i=0}^N$, $\{\mathbf{u}_i\}_{i=0}^N$, plus midpoint variables $\{\mathbf{x}_{i+\frac12},\mathbf{u}_{i+\frac12}\}_{i=0}^{N-1}$. Solve

$$
\begin{aligned}
\min\ & c_T(\mathbf{x}_N)\; +\; \sum_{i=0}^{N-1} \tfrac{h_i}{6}\Big[ c(\mathbf{x}_i,\mathbf{u}_i) + 4\,c(\mathbf{x}_{i+\frac12},\mathbf{u}_{i+\frac12}) + c(\mathbf{x}_{i+1},\mathbf{u}_{i+1}) \Big]\\
\text{s.t.}\ & \underbrace{\mathbf{x}_{i+1}-\mathbf{x}_i - \tfrac{h_i}{6}\Big[\mathbf{f}(\mathbf{x}_i,\mathbf{u}_i) + 4\,\mathbf{f}(\mathbf{x}_{i+\frac12},\mathbf{u}_{i+\frac12}) + \mathbf{f}(\mathbf{x}_{i+1},\mathbf{u}_{i+1})\Big]}_{\text{Simpson defect over }[t_i,t_{i+1}]} = \mathbf{0},\\
& \underbrace{\mathbf{x}_{i+\frac12} - \tfrac{\mathbf{x}_i+\mathbf{x}_{i+1}}{2} - \tfrac{h_i}{8}\Big[\mathbf{f}(\mathbf{x}_i,\mathbf{u}_i) - \mathbf{f}(\mathbf{x}_{i+1},\mathbf{u}_{i+1})\Big]}_{\text{midpoint collocation (slope matching at }t_{i+\frac12}\text{)}} = \mathbf{0},\\
& \mathbf{g}(\mathbf{x}_i,\mathbf{u}_i) \le \mathbf{0},\ \ \mathbf{g}(\mathbf{x}_{i+\frac12},\mathbf{u}_{i+\frac12}) \le \mathbf{0},\ \ \mathbf{g}(\mathbf{x}_{i+1},\mathbf{u}_{i+1}) \le \mathbf{0},\\
& \mathbf{x}_{\min} \le \mathbf{x}_i,\mathbf{x}_{i+\frac12} \le \mathbf{x}_{\max},\ \ \mathbf{u}_{\min} \le \mathbf{u}_i,\mathbf{u}_{i+\frac12} \le \mathbf{u}_{\max},\\
& \mathbf{x}_0 = \mathbf{x}(t_0),\quad i=0,\ldots,N-1.
\end{aligned}
$$
```

**Collocation nodes recap:** $\tau=0,\ \tfrac12,\ 1$.

* The **midpoint** is where we explicitly **match the ODE slope** (collocation).
* The **three nodes together** are used for the **Simpson integral** of $\mathbf{f}$ (state update) and of $c$ (cost), keeping physics and objective synchronized.


# Examples

## Compressor Surge Problem 

A compressor is a machine that raises the pressure of a gas by squeezing it into a smaller volume. You find them in natural gas pipelines, jet engines, and factories. But compressors can run into trouble if the flow of gas becomes too small. In that case, the machine can "stall" much like an airplane wing at too high an angle. Instead of moving forward, the gas briefly pushes back, creating strong pressure oscillations that can damage the compressor and anything connected to it.

To prevent this, engineers often add a close-coupled valve (CCV) at the outlet. The valve can quickly adjust the flow to keep the compressor away from these unstable conditions. Our goal is to design a control strategy for operating this valve so that the compressor never enters surge.

Following  {cite:p}`Gravdahl1997` and {cite}`Grancharova2012`, we model the compressor using a simplified second-order representation:

$$
\begin{aligned}
\dot{x}_1 &= B(\Psi_e(x_1) - x_2 - u) \\
\dot{x}_2 &= \frac{1}{B}(x_1 - \Phi(x_2))
\end{aligned}
$$

Here, $\mathbf{x} = [x_1, x_2]^T$ represents the state variables:

- $x_1$ is the normalized mass flow through the compressor.
- $x_2$ is the normalized pressure ratio across the compressor.

The control input $u$ denotes the normalized mass flow through a CCV.
The functions $\Psi_e(x_1)$ and $\Phi(x_2)$ represent the characteristics of the compressor and valve, respectively:

$$
\begin{aligned}
\Psi_e(x_1) &= \psi_{c0} + H\left(1 + 1.5\left(\frac{x_1}{W} - 1\right) - 0.5\left(\frac{x_1}{W} - 1\right)^3\right) \\
\Phi(x_2) &= \gamma \operatorname{sign}(x_2) \sqrt{|x_2|}
\end{aligned}
$$

The system parameters are given as $\gamma = 0.5$, $B = 1$, $H = 0.18$, $\psi_{c0} = 0.3$, and $W = 0.25$.

One possible way to pose the problem {cite}`Grancharova2012` is by penalizing deviations from the setpoints using a quadratic penalty in the instantaneous cost function as well as in the terminal one. Furthermore, we also penalize taking large actions (which are energy hungry and potentially unsafe) within the integral term. The idea of penalizing deviations throughout is a natural way of posing the problem when solving it via single shooting. Another alternative, which we will explore below, is to set the desired setpoint as a hard terminal constraint. 

The control objective is to stabilize the system and prevent surge, formulated as a continuous-time optimal control problem (COCP) in the Bolza form:

$$
\begin{aligned}
\text{minimize} \quad & \left[ \int_0^T \alpha(\mathbf{x}(t) - \mathbf{x}^*)^T(\mathbf{x}(t) - \mathbf{x}^*) + \kappa u(t)^2 \, dt\right] + \beta(\mathbf{x}(T) - \mathbf{x}^*)^T(\mathbf{x}(T) - \mathbf{x}^*) + R v^2  \\
\text{subject to} \quad & \dot{x}_1(t) = B(\Psi_e(x_1(t)) - x_2(t) - u(t)) \\
& \dot{x}_2(t) = \frac{1}{B}(x_1(t) - \Phi(x_2(t))) \\
& u_{\text{min}} \leq u(t) \leq u_{\text{max}} \\
& -x_2(t) + 0.4 \leq v \\
& -v \leq 0 \\
& \mathbf{x}(0) = \mathbf{x}_0
\end{aligned}
$$

The parameters $\alpha$, $\beta$, $\kappa$, and $R$ are non-negative weights that allow the designer to prioritize different aspects of performance (e.g., tight setpoint tracking vs. smooth control actions). We also constraint the control input to be within $0 \leq u(t) \leq 0.3$ due to the physical limitations of the valve.

The authors in {cite}`Grancharova2012` also add a soft path constraint $x_2(t) \geq 0.4$ to ensure that we maintain a minimum pressure at all time. This is implemented as a soft constraint using slack variables. The reason that we have the term $R v^2$ in the objective is to penalizes violations of the soft constraint: we allow for deviations, but don't want to do it too much.  

In the experiment below, we choose the setpoint $\mathbf{x}^* = [0.40, 0.60]^T$ as it corresponds to an unstable equilibrium point. If we were to run the system without applying any control, we would see that the system starts to oscillate. 

```{code-cell} ipython3
:tags: [hide-input]
:load: code/compressor_surge_single_shooting.py
```

### Solution by Trapezoidal Collocation

Another way to pose the problem is by imposing a terminal state constraint on the system rather than through a penalty in the integral term. In the following experiment, we use a problem formulation of the form: 

$$
\begin{aligned}
\text{minimize} \quad & \left[ \int_0^T \kappa u(t)^2 \, dt\right] \\
\text{subject to} \quad & \dot{x}_1(t) = B(\Psi_e(x_1(t)) - x_2(t) - u(t)) \\
& \dot{x}_2(t) = \frac{1}{B}(x_1(t) - \Phi(x_2(t))) \\
& u_{\text{min}} \leq u(t) \leq u_{\text{max}} \\
& \mathbf{x}(0) = \mathbf{x}_0 \\
& \mathbf{x}(T) = \mathbf{x}^\star
\end{aligned}
$$

We then find a control function $u(t)$ and state trajectory $x(t)$ using the trapezoidal collocation method. 

```{code-cell} ipython3
:tags: [hide-input]
:load: code/compressor_surge_trapezoidal_collocation.py
```

You can try to vary the number of collocation points in the code and observe how the state trajectory progressively matches the ground truth (the line denoted "integrated solution"). Note that this version of the code also lacks bound constraints on the variable $x_2$ to ensure a minimum pressure, as we did earlier. Consider this a good exercise to try on your own. 

### System Identification as Trajectory Optimization (Compressor Surge)

We now turn the compressor surge model into a simple system identification task: estimate unknown parameters (here, the scalar $B$) from measured trajectories. This can be viewed as a trajectory optimization problem: choose parameters (and optionally states) to minimize reconstruction error while enforcing the dynamics.

Given time-aligned data $\{(\mathbf{u}_k,\mathbf{y}_k)\}_{k=0}^{N}$, model states $\mathbf{x}_k\in\mathbb{R}^d$, outputs $\mathbf{y}_k\approx \mathbf{h}(\mathbf{x}_k;\boldsymbol{\theta})$, step $\Delta t$, and dynamics $\mathbf{f}(\mathbf{x},\mathbf{u};\boldsymbol{\theta})$, the simultaneous (full-discretization) viewpoint is

$$
\begin{aligned}
\min_{\boldsymbol{\theta},\,\{\mathbf{x}_k\}} \quad & \sum_{k\in K}\;\big\|\mathbf{y}_k - \mathbf{h}(\mathbf{x}_k;\boldsymbol{\theta})\big\|_2^2 \\
\text{s.t.}\quad & \mathbf{x}_{k+1} - \mathbf{x}_k - \Delta t\,\mathbf{f}(\mathbf{x}_k,\mathbf{u}_k;\boldsymbol{\theta}) = \mathbf{0},\quad k=0,\ldots,N-1, \\
& \mathbf{x}_0 \;\text{given},
\end{aligned}
$$

while the single-shooting (recursive elimination) variant eliminates the states by simulating forward from $\mathbf{x}_0$:

$$
J(\boldsymbol{\theta}) := \sum_{k\in K}\;\big\|\mathbf{y}_k - \mathbf{h}(\boldsymbol{\phi}_k(\boldsymbol{\theta};\mathbf{x}_0,\mathbf{u}_{0:N-1})\big\|_2^2,\quad \min_{\boldsymbol{\theta}} J(\boldsymbol{\theta}),
$$

where $\boldsymbol{\phi}_k$ denotes the state reached at step $k$ by an RK4 rollout under parameter $\boldsymbol{\theta}$. In our demo the data grid and rollout grid coincide, so $\boldsymbol{\phi}_k = \mathbf{x}_k$ and no interpolation is required. We will identify $B$ by fitting the model to data generated from the ground-truth $B=1$ system under randomized initial conditions and small input perturbations.

```{code-cell} ipython3
:tags: [hide-input]
:load: code/compressor_surge_data_collection.py
```

```{code-cell} ipython3
:tags: [hide-input]
:load: code/compressor_surge_direct_single_shooting_rk4_paramid.py
```

## Flight Trajectory Optimization

We consider a concrete task: computing a fuel-optimal trajectory between Montréal–Trudeau (CYUL) and Toronto Pearson (CYYZ), taking into account both aircraft dynamics and wind conditions along the route. The wind field comes from **ERA5**, a global atmospheric dataset produced by the ECMWF. It combines historical observations from satellites, aircraft, and surface stations with a weather model to reconstruct the state of the atmosphere across space and time. In climate science, this is called a *reanalysis*.

ERA5 data is stored in **GRIB files**, a compact format widely used in meteorology. Each file contains a **gridded field**: values of wind and other variables arranged on a regular 4D lattice over longitude, latitude, altitude, and time. Since the aircraft rarely sits exactly on a grid point, we interpolate the wind components it sees as it moves.

The aircraft is modeled as a point mass with state

$$
\mathbf{x}(t) = (x(t), y(t), h(t), m(t)),
$$

where $(x, y)$ is horizontal position, $h$ is altitude, and $m$ is remaining mass. Controls are Mach number $M(t)$, vertical speed $v\_s(t)$, and heading angle $\psi(t)$. The equations of motion combine airspeed and wind:

$$
\begin{aligned}
\dot x &= v(M,h)\cos\psi\cos\gamma + u_w(x,y,h,t), \\
\dot y &= v(M,h)\sin\psi\cos\gamma + v_w(x,y,h,t), \\
\dot h &= v_s, \\
\dot m &= -\,\mathrm{FF}(T(h,M,v_s), h, M, v_s),
\end{aligned}
$$

where $\gamma = \arcsin(v\_s / v)$ is the flight path angle and $\mathrm{FF}$ is the fuel flow rate based on current conditions. The wind terms $u\_w$ and $v\_w$ are taken from ERA5 and interpolated in space and time.

The optimization minimizes fuel burn over the CYUL–CYYZ leg. But the same setup could be used to minimize arrival time, or some weighted combination of time, cost, and emissions.

We use **OpenAP.top**, which solves the problem using direct collocation at **Legendre–Gauss–Lobatto (LGL)** points. Each trajectory segment is mapped to the unit interval, the state is interpolated by Lagrange polynomials at nonuniform LGL nodes, and the dynamics are enforced at those points. Integration is done with matching quadrature weights.

This setup lets us optimize trajectories under realistic conditions by feeding in the appropriate ERA5 GRIB file (e.g., `era5_mtl_20230601_12.grib`). The result accounts for wind patterns (eg. headwinds, tailwinds, shear) along the corridor between Montréal and Toronto.


```{code-cell} ipython3
:tags: [hide-input]
# OpenAP.top demo with optional wind overlay – docs: https://github.com/junzis/openap-top
from openap import top
import matplotlib.pyplot as plt
import os

# Montreal region route (Canada): CYUL (Montréal–Trudeau) → CYYZ (Toronto)
opt = top.CompleteFlight("A320", "CYUL", "CYYZ", m0=0.85)

# Optional: point to a local ERA5/GRIB file to enable wind (set env var OPENAP_WIND_GRIB)
# If not set, look for a default small file produced by `_static/openap_fetch_era5.py`.
fgrib = os.environ.get("OPENAP_WIND_GRIB", "_static/era5_mtl_20230601_12.grib")
windfield = None
if fgrib and os.path.exists(fgrib):
    try:
        windfield = top.tools.read_grids(fgrib)
        opt.enable_wind(windfield)
    except Exception:
        windfield = None  # fall back silently if GRIB reading deps are missing

# Solve for a fuel-optimal trajectory (CasADi direct collocation under the hood)
flight = opt.trajectory(objective="fuel")

# Visualize; overlay wind barbs if windfield available
if windfield is not None:
    ax = top.vis.trajectory(flight, windfield=windfield, barb_steps=15)
else:
    ax = top.vis.trajectory(flight)

title = "OpenAP.top fuel-optimal trajectory (A320: CYUL → CYYZ)"
if hasattr(ax, "set_title"):
    ax.set_title(title)
else:
    plt.title(title)

```

## Hydro Cascade Scheduling with Physical Routing and Multiple Shooting

Earlier in the book, we introduced a simplified view of hydro reservoir control, where the water level evolves in discrete time by accounting for inflow and outflow, with precipitation treated as a noisy input. While useful for learning and control design, this model abstracts away much of the physical behavior of actual rivers and dams.

In this chapter, we move toward a more realistic setup. We consider a series of dams arranged in a cascade, where the actions taken upstream influence downstream levels with a delay. The amount of power produced depends not only on how much water flows through the turbines, but also on the head (the vertical distance between the reservoir surface and the turbine outlet). The larger the head, the more potential energy is available for conversion into electricity, and the higher the power output.

To capture these effects, we follow a modeling approach inspired by the Saint-Venant equations, which describe how water levels and flows evolve in open channels. Instead of solving the full PDEs, we use a reduced model that approximates each dammed section of river (called a reach) as a lumped system governed by an ordinary differential equation. The main variable of interest is the water level $h_r(t)$, which changes over time depending on how much water enters, how much is discharged through the turbines $q_r(t)$, and how much is spilled $s_r(t)$. The mass balance for reach $r$ is written as:

$$
\frac{d h_r(t)}{dt} = \frac{1}{A_r} \left( z_r(t) - q_r(t) - s_r(t) \right),
$$

where $A_r$ is the surface area of the reservoir, assumed constant. The inflow $z_r(t)$ to a reach either comes from nature (for the first dam), or from the upstream turbine and spill discharge, delayed by a travel time $\tau_{r-1}$:

$$
z_1(t) = \text{inflow}(t), \qquad
z_r(t) = q_{r-1}(t - \tau_{r-1}) + s_{r-1}(t - \tau_{r-1}), \quad \text{for } r > 1.
$$

Power generation at each reach depends on how much water is discharged and the available head:

$$
P_r(t) = \rho g \eta \, q_r(t) \, H_r(h_r(t)),
$$

where $\rho$ is water density, $g$ is gravitational acceleration, $\eta$ is turbine efficiency, and $H_r(h_r(t))$ denotes the head as a function of the water level. In some models, the head is approximated as the difference between the current level and a fixed tailwater height (the water level downstream of the dam, after it has passed through the turbine).

The operator's goal is to meet a target generation profile $P^\text{ref}(t)$, such as one dictated by a market dispatch or load-following constraint. This leads to an objective that minimizes the deviation from the target over the full horizon:

$$
\min_{\{q_r(t), s_r(t)\}} \int_0^T \left( \sum_{r=1}^R P_r(t) - P^\text{ref}(t) \right)^2 dt.
$$

In practice, this is combined with operational constraints: turbine capacity $0 \le q_r(t) \le \bar{q}_r$, spillway limits $0 \le s_r(t) \le \bar{s}_r$, and safe level bounds $h_r^{\min} \le h_r(t) \le h_r^{\max}$. Depending on the use case, one may also penalize spill to encourage water conservation, or penalize fast changes in levels for ecological reasons.

What makes this problem particularly interesting is the coupling across space and time. An upstream reach cannot simply act in isolation: if the operator wants reach $r$ to produce power at a specific time, the water must be released by reach $r-1$ sufficiently in advance. This coordination is further complicated by delays, nonlinearities in head-dependent power, and limited storage capacity.

We solve the problem using **multiple shooting**. Each reach is divided into local simulation segments over short time windows. Within each segment, the dynamics are integrated forward using the ODEs, and continuity constraints are added to ensure that the water levels match across segment boundaries. At the same time, the inflows passed from upstream reaches must arrive at the right time and be consistent with previous decisions. In discrete time, this gives rise to a set of state-update equations:

$$
h_r^{k+1} = h_r^k + \Delta t \cdot \frac{1}{A_r}(z_r^k - q_r^k - s_r^k),
$$

with delays handled by shifting $z_r^k$ according to the appropriate travel time. These constraints are enforced as part of a nonlinear program, alongside the power tracking objective and control bounds.

Compared to the earlier inflow-outflow model, this richer setup introduces more structure, but also more opportunity. The cascade now behaves like a coordinated team: upstream reservoirs can store water in anticipation of future needs, while downstream dams adjust their output to match arrivals and avoid overflows. The optimization reveals not just a schedule, but a strategy for how the entire system should act together to meet demand.

```{code-cell} ipython3
:tags: [hide-input]
:load: _static/hydro.py

```

The figure shows the result of a multiple-shooting optimization applied to a three-reach hydroelectric cascade. The time horizon is discretized into 16 intervals, and SciPy's `trust-constr` solver is used to find a feasible control sequence that satisfies mass balance, turbine and spillway limits, and Muskingum-style routing dynamics. Each reach integrates its own local ODE, with continuity constraints linking the flows between reaches.

The top-left panel shows the water levels in each reservoir. We observe that upstream reservoirs tend to increase their levels ahead of discharge events, building potential energy before releasing water downstream. The top-right panel shows turbine discharges for each reach. These vary smoothly and are temporally coordinated across the system. The bottom-right panel compares the total generation to a synthetic demand profile, which is generated by a sum of time-shifted sigmoids and normalized to be feasible given turbine capacities. The optimized schedule (orange) tracks this demand closely, while the initial guess (blue) lags behind. The bottom-left panel plots the routed inflows between reaches, which display the expected lag and smoothing effects from Muskingum routing. The interplay between these plots shows how the system anticipates, stores, and routes water to meet time-varying generation targets within physical and operational limits.