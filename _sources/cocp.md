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

The discrete-time problems of the previous chapter already suggested how to proceed: we convert a continuous problem into one over finitely many numbers by deciding where to look at the trajectories and how to interpolate between those looks. We place a mesh $t_0<t_1<\cdots<t_N=t_f$ and, inside each window $[t_k,t_{k+1}]$, select a small set of interior fractions $\{\xi_i\}$ on the reference interval $[0,1]$. The running cost is additive over windows, so we write it as a sum of local integrals, map each window to $[0,1]$, and approximate each local integral by a quadrature rule with nodes $\xi_i$ and weights $w_i$. This produces

$$
\int_{t_0}^{t_f} c(\mathbf{x}(t),\mathbf{u}(t))\,dt
\;\approx\;
\sum_{k=0}^{N-1} h_k \sum_{i=1}^q w_i\, c\!\big(\mathbf{x}(t_k+h_k\xi_i),\,\mathbf{u}(t_k+h_k\xi_i)\big),
$$

with $h_k=t_{k+1}-t_k$. The dynamics are treated in the same way by the fundamental theorem of calculus,

$$
\mathbf{x}(t_{k+1})-\mathbf{x}(t_k)=\int_{t_k}^{t_{k+1}} \mathbf{f}(\mathbf{x}(t),\mathbf{u}(t))\,dt
\;\approx\;
h_k \sum_{i=1}^q b_i\, \mathbf{f}\!\big(\mathbf{x}(t_k+h_k\xi_i),\,\mathbf{u}(t_k+h_k\xi_i)\big),
$$

so the places where we “pay” running cost are the same places where we “account” for state changes. Path constraints and bounds are then enforced at the same interior times. In the infinite-horizon discounted case, the same formulas apply with an extra factor $e^{-\rho(t_k+h_k\xi_i)}$ multiplying the weights in the cost.

The values $\mathbf{x}(t_k+h_k\xi_i)$ and $\mathbf{u}(t_k+h_k\xi_i)$ do not exist a priori. We create them by a finite representation. One option is shooting: parameterize $\mathbf{u}$ on the mesh, integrate the ODE across each window with a chosen numerical step, and read interior values from that step. Another is collocation: represent $\mathbf{x}$ inside each window by a local polynomial and choose its coefficients so that the ODE holds at the interior nodes. Both constructions lead to the same structure: a nonlinear program whose objective is a composite quadrature of the running term (plus any terminal term in the Bolza case) and whose constraints are algebraic relations that encode the ODE and the pointwise inequalities at the selected nodes.

Specific choices recover familiar schemes. If we use the left endpoint as the single interior node, we obtain the forward Euler transcription. If we use both endpoints with equal weights, we obtain the trapezoidal transcription. Higher-order rules arise when we include interior nodes and richer polynomials for $\mathbf{x}$. What matters here is the unifying picture: choose nodes, translate integrals into weighted sums, and couple those evaluations to a finite trajectory representation so that cost and physics are enforced at the same places. This is the organizing idea that will guide the rest of the chapter.



## Discretizing cost and dynamics together

In a continuous-time OCP, integrals appear twice: in the objective, which accumulates running cost over time, and implicitly in the dynamics, since state changes over any interval are the integral of the vector field. To compute, we must approximate both the integrals and the unknown functions $\mathbf{x}(t)$ and $\mathbf{u}(t)$ with finitely many numbers that an optimizer can manipulate.

A natural way to do this is to lay down a finite set of time points (a mesh) over the horizon. You can think of the mesh as a grid we overlay on the “true” trajectories that exist as mathematical objects but are not directly accessible. Our aim is to approximate those trajectories and their integrals using values and simple models tied to the mesh. Using the same mesh for both the cost and the dynamics keeps the representation coherent: we evaluate what we pay and how the state changes at consistent times.

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

When we approximate this integral, we introduce interior evaluation points $t_k^{(i)}\in[t_k,t_{k+1}]$. Using the **same points** in the cost and in the dynamics ties $\mathbf{x}$ and $\mathbf{u}$ together coherently: the places where we “pay” for running cost are also the places where we enforce the ODE. This avoids a mismatch between where we approximate the objective and where we impose feasibility.

Third, the decomposition yields a nonlinear program with **sparse structure**. Each interval contributes a small block to the objective and constraints that depends only on variables from that interval (and its endpoints). Modern solvers exploit this banded sparsity to scale to long horizons. 

With the split justified, we standardize the approximation. Map each interval to a reference domain via $t=t_k+h_k\tau$ with $\tau\in[0,1]$ and $dt=h_k\,d\tau$. A **quadrature rule on $[0,1]$** is specified by evaluation points $\{\xi_i\}_{i=1}^q \subset [0,1]$ and positive weights $\{w_i\}_{i=1}^q$ such that, for a smooth $\phi$,

$$
\int_0^1 \phi(\tau)\,d\tau \;\approx\; \sum_{i=1}^q w_i\,\phi(\xi_i).
$$ 

Applying it on each interval gives

$$
\int_{t_k}^{t_{k+1}} c(\mathbf{x}(t),\mathbf{u}(t))\,dt
\;\approx\;
h_k\sum_{i=1}^q w_i\, c\!\big(\mathbf{x}(t_k+h_k\xi_i),\,\mathbf{u}(t_k+h_k\xi_i)\big).
$$


Summing these window contributions gives a composite approximation of the integral over $[t_0,t_f]$:

$$
\int_{t_0}^{t_f} c(\mathbf{x}(t),\mathbf{u}(t))\,dt
\;\approx\;
\sum_{k=0}^{N-1} h_k \sum_{i=1}^q w_i\, c\!\big(\mathbf{x}(t_k^{(i)}),\mathbf{u}(t_k^{(i)})\big).
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
h_k\sum_{i=1}^q b_i\, \mathbf{f}\!\big(\mathbf{x}(t_k^{(i)}),\mathbf{u}(t_k^{(i)})\big),
$$

where $\{b_i\}$ are the weights used for the ODE. Path constraints $\mathbf{g}(\mathbf{x}(t),\mathbf{u}(t))\le 0$ are imposed at selected nodes $t_k^{(i)}$ in the same spirit. Using the same evaluation points for cost and dynamics keeps the representation coherent: we “pay” running cost and “account” for state changes at the same times.



### How do values at interior points arise? (step functions vs interpolating functions)

Once we select a mesh $t_0<\cdots<t_N$ and interior fractions $\{\xi_i\}_{i=1}^q$ per window $[t_k,t_{k+1}]$, we need $\mathbf{x}(t_k^{(i)})$ and $\mathbf{u}(t_k^{(i)})$ at the evaluation times $t_k^{(i)} := t_k + h_k\xi_i$. These values do not preexist. They come from one of two constructions that align with the standard quadrature taxonomy: **step-function based** and **interpolating-function based** rules.

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

With a quadratic interpolation that includes the midpoint, Simpson’s rule appears in the cost and the Hermite–Simpson relations tie $\mathbf{x}_{k+\frac12}$ to endpoint values and slopes. More generally, **collocation** chooses interior nodes on $[t_k,t_{k+1}]$ (equally spaced gives Newton–Cotes like trapezoid or Simpson; Gaussian points give Gauss, Radau, or Lobatto schemes) and enforces the ODE at those nodes:

$$
\frac{d}{dt}\mathbf{x}(t_k^{(i)})=\mathbf{f}\!\big(\mathbf{x}(t_k^{(i)}),\mathbf{u}(t_k^{(i)}),t_k^{(i)}\big),
$$

with continuity at endpoints. The interior values $\mathbf{x}(t_k^{(i)})$ are **evaluations of the decision polynomials**; $\mathbf{u}(t_k^{(i)})$ follows from the chosen control interpolation (constant, linear, or quadratic). The running cost is evaluated by the same interpolatory quadrature at the same nodes, which keeps “where we pay” aligned with “where we enforce.”

# The Interpolation Problem: Functions from Constraints

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

### How Many Coefficients Do We Need?

Each constraint adds one equation to the system. To find a unique solution, we need the number of unknowns (the $c_n$) to match the number of constraints. Since a degree-$N$ polynomial has $N+1$ coefficients, we need:

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

### Interpolating ODE trajectories (collocation)

We now specialize the interpolation viewpoint to trajectories governed by an ODE

$$
\dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t),\mathbf{u}(t),t).
$$

On each mesh interval $[t_i,t_{i+1}]$ with $h_i:=t_{i+1}-t_i$, choose collocation nodes $\{\xi_j\}_{j=0}^s\subset[0,1]$ and map $t=t_i+h_i\,\xi$. Use the monomial basis $\phi_n(\xi)=\xi^n$ to approximate the (unknown) trajectory by

$$
\mathbf{x}_h(\xi) = \sum_{n=0}^s \mathbf{a}_n\,\xi^n,\quad \mathbf{a}_n\in\mathbb{R}^d.
$$

Differentiating with respect to time (using $\tfrac{d}{dt}=\tfrac{1}{h_i}\tfrac{d}{d\xi}$) gives, at the collocation nodes $\xi_j$,

$$
\dot{\mathbf{x}}_h(t_i+h_i\,\xi_j) 
\;=\; \frac{1}{h_i} \sum_{n=1}^s n\,\mathbf{a}_n\,\xi_j^{\,n-1}.
$$

Collocation enforces that these polynomial slopes match the ODE at the same nodes (this is exactly “interpolation with derivative constraints,” where the slopes come from the ODE):

$$
\frac{1}{h_i} \sum_{n=1}^s n\,\mathbf{a}_n\,\xi_j^{\,n-1}
\;=\; \mathbf{f}\Big( \sum_{n=0}^s \mathbf{a}_n\,\xi_j^{\,n},\ \mathbf{U}_j,\ t_i+h_i\,\xi_j \Big),\quad j=0,\ldots,s.
$$

Endpoint consistency provides linear constraints on the coefficients. If endpoints are included among $\{\xi_j\}$ (Lobatto-type nodes),

$$
\mathbf{x}_i = \mathbf{x}_h(0) = \mathbf{a}_0,\qquad
\mathbf{x}_{i+1} = \mathbf{x}_h(1) = \sum_{n=0}^s \mathbf{a}_n.
$$

Controls can be piecewise-constant/linear or also parameterized by a low-degree polynomial $\mathbf{u}_h(\xi)=\sum_{n=0}^{s_u} \mathbf{b}_n\,\xi^n$ with nodal values $\mathbf{U}_j=\mathbf{u}_h(\xi_j)$. The running cost on $[t_i,t_{i+1}]$ is evaluated with the same nodes and quadrature weights $\{w_j\}$:

$$
\int_{t_i}^{t_{i+1}} c\,dt \;\approx\; h_i\sum_{j=0}^s w_j\, c\big(\mathbf{x}_h(\xi_j),\mathbf{u}_h(\xi_j), t_i+h_i\,\xi_j\big).
$$

In words: pick a small set of points inside the interval, write a low-degree polynomial for $\mathbf{x}(t)$, and enforce that its time-derivative equals the ODE at those points. Endpoint equalities give value constraints; the ODE gives slope constraints. Stitch intervals by continuity, and use the same nodes and weights to evaluate the running cost. This is precisely the interpolation-with-derivatives idea, with derivatives supplied by the ODE.

```{prf:example} Two worked collocation examples (solve the local linear system)
Fix a window $[t_i,t_{i+1}]$ with step $h_i=t_{i+1}-t_i$ and map it to $\xi\in[0,1]$ via $t=t_i+h_i\,\xi$. Approximate the (unknown) scalar state by a degree-1 polynomial

$$
x_h(\xi)=a_0+a_1\,\xi,\qquad \dot{x}_h(t)=\frac{1}{h_i}\,a_1.
$$

We will impose a value constraint at the left endpoint $x_h(0)=x_i$ and one collocation condition $\dot{x}_h(t_i+h_i\xi_\star)=f\big(x_h(\xi_\star),u(\xi_\star),t_i+h_i\xi_\star\big)$ at a single node $\xi_\star$. This produces a tiny linear system for the coefficients $a_0,a_1$; evaluating $x_h(1)$ gives the step update.

- Forward (explicit) Euler: choose the left-endpoint node $\xi_\star=0$ and piecewise-constant control $u(\xi_\star)=u_i$. The constraints are

$$
\underbrace{\begin{bmatrix}1 & 0\\ 0 & 1/h_i\end{bmatrix}}_{\text{linear in }(a_0,a_1)}
\begin{bmatrix}a_0\\ a_1\end{bmatrix}
=
\begin{bmatrix}x_i\\ f(x_i,u_i,t_i)\end{bmatrix}.
$$

Hence $a_0=x_i$ and $a_1=h_i f(x_i,u_i,t_i)$, so

$$
x_{i+1}:=x_h(1)=x_i+h_i f(x_i,u_i,t_i),
$$

which is the forward Euler update recovered as one-point collocation.

- Backward (implicit) Euler: choose the right-endpoint node $\xi_\star=1$ and use $u(\xi_\star)=u_{i+1}$. Then

$$
\frac{1}{h_i}a_1=f\big(x_h(1),u_{i+1},t_{i+1}\big)=f(x_{i+1},u_{i+1},t_{i+1}),\qquad a_0=x_i,
$$

so $x_{i+1}=x_i+h_i f(x_{i+1},u_{i+1},t_{i+1})$, the backward Euler relation. For the scalar linear ODE $\dot{x}=\lambda x$ this becomes a 2×2 linear system in $(a_0,a_1)$ that yields the closed form

$$
x_{i+1}=\frac{1}{1-h_i\lambda}\,x_i.
$$

```

# Applying the recipe: concrete transcriptions

The mesh and interior nodes are the common scaffold. What distinguishes one transcription from another is how we obtain values at those nodes and how we approximate the two integrals that appear implicitly and explicitly: the integral of the running cost and the integral that carries the state forward. In other words, we now commit to two design choices that mirror the previous section: a finite representation for $\mathbf{x}(t)$ and $\mathbf{u}(t)$ over each interval $[t_i,t_{i+1}]$, and a quadrature rule whose nodes and weights are used consistently for both cost and dynamics. The result is always a sparse nonlinear program; the differences are in where we sample and how we tie samples together.

Below, each transcription should be read as “same grid, same interior points, same evaluations for cost and physics,” with only the local representation changing.

*Euler (step functions; rectangle rule).*
Here we treat $\mathbf{u}$ as piecewise constant and evaluate both cost and dynamics at the left endpoint. Interior values are not independent variables; they are whatever the step-function model implies. This is the simplest way to align “what we pay” with “how we advance,” and it reproduces the forward Euler update inside the NLP.

```{prf:definition} Euler Transcription (Discrete Bolza NLP)
Let $t_0<\cdots<t_N$ with $h_i:=t_{i+1}-t_i$. Decision variables are $\{\mathbf{x}_i\}_{i=0}^N$, $\{\mathbf{u}_i\}_{i=0}^{N-1}$. Given running cost $c$, terminal cost $c_T$, dynamics $\mathbf{f}$, and path constraint $\mathbf{g}\le \mathbf{0}$, solve

$$
\begin{aligned}
\min_{\{\mathbf{x}_i,\mathbf{u}_i\}}\ & c_T(\mathbf{x}_N)\; +\; \sum_{i=0}^{N-1} h_i\, c(\mathbf{x}_i,\mathbf{u}_i)\\
\text{s.t.}\ & \mathbf{x}_{i+1}-\mathbf{x}_i - h_i\,\mathbf{f}(\mathbf{x}_i,\mathbf{u}_i) = \mathbf{0},\quad i=0,\ldots,N-1,\\
& \mathbf{g}(\mathbf{x}_i,\mathbf{u}_i) \le \mathbf{0},\quad i=0,\ldots,N-1,\\
& \mathbf{x}_{\min} \le \mathbf{x}_i \le \mathbf{x}_{\max},\ \ \mathbf{u}_{\min} \le \mathbf{u}_i \le \mathbf{u}_{\max},\\
& \mathbf{x}_0 = \mathbf{x}(t_0).
\end{aligned}
$$
```

*Trapezoidal collocation (linear interpolation; end nodes).*
Now we let $\mathbf{x}$ vary linearly over the interval and evaluate both cost and dynamics at the two endpoints with equal weights. This matches the trapezoid rule in the objective and the trapezoidal defect in the dynamics, so cost accumulation and state accounting occur at the same two places.

```{prf:definition} Trapezoidal Collocation 
Let $t_0<\cdots<t_N$ with $h_i:=t_{i+1}-t_i$. Decision variables are $\{\mathbf{x}_i\}_{i=0}^N$, $\{\mathbf{u}_i\}_{i=0}^N$. Solve

$$
\begin{aligned}
\min_{\{\mathbf{x}_i,\mathbf{u}_i\}}\ & c_T(\mathbf{x}_N)\; +\; \sum_{i=0}^{N-1} \tfrac{h_i}{2}\,\Big[c(\mathbf{x}_i,\mathbf{u}_i)+c(\mathbf{x}_{i+1},\mathbf{u}_{i+1})\Big]\\
\text{s.t.}\ & \mathbf{x}_{i+1}-\mathbf{x}_i - \tfrac{h_i}{2}\Big[\mathbf{f}(\mathbf{x}_i,\mathbf{u}_i)+\mathbf{f}(\mathbf{x}_{i+1},\mathbf{u}_{i+1})\Big] = \mathbf{0},\quad i=0,\ldots,N-1,\\
& \mathbf{g}(\mathbf{x}_i,\mathbf{u}_i) \le \mathbf{0},\ \ \mathbf{g}(\mathbf{x}_{i+1},\mathbf{u}_{i+1}) \le \mathbf{0},\\
& \mathbf{x}_{\min} \le \mathbf{x}_i \le \mathbf{x}_{\max},\ \ \mathbf{u}_{\min} \le \mathbf{u}_i \le \mathbf{u}_{\max},\\
& \mathbf{x}_0 = \mathbf{x}(t_0).
\end{aligned}
$$
```

*Hermite–Simpson (quadratic interpolation; midpoint included).*  
Adding a midpoint enriches the local model from linear to quadratic. Practically, this buys two things at once: (i) higher accuracy at low cost—Hermite–Simpson delivers fourth‑order state accuracy with a single interior node, so fewer intervals are needed for a given error; and (ii) tight alignment between physics and objective—Simpson’s rule in the cost and the Hermite–Simpson defect in the dynamics use the same three evaluation sites (left, middle, right). Introducing midpoint variables makes the interior state explicit so the ODE can be matched there, which reduces collocation defects and typically improves conditioning compared to trapezoid on smooth problems. In short, we evaluate and enforce at the same places we pay, and the extra node yields a noticeable accuracy boost without a large increase in variables.

```{prf:definition} Hermite–Simpson Transcription 
Let $t_0<\cdots<t_N$ with $h_i:=t_{i+1}-t_i$ and midpoints $t_{i+\frac12}$. Decision variables are $\{\mathbf{x}_i\}_{i=0}^N$, $\{\mathbf{u}_i\}_{i=0}^N$, plus midpoint variables $\{\mathbf{x}_{i+\frac12},\mathbf{u}_{i+\frac12}\}_{i=0}^{N-1}$. Solve

$$
\begin{aligned}
\min\ & c_T(\mathbf{x}_N)\; +\; \sum_{i=0}^{N-1} \tfrac{h_i}{6}\Big[ c(\mathbf{x}_i,\mathbf{u}_i) + 4\,c(\mathbf{x}_{i+\frac12},\mathbf{u}_{i+\frac12}) + c(\mathbf{x}_{i+1},\mathbf{u}_{i+1}) \Big]\\
\text{s.t.}\ & \mathbf{x}_{i+1}-\mathbf{x}_i - \tfrac{h_i}{6}\Big[\mathbf{f}(\mathbf{x}_i,\mathbf{u}_i) + 4\,\mathbf{f}(\mathbf{x}_{i+\frac12},\mathbf{u}_{i+\frac12}) + \mathbf{f}(\mathbf{x}_{i+1},\mathbf{u}_{i+1})\Big] = \mathbf{0},\\
& \mathbf{x}_{i+\frac12} - \tfrac{\mathbf{x}_i+\mathbf{x}_{i+1}}{2} - \tfrac{h_i}{8}\Big[\mathbf{f}(\mathbf{x}_i,\mathbf{u}_i) - \mathbf{f}(\mathbf{x}_{i+1},\mathbf{u}_{i+1})\Big] = \mathbf{0},\\
& \mathbf{g}(\mathbf{x}_i,\mathbf{u}_i) \le \mathbf{0},\ \ \mathbf{g}(\mathbf{x}_{i+\frac12},\mathbf{u}_{i+\frac12}) \le \mathbf{0},\ \ \mathbf{g}(\mathbf{x}_{i+1},\mathbf{u}_{i+1}) \le \mathbf{0},\\
& \mathbf{x}_{\min} \le \mathbf{x}_i,\mathbf{x}_{i+\frac12} \le \mathbf{x}_{\max},\ \ \mathbf{u}_{\min} \le \mathbf{u}_i,\mathbf{u}_{i+\frac12} \le \mathbf{u}_{\max},\\
& \mathbf{x}_0 = \mathbf{x}(t_0),\quad i=0,\ldots,N-1.
\end{aligned}
$$
```

*RK4 transcription (integrator stages as algebraic variables).*
Runge–Kutta methods approximate the interval integral by evaluating the vector field at a set of staged points and combining them with fixed weights. Introducing the stages as variables brings the usual RK4 update inside the NLP, and again the evaluation sites used to advance the state are the same places where we evaluate the running cost.

```{prf:definition} RK4 Transcription 
Let $t_0<\cdots<t_N$ with $h_i:=t_{i+1}-t_i$. Decision variables are $\{\mathbf{x}_i\}_{i=0}^N$, $\{\mathbf{u}_i\}_{i=0}^N$, and stage vectors $\{\mathbf{s}^{(1)}_i,\mathbf{s}^{(2)}_i,\mathbf{s}^{(3)}_i,\mathbf{s}^{(4)}_i\}_{i=0}^{N-1}$ and midpoint controls $\{\bar{\mathbf{u}}_i\}$. Define the RK stages

$$
\begin{aligned}
\mathbf{s}^{(1)}_i &= \mathbf{f}(\mathbf{x}_i,\mathbf{u}_i),\\
\mathbf{s}^{(2)}_i &= \mathbf{f}\!\big(\mathbf{x}_i + \tfrac{h_i}{2}\,\mathbf{s}^{(1)}_i,\ \bar{\mathbf{u}}_i\big),\\
\mathbf{s}^{(3)}_i &= \mathbf{f}\!\big(\mathbf{x}_i + \tfrac{h_i}{2}\,\mathbf{s}^{(2)}_i,\ \bar{\mathbf{u}}_i\big),\\
\mathbf{s}^{(4)}_i &= \mathbf{f}\!\big(\mathbf{x}_i + h_i\,\mathbf{s}^{(3)}_i,\ \mathbf{u}_{i+1}\big).
\end{aligned}
$$

Solve

$$
\begin{aligned}
\min\ & c_T(\mathbf{x}_N)\; +\; \sum_{i=0}^{N-1} \tfrac{h_i}{6}\Big[ c(\mathbf{x}_i,\mathbf{u}_i) + 4\,c\!\big(\mathbf{x}_i + \tfrac{h_i}{2}\,\mathbf{s}^{(2)}_i,\ \bar{\mathbf{u}}_i\big) + c(\mathbf{x}_{i+1},\mathbf{u}_{i+1}) \Big]\\
\text{s.t.}\ & \mathbf{x}_{i+1}-\mathbf{x}_i - \tfrac{h_i}{6}\Big[\mathbf{s}^{(1)}_i + 2\mathbf{s}^{(2)}_i + 2\mathbf{s}^{(3)}_i + \mathbf{s}^{(4)}_i\Big] = \mathbf{0},\\
& \mathbf{g}(\mathbf{x}_i,\mathbf{u}_i) \le \mathbf{0},\ \ \mathbf{g}(\mathbf{x}_{i+1},\mathbf{u}_{i+1}) \le \mathbf{0},\ \ \mathbf{g}\!\big(\mathbf{x}_i + \tfrac{h_i}{2}\,\mathbf{s}^{(2)}_i,\ \bar{\mathbf{u}}_i\big) \le \mathbf{0},\\
& \mathbf{x}_{\min} \le \mathbf{x}_i \le \mathbf{x}_{\max},\ \ \mathbf{u}_{\min} \le \mathbf{u}_i,\bar{\mathbf{u}}_i \le \mathbf{u}_{\max},\\
& \mathbf{x}_0 = \mathbf{x}(t_0),\quad i=0,\ldots,N-1.
\end{aligned}
$$
```

### Choosing among them

All four transcriptions follow the same organizing idea from earlier: pick evaluation points, translate both integrals into weighted sums at those points, and couple those evaluations to a finite trajectory representation. Euler uses step functions and a single point per interval. Trapezoid and Hermite–Simpson use low-degree polynomials and endpoint or midpoint nodes, which brings collocation into play. RK4 uses staged vector-field samples in place of an explicit interpolant. In every case the mesh can be nonuniform and refined locally, and the resulting NLP keeps a banded sparsity that solvers exploit.

## A brief note on reconstruction

Once the finite problem is solved, the discrete solution $\{\mathbf{x}_i,\mathbf{u}_i\}$ and any interior values determine a continuous-time approximation on each interval. Linear reconstruction matches Euler and trapezoid; cubic Hermite for $\mathbf{x}(t)$ with quadratic $\mathbf{u}(t)$ matches Hermite–Simpson; for RK4, piecewise-constant or piecewise-linear controls on the half-steps are consistent with the staged evaluations. The reconstruction should mirror the choice made in the transcription so that what is plotted reflects what was optimized.

# Example: Compressor Surge Problem 

Compressors are mechanical devices used to increase the pressure of a gas by reducing its volume. They are found in many industrial settings, from natural gas pipelines to jet engines. However, compressors can suffer from a dangerous phenomenon called "surge" when the gas flow through the compressor falls too much below its design capacity. This can happen under different circumstances such as: 

- In a natural gas pipeline system, when there is less customer demand (e.g., during warm weather when less heating is needed) the flow through the compressor lowers.
- In a jet engine, when the pilot reduces thrust during landing, less air flows through the engine's compressors.
- In factory, the compressor might be connected through some equipment downstream via a valve. Closing it partially restricts gas flow, similar to pinching a garden hose, and can lead to compressor surge.

As the gas flow decreases, the compressor must work harder to maintain a steady flow. If the flow becomes too low, it can lead to a "breakdown": a phenomenon similar to an airplane stalling at low speeds or high angles of attack. In a compressor, when this breakdown occurs the gas briefly flows backward instead of moving forward, which in turns can cause violent oscillations in pressure which can damage the compressor and the equipments depending on it. One way to address this problem is by installing a close-coupled valve (CCV), which is a device connected at the output of the compressor to quickly modulate the flow. Our aim is not to devise a optimal control approach to ensure that the compressor does not experience a surge by operating this CCV appropriately. 

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

One possible way to pose the problem {cite}`Grancharova2012` is by penalizing for the deviations to the setpoints using a quadratic penalty in the instantenous cost function as well as in the terminal one. Furthermore, we also penalize for taking large actions (which are energy hungry, and potentially unsafe) within the integral term. The idea of penalzing for deviations throughout is natural way of posing the problem when solving it via single shooting. Another alternative which we will explore below is to set the desired setpoint as a hard terminal constraint. 

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

In the experiment below, we choose the setpoint $\mathbf{x}^* = [0.40, 0.60]^T$ as it corresponds to to an unstable equilibrium point. If we were to run the system without applying any control, we would see that the system starts to oscillate. 

```{code-cell} ipython3
:tags: [hide-input]
:load: code/compressor_surge_single_shooting.py
```

## Solution by Trapezoidal Collocation

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

You can try to vary the number of collocation points in the code an observe how the state trajectory progressively matches the ground thruth (the line denoted "integrated solution"). Note that this version of the code also lacks bound constraints on the variable $x_2$ to ensure a minimum pressure, as we did earlier. Consider this a good exercise for you to try on your own. 

## System Identification as Trajectory Optimization (Compressor Surge)

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

<!-- ## Parameterization of $f$ and Neural ODEs
In our compressor surge problem, we were provided with a physically-motivated form for the function $f$. This set of equations was likely derived by scientists with deep knowledge of the physical phenomena at play (i.e., gas compression). However, in complex systems, the underlying physics might not be well understood or too complicated to model explicitly. In such cases, we might opt for a more flexible, data-driven approach.

Instead of specifying a fixed structure for $f$, we could use a "black box" model such as a neural network to learn the dynamics directly from data. 
The optimization problem remains conceptually the same as that of parameter identification. However, we are now optimizing over the parameters of the neural network that defines $f$.

Another possibility is to blend the two approaches and use a grey-box model. In this approach, we typically use a physics-informed parameterization which we then supplement with a black-box model to account for the discrepancies in the observations. Mathematically, this can be expressed as:

$$
\dot{\mathbf{x}}(t) = f_{\text{physics}}(\mathbf{x}, t; \boldsymbol{\theta}_{\text{physics}}) + f_{\text{NN}}(\mathbf{x}, t; \boldsymbol{\theta}_{\text{NN}})
$$

where $f_{\text{physics}}$ is the physics-based model with parameters $\boldsymbol{\theta}_{\text{physics}}$, and $f_{\text{NN}}$ is a neural network with parameters $\boldsymbol{\theta}_{\text{NN}}$ that captures unmodeled dynamics.

We then learn the parameters of the black-box model in tandem with the output of the given physics-based model. You can think of the combination of these two models as a neural network of its own, with the key difference being that one subnetwork (the physics-based one) has frozen weights (non-adjustable parameters).

This approach is easy to implement using automatic differentiation techniques and allows us to leverage prior knowledge to make the data-driven modelling more sample efficient. From a learning perspective, it amounts to providing inductive biases to make learning more efficient and to generalize better.  -->
