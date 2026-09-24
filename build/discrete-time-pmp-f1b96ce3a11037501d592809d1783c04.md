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
# Adjoints and the Discrete-Time Pontryagin Principle

Finite-horizon optimal control produces a structured nonlinear program and the
KKT conditions characterize its local solutions. How do those multipliers
organize themselves when the equality constraints are a forward dynamical
recursion?

If we take the Bolza formulation of the DOCP and apply the KKT conditions directly, we obtain an optimization system with many multipliers and constraints. Written in raw form, it looks like any other nonlinear program. But in control, this structure has a long history and a name of its own: the **Pontryagin principle**. In fact, the discrete-time version can be seen as the structured KKT system that results from introducing multipliers for the dynamics and collecting terms stage by stage.

We work with the Bolza program

$$
\begin{aligned}
\min_{\{\mathbf{x}_t,\mathbf{u}_t\}} \quad & c_T(\mathbf{x}_T)\;+\;\sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t) \\
\text{s.t.}\quad & \mathbf{x}_{t+1}=\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t),\quad t=1,\dots,T-1,\\
& \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t)\le \mathbf{0},\quad \mathbf{u}_t\in \mathcal{U}_t,\\
& \mathbf{h}(\mathbf{x}_T)=\mathbf{0}\quad\text{(optional terminal equalities)}.
\end{aligned}
$$

Introduce **costates** $\boldsymbol{\lambda}_{t+1}\in\mathbb{R}^n$ for the dynamics, multipliers $\boldsymbol{\mu}_t\ge \mathbf{0}$ for path inequalities, and $\boldsymbol{\nu}$ for terminal equalities. The Lagrangian is

$$
\mathcal{L}
= c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t)
+ \sum_{t=1}^{T-1} \boldsymbol{\lambda}_{t+1}^\top\!\big(\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)-\mathbf{x}_{t+1}\big)
+ \sum_{t=1}^{T-1} \boldsymbol{\mu}_t^\top \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t)
+ \boldsymbol{\nu}^\top \mathbf{h}(\mathbf{x}_T).
$$

It is convenient to package the stagewise terms in a **Hamiltonian**

$$
H_t(\mathbf{x}_t,\mathbf{u}_t,\boldsymbol{\lambda}_{t+1},\boldsymbol{\mu}_t)
:= c_t(\mathbf{x}_t,\mathbf{u}_t)
+ \boldsymbol{\lambda}_{t+1}^\top \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)
+ \boldsymbol{\mu}_t^\top \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t).
$$

```{admonition} Check your understanding
:class: tip
Verify that the Hamiltonian $H_t$ collects exactly the terms from the Lagrangian that involve stage $t$: the stage cost, the dynamics constraint (weighted by the next costate), and the path constraints (weighted by their multipliers). Why does $\boldsymbol{\lambda}_{t+1}$ appear rather than $\boldsymbol{\lambda}_t$?
```

Then

$$
\mathcal{L} = c_T(\mathbf{x}_T)+\boldsymbol{\nu}^\top \mathbf{h}(\mathbf{x}_T)
+ \sum_{t=1}^{T-1}\Big[H_t(\mathbf{x}_t,\mathbf{u}_t,\boldsymbol{\lambda}_{t+1},\boldsymbol{\mu}_t)
- \boldsymbol{\lambda}_{t+1}^\top \mathbf{x}_{t+1}\Big].
$$

## Necessary conditions

How do state feasibility, costate recursion, control stationarity, and
complementarity emerge from the stagewise Lagrangian?

```{note} Gradient convention
Throughout this section, we use the **denominator layout** (gradient layout) convention:
- $\nabla_{\mathbf{x}} f(\mathbf{x})$ produces a **column vector** (gradient)
- $\frac{\partial f}{\partial \mathbf{x}}$ produces the Jacobian matrix
- For scalar functions: $\nabla_{\mathbf{x}} f = \left(\frac{\partial f}{\partial \mathbf{x}}\right)^\top$

This is the standard convention in optimization and control theory.
```

Taking first-order variations and collecting terms gives the discrete-time adjoint system, control stationarity, and complementarity. At a local minimum $\{\mathbf{x}_t^\star,\mathbf{u}_t^\star\}$ with multipliers $\{\boldsymbol{\lambda}_t^\star,\boldsymbol{\mu}_t^\star,\boldsymbol{\nu}^\star\}$:

**State dynamics (primal feasibility)**

$$
\mathbf{x}_{t+1}^\star=\mathbf{f}_t(\mathbf{x}_t^\star,\mathbf{u}_t^\star),\quad t=1,\dots,T-1.
$$

**Costate recursion (backward "adjoint" equation)**

$$
\boldsymbol{\lambda}_t^\star
= \nabla_{\mathbf{x}} H_t\big(\mathbf{x}_t^\star,\mathbf{u}_t^\star,\boldsymbol{\lambda}_{t+1}^\star,\boldsymbol{\mu}_t^\star\big)
= \nabla_{\mathbf{x}} c_t(\mathbf{x}_t^\star,\mathbf{u}_t^\star)
+ \big[\nabla_{\mathbf{x}} \mathbf{f}_t(\mathbf{x}_t^\star,\mathbf{u}_t^\star)\big]^\top \boldsymbol{\lambda}_{t+1}^\star
+ \big[\nabla_{\mathbf{x}} \mathbf{g}_t(\mathbf{x}_t^\star,\mathbf{u}_t^\star)\big]^\top \boldsymbol{\mu}_t^\star,
$$

with the **terminal condition**

$$
\boldsymbol{\lambda}_T^\star
= \nabla_{\mathbf{x}} c_T(\mathbf{x}_T^\star) + \big[\nabla_{\mathbf{x}} \mathbf{h}(\mathbf{x}_T^\star)\big]^\top \boldsymbol{\nu}^\star
\quad\text{(and \(\boldsymbol{\nu}^\star=\mathbf{0}\) if there are no terminal equalities).}
$$

**Control stationarity (first-order optimality in $\mathbf{u}_t$)**
If $\mathcal{U}_t=\mathbb{R}^m$ (no explicit set constraint), then

$$
\nabla_{\mathbf{u}} H_t\big(\mathbf{x}_t^\star,\mathbf{u}_t^\star,\boldsymbol{\lambda}_{t+1}^\star,\boldsymbol{\mu}_t^\star\big)=\mathbf{0}.
$$

If $\mathcal{U}_t$ imposes bounds or a convex set, the condition becomes the **variational inequality**

$$
\mathbf{0}\in \nabla_{\mathbf{u}} H_t(\cdot)\;+\;N_{\mathcal{U}_t}(\mathbf{u}_t^\star),
$$

where $N_{\mathcal{U}_t}(\cdot)$ is the normal cone to $\mathcal{U}_t$. For simple box bounds, this reduces to standard KKT sign and complementarity conditions on the components of $\mathbf{u}_t^\star$.

**Path-constraint multipliers (primal/dual feasibility and complementarity)**

$$
\mathbf{g}_t(\mathbf{x}_t^\star,\mathbf{u}_t^\star)\le \mathbf{0},\quad
\boldsymbol{\mu}_t^\star\ge \mathbf{0},\quad
\mu_{t,i}^\star\, g_{t,i}(\mathbf{x}_t^\star,\mathbf{u}_t^\star)=0\quad \text{for all }i,t.
$$

**Terminal equalities (if present)**

$$
\mathbf{h}(\mathbf{x}_T^\star)=\mathbf{0}.
$$

The triplet "forward state, backward costate, control stationarity" is the discrete-time Euler–Lagrange system tailored to control with dynamics. It is the same KKT logic as before, but organized stagewise through the Hamiltonian.

```{prf:proposition} Discrete-time Pontryagin necessary conditions (summary)
At a local minimum of the DOCP

$$
\min_{\{\mathbf{x}_t,\mathbf{u}_t\}}\ c_T(\mathbf{x}_T)+\sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t)
\quad\text{s.t.}\quad \mathbf{x}_{t+1}=\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t),\ \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t)\le \mathbf{0},\ \mathbf{h}(\mathbf{x}_T)=\mathbf{0},
$$

there exist multipliers $\{\boldsymbol{\lambda}_{t+1}\}$, $\{\boldsymbol{\mu}_t\ge\mathbf{0}\}$, and (if present) $\boldsymbol{\nu}$ such that, for $t=1,\dots,T-1$:

- State dynamics: $\ \mathbf{x}_{t+1}=\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)$.
- Backward costate recursion:

  $$
  \boldsymbol{\lambda}_t = \nabla_{\mathbf{x}} c_t(\mathbf{x}_t,\mathbf{u}_t)
  + \big[\nabla_{\mathbf{x}} \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)\big]^\top \boldsymbol{\lambda}_{t+1}
  + \big[\nabla_{\mathbf{x}} \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t)\big]^\top \boldsymbol{\mu}_t.
  $$
  
- Terminal condition: $\ \boldsymbol{\lambda}_T = \nabla_{\mathbf{x}} c_T(\mathbf{x}_T) + \big[\nabla_{\mathbf{x}} \mathbf{h}(\mathbf{x}_T)\big]^\top \boldsymbol{\nu}$.
- Control stationarity (unconstrained control): $\ \nabla_{\mathbf{u}} H_t(\cdot)=\mathbf{0}$; with a convex control set $\mathcal{U}_t$, $\ \mathbf{0}\in \nabla_{\mathbf{u}} H_t(\cdot)+N_{\mathcal{U}_t}(\mathbf{u}_t)$.
- Path inequalities: $\ \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t)\le \mathbf{0}$, $\ \boldsymbol{\mu}_t\ge\mathbf{0}$, and complementarity $\ \mu_{t,i}\,g_{t,i}(\mathbf{x}_t,\mathbf{u}_t)=0$ for all $i$.
- Terminal equalities (if present): $\ \mathbf{h}(\mathbf{x}_T)=\mathbf{0}$.

Here $H_t(\mathbf{x}_t,\mathbf{u}_t,\boldsymbol{\lambda}_{t+1},\boldsymbol{\mu}_t):=c_t(\mathbf{x}_t,\mathbf{u}_t)+\boldsymbol{\lambda}_{t+1}^\top\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)+\boldsymbol{\mu}_t^\top\mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t)$ is the stage Hamiltonian.
```

**Recap.** The discrete-time Pontryagin principle is the KKT system for trajectory optimization, organized to exploit temporal structure. It has a forward-backward decomposition: states propagate forward through the dynamics, while costates propagate backward through the adjoint equation. The Hamiltonian $H_t$ packages together the stage cost, the dynamics (weighted by the next costate), and any path constraints (weighted by their multipliers). Control stationarity says that optimal controls minimize the Hamiltonian at each stage. Complementarity ensures that only binding constraints carry nonzero multipliers. This structure underlies both analytical solution methods (such as LQR) and numerical algorithms (such as the adjoint method for gradient computation).

## The adjoint equation as reverse accumulation

The costate equations characterize stationarity, but can the same backward
recursion compute every control derivative with one reverse sweep?

Optimization needs sensitivities. In trajectory problems we adjust decisions (controls or parameters) to reduce an objective while respecting dynamics and constraints. First‑order methods in the unconstrained case (e.g., gradient descent, L‑BFGS, Adam) require the gradient of the objective with respect to all controls, and constrained methods (SQP, interior‑point) require gradients of the Lagrangian, i.e., of costs and constraints. The discrete‑time adjoint equations provide these derivatives in a way that scales to long horizons and many decision variables.

Consider

$$
J = c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t),
\qquad \mathbf{x}_{t+1}=\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t).
$$

A single forward rollout computes and stores the trajectory $\mathbf{x}_{1:T}$. A single backward sweep then applies the reverse‑mode chain rule stage by stage.

```{admonition} Before reading on
:class: tip
Try applying the chain rule to compute how a perturbation to the state at time $t$ affects the total cost $J$. Start from $\partial J / \partial \mathbf{x}_T$ and work backward. What pattern emerges?
```

Defining the costate by

$$
\boldsymbol{\lambda}_T = \nabla_{\mathbf{x}} c_T(\mathbf{x}_T),\qquad
\boldsymbol{\lambda}_t = \nabla_{\mathbf{x}} c_t(\mathbf{x}_t,\mathbf{u}_t) + \big[\nabla_{\mathbf{x}} \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)\big]^\top \boldsymbol{\lambda}_{t+1},\quad t=T-1,\dots,1,
$$

yields exactly the discrete‑time adjoint (PMP) recursion.

```{admonition} Check your understanding
:class: tip
What terms would you expect to appear in the gradient $\nabla_{\mathbf{u}_t} J$? The control $\mathbf{u}_t$ affects $J$ in two ways: directly through the stage cost $c_t$, and indirectly by changing the next state $\mathbf{x}_{t+1}$. How should these contributions combine?
```

The gradient with respect to each control follows from the same reverse pass:

$$
\nabla_{\mathbf{u}_t} J = \nabla_{\mathbf{u}} c_t(\mathbf{x}_t,\mathbf{u}_t) + \big[\nabla_{\mathbf{u}} \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)\big]^\top \boldsymbol{\lambda}_{t+1}.
$$

This reverse accumulation produces every control gradient with one forward
rollout and one backward adjoint pass. The costate
$\boldsymbol{\lambda}_t$ measures the marginal effect of perturbing the state
at time $t$ on the total objective. Each control gradient combines a direct
contribution from $c_t$ with an indirect contribution through the next state.
Backpropagation through an unrolled dynamical system performs the same
calculation.

Finite differences instead perturb one decision at a time and rerun the
system. They require on the order of $p$ rollouts for $p=(T-1)m$ control
variables and introduce a finite-difference step size. Forward-mode
sensitivities propagate a separate Jacobian-vector product for each parameter
direction, so their work also scales with $p$. Reverse mode propagates one
costate vector backward and reads all partial derivatives from that sweep. For
a scalar objective, this replaces one rollout per parameter by one
forward-backward pass, at the cost of storing or checkpointing the state
trajectory.

The adjoint recursion is therefore the reverse-mode derivative of the
trajectory objective. States carry the nominal trajectory forward, costates
carry its sensitivity backward, and the local control derivatives combine the
two at each stage.

## Summary and Outlook

The KKT conditions organize local optimality into primal feasibility,
stationarity, dual feasibility, and complementarity. Applied along a trajectory,
they give the discrete-time Pontryagin principle. States propagate forward,
costates propagate backward, and the Hamiltonian supplies the local control
stationarity condition. The same backward recursion computes all control
gradients with one reverse pass.

The necessary conditions do not choose how states and actions should be exposed
to a numerical solver. Should the states remain decision variables, be
eliminated by forward simulation, or appear only at segment boundaries?
[Numerical trajectory optimization](numerical-trajectory-optimization.md)
compares those formulations.

## Exercises

:::{exercise} Costate recursion as sensitivity
:label: ex-trajectories-costate

Consider an unconstrained DOCP with objective $J = c_T(x_T) + \sum_{t=1}^{T-1} c_t(x_t, u_t)$ and dynamics $x_{t+1} = f_t(x_t, u_t)$.

**(a)** Define the value-to-go from time $t$ as $V_t(x_t) = \min_{u_{t:T-1}} \left[ c_T(x_T) + \sum_{s=t}^{T-1} c_s(x_s, u_s) \right]$. Show that at an optimal trajectory, $\nabla_{x_t} V_t(x_t^\star) = \lambda_t^\star$, where $\lambda_t$ is the costate.

**(b)** Interpret the costate economically: what does $\lambda_t$ measure?

:::

:::{solution} ex-trajectories-costate
:class: dropdown

By definition, $V_t(x_t)$ is the minimum future cost starting from $x_t$. At the optimal trajectory, the envelope theorem gives $\nabla_{x_t} V_t = \lambda_t$, the marginal value of the state. Economically, $\lambda_t$ measures how much the optimal cost would decrease if we could perturb the state $x_t$ by a small amount—it is the "shadow price" of the state at time $t$.
:::

---

:::{exercise} Adjoint gradient verification
:label: ex-trajectories-adjoint

For the system $x_{t+1} = x_t^2 + u_t$ with $x_1 = 0.5$, $T = 4$, and objective $J = x_4$:

**(a)** Compute the gradient $\nabla_{u} J$ using finite differences (perturb each $u_t$ by $\epsilon = 10^{-5}$).

**(b)** Compute the gradient using the adjoint method: first simulate forward to get $x_{1:4}$, then propagate costates backward using $\lambda_4 = 1$ and $\lambda_t = 2x_t \lambda_{t+1}$.

**(c)** Verify that the two methods give the same answer. Which is more efficient for large $T$?

:::

:::{solution} ex-trajectories-adjoint
:class: dropdown

For controls $u = [0, 0, 0]$: forward simulation gives $x_1 = 0.5$, $x_2 = 0.25$, $x_3 = 0.0625$, $x_4 = 0.00390625$.

Adjoint: $\lambda_4 = 1$, $\lambda_3 = 2(0.0625)(1) = 0.125$, $\lambda_2 = 2(0.25)(0.125) = 0.0625$, $\lambda_1 = 2(0.5)(0.0625) = 0.0625$.

Control gradients: $\nabla_{u_t} J = \lambda_{t+1}$, so $\nabla_u J = [\lambda_2, \lambda_3, \lambda_4] = [0.0625, 0.125, 1.0]$.

Finite differences should match. The adjoint is $O(T)$ work regardless of the number of controls; finite differences require $O(T \cdot m)$ rollouts for $m$-dimensional control.
:::

---
