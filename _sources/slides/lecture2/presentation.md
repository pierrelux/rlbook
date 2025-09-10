---
marp: true
theme: mila
css: mila.css
paginate: true
class: lead
fragment: false
math: mathjax

style: |
  .two-columns {
    display: flex;
    gap: 15px;
    margin-top: 2px;
  }
  .column {
    flex: 1;
    border: 1px solid #ccc;
    padding: 15px;
    border-radius: 8px;
    background-color: #f9f9f9;
    font-size: 0.9em;
  }
---

# Trajectory Optimization: Discrete & Continuous Time
## Lecture 2: September 10, 2025
Pierre-Luc Bacon  
Université de Montréal

---

## Discrete-Time OCP (Bolza Form)

- Problem data: dynamics $\mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t)$, initial $\mathbf{x}_1$
- Costs: stage $c_t(\mathbf{x}_t,\mathbf{u}_t)$, terminal $c_T(\mathbf{x}_T)$

$$
\begin{aligned}
    \text{minimize} \quad & c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t) \\
    \text{subject to} \quad & \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t) \\
                            & \mathbf{g}_t(\mathbf{x}_t,\mathbf{u}_t) \leq \mathbf{0} \\
                            & \mathbf{x}_{\text{min}} \leq \mathbf{x}_t \leq \mathbf{x}_{\text{max}} \\
                            & \mathbf{u}_{\text{min}} \leq \mathbf{u}_t \leq \mathbf{u}_{\text{max}} \\
    \text{given} \quad & \mathbf{x}_1 = \mathbf{x}_0 \enspace .
\end{aligned}
$$

- Variants: Lagrange (no terminal), Mayer (terminal only). Reduce Bolza→Mayer via state augmentation.

---

## Simultaneous (Direct Transcription)

- Keep all $\mathbf{x}_{1:T},\mathbf{u}_{1:T-1}$; enforce dynamics as equalities
- Stack variables $\mathbf{z}=[\mathbf{x}_{1:T},\mathbf{u}_{1:T-1}]$

$$
\begin{aligned}
\min_{\mathbf{z}}\ & c_T(\mathbf{x}_T)+\sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t) \\
\text{s.t.}\ & \mathbf{x}_{t+1}-\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)=\mathbf{0},\ t=1{:}T-1, \\
& g_i(\mathbf{x}_k,\mathbf{u}_k)=0,\ h_i(\mathbf{x}_k,\mathbf{u}_k)\le 0\ (k\in K_i).
\end{aligned}
$$

- Structure: block bi-diagonal Jacobian; sparse KKT; good with state/path constraints

---

## Single Shooting (Recursive Elimination)

- Eliminate states by rollout: $\boldsymbol{\phi}_{t+1}=\mathbf{f}_t(\boldsymbol{\phi}_t,\mathbf{u}_t),\ \boldsymbol{\phi}_1=\mathbf{x}_1$
- Optimize controls only:

$$
\min_{\mathbf{u}_{1:T-1}}\; c_T\!\big(\boldsymbol{\phi}_T\big)+\sum_{t=1}^{T-1} c_t\!\big(\boldsymbol{\phi}_t,\mathbf{u}_t\big)\quad\text{s.t.}\ \mathbf{u}_t\in\mathcal{U}_t.
$$

- Pros: small NLP, fits autodiff/JIT; Cons: conditioning degrades on long horizons, harder state constraints

---

## Multiple Shooting (Segmented Rollouts)

- Segment horizon; decision vars: segment start states $\{\mathbf{x}_k\}$ and controls
- Continuity constraints couple simulated endpoints:

$$
\mathbf{x}_{k+1} - \Phi(\mathbf{x}_k, \mathbf{u}_{\text{segment }k}) = 0.
$$

- Interpolates between single ($K{=}1$) and full transcription ($K{=}T$); shorter dependency chains, better conditioning

---

## KKT Conditions (Smooth NLP)

$$
\min_{\mathbf{z}} F(\mathbf{z})\ \text{s.t.}\ G(\mathbf{z})=0,\ H(\mathbf{z})\ge 0,\ \ \mathcal{L}=F+\boldsymbol{\lambda}^\top G+\boldsymbol{\mu}^\top H,\ \boldsymbol{\mu}\ge 0.
$$

- Stationarity: $\nabla_{\mathbf{z}}\mathcal{L}(\mathbf{z}^\star,\boldsymbol{\lambda}^\star,\boldsymbol{\mu}^\star)=0$
- Primal feasibility: $G(\mathbf{z}^\star)=0,\ H(\mathbf{z}^\star)\ge 0$
- Dual feasibility: $\boldsymbol{\mu}^\star\ge 0$; Complementarity: $\mu_i^\star H_i(\mathbf{z}^\star)=0$
- LICQ (or Slater) ensures multipliers exist, linearized system well-posed

---


## Adjoint = Reverse Accumulation (Discrete PMP)

$$
\boldsymbol{\lambda}_T=\nabla_{\mathbf{x}} c_T(\mathbf{x}_T),\quad \boldsymbol{\lambda}_t=\nabla_{\mathbf{x}} c_t+\big[\nabla_{\mathbf{x}}\mathbf{f}_t\big]^\top \boldsymbol{\lambda}_{t+1},\quad \nabla_{\mathbf{u}_t}J=\nabla_{\mathbf{u}} c_t+\big[\nabla_{\mathbf{u}}\mathbf{f}_t\big]^\top \boldsymbol{\lambda}_{t+1}.
$$

One forward rollout + one backward sweep yields all gradients.

---

## Saddle Point & Arrow–Hurwicz (Primal–Dual)

$$
\begin{aligned}
\mathbf{z}^{k+1} &= \mathbf{z}^{k} - \alpha_k\big(\nabla F + \nabla G^\top \boldsymbol{\lambda}^k + \nabla H^\top \boldsymbol{\mu}^k\big),\\
\boldsymbol{\lambda}^{k+1} &= \boldsymbol{\lambda}^k + \beta_k\,G(\mathbf{z}^k),\\
\boldsymbol{\mu}^{k+1} &= \Pi_{\ge 0}\big(\boldsymbol{\mu}^k + \beta_k\,H(\mathbf{z}^k)\big).
\end{aligned}
$$

- Convex: convergence with proper steps; Nonconvex: stabilize via augmented Lagrangian

---

## SQP as Newton on KKT (Equalities)

$$
\begin{bmatrix}
\nabla^2_{\mathbf{z}\mathbf{z}}\mathcal{L}(\mathbf{z}^k,\boldsymbol{\lambda}^k) & \nabla G(\mathbf{z}^k)^\top \\
\nabla G(\mathbf{z}^k) & 0
\end{bmatrix}
\begin{bmatrix}
\Delta\mathbf{z}\\ \Delta\boldsymbol{\lambda}
\end{bmatrix}
= -\begin{bmatrix}
\nabla_{\mathbf{z}}\mathcal{L}(\mathbf{z}^k,\boldsymbol{\lambda}^k)\\ G(\mathbf{z}^k)
\end{bmatrix}.
$$

- With inequalities: solve QP subproblems (line search/trust region); exploits banded structure in OCPs

---

## Continuous-Time OCP (Bolza)

$$
\begin{aligned}
\min_{\mathbf{u}(\cdot)}\ & c(\mathbf{x}(t_f)) + \int_{t_0}^{t_f} c(\mathbf{x}(t),\mathbf{u}(t))\,dt \\
\text{s.t.}\ & \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t),\mathbf{u}(t)),\ \ \mathbf{g}(\mathbf{x}(t),\mathbf{u}(t))\le 0, \\
& \mathbf{x}_{\min}\le \mathbf{x}(t)\le \mathbf{x}_{\max},\ \ \mathbf{u}_{\min}\le \mathbf{u}(t)\le \mathbf{u}_{\max},\\
& \mathbf{x}(t_0)=\mathbf{x}_0.
\end{aligned}
$$

- Lagrange or Bolza to Mayer via state augmentation: $\dot{z}=c(\mathbf{x},\mathbf{u}),\ z(t_0)=0,\ \min z(t_f)$

---

## Discretization via Quadrature (Shared Nodes)

- Mesh: $t_0<\cdots<t_N$, steps $h_k$
- Quadrature on each window with nodes $\xi_i\in[0,1]$, weights $w_i$

$$
\int_{t_k}^{t_{k+1}} c\,dt \approx h_k \sum_{i=1}^q w_i\, c\big(\mathbf{x}(t_k+h_k\xi_i),\mathbf{u}(t_k+h_k\xi_i)\big).
$$

- Dynamics via matched quadrature:

$$
\mathbf{x}_{k+1}-\mathbf{x}_k \approx h_k \sum_{i=1}^q b_i\, \mathbf{f}\big(\mathbf{x}(t_k+h_k\xi_i),\mathbf{u}(t_k+h_k\xi_i)\big).
$$

---

## Interpolation: Vandermonde System

- Polynomial interpolation (monomial basis): $f(x)=\sum_{n=0}^N c_n x^n$
- Matching values $f(x_i)=y_i$ yields Vandermonde system:

$$
\begin{bmatrix}
1 & x_0 & x_0^2 & \cdots & x_0^N \\
1 & x_1 & x_1^2 & \cdots & x_1^N \\
\vdots & \vdots & \vdots & & \vdots \\
1 & x_m & x_m^2 & \cdots & x_m^N
\end{bmatrix}
\begin{bmatrix}c_0\\c_1\\\vdots\\c_N\end{bmatrix}
=
\begin{bmatrix}y_0\\y_1\\\vdots\\y_m\end{bmatrix}.
$$

- Other bases (Legendre/Chebyshev/Lagrange) similarly form a linear system

---

## Interpolation with Slope Constraints

- If some derivatives are prescribed: $f'(z_j)=s_j$, use basis derivatives

$$
f'(x)=\sum_{n=0}^N c_n\,\phi_n'(x),\quad \Rightarrow\quad f'(z_j)=\sum_{n=0}^N c_n\,\phi_n'(z_j)=s_j.
$$

- Assemble a square system combining value and slope rows; solve for $\mathbf{c}$

---

## Collocation (ODE-Constrained Interpolation)

- Approximate $\mathbf{x}(t)$ on each window; enforce ODE at nodes $t_k^{(i)}$

$$
\frac{d}{dt}\mathbf{x}_h(t_k^{(i)}) = \mathbf{f}\big(\mathbf{x}_h(t_k^{(i)}),\mathbf{u}_h(t_k^{(i)}), t_k^{(i)}\big).
$$

- Use same nodes/weights for cost and dynamics; stitch intervals by endpoint continuity

---

## Collocation Instantiation: Euler (Rectangle Rule)

$$
\begin{aligned}
\min\ & c_T(\mathbf{x}_N)+\sum_{i=0}^{N-1} h_i\,c(\mathbf{x}_i,\mathbf{u}_i)\\
\text{s.t.}\ & \mathbf{x}_{i+1}-\mathbf{x}_i-h_i\,\mathbf{f}(\mathbf{x}_i,\mathbf{u}_i)=0,\\
& \mathbf{g}(\mathbf{x}_i,\mathbf{u}_i)\le 0,\ \ \mathbf{x}_0=\mathbf{x}(t_0).
\end{aligned}
$$

---

## Collocation Instantiation: Trapezoid

$$
\begin{aligned}
\min\ & c_T(\mathbf{x}_N)+\sum_{i=0}^{N-1} \tfrac{h_i}{2}\,[c(\mathbf{x}_i,\mathbf{u}_i)+c(\mathbf{x}_{i+1},\mathbf{u}_{i+1})]\\
\text{s.t.}\ & \mathbf{x}_{i+1}-\mathbf{x}_i-\tfrac{h_i}{2}[\mathbf{f}(\mathbf{x}_i,\mathbf{u}_i)+\mathbf{f}(\mathbf{x}_{i+1},\mathbf{u}_{i+1})]=0.
\end{aligned}
$$

---

## Collocation Instantiation: Hermite–Simpson

Midpoints $t_{i+1/2}$ with variables $(\mathbf{x}_{i+1/2},\mathbf{u}_{i+1/2})$

$$
\begin{aligned}
\mathbf{x}_{i+1}-\mathbf{x}_i &- \tfrac{h_i}{6}[\mathbf{f}(\mathbf{x}_i,\mathbf{u}_i)+4\,\mathbf{f}(\mathbf{x}_{i+1/2},\mathbf{u}_{i+1/2})+\mathbf{f}(\mathbf{\,x}_{i+1},\mathbf{u}_{i+1})]=0,\\
\mathbf{x}_{i+1/2} &- \tfrac{\mathbf{x}_i+\mathbf{x}_{i+1}}{2} - \tfrac{h_i}{8}[\mathbf{f}(\mathbf{x}_i,\mathbf{u}_i)-\mathbf{f}(\mathbf{x}_{i+1},\mathbf{u}_{i+1})]=0.
\end{aligned}
$$

---

## Collocation Instantiation: RK4-as-Transcription

Stages $\mathbf{s}^{(1..4)}_i$, midpoint control $\bar{\mathbf{u}}_i$

$$
\begin{aligned}
\mathbf{s}^{(1)}_i&=\mathbf{f}(\mathbf{x}_i,\mathbf{u}_i),\ \ \mathbf{s}^{(2)}_i=\mathbf{f}\!\big(\mathbf{x}_i+\tfrac{h_i}{2}\mathbf{s}^{(1)}_i,\bar{\mathbf{u}}_i\big),\\
\mathbf{s}^{(3)}_i&=\mathbf{f}\!\big(\mathbf{x}_i+\tfrac{h_i}{2}\mathbf{s}^{(2)}_i,\bar{\mathbf{u}}_i\big),\ \ \mathbf{s}^{(4)}_i=\mathbf{f}\!\big(\mathbf{x}_i+h_i\mathbf{s}^{(3)}_i,\mathbf{u}_{i+1}\big),\\
\mathbf{x}_{i+1}&=\mathbf{x}_i+\tfrac{h_i}{6}\big[\mathbf{s}^{(1)}_i+2\mathbf{s}^{(2)}_i+2\mathbf{s}^{(3)}_i+\mathbf{s}^{(4)}_i\big].
\end{aligned}
$$

Cost nodes align with staged evaluations.

---

