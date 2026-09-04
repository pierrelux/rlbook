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
# Approximate Bellman Equations

Weighted-residual methods specify approximation spaces and tests for a general
functional equation. What changes when the residual contains a Bellman
operator whose contraction properties are responsible for convergence?

Consider the Bellman optimality equation $v(s) = \Bellman v(s) = \max_{a \in \mathcal{A}_s} \{ r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j) \}$. For a candidate approximation $\hat{v}(s) = \sum_{i=1}^n \theta_i \varphi_i(s)$, the residual is:

$$
R(s; \theta) = \Bellman\hat{v}(s) - \hat{v}(s) = \max_{a \in \mathcal{A}_s} \left\{ r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) \hat{v}(j) \right\} - \sum_{i=1}^n \theta_i \varphi_i(s).
$$

We examine how collocation and Galerkin, the two most common weighted residual methods for Bellman equations, specialize the general solution approaches from Step 4.

### Collocation

For collocation, we choose $n$ states $\{s_1, \ldots, s_n\}$ and require the Bellman equation to hold exactly at these points:

$$
\sum_{j=1}^n \theta_j \varphi_j(s_i) = \max_{a \in \mathcal{A}_{s_i}} \left\{ r(s_i,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s_i,a) \sum_{\ell=1}^n \theta_\ell \varphi_\ell(j) \right\}, \quad i = 1, \ldots, n.
$$

It helps to define the **parametric Bellman operator** $\mathrm{L}_\varphi: \mathbb{R}^n \to \mathbb{R}^n$ by $[\mathrm{L}_\varphi(\theta)]_i = [\Bellman\hat{v}(\cdot; \theta)](s_i)$, the Bellman operator evaluated at collocation point $s_i$. Let $\boldsymbol{\Phi}$ be the $n \times n$ matrix with entries $\Phi_{ij} = \varphi_j(s_i)$. Then the collocation equations become $\boldsymbol{\Phi} \theta = \mathrm{L}_\varphi(\theta)$.

Under function iteration, the current coefficients $\theta^{(k)}$ produce the
target values
$t_i^{(k)}=[\mathrm{L}_\varphi(\theta^{(k)})]_i$ at the collocation points.
The linear system
$\boldsymbol{\Phi}\theta^{(k+1)}=t^{(k)}$ then interpolates those values. Each
iteration therefore applies the Bellman operator and constructs its polynomial
interpolant at the selected points.

```{prf:algorithm} Collocation with Function Iteration
:label: collocation-function-iteration

**Input** Collocation points $\{s_1, \ldots, s_n\}$, basis functions $\{\varphi_1, \ldots, \varphi_n\}$, initial $\theta^{(0)}$, tolerance $\varepsilon > 0$

**Output** Converged coefficients $\theta^*$

1. Form collocation matrix $\boldsymbol{\Phi}$ with $\Phi_{ij} = \varphi_j(s_i)$
2. $k \leftarrow 0$
3. **repeat**
    1. For each $i = 1, \ldots, n$:
        1. $t_i^{(k)} \leftarrow \max_{a \in \mathcal{A}_{s_i}} \left\{ r(s_i, a) + \gamma \sum_{j \in \mathcal{S}} p(j|s_i, a) \sum_{\ell=1}^n \theta_\ell^{(k)} \varphi_\ell(j) \right\}$
    2. Solve $\boldsymbol{\Phi} \theta^{(k+1)} = t^{(k)}$
    3. $k \leftarrow k + 1$
4. **until** $\|\theta^{(k)} - \theta^{(k-1)}\| < \varepsilon$
5. **return** $\theta^{(k)}$
```

When the state space is continuous, we approximate expectations using numerical quadrature (Gauss-Hermite for normal shocks, etc.). The method is simple and robust when the finite-dimensional approximation preserves contraction, but can be slow for large discount factors.

**Newton's method** for collocation treats the problem as rootfinding: $G(\theta) = \boldsymbol{\Phi} \theta - \mathrm{L}_\varphi(\theta) = 0$. The Jacobian is $J_G = \boldsymbol{\Phi} - J_{\mathrm{L}_\varphi}$, where the Envelope Theorem (Step 4) gives us $[J_{\mathrm{L}_\varphi}]_{ij} = \gamma \sum_{s'} p(s'|s_i, a_i^*(\theta)) \varphi_j(s')$. Here $a_i^*(\theta)$ is the optimal action at collocation point $s_i$ given the current coefficients.

```{prf:algorithm} Collocation with Newton's Method
:label: collocation-newton

**Input** Collocation points $\{s_1, \ldots, s_n\}$, basis functions $\{\varphi_1, \ldots, \varphi_n\}$, initial $\theta^{(0)}$, tolerance $\varepsilon > 0$

**Output** Converged coefficients $\theta^*$

1. Form collocation matrix $\boldsymbol{\Phi}$ with $\Phi_{ij} = \varphi_j(s_i)$
2. $k \leftarrow 0$
3. **repeat**
    1. For each $i = 1, \ldots, n$:
        1. $t_i^{(k)} \leftarrow \max_{a \in \mathcal{A}_{s_i}} \left\{ r(s_i, a) + \gamma \sum_{j \in \mathcal{S}} p(j|s_i, a) \sum_{\ell=1}^n \theta_\ell^{(k)} \varphi_\ell(j) \right\}$
        2. Store $a_i^* \in \arg\max$ achieving the maximum
    2. Compute Jacobian: $[J_{\mathrm{L}_\varphi}]_{ij} = \gamma \sum_{j \in \mathcal{S}} p(j|s_i, a_i^*) \varphi_j(j)$ for all $i,j$
    3. Solve $(\boldsymbol{\Phi} - J_{\mathrm{L}_\varphi}) \Delta\theta = \boldsymbol{\Phi} \theta^{(k)} - t^{(k)}$
    4. $\theta^{(k+1)} \leftarrow \theta^{(k)} - \Delta\theta$
    5. $k \leftarrow k + 1$
4. **until** $\|\Delta\theta\| < \varepsilon$
5. **return** $\theta^{(k)}$
```

This converges rapidly near the solution but requires good initialization and more computation per iteration than function iteration. The method is equivalent to policy iteration: each step evaluates the value of the current greedy policy, then improves it.

Why is collocation popular for Bellman equations? Because it avoids integration when testing the residual. We only evaluate the Bellman operator at $n$ discrete points. In contrast, Galerkin requires integrating the residual against each basis function.

#### Worked Example: Collocation on the Optimal Stopping Problem

Returning to our motivating example, let us trace through the collocation algorithm with $n = 4$ polynomial basis functions at Chebyshev nodes:

```{code-cell} python
:tags: [hide-input]
import numpy as np
from scipy.integrate import quad

gamma = 0.9
n = 4

# Chebyshev nodes on [0, 1]
k = np.arange(1, n + 1)
nodes = 0.5 + 0.5 * np.cos((2*k - 1) * np.pi / (2*n))
nodes = np.sort(nodes)

# Vandermonde matrix
Phi = np.vander(nodes, n, increasing=True)

# Exact solution
v_bar_exact = (1 - np.sqrt(1 - gamma**2)) / gamma**2
s_star_exact = gamma * v_bar_exact
def v_exact(s):
    return np.where(s >= s_star_exact, s, gamma * v_bar_exact)

# Collocation iteration
theta = np.zeros(n)
print("Collocation iteration trace:")
print(f"{'Iter':<6} {'||theta||':<12} {'Max error':<12}")
print("-" * 30)

for iteration in range(15):
    def v_approx(s, th=theta):
        return sum(th[j] * s**j for j in range(n))
    v_bar, _ = quad(v_approx, 0, 1)
    test_points = np.linspace(0, 1, 100)
    max_error = max(abs(v_approx(s) - v_exact(s)) for s in test_points)
    print(f"{iteration:<6} {np.linalg.norm(theta):<12.6f} {max_error:<12.6f}")
    
    targets = np.maximum(nodes, gamma * v_bar)
    theta_new = np.linalg.solve(Phi, targets)
    if np.linalg.norm(theta_new - theta) < 1e-10:
        print(f"\nConverged in {iteration + 1} iterations")
        break
    theta = theta_new
```

### Galerkin

For Galerkin, we use the basis functions themselves as test functions. The conditions are:

$$
\int_{\mathcal{S}} [\Bellman\hat{v}(s; \theta) - \hat{v}(s; \theta)] \varphi_i(s) w(s) ds = 0, \quad i = 1, \ldots, n.
$$

where $w(s)$ is a weight function (often the stationary distribution $d^\pi(s)$ in RL applications, or simply $w(s) = 1$). Expanding this:

$$
\int_{\mathcal{S}} \left[ \max_a \left\{ r(s,a) + \gamma \mathbb{E}[v(s')] \right\} - \sum_j \theta_j \varphi_j(s) \right] \varphi_i(s) w(s) ds = 0.
$$

**Function iteration** for Galerkin works differently than for collocation. Given $\theta^{(k)}$, we cannot simply evaluate the Bellman operator and fit. Instead, we must solve an integral equation. At each iteration, we seek $\theta^{(k+1)}$ satisfying:

$$
\int_{\mathcal{S}} \sum_j \theta_j^{(k+1)} \varphi_j(s) \varphi_i(s) w(s) ds = \int_{\mathcal{S}} [\Bellman\hat{v}(s; \theta^{(k)})] \varphi_i(s) w(s) ds.
$$

```{prf:algorithm} Galerkin with Function Iteration
:label: galerkin-function-iteration

**Input** Basis functions $\{\varphi_1, \ldots, \varphi_n\}$, weight function $w(s)$, initial $\theta^{(0)}$, tolerance $\varepsilon > 0$

**Output** Converged coefficients $\theta^*$

1. Compute mass matrix $M_{ij} = \int_{\mathcal{S}} \varphi_i(s) \varphi_j(s) w(s) ds$ via numerical integration
2. $k \leftarrow 0$
3. **repeat**
    1. For each $i = 1, \ldots, n$:
        1. $b_i^{(k)} \leftarrow \int_{\mathcal{S}} [\Bellman\hat{v}(s; \theta^{(k)})] \varphi_i(s) w(s) ds$ via numerical integration
    2. Solve $M \theta^{(k+1)} = b^{(k)}$
    3. $k \leftarrow k + 1$
4. **until** $\|\theta^{(k)} - \theta^{(k-1)}\| < \varepsilon$
5. **return** $\theta^{(k)}$
```

The left side is a linear system (the "mass matrix" $M_{ij} = \int \varphi_i \varphi_j w$), and the right side requires integrating the Bellman operator output against each test function. When the basis functions are orthogonal polynomials with matching weight $w$, the mass matrix is diagonal, simplifying the solve. But we still need numerical integration to evaluate the right side. This makes Galerkin substantially more expensive than collocation per iteration.

**Newton's method** for Galerkin similarly requires integration. The residual is $R(s; \theta) = \Bellman\hat{v}(s; \theta) - \hat{v}(s; \theta)$, and we need $G_i(\theta) = \int R(s; \theta) \varphi_i(s) w(s) ds = 0$. The Jacobian entry is:

$$
J_{ij} = \int \left[ \frac{\partial \Bellman\hat{v}(s; \theta)}{\partial \theta_j} - \varphi_j(s) \right] \varphi_i(s) w(s) ds.
$$

```{prf:algorithm} Galerkin with Newton's Method
:label: galerkin-newton

**Input** Basis functions $\{\varphi_1, \ldots, \varphi_n\}$, weight function $w(s)$, initial $\theta^{(0)}$, tolerance $\varepsilon > 0$

**Output** Converged coefficients $\theta^*$

1. $k \leftarrow 0$
2. **repeat**
    1. For each $i = 1, \ldots, n$:
        1. $G_i^{(k)} \leftarrow \int_{\mathcal{S}} [\Bellman\hat{v}(s; \theta^{(k)}) - \hat{v}(s; \theta^{(k)})] \varphi_i(s) w(s) ds$
        2. For each $j = 1, \ldots, n$:
            1. $J_{ij} \leftarrow \int_{\mathcal{S}} \left[ \gamma \mathbb{E}[\varphi_j(s') \mid s, a^*(s;\theta^{(k)})] - \varphi_j(s) \right] \varphi_i(s) w(s) ds$
    2. Solve $J \Delta\theta = G^{(k)}$
    3. $\theta^{(k+1)} \leftarrow \theta^{(k)} - \Delta\theta$
    4. $k \leftarrow k + 1$
3. **until** $\|\Delta\theta\| < \varepsilon$
4. **return** $\theta^{(k)}$
```

The Envelope Theorem gives $\frac{\partial \Bellman\hat{v}(s; \theta)}{\partial \theta_j} = \gamma \mathbb{E}[\varphi_j(s') \mid s, a^*(s;\theta)]$, so we must integrate expected basis function values (under optimal actions) against test functions and weight. This requires both numerical integration and careful tracking of optimal actions across the state space, making it substantially more complex than collocation's pointwise evaluation.

The advantage of Galerkin over collocation lies in its theoretical properties: when using orthogonal polynomials, Galerkin provides optimal approximation in the weighted $L^2$ norm. For smooth problems, this can yield better accuracy per degree of freedom than collocation. In practice, collocation's computational simplicity usually outweighs Galerkin's theoretical optimality for Bellman equations, especially in high-dimensional problems where integration becomes prohibitively expensive.

The algorithms above reduce the infinite-dimensional Bellman fixed-point problem to finite-dimensional coefficient computation. Collocation avoids integration entirely by requiring exact satisfaction at discrete points, while Galerkin imposes weighted orthogonality conditions requiring numerical quadrature. Both can be solved via function iteration (when contraction is preserved) or Newton's method (for faster convergence near the solution). The discrete MDP specialization below reveals connections to algorithms widely used in reinforcement learning.

```{admonition} Exercises: Collocation and Galerkin
:class: hint dropdown

1. **Effect of collocation points.** For the optimal stopping problem, compare collocation with $n=4$ Chebyshev nodes versus $n=4$ equally-spaced nodes. Which choice gives smaller maximum error?

2. **Threshold location.** The optimal stopping policy has a threshold structure. Using your polynomial approximation $\hat{v}(s)$, estimate the threshold by finding where $\hat{v}(s) = s$. Compare to the exact threshold.

3. **Orthogonal polynomials.** For $\mathcal{S} = [-1, 1]$, write out the Galerkin conditions using Chebyshev polynomials $T_i$ and weight $w(x) = 1/\sqrt{1-x^2}$. What makes this choice computationally convenient?

4. **Newton vs. function iteration.** How does the iteration count depend on $\gamma$? Try $\gamma \in \{0.5, 0.9, 0.99\}$.
```

### Galerkin for Discrete MDPs: LSTD and LSPI

When the state space is discrete and finite, the Galerkin conditions simplify dramatically. The integrals become sums, and we can write everything in matrix form. This specialization shows the connection to algorithms widely used in reinforcement learning.

For a discrete state space $\mathcal{S} = \{s_1, \ldots, s_m\}$, the Galerkin orthogonality conditions

$$
\int_{\mathcal{S}} [\Bellman\hat{v}(s; \theta) - \hat{v}(s; \theta)] \varphi_i(s) w(s) ds = 0
$$

become weighted sums over states:

$$
\sum_{s \in \mathcal{S}} \xi(s) [\Bellman\hat{v}(s; \theta) - \hat{v}(s; \theta)] \varphi_i(s) = 0, \quad i = 1, \ldots, n,
$$

where $\xi(s) \geq 0$ with $\sum_s \xi(s) = 1$ is a probability distribution over states. Define the feature matrix $\boldsymbol{\Phi} \in \mathbb{R}^{m \times n}$ with entries $\Phi_{si} = \varphi_i(s)$ (each row contains the features for one state), and let $\boldsymbol{\Xi} = \text{diag}(\xi)$ be the diagonal matrix with the state distribution on the diagonal.

#### Policy Evaluation: LSTD

For **policy evaluation** with a fixed policy $\pi$, the Bellman operator is linear:

$$
[\BellmanPi \hat{v}](s) = r(s, \pi(s)) + \gamma \sum_{j \in \mathcal{S}} p(j|s, \pi(s)) \hat{v}(j).
$$

With linear function approximation $\hat{v}(s) = \boldsymbol{\varphi}(s)^\top \theta = \sum_i \theta_i \varphi_i(s)$, this becomes:

$$
[\BellmanPi \hat{v}](s) = r(s, \pi(s)) + \gamma \sum_{j \in \mathcal{S}} p(j|s, \pi(s)) \sum_i \theta_i \varphi_i(j).
$$

Let $\mathbf{r}_\pi \in \mathbb{R}^m$ be the vector of rewards $[\mathbf{r}_\pi]_s = r(s, \pi(s))$, and $\mathbf{P}_\pi \in \mathbb{R}^{m \times m}$ be the transition matrix with $[\mathbf{P}_\pi]_{sj} = p(j|s, \pi(s))$. Then $\BellmanPi \hat{v} = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi \boldsymbol{\Phi} \theta$ in vector form.

The Galerkin conditions require $\langle \BellmanPi \hat{v} - \hat{v}, \varphi_i \rangle_\xi = 0$ for all basis functions, which in matrix form is:

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \boldsymbol{\Phi} \theta - \boldsymbol{\Phi} \theta) = \mathbf{0}.
$$

Rearranging:

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_\pi \boldsymbol{\Phi}) \theta = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{r}_\pi.
$$ (eq:lstd-galerkin)

This is the **LSTD (Least Squares Temporal Difference)** solution. The matrix $\mathbf{A} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_\pi \boldsymbol{\Phi})$ and vector $\mathbf{b} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{r}_\pi$ give the linear system $\mathbf{A} \theta = \mathbf{b}$.

When $\xi$ is the stationary distribution of policy $\pi$ (so $\xi^\top \mathbf{P}_\pi = \xi^\top$), this system has a unique solution, and the projected Bellman operator $\Proj \BellmanPi$ is a contraction in the weighted $L^2$ norm $\|\cdot\|_\xi$. This is the theoretical foundation for TD learning with linear function approximation. The fixed point computed here is the same one that TD(0) converges to stochastically; we derive the incremental algorithm in the Monte Carlo chapter.

*Check that the dimensions work out: if we have $m$ states and $n$ basis functions, what are the dimensions of $\boldsymbol{\Phi}$, $\boldsymbol{\Xi}$, $\mathbf{P}_\pi$, and the matrix $\mathbf{A} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi}(\boldsymbol{\Phi} - \gamma \mathbf{P}_\pi \boldsymbol{\Phi})$?*

##### Worked Example: LSTD for Policy Evaluation

To illustrate LSTD concretely, consider a 3-state Markov chain under a fixed policy:

$$
\mathbf{P}_\pi = \begin{pmatrix} 0.7 & 0.2 & 0.1 \\ 0.3 & 0.4 & 0.3 \\ 0.1 & 0.3 & 0.6 \end{pmatrix}, \quad \mathbf{r}_\pi = \begin{pmatrix} 1 \\ 2 \\ 0 \end{pmatrix}.
$$

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

#### The Bellman Optimality Equation: Function Iteration and Newton's Method

For the **Bellman optimality equation**, the max operator introduces nonlinearity:

$$
[\Bellman\hat{v}](s) = \max_{a \in \mathcal{A}_s} \left\{ r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) \hat{v}(j) \right\}.
$$

The Galerkin conditions become:

$$
F(\theta) \equiv \boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\Bellman\hat{v}(\cdot; \theta) - \boldsymbol{\Phi} \theta) = \mathbf{0},
$$

where the Bellman operator must be evaluated at each state $s$ to find the optimal action and compute the target value. This is a system of $n$ nonlinear equations in $n$ unknowns.

**Function iteration** applies the Bellman operator and projects back. Given $\theta^{(k)}$, compute the greedy policy $\pi^{(k)}(s) = \arg\max_a \{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) \boldsymbol{\varphi}(j)^\top \theta^{(k)}\}$ at each state, then solve:

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_{\pi^{(k)}} \boldsymbol{\Phi}) \theta^{(k+1)} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{r}_{\pi^{(k)}}.
$$

This evaluates the current greedy policy using LSTD, then implicitly improves by computing a new greedy policy at the next iteration. However, convergence can be slow when the finite-dimensional approximation poorly preserves contraction.

**Newton's method** treats $G(\theta) = 0$ as a rootfinding problem and uses the Jacobian to accelerate convergence. The Jacobian of $G$ is:

$$
J_G(\theta) = \frac{\partial G}{\partial \theta} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \left( \frac{\partial \Bellman\hat{v}(\cdot; \theta)}{\partial \theta} - \boldsymbol{\Phi} \right).
$$

To compute $\frac{\partial \Bellman\hat{v}(s; \theta)}{\partial \theta_j}$, we use the Envelope Theorem from Step 4. At the current $\theta^{(k)}$, let $a^*(s; \theta^{(k)})$ be the optimal action at state $s$. Then:

$$
\frac{\partial [\Bellman\hat{v}](s; \theta^{(k)})}{\partial \theta_j} = \gamma \sum_{j \in \mathcal{S}} p(j|s, a^*(s; \theta^{(k)})) \varphi_j(j).
$$

Define the policy $\pi^{(k)}(s) = a^*(s; \theta^{(k)})$. The Jacobian becomes:

$$
J_G(\theta^{(k)}) = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\gamma \mathbf{P}_{\pi^{(k)}} \boldsymbol{\Phi} - \boldsymbol{\Phi}) = -\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_{\pi^{(k)}} \boldsymbol{\Phi}).
$$

The Newton update $\theta^{(k+1)} = \theta^{(k)} - J_G(\theta^{(k)})^{-1} G(\theta^{(k)})$ simplifies. We have:

$$
G(\theta^{(k)}) = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\Bellman\hat{v}(\cdot; \theta^{(k)}) - \boldsymbol{\Phi} \theta^{(k)}).
$$

At each state $s$, the greedy value is $[\Bellman\hat{v}(\cdot; \theta^{(k)})](s) = r(s, \pi^{(k)}(s)) + \gamma \sum_j p(j|s, \pi^{(k)}(s)) \boldsymbol{\varphi}(j)^\top \theta^{(k)}$, which equals $[\mathrm{L}_{\pi^{(k)}} \hat{v}(\cdot; \theta^{(k)})](s)$. Thus:

$$
G(\theta^{(k)}) = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\mathbf{r}_{\pi^{(k)}} + \gamma \mathbf{P}_{\pi^{(k)}} \boldsymbol{\Phi} \theta^{(k)} - \boldsymbol{\Phi} \theta^{(k)}).
$$

The Newton step becomes:

$$
\theta^{(k+1)} = \theta^{(k)} + [\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_{\pi^{(k)}} \boldsymbol{\Phi})]^{-1} \boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\mathbf{r}_{\pi^{(k)}} + \gamma \mathbf{P}_{\pi^{(k)}} \boldsymbol{\Phi} \theta^{(k)} - \boldsymbol{\Phi} \theta^{(k)}).
$$

Multiplying through and simplifying:

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_{\pi^{(k)}} \boldsymbol{\Phi}) \theta^{(k+1)} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{r}_{\pi^{(k)}}.
$$

This is **LSPI (Least Squares Policy Iteration)**. Each Newton step:
1. Computes the greedy policy $\pi^{(k)}(s) = \arg\max_a \{r(s,a) + \gamma \sum_j p(j|s,a) \boldsymbol{\varphi}(j)^\top \theta^{(k)}\}$
2. Solves the LSTD equation for this policy to get $\theta^{(k+1)}$

Newton's method for the Galerkin-projected Bellman optimality equation is equivalent to policy iteration in the function approximation setting. Just as Newton's method for collocation corresponded to policy iteration (Step 4), Newton's method for discrete Galerkin gives LSPI.

Galerkin projection with linear function approximation reduces policy iteration to a sequence of linear systems, each solvable in closed form. For discrete MDPs, we can compute the matrices $\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \boldsymbol{\Phi}$ and $\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{P}_\pi \boldsymbol{\Phi}$ exactly.

## Extension to Nonlinear Approximators

What remains of the residual formulation when the value function is represented
by a nonlinear model rather than a linear basis expansion?

The weighted residual methods developed so far have focused on linear function classes: polynomial bases, piecewise linear interpolants, and linear combinations of fixed basis functions. Neural networks, kernel methods, and decision trees do not fit this template. How does the framework extend to nonlinear approximators?

Recall the Galerkin approach for linear approximation $v_{\boldsymbol{\theta}} = \sum_{i=1}^d \theta_i \varphi_i$. The orthogonality conditions $\langle v - \Bellman v, \varphi_i \rangle_w = 0$ for all $i$ define a linear system with a closed-form solution. These equations arise from minimizing $\|v - \Bellman v\|_w^2$ over the subspace, since at the minimum, the gradient with respect to each coefficient must vanish. The connection between norm minimization and orthogonality holds generally. For any norm $\|\cdot\|_w$ induced by an inner product $\langle \cdot, \cdot \rangle_w$, minimizing $\|f(\boldsymbol{\theta})\|_w^2$ with respect to parameters requires $\frac{\partial}{\partial \theta_i} \|f(\boldsymbol{\theta})\|_w^2 = 0$. Since $\|f\|_w^2 = \langle f, f \rangle_w$, the chain rule gives $2\langle f, \frac{\partial f}{\partial \theta_i} \rangle_w = 0$. Minimizing the residual norm is thus equivalent to requiring orthogonality $\langle f, \frac{\partial f}{\partial \theta_i} \rangle_w = 0$ for all $i$. The equivalence holds for any choice of inner product: weighted $L^2$ integrals for Galerkin, sums over collocation points for collocation, or sampled expectations for neural networks.

For nonlinear function classes parameterized by $\boldsymbol{\theta} \in \mathbb{R}^p$ (neural networks, kernel expansions), the same minimization principle applies:

$$
\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \|v_{\boldsymbol{\theta}} - \Bellman v_{\boldsymbol{\theta}}\|_w^2.
$$

The first-order stationarity condition yields orthogonality:

$$
\Big\langle v_{\boldsymbol{\theta}} - \Bellman v_{\boldsymbol{\theta}}, \frac{\partial v_{\boldsymbol{\theta}}}{\partial \theta_i} \Big\rangle_w = 0 \quad \text{for all } i.
$$

The test functions are now the partial derivatives $\frac{\partial v_{\boldsymbol{\theta}}}{\partial \theta_i}$, which span the tangent space to the manifold $\{v_{\boldsymbol{\theta}} : \boldsymbol{\theta} \in \mathbb{R}^p\}$ at the current parameters. In the linear case $v_{\boldsymbol{\theta}} = \sum_i \theta_i \varphi_i$, the partial derivative $\frac{\partial v_{\boldsymbol{\theta}}}{\partial \theta_i} = \varphi_i$ recovers the fixed basis functions of Galerkin. For nonlinear parameterizations, the test functions change with $\boldsymbol{\theta}$, and the orthogonality conditions define a nonlinear system solved by iterative gradient descent.

The **dual pairing** formulation {cite}`LegrandJunca2025` extends this framework to settings where test objects need not be regular functions. We have been informal about this distinction in our treatment of collocation, but the Dirac deltas $\delta(x - x_i)$ used there are not classical functions. They are distributions, defined rigorously only through their action on test functions via $\langle \Residual(v), \delta(x - x_i) \rangle = (\Residual v)(x_i)$. The simple calculus argument for orthogonality does not apply directly to such objects; the dual pairing framework provides the proper mathematical foundation. The induced dual norm $\|\Residual(v)\|_* = \sup_{\|w\|=1} |\langle \Residual(v), w \rangle|$ measures residuals by their worst-case effect on test functions, a perspective that has inspired adversarial formulations {cite}`Zang2020` where both trial and test functions are learned.

The minimum residual framework thus connects classical projection methods to modern function approximation. The unifying principle is orthogonality of residuals to test functions. Linear methods use fixed test functions and admit closed-form solutions. Nonlinear methods use parameter-dependent test functions and require iterative optimization.

We now turn to the question of convergence: when does the iteration $v_{k+1} = \Proj \Bellman v_k$ converge?

## Monotone Projection and the Preservation of Contraction

Which approximation maps preserve order and sup-norm contraction when composed
with a Bellman operator?

The informal discussion of shape preservation hints at a deeper theoretical question: **when does the function iteration method converge?** Recall from our discussion of collocation that function iteration proceeds in two steps:

1. Apply the Bellman operator at collocation points: $t^{(k)} = v(\theta^{(k)})$ where $t_i^{(k)} = \Bellman\hat{v}^{(k)}(s_i)$
2. Fit new coefficients to match these targets: $\boldsymbol{\Phi} \theta^{(k+1)} = t^{(k)}$, giving $\theta^{(k+1)} = \boldsymbol{\Phi}^{-1} v(\theta^{(k)})$

We can reinterpret this iteration in **function space** rather than coefficient space. Let $\Proj$ be the **projection operator** that takes any function $f$ and returns its approximation in $\text{span}\{\varphi_1, \ldots, \varphi_n\}$. For collocation, $\Proj$ is the interpolation operator: $(\Proj f)(s)$ is the unique linear combination of basis functions that matches $f$ at the collocation points. Then Step 2 can be written as: fit $\hat{v}^{(k+1)}$ so that $\hat{v}^{(k+1)}(s_i) = \Bellman\hat{v}^{(k)}(s_i)$ for all collocation points, which means $\hat{v}^{(k+1)} = \Proj(\Bellman\hat{v}^{(k)})$.

In other words, function iteration is equivalent to **projected value iteration in function space**:

$$
\hat{v}^{(k+1)} = \Proj \Bellman \hat{v}^{(k)}.
$$

We know that standard value iteration $v_{k+1} = \Bellman v_k$ converges because $\Bellman$ is a $\gamma$-contraction in the sup norm. But now we're iterating with the **composed operator** $\Proj \Bellman$ instead of $\Bellman$ alone.

This $\Proj \Bellman$ structure is not specific to collocation. It is inherent in all projection methods. The general pattern is always the same: apply the Bellman operator to get a target function $\Bellman\hat{v}^{(k)}$, then project it back onto our approximation space to get $\hat{v}^{(k+1)}$. The projection step defines an operator $\Proj$ that depends on our choice of test functions:

- For **collocation**, $\Proj$ interpolates values at collocation points
- For **Galerkin**, $\Proj$ is orthogonal projection with respect to $\langle \cdot, \cdot \rangle_w$  
- For **least squares**, $\Proj$ minimizes the weighted residual norm

But regardless of which projection method we use, iteration takes the form $\hat{v}^{(k+1)} = \Proj \Bellman\hat{v}^{(k)}$.

The central question is whether the composition $\Proj \Bellman$ inherits the contraction property of $\Bellman$. If not, the iteration may diverge, oscillate, or converge to a spurious fixed point even though the original problem is well-posed.

### Monotone Approximators and Stability

The answer turns out to depend on specific properties of the approximation operator $\Proj$. This theory was developed independently across multiple research communities: computational economics {cite}`Judd1992,Judd1996,McGrattan1997,SantosVigoAguiar1998`, economic dynamics {cite}`Stachurski2009`, and reinforcement learning {cite}`Gordon1995,Gordon1999`. These communities arrived at essentially the same mathematical conditions.

#### Monotonicity Implies Nonexpansiveness

It turns out that approximation operators satisfying simple structural properties automatically preserve contraction.

```{prf:proposition} Monotone operators are nonexpansive (Stachurski)
:label: monotone-nonexpansive

Let $\Proj: \mathcal{V} \to \mathcal{V}$ be a linear operator on the space $\mathcal{V}$ of bounded real-valued functions on $\mathcal{S}$. If $\Proj$ satisfies:

1. **Monotonicity**: $f \leq g$ pointwise implies $\Proj f \leq \Proj g$
2. **Constant preservation**: $\Proj\mathbf{1} = \mathbf{1}$ where $\mathbf{1}$ is the constant function equal to $1$

Then $\Proj$ is nonexpansive in the sup norm: $\|\Proj f - \Proj g\|_\infty \leq \|f - g\|_\infty$ for all $f, g \in \mathcal{V}$.
```

```{prf:proof}
Let $M = \|f - g\|_\infty$. Then $-M \leq f(s) - g(s) \leq M$ for all $s$, which can be written as $g - M\mathbf{1} \leq f \leq g + M\mathbf{1}$. By monotonicity, $\Proj(g - M\mathbf{1}) \leq \Proj f \leq \Proj(g + M\mathbf{1})$. By linearity and constant preservation, $\Proj g - M\mathbf{1} \leq \Proj f \leq \Proj g + M\mathbf{1}$, which means $|\Proj f(s) - \Proj g(s)| \leq M$ for all $s$. Therefore $\|\Proj f - \Proj g\|_\infty \leq \|f - g\|_\infty$.
```

This proposition shows that monotonicity and constant preservation automatically imply nonexpansiveness. There is no need to verify this separately. The intuition is that a monotone, constant-preserving operator acts like a weighted average that respects order structure and cannot amplify differences between functions.

#### Preservation of Contraction

Combining nonexpansiveness with the contraction property of the Bellman operator yields the main stability result.

```{prf:theorem} Stability of projected value iteration (Santos-Vigo-Aguiar)
:label: santos-vigo-aguiar-stability

Let $\Bellman: \mathcal{V} \to \mathcal{V}$ be a $\gamma$-contraction on the space $\mathcal{V}$ of bounded functions with respect to the sup norm. Let $\Proj: \mathcal{V} \to \mathcal{V}$ be a linear approximation operator satisfying monotonicity and constant preservation.

Then the composed operator $\Proj \Bellman$ is a $\gamma$-contraction, and projected value iteration $v_{k+1} = \Proj \Bellman v_k$ converges globally to a unique fixed point $v_\Proj \in \text{Range}(\Proj)$ with approximation error:

$$
\|v_\Proj - v^*\|_\infty \leq \frac{1}{1-\gamma} \|\Proj v^* - v^*\|_\infty,
$$

where $v^*$ is the true value function.
```

```{prf:proof}
By {prf:ref}`monotone-nonexpansive`, $\Proj$ is nonexpansive since it satisfies monotonicity and constant preservation. Since $\Bellman$ is a $\gamma$-contraction, we have $\|\Bellman f - \Bellman g\|_\infty \leq \gamma\|f-g\|_\infty$. Therefore:

$$
\|\Proj \Bellman f - \Proj \Bellman g\|_\infty \leq \|\Bellman f - \Bellman g\|_\infty \leq \gamma\|f-g\|_\infty,
$$

showing that $\Proj \Bellman$ is a $\gamma$-contraction. The error bound follows from fixed-point analysis: $v^* - v_\Proj = (I - \Proj \Bellman)^{-1}(v^* - \Proj v^*)$, and since $\Proj \Bellman$ is a $\gamma$-contraction, $\|(I - \Proj \Bellman)^{-1}\| \leq (1-\gamma)^{-1}$.
```

This error bound tells us that the fixed-point error is controlled by how well $\Proj$ can represent $v^*$. If $v^* \in \text{Range}(\Proj)$, then $\Proj v^* = v^*$ and the error vanishes. Otherwise, the error is proportional to the approximation error $\|\Proj v^* - v^*\|_\infty$, amplified by the factor $(1-\gamma)^{-1}$.

#### Averagers in Discrete-State Problems

For discrete-state problems, the monotonicity conditions have a natural interpretation as **averaging with nonnegative weights**. This characterization was developed by Gordon in the context of reinforcement learning.

```{prf:definition} Averager (Gordon)
:label: gordon-averager

An operator $\Proj: \mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}^{|\mathcal{S}|}$ is an **averager** if $\Proj v = Wv$ where $W$ is a $|\mathcal{S}| \times |\mathcal{S}|$ stochastic matrix: $w_{ij} \geq 0$ and $\sum_j w_{ij} = 1$ for all $i$.
```

Averagers automatically satisfy the monotonicity conditions: linearity follows from matrix multiplication, monotonicity follows from nonnegativity of entries, and constant preservation follows from row sums equaling one.

```{prf:theorem} Stability with averagers (Gordon)
:label: gordon-stability

If $\Proj$ is an averager and $\Bellman$ is the Bellman operator (a $\gamma$-contraction), then $\Proj \Bellman$ is a $\gamma$-contraction, and value iteration $v_{k+1} = \Proj \Bellman v_k$ converges to a unique fixed point.
```

This specializes the Santos-Vigo-Aguiar theorem to discrete states, expressed in the probabilistic language of stochastic matrices. The stochastic matrix characterization connects to Markov chain theory: $\Proj v$ represents expected values after one transition, and the monotonicity property reflects the fact that expectations preserve order.

**Examples of averagers** include state aggregation (averaging values within groups), K-nearest neighbors (averaging over nearest states), kernel smoothing with positive kernels, and multilinear interpolation on grids (barycentric weights are nonnegative and sum to one). **Counterexamples** include linear least squares regression (projection matrix may have negative entries) and high-order polynomial interpolation (Runge phenomenon produces negative weights).

The following table summarizes which common approximation operators satisfy the monotonicity conditions:

| **Method** | **Monotone?** | **Notes** |
|:-----------|:--------------|:----------|
| Piecewise linear interpolation | Yes | Always an averager; guaranteed stability |
| Multilinear interpolation (grid) | Yes | Barycentric weights are nonnegative and sum to one |
| Shape-preserving splines (Schumaker) | Yes | Designed to maintain monotonicity |
| State aggregation | Yes | Exact averaging within groups |
| Kernel smoothing (positive kernels) | Yes | If kernel integrates to one |
| High-order polynomial interpolation | No | Oscillations violate monotonicity (Runge phenomenon) |
| Least squares projection (arbitrary basis) | No | Projection matrix may have negative entries |
| Fourier/spectral methods | No | Not monotone-preserving in general |
| Neural networks | No | Highly flexible but no monotonicity guarantees |

The distinction between "safe" (monotone) and "potentially unstable" (non-monotone) approximators provides rigorous foundation for the folk wisdom that linear interpolation is reliable while high-order polynomials can be dangerous for value iteration. But notice that the table's verdict on "least squares projection" is somewhat abstract. It doesn't specifically address the three weighted residual methods we introduced at the start of this chapter.

The choice of solution method determines which approximation operators are safe to use. Successive approximation (fixed-point iteration) requires monotone approximators to guarantee convergence. Rootfinding methods like Newton's method do not require monotonicity. Stability depends on numerical properties of the Jacobian rather than contraction preservation. These considerations suggest hybrid strategies. One approach runs a few iterations with a monotone method to generate an initial guess, then switches to Newton's method with a smooth approximation for rapid final convergence. 

### Connecting Back to Collocation, Galerkin, and Least Squares

We have now developed a general stability theory for projected value iteration and surveyed which approximation operators are monotone. But what does this mean for the three specific weighted residual methods we introduced at the start of this chapter: **collocation**, **Galerkin**, and **least squares**? Each method defines a different projection operator $\Proj$, and we now need to determine which satisfy the monotonicity conditions that guarantee convergence.

Collocation with piecewise linear interpolation is monotone. When we use collocation with piecewise linear basis functions on a grid, the projection operator performs linear interpolation between grid points. At any state $s$ between grid points $s_i$ and $s_{i+1}$, the interpolated value is:

$$
(\Proj v)(s) = \frac{s_{i+1} - s}{s_{i+1} - s_i} v(s_i) + \frac{s - s_i}{s_{i+1} - s_i} v(s_{i+1}).
$$

The interpolation weights (barycentric coordinates) are nonnegative and sum to one, making this an averager in Gordon's sense. Therefore collocation with piecewise linear bases satisfies the monotonicity conditions and the Santos-Vigo-Aguiar stability theorem applies. The folk wisdom that "linear interpolation is safe for value iteration" has rigorous theoretical foundation.

Galerkin projection is generally not monotone. The Galerkin projection operator for a general basis $\{\varphi_1, \ldots, \varphi_n\}$ has the form:

$$
\Proj = \boldsymbol{\Phi}(\boldsymbol{\Phi}^\top \mathbf{W} \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \mathbf{W},
$$

where $\mathbf{W}$ is a diagonal weight matrix and $\boldsymbol{\Phi}$ contains the basis function evaluations. This projection matrix typically has **negative entries**. To see why, consider a simple example with polynomial basis functions $\{1, x, x^2\}$ on $[-1, 1]$. The projection of a function onto this space involves computing $(\boldsymbol{\Phi}^\top \mathbf{W} \boldsymbol{\Phi})^{-1}$, and the resulting operator can map nonnegative functions to functions with negative values. This is the same phenomenon underlying the Runge phenomenon in high-order polynomial interpolation: the projection weights oscillate in sign.

Since Galerkin projection is not monotone, the sup norm contraction theory does not guarantee convergence of projected value iteration $v_{k+1} = \Proj \Bellman v_k$ with Galerkin.

Least squares methods share the non-monotonicity issue. The least squares projection operator minimizes $\|\Residual(\hat{f})\|_w^2$ and has the same mathematical form as Galerkin projection. It is a linear projection onto $\text{span}\{\varphi_1, \ldots, \varphi_n\}$ with respect to a weighted inner product. Like Galerkin, the projection matrix typically contains negative entries and violates monotonicity.

The monotone approximator framework successfully covers collocation with simple bases, but leaves two important methods, Galerkin and least squares, without convergence guarantees. These methods are used in least-squares temporal difference learning (LSTD) and modern reinforcement learning with linear function approximation. We need a different analytical framework to understand when these non-monotone projections lead to convergent algorithms.

Monotone projections (piecewise linear interpolation, state aggregation) automatically preserve the Bellman operator's contraction property, guaranteeing convergence of projected value iteration. Non-monotone projections (Galerkin, high-order polynomials) may destroy contraction in the sup norm, requiring either different solution methods (Newton) or analysis in different norms. The next section develops the latter approach for policy evaluation.

```{admonition} Exercises: Monotonicity and Convergence
:class: hint dropdown

1. **Verifying monotonicity.** Consider piecewise linear interpolation on a 5-point grid. Write out the interpolation weights for a point between grid points 2 and 3. Verify that all weights are nonnegative and sum to one.

2. **A non-monotone example.** Using Lagrange interpolation with 4 equally spaced nodes on $[0, 1]$, compute the interpolation weights for the point $x = 0.1$. Show that some weights are negative.

3. **State aggregation.** Consider a discrete MDP with states $\{1, 2, 3, 4\}$ aggregated into two groups: $\{1, 2\}$ and $\{3, 4\}$. Write out the aggregation operator as a matrix.

4. **Contraction constant.** For the composed operator $\Proj \Bellman$ with a monotone $\Proj$, prove that the contraction constant is exactly $\gamma$.
```

## Beyond Monotone Approximators

If an orthogonal projection is not monotone, which weighting and policy
conditions can still make the projected Bellman map contractive?

The monotone approximator theory gives us a clean sufficient condition for convergence: if $\Proj$ is monotone (and constant-preserving), then $\Proj$ is non-expansive in the sup norm $\|\cdot\|_\infty$. Since $\Bellman$ is a $\gamma$-contraction in the sup norm, their composition $\Proj \Bellman$ is also a $\gamma$-contraction in the sup norm, guaranteeing convergence of projected value iteration.

But what if $\Proj$ is not monotone? Can we still guarantee convergence? Galerkin and least squares projections typically violate monotonicity, yet they are widely used in practice, particularly in reinforcement learning through least-squares temporal difference learning (LSTD). In general, proving convergence for non-monotone projections is difficult. However, for the special case of **policy evaluation**, computing the value function $v_\pi$ of a fixed policy $\pi$, we can establish convergence by working in a different norm.

### The Policy Evaluation Problem and LSTD

Consider the policy evaluation problem: given policy $\pi$, we want to solve the policy Bellman equation $v_\pi = r_\pi + \gamma \mathbf{P}_\pi v_\pi$, where $r_\pi$ and $\mathbf{P}_\pi$ are the reward vector and transition matrix under $\pi$. This is the core computational task in policy iteration, actor-critic algorithms, and temporal difference learning. In reinforcement learning, we typically learn from sampled experience: trajectories $(s_0, a_0, r_1, s_1, a_1, r_2, s_2, \ldots)$ generated by following $\pi$. If the Markov chain induced by $\pi$ is ergodic, the state distribution converges to a stationary distribution $\xi$ satisfying $\xi^\top \mathbf{P}_\pi = \xi^\top$.

This distribution determines which states appear frequently in our data. States visited often contribute more samples and have more influence on any learned approximation. States visited rarely contribute little. For a linear approximation $v_\theta(s) = \sum_j \theta_j \varphi_j(s)$, the **least-squares temporal difference (LSTD)** algorithm computes coefficients by solving:

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_\pi \boldsymbol{\Phi}) \boldsymbol{\theta} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{r}_\pi,
$$

where $\boldsymbol{\Phi}$ is the matrix of basis function evaluations and $\boldsymbol{\Xi} = \text{diag}(\xi)$. We write this matrix equation for analysis purposes, but the actual algorithm does not compute it this way. For large state spaces, we cannot enumerate all states to form $\boldsymbol{\Phi}$ or explicitly represent the transition matrix $\mathbf{P}_\pi$. Instead, the practical algorithm accumulates sums from sampled transitions $(s, r, s')$, incrementally building the matrices $\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \boldsymbol{\Phi}$ and $\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{P}_\pi \boldsymbol{\Phi}$ without ever forming the full objects. The algorithm is derived from first principles through temporal difference learning, and the Galerkin perspective provides an interpretation of what it computes.

### LSTD as Projected Bellman Equation

To see what this equation means, let $\hat{v} = \boldsymbol{\Phi} \boldsymbol{\theta}$ be the solution. Expanding the parentheses:

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \boldsymbol{\Phi} \boldsymbol{\theta} - \gamma \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{P}_\pi \boldsymbol{\Phi} \boldsymbol{\theta} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{r}_\pi.
$$

Moving all terms to the left side and factoring out $\boldsymbol{\Phi}^\top \boldsymbol{\Xi}$:

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} \boldsymbol{\theta} - \gamma \mathbf{P}_\pi \boldsymbol{\Phi} \boldsymbol{\theta} - \mathbf{r}_\pi) = \mathbf{0}.
$$

Since $\hat{v} = \boldsymbol{\Phi} \boldsymbol{\theta}$ and the policy Bellman operator is $\BellmanPi \hat{v} = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi \hat{v}$, we can write:

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\hat{v} - \BellmanPi \hat{v}) = \mathbf{0}.
$$

Let $\boldsymbol{\varphi}_j$ denote the $j$-th column of $\boldsymbol{\Phi}$, which contains the evaluations of the $j$-th basis function at all states. The equation above says that for each $j$:

$$
\boldsymbol{\varphi}_j^\top \boldsymbol{\Xi} (\hat{v} - \BellmanPi \hat{v}) = 0.
$$

But $\boldsymbol{\varphi}_j^\top \boldsymbol{\Xi} (\hat{v} - \BellmanPi \hat{v})$ is exactly the $\xi$-weighted inner product $\langle \boldsymbol{\varphi}_j, \hat{v} - \BellmanPi \hat{v} \rangle_\xi$. So the residual $\hat{v} - \BellmanPi \hat{v}$ is orthogonal to every basis function, and therefore orthogonal to the entire subspace $\text{span}(\boldsymbol{\Phi})$. 

By definition, the orthogonal projection $\Proj y$ of a vector $y$ onto a subspace is the unique vector in that subspace such that $y - \Proj y$ is orthogonal to the subspace. Here, $\hat{v}$ lies in $\text{span}(\boldsymbol{\Phi})$ (since $\hat{v} = \boldsymbol{\Phi} \boldsymbol{\theta}$), and we have just shown that $\BellmanPi \hat{v} - \hat{v}$ is orthogonal to $\text{span}(\boldsymbol{\Phi})$. Therefore, $\hat{v} = \Proj \BellmanPi \hat{v}$, where $\Proj$ is orthogonal projection onto $\text{span}(\boldsymbol{\Phi})$ with respect to the $\xi$-weighted inner product:

$$
\langle u, v \rangle_\xi = u^\top \boldsymbol{\Xi} v, \qquad \|v\|_\xi = \sqrt{v^\top \boldsymbol{\Xi} v}, \qquad \Proj = \boldsymbol{\Phi}(\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \boldsymbol{\Xi}.
$$

The weighting by $\xi$ is not arbitrary. Temporal difference learning performs stochastic updates using individual transitions: $\theta_{k+1} = \theta_k + \alpha_k (r + \gamma v_{\theta_k}(s') - v_{\theta_k}(s)) \nabla v_{\theta_k}(s)$, with states sampled from $\xi$. The ODE analysis of this stochastic process (Borkar-Meyn theory) shows convergence to a fixed point, which can be expressed in closed form as the $\xi$-weighted projected Bellman operator. LSTD is an algorithm that computes this analytical fixed point.

### Orthogonal Projection is Non-Expansive

Suppose $\xi$ is the steady-state distribution: $\xi^\top \mathbf{P}_\pi = \xi^\top$. Our goal is to establish that $\Proj \BellmanPi$ is a contraction in $\|\cdot\|_\xi$. If we can establish that $\Proj$ is non-expansive in this norm and that $\BellmanPi$ is a $\gamma$-contraction in $\|\cdot\|_\xi$, then their composition will be a $\gamma$-contraction:

$$
\|\Proj \BellmanPi v - \Proj \BellmanPi w\|_\xi \leq \|\BellmanPi v - \BellmanPi w\|_\xi \leq \gamma \|v - w\|_\xi.
$$


First, we establish that orthogonal projection is non-expansive. For any vector $v$, we can decompose $v = \Proj v + (v - \Proj v)$, where $(v - \Proj v)$ is orthogonal to the subspace $\text{span}(\boldsymbol{\Phi})$. By the Pythagorean theorem in the $\|\cdot\|_\xi$ inner product:

$$
\|v\|_\xi^2 = \|\Proj v\|_\xi^2 + \|v - \Proj v\|_\xi^2.
$$

Since $\|v - \Proj v\|_\xi^2 \geq 0$, we have:

$$
\|v\|_\xi^2 \geq \|\Proj v\|_\xi^2.
$$

Taking square roots of both sides (which preserves the inequality since both norms are non-negative):

$$
\|\Proj v\|_\xi \leq \|v\|_\xi.
$$

This holds for all $v$, so $\Proj$ is non-expansive in $\|\cdot\|_\xi$.

### Contraction of $\BellmanPi$ in $\|\cdot\|_\xi$

To show $\BellmanPi = r_\pi + \gamma \mathbf{P}_\pi$ is a $\gamma$-contraction, we need to verify:

$$
\|\BellmanPi v - \BellmanPi w\|_\xi = \|\gamma \mathbf{P}_\pi (v - w)\|_\xi = \gamma \|\mathbf{P}_\pi (v - w)\|_\xi.
$$

This will be at most $\gamma \|v - w\|_\xi$ if $\mathbf{P}_\pi$ is non-expansive, meaning $\|\mathbf{P}_\pi z\|_\xi \leq \|z\|_\xi$ for any vector $z$. We therefore need to establish that $\mathbf{P}_\pi$ is non-expansive in $\|\cdot\|_\xi$. 

*Before reading the proof below, try to show that $\mathbf{P}_\pi$ is non-expansive in $\|\cdot\|_\xi$. Hint: what property of $\xi$ relates it to $\mathbf{P}_\pi$?*

Consider the squared norm of $\mathbf{P}_\pi z$. By definition of the weighted norm:

$$
\|\mathbf{P}_\pi z\|_\xi^2 = \sum_s \xi(s) [(\mathbf{P}_\pi z)(s)]^2.
$$

The $s$-th component of $\mathbf{P}_\pi z$ is $(\mathbf{P}_\pi z)(s) = \sum_{s'} p(s'|s,\pi(s)) z(s')$. This is a weighted average of the values $z(s')$ with weights $p(s'|s,\pi(s))$ that sum to one. Therefore:

$$
\|\mathbf{P}_\pi z\|_\xi^2 = \sum_s \xi(s) \left[\sum_{s'} p(s'|s,\pi(s)) z(s')\right]^2.
$$

Since the function $x \mapsto x^2$ is convex, Jensen's inequality applied to the probability distribution $p(\cdot|s,\pi(s))$ gives:

$$
\left[\sum_{s'} p(s'|s,\pi(s)) z(s')\right]^2 \leq \sum_{s'} p(s'|s,\pi(s)) z(s')^2.
$$

Substituting this into the norm expression:

$$
\|\mathbf{P}_\pi z\|_\xi^2 \leq \sum_s \xi(s) \sum_{s'} p(s'|s,\pi(s)) z(s')^2 = \sum_{s'} z(s')^2 \sum_s \xi(s) p(s'|s,\pi(s)).
$$

The stationarity condition $\xi^\top \mathbf{P}_\pi = \xi^\top$ means $\sum_s \xi(s) p(s'|s,\pi(s)) = \xi(s')$ for all $s'$. Therefore:

$$
\|\mathbf{P}_\pi z\|_\xi^2 \leq \sum_{s'} z(s')^2 \xi(s') = \|z\|_\xi^2.
$$

Taking square roots, $\|\mathbf{P}_\pi z\|_\xi \leq \|z\|_\xi$, so $\mathbf{P}_\pi$ is non-expansive in $\|\cdot\|_\xi$. This makes $\BellmanPi = r_\pi + \gamma \mathbf{P}_\pi$ a $\gamma$-contraction in $\|\cdot\|_\xi$. Composing with the non-expansive projection:

$$
\|\Proj \BellmanPi v - \Proj \BellmanPi w\|_\xi \leq \|\BellmanPi v - \BellmanPi w\|_\xi \leq \gamma \|v - w\|_\xi.
$$

By Banach's fixed-point theorem, $\Proj \BellmanPi$ has a unique fixed point and iterates converge from any initialization.

### Interpretation: The On-Policy Condition

The result shows that convergence depends on matching the weighting to the operator. We cannot choose an arbitrary weighted $L^2$ norm and expect $\Proj \BellmanPi$ to be a contraction. Instead, the weighting $\xi$ must have a specific relationship with the transition matrix $\mathbf{P}_\pi$ in the operator $\BellmanPi$: namely, $\xi$ must be the stationary distribution of $\mathbf{P}_\pi$. This is what makes the weighted geometry compatible with the operator's structure. When this match holds, Jensen's inequality gives us non-expansiveness of $\mathbf{P}_\pi$ in the $\|\cdot\|_\xi$ norm, and the composition $\Proj \BellmanPi$ inherits the contraction property.

In reinforcement learning, this has a practical interpretation. When we learn by following policy $\pi$ and collecting transitions $(s, a, r, s')$, the states we visit are distributed according to the stationary distribution of $\pi$. This is **on-policy learning**. The LSTD algorithm uses data sampled from this distribution, which means the empirical weighting naturally matches the operator structure. Our analysis shows that the iterative algorithm $v_{k+1} = \Proj \BellmanPi v_k$ converges to the same fixed point that LSTD computes in closed form.

This is fundamentally different from the monotone approximator theory. There, we required structural properties of $\Proj$ itself (monotonicity, constant preservation) to guarantee that $\Proj$ preserves the sup-norm contraction property of $\Bellman$. Here, we place no such restriction on $\Proj$. Galerkin projection is not monotone. Instead, convergence depends on matching the norm to the operator. When $\xi$ does not match the stationary distribution, as in off-policy learning where data comes from a different behavior policy, the Jensen inequality argument breaks down. The operator $\mathbf{P}_\pi$ need not be non-expansive in $\|\cdot\|_\xi$, and $\Proj \BellmanPi$ may fail to contract. This explains divergence phenomena such as Baird's counterexample {cite}`Baird1995`.

```{admonition} Exercises: LSTD and the On-Policy Condition
:class: hint dropdown

1. **Computing the stationary distribution.** For the 3-state Markov chain in the LSTD example, compute the stationary distribution $\xi$ satisfying $\xi^\top \mathbf{P}_\pi = \xi^\top$.

2. **LSTD with stationary weighting.** Recompute the LSTD solution using the stationary distribution instead of uniform weighting. Compare the approximation error.

3. **Off-policy divergence.** Consider a weighting $\xi' = (0.9, 0.05, 0.05)$ that does not match the stationary distribution. Implement projected value iteration and observe whether it converges.

4. **Proving non-expansiveness fails off-policy.** For $\xi' = (0.9, 0.05, 0.05)$, find a vector $z$ such that $\|\mathbf{P} z\|_{\xi'} > \|z\|_{\xi'}$.
```

### The Bellman Optimality Case

Can we extend this weighted $L^2$ analysis to the Bellman optimality operator $\Bellman v = \max_a [r_a + \gamma \mathbf{P}_a v]$? The answer is no, at least not with this approach. The obstacle appears at the Jensen inequality step. For policy evaluation, we had:

$$
\|\mathbf{P}_\pi z\|_\xi^2 = \sum_s \xi(s) \left[\sum_{s'} p(s'|s,\pi(s)) z(s')\right]^2.
$$

The inner term is a convex combination of the values $z(s')$, which allowed us to apply Jensen's inequality to the convex function $x \mapsto x^2$. For the optimal Bellman operator, we would need to bound:

$$
\left[\max_{a} \sum_{s'} p(s'|s,a) z(s')\right]^2.
$$

But the maximum of convex combinations is not itself a convex combination. It is a pointwise maximum. Jensen's inequality does not apply. We cannot conclude that $\max_a [\mathbf{P}_a z]$ is non-expansive in any weighted $L^2$ norm.

Is convergence of $\Proj \Bellman$ with Galerkin projection impossible, or merely difficult to prove? The situation is subtle. In practice, fitted Q-iteration and approximate value iteration with neural networks often work well, suggesting that some form of stability exists. But there are also well-documented divergence examples (e.g., Q-learning with linear function approximation can diverge). The theoretical picture remains incomplete. Some results exist for restricted function classes or under strong assumptions on the MDP structure, but no general convergence guarantee like the policy evaluation result is available. The interplay between the max operator, the projection, and the norm geometry is not well understood. This is an active area of research in reinforcement learning theory.

Despite these theoretical gaps, the practical algorithm template is straightforward. We now present fitted-value iteration as a meta-algorithm that combines any supervised learning method with the Bellman operator.

## Fitted-Value/Q Iteration (FVI/FQI)

How does projected fixed-point iteration become a repeated supervised fitting
problem for values or action values?

We have developed weighted residual methods through abstract functional equations: choose test functions, impose orthogonality conditions $\langle R, p_i \rangle_w = 0$, solve for coefficients. What are we actually computing when we solve these equations by successive approximation? The answer is simpler than the formalism suggests: **function iteration with a fitting step**.

Recall that the weighted residual conditions $\langle v - \Bellman v, p_i \rangle_w = 0$ define a fixed-point problem $v = \Proj \Bellman v$, where $\Proj$ is a projection operator onto $\text{span}(\boldsymbol{\Phi})$. We can solve this by iteration: $v_{k+1} = \Proj \Bellman v_k$. Under appropriate conditions (monotonicity of $\Proj$, or matching the weight to the operator for policy evaluation), this converges to a solution.

In parameter space, this iteration becomes a fitting procedure. Consider Galerkin projection with a finite state space of $n$ states. Let $\boldsymbol{\Phi}$ be the $n \times d$ matrix of basis evaluations, $\mathbf{W}$ the diagonal weight matrix, and $\mathbf{y}$ the vector of Bellman operator evaluations: $y_i = (\Bellman v_k)(s_i)$. The projection is:

$$
\boldsymbol{\theta}_{k+1} = (\boldsymbol{\Phi}^\top \mathbf{W} \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \mathbf{W} \mathbf{y}.
$$

This is weighted least-squares regression of
$\boldsymbol{\Phi}\boldsymbol{\theta}$ on the targets $\mathbf{y}$.
Collocation instead requires the exact interpolation
$\boldsymbol{\Phi}\boldsymbol{\theta}_{k+1}=\mathbf{y}$ at the selected
points. In continuous state spaces, sampled states can approximate the
Galerkin integrals and produce a finite-dimensional regression problem.

This extends beyond linear basis functions. Neural networks, decision trees, and kernel methods all implement variants of this procedure. Given data $\{(s_i, y_i)\}$ where $y_i = (\Bellman v_k)(s_i)$, each method produces a function $v_{k+1}: \mathcal{S} \to \mathbb{R}$ from the targets. The projection operator $\Proj$ is one such approximation rule. Galerkin uses weighted projection, while square collocation uses exact interpolation at the selected points.

```{prf:algorithm} Fitted-Value Iteration
:label: fitted-value-iteration

**Inputs:** Finite state set $\mathcal{S}$ (or sample $\{s_i\}_{i=1}^n$), discount factor $\gamma$, function class $\mathcal{F}$, fitting procedure $\mathtt{fit}$, convergence tolerance $\epsilon$

**Output:** Approximate value function $\hat{v} \approx v^*$

1. Initialize $v_0 \in \mathcal{F}$ arbitrarily
2. Set $k \leftarrow 0$
3. **repeat**
4. $\quad$ **for** each state $s_i \in \mathcal{S}$ **do**
5. $\quad\quad$ Compute target: $y_i \leftarrow \displaystyle\max_{a \in \mathcal{A}} \Big\{ r(s_i, a) + \gamma \sum_{s'} p(s' \mid s_i, a) v_k(s') \Big\}$
6. $\quad$ **end for**
7. $\quad$ Fit new approximation: $v_{k+1} \leftarrow \mathtt{fit}\big(\{(s_i, y_i)\}_{i=1}^n; \mathcal{F}\big)$
8. $\quad$ $k \leftarrow k+1$
9. **until** $\|v_k - v_{k-1}\| < \epsilon$ (or maximum iterations reached)
10. **return** $v_k$
```

The operation $\mathtt{fit}$ may solve a linear system, run gradient descent,
or train an ensemble. For a linear space $\mathcal{F}$, weighted squared-error
fitting gives the Galerkin projection. A square collocation system gives exact
interpolation when its evaluation matrix is nonsingular. Fitted-value iteration
alternates between generating Bellman targets and constructing a new function
from them.

The following code demonstrates fitted-value iteration on the optimal stopping problem:

```{code-cell} python
:tags: [hide-input]
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

gamma = 0.9
v_bar_exact = (1 - np.sqrt(1 - gamma**2)) / gamma**2
s_star_exact = gamma * v_bar_exact
def v_exact(s):
    return np.where(s >= s_star_exact, s, gamma * v_bar_exact)

def fitted_value_iteration(s_grid, gamma, degree, max_iter=50, tol=1e-6):
    X = s_grid.reshape(-1, 1)
    v = np.zeros(len(s_grid))
    
    for k in range(max_iter):
        # Use trapezoidal rule for E[v] under uniform distribution on [0,1]
        v_bar = np.trapezoid(v, s_grid)
        targets = np.maximum(s_grid, gamma * v_bar)
        
        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-6))
        model.fit(X, targets)
        v_new = model.predict(X)
        
        if np.linalg.norm(v_new - v) < tol:
            return v_new, k + 1
        v = v_new
    return v, max_iter

s_grid = np.linspace(0, 1, 50)
print(f"{'Degree':<10} {'Iterations':<12} {'Max Error':<12}")
print("-" * 34)
for deg in [3, 5, 8]:
    v, iters = fitted_value_iteration(s_grid, gamma, deg)
    max_error = np.max(np.abs(v - v_exact(s_grid)))
    print(f"{deg:<10} {iters:<12} {max_error:<12.6f}")
```

A limitation of FVI/FQI is that it assumes we can evaluate the Bellman operator exactly. Computing $y_i = (\Bellman v_k)(s_i)$ requires knowing transition probabilities and summing over all next states. In practice, we often have only a simulator or observed data. The next chapter shows how to approximate these expectations from samples, connecting the fitted-value iteration framework to simulation-based methods.

## Summary

Projected Bellman iteration composes an approximation map with a Bellman
operator. Monotone interpolation and state aggregation preserve sup-norm
contraction, while non-monotone projections require a compatible weighting and
can lose the fixed-point guarantee. Fitted value and Q iteration expose the
computational pattern: evaluate Bellman targets, fit an approximator, and
repeat.

Exact target evaluation still assumes access to the transition probabilities
or an exact expectation. How can the same Bellman update be estimated when the
model supplies only samples? [Monte Carlo Bellman estimation](monte-carlo-bellman-estimation.md)
replaces the exact integral by sampled averages and makes their variance and
maximization bias explicit.

## Self-checks

:::{exercise} Weighting shift
:label: ex-projection-check-3

If the projection weighting distribution puts almost no mass on an important state region, what failure should you expect?
:::

:::{solution} ex-projection-check-3
:class: dropdown

The approximation may be accurate under the weighted norm yet poor in that neglected region, leading to bad values or decisions there.
:::
