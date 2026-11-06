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

The [previous chapter](weighted-residual-methods.md) built a template for any
functional equation: choose a basis, define the residual, test it against a
finite set of conditions, and solve the resulting system by function iteration
or by Newton's method. What changes when the residual is a Bellman residual,
$R(s;\theta) = \Bellman\hat{v}(s;\theta) - \hat{v}(s;\theta)$?

Two things, and they organize this chapter. First, the Bellman operator is one
we know how to iterate and how to linearize. Value iteration is successive
approximation on $\Bellman$ ({prf:ref}`value-iteration`), and policy iteration
is the Newton-Kantorovich method on the same equation
({ref}`sec-newton-kantorovich-pi`). Once a projection is inserted, the two
generic solvers of {ref}`sec-wr-solve` become **projected value iteration** and
**projected policy iteration**, and in the Galerkin case with a finite state
space they are the least-squares temporal difference (LSTD) and least-squares
policy iteration (LSPI) algorithms of reinforcement learning. Second, the
Bellman operator is a $\gamma$-contraction in the sup norm, which is what made
value iteration converge ({prf:ref}`value-iteration-convergence`). Function
iteration on the projected equation is $\hat{v}_{k+1} = \Proj\Bellman\hat{v}_k$,
and the question the previous chapter left open is whether the composed
operator $\Proj\Bellman$ is still a contraction. The answer has three parts. It
is, in the sup norm, when $\Proj$ has nonnegative weights, which continues the
discussion of cardinal functions in the
[finite-horizon chapter](finite-horizon-dp.md#handling-continuous-spaces-with-interpolation).
It is, in a weighted $L^2$ norm, for policy evaluation with Galerkin projection
when the weighting is the stationary distribution of the policy. And for the
optimality operator with a non-monotone projection there is no general
guarantee, which is the situation in which the fitted value and Q iteration
methods of the next chapters are nonetheless used.

::::{admonition} Learning Goals
:class: note

After completing this chapter, you should be able to:
- Write function iteration and Newton's method for a projected Bellman equation as projected value iteration and projected policy iteration
- Derive the LSTD equations as Galerkin projection of the policy evaluation equation, and LSPI as Newton's method on the projected optimality equation
- State and prove the conditions under which projected value iteration converges: monotone projections in the sup norm, on-policy weighting in the $\xi$-norm
- Explain why neither argument covers the optimality operator with a non-monotone projection
- Describe fitted value iteration as the computational pattern shared by every projection method

**Prerequisites:** the weighted-residual template ([previous chapter](weighted-residual-methods.md)), the contraction and Newton-Kantorovich material of [infinite-horizon MDPs](infinite-horizon-mdps.md), interpolation with cardinal functions in [finite-horizon DP](finite-horizon-dp.md).
::::

(sec-abe-projected)=
## Projected Value Iteration and Projected Policy Iteration

What do the two generic solvers of the previous chapter compute when the
operator is a Bellman operator, and which known algorithms do they reduce to?

Consider the Bellman optimality equation $v(s) = \Bellman v(s) = \max_{a \in \mathcal{A}_s} \{ r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j) \}$. For a candidate approximation $\hat{v}(s;\theta) = \sum_{i=1}^n \theta_i \varphi_i(s)$, the residual is

$$
R(s; \theta) = \Bellman\hat{v}(s;\theta) - \hat{v}(s;\theta) = \max_{a \in \mathcal{A}_s} \left\{ r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) \hat{v}(j;\theta) \right\} - \sum_{i=1}^n \theta_i \varphi_i(s).
$$

Each test criterion of {ref}`sec-wr-conditions` defines a map $\Proj$ from functions to $\mathcal{V} = \text{span}\{\varphi_1, \ldots, \varphi_n\}$: collocation interpolates at the chosen points, Galerkin takes the $w$-orthogonal projection, and least squares fits in the weighted norm. The two solvers of {ref}`sec-wr-solve` then read as follows.

**Function iteration** applies the operator and projects the result,

$$
\hat{v}_{k+1} = \Proj \Bellman \hat{v}_k.
$$ (eq:projected-vi)

This is value iteration with a projection after each sweep: **projected value iteration**. Nothing in it depends on the max. Replacing $\Bellman$ by the policy evaluation operator $\BellmanPi$ gives projected policy evaluation, and replacing it by the smooth operator $\mathrm{L}_\beta$ of {ref}`sec-regularized-operators` gives projected smooth value iteration.

**Newton's method** linearizes the residual map. For the Bellman operator the envelope theorem of {ref}`sec-wr-newton` gives the derivative of $\Bellman\hat{v}$ with respect to $\theta$ by freezing the greedy action, so the Jacobian involves the transition matrix $\mathbf{P}_{\pi_k}$ of the greedy policy $\pi_k$ at the current iterate. The Newton step then solves a *linear* projected equation for $\pi_k$, which is projected policy evaluation, and the next greedy policy is read off the result. Newton's method on the projected optimality equation is therefore **projected policy iteration**, the function-approximation version of the equivalence in {ref}`sec-newton-kantorovich-pi`. Two remarks carry over from the exact case. With the smooth Bellman operator the Jacobian is $\gamma \mathbf{P}_{\pi_k}$ for the softmax policy, with no ties and no active sets to track ({ref}`sec-smooth-pi-newton`). And the "run a few sweeps of function iteration, then switch to Newton" recommendation at the end of {ref}`sec-wr-solve` is exactly how Rust solved his bus-engine model on a grid ({ref}`sec-rust-inverse`).

The rest of this section writes out both solvers for collocation and for Galerkin, then specializes to a finite state space, where the Galerkin equations become LSTD and LSPI.

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
interpolant at the selected points: this is {eq}`eq:projected-vi` with $\Proj$
the interpolation operator. With piecewise-linear cardinal functions on a grid
it is the stationary version of the interpolated backward recursion
{eq}`eq:interpolated-bellman`.

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

When the state space is continuous, the expectation inside the max is an integral, which we approximate by numerical quadrature (Gauss-Hermite for normal shocks, and so on); the [next chapter](monte-carlo-bellman-estimation.md) treats this integration problem in its own right. The method is simple and robust when the interpolation preserves contraction, a question settled in {ref}`sec-abe-convergence`, but can be slow for large discount factors.

**Newton's method** for collocation treats the problem as rootfinding: $G(\theta) = \boldsymbol{\Phi} \theta - \mathrm{L}_\varphi(\theta) = 0$. The Jacobian is $J_G = \boldsymbol{\Phi} - J_{\mathrm{L}_\varphi}$, where the envelope theorem gives $[J_{\mathrm{L}_\varphi}]_{ij} = \gamma \sum_{s' \in \mathcal{S}} p(s'|s_i, a_i^*(\theta)) \varphi_j(s')$. Here $a_i^*(\theta)$ is the optimal action at collocation point $s_i$ given the current coefficients.

```{prf:algorithm} Collocation with Newton's Method
:label: collocation-newton

**Input** Collocation points $\{s_1, \ldots, s_n\}$, basis functions $\{\varphi_1, \ldots, \varphi_n\}$, initial $\theta^{(0)}$, tolerance $\varepsilon > 0$

**Output** Converged coefficients $\theta^*$

1. Form collocation matrix $\boldsymbol{\Phi}$ with $\Phi_{ij} = \varphi_j(s_i)$
2. $k \leftarrow 0$
3. **repeat**
    1. For each $i = 1, \ldots, n$:
        1. $t_i^{(k)} \leftarrow \max_{a \in \mathcal{A}_{s_i}} \left\{ r(s_i, a) + \gamma \sum_{s' \in \mathcal{S}} p(s'|s_i, a) \sum_{\ell=1}^n \theta_\ell^{(k)} \varphi_\ell(s') \right\}$
        2. Store $a_i^* \in \arg\max$ achieving the maximum
    2. Compute Jacobian: $[J_{\mathrm{L}_\varphi}]_{ij} = \gamma \sum_{s' \in \mathcal{S}} p(s'|s_i, a_i^*) \varphi_j(s')$ for all $i,j$
    3. Solve $(\boldsymbol{\Phi} - J_{\mathrm{L}_\varphi}) \Delta\theta = \boldsymbol{\Phi} \theta^{(k)} - t^{(k)}$
    4. $\theta^{(k+1)} \leftarrow \theta^{(k)} - \Delta\theta$
    5. $k \leftarrow k + 1$
4. **until** $\|\Delta\theta\| < \varepsilon$
5. **return** $\theta^{(k)}$
```

Writing $\pi_k$ for the greedy policy stored in step 3.1.2 and $\mathbf{P}_{\pi_k}$ for the $n \times |\mathcal{S}|$ matrix of its transition probabilities from the collocation points, the Newton step is $(\boldsymbol{\Phi} - \gamma \mathbf{P}_{\pi_k} \boldsymbol{\Phi}_{\mathcal{S}}) \theta^{(k+1)} = \mathbf{r}_{\pi_k}$, where $\boldsymbol{\Phi}_{\mathcal{S}}$ evaluates the basis at the successor states. This is the policy evaluation equation $(\mathbf{I} - \gamma \mathbf{P}_{\pi_k}) v = \mathbf{r}_{\pi_k}$ of {prf:ref}`policy-iteration-standard`, restricted to the collocation points and solved within the span of the basis. Each Newton step evaluates the current greedy policy and the next step improves it, so the algorithm is policy iteration with an interpolated value function. It converges rapidly near the solution but requires good initialization and more computation per iteration than function iteration.

Why is collocation popular for Bellman equations? Because it avoids integration when testing the residual. We only evaluate the Bellman operator at $n$ discrete points. In contrast, Galerkin requires integrating the residual against each basis function.

#### Worked Example: Collocation on the Optimal Stopping Problem

Returning to the motivating example of the previous chapter, let us trace through the collocation algorithm with $n = 4$ polynomial basis functions at Chebyshev nodes:

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

for iteration in range(60):
    def v_approx(s, th=theta):
        return sum(th[j] * s**j for j in range(n))
    v_bar, _ = quad(v_approx, 0, 1)
    test_points = np.linspace(0, 1, 100)
    max_error = max(abs(v_approx(s) - v_exact(s)) for s in test_points)
    if iteration % 5 == 0:
        print(f"{iteration:<6} {np.linalg.norm(theta):<12.6f} {max_error:<12.6f}")
    
    targets = np.maximum(nodes, gamma * v_bar)
    theta_new = np.linalg.solve(Phi, targets)
    if np.linalg.norm(theta_new - theta) < 1e-10:
        print(f"\nConverged in {iteration + 1} iterations; max error {max_error:.6f}")
        break
    theta = theta_new
```

The iteration converges linearly at the rate $\gamma$, as value iteration does, and the remaining error is the interpolation error of a cubic through four nodes on a function with a kink at $s^*$.

### Galerkin

For Galerkin, we use the basis functions themselves as test functions. The conditions are:

$$
\int_{\mathcal{S}} [\Bellman\hat{v}(s; \theta) - \hat{v}(s; \theta)] \varphi_i(s) w(s) ds = 0, \quad i = 1, \ldots, n.
$$

where $w(s)$ is a weight function: the uniform weight $w(s) = 1$, or a probability density over states, which in reinforcement learning is usually the stationary distribution of a policy and will be written $\xi$ below. Expanding this:

$$
\int_{\mathcal{S}} \left[ \max_a \left\{ r(s,a) + \gamma \mathbb{E}[v(s')] \right\} - \sum_j \theta_j \varphi_j(s) \right] \varphi_i(s) w(s) ds = 0.
$$

**Function iteration** projects the target $\Bellman\hat{v}(\cdot;\theta^{(k)})$ onto the basis, as in {ref}`sec-wr-function-iteration`. We seek $\theta^{(k+1)}$ satisfying:

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

The left side is a linear system with the mass matrix $M_{ij} = \int \varphi_i \varphi_j w$, and the right side requires integrating the Bellman operator output against each test function. When the basis functions are orthogonal polynomials with matching weight $w$, the mass matrix is diagonal, simplifying the solve. But we still need numerical integration to evaluate the right side, which makes Galerkin substantially more expensive than collocation per iteration.

**Newton's method** for Galerkin similarly requires integration. The residual is $R(s; \theta) = \Bellman\hat{v}(s; \theta) - \hat{v}(s; \theta)$, and we need $G_i(\theta) = \int R(s; \theta) \varphi_i(s) w(s) ds = 0$. The envelope theorem gives $\frac{\partial \Bellman\hat{v}(s; \theta)}{\partial \theta_j} = \gamma \mathbb{E}[\varphi_j(s') \mid s, a^*(s;\theta)]$, so the Jacobian entry is:

$$
J_{ij} = \int \left[ \gamma \mathbb{E}[\varphi_j(s') \mid s, a^*(s;\theta)] - \varphi_j(s) \right] \varphi_i(s) w(s) ds.
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

We must integrate expected basis function values under the greedy actions against the test functions, which requires both numerical integration and tracking of greedy actions across the state space. The advantage of Galerkin lies in its approximation properties: with orthogonal polynomials it is the best approximation in the weighted $L^2$ norm, which for smooth problems can yield better accuracy per degree of freedom than collocation. In practice, collocation's computational simplicity usually wins for Bellman equations, especially in high dimension where integration becomes prohibitively expensive. The exception is the finite-state case, where the integrals are sums and Galerkin is as cheap as collocation. We turn to it now.

```{admonition} Exercises: Collocation and Galerkin
:class: hint dropdown

1. **Effect of collocation points.** For the optimal stopping problem, compare collocation with $n=4$ Chebyshev nodes versus $n=4$ equally-spaced nodes. Which choice gives smaller maximum error?

2. **Threshold location.** The optimal stopping policy has a threshold structure. Using your polynomial approximation $\hat{v}(s)$, estimate the threshold by finding where $\hat{v}(s) = s$. Compare to the exact threshold.

3. **Orthogonal polynomials.** For $\mathcal{S} = [-1, 1]$, write out the Galerkin conditions using Chebyshev polynomials $T_i$ and weight $w(x) = 1/\sqrt{1-x^2}$. What makes this choice computationally convenient?

4. **Newton vs. function iteration.** How does the iteration count depend on $\gamma$? Try $\gamma \in \{0.5, 0.9, 0.99\}$.
```

(sec-abe-lstd)=
### Discrete State Spaces: LSTD and LSPI

When the state space is discrete and finite, the Galerkin integrals become sums and everything can be written in matrix form. For $\mathcal{S} = \{s_1, \ldots, s_m\}$, the Galerkin orthogonality conditions become weighted sums over states:

$$
\sum_{s \in \mathcal{S}} \xi(s) [\Bellman\hat{v}(s; \theta) - \hat{v}(s; \theta)] \varphi_i(s) = 0, \quad i = 1, \ldots, n,
$$

where $\xi(s) \geq 0$ with $\sum_s \xi(s) = 1$ is a probability distribution over states. Define the feature matrix $\boldsymbol{\Phi} \in \mathbb{R}^{m \times n}$ with entries $\Phi_{si} = \varphi_i(s)$ (each row contains the features for one state), and let $\boldsymbol{\Xi} = \text{diag}(\xi)$ be the diagonal matrix with the state distribution on the diagonal. The $\xi$-weighted inner product, norm, and orthogonal projection onto $\text{span}(\boldsymbol{\Phi})$ are

$$
\langle u, v \rangle_\xi = u^\top \boldsymbol{\Xi} v, \qquad \|v\|_\xi = \sqrt{v^\top \boldsymbol{\Xi} v}, \qquad \Proj = \boldsymbol{\Phi}(\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \boldsymbol{\Xi}.
$$ (eq:xi-projection)

**Policy evaluation.** For a fixed policy $\pi$ the Bellman operator is linear, $\BellmanPi \hat{v} = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi \boldsymbol{\Phi} \theta$, with $[\mathbf{r}_\pi]_s = r(s, \pi(s))$ and $[\mathbf{P}_\pi]_{sj} = p(j|s, \pi(s))$. The Galerkin conditions $\langle \BellmanPi \hat{v} - \hat{v}, \varphi_i \rangle_\xi = 0$ for all $i$ read, in matrix form,

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \boldsymbol{\Phi} \theta - \boldsymbol{\Phi} \theta) = \mathbf{0},
\qquad\text{that is}\qquad
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_\pi \boldsymbol{\Phi}) \theta = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{r}_\pi.
$$ (eq:lstd-galerkin)

This is the **LSTD (least-squares temporal difference)** solution of {cite:t}`BradtkeBarto1996`: a linear system $\mathbf{A} \theta = \mathbf{b}$ with $\mathbf{A} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_\pi \boldsymbol{\Phi})$ and $\mathbf{b} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{r}_\pi$. The left form of {eq}`eq:lstd-galerkin` says that the residual $\BellmanPi\hat{v} - \hat{v}$ is $\xi$-orthogonal to every column of $\boldsymbol{\Phi}$, hence to $\text{span}(\boldsymbol{\Phi})$, while $\hat{v}$ itself lies in that span. By the definition of orthogonal projection, this is the same as

$$
\hat{v} = \Proj \BellmanPi \hat{v},
$$ (eq:projected-bellman-pi)

the **projected Bellman equation**. LSTD computes the fixed point of the projected operator $\Proj\BellmanPi$ in closed form; projected value iteration {eq}`eq:projected-vi` computes the same point by iterating, if it converges. We write the matrix equation for analysis. The practical algorithm never forms $\boldsymbol{\Phi}$ over all states or $\mathbf{P}_\pi$ explicitly: it accumulates $\mathbf{A}$ and $\mathbf{b}$ from sampled transitions $(s, r, s')$, since $\mathbf{A} = \mathbb{E}_{s \sim \xi}[\boldsymbol{\varphi}(s)(\boldsymbol{\varphi}(s) - \gamma \boldsymbol{\varphi}(s'))^\top]$ and $\mathbf{b} = \mathbb{E}_{s \sim \xi}[\boldsymbol{\varphi}(s) r]$. That sampled form, and the incremental TD(0) algorithm that converges to the same fixed point, belong to the chapters that replace exact expectations by samples.

*Check that the dimensions work out: if we have $m$ states and $n$ basis functions, what are the dimensions of $\boldsymbol{\Phi}$, $\boldsymbol{\Xi}$, $\mathbf{P}_\pi$, and $\mathbf{A}$?*

#### Worked Example: LSTD for Policy Evaluation

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

**The optimality equation.** With the max back in, the Galerkin conditions are a system of $n$ nonlinear equations in $n$ unknowns:

$$
G(\theta) \equiv \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \big(\Bellman\hat{v}(\cdot; \theta) - \boldsymbol{\Phi} \theta\big) = \mathbf{0}.
$$

The two solvers now give genuinely different updates, and it is worth seeing both in matrix form. Let $\pi_k$ be the greedy policy at the current coefficients, $\pi_k(s) = \arg\max_a \{r(s,a) + \gamma \sum_j p(j|s,a) \boldsymbol{\varphi}(j)^\top \theta^{(k)}\}$, so that $\Bellman\hat{v}(\cdot;\theta^{(k)}) = \mathbf{r}_{\pi_k} + \gamma \mathbf{P}_{\pi_k} \boldsymbol{\Phi} \theta^{(k)}$.

*Projected value iteration* fits the basis to the current Bellman targets:

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \boldsymbol{\Phi}\, \theta^{(k+1)} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \big(\mathbf{r}_{\pi_k} + \gamma \mathbf{P}_{\pi_k} \boldsymbol{\Phi} \theta^{(k)}\big).
$$ (eq:projected-vi-matrix)

The unknown appears only on the left, through the mass matrix $\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \boldsymbol{\Phi}$. This is {eq}`eq:projected-vi` written in coordinates, a weighted least-squares regression of $\boldsymbol{\Phi}\theta$ on the targets.

*Newton's method* linearizes $G$. By the envelope theorem the Jacobian at $\theta^{(k)}$ is $J_G = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\gamma \mathbf{P}_{\pi_k} \boldsymbol{\Phi} - \boldsymbol{\Phi})$, and the Newton update $\theta^{(k+1)} = \theta^{(k)} - J_G^{-1} G(\theta^{(k)})$ becomes, after multiplying through by $J_G$ and cancelling the terms in $\theta^{(k)}$,

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_{\pi_k} \boldsymbol{\Phi})\, \theta^{(k+1)} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{r}_{\pi_k}.
$$ (eq:lspi)

This is the LSTD system {eq}`eq:lstd-galerkin` for the greedy policy $\pi_k$. Newton's method on the projected optimality equation therefore alternates two steps: compute the greedy policy of the current coefficients, then evaluate it by LSTD. That is **LSPI (least-squares policy iteration)** {cite}`LagoudakisParr2003`, and it is projected policy iteration exactly as announced at the start of the section. Each step solves a linear system in closed form, so the sequence of greedy policies converges in a few steps when it converges. It need not: unlike exact policy iteration, whose improvement step is monotone, the projected evaluation can rank policies inconsistently and LSPI can cycle among a few policies. This is the function-approximation counterpart of the question we take up next.

```{admonition} Exercises: LSTD and LSPI
:class: hint dropdown

1. **Computing the stationary distribution.** For the 3-state Markov chain above, compute the stationary distribution $\xi$ satisfying $\xi^\top \mathbf{P}_\pi = \xi^\top$.

2. **LSTD with stationary weighting.** Recompute the LSTD solution using the stationary distribution instead of uniform weighting. Compare the approximation error.

3. **Two updates.** Implement {eq}`eq:projected-vi-matrix` and {eq}`eq:lspi` on a small MDP with two actions. Confirm that both converge to the same coefficients and compare the number of iterations.
```

(sec-abe-convergence)=
## When Does Projected Value Iteration Converge?

Which approximation maps preserve the contraction of the Bellman operator, in
which norm, and what happens when they do not?

Value iteration converges because $\Bellman$ is a $\gamma$-contraction in the sup norm ({prf:ref}`banach-fixed-point`). Projected value iteration {eq}`eq:projected-vi` iterates the composed operator $\Proj \Bellman$ instead, and $\Proj$ depends on the method:

- For **collocation**, $\Proj$ interpolates values at the collocation points
- For **Galerkin**, $\Proj$ is orthogonal projection with respect to $\langle \cdot, \cdot \rangle_w$
- For **least squares**, $\Proj$ minimizes the weighted residual norm

The question is whether $\Proj \Bellman$ inherits the contraction. If not, the iteration may diverge, oscillate, or converge to a spurious fixed point even though the original problem is well-posed. The finite-horizon chapter met the same question for backward recursion: {prf:ref}`prop-interp-error-accumulation` showed that the error of the stored values is passed on unchanged from stage to stage when the interpolation weights are nonnegative ($\Lambda = 1$) and can be amplified by $\Lambda > 1$ otherwise. Over an infinite horizon, amplification at every stage is fatal, and the $\Lambda = 1$ condition becomes the first of two sufficient conditions below.

### Monotone Projections Preserve the Sup-Norm Contraction

The answer depends on structural properties of the approximation operator $\Proj$. This theory was developed independently across several research communities: computational economics {cite}`Judd1992,Judd1996,McGrattan1997,SantosVigoAguiar1998`, economic dynamics {cite}`Stachurski2009`, and reinforcement learning {cite}`Gordon1995,Gordon1999`. They arrived at essentially the same conditions.

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

The intuition is that a monotone, constant-preserving operator acts like a weighted average that respects order and cannot amplify differences between functions. For an interpolation operator $(\Proj f)(s) = \sum_j \ell_j(s) f(s_j)$ with cardinal functions $\ell_j$, the two conditions say that the weights are nonnegative and sum to one, which is exactly $\Lambda = \sup_s \sum_j |\ell_j(s)| = 1$ in the finite-horizon chapter. Combining nonexpansiveness with the contraction of the Bellman operator yields the main stability result.

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
By {prf:ref}`monotone-nonexpansive`, $\Proj$ is nonexpansive. Since $\Bellman$ is a $\gamma$-contraction,

$$
\|\Proj \Bellman f - \Proj \Bellman g\|_\infty \leq \|\Bellman f - \Bellman g\|_\infty \leq \gamma\|f-g\|_\infty,
$$

so $\Proj \Bellman$ is a $\gamma$-contraction and {prf:ref}`banach-fixed-point` gives the unique fixed point and global convergence. For the error bound, use $v_\Proj = \Proj\Bellman v_\Proj$ and $v^* = \Bellman v^*$:

$$
\|v_\Proj - v^*\|_\infty \leq \|\Proj\Bellman v_\Proj - \Proj\Bellman v^*\|_\infty + \|\Proj \Bellman v^* - v^*\|_\infty \leq \gamma \|v_\Proj - v^*\|_\infty + \|\Proj v^* - v^*\|_\infty,
$$

and rearranging gives the claim.
```

The fixed-point error is controlled by how well $\Proj$ can represent $v^*$. If $v^* \in \text{Range}(\Proj)$, then $\Proj v^* = v^*$ and the error vanishes. Otherwise the error is the best-approximation error amplified by $(1-\gamma)^{-1}$, the infinite-horizon counterpart of the sum $\delta_2 + \cdots + \delta_T$ in {prf:ref}`prop-interp-error-accumulation`.

For discrete-state problems, the monotonicity conditions have a natural reading as **averaging with nonnegative weights**. This characterization is due to Gordon.

```{prf:definition} Averager (Gordon)
:label: gordon-averager

An operator $\Proj: \mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}^{|\mathcal{S}|}$ is an **averager** if $\Proj v = Wv$ where $W$ is a $|\mathcal{S}| \times |\mathcal{S}|$ stochastic matrix: $w_{ij} \geq 0$ and $\sum_j w_{ij} = 1$ for all $i$.
```

Averagers automatically satisfy the monotonicity conditions: linearity follows from matrix multiplication, monotonicity from nonnegativity of the entries, and constant preservation from the row sums.

```{prf:theorem} Stability with averagers (Gordon)
:label: gordon-stability

If $\Proj$ is an averager and $\Bellman$ is the Bellman operator (a $\gamma$-contraction), then $\Proj \Bellman$ is a $\gamma$-contraction, and value iteration $v_{k+1} = \Proj \Bellman v_k$ converges to a unique fixed point.
```

This specializes the Santos-Vigo-Aguiar theorem to discrete states in the language of stochastic matrices, and it gives the averaging a Markov-chain reading: $\Proj v$ is an expected value after one random transition, and expectations preserve order. The finite-horizon chapter made the same observation about linear interpolation, which replaces a deterministic successor by a random grid node with the interpolation weights as probabilities. Composing that chain with the model's own transitions gives $\Proj\Bellman$ as the Bellman operator of a finite-state MDP, which is why it contracts.

**Examples of averagers** include state aggregation (averaging values within groups), $k$-nearest neighbors (averaging over nearest states), kernel smoothing with positive kernels, and multilinear interpolation on grids (barycentric weights are nonnegative and sum to one). **Counterexamples** include linear least squares regression (the projection matrix may have negative entries) and high-order polynomial interpolation (the Runge phenomenon produces negative weights). The table summarizes which common approximation operators satisfy the monotonicity conditions:

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

Read against the three methods of this chapter, the table says the following. **Collocation with piecewise linear interpolation is monotone.** Between grid points $s_i$ and $s_{i+1}$ the interpolant is

$$
(\Proj v)(s) = \frac{s_{i+1} - s}{s_{i+1} - s_i} v(s_i) + \frac{s - s_i}{s_{i+1} - s_i} v(s_{i+1}),
$$

an averager in Gordon's sense, so {prf:ref}`santos-vigo-aguiar-stability` applies and the folk wisdom that linear interpolation is safe for value iteration has a rigorous foundation. Collocation with the cubic through four Chebyshev nodes in the worked example is *not* monotone, and the iteration there converged because the operator happened to be mild, not because the theory guaranteed it. **Galerkin projection is generally not monotone.** The projection matrix $\Proj = \boldsymbol{\Phi}(\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \boldsymbol{\Xi}$ typically has negative entries: for the polynomial basis $\{1, x, x^2\}$ on $[-1, 1]$ it maps some nonnegative functions to functions with negative values, the same sign oscillation as in the Runge phenomenon. **Least squares shares the problem**, since its projection has the same form. So the sup-norm theory covers collocation with simple bases and leaves Galerkin and least squares, the projections behind LSTD and linear function approximation in reinforcement learning, without a guarantee.

The choice of solver also matters. Function iteration needs $\Proj\Bellman$ to contract. Newton's method does not; its convergence depends on the conditioning of the Jacobian and on the starting point. This is the reason for the hybrid strategy of {ref}`sec-wr-solve`: a few iterations with a monotone, low-order approximation to get a starting guess, then Newton's method with a smooth, high-order one.

```{admonition} Exercises: Monotonicity and Convergence
:class: hint dropdown

1. **Verifying monotonicity.** Consider piecewise linear interpolation on a 5-point grid. Write out the interpolation weights for a point between grid points 2 and 3. Verify that all weights are nonnegative and sum to one.

2. **A non-monotone example.** Using Lagrange interpolation with 4 equally spaced nodes on $[0, 1]$, compute the interpolation weights for the point $x = 0.1$. Show that some weights are negative.

3. **State aggregation.** Consider a discrete MDP with states $\{1, 2, 3, 4\}$ aggregated into two groups: $\{1, 2\}$ and $\{3, 4\}$. Write out the aggregation operator as a matrix.

4. **Contraction constant.** For the composed operator $\Proj \Bellman$ with a monotone $\Proj$, prove that the contraction constant is exactly $\gamma$.
```

### Orthogonal Projection and Policy Evaluation: The On-Policy Condition

If an orthogonal projection is not monotone, what can still make the projected Bellman operator contract? For the special case of **policy evaluation**, the answer is to change the norm. We work with the finite-state Galerkin setting of {ref}`sec-abe-lstd`, where the projected equation is {eq}`eq:projected-bellman-pi` and $\Proj$ is the $\xi$-orthogonal projection {eq}`eq:xi-projection`. The result below is due to {cite:t}`TsitsiklisVanRoy1997`, who proved it to explain the convergence of TD(0) with linear function approximation.

In reinforcement learning the weighting $\xi$ is not a free choice. Data come from trajectories $(s_0, a_0, r_1, s_1, a_1, r_2, \ldots)$ generated by following $\pi$, and if the Markov chain induced by $\pi$ is ergodic, the state distribution converges to the stationary distribution $\xi$ satisfying $\xi^\top \mathbf{P}_\pi = \xi^\top$. States visited often contribute more samples and weigh more in any fit; states visited rarely contribute little. Suppose, then, that $\xi$ is the stationary distribution of $\pi$. We will show that $\Proj$ is nonexpansive in $\|\cdot\|_\xi$ and that $\BellmanPi$ is a $\gamma$-contraction in $\|\cdot\|_\xi$, so that their composition is a $\gamma$-contraction:

$$
\|\Proj \BellmanPi v - \Proj \BellmanPi w\|_\xi \leq \|\BellmanPi v - \BellmanPi w\|_\xi \leq \gamma \|v - w\|_\xi.
$$

**Orthogonal projection is nonexpansive.** For any vector $v$, decompose $v = \Proj v + (v - \Proj v)$, where $(v - \Proj v)$ is $\xi$-orthogonal to $\text{span}(\boldsymbol{\Phi})$. By the Pythagorean theorem in the $\xi$-inner product,

$$
\|v\|_\xi^2 = \|\Proj v\|_\xi^2 + \|v - \Proj v\|_\xi^2 \geq \|\Proj v\|_\xi^2,
$$

so $\|\Proj v\|_\xi \leq \|v\|_\xi$ for all $v$. Note what was *not* needed: no sign condition on the entries of $\Proj$. The sup-norm argument failed for this projection; the $\xi$-norm argument does not see the negative weights at all.

**$\BellmanPi$ contracts in $\|\cdot\|_\xi$.** Since $\BellmanPi v - \BellmanPi w = \gamma \mathbf{P}_\pi (v - w)$, it suffices to show that $\mathbf{P}_\pi$ is nonexpansive: $\|\mathbf{P}_\pi z\|_\xi \leq \|z\|_\xi$ for every $z$.

*Before reading on, try to show this yourself. Hint: what property of $\xi$ relates it to $\mathbf{P}_\pi$?*

The $s$-th component of $\mathbf{P}_\pi z$ is $\sum_{s'} p(s'|s,\pi(s)) z(s')$, a weighted average of the values $z(s')$. Since $x \mapsto x^2$ is convex, Jensen's inequality for the distribution $p(\cdot|s,\pi(s))$ gives

$$
\left[\sum_{s'} p(s'|s,\pi(s)) z(s')\right]^2 \leq \sum_{s'} p(s'|s,\pi(s)) z(s')^2.
$$

Weighting by $\xi$ and summing over $s$,

$$
\|\mathbf{P}_\pi z\|_\xi^2 \leq \sum_s \xi(s) \sum_{s'} p(s'|s,\pi(s)) z(s')^2 = \sum_{s'} z(s')^2 \sum_s \xi(s) p(s'|s,\pi(s)) = \sum_{s'} z(s')^2 \xi(s') = \|z\|_\xi^2,
$$

where the last step is stationarity, $\sum_s \xi(s) p(s'|s,\pi(s)) = \xi(s')$. So $\mathbf{P}_\pi$ is nonexpansive and $\BellmanPi$ is a $\gamma$-contraction in $\|\cdot\|_\xi$. Composing with the nonexpansive projection, $\Proj \BellmanPi$ is a $\gamma$-contraction in $\|\cdot\|_\xi$, and by {prf:ref}`banach-fixed-point` it has a unique fixed point to which projected policy evaluation converges from any initialization. That fixed point is the LSTD solution {eq}`eq:lstd-galerkin`, and the argument of {prf:ref}`santos-vigo-aguiar-stability` gives $\|v_\Proj - v_\pi\|_\xi \leq (1-\gamma)^{-1} \|\Proj v_\pi - v_\pi\|_\xi$; the sharper constant $(1-\gamma^2)^{-1/2}$ of {cite:t}`TsitsiklisVanRoy1997` uses the inner-product structure.

**The on-policy condition.** Convergence depends on matching the weighting to the operator. We cannot choose an arbitrary weighted $L^2$ norm and expect $\Proj \BellmanPi$ to be a contraction: $\xi$ must be the stationary distribution of the very $\mathbf{P}_\pi$ that appears in $\BellmanPi$. In reinforcement learning this is **on-policy learning**. When we collect transitions by following $\pi$, the empirical weighting is the stationary distribution of $\pi$, and the projection is compatible with the operator. When $\xi$ is the stationary distribution of a different behavior policy, as in off-policy learning, the Jensen step fails, $\mathbf{P}_\pi$ need not be nonexpansive in $\|\cdot\|_\xi$, and $\Proj\BellmanPi$ may fail to contract. This is the mechanism behind divergence examples such as Baird's counterexample {cite}`Baird1995`.

This is a different kind of result from the monotone theory. There, the structure of $\Proj$ (nonnegative weights) preserved the sup-norm contraction of any Bellman operator. Here, nothing is assumed about $\Proj$ beyond orthogonality, and the compatibility between the weighting and the operator does the work.

```{admonition} Exercises: The On-Policy Condition
:class: hint dropdown

1. **Non-expansiveness fails off-policy.** For the 3-state chain of the LSTD example and the weighting $\xi' = (0.9, 0.05, 0.05)$, find a vector $z$ such that $\|\mathbf{P}_\pi z\|_{\xi'} > \|z\|_{\xi'}$.

2. **Losing the argument is not losing convergence.** Implement projected policy evaluation $v_{k+1} = \Proj \BellmanPi v_k$ for the same chain with $\xi'$. It still converges: compute the spectral radius of $\gamma \Proj \mathbf{P}_\pi$ and check that it equals $\gamma$ for this feature set whatever the weighting. Explain why losing nonexpansiveness does not imply divergence. Then drop the constant feature, keep $\varphi(s) = s$ alone, and find a weighting concentrated on state 1 under which the spectral radius exceeds one and the iteration diverges.

3. **Baird's counterexample.** Look up Baird's star MDP and reproduce the divergence of projected policy evaluation under its off-policy weighting.
```

### The Optimality Operator

Can the weighted $L^2$ argument be extended to the Bellman optimality operator $\Bellman v = \max_a [\mathbf{r}_a + \gamma \mathbf{P}_a v]$? Not with this approach. The obstacle is the Jensen step. For policy evaluation, the inner term $\sum_{s'} p(s'|s,\pi(s)) z(s')$ is a convex combination of the values $z(s')$. For the optimality operator we would need to bound

$$
\left[\max_{a} \sum_{s'} p(s'|s,a) z(s')\right]^2,
$$

and the maximum of convex combinations is not a convex combination. Jensen's inequality does not apply, and there is no single stationary distribution to match, since each action has its own transition matrix. We cannot conclude that $\Bellman$ is nonexpansive in any weighted $L^2$ norm.

Is convergence of $\Proj \Bellman$ with a non-monotone projection impossible, or merely unproved? Both happen. Projected value iteration with linear least squares can diverge on small examples, and Q-learning with linear function approximation can diverge. Yet fitted Q-iteration and approximate value iteration with neural networks often work well in practice. No general guarantee like the policy-evaluation result exists; results are available for restricted function classes or under strong assumptions on the MDP, and the interplay between the max, the projection, and the norm is an active research area. The alternative solver does not escape the problem: LSPI {eq}`eq:lspi` replaces the max by a sequence of policy evaluations, each of which converges under the on-policy condition, but the sequence of policies has no monotonicity guarantee and can cycle.

Despite these gaps, the algorithm template is the same for every projection method, and it is simple. We now state it in the form used by the rest of the book.

## Nonlinear Approximators

What remains of the residual formulation when the value function is represented
by a neural network rather than a linear basis expansion?

The methods so far used linear function classes: polynomial bases, piecewise-linear interpolants, linear combinations of fixed features. Neural networks, kernel methods, and decision trees do not fit this template. The right way to extend it is through the projection step of {eq}`eq:projected-vi`, not through the Galerkin conditions themselves.

Recall what projected value iteration does in coordinates, {eq}`eq:projected-vi-matrix`: it fits $\hat{v}(\cdot;\theta)$ to the fixed targets $y = \Bellman\hat{v}_k$ by minimizing $\|\hat{v}(\cdot;\theta) - y\|_w^2$. For any model $v_{\boldsymbol{\theta}}$ with parameters $\boldsymbol{\theta} \in \mathbb{R}^p$, the same fit is

$$
\boldsymbol{\theta}_{k+1} = \arg\min_{\boldsymbol{\theta}} \|v_{\boldsymbol{\theta}} - \Bellman v_{\boldsymbol{\theta}_k}\|_w^2,
$$

and its first-order stationarity condition is an orthogonality condition:

$$
\Big\langle v_{\boldsymbol{\theta}} - \Bellman v_{\boldsymbol{\theta}_k}, \frac{\partial v_{\boldsymbol{\theta}}}{\partial \theta_i} \Big\rangle_w = 0 \quad \text{for all } i.
$$

The test functions are the partial derivatives $\partial v_{\boldsymbol{\theta}}/\partial\theta_i$, which span the tangent space of the manifold $\{v_{\boldsymbol{\theta}}\}$ at the current parameters. In the linear case $v_{\boldsymbol{\theta}} = \sum_i \theta_i \varphi_i$ the derivatives are the fixed basis functions and we recover Galerkin. For a nonlinear model the test functions move with $\boldsymbol{\theta}$, the condition is nonlinear, and it is solved by gradient descent on the squared error. Because the target $\Bellman v_{\boldsymbol{\theta}_k}$ is held fixed during the fit, the gradient only differentiates $v_{\boldsymbol{\theta}}$. This is the **semi-gradient** of temporal difference learning. Bellman residual minimization, the least-squares criterion of {ref}`sec-wr-conditions`, would instead minimize $\|v_{\boldsymbol{\theta}} - \Bellman v_{\boldsymbol{\theta}}\|_w^2$ with the same $\boldsymbol{\theta}$ on both sides. Its test functions are $\partial v_{\boldsymbol{\theta}}/\partial\theta_i - \partial(\Bellman v_{\boldsymbol{\theta}})/\partial\theta_i$, the **full gradient**, and computing them requires differentiating through the operator, which with a hard max means the envelope theorem and with the smooth operator of the [regularized DP chapter](regularized-dp.md) is an ordinary derivative. The two criteria have different fixed points; the chapters on [fitted Q iteration](fitted-q-iteration.md) use the first, and the semi-gradient versus full-gradient distinction returns there.

The **dual pairing** formulation {cite}`LegrandJunca2025` extends this framework to settings where test objects need not be regular functions. We have been informal about this distinction in our treatment of collocation, but the Dirac deltas $\delta(x - x_i)$ used there are not classical functions. They are distributions, defined rigorously only through their action on test functions via $\langle \Residual(v), \delta(x - x_i) \rangle = (\Residual v)(x_i)$. The simple calculus argument for orthogonality does not apply directly to such objects; the dual pairing framework provides the proper mathematical foundation. The induced dual norm $\|\Residual(v)\|_* = \sup_{\|w\|=1} |\langle \Residual(v), w \rangle|$ measures residuals by their worst-case effect on test functions, a perspective that has inspired adversarial formulations {cite}`Zang2020` where both trial and test functions are learned.

The unifying principle across linear and nonlinear methods is orthogonality of a residual to a set of test functions. Linear methods use fixed test functions and admit closed-form solutions. Nonlinear methods use parameter-dependent test functions and require iterative optimization. In both cases, the convergence theory of {ref}`sec-abe-convergence` is what is given up: a neural network is not an averager, and nothing matches its fitting norm to the operator.

## Fitted Value Iteration

How does projected fixed-point iteration become a repeated supervised fitting
problem for values or action values?

We have developed projection methods through functional equations: choose test functions, impose orthogonality conditions, solve for coefficients. Solved by function iteration, every one of them computes the same thing: **Bellman targets, then a fit**. In parameter space, with a finite state space of $m$ states, $n$ features, and weighting $\boldsymbol{\Xi}$, the Galerkin projection of the targets $\mathbf{y}$ with $y_s = (\Bellman v_k)(s)$ is {eq}`eq:projected-vi-matrix`,

$$
\boldsymbol{\theta}_{k+1} = (\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{y},
$$

a weighted least-squares regression of $\boldsymbol{\Phi}\boldsymbol{\theta}$ on the targets. Collocation instead requires the exact interpolation $\boldsymbol{\Phi}\boldsymbol{\theta}_{k+1} = \mathbf{y}$ at the selected points. In continuous state spaces, sampled states replace the Galerkin integrals and again produce a regression problem. Neural networks, decision trees, and kernel methods all implement the same procedure with a different fitting rule: given data $\{(s_i, y_i)\}$ with $y_i = (\Bellman v_k)(s_i)$, each produces a function $v_{k+1}: \mathcal{S} \to \mathbb{R}$ from the targets.

```{prf:algorithm} Fitted-Value Iteration
:label: fitted-value-iteration

**Inputs:** Finite state set $\mathcal{S}$ (or sample $\{s_i\}_{i=1}^m$), discount factor $\gamma$, function class $\mathcal{F}$, fitting procedure $\mathtt{fit}$, convergence tolerance $\epsilon$

**Output:** Approximate value function $\hat{v} \approx v^*$

1. Initialize $v_0 \in \mathcal{F}$ arbitrarily
2. Set $k \leftarrow 0$
3. **repeat**
4. $\quad$ **for** each state $s_i \in \mathcal{S}$ **do**
5. $\quad\quad$ Compute target: $y_i \leftarrow \displaystyle\max_{a \in \mathcal{A}} \Big\{ r(s_i, a) + \gamma \sum_{s'} p(s' \mid s_i, a) v_k(s') \Big\}$
6. $\quad$ **end for**
7. $\quad$ Fit new approximation: $v_{k+1} \leftarrow \mathtt{fit}\big(\{(s_i, y_i)\}_{i=1}^m; \mathcal{F}\big)$
8. $\quad$ $k \leftarrow k+1$
9. **until** $\|v_k - v_{k-1}\| < \epsilon$ (or maximum iterations reached)
10. **return** $v_k$
```

The operation $\mathtt{fit}$ may solve a linear system, run gradient descent,
or train an ensemble. For a linear space $\mathcal{F}$, weighted squared-error
fitting gives the Galerkin projection. A square collocation system gives exact
interpolation when its evaluation matrix is nonsingular. The convergence
results of {ref}`sec-abe-convergence` apply when $\mathtt{fit}$ is an averager,
and not otherwise. The following code runs fitted-value iteration on the
optimal stopping problem with polynomial regression of increasing degree, a
non-monotone fit that converges here all the same:

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

Replacing the state value by the action value $q(s,a)$ turns the target into $y_i = r(s_i,a_i) + \gamma \sum_{s'} p(s'|s_i,a_i) \max_{a'} q_k(s',a')$ and the algorithm into **fitted Q-iteration**, which the [chapter of that name](fitted-q-iteration.md) develops from batch data to DQN. Both assume that the Bellman operator can be evaluated exactly: computing $y_i$ requires the transition probabilities and a sum over next states. In practice we often have only a simulator or observed data. The next chapter shows how to approximate these expectations from samples.

## Summary and Outlook

Inserting a projection into the Bellman equation turns the two generic
solvers of the previous chapter into projected value iteration,
$\hat{v}_{k+1} = \Proj\Bellman\hat{v}_k$, and projected policy iteration,
which evaluates each greedy policy within the span of the basis. For a finite
state space and Galerkin projection these are the LSTD and LSPI algorithms.
Whether the projected iteration converges depends on whether $\Proj$ keeps the
contraction of $\Bellman$: monotone interpolation and state aggregation keep it
in the sup norm, orthogonal projection keeps it in the $\xi$-norm for policy
evaluation under the on-policy weighting, and the optimality operator with a
non-monotone projection has no general guarantee. Fitted value and Q iteration
expose the computational pattern shared by all of these methods: evaluate
Bellman targets, fit an approximator, repeat.

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

:::{exercise} Two solvers, two algorithms
:label: ex-projection-check-4

Projected value iteration and LSPI both use the greedy policy of the current coefficients. What does each do with it?
:::

:::{solution} ex-projection-check-4
:class: dropdown

Projected value iteration uses it once, to compute Bellman targets, and then fits the basis to those targets. LSPI solves the projected policy evaluation equation {eq}`eq:lstd-galerkin` for that policy, which is a Newton step on the projected optimality equation.
:::
