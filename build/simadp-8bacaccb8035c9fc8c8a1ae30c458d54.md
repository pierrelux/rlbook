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

# Simulation-Based Approximate Dynamic Programming 

The projection methods from the previous chapter showed how to transform the infinite-dimensional fixed-point problem $\Bellman v = v$ into a finite-dimensional one by choosing basis functions $\{\varphi_i\}$ and imposing conditions that make the residual $R(s) = \Bellman v(s) - v(s)$ small. Different projection conditions (Galerkin orthogonality, collocation at points, least squares minimization) yield different finite-dimensional systems to solve.

However, we left unresolved the question of how to evaluate the Bellman operator itself. Applying $\Bellman$ at any state requires computing an expectation:

$$
(\Bellman v)(s) = \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \int v(s')p(ds'|s,a)\right\}
$$

For discrete state spaces with a manageable number of states, this expectation is a finite sum we can compute exactly. For continuous or high-dimensional state spaces, we need numerical integration. The projection methods framework is compatible with any quadrature scheme, but leaves the choice of integration method unspecified.

This chapter addresses the **integration subproblem** that arises at two levels in approximate dynamic programming. First, we must evaluate the transition expectation $\int v(s')p(ds'|s,a)$ within the Bellman operator itself. Second, when using projection methods like Galerkin or least squares, we encounter outer integrations for enforcing orthogonality conditions or minimizing residuals over distributions of states. Both require numerical approximation in continuous or high-dimensional spaces.

We begin by examining deterministic numerical integration methods: quadrature rules that approximate integrals by evaluating integrands at carefully chosen points with associated weights. We discuss how to coordinate the choice of quadrature with the choice of basis functions to balance approximation accuracy and computational cost. Then we turn to **Monte Carlo integration**, which approximates expectations using random samples rather than deterministic quadrature points. This shift from deterministic to stochastic integration is what brings us into machine learning territory. When we replace exact transition probabilities with samples drawn from simulations or real interactions, projection methods combined with Monte Carlo integration become what the operations research community calls **simulation-based approximate dynamic programming** and what the machine learning community calls **reinforcement learning**. By relying on samples rather than explicit probability functions, we move from model-based planning to data-driven learning.

## Evaluating the Bellman Operator with Numerical Quadrature

Before turning to Monte Carlo methods, we examine the structure of the numerical integration problem. When we apply the Bellman operator to an approximate value function $\hat{v}(s; \theta) = \sum_i \theta_i \varphi_i(s)$, we must evaluate integrals of the form:

$$ \int \hat{v}(s'; \theta) \, p(s'|s,a) \, ds' = \int \left(\sum_i \theta_i \varphi_i(s')\right) p(s'|s,a) \, ds' = \sum_i \theta_i \int \varphi_i(s') \, p(s'|s,a) \, ds' $$

This shows two independent approximations:

1. **Value function approximation**: We represent $v$ using basis functions $\{\varphi_i\}$ and coefficients $\theta$
2. **Quadrature approximation**: We approximate each integral $\int \varphi_i(s') p(s'|s,a) ds'$ numerically

These choices are independent but should be coordinated. To see why, consider what happens in projection methods when we iterate $\hat{v}^{(k+1)} = \Proj \Bellman \hat{v}^{(k)}$. In practice, we cannot evaluate $\Bellman$ exactly due to the integrals, so we compute $\hat{v}^{(k+1)} = \Proj \widehat{\Bellman}_Q \hat{v}^{(k)}$ instead, where $\widehat{\Bellman}_Q$ denotes the Bellman operator with numerical quadrature.

The error in a single iteration can be bounded by the triangle inequality. Let $v$ denote the true fixed point $v = \Bellman v$. Then:

$$
\begin{aligned}
\|v - \hat{v}^{(k+1)}\| &= \|v - \Proj \widehat{\Bellman}_Q \hat{v}^{(k)}\| \\
&\le \|v - \Proj \Bellman v\| + \|\Proj \Bellman v - \Proj \Bellman \hat{v}^{(k)}\| + \|\Proj \Bellman \hat{v}^{(k)} - \Proj \widehat{\Bellman}_Q \hat{v}^{(k)}\| \\
&= \underbrace{\|v - \Proj \Bellman v\|}_{\text{best approximation error}} + \underbrace{\|\Proj \Bellman v - \Proj \Bellman \hat{v}^{(k)}\|}_{\text{contraction of iterate error}} + \underbrace{\|\Proj \Bellman \hat{v}^{(k)} - \Proj \widehat{\Bellman}_Q \hat{v}^{(k)}\|}_{\text{quadrature error}}
\end{aligned}
$$

The first term is the best we can do with our basis (how well $\Proj$ approximates the true solution). The second term decreases with iterations when $\Proj \Bellman$ is a contraction. The third term is the error from replacing exact integrals with quadrature, and it does not vanish as iterations proceed.

To make this concrete, consider evaluating $(\Bellman \hat{v})(s)$ for our current approximation $\hat{v}(s; \theta) = \sum_i \theta_i \varphi_i(s)$. We want:

$$ (\Bellman \hat{v})(s) = \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_i \theta_i \int \varphi_i(s') p(s'|s,a) ds' \right\} $$

But we compute instead:

$$ (\widehat{\Bellman}_Q \hat{v})(s) = \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_i \theta_i \sum_j w_j \varphi_i(s'_j) \right\} $$

where $\{(s'_j, w_j)\}$ are quadrature nodes and weights. If the quadrature error $\|(\Bellman \hat{v})(s) - (\widehat{\Bellman}_Q \hat{v})(s)\|$ is large relative to the basis approximation quality, we cannot exploit the expressive power of the basis. For instance, degree-10 Chebyshev polynomials represent smooth functions to $O(10^{-8})$ accuracy, but combined with rectangle-rule quadrature (error $O(h^2) \approx 10^{-2}$), the quadrature term dominates the error bound. We pay the cost of storing and manipulating 10 coefficients but achieve only $O(h^2)$ convergence in the quadrature mesh size.

This echoes the coordination principle from continuous optimal control (Chapter on trajectory optimization): when transcribing a continuous-time problem, we use the same quadrature nodes for both the running cost integral and the dynamics integral. There, coordination ensures that "where we pay" aligns with "where we enforce" the dynamics. Here, coordination ensures that integration accuracy matches approximation accuracy. Both are instances of balancing multiple sources of error in numerical methods.

Standard basis-quadrature pairings achieve this balance:

- Piecewise constant or linear elements with midpoint or trapezoidal rules
- Chebyshev polynomials with Gauss-Chebyshev quadrature
- Legendre polynomials with Gauss-Legendre quadrature
- Hermite polynomials with Gauss-Hermite quadrature (for Gaussian shocks)

To make this concrete, we examine what these pairings look like in practice for collocation and Galerkin projection.

### Orthogonal Collocation with Chebyshev Polynomials

Consider approximating the value function using Chebyshev polynomials of degree $n-1$:

$$
\hat{v}(s; \theta) = \sum_{i=0}^{n-1} \theta_i T_i(s)
$$

For orthogonal collocation, we place collocation points at the zeros of $T_n(s)$, denoted $\{s_j\}_{j=1}^n$. At each collocation point, we require the Bellman equation to hold exactly:

$$
\hat{v}(s_j; \theta) = \max_{a \in \mathcal{A}} \left\{r(s_j,a) + \gamma \int \hat{v}(s'; \theta) p(ds'|s_j,a)\right\}
$$

The integral on the right must be approximated using quadrature. With Chebyshev-Gauss quadrature using the same nodes $\{s_k\}_{k=1}^n$ and weights $\{w_k\}_{k=1}^n$, this becomes:

$$
\hat{v}(s_j; \theta) = \max_{a \in \mathcal{A}} \left\{r(s_j,a) + \gamma \sum_{k=1}^n w_k \hat{v}(s_k; \theta) p(s_k|s_j,a)\right\}
$$

Substituting the basis representation $\hat{v}(s_k; \theta) = \sum_{i=0}^{n-1} \theta_i T_i(s_k)$:

$$
\sum_{i=0}^{n-1} \theta_i T_i(s_j) = \max_{a \in \mathcal{A}} \left\{r(s_j,a) + \gamma \sum_{k=1}^n w_k p(s_k|s_j,a) \sum_{i=0}^{n-1} \theta_i T_i(s_k)\right\}
$$

Rearranging:

$$
\sum_{i=0}^{n-1} \theta_i T_i(s_j) = \max_{a \in \mathcal{A}} \left\{r(s_j,a) + \gamma \sum_{i=0}^{n-1} \theta_i \underbrace{\sum_{k=1}^n w_k T_i(s_k) p(s_k|s_j,a)}_{B_{ji}^a}\right\}
$$

This yields a system of $n$ nonlinear equations (one per collocation point):

$$
\sum_{i=0}^{n-1} T_i(s_j) \theta_i = \max_{a \in \mathcal{A}} \left\{r(s_j,a) + \gamma \sum_{i=0}^{n-1} B_{ji}^a \theta_i\right\}, \quad j=1,\ldots,n
$$

The matrix elements $B_{ji}^a$ can be precomputed once the quadrature nodes, weights, and transition probabilities are known. Solving this system gives the coefficient vector $\theta$.

### Galerkin Projection with Hermite Polynomials

For a problem with Gaussian shocks, we might use Hermite polynomials $\{H_i(s)\}_{i=0}^{n-1}$ weighted by the Gaussian density $\phi(s)$. The Galerkin condition requires:

$$
\int \left(\Bellman \hat{v}(s; \theta) - \hat{v}(s; \theta)\right) H_j(s) \phi(s) ds = 0, \quad j=0,\ldots,n-1
$$

Expanding the Bellman operator:

$$
\int \left[\max_{a \in \mathcal{A}} \left\{r(s,a) + \gamma \int \hat{v}(s'; \theta) p(ds'|s,a)\right\} - \hat{v}(s; \theta)\right] H_j(s) \phi(s) ds = 0
$$

We approximate this outer integral using Gauss-Hermite quadrature with nodes $\{s_\ell\}_{\ell=1}^m$ and weights $\{w_\ell\}_{\ell=1}^m$:

$$
\sum_{\ell=1}^m w_\ell \left[\max_{a \in \mathcal{A}} \left\{r(s_\ell,a) + \gamma \int \hat{v}(s'; \theta) p(ds'|s_\ell,a)\right\} - \hat{v}(s_\ell; \theta)\right] H_j(s_\ell) = 0
$$

The inner integral (transition expectation) is also approximated using Gauss-Hermite quadrature:

$$
\int \hat{v}(s'; \theta) p(ds'|s_\ell,a) \approx \sum_{k=1}^m w_k \hat{v}(s_k; \theta) p(s_k|s_\ell,a)
$$

Substituting the basis representation and collecting terms:

$$
\sum_{\ell=1}^m w_\ell H_j(s_\ell) \max_{a \in \mathcal{A}} \left\{r(s_\ell,a) + \gamma \sum_{i=0}^{n-1} \theta_i \sum_{k=1}^m w_k H_i(s_k) p(s_k|s_\ell,a)\right\} = \sum_{\ell=1}^m w_\ell H_j(s_\ell) \sum_{i=0}^{n-1} \theta_i H_i(s_\ell)
$$

This gives $n$ nonlinear equations in $n$ unknowns. The right-hand side simplifies using orthogonality: when using the same quadrature nodes for projection and integration, $\sum_{\ell=1}^m w_\ell H_j(s_\ell) H_i(s_\ell) = \delta_{ij} \|H_j\|^2$.

In both cases, the basis-quadrature pairing ensures that:
1. The quadrature nodes appear in both the transition expectation and the outer projection
2. The quadrature accuracy matches the polynomial approximation order
3. Precomputed matrices capture the dynamics, making iterations efficient

## Monte Carlo Integration 

Deterministic quadrature rules work well when the state space has low dimension, the transition density is smooth, and we can evaluate it cheaply at arbitrary points. In many stochastic control problems none of these conditions truly hold. The state may be high dimensional, the dynamics may be given by a simulator rather than an explicit density, and the cost of each call to the model may be large. In that regime, deterministic quadrature becomes brittle. Monte Carlo methods offer a different way to approximate expectations, one that relies only on the ability to **sample** from the relevant distributions.

### Monte Carlo as randomized quadrature

Consider a single expectation of the form

$$
J = \int f(x)\,p(dx) = \mathbb{E}[f(X)],
$$

where $X \sim p$ and $f$ is some integrable function. Monte Carlo integration approximates $J$ by drawing independent samples $X^{(1)},\ldots,X^{(N)} \sim p$ and forming the sample average

$$
\hat{J}_N \equiv \frac{1}{N}\sum_{n=1}^N f\bigl(X^{(n)}\bigr).
$$

This estimator has two basic properties that will matter throughout this chapter.

First, it is **unbiased**:

$$
\mathbb{E}[\hat{J}_N]
= \mathbb{E}\left[\frac{1}{N}\sum_{n=1}^N f(X^{(n)})\right]
= \frac{1}{N}\sum_{n=1}^N \mathbb{E}[f(X^{(n)})]
= \mathbb{E}[f(X)]
= J.
$$

Second, its variance scales as $1/N$. If we write

$$
\sigma^2 \equiv \mathrm{Var}[f(X)],
$$

then independence of the samples gives

$$
\mathrm{Var}(\hat{J}_N)
= \frac{1}{N^2}\sum_{n=1}^N \mathrm{Var}\bigl(f(X^{(n)})\bigr)
= \frac{\sigma^2}{N}.
$$

The central limit theorem then says that for large $N$,

$$
\sqrt{N}\,(\hat{J}_N - J) \Longrightarrow \mathcal{N}(0,\sigma^2),
$$

so the integration error decays at rate $O(N^{-1/2})$. This rate is slow compared to high-order quadrature in low dimension, but it has one crucial advantage: it does not explicitly depend on the dimension of $x$. Monte Carlo integration pays in variance, not in an exponential growth in the number of nodes.

It is often helpful to view Monte Carlo as **randomized quadrature**. A deterministic quadrature rule selects nodes $x_j$ and weights $w_j$ in advance and computes

$$
\sum_j w_j f(x_j).
$$

Monte Carlo can be written in the same form: if we draw $X^{(n)}$ from density $p$, the sample average

$$
\hat{J}_N = \frac{1}{N}\sum_{n=1}^N f(X^{(n)})
$$

is just a quadrature rule with random nodes and equal weights. More advanced Monte Carlo schemes, such as importance sampling, change both the sampling distribution and the weights, but the basic idea remains the same.

### Monte Carlo evaluation of the Bellman operator

We now apply this to the Bellman operator. For a fixed value function $v$ and a given state-action pair $(s,a)$, the transition part of the Bellman operator is

$$
\int v(s')\,p(ds' \mid s,a)
= \mathbb{E}\left[v(S') \mid S = s, A = a\right].
$$

If we can simulate next states $S'^{(1)},\ldots,S'^{(N)}$ from the transition kernel $p(\cdot \mid s,a)$, either by calling a simulator or by interacting with the environment, we can approximate this expectation by

$$
\widehat{\mathbb{E}}_N\bigl[v(S') \mid s,a\bigr]
\equiv
\frac{1}{N}\sum_{n=1}^N v\bigl(S'^{(n)}\bigr).
$$

If the immediate reward is also random, say

$$
r = r(S,A,S') \quad \text{with} \quad (S', r) \sim p(\cdot \mid s,a),
$$

we can approximate the full one-step return

$$
\mathbb{E}\bigl[r + \gamma v(S') \mid S = s, A = a\bigr]
$$

by

$$
\widehat{G}_N(s,a)
\equiv
\frac{1}{N}\sum_{n=1}^N \bigl[r^{(n)} + \gamma v(S'^{(n)})\bigr],
$$

where $(r^{(n)}, S'^{(n)})$ are independent samples given $(s,a)$. Again, this is an unbiased estimator of the Bellman expectation for fixed $v$.

Plugging this into the Bellman operator gives a **Monte Carlo Bellman operator**:

$$
(\widehat{\Bellman}_N v)(s)
\equiv
\max_{a \in \mathcal{A}_s}
\left\{
\widehat{G}_N(s,a)
\right\}.
$$

The expectation inside the braces is now replaced by a random sample average. In model-based settings, we implement this by simulating many next states for each candidate action $a$ at state $s$. In model-free settings, we obtain samples from real interactions and re-use them to estimate the expectation.

At this stage nothing about approximation or projection has entered yet. For a fixed value function $v$, Monte Carlo provides unbiased, noisy evaluations of $(\Bellman v)(s)$. The approximation question arises once we couple this stochastic evaluation with basis functions and projections.

### Sampling the outer expectations

Projection methods introduce a second layer of integration. In Galerkin and least squares schemes, we choose a distribution $\mu$ over states (and sometimes actions) and enforce conditions of the form

$$
\int R(s; \theta)\,p_i(s)\, \mu(ds) = 0
\quad \text{or} \quad
\int R(s; \theta)^2\,\mu(ds) \text{ is minimized}.
$$

Here $R(s; \theta)$ is the residual function, such as

$$
R(s; \theta) = (\Bellman \hat{v})(s;\theta) - \hat{v}(s;\theta),
$$

and $p_i$ are test functions or derivatives of the residual with respect to parameters.

Note a subtle but important shift in perspective from the previous chapter. There, the weight function $w(s)$ in the inner product $\langle f, g \rangle_w = \int f(s) g(s) w(s) ds$ could be any positive weight function (not necessarily normalized). For Monte Carlo integration, however, we need a probability distribution we can sample from. We write $\mu$ for this sampling distribution and express projection conditions as integrals with respect to $\mu$: $\int R(s; \theta) p_i(s) \mu(ds)$. If a problem was originally formulated with an unnormalized weight $w(s)$, we must either (i) normalize it to define $\mu$, or (ii) use importance sampling with a different $\mu$ and reweight samples by $w(s)/\mu(s)$. In reinforcement learning, $\mu$ is typically the empirical state visitation distribution from collected trajectories.

These outer integrals over $s$ are generally not easier to compute than the inner transition expectations. Monte Carlo gives a way to approximate them as well. If we can draw states $S^{(1)},\ldots,S^{(M)} \sim \mu$, we can approximate, for example, the Galerkin orthogonality conditions by

$$
\int R(s; \theta)\,p_i(s)\,\mu(ds)
\approx
\frac{1}{M}\sum_{m=1}^M R\bigl(S^{(m)}; \theta\bigr)\,p_i\bigl(S^{(m)}\bigr).
$$

Similarly, a least squares objective

$$
\int R(s; \theta)^2\,\mu(ds)
$$

is approximated by the empirical risk

$$
\frac{1}{M}\sum_{m=1}^M R\bigl(S^{(m)}; \theta\bigr)^2.
$$

If we now substitute Monte Carlo estimates for both the inner transition expectations and the outer projection conditions, we obtain fully simulation-based schemes. We no longer need explicit access to the transition kernel $p(\cdot \mid s,a)$ or the state distribution $\mu$. It is enough to be able to **sample** from them, either through a simulator or by interacting with the real system. 

### From quadrature-based planning to sample-based learning

We can now see the parallel between the two levels of approximation discussed earlier and their Monte Carlo counterparts.

For the Bellman operator, we move from

$$
\int v(s')\,p(ds' \mid s,a)
\quad \text{to} \quad
\frac{1}{N}\sum_{n=1}^N v\bigl(S'^{(n)}\bigr).
$$

For projection, we move from exact integrals over $\mu$ to empirical averages over sampled states. If we write $\Proj_M$ for the projection operator defined by empirical averages over $M$ samples, and $\widehat{\Bellman}_N$ for the Monte Carlo Bellman operator based on $N$ transition samples, the iteration implemented in practice has the form

$$
\hat{v}^{(k+1)} = \Proj_M\,\widehat{\Bellman}_N \hat{v}^{(k)}.
$$

The structure of the error decomposition remains similar to the quadrature case. There is still a best approximation error from the finite basis, and a contraction effect from repeated application of a suitable operator. What changes is the nature of the integration error. Deterministic quadrature introduces a fixed bias that does not average out with iterations. Monte Carlo introduces **random error** with zero mean (for fixed $\hat{v}^{(k)}$), but nonzero variance that depends on $N$ and the variability of the one-step returns.

This distinction becomes important once we turn to learning schemes in which the same samples are reused across iterations, or where the parameters $\theta$ that define $\hat{v}$ are updated using noisy Bellman targets. In that setting, Monte Carlo integration interacts with function approximation in more subtle ways. It will generate estimators that are no longer unbiased for the **fixed point** of the projected Bellman equation, even though each individual Monte Carlo estimate of an expectation is unbiased for its integrand.

One particular issue arises when we combine Monte Carlo sampling with the maximization in the Bellman operator. While the Monte Carlo estimate of the expected return for any individual action is unbiased, taking the maximum over these noisy estimates introduces a systematic upward bias. This overestimation bias can compound through iterations and lead to poor policies.


## Amortizing Action Selection via Q-Functions

Monte Carlo integration enables model-free approximate dynamic programming: we no longer need explicit transition probabilities $p(s'|s,a)$, only the ability to sample next states. However, one computational challenge remains. The standard formulation of an optimal decision rule is

$$
\pi(s) = \arg\max_{a \in \mathcal{A}} \left\{r(s,a) + \gamma \int v(s')p(ds'|s,a)\right\}.
$$

Even with an optimal value function $v^*$ in hand, extracting an action at state $s$ requires evaluating the transition expectation for each candidate action. In the model-free setting, this means we must draw Monte Carlo samples from each action's transition distribution every time we select an action. This repeated sampling "at inference time" wastes computation, especially when the same state is visited multiple times.

We can amortize this computation by working at a different level of representation. Define the **state-action value function** (or **Q-function**)

$$
q(s,a) = r(s,a) + \gamma \int v(s')p(ds'|s,a).
$$

The Q-function caches the result of evaluating each action at each state. Once we have $q$, action selection reduces to a finite maximization:

$$
\pi(s) = \arg\max_{a \in \mathcal{A}(s)} q(s,a).
$$

No integration appears in this expression. The transition expectation has been precomputed and stored in $q$ itself.

The optimal Q-function $q^*$ satisfies its own Bellman equation. Substituting the definition of $v^*(s) = \max_a q^*(s,a)$ into the expression for $q$:

$$
q^*(s,a) = r(s,a) + \gamma \int p(ds'|s,a) \max_{a' \in \mathcal{A}(s')} q^*(s', a').
$$

This defines a Bellman operator on Q-functions:

$$
(\Bellman q)(s,a) = r(s,a) + \gamma \int p(ds'|s,a)\max_{a' \in \mathcal{A}(s')} q(s', a').
$$

Like the Bellman operator on value functions, $\Bellman$ is a $\gamma$-contraction in the sup-norm, guaranteeing a unique fixed point $q^*$. We can thus apply the same projection and Monte Carlo techniques developed for value functions to Q-functions. The computational cost shifts from action selection (repeated sampling at decision time) to training (evaluating expectations during the iterations of approximate value iteration). Once $q$ is learned, acting is cheap.

```{prf:algorithm} Parametric Q-Value Iteration
:label: simadp-parametric-q-value-iteration

**Input** Given an MDP $(S, A, P, R, \gamma)$, base points $\mathcal{B} \subset S$, function approximator class $q(s,a; \boldsymbol{\theta})$, maximum iterations $N$, tolerance $\varepsilon > 0$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ (e.g., for zero initialization)
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D} \leftarrow \emptyset$
    2. For each $(s,a) \in \mathcal{B} \times A$:
        1. $y_{s,a} \leftarrow r(s,a) + \gamma \int p(ds'|s,a)\max_{a' \in A} q(s',a'; \boldsymbol{\theta}_n)$
        2. $\mathcal{D} \leftarrow \mathcal{D} \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D})$
    4. $\delta \leftarrow \frac{1}{|\mathcal{D}||A|}\sum_{(s,a) \in \mathcal{D} \times A} (q(s,a; \boldsymbol{\theta}_{n+1}) - q(s,a; \boldsymbol{\theta}_n))^2$
    5. $n \leftarrow n + 1$
4. **until** ($\delta < \varepsilon$ or $n \geq N$)
5. **return** $\boldsymbol{\theta}_n$
```

### Upward Bias from Maximizing Noisy Estimates

The source of the problem is simple to state: we are now optimizing over noisy estimates of expected returns. Each Monte Carlo estimate is, by itself, unbiased. The bias appears when we take a maximum over several such noisy estimates and then feed that maximum back into the next iteration.

Fix some state $s$ and a current value function $v$. For each action $a$, define the true expected continuation value

$$
\mu(s,a) \equiv \int v(s')\,p(ds' \mid s,a),
$$

so that the exact Bellman update is

$$
(\Bellman v)(s) = \max_{a \in \mathcal{A}_s} \left\{ r(s,a) + \gamma \mu(s,a)\right\}.
$$

Under Monte Carlo integration with $N$ samples, we replace $\mu(s,a)$ by the empirical mean

$$
\hat{\mu}_N(s,a) \equiv \frac{1}{N} \sum_{i=1}^N v(s'_{i}), \quad s'_{i} \sim p(\cdot \mid s,a),
$$

and define the empirical Bellman update

$$
(\widehat{\Bellman}_N v)(s) = \max_{a \in \mathcal{A}_s} \left\{ r(s,a) + \gamma \hat{\mu}_N(s,a)\right\}.
$$

For any fixed $s$ and $a$, the Monte Carlo estimator is unbiased:

$$
\mathbb{E}\big[\hat{\mu}_N(s,a)\big] = \mu(s,a).
$$

If we removed the maximization over actions, we would have an unbiased approximation of $(\Bellman v)(s)$. The problem is that we do not remove it. Instead, we compute

$$
\max_a \left\{ r(s,a) + \gamma \hat{\mu}_N(s,a)\right\},
$$

where each term inside the maximum is a random quantity. The nonlinearity of the max operator breaks unbiasedness.

### A Simple Inequality

To see this clearly, suppress $s$ in the notation and consider random variables

$$
Y_a \equiv r(a) + \gamma \hat{\mu}_N(a), \quad a \in \mathcal{A},
$$

with expectations

$$
\mathbb{E}[Y_a] = r(a) + \gamma \mu(a).
$$

Let $a^*$ be any action that maximizes the **true** expected return:

$$
a^* \in \arg\max_{a} \left\{r(a) + \gamma \mu(a)\right\}.
$$

Then, for every outcome of the randomness,

$$
\max_a Y_a \ge Y_{a^*}.
$$

This is not a probabilistic statement but a simple algebraic fact: the maximum of a collection of numbers is at least as large as any particular element of that collection. Since this inequality holds for every realization of the random variables, we can take expectations on both sides. Expectation is monotone: if $X \ge Z$ almost surely, then $\mathbb{E}[X] \ge \mathbb{E}[Z]$. Applying this gives

$$
\mathbb{E}\big[\max_a Y_a\big] \ge \mathbb{E}[Y_{a^*}]
= r(a^*) + \gamma \mu(a^*)
= (\Bellman v)(s).
$$

So we obtain the inequality

$$
\mathbb{E}\big[(\widehat{\Bellman}_N v)(s)\big]
=
\mathbb{E}\Big[\max_a \left\{r(s,a) + \gamma \hat{\mu}_N(s,a)\right\}\Big]
\ge
(\Bellman v)(s).
$$

The inequality is strict as soon as there is more than one action and at least one of the $\hat{\mu}_N(s,a)$ has nonzero variance. In other words, even if each action's Monte Carlo estimate is unbiased, the maximization step introduces an upward bias.

This is a general phenomenon: for any collection of random variables $\{Y_a\}$,

$$
\mathbb{E}[\max_a Y_a] \ge \max_a \mathbb{E}[Y_a],
$$

because the maximum of random variables is a convex function of the vector $(Y_a)_a$. We have simply applied that fact in the specific setting of our empirical Bellman operator.

### A One-State Thought Experiment

It is instructive to strip away all structure and consider a one-state MDP with two actions $a_1$ and $a_2$. Suppose that, under the current value function $v$, both actions have the same true continuation value:

$$
\mu(a_1) = \mu(a_2) = \mu.
$$

We estimate each using

$$
\hat{\mu}_N(a_i) = \mu + \varepsilon_i, \quad i = 1,2,
$$

where $\varepsilon_1$ and $\varepsilon_2$ are zero-mean noise terms (for instance, independent Gaussian random variables with variance $\sigma^2/N$).

Then

$$
(\Bellman v) = r + \gamma \mu
$$

is the true next value, while the empirical update is

$$
\begin{aligned}
(\widehat{\Bellman}_N v) &= r + \gamma \max\{\hat{\mu}_N(a_1), \hat{\mu}_N(a_2)\} \\
&= r + \gamma \left(\mu + \max\{\varepsilon_1,\varepsilon_2\}\right).
\end{aligned}
$$

Since $\varepsilon_1$ and $\varepsilon_2$ are symmetric around zero, the expected maximum of the two is strictly positive:

$$
\mathbb{E}[\max\{\varepsilon_1,\varepsilon_2\}] > 0.
$$

Consequently,

$$
\begin{aligned}
\mathbb{E}\big[(\widehat{\Bellman}_N v)\big]
&= r + \gamma \left(\mu + \mathbb{E}[\max\{\varepsilon_1,\varepsilon_2\}]\right) \\
&> r + \gamma \mu \\
&= (\Bellman v).
\end{aligned}
$$

Nothing in the problem "wants" action $a_1$ or $a_2$ to be better. The only asymmetry comes from the random noise. Yet by always picking the empirically best action, we systematically favor "lucky" noise realizations. The algorithm interprets these lucky draws as evidence that the optimal value is higher than it truly is.

### Bias Accumulation Under Iteration

The Monte Carlo value iteration algorithm applies this empirical operator repeatedly:

$$
v_{k+1} \approx \widehat{\Bellman}_N v_k.
$$

At every iteration, for every state, we:

1. Generate noisy estimates of expected returns for each action.
2. Take a maximum over these noisy estimates.
3. Use this maximum as the new value estimate and feed it into the next iteration.

The upward bias therefore does not cancel out over time. Instead, it can be amplified in two ways.

First, the algorithm is recursive: the next round of Monte Carlo estimates is taken around a value function that is already slightly too optimistic. When we maximize over noisy estimates again, the new "bump" from favorable noise is added on top of the previous one.

Second, once a state–action pair has been overestimated through a lucky sequence of samples, it becomes more attractive in the maximization step. The algorithm will tend to pick it again, which increases the chance of reinforcing that overestimate. Negative noise, in contrast, tends to be ignored because it lowers an action's estimated value and makes it less likely to be selected.

The result is an algorithm that, in expectation, computes a value function that satisfies

$$
\mathbb{E}[v_k] \ge v^*
$$

componentwise, where $v^*$ is the true optimal value function. How large the gap becomes depends on the variance of the Monte Carlo estimates, the number of actions, and the number of iterations. But the important qualitative fact is that the gap is systematically upward.

### Implications for Value Iteration

In the exact, model-based setting, value iteration has two reassuring properties:

1. The Bellman operator is a contraction, so repeated application converges to $v^*$.
2. The sequence of values remains "honest" in the sense that each iterate is the result of applying a deterministic operator to the previous iterate.

Once we replace $\Bellman$ with $\widehat{\Bellman}_N$, both properties need to be revisited. Our operator is now random, and, because of the maximization over noisy estimates, it is optimistic on average. The algorithm tends to overstate the achievable return, even if the number of samples $N$ is held fixed and the number of iterations goes to infinity.

This upward bias appears in many reinforcement learning algorithms that bootstrap and maximize over noisy value estimates, and it motivates a whole family of variance-reduction and "double" methods that we will encounter later in the chapter.

## Learning the Bias Correction

We now turn to methods for addressing the overestimation bias introduced by maximizing over noisy Monte Carlo estimates. The approach we examine here, developed by Keane and Wolpin {cite}`Keane1994` in the context of dynamic discrete choice models, takes a direct approach: learn a model of the bias itself, then subtract it from the Bellman updates.

For a given value function $v$ and Monte Carlo sample size $N$, define the bias at state $s$ as

$$
\delta(s) = \mathbb{E}\big[(\widehat{\Bellman}_N v)(s)\big] - (\Bellman v)(s).
$$

If we could compute $\delta(s)$, we could correct our empirical updates by subtracting it. While we cannot compute either the expectation $\mathbb{E}[(\widehat{\Bellman}_N v)(s)]$ or the exact Bellman application $(\Bellman v)(s)$ directly, the bias $\delta(s)$ depends on observable quantities:

1. The number of Monte Carlo samples $N$: more samples reduce variance and thus reduce the bias from maximization.
2. The number of actions $|\mathcal{A}_s|$: more actions increase the chance that at least one gets a favorable draw.
3. The spread of action values: when actions have similar values, small noise can flip the maximizer.

Rather than computing $\delta(s)$ analytically, Keane and Wolpin proposed learning it empirically. The strategy is to perform high-fidelity simulation at a subset of states to estimate the bias, then fit a regression model that predicts the bias from simple features. This learned correction can then be applied throughout the state space.

The method follows the "simulate on a subset, interpolate everywhere" template common in econometric dynamic programming. At a carefully chosen set of states, we simulate the Bellman operator with both high and low fidelity to estimate the bias. We then build a regression surrogate that predicts this bias from features of the state and action-value distribution. During routine value iteration, we use the fast low-fidelity Monte Carlo estimate and subtract the predicted bias correction.

To formalize this, consider a state $s$ with current value function $v$. Let

$$
\mu(s,a) = \int v(s') \, p(ds' \mid s,a), \qquad \hat{\mu}_N(s,a) = \frac{1}{N}\sum_{i=1}^N v(s'_i)
$$

where $s'_i \sim p(\cdot \mid s,a)$. The Monte Carlo estimate uses $\max_a \hat{\mu}_N(s,a)$. The noise-free target for the maximization step is $\max_a \mu(s,a)$. The gap

$$
\delta(s) = \mathbb{E}\big[\max_a \hat{\mu}_N(s,a)\big] - \max_a \mu(s,a) \geq 0
$$

is what we aim to learn. Once we have an estimator $\hat{\delta}(s)$, we can form a bias-corrected update:

$$
r(s,a^\star) + \gamma\Big(\max_a \hat{\mu}_N(s,a) - \hat{\delta}(s)\Big).
$$

The bias depends on the dispersion and cardinality of the action set. If one action dominates, $\delta(s)$ is small. If several actions have similar values, $\delta(s)$ grows because noise is more likely to change the maximizer. Useful features for predicting the bias include:

- Gaps to the best action: $g_a(s) = \max_{a'} \bar{\mu}(s,a') - \bar{\mu}(s,a)$, where $\bar{\mu}$ is a low-variance estimate of action values (from a fitted value model, large-$N$ simulation, or target network).
- Aggregates of the gap distribution: $\min_a g_a$ (zero by construction), $\text{mean}(g_a)$, $\text{std}(g_a)$, the top-2 gap.
- Number of actions: $\log|\mathcal{A}_s|$.
- Spread measures: $\log\sum_a \exp(\bar{\mu}(s,a)) - \max_a \bar{\mu}(s,a)$, which captures how "soft" the maximum is.

These features are cheap to compute and track regimes where maximization bias is large.

The procedure has two phases. First, we perform a high-cost simulation pass at a subset of states to estimate the bias. Second, we fit a regression model to predict the bias from features.

1. Choose a subset of states $\mathcal{S}^\star$ (space-filling design, or stratified sampling from a replay buffer).
2. For each $s \in \mathcal{S}^\star$:
   - Compute a high-precision reference $\mu^{\text{ref}}(s,a)$ using many draws (or exact integration in tractable models).
   - Form the deterministic max $M^{\text{ref}}(s) = \max_a \mu^{\text{ref}}(s,a)$.
   - Independently, compute a noisy max $\widehat{M}_N(s) = \max_a \hat{\mu}_N(s,a)$ using the same $N$ that will be used in routine value iteration.
   - Record the gap $y(s) = \widehat{M}_N(s) - M^{\text{ref}}(s)$.
   - Construct the feature vector $\phi(s)$ from the gaps and spreads of $\mu^{\text{ref}}(s,\cdot)$.
3. Fit a regression model $y \approx g_\eta(\phi)$ (polynomial, spline, or shallow network).
4. Freeze the fitted model $g_{\hat{\eta}}$. During routine value iteration, replace the raw noisy max with the bias-corrected expression that subtracts $g_{\hat{\eta}}(\phi(s))$.

The algorithm can be summarized as:

```{prf:algorithm} Keane-Wolpin Bias-Corrected Update
:label: keane-wolpin-bias-correction

**Input:** MDP, current value model $v$, sample size $N$, learned correction $g_\eta$

**Output:** Bias-corrected value estimate at state $s$

1. For each action $a \in \mathcal{A}_s$:
   - Draw $N$ next states $s'_1, \ldots, s'_N \sim p(\cdot \mid s,a)$
   - Compute $\hat{\mu}_N(s,a) = \frac{1}{N}\sum_{i=1}^N v(s'_i)$
2. Compute raw max: $\widehat{M}(s) = \max_a \hat{\mu}_N(s,a)$
3. Construct features $\phi(s)$ from a low-variance estimate of action values (target network or large-$N$ cache)
4. Return bias-corrected target: $T(s) = r(s,a^\star) + \gamma\big[\widehat{M}(s) - g_\eta(\phi(s))\big]$
```

Note that step 3 should use a separate, lower-variance estimate of action values to construct features, not the same noisy samples from step 1. This can be a target network, a large-$N$ precompute, or a smoothed critic. This avoids using the same randomness twice and stabilizes the feature computation.

**Properties.** If $g_{\hat{\eta}}(\phi)$ estimates $\mathbb{E}[\widehat{M}_N - M \mid \phi]$ accurately, subtracting it removes the systematic overestimation while leaving only zero-mean noise. The bias decreases, and because we replace many Monte Carlo draws with a deterministic function call, the variance also decreases. The method is flexible: the feature set can be tailored to the problem, and simple low-order polynomials or splines in the gap features work well in practice.

**Limitations.** The method requires computing sample variances or maintaining high-precision reference estimates, which increases memory and computational cost. If the regression model is misspecified, the correction can be ineffective or harmful. The bias function itself is estimated from data and thus subject to estimation error; if the bias estimate is noisy, subtracting it can increase variance. There is also a circular dependency: the bias depends on the current value function, which we are trying to estimate. Errors in $v$ affect the bias estimate, which in turn affects the next value estimate.

While influential in econometrics for dynamic discrete choice estimation, the Keane-Wolpin approach has seen limited use in reinforcement learning, partly due to computational overhead and partly because simpler alternatives (such as double Q-learning, which we discuss next) offer effective bias mitigation without the need for explicit bias modeling.

The Keane-Wolpin method represents an explicit parametric approach: treat bias estimation as a supervised learning problem and learn a function that predicts the systematic overestimation. This contrasts with the implicit approaches we encounter next, which mitigate bias by changing how value estimates are used rather than by explicitly modeling and subtracting the bias.
