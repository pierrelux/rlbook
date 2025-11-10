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

We begin by examining deterministic numerical integration methods—quadrature rules that approximate integrals by evaluating integrands at carefully chosen points with associated weights. We discuss how to coordinate the choice of quadrature with the choice of basis functions to balance approximation accuracy and computational cost. Then we turn to **Monte Carlo integration**, which approximates expectations using random samples rather than deterministic quadrature points. This shift from deterministic to stochastic integration is what brings us into machine learning territory. When we replace exact transition probabilities with samples drawn from simulations or real interactions, projection methods combined with Monte Carlo integration become what the operations research community calls **simulation-based approximate dynamic programming** and what the machine learning community calls **reinforcement learning**. By relying on samples rather than explicit probability functions, we move from model-based planning to data-driven learning.

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

is just a quadrature rule with random nodes and equal weights. More sophisticated Monte Carlo schemes, such as importance sampling, change both the sampling distribution and the weights, but the basic idea remains the same.

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

Note a subtle but important shift in perspective from the previous chapter. There, the weight function $w(s)$ in the inner product $\langle f, g \rangle_w = \int f(s) g(s) w(s) ds$ could be any positive weight function (not necessarily normalized). For Monte Carlo integration, however, we need $\mu$ to be a probability distribution that we can sample from. If the original problem was formulated with an unnormalized weight $w(s)$, we must either (i) normalize it to obtain $\mu(s) = w(s)/\int w(s') ds'$, which changes the problem statement, or (ii) use importance sampling with a different sampling distribution $\mu(s)$ and reweight samples by $w(s)/\mu(s)$. In reinforcement learning, we typically work under the assumption that $\mu$ is the empirical visitation distribution from collected trajectories, or (when the policy induces an ergodic Markov chain) the stationary distribution $\xi$ satisfying $\xi^\top P_\pi = \xi^\top$.

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

If we now substitute Monte Carlo estimates for both the inner transition expectations and the outer projection conditions, we obtain fully simulation-based schemes. We no longer need explicit access to the transition kernel $p(\cdot \mid s,a)$ or the state distribution $\mu$. It is enough to be able to **sample** from them, either through a simulator or by interacting with the real system. This is exactly the setting of reinforcement learning.

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


### Overestimation Bias in Monte Carlo Value Iteration

In statistics, bias refers to a systematic error where an estimator consistently deviates from the true parameter value. For an estimator $\hat{\theta}$ of a parameter $\theta$, we define bias as: $\text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta$. While bias is not always problematic (sometimes we deliberately introduce bias to reduce variance, as in ridge regression), uncontrolled bias can lead to significantly distorted results. In the context of value iteration, this distortion gets amplified even more due to the recursive nature of the algorithm.

Consider how the Bellman operator works in value iteration. At iteration n, we have a value function estimate $v_i(s)$ and aim to improve it by applying the Bellman operator $\Bellman$. The ideal update would be:

$$(\mathrm{{L}}v_i)(s) = \max_{a \in \mathcal{A}(s)} \left\{ r(s,a) + \gamma \int v_i(s') p(ds'|s,a) \right\}$$

However, we can't compute this integral exactly and use Monte Carlo integration instead, drawing $N$ next-state samples for each state and action pair. The bias emerges when we take the maximum over actions:

$$(\widehat{\Bellman}v_i)(s) = \max_{a \in \mathcal{A}(s)} \hat{q}_i(s,a), \enspace \text{where} \enspace \hat{q}_i(s,a) \equiv r(s,a) + \frac{\gamma}{N} \sum_{j=1}^N v_i(s'_j), \quad s'_j \sim p(\cdot|s,a)$$

White the Monte Carlo estimate $\hat{q}_n(s,a)$ is unbiased for any individual action, the empirical Bellman operator is biased upward due to Jensen's inequality, which states that for any convex function $f$, we have $\mathbb{E}[f(X)] \geq f(\mathbb{E}[X])$. Since the maximum operator is convex, this implies:

$$\mathbb{E}[(\widehat{\Bellman}v_i)(s)] = \mathbb{E}\left[\max_{a \in \mathcal{A}(s)} \hat{q}_i(s,a)\right] \geq \max_{a \in \mathcal{A}(s)} \mathbb{E}[\hat{q}_i(s,a)] = (\Bellman v_i)(s)$$

This means that our Monte Carlo approximation of the Bellman operator is biased upward:

$$b_i(s) = \mathbb{E}[(\widehat{\Bellman}v_i)(s)] - (\Bellman v_i)(s) \geq 0$$

Even worse, this bias compounds through iterations as each new value function estimate $v_{n+1}$ is based on targets generated by the biased operator $\widehat{\Bellman}$, creating a nested structure of bias accumulation.
This bias remains nonnegative at every step, and each application of the Bellman operator potentially adds more upward bias. As a result, instead of converging to the true value function $v^*$, the algorithm typically stabilizes at a biased approximation that systematically overestimates true values.

### The Keane-Wolpin Bias Correction Algorithm

Keane and Wolpin proposed to de-bias such estimators by essentially "learning" the bias, then subtracting it when computing the empirical Bellman operator. If we knew this bias function, we could subtract it from our empirical estimate to get an unbiased estimate of the true Bellman operator:

$$
(\widehat{\Bellman}v_n)(s) - \text{bias}(s) = (\widehat{\Bellman}v_n)(s) - (\mathbb{E}[(\widehat{\Bellman}v_n)(s)] - (\Bellman v_n)(s)) \approx (\Bellman v_n)(s)
$$

This equality holds in expectation, though any individual estimate would still have variance around the true value.

So how can we estimate the bias function? The Keane-Wolpin manages this using an important fact from extreme value theory: for normal random variables, the difference between the expected maximum and maximum of expectations scales with the standard deviation:

$$
\mathbb{E}\left[\max_{a \in \mathcal{A}} \hat{q}_i(s,a)\right] - \max_{a \in \mathcal{A}} \mathbb{E}[\hat{q}_i(s,a)] \approx c \cdot \sqrt{\max_{a \in \mathcal{A}} \text{Var}_i(s,a)}
$$

The variance term $\max_{a \in \mathcal{A}} \text{Var}_i(s,a)$ will typically be dominated by the action with the largest value -- the greedy action $a^*_i(s)$. Rather than deriving the constant $c$ theoretically, Keane-Wolpin proposed learning the relationship between variance and bias empirically through these steps:

1. Select a small set of "benchmark" states (typically 20-50) that span the state space
2. For these states, compute more accurate value estimates using many more Monte Carlo samples (10-100x more than usual)
3. Compute the empirical bias at each benchmark state $s$:

   $$\hat{b}_i(s) = (\hat{\Bellman}v_i)(s) - (\hat{\Bellman}_{\text{accurate}}v_i)(s)$$

4. Fit a linear relationship between this bias and the variance at the greedy action:

   $$\hat{b}_i(s) = \alpha_i \cdot \text{Var}_i(s,a^*_i(s)) + \epsilon$$

This creates a dataset of pairs $(\text{Var}_i(s,a^*_i(s)), \hat{b}_i(s))$ that can be used to estimate $\alpha_i$ through ordinary least squares regression. Once we have learned this bias function $\hat{b}$, we can define the bias-corrected Bellman operator:

$$ 
(\widetilde{\Bellman}v_i)(s) \triangleq (\hat{\Bellman}v_i)(s) - \hat{b}(s)
$$

While this bias correction approach has been influential in econometrics, it hasn't gained much traction in the machine learning community. A major drawback is the need for accurate operator estimation at benchmark states, which requires allocating substantially more samples to these states. In the next section, we'll explore an alternative strategy that, while requiring the maintenance of two sets of value estimates, achieves bias correction without demanding additional samples.

### Decoupling Selection and Evaluation

A simpler approach to addressing the upward bias is to maintain two separate q-function estimates - one for action selection and another for evaluation. We first examine the corresponding Monte Carlo value iteration algorithm, then establish why this works mathematically. Assume a Monte Carlo integration setup over Q factors: 


```{prf:algorithm} Double Monte Carlo Q-Value Iteration
:label: double-mc-value-iteration

**Input**: MDP $(S, A, P, R, \gamma)$, number of samples $N$, tolerance $\varepsilon > 0$, maximum iterations $K$
**Output**: Q-functions $q^A, q^B$

1. Initialize $q^A_0(s,a) = q^B_0(s,a) = 0$ for all $s \in S, a \in A$
2. $i \leftarrow 0$
3. repeat
    1. For each $s \in S, a \in A$:
        1. Draw $s'_j \sim p(\cdot|s,a)$ for $j = 1,\ldots,N$
        2. For network A:
            1. $a^*_i \leftarrow \arg\max_{a'} q^A_i(s'_j,a')$
            2. $q^A_{i+1}(s,a) \leftarrow r(s,a) + \frac{\gamma}{N} \sum_{j=1}^N q^B_i(s'_j,a^*_i)$
        3. For network B:
            1. $b^*_i \leftarrow \arg\max_{a'} q^B_i(s'_j,a')$
            2. $q^B_{i+1}(s,a) \leftarrow r(s,a) + \frac{\gamma}{N} \sum_{j=1}^N q^A_i(s'_j,b^*_i)$
    2. $\delta \leftarrow \max(\|q^A_{i+1} - q^A_i\|_{\infty}, \|q^B_{i+1} - q^B_i\|_{\infty})$
    3. $i \leftarrow i + 1$
4. until $\delta < \varepsilon$ or $i \geq K$
5. return final $q^A_{i+1}, q^B_{i+1}$
```
In this algorithm, we maintain two separate Q-functions ($q^A$ and $q^B$) and use them asymmetrically: when updating $q^A$, we use network A to select the best action ($a^*_i = \arg\max_{a'} q^A_i(s'_j,a')$) but then evaluate that action using network B's estimates ($q^B_i(s'_j,a^*_i)$). We do the opposite for updating $q^B$. You can see this separation in steps 3.2.2 and 3.2.3 of the algorithm, where for each network update, we first use one network to pick the action and then plug that chosen action into the other network for evaluation. We will see that this decomposition helps mitigate the positive bias that occurs due to Jensen's inequality. 

#### An HVAC analogy

Consider a building where each HVAC unit $i$ has some true maximum power draw $\mu_i$ under worst-case conditions. Suppose we lack access to manufacturer datasheets and must estimate these maxima from actual measurements. The challenge is that power draw fluctuates with environmental conditions. If we use a single day's measurements and look at the highest power draw, we systematically overestimate the true maximum draw across all units. 

To see this, let $X_A^i$ be unit i's power draw on day A and $X_B^i$ be unit i's power draw on day B. While both measurements are unbiased $\mathbb{E}[X_A^i] = \mathbb{E}[X_B^i] = \mu_i$, their maximum is not due to Jensen's inequality:

$$
\mathbb{E}[\max_i X_A^i] \geq \max_i \mathbb{E}[X_A^i] = \max_i \mu_i
$$

Intuitively, this problem occurs because reading tends to come from units that experienced particularly demanding conditions (e.g., direct sunlight, full occupancy, peak humidity) rather than just those with high true maximum draw. To estimate the true maximum power draw more accurately, we use the following measurement protocol:

1. Use day A measurements to **select** which unit hit the highest peak
2. Use day B measurements to **evaluate** that unit's power consumption

This yields the estimator:

$$
Y = X_B^{\arg\max_i X_A^i}
$$

We can show that by decoupling selection and evaluation in this fashion, our estimator $Y$ will no longer systematically overestimate the true maximum draw. First, observe that $\arg\max_i X_A^i$ is a random variable (call it $J$) - it tells us which unit had highest power draw on day A. It has some probability distribution based on day A's conditions:
   $P(J = j) = P(\arg\max_i X_A^i = j)$.
Using the law of total expectation:

$$
\begin{align*}
\mathbb{E}[Y] = \mathbb{E}[X_B^J] &= \mathbb{E}[\mathbb{E}[X_B^J \mid J]] \text{ (by tower property)} \\
&= \sum_{j=1}^n \mathbb{E}[X_B^j \mid J = j] P(J = j) \\
&= \sum_{j=1}^n \mathbb{E}[X_B^j \mid \arg\max_i X_A^i = j] P(\arg\max_i X_A^i = j)
\end{align*}
$$

Note that unit j's power draw on day B ($X_B^j$) is independent of whether it had the highest reading on day A ($\{\arg\max_i X_A^i = j\}$). An extreme cold event on day A should not affect day B's readings (especially in Quebec where the weather tends to vary widely from day to day). Therefore:

$$
\mathbb{E}[X_B^j \mid \arg\max_i X_A^i = j] = \mathbb{E}[X_B^j] = \mu_j
$$

This tells us that the two-day estimator is now an average of the true underlying power consumptions:

$$
\mathbb{E}[Y] = \sum_{j=1}^n \mu_j P(\arg\max_i X_A^i = j)
$$


To analyze $ \mathbb{E}[Y] $ more closely, we use a general result: if we have a real-valued function $ f $ defined on a discrete set of units $ \{1, \dots, n\} $ and a probability distribution $ q(\cdot) $ over these units, then the maximum value of $ f $ across all units is at least as large as the weighted sum of $ f $ values with weights $ q $. Formally,

$$
\max_{j \in \{1, \dots, n\}} f(j) \geq \sum_{j=1}^n q(j) f(j).
$$

Applying this to our setting, we set $ f(j) = \mu_j $ (the true maximum power draw for unit $ j $) and $ q(j) = P(J = j) $ (the probability that unit $ j $ achieves the maximum reading on day A). This gives us:

$$
\max_{j \in \{1, \dots, n\}} \mu_j \geq \sum_{j=1}^n P(J = j) \mu_j = \mathbb{E}[Y].
$$

Therefore, the expected value of $ Y $ (our estimator) will always be less than or equal to the true maximum value $ \max_j \mu_j $. In other words, $ Y $ provides a **conservative estimate** of the true maximum: it tends not to overestimate $ \max_j \mu_j $ but instead approximates it as closely as possible without systematic upward bias.

#### Consistency

Even though $ Y $ is not an unbiased estimator of $ \max_j \mu_j $ (since $ \mathbb{E}[Y] \leq \max_j \mu_j $), it is **consistent**. As more independent days (or measurements) are observed, the selection-evaluation procedure becomes more effective at isolating the intrinsic maximum, reducing the influence of day-specific environmental fluctuations. Over time, this approach yields a stable and increasingly accurate approximation of $ \max_j \mu_j $.

To show that $ Y $ is a consistent estimator of $ \max_i \mu_i $, we want to demonstrate that as the number of independent measurements (days, in this case) increases, $ Y $ converges in probability to $ \max_i \mu_i $. Suppose we have $ m $ independent days of measurements for each unit. Denote:
- $ X_A^{(k),i} $ as the power draw for unit $ i $ on day $ A_k $, where $ k \in \{1, \dots, m\} $.
- $ J_m = \arg\max_i \left( \frac{1}{m} \sum_{k=1}^m X_A^{(k),i} \right) $, which identifies the unit with the highest average power draw over $ m $ days.

The estimator we construct is:
$
Y_m = X_B^{(J_m)},
$
where $ X_B^{(J_m)} $ is the power draw of the selected unit $ J_m $ on an independent evaluation day $ B $.
We will now show that $ Y_m $ converges to $ \max_i \mu_i $ as $ m \to \infty $. This involves two main steps:

1. **Consistency of the Selection Step $ J_m $**: As $ m \to \infty $, the unit selected by $ J_m $ will tend to be the one with the true maximum power draw $ \max_i \mu_i $.
2. **Convergence of $ Y_m $ to $ \mu_{J_m} $**: Since the evaluation day $ B $ measurement $ X_B^{(J_m)} $ is unbiased with expectation $ \mu_{J_m} $, as $ m \to \infty $, $ Y_m $ will converge to $ \mu_{J_m} $, which in turn converges to $ \max_i \mu_i $.

The average power draw over $ m $ days for each unit $ i $ is:

$$
\frac{1}{m} \sum_{k=1}^m X_A^{(k),i}.
$$

By the law of large numbers, as $ m \to \infty $, this sample average converges to the true expected power draw $ \mu_i $ for each unit $ i $:

$$
\frac{1}{m} \sum_{k=1}^m X_A^{(k),i} \xrightarrow{m \to \infty} \mu_i.
$$

Since $ J_m $ selects the unit with the highest sample average, in the limit, $ J_m $ will almost surely select the unit with the highest true mean, $ \max_i \mu_i $. Thus, as $ m \to \infty $,
$$
\mu_{J_m} \to \max_i \mu_i.
$$

Given that $ J_m $ identifies the unit with the maximum true mean power draw in the limit, we now look at $ Y_m = X_B^{(J_m)} $, which is the power draw of unit $ J_m $ on the independent evaluation day $ B $.

Since $ X_B^{(J_m)} $ is an unbiased estimator of $ \mu_{J_m} $, we have:

$$
\mathbb{E}[Y_m \mid J_m] = \mu_{J_m}.
$$

As $ m \to \infty $, $ \mu_{J_m} $ converges to $ \max_i \mu_i $. Thus, $ Y_m $ will also converge in probability to $ \max_i \mu_i $ because $ Y_m $ is centered around $ \mu_{J_m} $ and $ J_m $ converges to the index of the unit with $ \max_i \mu_i $.


Combining these two steps, we conclude that:

$$
Y_m \xrightarrow{m \to \infty} \max_i \mu_i \text{ in probability}.
$$

This establishes the **consistency** of $ Y $ as an estimator for $ \max_i \mu_i $: as the number of independent measurements grows, $ Y_m $ converges to the true maximum power draw $ \max_i \mu_i $.

## Q-Factor Representation

As we discussed above, Monte Carlo integration is the method of choice when it comes to approximating the effect of the Bellman operator. This is due to both its computational advantages in higher dimensions and its compatibility with the model-free assumption. However, there is an additional important detail that we have neglected to properly cover: extracting actions from values in a model-free fashion. While we can obtain a value function using the Monte Carlo approach described above, we still face the challenge of extracting an optimal policy from this value function.

More precisely, recall that an optimal decision rule takes the form:

$$
d(s) = \arg\max_{a \in \mathcal{A}} \left\{r(s,a) + \gamma \int v(s')p(ds'|s,a)\right\}
$$

Therefore, even given an optimal value function $v$, deriving an optimal policy would still require Monte Carlo integration every time we query the decision rule/policy at a state.

Rather than approximating a state-value function, we can instead approximate a state-action value function. These two functions are related: the value function is the expectation of the Q-function (called Q-factors by some authors in the operations research literature) over the conditional distribution of actions given the current state:

$$
v(s) = \mathbb{E}[q(s,a)|s]
$$

If $q^*$ is an optimal state-action value function, then $v^*(s) = \max_a q^*(s,a)$. Just as we had a Bellman operator for value functions, we can also define an optimality operator for Q-functions. In component form:

$$
(\Bellman q)(s,a) = r(s,a) + \gamma \int p(ds'|s,a)\max_{a' \in \mathcal{A}(s')} q(s', a')
$$

Furthermore, this operator for Q-functions is also a contraction in the sup-norm and therefore has a unique fixed point $q^*$.

The advantage of iterating over Q-functions rather than value functions is that we can immediately extract optimal actions without having to represent the reward function or transition dynamics directly, nor perform numerical integration. Indeed, an optimal decision rule at state $s$ is obtained as:

$$
d(s) = \arg\max_{a \in \mathcal{A}(s)} q(s,a)
$$

With this insight, we can adapt our parametric value iteration algorithm to work with Q-functions:

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

## Initialization and Warmstarting

Parametric dynamic programming involves solving a sequence of related optimization problems, one for each fitting procedure at each iteration. While we've presented these as independent fitting problems, in practice we can leverage the relationship between successive iterations through careful initialization. This "warmstarting" strategy can significantly impact both computational efficiency and solution quality.

The basic idea is simple: rather than starting each fitting procedure from scratch, we initialize the function approximator with parameters from the previous iteration. This can speed up convergence since successive Q-functions tend to be similar. However, recent work suggests that persistent warmstarting might sometimes be detrimental, potentially leading to a form of overfitting. Alternative "reset" strategies that occasionally reinitialize parameters have shown promise in mitigating this issue.

Warmstarting can be incorporated into parametric Q-learning with one-step Monte Carlo integration as follows:

```{prf:algorithm} Warmstarted Parametric Q-Learning with N=1 Monte Carlo Integration
:label: warmstarted-q-learning

**Input** Given dataset $\mathcal{D}$ with transitions $(s, a, r, s')$, function approximator class $q(s,a; \boldsymbol{\theta})$, maximum iterations $N$, tolerance $\varepsilon > 0$, warmstart frequency $k$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D} \leftarrow \emptyset$
    2. For each $(s,a,r,s') \in \mathcal{D}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a'} q(s',a'; \boldsymbol{\theta}_n)$  // One-step Monte Carlo estimate
        2. $\mathcal{D} \leftarrow \mathcal{D} \cup \{((s,a), y_{s,a})\}$
    3. **if** $n \bmod k = 0$:  // Reset parameters periodically
        1. Initialize $\boldsymbol{\theta}_{temp}$ randomly
    4. **else**:
        1. $\boldsymbol{\theta}_{temp} \leftarrow \boldsymbol{\theta}_n$  // Warmstart from previous iteration
    5. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}, \boldsymbol{\theta}_{temp})$  // Initialize optimizer with $\boldsymbol{\theta}_{temp}$
    4. $\delta \leftarrow \frac{1}{|\mathcal{D}|}\sum_{(s,a) \in \mathcal{D}} (q(s,a; \boldsymbol{\theta}_{n+1}) - q(s,a; \boldsymbol{\theta}_n))^2$
    5. $n \leftarrow n + 1$
4. **until** ($\delta < \varepsilon$ or $n \geq N$)
5. **return** $\boldsymbol{\theta}_n$
```

The main addition here is the periodic reset of parameters (controlled by frequency $k$) which helps balance the benefits of warmstarting with the need to avoid potential overfitting. When $k=\infty$, we get traditional persistent warmstarting, while $k=1$ corresponds to training from scratch each iteration.

## Inner Loop Convergence

Beyond the choice of initialization and whether to chain optimization problems through warmstarting, we can also control how we terminate the inner optimization procedure. In the templates presented above, we implicitly assumed that $\texttt{fit}$ is run to convergence. However, this need not be the case, and different implementations handle this differently.

For example, scikit-learn's MLPRegressor terminates based on several criteria: when the improvement in loss falls below a tolerance (default `tol=1e-4`), when it reaches the maximum number of iterations (default `max_iter=200`), or when the loss fails to improve for `n_iter_no_change` consecutive epochs. In contrast, ExtraTreesRegressor builds trees deterministically to completion based on its splitting criteria, with termination controlled by parameters like `min_samples_split` and `max_depth`.

The intuition for using early stopping in the inner optimization mirrors that of modified policy iteration in exact dynamic programming. Just as modified policy iteration truncates the Neumann series during policy evaluation rather than solving to convergence, we might only partially optimize our function approximator at each iteration. While this complicates the theoretical analysis, it often works well in practice and can be computationally more efficient.

This perspective helps us understand modern deep reinforcement learning algorithms. For instance, DQN can be viewed as an instance of fitted Q-iteration where the inner optimization is intentionally limited. We can formalize this approach as follows:

```{prf:algorithm} Early-Stopping Fitted Q-Iteration
:label: early-stopping-fqi

**Input** Given dataset $\mathcal{D}$ with transitions $(s, a, r, s')$, function approximator $q(s,a; \boldsymbol{\theta})$, maximum outer iterations $N_{outer}$, maximum inner iterations $N_{inner}$, outer tolerance $\varepsilon_{outer}$, inner tolerance $\varepsilon_{inner}$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{P} \leftarrow \emptyset$
    2. For each $(s,a,r,s') \in \mathcal{D}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a'} q(s',a'; \boldsymbol{\theta}_n)$
        2. $\mathcal{P} \leftarrow \mathcal{P} \cup \{((s,a), y_{s,a})\}$
    3. // Inner optimization loop with early stopping
    4. $\boldsymbol{\theta}_{temp} \leftarrow \boldsymbol{\theta}_n$
    5. $k \leftarrow 0$
    6. **repeat**
        1. Update $\boldsymbol{\theta}_{temp}$ using one step of optimizer on $\mathcal{P}$
        2. Compute inner loop loss $\delta_{inner}$
        3. $k \leftarrow k + 1$
    7. **until** ($\delta_{inner} < \varepsilon_{inner}$ or $k \geq N_{inner}$)
    8. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_{temp}$
    9. $\delta_{outer} \leftarrow \frac{1}{|\mathcal{D}|}\sum_{(s,a) \in \mathcal{D}} (q(s,a; \boldsymbol{\theta}_{n+1}) - q(s,a; \boldsymbol{\theta}_n))^2$
    10. $n \leftarrow n + 1$
4. **until** ($\delta_{outer} < \varepsilon_{outer}$ or $n \geq N_{outer}$)
5. **return** $\boldsymbol{\theta}_n$
```
This formulation makes explicit the two-level optimization structure and allows us to control the trade-off between inner loop optimization accuracy and overall computational efficiency. When $N_{inner}=1$, we recover something closer to DQN's update rule, while larger values of $N_{inner}$ bring us closer to the full fitted Q-iteration approach.

# Example Methods

There are several moving parts we can swap in and out when working with parametric dynamic programming - from the function approximator we choose, to how we warm start things, to the specific methods we use for numerical integration and inner optimization. In this section, we'll look at some concrete examples and see how they fit into this general framework.

## Kernel-Based Reinforcement Learning (2002)

Ormoneit and Sen's Kernel-Based Reinforcement Learning (KBRL) {cite}`Ormoneit2002` helped establish the general paradigm of batch reinforcement learning later advocated by {cite}`ErnstGW05`. KBRL is a purely offline method that first collects a fixed set of transitions and then uses kernel regression to solve the optimal control problem through value iteration on this dataset. While the dominant approaches at the time were online methods like temporal difference, KBRL showed that another path to developping reinforcement learning algorithm was possible: one that capable of leveraging advances in supervised learning to provide both theoretical and practical benefits. 

As the name suggests, KBRL uses kernel based regression within the general framework of outlined above. 

```{prf:algorithm} Kernel-Based Q-Value Iteration
:label: kernel-based-q-iteration

**Input** Given an MDP $(S, A, P, R, \gamma)$, dataset $\mathcal{D}$ with observed transitions $(s, a, r, s')$, kernel bandwidth $b$, maximum iterations $N$, tolerance $\varepsilon > 0$

**Output** Kernel-based Q-function approximation

1. Initialize $\hat{Q}_0$ to zero everywhere
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D} \leftarrow \emptyset$
    2. For each $(s, a, r, s') \in \mathcal{D}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} \hat{Q}_n(s', a')$
        2. $\mathcal{D} \leftarrow \mathcal{D} \cup \{((s,a), y_{s,a})\}$
    3. $\hat{Q}_{n+1}(s,a) \leftarrow \sum_{(s_i,a_i,r_i,s_i') \in \mathcal{D}} k_b(s_i, s)\mathbb{1}[a_i=a] y_{s_i,a_i} / \sum_{(s_i,a_i,r_i,s_i') \in \mathcal{D}} k_b(s_i, s)\mathbb{1}[a_i=a]$
    4. $\delta \leftarrow \frac{1}{|\mathcal{D}|}\sum_{(s,a,r,s') \in \mathcal{D}} (\hat{Q}_{n+1}(s,a) - \hat{Q}_n(s,a))^2$
    5. $n \leftarrow n + 1$
4. **until** ($\delta < \varepsilon$ or $n \geq N$)
5. **return** $\hat{Q}_n$
```
Step 3 is where KBRL uses kernel regression with a normalized weighting kernel:

$$k_b(x^l_t, x) = \frac{\phi(\|x^l_t - x\|/b)}{\sum_{l'} \phi(\|x^l_{t'} - x\|/b)}$$

where $\phi$ is a kernel function (often Gaussian) and $b$ is the bandwidth parameter. Each iteration reuses the entire fixed dataset to re-estimate Q-values through this kernel regression.

An important theoretical contribution of KBRL is showing that this kernel-based approach ensures convergence of the Q-function sequence. The authors prove that, with appropriate choice of kernel bandwidth decreasing with sample size, the method is consistent - the estimated Q-function converges to the true Q-function as the number of samples grows.

The main practical limitation of KBRL is computational - being a batch method, it requires storing and using all transitions at each iteration, leading to quadratic complexity in the number of samples. The authors acknowledge this limitation for online settings, suggesting that modifications like discarding old samples or summarizing data clusters would be needed for online applications. Ernst's later work with tree-based methods would help address this limitation while maintaining many of the theoretical advantages of the batch approach.

## Ernst's Fitted Q Iteration (2005)

Ernst's {cite}`ErnstGW05` specific instantiation of parametric q-value iteration uses extremely randomized trees, an extension to random forests proposed by  {cite:t}`Geurts2006`. This algorithm became particularly well-known, partly because it was one of the first to demonstrate the advantages of offline reinforcement learning in practice on several challenging benchmarks at the time. 

Random Forests and Extra-Trees differ primarily in how they construct individual trees. Random Forests creates diversity in two ways: it resamples the training data (bootstrap) for each tree, and at each node it randomly selects a subset of features but then searches exhaustively for the best cut-point within each selected feature. In contrast, Extra-Trees uses the full training set for each tree and injects randomization differently: at each node, it randomly selects both features and cut-points without searching for the optimal one. It then picks the best among these completely random splits according to a variance reduction criterion. This double randomization - in both feature and cut-point selection - combined with using the full dataset makes Extra-Trees faster than Random Forests while maintaining similar predictive accuracy.

An important implementation detail concerns how tree structures can be reused across iterations of fitted Q iteration. With parametric methods like neural networks, warmstarting is straightforward - you simply initialize the weights with values from the previous iteration. For decision trees, the situation is more subtle because the model structure is determined by how splits are chosen at each node. When the number of candidate splits per node is $K=1$ (totally randomized trees), the algorithm selects both the splitting variable and threshold purely at random, without looking at the target values (the Q-values we're trying to predict) to evaluate the quality of the split. This means the tree structure only depends on the input variables and random choices, not on what we're predicting. As a result, we can build the trees once in the first iteration and reuse their structure throughout all iterations, only updating the prediction values at the leaves.

Standard Extra-Trees ($K>1$), however, uses target values to choose the best among K random splits by calculating which split best reduces the variance of the predictions. Since these target values change in each iteration of fitted Q iteration (as our estimate of Q evolves), we must rebuild the trees completely. While this is computationally more expensive, it allows the trees to better adapt their structure to capture the evolving Q-function.

The complete algorithm can be formalized as follows:

```{prf:algorithm} Extra-Trees Fitted Q Iteration
:label: extra-trees-fqi

**Input** Given an MDP $(S, A, P, R, \gamma)$, dataset $\mathcal{D}$ with observed transitions $(s, a, r, s')$, Extra-Trees parameters $(K, n_{min}, M)$, maximum iterations $N$, tolerance $\varepsilon > 0$

**Output** Extra-Trees model for Q-function approximation

1. Initialize $\hat{Q}_0$ to zero everywhere
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D} \leftarrow \emptyset$
    2. For each $(s, a, r, s') \in \mathcal{D}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} \hat{Q}_n(s', a')$
        2. $\mathcal{D} \leftarrow \mathcal{D} \cup \{((s,a), y_{s,a})\}$
    3. $\hat{Q}_{n+1} \leftarrow \text{BuildExtraTrees}(\mathcal{D}, K, n_{min}, M)$
    4. $\delta \leftarrow \frac{1}{|\mathcal{D}|}\sum_{(s,a,r,s') \in \mathcal{D}} (\hat{Q}_{n+1}(s,a) - \hat{Q}_n(s,a))^2$
    5. $n \leftarrow n + 1$
4. **until** ($\delta < \varepsilon$ or $n \geq N$)
5. **return** $\hat{Q}_n$
```
## Neural Fitted Q Iteration (2005)

Riedmiller's Neural Fitted Q Iteration (NFQI) {cite}`Riedmiller05` is a natural instantiation of parametric Q-value iteration where:

1. The function approximator $q(s,a; \boldsymbol{\theta})$ is a multi-layer perceptron
2. The $\texttt{fit}$ function uses Rprop optimization trained to convergence on each iteration's pattern set
3. The expected next-state values are estimated through Monte Carlo integration with $N=1$, using the observed next states from transitions

Specifically, rather than using numerical quadrature which would require known transition probabilities, NFQ approximates the expected future value using observed transitions:

$$
\int q_n(s',a')p(ds'|s,a) \approx q_n(s'_{observed},a')
$$

where $s'_{observed}$ is the actual next state that was observed after taking action $a$ in state $s$. This is equivalent to Monte Carlo integration with a single sample, making the algorithm fully model-free.

The algorithm follows from the parametric Q-value iteration template:

```{prf:algorithm} Neural Fitted Q Iteration
:label: neural-fitted-q-iteration

**Input** Given an MDP $(S, A, P, R, \gamma)$, dataset $\mathcal{D}$ with observed transitions $(s, a, r, s')$, MLP architecture $q(s,a; \boldsymbol{\theta})$, maximum iterations $N$, tolerance $\varepsilon > 0$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D} \leftarrow \emptyset$
    2. For each $(s,a,r,s') \in \mathcal{D}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a'} q(s',a'; \boldsymbol{\theta}_n)$  // Monte Carlo estimate with one sample
        2. $\mathcal{D} \leftarrow \mathcal{D} \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{n+1} \leftarrow \text{Rprop}(\mathcal{D})$ // Train MLP to convergence
    4. $\delta \leftarrow \frac{1}{|\mathcal{D}||A|}\sum_{(s,a) \in \mathcal{D} \times A} (q(s,a; \boldsymbol{\theta}_{n+1}) - q(s,a; \boldsymbol{\theta}_n))^2$
    5. $n \leftarrow n + 1$
4. **until** ($\delta < \varepsilon$ or $n \geq N$)
5. **return** $\boldsymbol{\theta}_n$
```

While NFQI was originally introduced as an offline method with base points collected a priori, the authors also present a variant where base points are collected incrementally. In this online variant, new transitions are gathered using the current policy (greedy with respect to $Q_k$) and added to the experience set. This approach proves particularly useful when random exploration cannot efficiently collect representative experiences.

## Deep Q Networks (2013)

DQN {cite}`mnih2013atari` is a close relative of NFQI - in fact, Riedmiller, the author of NFQI, was also an author on the DQN paper. What at first glance might look like a different algorithm can actually be understood as a special case of parametric dynamic programming with practical adaptations. We build this connection step by step.

First, we start with basic parametric Q-value iteration using a neural network:

```{prf:algorithm} Basic Offline Neural Fitted Q-Value Iteration
:label: basic-q-iteration

**Input** Given an MDP $(S, A, P, R, \gamma)$, dataset of transitions $\mathcal{T}$, neural network $q(s,a; \boldsymbol{\theta})$, maximum iterations $N$, tolerance $\varepsilon > 0$, initialization $\boldsymbol{\theta}_0$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D}_n \leftarrow \emptyset$  // Regression dataset
    2. For each $(s,a,r,s') \in \mathcal{T}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_n)$
        2. $\mathcal{D}_n \leftarrow \mathcal{D}_n \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}_n, \boldsymbol{\theta}_0)$ // Fit neural network using built-in convergence criterion 
    4. $n \leftarrow n + 1$
4. **until** training complete
5. **return** $\boldsymbol{\theta}_n$
```
Next, we open up the `fit` procedure to show the inner optimization loop using gradient descent:

```{prf:algorithm} Fitted Q-Value Iteration with Explicit Inner Loop
:label: q-iteration-inner-loop

**Input** Given MDP $(S, A, P, R, \gamma)$, dataset of transitions $\mathcal{T}$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, convergence test $\texttt{has\_converged}(\cdot)$, initialization $\boldsymbol{\theta}_0$, regression loss function $\mathcal{L}$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $n \leftarrow 0$  // Outer iteration index
3. **repeat**
    1. $\mathcal{D}_n \leftarrow \emptyset$  // Regression dataset
    2. For each $(s,a,r,s') \in \mathcal{T}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_n)$
        2. $\mathcal{D}_n \leftarrow \mathcal{D}_n \cup \{((s,a), y_{s,a})\}$
    3. // Inner optimization loop
    4. $\boldsymbol{\theta}^{(0)} \leftarrow \boldsymbol{\theta}_0$  // Start from initial parameters
    5. $k \leftarrow 0$  // Inner iteration index
    6. **repeat**
        1. $\boldsymbol{\theta}^{(k+1)} \leftarrow \boldsymbol{\theta}^{(k)} - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}^{(k)}; \mathcal{D}_n)$
        2. $k \leftarrow k + 1$
    7. **until** $\texttt{has\_converged}(\boldsymbol{\theta}^{(0)}, ..., \boldsymbol{\theta}^{(k)}, \mathcal{D}_n)$
    8. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}^{(k)}$
    9. $n \leftarrow n + 1$
4. **until** training complete
5. **return** $\boldsymbol{\theta}_n$
```

### Warmstarting and Partial Fitting

A natural modification is to initialize the inner optimization loop with the previous iteration's parameters - a strategy known as warmstarting - rather than starting from $\boldsymbol{\theta}_0$ each time. Additionally, similar to how modified policy iteration performs partial policy evaluation rather than solving to convergence, we can limit ourselves to a fixed number of optimization steps. These pragmatic changes, when combined, yield:

```{prf:algorithm} Neural Fitted Q-Iteration with Warmstarting and Partial Optimization
:label: nfqi-warmstart-partial

**Input** Given MDP $(S, A, P, R, \gamma)$, dataset of transitions $\mathcal{T}$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, number of steps $K$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $n \leftarrow 0$  // Outer iteration index
3. **repeat**
    1. $\mathcal{D}_n \leftarrow \emptyset$  // Regression dataset
    2. For each $(s,a,r,s') \in \mathcal{T}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_n)$
        2. $\mathcal{D}_n \leftarrow \mathcal{D}_n \cup \{((s,a), y_{s,a})\}$
    3. // Inner optimization loop with warmstart and fixed steps
    4. $\boldsymbol{\theta}^{(0)} \leftarrow \boldsymbol{\theta}_n$  // Warmstart from previous iteration
    5. For $k = 0$ to $K-1$:
        1. $\boldsymbol{\theta}^{(k+1)} \leftarrow \boldsymbol{\theta}^{(k)} - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}^{(k)}; \mathcal{D}_n)$
    6. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}^{(K)}$
    7. $n \leftarrow n + 1$
4. **until** training complete
5. **return** $\boldsymbol{\theta}_n$
```

### Flattening the Updates with Target Swapping

Now rather than maintaining two sets of indices for the outer and inner levels, we could also "flatten" this algorithm under a single loop structure using modulo arithmetic. We can rewrite it as follows:

```{prf:algorithm} Flattened Neural Fitted Q-Iteration
:label: nfqi-flattened-swap

**Input** Given MDP $(S, A, P, R, \gamma)$, dataset of transitions $\mathcal{T}$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, target update frequency $K$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_0$  // Initialize target parameters
3. $t \leftarrow 0$  // Single iteration counter
4. **while** training:
    1. $\mathcal{D}_t \leftarrow \emptyset$  // Regression dataset
    2. For each $(s,a,r,s') \in \mathcal{T}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_{target})$  // Use target parameters
        2. $\mathcal{D}_t \leftarrow \mathcal{D}_t \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t; \mathcal{D}_t)$
    4. If $t \bmod K = 0$:  // Every K steps
        1. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_t$  // Update target parameters
    5. $t \leftarrow t + 1$
4. **return** $\boldsymbol{\theta}_t$
```

The flattened version with target parameters achieves exactly the same effect as our previous nested-loop structure with warmstarting and K gradient steps. In the nested version, we would create a dataset using parameters $\boldsymbol{\theta}_n$, then perform K gradient steps to obtain $\boldsymbol{\theta}_{n+1}$. In our flattened version, we maintain a separate $\boldsymbol{\theta}_{target}$ that gets updated every K steps, ensuring that the dataset $\mathcal{D}_n$ is created using the same parameters for K consecutive iterations - just as it would be in the nested version. The only difference is that we've restructured the algorithm to avoid explicitly nesting the loops, making it more suitable for continuous online training which we are about to introduce. The periodic synchronization of $\boldsymbol{\theta}_{target}$ with the current parameters $\boldsymbol{\theta}_n$ effectively marks the boundary of what would have been the outer loop in our previous version.

### Exponential Moving Average Targets

An alternative to this periodic swap of parameters is to use an exponential moving average (EMA) of the parameters:

```{prf:algorithm} Flattened Neural Fitted Q-Iteration with EMA
:label: nfqi-flattened-ema

**Input** Given MDP $(S, A, P, R, \gamma)$, dataset of transitions $\mathcal{T}$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, EMA rate $\tau$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_0$  // Initialize target parameters
3. $n \leftarrow 0$  // Single iteration counter
4. **while** training:
    1. $\mathcal{D}_n \leftarrow \emptyset$  // Regression dataset
    2. For each $(s,a,r,s') \in \mathcal{T}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_{target})$  // Use target parameters
        2. $\mathcal{D}_n \leftarrow \mathcal{D}_n \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_n - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_n; \mathcal{D}_n)$
    4. $\boldsymbol{\theta}_{target} \leftarrow \tau\boldsymbol{\theta}_{n+1} + (1-\tau)\boldsymbol{\theta}_{target}$  // Smooth update of target parameters
    5. $n \leftarrow n + 1$
4. **return** $\boldsymbol{\theta}_n$
```

Note that the original DQN used the periodic swap of parameters rather than EMA targets. 
EMA targets (also called "Polyak averaging") started becoming popular in deep RL with DDPG {cite}`lillicrap2015continuous` where they used a "soft" target update: $\boldsymbol{\theta}_{target} \leftarrow \tau\boldsymbol{\theta} + (1-\tau)\boldsymbol{\theta}_{target}$ with a small $\tau$ (like 0.001). This has since become a common choice in many algorithms like TD3 {cite}`fujimoto2018addressing` and SAC {cite}`haarnoja2018soft`.

### Online Data Collection and Experience Replay

Rather than using offline data, we now consider a modification where we incrementally gather samples under our current policy. A common exploration strategy is $\varepsilon$-greedy: with probability $\varepsilon$ we select a random action, and with probability $1-\varepsilon$ we select the greedy action $\arg\max_a q(s,a;\boldsymbol{\theta}_n)$. This ensures we maintain some exploration even as our Q-function estimates improve. Typically $\varepsilon$ is annealed over time, starting with a high value (e.g., 1.0) to encourage early exploration and gradually decreasing to a small final value (e.g., 0.01) to maintain a minimal level of exploration while mostly exploiting our learned policy.

```{prf:algorithm} Flattened Online Neural Fitted Q-Iteration
:label: online-nfqi-flattened

**Input** Given MDP $(S, A, P, R, \gamma)$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, target update frequency $K$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_0$  // Initialize target parameters
3. Initialize $\mathcal{T} \leftarrow \emptyset$  // Initialize transition dataset
4. $n \leftarrow 0$  // Single iteration counter
5. **while** training:
    1. Observe current state $s$
    2. Select action $a$ using policy derived from $q(s,\cdot;\boldsymbol{\theta}_n)$ (e.g., ε-greedy)
    3. Execute $a$, observe reward $r$ and next state $s'$
    4. $\mathcal{T}_n \leftarrow \mathcal{T}_n \cup \{(s,a,r,s')\}$  // Add transition to dataset
    5. $\mathcal{D}_n \leftarrow \emptyset$  // Regression dataset
    6. For each $(s,a,r,s') \in \mathcal{T}_n$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_{target})$  // Use target parameters
        2. $\mathcal{D}_n \leftarrow \mathcal{D}_n \cup \{((s,a), y_{s,a})\}$
    7. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_n - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_n; \mathcal{D}_n)$
    8. If $n \bmod K = 0$:  // Every K steps
        1. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_n$  // Update target parameters
    9. $n \leftarrow n + 1$
6. **return** $\boldsymbol{\theta}_n$
```

This version faces two practical challenges. First, the transition dataset $\mathcal{T}_n$ grows unbounded over time, creating memory issues. Second, computing gradients over the entire dataset becomes increasingly expensive. These are common challenges in online learning settings, and the standard solutions from supervised learning apply here:
1. Use a fixed-size circular buffer (often called replay buffer, in reference to "experience replay" by {cite}`lin1992self`) to limit memory usage
2. Compute gradients on mini-batches rather than the full dataset

We can modify our algorithm to incorporate these ideas as follows:

```{prf:algorithm} Deep-Q Network 
:label: online-nfqi-replay

**Input** Given MDP $(S, A, P, R, \gamma)$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, target update frequency $K$, replay buffer size $B$, mini-batch size $b$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_0$  // Initialize target parameters
3. Initialize replay buffer $\mathcal{R}$ with capacity $B$
4. $n \leftarrow 0$  // Single iteration counter
5. **while** training:
    1. Observe current state $s$
    2. Select action $a$ using policy derived from $q(s,\cdot;\boldsymbol{\theta}_n)$ (e.g., ε-greedy)
    3. Execute $a$, observe reward $r$ and next state $s'$
    4. Store $(s,a,r,s')$ in $\mathcal{R}$, replacing oldest if full  // Circular buffer update
    5. $\mathcal{D}_n \leftarrow \emptyset$  // Regression dataset
    6. Sample mini-batch of $b$ transitions $(s_i,a_i,r_i,s_i')$ from $\mathcal{R}$
    7. For each sampled $(s_i,a_i,r_i,s_i')$:
        1. $y_i \leftarrow r_i + \gamma \max_{a' \in A} q(s_i',a'; \boldsymbol{\theta}_{target})$  // Use target parameters
        2. $\mathcal{D}_n \leftarrow \mathcal{D}_n \cup \{((s_i,a_i), y_i)\}$
    8. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_n - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_n; \mathcal{D}_n)$ // Replace by RMSProp to obtain DQN
    9. If $n \bmod K = 0$:  // Every K steps
        1. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_n$  // Update target parameters
    10. $n \leftarrow n + 1$
6. **return** $\boldsymbol{\theta}_n$
```

This formulation reveals the replay ratio (or data reuse ratio) in deep reinforcement learning. In our algorithm, for each new transition we collect, we sample a mini-batch of size b from our replay buffer and perform one update. This means we're reusing past experiences at a ratio of b:1 - for every new piece of data, we're learning from b experiences. This ratio can be tuned as a hyperparameter. Higher ratios mean more computation per environment step but better data efficiency, as we're extracting more learning from each collected transition. Experience replay allows us to decouple the rate of data collection from the rate of learning updates. Some modern algorithms like SAC or TD3 explicitly tune this ratio, sometimes using multiple gradient steps per environment step to achieve higher data efficiency.

### Double-Q Network Variant

As we saw earlier, the max operator in the target computation can lead to overestimation of Q-values. This happens because we use the same network to both select and evaluate actions in the target computation: $y_i \leftarrow r_i + \gamma \max_{a' \in A} q(s_i',a'; \boldsymbol{\theta}_{target})$. The max operator means we're both choosing the action that looks best under our current estimates and then using that same set of estimates to evaluate how good that action is, potentially compounding any optimization bias.

Double DQN {cite:t}`van2016deep` addresses this by using the current network parameters to select actions but the target network parameters to evaluate them. This leads to a simple modification of the DQN algorithm:

```{prf:algorithm} Double Deep-Q Network
:label: double-dqn

**Input** Given MDP $(S, A, P, R, \gamma)$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, target update frequency $K$, replay buffer size $B$, mini-batch size $b$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_0$  // Initialize target parameters
3. Initialize replay buffer $\mathcal{R}$ with capacity $B$
4. $n \leftarrow 0$  // Single iteration counter
5. **while** training:
    1. Observe current state $s$
    2. Select action $a$ using policy derived from $q(s,\cdot;\boldsymbol{\theta}_n)$ (e.g., ε-greedy)
    3. Execute $a$, observe reward $r$ and next state $s'$
    4. Store $(s,a,r,s')$ in $\mathcal{R}$, replacing oldest if full
    5. $\mathcal{D}_n \leftarrow \emptyset$  // Regression dataset
    6. Sample mini-batch of $b$ transitions $(s_i,a_i,r_i,s_i')$ from $\mathcal{R}$
    7. For each sampled $(s_i,a_i,r_i,s_i')$:
        1. $a^*_i \leftarrow \arg\max_{a' \in A} q(s_i',a'; \boldsymbol{\theta}_n)$  // Select action using current network
        2. $y_i \leftarrow r_i + \gamma q(s_i',a^*_i; \boldsymbol{\theta}_{target})$  // Evaluate using target network
        3. $\mathcal{D}_n \leftarrow \mathcal{D}_n \cup \{((s_i,a_i), y_i)\}$
    8. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_n - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_n; \mathcal{D}_n)$
    9. If $n \bmod K = 0$:  // Every K steps
        1. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_n$  // Update target parameters
    10. $n \leftarrow n + 1$
6. **return** $\boldsymbol{\theta}_n$
```

The main difference from the original DQN is in step 7, where we now separate action selection from action evaluation. Rather than directly taking the max over the target network's Q-values, we first select the action using our current network ($\boldsymbol{\theta}_n$) and then evaluate that specific action using the target network ($\boldsymbol{\theta}_{target}$). This simple change has been shown to lead to more stable learning and better final performance across a range of tasks.

## Deep Q Networks with Resets (2022)

In flattening neural fitted Q-iteration, our field had lost sight of an important structural element: the choice of inner-loop initializer inherent in the original FQI algorithm. The traditional structure explicitly separated outer iterations (computing targets) from inner optimization (fitting to those targets), with each inner optimization starting fresh from parameters $\boldsymbol{\theta}_0$. 

The flattened version with persistent warmstarting seemed like a natural optimization - why throw away learned parameters? However, recent work {cite}`Doro2023` has shown that persistent warmstarting can actually be detrimental to learning. Neural networks tend to lose their ability to learn and generalize over the course of training, suggesting that occasionally starting fresh from $\boldsymbol{\theta}_0$ can be beneficial. In the context of DQN, this looks algorithmically as follows:

```{prf:algorithm} DQN with Hard Resets
:label: dqn-hard-resets

**Input** Given MDP $(S, A, P, R, \gamma)$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, reset interval $K$, replay buffer size $B$, mini-batch size $b$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_0$  // Initialize target parameters
3. Initialize replay buffer $\mathcal{R}$ with capacity $B$
4. $n \leftarrow 0$  // Single iteration counter
5. **while** training:
    1. Observe current state $s$
    2. Select action $a$ using policy derived from $q(s,\cdot;\boldsymbol{\theta}_n)$ (e.g., ε-greedy)
    3. Execute $a$, observe reward $r$ and next state $s'$
    4. Store $(s,a,r,s')$ in $\mathcal{R}$, replacing oldest if full
    5. $\mathcal{D}_n \leftarrow \emptyset$  // Regression dataset
    6. Sample mini-batch of $b$ transitions $(s_i,a_i,r_i,s_i')$ from $\mathcal{R}$
    7. For each sampled $(s_i,a_i,r_i,s_i')$:
        1. $y_i \leftarrow r_i + \gamma \max_{a' \in A} q(s_i',a'; \boldsymbol{\theta}_{target})$
        2. $\mathcal{D}_n \leftarrow \mathcal{D}_n \cup \{((s_i,a_i), y_i)\}$
    8. If $n \bmod K = 0$:  // Periodic reset
        1. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_0$  // Reset to initial parameters
    9. Else:
        1. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_n - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_n; \mathcal{D}_n)$
    10. If $n \bmod K = 0$:  // Every K steps
        1. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_n$  // Update target parameters
    11. $n \leftarrow n + 1$
6. **return** $\boldsymbol{\theta}_n$
```

This algorithm change allows us to push the limits of our update ratio - the number of gradient steps we perform per environment interaction. Without resets, increasing this ratio leads to diminishing returns as the network's ability to learn degrades. However, by periodically resetting the parameters while maintaining our dataset of transitions, we can perform many more updates per interaction, effectively making our algorithm more "offline" and thus more sample efficient. 

The hard reset strategy, while effective, can be too aggressive in some settings as it completely discards learned parameters. An alternative approach is to use a softer form of reset, adapting the "Shrink and Perturb" technique originally introduced by {cite:t}`ash2020warm` in the context of continual learning. In their work, they found that neural networks that had been trained on one task could better adapt to new tasks if their parameters were partially reset - interpolated with a fresh initialization - rather than either kept intact or completely reset.

We can adapt this idea to our setting. Instead of completely resetting to $\boldsymbol{\theta}_0$, we can perform a soft reset by interpolating between our current parameters and a fresh random initialization:

```{prf:algorithm} DQN with Shrink and Perturb
:label: dqn-soft-resets

**Input** Given MDP $(S, A, P, R, \gamma)$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, reset interval $K$, replay buffer size $B$, mini-batch size $b$, interpolation coefficient $\beta$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_0$  // Initialize target parameters
3. Initialize replay buffer $\mathcal{R}$ with capacity $B$
4. $n \leftarrow 0$  // Single iteration counter
5. **while** training:
    1. Observe current state $s$
    2. Select action $a$ using policy derived from $q(s,\cdot;\boldsymbol{\theta}_n)$ (e.g., ε-greedy)
    3. Execute $a$, observe reward $r$ and next state $s'$
    4. Store $(s,a,r,s')$ in $\mathcal{R}$, replacing oldest if full
    5. $\mathcal{D}_n \leftarrow \emptyset$  // Regression dataset
    6. Sample mini-batch of $b$ transitions $(s_i,a_i,r_i,s_i')$ from $\mathcal{R}$
    7. For each sampled $(s_i,a_i,r_i,s_i')$:
        1. $y_i \leftarrow r_i + \gamma \max_{a' \in A} q(s_i',a'; \boldsymbol{\theta}_{target})$
        2. $\mathcal{D}_n \leftarrow \mathcal{D}_n \cup \{((s_i,a_i), y_i)\}$
    8. If $n \bmod K = 0$:  // Periodic soft reset
        1. Sample $\boldsymbol{\phi} \sim$ initializer  // Fresh random parameters
        2. $\boldsymbol{\theta}_{n+1} \leftarrow \beta\boldsymbol{\theta}_n + (1-\beta)\boldsymbol{\phi}$
    9. Else:
        1. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_n - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_n; \mathcal{D}_n)$
    10. If $n \bmod K = 0$:  // Every K steps
        1. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_n$  // Update target parameters
    11. $n \leftarrow n + 1$
6. **return** $\boldsymbol{\theta}_n$
```

The interpolation coefficient $\beta$ controls how much of the learned parameters we retain, with $\beta = 0$ recovering the hard reset case and $\beta = 1$ corresponding to no reset at all. This provides a more flexible approach to restoring learning capability while potentially preserving useful features that have been learned. Like hard resets, this softer variant still enables high update ratios by preventing the degradation of learning capability, but does so in a more gradual way.
