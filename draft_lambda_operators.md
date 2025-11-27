# Multi-Step Bellman Operators and Weighted Residual Methods

## Motivation: Beyond One-Step Lookahead

The classical Bellman optimality operator uses one-step lookahead:

$$
[\mathcal{T} v](s) = \max_{a} \left\{ r(s,a) + \gamma \mathbb{E}[v(s') \mid s,a] \right\}.
$$

This operator is a $\gamma$-contraction in the sup norm, ensuring that value iteration $v_{k+1} = \mathcal{T} v_k$ converges. But when we work with function approximation via projection methods, we iterate the **composed operator** $\Pi \mathcal{T}$, which may not be a contraction. As we saw in the previous sections, whether $\Pi \mathcal{T}$ contracts depends on properties of the projection operator $\Pi$—monotonicity for sup-norm analysis, or matching the weighting to the transition operator for weighted $L^2$ analysis.

This raises a natural question: might we obtain better convergence properties by modifying the Bellman operator itself? Rather than one-step lookahead, consider **multi-step** operators:

$$
[\mathcal{T}^{(n)} v](s) = \max_{a_0} \mathbb{E}\left[ \sum_{t=0}^{n-1} \gamma^t r_t + \gamma^n v(s_n) \,\Big|\, s_0 = s, a_0 \right],
$$

where the expectation is over trajectories following optimal actions at states $s_1, \ldots, s_{n-1}$. This is the $n$-step Bellman optimality operator: it backs up rewards from $n$ steps into the future, applying the max operator at each intermediate state.

For policy evaluation with a fixed policy $\pi$, the multi-step operator is simpler:

$$
[\mathcal{T}_\pi^{(n)} v](s) = \mathbb{E}_\pi\left[ \sum_{t=0}^{n-1} \gamma^t r_t + \gamma^n v(s_n) \,\Big|\, s_0 = s \right].
$$

These multi-step operators define different fixed-point equations. Just as $v^* = \mathcal{T} v^*$, we have $v^* = \mathcal{T}^{(n)} v^*$ for any $n$. But when we apply weighted residual methods with projection $\Pi$, the approximations $\hat{v} = \Pi \mathcal{T}^{(n)} \hat{v}$ may differ for different $n$. The bias-variance tradeoff appears: larger $n$ gives lower bias (more accurate returns) but higher variance (longer trajectories). More fundamentally, the **spectral properties** of $\Pi \mathcal{T}^{(n)}$ change with $n$, potentially affecting whether the operator contracts.

## Compound Operators via Geometric Averaging

Rather than committing to a single horizon $n$, we can form a **weighted average** over all horizons. Consider the $\lambda$-weighted compound operator:

$$
\mathcal{T}^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \mathcal{T}^{(n)}.
$$

The weight $(1-\lambda)\lambda^{n-1}$ is a geometric distribution with parameter $\lambda \in [0,1)$. This averages over all possible backup depths, with exponentially decaying weights. When $\lambda = 0$, we recover $\mathcal{T}^\lambda = \mathcal{T}$, the one-step operator. As $\lambda \to 1$, we give increasing weight to longer backups.

**Why geometric weights?** Several justifications:

1. **Maximum entropy**: Among all distributions on $\{1, 2, 3, \ldots\}$ with a given mean, the geometric distribution has maximum entropy. This makes it the "least informative" choice given a desired average depth.

2. **Memoryless property**: The geometric distribution is memoryless: $P(N > n+k \mid N > n) = P(N > k)$. This time-consistency simplifies analysis.

3. **Resolvent interpretation**: The series $(1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} \mathcal{T}^{(n)}$ resembles a Neumann series or resolvent expansion from functional analysis.

4. **Computational tractability**: As we will see, geometric averaging enables recursive computation via auxiliary state variables (eligibility traces), avoiding explicit summation.

### Connection to Random Stopping Times

An equivalent perspective interprets $\mathcal{T}^\lambda$ via random stopping. Define a geometric stopping time $\tau \sim \text{Geom}(1-\lambda)$: at each time $t$, we stop with probability $1-\lambda$, independently. Then:

$$
[\mathcal{T}^\lambda v](s) = \mathbb{E}_\tau\left[ [\mathcal{T}^{(\tau)} v](s) \right].
$$

The operator averages over random-depth backups. This viewpoint appears in {cite}`Watkins1989` and {cite}`DayanSejnowski1994`, motivated by biological considerations of memory traces in reinforcement learning. While the historical motivation was psychological, the mathematical structure stands on its own: we are simply considering a family of Bellman-like operators parameterized by $\lambda$.

### For Bellman Optimality: Tracking Greedy Actions

For the Bellman optimality operator, the multi-step backup must account for the max at each intermediate state:

$$
[\mathcal{T}^{(n)} v](s) = \max_{a_0} \mathbb{E}\left[ \sum_{t=0}^{n-1} \gamma^t r(s_t, a_t) + \gamma^n v(s_n) \,\Big|\, s_0 = s, a_0, a_t = \arg\max_{a'} [r(s_t,a') + \gamma \mathbb{E}[v(s_{t+1}) \mid s_t, a']] \right].
$$

Each $a_t$ for $t \geq 1$ is the greedy action at state $s_t$ with respect to the current value function $v$. The compound operator $\mathcal{T}^\lambda$ for optimality thus involves averaging over trajectories where actions are chosen greedily at each intermediate step:

$$
[\mathcal{T}^\lambda v](s) = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \max_{a_0} \mathbb{E}_{\text{greedy}}\left[ \sum_{t=0}^{n-1} \gamma^t r_t + \gamma^n v(s_n) \,\Big|\, s_0=s, a_0 \right].
$$

**Key observation**: The max appears only at the initial action $a_0$. Subsequent actions are determined by the current value estimate $v$. This asymmetry—optimizing only the first action while following a greedy policy thereafter—is characteristic of **one-step lookahead with multi-step evaluation**. It is the natural extension of the Bellman optimality principle to multi-step backups.

## Weighted Residual Methods with $\mathcal{T}^\lambda$: The Two-Level Structure

Just as with the standard Bellman operator $\mathcal{T}$, applying weighted residual methods to $\mathcal{T}^\lambda$ follows a two-level structure:

**Outer level (infinite-dimensional)**: We seek a solution to the functional equation
$$
v = \mathcal{T}^\lambda v.
$$
Since we cannot represent $v$ exactly on a computer, we work with a finite-dimensional approximation.

**Inner level (finite-dimensional)**: We parameterize $v$ using basis functions $\{\varphi_1, \ldots, \varphi_n\}$ as $v_\theta(s) = \sum_{i=1}^n \theta_i \varphi_i(s)$, and seek coefficients $\theta \in \mathbb{R}^n$ such that the **residual** 
$$
R^\lambda(s; \theta) = [\mathcal{T}^\lambda v_\theta](s) - v_\theta(s)
$$
is "small" in a sense made precise by our choice of weighted residual method.

### From Functional Equation to Finite-Dimensional System

For different choices of test functions, we obtain different finite-dimensional conditions:

**Collocation**: Choose $n$ states $\{s_1, \ldots, s_n\}$ and require $R^\lambda(s_i; \theta) = 0$ for all $i$:
$$
\sum_{j=1}^n \theta_j \varphi_j(s_i) = [\mathcal{T}^\lambda v_\theta](s_i), \quad i = 1, \ldots, n.
$$

**Galerkin**: Require orthogonality $\langle R^\lambda(\cdot; \theta), \varphi_i \rangle_w = 0$ for all basis functions:
$$
\int_{\mathcal{S}} \left( v_\theta(s) - [\mathcal{T}^\lambda v_\theta](s) \right) \varphi_i(s) w(s) ds = 0, \quad i = 1, \ldots, n.
$$

**Least squares**: Minimize the weighted squared residual:
$$
\min_\theta \int_{\mathcal{S}} [R^\lambda(s; \theta)]^2 w(s) ds = \min_\theta \left\| v_\theta - \mathcal{T}^\lambda v_\theta \right\|_w^2.
$$

Each method reduces the infinite-dimensional functional equation to a system of $n$ (generally nonlinear) equations in $n$ unknowns. We now have two fundamentally different approaches to solve this finite-dimensional problem.

### Two Solution Strategies for the Finite-Dimensional Problem

#### Strategy 1: Successive Approximation (Fixed-Point Iteration)

The most natural approach exploits the fixed-point structure. In function space, we would iterate:
$$
v_{k+1} = \mathcal{T}^\lambda v_k.
$$

But since we work in a finite-dimensional subspace, each iterate must be projected back. This gives **projected fixed-point iteration**:
$$
v_{k+1} = \Pi \mathcal{T}^\lambda v_k,
$$
where $\Pi$ is the projection operator onto $\text{span}\{\varphi_1, \ldots, \varphi_n\}$ defined by our weighted residual method.

In coefficient space, this becomes a two-step procedure:

**Step 1 (Evaluate)**: Given current coefficients $\theta^{(k)}$, evaluate the compound operator at representative points to get targets
$$
t_i^{(k)} = [\mathcal{T}^\lambda v_{\theta^{(k)}}](s_i).
$$

**Step 2 (Fit)**: Find new coefficients $\theta^{(k+1)}$ such that $v_{\theta^{(k+1)}}$ approximates these targets according to our projection method.

The specifics of Step 2 depend on the projection method:
- **Collocation**: Solve $\boldsymbol{\Phi} \theta^{(k+1)} = t^{(k)}$ where $\Phi_{ij} = \varphi_j(s_i)$
- **Galerkin**: Solve $M\theta^{(k+1)} = b^{(k)}$ where $M_{ij} = \langle \varphi_i, \varphi_j \rangle_w$ and $b_i^{(k)} = \langle t^{(k)}, \varphi_i \rangle_w$
- **Least squares**: Solve $\min_\theta \sum_i w_i (t_i^{(k)} - \boldsymbol{\varphi}(s_i)^\top \theta)^2$

This is **fitted value iteration** with the $\mathcal{T}^\lambda$ operator instead of $\mathcal{T}$. The algorithm remains conceptually identical: evaluate operator, fit to targets, repeat.

```{prf:algorithm} Fitted Value Iteration with $\mathcal{T}^\lambda$ (Collocation)
:label: fitted-vi-lambda-collocation

**Input** Collocation points $\{s_1, \ldots, s_n\}$, basis functions $\{\varphi_1, \ldots, \varphi_n\}$, compound parameter $\lambda \in [0,1)$, initial $\theta^{(0)}$, tolerance $\varepsilon > 0$

**Output** Converged coefficients $\theta^*$

1. Form collocation matrix $\boldsymbol{\Phi}$ with $\Phi_{ij} = \varphi_j(s_i)$
2. $k \leftarrow 0$
3. **repeat**
    1. **for** each $i = 1, \ldots, n$ **do**  (Evaluate $\mathcal{T}^\lambda v_{\theta^{(k)}}$ at collocation points)
        1. $t_i^{(k)} \leftarrow [\mathcal{T}^\lambda v_{\theta^{(k)}}](s_i)$
           
           (This is $(1-\lambda)\sum_{m=1}^{\infty} \lambda^{m-1} [\mathcal{T}^{(m)} v_{\theta^{(k)}}](s_i)$, computed as described below)
    2. Solve $\boldsymbol{\Phi} \theta^{(k+1)} = t^{(k)}$  (Fit: interpolate the targets)
    3. $k \leftarrow k + 1$
4. **until** $\|\theta^{(k)} - \theta^{(k-1)}\| < \varepsilon$
5. **return** $\theta^{(k)}$
```

**Key observation**: This algorithm has the same structure as standard fitted value iteration, but uses $\mathcal{T}^\lambda$ in Step 3.1 instead of $\mathcal{T}$. The question of *how* to compute $[\mathcal{T}^\lambda v](s)$ is separate from the two-level iteration structure. We will address computation below.

For Galerkin projection, the algorithm structure is similar, but Step 3.2 requires solving a weighted least-squares problem rather than exact interpolation:

```{prf:algorithm} Fitted Value Iteration with $\mathcal{T}^\lambda$ (Galerkin)
:label: fitted-vi-lambda-galerkin

**Input** Basis functions $\{\varphi_1, \ldots, \varphi_n\}$, weight function $w(s)$, compound parameter $\lambda \in [0,1)$, initial $\theta^{(0)}$, tolerance $\varepsilon > 0$

**Output** Converged coefficients $\theta^*$

1. Compute mass matrix $M_{ij} = \int_{\mathcal{S}} \varphi_i(s) \varphi_j(s) w(s) ds$
2. $k \leftarrow 0$
3. **repeat**
    1. **for** each $i = 1, \ldots, n$ **do**  (Evaluate projected $\mathcal{T}^\lambda v_{\theta^{(k)}}$)
        1. $b_i^{(k)} \leftarrow \int_{\mathcal{S}} [\mathcal{T}^\lambda v_{\theta^{(k)}}](s) \varphi_i(s) w(s) ds$
    2. Solve $M \theta^{(k+1)} = b^{(k)}$  (Fit: Galerkin projection)
    3. $k \leftarrow k + 1$
4. **until** $\|\theta^{(k)} - \theta^{(k-1)}\| < \varepsilon$
5. **return** $\theta^{(k)}$
```

In both algorithms, convergence depends on whether $\Pi \mathcal{T}^\lambda$ (the composition of projection with the compound operator) is a contraction. The parameter $\lambda$ provides a tuning knob: we hope to choose it such that $\Pi \mathcal{T}^\lambda$ contracts even when $\Pi \mathcal{T}$ does not.

#### Strategy 2: Rootfinding (Newton's Method)

Alternatively, we can treat the projection conditions as a rootfinding problem $G(\theta) = 0$ where:
- **Collocation**: $G_i(\theta) = v_{\theta}(s_i) - [\mathcal{T}^\lambda v_\theta](s_i)$
- **Galerkin**: $G_i(\theta) = \langle v_\theta - \mathcal{T}^\lambda v_\theta, \varphi_i \rangle_w$

Newton's method then iterates:
$$
\theta^{(k+1)} = \theta^{(k)} - J_G(\theta^{(k)})^{-1} G(\theta^{(k)}),
$$
requiring computation of the Jacobian $J_{ij} = \frac{\partial G_i}{\partial \theta_j}$.

### Why Consider $\mathcal{T}^\lambda$ Instead of $\mathcal{T}$?

The motivation is entirely about the **inner level** (finite-dimensional) convergence properties. The standard theory guarantees that $\mathcal{T}$ is a $\gamma$-contraction in the sup norm. But we care about whether **$\Pi \mathcal{T}$ contracts**, not just $\mathcal{T}$. As we saw in the previous chapter:
- Monotone projections preserve contraction in $\|\cdot\|_\infty$
- Orthogonal projections preserve contraction in $\|\cdot\|_\xi$ for policy evaluation when $\xi$ is the stationary distribution
- For general projections (e.g., Galerkin with polynomial bases), neither guarantee applies

The compound operator $\mathcal{T}^\lambda$ introduces an additional degree of freedom. Even if $\Pi \mathcal{T}$ fails to contract (making Strategy 1 diverge or converge slowly), perhaps $\Pi \mathcal{T}^\lambda$ does for some $\lambda \in (0,1)$. The parameter $\lambda$ controls **temporal smoothing** of the backup: small $\lambda$ focuses on immediate rewards, large $\lambda$ incorporates long-term dependencies. This smoothing can improve the conditioning of the projected operator composition.

**Convergence question**: Under what conditions does the successive approximation
$$
\theta^{(k+1)} = [\text{fit to } \{(s_i, [\mathcal{T}^\lambda v_{\theta^{(k)}}](s_i))\}]
$$
converge? This requires $\Pi \mathcal{T}^\lambda$ to be a contraction in some norm. Partial results exist:
- For tabular representations ($\Pi = I$), $\lambda$ makes no difference: $v^* = \mathcal{T}^\lambda v^*$ for any $\lambda$
- For policy evaluation with linear function approximation and on-policy sampling, TD($\lambda$) converges for all $\lambda \in [0,1]$ {cite}`TsitsiklisVanRoy1997`
- For Bellman optimality with general projections, convergence is not guaranteed even with $\lambda$. Divergence examples exist {cite}`Baird1995`

The weighted residual framework makes clear what we are solving: the fixed-point equation $v = \Pi \mathcal{T}^\lambda v$ in coefficient space. Whether we use successive approximation (Strategy 1) or Newton's method (Strategy 2) to solve the finite-dimensional problem is a separate algorithmic choice, independent of the operator $\mathcal{T}^\lambda$ itself.

## Computing $[\mathcal{T}^\lambda v](s)$: The Role of Recursive Filtering

Having established the two-level iteration structure, we now address a critical implementation question: **How do we compute $[\mathcal{T}^\lambda v_\theta](s_i)$ in Step 3.1 of the fitted value iteration algorithms?**

The definition $\mathcal{T}^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \mathcal{T}^{(n)}$ requires evaluating infinitely many multi-step operators. This appears intractable. We will see that the geometric structure enables a **recursive decomposition** that reduces computation to tracking a single auxiliary variable.

**Important conceptual point**: The recursive filtering machinery described below is purely about *implementing* the operator evaluation $[\mathcal{T}^\lambda v](s)$ efficiently. It does not change the two-level structure of fitted value iteration. Whether we compute $[\mathcal{T}^\lambda v](s)$ via the explicit sum, via recursion, or via any other method, the outer iterate $\theta^{(k+1)} = \text{fit}[\mathcal{T}^\lambda v_{\theta^{(k)}}]$ remains the same.

This is analogous to numerical quadrature: when fitted value iteration with $\mathcal{T}$ requires computing $\mathbb{E}[v(s')]$, we can use Monte Carlo sampling, Gauss-Hermite quadrature, or any other integration method. The choice of integration technique doesn't change the fitted value iteration structure—it's an implementation detail of evaluating the Bellman operator. Similarly, recursive filtering is an implementation detail of evaluating the compound operator $\mathcal{T}^\lambda$.

### The Key Recursion

Consider the $n$-step return at a specific state $s$:

$$
G^{(n)}_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n v_\theta(s_{t+n}).
$$

The $\lambda$-return is the geometric average:

$$
G^\lambda_t = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G^{(n)}_t.
$$

We can expand this sum:

\begin{align}
G^\lambda_t &= (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \left( \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n v_\theta(s_{t+n}) \right) \\
&= (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \sum_{k=0}^{n-1} \gamma^k r_{t+k} + (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \gamma^n v_\theta(s_{t+n}).
\end{align}

The second term simplifies: $(1-\lambda) \sum_{n=1}^{\infty} (\lambda\gamma)^{n-1} \gamma v_\theta(s_{t+n})$. But the first term involves nested sums. Let's approach this differently.

**Telescoping approach**: Write $G^{(n+1)}_t = r_t + \gamma G^{(n)}_{t+1}$. Then:

\begin{align}
G^\lambda_t &= (1-\lambda) G^{(1)}_t + (1-\lambda) \sum_{n=1}^{\infty} \lambda^n G^{(n+1)}_t \\
&= (1-\lambda) [r_t + \gamma v_\theta(s_{t+1})] + (1-\lambda) \sum_{n=1}^{\infty} \lambda^n [r_t + \gamma G^{(n)}_{t+1}] \\
&= (1-\lambda) r_t + (1-\lambda)\lambda \sum_{n=1}^{\infty} \lambda^{n-1} [r_t + \gamma G^{(n)}_{t+1}] + (1-\lambda)\gamma v_\theta(s_{t+1}) \\
&= r_t + \gamma \left[ (1-\lambda) v_\theta(s_{t+1}) + \lambda G^\lambda_{t+1} \right].
\end{align}

This gives the recursion:

$$
G^\lambda_t = r_t + \gamma \left[ (1-\lambda) v_\theta(s_{t+1}) + \lambda G^\lambda_{t+1} \right].
$$

**Interpretation**: The $\lambda$-return mixes the one-step bootstrap $(1-\lambda)v_\theta(s_{t+1})$ with the recursive $\lambda$-return $G^\lambda_{t+1}$ from the next state. This recursion reduces the infinite sum to a single backward pass through a trajectory.

### State-Space Representation

We can view this recursion as a **linear time-invariant (LTI) dynamical system**. Define the auxiliary variable (eligibility trace):

$$
z_t = \sum_{k=0}^{t} (\lambda\gamma)^{t-k} \nabla v_\theta(s_k),
$$

where $\nabla v_\theta(s) = \frac{\partial v_\theta(s)}{\partial \theta}$ is the gradient of the value function. This trace accumulates gradients with exponential decay rate $\lambda\gamma$.

The trace satisfies a first-order recursion:

$$
z_{t+1} = \lambda\gamma z_t + \nabla v_\theta(s_{t+1}).
$$

This is the **backward view**: we maintain $z_t$ as a state variable and update it recursively as we observe transitions $(s_t, r_t, s_{t+1})$. The temporal difference error at time $t$ is:

$$
\delta_t = r_t + \gamma v_\theta(s_{t+1}) - v_\theta(s_t).
$$

The gradient update for the $\lambda$-return objective can be written:

$$
\nabla_\theta \mathcal{L} = \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t \delta_t z_t \right].
$$

**State-space form**: Define state $z_t \in \mathbb{R}^p$, input $u_t = \nabla v_\theta(s_t)$, and output $y_t = \delta_t z_t$. Then:

\begin{align}
z_{t+1} &= \lambda\gamma z_t + u_t, \\
y_t &= \delta_t z_t.
\end{align}

This is a first-order LTI system with state transition matrix $A = \lambda\gamma$ (scalar), input matrix $B = 1$, and time-varying output map $C_t = \delta_t$. The eligibility trace acts as a **low-pass filter** that smooths gradients over time, with cutoff determined by $\lambda\gamma$.

### Convolution Perspective

Alternatively, we can view the eligibility trace computation as a **discrete convolution**. The trace at time $t$ is:

$$
z_t = \sum_{k=0}^{t} (\lambda\gamma)^{t-k} \nabla v_\theta(s_k) = h * u,
$$

where $h_k = (\lambda\gamma)^k$ is the impulse response of an exponential filter and $u_k = \nabla v_\theta(s_k)$ is the input sequence. This convolution perspective connects to signal processing: the eligibility trace is the output of passing the gradient sequence through an IIR (infinite impulse response) filter with transfer function:

$$
H(z) = \frac{1}{1 - \lambda\gamma z^{-1}}.
$$

This filter has a single pole at $z = \lambda\gamma$, giving exponential decay. The parameter $\lambda$ controls the decay rate: small $\lambda$ gives fast decay (short memory), large $\lambda$ gives slow decay (long memory).

### The Three-Level Hierarchy: Outer, Inner, Implementation

We can now see the complete hierarchy of abstraction in weighted residual methods with compound operators:

**Level 1 (Outer/Functional)**: 
$$v = \mathcal{T}^\lambda v$$
An infinite-dimensional fixed-point equation in function space. This is the problem statement.

**Level 2 (Inner/Parametric)**: 
After choosing a finite-dimensional approximation $v_\theta = \sum_i \theta_i \varphi_i$ and projection method, we obtain a finite-dimensional problem. We have **two solution strategies**:

- **Strategy A (Successive Approximation)**:
  $$\theta^{(k+1)} = \arg\min_\theta \sum_i w_i \left( v_\theta(s_i) - [\mathcal{T}^\lambda v_{\theta^{(k)}}](s_i) \right)^2$$
  This is fitted value iteration: evaluate operator at current parameters, fit new parameters to targets. Converges when $\Pi \mathcal{T}^\lambda$ is a contraction.

- **Strategy B (Newton's Method)**:
  Treat $G(\theta) = v_\theta - \Pi \mathcal{T}^\lambda v_\theta = 0$ as a rootfinding problem and iterate:
  $$\theta^{(k+1)} = \theta^{(k)} - J_G(\theta^{(k)})^{-1} G(\theta^{(k)})$$
  This is equivalent to policy iteration in the Q-function case. Requires computing Jacobians (Envelope Theorem for optimization operators).

**Level 3 (Implementation/Computational)**:
How do we compute $[\mathcal{T}^\lambda v_{\theta^{(k)}}](s_i)$ in the evaluation step?

- **If we have exact MDP knowledge**: Compute expectations via dynamic programming:
  $$[\mathcal{T}^{(n)} v](s) = \max_a \left\{ r(s,a) + \gamma \sum_{s'} p(s'|s,a) [\mathcal{T}^{(n-1)} v](s') \right\}$$
  Then form the geometric average $(1-\lambda)\sum_{n=1}^{N} \lambda^{n-1} [\mathcal{T}^{(n)} v](s)$ by truncating at large $N$.

- **If we only have a simulator**: Sample trajectories from $s$, compute returns $G^\lambda_t$ along trajectories, and average. The eligibility trace recursion provides memory-efficient computation:
  $$z_{t+1} = \lambda\gamma z_t + \nabla v_\theta(s_t), \quad \nabla_\theta L = \mathbb{E}\left[\sum_t \delta_t z_t\right]$$

The three levels are **cleanly separated**:
- **Level 1**: What equation are we solving? (functional equation, operator choice)
- **Level 2**: How do we reduce it to finite dimensions and solve the resulting system? (projection method, successive approximation vs rootfinding)
- **Level 3**: How do we evaluate the operator given our computational constraints? (exact DP, Monte Carlo, eligibility traces)

The eligibility trace filter lives entirely at **Level 3**—it is a computational device for Monte Carlo estimation, analogous to choosing Gauss-Hermite quadrature vs Monte Carlo for computing $\mathbb{E}[v(s')]$ in the standard Bellman operator. It does not change the problem formulation (Level 1) or the coefficient iteration structure (Level 2).

**Why this hierarchy matters**: 
1. **Conceptual clarity**: We can reason about convergence at Level 2 (does $\Pi \mathcal{T}^\lambda$ contract?) independently of Level 3 concerns (how do we compute it?).
2. **Modularity**: We can swap Level 3 implementations (exact DP ↔ Monte Carlo ↔ importance sampling) without changing Levels 1-2.
3. **Avoiding confusion**: Eligibility traces are often presented as if they define the algorithm. In fact, they're an implementation optimization for a particular case (Monte Carlo estimation at Level 3).

## Worked Example: Policy Evaluation with Linear Approximation

To make the two-level structure concrete, consider **policy evaluation** with a fixed policy $\pi$ and linear function approximation $v_\theta(s) = \boldsymbol{\varphi}(s)^\top \boldsymbol{\theta}$. 

**Outer level**: We seek $v_\pi$ satisfying $v_\pi = \mathcal{T}^\lambda_\pi v_\pi$, where:
$$
[\mathcal{T}^\lambda_\pi v](s) = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \mathbb{E}_\pi\left[ \sum_{t=0}^{n-1} \gamma^t r_t + \gamma^n v(s_n) \,\Big|\, s_0=s \right].
$$

**Inner level (Galerkin projection)**: We parameterize as $v_\theta = \boldsymbol{\Phi}\boldsymbol{\theta}$ and seek $\theta$ such that:
$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (v_\theta - \mathcal{T}^\lambda_\pi v_\theta) = \mathbf{0},
$$
where $\boldsymbol{\Xi} = \text{diag}(\xi)$ with $\xi$ the state distribution (often the stationary distribution under $\pi$).

Expanding this condition:
$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \boldsymbol{\Phi} \boldsymbol{\theta} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\mathcal{T}^\lambda_\pi v_\theta).
$$

For **successive approximation** (fitted value iteration at Level 2), we iterate:
$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \boldsymbol{\Phi} \boldsymbol{\theta}^{(k+1)} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\mathcal{T}^\lambda_\pi v_{\theta^{(k)}}).
$$

This is precisely the **LSTD($\lambda$)** (Least-Squares Temporal Difference with $\lambda$) fixed point. When $\lambda = 0$, we recover standard LSTD. For $\lambda > 0$, we solve for the projection of a multi-step operator.

**Level 3 (Implementation)**: How do we compute the right-hand side $\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\mathcal{T}^\lambda_\pi v_{\theta^{(k)}})$?

- **With exact MDP knowledge**: Sum over states and compute expectations:
  $$[\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\mathcal{T}^\lambda_\pi v)]_i = \sum_s \xi(s) \varphi_i(s) [\mathcal{T}^\lambda_\pi v](s)$$
  
- **With only a simulator**: Sample trajectories following $\pi$, compute $\lambda$-returns along each trajectory, and use eligibility traces to efficiently accumulate the sum $\sum_t \delta_t z_t$ where $z_t$ is the filtered gradient history.

The two-level structure (outer functional equation, inner coefficient iteration) is **independent** of whether we have a model or only samples. The sampling concern is purely at Level 3.

## Bellman Optimality with $\lambda$-Returns: Fitted Q-Iteration

For the Bellman optimality operator with linear Q-function approximation $Q_\theta(s,a) = \boldsymbol{\varphi}(s,a)^\top \boldsymbol{\theta}$, the structure is analogous.

**Outer level**: We seek $Q^*$ satisfying $Q^* = \mathcal{T}^\lambda_Q Q^*$, where:
$$
[\mathcal{T}^{(n)}_Q Q](s,a) = \mathbb{E}\left[ \sum_{t=0}^{n-1} \gamma^t r_t + \gamma^n \max_{a'} Q(s_n, a') \,\Big|\, s_0=s, a_0=a, \pi^Q_{\text{greedy}} \text{ thereafter} \right],
$$
and $\mathcal{T}^\lambda_Q = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \mathcal{T}^{(n)}_Q$.

**Inner level (Collocation)**: Choose state-action pairs $\{(s_i, a_i)\}_{i=1}^n$ and require:
$$
\boldsymbol{\varphi}(s_i, a_i)^\top \boldsymbol{\theta} = [\mathcal{T}^\lambda_Q Q_\theta](s_i, a_i), \quad i = 1, \ldots, n.
$$

**Successive approximation** (fitted Q-iteration at Level 2):
$$
\boldsymbol{\Phi} \boldsymbol{\theta}^{(k+1)} = t^{(k)}, \quad t_i^{(k)} = [\mathcal{T}^\lambda_Q Q_{\theta^{(k)}}](s_i, a_i),
$$
where $\boldsymbol{\Phi}$ is the feature matrix with rows $\boldsymbol{\varphi}(s_i, a_i)^\top$.

**Level 3 (Implementation)**: Computing $[\mathcal{T}^\lambda_Q Q](s,a)$ requires:

1. **Multi-step returns with greedy action selection**: For each $n$, compute
   $$[\mathcal{T}^{(n)}_Q Q](s,a) = \mathbb{E}\left[\sum_{t=0}^{n-1} \gamma^t r_t + \gamma^n \max_{a'} Q(s_n, a') \,\Big|\, s_0=s, a_0=a, \text{greedy thereafter}\right]$$

2. **Geometric averaging**: $[\mathcal{T}^\lambda_Q Q](s,a) = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} [\mathcal{T}^{(n)}_Q Q](s,a)$

With a simulator, we sample trajectories and use eligibility traces to efficiently compute returns. The eligibility trace for Q-learning is:
$$
z_t = \sum_{k=0}^{t} (\lambda\gamma)^{t-k} \mathbb{I}[a_k \text{ is greedy}] \nabla Q_\theta(s_k, a_k).
$$

The indicator $\mathbb{I}[a_k \text{ is greedy}]$ implements a **cut-off**: the trace resets whenever a non-greedy action is taken. This is **Watkins's Q($\lambda$)** {cite}`Watkins1989`.

**Why the cut-off?** The multi-step operator $\mathcal{T}^{(n)}_Q$ assumes greedy actions at intermediate states. When we take a non-greedy action, we're no longer on a trajectory that $\mathcal{T}^{(n)}_Q$ would generate. The cut-off ensures we only accumulate eligibility along greedy sub-trajectories. This is not a heuristic—it follows directly from the definition of $\mathcal{T}^{(n)}_Q$.

**Peng's Q($\lambda$)** {cite}`Peng1993` omits the cut-off:
$$
z_t = \sum_{k=0}^{t} (\lambda\gamma)^{t-k} \nabla Q_\theta(s_k, a_k).
$$
This computes $\lambda$-returns for the **actual trajectory** (behavior policy), not the greedy trajectory. It solves a different Level 1 equation—not $Q = \mathcal{T}^\lambda_Q Q$ but $Q = \mathcal{T}^\lambda_{\pi_b} Q$ for behavior policy $\pi_b$. When $\pi_b$ is $\varepsilon$-greedy, Peng's approach has lower variance but solves a biased problem (not the Bellman optimality equation).

## Open Questions and Future Directions

### Convergence Theory for $\Pi \mathcal{T}^\lambda$

The fundamental question remains: for which projection operators $\Pi$ and which values of $\lambda$ does $\Pi \mathcal{T}^\lambda$ contract? We have partial answers:
- For policy evaluation with linear function approximation and on-policy sampling, convergence holds for all $\lambda \in [0,1]$ {cite}`TsitsiklisVanRoy1997`
- For optimality operators, even with linear approximation, convergence is not guaranteed. Divergence examples exist {cite}`Baird1995`

A complete theory would characterize:
1. When does $\lambda > 0$ improve contraction rate compared to $\lambda = 0$?
2. Is there an optimal choice of $\lambda$ for a given projection operator and MDP?
3. Can we adaptively adjust $\lambda$ during learning to improve stability?

### Spectral Analysis of $\Pi \mathcal{T}^\lambda$: When Does $\lambda$ Help?

For policy evaluation with linear function approximation, the iteration $\theta^{(k+1)} = \text{fit}[\mathcal{T}^\lambda_\pi v_{\theta^{(k)}}]$ is a linear map in coefficient space. Let $A_\lambda \in \mathbb{R}^{n \times n}$ represent this map: $\theta^{(k+1)} = A_\lambda \theta^{(k)}$ (assuming zero rewards for simplicity). Convergence depends on the **spectral radius** $\rho(A_\lambda) = \max_i |\mu_i|$ where $\{\mu_i\}$ are eigenvalues of $A_\lambda$.

**Key question**: How does $\rho(A_\lambda)$ depend on $\lambda$?

For **Galerkin projection** with policy $\pi$, the matrix $A_\lambda$ can be expressed in terms of the standard projected Bellman operator. Let $A_0 = \Pi_\xi \mathcal{T}_\pi$ be the $\lambda=0$ operator (standard TD), where $\Pi_\xi = \boldsymbol{\Phi}(\boldsymbol{\Phi}^\top\boldsymbol{\Xi}\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^\top\boldsymbol{\Xi}$ is Galerkin projection and $\mathcal{T}_\pi = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi$ is the policy Bellman operator (in matrix form on the finite state space).

The compound operator satisfies:
\begin{align}
A_\lambda &= \Pi_\xi \mathcal{T}^\lambda_\pi = \Pi_\xi \left[(1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} \mathcal{T}^n_\pi\right] \\
&= (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} (\Pi_\xi \mathcal{T}_\pi)^n \quad \text{(if projection and operator commute)} \\
&= (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} A_0^n.
\end{align}

**Problem**: Projection and operator generally **do not commute**: $\Pi_\xi \mathcal{T}_\pi \mathcal{T}_\pi \neq \Pi_\xi \mathcal{T}_\pi \cdot \Pi_\xi \mathcal{T}_\pi$ in general. The exact relationship between eigenvalues of $A_\lambda$ and $A_0$ is complex.

However, we can derive bounds. If $A_0$ has spectral radius $\rho(A_0) < 1$ (so TD converges), then for small enough $\lambda$, $A_\lambda$ also contracts. As $\lambda \to 1$, the operator averages over increasingly long backup horizons, which can either help or hurt depending on the structure of $\Pi_\xi \mathcal{T}_\pi$.

**Empirical observations** from the reinforcement learning literature:
- Intermediate values $\lambda \in [0.8, 0.95]$ often work best in practice
- Very small $\lambda$ gives high variance (short bootstraps)
- $\lambda = 1$ (Monte Carlo) has zero bias but maximum variance
- The optimal $\lambda$ is problem-dependent and relates to the eigenvalue structure

A complete theory characterizing $\rho(A_\lambda)$ as a function of $\lambda$ remains an open problem.

### Connection to Resolvent Theory and Operator Preconditioning

The compound operator has a closed-form expression as a Neumann series sum. For policy evaluation (linear operator), assuming $\|\lambda \mathcal{T}_\pi\| < 1$:

$$
\mathcal{T}^\lambda_\pi = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \mathcal{T}^n_\pi = (1-\lambda) \mathcal{T}_\pi (I - \lambda \mathcal{T}_\pi)^{-1}.
$$

This is a **resolvent-like** operator. In functional analysis, the resolvent $(I - \alpha A)^{-1}$ is central to understanding operator spectra via the resolvent identity and analytic continuation. The compound operator $\mathcal{T}^\lambda$ is not quite a resolvent (there's an extra $\mathcal{T}_\pi$ factor and the $(1-\lambda)$ scaling), but it shares structural similarities.

**Preconditioning interpretation**: In iterative methods for linear systems $Ax = b$, preconditioning replaces the system with $M^{-1}Ax = M^{-1}b$ where $M$ is chosen so that $M^{-1}A$ has better spectral properties (clustered eigenvalues, smaller condition number). Can we view $(I - \lambda \mathcal{T}_\pi)^{-1}$ as a preconditioner for $\mathcal{T}_\pi$?

The analogy is imperfect because we're not solving a linear system directly—we're finding a fixed point. But the intuition carries over: we hope that $(1-\lambda)\mathcal{T}_\pi(I - \lambda\mathcal{T}_\pi)^{-1}$ has better contraction properties than $\mathcal{T}_\pi$ when composed with projection $\Pi$.

**Connection to relaxation methods**: In numerical linear algebra, **successive over-relaxation (SOR)** accelerates iterative solvers by using updates of the form $x^{(k+1)} = (1-\omega)x^{(k)} + \omega x^{(k+1)}_{\text{Gauss-Seidel}}$. The parameter $\omega$ blends old and new iterates. Similarly, $\lambda$ in TD($\lambda$) blends short and long backups. Is there a deeper connection? Can optimal SOR theory (choosing $\omega$ to minimize spectral radius) inform optimal $\lambda$ selection?

### Geometric Interpretation: Smoothing in Value Function Space

Another perspective on why $\lambda$ might help: the compound operator $\mathcal{T}^\lambda$ is a **smoothing operator**. By averaging over multiple backup depths, it reduces the sensitivity to errors at any single depth.

Consider the residual at iteration $k$: $e^{(k)} = v^* - v^{(k)}$. Standard value iteration propagates this error via:
$$
e^{(k+1)} = v^* - \mathcal{T} v^{(k)} = \mathcal{T} v^* - \mathcal{T} v^{(k)} = \gamma \mathbf{P}_\pi e^{(k)}.
$$

The error propagates through the transition operator $\mathbf{P}_\pi$ scaled by $\gamma$. With projection, we have $e^{(k+1)} = v^* - \Pi \mathcal{T} v^{(k)}$, and the projection introduces additional error in directions outside $\text{span}(\boldsymbol{\Phi})$.

With the compound operator:
$$
e^{(k+1)} = v^* - \Pi\mathcal{T}^\lambda v^{(k)} = v^* - \Pi (1-\lambda)\sum_n \lambda^{n-1} \mathcal{T}^n v^{(k)}.
$$

The error now depends on an exponentially weighted average of $\mathcal{T}^n v^{(k)}$. If projection error has high-frequency components (oscillations due to polynomial bases), averaging over multiple depths might smooth them out, improving the effective contraction rate.

This geometric intuition suggests $\lambda$ acts as a low-pass filter on value function updates, similar to momentum in gradient descent or moving averages in time series. But making this rigorous requires analyzing how projection error components transform under $\mathcal{T}^n$ and how geometric averaging affects their spectrum—an open research direction.

### Non-Geometric Weightings

We used geometric weights $(1-\lambda)\lambda^{n-1}$ because they enable recursive computation. But are they optimal for improving projection-composition contraction? One could consider:
- Truncated sums: $\mathcal{T}^{[N]} = N^{-1} \sum_{n=1}^{N} \mathcal{T}^{(n)}$ (uniform average up to horizon $N$)
- Polynomial weightings: $w_n = c n^{-\alpha}$ for heavy tails
- Data-adaptive weightings: learn weights from observed variance

These lose the recursive structure but might have better theoretical properties. Implementing them requires storing trajectory segments, trading computation for potentially better approximation.

### State-Space Models and Control Theory

The state-space representation of eligibility traces connects to control theory. Eligibility is a **controlled dynamical system** where the control input is the gradient sequence. Tools from control theory might apply:
- Observability/controllability analysis
- Optimal filtering (Kalman-like) for noisy gradient estimates
- Stability analysis via Lyapunov functions

Can we design better "filters" than the exponential decay $(\lambda\gamma)^k$? For example, a second-order filter:

$$
z_{t+1} = A z_t + B u_t, \quad A = \begin{bmatrix} a_1 & a_2 \\ 1 & 0 \end{bmatrix}, \quad B = \begin{bmatrix} 1 \\ 0 \end{bmatrix},
$$

would allow richer temporal dynamics. This increases state dimension but might better match the temporal structure of value propagation in specific MDPs.

### Extension to Continuous Time

The discrete-time geometric stopping naturally extends to **continuous-time exponential stopping**. Define an exponentially distributed stopping time $\tau \sim \text{Exp}(\lambda)$, and consider:

$$
[\mathcal{T}^\lambda v](s) = \mathbb{E}_\tau\left[ \int_0^\tau e^{-\rho t} r(s_t, a_t) dt + e^{-\rho \tau} v(s_\tau) \right],
$$

where $s_t$ evolves according to a continuous-time Markov process. This connects to **Hamilton-Jacobi-Bellman (HJB) equations** with random horizons. The eligibility trace becomes a continuous-time filter:

$$
\frac{d z_t}{dt} = -\lambda z_t + \nabla v_\theta(s_t).
$$

This is a first-order linear ODE. Do the convergence properties differ in continuous time? Are there advantages to working with continuous formulations (e.g., for robot control with fast sampling rates)?

## Practical Recommendations

Having developed the theory, we now consider practical guidance for implementing weighted residual methods with compound operators.

### When to Use $\lambda > 0$

**Use multi-step operators ($\lambda \in (0.5, 0.95)$) when**:
1. The projected operator $\Pi \mathcal{T}$ contracts slowly or not at all (high discount factor $\gamma$, non-monotone projection)
2. You have access to trajectory data (simulation or real experience)
3. The variance-bias tradeoff favors some bias reduction (noisy rewards, sparse features)

**Stick with one-step ($\lambda = 0$) when**:
1. The projection is monotone (linear interpolation, state aggregation) and $\gamma$ is moderate
2. You only have single-transition samples (no trajectories)
3. Off-policy learning is required (multi-step methods complicate importance sampling)

**Consider Monte Carlo ($\lambda = 1$) when**:
1. Episodes are short (low variance)
2. Bootstrapping is unreliable (poor initial value function, very sparse features)
3. Theoretical simplicity is valued (unbiased returns, no bootstrap error)

### Choosing $\lambda$ in Practice

There is no universal optimal $\lambda$. Several approaches:

**Grid search**: Try $\lambda \in \{0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0\}$ on a validation set. Often $\lambda \in [0.8, 0.95]$ works well, but this is problem-dependent.

**Adaptive $\lambda$**: Some methods adjust $\lambda$ during learning:
- Start with large $\lambda$ (more Monte Carlo) for exploration, decay toward small $\lambda$ (more bootstrapping) as value function improves
- Use different $\lambda$ for different states (learned via meta-gradient methods)

**Cross-validation**: For episodic tasks, use held-out episodes to measure prediction error at different $\lambda$ values.

**Theory-guided**: If you can estimate the spectrum of $\Pi \mathcal{T}$ (eigenvalues of the projected operator), choose $\lambda$ to minimize the spectral radius of $\Pi \mathcal{T}^\lambda$. This requires sophisticated analysis but could be automated.

### Computational Considerations

**Model-based (Level 3: Exact DP)**:
- Compute $[\mathcal{T}^{(n)} v](s)$ recursively up to horizon $N$ where $\lambda^N$ is negligible
- Storage: $O(|\mathcal{S}| \cdot N)$ for intermediate $n$-step values
- Useful when transitions are cheap to evaluate (tabular or low-dimensional problems with analytic models)

**Model-free (Level 3: Monte Carlo + Eligibility)**:
- Sample trajectories, maintain eligibility trace $z_t = \lambda\gamma z_{t-1} + \nabla v_\theta(s_t)$
- Storage: $O(|\theta|)$ for the trace (constant per trajectory length)
- Essential when only a simulator is available
- Implementation: Use trace accumulation for LSTD($\lambda$) or online TD($\lambda$)

**Hybrid**:
- Use short exact DP for $n \leq n_0$ (cheap), then Monte Carlo with traces for $n > n_0$ (expensive)
- Useful when partial model is available (e.g., reward function known, transitions require simulation)

### Algorithmic Variants at Level 2

Recall that at Level 2 (parametric/coefficient space), we can use either:

**Successive approximation** (fitted value iteration):
- Simple to implement: evaluate operator, fit, repeat
- Requires $\Pi \mathcal{T}^\lambda$ to be a contraction
- Works well with monotone projections or on-policy Galerkin

**Newton's method** (policy iteration):
- Faster convergence near solution (quadratic vs. linear)
- Requires Jacobian computation (Envelope Theorem for max operators)
- More robust to weak contraction or non-monotone projections
- Equivalent to policy iteration for Q-functions

**Hybrid strategy**:
1. Run a few iterations of fitted value iteration to get into the basin of attraction
2. Switch to Newton's method for rapid final convergence
3. Useful when contraction is weak ($\gamma$ close to 1) or projection is non-monotone

### Implementation Checklist

When implementing TD($\lambda$) or Q($\lambda$) with the three-level perspective:

**Level 1 (Operator choice)**:
- [ ] Are you solving policy evaluation ($\mathcal{T}_\pi$) or optimality ($\mathcal{T}_Q$)?
- [ ] What is the target operator? (standard one-step or $\lambda$-compound)
- [ ] For Q($\lambda$): Watkins (greedy trajectory) or Peng (behavior trajectory)?

**Level 2 (Projection and solution method)**:
- [ ] What is the approximation architecture? (linear features, neural network, etc.)
- [ ] What projection method? (collocation, Galerkin, least-squares)
- [ ] What solution strategy? (successive approximation, Newton, stochastic gradient)
- [ ] Do you have a convergence guarantee? (monotone projection, matched weighting, etc.)

**Level 3 (Evaluation implementation)**:
- [ ] Model-based or model-free?
- [ ] If model-free: full trajectories or single transitions?
- [ ] If using traces: correct decay rate $\lambda\gamma$? Correct reset on episode boundaries?
- [ ] For Q($\lambda$): correct greedy cutoff implementation?

## Conclusion and Integration with the Main Chapter

We have developed compound Bellman operators $\mathcal{T}^\lambda$ as a natural extension of the weighted residual framework from the main chapter. The key insights:

1. **The three-level hierarchy** cleanly separates:
   - **What** equation we solve (functional equation, operator choice)
   - **How** we reduce it to finite dimensions (projection, successive approximation vs. Newton)
   - **Implementation** details (exact DP, Monte Carlo, eligibility traces)

2. **Eligibility traces are not fundamental** to the formulation—they are an efficient implementation technique (Level 3) for Monte Carlo estimation when we lack a model.

3. **The forward/backward view distinction** is purely computational (Level 3), not conceptual. Both views compute the same $\lambda$-return, just reorganized for different computational constraints.

4. **The motivation for $\lambda$** is about improving finite-dimensional convergence (Level 2): even if $\Pi \mathcal{T}$ fails to contract, $\Pi \mathcal{T}^\lambda$ might contract for appropriate $\lambda$. This is about the parametric iteration structure, not about psychological intuitions regarding memory traces.

5. **For Bellman optimality** (Q($\lambda$)), the multi-step operator naturally involves greedy action selection at intermediate states. The eligibility cutoff in Watkins's Q($\lambda$) follows directly from this definition, not from algorithmic considerations.

6. **Open theoretical questions** remain about when $\Pi \mathcal{T}^\lambda$ contracts, how to choose $\lambda$ optimally, and connections to resolvent theory and operator preconditioning.

This perspective integrates naturally with the main chapter's development:
- We previously asked: when does $\Pi \mathcal{T}$ (composition of projection with Bellman operator) contract?
- Now we ask: can we modify the operator via $\lambda$-averaging to improve contraction properties?
- Both questions live at the **Level 2** (parametric iteration) of the hierarchy

The machinery of weighted residual methods (test functions, orthogonality conditions, Galerkin vs. collocation) applies identically whether we use $\mathcal{T}$ or $\mathcal{T}^\lambda$. The compound operator extends the framework without changing its fundamental structure. This unification reveals TD($\lambda$), Q($\lambda$), and related methods as natural instances of weighted residual methods applied to a family of operators, demystifying the traditional presentation and connecting reinforcement learning to classical numerical analysis.

## Summary: The Three-Level Structure of TD($\lambda$) and Q($\lambda$)

We have extended the weighted residual framework to compound Bellman operators $\mathcal{T}^\lambda$ formed by geometric averaging over multi-step operators. The key insight is the **three-level hierarchy** that cleanly separates concerns:

### The Hierarchy

| **Level** | **What it specifies** | **Example for TD($\lambda$)** |
|:----------|:---------------------|:------------------------------|
| **1. Outer (Functional)** | Which infinite-dimensional equation to solve | $v_\pi = \mathcal{T}^\lambda_\pi v_\pi$ where $\mathcal{T}^\lambda_\pi = (1-\lambda)\sum_n \lambda^{n-1} \mathcal{T}^{(n)}_\pi$ |
| **2. Inner (Parametric)** | How to reduce to finite dimensions and solve | Galerkin: $\boldsymbol{\Phi}^\top\boldsymbol{\Xi}(v_\theta - \mathcal{T}^\lambda_\pi v_\theta) = 0$<br>Successive approx: $\theta^{(k+1)} = \text{fit}[\mathcal{T}^\lambda_\pi v_{\theta^{(k)}}]$ |
| **3. Implementation** | How to compute operator evaluations | Exact: $[\mathcal{T}^\lambda v](s) = (1-\lambda)\sum_n \lambda^{n-1}[\mathcal{T}^n v](s)$<br>Sampled: Eligibility traces $z_{t+1} = \lambda\gamma z_t + \nabla v(s_t)$ |

### How Standard Algorithms Fit

| **Algorithm** | **Level 1 (Operator)** | **Level 2 (Projection)** | **Level 3 (Evaluation)** |
|:--------------|:-----------------------|:-------------------------|:-------------------------|
| **Fitted VI** | $v = \mathcal{T} v$ | Collocation or Galerkin<br>Successive approximation | Exact DP or MC |
| **LSTD** | $v_\pi = \mathcal{T}_\pi v_\pi$ | Galerkin with $\xi$-weighting<br>Closed-form solution | Exact (model-based) |
| **TD($\lambda$)** | $v_\pi = \mathcal{T}^\lambda_\pi v_\pi$ | Galerkin with $\xi$-weighting<br>Stochastic gradient | MC + eligibility traces |
| **LSTD($\lambda$)** | $v_\pi = \mathcal{T}^\lambda_\pi v_\pi$ | Galerkin with $\xi$-weighting<br>Closed-form solution | MC + trace-based matrix estimation |
| **Q($\lambda$)** (Watkins) | $Q^* = \mathcal{T}^\lambda_Q Q^*$ | Collocation<br>Successive approximation | MC + eligibility with greedy cutoff |
| **Q($\lambda$)** (Peng) | $Q = \mathcal{T}^\lambda_{\pi_b} Q$ | Collocation<br>Successive approximation | MC + eligibility (no cutoff) |

### Key Points

1. **Motivation for $\lambda$**: The parameter $\lambda$ provides an additional degree of freedom at Level 1. Even if $\Pi \mathcal{T}$ fails to contract, perhaps $\Pi \mathcal{T}^\lambda$ does for appropriate $\lambda$. This is about improving finite-dimensional convergence (Level 2), not changing the underlying optimal value function (which satisfies $v^* = \mathcal{T}^\lambda v^*$ for any $\lambda$).

2. **Eligibility traces are Level 3**: The forward/backward view distinction, state-space models, and filtering are purely about efficient Monte Carlo implementation. They don't change what equation we're solving (Level 1) or the structure of coefficient iteration (Level 2).

3. **Watkins vs Peng**: The difference is at **Level 1** (which operator). Watkins solves the Bellman optimality equation with multi-step greedy backups. Peng solves the behavior policy equation. The eligibility cutoff follows from the operator definition, not from algorithmic considerations.

4. **Convergence is a Level 2 question**: Whether successive approximation $\theta^{(k+1)} = \text{fit}[\mathcal{T}^\lambda v_{\theta^{(k)}}]$ converges depends on whether $\Pi \mathcal{T}^\lambda$ is a contraction. This is independent of whether we use exact DP or Monte Carlo (Level 3). The theory remains incomplete:
   - For policy evaluation with on-policy sampling and linear FA, TD($\lambda$) converges for all $\lambda \in [0,1]$
   - For Bellman optimality with general projections, even with $\lambda$, divergence examples exist
   - The spectral properties of $\Pi \mathcal{T}^\lambda$ and optimal choice of $\lambda$ are not fully understood

5. **Connection to resolvent theory**: The compound operator $(1-\lambda)\sum_n \lambda^{n-1}\mathcal{T}^n = (1-\lambda)(I - \lambda\mathcal{T})^{-1}\mathcal{T}$ resembles a resolvent or preconditioned operator. Can ideas from numerical linear algebra (preconditioning to improve condition numbers) provide insight into when $\Pi \mathcal{T}^\lambda$ has better contraction properties than $\Pi \mathcal{T}$?

### Why This Perspective Matters

The traditional presentation of TD($\lambda$) and Q($\lambda$) often conflates the three levels:
- Eligibility traces are presented as if they *are* the algorithm, obscuring that they're an implementation optimization
- The forward/backward view equivalence is emphasized, but this is purely a Level 3 computational identity
- The role of $\lambda$ in improving projected operator contraction (Level 2 concern) is rarely made explicit

By separating the levels, we gain:
- **Conceptual clarity**: What problem are we solving vs. how are we solving it vs. how do we implement it?
- **Modularity**: Can swap implementations at each level independently
- **Research directions**: Can separately study operator properties (Level 1), contraction theory (Level 2), and efficient estimation (Level 3)

This perspective also extends naturally to modern function approximation (neural networks, kernels) and connects to classical numerical analysis (operator preconditioning, resolvent theory), avoiding the folklore narrative based on psychological intuitions about memory traces.

