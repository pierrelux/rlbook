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
# Policy Gradients and Actor-Critic

The previous chapters optimized policies through Q-functions, soft Bellman
relations, or path-consistency equations. Can expected return itself supply the
policy objective without differentiating the environment dynamics? The score
function estimator applies directly to trajectories because the policy terms
are the only factors in their probability law that depend on the policy
parameters.

The estimator requires only the ability to evaluate and differentiate $\log
\pi_{\boldsymbol{w}}(a|s)$, so it also works with discrete actions where
reparameterization is unavailable.

Let $G(\tau) \equiv \sum_{t=0}^T r(s_t, a_t)$ be the sum of undiscounted rewards in a trajectory $\tau$. The stochastic optimization problem we face is to maximize:

$$
J(\boldsymbol{w}) = \mathbb{E}_{\tau \sim p(\tau;\boldsymbol{w})}[G(\tau)]
$$

where $\tau = (s_0,a_0,s_1,a_1,...)$ is a trajectory and $G(\tau)$ is the total return. 
Applying the score function estimator, we get:

$$
\begin{align*}
\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) &= \nabla_{\boldsymbol{w}}\mathbb{E}_{\tau}[G(\tau)] \\
&= \mathbb{E}_{\tau}\left[G(\tau)\nabla_{\boldsymbol{w}}\log p(\tau;\boldsymbol{w})\right] \\
&= \mathbb{E}_{\tau}\left[G(\tau)\nabla_{\boldsymbol{w}}\sum_{t=0}^T\log \pi_{\boldsymbol{w}}(a_t|s_t)\right] \\
&= \mathbb{E}_{\tau}\left[G(\tau)\sum_{t=0}^T\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\right]
\end{align*}
$$

We have eliminated the need to know the transition probabilities in this estimator since the probability of a trajectory factorizes as:

$$
p(\tau;\boldsymbol{w}) = p(s_0)\prod_{t=0}^T \pi_{\boldsymbol{w}}(a_t|s_t)p(s_{t+1}|s_t,a_t)
$$

Therefore, only the policy depends on $\boldsymbol{w}$. When taking the logarithm of this product, we get a sum where all the $\boldsymbol{w}$-independent terms vanish. The final estimator samples trajectories under the distribution $p(\tau; \boldsymbol{w})$ and computes:

$$
\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) \approx \frac{1}{N}\sum_{i=1}^N\left[G(\tau^{(i)})\sum_{t=0}^T\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t^{(i)}|s_t^{(i)})\right]
$$

This is a direct application of the score function estimator. However, we rarely use this form in practice and instead make several improvements to further reduce the variance.

### Leveraging Conditional Independence 

Given the Markov property of the MDP, rewards $r_k$ for $k < t$ are conditionally independent of action $a_t$ given the history $h_t = (s_0,a_0,...,s_{t-1},a_{t-1},s_t)$. This allows us to only need to consider future rewards when computing policy gradients.

$$
\begin{align*}
\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) &= \mathbb{E}_{\tau}\left[\sum_{t=0}^T\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\sum_{k=0}^T r_k\right] \\
&= \mathbb{E}_{\tau}\left[\sum_{t=0}^T\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\left(\sum_{k=0}^{t-1} r_k + \sum_{k=t}^T r_k\right)\right] \\
&= \mathbb{E}_{\tau}\left[\sum_{t=0}^T\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\sum_{k=t}^T r_k\right]
\end{align*}
$$

The conditional independence assumption means that the term $\mathbb{E}_{\tau}\left[\sum_{t=0}^T\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\sum_{k=0}^{t-1} r_k \right]$ vanishes. To see this, factor the trajectory distribution as:

$$
p(\tau) = p(s_0,...,s_t,a_0,...,a_{t-1}) \pi_{\boldsymbol{w}}(a_t|s_t) p(s_{t+1},...,s_T,a_{t+1},...,a_T|s_t,a_t)
$$

We can now re-write a single term of this summation as: 

$$
\mathbb{E}_{\tau}\left[\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\sum_{k=0}^{t-1} r_k\right] = \mathbb{E}_{s_{0:t},a_{0:t-1}}\left[\sum_{k=0}^{t-1} r_k \, \mathbb{E}_{a_t}\left[\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\right]\right]
$$

The inner expectation is zero because 

$$
\begin{align*}
\mathbb{E}_{a_t}\left[\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\right] &= \int \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\pi_{\boldsymbol{w}}(a_t|s_t)da_t \\
&= \int \frac{\nabla_{\boldsymbol{w}}\pi_{\boldsymbol{w}}(a_t|s_t)}{\pi_{\boldsymbol{w}}(a_t|s_t)}\pi_{\boldsymbol{w}}(a_t|s_t)da_t \\
&= \int \nabla_{\boldsymbol{w}}\pi_{\boldsymbol{w}}(a_t|s_t)da_t \\
&= \nabla_{\boldsymbol{w}}\int \pi_{\boldsymbol{w}}(a_t|s_t)da_t \\
&= \nabla_{\boldsymbol{w}}1 = 0
\end{align*}
$$ (eq:score-function-zero-mean)

The Monte Carlo estimator becomes:

$$
\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) \approx \frac{1}{N}\sum_{i=1}^N\left[\sum_{t=0}^T\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t^{(i)}|s_t^{(i)})\sum_{k=t}^T r_k^{(i)}\right]
$$

This gives us the REINFORCE algorithm:

```{prf:algorithm} REINFORCE
:label: reinforce

**Input:** Policy parameterization $\pi_{\boldsymbol{w}}(a|s)$  
**Output:** Updated policy parameters $\boldsymbol{w}$  
**Hyperparameters:** Learning rate $\alpha$, number of episodes $N$, episode length $T$

1. Initialize parameters $\boldsymbol{w}$
2. For episode = 1, ..., $N$ do:
   1. Collect trajectory $\tau = (s_0, a_0, r_0, ..., s_T, a_T, r_T)$ using policy $\pi_{\boldsymbol{w}}(a|s)$
   2. Compute returns: $G_t = \sum_{k=t}^T r_k$ for $t = 0, ..., T$
   3. Compute gradient estimate: $\hat{g} = \sum_{t=0}^T \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t) G_t$
   4. Update policy: $\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha \hat{g}$
3. Return $\boldsymbol{w}$
```

The benefit of this estimator compared to the naive one (which would weight each score function by the full trajectory return $G(\tau)$) is that it generally has less variance. This variance reduction arises from the conditional independence structure we exploited: past rewards do not depend on future actions. More formally, this estimator is an instance of a variance reduction technique known as the Extended Conditional Monte Carlo Method.

#### The Surrogate Loss Perspective

The algorithm above computes a gradient estimate $\hat{g}$ explicitly. In practice, implementations using automatic differentiation frameworks take a different approach: they define a **surrogate loss** whose gradient matches the REINFORCE estimator. For a single trajectory, consider:

$$
L_{\text{surrogate}}(\boldsymbol{w}) = -\sum_{t=0}^T \log \pi_{\boldsymbol{w}}(a_t|s_t) \, G_t
$$

where the returns $G_t$ and actions $a_t$ are treated as fixed constants (detached from the computation graph). Taking the gradient with respect to $\boldsymbol{w}$:

$$
\nabla_{\boldsymbol{w}} L_{\text{surrogate}} = -\sum_{t=0}^T \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t) \, G_t
$$

Minimizing this surrogate loss via gradient descent yields the same update as maximizing expected return via REINFORCE. The negative sign converts our maximization problem into a minimization suitable for standard optimizers.

This surrogate loss is **not** the expected return $J(\boldsymbol{w})$ we are trying to maximize. It is a computational device that produces the correct gradient at the current parameter values. Several properties distinguish it from a true loss function:

1. **It changes each iteration.** The returns $G_t$ come from trajectories sampled under the current policy. After updating $\boldsymbol{w}$, we must collect new trajectories and construct a new surrogate loss.

2. **Its value is not meaningful.** Unlike supervised learning where the loss measures prediction error, the numerical value of $L_{\text{surrogate}}$ has no direct interpretation. Only its gradient matters.

3. **It is valid only locally.** The surrogate loss provides the correct gradient only at the parameters used to collect the data. Moving far from those parameters invalidates the gradient estimate.

This perspective explains why policy gradient code often looks different from the pseudocode above. Instead of computing $\hat{g}$ explicitly, implementations define the surrogate loss and call `loss.backward()`:

```python
# Surrogate loss implementation (single trajectory)
log_probs = [policy.log_prob(a_t, s_t) for s_t, a_t in trajectory]
returns = compute_returns(rewards)
surrogate_loss = -sum(lp * G for lp, G in zip(log_probs, returns))
surrogate_loss.backward()  # computes REINFORCE gradient
optimizer.step()
```

### Variance Reduction via Control Variates

Recall that the REINFORCE gradient estimator, after leveraging conditional independence, takes the form:

$$
\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) \approx \frac{1}{N}\sum_{i=1}^N\left[\sum_{t=0}^T\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t^{(i)}|s_t^{(i)})\sum_{k=t}^T r_k^{(i)}\right]
$$

This is a sum over trajectories and timesteps. The gradient contribution at timestep $t$ of trajectory $i$ is:

$$
\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t^{(i)}|s_t^{(i)})\sum_{k=t}^T r_k^{(i)}
$$

While unbiased, this estimator suffers from high variance because the return $\sum_{k=t}^T r_k$ can vary significantly across trajectories even for the same state-action pair. The control variate method provides a principled way to reduce this variance.

#### General Control Variate Theory

For a general estimator $Z$ of some quantity $\mu = \mathbb{E}[Z]$, and a control variate $C$ with known expectation $\mathbb{E}[C]=0$, we can construct:

$$
Z_{\text{cv}} = Z - \alpha C
$$

This remains unbiased since $\mathbb{E}[Z_{\text{cv}}] = \mathbb{E}[Z] - \alpha\mathbb{E}[C] = \mathbb{E}[Z]$. The variance is:

$$
\text{Var}(Z_{\text{cv}}) = \text{Var}(Z) + \alpha^2\text{Var}(C) - 2\alpha\text{Cov}(Z,C)
$$

The $-2\alpha\text{Cov}(Z,C)$ term is what enables variance reduction. If $Z$ and $C$ are positively correlated, we can choose $\alpha > 0$ to make this term negative and large in magnitude, reducing the overall variance. However, the $\alpha^2\text{Var}(C)$ term grows quadratically with $\alpha$, so if we make $\alpha$ too large, this quadratic term will eventually dominate and the variance will increase rather than decrease. The variance as a function of $\alpha$ is a parabola opening upward, with a unique minimum. Setting $\frac{d}{d\alpha}\text{Var}(Z_{\text{cv}}) = 0$ gives:

$$
\alpha^* = \frac{\text{Cov}(Z,C)}{\text{Var}(C)}
$$

This is the coefficient from ordinary least squares regression: we predict the estimator $Z$ using the control variate $C$ as the predictor. Since $\mathbb{E}[C] = 0$, the linear model is $Z \approx \mathbb{E}[Z] + \alpha^* C$, where $\alpha^*$ is the OLS slope coefficient. The control variate estimator $Z_{\text{cv}} = Z - \alpha^* C$ computes the residual: the part of $Z$ that cannot be explained by $C$. 

Substituting $\alpha^*$ into the variance formula yields:

$$
\text{Var}(Z_{\text{cv}}) = \text{Var}(Z) - \frac{[\text{Cov}(Z,C)]^2}{\text{Var}(C)} = (1 - R^2) \text{Var}(Z)
$$

where $R^2 = \frac{[\text{Cov}(Z,C)]^2}{\text{Var}(Z)\text{Var}(C)}$ is the coefficient of determination from regressing $Z$ on $C$. The variance reduction is $R^2 \text{Var}(Z)$: the better $C$ predicts $Z$, the more variance we eliminate.

#### Application to REINFORCE

In the reinforcement learning setting, our REINFORCE gradient estimator is a sum over timesteps: $\sum_{t=0}^T Z_t$ where each $Z_t$ represents the gradient contribution at timestep $t$. We apply control variates separately to each term. Since $\text{Var}(\sum_t Z_t) = \sum_t \text{Var}(Z_t) + \sum_{t \neq s} \text{Cov}(Z_t, Z_s)$, reducing the variance of each $Z_t$ reduces the total variance, though we do not explicitly address the cross-timestep covariance terms.

For a given trajectory at state $s_t$, the gradient contribution at time $t$ is:

$$
Z_t = \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\sum_{k=t}^T r_k
$$

This is the product of the score function $\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)$ and the return-to-go $\sum_{k=t}^T r_k$. We can subtract any state-dependent function $b(s_t)$ from the return without introducing bias, as long as $b(s_t)$ does not depend on $a_t$. This is because:

$$
\mathbb{E}_{a_t \sim \pi_{\boldsymbol{w}}(\cdot|s_t)}\left[\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)b(s_t)\right] = b(s_t)\mathbb{E}_{a_t}\left[\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\right] = 0
$$

where the last equality follows from the score function identity {eq}`eq:score-function-zero-mean`.

We can now define our control variate as:

$$
C_t = \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t) \cdot b(s_t)
$$

where $b(s_t)$ is a **baseline function** that depends only on the state. This satisfies $\mathbb{E}[C_t|s_t] = 0$. Our control variate estimator becomes:

$$
Z_{t,\text{cv}} = Z_t - C_t = \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\left(\sum_{k=t}^T r_k - b(s_t)\right)
$$

The optimal baseline $b^*(s_t)$ minimizes the variance. To find it, consider the scalar parameter case for simplicity. Write $g(a_t) \equiv \nabla_w \log \pi_w(a_t|s_t)$ and $G_t = \sum_{k=t}^T r_k$. We want to minimize:

$$
b^*(s_t) = \arg\min_{b} \text{Var}_{a_t \sim \pi_{\boldsymbol{w}}(\cdot|s_t)}\left[g(a_t)(G_t - b)\right]
$$

Since the mean does not depend on $b$, minimizing the variance is equivalent to minimizing the second moment $\mathbb{E}[g(a_t)^2(G_t - b)^2|s_t]$. Expanding and taking the derivative with respect to $b$ gives:

$$
b^*(s_t) = \frac{\mathbb{E}_{a_t|s_t}\left[g(a_t)^2 G_t\right]}{\mathbb{E}_{a_t|s_t}\left[g(a_t)^2\right]}
$$

For vector-valued parameters $\boldsymbol{w}$, we minimize a scalar proxy such as the trace of the covariance matrix, which yields the same formula with $\|\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\|^2$ in place of $g(a_t)^2$:

$$
b^*(s_t) = \frac{\mathbb{E}_{a_t|s_t}\left[\|\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\|^2 G_t\right]}{\mathbb{E}_{a_t|s_t}\left[\|\nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\|^2\right]}
$$

This is the exact optimal baseline: a weighted average of returns where the weights are the squared norms of the score function. In practice, we treat the squared norm as roughly constant across actions at a given state, which leads to the simpler and widely used choice:

$$
b(s_t) \approx \mathbb{E}[G_t|s_t] = v^{\pi_{\boldsymbol{w}}}(s_t)
$$

With this approximation, the variance-reduced gradient contribution at timestep $t$ becomes:

$$
Z_{\text{cv},t} = \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\left(\sum_{k=t}^T r_k - v^{\pi_{\boldsymbol{w}}}(s_t)\right)
$$

The term in parentheses is exactly the **advantage function**: $A^{\pi_{\boldsymbol{w}}}(s_t, a_t) = q^{\pi_{\boldsymbol{w}}}(s_t, a_t) - v^{\pi_{\boldsymbol{w}}}(s_t)$, where the Q-function is approximated by the Monte Carlo return $\sum_{k=t}^T r_k$. The full gradient estimate for a trajectory is then the sum over all timesteps:

$$
\hat{g} = \sum_{t=0}^T Z_{\text{cv},t} = \sum_{t=0}^T \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)\left(G_t - v^{\pi_{\boldsymbol{w}}}(s_t)\right)
$$


In practice, we do not have access to the true value function and must learn it. Unlike the methods in the [amortization chapter](amortized-action-optimization.md), where we learned value functions to approximate the *optimal* Q-function, here our goal is **policy evaluation**: estimating the value of the *current* policy $\pi_{\boldsymbol{w}}$. The same function approximation techniques apply, but we target $v^{\pi_{\boldsymbol{w}}}$ rather than $v^*$. The simplest approach is to regress from states to Monte Carlo returns, learning what {cite:t}`williams1992reinforce` called a "baseline": 

```{prf:algorithm} Policy Gradient with Simple Baseline
:label: policy-grad-baseline

**Input:** Policy parameterization $\pi_{\boldsymbol{w}}(a|s)$, baseline function $b(s;\boldsymbol{\theta})$  
**Output:** Updated policy parameters $\boldsymbol{w}$  
**Hyperparameters:** Learning rates $\alpha_w$, $\alpha_\theta$, number of episodes $N$, episode length $T$

1. Initialize parameters $\boldsymbol{w}$, $\boldsymbol{\theta}$
2. For episode = 1, ..., $N$ do:
   1. Collect trajectory $\tau = (s_0, a_0, r_0, ..., s_T, a_T, r_T)$ using policy $\pi_{\boldsymbol{w}}(a|s)$
   2. Compute returns: $G_t = \sum_{k=t}^T r_k$ for $t = 0, ..., T$
   3. Update baseline: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha_\theta \nabla_{\boldsymbol{\theta}}\sum_{t=0}^T (G_t - b(s_t;\boldsymbol{\theta}))^2$
   4. Compute gradient estimate: $\hat{g} = \sum_{t=0}^T \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t)(G_t - b(s_t;\boldsymbol{\theta}))$
   5. Update policy: $\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha_w \hat{g}$
3. Return $\boldsymbol{w}$
```

When implementing this algorithm nowadays, we always use mini-batching to make full use of our GPUs. Therefore, a more representative variant for this algorithm would be: 


```{prf:algorithm} Policy Gradient with Optimal Control Variate and Mini-batches
:label: policy-grad-cv-batch

**Input:** Policy parameterization $\pi_{\boldsymbol{w}}(a|s)$, value function $v(s;\boldsymbol{\theta})$  
**Output:** Updated policy parameters $\boldsymbol{w}$  
**Hyperparameters:** Learning rates $\alpha_w$, $\alpha_\theta$, number of iterations $N$, episode length $T$, batch size $B$, mini-batch size $M$

1. Initialize parameters $\boldsymbol{w}$, $\boldsymbol{\theta}$
2. For iteration = 1, ..., N:
    1. Initialize empty buffer $\mathcal{D}$
    2. For b = 1, ..., B:
        1. Collect trajectory $\tau_b = (s_0, a_0, r_0, ..., s_T, a_T, r_T)$ using policy $\pi_{\boldsymbol{w}}(a|s)$
        2. Compute returns: $G_t = \sum_{k=t}^T r_k$ for all t
        3. Store tuple $(s_t, a_t, G_t)_{t=0}^T$ in $\mathcal{D}$
    3. For value_epoch = 1, ..., K:
        1. Sample mini-batch $\mathcal{B}_v$ of size $M$ from $\mathcal{D}$
        2. Compute value loss: $L_v = \frac{1}{M}\sum_{(s,a,G) \in \mathcal{B}_v} (v(s;\boldsymbol{\theta}) - G)^2$
        3. Update value function: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha_\theta \nabla_{\boldsymbol{\theta}}L_v$
    4. Compute advantages: $A(s,a) = G - v(s;\boldsymbol{\theta})$ for all $(s,a,G) \in \mathcal{D}$
    5. Normalize advantages: $A \leftarrow \frac{A - \mu_A}{\sigma_A}$
    6. For policy_epoch = 1, ..., J:
        1. Sample mini-batch $\mathcal{B}_\pi$ of size $M$ from $\mathcal{D}$
        2. Compute policy loss: $L_\pi = -\frac{1}{M}\sum_{(s,a,A) \in \mathcal{B}_\pi} \log \pi_{\boldsymbol{w}}(a|s)A$
        3. Update policy: $\boldsymbol{w} \leftarrow \boldsymbol{w} - \alpha_w \nabla_{\boldsymbol{w}}L_\pi$
3. Return $\boldsymbol{w}$
```

The value function is trained by regressing states directly to their sampled Monte Carlo returns $G$. Advantage normalization (step 2.5) is not part of the optimal baseline derivation but improves optimization in practice and is standard in modern implementations.

### Generalized Advantage Estimation

The baseline construction gave us a gradient estimator of the form:

$$
\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) \approx \frac{1}{N}\sum_{i=1}^N \sum_{t=0}^T \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t^{(i)}|s_t^{(i)}) \left(G_t^{(i)} - v(s_t^{(i)})\right)
$$

where $G_t = \sum_{k=t}^T r_k$ is the Monte Carlo return from time $t$. For each visited state-action pair $(s_t, a_t)$, the term in parentheses

$$
\widehat{A}_t^{\text{MC}} = G_t - v(s_t)
$$

is a Monte Carlo estimate of the advantage $A^{\pi}(s_t, a_t) = q^{\pi}(s_t, a_t) - v^{\pi}(s_t)$. If the baseline equals the true value function, $v = v^{\pi}$, then $\mathbb{E}[\widehat{A}_t^{\text{MC}} | s_t, a_t] = A^{\pi}(s_t, a_t)$, so this estimator is unbiased.

However, as an estimator it has two limitations. First, it has high variance because $G_t$ depends on all future rewards. Second, it uses the value function only as a baseline, not as a predictor of long-term returns. We essentially discard the information in $v(s_{t+1}), v(s_{t+2}), \ldots$

GAE addresses these issues by constructing a family of estimators that interpolate between pure Monte Carlo and pure bootstrapping. A parameter $\lambda$ controls the bias-variance tradeoff.

#### Decomposing the Monte Carlo Advantage

Fix a value function $v(s)$ (not necessarily equal to $v^{\pi}$) and define the one-step **residual**:

$$
\delta_t = r_t + \gamma v(s_{t+1}) - v(s_t)
$$

Start from the Monte Carlo advantage and add and subtract $\gamma v(s_{t+1})$:

$$
\begin{align*}
G_t - v(s_t) &= r_t + \gamma G_{t+1} - v(s_t) \\
&= r_t + \gamma v(s_{t+1}) - v(s_t) + \gamma(G_{t+1} - v(s_{t+1})) \\
&= \delta_t + \gamma(G_{t+1} - v(s_{t+1}))
\end{align*}
$$

Applying this decomposition recursively yields:

$$
G_t - v(s_t) = \sum_{l=0}^{T-t} \gamma^l \delta_{t+l}
$$

The Monte Carlo advantage is exactly the discounted sum of future residuals. This is an algebraic identity, not an approximation.

The sequence $\{\delta_{t+l}\}_{l \geq 0}$ provides incremental corrections to the value function as we move forward in time. The term $\delta_t$ depends only on $(s_t, a_t, s_{t+1})$; $\delta_{t+1}$ depends on $(s_{t+1}, a_{t+1}, s_{t+2})$, and so on. As $l$ increases, the corrections become more noisy (they depend on more random outcomes) and more sensitive to errors in the value function at later states. Although the full sum is unbiased when $v = v^{\pi}$, it can have high variance and can be badly affected by approximation error in $v$.

#### GAE as a Shrinkage Estimator

The decomposition above suggests a family of estimators that downweight residuals farther in the future. Let $\lambda \in [0,1]$ and define:

$$
A_t^{\lambda} = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}
$$ (eq:gae-definition)

This is the **generalized advantage estimator** $A_t^{\text{GAE}(\gamma,\lambda)}$.

Two special cases illustrate the extremes. When $\lambda = 1$, we recover the Monte Carlo advantage:

$$
A_t^{\lambda=1} = \sum_{l=0}^{T-t} \gamma^l \delta_{t+l} = G_t - v(s_t)
$$

When $\lambda = 0$, we keep only the immediate residual:

$$
A_t^{\lambda=0} = \delta_t = r_t + \gamma v(s_{t+1}) - v(s_t)
$$

Intermediate values $0 < \lambda < 1$ interpolate between these extremes. The influence of $\delta_{t+l}$ decays geometrically as $(\gamma\lambda)^l$. The parameter $\lambda$ acts as a shrinkage parameter: small $\lambda$ shrinks the estimator toward the one-step residual; large $\lambda$ allows the estimator to behave more like the Monte Carlo advantage.

If $v = v^{\pi}$ is the true value function, then $\mathbb{E}[\delta_t | s_t, a_t] = A^{\pi}(s_t, a_t)$ and $\mathbb{E}[\delta_{t+l} | s_t, a_t] = 0$ for $l \geq 1$. In this case:

$$
\mathbb{E}[A_t^{\lambda} | s_t, a_t] = \sum_{l=0}^{T-t} (\gamma\lambda)^l \mathbb{E}[\delta_{t+l} | s_t, a_t] = A^{\pi}(s_t, a_t)
$$

for all $\lambda \in [0,1]$. When the value function is exact, GAE is unbiased regardless of $\lambda$; changing $\lambda$ only affects variance.

In practice, we approximate $v^{\pi}$ with a function approximator, and the residuals $\delta_{t+l}$ inherit approximation error. Distant residuals involve multiple applications of the approximate value function and are more contaminated by modeling error. Downweighting them (choosing $\lambda < 1$) introduces bias but can reduce variance and limit the impact of those errors.

#### Mixture of Multi-Step Estimators

Another perspective on GAE comes from multi-step returns. Define the $k$-step return from time $t$:

$$
G_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l r_{t+l} + \gamma^k v(s_{t+k})
$$

and the corresponding $k$-step advantage estimator $A_t^{(k)} = G_t^{(k)} - v(s_t)$. Each $A_t^{(k)}$ uses $k$ rewards before bootstrapping; larger $k$ means more variance but less bootstrapping error.

The GAE estimator can be written as a geometric mixture:

$$
A_t^{\lambda} = (1-\lambda) \sum_{k=1}^{T-t} \lambda^{k-1} A_t^{(k)}
$$

GAE is a weighted average of the $k$-step advantage estimators, with shorter horizons weighted more heavily when $\lambda$ is small.

#### Using GAE in the Policy Gradient

Once we choose $\lambda$, we plug $A_t^{\lambda}$ in place of $G_t - v(s_t)$ in the policy gradient estimator:

$$
\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) \approx \frac{1}{N}\sum_{i=1}^N \sum_{t=0}^T \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t^{(i)}|s_t^{(i)}) A_t^{\lambda,(i)}
$$

We still use a control variate to reduce variance (the baseline $v$), but now we construct the advantage target by smoothing the sequence of residuals $\{\delta_t\}$ with a geometrically decaying kernel.

For the value function, it is convenient to define the **$\lambda$-return**:

$$
G_t^{\lambda} = A_t^{\lambda} + v(s_t)
$$

When $\lambda = 1$, $G_t^{\lambda}$ reduces to the Monte Carlo return; when $\lambda = 0$, it becomes the one-step bootstrapped target $r_t + \gamma v(s_{t+1})$.

```{prf:algorithm} Policy Gradient with GAE and Mini-batches
:label: policy-grad-gae-batch

**Input:** Policy parameterization $\pi_{\boldsymbol{w}}(a|s)$, value function $v(s;\boldsymbol{\theta})$  
**Output:** Updated policy parameters $\boldsymbol{w}$  
**Hyperparameters:** Learning rates $\alpha_w$, $\alpha_\theta$, number of iterations $N$, episode length $T$, batch size $B$, mini-batch size $M$, discount $\gamma$, GAE parameter $\lambda$

1. Initialize parameters $\boldsymbol{w}$, $\boldsymbol{\theta}$
2. For iteration = 1, ..., $N$:
    1. Initialize empty buffer $\mathcal{D}$
    2. For $b = 1, ..., B$:
        1. Collect trajectory $\tau_b = (s_0, a_0, r_0, ..., s_T, a_T, r_T)$ using policy $\pi_{\boldsymbol{w}}(a|s)$
        2. Compute value predictions $v_t = v(s_t;\boldsymbol{\theta})$ for $t = 0, \ldots, T$ and set $v_{T+1} = 0$
        3. Compute residuals: $\delta_t = r_t + \gamma v_{t+1} - v_t$ for $t = 0, \ldots, T$
        4. Compute GAE advantages backwards:
            1. Set $A_T = \delta_T$
            2. For $t = T-1, ..., 0$: $A_t = \delta_t + (\gamma\lambda) A_{t+1}$
        5. Compute $\lambda$-returns: $G_t^{\lambda} = A_t + v_t$ for $t = 0, \ldots, T$
        6. Store tuples $(s_t, a_t, A_t, G_t^{\lambda})_{t=0}^T$ in $\mathcal{D}$
    3. For value_epoch = 1, ..., $K$:
        1. Sample mini-batch $\mathcal{B}_v$ of size $M$ from $\mathcal{D}$
        2. Compute value loss: $L_v = \frac{1}{M}\sum_{(s,\cdot,\cdot,G^{\lambda}) \in \mathcal{B}_v} (v(s;\boldsymbol{\theta}) - G^{\lambda})^2$
        3. Update value function: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha_\theta \nabla_{\boldsymbol{\theta}}L_v$
    4. Normalize advantages: $A \leftarrow \frac{A - \mu_A}{\sigma_A}$
    5. For policy_epoch = 1, ..., $J$:
        1. Sample mini-batch $\mathcal{B}_\pi$ of size $M$ from $\mathcal{D}$
        2. Compute policy loss: $L_\pi = -\frac{1}{M}\sum_{(s,a,A,\cdot) \in \mathcal{B}_\pi} \log \pi_{\boldsymbol{w}}(a|s) A$
        3. Update policy: $\boldsymbol{w} \leftarrow \boldsymbol{w} - \alpha_w \nabla_{\boldsymbol{w}}L_\pi$
3. Return $\boldsymbol{w}$
```

When $\lambda = 1$, this reduces (up to advantage normalization) to the Monte Carlo baseline algorithm earlier in the chapter. When $\lambda = 0$, advantages become the one-step residuals $\delta_t$, and the $\lambda$-returns reduce to standard one-step bootstrapped targets.

#### Actor-Critic as the $\lambda = 0$ Limit

The case $\lambda = 0$ is particularly simple. The advantage becomes:

$$
A_t^{\lambda=0} = \delta_t = r_t + \gamma v(s_{t+1}) - v(s_t)
$$

and the policy update reduces to:

$$
\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha_w \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t) \delta_t
$$

while the value update becomes a standard one-step regression toward $r_t + \gamma v(s_{t+1})$. This gives the online actor-critic algorithm:

```{prf:algorithm} Actor-Critic with One-Step Residuals
:label: actor-critic-td0

**Input:** Policy parameterization $\pi_{\boldsymbol{w}}(a|s)$, value function $v(s;\boldsymbol{\theta})$  
**Output:** Updated policy parameters $\boldsymbol{w}$  
**Hyperparameters:** Learning rates $\alpha_w$, $\alpha_\theta$, number of episodes $N$, episode length $T$, discount $\gamma$

1. Initialize parameters $\boldsymbol{w}$, $\boldsymbol{\theta}$
2. For episode = 1, ..., $N$ do:
   1. Initialize state $s_0$
   2. For $t = 0, ..., T$ do:
      1. Sample action: $a_t \sim \pi_{\boldsymbol{w}}(\cdot|s_t)$
      2. Execute $a_t$, observe $r_t$, $s_{t+1}$
      3. Compute residual: $\delta_t = r_t + \gamma v(s_{t+1};\boldsymbol{\theta}) - v(s_t;\boldsymbol{\theta})$
      4. Update value function: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha_\theta \delta_t \nabla_{\boldsymbol{\theta}}v(s_t;\boldsymbol{\theta})$
      5. Update policy: $\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha_w \nabla_{\boldsymbol{w}}\log \pi_{\boldsymbol{w}}(a_t|s_t) \delta_t$
3. Return $\boldsymbol{w}$
```

This algorithm was derived by Sutton in his 1984 thesis as an "adaptive heuristic" for temporal credit assignment. In the language of this chapter, it is the $\lambda = 0$ member of the GAE family: it uses the most local residual $\delta_t$ as both the target for the value function and the advantage estimate for the policy gradient.

## Likelihood Ratio Methods in Reinforcement Learning

How do importance ratios and constrained or clipped surrogates reuse
trajectories collected under an earlier policy?

The score function estimator from the previous section is a special case of the likelihood ratio method where the proposal distribution equals the target distribution. We now consider the general case where they differ.

Recall the likelihood ratio gradient estimator from the beginning of this chapter. For objective $J(\theta) = \mathbb{E}_{x \sim p(x;\theta)}[f(x)]$ and any proposal distribution $q(x)$:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{x \sim q(x)}\left[f(x) \rho(x; q, \theta) \nabla_\theta \log p(x;\theta)\right]
$$

where $\rho(x; q, \theta) = \frac{p(x;\theta)}{q(x)}$ is the likelihood ratio. The partial derivative $\frac{\partial \rho}{\partial \theta} = \rho \nabla_\theta \log p$ holds because $x$ is treated as fixed, having been sampled from $q$, which does not depend on $\theta$.

In reinforcement learning, let $x = \tau$ be a trajectory, $f(\tau) = G(\tau)$ the return, $p(\tau;\boldsymbol{w})$ the trajectory distribution under policy $\pi_{\boldsymbol{w}}$, and $q(\tau)$ the trajectory distribution under some other policy $\pi_q$. The gradient becomes:

$$
\nabla_{\boldsymbol{w}} J(\boldsymbol{w}) = \mathbb{E}_{\tau \sim \pi_q}\left[G(\tau) \rho(\tau) \sum_{t=0}^T \nabla_{\boldsymbol{w}} \log \pi_{\boldsymbol{w}}(a_t|s_t)\right]
$$

where the trajectory likelihood ratio simplifies because transition probabilities cancel:

$$
\rho(\tau) = \frac{p(\tau;\boldsymbol{w})}{q(\tau)} = \prod_{t=0}^T \frac{\pi_{\boldsymbol{w}}(a_t|s_t)}{\pi_q(a_t|s_t)} = \prod_{t=0}^T \rho_t
$$

This product of $T+1$ ratios can become extremely large or small as $T$ grows, leading to high variance. The temporal structure provides some relief: since $\mathbb{E}_{a \sim \pi_q}[\rho] = 1$, future ratios $\rho_{k}$ for $k > t$ that do not affect the reward $r_t$ can be marginalized out. However, past ratios $\rho_{0:t-1}$ are still needed to correctly weight the probability of reaching state $s_t$.

In practice, algorithms like PPO and TRPO make an additional approximation: they use only the per-step ratio $\rho_t$ rather than the cumulative product $\rho_{0:t}$. This ignores the mismatch between the state distributions induced by the two policies. Combined with a baseline $b(s_t)$, the approximate estimator is:

$$
\nabla_{\boldsymbol{w}} J(\boldsymbol{w}) \approx \mathbb{E}_{\tau \sim \pi_q}\left[\sum_{t=0}^T \rho_t \nabla_{\boldsymbol{w}} \log \pi_{\boldsymbol{w}}(a_t|s_t) (G_t - b(s_t))\right]
$$ (eq:lr-estimator)

This approximation corresponds to maximizing the **importance-weighted surrogate objective**:

$$
L^{\text{IS}}(\boldsymbol{w}) = \mathbb{E}_{\tau \sim \pi_q}\left[\sum_{t=0}^T \rho_t A_t\right]
$$ (eq:importance-weighted-surrogate)

where $A_t = G_t - b(s_t)$. Taking the gradient with respect to $\boldsymbol{w}$, only $\rho_t$ depends on $\boldsymbol{w}$ (since trajectories are sampled from $\pi_q$):

$$
\nabla_{\boldsymbol{w}} L^{\text{IS}}(\boldsymbol{w}) = \mathbb{E}_{\tau \sim \pi_q}\left[\sum_{t=0}^T A_t \nabla_{\boldsymbol{w}} \rho_t\right]
$$

The gradient of the ratio is:

$$
\nabla_{\boldsymbol{w}} \rho_t = \nabla_{\boldsymbol{w}} \frac{\pi_{\boldsymbol{w}}(a_t|s_t)}{\pi_q(a_t|s_t)} = \frac{\nabla_{\boldsymbol{w}} \pi_{\boldsymbol{w}}(a_t|s_t)}{\pi_q(a_t|s_t)} = \rho_t \nabla_{\boldsymbol{w}} \log \pi_{\boldsymbol{w}}(a_t|s_t)
$$

Substituting back:

$$
\nabla_{\boldsymbol{w}} L^{\text{IS}}(\boldsymbol{w}) = \mathbb{E}_{\tau \sim \pi_q}\left[\sum_{t=0}^T \rho_t \nabla_{\boldsymbol{w}} \log \pi_{\boldsymbol{w}}(a_t|s_t) A_t\right]
$$

This matches equation {eq}`eq:lr-estimator`. When $\pi_q = \pi_{\boldsymbol{w}}$, the ratios $\rho_t = 1$ and we recover the score function estimator. The approximation error grows as the policies diverge, which motivates the trust region and clipping mechanisms discussed below.

### Variance and the Dominance Condition

The ratio $\rho_t = \pi_{\boldsymbol{w}}(a_t|s_t)/\pi_q(a_t|s_t)$ is well-behaved only when the two policies are similar. If $\pi_{\boldsymbol{w}}$ assigns high probability to an action where $\pi_q$ assigns low probability, the ratio explodes. For example, if $\pi_q(a|s) = 0.01$ and $\pi_{\boldsymbol{w}}(a|s) = 0.5$, then $\rho = 50$, amplifying any noise in the advantage estimate.

Importance sampling also requires the **dominance condition**: the support of $\pi_{\boldsymbol{w}}$ must be contained in the support of $\pi_q$. If $\pi_{\boldsymbol{w}}(a|s) > 0$ but $\pi_q(a|s) = 0$, the ratio is undefined. Stochastic policies typically have full support, but the ratio can still become arbitrarily large as $\pi_q(a|s) \to 0$.

A common use case is to set $\pi_q = \pi_{\boldsymbol{w}_{\text{old}}}$, a previous version of the policy. This allows reusing data across multiple gradient steps: collect trajectories once, then update $\boldsymbol{w}$ several times. But each update moves $\boldsymbol{w}$ further from $\boldsymbol{w}_{\text{old}}$, making the ratios more extreme. Eventually, the gradient signal is dominated by a few samples with large weights.

### Proximal Policy Optimization

The variance issues suggest a natural solution: keep the ratio $\rho_t$ close to 1 by ensuring the new policy stays close to the behavior policy. This keeps the importance-weighted surrogate $L^{\text{IS}}(\boldsymbol{w})$ from {eq}`eq:importance-weighted-surrogate` well-behaved.

Trust Region Policy Optimization (TRPO) formalizes this by adding a constraint on the KL divergence between the old and new policies:

$$
\max_{\boldsymbol{w}} L^{\text{IS}}(\boldsymbol{w}) \quad \text{subject to} \quad \mathbb{E}_s\left[D_{\text{KL}}(\pi_{\boldsymbol{w}_{\text{old}}}(\cdot|s) \| \pi_{\boldsymbol{w}}(\cdot|s))\right] \leq \delta
$$

The KL constraint ensures that the two distributions remain similar, which bounds how extreme the importance weights can become. This is a constrained optimization problem, and one could in principle apply standard methods such as projected gradient descent or augmented Lagrangian approaches (as discussed in the [trajectory optimization chapter](discrete-time-optimal-control.md)). TRPO takes a different approach: it uses a second-order Taylor approximation of the KL constraint around the current parameters and solves the resulting trust region subproblem using conjugate gradient methods. This involves computing the Fisher information matrix (the Hessian of the KL divergence), which adds computational overhead.

Proximal Policy Optimization (PPO) achieves similar behavior through a simpler mechanism: rather than constraining the distributions to be similar, it directly clips the ratio $\rho_t$ to prevent it from moving too far from 1. This is a construction-level guarantee rather than an optimization-level constraint.

#### From Trajectory Expectations to State-Action Averages

Before defining the PPO objective, we need to clarify the relationship between the trajectory-level surrogate {eq}`eq:importance-weighted-surrogate` and the state-action level objective that PPO actually optimizes. The importance-weighted surrogate is defined as an expectation over trajectories:

$$
L^{\text{IS}}(\boldsymbol{w}) = \mathbb{E}_{\tau \sim \pi_q}\left[\sum_{t=0}^T \rho_t A_t\right]
$$

We can rewrite this as an expectation over state-action pairs by introducing a sampling distribution. For a finite horizon $T$, define the **averaged time-marginal distribution**:

$$
\xi_{\pi_q}(s, a) = \frac{1}{T+1} \sum_{t=0}^{T} d_t^{\pi_q}(s) \pi_q(a|s)
$$ (eq:time-marginal)

where $d_t^{\pi_q}(s)$ is the probability of being in state $s$ at time $t$ when following policy $\pi_q$ from the initial distribution. This is the uniform mixture over the time-indexed state-action distributions: we pick a timestep $t$ uniformly at random from $\{0, 1, \ldots, T\}$, then sample $(s, a)$ from the joint distribution at that timestep.

With this definition, the trajectory expectation becomes:

$$
\mathbb{E}_{\tau \sim \pi_q}\left[\sum_{t=0}^T \rho_t A_t\right] = (T+1) \cdot \mathbb{E}_{(s,a) \sim \xi_{\pi_q}}\left[\rho(s, a) A(s, a)\right]
$$

The factor $(T+1)$ is just a constant that does not affect the optimization. This reformulation shows that the importance-weighted surrogate is equivalent to an expectation over state-action pairs drawn from the averaged time-marginal distribution. This is not a stationary distribution or a discounted visitation distribution, but the empirical mixture induced by the finite-horizon rollout procedure.

#### The Clipped Surrogate Objective

PPO replaces the linear importance-weighted term $\rho A$ with a clipped version. For a state-action pair $(s, a)$ with advantage $A$ and importance ratio $\rho(\boldsymbol{w}) = \pi_{\boldsymbol{w}}(a|s) / \pi_{\boldsymbol{w}_{\text{old}}}(a|s)$, define the **per-sample clipped objective**:

$$
\ell^{\text{CLIP}}(\boldsymbol{w}; s, a, A) = \min\left(\rho(\boldsymbol{w}) A, \, \text{clip}(\rho(\boldsymbol{w}), 1-\epsilon, 1+\epsilon) A\right)
$$ (eq:ppo-per-sample)

where $\epsilon$ is a hyperparameter (typically 0.1 or 0.2) and $\text{clip}(x, a, b) = \max(a, \min(x, b))$ restricts $x$ to the interval $[a, b]$.

The **population-level PPO objective** is then:

$$
L^{\text{CLIP}}(\boldsymbol{w}) = \mathbb{E}_{(s,a,A) \sim \xi_{\pi_{\boldsymbol{w}_{\text{old}}}}}\left[\ell^{\text{CLIP}}(\boldsymbol{w}; s, a, A)\right]
$$ (eq:ppo-clip-population)

where the expectation is taken over the averaged time-marginal distribution {eq}`eq:time-marginal` induced by $\pi_{\boldsymbol{w}_{\text{old}}}$.

In practice, we never compute this expectation exactly. Instead, we collect a batch of transitions $\mathcal{D} = \{(s_t^{(i)}, a_t^{(i)}, A_t^{(i)})\}$ by running $\pi_{\boldsymbol{w}_{\text{old}}}$ and approximate the expectation with an empirical average:

$$
\hat{L}^{\text{CLIP}}(\boldsymbol{w}; \mathcal{D}) = \frac{1}{|\mathcal{D}|} \sum_{(s,a,A) \in \mathcal{D}} \ell^{\text{CLIP}}(\boldsymbol{w}; s, a, A)
$$ (eq:ppo-clip)

This is the same plug-in approximation used in [fitted Q-iteration](fitted-q-iteration.md): replace the unknown population distribution with the empirical distribution $\hat{P}_{\mathcal{D}}$ induced by the collected batch, then compute the sample average. The empirical surrogate $\hat{L}^{\text{CLIP}}$ is simply an expectation under $\hat{P}_{\mathcal{D}}$. No assumptions about stationarity or discounted visitation are needed. We just average over the transitions we collected.

#### Intuition for the Clipping Mechanism

The $\min$ operator in {eq}`eq:ppo-per-sample` selects the more pessimistic estimate. Consider the two cases:

- **Positive advantage** ($A > 0$): The action is better than average, so we want to increase $\pi_{\boldsymbol{w}}(a|s)$. The unclipped term $\rho A$ increases with $\rho$. The clipped term stops increasing once $\rho > 1 + \epsilon$. Taking the minimum means we get the benefit of increasing $\rho$ only up to $1 + \epsilon$.

- **Negative advantage** ($A < 0$): The action is worse than average, so we want to decrease $\pi_{\boldsymbol{w}}(a|s)$. The unclipped term $\rho A$ becomes less negative (improves) as $\rho$ decreases. The clipped term stops improving once $\rho < 1 - \epsilon$. Taking the minimum means we get the benefit of decreasing $\rho$ only down to $1 - \epsilon$.

In both cases, the clipping removes the incentive to move the probability ratio beyond the interval $[1-\epsilon, 1+\epsilon]$. This keeps the new policy close to the old policy without explicitly computing or constraining the KL divergence.

```{prf:algorithm} Proximal Policy Optimization (PPO-Clip)
:label: ppo

**Input:** Policy $\pi_{\boldsymbol{w}}(a|s)$, value function $v(s;\boldsymbol{\theta})$  
**Output:** Updated parameters $\boldsymbol{w}$  
**Hyperparameters:** Clip parameter $\epsilon$, learning rates $\alpha_w$, $\alpha_\theta$, number of iterations $N$, batch size $B$, mini-batch size $M$, epochs per iteration $K$, GAE parameters $\gamma$, $\lambda$

1. Initialize parameters $\boldsymbol{w}$, $\boldsymbol{\theta}$
2. For iteration = 1, ..., $N$:
    1. Collect batch $\mathcal{D}$ of $B$ trajectories using policy $\pi_{\boldsymbol{w}}$
    2. Compute GAE advantages $A_t$ and $\lambda$-returns $G_t^{\lambda}$ for all timesteps
    3. Store old log-probabilities: $\log \pi_{\text{old}}(a_t|s_t) = \log \pi_{\boldsymbol{w}}(a_t|s_t)$ for all $(s_t, a_t)$
    4. Normalize advantages: $A \leftarrow \frac{A - \mu_A}{\sigma_A}$
    5. For epoch = 1, ..., $K$:
        1. Shuffle $\mathcal{D}$ and partition into mini-batches of size $M$
        2. For each mini-batch $\mathcal{B}$:
            1. Compute ratios: $\rho = \exp(\log \pi_{\boldsymbol{w}}(a|s) - \log \pi_{\text{old}}(a|s))$ for all $(s, a) \in \mathcal{B}$
            2. Compute per-sample objectives: $\ell^{\text{CLIP}} = \min(\rho A, \text{clip}(\rho, 1-\epsilon, 1+\epsilon) A)$
            3. Compute empirical surrogate: $\hat{L}^{\text{CLIP}} = \frac{1}{|\mathcal{B}|}\sum_{(s,a,A) \in \mathcal{B}} \ell^{\text{CLIP}}$
            4. Compute value loss: $L_v = \frac{1}{|\mathcal{B}|}\sum_{(s,G^{\lambda}) \in \mathcal{B}} (v(s;\boldsymbol{\theta}) - G^{\lambda})^2$
            5. Update policy: $\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha_w \nabla_{\boldsymbol{w}} \hat{L}^{\text{CLIP}}$
            6. Update value function: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha_\theta \nabla_{\boldsymbol{\theta}} L_v$
3. Return $\boldsymbol{w}$
```

The algorithm collects a batch of trajectories, then performs $K$ epochs of mini-batch updates on the same data. The empirical surrogate $\hat{L}^{\text{CLIP}}$ approximates the population objective {eq}`eq:ppo-clip-population` using samples from the averaged time-marginal distribution. The ratio $\rho$ is computed in log-space for numerical stability. Clipping removes the local incentive to move a sampled probability ratio beyond the chosen interval. It does not guarantee that the sampled updates will find a successful policy, and it says nothing about physical constraints absent from the reward or environment.

### Experiment: PPO on the SwingRL Plant

The SwingRL model from the [modeling chapter](modeling-controlled-systems.md#internal-actuation-in-swingrl)
provides a matched comparison between supplied structure and sampled policy
optimization. Both controllers act on the same articulated standing rider, use
the same two bounded commands, receive the same observations, and count one
full unwrapped rotation as success. The structured controller receives a
phase-locked pumping rule whose phases were selected by a model sweep. PPO
receives trajectories and the environment reward.

The experiment asks whether the clipped policy-gradient update discovers a
full rotation under one fixed, reproducible protocol. The protocol was chosen
before inspecting the five final runs.

| quantity | value |
|---|---:|
| policy and value networks | separate $2 \times 64$ tanh networks |
| requested interactions | 1,000,000 per seed |
| rollout batch | 8 environments $\times$ 256 steps |
| PPO epochs and minibatch | 4 epochs, 256 samples |
| optimizer | Adam, learning rate $3\times10^{-4}$ with linear decay |
| discount and GAE | $\gamma=0.995$, $\lambda=0.95$ |
| clipping and entropy coefficients | $0.2$ and $0.001$ |
| seeds | 0, 1, 2, 3, 4 |
| evaluation | 100 fixed initial states every 50,000 requested interactions |

Each complete rollout batch contains 2,048 transitions, so the last update is
recorded at 1,001,472 interactions. A checkpoint is the first complete update
at or beyond its nominal 50,000-interaction target. Evaluation angles are
uniform between $-5^\circ$ and $5^\circ$, angular velocities are uniform
between $-0.1$ and $0.1$ rad/s, and the deterministic policy uses the tanh of
the Gaussian mean. The environment and the structured baseline use the same
100 states.

```{code-cell} python
:tags: [remove-input]
:label: fig-swing-ppo-replay
:caption: Recorded training replay for seed 0. Checkpoints occur every 50,000 requested interactions, with complete rollout batches producing the recorded counts shown by the slider. Each checkpoint uses the saved deterministic policy from the same $3^\circ$ initial angle. The curve shows raw completed-episode returns and an eight-point trailing mean. The 21 saved policies are compressed into 63 seconds; the overlay reports the original elapsed training time.

import sys
from pathlib import Path
from IPython.display import HTML, display

sys.path.insert(0, "code")
from swing_ppo import replay_player_html

display(HTML(replay_player_html(Path("_static/swing_ppo"))))
```

The replay is an illustration from one prespecified run, not the evidential
comparison. Future observations remain hidden until the movie reaches their
checkpoint. The five-seed result below carries the comparison across runs.

{download}`Download the recorded seed-0 replay <_static/swing_ppo/training_replay.mp4>`

```{figure} _static/swing_ppo/swing_ppo_learning.png
:label: fig-swing-ppo-learning
:alt: Two plots compare five PPO seeds with a structured SwingRL controller. PPO remains at zero success while its mean return improves slightly. The structured controller succeeds from every evaluation state and has a much larger return.
:width: 100%

Five prespecified PPO seeds evaluated on the same 100 fixed initial states. Thin blue traces show individual seeds; the solid trace and band show the mean and a two-sided 95% $t$ interval with seed as the statistical unit. PPO improves its return early but never completes a rotation. The structured controller succeeds from every state and obtains mean return 191.8; this value is annotated off scale in the return panel so that the smaller PPO change remains visible.
```

The final policies settle on low-motion behavior. Across the five showcase
rollouts, the largest angle lies between $5.45^\circ$ and $5.62^\circ$. Return
improves because the policy changes the rider while keeping effort and time
penalties modest, but the behavior never approaches the $360^\circ$ success
condition.

| controller | training transitions | held-out success | mean return | worst suspension tension |
|---|---:|---:|---:|---:|
| structured phase controller | 0 | 100% | 191.79 | $-637$ N |
| PPO, mean over five final policies | 1,001,472 per seed | 0% | $-5.11$ | $+387$ N |

The comparison does not establish a general ranking between control and RL.
The structured controller receives the oscillation phase and a controller
family adapted to the mechanism, while PPO must infer useful coordination from
sampled returns. Conversely, the structured controller's nominal success
requires negative suspension tension during 7.37 percent of its active steps.
A rigid rod can supply that outward force; a playground chain cannot. PPO
avoids the violation by barely moving, which is feasible but does not solve the
task.

The two failures answer different audit questions. Optimization has not found
a successful sampled policy under this protocol. The successful structured
policy exposes an inadequate rigid-link model. More interactions or a revised
learning objective might address the first failure. A unilateral chain model
or an explicit tension constraint is required for the second.

:::{admonition} Model audit
:class: tip
Suppose the success bonus were increased until PPO learned to rotate the rigid
link. Which reported metric would still prevent the result from establishing a
successful playground-swing controller? Identify the missing physical mode and
describe an evaluation that would expose it.
:::

The [complete Python implementation](code/swing_ppo.py) includes policy
sampling, tanh-corrected log probabilities, GAE, PPO updates, batched SwingRL
evaluation, checkpoint serialization, and replay rendering. The
[experiment record](artifacts/swing_ppo/README.md) specifies the artifact
layout and reproduction command. Policy training is deliberately absent from
the MyST build; the page reads the recorded checkpoints, tables, figures, and
movie.

## The Policy Gradient Theorem

How can the trajectory-level score estimator be rewritten as an expectation
over discounted state visitation and action values?

The algorithms developed so far (REINFORCE, actor-critic, GAE, and PPO) all estimate policy gradients from sampled trajectories. We now establish the theoretical foundation for these estimators by deriving the policy gradient theorem in the discounted infinite-horizon setting.

{cite:t}`sutton1999policy` provided the original derivation. Here we present an alternative approach using the Implicit Function Theorem, which frames policy optimization as a bilevel problem:

$$
\max_{\mathbf{w}} \alpha^\top \mathbf{v}_\gamma^{\pi_{\boldsymbol{w}}}
$$

subject to:

$$
(\mathbf{I} - \gamma \mathbf{P}_{\pi_{\boldsymbol{w}}}) \mathbf{v}_\gamma^{\pi_{\boldsymbol{w}}} = \mathbf{r}_{\pi_{\boldsymbol{w}}}
$$

The Implicit Function Theorem states that if there is a solution to the problem $F(\mathbf{v}, \mathbf{w}) = 0$, then we can "reparameterize" our problem as $F(\mathbf{v}(\mathbf{w}), \mathbf{w})$ where $\mathbf{v}(\mathbf{w})$ is an implicit function of $\mathbf{w}$. If the Jacobian $\frac{\partial F}{\partial \mathbf{v}}$ is invertible, then:

$$
\frac{d\mathbf{v}(\mathbf{w})}{d\mathbf{w}} = -\left(\frac{\partial F(\mathbf{v}(\mathbf{w}), \mathbf{w})}{\partial \mathbf{v}}\right)^{-1}\frac{\partial F(\mathbf{v}(\mathbf{w}), \mathbf{w})}{\partial \mathbf{w}}
$$

Here we made it clear in our notation that the derivative must be evaluated at root $(\mathbf{v}(\mathbf{w}), \mathbf{w})$ of $F$. For the remaining of this derivation, we will drop this dependence to make notation more compact. 

Applying this to our case with $F(\mathbf{v}, \mathbf{w}) = (\mathbf{I} - \gamma \mathbf{P}_{\pi_{\boldsymbol{w}}})\mathbf{v} - \mathbf{r}_{\pi_{\boldsymbol{w}}}$:

$$
\frac{\partial \mathbf{v}_\gamma^{\pi_{\boldsymbol{w}}}}{\partial \mathbf{w}} = (\mathbf{I} - \gamma \mathbf{P}_{\pi_{\boldsymbol{w}}})^{-1}\left(\frac{\partial \mathbf{r}_{\pi_{\boldsymbol{w}}}}{\partial \mathbf{w}} + \gamma \frac{\partial \mathbf{P}_{\pi_{\boldsymbol{w}}}}{\partial \mathbf{w}}\mathbf{v}_\gamma^{\pi_{\boldsymbol{w}}}\right)
$$

Then:

$$
\begin{align*}
\nabla_{\mathbf{w}}J(\mathbf{w}) &= \alpha^\top \frac{\partial \mathbf{v}_\gamma^{\pi_{\boldsymbol{w}}}}{\partial \mathbf{w}} \\
&= \mathbf{x}_\alpha^\top\left(\frac{\partial \mathbf{r}_{\pi_{\boldsymbol{w}}}}{\partial \mathbf{w}} + \gamma \frac{\partial \mathbf{P}_{\pi_{\boldsymbol{w}}}}{\partial \mathbf{w}}\mathbf{v}_\gamma^{\pi_{\boldsymbol{w}}}\right)
\end{align*}
$$

where we have defined the discounted state visitation distribution:

$$
\mathbf{x}_\alpha^\top \equiv \alpha^\top(\mathbf{I} - \gamma \mathbf{P}_{\pi_{\boldsymbol{w}}})^{-1}.
$$

Recall the vector notation for MDPs from the [infinite-horizon MDP chapter](infinite-horizon-mdps.md):

$$
\begin{align*}
\mathbf{r}_\pi(s) &\equiv \sum_{a \in \mathcal{A}_s} \pi(a \mid s) \, r(s, a), \\
[\mathbf{P}_\pi]_{s,s'} &\equiv \sum_{a \in \mathcal{A}_s} \pi(a \mid s) \, p(s' \mid s, a).
\end{align*}
$$

Taking derivatives with respect to $\mathbf{w}$ gives: 

$$
\begin{align*}
\left[\frac{\partial \mathbf{r}_{\pi_{\boldsymbol{w}}}}{\partial \mathbf{w}}\right]_s &= \sum_{a \in \mathcal{A}_s} \nabla_{\mathbf{w}}\pi_{\boldsymbol{w}}(a \mid s) \, r(s,a), \\
\left[\frac{\partial \mathbf{P}_{\pi_{\boldsymbol{w}}}}{\partial \mathbf{w}}\mathbf{v}_\gamma^{\pi_{\boldsymbol{w}}}\right]_s &= \sum_{a \in \mathcal{A}_s} \nabla_{\mathbf{w}}\pi_{\boldsymbol{w}}(a \mid s)\sum_{s'} p(s' \mid s,a) \, v_\gamma^{\pi_{\boldsymbol{w}}}(s').
\end{align*}
$$

Substituting back:

$$
\begin{align*}
\nabla_{\mathbf{w}}J(\mathbf{w}) &= \sum_s x_\alpha(s)\left(\sum_a \nabla_{\mathbf{w}}\pi_{\boldsymbol{w}}(a \mid s) \, r(s,a) + \gamma\sum_a \nabla_{\mathbf{w}}\pi_{\boldsymbol{w}}(a \mid s)\sum_{s'} p(s' \mid s,a) \, v_\gamma^{\pi_{\boldsymbol{w}}}(s')\right) \\
&= \sum_s x_\alpha(s)\sum_a \nabla_{\mathbf{w}}\pi_{\boldsymbol{w}}(a \mid s)\left(r(s,a) + \gamma \sum_{s'} p(s' \mid s,a) \, v_\gamma^{\pi_{\boldsymbol{w}}}(s')\right)
\end{align*}
$$

This is the policy gradient theorem, where $x_\alpha(s)$ is the discounted state visitation distribution and the term in parentheses is the state-action value function $q^{\pi_{\boldsymbol{w}}}(s,a)$.


### Normalized Discounted State Visitation Distribution

The discounted state visitation $x_\alpha(s)$ is not normalized. Therefore the expression we obtained above is not an expectation. However, we can transform it into one by normalizing by $1 - \gamma$. Note that for any initial distribution $\alpha$:

$$
\sum_s x_\alpha(s) = \alpha^\top(\mathbf{I} - \gamma \mathbf{P}_{\pi_{\boldsymbol{w}}})^{-1}\mathbf{1} = \frac{\alpha^\top\mathbf{1}}{1-\gamma} = \frac{1}{1-\gamma}
$$

Therefore, defining the normalized state distribution $\xi_\alpha(s) = (1-\gamma)x_\alpha(s)$, we can write:

$$
\begin{align*}
\nabla_{\mathbf{w}}J(\mathbf{w}) &= \frac{1}{1-\gamma}\sum_s \xi_\alpha(s)\sum_a \nabla_{\mathbf{w}}\pi_{\boldsymbol{w}}(a \mid s)\left(r(s,a) + \gamma \sum_{s'} p(s' \mid s,a) \, v_\gamma^{\pi_{\boldsymbol{w}}}(s')\right) \\
&= \frac{1}{1-\gamma}\mathbb{E}_{s\sim\xi_\alpha}\left[\sum_a \nabla_{\mathbf{w}}\pi_{\boldsymbol{w}}(a \mid s) \, q^{\pi_{\boldsymbol{w}}}(s,a)\right]
\end{align*}
$$

Now we have expressed the policy gradient theorem in terms of expectations under the normalized discounted state visitation distribution. But what does sampling from $\xi_\alpha$ mean? Recall that $\mathbf{x}_\alpha^\top = \alpha^\top(\mathbf{I} - \gamma \mathbf{P}_{\pi_{\boldsymbol{w}}})^{-1}$. Using the Neumann series expansion (valid when $\|\gamma \mathbf{P}_{\pi_{\boldsymbol{w}}}\| < 1$, which holds for $\gamma < 1$ since $\mathbf{P}_{\pi_{\boldsymbol{w}}}$ is a stochastic matrix) we have:

$$
\boldsymbol{\xi}_\alpha^\top = (1-\gamma)\alpha^\top\sum_{k=0}^{\infty} (\gamma \mathbf{P}_{\pi_{\boldsymbol{w}}})^k
$$

We can then factor out the first term from this summation to obtain:

$$
\begin{align*}
\boldsymbol{\xi}_\alpha^\top &= (1-\gamma)\alpha^\top\sum_{k=0}^{\infty} (\gamma \mathbf{P}_{\pi_{\boldsymbol{w}}})^k \\
&= (1-\gamma)\alpha^\top + (1-\gamma)\alpha^\top\sum_{k=1}^{\infty} (\gamma \mathbf{P}_{\pi_{\boldsymbol{w}}})^k \\
&= (1-\gamma)\alpha^\top + (1-\gamma)\alpha^\top\gamma\mathbf{P}_{\pi_{\boldsymbol{w}}}\sum_{k=0}^{\infty} (\gamma \mathbf{P}_{\pi_{\boldsymbol{w}}})^k \\
&= (1-\gamma)\alpha^\top + \gamma\boldsymbol{\xi}_\alpha^\top \mathbf{P}_{\pi_{\boldsymbol{w}}}
\end{align*}
$$


The balance equation:

$$
\boldsymbol{\xi}_\alpha^\top = (1-\gamma)\alpha^\top + \gamma\boldsymbol{\xi}_\alpha^\top \mathbf{P}_{\pi_{\boldsymbol{w}}}
$$


shows that $\boldsymbol{\xi}_\alpha$ is a mixture distribution: with probability $1-\gamma$ you draw a state from the initial distribution $\alpha$ (reset), and with probability $\gamma$ you follow the policy dynamics $\mathbf{P}_{\pi_{\boldsymbol{w}}}$ from the current state (continue). This interpretation directly connects to the geometric process: at each step you either terminate and resample from $\alpha$ (with probability $1-\gamma$) or continue following the policy (with probability $\gamma$).

```{code-cell} python
import numpy as np

def sample_from_discounted_visitation(
    alpha, 
    policy, 
    transition_model, 
    gamma, 
    n_samples=1000
):
    """Sample states from the discounted visitation distribution.
    
    Args:
        alpha: Initial state distribution (vector of probabilities)
        policy: Function (state -> action probabilities)
        transition_model: Function (state, action -> next state probabilities)
        gamma: Discount factor
        n_samples: Number of states to sample
    
    Returns:
        Array of sampled states
    """
    samples = []
    n_states = len(alpha)
    rng = np.random.default_rng(2026)
    
    # Initialize state from alpha
    current_state = rng.choice(n_states, p=alpha)
    
    for _ in range(n_samples):
        samples.append(current_state)
        
        # With probability (1-gamma): reset
        if rng.random() > gamma:
            current_state = rng.choice(n_states, p=alpha)
        # With probability gamma: continue
        else:
            # Sample action from policy
            action_probs = policy(current_state)
            action = rng.choice(len(action_probs), p=action_probs)
            
            # Sample next state from transition model
            next_state_probs = transition_model(current_state, action)
            current_state = rng.choice(n_states, p=next_state_probs)
    
    return np.array(samples)

# Example usage for a simple 2-state MDP
alpha = np.array([0.7, 0.3])  # Initial distribution
policy = lambda s: np.array([0.8, 0.2])  # Dummy policy
transition_model = lambda s, a: np.array([0.9, 0.1])  # Dummy transitions
gamma = 0.9

samples = sample_from_discounted_visitation(alpha, policy, transition_model, gamma)

# Check empirical distribution
print("Empirical state distribution:")
print(np.bincount(samples) / len(samples))
```

While the math shows that sampling from the discounted visitation distribution $\boldsymbol{\xi}_\alpha$ would give us unbiased policy gradient estimates, Thomas (2014) demonstrated that this implementation can be detrimental to performance in practice. The issue arises because terminating trajectories early (with probability $1-\gamma$) reduces the effective amount of data we collect from each trajectory. This early termination weakens the learning signal, as many trajectories don't reach meaningful terminal states or rewards.

Therefore, in practice, we typically sample complete trajectories from the undiscounted process (running the policy until natural termination or a fixed horizon) while still using $\gamma$ in the advantage estimation. This approach preserves the full learning signal from each trajectory and has been empirically shown to lead to better performance. 

This is one of several cases in RL where the theoretically optimal procedure differs from the best practical implementation.

### The Actor-Critic Architecture

The policy gradient theorem shows that the gradient depends on the action-value function $q^{\pi_{\boldsymbol{w}}}(s,a)$. In practice, we do not have access to the true $q$-function and must estimate it. This leads to the **actor-critic** architecture: the **actor** maintains the policy $\pi_{\boldsymbol{w}}$, while the **critic** maintains an estimate of the value function.

This architecture traces back to Sutton's 1984 thesis, where he proposed the Adaptive Heuristic Critic. The actor uses the critic's value estimates to compute advantage estimates for the policy gradient, while the critic learns from the same trajectories generated by the actor. The algorithms we developed earlier (REINFORCE with baseline, GAE, and the one-step actor-critic) are all instances of this architecture.

We are simultaneously learning two functions that depend on each other, which creates a stability challenge. The actor's gradient uses the critic's estimates, but the critic is trained on data generated by the actor's policy. If both change too quickly, the learning process can become unstable.

{cite:t}`konda2002thesis` analyzed this coupled learning problem and established convergence guarantees under a **two-timescale** condition: the critic must update faster than the actor. Intuitively, the critic needs to "track" the current policy's value function before the actor uses those estimates to update. If the actor moves too fast, it uses stale or inaccurate value estimates, leading to poor gradient estimates.

In practice, this is implemented by using different learning rates: a larger learning rate $\alpha_\theta$ for the critic and a smaller learning rate $\alpha_w$ for the actor, with $\alpha_\theta > \alpha_w$. Alternatively, one can perform multiple critic updates per actor update. The soft actor-critic algorithm discussed earlier in the [amortization chapter](amortized-action-optimization.md) follows this same principle, inheriting the actor-critic structure while incorporating entropy regularization and learning Q-functions directly.

The actor-critic architecture also connects to the bilevel optimization perspective of the policy gradient theorem: the outer problem optimizes the policy, while the inner problem solves for the value function given that policy. The two-timescale condition ensures that the inner problem is approximately solved before taking a step on the outer problem.

## Reparameterization Methods in Reinforcement Learning

When actions and dynamics admit differentiable sampling paths, how does
pathwise differentiation change the variance and model requirements of policy
optimization?

When dynamics are known or can be learned, reparameterization provides an alternative to score function methods. By expressing actions and state transitions as deterministic functions of noise, we can backpropagate through trajectories to compute policy gradients with lower variance than score function estimators.

### Stochastic Value Gradients

The reparameterization trick requires that we can express our random variable as a deterministic function of noise. In reinforcement learning, this applies naturally when we have a learned model of the dynamics. Consider a stochastic policy $\pi_{\boldsymbol{w}}(a|s)$ that we can reparameterize as $a = \pi_{\boldsymbol{w}}(s,\epsilon)$ where $\epsilon \sim p(\epsilon)$, and a dynamics model $s' = f(s,a,\xi)$ where $\xi \sim p(\xi)$ represents environment stochasticity. Both transformations are deterministic given the noise variables.

With these reparameterizations, we can write an $n$-step return as a differentiable function of the noise:

$$
R_n(s_0,\{\epsilon_i\},\{\xi_i\}) = \sum_{i=0}^{n-1} \gamma^i r(s_i,a_i)
$$

where $a_i = \pi_{\boldsymbol{w}}(s_i,\epsilon_i)$ and $s_{i+1} = f(s_i,a_i,\xi_i)$ for $i=0,...,n-1$. The objective becomes:

$$
J(\boldsymbol{w}) = \mathbb{E}_{\{\epsilon_i\},\{\xi_i\}}[R_n(s_0,\{\epsilon_i\},\{\xi_i\})]
$$

We can now apply the reparameterization gradient estimator:

$$
\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) = \mathbb{E}_{\{\epsilon_i\},\{\xi_i\}}\left[\nabla_{\boldsymbol{w}}R_n(s_0,\{\epsilon_i\},\{\xi_i\})\right]
$$

This gradient can be computed by automatic differentiation through the sequence of policy and model evaluations. The computation requires backpropagating through $n$ steps of model rollouts, which becomes expensive for large $n$ but avoids the high variance of score function estimators.

The Stochastic Value Gradients (SVG) framework {cite:p}`Heess2015` uses this approach while introducing a hybrid objective that combines model rollouts with value function bootstrapping:

$$
J^{\text{SVG}(n)}(\boldsymbol{w}) = \mathbb{E}_{\{\epsilon_i\},\{\xi_i\}}\left[\sum_{i=0}^{n-1} \gamma^i r(s_i,a_i) + \gamma^n q(s_n,a_n;\theta)\right]
$$

The terminal value function $q(s_n,a_n;\theta)$ approximates the value beyond horizon $n$, allowing shorter rollouts while still capturing long-term value. This creates a spectrum of algorithms parameterized by $n$.

#### SVG(0): Model-Free Reparameterization

When $n=0$, the objective collapses to:

$$
J^{\text{SVG}(0)}(\boldsymbol{w}) = \mathbb{E}_{s \sim \rho}\mathbb{E}_{\epsilon \sim p(\epsilon)}\left[q(s,\pi_{\boldsymbol{w}}(s,\epsilon);\theta)\right]
$$

No model is required. We simply differentiate the critic with respect to actions sampled from the reparameterized policy. This is the approach used in DDPG {cite:p}`lillicrap2015continuous` (with a deterministic policy where $\epsilon$ is absent) and SAC {cite:p}`haarnoja2018soft` (where $\epsilon$ produces the stochastic component). The gradient is:

$$
\nabla_{\boldsymbol{w}} J^{\text{SVG}(0)} = \mathbb{E}_{s,\epsilon}\left[\nabla_a q(s,a;\theta)\big|_{a=\pi_{\boldsymbol{w}}(s,\epsilon)} \nabla_{\boldsymbol{w}} \pi_{\boldsymbol{w}}(s,\epsilon)\right]
$$

This requires only that the critic $q$ be differentiable with respect to actions, not a learned dynamics model. All bias comes from errors in the value function approximation.

#### SVG(1) to SVG($n$): Model-Based Rollouts

For $n \geq 1$, we unroll a learned dynamics model for $n$ steps before bootstrapping with the critic. Consider SVG(1):

$$
J^{\text{SVG}(1)}(\boldsymbol{w}) = \mathbb{E}_{s,\epsilon,\xi}\left[r(s,\pi_{\boldsymbol{w}}(s,\epsilon)) + \gamma q(f(s,\pi_{\boldsymbol{w}}(s,\epsilon),\xi), \pi_{\boldsymbol{w}}(s',\epsilon');\theta)\right]
$$

where $s' = f(s,\pi_{\boldsymbol{w}}(s,\epsilon),\xi)$ is the next state predicted by the model. The gradient now flows through both the reward and the model transition. Increasing $n$ propagates reward information more directly through the model rollout, reducing reliance on the critic. However, model errors compound over the horizon. If the model is inaccurate, longer rollouts can degrade performance.

#### SVG($\infty$): Pure Model-Based Optimization

As $n \to \infty$, we eliminate the critic entirely:

$$
J^{\text{SVG}(\infty)}(\boldsymbol{w}) = \mathbb{E}_{\{\epsilon_i\},\{\xi_i\}}\left[\sum_{i=0}^{T-1} \gamma^i r(s_i,\pi_{\boldsymbol{w}}(s_i,\epsilon_i))\right]
$$

This is pure model-based policy optimization, differentiating through the entire trajectory. Approaches like PILCO {cite:p}`deisenroth2011pilco` and Dreamer {cite:p}`hafner2019dreamer` operate in this regime. With an accurate model, this provides the most direct gradient signal. The tradeoff is computational: backpropagating through hundreds of time steps is expensive, and gradient magnitudes can explode or vanish over long horizons.

The choice of $n$ reflects a fundamental bias-variance tradeoff. Small $n$ relies on the critic for long-term value estimation, inheriting its approximation errors. Large $n$ relies on the model, accumulating its prediction errors. In practice, intermediate values like $n=5$ or $n=10$ often work well when combined with a reasonably accurate learned model.

#### Noise Inference for Off-Policy Learning

A subtle issue arises when combining reparameterization with experience replay. SVG naturally supports off-policy learning: states $s$ can be sampled from a replay buffer rather than the current policy. However, reparameterization requires the noise variables $\epsilon$ that generated each action.

For on-policy data, we can simply store $\epsilon$ alongside each transition $(s, a, r, s')$. For off-policy data collected under a different policy, the noise is unknown. To apply reparameterization gradients to such data, we must *infer* the noise that would have produced the observed action under the current policy.

For invertible policies, this is straightforward. If $a = \pi_{\boldsymbol{w}}(s, \epsilon)$ with $\epsilon \sim \mathcal{N}(0, I)$, and the policy takes the form $a = \mu_{\boldsymbol{w}}(s) + \sigma_{\boldsymbol{w}}(s) \odot \epsilon$ (as in a Gaussian policy), we can recover the noise exactly:

$$
\epsilon = \frac{a - \mu_{\boldsymbol{w}}(s)}{\sigma_{\boldsymbol{w}}(s)}
$$

This recovered $\epsilon$ can then be used for gradient computation. However, this introduces a subtle dependence: the inferred $\epsilon$ depends on the current policy parameters $\boldsymbol{w}$, not just the data. As the policy changes during training, the same action $a$ corresponds to different noise values.

For dynamics noise $\xi$, the situation is more complex. If we have a probabilistic model $s' = f(s, a, \xi)$ and observe the actual next state $s'$, we could in principle infer $\xi$. In practice, environment stochasticity is often treated as irreducible: we cannot replay the exact same noise realization. SVG handles this by either: (1) using deterministic models and ignoring environment stochasticity, (2) re-simulating from the model rather than using observed next states, or (3) using importance weighting to correct for the distribution mismatch.

The noise inference perspective connects reparameterization gradients to the broader question of credit assignment in RL. By explicitly tracking which noise realizations led to which outcomes, we can more precisely attribute value to policy parameters rather than to lucky or unlucky samples.

When dynamics are deterministic or can be accurately reparameterized, SVG-style methods offer an efficient alternative to the score function methods developed in the previous section. However, many reinforcement learning problems involve unknown dynamics or dynamics that resist accurate modeling. In those settings, score function methods remain the primary tool since they require only the ability to sample trajectories under the policy.

## Summary

The trajectory score supplies a model-free policy gradient because the
transition terms do not depend on the policy parameters. Conditional returns,
state-dependent baselines, generalized advantage estimates, and learned
critics reduce its variance. Importance ratios then compare data from an older
policy with the current one, while PPO clips those ratios to limit the update.

The SwingRL experiment separated a correct PPO implementation from a
successful control result. Five million total training interactions improved
the shaped return but produced no full rotations. The structured controller
solved the nominal rigid-link problem without policy training, then failed a
different test because the resulting trajectory required a chain to push.
Algorithm diagnostics, task success, and model validity therefore remain
separate parts of the evaluation.

The policy-gradient theorem expresses the same derivative through discounted
state visitation and action values. Approximating those values produces the
actor-critic architecture, with a faster critic update supplying the signal
used by the actor.

When dynamics models are available, reparameterization through stochastic
value gradients supplies a lower-variance alternative. SVG(0) recovers
actor-critic methods such as DDPG and SAC, while SVG($\infty$) differentiates
through a complete simulated trajectory. The resulting choice is now explicit:
score methods trade variance for minimal model assumptions; pathwise methods
trade stronger differentiability assumptions for lower-variance credit
assignment.

## Self-checks

:::{exercise} Score-function requirement
:label: ex-pg-check-1

Does the likelihood-ratio policy-gradient estimator require differentiating the environment dynamics? Why?
:::

:::{solution} ex-pg-check-1
:class: dropdown

No. It differentiates the log probability of sampled actions under the policy and weights that score by sampled returns or advantages; the environment need only generate trajectories.
:::

:::{exercise} Baseline bias
:label: ex-pg-check-2

Why can a state-dependent baseline reduce variance without biasing the score-function policy gradient?
:::

:::{solution} ex-pg-check-2
:class: dropdown

Conditioned on a state, the expected policy score is zero: $\sum_a\pi(a|s)\nabla\log\pi(a|s)=0$. Multiplying a state-only baseline by that score therefore has zero expectation.
:::
