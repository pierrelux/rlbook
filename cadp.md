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

# Policy Parametrization Methods

In the previous chapter, we explored various approaches to approximate dynamic programming, focusing on ways to handle large state spaces through function approximation. However, these methods still face significant challenges when dealing with large or continuous action spaces. The need to maximize over actions during the Bellman operator evaluation becomes computationally prohibitive as the action space grows.

This chapter explores a natural evolution of these ideas: rather than exhaustively searching over actions, we can parameterize and directly optimize the policy itself. We begin by examining how fitted Q methods, while powerful for handling large state spaces, still struggle with action space complexity. 

# Embedded Optimization

Recall that in fitted Q methods, the main idea is to compute the Bellman operator only at a subset of all states, relying on function approximation to generalize to the remaining states. At each step of the successive approximation loop, we build a dataset of input state-action pairs mapped to their corresponding optimality operator evaluations: 

$$
\mathcal{D}_n = \{((s, a), (Lq)(s, a; \boldsymbol{\theta}_n)) \mid (s,a) \in \mathcal{B}\}
$$

This dataset is then fed to our function approximator (neural network, random forest, linear model) to obtain the next set of parameters:

$$
\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}_n)
$$

While this strategy allows us to handle very large or even infinite (continuous) state spaces, it still requires maximizing over actions ($\max_{a \in A}$) during the dataset creation when computing the operator $L$ for each basepoint. This maximization becomes computationally expensive for large action spaces. A natural improvement is to add another level of optimization: for each sample added to our regression dataset, we can employ numerical optimization methods to find actions that maximize the Bellman operator for the given state.

```{prf:algorithm} Fitted Q-Iteration with Explicit Optimization
:label: fitted-q-iteration-explicit

**Input** Given an MDP $(S, A, P, R, \gamma)$, base points $\mathcal{B}$, function approximator class $q(s,a; \boldsymbol{\theta})$, maximum iterations $N$, tolerance $\varepsilon > 0$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ (e.g., for zero initialization)
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D} \leftarrow \emptyset$ // Regression Dataset
    2. For each $(s,a,r,s') \in \mathcal{B}$: // Assumes Monte Carlo Integration with one sample
        1. $y_{s,a} \leftarrow r + \gamma \texttt{maximize}(q(s', \cdot; \boldsymbol{\theta}_n))$ // $s'$ and $\boldsymbol{\theta}_n$ are kept fixed
        2. $\mathcal{D} \leftarrow \mathcal{D} \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D})$
    4. $\delta \leftarrow \frac{1}{|\mathcal{D}||A|}\sum_{(s,a) \in \mathcal{D} \times A} (q(s,a; \boldsymbol{\theta}_{n+1}) - q(s,a; \boldsymbol{\theta}_n))^2$
    5. $n \leftarrow n + 1$
4. **until** ($\delta < \varepsilon$ or $n \geq N$)
5. **return** $\boldsymbol{\theta}_n$
```

The above pseudocode introduces a generic $\texttt{maximize}$ routine which represents any numerical optimization method that searches for an action maximizing the given function. This approach is versatile and can be adapted to different types of action spaces. For continuous action spaces, we can employ standard nonlinear optimization methods like gradient descent or L-BFGS (e.g., using scipy.optimize.minimize). For large discrete action spaces, we can use integer programming solvers - linear integer programming if the Q-function approximator is linear in actions, or mixed-integer nonlinear programming (MINLP) solvers for nonlinear Q-functions. The choice of solver depends on the structure of our Q-function approximator and the constraints on our action space.

## Amortized Optimization Approach

This process is computationally intensive. A natural question is whether we can "amortize" some of this computation by replacing the explicit optimization for each sample with a direct mapping that gives us an approximate maximizer directly. 
For Q-functions, recall that the operator is given by:

$$
(\mathrm{L}q)(s,a) = r(s,a) + \gamma \int p(ds'|s,a)\max_{a' \in \mathcal{A}(s')} q(s', a')
$$

If $q^*$ is the optimal state-action value function, then $v^*(s) = \max_a q^*(s,a)$, and we can derive the optimal policy directly by computing the decision rule:

$$
d^\star(s) = \arg\max_{a \in \mathcal{A}(s)} q^\star(s,a)
$$

Since $q^*$ is a fixed point of $L$, we can write:

$$
\begin{align*}
q^\star(s,a) &= (Lq^*)(s,a) \\
&= r(s,a) + \gamma \int p(ds'|s,a) \max_{a' \in \mathcal{A}(s')} q^\star(s', a') \\
&= r(s,a) + \gamma \int p(ds'|s,a) q^\star(s', d^\star(s'))
\end{align*}
$$

Note that $d^\star$ is implemented by our $\texttt{maximize}$ numerical solver in the procedure above. A practical strategy would be to collect these maximizer values at each step and use them to train a function approximator that directly predicts these solutions. Due to computational constraints, we might want to compute these exact maximizer values only for a subset of states, based on some computational budget, and use the fitted decision rule to generalize to the remaining states. This leads to the following amortized version:

```{prf:algorithm} Fitted Q-Iteration with Amortized Optimization
:label: fitted-q-iteration-amortized

**Input** Given an MDP $(S, A, P, R, \gamma)$, base points $\mathcal{B}$, subset for exact optimization $\mathcal{B}_{\text{opt}} \subset \mathcal{B}$, Q-function approximator $q(s,a; \boldsymbol{\theta})$, policy approximator $d(s; \boldsymbol{w})$, maximum iterations $N$, tolerance $\varepsilon > 0$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function, $\boldsymbol{w}$ for policy

1. Initialize $\boldsymbol{\theta}_0$, $\boldsymbol{w}_0$
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D}_q \leftarrow \emptyset$ // Q-function regression dataset
    2. $\mathcal{D}_d \leftarrow \emptyset$ // Policy regression dataset
    3. For each $(s,a,r,s') \in \mathcal{B}$:
        1. // Determine next state's action using either exact optimization or approximation
        2. **if** $s' \in \mathcal{B}_{\text{opt}}$ **then**
            1. $a^*_{s'} \leftarrow \texttt{maximize}(q(s', \cdot; \boldsymbol{\theta}_n))$
            2. $\mathcal{D}_d \leftarrow \mathcal{D}_d \cup \{(s', a^*_{s'})\}$
        3. **else**
            1. $a^*_{s'} \leftarrow d(s'; \boldsymbol{w}_n)$
        4. // Compute Q-function target using chosen action
        5. $y_{s,a} \leftarrow r + \gamma q(s', a^*_{s'}; \boldsymbol{\theta}_n)$
        6. $\mathcal{D}_q \leftarrow \mathcal{D}_q \cup \{((s,a), y_{s,a})\}$
    4. // Update both function approximators
    5. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}_q)$
    6. $\boldsymbol{w}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}_d)$
    7. // Compute convergence criteria
    8. $\delta_q \leftarrow \frac{1}{|\mathcal{D}_q|}\sum_{(s,a) \in \mathcal{D}_q} (q(s,a; \boldsymbol{\theta}_{n+1}) - q(s,a; \boldsymbol{\theta}_n))^2$
    9. $\delta_d \leftarrow \frac{1}{|\mathcal{D}_d|}\sum_{(s,a^*) \in \mathcal{D}_d} \|a^* - d(s; \boldsymbol{w}_{n+1})\|^2$
    10. $n \leftarrow n + 1$
4. **until** ($\max(\delta_q, \delta_d) < \varepsilon$ or $n \geq N$)
5. **return** $\boldsymbol{\theta}_n$, $\boldsymbol{w}_n$
```

An important observation about this procedure is that the policy $d(s; \boldsymbol{w})$ is being trained on a dataset $\mathcal{D}_d$ containing optimal actions computed with respect to an evolving Q-function. Specifically, at iteration n, we collect pairs $(s', a^*_{s'})$ where $a^*_{s'} = \arg\max_a q(s', a; \boldsymbol{\theta}_n)$. However, after updating to $\boldsymbol{\theta}_{n+1}$, these actions may no longer be optimal with respect to the new Q-function.

A natural approach to handle this staleness would be to maintain only the most recent optimization data. We could modify our procedure to keep a sliding window of K iterations, where at iteration n, we only use data from iterations max(0, n-K) to n. This would be implemented by augmenting each entry in $\mathcal{D}_d$ with a timestamp:

$$
\mathcal{D}_d^t = \{(s', a^*_{s'}, t) \mid t \in \{n-K,\ldots,n\}\}
$$

where t indicates the iteration at which the optimal action was computed. When fitting the policy network, we would then only use data points that are at most K iterations old:

$$
\boldsymbol{w}_{n+1} \leftarrow \texttt{fit}(\{(s', a^*_{s'}) \mid (s', a^*_{s'}, t) \in \mathcal{D}_d^t, n-K \leq t \leq n\})
$$

This introduces a trade-off between using more data (larger K) versus using more recent, accurate data (smaller K). The choice of K would depend on how quickly the Q-function evolves and the computational budget available for computing exact optimal actions.

Now the main issue with this approach, apart from the intrinsic out-of-distribution drift that we are trying to track, is that it requires "ground truth" - samples of optimal actions computed by the actual solver. This raises an intriguing question: how few samples do we actually need? Could we even envision eliminating the solver entirely? What seems impossible at first glance turns out to be achievable. The intuition is that as our policy improves at selecting actions, we can bootstrap from these increasingly better choices. As we continuously amortize these improving actions over time, it creates a virtuous cycle of self-improvement towards the optimal policy. But for this bootstrapping process to work, we need careful management - move too quickly and the process may become unstable. Let's examine how this balance can be achieved.


# Deterministic Parametrized Policies 


In this section, we consider deterministic parametrized policies of the form $d(s; \boldsymbol{w})$ which directly output an action given a state. This approach differs from stochastic policies that output probability distributions over actions, making it particularly suitable for continuous control problems where the optimal policy is often deterministic. We'll see how fitted Q-value methods can be naturally extended to simultaneously learn both the Q-function and such a deterministic policy.


## Neural Fitted Q-iteration for Continuous Actions (NFQCA)

To develop this approach, let's first consider an idealized setting where we have access to $q^\star$, the optimal Q-function. Then we can state our goal as finding policy parameters $\boldsymbol{w}$ that maximize $q^\star$ with respect to the actions chosen by our policy across the state space:

$$
\max_{\boldsymbol{w}} q^*(s, d(s; \boldsymbol{w})) \quad \text{for all } s
$$

However, it's computationally infeasible to satisfy this condition for every possible state $s$, especially in large or continuous state spaces. To address this, we assume a distribution of states, denoted $\mu(s)$, and take the expectation, leading to the problem:

$$
\max_{\boldsymbol{w}} \mathbb{E}_{s \sim \mu(s)}[q^*(s, d(s; \boldsymbol{w}))]
$$

However in practice, we do not have access to $q^*$. Instead, we need to approximate $q^*$ with a Q-function $q(s, a; \boldsymbol{\theta})$, parameterized by $\boldsymbol{\theta}$, which we will learn simultaneously with the policy function $d(s; \boldsymbol{w})$. Given a samples of initial states drawn from $\mu$, we then maximize this objective via a Monte Carlo surrogate problem:  


$$
\max_{\boldsymbol{w}} \mathbb{E}_{s \sim \mu(s)}[q(s, d(s; \boldsymbol{w}); \boldsymbol{\theta})] \approx
\max_{\boldsymbol{w}} \frac{1}{|\mathcal{B}|} \sum_{s \in \mathcal{B}}  q(s, d(s; \boldsymbol{w}); \boldsymbol{\theta})
$$

When using neural networks to parametrize $q$ and $d$, we obtain the Neural Fitted Q-Iteration with Continuous Actions (NFQCA) algorithm proposed by {cite}`Hafner2011`.

```{prf:algorithm} Neural Fitted Q-Iteration with Continuous Actions (NFQCA)
:label: nfqca

**Input** MDP $(S, A, P, R, \gamma)$, base points $\mathcal{B}$, Q-function $q(s,a; \boldsymbol{\theta})$, policy $d(s; \boldsymbol{w})$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function, $\boldsymbol{w}$ for policy

1. Initialize $\boldsymbol{\theta}_0$, $\boldsymbol{w}_0$
2. **for** $n = 0,1,2,...$ **do**
    1. $\mathcal{D}_q \leftarrow \emptyset$
    2. For each $(s,a,r,s') \in \mathcal{B}$:
        1. $a'_{s'} \leftarrow d(s'; \boldsymbol{w}_n)$
        2. $y_{s,a} \leftarrow r + \gamma q(s', a'_{s'}; \boldsymbol{\theta}_n)$
        3. $\mathcal{D}_q \leftarrow \mathcal{D}_q \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}_q)$
    4. $\boldsymbol{w}_{n+1} \leftarrow \texttt{minimize}_{\boldsymbol{w}} -\frac{1}{|\mathcal{B}|} \sum_{(s,a,r,s') \in \mathcal{B}} q(s, d(s; \boldsymbol{w}); \boldsymbol{\theta}_{n+1})$
3. **return** $\boldsymbol{\theta}_n$, $\boldsymbol{w}_n$
```


In practice, both the `fit` and `minimize` operations above are implemented using gradient descent. For the Q-function, the `fit` operation minimizes the mean squared error between the network's predictions and the target values:

$$
\texttt{fit}(\mathcal{D}_q) = \arg\min_{\boldsymbol{\theta}} \frac{1}{|\mathcal{D}_q|} \sum_{((s,a), y) \in \mathcal{D}_q} (q(s,a; \boldsymbol{\theta}) - y)^2
$$

For the policy update, the `minimize` operation uses gradient descent on the composition of the "critic" network $q$ and the "actor" network $d$. This results in the following update rule:

$$
\boldsymbol{w}_{n+1} = \boldsymbol{w}_n + \alpha \nabla_{\boldsymbol{w}} \left(\frac{1}{|\mathcal{B}|} \sum_{(s,a,r,s') \in \mathcal{B}} q(s, d(s; \boldsymbol{w}); \boldsymbol{\theta}_{n+1})\right)
$$

where $\alpha$ is the learning rate. Both operations can be efficiently implemented using modern automatic differentiation libraries and stochastic gradient descent variants like Adam or RMSProp.

## Deep Deterministic Policy Gradient (DDPG)

Just as DQN adapted Neural Fitted Q-Iteration to the online setting, DDPG {cite}`lillicrap2015continuous` extends NFQCA to learn from data collected online. Like NFQCA, DDPG simultaneously learns a Q-function and a deterministic policy that maximizes it, but differs in how it collects and processes data.

Instead of maintaining a fixed set of basepoints, DDPG uses a replay buffer that continuously stores new transitions as the agent interacts with the environment. Since the policy is deterministic, exploration becomes challenging. DDPG addresses this by adding noise to the policy's actions during data collection:

$$
a = d(s; \boldsymbol{w}) + \mathcal{N}
$$

where $\mathcal{N}$ represents exploration noise drawn from an Ornstein-Uhlenbeck (OU) process. The OU process is particularly well-suited for control tasks as it generates temporally correlated noise, leading to smoother exploration trajectories compared to independent random noise. It is defined by the stochastic differential equation:

$$
d\mathcal{N}_t = \theta(\mu - \mathcal{N}_t)dt + \sigma dW_t
$$

where $\mu$ is the long-term mean value (typically set to 0), $\theta$ determines how strongly the noise is pulled toward this mean, $\sigma$ scales the random fluctuations, and $dW_t$ is a Wiener process (continuous-time random walk). For implementation, we discretize this continuous-time process using the Euler-Maruyama method:

$$
\mathcal{N}_{t+1} = \mathcal{N}_t + \theta(\mu - \mathcal{N}_t)\Delta t + \sigma\sqrt{\Delta t}\epsilon_t
$$

where $\Delta t$ is the time step and $\epsilon_t \sim \mathcal{N}(0,1)$ is standard Gaussian noise. Think of this process like a spring mechanism: when the noise value $\mathcal{N}_t$ deviates from $\mu$, the term $\theta(\mu - \mathcal{N}_t)\Delta t$ acts like a spring force, continuously pulling it back. Unlike a spring, however, this return to $\mu$ is not oscillatory - it's more like motion through a viscous fluid, where the force simply decreases as the noise gets closer to $\mu$. The random term $\sigma\sqrt{\Delta t}\epsilon_t$ then adds perturbations to this smooth return trajectory. This creates noise that wanders away from $\mu$ (enabling exploration) but is always gently pulled back (preventing the actions from wandering too far), with $\theta$ controlling the strength of this pulling force.

The policy gradient update follows the same principle as NFQCA:

$$
\nabla_{\boldsymbol{w}} \mathbb{E}_{s \sim \mu(s)}[q(s, d(s; \boldsymbol{w}); \boldsymbol{\theta})]
$$

We then embed this exploration mechanism into the data collection procedure and use the same flattened FQI structure that we adopted in DQN. Similar to DQN, flattening the outer-inner optimization structure leads to the need for target networks - both for the Q-function and the policy.

```{prf:algorithm} Deep Deterministic Policy Gradient (DDPG)
:label: ddpg

**Input** MDP $(S, A, P, R, \gamma)$, Q-network $q(s,a; \boldsymbol{\theta})$, policy network $d(s; \boldsymbol{w})$, learning rates $\alpha_q, \alpha_d$, replay buffer size $B$, mini-batch size $b$, target update frequency $K$

**Initialize**
1. Parameters $\boldsymbol{\theta}_0$, $\boldsymbol{w}_0$ randomly
2. Target parameters: $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_0$, $\boldsymbol{w}_{target} \leftarrow \boldsymbol{w}_0$
3. Initialize replay buffer $\mathcal{R}$ with capacity $B$
4. Initialize exploration noise process $\mathcal{N}$
5. $n \leftarrow 0$

6. **while** training:
    1. Observe current state $s$
    2. Select action with noise: $a = d(s; \boldsymbol{w}_n) + \mathcal{N}$
    3. Execute $a$, observe reward $r$ and next state $s'$
    4. Store $(s,a,r,s')$ in $\mathcal{R}$, replacing oldest if full
    5. Sample mini-batch of $b$ transitions $(s_i,a_i,r_i,s'_i)$ from $\mathcal{R}$
    6. For each sampled transition:
        1. $y_i \leftarrow r_i + \gamma q(s'_i, d(s'_i; \boldsymbol{w}_{target}); \boldsymbol{\theta}_{target})$
    7. Update Q-network: $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_n - \alpha_q \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_i(y_i - q(s_i,a_i;\boldsymbol{\theta}_n))^2$
    8. Update policy: $\boldsymbol{w}_{n+1} \leftarrow \boldsymbol{w}_n + \alpha_d \frac{1}{b}\sum_i \nabla_a q(s_i,a;\boldsymbol{\theta}_{n+1})|_{a=d(s_i;\boldsymbol{w}_n)} \nabla_{\boldsymbol{w}} d(s_i;\boldsymbol{w}_n)$
    9. If $n \bmod K = 0$:
        1. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_n$
        2. $\boldsymbol{w}_{target} \leftarrow \boldsymbol{w}_n$
    10. $n \leftarrow n + 1$
    
**return** $\boldsymbol{\theta}_n$, $\boldsymbol{w}_n$
```


# Stochastic Policy Parameterization

## Soft Actor Critic
Adapting the intuition of NFQCA to the smooth Bellman optimality equations leads us to the soft actor-critic algorithm {cite}`haarnoja2018soft`. To understand this connection, let's first examine how the smooth Bellman equations emerge naturally from entropy regularization.

Consider the standard Bellman operator augmented with an entropy term. The smooth Bellman operator $\mathrm{L}_\beta$ takes the form:

$$
(\mathrm{L}_\beta v)(s) = \max_{d \in D^{MR}}\{\mathbb{E}_{a \sim d}[r(s,a) + \gamma v(s')] + \beta\mathcal{H}(d)\}
$$

where $\mathcal{H}(d) = -\mathbb{E}_{a \sim d}[\log d(a|s)]$ represents the entropy of the policy. To find the solution to the optimization problem embedded in the operator $\mathrm{L}_\beta$, we set the functional derivative of the objective with respect to the decision rule to zero:

$$
\frac{\delta}{\delta d(a|s)} \left[\int_A d(a|s)(r(s,a) + \gamma v(s'))da - \beta\int_A d(a|s)\log d(a|s)da \right] = 0
$$

Enforcing that $\int_A d(a|s)da = 1$ leads to the following Lagrangian:

$$
r(s,a) + \gamma v(s') - \beta(1 + \log d(a|s)) - \lambda(s) = 0
$$

Solving for $d$ shows that the optimal policy is a Boltzmann distribution 

$$
d^*(a|s) = \frac{\exp(\frac{1}{\beta}(r(s,a) + \gamma \mathbb{E}_{s'}[v(')]))}{Z(s)}
$$

When we substitute this optimal policy back into the entropy-regularized objective, we obtain:

$$
\begin{align*}
v(s) &= \mathbb{E}_{a \sim d^*}[r(s,a) + \gamma v(s')] + \beta\mathcal{H}(d^*) \\
&= \beta \log \int_A \exp(\frac{1}{\beta}(r(s,a) + \gamma \mathbb{E}_{s'}[v(s')]))da
\end{align*}
$$
 
As we saw at the beginning of this chapter, the smooth Bellman optimality operator for Q-factors is defined as:

$$
(\mathrm{L}_\beta q)(s,a) = r(s,a) + \gamma \mathbb{E}_{s'}\left[\beta \log \int_A \exp(\frac{1}{\beta}q(s',a'))da'\right]
$$

This operator maintains the contraction property of its standard counterpart, guaranteeing a unique fixed point $q^*$. The optimal policy takes the form:

$$
d^*(a|s) = \frac{\exp(\frac{1}{\beta}q^*(s,a))}{Z(s)}
$$

where $Z(s) = \int_A \exp(\frac{1}{\beta}q^*(s,a))da$. The optimal value function can be recovered as:

$$
v^*(s) = \beta \log \int_A \exp(\frac{1}{\beta}q^*(s,a))da
$$

### Fitted Q-Iteration for the Smooth Bellman Equations

Following the principles of fitted value iteration, we can approximate approximate the effect of the smooth Bellman operator by computing it exactly at a number of basepoints and generalizing elsewhere using function approximation. Concretely, given a collection of states $s_i$ and actions $a_i$, we would compute regression target values:

$$
y_i = r(s_i,a_i) + \gamma \mathbb{E}_{s'}\left[\beta \log \int_A \exp(\frac{1}{\beta}q_\theta(s',a'))da'\right]
$$

and fit our Q-function approximator by minimizing:

$$
\min_\theta \sum_i (q_\theta(s_i,a_i) - y_i)^2
$$

The expectation over next states can be handled through Monte Carlo estimation using samples from the environment: given a transition $(s_i,a_i,s'_i)$, we can approximate:

$$
\mathbb{E}_{s'}\left[\beta \log \int_A \exp(\frac{1}{\beta}q_\theta(s',a'))da'\right] \approx \beta \log \int_A \exp(\frac{1}{\beta}q_\theta(s'_i,a'))da'
$$

However, we still face the challenge of computing the integral over actions. This motivates maintaining separate function approximators for both Q and V, using samples from the current policy to estimate the value function:

$$
v_\psi(s) \approx \mathbb{E}_{a \sim d(\cdot|s;\phi)}\left[q_\theta(s,a) - \beta \log d(a|s;\phi)\right]
$$

By maintaining both approximators, we can estimate targets using sampled actions from our policy. Specifically, if we have a transition $(s_i,a_i,s'_i)$ and sample $a'_i \sim d(\cdot|s'_i;\phi)$, our target becomes:

$$
y_i = r(s_i,a_i) + \gamma\left(q_\theta(s'_i,a'_i) - \beta \log d(a'_i|s'_i;\phi)\right)
$$

This is a remarkable idea! One that exists only due to the dual representation of the smooth Bellman equations as an entropy-regularized problem which transforms the intractable log-sum-exp into a form we can estimate efficiently through sampling.

### Approximating Boltzmann Policies by Gaussians

The entropy-regularized objective and the smooth Bellman equation are mathematically equivalent. However, both formulations face a practical challenge: they require evaluating an intractable integral due to the Boltzmann distribution. Soft Actor-Critic (SAC) addresses this problem by approximating the optimal policy with a simpler, more tractable Gaussian distribution. Given the optimal soft policy:

$$
d^*(a|s) = \frac{\exp(\frac{1}{\beta}q^*(s,a))}{Z(s)}
$$

we seek to approximate it with a Gaussian policy:

$$
d(a|s;\phi) = \mathcal{N}(\mu_\phi(s), \sigma_\phi(s))
$$

This approximation task naturally raises the question of how to measure the "closeness" between the target Boltzmann distribution and a candidate Gaussian approximation. Following common practice in deep learning, we employ the Kullback-Leibler (KL) divergence as our measure of distributional distance. To find the best approximation, we minimize the KL divergence between our policy and the optimal policy, using our current estimate $q_\theta$ of $q^*$:

$$
\operatorname{minimize}_{\phi} \mathbb{E}_{s \sim \mu(s)}\left[D_{KL}\left(d(\cdot|s;\phi) \| \frac{\exp(\frac{1}{\beta}q_\theta(s,\cdot))}{Z(s)}\right)\right]
$$


However, an important question remains: how can we solve this optimization problem when it involves the intractable partition function $Z(s)$? To see this, recall that for two distributions p and q, the KL divergence takes the form $D_{KL}(p\|q) = \mathbb{E}_{x \sim p}[\log p(x) - \log q(x)]$. Let's denote the target Boltzmann distribution based on our current Q-estimate as:

$$
d_\theta(a|s) = \frac{\exp(\frac{1}{\beta}q_\theta(s,a))}{Z_\theta(s)}
$$

Then the KL minimization becomes:

$$
\begin{align*}
D_{KL}(d(\cdot|s;\phi)\|d_\theta) &= \mathbb{E}_{a \sim d(\cdot|s;\phi)}[\log d(a|s;\phi) - \log d_\theta(a|s)] \\
&= \mathbb{E}_{a \sim d(\cdot|s;\phi)}\left[\log d(a|s;\phi) - \log \left(\frac{\exp(\frac{1}{\beta}q_\theta(s,a))}{Z_\theta(s)}\right)\right] \\
&= \mathbb{E}_{a \sim d(\cdot|s;\phi)}\left[\log d(a|s;\phi) - \frac{1}{\beta}q_\theta(s,a) + \log Z_\theta(s)\right]
\end{align*}
$$

Since $\log Z(s)$ is constant with respect to $\phi$, minimizing this KL divergence is equivalent to:

$$
\operatorname{minimize}_{\phi} \mathbb{E}_{s \sim \mu(s)}\mathbb{E}_{a \sim d(\cdot|s;\phi)}[\log d(a|s;\phi) - \frac{1}{\beta}q_\theta(s,a)]
$$

### Reparameterizating the Objective 

One last challenge remains: $\phi$ appears in the distribution underlying the inner expectation, not just in the integrand. This setting departs from standard empirical risk minimization (ERM) in supervised learning where the distribution of the data (e.g., cats and dogs in image classification) remains fixed regardless of model parameters. Here, however, the "data" - our sampled actions - depends directly on the parameters $\phi$ we're trying to optimize.

This dependence prevents us from simply using sample average estimators and differentiating through them, as we typically do in supervised learning. The challenge of correctly and efficiently estimating such derivatives has been extensively studied in the simulation literature under the umbrella of "derivative estimation." SAC adopts a particular solution known as the reparameterization trick in deep learning (or the IPA estimator in simulation literature). This approach transforms the problem by pushing $\phi$ inside the expectation through a change of variables.

To address this, we can express our Gaussian policy through a deterministic function $f_\phi$ that transforms noise samples to actions:

$$
f_\phi(s,\epsilon) = \mu_\phi(s) + \sigma_\phi(s)\epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
$$

This transformation allows us to rewrite our objective using an expectation over the fixed noise distribution:

$$
\begin{align*}
&\mathbb{E}_{s \sim \mu(s)}\mathbb{E}_{\epsilon \sim \mathcal{N}(0,1)}[\log d(f_\phi(s,\epsilon)|s;\phi) - \frac{1}{\beta}q_\theta(s,f_\phi(s,\epsilon))]
\end{align*}
$$


Now $\phi$ appears only in the integrand through the function $f_\phi$, not in the sampling distribution. The objective involves two terms. First, the log-probability of our Gaussian policy has a simple closed form:


$$
\log d(f_\phi(s,\epsilon)|s;\phi) = -\frac{1}{2}\log(2\pi\sigma_\phi(s)^2) - \frac{(f_\phi(s,\epsilon)-\mu_\phi(s))^2}{2\sigma_\phi(s)^2}
$$
Second, $\phi$ enters through the composition of $q^\star$ with $f_\phi$: $q^\star(s,f_\phi(s,\epsilon))$. The chain rule for this composition would involve derivatives of both functions. While this might be problematic if the Q-factors were to come from outside of our control (ie. not in the computational graph), but since SAC learns it simultaneously with the policy, then we can simply compute all required derivatives through automatic differentiation. 

This composition of policy and value functions - where $f_\phi$ enters as input to $q_\theta$ - directly parallels the structure we encountered in deterministic policy methods like NFQCA and DDPG. In those methods, we optimized:

$$
\max_{\phi} \mathbb{E}_{s \sim \mu(s)}[q_\theta(s, f_\phi(s))]
$$

where $f_\phi(s)$ was a deterministic policy. SAC extends this idea to stochastic policies by having $f_\phi$ transform both state and noise:

$$
\max_{\phi} \mathbb{E}_{s \sim \mu(s)}\mathbb{E}_{\epsilon \sim \mathcal{N}(0,1)}[q_\theta(s,f_\phi(s,\epsilon))]
$$

Thus, rather than learning a single action for each state as in DDPG, we learn a function that transforms random noise into actions, explicitly parameterizing a distribution over actions while maintaining the same underlying principle of differentiating through composed policy and value functions.