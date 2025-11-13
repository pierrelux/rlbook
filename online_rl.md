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

# Online Reinforcement Learning

The [previous chapter](batch_rl.md) established batch reinforcement learning algorithms that work with fixed offline datasets. This chapter extends those ideas to online learning, where we continuously collect new data while simultaneously improving our policy. We introduce replay buffers as the key data structure enabling this transition, develop the DQN algorithm and its variants, and examine the spectrum from sample average approximation to stochastic approximation.

## Online Data Collection and Replay Buffers

The [batch algorithms chapter](batch_rl.md) assumed a fixed transition dataset $\mathcal{D}$. In practice, we often want to collect data online while learning. This requires three modifications:

1. **Online data collection**: Use the current Q-function to act in the environment with an exploration strategy (e.g., $\varepsilon$-greedy)
2. **Replay buffer**: Store transitions in a finite-capacity buffer $\mathcal{R}$ rather than keeping all data
3. **Mini-batch sampling**: Sample a small batch from $\mathcal{R}$ at each step rather than using the full dataset

The replay buffer has a natural interpretation as a nonparametric model of the environment. The buffer $\mathcal{R} = \{(s_i, a_i, r_i, s'_i)\}$ represents an empirical approximation $\hat{p}(s', r | s, a)$ of the true transition distribution. When we sample from $\mathcal{R}$ to compute targets, we're performing Monte Carlo integration using this empirical distribution instead of the true one. This connects replay buffer methods to kernel-based RL {cite}`Ormoneit2002`, which makes the nonparametric modeling explicit.

However, the replay distribution differs from the current policy's state-action visitation. Early transitions come from an exploratory policy; later we sample these alongside new transitions from the improved policy. The buffer contains a mixture that doesn't match any single policy. This is what makes replay buffer learning off-policy.

Putting these pieces together yields the Deep Q-Network algorithm:

```{prf:algorithm} Deep Q-Network (DQN)
:label: dqn

**Input**: MDP $(S, A, P, R, \gamma)$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, target update frequency $K$, replay buffer size $B$, mini-batch size $b$

**Output**: Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_0$
3. Initialize replay buffer $\mathcal{R}$ with capacity $B$
4. $n \leftarrow 0$
5. **while** training:
    1. Observe current state $s$
    2. Select action $a$ using $\varepsilon$-greedy policy: $a = \begin{cases} \arg\max_{a'} q(s,a';\boldsymbol{\theta}_n) & \text{with probability } 1-\varepsilon \\ \text{random action} & \text{with probability } \varepsilon \end{cases}$
    3. Execute $a$, observe reward $r$ and next state $s'$
    4. Store $(s,a,r,s')$ in $\mathcal{R}$, replacing oldest if full
    5. Sample mini-batch of $b$ transitions $(s_i,a_i,r_i,s_i')$ from $\mathcal{R}$
    6. $\mathcal{D}_n^{\text{fit}} \leftarrow \emptyset$
    7. For each sampled $(s_i,a_i,r_i,s_i')$:
        1. $y_i \leftarrow r_i + \gamma \max_{a' \in A} q(s_i',a'; \boldsymbol{\theta}_{target})$
        2. $\mathcal{D}_n^{\text{fit}} \leftarrow \mathcal{D}_n^{\text{fit}} \cup \{((s_i,a_i), y_i)\}$
    8. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_n - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_n; \mathcal{D}_n^{\text{fit}})$
    9. If $n \bmod K = 0$:
        1. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_n$
    10. $n \leftarrow n + 1$
6. **return** $\boldsymbol{\theta}_n$
```

Note the replay ratio implicit in this algorithm: for each new transition we collect (step 5.4), we sample a mini-batch of size $b$ (step 5.5) and perform one update. We're reusing past experiences at a ratio of $b$:1. This ratio is a tunable hyperparameter. Higher ratios mean more computation per environment step but better data efficiency.

## Decoupling Selection and Evaluation in DQN

As discussed in the [batch RL chapter](batch_rl.md), the max operator in target computation can lead to overestimation: $y_i = r_i + \gamma \max_{a'} q(s_i',a'; \boldsymbol{\theta}_{target})$. We use the same network to both select the best-looking action and evaluate it, potentially compounding estimation errors.

Double DQN {cite}`van2016deep` addresses this by using the current network to select actions but the target network to evaluate them:

```{prf:algorithm} Double Deep Q-Network (Double DQN)
:label: double-dqn

**Input**: MDP $(S, A, P, R, \gamma)$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, target update frequency $K$, replay buffer size $B$, mini-batch size $b$

**Output**: Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_0$
3. Initialize replay buffer $\mathcal{R}$ with capacity $B$
4. $n \leftarrow 0$
5. **while** training:
    1. Observe current state $s$
    2. Select action $a$ using $\varepsilon$-greedy policy based on $q(s,\cdot;\boldsymbol{\theta}_n)$
    3. Execute $a$, observe reward $r$ and next state $s'$
    4. Store $(s,a,r,s')$ in $\mathcal{R}$, replacing oldest if full
    5. Sample mini-batch of $b$ transitions $(s_i,a_i,r_i,s_i')$ from $\mathcal{R}$
    6. $\mathcal{D}_n^{\text{fit}} \leftarrow \emptyset$
    7. For each sampled $(s_i,a_i,r_i,s_i')$:
        1. $a^*_i \leftarrow \arg\max_{a' \in A} q(s_i',a'; \boldsymbol{\theta}_n)$
        2. $y_i \leftarrow r_i + \gamma q(s_i',a^*_i; \boldsymbol{\theta}_{target})$
        3. $\mathcal{D}_n^{\text{fit}} \leftarrow \mathcal{D}_n^{\text{fit}} \cup \{((s_i,a_i), y_i)\}$
    8. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_n - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_n; \mathcal{D}_n^{\text{fit}})$
    9. If $n \bmod K = 0$:
        1. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_n$
    10. $n \leftarrow n + 1$
6. **return** $\boldsymbol{\theta}_n$
```

The key change is in step 5.7: we first select the action using our current network $\boldsymbol{\theta}_n$ (step 5.7.1), then evaluate that specific action using the target network $\boldsymbol{\theta}_{target}$ (step 5.7.2). Since the two networks differ (the target updates less frequently), their estimation errors are less correlated, mitigating the overestimation bias.

## Smooth Bellman Operators and Entropy Regularization

The [chapter on smooth Bellman optimality equations](regmdp.md) showed that replacing $\max_a$ with $\frac{1}{\beta} \log \sum_a \exp(\beta \cdot)$ yields a smooth operator corresponding to entropy-regularized optimization. Applying the fitted Q-iteration framework with this smooth operator requires only one change: replace the target computation

$$
y_i = r_i + \gamma \max_{a'} q(s'_i, a'; \theta^{(k)})
$$

with the smooth version

$$
y_i = r_i + \gamma \frac{1}{\beta} \log \sum_{a'} \exp(\beta q(s'_i, a'; \theta^{(k)})).
$$

The regression problem $\min_\theta \sum_i \ell(q(s_i, a_i; \theta), y_i)$ remains unchanged. The learned policy is the softmax $\pi(a|s) = \exp(\beta q(s,a))/\sum_{a'} \exp(\beta q(s,a'))$ rather than the greedy $\arg\max_a q(s,a)$.

For discrete action spaces, this modification is straightforward. The logsumexp can be computed stably using the log-sum-exp trick (subtract the maximum before exponentiating to prevent overflow), and gradient computation through the softmax is standard in neural network frameworks. Soft Q-learning applies this to the DQN architecture: use smooth targets, sample from the softmax policy for exploration, and maintain replay buffers and target networks as in DQN.

For continuous action spaces, evaluating the logsumexp requires integration or sampling over the continuous action set, which is expensive. This motivated the development of soft actor-critic {cite}`haarnoja2018soft`, which is discussed in the [continuous action chapter](cadp.md). The soft actor-critic approach uses the reparameterization trick and analytically tractable policy distributions (typically Gaussian) to compute the entropy term and its gradients efficiently.

## From Sample Average Approximation to Stochastic Approximation

The algorithms we have examined so far (FQI, NFQI, DQN) all solve optimization problems of the form $\min_\theta \frac{1}{N}\sum_{i=1}^N \ell_i(\theta)$, where $\ell_i(\theta)$ is the loss on the $i$-th sample (e.g., squared error, Huber loss). We compute the empirical average over a dataset or mini-batch, then minimize this average. This is sample average approximation (SAA): approximate an expectation with a sample average, then solve the resulting deterministic optimization problem.

Classical Q-learning {cite}`SuttonBarto2018` takes a different approach. Instead of accumulating samples and solving $\min_\theta \mathbb{E}[\ell(\theta)]$, it performs incremental updates:

$$
\theta \leftarrow \theta - \alpha \nabla_\theta \ell(\theta),
$$

using a single sample at a time. This is stochastic approximation (SA): directly update parameters using noisy gradient estimates without explicitly forming sample averages.

For Q-learning with a linear approximation $q(s,a; \theta) = \theta^\top \varphi(s,a)$, the update at transition $(s, a, r, s')$ is:

$$
\theta \leftarrow \theta + \alpha \left[r + \gamma \max_{a'} \theta^\top \varphi(s', a') - \theta^\top \varphi(s,a)\right] \varphi(s,a).
$$

This is a single gradient step on the squared TD error $(r + \gamma \max_{a'} q(s',a') - q(s,a))^2$ with step size $\alpha$. Each transition yields one update, and the parameter vector evolves continuously as new data arrives.

Comparing the approaches: SAA (FQI, NFQI, DQN) accumulates samples, computes targets, and solves $\min_\theta \sum_i (q(s_i, a_i; \theta) - y_i)^2$ via multiple gradient steps. SA (Q-learning) takes one gradient step per sample. The design choice is the number of inner-loop gradient steps: full convergence (SAA), many steps (DQN with $K$ steps per sample), or single step (SA). As $K$ decreases from full convergence to one, we transition from sample average approximation to stochastic approximation.

Stochastic approximation has a rich convergence theory. Under appropriate conditions on the step size sequence $\{\alpha_k\}$ (typically $\sum_k \alpha_k = \infty$ and $\sum_k \alpha_k^2 < \infty$, satisfied by $\alpha_k = 1/k$) and the sampling distribution, SA iterates converge almost surely to a local minimum. The ODE method provides a framework for analyzing these algorithms by relating the discrete stochastic updates to a continuous-time ordinary differential equation that describes the limiting behavior. We defer detailed convergence analysis to later treatment, but note that the distinction between SAA and SA is fundamental: SAA separates sampling from optimization while SA interleaves them completely.

Q-learning is thus the limiting case of our template: $K=1$ inner optimization step, online data collection (no replay buffer), and typically tabular or linear function approximation. Modern deep RL usually employs intermediate designs (DQN with $K=1$ to $4$ steps, replay buffers, neural networks) that blend SAA and SA characteristics.

## Synthesis and Connections

This chapter, together with the [simulation-based methods chapter](simadp.md) and [batch RL chapter](batch_rl.md), has shown how projection methods from the [projection methods chapter](projdp.md) combine with Monte Carlo integration to enable approximate dynamic programming from sampled experience. Q-function representations amortize action selection, and different design choices (function approximators, optimization strategies, data collection modes) generate the major value-based algorithms in reinforcement learning.

### Target Networks in Approximate Value Iteration

Understanding the "target network" in DQN requires understanding the nested loop structure of approximate value iteration. This structure was already present in batch methods like FQI and NFQI, but the 2013 deep learning RL community was primarily familiar with strict online methods like Q-learning that use $K=1$ gradient step per transition with no replay. The OR and approximate dynamic programming communities had used batch methods since the 1990s, but operated in separate research circles.

Consider the nested loop structure of fitted Q-iteration:

$$
\begin{aligned}
&\textbf{repeat } k = 0, 1, 2, \ldots \quad \text{\small // Value iteration (function iteration)} \\
&\quad \text{Compute targets: } y_i \leftarrow r_i + \gamma \max_{a'} q(s'_i, a'; \boldsymbol{\theta}_k) \text{ for all } i \\
&\quad \boldsymbol{\theta}_{k+1} \leftarrow \texttt{fit}\left(\{((s_i, a_i), y_i)\}, \boldsymbol{\theta}_k, K\right) \quad \text{\small // Solve regression problem} \\
&\textbf{until } \text{convergence}
\end{aligned}
$$

where $\texttt{fit}$ minimizes

$$
\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N}\sum_i \ell(q(s_i, a_i; \boldsymbol{\theta}), y_i)
$$

using $K$ gradient descent steps (potentially $K=\infty$, running to convergence). The loss $\ell$ is typically squared error, Huber loss, or another regression loss.

This is not ordinary supervised learning. We solve a sequence of regression problems, not a single one. At each outer iteration, we generate new targets $\{y_i\}$ from our current Q-function, creating a moving target. The targets $\{y_i\}$ must remain fixed during the inner optimization. This defines what it means to apply the Bellman operator and then project. We compute them once using $\boldsymbol{\theta}_k$, then solve the regression problem with these fixed targets.

When we flatten this nested structure into a single loop (convenient for online implementation), we maintain a separate "target network" $\boldsymbol{\theta}_{\text{target}}$ that plays the role of $\boldsymbol{\theta}_k$: it provides fixed targets for $K$ gradient steps, then synchronizes via $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}$. This periodic update marks the outer iteration boundary. The target network is not a design choice but what approximate value iteration means when written as a single loop. Without it (recomputing targets after every gradient step), we would be solving a different fixed-point problem (residual minimization) that requires different analysis.

The gradient $\partial \mathcal{L}/\partial \boldsymbol{\theta}$ automatically treats $y_i$ as constants because they are constants (computed once per outer iteration). No "stop_gradient" operation is needed; the partial derivative captures this naturally. The computational graph perspective that requires explicit annotations conflates partial and total derivatives.

The DQN paper acknowledges NFQ as "perhaps the most similar prior work" but characterizes it as using "batch updates" with cost proportional to dataset size, contrasting with their "stochastic gradient updates." Both methods actually use mini-batch gradients. The difference is that NFQ runs optimization to convergence ($K=\infty$, many epochs through the data) while DQN truncates to $K=1$ (single gradient step per environment interaction). DQN demonstrated that this truncation works well when combined with large replay buffers, deep convolutional networks, and careful engineering, achieving success on high-dimensional Atari games. This was a significant empirical achievement that brought approximate dynamic programming methods to the attention of the deep learning community.

The algorithm space is defined by four independent design choices: (1) truncation parameter $K$, (2) data collection mode (offline, online with replay, pure online), (3) initialization (cold or warm start), and (4) operator choice (hard max or smooth). Target networks are the implementation of approximate value iteration's nested structure, not a fifth design dimension.

### Broader Context

The methods developed across these three chapters search in the space of value functions. The [projection methods chapter](projdp.md) established the projection framework with exact operator evaluations. The [simulation-based chapter](simadp.md) extended it to Monte Carlo integration, the [batch RL chapter](batch_rl.md) developed offline algorithms, and this chapter completed the picture with online learning. The [next chapter](cadp.md) takes a complementary approach: rather than searching for value functions and extracting policies through maximization, we directly parameterize and optimize the policy itself. Just as value-based methods progressed from exact evaluation to Monte Carlo sampling, policy parametrization methods face the same challenge of evaluating expectations, leading to similar Monte Carlo techniques applied to policy gradients rather than Bellman operators.

