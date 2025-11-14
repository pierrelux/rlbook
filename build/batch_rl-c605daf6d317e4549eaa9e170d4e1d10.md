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

# Batch Reinforcement Learning

The [previous chapter](simadp.md) established the theoretical foundations of simulation-based approximate dynamic programming: Monte Carlo integration for evaluating expectations, Q-functions for efficient action selection, and techniques for mitigating overestimation bias. Those developments assumed we could sample freely from transition distributions and choose optimization parameters without constraint. This chapter grounds those ideas in practical algorithms by starting with the batch (offline) setting, where we learn from a fixed dataset of transitions, and then developing a unified framework that encompasses both offline and online methods through systematic variations in data collection, optimization strategy, and function approximation.

## A Unified View of Reinforcement Learning Algorithms

All value-based reinforcement learning algorithms share the same two-level structure. At iteration $n$, the outer loop applies the Bellman operator $\Bellman$ to construct targets $y_i^{(n)} = (\widehat{\Bellman}_N q)(s_i, a_i; \boldsymbol{\theta}^{(n)})$ at observed state-action pairs $(s_i, a_i)$, where $\widehat{\Bellman}_N$ denotes Monte Carlo approximation with $N$ samples from the transition distribution. The inner loop solves the regression problem $\min_{\boldsymbol{\theta}} \sum_i \ell(q(s_i, a_i; \boldsymbol{\theta}), y_i^{(n)})$ to find parameters that match these targets. We can write this abstractly as:

$$
\begin{aligned}
&\textbf{repeat } n = 0, 1, 2, \ldots \\
&\quad \text{Construct } (X^{(n)}, y^{(n)}) \text{ where } y_i^{(n)} = (\widehat{\Bellman}_N q)(s_i, a_i; \boldsymbol{\theta}^{(n)}) \text{ for each } (s_i, a_i) \in X^{(n)} \\
&\quad \boldsymbol{\theta}^{(n+1)} \leftarrow \texttt{fit}(X^{(n)}, y^{(n)}, \boldsymbol{\theta}_{\text{init}}, K) \\
&\textbf{until } \text{convergence}
\end{aligned}
$$

The `fit` operation minimizes the regression loss using $K$ optimization steps (gradient descent for neural networks, tree construction for ensembles, matrix inversion for linear models) starting from initialization $\boldsymbol{\theta}_{\text{init}}$. Standard supervised learning uses random initialization ($\boldsymbol{\theta}_{\text{init}} = \boldsymbol{\theta}_0$) and runs to convergence ($K = \infty$). Reinforcement learning algorithms vary these choices: warm starting from the previous iteration ($\boldsymbol{\theta}_{\text{init}} = \boldsymbol{\theta}^{(n)}$), partial optimization ($K \in \{10, \ldots, 100\}$), or single-step updates ($K=1$).

The input points $X^{(n)}$ may stay fixed (batch setting) or change (online setting), but the targets $y^{(n)}$ always change because they depend on the evolving Q-function. In practice, we typically have $N=1$ (single observed next state per transition), giving $(\widehat{\Bellman}_1 q)(s_i, a_i; \boldsymbol{\theta}^{(n)}) = r_i + \gamma \max_{a'} q(s'_i, a'; \boldsymbol{\theta}^{(n)})$ for transition tuples $(s_i, a_i, r_i, s'_i)$.

### Design Choices and Algorithm Space

This template provides a blueprint for instantiating concrete algorithms. Six design axes generate algorithmic diversity: the function approximator (trees, neural networks, linear models), the Bellman operator (hard max vs smooth logsumexp), the inner optimization strategy (full convergence, $K$ steps, or single step), the initialization scheme (cold vs warm start), the data collection mechanism (offline, online, replay buffer), and bias mitigation approaches (none, double Q-learning, learned correction). While individual algorithms include additional refinements, these axes capture the primary sources of variation. The table below shows how several well-known methods instantiate this template:

| **Algorithm** | **Approximator** | **Bellman** | **Inner Loop** | **Initialization** | **Data** | **Bias Fix** |
|:--------------|:-----------------|:------------|:---------------|:-------------------|:---------|:-------------|
| FQI {cite}`ernst2005tree` | Extra Trees | Hard | Full | Cold | Offline | None |
| NFQI {cite}`riedmiller2005neural` | Neural Net | Hard | Full | Warm | Offline | None |
| DQN {cite}`mnih2013atari` | Deep NN | Hard | K=1 | Warm | Replay | None |
| Double DQN {cite}`van2016deep` | Deep NN | Hard | K=1 | Warm | Replay | Double Q |
| Q-learning {cite}`SuttonBarto2018` | Tabular/Linear | Hard | K=1 | Warm | Online | None |
| Soft Q {cite}`haarnoja2017reinforcement` | Neural Net | Smooth | K steps | Warm | Replay | None |

This table omits continuous action methods (NFQCA, DDPG, SAC), which introduce an additional design dimension. We address those in the [continuous action chapter](cadp.md). The initialization choice becomes particularly important when moving from batch to online algorithms.

### The Empirical Distribution Perspective: Unifying Offline and Online Learning

Both offline and online reinforcement learning operate by repeatedly solving supervised learning problems where training data is sampled from observed transitions. Given a dataset $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$ of observed transitions, the **empirical distribution** is simply the uniform distribution over these $N$ tuples:

$$
\hat{P}_{\mathcal{D}} = \text{Uniform}(\mathcal{D})
$$

Operationally, sampling from $\hat{P}_{\mathcal{D}}$ means: pick an index $i$ uniformly at random from $\{1, \ldots, N\}$ and use transition $(s_i, a_i, r_i, s'_i)$. This is our approximation to the unknown true distribution from which the data was generated.

This connects to the statistical bootstrap principle: we treat the empirical distribution over observed data as if it were the true population distribution. By sampling from $\mathcal{D}$, we implicitly assume our finite sample represents the environment's true transition dynamics $P(r, s' | s, a)$ weighted by the state-action visitation of whatever policy generated the data.

```{prf:remark} Mathematical Formulation of Empirical Distributions
:class: dropdown

More formally, $\hat{P}_{\mathcal{D}}$ is a discrete probability measure defined by $\hat{P}_{\mathcal{D}} = \frac{1}{N}\sum_{i=1}^N \delta_{\tau_i}$, where $\tau_i = (s_i, a_i, r_i, s'_i)$ and $\delta_{\tau_i}$ denotes the Dirac point mass at transition $\tau_i$. This is a discrete distribution over $N$ points regardless of whether the state and action spaces are continuous or discrete. For any measurable set $A \subseteq \mathcal{S} \times \mathcal{A} \times \mathbb{R} \times \mathcal{S}$:

$$
\hat{P}_{\mathcal{D}}(A) = \frac{1}{N}\sum_{i=1}^N \mathbb{1}[\tau_i \in A] = \frac{|\{i : \tau_i \in A\}|}{N}
$$

The empirical distribution assigns probability $1/N$ to each observed tuple and probability zero everywhere else.
```

#### Two Sources of Approximation

When we evaluate the Bellman operator using transition data, we face two distinct approximations:

1. **Monte Carlo approximation of expectations**: For a given $(s, a)$ pair, the Bellman operator requires computing $\mathbb{E}_{s' \sim P(\cdot|s,a)}[r + \gamma \max_{a'} q(s', a')]$. In practice, we have $N=1$ next-state sample per transition tuple, so for $(s_i, a_i, r_i, s'_i)$:

$$
y_i = r_i + \gamma \max_{a'} q(s'_i, a'; \boldsymbol{\theta}^{(n)})
$$

This provides an unbiased Monte Carlo estimate of $(\Bellman q)(s_i, a_i)$ for the specific state-action pair $(s_i, a_i)$, assuming $s'_i$ was drawn from the true $P(\cdot | s_i, a_i)$.

2. **Empirical distribution approximation**: The collection of state-action pairs $\{(s_i, a_i)\}$ we evaluate at comes from $\hat{P}_{\mathcal{D}}$, which may poorly represent the states we actually care about (e.g., those visited by the optimal policy, or those under the MDP's stationary distribution).

The distinction between offline and online learning reduces to whether $\hat{P}_{\mathcal{D}}$ remains static or evolves:

**Offline (fixed empirical distribution)**: Dataset $\mathcal{D}$ is collected once and never changes. At each iteration $n$, we sample transitions from $\mathcal{D}$ (typically the full dataset, or mini-batches sampled uniformly), compute targets using the current Q-function, and fit. This is sample average approximation: repeatedly solving optimization problems defined by the same fixed empirical distribution, with targets that change as $q(\cdot; \boldsymbol{\theta}^{(n)})$ evolves.

**Online (dynamic empirical distribution)**: Dataset grows as $\mathcal{D}_t = \mathcal{D}_{t-1} \cup \{(s_t, a_t, r_t, s'_{t+1})\}$ by adding newly observed transitions. At each step, we sample from the current $\mathcal{D}_t$ to form targets. The empirical distribution shifts continuously as new data arrives. This is stochastic approximation with a time-varying empirical distribution.

A replay buffer implements efficient sampling from the empirical distribution with finite capacity. When capacity is reached, old transitions are discarded (typically FIFO), so the empirical distribution represents a sliding window over recent experience. Three cases span the design space:

- **Classical FQI**: Fixed $\mathcal{D}$, use entire dataset each outer iteration (no sampling needed), run inner loop to convergence ($K=\infty$). Pure sample average approximation.
- **DQN**: Growing $\mathcal{D}_t$ with finite capacity $B$, sample mini-batch of size $b < B$ uniformly with replacement from buffer each step, single gradient step ($K=1$). Stochastic approximation on a time-varying empirical distribution with data reuse.
- **Online Q-learning**: Capacity $= 1$, use each newly observed transition exactly once for one gradient step, then discard. Pure stochastic approximation without reuse.

The replay ratio (number of optimization steps per environment transition) quantifies data reuse. For a dataset of size $N$ and mini-batch size $b$:

- Offline: replay ratio $\approx K \cdot N_{\text{epochs}} \cdot N / b$
- DQN: replay ratio $= b$ (one gradient step per new transition, using batch of $b$ samples)
- Online Q-learning: replay ratio $= 1$

Higher replay ratios amortize environment interaction costs but increase the risk of overfitting to the empirical distribution, especially when it's based on outdated policies. The choice of $K$ (optimization steps per outer iteration) and buffer capacity jointly control the position along the SAA-SA continuum.

```{prf:remark} Two Meanings of "Bootstrap" in Reinforcement Learning
:class: dropdown

The term "bootstrap" appears in reinforcement learning with two distinct meanings that should not be confused:

1. **Statistical bootstrapping** (this section): Resampling from the empirical distribution $\hat{P}_{\mathcal{D}}$ to approximate expectations. We sample observed transition tuples $(s_i, a_i, r_i, s'_i)$ uniformly from our dataset. This is the classical bootstrap from statistics: using the empirical distribution as a proxy for the true population distribution.

2. **Temporal difference bootstrapping**: Using current value estimates $q(s', a')$ to form targets, rather than complete Monte Carlo returns. When we write $y_i = r_i + \gamma \max_{a'} q(s'_i, a')$, we "bootstrap" from the estimated value at $s'_i$ instead of rolling out a full trajectory.

Both forms of bootstrapping are present simultaneously in fitted Q-iteration: we resample transitions from the empirical distribution (statistical bootstrap) and use Q-function estimates in targets (TD bootstrap). The two concepts are independent—we could have statistical bootstrapping without TD bootstrapping (sample from $\hat{P}_{\mathcal{D}}$ but use full Monte Carlo returns) or TD bootstrapping without statistical bootstrapping (use a fixed set of transitions without resampling).
```

```{prf:remark} Notation: Transition Data vs Regression Data
:class: dropdown

We maintain a careful distinction between two types of datasets throughout:

- **Transition dataset** $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}$: Raw experience tuples collected from the environment (offline) or stored in replay buffer (online). This is the "environment data" that never changes once collected.

- **Regression dataset** $\mathcal{D}_n^{\text{fit}} = \{((s_i, a_i), y_i)\}$: Supervised learning data with state-action pairs and computed target values $y_i$. This changes every outer iteration $n$ as we recompute targets using the current Q-function.

The relationship: at iteration $n$, we construct $\mathcal{D}_n^{\text{fit}}$ from $\mathcal{D}$ (or from buffer sample $\mathcal{B}.\text{sample}(b)$) by evaluating the Bellman operator:
$$
y_i = r_i + \gamma \max_{a'} q(s'_i, a'; \boldsymbol{\theta}_n) \quad \text{or} \quad y_i = r_i + \gamma \frac{1}{\beta}\log\sum_{a'} \exp(\beta q(s'_i, a'; \boldsymbol{\theta}_n))
$$

This distinction matters pedagogically: the **transition data** is fixed (offline) or evolves via online collection, while the **regression data** evolves via target recomputation. Fitted Q-iteration is the outer loop over target recomputation, not the inner loop over gradient steps.
```

The `fit` operation itself has the following signature and semantics:

```{prf:algorithm} Fit Operation
:label: fit-operation

**Input**: Regression dataset $\mathcal{D}^{\text{fit}} = \{((s_i, a_i), y_i)\}$, initialization $\boldsymbol{\theta}_{\text{init}}$, truncation steps $K \in \mathbb{N} \cup \{\infty\}$, learning rate $\alpha$, loss function $\mathcal{L}$

**Output**: Parameters $\boldsymbol{\theta}$ minimizing $\mathcal{L}(\boldsymbol{\theta}; \mathcal{D}^{\text{fit}}) = \frac{1}{|\mathcal{D}^{\text{fit}}|}\sum_{((s,a), y) \in \mathcal{D}^{\text{fit}}} \ell(q(s,a; \boldsymbol{\theta}), y)$

1. $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}_{\text{init}}$
2. $k \leftarrow 0$
3. **if** $K = \infty$:
    1. **repeat**
        1. $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}; \mathcal{D}^{\text{fit}})$
        2. $k \leftarrow k + 1$
    2. **until** convergence
4. **else**:
    1. **for** $k = 0$ to $K-1$:
        1. $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}; \mathcal{D}^{\text{fit}})$
5. **return** $\boldsymbol{\theta}$
```

This makes explicit the three key parameters:
- **Initialization** $\boldsymbol{\theta}_{\text{init}}$: Cold start ($\boldsymbol{\theta}_0$) resets to a fixed point each iteration; warm start ($\boldsymbol{\theta}_n$) continues from the previous solution
- **Truncation** $K$: Controls SAA ($K=\infty$), partial optimization ($K \in \{10, \ldots, 100\}$), or stochastic approximation ($K=1$)
- **Regression dataset** $\mathcal{D}^{\text{fit}}$: Supervised learning data with state-action pairs and computed targets, derived from transition data in buffer

## Loss Functions and Statistical Perspectives on Fitted Q-Iteration

### Statistical Interpretation of Loss Functions

The `fit` operation takes a loss function $\mathcal{L}$ as a parameter, which we have been treating as squared error by default. The loss function encodes assumptions about the statistical relationship between state-action pairs and the Bellman targets we are trying to match. Different loss functions correspond to different implicit noise models, and in deep reinforcement learning, where targets are noisy and network capacity is high, the choice of loss can substantially affect performance.

The standard squared error loss $\ell(q(s,a; \boldsymbol{\theta}), y) = (q(s,a; \boldsymbol{\theta}) - y)^2$ corresponds to maximum likelihood estimation under the assumption that targets are corrupted by additive Gaussian noise. If we write the relationship as $y = q^*(s,a) + \varepsilon$ where $\varepsilon \sim \mathcal{N}(0, \sigma^2)$, minimizing squared error is equivalent to maximizing the likelihood of the observed targets. This statistical interpretation is standard in regression: L2 loss assumes Gaussian noise.

In reinforcement learning, however, the "noise" in our targets comes from multiple sources with different characteristics. Monte Carlo sampling contributes variance that may have heavy tails, especially early in learning when the policy explores poorly. Function approximation error introduces bias. Bootstrap targets compound errors recursively. The resulting noise distribution need not be Gaussian. Outliers from bad exploration or initialization can create targets with large errors, and squared loss penalizes these quadratically, potentially dominating the gradient signal. Other score functions, designed for robustness to non-Gaussian noise, may perform better.

### Classification-Based Q-Learning

One approach that has shown substantial empirical gains in deep Q-learning is to replace scalar regression with classification. Instead of predicting a single real-valued Q-value, we predict a probability distribution over a discrete set of possible values and use cross-entropy loss. Q-values are continuous quantities, but the discrete representation can be viewed as a form of quantization that provides robustness to the noise structure in TD learning.

To make this concrete, suppose we choose a finite grid of values $z_1 < z_2 < \cdots < z_K$ spanning the range of plausible returns. Instead of outputting a scalar $q(s,a; \boldsymbol{\theta}) \in \mathbb{R}$, the network outputs logits $\ell_{\boldsymbol{\theta}}(s,a) \in \mathbb{R}^K$, which are converted to probabilities via softmax:

$$
p_{\boldsymbol{\theta}}(k \mid s,a) = \frac{\exp(\ell_{\boldsymbol{\theta}}(s,a)_k)}{\sum_{j=1}^K \exp(\ell_{\boldsymbol{\theta}}(s,a)_j)}.
$$

The Q-value is then the expected value under this categorical distribution:

$$
q(s,a; \boldsymbol{\theta}) = \sum_{k=1}^K z_k \, p_{\boldsymbol{\theta}}(k \mid s,a).
$$

This representation expresses a scalar as the mean of a discrete distribution. The function approximator learns the distribution, and the Q-value emerges as a derived quantity.

Training requires converting each scalar TD target $y_i$ into a target distribution $q(\cdot \mid y_i)$ over the bins. The simplest and most direct encoding is the two-hot representation. If the target $y_i$ falls between bins $z_j$ and $z_{j+1}$, we place probability mass on these two neighboring bins proportional to how close $y_i$ is to each:

$$
q_j(y_i) = \frac{z_{j+1} - y_i}{z_{j+1} - z_j}, \quad q_{j+1}(y_i) = \frac{y_i - z_j}{z_{j+1} - z_j},
$$

with $q_k(y_i) = 0$ for all other bins $k \notin \{j, j+1\}$. This is simply linear interpolation expressed as a categorical distribution. The expectation $\sum_k z_k q_k(y_i) = y_i$ recovers the original scalar target exactly. The loss becomes cross-entropy between the target distribution and the predicted distribution:

$$
\mathcal{L}_{\text{CE}}(\boldsymbol{\theta}; \mathcal{D}^{\text{fit}}) = -\frac{1}{|\mathcal{D}^{\text{fit}}|} \sum_{((s_i,a_i), y_i) \in \mathcal{D}^{\text{fit}}} \sum_{k=1}^K q_k(y_i) \log p_{\boldsymbol{\theta}}(k \mid s_i, a_i).
$$

This formulation replaces the scalar prediction problem with a categorical one, and L2 loss with cross-entropy. The two-hot encoding ensures that no information is lost in the conversion: the target distribution's mean equals the original scalar target.

#### Connection to Monotone Approximators

The two-hot encoding is not new mathematics. It is linear interpolation expressed in probabilistic language. The weights $q_j(y_i)$ and $q_{j+1}(y_i)$ are barycentric coordinates: non-negative weights summing to one that represent $y_i$ as a convex combination of the neighboring grid points $z_j$ and $z_{j+1}$. This is identical to the linear interpolation formula used in the [dynamic programming chapter](dp.md) for handling continuous state spaces (see Algorithm {prf:ref}`backward-recursion-interp`). When we discretize a continuous state space and interpolate between grid points, we compute exactly these weights to estimate the value function at off-grid states.

This connection places the two-hot encoding within the framework developed in the [projection methods chapter](projdp.md). The target construction satisfies the conditions for a monotone approximator in Gordon's sense: it maps value functions to value functions using a stochastic matrix with non-negative entries and rows summing to one (Definition {prf:ref}`gordon-averager`). Such operators preserve the contraction property of the Bellman operator (Theorem {prf:ref}`gordon-stability`), guaranteeing convergence of value iteration. As shown in the monotone approximator analysis, piecewise linear interpolation always provides stability because the barycentric weights are non-negative and sum to one. The two-hot encoding inherits this structure: the target at bin $k$ is a weighted average of the neighbors, bounded by the minimum and maximum values in the local region.

The neural network $q(s,a; \boldsymbol{\theta})$ that predicts the categorical distribution, however, does not preserve monotonicity. The function class is highly flexible: learned features, nonlinear activations, and high-dimensional parameter spaces provide no guarantees about order preservation or bounded sensitivity. What we have in classification-based Q-learning is a hybrid: monotone targets paired with a non-monotone function approximator. The target distribution $q(\cdot \mid y_i)$ has the averager structure from classical dynamic programming theory, while the predicted distribution $p_{\boldsymbol{\theta}}(\cdot \mid s_i, a_i)$ comes from a flexible learned representation.

This asymmetry suggests an interpretation. Classical dynamic programming theory requires monotone approximators throughout to guarantee convergence: both the projection step and the function representation must preserve contraction. Deep reinforcement learning relaxes this by using non-monotone function classes but retains structure on the target side. The two-hot encoding provides implicit regularization through its interpolation structure: targets are smooth, locally bounded, and order-preserving, even though the approximator learns arbitrary features. This may partially explain why classification loss stabilizes learning beyond the cross-entropy robustness discussed below. The target encoding imposes geometric constraints inherited from monotone approximation theory, while the neural network retains the flexibility to represent complex value functions.

#### Alternative Encodings and Distributional Connections

An alternative to the two-hot encoding is the histogram loss with Gaussian smoothing (HL-Gauss), which treats the scalar target $y_i$ as the mean of a Gaussian and projects that Gaussian onto the histogram bins to produce a smoothed categorical distribution. This adds a form of label smoothing: instead of placing mass only on the two immediate neighbors, the Gaussian tail spreads small amounts of probability to nearby bins. This regularization can improve generalization, particularly when targets are noisy.

The classification approach has connections to distributional reinforcement learning, particularly the C51 algorithm {cite}`bellemare2017distributional`, which applies the Bellman operator to entire return distributions rather than scalar expectations. C51 uses a categorical representation and KL divergence (equivalently, cross-entropy) between distributions. However, the classification framework can be applied without adopting the full distributional Bellman machinery. Using cross-entropy loss on scalar TD targets already provides substantial benefits, suggesting that the loss geometry itself, not just the distributional perspective, contributes to improved performance.

### Advantages of Cross-Entropy Loss

Cross-entropy loss provides several advantages over squared error in the deep Q-learning setting. First, it is more robust to outliers. Squared error penalizes large deviations quadratically, so a single bad target with $|y_i - q(s_i,a_i; \boldsymbol{\theta})| = 100$ contributes 10,000 to the loss. Cross-entropy, by contrast, penalizes incorrect categorical predictions logarithmically. A target that falls in a low-probability region of the predicted distribution incurs a large but bounded gradient. This reduced sensitivity to extreme TD errors can stabilize learning when early targets are unreliable.

Second, the categorical representation encodes ordinal structure. Neighboring bins represent similar values, and the two-hot or Gaussian-smoothed encodings ensure that nearby bins receive similar probabilities. This is a form of structured prediction: the network learns that a Q-value of 5.1 is similar to 5.0 and 4.9, which L2 regression does not explicitly encode. The ordinal structure acts as an inductive bias that can improve sample efficiency.

Third, cross-entropy loss has been shown to scale better with network capacity than squared error in deep Q-learning. Farebrother et al. {cite}`farebrother2024stop` found that standard DQN and offline methods like CQL degrade when the Q-network is scaled up to large ResNets or mixture-of-experts architectures, but using classification loss (specifically HL-Gauss) maintains or improves performance as network size increases. This suggests that L2 loss may be overfitting to noise in the targets when given high-capacity approximators, while cross-entropy's implicit regularization prevents this degradation. The same study showed gains across online Atari, offline Atari, and robotic manipulation tasks.

This provides a third approach to mitigating the overestimation bias discussed earlier in this chapter. Keane-Wolpin bias correction explicitly learns and subtracts the bias term, and double Q-learning decouples selection from evaluation to eliminate evaluation bias. Classification loss takes a different route: it changes the projection geometry itself. Instead of projecting onto the function class using L2 distance (which amplifies outliers), we project using KL divergence (cross-entropy), which provides robustness to the noisy target distribution. All three approaches address the same fundamental problem—noisy, biased targets—but through different mechanisms.

### Practical Implementation

Practical implementation requires choosing the number of bins $K$ and their range $[z_1, z_K]$. Typical choices use $K \in \{51, 101, 201\}$ bins uniformly spaced over the expected return range. The range can be estimated from domain knowledge or learned adaptively during training. The network architecture changes minimally: instead of a single scalar output per action, we output $K$ logits per action. For discrete action spaces with $|\mathcal{A}|$ actions, this means a final layer of size $K \times |\mathcal{A}|$ rather than $|\mathcal{A}|$. The computational overhead is modest, and the implementation can use standard cross-entropy loss functions available in deep learning libraries. The main conceptual shift is viewing Q-learning as categorical prediction rather than scalar regression.

## Unified Q-Iteration Template

With these components, we can state the unified template that encompasses all value-based reinforcement learning algorithms:

```{prf:algorithm} Unified Simulation-Based Q-Iteration Template
:label: unified-q-iteration

**Input**: MDP $(S, A, P, R, \gamma)$, Q-function $q(s,a; \boldsymbol{\theta})$, Bellman operator type $\mathcal{T} \in \{\max, \text{softmax}\}$, initial parameters $\boldsymbol{\theta}_0$, offline transition dataset $\mathcal{D}$ (or $\emptyset$), buffer capacity $B$, mini-batch size $b$, truncation steps $K$, warmstart flag $w \in \{\text{cold}, \text{warm}\}$, online collection flag $c \in \{\text{offline}, \text{online}\}$

**Output**: Learned Q-function parameters $\boldsymbol{\theta}$

1. Initialize $\boldsymbol{\theta}_0$
2. Initialize buffer $\mathcal{B} \leftarrow \mathcal{D}$ with capacity $B$ (or $\mathcal{B} \leftarrow \emptyset$ if no offline data)
3. $n \leftarrow 0$
4. **while** training:
    1. **// Data collection (if online)**
    2. **if** $c = \text{online}$:
        1. Observe current state $s$
        2. Select action $a$ using exploration policy based on $q(s, \cdot; \boldsymbol{\theta}_n)$
        3. Execute $a$, observe $r, s'$
        4. $\mathcal{B}.\text{add}(s, a, r, s')$
    3. **// Sample mini-batch from buffer**
    4. $\{(s_i, a_i, r_i, s'_i)\}_{i=1}^b \leftarrow \mathcal{B}.\text{sample}(b)$
    5. **// Compute targets from transitions**
    6. $\mathcal{D}_n^{\text{fit}} \leftarrow \emptyset$
    7. **for** each sampled transition $(s_i, a_i, r_i, s'_i)$:
        1. $y_i \leftarrow r_i + \gamma \cdot \begin{cases} \max_{a'} q(s'_i, a'; \boldsymbol{\theta}_{\text{target}}) & \text{if } \mathcal{T} = \max \\ \frac{1}{\beta}\log\sum_{a'} \exp(\beta q(s'_i, a'; \boldsymbol{\theta}_{\text{target}})) & \text{if } \mathcal{T} = \text{softmax} \end{cases}$
        2. $\mathcal{D}_n^{\text{fit}} \leftarrow \mathcal{D}_n^{\text{fit}} \cup \{((s_i, a_i), y_i)\}$
    8. **// Fit Q-function with warmstart and truncation**
    9. $\boldsymbol{\theta}_{\text{init}} \leftarrow \begin{cases} \boldsymbol{\theta}_0 & \text{if } w = \text{cold} \\ \boldsymbol{\theta}_n & \text{if } w = \text{warm} \end{cases}$
    10. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}_n^{\text{fit}}, \boldsymbol{\theta}_{\text{init}}, K)$
    11. **// Update target network (if using)**
    12. Update $\boldsymbol{\theta}_{\text{target}}$ (e.g., $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_{n+1}$ periodically or via EMA)
    13. $n \leftarrow n + 1$
5. **return** $\boldsymbol{\theta}_n$
```

## Batch Algorithms: FQI and NFQI

The offline (batch) setting begins with a fixed dataset $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$ collected once before learning. This data might come from a previous controller, from human demonstrations, or from exploratory interactions. The task is to extract the best Q-function approximation from this data without additional environment interactions.

Fitted Q-iteration (FQI) and Neural Fitted Q-iteration (NFQI) are **approximate versions of Value Iteration**. They implement the classic successive approximation scheme $q^{(k+1)} = \Bellman q^{(k)}$ using function approximation and Monte Carlo integration. This creates a **nested two-loop structure**:

- **Outer loop** (Value Iteration): Compute Bellman targets from current Q-function, iterate until convergence
- **Inner loop** (Regression): Fit the function approximator to the targets, solving $\min_\theta \sum_i \ell(q(s_i, a_i; \theta), y_i)$

The outer loop is indexed by $k$ or $n$, the inner loop is the iterative optimization algorithm (gradient descent, tree training, etc.) that implements the projection onto the function class.

At iteration $k$, we have Q-function parameters $\theta^{(k)}$. For each transition $(s_i, a_i, r_i, s'_i)$ in $\mathcal{D}$, we compute a target value:

$$
y_i^{(k)} = r_i + \gamma \max_{a' \in \mathcal{A}} q(s'_i, a'; \theta^{(k)}).
$$

This evaluates the Bellman operator at the sampled next state $s'_i$. We then solve the regression problem:

$$
\theta^{(k+1)} = \arg\min_\theta \sum_{i=1}^N \left(q(s_i, a_i; \theta) - y_i^{(k)}\right)^2.
$$

The choice of function approximator determines how we solve this problem. Ernst et al. {cite}`ernst2005tree` used extremely randomized trees, an ensemble method that partitions the state-action space into regions and fits piecewise constant Q-values. Trees handle high-dimensional inputs naturally and the ensemble reduces overfitting. The resulting method, simply called FQI (Fitted Q-Iteration), demonstrated that batch reinforcement learning could work with complex function approximators on continuous-state problems.

Riedmiller {cite}`riedmiller2005neural` replaced the tree ensemble with a neural network, yielding Neural Fitted Q-Iteration (NFQI). The neural network $q(s,a; \theta)$ provides smooth interpolation and leverages gradient-based optimization (RProp, chosen for its insensitivity to hyperparameter choices). NFQI runs the inner optimization to convergence at each iteration: train the network until the loss stops decreasing, then compute new targets using the converged Q-function. This full optimization ensures the network accurately represents the projected Bellman operator before moving to the next iteration. NFQI uses warm starting: the network is initialized once at the beginning and continues learning from the previous iteration's weights rather than resetting. RProp itself operates on mini-batches or the full dataset per gradient update, but critically, NFQI performs many such updates (multiple epochs through the data) at each outer iteration until convergence. This is $K=\infty$ in our framework.

An important variant described by Riedmiller uses **incremental data collection**. After each outer iteration $k$, new episodes are collected using the current greedy policy $\pi_k(s) = \arg\max_a q(s,a;\theta_k)$, and these transitions are **added to the existing dataset** (not replacing it). Future iterations train on the accumulated set of all transitions collected so far. This sits between pure offline learning (fixed dataset collected once) and online learning (immediate use and discard). The growing dataset improves exploration beyond random actions while retaining the sample efficiency of batch learning. Riedmiller used this variant for the Mountain Car and Cart-Pole benchmarks where random exploration is insufficient.

For episodic tasks with goal and forbidden regions, Riedmiller uses a modified target structure. Let $S^+$ denote goal states and $S^-$ denote forbidden states. The targets become:

$$
y_i = \begin{cases}
c(s_i, a_i, s'_i) & \text{if } s'_i \in S^+ \text{ (goal reached)} \\
C^- & \text{if } s'_i \in S^- \text{ (forbidden state, typically } C^- = 1.0\text{)} \\
c(s_i, a_i, s'_i) + \gamma \max_{a'} q(s'_i, a'; \theta_k) & \text{otherwise}
\end{cases}
$$

Additionally, the **hint-to-goal heuristic** adds synthetic transitions $(s, a, s')$ where $s \in S^+$ with target value $c(s,a,s') = 0$ to explicitly clamp the Q-function to zero in the goal region. This stabilizes learning by encoding the boundary condition without requiring additional prior knowledge.

The following algorithm shows the basic offline structure:

```{prf:algorithm} Fitted Q-Iteration (Batch)
:label: fitted-q-iteration-batch

**Input:** Dataset $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$, function approximator class $q(s,a; \theta)$, discount factor $\gamma$, maximum iterations $K$, convergence tolerance $\varepsilon$

**Output:** Learned Q-function parameters $\theta$

1. Initialize $\theta_0$ (e.g., random initialization)
2. $k \leftarrow 0$
3. **repeat**  $\quad$ // **Outer loop: Value Iteration**
4. $\quad$ **// Apply Bellman operator: compute targets**
5. $\quad$ **for** each transition $(s_i, a_i, r_i, s'_i) \in \mathcal{D}$ **do**
6. $\quad\quad$ $y_i \leftarrow r_i + \gamma \max_{a' \in \mathcal{A}} q(s'_i, a'; \theta_k)$
7. $\quad$ **end for**
8. $\quad$ **// Inner loop: Fit Q-function to targets (projection step)**
9. $\quad$ $\theta_{k+1} \leftarrow \arg\min_\theta \sum_{i=1}^N \ell(q(s_i, a_i; \theta), y_i)$
10. $\quad$ $k \leftarrow k+1$
11. **until** $\|\theta_k - \theta_{k-1}\| < \varepsilon$ or $k \geq K$
12. **return** $\theta_k$
```

Line 9 hides an entire **inner optimization loop**. The loss $\ell$ is typically squared error $(q - y)^2$ but can be Huber loss (used in DQN) or other regression losses. For FQI with trees, this loop trains the ensemble until completion. For NFQI with neural networks, this loop runs gradient descent until convergence (measured by validation loss or a fixed number of epochs). Each outer iteration $k$ involves one full pass through this inner loop. The algorithm reuses the same offline dataset $\mathcal{D}$ at every outer iteration. The transitions never change, but the targets $y_i$ are recomputed using the updated Q-function $q(\cdot, \cdot; \theta_k)$.

## From Nested to Flattened Q-Iteration

To understand modern deep RL algorithms like DQN, we need to see how they emerge from the basic fitted Q-iteration template through a series of natural modifications. We'll build this up step by step, starting with the basic offline setting and gradually introducing warmstarting, partial optimization, and online data collection.

### Basic Offline Neural Fitted Q-Iteration

We start with the simplest case: a fixed dataset $\mathcal{D}$ of transitions, where we use a neural network as our function approximator and fit it to convergence at each iteration:

```{prf:algorithm} Basic Offline Neural Fitted Q-Value Iteration
:label: basic-nfqi

**Input**: MDP $(S, A, P, R, \gamma)$, offline transition dataset $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}$, neural network $q(s,a; \boldsymbol{\theta})$, initialization $\boldsymbol{\theta}_0$

**Output**: Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D}_n^{\text{fit}} \leftarrow \emptyset$
    2. For each $(s,a,r,s') \in \mathcal{D}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_n)$
        2. $\mathcal{D}_n^{\text{fit}} \leftarrow \mathcal{D}_n^{\text{fit}} \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}_n^{\text{fit}}, \boldsymbol{\theta}_{\text{init}}=\boldsymbol{\theta}_n, K=\infty)$
    4. $n \leftarrow n + 1$
4. **until** training complete
5. **return** $\boldsymbol{\theta}_n$
```

This algorithm has a nested structure: the outer loop (indexed by $n$) computes targets using the current Q-function, while `fit` performs an inner optimization to minimize the regression loss. The network uses warm starting: `fit` initializes from the previous iteration's parameters $\boldsymbol{\theta}_n$ and continues training, not from a fixed $\boldsymbol{\theta}_0$. Note the distinction: $\mathcal{D}$ is the fixed offline transition dataset, while $\mathcal{D}_n^{\text{fit}}$ is the regression dataset with computed targets that changes each iteration.

### Opening Up the Inner Loop

To make the structure explicit, let's expand the `fit` procedure to show the inner gradient descent loop:

```{prf:algorithm} Fitted Q-Value Iteration with Explicit Inner Loop
:label: nfqi-explicit-inner

**Input**: MDP $(S, A, P, R, \gamma)$, offline transition dataset $\mathcal{D}$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, convergence test $\texttt{has\_converged}(\cdot)$, initialization $\boldsymbol{\theta}_0$, loss $\mathcal{L}$

**Output**: Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D}_n^{\text{fit}} \leftarrow \emptyset$
    2. For each $(s,a,r,s') \in \mathcal{D}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_n)$
        2. $\mathcal{D}_n^{\text{fit}} \leftarrow \mathcal{D}_n^{\text{fit}} \cup \{((s,a), y_{s,a})\}$
    3. // Inner optimization loop (warm start from previous iteration)
    4. $\boldsymbol{\theta}^{(0)} \leftarrow \boldsymbol{\theta}_n$
    5. $k \leftarrow 0$
    6. **repeat**
        1. $\boldsymbol{\theta}^{(k+1)} \leftarrow \boldsymbol{\theta}^{(k)} - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}^{(k)}; \mathcal{D}_n^{\text{fit}})$
        2. $k \leftarrow k + 1$
    7. **until** $\texttt{has\_converged}(\boldsymbol{\theta}^{(0)}, ..., \boldsymbol{\theta}^{(k)}, \mathcal{D}_n^{\text{fit}})$
    8. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}^{(k)}$
    9. $n \leftarrow n + 1$
4. **until** training complete
5. **return** $\boldsymbol{\theta}_n$
```

Now the two-level structure is explicit: targets are computed using $\boldsymbol{\theta}_n$ (step 3.2.1) and remain fixed throughout the inner loop. Each inner optimization warm starts from the previous outer iteration's parameters $\boldsymbol{\theta}_n$ (step 3.4) and runs until convergence.

### Partial Optimization

NFQI as described runs the inner loop to convergence ($K=\infty$). Similar to modified policy iteration, we can perform partial optimization by taking only $K$ gradient steps rather than running to convergence. This trades off fit quality for computational cost per outer iteration:

```{prf:algorithm} Neural Fitted Q-Iteration with Partial Optimization
:label: nfqi-partial

**Input**: MDP $(S, A, P, R, \gamma)$, offline transition dataset $\mathcal{D}$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, number of steps $K$, initial parameters $\boldsymbol{\theta}_0$

**Output**: Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D}_n^{\text{fit}} \leftarrow \emptyset$
    2. For each $(s,a,r,s') \in \mathcal{D}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_n)$
        2. $\mathcal{D}_n^{\text{fit}} \leftarrow \mathcal{D}_n^{\text{fit}} \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}_n^{\text{fit}}, \boldsymbol{\theta}_{\text{init}}=\boldsymbol{\theta}_n, K=K)$
    4. $n \leftarrow n + 1$
4. **until** training complete
5. **return** $\boldsymbol{\theta}_n$
```

With partial optimization, we take only $K$ gradient steps per outer iteration. The template combines Monte Carlo integration (using observed next states $s'$), warm starting (continuing from $\boldsymbol{\theta}_n$), and partial optimization ($K$ steps instead of full convergence).

### Flattening the Nested Structure

The nested loop structure is conceptually clear but algorithmically cumbersome. We can flatten it into a single loop using a target network $\boldsymbol{\theta}_{target}$ that gets updated every $K$ steps. This is algorithmically equivalent but removes the explicit nesting:

```{prf:algorithm} Flattened Neural Fitted Q-Iteration
:label: nfqi-flattened

**Input**: MDP $(S, A, P, R, \gamma)$, offline transition dataset $\mathcal{D}$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, target update frequency $K$

**Output**: Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_0$
3. $t \leftarrow 0$
4. **while** training:
    1. $\mathcal{D}_t^{\text{fit}} \leftarrow \emptyset$
    2. For each $(s,a,r,s') \in \mathcal{D}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_{target})$
        2. $\mathcal{D}_t^{\text{fit}} \leftarrow \mathcal{D}_t^{\text{fit}} \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t; \mathcal{D}_t^{\text{fit}})$
    4. If $t \bmod K = 0$:
        1. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_t$
    5. $t \leftarrow t + 1$
5. **return** $\boldsymbol{\theta}_t$
```

The target network $\boldsymbol{\theta}_{target}$ plays exactly the role that $\boldsymbol{\theta}_n$ played in the nested version: it provides fixed targets for $K$ gradient steps. The periodic synchronization $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_t$ marks what would have been the boundary of the outer loop. This flattened structure is more suitable for online learning, where we continuously collect data.

The target network is thus not a magical stabilization trick but simply the natural consequence of flattening NFQI's nested structure while preserving the property that targets remain fixed for multiple gradient steps.

An alternative to periodic updates is exponential moving average (EMA), which updates the target network smoothly at every step:

```{prf:algorithm} Flattened Neural Fitted Q-Iteration with EMA
:label: nfqi-flattened-ema

**Input**: MDP $(S, A, P, R, \gamma)$, offline transition dataset $\mathcal{D}$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, EMA rate $\tau$

**Output**: Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_0$
3. $t \leftarrow 0$
4. **while** training:
    1. $\mathcal{D}_t^{\text{fit}} \leftarrow \emptyset$
    2. For each $(s,a,r,s') \in \mathcal{D}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_{target})$
        2. $\mathcal{D}_t^{\text{fit}} \leftarrow \mathcal{D}_t^{\text{fit}} \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t; \mathcal{D}_t^{\text{fit}})$
    4. $\boldsymbol{\theta}_{target} \leftarrow \tau\boldsymbol{\theta}_{t+1} + (1-\tau)\boldsymbol{\theta}_{target}$
    5. $t \leftarrow t + 1$
5. **return** $\boldsymbol{\theta}_t$
```

EMA targets (also called Polyak averaging) became popular with DDPG {cite}`lillicrap2015continuous` and are now standard in continuous control algorithms like TD3 {cite}`fujimoto2018addressing` and SAC {cite}`haarnoja2018soft`, typically using small $\tau$ values like 0.001.

With the batch algorithms established and the nested-to-flattened progression clear, the [next chapter](online_rl.md) extends these ideas to online learning with replay buffers, yielding algorithms like DQN that collect data and learn simultaneously.

