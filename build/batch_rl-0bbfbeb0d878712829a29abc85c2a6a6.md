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

The [previous chapter](simadp.md) established Monte Carlo integration for approximate dynamic programming and introduced Q-functions to amortize action selection. However, combining Monte Carlo with the maximization in the Bellman operator introduces a systematic problem: while individual expectations are unbiased, taking the maximum over noisy estimates creates upward bias. This chapter examines the sources of this bias, approaches for mitigating it, and develops batch reinforcement learning algorithms that work with fixed offline datasets.

## Overestimation Bias and Mitigation Strategies

Monte Carlo integration enables simulation-based approximate dynamic programming. However, combining Monte Carlo with the maximization in the Bellman operator introduces a systematic problem. While individual expectations are unbiased, taking the maximum over noisy estimates creates upward bias. This section examines the sources of this bias and two approaches for mitigating it.

### Sources of Overestimation Bias

When we apply Monte Carlo integration to the Bellman operator, each individual expectation is unbiased. The problem arises when we maximize over these noisy estimates. To make this precise, consider a given state-action pair $(s,a)$ and value function $v$. Define the true continuation value

$$
\mu(s,a) \equiv \int v(s')\,p(ds' \mid s,a),
$$

and its Monte Carlo approximation with $N$ samples:

$$
\hat{\mu}_N(s,a) \equiv \frac{1}{N}\sum_{i=1}^N v(s'_i), \quad s'_i \sim p(\cdot|s,a).
$$

Each estimator is unbiased: $\mathbb{E}[\hat{\mu}_N(s,a)] = \mu(s,a)$. The Monte Carlo Bellman operator is

$$
(\widehat{\Bellman}_N v)(s) = \max_{a \in \mathcal{A}_s} \left\{ r(s,a) + \gamma \hat{\mu}_N(s,a)\right\}.
$$

While each $\hat{\mu}_N(s,a)$ is unbiased, the maximization introduces systematic overestimation. To see why, let $a^*$ be the truly optimal action. The maximum of any collection is at least as large as any particular element:

$$
\mathbb{E}\big[\max_a \{r(s,a) + \gamma \hat{\mu}_N(s,a)\}\big] \ge \mathbb{E}\big[r(s,a^*) + \gamma \hat{\mu}_N(s,a^*)\big] = r(s,a^*) + \gamma \mu(s,a^*) = (\Bellman v)(s).
$$

The inequality is strict whenever multiple actions have nonzero variance. The maximization selects whichever action happens to receive a positive noise realization, and that inflated estimate contributes to the target value. This is Jensen's inequality for the max operator: $\mathbb{E}[\max_a Y_a] \ge \max_a \mathbb{E}[Y_a]$ for random variables $\{Y_a\}$, since the max is convex. Even though we start with unbiased estimators, taking the maximum breaks unbiasedness.

Operationally, this means that repeatedly sampling fresh states and taking the max will not, on average, converge to the true value but systematically land above it. Unlike noise that averages out, this bias persists no matter how many samples we draw. Worse, it compounds across iterations as overestimates feed into future target computations.

```{prf:remark} Connection to deterministic policies
This inequality also underlies why deterministic policies are optimal in MDPs (Theorem {prf:ref}`stoch-policy-reduction`): $\max_a w(a) \ge \sum_a q(a) w(a)$ shows randomization cannot improve expected returns. The same mathematical principle appears in two contexts: (1) deterministic policies suffice, (2) maximization over noisy estimates creates upward bias.
```

Monte Carlo value iteration applies $v_{k+1} = \widehat{\Bellman}_N v_k$ repeatedly. The overestimation bias does not stay confined to a single iteration: it accumulates through the recursion. To see why, note that each iteration's output becomes the next iteration's input. At iteration 2, we compute Monte Carlo estimates $\hat{\mu}_N(s,a) = \frac{1}{N}\sum_{i=1}^N v_1(s'_i)$, but $v_1$ is already biased upward from iteration 1. We are now averaging an overestimated function before we even apply the maximization, which adds another layer of bias on top of what was inherited. This pattern repeats at each iteration, creating a feedback loop where $\mathbb{E}[v_k] \ge \mathbb{E}[v_{k-1}] \ge \cdots \ge v^*$. Favorable noise at iteration $k$ becomes embedded in the value function and treated as truth by iteration $k+1$.

### Learning the Bias Correction

One approach, developed by Keane and Wolpin {cite}`Keane1994` in the context of dynamic discrete choice models, treats the bias itself as a quantity to be learned and subtracted. For a given value function $v$ and Monte Carlo sample size $N$, define the bias at state $s$ as

$$
\delta(s) = \mathbb{E}\big[(\widehat{\Bellman}_N v)(s)\big] - (\Bellman v)(s) \ge 0.
$$

While we cannot compute $\delta(s)$ directly (we lack both the expectation and the exact Bellman application), the bias has structure. It depends on observable quantities: the number of actions $|\mathcal{A}_s|$, the sample size $N$, and the spread of action values. When one action dominates, $\delta(s)$ is small. When several actions have similar values, noise is more likely to flip the maximizer, increasing $\delta(s)$.

Rather than deriving $\delta(s)$ analytically, Keane and Wolpin proposed learning it empirically. The strategy follows a "simulate on a subset, interpolate everywhere" template common in econometric dynamic programming. At a carefully chosen set of states, we run both high-fidelity simulation (many samples or exact integration) and the low-fidelity $N$-sample estimate used in value iteration. The gap between these estimates provides training data for the bias. We then fit a regression model $g_\eta$ that predicts $\delta(s)$ from features of the state and action-value distribution. During value iteration, we subtract the predicted bias from the raw Monte Carlo maximum.

Useful features for predicting the bias include the spread of action values (from a separate high-fidelity simulation), the number of actions, and gaps to the best action. These are cheap to compute and track regimes where maximization bias is large. The procedure can be summarized as:

```{prf:algorithm} Keane-Wolpin Bias-Corrected Value Iteration
:label: keane-wolpin-bias-correction

**Input:** MDP, current value $v_k$, sample size $N$, learned bias correction $g_\eta$

**Output:** Bias-corrected next value $v_{k+1}$

1. **For** each state $s$ **do**:
   - **For** each action $a$ **do**:
     - Draw $N$ samples: $s'_1, \ldots, s'_N \sim p(\cdot|s,a)$
     - Compute $\hat{\mu}_N(s,a) = \frac{1}{N}\sum_{i=1}^N v_k(s'_i)$
   - Compute raw max: $M(s) = \max_a \{r(s,a) + \gamma \hat{\mu}_N(s,a)\}$
   - Construct features $\phi(s)$ from action-value spread (using separate high-fidelity estimate)
   - Bias-corrected value: $v_{k+1}(s) = M(s) - g_\eta(\phi(s))$
```

The regression model $g_\eta$ is trained offline by comparing high and low-fidelity simulations at a grid of states, then applied during each value iteration step. This approach has been influential in econometrics for structural estimation problems. However, it has seen limited adoption in reinforcement learning. The computational overhead is substantial: it requires high-fidelity simulation at training states, careful feature engineering, and maintaining the regression model throughout learning. More critically, the circular dependency between the bias estimate and the value function can amplify errors if $g_\eta$ is misspecified. We now turn to a simpler alternative that avoids explicit bias modeling.

### Decoupling Selection and Evaluation

An alternative approach modifies the estimator itself to break the coupling that creates bias. In the standard Monte Carlo update $\max_a \{r(s,a) + \gamma \hat{\mu}_N(s,a)\}$, the same noisy estimate both selects which action looks best and provides the value assigned to that action. To see the problem, decompose the estimator into its mean and noise:

$$
\hat{\mu}_N(s,a) = \mu(s,a) + \varepsilon_a,
$$

where $\varepsilon_a$ is zero-mean noise. Whichever action happens to have the largest $\varepsilon_a$ gets selected, and that same positive noise inflates the target value:

$$
Y = r(s,a^\star) + \gamma \mu(s,a^\star) + \gamma \varepsilon_{a^\star},
$$

where $a^\star = \arg\max_a \{r(s,a) + \gamma \mu(s,a) + \gamma \varepsilon_a\}$. This coupling (using the same random variable $\varepsilon_{a^\star}$ for both selection and evaluation) produces $\mathbb{E}[Y] \ge \max_a \{r(s,a) + \gamma \mu(s,a)\}$.

Double Q-learning {cite}`van2016deep` breaks this coupling by maintaining two independent Monte Carlo estimates. Conceptually, for each action $a$ at state $s$, we would draw two separate sets of samples from $p(\cdot|s,a)$:

$$
\hat{\mu}^{(1)}_N(s,a) = \mu(s,a) + \varepsilon^{(1)}_a, \quad \hat{\mu}^{(2)}_N(s,a) = \mu(s,a) + \varepsilon^{(2)}_a,
$$

where $\varepsilon^{(1)}_a$ and $\varepsilon^{(2)}_a$ are **independent zero-mean noise terms**. We use the first estimate to select the action but the second to evaluate it:

$$
a^\star = \arg\max_{a} \left\{r(s,a) + \gamma \hat{\mu}^{(1)}_N(s,a)\right\}, \quad Y = r(s,a^\star) + \gamma \hat{\mu}^{(2)}_N(s,a^\star).
$$

Note that $a^\star$ is a random variable. It is the result of maximizing over noisy estimates $\hat{\mu}^{(1)}_N(s,a)$, and therefore depends on the noise $\varepsilon^{(1)}$. The target value $Y$ uses the second estimator $\hat{\mu}^{(2)}_N(s,a^\star)$, evaluating it at the randomly selected action $a^\star$. Expanding the target value to make the dependencies explicit:

$$
Y = r(s,a^\star) + \gamma \hat{\mu}^{(2)}_N(s,a^\star) = r(s,a^\star) + \gamma \mu(s,a^\star) + \gamma \varepsilon^{(2)}_{a^\star},
$$

where $a^\star$ itself depends on $\varepsilon^{(1)}$. To see why this eliminates evaluation bias, write the expectation as a nested expectation:

$$
\mathbb{E}[Y] = \mathbb{E}_{\varepsilon^{(1)}}\Big[\mathbb{E}_{\varepsilon^{(2)}}\big[r(s,a^\star) + \gamma \mu(s,a^\star) + \gamma \varepsilon^{(2)}_{a^\star} \mid a^\star\big]\Big].
$$

The inner expectation, conditioned on the selected action $a^\star$, equals:

$$
\mathbb{E}_{\varepsilon^{(2)}}\big[r(s,a^\star) + \gamma \mu(s,a^\star) + \gamma \varepsilon^{(2)}_{a^\star} \mid a^\star\big] = r(s,a^\star) + \gamma \mu(s,a^\star),
$$

because $\mathbb{E}[\varepsilon^{(2)}_{a^\star} \mid a^\star] = 0$. Why does this conditional expectation vanish? Because $a^\star$ is a random variable determined entirely by $\varepsilon^{(1)}$ (through the argmax operation), while $\varepsilon^{(2)}$ is independent of $\varepsilon^{(1)}$. Therefore, $\varepsilon^{(2)}$ is independent of $a^\star$. By the fundamental property of independence, for any random variables $X$ and $Y$ where $X \perp Y$, we have $\mathbb{E}[X \mid Y] = \mathbb{E}[X]$. Applying this:

$$
\mathbb{E}[\varepsilon^{(2)}_{a^\star} \mid a^\star] = \mathbb{E}[\varepsilon^{(2)}_{a^\star}] = 0,
$$

where the final equality holds because each $\varepsilon^{(2)}_a$ has zero mean. Taking the outer expectation:

$$
\mathbb{E}[Y] = \mathbb{E}_{\varepsilon^{(1)}}\big[r(s,a^\star) + \gamma \mu(s,a^\star)\big].
$$

The evaluation noise contributes zero on average because it is independent of the selection. However, double Q-learning does not eliminate all bias. Two distinct sources of bias arise when maximizing over noisy estimates:

1. **Selection bias**: Using $\arg\max_a \{r(s,a) + \gamma \mu(s,a) + \gamma \varepsilon^{(1)}_a\}$ tends to select actions that received positive noise. The noise $\varepsilon^{(1)}$ causes us to favor actions that got lucky, so $a^\star$ is not uniformly distributed but skewed toward actions with positive $\varepsilon^{(1)}_a$. This means $\mathbb{E}_{\varepsilon^{(1)}}[\mu(s,a^\star)] \ge \max_a \mu(s,a)$.

2. **Evaluation bias**: In the standard estimator, we use the *same* noisy estimate $\hat{\mu}_N(s,a^\star)$ that selected $a^\star$ to also evaluate it. If $a^\star$ was selected because $\varepsilon_{a^\star}$ happened to be large and positive, that same inflated estimate provides the target value. This creates additional bias beyond the selection effect.

Double Q-learning eliminates evaluation bias by using independent noise for evaluation: even though $a^\star$ is biased toward actions with positive $\varepsilon^{(1)}$, the evaluation uses $\varepsilon^{(2)}$ which is independent of the selection, so $\mathbb{E}[\varepsilon^{(2)}_{a^\star} \mid a^\star] = 0$. Selection bias remains, but removing evaluation bias substantially reduces total overestimation.

```{prf:remark} Implementation via different Q-functions
:class: dropdown

The conceptual framework above assumes we can draw multiple independent samples from $p(\cdot|s,a)$ for each state-action pair. In practice, this would require resetting a simulator to the same state and sampling multiple times, which is often infeasible. The practical implementation achieves the same effect differently: maintain two Q-functions that are trained on different data or updated at different times (e.g., one is a slowly-updating target network). Since the two Q-functions experience different noise realizations during training, their errors remain less correlated than if we used a single Q-function for both selection and evaluation. This is how Double DQN works, as we'll see in the [next chapter on online learning](online_rl.md).
```

The following algorithm implements this principle with Monte Carlo integration. We maintain two Q-functions and alternate which one selects actions and which one evaluates them. The bias reduction mechanism applies whether we store Q-values in a table or use function approximation.

Note that this algorithm is written in the theoretical form that assumes we can draw $N$ samples from each $(s,a)$ pair. In practice with single-trajectory data ($N=1$, one quadruple $(s,a,r,s')$ per transition), the double Q-learning principle still applies but we work with the single observed next state $s'$ and maintain two Q-functions that are trained on different subsets of data or updated at different frequencies to achieve the independence needed for bias reduction.

```{prf:algorithm} Double Q Value Iteration with Monte Carlo
:label: double-q-value-iteration

**Input:** MDP, state sample $\mathcal{S}$, Monte Carlo sample size $N$, maximum iterations $K$

**Output:** Two Q-functions $q^{(1)}$, $q^{(2)}$

1. Initialize $q^{(1)}_0(s,a)$ and $q^{(2)}_0(s,a)$ for all $s,a$
2. $k \leftarrow 0$
3. **repeat**
4. $\quad$ **for** each $s \in \mathcal{S}$ **do**
5. $\quad\quad$ **for** each $a \in \mathcal{A}(s)$ **do**
6. $\quad\quad\quad$ Draw $N$ next states: $s'_1, \ldots, s'_N \sim p(\cdot \mid s,a)$
7. $\quad\quad\quad$ **// Compute targets using decoupled selection-evaluation**
8. $\quad\quad\quad$ **for** $i = 1, \ldots, N$ **do**
9. $\quad\quad\quad\quad$ $a^{(1)}_i \leftarrow \arg\max_{a'} q^{(1)}_k(s'_i, a')$
10. $\quad\quad\quad\quad$ $a^{(2)}_i \leftarrow \arg\max_{a'} q^{(2)}_k(s'_i, a')$
11. $\quad\quad\quad$ **end for**
12. $\quad\quad\quad$ $q^{(1)}_{k+1}(s,a) \leftarrow r(s,a) + \frac{\gamma}{N}\sum_{i=1}^N q^{(2)}_k(s'_i, a^{(1)}_i)$
13. $\quad\quad\quad$ $q^{(2)}_{k+1}(s,a) \leftarrow r(s,a) + \frac{\gamma}{N}\sum_{i=1}^N q^{(1)}_k(s'_i, a^{(2)}_i)$
14. $\quad\quad$ **end for**
15. $\quad$ **end for**
16. $\quad$ $k \leftarrow k+1$
17. **until** convergence or $k \geq K$
18. **return** $q^{(1)}_k$, $q^{(2)}_k$
```

At lines 9 and 12, $q^{(1)}$ selects the action but $q^{(2)}$ evaluates it. The noise in $q^{(1)}$ influences which action is chosen, but the independent noise in $q^{(2)}$ does not inflate the evaluation. Lines 10 and 13 apply the same principle symmetrically for updating $q^{(2)}$.

## A Unified View of Reinforcement Learning Algorithms

We have developed three foundational components: (1) fitted-value iteration from the [projection methods chapter](projdp.md), which computes targets from the Bellman operator and fits a new approximation; (2) Q-functions that amortize action selection by caching state-action values; (3) Monte Carlo integration using single-sample transitions $(s, a, r, s')$ from trajectories.

These combine into a template: $\hat{q}^{(k+1)} = \text{fit}(\{((s_i, a_i), y_i)\})$, where targets $y_i = r_i + \gamma \max_{a'} q(s'_i, a'; \theta^{(k)})$ use the single observed next state $s'_i$. Different instantiations yield the major value-based reinforcement learning algorithms. This section systematically maps the design space, showing how FQI, NFQI, DQN, Q-learning, and their variants arise as specific configurations.

### Design Choices and Algorithm Space

Six independent choices define an algorithm:

| **Design Choice** | **Options** |
|:------------------|:------------|
| Function approximator | Linear basis, Decision trees, Neural networks, Kernel methods |
| Bellman operator | Hard max, Smooth logsumexp |
| Inner optimization | Full convergence, K steps, Single step (K=1) |
| Inner loop initialization | Cold start (from $\theta_0$), Warm start (from $\theta^{(k)}$) |
| Data collection | Offline (fixed batch), Online, Online with replay buffer |
| Bias mitigation | None, Double Q-learning, Learned correction |

Different combinations of these choices yield different algorithms. The table below shows how several well-known methods correspond to specific configurations:

| **Algorithm** | **Approximator** | **Bellman** | **Inner Loop** | **Initialization** | **Data** | **Bias Fix** |
|:--------------|:-----------------|:------------|:---------------|:-------------------|:---------|:-------------|
| FQI {cite}`ernst2005tree` | Extra Trees | Hard | Full | Cold | Offline | None |
| NFQI {cite}`riedmiller2005neural` | Neural Net | Hard | Full | Warm | Offline | None |
| DQN {cite}`mnih2013atari` | Deep NN | Hard | K=1 | Warm | Replay | None |
| Double DQN {cite}`van2016deep` | Deep NN | Hard | K=1 | Warm | Replay | Double Q |
| Q-learning {cite}`SuttonBarto2018` | Tabular/Linear | Hard | K=1 | Warm | Online | None |
| Soft Q {cite}`haarnoja2017reinforcement` | Neural Net | Smooth | K steps | Warm | Replay | None |

This table omits continuous action methods (NFQCA, DDPG, SAC), which introduce an additional design dimension. We address those in the [continuous action chapter](cadp.md). The initialization choice becomes particularly important when moving from batch to online algorithms.

### Buffer Abstraction: Unifying Offline and Online Data Collection

The distinction between offline and online data collection can be unified through a **buffer abstraction** with different update policies. A buffer $\mathcal{B}$ is a finite-capacity data structure that stores transitions $(s, a, r, s')$ and supports two operations:

- $\mathcal{B}.\text{add}(s, a, r, s')$: Add a transition (may discard oldest if at capacity)
- $\mathcal{B}.\text{sample}(b)$: Sample a mini-batch of $b$ transitions

Different data collection regimes correspond to different buffer initialization and update policies:

**Offline (Fixed Batch)**: Initialize $\mathcal{B}$ with pre-collected dataset $\mathcal{D}$, set capacity $= |\mathcal{D}|$, and disable the `add` operation. All algorithms sample from the same fixed distribution throughout training. This is pure sample average approximation: we separate data collection from learning.

**Online (No Replay)**: Set capacity $= 1$ and immediately sample each newly added transition. After processing, discard it. This is pure stochastic approximation: each transition is used exactly once for a single gradient step. Classical Q-learning follows this pattern.

**Online with Replay Buffer**: Initialize $\mathcal{B} = \emptyset$ with finite capacity $B$. As we interact with the environment, continuously `add` new transitions (overwriting oldest when full). Sample mini-batches from $\mathcal{B}$ for updates. The buffer contains a mixture of transitions from different policies: recent ones from the current policy and older ones from past policies. This hybrid approach enables data reuse while continuously incorporating new experience.

**Offline-to-Online (Warm Start)**: Initialize $\mathcal{B}$ with offline dataset $\mathcal{D}$ at capacity $B \ge |\mathcal{D}|$. Enable `add` operations so new online transitions gradually replace offline data. This smoothly transitions from pure offline learning to online learning as the buffer composition shifts.

**Fixed Offline with Online Evaluation**: Maintain two buffers: $\mathcal{B}_{\text{train}}$ (fixed offline data) and $\mathcal{B}_{\text{eval}}$ (online data for evaluation only). Sample training batches exclusively from $\mathcal{B}_{\text{train}}$ but collect online data to track performance. This tests offline-learned policies without contaminating the training distribution.

The buffer abstraction also clarifies the connection between data reuse and optimization strategy. Define the **replay ratio** as the number of gradient steps per environment transition. For capacity-$B$ buffer with mini-batch size $b$:

- Offline algorithms: replay ratio $\approx K \cdot N_{\text{epochs}} \cdot B / b$ (multiple epochs over fixed data)
- DQN: replay ratio $= b$ (one gradient step per transition, using batch of size $b$)
- Online Q-learning: replay ratio $= 1$ (one gradient step per transition, no reuse)

Higher replay ratios amortize environment interaction costs but risk overfitting to stale data or biased distributions. The extreme case (running inner loop to full convergence $K=\infty$ on a fixed batch) is classical fitted Q-iteration. The opposite extreme ($K=1$, capacity $=1$) is pure stochastic approximation.

The SAA vs SA distinction is not binary but a continuum controlled by truncation steps $K$ and buffer capacity $B$. The unified template accommodates all cases:

$$
\boldsymbol{\theta}_{n+1} = \texttt{fit}(\mathcal{B}.\text{sample}(b), \boldsymbol{\theta}_{\text{init}}, K)
$$

where the choice of $\boldsymbol{\theta}_{\text{init}} \in \{\boldsymbol{\theta}_0, \boldsymbol{\theta}_n\}$ controls warm vs cold starting, and the buffer policy determines the data distribution.

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

The two-hot encoding is not new mathematics. It is linear interpolation expressed in probabilistic language. The weights $q_j(y_i)$ and $q_{j+1}(y_i)$ are barycentric coordinates: non-negative weights summing to one that represent $y_i$ as a convex combination of the neighboring grid points $z_j$ and $z_{j+1}$. This is identical to the linear interpolation formula used in the [dynamic programming chapter](dp.md) for handling continuous state spaces (see Algorithm {prf:ref}`backward-recursion-interp`). When we discretize a continuous state space and interpolate between grid points, we compute exactly these weights to estimate the value function at off-grid states.

This connection places the two-hot encoding within the framework developed in the [projection methods chapter](projdp.md). The target construction satisfies the conditions for a monotone approximator in Gordon's sense: it maps value functions to value functions using a stochastic matrix with non-negative entries and rows summing to one (Definition {prf:ref}`gordon-averager`). Such operators preserve the contraction property of the Bellman operator (Theorem {prf:ref}`gordon-stability`), guaranteeing convergence of value iteration. Linear interpolation appears in the table of monotone approximators (projdp.md, line 703) as a method that always provides stability. The two-hot encoding inherits this structure: the target at bin $k$ is a weighted average of the neighbors, bounded by the minimum and maximum values in the local region.

The neural network $q(s,a; \boldsymbol{\theta})$ that predicts the categorical distribution, however, does not preserve monotonicity. The function class is highly flexible: learned features, nonlinear activations, and high-dimensional parameter spaces provide no guarantees about order preservation or bounded sensitivity. Neural networks are explicitly listed as non-monotone in the projection methods table (projdp.md, line 711). What we have in classification-based Q-learning is a hybrid: monotone targets paired with a non-monotone function approximator. The target distribution $q(\cdot \mid y_i)$ has the averager structure from classical dynamic programming theory, while the predicted distribution $p_{\boldsymbol{\theta}}(\cdot \mid s_i, a_i)$ comes from a flexible learned representation.

This asymmetry suggests an interpretation. Classical dynamic programming theory requires monotone approximators throughout to guarantee convergence: both the projection step and the function representation must preserve contraction. Deep reinforcement learning relaxes this by using non-monotone function classes but retains structure on the target side. The two-hot encoding provides implicit regularization through its interpolation structure: targets are smooth, locally bounded, and order-preserving, even though the approximator learns arbitrary features. This may partially explain why classification loss stabilizes learning beyond the cross-entropy robustness discussed below. The target encoding imposes geometric constraints inherited from monotone approximation theory, while the neural network retains the flexibility to represent complex value functions.

An alternative to the two-hot encoding is the histogram loss with Gaussian smoothing (HL-Gauss), which treats the scalar target $y_i$ as the mean of a Gaussian and projects that Gaussian onto the histogram bins to produce a smoothed categorical distribution. This adds a form of label smoothing: instead of placing mass only on the two immediate neighbors, the Gaussian tail spreads small amounts of probability to nearby bins. This regularization can improve generalization, particularly when targets are noisy.

The classification approach has connections to distributional reinforcement learning, particularly the C51 algorithm {cite}`bellemare2017distributional`, which applies the Bellman operator to entire return distributions rather than scalar expectations. C51 uses a categorical representation and KL divergence (equivalently, cross-entropy) between distributions. However, the classification framework can be applied without adopting the full distributional Bellman machinery. Using cross-entropy loss on scalar TD targets already provides substantial benefits, suggesting that the loss geometry itself, not just the distributional perspective, contributes to improved performance.

Cross-entropy loss provides several advantages over squared error in the deep Q-learning setting. First, it is more robust to outliers. Squared error penalizes large deviations quadratically, so a single bad target with $|y_i - q(s_i,a_i; \boldsymbol{\theta})| = 100$ contributes 10,000 to the loss. Cross-entropy, by contrast, penalizes incorrect categorical predictions logarithmically. A target that falls in a low-probability region of the predicted distribution incurs a large but bounded gradient. This reduced sensitivity to extreme TD errors can stabilize learning when early targets are unreliable.

Second, the categorical representation encodes ordinal structure. Neighboring bins represent similar values, and the two-hot or Gaussian-smoothed encodings ensure that nearby bins receive similar probabilities. This is a form of structured prediction: the network learns that a Q-value of 5.1 is similar to 5.0 and 4.9, which L2 regression does not explicitly encode. The ordinal structure acts as an inductive bias that can improve sample efficiency.

Third, cross-entropy loss has been shown to scale better with network capacity than squared error in deep Q-learning. Farebrother et al. {cite}`farebrother2024stop` found that standard DQN and offline methods like CQL degrade when the Q-network is scaled up to large ResNets or mixture-of-experts architectures, but using classification loss (specifically HL-Gauss) maintains or improves performance as network size increases. This suggests that L2 loss may be overfitting to noise in the targets when given high-capacity approximators, while cross-entropy's implicit regularization prevents this degradation. The same study showed gains across online Atari, offline Atari, and robotic manipulation tasks.

This provides a third approach to mitigating the overestimation bias discussed earlier in this chapter. Keane-Wolpin bias correction explicitly learns and subtracts the bias term, and double Q-learning decouples selection from evaluation to eliminate evaluation bias. Classification loss takes a different route: it changes the projection geometry itself. Instead of projecting onto the function class using L2 distance (which amplifies outliers), we project using KL divergence (cross-entropy), which provides robustness to the noisy target distribution. All three approaches address the same fundamental problem—noisy, biased targets—but through different mechanisms.

Practical implementation requires choosing the number of bins $K$ and their range $[z_1, z_K]$. Typical choices use $K \in \{51, 101, 201\}$ bins uniformly spaced over the expected return range. The range can be estimated from domain knowledge or learned adaptively during training. The network architecture changes minimally: instead of a single scalar output per action, we output $K$ logits per action. For discrete action spaces with $|\mathcal{A}|$ actions, this means a final layer of size $K \times |\mathcal{A}|$ rather than $|\mathcal{A}|$. The computational overhead is modest, and the implementation can use standard cross-entropy loss functions available in deep learning libraries. The main conceptual shift is viewing Q-learning as categorical prediction rather than scalar regression.

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

This template generates specific algorithms via instantiation:

- **FQI/NFQI**: $\mathcal{D}$ = offline data, $B = |\mathcal{D}|$, $c = \text{offline}$, $w = \text{cold}$, $K = \infty$, $\mathcal{T} = \max$
- **DQN**: $\mathcal{D} = \emptyset$, $B$ finite, $c = \text{online}$, $w = \text{warm}$, $K \in \{1, \ldots, 4\}$, $\mathcal{T} = \max$, periodic target updates
- **Q-learning**: $\mathcal{D} = \emptyset$, $B = 1$, $c = \text{online}$, $w = \text{warm}$, $K = 1$, $\mathcal{T} = \max$, no target network
- **Soft Q**: $\mathcal{D} = \emptyset$, $B$ finite, $c = \text{online}$, $w = \text{warm}$, $K \in \{1, \ldots, 4\}$, $\mathcal{T} = \text{softmax}$

The template makes explicit that the key algorithmic differences lie in:
1. **Buffer policy**: Capacity $B$ and initialization $\mathcal{B}_0$ determine offline vs online
2. **Optimization strategy**: $K$ and $w$ control SAA vs SA spectrum
3. **Operator choice**: Hard vs smooth Bellman determines policy class
4. **Target stability**: How $\boldsymbol{\theta}_{\text{target}}$ tracks $\boldsymbol{\theta}_n$

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

