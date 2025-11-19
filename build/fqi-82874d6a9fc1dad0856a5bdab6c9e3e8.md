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

# Fitted Q-Iteration Methods

The [previous chapter](simadp.md) established the theoretical foundations of simulation-based approximate dynamic programming: Monte Carlo integration for evaluating expectations, Q-functions for efficient action selection, and techniques for mitigating overestimation bias. Those developments assumed we could sample freely from transition distributions and choose optimization parameters without constraint. This chapter develops a unified framework for fitted Q-iteration algorithms that spans both offline and online settings. We begin with batch algorithms that learn from fixed datasets, then show how the same template generates online methods like DQN through systematic variations in data collection, optimization strategy, and function approximation.

## Design Choices in FQI Methods

All FQI methods share the same two-level structure built on three core ingredients: a buffer $\mathcal{B}_t$ of transitions inducing an empirical distribution $\hat{P}_{\mathcal{B}_t}$, a target map $T_q$ derived from the current Q-function, and an optimization procedure to fit the Q-function to the resulting targets. At iteration $n$, the outer loop applies the Bellman operator to construct targets $y_i^{(n)} = T_{q_n}(s_i, a_i, r_i, s'_i)$ by sampling transitions from $\hat{P}_{\mathcal{B}_n}$. The inner loop solves the regression problem $\min_{\boldsymbol{\theta}} \mathbb{E}_{((s,a),y) \sim \hat{P}_n^{\text{fit}}}[\ell(q(s, a; \boldsymbol{\theta}), y)]$ to find parameters that match these targets. We can write this abstractly as:

$$
\begin{aligned}
&\textbf{repeat } n = 0, 1, 2, \ldots \\
&\quad \text{Sample transitions from } \hat{P}_{\mathcal{B}_n} \text{ and construct targets via } T_{q_n} \\
&\quad \boldsymbol{\theta}^{(n+1)} \leftarrow \texttt{fit}(\hat{P}_n^{\text{fit}}, \boldsymbol{\theta}_{\text{init}}, K) \\
&\textbf{until } \text{convergence}
\end{aligned}
$$

The `fit` operation minimizes the regression loss using $K$ optimization steps (gradient descent for neural networks, tree construction for ensembles, matrix inversion for linear models) starting from initialization $\boldsymbol{\theta}_{\text{init}}$. Standard supervised learning uses random initialization ($\boldsymbol{\theta}_{\text{init}} = \boldsymbol{\theta}_0$) and runs to convergence ($K = \infty$). Reinforcement learning algorithms vary these choices: warm starting from the previous iteration ($\boldsymbol{\theta}_{\text{init}} = \boldsymbol{\theta}^{(n)}$), partial optimization ($K \in \{10, \ldots, 100\}$), or single-step updates ($K=1$).

The buffer $\mathcal{B}_t$ may stay fixed (batch setting) or change (online setting), but the targets always change because they depend on the evolving target map $T_{q_n}$. In practice, we typically have one observed next state per transition, giving $T_q(s_i, a_i, r_i, s'_i) = r_i + \gamma \max_{a'} q(s'_i, a'; \boldsymbol{\theta})$ for transition tuples $(s_i, a_i, r_i, s'_i)$.


```{prf:remark} Notation: Buffer vs Regression Distribution
:class: dropdown

We maintain a careful distinction between two empirical distributions throughout:

- **Buffer distribution** $\hat{P}_{\mathcal{B}_t}$ over transitions $\tau = (s, a, r, s')$: The empirical distribution induced by the replay buffer $\mathcal{B}_t$, which contains raw experience tuples. This is fixed (offline) or evolves via online collection (adding new transitions, dropping old ones).

- **Regression distribution** $\hat{P}_t^{\text{fit}}$ over pairs $((s,a), y)$: The empirical distribution over supervised learning targets. This changes every outer iteration $n$ as we recompute targets using the current target map $T_{q_n}$.

The relationship: at iteration $n$, we construct $\hat{P}_n^{\text{fit}}$ from $\hat{P}_{\mathcal{B}_n}$ by applying the target map:
$$
\hat{P}_n^{\text{fit}} = (\mathrm{id}, T_{q_n})_\# \hat{P}_{\mathcal{B}_n}
$$
where $T_{q_n}(s,a,r,s') = r + \gamma \max_{a'} q(s', a'; \boldsymbol{\theta}_n)$ or uses the smooth logsumexp operator.

This distinction matters pedagogically: the **buffer distribution** $\hat{P}_{\mathcal{B}_t}$ is fixed (offline) or evolves via online collection, while the **regression distribution** $\hat{P}_t^{\text{fit}}$ evolves via target recomputation. Fitted Q-iteration is the outer loop over target recomputation, not the inner loop over gradient steps.
```


This template provides a blueprint for instantiating concrete algorithms. Six design axes generate algorithmic diversity: the function approximator (trees, neural networks, linear models), the Bellman operator (hard max vs smooth logsumexp, discussed in the [regularized MDP chapter](regmdp.md)), the inner optimization strategy (full convergence, $K$ steps, or single step), the initialization scheme (cold vs warm start), the data collection mechanism (offline, online, replay buffer), and bias mitigation approaches (none, double Q-learning, learned correction). While individual algorithms include additional refinements, these axes capture the primary sources of variation. The table below shows how several well-known methods instantiate this template:

| **Algorithm** | **Approximator** | **Bellman** | **Inner Loop** | **Initialization** | **Data** | **Bias Fix** |
|:--------------|:-----------------|:------------|:---------------|:-------------------|:---------|:-------------|
| FQI {cite}`ernst2005tree` | Extra Trees | Hard | Full | Cold | Offline | None |
| NFQI {cite}`riedmiller2005neural` | Neural Net | Hard | Full | Warm | Offline | None |
| Q-learning {cite}`watkins1989learning` | Any | Hard | K=1 | Warm | Online | None |
| DQN {cite}`mnih2013atari` | Deep NN | Hard | K=1 | Warm | Replay | None |
| Double DQN {cite}`van2016deep` | Deep NN | Hard | K=1 | Warm | Replay | Double Q |
| Soft Q {cite}`haarnoja2017reinforcement` | Neural Net | Smooth | K steps | Warm | Replay | None |

This table omits continuous action methods (NFQCA, DDPG, SAC), which introduce an additional design dimension. We address those in the [continuous action chapter](cadp.md). The initialization choice becomes particularly important when moving from batch to online algorithms.

### Plug-In Approximation with Empirical Distributions

The exact Bellman operator involves expectations under the true transition law (combined with the behavior policy), which we denote abstractly by $P$ over transitions $\tau = (s,a,r,s')$:

$$
(\Bellman q)(s,a) = r(s,a) + \gamma \int \max_{a'} q(s',a')\, P(ds' \mid s,a)
$$

In fitted Q-iteration we never see $P$ directly. Instead, we collect a finite set of transitions in a buffer $\mathcal{B}_t = \{\tau_1,\dots,\tau_{|\mathcal{B}_t|}\}$ and work with the **empirical distribution**

$$
\hat{P}_{\mathcal{B}_t} = \frac{1}{|\mathcal{B}_t|} \sum_{\tau \in \mathcal{B}_t} \delta_\tau
$$

where $\delta_\tau$ denotes a Dirac delta (point mass) centered at transition $\tau$. Each $\delta_\tau$ is a probability distribution that places mass 1 at the single point $\tau$ and mass 0 everywhere else. The sum creates a mixture distribution: a uniform distribution over the finite set of observed transitions. Sampling from $\hat{P}_{\mathcal{B}_t}$ means picking one transition uniformly at random from the buffer.

For any integrand $g(\tau)$ (loss, TD error, gradient term), expectations under $P$ are approximated by expectations under $\hat{P}_{\mathcal{B}_t}$ using the **sample average estimator**:

$$
\mathbb{E}_{\tau \sim P}\big[g(\tau)\big] \;\approx\; \mathbb{E}_{\tau \sim \hat{P}_{\mathcal{B}_t}}\big[g(\tau)\big] = \frac{1}{|\mathcal{B}_t|} \sum_{\tau \in \mathcal{B}_t} g(\tau)
$$

This is exactly the sample average estimator from the Monte Carlo chapter, applied now to transitions. Conceptually, fitted Q-iteration performs **plug-in approximate dynamic programming**. The plug-in principle is a general approach from statistics: when an algorithm requires an unknown population quantity, substitute its sample-based estimator. Here, we replace the unknown transition law $P$ with the empirical distribution $\hat{P}_{\mathcal{B}_t}$ and run value iteration using this empirical Bellman operator:

$$
(\widehat{\Bellman}_{\mathcal{B}_t} q)(s,a) \triangleq r(s,a) + \gamma\; \mathbb{E}_{(r',s') \mid (s,a)\sim \hat{P}_{\mathcal{B}_t}} \Big[\max_{a'} q(s',a')\Big]
$$

From a computational viewpoint, we could describe all of this using sample averages and mini-batch gradients. The empirical distribution notation provides three benefits. First, it unifies offline and online algorithms: both perform value iteration under an empirical law $\hat{P}_{\mathcal{B}_t}$, differing only in whether $\mathcal{B}_t$ remains fixed or evolves. Second, it shows that methods like DQN perform stochastic optimization of a sample average approximation objective $\mathbb{E}_{\tau \sim \hat{P}_{\mathcal{B}_t}}[\ell(\cdot)]$, not some ad hoc non-stationary procedure. Third, it cleanly separates two sources of approximation that we will examine shortly: statistical bootstrap (resampling from $\hat{P}_{\mathcal{B}_t}$) versus temporal-difference bootstrap (using estimated values in targets).

```{prf:remark} Mathematical Formulation of Empirical Distributions
:class: dropdown

The empirical distribution $\hat{P}_{\mathcal{B}_t} = \frac{1}{|\mathcal{B}_t|}\sum_{\tau \in \mathcal{B}_t} \delta_{\tau}$ is a discrete probability measure over $|\mathcal{B}_t|$ points regardless of whether state and action spaces are continuous or discrete. For any measurable set $A \subseteq \mathcal{S} \times \mathcal{A} \times \mathbb{R} \times \mathcal{S}$:

$$
\hat{P}_{\mathcal{B}_t}(A) = \frac{1}{|\mathcal{B}_t|}\sum_{\tau \in \mathcal{B}_t} \mathbb{1}[\tau \in A] = \frac{|\{\tau \in \mathcal{B}_t : \tau \in A\}|}{|\mathcal{B}_t|}
$$

The empirical distribution assigns probability $1/|\mathcal{B}_t|$ to each observed tuple and zero elsewhere.
```

### Data, Buffers, and the Unified Template

Fitted Q-iteration is built around three ingredients at any time $t$:

1. A **replay buffer** $\mathcal{B}_t$ containing transitions, inducing an empirical distribution $\hat{P}_{\mathcal{B}_t}$
2. A **target map** $T_q : (s,a,r,s') \mapsto y$ derived from the Bellman operator (hard max or soft logsumexp)
3. A **loss function** $\ell$ and **optimization budget** (replay ratio, number of gradient steps)

Pushing transitions through the target map transforms $\hat{P}_{\mathcal{B}_t}$ into a regression distribution $\hat{P}_t^{\text{fit}}$ over pairs $((s,a), y)$, as described in the notation remark above. The inner loop minimizes the empirical risk $\mathbb{E}_{((s,a),y)\sim \hat{P}_t^{\text{fit}}} [\ell(q(s,a;\boldsymbol{\theta}), y)]$ via stochastic gradient descent on mini-batches.

Different algorithms correspond to different ways of evolving the buffer $\mathcal{B}_t$ and different replay ratios:

- **Offline FQI.** We start from a fixed dataset $\mathcal{D}$ and never collect new data. The buffer is constant, $\mathcal{B}_t \equiv \mathcal{D}$ and $\hat{P}_{\mathcal{B}_t} \equiv \hat{P}_{\mathcal{D}}$, and only the target map $T_{q_t}$ changes as the Q-function evolves.

- **Replay (DQN-style).** The buffer is a **circular buffer** of fixed capacity $B$. At each interaction step we append the new transition and, if the buffer is full, drop the oldest one: $\mathcal{B}_t = \{\tau_{t-B+1},\ldots,\tau_t\}$ and $\hat{P}_{\mathcal{B}_t} = \frac{1}{|\mathcal{B}_t|} \sum_{\tau \in \mathcal{B}_t} \delta_{\tau}$. The empirical distribution slides forward through time, but at each update we still sample uniformly from the current buffer.

- **Fully online Q-learning.** Tabular Q-learning is the degenerate case with buffer size $B=1$: we only keep the most recent transition. Then $\hat{P}_{\mathcal{B}_t}$ is supported on a single point and each update uses that one sample once.

The **replay ratio** (optimization steps per environment transition) quantifies data reuse:

$$
\text{Replay ratio} = \begin{cases}
K \cdot N_{\text{epochs}} \cdot N / b & \text{Offline (batch size } b \text{ from } N \text{ transitions)} \\
b & \text{DQN (one step per transition, batch size } b \text{)} \\
1 & \text{Online Q-learning}
\end{cases}
$$

where $K$ is the number of gradient steps per outer iteration. Large replay ratios reuse the same empirical distribution $\hat{P}_{\mathcal{B}_t}$ many times, implementing sample average approximation (offline FQI). Small replay ratios use each sample once, implementing stochastic approximation (online Q-learning). Higher replay ratios reduce variance of estimates under $\hat{P}_{\mathcal{B}_t}$ but risk overfitting to the idiosyncrasies of the current empirical law, especially when it reflects outdated policies or narrow coverage.

```{prf:remark} Two Notions of Bootstrap
:class: dropdown

This perspective separates two different "bootstraps" that appear in FQI:

1. **Statistical bootstrap over data.** By sampling with replacement from $\hat{P}_{\mathcal{B}_t}$, we approximate expectations under the (unknown) transition distribution $P$ with expectations under the empirical distribution $\hat{P}_{\mathcal{B}_t}$. This is identical to bootstrap resampling in statistics. Mini-batch training is exactly this: we treat the observed transitions as if they were the entire population and approximate expectations by repeatedly resampling from the empirical law. The replay ratio controls how many such bootstrap samples we take per environment interaction.

2. **Temporal-difference bootstrap over values.** When we compute $y = r + \gamma \max_{a'} q(s',a';\boldsymbol{\theta})$, we replace the unknown continuation value by our current estimate. This is the TD notion of bootstrapping and the source of maximization bias studied in the previous chapter.

FQI, DQN, and their variants combine both: the empirical distribution $\hat{P}_{\mathcal{B}_t}$ encodes how we reuse data (statistical bootstrap), and the target map $T_q$ encodes how we bootstrap values (TD bootstrap). Most bias-correction techniques (Keane–Wolpin, Double Q-learning, Gumbel losses) modify the second kind of bootstrapping while leaving the statistical bootstrap unchanged.

Every algorithm in this chapter minimizes an empirical risk of the form $\mathbb{E}_{((s,a),y)\sim \hat{P}_t^{\text{fit}}} [\ell(q(s,a;\boldsymbol{\theta}), y)]$, where expectations are computed via the sample average estimator over the buffer. Algorithmic diversity arises from choices of buffer evolution, target map, loss, and optimization schedule.
```

```{prf:remark} Unified FQI Template
:label: unified-fqi-template

At any time $t$, fitted Q-iteration is completely characterized by four components:

1. A **replay buffer** $\mathcal{B}_t$ and its empirical distribution $\hat{P}_{\mathcal{B}_t}$
2. A **target map** $T_{q_t}$ derived from the current Q-function
3. A **loss function** $\ell$ specifying a noise model
4. An **optimization budget** (replay ratio, number of gradient steps $K$)

The algorithms we study differ only in how they instantiate these choices:

- **Offline FQI**: Fixes $\mathcal{B}_t \equiv \mathcal{D}$ once, uses hard-max target map $T_q(s,a,r,s') = r + \gamma \max_{a'} q(s',a')$, squared loss $\ell(q,y) = (q-y)^2$, and large optimization budgets ($K=\infty$ or $K \gg 1$).

- **DQN**: Uses circular buffer $\mathcal{B}_t = \{\tau_{t-B+1}, \ldots, \tau_t\}$ with capacity $B$, same hard-max target map, squared loss, and minimal optimization budget ($K=1$ step per transition).

- **Double DQN**: Changes only the target map to $T_q(s,a,r,s') = r + \gamma q(s', \arg\max_{a'} q_{\text{online}}(s',a'); \boldsymbol{\theta}_{\text{target}})$, decoupling selection from evaluation.

- **Classification-based Q-learning**: Changes only the loss to cross-entropy over value bins and the target representation to categorical distributions.

This unified view makes it clear that the transition from offline to online is continuous (varying buffer size $B$ and replay ratio), and that algorithmic innovations typically modify just one or two components while leaving the rest unchanged.
```

## Batch Algorithms: Ernst's FQI and NFQI

We begin with the simplest case from the buffer perspective. We are given a fixed transition dataset $\mathcal{D} = \{(s_i,a_i,r_i,s'_i)\}_{i=1}^N$ and never collect new data. The replay buffer is frozen:

$$
\mathcal{B}_t \equiv \mathcal{D}, \qquad \hat{P}_{\mathcal{B}_t} \equiv \hat{P}_{\mathcal{D}}
$$

so the only thing that changes across iterations is the target map $T_{q_n}$ induced by the current Q-function. Every outer iteration of FQI samples from the same empirical distribution $\hat{P}_{\mathcal{D}}$ but pushes it through a new target map, producing a new regression distribution $\hat{P}_n^{\text{fit}}$.

At each outer iteration $n$, we construct the input set $X^{(n)} = \{(s_i, a_i)\}_{i=1}^N$ from the same transitions (the state-action pairs remain fixed), compute targets $y_i^{(n)} = T_{q_n}(s_i, a_i, r_i, s'_i) = r_i + \gamma \max_{a'} q(s'_i, a'; \boldsymbol{\theta}^{(n)})$ using the current Q-function, and solve the regression problem $\boldsymbol{\theta}^{(n+1)} \leftarrow \texttt{fit}(X^{(n)}, y^{(n)}, \boldsymbol{\theta}_{\text{init}}, K)$. The buffer $\mathcal{B}_t = \mathcal{D}$ never changes, but the targets change at every iteration because they depend on the evolving target map $T_{q_n}$. Each transition $(s_i, a_i, r_i, s'_i)$ provides a single Monte Carlo sample $s'_i$ for evaluating the Bellman operator at $(s_i, a_i)$, giving us $\widehat{\Bellman}_1 q$ with $N=1$.

The following algorithm is simply approximate value iteration where expectations under the transition kernel $P$ are replaced by expectations under the fixed empirical distribution $\hat{P}_{\mathcal{D}}$:

```{prf:algorithm} Generic Fitted Q-Iteration (Batch)
:label: fitted-q-iteration-batch

**Input:** Dataset $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$, function approximator $q(s,a; \boldsymbol{\theta})$, discount factor $\gamma$, maximum iterations $n_{\max}$

**Output:** Learned Q-function parameters $\boldsymbol{\theta}$

1. Initialize $\boldsymbol{\theta}_0$ 
2. $n \leftarrow 0$
3. **repeat**  $\quad$ // **Outer loop: Value Iteration**
4. $\quad$ **// Construct regression dataset with Bellman targets**
5. $\quad$ $X^{(n)} \leftarrow \{(s_i, a_i) : (s_i, a_i, r_i, s'_i) \in \mathcal{D}\}$
6. $\quad$ **for** each $(s_i, a_i, r_i, s'_i) \in \mathcal{D}$ **do**
7. $\quad\quad$ $y_i^{(n)} \leftarrow r_i + \gamma \max_{a' \in \mathcal{A}} q(s'_i, a'; \boldsymbol{\theta}_n)$
8. $\quad$ **end for**
9. $\quad$ **// Inner loop: Fit Q-function to targets (projection step)**
10. $\quad$ $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(X^{(n)}, y^{(n)}, \boldsymbol{\theta}_{\text{init}}, K)$
11. $\quad$ $n \leftarrow n+1$
12. **until** convergence or $n \geq n_{\max}$
13. **return** $\boldsymbol{\theta}_n$
```

The `fit` operation in line 10 abstracts the inner optimization loop that minimizes $\sum_{i=1}^N \ell(q(s_i, a_i; \boldsymbol{\theta}), y_i^{(n)})$. This line hides the key algorithmic choice: which function approximator to use and how to optimize it. The initialization $\boldsymbol{\theta}_{\text{init}}$ and number of optimization steps $K$ control whether we use cold or warm starting and whether we optimize to convergence or perform partial updates.

**Fitted Q-Iteration (FQI)**: Ernst et al. {cite}`ernst2005tree` instantiate this template with extremely randomized trees (extra trees), an ensemble method that partitions the state-action space into regions with piecewise constant Q-values. The `fit` operation trains the ensemble until completion using the tree construction algorithm. Trees handle high-dimensional inputs naturally and the ensemble reduces overfitting. FQI uses cold start initialization: $\boldsymbol{\theta}_{\text{init}} = \boldsymbol{\theta}_0$ (randomly initialized) at every iteration, since trees don't naturally support incremental refinement. The loss $\ell$ is squared error. This method demonstrated that batch reinforcement learning could work with complex function approximators on continuous-state problems.

**Neural Fitted Q-Iteration (NFQI)**: Riedmiller {cite}`riedmiller2005neural` replaces the tree ensemble with a neural network $q(s,a; \boldsymbol{\theta})$, providing smooth interpolation across the state-action space. The `fit` operation runs gradient-based optimization (RProp, chosen for its insensitivity to hyperparameter choices) to convergence: train the network until the loss stops decreasing (multiple epochs through the full dataset $\mathcal{D}$), corresponding to $K=\infty$ in our framework. NFQI uses warm start initialization: $\boldsymbol{\theta}_{\text{init}} = \boldsymbol{\theta}_n$ at iteration $n$, meaning the network continues learning from the previous iteration's weights rather than resetting. This ensures the network accurately represents the projected Bellman operator before moving to the next outer iteration. For episodic tasks with goal and forbidden regions, Riedmiller uses modified target computations (detailed below).

```{prf:remark} Goal State Heuristics in NFQI
:class: dropdown

For episodic tasks with goal states $S^+$ and forbidden states $S^-$, Riedmiller modifies the target structure:

$$
y_i^{(n)} = \begin{cases}
c(s_i, a_i, s'_i) & \text{if } s'_i \in S^+ \text{ (goal reached)} \\
C^- & \text{if } s'_i \in S^- \text{ (forbidden state, typically } C^- = 1.0\text{)} \\
c(s_i, a_i, s'_i) + \gamma \max_{a'} q(s'_i, a'; \boldsymbol{\theta}_n) & \text{otherwise}
\end{cases}
$$

where $c(s, a, s')$ is the immediate cost. Goal states have zero future cost (no bootstrapping), forbidden states have high penalty, and regular states use the standard Bellman backup. Additionally, the **hint-to-goal heuristic** adds synthetic transitions $(s, a, s')$ where $s \in S^+$ with target value $c(s,a,s') = 0$ to explicitly clamp the Q-function to zero in the goal region. This stabilizes learning by encoding the boundary condition without requiring additional prior knowledge.
```

## From Nested to Flattened Q-Iteration

Fitted Q-iteration has an inherently nested structure: an outer loop performs approximate value iteration by computing Bellman targets, and an inner loop performs regression by fitting the function approximator to those targets. This nested structure shows that FQI is approximate dynamic programming with function approximation, distinct from supervised learning with changing targets.

When the inner loop uses gradient descent for $K$ steps, we have:

$$
\texttt{fit}(\mathcal{D}_n^{\text{fit}}, \boldsymbol{\theta}_n, K) = \boldsymbol{\theta}_n - \alpha \sum_{k=0}^{K-1} \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_n^{(k)}; \mathcal{D}_n^{\text{fit}})
$$

This is a sequence of updates $\boldsymbol{\theta}_n^{(k+1)} = \boldsymbol{\theta}_n^{(k)} - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_n^{(k)}; \mathcal{D}_n^{\text{fit}})$ for $k = 0, \ldots, K-1$ starting from $\boldsymbol{\theta}_n^{(0)} = \boldsymbol{\theta}_n$. Since these inner updates are themselves a loop, we can algebraically rewrite the nested loops as a single flattened loop. This flattening is purely representational. The algorithm remains approximate value iteration, but the presentation obscures the conceptual structure.

In the flattened form, the parameters used for computing targets are called the **target network** $\boldsymbol{\theta}_{\text{target}}$, which corresponds to $\boldsymbol{\theta}_n$ in the nested form. The target network gets updated every $K$ steps, marking the boundaries between outer iterations. Many modern algorithms, especially those that collect data online like DQN, are presented in flattened form. This can make them appear different from batch methods when they are the same template with different design choices.

Tree ensemble methods like random forests or extra trees have no continuous parameter space and no gradient-based optimization. The `fit` operation builds the entire tree structure in one pass. There's no sequence of incremental updates to unfold into a single loop. Ernst's FQI {cite}`ernst2005tree` retains the explicit nested structure with cold start initialization at each outer iteration, while neural methods can be flattened.

### Making the Nested Structure Explicit

To see how flattening works, we first make the nested structure completely explicit by expanding the `fit` operation to show the inner gradient descent loop. In terms of the buffer notation, the inner loop approximately minimizes the empirical risk:

$$
\mathbb{E}_{((s,a),y)\sim \hat{P}_n^{\text{fit}}}[\ell(q(s,a;\boldsymbol{\theta}), y)]
$$

induced by the fixed buffer $\mathcal{B}_n = \mathcal{D}$ and target map $T_{q_n}$. Starting from the generic batch FQI template (Algorithm {prf:ref}`fitted-q-iteration-batch`), we replace the abstract `fit` call with explicit gradient updates:

```{prf:algorithm} Neural Fitted Q-Iteration with Explicit Inner Loop
:label: nfqi-explicit-inner

**Input**: MDP $(S, A, P, R, \gamma)$, offline transition dataset $\mathcal{D}$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, inner optimization steps $K$, initialization $\boldsymbol{\theta}_0$

**Output**: Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $n \leftarrow 0$  $\quad$ // **Outer loop counter (value iteration)**
3. **repeat**
    1. **// Compute Bellman targets using current Q-function**
    2. $\mathcal{D}_n^{\text{fit}} \leftarrow \emptyset$
    3. **for** each $(s,a,r,s') \in \mathcal{D}$ **do**
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_n)$
        2. $\mathcal{D}_n^{\text{fit}} \leftarrow \mathcal{D}_n^{\text{fit}} \cup \{((s,a), y_{s,a})\}$
    4. **// Inner optimization loop: fit to targets via gradient descent**
    5. $\boldsymbol{\theta}_n^{(0)} \leftarrow \boldsymbol{\theta}_n$ $\quad$ // Warm start from previous outer iteration
    6. $k \leftarrow 0$ $\quad$ // **Inner loop counter (regression)**
    7. **repeat**
        1. $\boldsymbol{\theta}_n^{(k+1)} \leftarrow \boldsymbol{\theta}_n^{(k)} - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_n^{(k)}; \mathcal{D}_n^{\text{fit}})$
        2. $k \leftarrow k + 1$
    8. **until** $k = K$ $\quad$ // Partial optimization: exactly K gradient steps
    9. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_n^{(K)}$
    10. $n \leftarrow n + 1$
4. **until** training complete
5. **return** $\boldsymbol{\theta}_n$
```

This makes the two-level structure completely transparent. The outer loop (indexed by $n$) computes targets $y_{s,a} = r + \gamma \max_{a'} q(s',a'; \boldsymbol{\theta}_n)$ using the parameters $\boldsymbol{\theta}_n$ from the end of the previous outer iteration. These targets remain **fixed** throughout the entire inner loop. The inner loop (indexed by $k$) performs $K$ gradient steps to fit $q(s,a; \boldsymbol{\theta})$ to the regression dataset $\mathcal{D}_n^{\text{fit}} = \{((s_i, a_i), y_i)\}$, warm starting from $\boldsymbol{\theta}_n$ (the parameters that computed the targets). After $K$ steps, the inner loop produces $\boldsymbol{\theta}_{n+1} = \boldsymbol{\theta}_n^{(K)}$, which becomes the starting point for the next outer iteration.

The notation $\boldsymbol{\theta}_n^{(k)}$ indicates that we are at inner step $k$ within outer iteration $n$. The targets depend only on $\boldsymbol{\theta}_n = \boldsymbol{\theta}_n^{(0)}$, not on the intermediate inner iterates $\boldsymbol{\theta}_n^{(k)}$ for $k > 0$.

### Flattening into a Single Loop

We can now flatten the nested structure by treating all gradient steps uniformly and using a global step counter $t$ instead of separate outer/inner counters. We introduce a **target network** $\boldsymbol{\theta}_{\text{target}}$ that holds the parameters used for computing targets. This target network gets updated every $K$ steps, which marks what would have been the boundary between outer iterations. The transformation works as follows: we replace the outer counter $n$ and inner counter $k$ with a single counter $t$, where $t = nK + k$. When $n$ advances from $n$ to $n+1$, this corresponds to $K$ steps of $t$: from $t = nK$ to $t = (n+1)K$. The target network $\boldsymbol{\theta}_{\text{target}}$ equals $\boldsymbol{\theta}_n$ throughout outer iteration $n$ and gets updated via $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_t$ every $K$ steps (when $t \bmod K = 0$). Parameters that were $\boldsymbol{\theta}_n^{(k)}$ in nested form become $\boldsymbol{\theta}_t$ in flattened form.

This transformation is purely algebraic. No algorithmic behavior changes, only the presentation:

```{prf:algorithm} Flattened Neural Fitted Q-Iteration
:label: nfqi-flattened

**Input**: MDP $(S, A, P, R, \gamma)$, offline transition dataset $\mathcal{D}$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, target update frequency $K$, initialization $\boldsymbol{\theta}_0$

**Output**: Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_0$
3. $t \leftarrow 0$ $\quad$ // **Single flattened loop counter**
4. **while** training **do**
    1. **// Compute targets using fixed target network**
    2. $\mathcal{D}_t^{\text{fit}} \leftarrow \emptyset$
    3. **for** each $(s,a,r,s') \in \mathcal{D}$ **do**
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_{\text{target}})$
        2. $\mathcal{D}_t^{\text{fit}} \leftarrow \mathcal{D}_t^{\text{fit}} \cup \{((s,a), y_{s,a})\}$
    4. **// Gradient step on online network**
    5. $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t; \mathcal{D}_t^{\text{fit}})$
    6. **// Periodic target network update (marks outer iteration boundary)**
    7. **if** $t \bmod K = 0$ **then**
        1. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_t$
    8. $t \leftarrow t + 1$
5. **return** $\boldsymbol{\theta}_t$
```

At step $t$, we have $n = \lfloor t/K \rfloor$ (outer iteration) and $k = t \bmod K$ (position within inner loop). The target network $\boldsymbol{\theta}_{\text{target}}$ equals $\boldsymbol{\theta}_n$ throughout the $K$ steps from $t = nK$ to $t = (n+1)K - 1$, then gets updated to $\boldsymbol{\theta}_{n+1}$ at $t = (n+1)K$. The parameters $\boldsymbol{\theta}_t$ correspond to $\boldsymbol{\theta}_n^{(k)}$ in the nested form. The flattening reindexes the iteration structure: outer iteration 3, inner step 7 becomes step 37.

Flattening replaces the pair $(n,k)$ by a single global step index $t$, but the underlying empirical distribution $\hat{P}_{\mathcal{D}}$ remains the same. We still sample from the fixed offline dataset throughout.

The target network arises directly from flattening the nested FQI structure. When DQN is presented with a target network that updates every $K$ steps, this is approximate value iteration in flattened form. The algorithm still performs outer loop (value iteration) and inner loop (regression), but the presentation obscures this structure. The periodic target updates mark the boundaries between outer iterations. DQN is batch approximate DP in flattened form, using online data collection with a replay buffer.

### Smooth Target Updates via Exponential Moving Average

An alternative to periodic hard updates is **exponential moving average (EMA)** (also called Polyak averaging), which updates the target network smoothly at every step rather than synchronizing it every $K$ steps:

```{prf:algorithm} Flattened Neural Fitted Q-Iteration with EMA Target Updates
:label: nfqi-flattened-ema

**Input**: MDP $(S, A, P, R, \gamma)$, offline transition dataset $\mathcal{D}$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, EMA rate $\tau \in (0, 1)$, initialization $\boldsymbol{\theta}_0$

**Output**: Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_0$
3. $t \leftarrow 0$ $\quad$ // **Single flattened loop counter**
4. **while** training **do**
    1. **// Compute targets using fixed target network**
    2. $\mathcal{D}_t^{\text{fit}} \leftarrow \emptyset$
    3. **for** each $(s,a,r,s') \in \mathcal{D}$ **do**
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_{\text{target}})$
        2. $\mathcal{D}_t^{\text{fit}} \leftarrow \mathcal{D}_t^{\text{fit}} \cup \{((s,a), y_{s,a})\}$
    4. **// Gradient step on online network**
    5. $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t; \mathcal{D}_t^{\text{fit}})$
    6. **// ← CHANGED: Smooth EMA update at every step**
    7. $\boldsymbol{\theta}_{\text{target}} \leftarrow \tau\boldsymbol{\theta}_{t+1} + (1-\tau)\boldsymbol{\theta}_{\text{target}}$
    8. $t \leftarrow t + 1$
5. **return** $\boldsymbol{\theta}_t$
```

With EMA updates, the target network slowly tracks the online network instead of making discrete jumps. For small $\tau$ (typically $\tau \in [0.001, 0.01]$), the target lags behind the online network by roughly $1/\tau$ steps. This provides smoother learning dynamics and avoids the discontinuous changes in targets that occur with periodic hard updates. The EMA approach became popular with DDPG {cite}`lillicrap2015continuous` for continuous control and is now standard in algorithms like TD3 {cite}`fujimoto2018addressing` and SAC {cite}`haarnoja2018soft`.

## Online Algorithms: DQN and Extensions

We now keep the same fitted Q-iteration template but allow the buffer $\mathcal{B}_t$ and its empirical distribution $\hat{P}_{\mathcal{B}_t}$ to evolve while we learn. Instead of repeatedly sampling from a fixed $\hat{P}_{\mathcal{D}}$, we collect new transitions during learning and store them in a circular replay buffer. 

Deep Q-Network (DQN) instantiates the online template with moderate choices along the design axes: buffer capacity $B \approx 10^6$, mini-batch size $b \approx 32$, and target network update frequency $K \approx 10^4$. Crucially, DQN is not an ad hoc collection of tricks. It is fitted Q-iteration in flattened form (as developed in the nested-to-flattened transformation earlier) with an evolving buffer $\mathcal{B}_t$ and low replay ratio.

### Deep Q-Network (DQN)

Deep Q-Network (DQN) maintains a circular replay buffer of capacity $B$ (typically $B \approx 10^6$). At each environment step, we store the new transition and sample a mini-batch of size $b$ from the buffer for training. This increases the replay ratio, reducing gradient variance at the cost of older, potentially off-policy data. 

DQN uses a separate target network $\boldsymbol{\theta}_{\text{target}}$ that updates every $K$ steps (typically $K \approx 10^4$). As shown in the nested-to-flattened section, this target network is not a stabilization trick but the natural consequence of periodic outer-iteration boundaries in flattened FQI. The target network keeps targets fixed for $K$ gradient steps, which corresponds to the inner loop in the nested view:

```{prf:algorithm} Deep Q-Network (DQN)
:label: dqn

**Input**: MDP $(S, A, P, R, \gamma)$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, target update frequency $K$, replay buffer capacity $B$, mini-batch size $b$, exploration rate $\varepsilon$

**Output**: Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_0$
3. Initialize replay buffer $\mathcal{B}$ with capacity $B$
4. $t \leftarrow 0$
5. **while** training **do**
    1. Observe current state $s$
    2. Select action: $a \leftarrow \begin{cases} \arg\max_{a'} q(s,a';\boldsymbol{\theta}_t) & \text{with probability } 1-\varepsilon \\ \text{random action} & \text{with probability } \varepsilon \end{cases}$
    3. Execute $a$, observe reward $r$ and next state $s'$
    4. Store $(s,a,r,s')$ in $\mathcal{B}$, replacing oldest if full
    5. Sample mini-batch of $b$ transitions $\{(s_i,a_i,r_i,s_i')\}_{i=1}^b$ from $\mathcal{B}$
    6. For each sampled transition $(s_i,a_i,r_i,s_i')$:
        1. $y_i \leftarrow r_i + \gamma \max_{a' \in A} q(s_i',a'; \boldsymbol{\theta}_{\text{target}})$
    7. $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_{i=1}^b (q(s_i,a_i;\boldsymbol{\theta}_t) - y_i)^2$
    8. **if** $t \bmod K = 0$ **then**
        1. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_t$
    9. $t \leftarrow t + 1$
6. **return** $\boldsymbol{\theta}_t$
```

Double DQN addresses overestimation bias by decoupling action selection from evaluation. The online network selects the action, while the target network evaluates it:

```{prf:algorithm} Double Deep Q-Network (Double DQN)
:label: double-dqn

**Input**: MDP $(S, A, P, R, \gamma)$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, target update frequency $K$, replay buffer capacity $B$, mini-batch size $b$, exploration rate $\varepsilon$

**Output**: Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_0$
3. Initialize replay buffer $\mathcal{B}$ with capacity $B$
4. $t \leftarrow 0$
5. **while** training **do**
    1. Observe current state $s$
    2. Select action: $a \leftarrow \begin{cases} \arg\max_{a'} q(s,a';\boldsymbol{\theta}_t) & \text{with probability } 1-\varepsilon \\ \text{random action} & \text{with probability } \varepsilon \end{cases}$
    3. Execute $a$, observe reward $r$ and next state $s'$
    4. Store $(s,a,r,s')$ in $\mathcal{B}$, replacing oldest if full
    5. Sample mini-batch of $b$ transitions $\{(s_i,a_i,r_i,s_i')\}_{i=1}^b$ from $\mathcal{B}$
    6. For each sampled transition $(s_i,a_i,r_i,s_i')$:
        1. $a^*_i \leftarrow \arg\max_{a' \in A} q(s_i',a'; \boldsymbol{\theta}_t)$
        2. $y_i \leftarrow r_i + \gamma q(s_i',a^*_i; \boldsymbol{\theta}_{\text{target}})$
    7. $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_{i=1}^b (q(s_i,a_i;\boldsymbol{\theta}_t) - y_i)^2$
    8. **if** $t \bmod K = 0$ **then**
        1. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_t$
    9. $t \leftarrow t + 1$
6. **return** $\boldsymbol{\theta}_t$
```

Double DQN leaves $\mathcal{B}_t$ and $\hat{P}_{\mathcal{B}_t}$ unchanged and modifies only the target map $T_q$ by decoupling action selection (online network) from evaluation (target network).

### Q-Learning: The Limiting Case

DQN and Double DQN use moderate settings: $B \approx 10^6$, $K \approx 10^4$, mini-batch size $b \approx 32$. We can ask: what happens at the extreme where we minimize buffer capacity, eliminate replay, and update targets every step? This limiting case gives classical Q-learning {cite}`watkins1989learning`: buffer capacity $B=1$, target network frequency $K=1$, and mini-batch size $b=1$. 

In the buffer perspective, Q-learning has $\mathcal{B}_t = \{(s_t, a_t, r_t, s'_t)\}$ with $|\mathcal{B}_t | = 1$. The empirical distribution $\hat{P}_{\mathcal{B}_t}$ collapses to a Dirac mass at the current transition. We use each transition exactly once then discard it. There is no separate target network: the parameters used for computing targets are immediately updated after each step ($K=1$, or equivalently $\tau=1$ in EMA). This makes Q-learning a **stochastic approximation** method with replay ratio 1.

**Stochastic approximation** is a general framework for solving equations of the form $\mathbb{E}[h(X; \boldsymbol{\theta})] = 0$ using noisy samples, without computing expectations explicitly. The classic example is root-finding: given function $g(\boldsymbol{\theta})$ whose expectation $\mathbb{E}[g(\boldsymbol{\theta}; Z)] = G(\boldsymbol{\theta})$ we want to solve $G(\boldsymbol{\theta}) = 0$, the Robbins-Monro procedure updates:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha_t g(\boldsymbol{\theta}_t; Z_t)
$$

using noisy samples $Z_t$ without ever computing $G(\boldsymbol{\theta})$ or its Jacobian. This is analogous to Newton's method in the deterministic case, but replaces exact gradients with stochastic estimates and avoids computing or inverting the Jacobian. Under diminishing step sizes ($\alpha_t \to 0$, $\sum_t \alpha_t = \infty$), the iterates converge to solutions of $G(\boldsymbol{\theta}) = 0$.

Q-learning fits this framework by solving the Bellman residual equation. Recall from the [projection methods chapter](projdp.md) that the Bellman equation $q^* = \Bellman q^*$ can be written as a residual equation $\Residual(q) \equiv \Bellman q - q = 0$. For a parameterized Q-function $q(s,a; \boldsymbol{\theta})$, the residual at observed transition $(s,a,r,s')$ is:

$$
R(s,a,r,s'; \boldsymbol{\theta}) = r + \gamma \max_{a'} q(s',a'; \boldsymbol{\theta}) - q(s,a; \boldsymbol{\theta})
$$

This is the TD error. Q-learning is a stochastic approximation method for solving $\mathbb{E}_{\tau \sim P}[R(\tau; \boldsymbol{\theta})] = 0$ where $P$ is the distribution of transitions under the behavior policy. Each observed transition provides a noisy sample of the residual, and the algorithm updates parameters using the gradient of the squared residual without computing expectations explicitly.

The algorithm works with any function approximator that supports incremental updates. The general form applies a gradient descent step to minimize the squared TD error. The only difference between linear and nonlinear cases is how the gradient $\nabla_{\boldsymbol{\theta}} q(s,a; \boldsymbol{\theta})$ is computed.

```{prf:algorithm} Q-Learning
:label: q-learning

**Input**: MDP $(S, A, P, R, \gamma)$, function approximator $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, exploration rate $\varepsilon$

**Output**: Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$
2. $t \leftarrow 0$
3. **while** training **do**
    1. Observe current state $s$
    2. Select action: $a \leftarrow \begin{cases} \arg\max_{a' \in A} q(s,a';\boldsymbol{\theta}_t) & \text{with probability } 1-\varepsilon \\ \text{random action} & \text{with probability } \varepsilon \end{cases}$
    3. Execute $a$, observe reward $r$ and next state $s'$
    4. Compute TD target: $y \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_t)$
    5. Update: $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} (q(s,a;\boldsymbol{\theta}_t) - y)^2$
    6. $t \leftarrow t + 1$
4. **return** $\boldsymbol{\theta}_t$
```

The gradient in line 5 depends on the choice of function approximator:

- **Tabular**: Uses one-hot features $\boldsymbol{\phi}(s,a) = \boldsymbol{e}_{(s,a)}$, giving table-lookup updates $Q(s,a) \leftarrow Q(s,a) + \alpha(r + \gamma \max_{a'} Q(s',a') - Q(s,a))$. This converges under standard stochastic approximation conditions (diminishing step sizes $\alpha_t \to 0$ with $\sum_t \alpha_t = \infty$, sufficient exploration).

- **Linear**: $q(s,a; \boldsymbol{\theta}) = \boldsymbol{\theta}^\top \boldsymbol{\phi}(s,a)$ where $\boldsymbol{\phi}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}^d$ is a feature map. The gradient is $\nabla_{\boldsymbol{\theta}} q(s,a; \boldsymbol{\theta}) = \boldsymbol{\phi}(s,a)$, giving the update $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha (y - \boldsymbol{\theta}_t^\top \boldsymbol{\phi}(s,a)) \boldsymbol{\phi}(s,a)$. Convergence is not guaranteed in general; the max operator combined with function approximation can cause instability.

- **Nonlinear**: $q(s,a; \boldsymbol{\theta})$ is a neural network with weights $\boldsymbol{\theta}$. The gradient $\nabla_{\boldsymbol{\theta}} q(s,a; \boldsymbol{\theta})$ is computed via backpropagation. Convergence guarantees do not exist.

## Regression Losses and Noise Models

Fix a particular time $t$ and buffer contents $\mathcal{B}_t$. Sampling from $\mathcal{B}_t$ and pushing transitions through the target map $T_{q_t}$ gives a regression distribution $\hat{P}_t^{\text{fit}}$ over pairs $((s,a), y)$. The `fit` operation in the inner loop is then a standard statistical estimation problem: given empirical samples from $\hat{P}_t^{\text{fit}}$, choose parameters $\boldsymbol{\theta}$ to minimize a loss:

$$
\mathcal{L}(\boldsymbol{\theta}; \mathcal{B}_t) \approx \mathbb{E}_{((s,a),y)\sim \hat{P}_t^{\text{fit}}} \big[\ell(q(s,a;\boldsymbol{\theta}), y)\big]
$$

The choice of loss $\ell$ implicitly specifies a noise model for the targets $y$ under $\hat{P}_t^{\text{fit}}$. Squared error corresponds to Gaussian noise, absolute error to Laplace noise, Gumbel regression to extreme-value noise, and classification losses to non-parametric noise models over value bins. Because Bellman targets are noisy, biased, and bootstrapped, this choice has a direct impact on how the algorithm interprets the empirical distribution $\hat{P}_t^{\text{fit}}$.

This section examines alternative loss functions for the regression step. The standard approach uses squared error, but the noise in Bellman targets has special structure due to the max operator and bootstrapping. Two strategies have shown empirical success: Gumbel regression, which uses the proper likelihood for extreme-value noise, and classification-based methods, which avoid parametric noise assumptions by working with distributions over value bins.

### Gumbel Regression

Extreme value theory tells us that the maximum of Gaussian errors has Gumbel-distributed tails. If we take this distribution seriously, maximum likelihood estimation should use a Gumbel likelihood rather than a Gaussian one. Garg, Tang, Kahn, and Levine {cite}`garg2023extreme` developed this idea in Extreme Q-Learning (XQL). Instead of modeling the Bellman error as additive Gaussian noise:

$$
y_i = q^*(s_i, a_i) + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

they model it as Gumbel noise:

$$
y_i = q^*(s_i, a_i) + \varepsilon_i, \quad \varepsilon_i \sim -\text{Gumbel}(0, \beta)
$$

The negative Gumbel distribution arises because we are modeling errors in targets that overestimate the true value. The corresponding maximum likelihood loss is Gumbel regression:

$$
\mathcal{L}_{\text{Gumbel}}(\boldsymbol{\theta}) = \sum_i \left[\frac{q(s_i, a_i; \boldsymbol{\theta}) - y_i}{\beta} + \exp\left(\frac{q(s_i, a_i; \boldsymbol{\theta}) - y_i}{\beta}\right)\right]
$$

The temperature parameter $\beta$ controls the heaviness of the tail. The score function (gradient with respect to $q$) is:

$$
\frac{\partial \mathcal{L}_{\text{Gumbel}}}{\partial q} = \frac{1}{\beta}\left[1 + \exp\left(\frac{q - y}{\beta}\right)\right]
$$

When $q < y$ (underestimation), the exponential term is small and the gradient is mild. When $q > y$ (overestimation), the gradient grows exponentially with the error. This asymmetry deliberately penalizes overestimation more heavily than underestimation. When targets are systematically biased upward due to the max operator, this loss geometry pushes the estimates toward conservative Q-values.

The Gumbel loss can be understood as the natural likelihood for problems involving max operators, just as the Gaussian is the natural likelihood for problems involving averages. The central limit theorem tells us that sums converge to Gaussians; extreme value theory tells us that maxima converge to Gumbel (for light-tailed base distributions). Squared error is optimal for Gaussian noise; Gumbel regression is optimal for Gumbel noise.

The practical advantage is that we do not need to estimate variances or compute weighted averages. The loss function itself handles the asymmetric error structure through its score function. XQL has shown improvements in both value-based and actor-critic methods, particularly in offline reinforcement learning where the max-operator bias compounds across iterations without corrective exploration.

### Classification-Based Q-Learning

From the buffer viewpoint, nothing changes upstream: we still sample transitions from $\hat{P}_{\mathcal{B}_t}$ and apply the same target map $T_{q_t}$. Classification-based Q-learning changes only the loss $\ell$ and target representation. Instead of regressing on a scalar $y\in\mathbb{R}$ with L2, we represent values as categorical distributions over bins and use cross-entropy loss.

Choose a finite grid $z_1 < z_2 < \cdots < z_K$ spanning plausible return values. The network outputs logits $\ell_{\boldsymbol{\theta}}(s,a) \in \mathbb{R}^K$ converted to probabilities:

$$
p_{\boldsymbol{\theta}}(k \mid s,a) = \frac{\exp(\ell_{\boldsymbol{\theta}}(s,a)_k)}{\sum_{j=1}^K \exp(\ell_{\boldsymbol{\theta}}(s,a)_j)}, \quad q(s,a; \boldsymbol{\theta}) = \sum_{k=1}^K z_k \, p_{\boldsymbol{\theta}}(k \mid s,a)
$$

Each scalar TD target $y_i$ is converted to a target distribution via the two-hot encoding. If $y_i$ falls between bins $z_j$ and $z_{j+1}$:

$$
q_j(y_i) = \frac{z_{j+1} - y_i}{z_{j+1} - z_j}, \quad q_{j+1}(y_i) = \frac{y_i - z_j}{z_{j+1} - z_j}, \quad q_k(y_i) = 0 \text{ for } k \notin \{j, j+1\}
$$

This is barycentric interpolation: $\sum_k z_k q_k(y_i) = y_i$ recovers the scalar exactly. The loss minimizes cross-entropy:

$$
\mathcal{L}_{\text{CE}}(\boldsymbol{\theta}) = -\mathbb{E}_{((s,a),y) \sim \hat{P}_t^{\text{fit}}}\left[ \sum_{k=1}^K q_k(y) \log p_{\boldsymbol{\theta}}(k \mid s, a) \right]
$$

which projects the target distribution onto the predicted distribution in KL geometry on the simplex $\Delta^{K-1}$ rather than L2 on $\mathbb{R}$. The gradient is $\nabla_{\ell_\theta} \mathcal{L}_{\text{CE}} = p_\theta - q$, bounded in magnitude regardless of target size.

This provides three sources of implicit robustness. First, gradient influence is bounded: each sample contributes $O(1)$ gradient magnitude per bin, unlike L2 where error magnitude $E$ contributes gradient proportional to $E$. Second, the finite grid $[z_1, z_K]$ clips extreme targets to boundary bins, preventing outliers from dominating the regression scale. Third, the two-hot encoding spreads mass across neighboring bins, providing label smoothing that averages noisy targets at the same $(s,a)$.

The two-hot weights $q_j(y_i), q_{j+1}(y_i)$ are barycentric coordinates, identical to linear interpolation in the [dynamic programming chapter](dp.md) (Algorithm {prf:ref}`backward-recursion-interp`). This places the encoding within Gordon's monotone approximator framework (Definition {prf:ref}`gordon-averager`): targets are convex combinations preserving order and boundedness. The neural network predicting $p_{\boldsymbol{\theta}}(\cdot \mid s,a)$ is non-monotone, making classification-based Q-learning a hybrid: monotone target structure paired with flexible function approximation.

Empirically, cross-entropy loss scales better with network capacity. Farebrother et al. {cite}`farebrother2024stop` found that L2-based DQN and CQL degrade when Q-networks scale to large ResNets, while classification loss (specifically HL-Gauss, which uses Gaussian smoothing instead of two-hot) maintains performance. The combination of KL geometry, quantization, and smoothing prevents overfitting to noisy targets that plagues L2 with high-capacity networks.

### Practical Considerations

Implementing classification-based Q-learning requires choosing the number of bins $K$ and their range $[z_1, z_K]$. Typical choices use $K \in \{51, 101, 201\}$ bins uniformly spaced over the expected return range. The range can be estimated from domain knowledge or learned adaptively during training. The network architecture changes minimally: instead of a single scalar output per action, we output $K$ logits per action. For discrete action spaces with $|\mathcal{A}|$ actions, this means a final layer of size $K \times |\mathcal{A}|$ rather than $|\mathcal{A}|$. The computational overhead is modest, and the implementation can use standard cross-entropy loss functions available in deep learning libraries. The main conceptual shift is viewing Q-learning as categorical prediction rather than scalar regression.

Gumbel regression and classification-based Q-learning represent two strategies for matching the loss function to the noise structure in TD targets. Gumbel regression commits to an extreme-value noise model and uses the corresponding likelihood. Classification avoids parametric assumptions by working with distributions over bins, changing both the target representation and the projection geometry from $L^2$ to KL divergence. The choice depends on whether the Gumbel assumption is justified and whether the computational overhead of maintaining $K$ bins per action is acceptable.

## Summary

Fitted Q-iteration has a two-level structure: an outer loop applies the Bellman operator to construct targets, an inner loop fits a function approximator to those targets. All algorithms in this chapter instantiate this template with different choices of buffer $\mathcal{B}_t$, target map $T_q$, loss $\ell$, and optimization budget $K$.

The empirical distribution $\hat{P}_{\mathcal{B}_t}$ unifies offline and online methods through plug-in approximation: replace unknown transition law $P$ with $\hat{P}_{\mathcal{B}_t}$ and minimize empirical risk $\mathbb{E}_{((s,a),y)\sim \hat{P}_t^{\text{fit}}} [\ell(q, y)]$. Offline uses fixed $\mathcal{B}_t \equiv \mathcal{D}$ (sample average approximation), online uses circular buffer (Q-learning with $B=1$ is stochastic approximation, DQN with large $B$ is hybrid).

Target networks arise from flattening the nested loops. Merging inner gradient steps with outer value iteration creates a single loop where the parameters for targets, $\boldsymbol{\theta}_{\text{target}}$, update every $K$ steps to mark outer-iteration boundaries.

The [next chapter](cadp.md) directly parameterizes and optimizes policies instead of searching over value functions.
