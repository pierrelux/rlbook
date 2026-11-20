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

# Deep Q-Networks: Target Networks, Replay and Online Data

The [previous chapter](monte-carlo-bellman-estimation.md) ended with fitted
Q-iteration on a fixed log of transitions: an outer loop that computes Bellman
targets and an inner loop that fits a function to them, run to completion.
This chapter opens the inner loop. Writing out the gradient steps of a neural
fit and merging them with the outer loop turns fitted Q-iteration into DQN,
with its target network and its online network, and letting the buffer evolve
during learning gives DQN proper, Double DQN, and, at the limit of one
transition per update, Q-learning. A synthesis section lays out the design
space that these choices span, and the chapter closes with losses that model
the noise of Bellman targets better than squared error.

How do the buffer, the Bellman target, the function approximator and the
optimization budget combine when only one observed successor is available
for each stored transition?

::::{admonition} Learning Goals
:class: note

After completing this chapter, you should be able to:
- Flatten the nested loops of neural fitted Q-iteration into DQN, and explain the roles of its online and target networks
- Explain what Double DQN changes in the target, and why Q-learning is the limit of one transition per update
- Place an algorithm in the design space of fitted Q-iteration by its buffer, target function, fit, initialization and loss
- Choose a regression loss for Bellman targets as a model of their noise

**Prerequisites:** fitted Q-iteration on a fixed set of transitions, action values, maximization bias and the error bounds of fitted value iteration ([fitted value methods](monte-carlo-bellman-estimation.md)).
::::

(sec-fqi-flattening)=
## From Nested Loops to Target Networks

Fitted Q-iteration has a nested structure: the outer loop computes Bellman
targets, and the inner loop fits the function approximator to them. For a
tree ensemble the inner loop is one call that builds the trees, and the
nesting is the only possible presentation. For a neural network the inner
loop is a sequence of gradient steps,

$$
\boldsymbol{\theta}_n^{(k+1)} = \boldsymbol{\theta}_n^{(k)} - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}\big(\boldsymbol{\theta}_n^{(k)}; \mathcal{D}_n^{\text{fit}}\big), \qquad k = 0, \ldots, K-1,
$$

starting from $\boldsymbol{\theta}_n^{(0)} = \boldsymbol{\theta}_n$ and
ending with $\boldsymbol{\theta}_{n+1} = \boldsymbol{\theta}_n^{(K)}$, where
$\mathcal{D}_n^{\text{fit}}$ is the regression data set of the $n$-th outer
iteration. Two loops of counted steps can be written as one, and the
rewriting shows where the target network of DQN comes from.

### Making the Nested Structure of Batch FQI Explicit

Starting from {prf:ref}`fitted-q-iteration-batch`, replacing the `fit` call
by $K$ explicit gradient steps gives the following.

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

The targets of outer iteration $n$ are computed with $\boldsymbol{\theta}_n$,
the parameters at the end of the previous outer iteration, and they stay
fixed for all $K$ inner steps. The inner iterates
$\boldsymbol{\theta}_n^{(k)}$ for $k > 0$ never enter a target. The fit is
warm-started from $\boldsymbol{\theta}_n$, the very parameters that computed
the targets, and it stops after $K$ steps rather than at convergence: these
are the two departures from supervised learning that neural fitted
Q-iteration introduced, and they are kept by everything that follows.

### Flattening into a Single Loop

Replace the pair of counters $(n, k)$ by a single step counter $t = nK + k$.
The parameters that compute the targets, $\boldsymbol{\theta}_n$, are held
in a separate copy that changes only when $t$ crosses a multiple of $K$, the
boundary between two outer iterations. That copy is the **target network**
$\boldsymbol{\theta}_{\text{target}}$. The parameters that the gradient
steps update, $\boldsymbol{\theta}_t$, correspond to the inner iterates
$\boldsymbol{\theta}_n^{(k)}$, and are called the **online network**, online
in the sense of being continually updated, not in the sense of how the data
were collected. Nothing else changes.

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

At step $t$, the outer iteration is $n = \lfloor t/K \rfloor$ and the inner
step is $k = t \bmod K$. The target network equals $\boldsymbol{\theta}_n$
for $t$ from $nK$ to $(n+1)K - 1$ and becomes $\boldsymbol{\theta}_{n+1}$ at
$t = (n+1)K$, and $\boldsymbol{\theta}_t$ is $\boldsymbol{\theta}_n^{(k)}$.
With $K = 10$, the inner step $k = 7$ of outer iteration $n = 3$ is the
global step $t = 37$. The data set is still the fixed $\mathcal{D}$; only the
indexing has changed. A target network, in this reading, is the boundary of
an outer iteration of fitted Q-iteration, kept in memory. When DQN is
presented with a target network updated every $K$ steps, it is this
flattened fitted Q-iteration with an evolving buffer, which the next section
adds.

### Smooth Target Updates by Exponential Moving Average

The periodic copy can be replaced by a target network that tracks the online
network a little at every step,

$$
\boldsymbol{\theta}_{\text{target}} \leftarrow \tau\,\boldsymbol{\theta}_{t+1} + (1-\tau)\,\boldsymbol{\theta}_{\text{target}},
$$

an **exponential moving average** (also called Polyak averaging) with rate
$\tau \in (0, 1)$. For small $\tau$, typically between $0.001$ and $0.01$,
the target lags the online network by roughly $1/\tau$ steps, and the targets
drift continuously instead of jumping every $K$ steps. The scheme became
standard with DDPG {cite}`lillicrap2015continuous` and is used by TD3
{cite}`fujimoto2018addressing` and SAC {cite}`haarnoja2018soft`.

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
    6. **// CHANGED: Smooth EMA update at every step**
    7. $\boldsymbol{\theta}_{\text{target}} \leftarrow \tau\boldsymbol{\theta}_{t+1} + (1-\tau)\boldsymbol{\theta}_{\text{target}}$
    8. $t \leftarrow t + 1$
5. **return** $\boldsymbol{\theta}_t$
```

(sec-fqi-online)=
## DQN: Fitted Q-Iteration on a Replay Buffer

The flattened algorithm sampled from a fixed data set. Let the data set
change instead: a **replay buffer** $\mathcal{B}_t$ of fixed capacity $B$
receives one new transition at every step of interaction and drops the
oldest when full, so that $\mathcal{B}_t = \{\tau_{t-B+1}, \ldots, \tau_t\}$
and its empirical distribution $\hat{P}_{\mathcal{B}_t}$ slides forward in
time. Each gradient step uses a mini-batch of $b$ transitions drawn
uniformly from the current buffer. The online network acquires a second
role: besides being the parameters under training, it is the policy that
collects the new transitions, through $\varepsilon$-greedy action selection.
"Online" now has both meanings at once, continually updated parameters and
data collected during learning.

Deep Q-Network (DQN) {cite}`mnih2013atari` is this algorithm with a
convolutional network and moderate settings: a buffer of about $10^6$
transitions, mini-batches of about 32, and a target network copied every
$K \approx 10^4$ steps. It is flattened fitted Q-iteration with an evolving
buffer, and each of its components has a place in the nested view: the
target network marks the outer iterations, the mini-batch gradient step is
the inner loop, and the buffer is the data set, now changing slowly under
the policy that the online network defines.

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
    2. **// Use online network to collect data**
    3. Select action: $a \leftarrow \begin{cases} \arg\max_{a'} q(s,a';\boldsymbol{\theta}_t) & \text{with probability } 1-\varepsilon \\ \text{random action} & \text{with probability } \varepsilon \end{cases}$
    4. Execute $a$, observe reward $r$ and next state $s'$
    5. Store $(s,a,r,s')$ in $\mathcal{B}$, replacing oldest if full
    6. Sample mini-batch of $b$ transitions $\{(s_i,a_i,r_i,s_i')\}_{i=1}^b$ from $\mathcal{B}$
    7. For each sampled transition $(s_i,a_i,r_i,s_i')$:
        1. $y_i \leftarrow r_i + \gamma \color{blue}{\max_{a' \in A} q(s_i',a'; \boldsymbol{\theta}_{\text{target}})}$ (both selection AND evaluation with target network)
    8. $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_{i=1}^b (q(s_i,a_i;\boldsymbol{\theta}_t) - y_i)^2$
    9. **if** $t \bmod K = 0$ **then**
        1. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_t$
    10. $t \leftarrow t + 1$
6. **return** $\boldsymbol{\theta}_t$
```

The deadly triad of the [previous chapter](monte-carlo-bellman-estimation.md#sec-mc-deadly-triad)
is all here: a neural network, bootstrapped targets, and a buffer whose
transitions were collected by earlier policies. The design choices are what
keep it in check. Targets are held fixed for $K$ steps by the target network,
each fit is warm-started, and the large buffer averages the data over many
policies, so that the distribution of the data used by each fit changes slowly even though
the policy changes at every step.

Recent analyses read these choices in the language of projected equations.
{cite:t}`ZhangYaoWhiteson2021` show that a target network, combined with a
ridge penalty on the parameters, gives a linear off-policy Q-learning method
that provably converges, and {cite:t}`Fellows2023` treat the target network as
a partially fitted iteration and identify conditions on the Jacobian of the TD
update under which it restores stability. {cite:t}`Gallici2025` show that
layer normalization in the value network makes TD stable without a target
network, even off-policy.

(sec-fqi-double-dqn)=
### Double DQN

The target of DQN, $\max_{a'} q(s', a'; \boldsymbol{\theta}_{\text{target}})$,
uses the target network twice, to select the best next action and to value
it, and the [previous chapter](monte-carlo-bellman-estimation.md#sec-mc-overestimation)
showed that such a maximum is biased upward by the reuse of the winning
noise. Double DQN {cite}`van2016deep` separates the two roles between the
two networks that DQN already maintains: the online network selects,
$a^*_i = \arg\max_{a'} q(s'_i, a'; \boldsymbol{\theta}_t)$, and the target
network evaluates, $y_i = r_i + \gamma\, q(s'_i, a^*_i; \boldsymbol{\theta}_{\text{target}})$.
The buffer and the fit are unchanged; only the target function differs.

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
    2. **// Use online network to collect data**
    3. Select action: $a \leftarrow \begin{cases} \arg\max_{a'} q(s,a';\boldsymbol{\theta}_t) & \text{with probability } 1-\varepsilon \\ \text{random action} & \text{with probability } \varepsilon \end{cases}$
    4. Execute $a$, observe reward $r$ and next state $s'$
    5. Store $(s,a,r,s')$ in $\mathcal{B}$, replacing oldest if full
    6. Sample mini-batch of $b$ transitions $\{(s_i,a_i,r_i,s_i')\}_{i=1}^b$ from $\mathcal{B}$
    7. For each sampled transition $(s_i,a_i,r_i,s_i')$:
        1. $a^*_i \leftarrow \color{red}{\arg\max_{a' \in A} q(s_i',a'; \boldsymbol{\theta}_t)}$ (action selection with online network)
        2. $y_i \leftarrow r_i + \gamma \color{green}{q(s_i',a^*_i; \boldsymbol{\theta}_{\text{target}})}$ (action evaluation with target network)
    8. $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_{i=1}^b (q(s_i,a_i;\boldsymbol{\theta}_t) - y_i)^2$
    9. **if** $t \bmod K = 0$ **then**
        1. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_t$
    10. $t \leftarrow t + 1$
6. **return** $\boldsymbol{\theta}_t$
```

The two networks are not independent estimates in the sense of the Monte
Carlo chapter's argument, since the target network is a delayed copy of the
online one. Their errors are nonetheless less correlated than the errors of
one network used twice, because they were fitted to different targets at
different times, and the reduction in overestimation that this buys is
substantial in practice.

### Q-Learning as the Limit of One Transition per Update

DQN uses a buffer of about $10^6$ transitions, a target network updated
every $10^4$ steps and mini-batches of 32. At the opposite extreme, with a
buffer of one transition, a target network updated at every step and a
mini-batch of one, the algorithm is Q-learning {cite}`watkins1989learning`.
Each transition is used once and discarded, $\hat{P}_{\mathcal{B}_t}$ is a
point mass at the current transition, and the parameters that compute the
target are the ones just updated.

Q-learning is a **stochastic approximation** method. The Robbins–Monro
procedure solves an equation $F(\boldsymbol{\theta}) = 0$ in which
$F(\boldsymbol{\theta}) = \mathbb{E}[f(\boldsymbol{\theta}; Z)]$ is an
expectation that is never computed, by the update
$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha_t f(\boldsymbol{\theta}_t; Z_t)$
on one noisy draw $Z_t$ at a time; under step sizes with $\alpha_t \to 0$
and $\sum_t \alpha_t = \infty$ the iterates converge to a solution. For
Q-learning the equation is the Bellman equation in residual form,
$\Bellman q - q = 0$, and the draw is a transition. The residual at a
transition $(s,a,r,s')$ is the **TD error**

$$
R(s,a,r,s'; \boldsymbol{\theta}) = r + \gamma \max_{a'} q(s',a'; \boldsymbol{\theta}) - q(s,a; \boldsymbol{\theta}),
$$

whose expectation over transitions is the Bellman residual at $(s,a)$,
averaged over the distribution of the behavior policy. Each transition is
one noisy sample of it, and the update takes a gradient step on the squared
TD error with the target held fixed, the semi-gradient of the approximate
Bellman chapter.

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

The gradient in line 5 depends on the function class. With one-hot features
$\boldsymbol{\phi}(s,a) = \boldsymbol{e}_{(s,a)}$ the update is the tabular
rule $Q(s,a) \leftarrow Q(s,a) + \alpha(r + \gamma \max_{a'} Q(s',a') - Q(s,a))$,
which converges under the stochastic-approximation conditions. With linear
features $q(s,a;\boldsymbol\theta) = \boldsymbol{\theta}^\top \boldsymbol{\phi}(s,a)$
the gradient is $\boldsymbol{\phi}(s,a)$, and convergence is not guaranteed:
least squares is not an averager, and the max operator keeps the on-policy
argument from applying. With a neural network the gradient comes from
backpropagation and there are no guarantees at all.

### Convergence Analysis by the ODE Method

Tabular Q-learning converges under diminishing step sizes and sufficient
exploration {cite}`watkins1992qlearning,jaakkola1994convergence,tsitsiklis1994asynchronous`.
The standard proof technique is the **ordinary differential equation (ODE)
method** {cite}`kushner2003stochastic,borkar2000ode`, which studies a
stochastic approximation through an associated deterministic dynamical
system. The argument has two steps. As the step sizes shrink, the stochastic
recursion $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha_t h(\boldsymbol{\theta}_t, Z_t)$
tracks the ODE $\dot{\boldsymbol{\theta}} = \bar{h}(\boldsymbol{\theta})$,
where $\bar{h}(\boldsymbol{\theta}) = \mathbb{E}[h(\boldsymbol{\theta}, Z)]$
is the expected update direction. Then the ODE is shown to converge, with a
Lyapunov function, and the stochastic algorithm converges to the same point.

For tabular Q-learning, the update at the visited pair $(s,a)$ is
$Q_{t+1}(s,a) = Q_t(s,a) + \alpha_t [r + \gamma \max_{a'} Q_t(s',a') - Q_t(s,a)]$,
and its expectation over the transition, given that $(s,a)$ is visited, is
the Bellman residual at $(s,a)$. The ODE is

$$
\frac{dQ(s,a)}{dt} = \sum_{s'} p(s'|s,a) \big[r(s,a,s') + \gamma \max_{a'} Q(s',a') \big] - Q(s,a),
$$

whose equilibria are the solutions of the Bellman optimality equation, that
is, $Q^*$ alone. The distance $\|Q - Q^*\|_\infty$ decreases along its
trajectories because the Bellman operator is a contraction, which is the
Lyapunov function. Exploration enters through the requirement that every
pair be visited infinitely often; under ergodicity the frequency with which
each $(s,a)$ is updated converges to the stationary distribution of the
behavior policy, which the ODE absorbs as a per-coordinate time scale. With
linear features the ODE no longer has the Bellman residual as its drift at
every coordinate, the projection enters, and the [approximate Bellman
chapter](approximate-bellman-equations.md) showed what can go wrong then.

(sec-fqi-design-space)=
## The Design Space of Fitted Q-Iteration

Every algorithm above has the same two-level structure, with three
ingredients at any time $t$:

1. A **buffer** $\mathcal{B}_t$ of transitions, with its empirical
   distribution $\hat{P}_{\mathcal{B}_t}$. It is a fixed data set
   $\mathcal{D}$ in the batch algorithms, a circular buffer of capacity $B$
   in DQN, and the single current transition in Q-learning.
2. A **target function** $g(s,a,r,s'; \boldsymbol{\theta})$ that computes a
   regression target from one transition and the parameters of the current
   Q-function: $r + \gamma \max_{a'} q(s',a';\boldsymbol\theta)$ for the hard
   maximum, the log-sum-exp of the [regularized chapter](regularized-dp.md)
   for the smooth one, and the decoupled selection and evaluation of Double
   DQN. Unlike the Bellman operator, which acts on functions, $g$ acts on
   tuples.
3. A **fit**: a loss $\ell$, an initialization $\boldsymbol{\theta}_{\text{init}}$,
   and an optimization budget $K$.

Pushing the buffer through the target function turns the distribution over
transitions into a distribution over regression pairs,
$\hat{P}_t^{\text{fit}} = (\mathrm{id}, g(\cdot;\boldsymbol\theta_t))_\#\hat{P}_{\mathcal{B}_t}$,
and the fit minimizes the empirical risk
$\mathbb{E}_{((s,a),y) \sim \hat{P}_t^{\text{fit}}}[\ell(q(s,a;\boldsymbol\theta), y)]$.
The buffer distribution changes only when data are collected; the
regression distribution changes every time the targets are recomputed. In
the abstract, the whole family is

$$
\begin{aligned}
&\textbf{repeat } n = 0, 1, 2, \ldots \\
&\quad \text{Sample transitions from } \hat{P}_{\mathcal{B}_n} \text{ and construct targets via } g(\cdot; \boldsymbol{\theta}_n) \\
&\quad \boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\hat{P}_n^{\text{fit}}, \boldsymbol{\theta}_{\text{init}}, K) \\
&\textbf{until } \text{convergence.}
\end{aligned}
$$

How often the data are reused is measured by the **replay ratio**, the
number of optimization steps per transition collected:

$$
\text{Replay ratio} = \begin{cases}
K \cdot N_{\text{epochs}} \cdot N / b & \text{batch (batch size } b \text{ from } N \text{ transitions)} \\
b & \text{DQN (one step per transition, batch size } b \text{)} \\
1 & \text{Q-learning.}
\end{cases}
$$

A large replay ratio reuses the same empirical distribution many times,
which is sample-average approximation; a replay ratio of one uses each
transition once, which is stochastic approximation. More reuse lowers the
variance of the estimates under $\hat{P}_{\mathcal{B}_t}$ and raises the
risk of fitting the idiosyncrasies of a buffer that reflects old policies or
narrow coverage.

```{prf:remark} Two notions of bootstrap
:class: dropdown

Two different "bootstraps" meet in fitted Q-iteration. Sampling with replacement from $\hat{P}_{\mathcal{B}_t}$ approximates expectations under the unknown $P$ by expectations under the empirical distribution, which is the bootstrap of statistics; mini-batch training is this resampling, and the replay ratio is the number of bootstrap samples per transition. Computing $y = r + \gamma \max_{a'} q(s',a';\boldsymbol{\theta})$ replaces the unknown continuation value by the current estimate, which is the bootstrap of temporal-difference learning and the source of the maximization bias of the previous chapter. The buffer encodes the first, the target function the second, and the bias corrections (double estimation, learned corrections, Gumbel losses) act on the second alone.
```

The table places the algorithms of this chapter, and of the next, in this
space.

| **Algorithm** | **Approximator** | **Target** | **Fit budget** | **Initialization** | **Data** | **Bias fix** |
|:--------------|:-----------------|:-----------|:---------------|:-------------------|:---------|:-------------|
| FQI {cite}`ernst2005tree` | Extra trees | Hard max | Full | Cold | Fixed batch | None |
| NFQI {cite}`riedmiller2005neural` | Neural net | Hard max | Full | Warm | Fixed batch | None |
| Q-learning {cite}`watkins1989learning` | Any | Hard max | $K=1$ | Warm | One transition | None |
| DQN {cite}`mnih2013atari` | Deep net | Hard max | $K=1$ | Warm | Replay buffer | None |
| Double DQN {cite}`van2016deep` | Deep net | Decoupled max | $K=1$ | Warm | Replay buffer | Double estimation |
| Soft Q {cite}`haarnoja2017reinforcement` | Neural net | Log-sum-exp | $K$ steps | Warm | Replay buffer | None |

The continuous-action methods NFQCA, DDPG and SAC add one more axis, the
actor that replaces the maximum in the target, and belong to the
[next chapter](amortized-action-optimization.md).

## Regression Losses and Noise Models

At a fixed time $t$, the fit is a statistical estimation problem: given
samples from $\hat{P}_t^{\text{fit}}$, choose $\boldsymbol{\theta}$ to
minimize $\mathbb{E}_{((s,a),y)\sim \hat{P}_t^{\text{fit}}}[\ell(q(s,a;\boldsymbol{\theta}), y)]$.
The loss $\ell$ is a noise model for the targets. Squared error is the
likelihood of Gaussian noise, absolute error of Laplace noise, Gumbel
regression of extreme-value noise, and a classification loss over value bins
makes no parametric assumption. Bellman targets are noisy, biased by the
maximum, and bootstrapped, so the choice is not neutral, and two
alternatives to squared error have done well in practice.

### Gumbel Regression

Extreme value theory says that the maximum of several Gaussian errors has
Gumbel-distributed tails. {cite:t}`garg2023extreme` took the distribution
seriously in Extreme Q-Learning (XQL): instead of modeling the error of a
target as Gaussian,

$$
y_i = q^*(s_i, a_i) + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2),
$$

they model it as Gumbel,

$$
y_i = q^*(s_i, a_i) + \varepsilon_i, \quad \varepsilon_i \sim \text{Gumbel}(0, \beta),
$$

a distribution skewed to the right, because a target built from a maximum
occasionally lands far above the value it estimates. With $u_i = q(s_i, a_i;
\boldsymbol\theta) - y_i$, the negative log-likelihood is, up to constants,
the Gumbel regression loss

$$
\mathcal{L}_{\text{Gumbel}}(\boldsymbol{\theta}) = \sum_i \left[\exp\left(\frac{u_i}{\beta}\right) - \frac{u_i}{\beta} - 1\right],
$$

with a temperature $\beta$ that sets the heaviness of the tail. The loss is
zero at $u_i = 0$ and convex, and its gradient with respect to $q$ is

$$
\frac{\partial \mathcal{L}_{\text{Gumbel}}}{\partial q} = \frac{1}{\beta}\left[\exp\left(\frac{q - y}{\beta}\right) - 1\right].
$$

When $q < y$ the exponential is small and the gradient tends to $-1/\beta$,
so a target far above the prediction pulls it up only linearly; when $q > y$
the gradient grows exponentially with the error. The loss penalizes overestimation
much more than underestimation, which pushes the fit toward conservative
values when the targets are biased upward by the maximum. Squared error is
the right likelihood for averages, by the central limit theorem; Gumbel
regression is the right likelihood for maxima of light-tailed noise, by
extreme value theory.

```{code-cell} python
:tags: [hide-input]
:label: fig-gumbel-loss
:caption: Gumbel regression against squared error, as functions of the error $q - y$ of a prediction. (a) The loss $\exp(u/\beta) - u/\beta - 1$ grows exponentially for overestimates ($q > y$) and linearly for underestimates; a smaller temperature $\beta$ makes the asymmetry sharper. (b) Its derivative with respect to $q$, which sets how hard the fit pushes back: steep for overestimates, flat near $-1/\beta$ for underestimates, while the derivative of squared error (dashed) is symmetric.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

code_directory = Path.cwd() / "code"
if str(code_directory) not in sys.path:
    sys.path.insert(0, str(code_directory))
from book_figure_theme import INK, MUTED, PALETTE, figure_style, show

error = np.linspace(-3, 3, 500)                       # q - y
betas = {0.5: PALETTE["vermilion"], 1.0: PALETTE["orange"], 2.0: PALETTE["blue"]}

with figure_style():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for beta, color in betas.items():
        ax.plot(error, np.exp(error / beta) - error / beta - 1, color=color)
        bx.plot(error, (np.exp(error / beta) - 1) / beta, color=color)
    ax.plot(error, error ** 2, color=MUTED, lw=1, ls="--")
    bx.plot(error, 2 * error, color=MUTED, lw=1, ls="--")
    for axis in (ax, bx):
        axis.axvline(0, color=MUTED, lw=.6)
        axis.set_xlabel("error $q - y$")
        axis.set_xlim(-3, 3)
    ax.set_ylim(0, 10)
    bx.set_ylim(-5, 10)
    ax.text(1.0, 9.2, r"$\beta = 0.5$", color=betas[0.5], fontsize=8.5, ha="right")
    ax.text(1.75, 8.0, r"$\beta = 1$", color=betas[1.0], fontsize=8.5, ha="right")
    ax.text(2.95, 3.2, r"$\beta = 2$", color=betas[2.0], fontsize=8.5, ha="right")
    ax.text(-2.0, 7.4, "squared\nerror", color=MUTED, fontsize=8.5)
    bx.text(-2.9, -3.8, "squared\nerror", color=MUTED, fontsize=8.5)
    ax.set_ylabel("loss")
    bx.set_ylabel(r"derivative with respect to $q$")
    ax.set_title("(a) Gumbel loss")
    bx.set_title("(b) Its gradient")
    fig.tight_layout(w_pad=2)
show(fig)
```

{numref}`fig-gumbel-loss` draws the loss and its gradient for three
temperatures. The loss handles the asymmetry of the noise on its own, with no variance to
estimate and no weights to compute, and XQL has improved both value-based
and actor-critic methods, in particular in offline reinforcement learning,
where the bias of the maximum compounds across iterations without new data
to correct it.

```{prf:remark} XQL and soft Q-learning: target function against loss
:class: dropdown

XQL and soft Q-learning both involve the Gumbel distribution, at different places in the design space. Soft Q-learning changes the target function, using the smooth Bellman operator of the [regularized chapter](regularized-dp.md),

$$
g^{\text{soft}}(s,a,r,s'; \boldsymbol{\theta}) = r + \gamma \frac{1}{\beta}\log\sum_{a'} \exp(\beta q(s',a'; \boldsymbol{\theta})),
$$

with the squared loss unchanged; the Gumbel distribution enters through the Gumbel-max construction of the log-sum-exp, which is about making the optimal policy stochastic, not about the noise of the targets. XQL keeps the hard maximum in the target and changes the loss to $\ell_{\text{Gumbel}}(q,y) = \exp((q-y)/\beta) - (q-y)/\beta - 1$, where the Gumbel distribution models the noise of the TD errors. The log-sum-exp reduces maximization bias by averaging over actions, a property of the target; the Gumbel loss penalizes overestimation, a property of the loss. The two can be combined.
```

### Classification-Based Q-Learning

Nothing changes upstream: transitions are sampled from $\hat{P}_{\mathcal{B}_t}$
and pushed through the same target function. What changes is the
representation of the value and the loss. Choose a grid
$z_1 < z_2 < \cdots < z_K$ that spans the plausible returns. The network
outputs logits $\ell_{\boldsymbol{\theta}}(s,a) \in \mathbb{R}^K$, converted
to probabilities and to a scalar value by

$$
p_{\boldsymbol{\theta}}(k \mid s,a) = \frac{\exp(\ell_{\boldsymbol{\theta}}(s,a)_k)}{\sum_{j=1}^K \exp(\ell_{\boldsymbol{\theta}}(s,a)_j)}, \quad q(s,a; \boldsymbol{\theta}) = \sum_{k=1}^K z_k \, p_{\boldsymbol{\theta}}(k \mid s,a).
$$

Each scalar target $y_i$ becomes a distribution over the bins by the
**two-hot encoding**. If $y_i$ falls between $z_j$ and $z_{j+1}$,

$$
q_j(y_i) = \frac{z_{j+1} - y_i}{z_{j+1} - z_j}, \quad q_{j+1}(y_i) = \frac{y_i - z_j}{z_{j+1} - z_j}, \quad q_k(y_i) = 0 \text{ for } k \notin \{j, j+1\}.
$$

These are barycentric coordinates, and $\sum_k z_k q_k(y_i) = y_i$ recovers
the scalar exactly: the encoding is the linear interpolation of the
[finite-horizon chapter](finite-horizon-dp.md) ({prf:ref}`backward-recursion-interp`)
applied to the value axis.

```{code-cell} python
:tags: [hide-input]
:label: fig-two-hot
:caption: Two-hot encoding on 11 bins spanning $[-5, 5]$. (a) The target $y = 0.7$ puts weight $0.7$ on the bin at $1$ and $0.3$ on the bin at $0$, the barycentric coordinates of $0.7$ between them, so that the weighted average of the bins is $0.7$ again. (b) Three targets, each spread over the two bins around it.

def two_hot(y, bins):
    """Weights q_k(y): linear interpolation of y between the two bins around it."""
    q = np.zeros(len(bins))
    y = np.clip(y, bins[0], bins[-1])
    j = min(np.searchsorted(bins, y, side="right") - 1, len(bins) - 2)
    t = (y - bins[j]) / (bins[j + 1] - bins[j])
    q[j], q[j + 1] = 1 - t, t
    return q

bins = np.linspace(-5, 5, 11)
with figure_style():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    q = two_hot(0.7, bins)
    ax.bar(bins, q, width=.55, color=PALETTE["blue"])
    ax.axvline(0.7, ymax=.88, color=PALETTE["vermilion"], lw=1.2, ls="--")
    ax.text(0.7, 1.0, "$y = 0.7$", color=PALETTE["vermilion"], fontsize=8.5, ha="center")
    for k in np.flatnonzero(q):
        side = 1 if bins[k] > 0.7 else -1
        ax.text(bins[k] + .35 * side, q[k] - .03, f"{q[k]:.1f}", ha="left" if side > 0 else "right",
                va="top", fontsize=8.5, color=INK)
    ax.text(-4.8, .55, rf"$\Sigma_k\, z_k\, q_k(y) = {bins @ q:.1f}$", fontsize=8.5, color=INK)
    ax.set_title("(a) One target")
    for y, color, shift in ((-2.3, PALETTE["green"], -.18), (0.7, PALETTE["blue"], 0), (3.8, PALETTE["purple"], .18)):
        bx.bar(bins + shift, two_hot(y, bins), width=.18, color=color)
        bx.text(y, 1.0, f"$y = {y}$", color=color, fontsize=8.5, ha="center")
    bx.set_title("(b) Three targets")
    for axis in (ax, bx):
        axis.set_xticks(bins, [f"${b:.0f}$" for b in bins])
        axis.set_xlabel("bin value $z_k$")
    ax.set_ylabel("weight $q_k(y)$")
    ax.set_ylim(0, 1.12)
    fig.tight_layout(w_pad=2)
show(fig)
```

In {numref}`fig-two-hot`, each target puts its mass on the two bins around
it, with a weight that decreases with the distance to each bin. The loss is
the cross-entropy

$$
\mathcal{L}_{\text{CE}}(\boldsymbol{\theta}) = -\mathbb{E}_{((s,a),y) \sim \hat{P}_t^{\text{fit}}}\left[ \sum_{k=1}^K q_k(y) \log p_{\boldsymbol{\theta}}(k \mid s, a) \right],
$$

which projects the target distribution onto the predicted one in the
geometry of the simplex rather than of the real line. Its gradient with
respect to the logits is $p_{\boldsymbol\theta} - q$, bounded whatever the
size of the target. This is where its robustness comes from. Each sample
contributes a bounded gradient per bin, where squared error contributes a
gradient proportional to the error; the finite grid clips extreme targets
to its end bins, so that an outlier cannot set the scale of the regression;
and the two-hot encoding spreads each target over two bins, a label
smoothing that averages noisy targets at the same $(s,a)$. The encoding is
an averager in the sense of {prf:ref}`gordon-averager`, a convex combination
that preserves order and bounds, paired with a network that is not, so that
classification-based Q-learning is a hybrid of a monotone target structure
and a flexible fit. {cite:t}`farebrother2024stop` found that DQN and CQL
with squared error degrade when the Q-network is scaled to a large ResNet,
while the classification loss (HL-Gauss, which smooths the target with a
Gaussian instead of the two-hot encoding) keeps its performance.

## Exercises

The first exercises are short checks. The others follow the chapter: the
loops and targets of DQN, then the design space.

:::{exercise} Two loops
:label: ex-fqi-check-1

What happens in the outer loop and the inner loop of fitted Q-iteration?
:::

:::{solution} ex-fqi-check-1
:class: dropdown

The outer loop freezes the current Q-function and constructs Bellman targets. The inner loop fits a new function approximator to those targets.
:::

:::{exercise} Frozen targets
:label: ex-fqi-check-3

What instability does a target network reduce?
:::

:::{solution} ex-fqi-check-3
:class: dropdown

It slows the movement of regression targets. Without it, each parameter update simultaneously changes the predictions and the targets being fitted, creating a rapidly moving objective.
:::

:::{exercise} Indices of a flattened loop
:label: ex-fqi-indices

DQN copies its target network every $K = 500$ gradient steps.

1. At global step $t = 12{,}345$, which outer iteration $n$ and which inner
   step $k$ of the nested algorithm is it?
2. Which parameters computed the targets used at that step, in the notation
   of {prf:ref}`nfqi-explicit-inner`?
3. With hard copies every $K$ steps, the parameters that compute a target are
   on average $(K-1)/2$ steps old. With an exponential moving average of rate
   $\tau$ they are on average $(1-\tau)/\tau$ steps old. Which $\tau$ matches
   $K = 10^4$?
:::

:::{solution} ex-fqi-indices
:class: dropdown

1. $n = \lfloor 12{,}345/500 \rfloor = 24$ and $k = 12{,}345 - 24 \times 500 = 345$.
2. $\boldsymbol{\theta}_{24}$, the parameters at the end of outer iteration 23, held in the target network since step $12{,}000$.
3. $(1-\tau)/\tau = (K-1)/2$ gives $\tau = 2/(K+1) \approx 2 \times 10^{-4}$. The typical values $\tau \in [10^{-3}, 10^{-2}]$ of DDPG and TD3 correspond to copies every 200 to 2,000 steps.
:::

:::{exercise} One Double DQN target
:label: ex-fqi-double-target

A transition has reward $r = 0.5$ and successor $s'$, and $\gamma = 0.9$. At
$s'$ the online network predicts $(1.0, 1.3, 0.9)$ for three actions, and the
target network predicts $(1.2, 1.0, 1.1)$.

1. Compute the DQN target and the Double DQN target.
2. Which of the two uses the same estimate to select and to evaluate?
3. Suppose the true values at $s'$ are $(1.0, 1.0, 1.0)$. Which target is
   closer to the truth, and which bias of {ref}`sec-mc-overestimation` does
   the other one show?
:::

:::{solution} ex-fqi-double-target
:class: dropdown

1. DQN: $0.5 + 0.9 \times \max(1.2, 1.0, 1.1) = 1.58$. Double DQN: the online network selects the second action, which the target network values at $1.0$, so $0.5 + 0.9 \times 1.0 = 1.40$.
2. DQN: the target network both selects the first action and values it.
3. The true target is $0.5 + 0.9 = 1.40$. Double DQN hits it, and DQN overestimates by $0.18$ because it values the action that the noise of its own estimate made look best, the evaluation bias.
:::

:::{exercise} The deadly triad, algorithm by algorithm
:label: ex-fqi-triad

For each algorithm, say which of the three elements of the deadly triad it
has (function approximation that is not an averager, bootstrapped targets,
data from a policy other than the one being evaluated), and what, if
anything, guarantees its convergence.

1. Tree-based FQI on a fixed batch, with partitions frozen after a few iterations.
2. Neural fitted Q-iteration on a fixed batch.
3. DQN.
4. Tabular Q-learning with $\varepsilon$-greedy exploration.
5. Fitted policy evaluation with linear features on data collected by the policy being evaluated.
:::

:::{solution} ex-fqi-triad
:class: dropdown

1. Bootstrapping and off-policy data, but the fit is an averager once the partitions are frozen, so {prf:ref}`santos-vigo-aguiar-stability` applies to the empirical Bellman operator.
2. All three. No guarantee; warm starts and fits run to convergence on fixed targets limit the damage.
3. All three, with a buffer that moves. No guarantee; the target network, the warm start and the large buffer are the mitigations.
4. Bootstrapping and off-policy data, but a table is an averager in the trivial sense. Convergence follows from the ODE method under decreasing step sizes and visits to every pair.
5. Linear features and bootstrapping, but on-policy data, so the projected operator contracts in the $\xi$-norm ({ref}`sec-abe-convergence`).
:::

:::{exercise} Losses as noise models
:label: ex-fqi-losses

1. Show that minimizing squared error is maximum likelihood when
   $y = q + \varepsilon$ with Gaussian $\varepsilon$, and that minimizing the
   Gumbel loss is maximum likelihood when $\varepsilon \sim \text{Gumbel}(0, \beta)$,
   whose density is $\tfrac1\beta\exp\big(-\varepsilon/\beta - e^{-\varepsilon/\beta}\big)$.
2. Two-hot encoding writes a target as a convex combination of two bin
   values. Which result of the chapter on convergence and error bounds does this make it
   an instance of, and which interpolation of the finite-horizon chapter does
   it reuse?
3. The satellite example scaled its targets by the current message's
   estimate. Explain why this is a change of noise model, in the same sense as
   1, and what a classification loss with a fixed grid of bins would have done
   with the event whose cost was 2,070.
:::

:::{solution} ex-fqi-losses
:class: dropdown

1. With Gaussian $\varepsilon$, the negative log-likelihood of $y$ is $(q - y)^2/2\sigma^2$ plus a constant. With Gumbel $\varepsilon = y - q$, it is $\varepsilon/\beta + e^{-\varepsilon/\beta}$ plus a constant; writing $u = q - y = -\varepsilon$ gives $e^{u/\beta} - u/\beta$, the Gumbel loss up to the constant $-1$.
2. It is an averager ({prf:ref}`gordon-averager`) applied on the value axis, and it reuses the linear interpolation with nonnegative weights of {prf:ref}`backward-recursion-interp`.
3. Fitting the ratio $y/(C\,10^{\rho})$ assumes that errors are proportional to the predicted cost rather than additive, a multiplicative noise model, so one large cost no longer dominates the squared error of its leaf. A classification loss would have clipped the 2,070 target to the top bin, which bounds its influence too, but loses the information of how large the cost was.
:::

:::{exercise} A buffer of your own design
:label: ex-fqi-open

Take the inference-scheduling MDP of the chapter. Design a behavior policy
that collects a data set of 50,000 transitions as small in coverage as you
can make it while keeping the fitted policy within 2% of the exact return,
or show that no such policy exists. Report the coverage, the disagreement
with the exact policy, and the return, and relate your design to the action
gap at the states where the exact policy switches between prefill and decode.
:::

:::{solution} ex-fqi-open
:class: dropdown

This exercise is open-ended. The chapter's two buffers bound the answer: the
broad one covers everything and still loses 5% of the return near the
switching curve, where the action values differ by at most 1.24. A good
design concentrates data on the states next to that curve, where the action
gap is small and errors in the ranking are likely, and spends little on
states far from it, where any reasonable fit ranks the actions correctly.
:::

## Summary

Writing out the gradient steps of a neural fit and merging them with the
outer loop produces the online and target networks of DQN: the target
network is the outer iteration's parameters kept in memory, and the online
network is the inner iterate, which also collects the data once the buffer
is allowed to evolve. Double DQN changes only the target function, to
decouple selection from evaluation, and Q-learning is the limit of one
transition per update, a stochastic approximation whose convergence the ODE
method establishes for tables. All of them occupy one design space, with a
buffer, a target function and a fit as its coordinates, and the loss of the
fit is a noise model for the targets, which Gumbel regression and
classification losses match better than squared error.

Fitted Q-iteration can now learn action values from fixed or evolving
buffers, but every target still requires $\max_a q(s,a)$. What happens when
that action search is itself a nonlinear program? [Amortized action
optimization](amortized-action-optimization.md) learns an actor to replace
the repeated search while retaining the fitted-Q critic.
