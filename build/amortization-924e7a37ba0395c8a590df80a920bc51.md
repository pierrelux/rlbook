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

# Fitted Q-Iteration for Continuous Action Spaces

The previous chapter showed how fitted Q-iteration handles large state spaces through function approximation. FQI maintains a unified structure across batch and online settings: a replay buffer $\mathcal{B}_t$ inducing empirical distribution $\hat{P}_{\mathcal{B}_t}$, a target map $T_q$ derived from the Bellman operator, a loss function $\ell$, and an optimization budget. Algorithms differ in how they instantiate these components (buffer evolution, hard vs soft Bellman, update frequency), but all follow the same template.

However, this framework breaks down when the action space becomes large or continuous. Computing Bellman targets requires evaluating $\max_{a' \in \mathcal{A}} q(s',a';\theta)$ for each next state $s'$. When actions are continuous ($\mathcal{A} \subset \mathbb{R}^m$), this maximization requires solving a nonlinear program at every target computation. For a replay buffer with millions of transitions, this becomes computationally prohibitive.

This chapter addresses the continuous action problem while maintaining the FQI framework. We develop three strategies, unified by a common theme: **amortization**. Rather than solving the optimization problem $\max_a q(s,a;\theta)$ repeatedly at inference time, we invest computational effort during training to learn a mapping that directly produces good actions. This trades training-time cost for inference-time speed.

The strategies we examine are:

1. **Explicit optimization** (Section 2): Solve the maximization numerically for a subset of states, accepting the computational cost for exact solutions.

2. **Policy network amortization** (Sections 3-6): Learn a deterministic or stochastic policy network $\pi_{\boldsymbol{w}}$ that approximates $\arg\max_a q(s,a;\theta)$, enabling fast action selection via a single forward pass.

3. **Model-based amortization** (Section 6.3): Use a dynamics model to reduce reliance on the critic by unrolling trajectories, amortizing value computation across time rather than action selection.

Each approach represents a different point in the computation-accuracy trade-off, and all fit within the FQI template by modifying how targets are computed. 

# Embedded Optimization

Recall that in fitted Q methods, the main idea is to compute the Bellman operator only at a subset of all states, relying on function approximation to generalize to the remaining states. At each step of the successive approximation loop, we build a dataset of input state-action pairs mapped to their corresponding optimality operator evaluations: 

$$
\mathcal{D}_n = \{((s, a), (\Bellman q)(s, a; \boldsymbol{\theta}_n)) \mid (s,a) \in \mathcal{B}\}
$$

This dataset is then fed to our function approximator (neural network, random forest, linear model) to obtain the next set of parameters:

$$
\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}_n)
$$

While this strategy allows us to handle very large or even infinite (continuous) state spaces, it still requires maximizing over actions ($\max_{a \in A}$) during the dataset creation when computing the operator $\Bellman$ for each basepoint. This maximization becomes computationally expensive for large action spaces. We can address this by adding another level of optimization: for each sample added to our regression dataset, we employ numerical optimization methods to find actions that maximize the Bellman operator for the given state.

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

While explicit optimization provides exact solutions, it becomes computationally expensive when we need to compute targets for millions of transitions in a replay buffer. Can we avoid solving an optimization problem at every decision? The answer is amortization.

## Amortized Optimization Approach

This process is computationally intensive. We can "amortize" some of this computation by replacing the explicit optimization for each sample with a direct mapping that gives us an approximate maximizer directly. 
For Q-functions, recall that the operator is given by:

$$
(\Bellman q)(s,a) = r(s,a) + \gamma \int p(ds'|s,a)\max_{a' \in \mathcal{A}(s')} q(s', a')
$$

If $q^*$ is the optimal state-action value function, then $v^*(s) = \max_a q^*(s,a)$, and we can derive the optimal policy directly by computing the decision rule:

$$
\pi^\star(s) = \arg\max_{a \in \mathcal{A}(s)} q^\star(s,a)
$$

Since $q^*$ is a fixed point of $\Bellman$, we can write:

$$
\begin{align*}
q^\star(s,a) &= (\Bellman q^*)(s,a) \\
&= r(s,a) + \gamma \int p(ds'|s,a) \max_{a' \in \mathcal{A}(s')} q^\star(s', a') \\
&= r(s,a) + \gamma \int p(ds'|s,a) q^\star(s', \pi^\star(s'))
\end{align*}
$$

Note that $\pi^\star$ is implemented by our $\texttt{maximize}$ numerical solver in the procedure above. A practical strategy would be to collect these maximizer values at each step and use them to train a function approximator that directly predicts these solutions. Due to computational constraints, we might want to compute these exact maximizer values only for a subset of states, based on some computational budget, and use the fitted decision rule to generalize to the remaining states. This leads to the following amortized version:

```{prf:algorithm} Fitted Q-Iteration with Amortized Optimization
:label: fitted-q-iteration-amortized

**Input** Given an MDP $(S, A, P, R, \gamma)$, base points $\mathcal{B}$, subset for exact optimization $\mathcal{B}_{\text{opt}} \subset \mathcal{B}$, Q-function approximator $q(s,a; \boldsymbol{\theta})$, policy approximator $\pi_{\boldsymbol{w}}$, maximum iterations $N$, tolerance $\varepsilon > 0$

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
            1. $a^*_{s'} \leftarrow \pi_{\boldsymbol{w}_n}(s')$
        4. // Compute Q-function target using chosen action
        5. $y_{s,a} \leftarrow r + \gamma q(s', a^*_{s'}; \boldsymbol{\theta}_n)$
        6. $\mathcal{D}_q \leftarrow \mathcal{D}_q \cup \{((s,a), y_{s,a})\}$
    4. // Update both function approximators
    5. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}_q)$
    6. $\boldsymbol{w}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}_d)$
    7. // Compute convergence criteria
    8. $\delta_q \leftarrow \frac{1}{|\mathcal{D}_q|}\sum_{(s,a) \in \mathcal{D}_q} (q(s,a; \boldsymbol{\theta}_{n+1}) - q(s,a; \boldsymbol{\theta}_n))^2$
    9. $\delta_d \leftarrow \frac{1}{|\mathcal{D}_d|}\sum_{(s,a^*) \in \mathcal{D}_d} \|a^* - \pi_{\boldsymbol{w}_{n+1}}(s)\|^2$
    10. $n \leftarrow n + 1$
4. **until** ($\max(\delta_q, \delta_d) \geq \varepsilon$ or $n \geq N$)
5. **return** $\boldsymbol{\theta}_n$, $\boldsymbol{w}_n$
```

Note that the policy $\pi_{\boldsymbol{w}}$ is being trained on a dataset $\mathcal{D}_d$ containing optimal actions computed with respect to an evolving Q-function. Specifically, at iteration n, we collect pairs $(s', a^*_{s'})$ where $a^*_{s'} = \arg\max_a q(s', a; \boldsymbol{\theta}_n)$. However, after updating to $\boldsymbol{\theta}_{n+1}$, these actions may no longer be optimal with respect to the new Q-function.

A natural approach to handle this staleness would be to maintain only the most recent optimization data. We could modify our procedure to keep a sliding window of K iterations, where at iteration n, we only use data from iterations max(0, n-K) to n. This would be implemented by augmenting each entry in $\mathcal{D}_d$ with a timestamp:

$$
\mathcal{D}_\pi^t = \{(s', a^*_{s'}, t) \mid t \in \{n-K,\ldots,n\}\}
$$

where t indicates the iteration at which the optimal action was computed. When fitting the policy network, we would then only use data points that are at most K iterations old:

$$
\boldsymbol{w}_{n+1} \leftarrow \texttt{fit}(\{(s', a^*_{s'}) \mid (s', a^*_{s'}, t) \in \mathcal{D}_\pi^t, n-K \leq t \leq n\})
$$

This introduces a trade-off between using more data (larger K) versus using more recent, accurate data (smaller K). The choice of K would depend on how quickly the Q-function evolves and the computational budget available for computing exact optimal actions.

The main limitation of this approach, beyond the out-of-distribution drift, is that it requires computing exact optimal actions via the solver for states in $\mathcal{B}_{\text{opt}}$. Can we reduce or eliminate this computational expense? As the policy improves at selecting actions, we can bootstrap from these increasingly better choices. Continuously amortizing these improving actions over time creates a virtuous cycle of self-improvement toward the optimal policy. However, this bootstrapping process requires careful management: moving too quickly can destabilize training.


# Deterministic Parametrized Policies 

In this section, we consider deterministic parametrized policies of the form $\pi_{\boldsymbol{w}}(s)$ which directly output an action given a state. This approach differs from stochastic policies that output probability distributions over actions, making it particularly suitable for continuous control problems where the optimal policy is often deterministic. We'll see how fitted Q-value methods can be naturally extended to simultaneously learn both the Q-function and such a deterministic policy.

## The Amortization Problem for Continuous Actions

When actions are continuous, $a \in \mathbb{R}^d$, extracting a greedy policy from a Q-function becomes computationally expensive. Consider a robot arm control task where the action is a $d$-dimensional torque vector. To act greedily given Q-function $q(s,a; \boldsymbol{\theta})$, we must solve:

$$
\pi(s) = \arg\max_{a \in \mathcal{A}} q(s, a; \boldsymbol{\theta}),
$$

where $\mathcal{A} \subset \mathbb{R}^d$ is a continuous set (often a box or polytope). This requires running an optimization algorithm at every time step. For neural network Q-functions, this means solving a nonlinear program whose objective involves forward passes through the network.

After training converges, the agent must select actions in real-time during deployment. Running interior-point methods or gradient-based optimizers at every decision creates unacceptable latency, especially in high-frequency control where decisions occur at 100Hz or faster.

The solution is to **amortize** the optimization cost by learning a separate policy network $\pi_{\boldsymbol{w}}(s)$ that directly outputs actions. During training, we optimize $\boldsymbol{w}$ so that $\pi_{\boldsymbol{w}}(s) \approx \arg\max_a q(s,a; \boldsymbol{\theta})$ for states we encounter. At deployment, action selection reduces to a single forward pass through the policy network: $a = \pi_{\boldsymbol{w}}(s)$. The computational cost of optimization is paid during training (where time is less constrained) rather than at inference.

This introduces a second approximation beyond the Q-function. We now have two function approximators: a **critic** $q(s,a; \boldsymbol{\theta})$ that estimates values, and an **actor** $\pi_{\boldsymbol{w}}(s)$ that selects actions. The critic is trained using Bellman targets as in standard fitted Q-iteration. The actor is trained to maximize the critic:

$$
\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha \mathbb{E}_s \left[\nabla_{\boldsymbol{w}} q(s, \pi_{\boldsymbol{w}}(s); \boldsymbol{\theta})\right],
$$

where the expectation is over states in the dataset or replay buffer. This gradient ascent pushes the actor toward actions that the critic considers valuable. By the chain rule, this equals $(\nabla_a q(s,a; \boldsymbol{\theta})|_{a=\pi_{\boldsymbol{w}}(s)}) \cdot (\nabla_{\boldsymbol{w}} \pi_{\boldsymbol{w}}(s))$, which can be efficiently computed via backpropagation through the composition of the two networks.


## Neural Fitted Q-iteration for Continuous Actions (NFQCA)

To develop this approach, let's first consider an idealized setting where we have access to $q^\star$, the optimal Q-function. Then we can state our goal as finding policy parameters $\boldsymbol{w}$ that maximize $q^\star$ with respect to the actions chosen by our policy across the state space:

$$
\max_{\boldsymbol{w}} q^*(s, \pi_{\boldsymbol{w}}(s)) \quad \text{for all } s
$$

However, it's computationally infeasible to satisfy this condition for every possible state $s$, especially in large or continuous state spaces. To address this, we assume a distribution of states, denoted $\mu(s)$, and take the expectation, leading to the problem:

$$
\max_{\boldsymbol{w}} \mathbb{E}_{s \sim \mu(s)}[q^*(s, \pi_{\boldsymbol{w}}(s))]
$$

However in practice, we do not have access to $q^*$. Instead, we need to approximate $q^*$ with a Q-function $q(s, a; \boldsymbol{\theta})$, parameterized by $\boldsymbol{\theta}$, which we will learn simultaneously with the policy function $\pi_{\boldsymbol{w}}(s)$. Given a samples of initial states drawn from $\mu$, we then maximize this objective via a Monte Carlo surrogate problem:  


$$
\max_{\boldsymbol{w}} \mathbb{E}_{s \sim \mu(s)}[q(s, \pi_{\boldsymbol{w}}(s); \boldsymbol{\theta})] \approx
\max_{\boldsymbol{w}} \frac{1}{|\mathcal{B}|} \sum_{s \in \mathcal{B}}  q(s, \pi_{\boldsymbol{w}}(s); \boldsymbol{\theta})
$$

When using neural networks to parametrize $q$ and $\pi$, we obtain the Neural Fitted Q-Iteration with Continuous Actions (NFQCA) algorithm proposed by {cite}`Hafner2011`.

```{prf:algorithm} Neural Fitted Q-Iteration with Continuous Actions (NFQCA)
:label: nfqca

**Input** MDP $(S, A, P, R, \gamma)$, base points $\mathcal{B}$, Q-function $q(s,a; \boldsymbol{\theta})$, policy $\pi_{\boldsymbol{w}}(s)$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function, $\boldsymbol{w}$ for policy

1. Initialize $\boldsymbol{\theta}_0$, $\boldsymbol{w}_0$
2. **for** $n = 0,1,2,...$ **do**
    1. $\mathcal{D}_q \leftarrow \emptyset$
    2. For each $(s,a,r,s') \in \mathcal{B}$:
        1. $a'_{s'} \leftarrow \pi_{\boldsymbol{w}_n}(s')$
        2. $y_{s,a} \leftarrow r + \gamma q(s', a'_{s'}; \boldsymbol{\theta}_n)$
        3. $\mathcal{D}_q \leftarrow \mathcal{D}_q \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}_q)$
    4. $\boldsymbol{w}_{n+1} \leftarrow \texttt{minimize}_{\boldsymbol{w}} -\frac{1}{|\mathcal{B}|} \sum_{(s,a,r,s') \in \mathcal{B}} q(s, \pi_{\boldsymbol{w}}(s); \boldsymbol{\theta}_{n+1})$
3. **return** $\boldsymbol{\theta}_n$, $\boldsymbol{w}_n$
```


In practice, both the `fit` and `minimize` operations above are implemented using gradient descent. For the Q-function, the `fit` operation minimizes the mean squared error between the network's predictions and the target values:

$$
\texttt{fit}(\mathcal{D}_q) = \arg\min_{\boldsymbol{\theta}} \frac{1}{|\mathcal{D}_q|} \sum_{((s,a), y) \in \mathcal{D}_q} (q(s,a; \boldsymbol{\theta}) - y)^2
$$

For the policy update, the `minimize` operation uses gradient descent on the composition of the "critic" network $q$ and the "actor" network $d$. This results in the following update rule:

$$
\boldsymbol{w}_{n+1} = \boldsymbol{w}_n + \alpha \nabla_{\boldsymbol{w}} \left(\frac{1}{|\mathcal{B}|} \sum_{(s,a,r,s') \in \mathcal{B}} q(s, \pi_{\boldsymbol{w}}(s); \boldsymbol{\theta}_{n+1})\right)
$$

where $\alpha$ is the learning rate. Both operations can be efficiently implemented using modern automatic differentiation libraries and stochastic gradient descent variants like Adam or RMSProp.

## Connection to Classical Optimal Control

The gradient-based approach to action selection in NFQCA has deep roots in classical optimal control theory. To see this connection, consider a deterministic MDP with continuous states and actions where the dynamics are given by $s' = f(s,a)$ and rewards by $r(s,a)$. The optimal value function $v^*$ solves the Bellman equation:

$$
v^*(s) = \max_{a \in \mathcal{A}} \left\{ r(s,a) + \gamma v^*(f(s,a)) \right\}
$$

Assume that for each state $s$, the maximizer is unique and interior, denoted by the optimal policy $\pi^*(s)$. Define the right-hand side of the Bellman equation:

$$
g(s,a) := r(s,a) + \gamma v^*(f(s,a))
$$

The optimal action $a^* = \pi^*(s)$ at state $s$ must satisfy the first-order condition:

$$
\frac{\partial g}{\partial a}(s,a^*) = 0
$$

Using the chain rule to differentiate $g(s,a)$ with respect to $a$, and evaluating at the optimal action $a^* = \pi^*(s)$, the first-order condition becomes:

$$
\frac{\partial r}{\partial a}(s,a^*) + \gamma \frac{\partial v^*}{\partial s}(f(s,a^*)) \frac{\partial f}{\partial a}(s,a^*) = 0
$$

This is the classical Euler equation for deterministic optimal control. Here $\frac{\partial v^*}{\partial s}$ is the gradient of the value function (a row vector) and $\frac{\partial f}{\partial a}$ is the Jacobian of the dynamics with respect to the action.

NFQCA's policy gradient has exactly this structure. When the Q-function $q(s,a;\theta)$ approximates the right-hand side of the Bellman equation, the gradient at the current policy action $\tilde{a} = d(s;w)$ is:

$$
\nabla_a q(s,\tilde{a};\theta) \approx \frac{\partial r}{\partial a}(s,\tilde{a}) + \gamma \frac{\partial v}{\partial s'}(f(s,\tilde{a}))\frac{\partial f}{\partial a}(s,\tilde{a})
$$

Setting this gradient to zero through the policy update $\nabla_w q(s, d(s;w); \theta) = 0$ implements the same optimality condition. The difference is computational:

- **Classical optimal control**: Solve the Euler equation analytically to find $\pi^*(s)$ for simple dynamics and rewards.
- **NFQCA**: Approximate $\pi^*(s)$ with a neural network $d(s;w)$ trained via gradient ascent on the Q-function, handling complex dynamics and rewards.

Both seek the same mathematical object (the action that satisfies the first-order optimality condition) but through different computational approaches. NFQCA can be viewed as learning to solve the Euler equation numerically across the state space, amortizing the computation by fitting a function approximator to the solution. This geometric perspective also provides intuition: the policy gradient points in the direction of steepest ascent in Q-value space, and at optimality, we reach a stationary point where no local improvement is possible.

## Deep Deterministic Policy Gradient (DDPG)

Just as DQN adapted Neural Fitted Q-Iteration to the online setting, DDPG {cite}`lillicrap2015continuous` extends NFQCA to learn from data collected online. Like NFQCA, DDPG simultaneously learns a Q-function and a deterministic policy that maximizes it, but differs in how it collects and processes data.

Instead of maintaining a fixed set of basepoints, DDPG uses a replay buffer that continuously stores new transitions as the agent interacts with the environment. Since the policy is deterministic, exploration becomes challenging. DDPG addresses this by adding noise to the policy's actions during data collection:

$$
a = \pi_{\boldsymbol{w}}(s) + \eta_t
$$

where $\eta_t$ represents exploration noise drawn from an Ornstein-Uhlenbeck (OU) process. The OU process is particularly well-suited for control tasks as it generates temporally correlated noise, leading to smoother exploration trajectories compared to independent random noise. It is defined by the stochastic differential equation:

$$
d\eta_t = \theta(\mu - \eta_t)dt + \sigma dW_t
$$

where $\mu$ is the long-term mean value (typically set to 0), $\theta$ determines how strongly the noise is pulled toward this mean, $\sigma$ scales the random fluctuations, and $dW_t$ is a Wiener process (continuous-time random walk). For implementation, we discretize this continuous-time process using the Euler-Maruyama method:

$$
\mathcal{N}_{t+1} = \mathcal{N}_t + \theta(\mu - \mathcal{N}_t)\Delta t + \sigma\sqrt{\Delta t}\epsilon_t
$$

where $\Delta t$ is the time step and $\epsilon_t \sim \mathcal{N}(0,1)$ is standard Gaussian noise. Think of this process like a spring mechanism: when the noise value $\mathcal{N}_t$ deviates from $\mu$, the term $\theta(\mu - \mathcal{N}_t)\Delta t$ acts like a spring force, continuously pulling it back. Unlike a spring, however, this return to $\mu$ is not oscillatory - it's more like motion through a viscous fluid, where the force simply decreases as the noise gets closer to $\mu$. The random term $\sigma\sqrt{\Delta t}\epsilon_t$ then adds perturbations to this smooth return trajectory. This creates noise that wanders away from $\mu$ (enabling exploration) but is always gently pulled back (preventing the actions from wandering too far), with $\theta$ controlling the strength of this pulling force.

The policy gradient update follows the same principle as NFQCA:

$$
\nabla_{\boldsymbol{w}} \mathbb{E}_{s \sim \mu(s)}[q(s, \pi_{\boldsymbol{w}}(s); \boldsymbol{\theta})]
$$

We then embed this exploration mechanism into the data collection procedure and use the same flattened FQI structure that we adopted in DQN. Similar to DQN, flattening the outer-inner optimization structure leads to the need for target networks - both for the Q-function and the policy.

```{prf:algorithm} Deep Deterministic Policy Gradient (DDPG)
:label: ddpg

**Input** MDP $(S, A, P, R, \gamma)$, Q-network $q(s,a; \boldsymbol{\theta})$, policy network $\pi_{\boldsymbol{w}}(s)$, learning rates $\alpha_q, \alpha_d$, replay buffer size $B$, mini-batch size $b$, target update frequency $K$

**Initialize**
1. Parameters $\boldsymbol{\theta}_0$, $\boldsymbol{w}_0$ randomly
2. Target parameters: $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_0$, $\boldsymbol{w}_{target} \leftarrow \boldsymbol{w}_0$
3. Initialize replay buffer $\mathcal{R}$ with capacity $B$
4. Initialize exploration noise process $\mathcal{N}$
5. $n \leftarrow 0$

6. **while** training:
    1. Observe current state $s$
    2. Select action with noise: $a = \pi_{\boldsymbol{w}_n}(s) + \mathcal{N}$
    3. Execute $a$, observe reward $r$ and next state $s'$
    4. Store $(s,a,r,s')$ in $\mathcal{R}$, replacing oldest if full
    5. Sample mini-batch of $b$ transitions $(s_i,a_i,r_i,s'_i)$ from $\mathcal{R}$
    6. For each sampled transition:
        1. $y_i \leftarrow r_i + \gamma q(s'_i, \pi_{\boldsymbol{w}_{target}}(s'_i); \boldsymbol{\theta}_{target})$
    7. Update Q-network: $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_n - \alpha_q \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_i(y_i - q(s_i,a_i;\boldsymbol{\theta}_n))^2$
    8. Update policy: $\boldsymbol{w}_{n+1} \leftarrow \boldsymbol{w}_n + \alpha_d \frac{1}{b}\sum_i \nabla_a q(s_i,a;\boldsymbol{\theta}_{n+1})|_{a=\pi_{\boldsymbol{w}_n}(s_i)} \nabla_{\boldsymbol{w}} \pi_{\boldsymbol{w}_n}(s_i)$
    9. If $n \bmod K = 0$:
        1. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_n$
        2. $\boldsymbol{w}_{target} \leftarrow \boldsymbol{w}_n$
    10. $n \leftarrow n + 1$
    
**return** $\boldsymbol{\theta}_n$, $\boldsymbol{w}_n$
```

## Twin Delayed Deep Deterministic Policy Gradient (TD3)

While DDPG provided a foundation for continuous control with deep RL, it suffers from similar overestimation issues as DQN. TD3 {cite}`fujimoto2018addressing` addresses these challenges through three key modifications: double Q-learning to reduce overestimation bias, delayed policy updates to reduce per-update error, and target policy smoothing to prevent exploitation of Q-function errors.

```{prf:algorithm} Twin Delayed Deep Deterministic Policy Gradient (TD3)
:label: td3

**Input** MDP $(S, A, P, R, \gamma)$, twin Q-networks $q^A(s,a; \boldsymbol{\theta}^A)$, $q^B(s,a; \boldsymbol{\theta}^B)$, policy network $\pi_{\boldsymbol{w}}(s)$, learning rates $\alpha_q, \alpha_d$, replay buffer size $B$, mini-batch size $b$, policy delay $d$, noise scale $\sigma$, noise clip $c$, exploration noise std $\sigma_{explore}$

**Initialize**
1. Parameters $\boldsymbol{\theta}^A_0$, $\boldsymbol{\theta}^B_0$, $\boldsymbol{w}_0$ randomly
2. Target parameters: $\boldsymbol{\theta}^A_{target} \leftarrow \boldsymbol{\theta}^A_0$, $\boldsymbol{\theta}^B_{target} \leftarrow \boldsymbol{\theta}^B_0$, $\boldsymbol{w}_{target} \leftarrow \boldsymbol{w}_0$
3. Initialize replay buffer $\mathcal{R}$ with capacity $B$
4. $n \leftarrow 0$

6. **while** training:
    1. Observe current state $s$
    2. Select action with Gaussian noise: $a = \pi_{\boldsymbol{w}_n}(s) + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma_{explore})$
    3. Execute $a$, observe reward $r$ and next state $s'$
    4. Store $(s,a,r,s')$ in $\mathcal{R}$, replacing oldest if full
    5. Sample mini-batch of $b$ transitions $(s_i,a_i,r_i,s'_i)$ from $\mathcal{R}$
    6. For each sampled transition:
        1. $\tilde{a}_i \leftarrow \pi_{\boldsymbol{w}_{target}}(s'_i) + \text{clip}(\mathcal{N}(0, \sigma), -c, c)$  // Add clipped noise
        2. $q_{target} \leftarrow \min(q^A(s'_i, \tilde{a}_i; \boldsymbol{\theta}^A_{target}), q^B(s'_i, \tilde{a}_i; \boldsymbol{\theta}^B_{target}))$
        3. $y_i \leftarrow r_i + \gamma q_{target}$
    7. Update Q-networks:
        1. $\boldsymbol{\theta}^A_{n+1} \leftarrow \boldsymbol{\theta}^A_n - \alpha_q \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_i(y_i - q^A(s_i,a_i;\boldsymbol{\theta}^A_n))^2$
        2. $\boldsymbol{\theta}^B_{n+1} \leftarrow \boldsymbol{\theta}^B_n - \alpha_q \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_i(y_i - q^B(s_i,a_i;\boldsymbol{\theta}^B_n))^2$
    8. If $n \bmod d = 0$:  // Delayed policy update
        1. Update policy: $\boldsymbol{w}_{n+1} \leftarrow \boldsymbol{w}_n + \alpha_d \frac{1}{b}\sum_i \nabla_a q^A(s_i,a;\boldsymbol{\theta}^A_{n+1})|_{a=\pi_{\boldsymbol{w}_n}(s_i)} \nabla_{\boldsymbol{w}} \pi_{\boldsymbol{w}_n}(s_i)$
        2. Soft update of target networks:
            1. $\boldsymbol{\theta}^A_{target} \leftarrow \tau\boldsymbol{\theta}^A_{n+1} + (1-\tau)\boldsymbol{\theta}^A_{target}$
            2. $\boldsymbol{\theta}^B_{target} \leftarrow \tau\boldsymbol{\theta}^B_{n+1} + (1-\tau)\boldsymbol{\theta}^B_{target}$
            3. $\boldsymbol{w}_{target} \leftarrow \tau\boldsymbol{w}_{n+1} + (1-\tau)\boldsymbol{w}_{target}$
    9. $n \leftarrow n + 1$
    
**return** $\boldsymbol{\theta}^A_n$, $\boldsymbol{\theta}^B_n$, $\boldsymbol{w}_n$
```
Similar to Double Q-learning, TD3 decouples selection from evaluation when forming the targets. However, instead of intertwining the two existing online and target networks, TD3 suggests learning two Q-functions simultaneously and uses their minimum when computing target values to help combat the overestimation bias further. 

Furthermore, when computing target Q-values, TD3 adds small random noise to the target policy's actions and clips it to keep the perturbations bounded. This regularization technique essentially implements a form of "policy smoothing" that prevents the policy from exploiting areas where the Q-function may have erroneously high values:

    $$\tilde{a} = \pi_{\boldsymbol{w}_{target}}(s') + \text{clip}(\mathcal{N}(0, \sigma), -c, c)$$

While DDPG used the OU process which generates temporally correlated noise, TD3's authors found that simple uncorrelated Gaussian noise works just as well for exploration. It is also easier to implement and tune since you only need to set a single parameter ($\sigma_{explore}$) for exploration rather than the multiple parameters required by the OU process ($\theta$, $\mu$, $\sigma$).


Finally, TD3 updates the policy network (and target networks) less frequently than the Q-networks, typically once every $d$ Q-function updates. This helps reduce the per-update error and gives the Q-functions time to become more accurate before they are used to update the policy.

# Soft Actor-Critic

Adapting NFQCA to the smooth Bellman optimality equations leads to the soft actor-critic algorithm {cite}`haarnoja2018soft`. To understand this connection, we first examine how the smooth Bellman equations follow from entropy regularization.

Consider the standard Bellman operator augmented with an entropy term. The smooth Bellman operator $\Bellman_\beta$ takes the form:

$$
(\Bellman_\beta v)(s) = \max_{\pi \in \Pi^{MR}}\{\mathbb{E}_{a \sim \pi}[r(s,a) + \gamma v(s')] + \beta\mathcal{H}(\pi)\}
$$

where $\mathcal{H}(\pi) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s)]$ represents the entropy of the policy. To find the solution to the optimization problem embedded in the operator $\Bellman_\beta$, we set the functional derivative of the objective with respect to the decision rule to zero:

$$
\frac{\delta}{\delta \pi(a|s)} \left[\int_A \pi(a|s)(r(s,a) + \gamma v(s'))da - \beta\int_A \pi(a|s)\log \pi(a|s)da \right] = 0
$$

Enforcing that $\int_A \pi(a|s)da = 1$ leads to the following Lagrangian:

$$
r(s,a) + \gamma v(s') - \beta(1 + \log \pi(a|s)) - \lambda(s) = 0
$$

Solving for $\pi$ shows that the optimal policy is a Boltzmann distribution 

$$
\pi^*(a|s) = \frac{\exp(\frac{1}{\beta}(r(s,a) + \gamma \mathbb{E}_{s'}[v(s')]))}{Z(s)}
$$

When we substitute this optimal policy back into the entropy-regularized objective, we obtain:

$$
\begin{align*}
v(s) &= \mathbb{E}_{a \sim \pi^*}[r(s,a) + \gamma v(s')] + \beta\mathcal{H}(\pi^*) \\
&= \beta \log \int_A \exp(\frac{1}{\beta}(r(s,a) + \gamma \mathbb{E}_{s'}[v(s')]))da
\end{align*}
$$
 
The smooth Bellman optimality operator for Q-factors is defined as:

$$
(\Bellman_\beta q)(s,a) = r(s,a) + \gamma \mathbb{E}_{s'}\left[\beta \log \int_A \exp(\frac{1}{\beta}q(s',a'))da'\right]
$$

This operator maintains the contraction property of its standard counterpart, guaranteeing a unique fixed point $q^*$. The optimal policy takes the form:

$$
\pi^*(a|s) = \frac{\exp(\frac{1}{\beta}q^*(s,a))}{Z(s)}
$$

where $Z(s) = \int_A \exp(\frac{1}{\beta}q^*(s,a))da$ is the partition function. The optimal value function can be recovered as:

$$
v^*(s) = \beta \log \int_A \exp(\frac{1}{\beta}q^*(s,a))da
$$

## Fitted Q-Iteration for the Smooth Bellman Equations

Following the principles of fitted value iteration, we can approximate the effect of the smooth Bellman operator by computing it exactly at a number of basepoints and generalizing elsewhere using function approximation. Concretely, given a collection of states $s_i$ and actions $a_i$, we would compute regression target values:

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
v_\psi(s) \approx \mathbb{E}_{a \sim \pi_{\phi}(\cdot|s)}\left[q_\theta(s,a) - \beta \log \pi_{\phi}(a|s)\right]
$$

By maintaining both approximators, we can estimate targets using sampled actions from our policy. Specifically, if we have a transition $(s_i,a_i,s'_i)$ and sample $a'_i \sim \pi_{\phi}(\cdot|s'_i)$, our target becomes:

$$
y_i = r(s_i,a_i) + \gamma\left(q_\theta(s'_i,a'_i) - \beta \log \pi_{\phi}(a'_i|s'_i)\right)
$$

This approach exists only due to the dual representation of the smooth Bellman operator as an entropy-regularized problem, which transforms the intractable log-sum-exp into a form we can estimate efficiently through sampling.

## Approximating Boltzmann Policies by Gaussians

The entropy-regularized objective and the smooth Bellman operator are mathematically equivalent. However, both formulations face a practical challenge: they require evaluating an intractable integral due to the Boltzmann distribution. Soft Actor-Critic (SAC) addresses this problem by approximating the optimal policy with a simpler, more tractable Gaussian distribution. Given the optimal soft policy:

$$
\pi^*(a|s) = \frac{\exp(\frac{1}{\beta}q^*(s,a))}{Z(s)}
$$

we seek to approximate it with a Gaussian policy:

$$
\pi_{\phi}(a|s) = \mathcal{N}(\mu_\phi(s), \sigma_\phi(s))
$$

This approximation task naturally raises the question of how to measure the "closeness" between the target Boltzmann distribution and a candidate Gaussian approximation. Following common practice in deep learning, we employ the Kullback-Leibler (KL) divergence as our measure of distributional distance. To find the best approximation, we minimize the KL divergence between our policy and the optimal policy, using our current estimate $q_\theta$ of $q^*$:

$$
\operatorname{minimize}_{\phi} \mathbb{E}_{s \sim \mu(s)}\left[D_{KL}\left(\pi_{\phi}(\cdot|s) \| \frac{\exp(\frac{1}{\beta}q_\theta(s,\cdot))}{Z(s)}\right)\right]
$$


However, an important question remains: how can we solve this optimization problem when it involves the intractable partition function $Z(s)$? To see this, recall that for two distributions p and q, the KL divergence takes the form $D_{KL}(p\|q) = \mathbb{E}_{x \sim p}[\log p(x) - \log q(x)]$. Let's denote the target Boltzmann distribution based on our current Q-estimate as:

$$
d_\theta(a|s) = \frac{\exp(\frac{1}{\beta}q_\theta(s,a))}{Z_\theta(s)}
$$

Then the KL minimization becomes:

$$
\begin{align*}
D_{KL}(\pi_{\phi}(\cdot|s)\|d_\theta) &= \mathbb{E}_{a \sim \pi_{\phi}(\cdot|s)}[\log \pi_{\phi}(a|s) - \log d_\theta(a|s)] \\
&= \mathbb{E}_{a \sim \pi_{\phi}(\cdot|s)}\left[\log \pi_{\phi}(a|s) - \log \left(\frac{\exp(\frac{1}{\beta}q_\theta(s,a))}{Z_\theta(s)}\right)\right] \\
&= \mathbb{E}_{a \sim \pi_{\phi}(\cdot|s)}\left[\log \pi_{\phi}(a|s) - \frac{1}{\beta}q_\theta(s,a) + \log Z_\theta(s)\right]
\end{align*}
$$

Since $\log Z(s)$ is constant with respect to $\phi$, minimizing this KL divergence is equivalent to:

$$
\operatorname{minimize}_{\phi} \mathbb{E}_{s \sim \mu(s)}\mathbb{E}_{a \sim \pi_{\phi}(\cdot|s)}[\log \pi_{\phi}(a|s) - \frac{1}{\beta}q_\theta(s,a)]
$$

## Reparameterizing the Objective 

One last challenge remains: $\phi$ appears in the distribution underlying the inner expectation, as well as in the integrand. This setting departs from standard empirical risk minimization (ERM) in supervised learning where the distribution of the data (e.g., cats and dogs in image classification) remains fixed regardless of model parameters. Here, however, the "data" - our sampled actions - depends directly on the parameters $\phi$ we're trying to optimize.

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
Second, $\phi$ enters through the composition of $q^\star$ with $f_\phi$: $q^\star(s,f_\phi(s,\epsilon))$. The chain rule for this composition involves derivatives of both functions. Since SAC learns the Q-function simultaneously with the policy, we can compute all required derivatives through automatic differentiation. 

This composition of policy and value functions - where $f_\phi$ enters as input to $q_\theta$ - directly parallels the structure we encountered in deterministic policy methods like NFQCA and DDPG. In those methods, we optimized:

$$
\max_{\phi} \mathbb{E}_{s \sim \mu(s)}[q_\theta(s, f_\phi(s))]
$$

where $f_\phi(s)$ was a deterministic policy. SAC extends this idea to stochastic policies by having $f_\phi$ transform both state and noise:

$$
\max_{\phi} \mathbb{E}_{s \sim \mu(s)}\mathbb{E}_{\epsilon \sim \mathcal{N}(0,1)}[q_\theta(s,f_\phi(s,\epsilon))]
$$

Thus, rather than learning a single action for each state as in DDPG, we learn a function that transforms random noise into actions, explicitly parameterizing a distribution over actions while maintaining the same underlying principle of differentiating through composed policy and value functions.


```{prf:algorithm} Soft Actor-Critic
:label: sac

**Input** MDP $(S, A, P, R, \gamma)$, Q-networks $q^1(s,a; \boldsymbol{\theta}^1)$, $q^2(s,a; \boldsymbol{\theta}^2)$, value network $v(s; \boldsymbol{\psi})$, policy network $\pi_{\boldsymbol{\phi}}(a|s)$, learning rates $\alpha_q, \alpha_v, \alpha_\pi$, replay buffer size $B$, mini-batch size $b$, target smoothing coefficient $\tau$

**Initialize**
1. Parameters $\boldsymbol{\theta}^1_0$, $\boldsymbol{\theta}^2_0$, $\boldsymbol{\psi}_0$, $\boldsymbol{\phi}_0$ randomly
2. Target parameters: $\boldsymbol{\bar{\psi}}_0 \leftarrow \boldsymbol{\psi}_0$
3. Initialize replay buffer $\mathcal{R}$ with capacity $B$

**while** training:
1. Observe current state $s$
2. Sample action from policy: $a \sim \pi_{\boldsymbol{\phi}}(a|s)$
3. Execute $a$, observe reward $r$ and next state $s'$
4. Store $(s, a, r, s')$ in $\mathcal{R}$, replacing oldest if full
5. Sample mini-batch of $b$ transitions $(s_i, a_i, r_i, s'_i)$ from $\mathcal{R}$

**Update Value Network:**
1. Compute target for value network:

   $$
   y_v = \mathbb{E}_{a' \sim \pi_{\boldsymbol{\phi}}(\cdot|s')} \left[ \min \left( q^1(s', a'; \boldsymbol{\theta}^1), q^2(s', a'; \boldsymbol{\theta}^2) \right) - \alpha \log \pi_{\boldsymbol{\phi}}(a'|s') \right]
   $$
2. Update $\boldsymbol{\psi}$ via gradient descent:

   $$
   \boldsymbol{\psi} \leftarrow \boldsymbol{\psi} - \alpha_v \nabla_{\boldsymbol{\psi}} \frac{1}{b} \sum_i (v(s_i; \boldsymbol{\psi}) - y_v)^2
   $$

**Update Q-Networks:**
1. Compute targets for Q-networks:

   $$
   y_q = r_i + \gamma \cdot v(s'_i; \boldsymbol{\bar{\psi}})
   $$
2. Update $\boldsymbol{\theta}^1$ and $\boldsymbol{\theta}^2$ via gradient descent:

   $$
   \boldsymbol{\theta}^j \leftarrow \boldsymbol{\theta}^j - \alpha_q \nabla_{\boldsymbol{\theta}^j} \frac{1}{b} \sum_i (q^j(s_i, a_i; \boldsymbol{\theta}^j) - y_q)^2, \quad j \in \{1, 2\}
   $$

**Update Policy Network:**
1. Sample actions $a \sim \pi_{\boldsymbol{\phi}}(\cdot|s_i)$ for each $s_i$ in the mini-batch
2. Update $\boldsymbol{\phi}$ via gradient ascent:

   $$
   \boldsymbol{\phi} \leftarrow \boldsymbol{\phi} + \alpha_\pi \nabla_{\boldsymbol{\phi}} \frac{1}{b} \sum_i \left[ \alpha \log \pi_{\boldsymbol{\phi}}(a|s_i) - q^1(s_i, a; \boldsymbol{\theta}^1) \right]
   $$

**Update Target Value Network:**

$$
\boldsymbol{\bar{\psi}} \leftarrow \tau \boldsymbol{\psi} + (1 - \tau) \boldsymbol{\bar{\psi}}
$$

**return** Learned parameters $\boldsymbol{\theta}^1$, $\boldsymbol{\theta}^2$, $\boldsymbol{\psi}$, $\boldsymbol{\phi}$
```

## Stochastic Value Gradients: Model-Based Amortization

The methods we've seen so far (NFQCA, DDPG, TD3, SAC) all amortize action selection by learning a policy network that directly outputs actions. This amortizes the $\arg\max$ operation across states. But there's another axis of amortization: across time.

Consider the Q-function in the Bellman target: $y = r + \gamma q(s',a';\theta)$. The Q-value itself represents an expectation over future rewards. Computing this exactly would require rolling out trajectories to termination. Instead, we bootstrap: we use the current Q-function estimate to approximate future value. This is temporal bootstrapping.

If we have a dynamics model $f(s,a,\xi)$, we can reduce our reliance on bootstrapping by explicitly unrolling the model for $n$ steps before using the critic. This trades model accuracy for reduced bootstrap error. The Stochastic Value Gradients (SVG) framework {cite}`Heess2015` formalizes this idea.

### The SVG Spectrum

For a stochastic policy $\pi_{\boldsymbol{w}}(a|s)$ and dynamics model $f(s,a,\xi)$ where $\xi$ is noise, the Stochastic Value Gradients (SVG) framework defines an n-step objective:

$$
J^{\text{SVG}(n)}(\boldsymbol{w}) = \mathbb{E}_{s,\{\epsilon_i\},\{\xi_i\}}\left[\sum_{i=0}^{n-1} \gamma^i r(s_i,a_i) + \gamma^n q(s_n,a_n;\theta)\right]
$$

where $a_i = \pi_{\boldsymbol{w}}(s_i,\epsilon_i)$, $s_{i+1} = f(s_i,a_i,\xi_i)$, and the noise variables $\epsilon_i, \xi_i$ enable reparameterization. This objective interpolates between:

- **SVG(0)**: Pure critic, no model unroll ($n=0$). Equivalent to DDPG/SAC with stochastic policy.
- **SVG($n$)**: Hybrid, unroll $n$ steps then bootstrap with critic. Trades model error for reduced bootstrapping.
- **SVG($\infty$)**: Pure model-based, no critic ($n \to \infty$). Full backpropagation through model rollouts.

The gradient can be computed via automatic differentiation through the reparameterized objective. SAC implements this for $n=0$, using the reparameterization trick to differentiate through the stochastic policy. SVG extends this approach to also differentiate through stochastic dynamics for $n > 0$.

DDPG and SAC amortize the $\arg\max$ over actions. SVG adds another axis of amortization: expectations over future states. By replacing bootstrapping with explicit model rollouts, SVG reduces bias at the cost of additional computation during training when the model is accurate.

For implementation details and the full SVG algorithm with noise inference, we refer to {cite}`Heess2015`. The critical point for our narrative is recognizing SVG as another instance of the amortization principle: paying computational cost during training (model rollouts) to reduce reliance on learned approximations (the critic).

# Comparison and Practical Guidance

We have developed several methods for continuous-action FQI, all sharing the amortization theme but differing along key design axes. The primary distinction is between deterministic and stochastic policies, which affects exploration, robustness, and the structure of the algorithm.

## Design Choices

| Algorithm | Policy Type | Buffer | Target Map | Networks | Key Feature |
|-----------|-------------|--------|------------|----------|-------------|
| NFQCA | Deterministic | Fixed dataset | $r + \gamma q(s', d(s'))$ | Q + $\pi$ | Batch RL, amortize argmax |
| DDPG | Deterministic | Replay | $r + \gamma q(s', d(s'))$ | Q + $\pi$ + targets | Online NFQCA + OU noise |
| TD3 | Deterministic | Replay | $r + \gamma \min(q^A, q^B)(s', \tilde{a})$ | 2Q + $\pi$ + targets | Double Q + delayed updates |
| SAC | Stochastic | Replay | $r + \gamma v(s')$ | 2Q + V + $\pi$ + targets | Entropy regularization |
| SVG(0) | Stochastic | Replay | $r + \gamma q(s', a')$ | Q + $\pi$ | Reparameterized DDPG |
| SVG($n$) | Stochastic | Replay | Model rollout + Q | Q + $\pi$ + model | Hybrid model-based |

## When to Use Each Method

**Explicit optimization** (Algorithm 1): Use when you have a small number of decision points, need exact solutions for safety-critical applications, or have expensive simulations that justify optimization cost per sample.

**NFQCA**: Use for offline RL with a fixed dataset when you can afford multiple training epochs. Good baseline for understanding the actor-critic structure.

**DDPG**: Use for continuous control with deterministic dynamics and low-dimensional action spaces ($m \leq 10$). Simple and effective when environment is not too noisy.

**TD3**: Use when DDPG shows instability due to overestimation bias. Currently a standard baseline for deterministic policies in benchmarks.

**SAC**: Use for most continuous control tasks. The entropy regularization provides robustness to hyperparameters and enables exploration. Currently state-of-the-art for many benchmarks. The main cost is maintaining three networks (two Q-networks plus value network).

**SVG($n > 0$)**: Use when you have an accurate dynamics model and want to reduce sample complexity. Model errors can compound during rollouts, so this works best when the model is learned on high-quality data or when physics provides accurate dynamics.

## Summary

Continuous action spaces pose a computational challenge for fitted Q-iteration: the $\max_a q(s',a')$ operation requires solving a nonlinear program at every target computation. This chapter developed three strategies to address this challenge, unified by the theme of **amortization**.

We invest computational effort during training to learn mappings that enable fast inference:

1. **Action selection amortization**: Replace runtime $\arg\max$ with a policy network $d(s;w)$ trained to approximate the maximizer.

2. **Value amortization**: In SAC, use a value network $v(s;\psi)$ to amortize the soft-max computation.

3. **Temporal amortization**: In SVG, use model rollouts to reduce reliance on the critic, amortizing expectations over future states.

All methods fit within the FQI framework from the previous chapter. They differ in:
- **Buffer evolution**: Fixed (NFQCA) vs replay (DDPG, TD3, SAC, SVG)
- **Target map**: Hard max, double Q, soft max, or model rollout
- **Policy class**: Deterministic (NFQCA, DDPG, TD3) vs stochastic (SAC, SVG)
- **Optimization**: Batch (NFQCA) vs online (DDPG, TD3, SAC, SVG)

The deterministic vs stochastic distinction is fundamental. Deterministic policies require explicit exploration noise (OU process, Gaussian) during training. Stochastic policies provide exploration through entropy, enabling more robust learning but requiring careful handling of the log-probability terms in the objective.

The connection to classical optimal control (Euler equation) shows that these methods implement first-order optimality conditions numerically, learning to solve for optimal actions across the state space. The amortization perspective clarifies the computational trade-offs: we trade training-time cost for inference-time speed, enabling real-time control in continuous action spaces.

The next chapter develops policy gradient methods that optimize policies directly without maintaining Q-functions, offering an alternative to the actor-critic approach developed here.

